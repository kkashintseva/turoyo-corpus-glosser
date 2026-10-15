"""FastAPI app for the interactive gloss editor.

Two ways to run this. Locally (`cli.py edit`, and every test in this repo)
it is offline by construction: bound to 127.0.0.1, no login, exactly as
before this module grew a network-facing sibling. Hosted (`serve.py`, on
Railway) it sits behind HTTP Basic and a real port — see `_configure_auth`
below and `docs/DEPLOY_RAILWAY_ROADMAP_RU.md`, Stage 2.

Routes are deliberately small and JSON-first; the browser holds the live
document and PATCHes single cells, so a save is never bigger than one edit.
"""
from __future__ import annotations

import base64
import csv
import copy
import json
import io
import os
import re
import secrets
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path
from datetime import date
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.background import BackgroundTask

from maalula_glosser import lexicon_additions as LA
from maalula_glosser import lexicon_edits as LE
from maalula_glosser import gloss_shape as GS
from maalula_glosser.editor import labels as L
from maalula_glosser.editor import lexeme_resolve as LR
from maalula_glosser.editor import relink as RL
from maalula_glosser.editor import mutes as MU
from maalula_glosser.editor import senses as SE
from maalula_glosser.editor import session as S
from maalula_glosser.editor import docx_export as X
from maalula_glosser.editor import history as H
from maalula_glosser.editor import edit_log as EL
from maalula_glosser.editor import locks as LK
from maalula_glosser.editor.dictionary_links import ArnoldLinks
from maalula_glosser.editor import preferences
from maalula_glosser.editor import guide as G
from maalula_glosser.reusable import active_readings, sync_journal
from maalula_glosser.editor.store import SessionStore
from maalula_glosser.state import state_root, writable
from maalula_glosser.ingest.igt_parser import parse_igt
from maalula_glosser.ingest.flextext_parser import (evidence_from_text, flex_text_to_document,
                                                     iter_flextext)
from maalula_glosser.flex_corpus import FlexCorpusStore, FlexCorpusTextGlosser
from maalula_glosser.flex_import import import_flextext_to_state
from maalula_glosser.lift_lexicon import (LiftLexiconStore, backfill_editor_sessions,
                                          preferred_sense_gloss)
from maalula_glosser.normalization import TranscriptionNormalizer
from maalula_glosser.ritter_dictionary import (consonant_skeleton,
                                                default_ritter_dictionary)
from maalula_glosser.turoyo_verbs import TuroyoVerbParser

WEB = "web/editor"

UNAUTHENTICATED_PATHS = {"/healthz"}

# A browser upload has no server path to size-limit by inspection first, so
# the request body itself is capped — see docs/DEPLOY_RAILWAY_ROADMAP_RU.md,
# Stage 5.
MAX_PARTIAL_IGT_UPLOAD_BYTES = 25 * 1024 * 1024
# Real FLEx corpus exports are substantially larger than the DOCX imports.
MAX_FLEXTEXT_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_LIFT_UPLOAD_BYTES = 50 * 1024 * 1024

# What one word of "checked, every gloss confirmed" is worth, credited to
# whoever ticked the line's "проверено" box — the "галочка" the payment
# tally in create_app is built on.
REVIEW_RATE_RUB_PER_WORD = 3


def _configure_auth(app: FastAPI) -> None:
    """HTTP Basic in front of every route but `/healthz`, opt-in.

    Deliberately **not** a hard requirement here: `create_app` is also what
    every test in this repo and the local `cli.py edit` command build, on
    127.0.0.1, offline, with neither variable ever set — requiring them here
    would refuse to build the app in both of those cases. The network-facing
    entry point (`serve.py`, Stage 3 of the deploy road-map) is where a
    missing credential must refuse to *start the server*; here, a missing
    credential means "not deployed", not "deployed insecurely", so the two
    checks live at different layers on purpose.

    Both variables set is the only configuration that enables auth. Neither
    set leaves the app exactly as it always was. Exactly one set is a
    half-finished deployment, not a valid third state — refused loudly,
    the same instant, rather than left to surface later as a silently
    unlocked editor on a public URL.
    """
    user = os.environ.get("MAALULA_USER")
    password = os.environ.get("MAALULA_PASSWORD")
    if not user and not password:
        return
    if not (user and password):
        raise RuntimeError(
            "MAALULA_USER and MAALULA_PASSWORD must both be set, or neither. "
            "One without the other is a half-configured login, not a choice "
            "to run without one.")

    @app.middleware("http")
    async def require_login(request: Request, call_next):
        if request.url.path in UNAUTHENTICATED_PATHS:
            return await call_next(request)
        scheme, _, encoded = request.headers.get("authorization", "").partition(" ")
        given_user = given_password = None
        if scheme.lower() == "basic" and encoded:
            try:
                decoded = base64.b64decode(encoded).decode("utf-8")
            except (ValueError, UnicodeDecodeError):
                decoded = ""
            given_user, _, given_password = decoded.partition(":")
        # compare_digest raises TypeError on a non-ASCII str (a Cyrillic
        # password is a real possibility for this project) — compare bytes
        # instead, which it accepts for any script.
        if not (given_user is not None
                and secrets.compare_digest(given_user.encode("utf-8"),
                                           user.encode("utf-8"))
                and secrets.compare_digest((given_password or "").encode("utf-8"),
                                           password.encode("utf-8"))):
            return Response(
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="Maalula gloss editor"'})
        return await call_next(request)


# Literal path segments that name a document-creation route, not an existing
# document — a POST here has nothing to lock yet, and none of these ever
# collides with a real `document_id` (those come from `stable_id`, which
# never produces a bare English phrase like this).
PSEUDO_DOCUMENT_IDS = {"from-corpus", "from-corpus-text",
                       "import-partial-igt", "upload-partial-igt",
                       "import-flextext", "upload-flextext",
                       "import-lift", "upload-lift"}
# Sub-paths of a real document that are exempt from holding the pen: the lock
# routes themselves (or nothing could ever be claimed) and discussion threads,
# which are a conversation channel open to every reader, not an edit — see
# the design doc, §1 "Не входит в объём".
LOCK_EXEMPT_PREFIXES = ("/lock", "/discussions")
# `rename_document` and `delete_document` live at the bare
# `/api/documents/{document_id}` path (no further segment) and are called
# from the homepage's session list (`index.html`), which has no concept of
# holding the pen at all — it never sends `X-Editor-Client`. Gating them
# would make rename/trash permanently unusable rather than serialize a real
# content edit, so they are file-management actions, not the kind of edit
# this lock exists to serialize.


def _configure_edit_lock(app: FastAPI, locks: LK.EditLocks) -> None:
    """Refuse a mutation on a document to anyone but its current pen-holder.

    Deliberately a path-based middleware rather than a per-route dependency:
    the alternative is threading a check through every one of the ~25
    mutating routes under `/api/documents/{document_id}/...`, each one a
    fresh chance to forget it. A new mutating route only needs to live under
    that prefix to be covered automatically — see `_configure_auth` above,
    which the same reasoning already justified for login.
    """
    @app.middleware("http")
    async def require_edit_lock(request: Request, call_next):
        if request.method not in ("POST", "PATCH", "DELETE"):
            return await call_next(request)
        match = re.match(r"^/api/documents/([^/]+)(/.*)?$", request.url.path)
        if not match:
            return await call_next(request)
        document_id, rest = match.group(1), match.group(2) or ""
        if (document_id in PSEUDO_DOCUMENT_IDS or rest == ""
                or rest.startswith(LOCK_EXEMPT_PREFIXES)):
            return await call_next(request)
        client_id = request.headers.get("x-editor-client", "")
        if not locks.holder_is(document_id, client_id):
            return JSONResponse(
                status_code=423,
                content={"detail": "read-only", "holder": locks.status(document_id)},
            )
        return await call_next(request)


def _warm(glosser: TextGlosser) -> None:
    """Pay a rebuilt glosser's ~5s of lazy loads (the compositional analyzer's
    tables, the v4 productive-fallback engine) right now, at construction,
    instead of on whichever request happens to be first after it — every
    confirmed correction and retraction rebuilds `glosser` from scratch (see
    `TextGlosser.remember`'s docstring), so this runs after each of those
    too. Measured: construction ~1.8s, an unwarmed first `gloss_sentence`
    call +5.8s more on top of that, every call after ~0ms — see
    docs/DEPLOY_RAILWAY_ROADMAP_RU.md, Stage 4.

    Never lets a warm-up problem become a startup or request failure: the
    first real request pays the same lazy loads itself either way, so
    warming only saves the wait when it works, the same trade `_v4()`
    already makes for the engine it warms here."""
    try:
        glosser.gloss_sentence("hanna bayta.")
    except Exception:
        pass


class NewDocument(BaseModel):
    text: str
    title: str = ""
    # one translation per line of `text`; the author decides where a line ends
    translation: str = ""


class FromCorpusBody(BaseModel):
    example_id: str
    title: str = ""


class FromCorpusTextBody(BaseModel):
    source_text: str
    title: str = ""


class PartialIGTImportBody(BaseModel):
    path: str
    title: str = ""


class FlexTextImportBody(BaseModel):
    path: str


class LiftImportBody(BaseModel):
    path: str
    ranges_path: str | None = None


class TranslationBody(BaseModel):
    text: str


class ReviewBody(BaseModel):
    reviewed: bool
    name: str = ""


class NewLineBody(BaseModel):
    text: str
    translation: str = ""
    after_sentence_id: str | None = None


class CellEdit(BaseModel):
    gloss: str
    reusable: bool = False
    # Optional exact FieldWorks choice from the gloss-cell picker.  Free text
    # leaves these empty and therefore never guesses a lexical entry.
    lexeme_id: str | None = None
    sense_id: str | None = None
    grammatical_info: str | None = None


class EditorPreferences(BaseModel):
    reusable_default: bool


class GuideText(BaseModel):
    text: str


class ReuseMode(BaseModel):
    mode: Literal["inherit", "reusable", "local"]


class SegmentationEdit(BaseModel):
    text: str


class ResolveAlternativeBody(BaseModel):
    alternative_index: int


class BulkGlossBody(BaseModel):
    morph_ids: list[str]
    gloss: str


class BulkReadingBody(BaseModel):
    source_id: str
    source_version: str
    targets: dict[str, str]


class TokenLemmaBody(BaseModel):
    lexeme_id: str


class NewLemmaBody(BaseModel):
    lemma: str
    root: str = ""
    gloss: str = ""
    grammatical_info: str = ""


class ReuseDefaultBody(BaseModel):
    # None: follow the project setting; True/False: this text only
    reusable_default: bool | None = None


class MuteBody(BaseModel):
    name: str = ""
    key: str
    label: str = ""
    muted: bool = True


class RelinkBody(BaseModel):
    # token_id -> the lexeme the author picked among the candidates
    choices: dict[str, str] = {}
    # token_ids the author unticked in the preview: left exactly as they are
    skip: list[str] = []


class BulkTopologyBody(BaseModel):
    proposal_id: str
    targets: dict[str, str]


class MergeBody(BaseModel):
    count: int = 2
    # `=`/`-` keep the cells apart, a space collapses the run into one gloss
    joiner: str = "="


class TokenEdit(BaseModel):
    surface: str
    regloss: bool = True


class VerbSuggestionAcceptBody(BaseModel):
    suggestion_id: str


class RenameBody(BaseModel):
    new: str
    confirm: bool = False


class DefinitionBody(BaseModel):
    text: str


class CorrectionBody(BaseModel):
    surface: str
    segmentation: str
    gloss: str
    note: str = ""


class CommentBody(BaseModel):
    token_id: str
    morph_id: str | None = None
    text: str


class CommentEditBody(BaseModel):
    text: str


class LockClaimBody(BaseModel):
    client_id: str
    name: str


class LockReleaseBody(BaseModel):
    client_id: str


class DiscussionBody(BaseModel):
    token_id: str
    morph_id: str | None = None
    author: str
    text: str


class DiscussionMessageBody(BaseModel):
    author: str
    text: str


class DocumentTitleBody(BaseModel):
    title: str


class LexicalConfirmBody(BaseModel):
    gloss: str
    gloss_language: str = "en"
    confirmed: list[str] = []
    excluded: list[str] = []
    note: str = ""
    part_of_speech: str | None = None


class PartOfSpeechBody(BaseModel):
    part_of_speech: str | None = None


class LexiconAdditionBody(BaseModel):
    lexeme_id: str
    lemma_canonical: str
    part_of_speech: str | None = None
    grammatical_info: str | None = None
    gloss: str | None = None
    gloss_language: str = "en"


class ReassignTarget(BaseModel):
    document_id: str
    token_id: str


class LexiconReassignBody(BaseModel):
    targets: list[ReassignTarget]
    target_lexeme_id: str | None = None
    target_lemma: str | None = None


class LexiconMergeBody(BaseModel):
    source_lexeme_id: str
    target_lexeme_id: str | None = None
    target_lemma: str | None = None


class LexemeRenameBody(BaseModel):
    lemma_canonical: str


def _resolve_lexicon_target(glosser, target_lexeme_id: str | None,
                            target_lemma: str | None) -> str:
    """An author has no way to discover a raw `lexeme_id` to type into a
    prompt — resolve a target by its citation-form word instead, the same
    way `lexicon_add`'s own duplicate-citation-form check already matches
    (`LA.nfc(...).strip()` equality). `target_lexeme_id` stays accepted
    as-is for programmatic callers (and to disambiguate the rare case
    below), but the UI only ever sends `target_lemma` now."""
    if target_lexeme_id:
        if target_lexeme_id not in glosser.analyzer.lexemes:
            raise HTTPException(404, f"no such lexeme: {target_lexeme_id}")
        return target_lexeme_id
    if not target_lemma:
        raise HTTPException(400, "provide target_lexeme_id or target_lemma")
    lemma = LA.nfc(target_lemma).strip()
    matches = [lexeme_id for lexeme_id, lexeme in glosser.analyzer.lexemes.items()
               if lexeme_id and LA.nfc(lexeme.get("lemma_canonical")).strip() == lemma]
    if not matches:
        raise HTTPException(404, f"no lexeme with citation form: {target_lemma}")
    if len(matches) > 1:
        # Инвариант #7: несколько лексем с одинаковой словарной формой — это
        # выбор для автора, не повод привязать к первой. Каждый кандидат
        # несёт root/form_count (та же форма, что и `candidate_details` в
        # очереди «Словника») именно затем, чтобы автор мог отличить их и
        # прислать сюда же явный target_lexeme_id.
        raise HTTPException(409, {
            "error": "this citation form belongs to more than one lexeme",
            "lemma_canonical": target_lemma,
            "candidates": [
                {"lexeme_id": cid,
                 "root": glosser.analyzer.lexemes[cid].get("root_canonical"),
                 "form_count": len(glosser.analyzer.forms_by_lexeme.get(cid, []))}
                for cid in sorted(matches)
            ],
        })
    return matches[0]


def GS_fold(surface: str) -> str:
    """A surface for comparison with a form list: no punctuation, no case."""
    from maalula_glosser.normalization.transcription import fold_allophones
    text = S.nfc(fold_allophones(surface or "")).replace("‿", "")
    return re.sub(r"^[^\w]+|[^\w]+$", "", text).casefold()


def _replace_lexical_head(cells: list[dict[str, Any]], meaning: str) -> bool:
    """Put `meaning` in place of the lexical head of the stem cell
    (`say.SC` -> `tell.SC`); True if a cell changed."""
    stem, _head = RL.host_stem(cells)      # b=blōt-a: the host, not `with`
    for cell in cells:
        gloss = cell.get("gloss") or ""
        if not GS.has_lexical_part(gloss) or (stem and S.nfc(cell.get("form") or "") != stem):
            continue
        head = GS.lexical_head(gloss)
        if not head or not gloss.startswith(head) or head == meaning:
            return False
        cell["gloss"] = meaning + gloss[len(head):]
        return True
    return False


def _find_token(document: dict[str, Any], token_id: str) -> tuple[dict, dict]:
    for sentence in document.get("sentences") or []:
        for token in sentence.get("tokens") or []:
            if token.get("token_id") == token_id:
                return sentence, token
    raise HTTPException(404, f"token {token_id} not found")


def _find_cell(token: dict[str, Any], morph_id: str) -> dict:
    for morpheme in token.get("morphemes") or []:
        if morpheme.get("morph_id") == morph_id:
            return morpheme
    raise HTTPException(404, f"morpheme {morph_id} not found")


def _find_comment(document: dict[str, Any], comment_id: str) -> tuple[dict, dict]:
    for sentence in document.get("sentences") or []:
        for comment in sentence.get("comments") or []:
            if comment.get("comment_id") == comment_id:
                return sentence, comment
    raise HTTPException(404, f"comment {comment_id} not found")


def _find_thread(document: dict[str, Any], thread_id: str) -> tuple[dict, dict]:
    for sentence in document.get("sentences") or []:
        for thread in sentence.get("discussions") or []:
            if thread.get("thread_id") == thread_id:
                return sentence, thread
    raise HTTPException(404, f"discussion {thread_id} not found")


def _find_message(thread: dict[str, Any], message_id: str) -> dict:
    for message in thread.get("messages") or []:
        if message.get("message_id") == message_id:
            return message
    raise HTTPException(404, f"message {message_id} not found")


# Which one-off repairs have already been applied to the saved sessions. Kept
# without a `.json` suffix on purpose: `SessionStore.list` globs `*.json` and
# would otherwise offer this file as an untitled document.
MIGRATIONS = ".migrations"


def _applied_migrations(store: SessionStore) -> set[str]:
    try:
        recorded = json.loads(
            (store.dir / MIGRATIONS).read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, NotADirectoryError):
        return set()
    return set(recorded) if isinstance(recorded, list) else set()


def _mark_migration(store: SessionStore, name: str) -> None:
    store.dir.mkdir(parents=True, exist_ok=True)
    done = sorted(_applied_migrations(store) | {name})
    (store.dir / MIGRATIONS).write_text(
        json.dumps(done, ensure_ascii=False), encoding="utf-8")


def _repair_saved_waybin(store: SessionStore, glosser: TextGlosser,
                         inventory: L.LabelInventory) -> int:
    """Repair the two editor states produced by the old Waybin bugs.

    The analyzer now handles initial capitalisation before composition, but
    saved documents keep their old cells.  Two shapes are mechanically safe to
    migrate:

    * untouched auto ``Wayb-in / on-MP.INDF`` → re-gloss with the fixed analyzer;
    * human-entered ``w-ayb-in / SC-be-MP`` → restore ``PST`` which the old
      inventory alias had changed to SC while the user was typing.

    No other token and no other human edit is touched.
    """
    repaired = 0
    for listed in store.list():
        document = store.load(listed["document_id"])
        changed = False
        for sentence in document.get("sentences") or []:
            for index, token in enumerate(sentence.get("tokens") or []):
                normalized = S.nfc(token.get("normalized")
                                   or token.get("surface") or "")
                normalized = normalized.strip(".,;:!?…").casefold()
                if normalized != "waybin":
                    continue
                morphemes = token.get("morphemes") or []
                forms = [S.nfc(m.get("form") or "").casefold()
                         for m in morphemes]
                glosses = [S.nfc(m.get("gloss") or "") for m in morphemes]

                # This is specifically the value the former PST→SC alias wrote.
                if forms == ["w", "ayb", "in"] and glosses[:2] == ["SC", "be"]:
                    morphemes[0]["gloss"] = "PST"
                    morphemes[0]["status"] = S.CONFIDENT
                    changed = True
                    repaired += 1
                    continue

                untouched = (not token.get("locked") and morphemes
                             and all(m.get("origin") == S.AUTO for m in morphemes))
                if not untouched or forms != ["wayb", "in"] \
                        or glosses != ["on", "MP.INDF"]:
                    continue
                result = glosser.gloss_token(
                    token.get("surface") or "Waybin",
                    sentence_initial=(index == 0))
                if result.get("segmentation") != "w-ayb-in" \
                        or result.get("gloss") != "PST-be-MP":
                    continue
                rebuilt = S.build_token(
                    {**result, "source_token": token.get("surface") or "Waybin"},
                    sentence.get("sentence_id"), index, inventory.apply_case)
                # Keep the stable identity and source spelling even if an older
                # version computed them slightly differently.
                rebuilt["token_id"] = token.get("token_id")
                rebuilt["surface"] = token.get("surface")
                rebuilt["sandhi_group"] = token.get("sandhi_group") or ""
                token.clear()
                token.update(rebuilt)
                changed = True
                repaired += 1
        if changed:
            store.save(document)
    return repaired


# The one editor session tracked in the checkout at its real document-id
# path (COPY'd there by the Dockerfile from the root-level, git-tracked
# "Muše Sanžar.json" — see the Dockerfile's own comment). Everything else
# under data/editor_sessions/ is gitignored runtime state that a fresh
# volume has no reason to start with; this one is the opposite: a real,
# already-transcribed-and-glossed text the author wants live from day one.
MUSHE_SANZAR_DOCUMENT_ID = "edoc_29b60ac34ce2a86b"


def _restore_mushe_sanzar(root: Path, store: SessionStore) -> bool:
    """Push the checkout's copy onto the volume once, even if the volume's
    `data/editor_sessions` already exists.

    `writable()` seeds `data/editor_sessions` from the checkout only the
    first time the volume's own copy of that directory doesn't exist yet —
    exactly wrong for this file, added to the checkout *after* a live
    Railway volume was already seeded (empty, since the directory is
    gitignored and had nothing to seed from at the time). Without this,
    the fix would sit in every future checkout and never reach the one
    volume it was meant for. Once applied, the volume's copy is authoritative
    again, the same as any other saved session — a later edit made live is
    never undone by a redeploy.

    Returns whether it actually copied anything: the caller only marks this
    migration done when it did. A build without the checkout file yet (a
    fork of this project, or simply before the Dockerfile carried it) must
    stay eligible to run once a later deploy does carry it, rather than
    being marked "done" for a boot that did nothing.
    """
    source = Path(root) / "data" / "editor_sessions" / f"{MUSHE_SANZAR_DOCUMENT_ID}.json"
    target = store.path(MUSHE_SANZAR_DOCUMENT_ID)
    if not source.is_file() or source.resolve() == target.resolve():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return True


def _glosser_profile() -> str:
    """The cleaned project has one language profile: Turoyo/FLEx."""
    return "flex"


def _make_glosser(root: Path) -> Any:
    return FlexCorpusTextGlosser(root)


def create_app(root: Path | str = ".", *, warm: bool = False) -> FastAPI:
    """`warm=True` pays `glosser`'s ~1.8s construction plus ~5.8s of lazy
    loads at startup instead of on the first real request — worth it for
    the one process that will actually face a user (`serve.py`), not for
    every test and `cli.py edit` invocation that builds this app and would
    otherwise all pay it too. The four rebuild sites below (a confirmed or
    retracted correction) warm unconditionally regardless of this flag:
    they only fire on an actual edit action, so only the handful of tests
    that exercise one pay for it — see docs/DEPLOY_RAILWAY_ROADMAP_RU.md,
    Stage 4."""
    root = Path(root).resolve()
    profile = _glosser_profile()
    app = FastAPI(title="Corpus gloss editor")
    locks = LK.EditLocks()
    # Registered before `_configure_auth`: Starlette runs middleware in the
    # reverse of registration order, so auth — registered second — ends up
    # outermost and refuses an unauthenticated request before the lock check
    # ever sees it, rather than leaking who holds the pen to a stranger.
    _configure_edit_lock(app, locks)
    _configure_auth(app)

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "profile": "flex", "missing_data": [],
                "flex_corpus": FlexCorpusStore(root).manifest(),
                "lift_lexicon": LiftLexiconStore(root).manifest()}

    templates = Jinja2Templates(directory=str(root / WEB / "templates"))
    app.mount("/static", StaticFiles(directory=str(root / WEB / "static")),
              name="static")

    # A curated correction lands by `git push`, i.e. into the checkout, not
    # onto an already-seeded volume — merge it in before anything reads the
    # journal, or a redeploy would carry the fix without it taking effect.
    sync_journal(root, L.CORRECTIONS)
    sync_journal(root, SE.LEXICAL_CORRECTIONS)

    glosser = _make_glosser(root)
    if warm:
        _warm(glosser)
    links = ArnoldLinks(root)
    inventory = (L.LabelInventory(root) if profile == "maalula" else
                 L.LabelInventory(root, inventory_path="config/gloss_inventory.flex.json",
                                  migrate_legacy=False))
    store = SessionStore(root)
    igt_normalizer = TranscriptionNormalizer([])
    flex_store = FlexCorpusStore(root)
    lift_store = LiftLexiconStore(root)
    # Ritter is a read-only suggestion layer. A missing/corrupt bundled file
    # must not prevent the corpus editor itself from starting.
    try:
        ritter_dictionary = default_ritter_dictionary()
    except (OSError, ValueError, UnicodeError):
        ritter_dictionary = None

    # The verb parser is an index over the *current* corpus analyzer.  Several
    # editor actions rebuild/reload ``glosser``; cache by analyzer identity so
    # those changes are picked up automatically on the next suggestion request
    # without rescanning the corpus on every click.
    verb_parser_cache: dict[str, Any] = {"analyzer": None, "parser": None}

    def _verb_parser() -> TuroyoVerbParser:
        analyzer = glosser.analyzer
        if verb_parser_cache["analyzer"] is not analyzer:
            verb_parser_cache["analyzer"] = analyzer
            verb_parser_cache["parser"] = TuroyoVerbParser(analyzer)
        return verb_parser_cache["parser"]

    def _refresh_flex_inventory() -> None:
        if profile != "flex":
            return
        changed = False
        for label in sorted(getattr(glosser.analyzer, "gloss_labels", set())):
            changed = inventory.register(label, category="FLEx corpus") or changed
        if changed:
            inventory.save()

    _refresh_flex_inventory()
    # Once, not on every start. Re-reading every saved text costs a third of a
    # second for four of them and grows with the shelf; worse, the repair
    # cannot tell the `SC` the old alias wrote from an `SC` the author means,
    # so leaving it armed would keep undoing a deliberate choice at each start.
    if profile == "maalula" and "waybin-pst" not in _applied_migrations(store):
        _repair_saved_waybin(store, glosser, inventory)
        _mark_migration(store, "waybin-pst")
    if profile == "maalula" and "mushe-sanzar-restore-2026-09-12" not in _applied_migrations(store):
        if _restore_mushe_sanzar(root, store):
            _mark_migration(store, "mushe-sanzar-restore-2026-09-12")
    # In memory on purpose: the journal serves the editing session and goes when
    # the server does. The documents are on disk and never depend on it. Every
    # entry is still mirrored to data/editor_sessions/edit_log.jsonl for
    # before/after statistics, independent of how long the journal itself lives.
    history = H.History(on_record=EL.sink(root))

    def _review_totals() -> dict[str, int]:
        """Rubles owed to each name that has ticked a "проверено" box
        anywhere in the project, one pass over the whole store — the same
        shape as `senses.text_titles_by_lexeme`, for the same reason: every
        document could have been touched, and there is no cheaper index."""
        totals: dict[str, int] = {}
        for listed in store.list():
            document = store.load(listed["document_id"])
            for sentence in document.get("sentences") or []:
                name = sentence.get("reviewed_by")
                count = sentence.get("reviewed_word_count")
                if name and count:
                    totals[name] = totals.get(name, 0) + count * REVIEW_RATE_RUB_PER_WORD
        return totals

    def _review_statistics() -> dict[str, Any]:
        """Project-wide statistics based ONLY on explicit review checkboxes."""
        editors: dict[str, dict[str, Any]] = {}
        documents: list[dict[str, Any]] = []

        total_lines = 0
        reviewed_lines_total = 0
        total_words = 0
        reviewed_words_total = 0

        for listed in store.list():
            document_id = listed["document_id"]
            document = store.load(document_id)
            sentences = document.get("sentences") or []

            document_total_lines = len(sentences)
            document_reviewed_lines = 0
            document_total_words = 0
            document_reviewed_words = 0

            reviewers: dict[str, dict[str, int]] = {}

            for sentence in sentences:
                line_words = sum(
                    1
                    for token in sentence.get("tokens") or []
                    if not S.is_speaker(token)
                )
                document_total_words += line_words

                name = S.nfc(sentence.get("reviewed_by") or "").strip()

                if not name:
                    continue

                count = sentence.get("reviewed_word_count")
                if count is None:
                    # Compatibility with an old checked row that predates
                    # reviewed_word_count.
                    count = line_words

                count = int(count or 0)

                document_reviewed_lines += 1
                document_reviewed_words += count

                reviewer = reviewers.setdefault(
                    name,
                    {"lines": 0, "words": 0},
                )
                reviewer["lines"] += 1
                reviewer["words"] += count

                person = editors.setdefault(
                    name,
                    {
                        "name": name,
                        "lines": 0,
                        "words": 0,
                        "texts": set(),
                    },
                )
                person["lines"] += 1
                person["words"] += count
                person["texts"].add(document_id)

            if document_total_lines and document_reviewed_lines == document_total_lines:
                status = "complete"
            elif document_reviewed_lines:
                status = "partial"
            else:
                status = "none"

            percent = (
                round(document_reviewed_lines * 100 / document_total_lines)
                if document_total_lines
                else 0
            )

            documents.append({
                "document_id": document_id,
                "title": (
                    document.get("title")
                    or listed.get("title")
                    or document_id
                ),
                "status": status,
                "total_lines": document_total_lines,
                "reviewed_lines": document_reviewed_lines,
                "pending_lines": document_total_lines - document_reviewed_lines,
                "percent": percent,
                "total_words": document_total_words,
                "reviewed_words": document_reviewed_words,
                "reviewers": [
                    {
                        "name": name,
                        "lines": values["lines"],
                        "words": values["words"],
                    }
                    for name, values in sorted(
                        reviewers.items(),
                        key=lambda item: (
                            -item[1]["words"],
                            item[0].casefold(),
                        ),
                    )
                ],
            })

            total_lines += document_total_lines
            reviewed_lines_total += document_reviewed_lines
            total_words += document_total_words
            reviewed_words_total += document_reviewed_words

        editor_rows = []
        for person in editors.values():
            editor_rows.append({
                "name": person["name"],
                "lines": person["lines"],
                "words": person["words"],
                "texts": len(person["texts"]),
                "payment": person["words"] * REVIEW_RATE_RUB_PER_WORD,
            })

        editor_rows.sort(
            key=lambda row: (-row["words"], row["name"].casefold())
        )

        status_rank = {
            "partial": 0,
            "none": 1,
            "complete": 2,
        }
        documents.sort(
            key=lambda row: (
                status_rank[row["status"]],
                row["title"].casefold(),
            )
        )

        complete = sum(
            1 for row in documents if row["status"] == "complete"
        )
        partial = sum(
            1 for row in documents if row["status"] == "partial"
        )
        untouched = sum(
            1 for row in documents if row["status"] == "none"
        )

        return {
            "editors": editor_rows,
            "documents": documents,
            "rate": REVIEW_RATE_RUB_PER_WORD,
            "summary": {
                "texts": len(documents),
                "complete": complete,
                "partial": partial,
                "none": untouched,
                "total_lines": total_lines,
                "reviewed_lines": reviewed_lines_total,
                "pending_lines": total_lines - reviewed_lines_total,
                "total_words": total_words,
                "reviewed_words": reviewed_words_total,
                "payment": (
                    reviewed_words_total * REVIEW_RATE_RUB_PER_WORD
                ),
                "percent": (
                    round(reviewed_lines_total * 100 / total_lines)
                    if total_lines
                    else 0
                ),
            },
        }


    def enrich(document: dict[str, Any]) -> dict[str, Any]:
        """Attach the read-only helpers the UI needs but the file need not keep:
        dictionary links and the label a cell would suggest."""
        links.annotate(document)
        active = {row.get("addition_id")
                  for row in active_readings(writable(root, L.CORRECTIONS))}
        lexical_active = {row.get("addition_id")
                          for row in active_readings(writable(root, SE.LEXICAL_CORRECTIONS))}
        # Exact morpheme readings are read-only UI helpers. If the
        # corpus/LIFT index has exactly one gloss for an unresolved component,
        # expose it immediately as `suggested`.
        #
        # enrich() is not saved, therefore this DOES NOT accept or persist the
        # proposal. The author's explicit edit is still required.
        morpheme_getter = getattr(
            glosser.analyzer,
            "morpheme_gloss_options",
            None,
        )
        suggestion_cache: dict[str, str | None] = {}

        for sentence in document["sentences"]:
            for token in sentence["tokens"]:
                if token.get("reusable_record"):
                    token["reusable_record"]["active"] = token["reusable_record"]["addition_id"] in active
                if token.get("lexical_record"):
                    token["lexical_record"]["active"] = token["lexical_record"]["addition_id"] in lexical_active

                # A segmented token may contain several independent lexical
                # morphemes/clitics. Resolve a lemma PER CELL, never one lemma
                # for the whole orthographic word.
                lift_index = getattr(glosser.analyzer, "lift", None)
                if lift_index is not None and lift_index.has_entries():
                    for cell in token.get("morphemes") or []:
                        # A manually/existing explicitly linked cell wins.
                        if cell.get("lexeme_id") and cell.get("lemma"):
                            continue

                        form = S.nfc(cell.get("form") or "").strip()
                        gloss = S.nfc(
                            cell.get("gloss")
                            or cell.get("suggested")
                            or ""
                        ).strip()

                        if not form:
                            continue

                        lexical_gloss = (
                            GS.lexical_head(gloss)
                            if gloss
                            else None
                        )

                        entry, sense, options = lift_index.match(
                            form,
                            gloss=lexical_gloss,
                        )

                        # Only an unambiguous exact lexical identity is shown.
                        if entry:
                            cell["lexeme_id"] = entry.get("lexeme_id")
                            cell["lemma"] = entry.get("lemma_canonical")
                            cell["lift_entry_id"] = entry.get("lift_id")
                            if sense:
                                cell["lift_sense_id"] = sense.get("sense_id")
                                cell["grammatical_info"] = (
                                    sense.get("grammatical_info")
                                    or cell.get("grammatical_info")
                                )

                if profile != "flex" or not callable(morpheme_getter):
                    continue

                for morpheme in token.get("morphemes") or []:
                    # Never replace a real analysis, whether FLEx or human.
                    if morpheme.get("gloss") or morpheme.get("suggested"):
                        continue

                    form = S.nfc(morpheme.get("form") or "").strip()
                    if not form:
                        continue

                    # Cache by this exact component, not by the whole token.
                    if form not in suggestion_cache:
                        labels = {
                            S.nfc(row.get("gloss") or "").strip()
                            for row in morpheme_getter(form, prefix="")
                            if S.nfc(row.get("gloss") or "").strip()
                        }

                        # Same label from several sources still counts as one
                        # unambiguous gloss. Different labels remain unresolved.
                        suggestion_cache[form] = (
                            next(iter(labels))
                            if len(labels) == 1
                            else None
                        )

                    suggestion = suggestion_cache[form]
                    if suggestion:
                        morpheme["suggested"] = suggestion
                        morpheme["status"] = S.UNRESOLVED

        return document

    # ---- pages ----
    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        documents = store.list()
        for row in documents:
            holder = locks.status(row.get("document_id"))
            row["holder_name"] = holder.get("name") if holder else None
        return templates.TemplateResponse(
            request, "index.html", {"documents": documents})

    @app.get("/doc/{document_id}", response_class=HTMLResponse)
    def editor(request: Request, document_id: str):
        try:
            document = store.load(document_id)
        except FileNotFoundError:
            raise HTTPException(404, "no such document")
        return templates.TemplateResponse(
            request, "editor.html", {"document_id": document_id})

    @app.get("/labels", response_class=HTMLResponse)
    def labels_page(request: Request):
        return templates.TemplateResponse(request, "labels.html", {})

    @app.get("/senses", response_class=HTMLResponse)
    def senses_page(request: Request):
        return templates.TemplateResponse(request, "senses.html", {})

    @app.get("/guide", response_class=HTMLResponse)
    def guide_page(request: Request):
        return templates.TemplateResponse(request, "guide.html", {})


    @app.get("/stats", response_class=HTMLResponse)
    def review_stats_page(request: Request):
        stats = _review_statistics()
        return templates.TemplateResponse(
            request,
            "stats.html",
            {"stats": stats},
        )


    @app.get("/api/editor-preferences")
    def get_preferences():
        return preferences.read(root)

    @app.patch("/api/editor-preferences")
    def set_preferences(body: EditorPreferences):
        return preferences.save(root, body.reusable_default)

    # ---- per-editor «не предлагать для …» ----
    @app.get("/api/editor-mutes")
    def get_mutes(name: str = ""):
        return MU.read(root, name)

    @app.post("/api/editor-mutes")
    def set_mute(body: MuteBody):
        try:
            return MU.set_muted(root, body.name, body.key, body.label, body.muted)
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    # ---- lexeme search for «Привязать к лемме…» ----
    @app.get("/api/lexicon/search")
    def lexicon_search(q: str = "", ids: str = "", limit: int = 25):
        """Lexemes whose citation form or root matches `q`: exact and prefix
        hits first. A leading √ searches roots only. `ids` (comma-separated)
        returns those lexemes as they are — a word's own candidates."""
        from maalula_glosser.normalization.transcription import fold_allophones
        analyzer = glosser.analyzer
        if ids:
            rows = []
            for lexeme_id in [i for i in ids.split(",") if i][:60]:
                record = analyzer.lexeme_record(lexeme_id)
                if not record:
                    continue
                gloss, _status = analyzer.lexical.resolve(lexeme_id)
                lift_entry = getattr(analyzer, "lift", None)
                entry = lift_entry.entry(lexeme_id) if lift_entry is not None else None
                senses = []
                if entry:
                    for sense in entry.get("senses") or []:
                        senses.append({
                            "sense_id": sense.get("sense_id"),
                            "gloss": preferred_sense_gloss(sense),
                            "grammatical_info": sense.get("grammatical_info"),
                        })
                rows.append({"lexeme_id": lexeme_id, "lemma": record.get("lemma_canonical"),
                             "root": record.get("root_canonical"), "gloss": gloss,
                             "verb": analyzer.is_verbal(lexeme_id),
                             "source": record.get("status"),
                             "lift_id": record.get("lift_id"),
                             "grammatical_infos": record.get("grammatical_infos") or [],
                             "senses": senses})
            return {"results": rows}
        query = S.nfc(fold_allophones(q)).strip().casefold()
        root_only = query.startswith("√")
        query = query.lstrip("√").strip()
        if not query:
            return {"results": []}
        scored = []
        for lexeme_id, lexeme in analyzer.lexemes.items():
            if not lexeme_id:
                continue
            lemma = S.nfc(lexeme.get("lemma_canonical") or "")
            if not lemma or lemma.startswith("["):
                continue
            root_text = S.nfc(lexeme.get("root_canonical") or "").casefold()
            folded = S.nfc(fold_allophones(lemma)).casefold()
            words = folded.replace("/", " ").split()
            form_texts = [S.nfc(fold_allophones(form.get("form_canonical") or "")).casefold()
                          for form in analyzer.forms_by_lexeme.get(lexeme_id, [])]
            if root_only:
                rank = 0 if root_text == query else (2 if root_text.startswith(query) else None)
            elif query in words or root_text == query or query in form_texts:
                rank = 0
            elif any(w.startswith(query) for w in words) or any(f.startswith(query) for f in form_texts):
                rank = 1
            elif query in folded or any(query in f for f in form_texts):
                rank = 3
            else:
                rank = None
            if rank is not None:
                scored.append((rank, len(lemma), lemma, lexeme_id, lexeme))
        scored.sort()
        results = []
        lift_index = getattr(analyzer, "lift", None)
        if profile == "flex" and not (lift_index and lift_index.has_entries()):
            seen_lemmas = set()
            wanted = max(1, min(limit, 60))
            for _rank, _len, lemma, lexeme_id, lexeme in scored:
                lemma_key = S.nfc(lemma).strip().casefold()
                if lemma_key in seen_lemmas:
                    continue
                seen_lemmas.add(lemma_key)
                related = [
                    (lid, rec) for lid, rec in analyzer.lexemes.items()
                    if S.nfc(rec.get("lemma_canonical") or "").strip().casefold()
                    == lemma_key
                ]
                senses = []
                for lid, _rec in related:
                    gloss, _status = analyzer.lexical.resolve(lid)
                    if gloss and gloss not in senses:
                        senses.append(gloss)
                results.append({
                    "lexeme_id": lexeme_id, "lexeme_ids": [lid for lid, _ in related],
                    "lemma": lemma, "root": lexeme.get("root_canonical"),
                    "gloss": " / ".join(senses) if senses else None,
                    "senses": senses,
                    "verb": any(analyzer.is_verbal(lid) for lid, _ in related),
                })
                if len(results) >= wanted:
                    break
        else:
            for _rank, _len, lemma, lexeme_id, lexeme in scored[:max(1, min(limit, 60))]:
                gloss, _status = analyzer.lexical.resolve(lexeme_id)
                entry = lift_index.entry(lexeme_id) if lift_index else None
                senses = []
                if entry:
                    for sense in entry.get("senses") or []:
                        senses.append({
                            "sense_id": sense.get("sense_id"),
                            "gloss": preferred_sense_gloss(sense),
                            "grammatical_info": sense.get("grammatical_info"),
                        })
                results.append({"lexeme_id": lexeme_id, "lemma": lemma,
                                "root": lexeme.get("root_canonical"), "gloss": gloss,
                                "verb": analyzer.is_verbal(lexeme_id),
                                "source": lexeme.get("status"),
                                "lift_id": lexeme.get("lift_id"),
                                "grammatical_infos": lexeme.get("grammatical_infos") or [],
                                "senses": senses})
        return {"results": results}


    # ---- direct read-only Turoyo verb parser ----
    @app.get("/api/verb-parser")
    def turoyo_verb_parser(q: str = "", limit: int = 12):
        """Return verb-parser suggestions for an arbitrary surface form.

        Read-only diagnostic/suggestion endpoint. It does not inspect or
        modify an editor token, so an unanalysed word from FLEXTEXT can be
        parsed without touching existing FLEx analyses.
        """
        surface = S.nfc(q).strip()

        if not surface:
            return {
                "surface": surface,
                "suggestions": [],
                "reason": "empty form",
            }

        parser = TuroyoVerbParser(glosser.analyzer)
        rows = parser.suggest(
            surface,
            limit=max(1, min(int(limit), 50)),
        )

        suggestions = [row.as_dict() for row in rows]

        return {
            "surface": surface,
            "suggestions": suggestions,
            "reason": (
                f"{len(suggestions)} suggestion(s)"
                if suggestions
                else "no current verb rule matched"
            ),
        }

    @app.get("/api/guide")
    def get_guide():
        text = G.read(root)
        return {"text": text, "html": G.render(text)}

    @app.patch("/api/guide")
    def set_guide(body: GuideText):
        text = G.save(root, body.text)
        return {"text": text, "html": G.render(text)}

    # ---- document lifecycle ----
    @app.post("/api/documents")
    def create_document(body: NewDocument):
        # translations arrive as a block whose lines pair with the text's lines
        translations = S.nfc(body.translation).splitlines()
        document = S.build_document(body.text, glosser, title=body.title,
                                    translations=[t.strip() for t in translations],
                                    case=inventory.apply_case)
        store.save(document)
        return {"document_id": document["document_id"]}

    def _import_partial_igt_docx(source: Path, title: str,
                                 *, display_name: str | None = None
                                 ) -> dict[str, Any]:
        """Shared by the local-path and browser-upload import routes below.

        Human IGT is imported verbatim as USER+locked; turns with no IGT are
        passed through the ordinary TextGlosser.
        """
        try:
            parsed = parse_igt(
                source,
                igt_normalizer,
                known_labels=(
                    set(inventory.registered())
                    | set(inventory.display_map)
                    | set(inventory.case_map.values())
                ),
                # The DOCX is the human authority here.  In particular, do not
                # collapse source NC1/NC2 or PST through pipeline aliases.
                preserve_source_gloss=True,
            )
            document, report = S.build_partial_igt_document(
                parsed,
                glosser,
                title=title,
                case=inventory.apply_case,
            )
        except S.SessionError as bad:
            raise HTTPException(400, str(bad))
        except (OSError, ValueError) as bad:
            raise HTTPException(400, f"cannot import {source.name}: {bad}")

        if display_name is not None:
            # An upload's `source` is a throwaway temp path; the report is
            # for a human, who should see the file they chose, not
            # /tmp/tmpXXXX.docx.
            report["source_file"] = display_name

        store.save(document)
        return {
            "document_id": document["document_id"],
            "title": document.get("title"),
            "report": report,
        }

    @app.post("/api/documents/import-partial-igt")
    def import_partial_igt(body: PartialIGTImportBody):
        """Import a DOCX in which some turns already carry human IGT, read
        from a path on the machine running the editor.

        Kept for the local, offline use of `cli.py edit` and
        `tests/unit/test_partial_igt_safety.py` — a path on the server means
        nothing to a browser talking to a hosted editor over the network,
        which is what `upload_partial_igt` below is for.
        """
        source = Path(body.path).expanduser()
        if not source.is_absolute():
            source = (root / source).resolve()
        else:
            source = source.resolve()
        if source.suffix.lower() != ".docx":
            raise HTTPException(400, "partial IGT import expects a .docx file")
        if not source.is_file():
            raise HTTPException(404, f"no such DOCX: {source}")
        return _import_partial_igt_docx(source, body.title or source.stem)

    @app.post("/api/documents/upload-partial-igt")
    def upload_partial_igt(file: UploadFile = File(...)):
        """Browser counterpart to `import_partial_igt`: the file travels in
        the request body instead of by a server-side path, since that's the
        only way a document reaches this editor once it's hosted rather than
        run locally. See docs/DEPLOY_RAILWAY_ROADMAP_RU.md, Stage 5.

        A plain `def`, not `async`: `_import_partial_igt_docx` below glosses
        every line, which for a real interview is tens of seconds of CPU —
        on an `async def` that runs on uvicorn's one event-loop thread and
        answers nothing else meanwhile, not even `/healthz`, for as long as
        the import takes. A plain `def` gets FastAPI's threadpool instead,
        the same as `import_partial_igt` already gets for the same reason.
        """
        name = file.filename or "upload.docx"
        if not name.lower().endswith(".docx"):
            raise HTTPException(400, "partial IGT import expects a .docx file")
        max_mb = MAX_PARTIAL_IGT_UPLOAD_BYTES // (1024 * 1024)
        if file.size is not None and file.size > MAX_PARTIAL_IGT_UPLOAD_BYTES:
            raise HTTPException(400, f"file too large (max {max_mb} MB)")
        # `.size` comes from a client-supplied Content-Length and isn't
        # guaranteed; this is the check that actually holds the limit.
        data = file.file.read()
        if len(data) > MAX_PARTIAL_IGT_UPLOAD_BYTES:
            raise HTTPException(400, f"file too large (max {max_mb} MB)")

        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        try:
            return _import_partial_igt_docx(tmp_path, Path(name).stem,
                                            display_name=name)
        finally:
            tmp_path.unlink(missing_ok=True)

    def _import_flextext_file(source: Path, *, display_name: str | None = None
                              ) -> dict[str, Any]:
        """Import all FLEx texts as editable sessions and rebuild corpus evidence."""
        name = display_name or source.name
        try:
            result = import_flextext_to_state(
                source, root, display_name=name,
                session_store=store, corpus_store=flex_store,
            )
            reload_corpus = getattr(glosser, "reload_corpus", None)
            if callable(reload_corpus):
                reload_corpus()
            if lift_store.manifest().get("entries"):
                result["report"]["lift_linking"] = backfill_editor_sessions(
                    root,
                    document_ids=[row["document_id"] for row in result.get("documents") or []],
                    session_store=store,
                )
            _refresh_flex_inventory()
            return result
        except (OSError, ValueError) as bad:
            raise HTTPException(400, f"cannot import {name}: {bad}")

    @app.post("/api/documents/import-flextext")
    def import_flextext(body: FlexTextImportBody):
        source = Path(body.path).expanduser()
        if not source.is_absolute():
            source = (root / source).resolve()
        else:
            source = source.resolve()
        if source.suffix.lower() not in {".flextext", ".xml"}:
            raise HTTPException(400, "FLEx import expects a .flextext or .xml file")
        if not source.is_file():
            raise HTTPException(404, f"no such FLEx file: {source}")
        if source.stat().st_size > MAX_FLEXTEXT_UPLOAD_BYTES:
            raise HTTPException(400, "file too large (max 100 MB)")
        return _import_flextext_file(source)

    @app.post("/api/documents/upload-flextext")
    def upload_flextext(file: UploadFile = File(...)):
        name = file.filename or "upload.flextext"
        if Path(name).suffix.lower() not in {".flextext", ".xml"}:
            raise HTTPException(400, "FLEx import expects a .flextext or .xml file")
        max_mb = MAX_FLEXTEXT_UPLOAD_BYTES // (1024 * 1024)
        if file.size is not None and file.size > MAX_FLEXTEXT_UPLOAD_BYTES:
            raise HTTPException(400, f"file too large (max {max_mb} MB)")
        tmp_path = None
        total = 0
        try:
            with tempfile.NamedTemporaryFile(suffix=Path(name).suffix or ".flextext",
                                              delete=False) as tmp:
                tmp_path = Path(tmp.name)
                while True:
                    chunk = file.file.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_FLEXTEXT_UPLOAD_BYTES:
                        raise HTTPException(400, f"file too large (max {max_mb} MB)")
                    tmp.write(chunk)
            return _import_flextext_file(tmp_path, display_name=name)
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

    @app.get("/api/flex-corpus")
    def flex_corpus_status():
        return {"profile": profile, **flex_store.manifest()}

    def _import_lift_file(source: Path, *, display_name: str | None = None,
                          ranges: Path | None = None,
                          ranges_display_name: str | None = None) -> dict[str, Any]:
        """Replace the FieldWorks source snapshot, then re-link corpus tokens.

        Author additions/senses live in separate append-only journals and are
        therefore untouched by replacing the LIFT source.
        """
        nonlocal glosser
        name = display_name or source.name
        try:
            report = lift_store.import_file(
                source, display_name=name, ranges=ranges,
                ranges_display_name=ranges_display_name)
            linking = backfill_editor_sessions(root, session_store=store)
            glosser = _make_glosser(root)
            if warm:
                _warm(glosser)
            rebuilds["n"] += 1
            return {"report": report, "linking": linking}
        except (OSError, ValueError) as bad:
            raise HTTPException(400, f"cannot import {name}: {bad}")

    @app.post("/api/documents/import-lift")
    def import_lift(body: LiftImportBody):
        source = Path(body.path).expanduser()
        source = ((root / source).resolve() if not source.is_absolute()
                  else source.resolve())
        if source.suffix.lower() != ".lift":
            raise HTTPException(400, "LIFT import expects a .lift file")
        if not source.is_file():
            raise HTTPException(404, f"no such LIFT file: {source}")
        if source.stat().st_size > MAX_LIFT_UPLOAD_BYTES:
            raise HTTPException(400, "file too large (max 50 MB)")
        ranges = None
        if body.ranges_path:
            ranges = Path(body.ranges_path).expanduser()
            ranges = ((root / ranges).resolve() if not ranges.is_absolute()
                      else ranges.resolve())
            if not ranges.is_file():
                raise HTTPException(404, f"no such LIFT ranges file: {ranges}")
        return _import_lift_file(source, ranges=ranges)

    @app.post("/api/documents/upload-lift")
    def upload_lift(file: UploadFile = File(...), ranges: UploadFile | None = File(None)):
        name = file.filename or "upload.lift"
        if Path(name).suffix.lower() != ".lift":
            raise HTTPException(400, "LIFT import expects a .lift file")
        max_mb = MAX_LIFT_UPLOAD_BYTES // (1024 * 1024)
        data = file.file.read(MAX_LIFT_UPLOAD_BYTES + 1)
        if len(data) > MAX_LIFT_UPLOAD_BYTES:
            raise HTTPException(400, f"file too large (max {max_mb} MB)")
        ranges_data = None
        ranges_name = None
        if ranges and ranges.filename:
            ranges_name = ranges.filename
            ranges_data = ranges.file.read(MAX_LIFT_UPLOAD_BYTES + 1)
            if len(ranges_data) > MAX_LIFT_UPLOAD_BYTES:
                raise HTTPException(400, f"ranges file too large (max {max_mb} MB)")
        lift_tmp = ranges_tmp = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".lift", delete=False) as tmp:
                tmp.write(data)
                lift_tmp = Path(tmp.name)
            if ranges_data is not None:
                with tempfile.NamedTemporaryFile(suffix=".lift-ranges", delete=False) as tmp:
                    tmp.write(ranges_data)
                    ranges_tmp = Path(tmp.name)
            return _import_lift_file(
                lift_tmp, display_name=name, ranges=ranges_tmp,
                ranges_display_name=ranges_name)
        finally:
            if lift_tmp is not None:
                lift_tmp.unlink(missing_ok=True)
            if ranges_tmp is not None:
                ranges_tmp.unlink(missing_ok=True)

    @app.get("/api/lift-lexicon")
    def lift_lexicon_status():
        return {"profile": profile, **lift_store.manifest()}

    @app.get("/api/documents/{document_id}")
    def get_document(document_id: str):
        try:
            return enrich(store.load(document_id))
        except FileNotFoundError:
            raise HTTPException(404, "no such document")

    @app.patch("/api/documents/{document_id}")
    def rename_document(document_id: str, body: DocumentTitleBody):
        try:
            document = store.load(document_id)
        except FileNotFoundError:
            raise HTTPException(404, "no such document")
        document["title"] = S.nfc(body.title).strip()
        store.save(document)
        return {"document_id": document_id, "title": document["title"]}

    @app.delete("/api/documents/{document_id}")
    def delete_document(document_id: str):
        """Trashes the session (SessionStore.trash), not `unlink`: a stray
        click on the session list should be recoverable, same reasoning as
        `renderDropLine` for one line inside a document."""
        try:
            store.trash(document_id)
        except FileNotFoundError:
            raise HTTPException(404, "no such document")
        history.forget(document_id)
        return {"trashed": True}

    # ---- editing one cell ----
    def _remember(token: dict) -> dict | None:
        """An explicitly reusable reading enters the curated surface layer.

        Same layer the label window writes to, so the fix reaches every text
        and survives a rebuild. It is registered in the live analyzer rather
        than rebuilt into it: rebuilding takes seconds, and this runs on every
        explicitly reusable edit. Nothing is recorded while a cell is still
        `??` — an unfinished word is not a reading.
        """
        reading = S.finished_reading(token)
        if not reading:
            return None
        found = S.resolve_reading(*reading, glosser.analyzer)
        try:
            row = L.record_correction(root, *reading,
                                      note="finished in the text editor",
                                      lemma_id=found.lexeme_id if found else None,
                                      original_surface=token.get("surface"))
        except ValueError:
            return None      # the cells no longer reconstruct the surface
        glosser.remember(row)
        rebuilds["n"] += 1   # the label window is looking at stale counts now
        return row

    @app.patch("/api/documents/{document_id}/morphemes/{morph_id}")
    def edit_cell(document_id: str, morph_id: str, body: CellEdit):
        nonlocal glosser
        document = store.load(document_id)
        target = host = None
        for sentence in document["sentences"]:
            for token in sentence["tokens"]:
                for morpheme in token["morphemes"]:
                    if morpheme["morph_id"] == morph_id:
                        target, host = morpheme, token
        if target is None:
            raise HTTPException(404, "no such morpheme")
        before = copy.deepcopy(host)
        gloss = S.morphology_text(body.gloss).strip()
        folded = inventory.canonical(gloss) if gloss else gloss
        alias = folded if folded != gloss else None
        gloss = folded
        if gloss == S.SPEAKER_GLOSS:
            S.mark_speaker(host, origin=S.USER)
            target = host["morphemes"][0]
        else:
            target["gloss"] = gloss
            target["suggested"] = ""
            target["origin"] = S.USER
            target["status"] = S.morpheme_status(gloss, "HIGH" if gloss else "LOW")
        selected_existing_sense = False
        if body.lexeme_id:
            record = glosser.analyzer.lexeme_record(body.lexeme_id)
            if not record:
                raise HTTPException(404, "no such lexical entry")
            sense = None
            lift_index = getattr(glosser.analyzer, "lift", None)
            lift_entry = lift_index.entry(body.lexeme_id) if lift_index is not None else None
            if lift_entry and body.sense_id:
                sense = next((row for row in lift_entry.get("senses") or []
                              if row.get("sense_id") == body.sense_id), None)
                if sense is None:
                    raise HTTPException(400, "that sense does not belong to this LIFT entry")
            grammatical_info = (body.grammatical_info
                                or (sense or {}).get("grammatical_info"))
            target["lexeme_id"] = body.lexeme_id
            target["lemma"] = record.get("lemma_canonical")
            target["grammatical_info"] = grammatical_info
            target["lift_entry_id"] = record.get("lift_id")
            target["lift_sense_id"] = (sense or {}).get("sense_id")

            # Only the lexical host owns the word-level lemma.  Bound-morph
            # LIFT entries can still be selected and retained on their own
            # cell without replacing the host lexeme of the word.
            stem, _head = RL.host_stem(host.get("morphemes") or [])
            if stem and S.nfc(target.get("form") or "") == S.nfc(stem):
                host["lemma_id"] = body.lexeme_id
                host["lemma"] = record.get("lemma_canonical")
                host["grammatical_infos"] = list(record.get("grammatical_infos") or [])
                host["grammatical_info"] = (grammatical_info
                                            or (record.get("grammatical_infos") or [None])[0]
                                            if len(record.get("grammatical_infos") or []) == 1
                                            else grammatical_info)
                host["lift_entry_id"] = record.get("lift_id")
                host["lift_guid"] = record.get("lift_guid")
                host["lift_sense_id"] = (sense or {}).get("sense_id")
                if sense and preferred_sense_gloss(sense):
                    host["lexical_gloss"] = preferred_sense_gloss(sense)
                elif GS.has_lexical_part(gloss):
                    host["lexical_gloss"] = GS.lexical_head(gloss)
                host.pop("lexeme_candidates", None)
            selected_existing_sense = bool(
                lift_entry and sense and preferred_sense_gloss(sense)
                and S.nfc(preferred_sense_gloss(sense)).casefold()
                == S.nfc(GS.lexical_head(gloss)).casefold())
        host["locked"] = True
        history.record(document_id, H.Entry("token", before, copy.deepcopy(host),
                                            token_id=host["token_id"]))
        taught = None
        should_teach_lexicon = (gloss and gloss != S.SPEAKER_GLOSS
                                and not selected_existing_sense
                                and (profile == "flex" or body.reusable))
        if should_teach_lexicon:
            try:
                taught = _teach_meaning_from_cell(document, host, target, before)
            except ValueError:
                taught = None
            if taught and taught["after"]:
                history.record(document_id,
                               H.Entry("reading_batch", taught["before"], taught["after"]))
        correction = _remember(host) if body.reusable and gloss != S.SPEAKER_GLOSS else None
        if correction:
            host["reusable_record"] = {key: correction[key] for key in
                                       ("addition_id", "surface_canonical", "segmentation", "gloss")}
            host["reusable_record"]["active"] = True
        store.save(document)
        if taught and taught.get("rebuild"):
            glosser = _make_glosser(root)
            _warm(glosser)
            rebuilds["n"] += 1
        return {"morpheme": target, "alias_applied": alias,
                "token": host, "correction": correction,
                "reusable": {"requested": body.reusable, "recorded": correction is not None},
                "lexical": ({"recorded": True, "gloss": taught["row"]["gloss"],
                             "updated": len(taught["after"])} if taught
                            else {"recorded": False,
                                  "selected_lift": selected_existing_sense}),
                "history": history.state(document_id)}

    @app.patch("/api/documents/{document_id}/reuse-default")
    def document_reuse_default(document_id: str, body: ReuseDefaultBody):
        """This text's own default for finished readings, over the project
        setting: one editor can keep «только здесь» for their text while
        the project default stays «для всех текстов»."""
        document = store.load(document_id)
        document["reuse_default"] = body.reusable_default
        store.save(document)
        return {"reuse_default": document["reuse_default"]}

    @app.post("/api/documents/{document_id}/tokens/{token_id}/lemma")
    def bind_token_lemma(document_id: str, token_id: str, body: TokenLemmaBody):
        """«Привязать к лемме…»: the author names the lexeme of this word.
        Morphology and glosses stay as they are; one undo step."""
        analyzer = glosser.analyzer
        record = analyzer.lexeme_record(body.lexeme_id)
        if not record:
            raise HTTPException(404, "no such lexeme")
        document = store.load(document_id)
        _, token = _find_token(document, token_id)
        before = copy.deepcopy(token)
        stem, head = RL.host_stem(token.get("morphemes") or [])
        gloss, _status = analyzer.lexical.resolve(body.lexeme_id)
        token["lemma_id"] = body.lexeme_id
        token["lemma"] = record.get("lemma_canonical")
        token["lexical_gloss"] = head or gloss or token.get("lexical_gloss")
        token["grammatical_infos"] = list(record.get("grammatical_infos") or [])
        token["grammatical_info"] = (record.get("grammatical_infos") or [None])[0] \
            if len(record.get("grammatical_infos") or []) == 1 else None
        token["lift_entry_id"] = record.get("lift_id")
        token["lift_guid"] = record.get("lift_guid")
        token["lift_sense_id"] = None
        token.pop("lexeme_candidates", None)
        token["relink_provenance"] = {"from": before.get("lemma_id"), "why": "выбрано автором"}
        if profile == "flex":
            # A manual link is also lexical evidence for this exact written
            # form. Persist it as a local overlay rather than modifying the
            # imported LIFT file. On the next occurrence (and after restart)
            # the form can therefore resolve to the same exact entry again.
            grammatical_info = token.get("grammatical_info")
            LA.extend_addition(
                root, body.lexeme_id, record.get("lemma_canonical") or token.get("lemma") or "",
                part_of_speech=(record.get("part_of_speech")
                                if record.get("part_of_speech") in LA.PARTS_OF_SPEECH else None),
                surface=token.get("surface") or "",
                title=document.get("title") or "",
                root_canonical=record.get("root_canonical"),
                grammatical_info=grammatical_info)
            remember_surface = getattr(analyzer, "remember_lexeme_surface", None)
            if callable(remember_surface):
                remember_surface(body.lexeme_id, token.get("surface") or "", grammatical_info)
            clear_cache = getattr(glosser, "_token_cache", None)
            if isinstance(clear_cache, dict):
                clear_cache.clear()
        links.annotate(document)
        history.record(document_id, H.Entry("token", before, copy.deepcopy(token),
                                            token_id=token_id))
        store.save(document)
        return {"token": token, "history": history.state(document_id)}

    @app.post("/api/documents/{document_id}/tokens/{token_id}/new-lemma")
    def new_token_lemma(document_id: str, token_id: str, body: NewLemmaBody):
        """«Новая лемма»: a lexeme the dictionary does not have yet, named by
        its citation form and root, with the meaning from the word's gloss;
        this word is bound to it. A citation form that already exists is a
        409 with the lexemes it belongs to — bind to one of them instead."""
        nonlocal glosser
        from maalula_glosser.pos import gloss_is_verbal
        lemma = S.nfc(body.lemma).strip()
        root_text = S.nfc(body.root).strip().lstrip("√").strip()
        meaning = S.nfc(body.gloss).strip()
        grammatical_info = S.nfc(body.grammatical_info).strip() or None
        if not lemma:
            raise HTTPException(400, "a citation form is required")
        analyzer = glosser.analyzer
        existing = [lexeme_id for lexeme_id, lexeme in analyzer.lexemes.items()
                    if lexeme_id and S.nfc(lexeme.get("lemma_canonical") or "").strip() == lemma]
        if existing:
            raise HTTPException(409, {
                "error": "this citation form already exists",
                "candidates": [{"lexeme_id": i, "lemma": lemma,
                                "root": analyzer.lexemes[i].get("root_canonical"),
                                "grammatical_infos": analyzer.lexemes[i].get("grammatical_infos") or [],
                                "lift_id": analyzer.lexemes[i].get("lift_id")}
                               for i in sorted(existing)]})
        document = store.load(document_id)
        _, token = _find_token(document, token_id)
        cells = token.get("morphemes") or []
        gloss_line = "".join((c.get("sep") or "") + (c.get("gloss") or "") for c in cells)
        current = token.get("lemma_id") or ""
        lexeme_id = (current if current.startswith("lexnew_") and current not in analyzer.lexemes
                     else S.stable_id("lexnew", lemma.casefold()))
        surface = analyzer.canonical(GS_fold(token.get("surface") or ""))
        owners = {form.get("lexeme_id") for form in analyzer.forms_by_surface.get(surface, [])
                  if form.get("lexeme_id") and form.get("lexeme_id") != lexeme_id}
        low_gi = (grammatical_info or "").casefold()
        category_is_verbal = any(marker in low_gi for marker in
                                 ("preterite", "infectum", "imperative", "verb"))
        try:
            addition = LA.record_addition(
                root, lexeme_id, lemma,
                part_of_speech="verb" if (category_is_verbal or gloss_is_verbal(gloss_line)) else None,
                surfaces=[] if owners or not surface else [{"form_canonical": surface, "frequency": 1}],
                titles=[document.get("title") or ""], root_canonical=root_text or None,
                grammatical_info=grammatical_info)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        if meaning:
            SE.record_lexical_correction(root, lexeme_id, lemma, meaning, "en")
        before = copy.deepcopy(token)
        token["lemma_id"] = lexeme_id
        token["lemma"] = lemma
        token["lexical_gloss"] = meaning or token.get("lexical_gloss")
        token["grammatical_info"] = grammatical_info
        token["grammatical_infos"] = [grammatical_info] if grammatical_info else []
        token["lift_entry_id"] = None
        token["lift_guid"] = None
        token["lift_sense_id"] = None
        token.pop("lexeme_candidates", None)
        token["relink_provenance"] = {"from": current or None, "why": "новая лемма автора"}
        # the lexicon changed (a new lexeme, maybe a root): full rebuild, as
        # in `lexicon_add`
        glosser = _make_glosser(root)
        _warm(glosser)
        rebuilds["n"] += 1
        links.annotate(document)
        history.record(document_id, H.Entry("token", before, copy.deepcopy(token),
                                            token_id=token_id))
        store.save(document)
        return {"token": token, "addition": addition, "history": history.state(document_id)}

    @app.patch("/api/documents/{document_id}/tokens/{token_id}/reuse-mode")
    def token_reuse_mode(document_id: str, token_id: str, body: ReuseMode):
        document = store.load(document_id)
        _, token = _find_token(document, token_id)
        token["reuse_mode"] = body.mode
        store.save(document)
        return {"token": token}

    @app.post("/api/documents/{document_id}/tokens/{token_id}/reusable/retract")
    def retract_token_rule(document_id: str, token_id: str):
        nonlocal glosser
        document = store.load(document_id)
        _, token = _find_token(document, token_id)
        record = token.get("reusable_record")
        if not record:
            raise HTTPException(400, "this token has no recorded reusable rule")
        result = L.retract_correction(root, record["addition_id"])
        record["active"] = False
        token["reuse_mode"] = "local"
        store.save(document)
        glosser = _make_glosser(root)
        _warm(glosser)
        rebuilds["n"] += 1
        return {"token": token, "retraction": result}

    # ---- lexical gloss: reusable by lexeme, not by surface ----
    #
    # Sibling of "reusable/retract" above, but for the MEANING the gloss
    # translates to rather than its segmentation — see
    # `docs/LEXICAL_REUSABLE_SPEC_RU.md`. Does not touch `similar_cells` or
    # `reusable_morphology.jsonl`: those stay exact-surface / morphology only.

    @app.get("/api/documents/{document_id}/lexeme-forms/{token_id}")
    def lexeme_forms(document_id: str, token_id: str):
        """Every attested form of this token's lexeme, for the confirm-time
        checklist — the lexical analogue of `/similar/{morph_id}`, but scoped
        to one lexeme across the whole corpus rather than to same-surface
        repeats inside this document."""
        document = store.load(document_id)
        _, token = _find_token(document, token_id)
        lexeme_id = token.get("lemma_id")
        if not lexeme_id:
            raise HTTPException(400, "this token has no lexeme identity yet")
        result = SE.sibling_forms(root, glosser.analyzer, lexeme_id, document=document)
        result["lexical_gloss"] = token.get("lexical_gloss")
        return result

    def _apply_meaning_in_text(document: dict, lexeme_id: str, meaning: str,
                               clicked: dict | None = None,
                               clicked_before: dict | None = None,
                               excluded: set | None = None) -> tuple[list, list, int]:
        """Every automatic word of the lexeme in this text gets `meaning` in
        its stem cell, and so does the word the author clicked. Hand-checked
        words keep what the author wrote there; they are only counted."""
        excluded = excluded or set()
        before, after, kept = [], [], 0
        for sentence in document.get("sentences") or []:
            for other in sentence.get("tokens") or []:
                if other.get("lemma_id") != lexeme_id:
                    continue
                if GS_fold(other.get("surface") or "") in excluded:
                    continue
                is_clicked = other is clicked
                cells = other.get("morphemes") or []
                if not is_clicked and any(c.get("origin") == S.USER for c in cells):
                    kept += 1
                    continue
                snapshot = clicked_before if is_clicked else copy.deepcopy(other)
                if _replace_lexical_head(cells, meaning) or is_clicked:
                    other["lexical_gloss"] = meaning
                    before.append(snapshot)
                    after.append(copy.deepcopy(other))
        return before, after, kept

    def _teach_meaning_from_cell(document: dict, token: dict, cell: dict,
                                 before_token: dict) -> dict | None:
        """Teach a lexical sense from a direct edit of the host-stem cell.

        In the FLEX/Turoyo profile this is deliberately polysemous: every
        lexical value the author types becomes a sense of the lemma, including
        a second or third sense of a lemma that already has one.  If the word
        came from a new raw text and has no lexeme yet, the edit also promotes
        it into the author lexicon immediately.  This is independent of the
        whole-word ``reusable`` switch: that switch still controls morphology,
        while the lexicon records lexical knowledge as soon as the author has
        explicitly entered it.

        The legacy Maalula profile keeps its former single-meaning behaviour.
        """
        new_head = GS.lexical_head(cell.get("gloss") or "")
        if not new_head or not GS.has_lexical_part(new_head):
            return None

        if profile == "flex":
            # Every explicitly segmented component may have its own lexical
            # identity.  An orthographic token is NOT restricted to one lemma:
            #
            #     b=u+televizyon
            #     b          -> lemma b
            #     =u         -> lemma u
            #     +televizyon -> lemma televizyon
            #
            # Purely grammatical labels are already rejected above by
            # GS.has_lexical_part().
            form = S.nfc(cell.get("form") or "").strip()
            if not form:
                return None

            cells = token.get("morphemes") or []
            stem, _existing_head = RL.host_stem(cells)
            is_host = bool(
                stem and S.nfc(stem).strip() == form
            )

            # Prefer an already explicit cell lemma. For the lexical host the
            # old word-level lemma is still useful as a fallback. Other
            # components default to their own written form.
            lemma = S.nfc(cell.get("lemma") or "").strip()

            if not lemma and is_host:
                word_lemma = S.nfc(token.get("lemma") or "").strip()
                if word_lemma and GS.looks_like_a_lemma(word_lemma):
                    lemma = word_lemma
                else:
                    lemma = (
                        S.nominal_citation(cells, stem)
                        or form
                    )

            if not lemma:
                lemma = form

            if not lemma:
                return None

            # A citation form copied into the gloss slot is not a meaning.
            if new_head.casefold() == lemma.casefold():
                return None

            analyzer = glosser.analyzer

            # Lexical identity now belongs first of all to this particular
            # morpheme, not to the complete orthographic token.
            lexeme_id = cell.get("lexeme_id")
            lift_entry = None
            lift_sense = None

            if (
                not lexeme_id
                and getattr(analyzer, "lift", None)
                and analyzer.lift.has_entries()
            ):
                lift_entry, lift_sense, options = analyzer.lift.match(
                    form,
                    gloss=new_head,
                )

                if lift_entry:
                    lexeme_id = lift_entry.get("lexeme_id")
                    lemma = lift_entry.get("lemma_canonical") or lemma

                elif options:
                    # Several real FieldWorks entries fit. Never silently
                    # choose one or mint a duplicate lexeme.
                    cell["lexeme_candidates"] = list(dict.fromkeys(
                        option.get("lexeme_id")
                        for option in options
                        if option.get("lexeme_id")
                    ))
                    return None

            if not lexeme_id:
                candidates = [
                    lid
                    for lid, record in analyzer.lexemes.items()
                    if record.get("status") != "lift"
                    and S.nfc(
                        record.get("lemma_canonical") or ""
                    ).casefold() == lemma.casefold()
                ]

                matching = []
                for lid in candidates:
                    existing, _status = analyzer.lexical.resolve(lid)
                    if (
                        existing
                        and S.nfc(existing).casefold()
                        == new_head.casefold()
                    ):
                        matching.append(lid)

                lexeme_id = (
                    matching[0]
                    if len(matching) == 1
                    else S.stable_id(
                        "lexnew",
                        lemma.casefold(),
                    )
                )

            record = analyzer.lexeme_record(lexeme_id)
            canonical_lemma = (
                (record or {}).get("lemma_canonical")
                or lemma
            )

            # Store lexical identity directly on this cell.
            cell["lexeme_id"] = lexeme_id
            cell["lemma"] = canonical_lemma
            cell.pop("lexeme_candidates", None)

            if record:
                gi = (
                    (lift_sense or {}).get("grammatical_info")
                    or analyzer.grammatical_info(
                        lexeme_id,
                        new_head,
                    )
                )

                cell["grammatical_info"] = (
                    gi or cell.get("grammatical_info")
                )
                cell["lift_entry_id"] = record.get("lift_id")
                cell["lift_sense_id"] = (
                    (lift_sense or {}).get("sense_id")
                )

            # Keep word-level metadata only as a compatibility mirror for the
            # true lexical host. Bound components no longer overwrite it.
            if is_host:
                token["lemma_id"] = lexeme_id
                token["lemma"] = canonical_lemma
                token["lexical_gloss"] = new_head

                if record:
                    token["grammatical_infos"] = list(
                        record.get("grammatical_infos") or []
                    )
                    token["grammatical_info"] = (
                        cell.get("grammatical_info")
                    )
                    token["lift_entry_id"] = record.get("lift_id")
                    token["lift_guid"] = record.get("lift_guid")
                    token["lift_sense_id"] = (
                        (lift_sense or {}).get("sense_id")
                    )

                token.pop("lexeme_candidates", None)

            addition = None

            if not record:
                addition = LA.extend_addition(
                    root,
                    lexeme_id,
                    canonical_lemma,
                    # Record the component itself as the lexical form,
                    # not the entire orthographic token.
                    surface=form,
                    title=document.get("title") or "",
                    grammatical_info=cell.get(
                        "grammatical_info"
                    ),
                )

            row = SE.record_lexical_correction(
                root,
                lexeme_id,
                canonical_lemma,
                new_head,
                "en",
                stem_form=form,
                note="added while glossing a FLEX/Turoyo morpheme",
            )

            lexical_record = {
                key: row[key]
                for key in (
                    "addition_id",
                    "lexeme_id",
                    "lemma_canonical",
                    "gloss",
                )
            }
            lexical_record["active"] = True

            # Each component owns its own lexical record.
            cell["lexical_record"] = lexical_record

            # Preserve the old token-level mirror only for the host.
            if is_host:
                token["lexical_record"] = dict(lexical_record)

            if addition is None:
                glosser.remember_lexical(row)

            remember_cell = getattr(
                glosser,
                "remember_morpheme_gloss",
                None,
            )
            if callable(remember_cell):
                remember_cell(form, new_head)

            return {
                "row": row,
                "before": [],
                "after": [],
                "addition": addition,
                "rebuild": addition is not None,
            }

        # Legacy Maalula behaviour: a direct edit teaches the single lexical
        # meaning only while the lexeme has none yet.
        lexeme_id = token.get("lemma_id")
        if not lexeme_id:
            return None
        stem, _head = RL.host_stem(before_token.get("morphemes") or [])
        if stem and S.nfc(cell.get("form") or "") != stem:
            return None
        old_cell = next((c for c in before_token.get("morphemes") or []
                         if c.get("morph_id") == cell.get("morph_id")), {})
        old_head = GS.lexical_head(old_cell.get("gloss") or old_cell.get("suggested") or "")
        lemma = S.nfc(token.get("lemma") or "")
        lemma_words = {w for w in re.split(r"[\s/()]+", lemma.lower()) if w}
        if old_head and old_head.lower() not in lemma_words and old_head.lower() != lemma.lower():
            return None
        if new_head.lower() in lemma_words:
            return None
        current, status = glosser.analyzer.lexical.resolve(lexeme_id)
        if current and status not in ("translation_pending", "no_sense", "unknown_lexeme"):
            return None
        row = SE.record_lexical_correction(root, lexeme_id, token.get("lemma"),
                                           new_head, "en")
        glosser.remember_lexical(row)
        token["lexical_record"] = {key: row[key] for key in
                                   ("addition_id", "lexeme_id", "lemma_canonical", "gloss")}
        token["lexical_record"]["active"] = True
        token["lexical_gloss"] = new_head
        before, after, _kept = _apply_meaning_in_text(document, lexeme_id, new_head,
                                                      clicked=None)
        return {"row": row, "before": before, "after": after,
                "addition": None, "rebuild": False}

    @app.post("/api/documents/{document_id}/tokens/{token_id}/lexical/confirm")
    def confirm_lexical(document_id: str, token_id: str, body: LexicalConfirmBody):
        """Confirm a lexeme's meaning; it applies to every form of the lexeme
        from the next token glossed on, not just this one."""
        nonlocal glosser
        document = store.load(document_id)
        _, token = _find_token(document, token_id)
        lexeme_id = token.get("lemma_id")
        try:
            row = SE.record_lexical_correction(
                root, lexeme_id, token.get("lemma"), body.gloss,
                body.gloss_language, confirmed=body.confirmed,
                excluded=body.excluded, note=body.note)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        glosser.remember_lexical(row)
        before_clicked = copy.deepcopy(token)
        token["lexical_record"] = {key: row[key] for key in
                                   ("addition_id", "lexeme_id", "lemma_canonical", "gloss")}
        token["lexical_record"]["active"] = True
        token["lexical_gloss"] = row["gloss"]
        # This text follows at once (see `_apply_meaning_in_text`).
        excluded = {GS_fold(x) for x in body.excluded}
        before, after, kept = _apply_meaning_in_text(
            document, lexeme_id, row["gloss"], clicked=token,
            clicked_before=before_clicked, excluded=excluded)
        if after:
            history.record(document_id, H.Entry("reading_batch", before, after))
        store.save(document)
        return {"token": token, "correction": row, "updated": len(after),
                "kept_checked": kept, "history": history.state(document_id)}

    @app.post("/api/documents/{document_id}/tokens/{token_id}/lexical/retract")
    def retract_lexical(document_id: str, token_id: str):
        nonlocal glosser
        document = store.load(document_id)
        _, token = _find_token(document, token_id)
        record = token.get("lexical_record")
        if not record:
            raise HTTPException(400, "this token has no recorded lexical rule")
        result = SE.retract_lexical_correction(root, record["addition_id"])
        record["active"] = False
        store.save(document)
        glosser = _make_glosser(root)
        _warm(glosser)
        rebuilds["n"] += 1
        return {"token": token, "retraction": result}

    @app.patch("/api/documents/{document_id}/tokens/{token_id}/segmentation")
    def edit_segmentation(document_id: str, token_id: str, body: SegmentationEdit):
        """Line 2, edited by hand: fix a morpheme or move a boundary.

        Line 1 is deliberately left alone. The transcription and the morphology
        are allowed to disagree — that is what this endpoint is for.
        """
        document = store.load(document_id)
        _sentence, token = _find_token(document, token_id)
        before = copy.deepcopy(token)
        # Asked once: this is the expensive part of the request, and it was
        # being run twice — six seconds became twelve on a word the analyzer
        # has to work at.
        proposed_options, analysis = S.propose_label_options(body.text, glosser)
        proposed_options = [[inventory.apply_case(label) for label in choices]
                            for choices in proposed_options]
        proposed = [choices[0] if choices else "" for choices in proposed_options]
        try:
            token["morphemes"] = S.resegment(
                token, body.text, proposed=proposed,
                proposed_options=proposed_options)
        except S.SessionError as bad:
            raise HTTPException(400, str(bad))
        # A hand-cut word can reach a lexeme the whole form never did —
        # `mḥammṣille` is unknown, `mḥammṣ-in` is ḥammeṣ — and the dictionary
        # link hangs off the lemma, so it is picked up here or it never appears.
        if analysis and analysis.get("lemma") and not token.get("lemma"):
            for field in ("lemma", "lemma_id", "lexical_gloss"):
                token[field] = analysis.get(field)
        token["locked"] = True
        links.annotate(document)
        history.record(document_id, H.Entry("token", before, copy.deepcopy(token),
                                            token_id=token_id))
        store.save(document)
        return {"token": token,
                "history": history.state(document_id)}

    @app.post("/api/documents/{document_id}/tokens/{token_id}/resolve-alternative")
    def resolve_alternative(document_id: str, token_id: str, body: ResolveAlternativeBody):
        """An author picks one of a token's `alternatives` — several
        competing segmentation+gloss readings the resolver refused to
        auto-guess between (invariant #7) — and it actually resolves.

        No rebuild afterward: `LR.resolve_ambiguous_occurrences` (via
        `session.resolve_ambiguous_token`) only ever resolves to a lexeme
        already in `glosser.analyzer.lexemes`, exactly like `lexicon_
        reassign` above — it never mints, so nothing analyzer-visible
        changes, only the touched saved documents.

        Only the primary document/token is re-annotated here (`links.
        annotate` stamps a per-token `dictionary` field derived from
        `lemma`, which this action can change): that matches every other
        per-token endpoint in this file (`edit_segmentation` etc.), the
        convention this endpoint otherwise follows throughout. Documents
        touched only by propagation are not re-annotated — that matches
        `lexicon_reassign`/`lexicon_merge`/`lexicon_delete` above, none of
        which re-annotate the (potentially many) documents their own
        multi-document token rewrites touch either; this endpoint's
        propagation walk is the same category of action as those, not the
        single-token category `edit_segmentation` is.
        """
        try:
            document = store.load(document_id)
        except FileNotFoundError:
            raise HTTPException(404, "no such document")
        _sentence, token = _find_token(document, token_id)
        before = copy.deepcopy(token)
        try:
            report = LR.resolve_ambiguous_occurrences(
                store, document_id, token_id, body.alternative_index, glosser.analyzer)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        reloaded = store.load(document_id)
        links.annotate(reloaded)
        store.save(reloaded)
        _, after_token = _find_token(reloaded, token_id)
        history.record(document_id, H.Entry("token", before, copy.deepcopy(after_token),
                                            token_id=token_id))
        return {"token": after_token, "history": history.state(document_id),
                "propagation": report}

    def _rearrange(document_id: str, token_id: str, change, operation: str,
                   count: int = 1, joiner: str = "="):
        """Run a change that alters how many columns a block has."""
        document = store.load(document_id)
        sentence, _token = _find_token(document, token_id)
        before = copy.deepcopy(sentence["tokens"])
        start = S._token_index(sentence, token_id)
        source_before = copy.deepcopy(sentence["tokens"][start:start + count])
        try:
            changed = change(sentence)
        except S.SessionError as bad:
            raise HTTPException(400, str(bad))
        links.annotate(document)
        history.record(document_id,
                       H.Entry("tokens", before, copy.deepcopy(sentence["tokens"]),
                               sentence_id=sentence["sentence_id"]))
        S.normalize_morphology(document)
        proposal = S.topology_offer(document, source_before,
                                    changed if isinstance(changed, list) else [changed],
                                    operation, joiner)
        store.save(document)
        return {"sentence_id": sentence["sentence_id"], "tokens": sentence["tokens"],
                "topology": proposal, "history": history.state(document_id)}

    @app.post("/api/documents/{document_id}/tokens/{token_id}/merge")
    def merge_columns(document_id: str, token_id: str, body: MergeBody):
        """Join this column with the ones after it — `la` to its host, or a run
        of words into one Arabic insert."""
        return _rearrange(document_id, token_id, lambda sentence: S.merge_tokens(
            sentence, token_id, body.count, body.joiner), "merge", body.count, body.joiner)

    @app.post("/api/documents/{document_id}/tokens/{token_id}/split")
    def split_column(document_id: str, token_id: str):
        """Break a column back into one column per written word, re-glossed."""
        return _rearrange(document_id, token_id, lambda sentence: S.split_token(
            sentence, token_id, glosser, inventory.apply_case), "split")

    @app.post("/api/documents/{document_id}/apply-topology")
    def apply_topology(document_id: str, body: BulkTopologyBody):
        document = store.load(document_id)
        try:
            before, after = S.apply_topologies(document, body.proposal_id, body.targets,
                                              glosser, inventory.apply_case)
        except S.SessionError as bad:
            raise HTTPException(409, str(bad))
        if after:
            history.record(document_id, H.Entry("topology_batch", before, after))
            store.save(document)
        return {"history": history.state(document_id)}

    @app.get("/api/documents/{document_id}/tokens/{token_id}/similar-reading")
    def similar_reading(document_id: str, token_id: str):
        document = store.load(document_id)
        _, source = _find_token(document, token_id)
        return S.similar_readings(document, source)

    @app.post("/api/documents/{document_id}/apply-reading")
    def apply_reading(document_id: str, body: BulkReadingBody):
        document = store.load(document_id)
        _, source = _find_token(document, body.source_id)
        offer = S.similar_readings(document, source)
        eligible = {match["token_id"]: match for match in offer["matches"]}
        if (body.source_version != offer["source_version"]
                or any(key not in eligible or eligible[key]["version"] != version
                       for key, version in body.targets.items())):
            raise HTTPException(409, "The reading changed; reopen the proposal.")
        before, after = [], []
        for sentence in document.get("sentences") or []:
            for token in sentence.get("tokens") or []:
                if token["token_id"] not in body.targets:
                    continue
                before.append(copy.deepcopy(token))
                S.apply_reading(sentence, token, source)
                after.append(copy.deepcopy(token))
        if after:
            history.record(document_id, H.Entry("reading_batch", before, after))
            store.save(document)
        return {"applied": len(after), "history": history.state(document_id)}

    # ---- «Перепривязать леммы»: preview (GET) and apply (POST) ----
    def _relink_plan(document: dict[str, Any]) -> tuple[RL.Relinker, list[dict[str, Any]]]:
        relinker = RL.Relinker(glosser.analyzer, RL.merges_of(root))
        return relinker, relinker.plan(document)

    @app.get("/api/documents/{document_id}/relink")
    def relink_preview(document_id: str):
        """What the button would change; nothing is written."""
        _relinker, changes = _relink_plan(store.load(document_id))
        return {"changes": changes, "counts": RL.summarize(changes),
                "labels": RL.ACTION_LABELS}

    @app.post("/api/documents/{document_id}/relink")
    def relink_apply(document_id: str, body: RelinkBody):
        """Recompute the plan (the document may have moved on since the
        preview), apply the author's picks, write it all as one undo step."""
        document = store.load(document_id)
        relinker, changes = _relink_plan(document)
        skip = set(body.skip)
        kept = []
        for change in changes:
            if change["token_id"] in skip:
                continue
            picked = body.choices.get(change["token_id"])
            if picked:
                if picked not in {c["lexeme_id"] for c in change["candidates"]}:
                    raise HTTPException(409, "The candidates changed; reopen the preview.")
                RL.choose(relinker, change, picked)
            elif change["action"] == "choose":
                # no pick: record the candidates only, the lemma stays as it is
                pass
            kept.append(change)
        before, after = RL.apply(document, kept)
        if after:
            history.record(document_id, H.Entry("reading_batch", before, after))
            store.save(document)
        return {"applied": len(after), "counts": RL.summarize(kept),
                "history": history.state(document_id)}

    @app.get("/api/documents/{document_id}/similar/{morph_id}")
    def similar_cells(document_id: str, morph_id: str):
        """Where else this morpheme is spelled the same, with its line."""
        return S.similar_cells(store.load(document_id), morph_id)

    @app.post("/api/documents/{document_id}/apply-gloss")
    def apply_gloss(document_id: str, body: BulkGlossBody):
        """Write one label into the cells the author ticked.

        Recorded as a single entry, so undoing a batch of sixty is one Ctrl+Z.
        """
        document = store.load(document_id)
        wanted = set(body.morph_ids)
        gloss = inventory.canonical(S.nfc(body.gloss).strip())
        before, after = [], []
        for sentence in document.get("sentences") or []:
            for token in sentence.get("tokens") or []:
                for morpheme in token.get("morphemes") or []:
                    if morpheme.get("morph_id") not in wanted:
                        continue
                    before.append(dict(morpheme))
                    morpheme["gloss"] = gloss
                    morpheme["suggested"] = ""
                    morpheme["origin"] = S.USER
                    morpheme["status"] = S.morpheme_status(
                        gloss, "HIGH" if gloss else "LOW")
                    after.append(dict(morpheme))
                    token["locked"] = True
        if not after:
            raise HTTPException(404, "none of those cells are in this document")
        history.record(document_id, H.Entry("cells", before, after))
        store.save(document)
        return {"applied": len(after), "morphemes": after,
                "history": history.state(document_id)}

    # ---- autocomplete + hover alternatives ----
    @app.get("/api/documents/{document_id}/suggest")
    def suggest(document_id: str, prefix: str = "", morph_id: str = "",
                all_morpheme: bool = False):
        """Autocomplete a gloss cell.

        In the FLEX profile an exact morpheme form has its own corpus
        dictionary.  Those attested meanings come first (with frequencies);
        the general grammatical-label inventory remains the fallback.  The UI
        is still a free-text input, so this endpoint never limits the author to
        the returned values.
        """
        document = store.load(document_id)
        corpus_options: list[dict[str, Any]] = []
        if profile == "flex" and morph_id:
            found = None
            found_token = None
            for sentence in document.get("sentences") or []:
                for token in sentence.get("tokens") or []:
                    for morpheme in token.get("morphemes") or []:
                        if morpheme.get("morph_id") == morph_id:
                            found = morpheme
                            found_token = token
                            break
                    if found is not None:
                        break
                if found is not None:
                    break
            if found is None:
                raise HTTPException(404, f"morpheme {morph_id} not found")
            getter = getattr(glosser.analyzer, "morpheme_gloss_options", None)
            if callable(getter):
                morph_prefix = "" if all_morpheme else prefix
                for row in getter(found.get("form") or "", prefix=morph_prefix):
                    count = int(row.get("count") or 0)
                    lift_options = row.get("lift_options") or []
                    if lift_options:
                        for option in lift_options:
                            gi = option.get("grammatical_info") or ""
                            lemma = option.get("lemma") or ""
                            details = ["FieldWorks", lemma, gi]
                            if count:
                                details.append(f"корпус ×{count}")
                            corpus_options.append({
                                "label": row.get("gloss") or "",
                                "note": " · ".join(x for x in details if x),
                                "source": "lift",
                                "count": count,
                                "lexeme_id": option.get("lexeme_id"),
                                "sense_id": option.get("sense_id"),
                                "lemma": lemma,
                                "grammatical_info": gi or None,
                                "lift_entry_id": option.get("lift_id"),
                            })
                        # A LIFT sense already represents the same string; do
                        # not add a second anonymous FLEx row for it below.
                        continue
                    if row.get("author"):
                        note = "ваш reusable-вариант"
                        if count:
                            note += f" · во FLEx ×{count}"
                    else:
                        note = f"словарь FLEx · ×{count}" if count else "словарь FLEx"
                    kinds = [kind for kind in (row.get("types") or []) if kind != "unknown"]
                    if kinds:
                        note += " · " + ", ".join(kinds[:3])
                    corpus_options.append({
                        "label": row.get("gloss") or "",
                        "note": note,
                        "source": row.get("source") or "flextext",
                        "count": count,
                    })

            # If an unknown written form was manually linked to a FieldWorks
            # entry, the cell still needs that ENTRY'S senses even though the
            # form itself is not yet a citation/variant spelling in LIFT.
            # A token-level lemma belongs only to the lexical host cell.
            # Never inject that entry's senses into every clitic/affix cell.
            #
            # Example:
            #   b =a +dḳ -oṭ =ani
            #
            # Editing `b` must not offer `prox.p` merely because the whole
            # token is linked to the demonstrative entry used by `=ani`.
            stem_form, _stem_head = RL.host_stem(
                (found_token or {}).get("morphemes") or []
            )
            is_host_cell = bool(
                stem_form
                and S.nfc(found.get("form") or "") == S.nfc(stem_form)
            )

            bound_lexeme_id = (
                (found_token or {}).get("lemma_id")
                if is_host_cell
                else None
            )

            lift_index = getattr(glosser.analyzer, "lift", None)
            bound_entry = (
                lift_index.entry(bound_lexeme_id)
                if lift_index and bound_lexeme_id
                else None
            )

            if bound_entry:
                prefix_key = S.nfc("" if all_morpheme else prefix).strip().casefold()
                seen_lift = {(row.get("lexeme_id"), row.get("sense_id"))
                             for row in corpus_options if row.get("source") == "lift"}
                for sense in bound_entry.get("senses") or []:
                    gloss = preferred_sense_gloss(sense) or ""
                    if not gloss or (prefix_key and not gloss.casefold().startswith(prefix_key)):
                        continue
                    key = (bound_lexeme_id, sense.get("sense_id"))
                    if key in seen_lift:
                        continue
                    gi = sense.get("grammatical_info")
                    corpus_options.insert(0, {
                        "label": gloss,
                        "note": " · ".join(x for x in
                            ["FieldWorks", bound_entry.get("lemma_canonical") or "", gi or "",
                             "привязано к этой entry"] if x),
                        "source": "lift",
                        "count": 0,
                        "lexeme_id": bound_lexeme_id,
                        "sense_id": sense.get("sense_id"),
                        "lemma": bound_entry.get("lemma_canonical"),
                        "grammatical_info": gi,
                        "lift_entry_id": bound_entry.get("lift_id"),
                    })
                    seen_lift.add(key)

        # With an untouched cell, exact dictionary choices are the useful
        # first view.  Do not bury them under the whole label inventory. Once
        # the user starts typing, matching general labels join the list.
        extra = L.document_labels(document, inventory)
        matches = [] if (corpus_options and all_morpheme) else inventory.suggest(prefix, extra=extra)
        seen = {row["label"] for row in corpus_options if row.get("label")}
        labels = list(corpus_options)
        for match in matches:
            if match in seen:
                continue
            labels.append({"label": match, "note": inventory.definition_for(match),
                           "source": "inventory"})
        return {"labels": labels, "morpheme_options": corpus_options}

    # ---- conservative Turoyo verb parser ---------------------------------
    # This is a suggestion layer only.  It has two independent guards:
    # ``TuroyoVerbParser.eligible`` refuses any imported/manual analysis, and
    # the POST route recomputes the candidate before applying it.  Thus an
    # analysed FLEx token cannot be changed even if a client calls the endpoint
    # directly instead of using the UI.
    @app.get("/api/documents/{document_id}/tokens/{token_id}/verb-suggestions")
    def verb_suggestions(document_id: str, token_id: str, limit: int = 12):
        document = store.load(document_id)
        _sentence, token = _find_token(document, token_id)
        return _verb_parser().suggestions_for_token(
            token, limit=max(1, min(limit, 30)))

    @app.post("/api/documents/{document_id}/tokens/{token_id}/verb-suggestions/accept")
    def accept_verb_suggestion(document_id: str, token_id: str,
                               body: VerbSuggestionAcceptBody):
        document = store.load(document_id)
        _sentence, token = _find_token(document, token_id)
        parser = _verb_parser()
        eligible, reason = parser.eligible(token)
        if not eligible:
            raise HTTPException(409, reason)

        # Recompute from the current token and current corpus index.  The
        # browser never gets to submit arbitrary morphology under the name of
        # a parser suggestion, and a stale suggestion cannot overwrite a token
        # that was edited in another action meanwhile.
        candidates = parser.suggest(token.get("surface") or "", limit=100)
        candidate = next((row for row in candidates
                          if row.suggestion_id == body.suggestion_id), None)
        if candidate is None:
            raise HTTPException(409, "Эта подсказка устарела; откройте список заново.")

        before = copy.deepcopy(token)
        cells = S.build_morphemes(
            token.get("token_id") or token_id,
            candidate.segmentation,
            candidate.gloss,
            token.get("surface") or "",
            "HIGH",
            case=inventory.apply_case,
        )
        # An explicit click by the editor is a human decision.  It is local to
        # this token: no reusable rule and no FLEx/LIFT source data are changed.
        for cell in cells:
            cell["origin"] = S.USER
            cell["suggested"] = ""
            cell["status"] = S.CONFIDENT if cell.get("gloss") else S.UNRESOLVED
            cell["alternatives"] = []
        token["morphemes"] = cells
        token["locked"] = True
        token["confidence"] = "HIGH"
        token["tier"] = "TUROYO_VERB_RULE+USER"
        token["uncertainty"] = []
        token["verb_parser_acceptance"] = {
            "suggestion_id": candidate.suggestion_id,
            "rule": candidate.rule,
            "rule_label": candidate.rule_label,
            "evidence_stem": candidate.evidence_stem,
            "evidence_gloss": candidate.evidence_gloss,
            "evidence_match": candidate.evidence_match,
            "accepted": S.now(),
        }
        history.record(document_id, H.Entry(
            "token", before, copy.deepcopy(token), token_id=token_id))
        store.save(document)
        return {
            "token": token,
            "accepted": candidate.as_dict(),
            "history": history.state(document_id),
        }

    def _ritter_hints(token: dict[str, Any], *, limit: int = 8) -> dict[str, Any]:
        """Ritter candidates for one token, without changing its analysis.

        The lookup is deliberately conservative: exact spelling first; then
        forms whose consonant skeleton is IDENTICAL and whose vowels may
        differ.  We query the written word, the lexical stem (when morphology
        has already isolated one), and the current lemma.  Repeated dictionary
        rows are collapsed, keeping the strongest/most direct hit.
        """
        if ritter_dictionary is None:
            return {"available": False, "queries": [], "results": []}

        requested: list[tuple[str, str]] = []
        surface = S.nfc(token.get("surface") or "").strip()
        normalized = S.nfc(token.get("normalized") or "").strip()
        stem, _head = RL.host_stem(token.get("morphemes") or [])
        lemma = S.nfc(token.get("lemma") or "").strip()

        for kind, value in (
            ("word", surface),
            ("normalized", normalized),
            ("stem", S.nfc(stem or "").strip()),
            ("lemma", lemma),
        ):
            if not value:
                continue
            folded = value.casefold()
            if any(previous.casefold() == folded for _kind, previous in requested):
                continue
            requested.append((kind, value))

        kind_rank = {"word": 0, "normalized": 1, "stem": 2, "lemma": 3}
        best: dict[tuple[str, str], dict[str, Any]] = {}
        # Ask for more per query, then collapse duplicates across the queries.
        per_query = max(16, min(60, limit * 4))
        for query_kind, query in requested:
            for match in ritter_dictionary.lookup(query, limit=per_query):
                row = match.as_dict()
                row["query"] = query
                row["query_kind"] = query_kind
                row["query_skeleton"] = consonant_skeleton(query)
                key = (str(row.get("entry_id") or ""),
                       str(row.get("matched_form") or ""))
                score = (
                    0 if row.get("kind") == "exact" else 1,
                    kind_rank.get(query_kind, 9),
                    int(row.get("vowel_distance") or 0),
                    int(row.get("line_number") or 0),
                )
                previous = best.get(key)
                if previous is None or score < previous["_score"]:
                    row["_score"] = score
                    best[key] = row

        rows = sorted(best.values(), key=lambda row: row["_score"])[:max(1, min(limit, 30))]
        for row in rows:
            row.pop("_score", None)
        return {
            "available": True,
            "queries": [
                {"kind": kind, "form": value,
                 "skeleton": consonant_skeleton(value)}
                for kind, value in requested
            ],
            "results": rows,
        }

    @app.get("/api/documents/{document_id}/tokens/{token_id}/ritter")
    def ritter_hints(document_id: str, token_id: str, limit: int = 12):
        document = store.load(document_id)
        _sentence, token = _find_token(document, token_id)
        return _ritter_hints(token, limit=max(1, min(limit, 30)))

    @app.get("/api/documents/{document_id}/tokens/{token_id}/alternatives")
    def alternatives(document_id: str, token_id: str):
        document = store.load(document_id)
        _sentence, token = _find_token(document, token_id)
        return {"alternatives": token.get("alternatives") or [],
                "dictionary": links.link_for(token.get("lemma"),
                                              token.get("normalized")),
                "ritter": _ritter_hints(token, limit=5)}

    @app.post("/api/documents/{document_id}/tokens/{token_id}/approve-yellow")
    def approve_yellow(document_id: str, token_id: str):
        """Explicitly accept only the currently proposed/uncertain cells
        of one token.

        Red unresolved cells without a proposal are deliberately untouched.
        This is a local human confirmation; it does not create a reusable rule
        and does not modify FLEx/LIFT source data.
        """
        document = store.load(document_id)
        _sentence, token = _find_token(document, token_id)
        before = copy.deepcopy(token)
        changed = 0

        for cell in token.get("morphemes") or []:
            gloss = S.nfc(cell.get("gloss") or "").strip()
            suggested = S.nfc(cell.get("suggested") or "").strip()
            status = cell.get("status")

            # A visible proposal with no accepted gloss.
            if not gloss and suggested:
                gloss = inventory.canonical(suggested)
                cell["gloss"] = gloss
                cell["suggested"] = ""
                cell["origin"] = S.USER
                cell["status"] = S.CONFIDENT
                cell["alternatives"] = []
                changed += 1
                continue

            # Existing automatic analysis which is yellow / uncertain.
            if gloss and status == S.UNCERTAIN:
                cell["origin"] = S.USER
                cell["status"] = S.CONFIDENT
                cell["suggested"] = ""
                changed += 1

        if not changed:
            return {
                "token": token,
                "approved": 0,
                "history": history.state(document_id),
            }

        # Human confirmation protects the accepted morphology from a later
        # automatic re-analysis of line 1.
        token["locked"] = True

        cells = token.get("morphemes") or []
        if cells and all(
            cell.get("status") == S.CONFIDENT
            for cell in cells
        ):
            token["confidence"] = "HIGH"
            token["uncertainty"] = []

        history.record(
            document_id,
            H.Entry("token", before, copy.deepcopy(token), token_id=token_id),
        )
        store.save(document)

        return {
            "token": token,
            "approved": changed,
            "history": history.state(document_id),
        }

    # ---- editing line 1: re-gloss a token ----
    @app.post("/api/documents/{document_id}/tokens/{token_id}/regloss")
    def regloss_token(document_id: str, token_id: str, body: TokenEdit):
        document = store.load(document_id)
        sentence, token = _find_token(document, token_id)
        before = copy.deepcopy(token)
        glossed = glosser.gloss_token(S.nfc(body.surface).strip())
        rebuilt = S.build_token(
            {**glossed, "source_token": body.surface}, token_id, 0,
            case=inventory.apply_case)
        # keep the cell's stable identity so the DOM node maps cleanly, but take
        # the freshly glossed morphology, lemma, alternatives and confidence
        # The transcription typed by the editor is authoritative.
        # The analyzer may normalize spelling internally for lookup, but its
        # rebuilt surface must never rewrite the user's source transcription.
        edited_surface = S.nfc(body.surface).strip()
        S.resurface(sentence, token_id, token["surface"], edited_surface)
        token["surface"] = edited_surface
        S.remember_surface_edit(token, before.get("surface") or "")
        fields = ["normalized", "confidence", "tier", "lemma", "lemma_id",
                  "lexical_gloss", "uncertainty", "alternatives", "morphemes"]
        # A token whose line 2 was written by hand is cut loose from line 1: the
        # transcription and the morphology are then two separate statements, and
        # editing one must not silently rewrite the other. Invariant 3 — human
        # decisions outrank generation.
        if token.get("locked"):
            fields.remove("morphemes")
        else:
            token["locked"] = False
        for field in fields:
            token[field] = rebuilt[field]
        links.annotate(document)
        history.record(document_id, H.Entry("token", before, copy.deepcopy(token),
                                            token_id=token_id))
        store.save(document)
        return {"token": token, "reglossed": not before.get("locked"),
                "history": history.state(document_id)}

    # ---- Del: keep line 1, restore the previous lines 2-3 ----
    @app.post("/api/documents/{document_id}/tokens/{token_id}/keep-morphology")
    def keep_morphology(document_id: str, token_id: str, body: dict):
        document = store.load(document_id)
        _sentence, token = _find_token(document, token_id)
        morphemes = body.get("morphemes")
        if isinstance(morphemes, list) and morphemes:
            before = copy.deepcopy(token)
            token["morphemes"] = morphemes
            token["locked"] = True
            history.record(document_id,
                           H.Entry("token", before, copy.deepcopy(token),
                                   token_id=token_id))
        store.save(document)
        return {"token": token, "history": history.state(document_id)}

    # ---- comments ----
    @app.post("/api/documents/{document_id}/comments")
    def add_comment(document_id: str, body: CommentBody):
        document = store.load(document_id)
        sentence, _token = _find_token(document, body.token_id)
        comment = {
            # Not S.stable_id(...): that hashes token_id/morph_id/S.now(),
            # and S.now() is second-granular — two comments added to the
            # same word inside one second hash to the same id, and PATCH/
            # DELETE (keyed by comment_id) would then hit whichever one
            # happens to come first. A comment has no natural content to
            # make the hash unique from, so it gets a real unique id instead.
            "comment_id": f"ecmt_{uuid.uuid4().hex[:16]}",
            "anchor": {"token_id": body.token_id, "morph_id": body.morph_id},
            "text": S.nfc(body.text).strip(),
            "created": S.now(),
        }
        sentence.setdefault("comments", []).append(comment)
        history.record(document_id,
                       H.Entry("comment", None, dict(comment),
                               sentence_id=sentence["sentence_id"]))
        store.save(document)
        return {"comment": comment, "history": history.state(document_id)}

    @app.patch("/api/documents/{document_id}/comments/{comment_id}")
    def edit_comment(document_id: str, comment_id: str, body: CommentEditBody):
        document = store.load(document_id)
        sentence, comment = _find_comment(document, comment_id)
        before = dict(comment)
        comment["text"] = S.nfc(body.text).strip()
        history.record(document_id,
                       H.Entry("comment", before, dict(comment),
                               sentence_id=sentence["sentence_id"]))
        store.save(document)
        return {"comment": comment, "history": history.state(document_id)}

    @app.delete("/api/documents/{document_id}/comments/{comment_id}")
    def delete_comment(document_id: str, comment_id: str):
        document = store.load(document_id)
        sentence, comment = _find_comment(document, comment_id)
        sentence["comments"] = [c for c in sentence.get("comments") or []
                                if c.get("comment_id") != comment_id]
        history.record(document_id,
                       H.Entry("comment", dict(comment), None,
                               sentence_id=sentence["sentence_id"]))
        store.save(document)
        return {"history": history.state(document_id)}

    # ---- the single-editor lock ----
    def _lock_view(document_id: str, client_id: str) -> dict[str, Any]:
        holder = locks.status(document_id)
        return {"holder": holder, "you_are_holder": locks.holder_is(document_id, client_id)}

    @app.get("/api/documents/{document_id}/lock")
    def get_lock(document_id: str, client_id: str = ""):
        store.load(document_id)  # 404s if the document does not exist
        return _lock_view(document_id, client_id)

    @app.post("/api/documents/{document_id}/lock/claim")
    def claim_lock(document_id: str, body: LockClaimBody):
        store.load(document_id)
        locks.claim(document_id, body.client_id, S.nfc(body.name).strip())
        return _lock_view(document_id, body.client_id)

    @app.post("/api/documents/{document_id}/lock/take-over")
    def take_over_lock(document_id: str, body: LockClaimBody):
        store.load(document_id)
        locks.take_over(document_id, body.client_id, S.nfc(body.name).strip())
        return _lock_view(document_id, body.client_id)

    @app.post("/api/documents/{document_id}/lock/release")
    def release_lock(document_id: str, body: LockReleaseBody):
        store.load(document_id)
        locks.release(document_id, body.client_id)
        return _lock_view(document_id, body.client_id)

    # ---- discussion threads ----
    # A conversation channel separate from `comments` (примечания): open to
    # every reader regardless of who holds the pen (exempted from
    # `_configure_edit_lock` above), never part of Undo/Redo — see the design
    # doc's "Не входит в объём" for why.
    @app.post("/api/documents/{document_id}/discussions")
    def start_discussion(document_id: str, body: DiscussionBody):
        document = store.load(document_id)
        sentence, _token = _find_token(document, body.token_id)
        author = S.nfc(body.author).strip()
        thread = {
            "thread_id": f"thrd_{uuid.uuid4().hex[:16]}",
            "anchor": {"token_id": body.token_id, "morph_id": body.morph_id},
            "resolved": False,
            "created": S.now(),
            "messages": [{
                "message_id": f"dmsg_{uuid.uuid4().hex[:16]}",
                "author": author,
                "text": S.nfc(body.text).strip(),
                "created": S.now(),
                "edited": None,
            }],
        }
        sentence.setdefault("discussions", []).append(thread)
        store.save(document)
        return {"thread": thread}

    @app.post("/api/documents/{document_id}/discussions/{thread_id}/messages")
    def reply_to_discussion(document_id: str, thread_id: str, body: DiscussionMessageBody):
        document = store.load(document_id)
        _sentence, thread = _find_thread(document, thread_id)
        thread["messages"].append({
            "message_id": f"dmsg_{uuid.uuid4().hex[:16]}",
            "author": S.nfc(body.author).strip(),
            "text": S.nfc(body.text).strip(),
            "created": S.now(),
            "edited": None,
        })
        store.save(document)
        return {"thread": thread}

    @app.patch("/api/documents/{document_id}/discussions/{thread_id}/messages/{message_id}")
    def edit_discussion_message(document_id: str, thread_id: str, message_id: str,
                                body: DiscussionMessageBody):
        document = store.load(document_id)
        _sentence, thread = _find_thread(document, thread_id)
        message = _find_message(thread, message_id)
        author = S.nfc(body.author).strip()
        if message["author"] != author:
            # A soft check, not a security boundary: nothing here is
            # cryptographically authenticated (see the design doc, §1). It
            # only guards against absent-mindedly editing someone else's
            # reply through the same shared login.
            raise HTTPException(403, "only the original author can edit this message")
        message["text"] = S.nfc(body.text).strip()
        message["edited"] = S.now()
        store.save(document)
        return {"thread": thread}

    @app.delete("/api/documents/{document_id}/discussions/{thread_id}/messages/{message_id}")
    def delete_discussion_message(document_id: str, thread_id: str, message_id: str,
                                  author: str):
        document = store.load(document_id)
        sentence, thread = _find_thread(document, thread_id)
        message = _find_message(thread, message_id)
        if message["author"] != S.nfc(author).strip():
            raise HTTPException(403, "only the original author can delete this message")
        thread["messages"] = [m for m in thread["messages"]
                              if m["message_id"] != message_id]
        if not thread["messages"]:
            sentence["discussions"] = [t for t in sentence.get("discussions") or []
                                       if t["thread_id"] != thread_id]
            store.save(document)
            return {"thread": None}
        store.save(document)
        return {"thread": thread}

    @app.post("/api/documents/{document_id}/discussions/{thread_id}/resolve")
    def resolve_discussion(document_id: str, thread_id: str):
        document = store.load(document_id)
        _sentence, thread = _find_thread(document, thread_id)
        thread["resolved"] = True
        store.save(document)
        return {"thread": thread}

    @app.post("/api/documents/{document_id}/discussions/{thread_id}/reopen")
    def reopen_discussion(document_id: str, thread_id: str):
        document = store.load(document_id)
        _sentence, thread = _find_thread(document, thread_id)
        thread["resolved"] = False
        store.save(document)
        return {"thread": thread}

    # ---- adding a line ----
    @app.post("/api/documents/{document_id}/lines")
    def add_line(document_id: str, body: NewLineBody):
        """Gloss a new transcription line and put it in the document.

        Appended at the end, or placed straight after a given line, so a turn
        can be inserted where it belongs rather than only at the bottom.
        """
        document = store.load(document_id)
        text = S.nfc(body.text).strip()
        if not text:
            raise HTTPException(400, "the line is empty")

        sentences = document.setdefault("sentences", [])
        at = len(sentences)
        if body.after_sentence_id:
            for index, sentence in enumerate(sentences):
                if sentence.get("sentence_id") == body.after_sentence_id:
                    at = index + 1
                    break
            else:
                raise HTTPException(404, f"no such line {body.after_sentence_id}")

        # One line in, one block out — pasting five lines of a transcript has to
        # give five blocks, the same rule that governs a new document, or the
        # whole turn arrives as one wall with a single translation slot.
        lines = S.split_lines(text)
        translations = S.split_lines(body.translation)
        added = []
        for offset, line in enumerate(lines):
            # a new line needs an id of its own; position is not enough, because
            # inserting again above it would collide with an id built from an index
            built = S.build_sentence(
                glosser.gloss_sentence(line), at + offset, document_id,
                translations[offset] if offset < len(translations) else "",
                inventory.apply_case)
            built["sentence_id"] = S.stable_id("esent", document_id, line,
                                               S.now(), offset)
            for position, token in enumerate(built["tokens"]):
                token["token_id"] = S.stable_id("etok", built["sentence_id"],
                                                position, token["surface"])
                for cell_index, morpheme in enumerate(token["morphemes"]):
                    morpheme["morph_id"] = S.stable_id("morph", token["token_id"],
                                                       cell_index, morpheme["form"])
            sentences.insert(at + offset, built)
            added.append(built)
        links.annotate(document)
        for offset, built in enumerate(added):
            history.record(document_id, H.Entry("line", None, copy.deepcopy(built),
                                                index=at + offset))
        store.save(document)
        return {"sentence": added[0], "sentences": added, "index": at,
                "history": history.state(document_id)}

    # ---- undo / redo ----
    @app.get("/api/documents/{document_id}/history")
    def history_state(document_id: str):
        return history.state(document_id)

    @app.post("/api/documents/{document_id}/undo")
    def undo(document_id: str):
        entry = history.pop_undo(document_id)
        if entry is None:
            return {"applied": None, **history.state(document_id)}
        document = store.load(document_id)
        changed = H.apply(document, entry, forward=False)
        store.save(document)
        EL.log_undo_redo(root, "undo", document_id, entry)
        return {"applied": changed, **history.state(document_id)}

    @app.post("/api/documents/{document_id}/redo")
    def redo(document_id: str):
        entry = history.pop_redo(document_id)
        if entry is None:
            return {"applied": None, **history.state(document_id)}
        document = store.load(document_id)
        changed = H.apply(document, entry, forward=True)
        store.save(document)
        EL.log_undo_redo(root, "redo", document_id, entry)
        return {"applied": changed, **history.state(document_id)}

    # ---- export ----
    MEDIA = {
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "json": "application/json",
        "csv": "text/csv; charset=utf-8",
    }

    @app.get("/api/documents/{document_id}/export/{fmt}")
    def export_document(document_id: str, fmt: str):
        if fmt not in MEDIA:
            raise HTTPException(404, f"unknown format {fmt}")
        try:
            document = store.load(document_id)
        except FileNotFoundError:
            raise HTTPException(404, "no such document")
        directory = writable(root, "data/editor_exports")
        produced = X.export_all(document, directory, inventory=inventory)
        path = produced[fmt]
        return FileResponse(path, media_type=MEDIA[fmt], filename=path.name)

    @app.get("/api/backup")
    def backup():
        """A ZIP of the whole persistent volume: on Railway, `MAALULA_STATE_DIR`
        is the only copy of every text and every edit (docs/USER_GUIDE_RU.md
        §8, docs/DEPLOY_RAILWAY_ROADMAP_RU.md Stage 8). Gated the same way
        every other route but /healthz is: `_configure_auth` requires a login
        whenever one is configured, and is wide open when it isn't — true for
        `cli.py edit` on 127.0.0.1, where a backup of the working checkout is
        not a secret to begin with.

        Built into a temp file, not memory: the archive only grows (every
        session, every retained `.trash/` copy), and holding both the
        zipfile's own buffer and `.getvalue()`'s copy in RAM at once does not
        age well as it does. `data/editor_exports/` is skipped — regenerable
        from the sessions this backup already keeps, and the one directory
        under the volume that grows with every download rather than every
        edit, so including it would make "the only copy of your work" mostly
        not the work.
        """
        base = state_root(root)
        skip = base / "data" / "editor_exports"
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(base.rglob("*")):
                if path.is_file() and skip not in path.parents:
                    bundle.write(path, path.relative_to(base))
        stamp = S.now().replace(":", "-")
        return FileResponse(
            tmp_path, media_type="application/zip",
            filename=f"maalula-backup-{stamp}.zip",
            background=BackgroundTask(tmp_path.unlink))

    # ---- label inventory window ----
    concordance_cache: dict[str, Any] = {"key": None, "value": None}
    rebuilds = {"n": 0}

    def _concordance_key() -> tuple[Any, ...]:
        """Cheap fingerprint of every curated/editable input to the window."""
        watched = [writable(root, L.INVENTORY), writable(root, L.CORRECTIONS)]
        watched.extend(sorted(store.dir.glob("*.json")))
        files = []
        for path in watched:
            try:
                stat = path.stat()
                files.append((str(path), stat.st_mtime_ns, stat.st_size))
            except FileNotFoundError:
                files.append((str(path), None, None))
        # A reusable correction reconstructs TextGlosser in memory. A counter
        # bumped on every rebuild invalidates the cache even on a filesystem
        # whose timestamps are too coarse to notice two rapid writes — and
        # unlike `id(analyzer)` it cannot repeat, which a freed address can.
        return (rebuilds["n"], *files)

    def _concordance() -> dict[str, list[dict[str, str]]]:
        """Combined corpus/text inventory, cached until an input changes."""
        key = _concordance_key()
        if concordance_cache["key"] != key:
            concordance_cache["key"] = key
            concordance_cache["value"] = L.combined_concordance(
                root, store, inventory, analyzer=glosser.analyzer)
        return concordance_cache["value"]

    @app.get("/api/labels")
    def label_inventory():
        usage = _concordance()
        rows = inventory.inventory_view(usage)
        for row in rows:
            # the window lists a few examples inline and loads the rest on demand
            row["usage_count"] = len(row["usage"])
            row["usage"] = row["usage"][:3]
        return {"rows": rows}

    @app.get("/api/labels/{label}/concordance")
    def label_concordance(label: str):
        canonical = inventory.canonical(label)
        # A row may be listed under a canonical label via a legacy/internal
        # spelling (PRS under NC). Highlight must follow the same rule.
        spellings = [canonical] + sorted(
            source for source, target in inventory.display_map.items()
            if target == canonical.upper())
        return {"label": canonical, "spellings": spellings,
                "rows": _concordance().get(canonical, [])}

    @app.post("/api/labels/{label}/definition")
    def set_definition(label: str, body: DefinitionBody):
        try:
            inventory.set_definition(label, body.text)
        except KeyError:
            raise HTTPException(404, f"no such label {label}")
        inventory.save()
        return {"label": inventory.canonical(label), "definition": body.text}

    @app.post("/api/corrections")
    def add_correction(body: CorrectionBody):
        """Fix a reading so it applies to every text containing the form."""
        nonlocal glosser
        found = S.resolve_reading(body.surface, body.segmentation, body.gloss,
                                  glosser.analyzer)
        try:
            row = L.record_correction(root, body.surface, body.segmentation,
                                      body.gloss, body.note,
                                      lemma_id=found.lexeme_id if found else None)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        # the analyzer caches the corrections layer at construction, so it has
        # to be rebuilt for the new reading to take effect without a restart
        glosser = _make_glosser(root)
        _warm(glosser)
        rebuilds["n"] += 1
        return {"correction": row,
                "occurrences": L.find_occurrences(store, body.surface)}

    @app.get("/api/occurrences/{surface}")
    def occurrences(surface: str):
        """Where a form can be edited: open texts first, else corpus sentences.

        A form listed in the label window may be in no document at all — the
        corpus is speaking, not the user's texts — so the caller is also handed
        the sentences a new document could be started from.
        """
        return {"surface": surface,
                "occurrences": L.find_occurrences(store, surface),
                "corpus_sentences": L.find_corpus_sentences(root, surface)}

    @app.get("/api/corpus-texts")
    def corpus_texts():
        return {"texts": L.corpus_texts(root)}

    @app.post("/api/documents/from-corpus-text")
    def create_from_corpus_text(body: FromCorpusTextBody):
        """Import a whole corpus text, so a form arrives with its context."""
        lines, translations = L.corpus_text_lines(root, body.source_text)
        if not lines:
            raise HTTPException(404, f"no such corpus text {body.source_text}")
        document = S.build_document(
            "\n".join(lines), glosser, title=body.title or body.source_text,
            translations=translations, case=inventory.apply_case)
        store.save(document)
        return {"document_id": document["document_id"],
                "lines": len(document["sentences"])}

    @app.post("/api/documents/from-corpus")
    def create_from_corpus(body: FromCorpusBody):
        example = L.corpus_sentence(root, body.example_id)
        if example is None:
            raise HTTPException(404, f"no such corpus sentence {body.example_id}")
        # A corpus turn is a paragraph, not a line. Break it into sentences so
        # the document arrives as editable blocks rather than one wall of text.
        from maalula_glosser.gloss_text import split_sentences

        lines = split_sentences(example["transcription"]) or [example["transcription"]]
        translations = split_sentences(example["translation"])
        # pair sentence for sentence only when the two actually line up;
        # otherwise the whole translation goes on the first block rather than
        # being silently misaligned against the wrong sentences
        if len(translations) != len(lines):
            translations = [example["translation"]]
        document = S.build_document(
            "\n".join(lines), glosser,
            title=body.title or example["marker"] or "Из корпуса",
            translations=translations, case=inventory.apply_case)
        store.save(document)
        return {"document_id": document["document_id"],
                "example_id": example["example_id"]}

    @app.patch("/api/documents/{document_id}/sentences/{sentence_id}/translation")
    def set_translation(document_id: str, sentence_id: str, body: TranslationBody):
        """A translation belongs to one line, so it is edited one line at a time."""
        document = store.load(document_id)
        for sentence in document.get("sentences") or []:
            if sentence.get("sentence_id") == sentence_id:
                before = sentence.get("translation", "")
                sentence["translation"] = S.nfc(body.text).strip()
                history.record(document_id,
                               H.Entry("translation", before,
                                       sentence["translation"],
                                       sentence_id=sentence_id))
                store.save(document)
                return {"sentence_id": sentence_id,
                        "translation": sentence["translation"],
                        "history": history.state(document_id)}
        raise HTTPException(404, f"no such line {sentence_id}")

    @app.patch("/api/documents/{document_id}/sentences/{sentence_id}/source")
    def rewrite_source_line(document_id: str, sentence_id: str,
                            body: TranslationBody):
        """Rewrite line 1 whole: punctuate it, or add a word in the middle.

        Words that survived the edit keep their glosses; only what appeared is
        analysed. Punctuation is not part of a word here, so adding a comma
        moves nothing in lines 2-3.
        """
        document = store.load(document_id)
        for sentence in document.get("sentences") or []:
            if sentence.get("sentence_id") != sentence_id:
                continue
            if not S.nfc(body.text).strip():
                raise HTTPException(400, "the line is empty")
            before = copy.deepcopy(sentence["tokens"])
            counts = S.rewrite_line(sentence, body.text, glosser,
                                    inventory.apply_case)
            links.annotate(document)
            history.record(document_id,
                           H.Entry("tokens", before,
                                   copy.deepcopy(sentence["tokens"]),
                                   sentence_id=sentence_id))
            store.save(document)
            return {"sentence": sentence, **counts,
                    "history": history.state(document_id)}
        raise HTTPException(404, f"no such line {sentence_id}")

    @app.post("/api/documents/{document_id}/sentences/{sentence_id}/regloss")
    def regloss_sentence(document_id: str, sentence_id: str):
        """Re-gloss this line from scratch (Alt+R): ignores any lock, hand
        edit or imported IGT reading its tokens carry, unlike the light-touch
        `rewrite_source_line` above which keeps what survived."""
        document = store.load(document_id)
        for sentence in document.get("sentences") or []:
            if sentence.get("sentence_id") != sentence_id:
                continue
            before = copy.deepcopy(sentence["tokens"])
            S.force_regloss_line(sentence, glosser, inventory.apply_case)
            links.annotate(document)
            history.record(document_id,
                           H.Entry("tokens", before,
                                   copy.deepcopy(sentence["tokens"]),
                                   sentence_id=sentence_id))
            store.save(document)
            return {"sentence": sentence, "history": history.state(document_id)}
        raise HTTPException(404, f"no such line {sentence_id}")

    @app.post("/api/documents/{document_id}/sentences/{sentence_id}/review")
    def set_reviewed(document_id: str, sentence_id: str, body: ReviewBody):
        """Tick or clear the "проверено" box: credits (or un-credits) whoever
        ticked it at REVIEW_RATE_RUB_PER_WORD per word, auto-glossed words
        included. Not an undo-able edit — a bookkeeping flag, not content."""
        document = store.load(document_id)
        for sentence in document.get("sentences") or []:
            if sentence.get("sentence_id") != sentence_id:
                continue
            if body.reviewed:
                name = S.nfc(body.name).strip()
                if not name:
                    raise HTTPException(
                        400, "Укажите своё имя вверху страницы, прежде чем "
                             "отмечать строку проверенной")
                word_count = sum(1 for token in sentence.get("tokens") or []
                                 if not S.is_speaker(token))
                sentence["reviewed_by"] = name
                sentence["reviewed_word_count"] = word_count
            else:
                sentence["reviewed_by"] = None
                sentence["reviewed_word_count"] = None
            store.save(document)
            return {"sentence_id": sentence_id,
                    "reviewed_by": sentence.get("reviewed_by"),
                    "reviewed_word_count": sentence.get("reviewed_word_count"),
                    "totals": _review_totals()}
        raise HTTPException(404, f"no such line {sentence_id}")

    @app.post("/api/documents/{document_id}/review-all")
    def set_document_reviewed(document_id: str, body: ReviewBody):
        """Mark every still-unreviewed sentence in the document as reviewed.

        Existing review attribution is preserved: if a sentence was already
        checked by another editor, this bulk action does not overwrite their
        name or payment credit.  The operation is deliberately bookkeeping,
        just like the per-sentence checkbox, so it is not added to undo/redo.
        """
        if not body.reviewed:
            raise HTTPException(
                400,
                "Массовая команда поддерживает только отметку текста как проверенного"
            )

        name = S.nfc(body.name).strip()
        if not name:
            raise HTTPException(
                400,
                "Укажите своё имя вверху страницы, прежде чем отмечать текст проверенным"
            )

        document = store.load(document_id)
        changed_sentences = 0
        changed_words = 0

        for sentence in document.get("sentences") or []:
            # Preserve another reviewer's existing attribution.
            if sentence.get("reviewed_by"):
                continue

            word_count = sum(
                1 for token in sentence.get("tokens") or []
                if not S.is_speaker(token)
            )
            sentence["reviewed_by"] = name
            sentence["reviewed_word_count"] = word_count
            changed_sentences += 1
            changed_words += word_count

        if changed_sentences:
            store.save(document)

        return {
            "reviewed": True,
            "reviewed_by": name,
            "changed_sentences": changed_sentences,
            "changed_words": changed_words,
            "sentences": [
                {
                    "sentence_id": sentence.get("sentence_id"),
                    "reviewed_by": sentence.get("reviewed_by"),
                    "reviewed_word_count": sentence.get("reviewed_word_count"),
                }
                for sentence in document.get("sentences") or []
            ],
            "totals": _review_totals(),
        }

    @app.delete("/api/documents/{document_id}/sentences/{sentence_id}")
    def delete_line(document_id: str, sentence_id: str):
        """Drop a whole interlinear block, comments and translation with it.

        Recorded as the mirror image of adding a line, so undo puts it back
        where it stood rather than at the end.
        """
        document = store.load(document_id)
        sentences = document.get("sentences") or []
        for index, sentence in enumerate(sentences):
            if sentence.get("sentence_id") != sentence_id:
                continue
            removed = copy.deepcopy(sentence)
            del sentences[index]
            history.record(document_id, H.Entry("line", removed, None,
                                                index=index))
            store.save(document)
            return {"sentence_id": sentence_id, "index": index,
                    "history": history.state(document_id)}
        raise HTTPException(404, f"no such line {sentence_id}")

    @app.post("/api/labels/{label}/rename")
    def rename_label(label: str, body: RenameBody):
        canonical = inventory.canonical(label)
        # A label may carry a caveat that makes a blanket rename risky. Surface
        # it and require an explicit confirmation instead of applying silently.
        note = inventory.note_for(canonical)
        if note and not body.confirm:
            return JSONResponse(status_code=409,
                                content={"needs_confirmation": True,
                                         "label": canonical, "note": note})
        try:
            outcome = inventory.rename(canonical, body.new)
        except KeyError:
            raise HTTPException(404, f"no such label {label}")
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        inventory.save()
        documents = L.rewrite_documents(store, canonical, inventory.canonical(body.new))
        return {**outcome, "label": inventory.canonical(body.new),
                "documents_updated": documents}

    # ---- lexical-gloss window: "Словник значений" ----
    #
    # By lemma/lexeme, not by morphological label — the sibling of the
    # `/labels` block above, backed by `editor/senses.py` instead of
    # `editor/labels.py`. See `docs/LEXICAL_REUSABLE_SPEC_RU.md` §6.

    def _lemma_for_lexeme(lexeme_id: str) -> str | None:
        """The citation form to show/confirm/link a lexeme by — from the
        corpus lexicon when it has one, otherwise from whatever the imported
        tokens themselves carry for a `lexnew_...` id the analyser never saw
        (`SE.unresolved_text_lexemes`)."""
        lexeme = glosser.analyzer.lexemes.get(lexeme_id) or {}
        lemma = lexeme.get("lemma_canonical")
        if lemma:
            return lemma
        unresolved = text_summary["unresolved"].get(lexeme_id)
        return unresolved["lemma"] if unresolved else None

    def _stem_form_for_lexeme(lexeme_id: str) -> str | None:
        stems = [S.nfc(x).strip() for x in
                 glosser.analyzer.stems_by_lexeme.get(lexeme_id, [])
                 if S.nfc(x).strip()]
        unique = list(dict.fromkeys(stems))
        if len(unique) == 1:
            return unique[0]
        lemma = S.nfc(_lemma_for_lexeme(lexeme_id) or "").strip()
        return lemma if lemma and GS.looks_like_a_lemma(lemma) else None

    @app.get("/api/senses")
    def senses_inventory():
        """Every lexeme with SOME resolved meaning: author-confirmed rows
        first-class, plus everything the corpus/project tiers already resolve
        but nobody has confirmed yet — so the window is a complete picture to
        browse, not only the (initially empty) reusable journal.

        `text_titles` per row and the top-level `text_sources` list back the
        window's source-filter checkboxes (§12): one pass over the saved
        texts here, not a per-lexeme store scan, since every row needs it.
        """
        # Turoyo/FLEX must not consult the inherited Maalula MASC analyses or
        # curated verbal list that may still exist in an untouched data/.
        analyses_by_lexeme = {}
        curated_pos = {}
        confirmed = SE.active_lexical_glosses(root)
        confirmed_senses = SE.active_lexical_senses(root)
        text_summary = SE.text_lexicon_summary(store, glosser.analyzer)
        text_senses = text_summary["senses"] if profile == "flex" else {}
        titles_by_lexeme = text_summary["titles"]

        def _extra(lexeme_id, lemma):
            return {
                "text_titles": sorted(titles_by_lexeme.get(lexeme_id, ())),
                "has_dictionary": bool(links.link_for(lemma)),
            }

        rows = []
        for lexeme_id, row in confirmed.items():
            lexeme = glosser.analyzer.lexemes.get(lexeme_id) or {}
            forms = glosser.analyzer.forms_by_lexeme.get(lexeme_id) or []
            lemma = row.get("lemma_canonical") or lexeme.get("lemma_canonical")
            rows.append({
                "lexeme_id": lexeme_id,
                "lemma": lemma,
                "gloss": row.get("gloss"),
                "gloss_language": row.get("gloss_language"),
                "form_count": len(forms),
                "excluded_count": len(row.get("excluded_surfaces") or []),
                "note": row.get("note"),
                "source": "author",
                "part_of_speech": SE.resolved_part_of_speech(
                    root, lexeme_id, row, analyses_by_lexeme,
                    lemma=lemma, curated=curated_pos),
                **_extra(lexeme_id, lemma),
            })
        for lexeme_id, (gloss, status) in SE.resolvable_lexemes(glosser.analyzer).items():
            if lexeme_id in confirmed:
                continue
            lexeme = glosser.analyzer.lexemes.get(lexeme_id) or {}
            forms = glosser.analyzer.forms_by_lexeme.get(lexeme_id) or []
            lemma = lexeme.get("lemma_canonical")
            rows.append({
                "lexeme_id": lexeme_id,
                "lemma": lemma,
                "gloss": gloss,
                "gloss_language": "en",
                "form_count": len(forms),
                "excluded_count": 0,
                "note": None,
                "source": status,
                "part_of_speech": SE.resolved_part_of_speech(
                    root, lexeme_id, None, analyses_by_lexeme,
                    lemma=lemma, curated=curated_pos),
                **_extra(lexeme_id, lemma),
            })
        # A real lexeme the texts use but that has no meaning yet (bound by
        # hand, or by «Леммы…»): listed too, with the texts' own gloss as a
        # suggestion to confirm — otherwise it is bound yet invisible here.
        listed_ids = {row["lexeme_id"] for row in rows}
        text_glosses = text_summary["glosses"]
        for lexeme_id in titles_by_lexeme:
            if lexeme_id in listed_ids or lexeme_id not in glosser.analyzer.lexemes:
                continue
            lexeme = glosser.analyzer.lexemes[lexeme_id]
            lemma = lexeme.get("lemma_canonical")
            if not lemma or str(lemma).startswith("["):
                continue
            forms = glosser.analyzer.forms_by_lexeme.get(lexeme_id) or []
            rows.append({
                "lexeme_id": lexeme_id,
                "lemma": lemma,
                "gloss": text_glosses.get(lexeme_id),
                "gloss_language": "en",
                "form_count": len(forms),
                "excluded_count": 0,
                "note": None,
                "source": "text",
                "part_of_speech": SE.resolved_part_of_speech(
                    root, lexeme_id, None, analyses_by_lexeme,
                    lemma=lemma, curated=curated_pos),
                **_extra(lexeme_id, lemma),
            })
            listed_ids.add(lexeme_id)
        # Vocabulary the analyser never saw at all — no `data/final/lexemes.jsonl`
        # entry, so `resolvable_lexemes` above cannot list it — but minted its
        # own lemma_id at import time (`session.overlay_user_igt`). Without
        # this pass a word like "čiṭwīra" is confirmed in the text yet
        # invisible in the Словник (docs/LEXICAL_REUSABLE_SPEC_RU.md).
        # Already computed in the one-pass text summary above.  Re-scanning
        # every saved FLEX document here made the vocabulary page needlessly
        # read the entire corpus twice.
        unresolved = text_summary["unresolved"]
        for lexeme_id, info in unresolved.items():
            if lexeme_id in confirmed:
                continue
            # A candidate id can only be stale if the lexeme it once named is
            # gone — data/final is never rewritten and additions are
            # append-only, so this should not happen, but a dangling id with
            # no lemma to show is not information the author can act on; skip
            # it rather than print a `None` where a word belongs.
            candidate_details = [
                {"lexeme_id": cid,
                 "lemma": glosser.analyzer.lexemes[cid].get("lemma_canonical"),
                 "root": glosser.analyzer.lexemes[cid].get("root_canonical"),
                 "form_count": len(glosser.analyzer.forms_by_lexeme.get(cid, []))}
                for cid in info.get("candidates") or ()
                if cid in glosser.analyzer.lexemes
                and glosser.analyzer.lexemes[cid].get("lemma_canonical")
            ]
            rows.append({**SE.queue_row(lexeme_id, info),
                        "has_dictionary": bool(links.link_for(info["lemma"])),
                        "candidate_details": candidate_details})

        if profile == "flex":
            # FLEX without a LIFT source keeps the older compatibility view:
            # technical lemma+gloss ids are grouped by citation string.  Once
            # FieldWorks LIFT is present, however, ENTRY IDENTITY IS PRIMARY:
            # two LIFT entries with the same spelling remain two separate
            # dictionary articles, with their own grammatical-info and senses.
            def collapse_legacy_flex(input_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
                grouped: dict[str, dict[str, Any]] = {}

                def add_sense(group: dict[str, Any], gloss: str | None, *,
                              source: str, lexeme_id: str,
                              addition_id: str | None = None,
                              language: str = "en", count: int = 0,
                              grammatical_info: str | None = None) -> None:
                    gloss = S.nfc(gloss or "").strip()
                    if not gloss:
                        return
                    key = gloss.casefold()
                    sense = group["_senses"].get(key)
                    if sense is None:
                        sense = {"gloss": gloss, "gloss_language": language or "en",
                                 "sources": set(), "lexeme_ids": set(),
                                 "author": False, "addition_ids": set(), "count": 0,
                                 "grammatical_info": grammatical_info}
                        group["_senses"][key] = sense
                    sense["sources"].add(source)
                    sense["lexeme_ids"].add(lexeme_id)
                    sense["author"] = sense["author"] or source == "author"
                    if addition_id:
                        sense["addition_ids"].add(addition_id)
                    sense["count"] += int(count or 0)
                    if not sense.get("grammatical_info") and grammatical_info:
                        sense["grammatical_info"] = grammatical_info

                for row in input_rows:
                    lemma = S.nfc(row.get("lemma") or "").strip()
                    key = lemma.casefold() if lemma else f"@{row.get('lexeme_id')}"
                    group = grouped.setdefault(key, {
                        "lexeme_id": row.get("lexeme_id"), "lemma": lemma,
                        "gloss": None, "gloss_language": "en",
                        "form_count": 0, "excluded_count": 0, "note": None,
                        "source": "lemma", "part_of_speech": row.get("part_of_speech"),
                        "grammatical_infos": list(row.get("grammatical_infos") or []),
                        "text_titles": set(), "has_dictionary": False,
                        "has_lift": False, "is_new": True,
                        "needs_citation_form": True,
                        "candidate_details": [], "_ids": set(), "_senses": {},
                    })
                    lid = row.get("lexeme_id")
                    if lid:
                        group["_ids"].add(lid)
                    group["text_titles"].update(row.get("text_titles") or [])
                    group["has_dictionary"] = group["has_dictionary"] or bool(row.get("has_dictionary"))
                    group["is_new"] = group["is_new"] and bool(row.get("is_new"))
                    group["needs_citation_form"] = (group["needs_citation_form"]
                                                       and bool(row.get("needs_citation_form")))
                    group["excluded_count"] += int(row.get("excluded_count") or 0)
                    if not group.get("part_of_speech") and row.get("part_of_speech"):
                        group["part_of_speech"] = row.get("part_of_speech")
                    row_gis = list(row.get("grammatical_infos") or [])
                    if lid:
                        row_gis.extend((glosser.analyzer.lexeme_record(lid) or {}).get("grammatical_infos") or [])
                    for gi in row_gis:
                        if gi and gi not in group["grammatical_infos"]:
                            group["grammatical_infos"].append(gi)
                    for candidate in row.get("candidate_details") or []:
                        if candidate not in group["candidate_details"]:
                            group["candidate_details"].append(candidate)
                    add_sense(group, row.get("gloss"), source=row.get("source") or "corpus",
                              lexeme_id=lid or "", language=row.get("gloss_language") or "en",
                              grammatical_info=row.get("grammatical_info"))
                    for sense in confirmed_senses.get(lid, []):
                        add_sense(group, sense.get("gloss"), source="author",
                                  lexeme_id=lid or "", addition_id=sense.get("addition_id"),
                                  language=sense.get("gloss_language") or "en")
                    for sense in text_senses.get(lid, []):
                        add_sense(group, sense.get("gloss"), source="text",
                                  lexeme_id=lid or "", count=sense.get("count") or 0,
                                  grammatical_info=row.get("grammatical_info"))

                collapsed = []
                for group in grouped.values():
                    ids = sorted(i for i in group.pop("_ids") if i)
                    group["lexeme_ids"] = ids
                    if ids:
                        author_ids = [i for i in ids
                                      if (glosser.analyzer.lexemes.get(i) or {}).get("status")
                                      == "author_addition"]
                        group["lexeme_id"] = author_ids[0] if author_ids else ids[0]
                    surfaces = {
                        S.nfc(form.get("form_canonical") or "")
                        for lid in ids
                        for form in glosser.analyzer.forms_by_lexeme.get(lid, [])
                        if S.nfc(form.get("form_canonical") or "")
                    }
                    group["form_count"] = len(surfaces)
                    senses = []
                    for sense in group.pop("_senses").values():
                        sense["sources"] = sorted(sense["sources"])
                        sense["lexeme_ids"] = sorted(i for i in sense["lexeme_ids"] if i)
                        sense["addition_ids"] = sorted(sense["addition_ids"])
                        senses.append(sense)
                    senses.sort(key=lambda x: (not x["author"], -int(x.get("count") or 0),
                                                x["gloss"].casefold()))
                    group["senses"] = senses
                    if senses:
                        group["gloss"] = senses[0]["gloss"]
                        group["gloss_language"] = senses[0].get("gloss_language") or "en"
                    group["source"] = "author" if any(x["author"] for x in senses) else "lemma"
                    group["text_titles"] = sorted(group["text_titles"])
                    collapsed.append(group)
                return collapsed

            lift_index = getattr(glosser.analyzer, "lift", None)
            if lift_index and lift_index.has_entries():
                lift_ids = set(lift_index.entries)
                exact_rows: list[dict[str, Any]] = []
                by_row = {row.get("lexeme_id"): row for row in rows if row.get("lexeme_id")}

                def _sense_key(gloss: str | None) -> str:
                    return S.nfc(gloss or "").strip().casefold()

                for lexeme_id, entry in lift_index.entries.items():
                    record = glosser.analyzer.lexeme_record(lexeme_id) or {}
                    existing_row = by_row.get(lexeme_id) or {}
                    forms = glosser.analyzer.forms_by_lexeme.get(lexeme_id) or []
                    unique_forms = {S.nfc(form.get("form_canonical") or "")
                                    for form in forms
                                    if S.nfc(form.get("form_canonical") or "")}
                    senses: list[dict[str, Any]] = []
                    by_gloss: dict[str, list[dict[str, Any]]] = {}
                    for lift_sense in entry.get("senses") or []:
                        gloss = preferred_sense_gloss(lift_sense)
                        if not gloss:
                            continue
                        sense = {
                            "sense_id": lift_sense.get("sense_id"),
                            "gloss": gloss,
                            "gloss_language": next((g.get("lang") for g in lift_sense.get("glosses") or []
                                                     if g.get("text") == gloss), "en") or "en",
                            "grammatical_info": lift_sense.get("grammatical_info"),
                            "sources": ["lift"],
                            "lexeme_ids": [lexeme_id],
                            "author": False,
                            "addition_ids": [],
                            "count": 0,
                        }
                        senses.append(sense)
                        by_gloss.setdefault(_sense_key(gloss), []).append(sense)

                    # Saved-text counts enrich the exact FieldWorks entry but
                    # never fuse it with another entry that happens to have the
                    # same citation form.
                    entry_gis = list(entry.get("grammatical_infos") or [])
                    default_entry_gi = entry_gis[0] if len(entry_gis) == 1 else None
                    for text_sense in text_senses.get(lexeme_id, []):
                        gloss = S.nfc(text_sense.get("gloss") or "").strip()
                        if not gloss:
                            continue
                        matches = by_gloss.get(_sense_key(gloss), [])
                        if len(matches) == 1:
                            target = matches[0]
                            if "text" not in target["sources"]:
                                target["sources"].append("text")
                            target["count"] += int(text_sense.get("count") or 0)
                        else:
                            dynamic = {
                                "sense_id": None, "gloss": gloss,
                                "gloss_language": "en",
                                "grammatical_info": default_entry_gi,
                                "sources": ["text"], "lexeme_ids": [lexeme_id],
                                "author": False, "addition_ids": [],
                                "count": int(text_sense.get("count") or 0),
                            }
                            senses.append(dynamic)
                            # A corpus-only meaning is still the same sense if
                            # the author confirms/edits it later.  Index a new
                            # dynamic sense only when no LIFT sense with this
                            # gloss existed; if several LIFT senses share the
                            # spelling, keep that ambiguity explicit.
                            if not matches:
                                by_gloss.setdefault(_sense_key(gloss), []).append(dynamic)

                    for author_sense in confirmed_senses.get(lexeme_id, []):
                        gloss = S.nfc(author_sense.get("gloss") or "").strip()
                        if not gloss:
                            continue
                        matches = by_gloss.get(_sense_key(gloss), [])
                        if len(matches) == 1:
                            target = matches[0]
                            if "author" not in target["sources"]:
                                target["sources"].append("author")
                            target["author"] = True
                            if author_sense.get("addition_id"):
                                target["addition_ids"].append(author_sense["addition_id"])
                        else:
                            senses.append({
                                "sense_id": None, "gloss": gloss,
                                "gloss_language": author_sense.get("gloss_language") or "en",
                                "grammatical_info": default_entry_gi,
                                "sources": ["author"], "lexeme_ids": [lexeme_id],
                                "author": True,
                                "addition_ids": ([author_sense.get("addition_id")]
                                                 if author_sense.get("addition_id") else []),
                                "count": 0,
                            })

                    exact_rows.append({
                        "lexeme_id": lexeme_id,
                        "lexeme_ids": [lexeme_id],
                        "lemma": entry.get("lemma_canonical"),
                        "gloss": senses[0]["gloss"] if senses else None,
                        "gloss_language": senses[0].get("gloss_language", "en") if senses else "en",
                        "form_count": len(unique_forms),
                        "excluded_count": int(existing_row.get("excluded_count") or 0),
                        "note": existing_row.get("note"),
                        "source": "lift",
                        "part_of_speech": record.get("part_of_speech"),
                        "grammatical_infos": list(entry.get("grammatical_infos") or []),
                        "morph_type": entry.get("morph_type"),
                        "lift_id": entry.get("lift_id"),
                        "lift_guid": entry.get("guid"),
                        "has_lift": True,
                        "has_dictionary": True,
                        "text_titles": sorted(titles_by_lexeme.get(lexeme_id, ())),
                        "is_new": False,
                        "needs_citation_form": False,
                        "candidate_details": [],
                        "senses": senses,
                    })

                # Old synthetic FLEX ids and author-only entries are still
                # visible, but only those legacy ids may be grouped by lemma.
                legacy = [row for row in rows if row.get("lexeme_id") not in lift_ids]
                rows = exact_rows + collapse_legacy_flex(legacy)
            else:
                rows = collapse_legacy_flex(rows)

        rows.sort(key=lambda r: (r["lemma"] or "").casefold())
        text_sources = sorted({title for titles in titles_by_lexeme.values()
                               for title in titles}
                              | {title for info in unresolved.values()
                                 for title in info["titles"]})
        # «только новые»: lexemes neither MASC nor Arnold had — minted from a
        # text, added by the author, or a service verb for an unlisted root
        for row in rows:
            lexeme_id = row.get("lexeme_id") or ""
            record = glosser.analyzer.lexemes.get(lexeme_id) or {}
            row["is_new"] = (lexeme_id.startswith(("lexnew_", "lexsvc_"))
                             or record.get("status") == "author_addition")
        return {"rows": rows, "text_sources": text_sources}

    @app.get("/api/senses/export.csv")
    def export_lexicon_csv():
        """Export the CURRENT lexicon as one UTF-8 CSV.

        The rows are taken from the same resolved inventory that powers the
        «Словник» page, so the export includes FieldWorks/LIFT entries plus
        local entries, author-added senses and local edits already reflected
        by `senses_inventory()`.  One CSV row is one lexical entry as shown in
        the lexicon.  Separate LIFT entries with the same spelling therefore
        remain separate rows.
        """
        inventory_data = senses_inventory()
        inventory_rows = inventory_data.get("rows") or []
        lift_index = getattr(glosser.analyzer, "lift", None)

        def _unique(values):
            seen = set()
            out = []
            for value in values:
                value = S.nfc(value or "").strip()
                if not value or value in seen:
                    continue
                seen.add(value)
                out.append(value)
            return out

        export_rows = []
        for row in inventory_rows:
            lemma = S.nfc(row.get("lemma") or "").strip()
            if not lemma:
                continue

            lexeme_ids = [x for x in (row.get("lexeme_ids") or
                                      [row.get("lexeme_id")]) if x]
            entry_id = " | ".join(lexeme_ids)

            # Forms known to the analyser include FieldWorks variants and
            # locally-added surface forms.  Do not repeat the citation form.
            variants = []
            for lexeme_id in lexeme_ids:
                for form in glosser.analyzer.forms_by_lexeme.get(lexeme_id, []) or []:
                    text = S.nfc(form.get("form_canonical") or "").strip()
                    if text and text != lemma:
                        variants.append(text)

                # Be conservative and also read preserved LIFT variants
                # directly when available; this makes the export robust even
                # if a future analyser stops materialising every LIFT variant
                # in `forms_by_lexeme`.
                if lift_index and getattr(lift_index, "entries", None):
                    entry = lift_index.entries.get(lexeme_id)
                    if entry:
                        for variant in entry.get("variants") or []:
                            for form in variant.get("forms") or []:
                                text = S.nfc(form.get("text") or "").strip()
                                if text and text != lemma:
                                    variants.append(text)

            senses = []
            sense_gis = []
            for sense in row.get("senses") or []:
                gloss = S.nfc(sense.get("gloss") or "").strip()
                if gloss:
                    senses.append(gloss)
                gi = S.nfc(sense.get("grammatical_info") or "").strip()
                if gi:
                    sense_gis.append(gi)
            if not senses and row.get("gloss"):
                senses.append(S.nfc(row.get("gloss") or "").strip())

            grammatical_infos = list(row.get("grammatical_infos") or [])
            grammatical_infos.extend(sense_gis)
            grammatical_infos = _unique(grammatical_infos)

            raw_source = row.get("source") or ""
            if raw_source == "lift" or row.get("has_lift"):
                source = "FieldWorks"
            elif row.get("is_new") or raw_source == "author":
                source = "Added"
            else:
                source = "Corpus"

            export_rows.append({
                "entry_id": entry_id,
                "lift_id": S.nfc(row.get("lift_id") or "").strip(),
                "guid": S.nfc(row.get("lift_guid") or "").strip(),
                "lemma": lemma,
                "grammatical_info": " | ".join(grammatical_infos),
                "part_of_speech": S.nfc(row.get("part_of_speech") or "").strip(),
                "morph_type": S.nfc(row.get("morph_type") or "").strip(),
                "senses": " | ".join(_unique(senses)),
                "variants": " | ".join(_unique(variants)),
                "source": source,
            })

        export_rows.sort(
            key=lambda r: (
                r["lemma"].casefold(),
                r["grammatical_info"].casefold(),
                r["entry_id"],
            )
        )

        fieldnames = [
            "entry_id",
            "lift_id",
            "guid",
            "lemma",
            "grammatical_info",
            "part_of_speech",
            "morph_type",
            "senses",
            "variants",
            "source",
        ]
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(export_rows)

        filename = f"turoyo_lexicon_{date.today().isoformat()}.csv"
        # UTF-8 BOM makes Excel/Numbers recognise Cyrillic and Turoyo
        # transcription without asking for an encoding.
        content = "\ufeff" + buffer.getvalue()
        return Response(
            content=content,
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            },
        )

    @app.post("/api/senses/seed-from-corpus")
    def senses_seed_from_corpus():
        """Load every not-yet-confirmed resolvable lexeme into the reusable
        layer in one action, as the project owner asked for: a one-time bulk
        promotion, each entry retractable individually afterwards exactly like
        a hand-confirmed one."""
        written = SE.seed_from_corpus(root, glosser.analyzer)
        return {"written": len(written)}

    def _related_flex_lexeme_ids(lexeme_id: str) -> list[str]:
        if profile != "flex":
            return [lexeme_id]
        record = glosser.analyzer.lexeme_record(lexeme_id) or {}
        # A LIFT entry is an identity boundary.  Homographic FieldWorks
        # entries (e.g. different TAM entries) must never leak into each
        # other's concordances merely because their citation forms match.
        if record.get("status") == "lift":
            return [lexeme_id]
        lemma = S.nfc(_lemma_for_lexeme(lexeme_id) or "").strip().casefold()
        if not lemma:
            return [lexeme_id]
        ids = {lexeme_id}
        for lid, other in glosser.analyzer.lexemes.items():
            if other.get("status") == "lift":
                continue
            if S.nfc(other.get("lemma_canonical") or "").strip().casefold() == lemma:
                ids.add(lid)
        for lid, info in SE.unresolved_text_lexemes(store, glosser.analyzer).items():
            if S.nfc(info.get("lemma") or "").strip().casefold() == lemma:
                ids.add(lid)
        return sorted(ids)

    @app.get("/api/senses/{lexeme_id}/concordance")
    def senses_concordance(lexeme_id: str):
        """Merges three sources, each tagged so the window can filter and
        render them differently (§11): `forms`/`text_occurrences` come from
        `sibling_forms` (corpus + this author's own saved texts), `dictionary`
        is the Arnold-scan link already built for the inline gloss tooltip
        (`ArnoldLinks`, reused as-is, not duplicated here)."""
        related_ids = _related_flex_lexeme_ids(lexeme_id)
        if profile == "flex" and len(related_ids) > 1:
            pieces = [SE.sibling_forms(root, glosser.analyzer, lid,
                                       limit=1000, store=store)
                      for lid in related_ids]
            forms = []
            seen_forms = set()
            for piece in pieces:
                for form in piece.get("forms") or []:
                    key = (form.get("surface_canonical"), form.get("segmentation"),
                           form.get("gloss"), form.get("lexical_gloss"))
                    if key in seen_forms:
                        continue
                    seen_forms.add(key)
                    forms.append(form)
            occurrences = []
            seen_occ = set()
            for piece in pieces:
                for occurrence in piece.get("text_occurrences") or []:
                    key = (occurrence.get("document_id"), occurrence.get("token_id"))
                    if key in seen_occ:
                        continue
                    seen_occ.add(key)
                    occurrences.append(occurrence)
            result = {"lexeme_id": lexeme_id, "lemma": _lemma_for_lexeme(lexeme_id),
                      "forms": forms, "text_occurrences": occurrences,
                      "truncated": any(p.get("truncated") for p in pieces),
                      "total_forms": sum(int(p.get("total_forms") or 0) for p in pieces),
                      "lexeme_ids": related_ids}
        else:
            result = SE.sibling_forms(root, glosser.analyzer, lexeme_id,
                                      limit=1000, store=store)
        lemma = _lemma_for_lexeme(lexeme_id)
        result["lemma"] = result.get("lemma") or lemma
        active_rows = [sense for lid in related_ids
                       for sense in SE.active_lexical_senses(root).get(lid, [])]
        excluded = sorted({surface for sense in active_rows
                           for surface in sense.get("excluded_surfaces") or []})
        active = {"excluded_surfaces": excluded} if active_rows else None
        result["active"] = active
        result["senses"] = active_rows
        result["part_of_speech"] = SE.resolved_part_of_speech(
            root, lexeme_id, active, lemma=lemma)
        lexeme_record = glosser.analyzer.lexeme_record(lexeme_id) or {}
        result["grammatical_infos"] = list(lexeme_record.get("grammatical_infos") or [])
        result["lift_entry_id"] = lexeme_record.get("lift_id")
        result["dictionary"] = links.link_for(lemma)
        return result

    @app.post("/api/senses/{lexeme_id}")
    def senses_confirm(lexeme_id: str, body: LexicalConfirmBody):
        nonlocal glosser
        try:
            row = SE.record_lexical_correction(
                root, lexeme_id, _lemma_for_lexeme(lexeme_id), body.gloss,
                body.gloss_language, confirmed=body.confirmed,
                excluded=body.excluded, note=body.note,
                part_of_speech=body.part_of_speech,
                stem_form=_stem_form_for_lexeme(lexeme_id) if profile == "flex" else None)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        glosser.remember_lexical(row)
        return {"correction": row}

    @app.post("/api/senses/{lexeme_id}/readings/{addition_id}/retract")
    def senses_retract_reading(lexeme_id: str, addition_id: str):
        """Retract one child sense without deleting the lemma or its siblings."""
        nonlocal glosser
        related = set(_related_flex_lexeme_ids(lexeme_id))
        active = [row for lid in related
                  for row in SE.active_lexical_senses(root).get(lid, [])
                  if row.get("addition_id") == addition_id]
        if not active:
            raise HTTPException(404, "no such active sense for this lemma")
        result = SE.retract_lexical_correction(root, addition_id)
        glosser = _make_glosser(root)
        _warm(glosser)
        rebuilds["n"] += 1
        return {"retraction": result}

    @app.post("/api/senses/{lexeme_id}/part-of-speech")
    def senses_set_part_of_speech(lexeme_id: str, body: PartOfSpeechBody):
        """Change only the part-of-speech hint, keeping whatever gloss/
        exclusions are already active (or, if this lexeme was only ever a
        corpus-tier suggestion, promoting it as-is — the same "confirm" a
        plain gloss edit already does)."""
        nonlocal glosser
        if body.part_of_speech is not None and body.part_of_speech not in SE.PARTS_OF_SPEECH:
            raise HTTPException(400, f"part_of_speech must be one of "
                                     f"{sorted(SE.PARTS_OF_SPEECH)} or null")
        active = SE.active_lexical_glosses(root).get(lexeme_id)
        if active:
            gloss, gloss_language = active["gloss"], active.get("gloss_language", "en")
            confirmed, excluded = active.get("confirmed_surfaces", []), active.get("excluded_surfaces", [])
            lemma = active.get("lemma_canonical") or _lemma_for_lexeme(lexeme_id)
        else:
            gloss, status = glosser.analyzer.lexical.resolve(lexeme_id)
            if not gloss:
                raise HTTPException(400, "this lexeme has no resolved meaning to attach a part of speech to")
            gloss_language, confirmed, excluded = "en", [], []
            lemma = _lemma_for_lexeme(lexeme_id)
        try:
            row = SE.record_lexical_correction(
                root, lexeme_id, lemma, gloss, gloss_language,
                confirmed=confirmed, excluded=excluded,
                note="изменена часть речи в окне «Словник значений»",
                part_of_speech=body.part_of_speech)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        glosser.remember_lexical(row)
        return {"correction": row}

    @app.post("/api/senses/{lexeme_id}/retract")
    def senses_retract(lexeme_id: str):
        nonlocal glosser
        active = SE.active_lexical_glosses(root).get(lexeme_id)
        if not active:
            raise HTTPException(404, f"no active reading for lexeme {lexeme_id}")
        result = SE.retract_lexical_correction(root, active["addition_id"])
        glosser = _make_glosser(root)
        _warm(glosser)
        rebuilds["n"] += 1
        return {"retraction": result}

    @app.post("/api/lexicon/backfill")
    def lexicon_backfill():
        """Дочистить документы, сохранённые до резолвера: слово, которое
        лексикон знает, получает его лексему вместо минтованной."""
        return LR.rewrite_minted_lexemes(store, glosser.analyzer)

    @app.post("/api/lexicon/backfill-reusable-lemma-ids")
    def backfill_reusable_lemma_ids():
        """Give already-active reusable-morphology corrections a chance to
        pick up a `lemma_id` they were recorded without. Unlike
        `lexicon_backfill`, a newly-recorded `lemma_id` here is a fresh row
        in the corrections journal itself, which `_correction_candidate`
        will surface on the next build — so, unlike that endpoint, this one
        rebuilds the glosser."""
        nonlocal glosser
        result = L.backfill_reusable_lemma_ids(root, glosser.analyzer)
        glosser = _make_glosser(root)
        _warm(glosser)
        rebuilds["n"] += 1
        return result

    def _staged_surfaces(lexeme_id: str) -> tuple[list[dict[str, Any]], list[str]]:
        """Поверхности продвигаемого слова и тексты, где оно встречается.

        Частота — число вхождений каждой поверхности в сохранённых документах
        автора: у слова из своего текста другого счётчика ещё нет.
        """
        counts: dict[str, int] = {}
        titles: set[str] = set()
        for hit in SE.text_occurrences_by_lexeme(store, lexeme_id):
            surface = hit.get("surface_canonical")
            if surface:
                counts[surface] = counts.get(surface, 0) + 1
            if hit.get("title"):
                titles.add(hit["title"])
        surfaces = [{"form_canonical": surface, "frequency": frequency}
                    for surface, frequency in sorted(counts.items())]
        return surfaces, sorted(titles)

    @app.post("/api/lexicon/additions")
    def lexicon_add(body: LexiconAdditionBody):
        """Продвинуть слово из очереди в словарик лексем/форм.

        Если введённая словарная форма уже принадлежит существующей лексеме —
        сливаем с ней, а не заводим дубль. Если она принадлежит более чем
        одной — выбор за автором (инвариант #7), а не поводом для третьей
        записи.
        """
        nonlocal glosser
        lemma = LA.nfc(body.lemma_canonical).strip()
        if not lemma:
            raise HTTPException(400, "a citation form is required")
        existing = [lexeme_id for lexeme_id, lexeme in glosser.analyzer.lexemes.items()
                    if lexeme_id and lexeme_id != body.lexeme_id
                    and LA.nfc(lexeme.get("lemma_canonical")).strip() == lemma]
        if len(existing) > 1:
            raise HTTPException(409, {
                "error": "this citation form already belongs to more than one lexeme",
                "lemma_canonical": lemma,
                "candidates": sorted(existing),
            })
        if len(existing) == 1:
            report = LR.attach_lexeme(store, body.lexeme_id, existing[0], lemma,
                                      gloss=body.gloss)
            if body.gloss:
                row = SE.record_lexical_correction(root, existing[0], lemma,
                                                    body.gloss, body.gloss_language)
                glosser.remember_lexical(row)
            return report
        surfaces, titles = _staged_surfaces(body.lexeme_id)
        for surface_row in surfaces:
            surface = surface_row["form_canonical"]
            owners = {form.get("lexeme_id")
                      for form in glosser.analyzer.forms_by_surface.get(surface, [])
                      if form.get("lexeme_id") and form.get("lexeme_id") != body.lexeme_id}
            if owners:
                raise HTTPException(409, {
                    "error": "this surface already belongs to a different lexeme",
                    "surface": surface,
                    "lexeme_id": sorted(owners)[0],
                    "candidates": sorted(owners),
                })
        try:
            addition = LA.record_addition(
                root, body.lexeme_id, lemma, part_of_speech=body.part_of_speech,
                surfaces=surfaces, titles=titles,
                grammatical_info=body.grammatical_info)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        if body.gloss:
            SE.record_lexical_correction(root, body.lexeme_id, lemma, body.gloss,
                                         body.gloss_language)
        # Новая лексема меняет self.lexemes/forms анализатора, а не только
        # журнал значений, поэтому нужен полный пересбор — тот же приём, что в
        # `senses_retract` (см. выше).
        glosser = _make_glosser(root)
        _warm(glosser)
        rebuilds["n"] += 1
        return {"addition": addition}

    @app.post("/api/lexicon/reassign")
    def lexicon_reassign(body: LexiconReassignBody):
        """Move specific, already-identified tokens onto a different,
        already-established lexeme — the scoped sibling of
        `/api/lexicon/additions`' merge branch (`attach_lexeme`), for when
        only some occurrences were mis-resolved, not every token of some
        staged lexeme.

        No rebuild: both lexemes named here already exist in
        `glosser.analyzer.lexemes`, so nothing analyzer-visible changes —
        only the named tokens in their saved documents.
        """
        target_lexeme_id = _resolve_lexicon_target(glosser, body.target_lexeme_id,
                                                   body.target_lemma)
        target = glosser.analyzer.lexemes[target_lexeme_id]
        if not body.targets:
            raise HTTPException(400, "targets must not be empty")
        lemma = target.get("lemma_canonical")
        gloss, _status = glosser.analyzer.lexical.resolve(target_lexeme_id)
        pairs = [(t.document_id, t.token_id) for t in body.targets]
        try:
            report = LR.reassign_tokens(store, pairs, target_lexeme_id, lemma, gloss=gloss)
        except FileNotFoundError as exc:
            raise HTTPException(404, f"no such document: {exc}")
        if report["tokens"] == 0:
            raise HTTPException(
                400, "none of the named tokens were found in their documents")
        return report

    @app.post("/api/lexicon/merge")
    def lexicon_merge(body: LexiconMergeBody):
        """Merge two already-established lexemes: every token pointing at
        `source_lexeme_id` is rewritten to `target_lexeme_id` (reusing
        `attach_lexeme`'s token-rewrite loop as-is — it already works for
        two arbitrary ids, not just a staged-lexeme promotion), and the merge
        itself becomes a journal row so a rebuilt analyzer folds the
        source's forms into the target and drops the source from the
        vocabulary for good."""
        nonlocal glosser
        if body.source_lexeme_id not in glosser.analyzer.lexemes:
            raise HTTPException(404, f"no such lexeme: {body.source_lexeme_id}")
        if (glosser.analyzer.lexemes.get(body.source_lexeme_id) or {}).get("status") == "lift":
            raise HTTPException(400, "FieldWorks entries stay separate; reassign individual tokens instead")
        target_lexeme_id = _resolve_lexicon_target(glosser, body.target_lexeme_id,
                                                   body.target_lemma)
        if body.source_lexeme_id == target_lexeme_id:
            raise HTTPException(400, "cannot merge a lexeme into itself")
        target = glosser.analyzer.lexemes[target_lexeme_id]
        lemma = target.get("lemma_canonical")
        gloss, _status = glosser.analyzer.lexical.resolve(target_lexeme_id)
        report = LR.attach_lexeme(store, body.source_lexeme_id, target_lexeme_id,
                                  lemma, gloss=gloss, clear_stale_gloss=True)
        # Document-first, then the journal (same order as `lexicon_delete`
        # below): the journal row needs the document walk's `touched_tokens`
        # to make retraction an actual undo (see `lexicon_edit_retract`), so
        # it cannot be written before the walk that produces them.
        edit = LE.record_edit(root, {
            "type": "merge", "source_lexeme_id": body.source_lexeme_id,
            "target_lexeme_id": target_lexeme_id,
            "touched_tokens": report.get("touched_tokens", [])})
        # Both self.lexemes and forms move: the vocabulary window and every
        # future resolve must stop seeing the source lexeme at all.
        glosser = _make_glosser(root)
        _warm(glosser)
        rebuilds["n"] += 1
        return {"merge": edit, "tokens": report}

    @app.delete("/api/lexicon/{lexeme_id}")
    def lexicon_delete(lexeme_id: str):
        """Delete an established lexeme. Its own journal entry excludes it
        from `glosser.analyzer.lexemes`/forms on the next rebuild; every
        saved-document token that pointed at it is walked (same
        document-walking mechanics as `lexicon_backfill`) and un-resolved
        back to the ordinary unresolved shape — a deleted lexeme's tokens
        re-enter the normal resolver/review queue, per product decision,
        rather than landing in any special "orphaned" state."""
        nonlocal glosser
        if lexeme_id not in glosser.analyzer.lexemes:
            raise HTTPException(404, f"no such lexeme: {lexeme_id}")
        if (glosser.analyzer.lexemes.get(lexeme_id) or {}).get("status") == "lift":
            raise HTTPException(400, "FieldWorks entries cannot be deleted here")
        report = LR.clear_lexeme_tokens(store, lexeme_id)
        # Document-first, then the journal, same order as `lexicon_merge`
        # above: the journal row needs the document walk's `touched_tokens`
        # to make retraction an actual undo (see `lexicon_edit_retract`).
        edit = LE.record_edit(root, {"type": "delete", "lexeme_id": lexeme_id,
                                     "touched_tokens": report.get("touched_tokens", [])})
        glosser = _make_glosser(root)
        _warm(glosser)
        rebuilds["n"] += 1
        return {"deletion": edit, "tokens": report}

    @app.post("/api/lexicon/{lexeme_id}/lemma")
    def lexicon_rename(lexeme_id: str, body: LexemeRenameBody):
        """Correct an established lexeme's citation form. A separate
        endpoint from `/api/senses/{lexeme_id}` (which confirms a MEANING
        and requires a non-empty `gloss`): renaming the citation form is
        orthogonal to whether a gloss exists yet, so folding it into that
        body would force an unrelated gloss requirement onto a pure lemma
        fix. Validation mirrors `lexicon_additions.record_addition`'s own
        citation-form check (not a corpus group, not gloss/segmentation-shaped)."""
        nonlocal glosser
        if lexeme_id not in glosser.analyzer.lexemes:
            raise HTTPException(404, f"no such lexeme: {lexeme_id}")
        if (glosser.analyzer.lexemes.get(lexeme_id) or {}).get("status") == "lift":
            raise HTTPException(400, "FieldWorks citation forms come from LIFT and are read-only here")
        lemma = LA.nfc(body.lemma_canonical).strip()
        if not lemma:
            raise HTTPException(400, "a citation form is required")
        if lemma.startswith("["):
            raise HTTPException(400, "a corpus group is not a lexical identity")
        if GS.is_grammatical_label(lemma) or not GS.looks_like_a_lemma(lemma):
            raise HTTPException(400, "this is not a citation form")
        edit = LE.record_edit(root, {
            "type": "rename", "lexeme_id": lexeme_id, "lemma_canonical": lemma})
        glosser = _make_glosser(root)
        _warm(glosser)
        rebuilds["n"] += 1
        return {"rename": edit}

    @app.post("/api/lexicon/edits/{edit_id}/retract")
    def lexicon_edit_retract(edit_id: str):
        """Undo one merge/delete/rename by the same tombstone convention as
        `senses_retract`: the row is never removed, only shadowed, and the
        analyzer must be rebuilt for the retraction to become visible again
        (a merged-away lexeme's re-appearance, a deleted lexeme coming back,
        or a citation form reverting).

        A merge/delete also rewrote saved-document tokens when it happened
        (`attach_lexeme`/`clear_lexeme_tokens`); tombstoning the journal row
        alone does not undo that. So this endpoint -- not `retract_edit`
        itself, which stays a pure journal operation -- also restores those
        tokens from the retracted row's own `touched_tokens`
        (`restore_touched_tokens`), mirroring how `lexicon_merge`/
        `lexicon_delete` already orchestrate journal-write + document-walk
        together. A rename carries no `touched_tokens` (nothing
        document-level to rewind), so this is a no-op for that type."""
        nonlocal glosser
        edit_row = LE.find_active_edit(root, edit_id)
        result = LE.retract_edit(root, edit_id)
        if result.get("already_inactive"):
            raise HTTPException(404, f"no active edit {edit_id}")
        if edit_row and edit_row.get("type") in ("merge", "delete"):
            LR.restore_touched_tokens(store, edit_row.get("touched_tokens") or [])
        glosser = _make_glosser(root)
        _warm(glosser)
        rebuilds["n"] += 1
        return {"retraction": result}

    return app
