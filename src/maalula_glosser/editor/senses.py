"""Reusable lexical-gloss journal: a confirmed meaning applies to every
surface form of the same lexeme, not just the token on screen.

Sibling of `editor/labels.py`'s `reusable_morphology.jsonl`, but keyed by
lexeme (`lexeme_id`, with the lemma string as a fallback) rather than by
exact surface form. This layer is about lexical MEANING only — the string a
translation shows — never about segmentation, POS, TAM or PNG. See
`docs/LEXICAL_REUSABLE_SPEC_RU.md` for the full design.
"""
from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any

from maalula_glosser.editor.session import is_speaker, join_parts
from maalula_glosser.gloss_shape import could_be_a_typed_meaning, looks_like_a_lemma
from maalula_glosser.io_utils import iter_jsonl, read_json, stable_id, write_json
from maalula_glosser.pos import citation_is_verbal, pos_from_labels
from maalula_glosser.reusable import active_readings
from maalula_glosser.state import writable

LEXICAL_CORRECTIONS = "data/corrections_v2/reusable_lexical_gloss.jsonl"
IGT_EXAMPLES = "data/final/igt_examples.jsonl"

# Above this many sibling forms, only the ones with an analysis or a corpus
# sentence (the ones an author can actually judge) are worth showing; open
# question #5 in the spec leaves the exact cutoff to be tuned later.
FORM_LIMIT = 60


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


def _head(lemma: str) -> str:
    """Head form of a multi-part lemma: mirrors `lexical_gloss.py::_head`."""
    text = nfc(lemma).strip()
    return text.split()[0] if text else ""


def _looks_lexical(gloss: str) -> bool:
    """`gloss_shape.could_be_a_typed_meaning`: this field has no morphology,
    so a word that merely collides with a structural label's name (`free`,
    `exist`) is a meaning here, while `3fs`/`mp` (a person/number feature in
    any case) and a label spelled the way labels are spelled (`FREE`) still
    are not."""
    return could_be_a_typed_meaning(gloss)


PARTS_OF_SPEECH = frozenset(("verb", "adjective", "noun"))

# Окно конкорданса: три слова слева и справа. Константа модуля, не параметр
# API — целая исходная строка в окне читалась как абзац, а не как контекст.
KWIC_WIDTH = 3


def kwic(words: list[str], hit_index: int,
         width: int = KWIC_WIDTH) -> tuple[str, str]:
    before = words[max(0, hit_index - width):hit_index]
    after = words[hit_index + 1:hit_index + 1 + width]
    return " ".join(before), " ".join(after)


def record_lexical_correction(root: Path | str, lexeme_id: str | None,
                              lemma: str | None, gloss: str,
                              gloss_language: str = "en",
                              confirmed: tuple[str, ...] | list[str] = (),
                              excluded: tuple[str, ...] | list[str] = (),
                              note: str = "",
                              part_of_speech: str | None = None,
                              stem_form: str | None = None) -> dict[str, Any]:
    """Record an author-confirmed lexeme meaning, applied to every form.

    Written to `reusable_lexical_gloss.jsonl`. Unlike
    `labels.record_correction`, there is no segmentation to reconstruct and
    no morpheme count to match: this layer carries a display gloss only.
    """
    root = Path(root)
    lexeme_id = (lexeme_id or "").strip() or None
    lemma = nfc(lemma).strip()
    gloss = nfc(gloss).strip()
    gloss_language = (gloss_language or "en").strip() or "en"
    if not lexeme_id and not lemma:
        raise ValueError("lexeme_id or lemma is required")
    if not gloss:
        raise ValueError("gloss is required")
    if "?" in gloss or "*" in gloss:
        raise ValueError("an uncertain gloss cannot become a reusable correction")
    if not _looks_lexical(gloss):
        raise ValueError(
            "this looks like a grammatical label, not a lexical meaning")
    if lemma.startswith("["):
        raise ValueError("a non-lexical corpus group has no lexical identity")
    if part_of_speech is not None and part_of_speech not in PARTS_OF_SPEECH:
        raise ValueError(f"part_of_speech must be one of {sorted(PARTS_OF_SPEECH)} or null")

    confirmed_set = sorted({nfc(s).strip() for s in confirmed if nfc(s).strip()})
    excluded_set = sorted({nfc(s).strip() for s in excluded if nfc(s).strip()})
    row = {
        "addition_id": stable_id("rl", lexeme_id or lemma, gloss),
        "lexeme_id": lexeme_id,
        "lemma_canonical": lemma or None,
        "lemma_head": _head(lemma) if lemma else None,
        "gloss": gloss,
        "gloss_language": gloss_language,
        "confirmed_surfaces": confirmed_set,
        "excluded_surfaces": excluded_set,
        "part_of_speech": part_of_speech,
        "stem_form": nfc(stem_form or "").strip() or None,
        "scope": "reusable_lexical",
        "explicit_reuse": True,
        "basis": "author-confirmed in the gloss editor",
        "note": note or "recorded from the lexical gloss confirmation",
    }
    path = writable(root, LEXICAL_CORRECTIONS)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        matches = [r for r in active_readings(path)
                  if r.get("addition_id") == row["addition_id"]]
        if matches:
            last = matches[-1]
            unchanged = (
                last.get("gloss") == row["gloss"]
                and last.get("gloss_language") == row["gloss_language"]
                and set(last.get("confirmed_surfaces") or ()) == set(confirmed_set)
                and set(last.get("excluded_surfaces") or ()) == set(excluded_set)
                and last.get("part_of_speech") == part_of_speech
                and last.get("stem_form") == row.get("stem_form")
                and last.get("explicit_reuse"))
            if unchanged:
                return last
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def retract_lexical_correction(root: Path | str, addition_id: str) -> dict:
    """Retract one lexical reading, by the same tombstone convention as
    `labels.retract_correction`."""
    from uuid import uuid4
    path = writable(root, LEXICAL_CORRECTIONS)
    if not any(row.get("addition_id") == addition_id
              for row in active_readings(path)):
        return {"target_addition_id": addition_id, "already_inactive": True}
    row = {"addition_id": f"retraction_{uuid4().hex}", "scope": "retracted",
           "target_addition_id": addition_id,
           "note": "Explicitly undone from the originating editor token"}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def active_lexical_glosses(root: Path | str) -> dict[str, dict]:
    """lexeme_id → the latest active reusable reading for it.

    The journal can hold several distinct readings for one lexeme over time
    (a corrected gloss gets a new `addition_id`); iterating in file order and
    overwriting per `lexeme_id` keeps whichever was recorded last, matching
    `_from_reusable_morphology`'s "last line wins" rule.
    """
    path = writable(root, LEXICAL_CORRECTIONS)
    out: dict[str, dict] = {}
    for row in active_readings(path):
        key = row.get("lexeme_id")
        if key:
            out[key] = row
    return out


def active_lexical_senses(root: Path | str) -> dict[str, list[dict]]:
    """lexeme_id → every active author-confirmed sense for that lexeme.

    ``active_lexical_glosses`` deliberately remains the compatibility view
    used by the old single-sense resolver (latest row wins).  The Turoyo/FLEX
    lexicon, however, is polysemous: one lemma can have several meanings, and
    the vocabulary window must not erase the older ones merely because a new
    sense was added.  This view therefore preserves all active rows.
    """
    path = writable(Path(root), LEXICAL_CORRECTIONS)
    out: dict[str, list[dict]] = {}
    for row in active_readings(path):
        key = row.get("lexeme_id")
        if key:
            out.setdefault(key, []).append(row)
    return out



def _text_summary_fingerprint(store, analyzer) -> tuple[str, list[Path]]:
    """Cheap invalidation key for the saved-text lexicon summary.

    Session JSON can be hundreds of MB after a full FLEXTEXT import.  Looking
    only at filenames/size/mtime lets the vocabulary page decide whether its
    compact cache is current without parsing those large files.  Lexeme ids are
    included because importing/replacing LIFT can turn a previously unresolved
    token id into a known lexical entry even when no session file changed.
    """
    paths = [path for path in sorted(store.dir.glob("*.json"))
             if path.is_file() and not path.name.startswith(".")] if store.dir.exists() else []
    digest = hashlib.sha256()
    for path in paths:
        try:
            stat = path.stat()
        except OSError:
            continue
        digest.update(path.name.encode("utf-8", "surrogatepass"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b":")
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        digest.update(b"\0")
    digest.update(b"|lexemes|")
    for lexeme_id in sorted(getattr(analyzer, "lexemes", {})):
        digest.update(str(lexeme_id).encode("utf-8", "surrogatepass"))
        digest.update(b"\0")
    return digest.hexdigest(), paths


def text_lexicon_summary(store, analyzer) -> dict[str, Any]:
    """One-pass saved-text index for the vocabulary page, persistently cached.

    Full imported FLEXTEXT sessions are large.  Re-reading 60k+ tokens merely
    to open the lexicon made that page noticeably slow.  The first call after a
    text/LIFT change scans each live session exactly once and writes a compact
    summary; subsequent page loads validate it using file metadata and read the
    compact cache instead.  Full KWIC contexts are still lazy and are read only
    after the author clicks a lexical entry/sense.
    """
    from collections import Counter
    from maalula_glosser.editor.relink import host_stem
    from maalula_glosser.gloss_shape import lexical_head as _head

    fingerprint, paths = _text_summary_fingerprint(store, analyzer)
    cache_path = writable(store.root, "data/editor_indexes/lexicon_summary.json")
    cached = read_json(cache_path, None)
    if isinstance(cached, dict) and cached.get("fingerprint") == fingerprint:
        titles = {lexeme_id: set(values or [])
                  for lexeme_id, values in (cached.get("titles") or {}).items()}
        unresolved = {}
        for lexeme_id, info in (cached.get("unresolved") or {}).items():
            restored = dict(info or {})
            restored["titles"] = set(restored.get("titles") or [])
            restored["candidates"] = set(restored.get("candidates") or [])
            unresolved[lexeme_id] = restored
        return {
            "titles": titles,
            "senses": cached.get("senses") or {},
            "glosses": cached.get("glosses") or {},
            "unresolved": unresolved,
        }

    titles: dict[str, set[str]] = {}
    sense_counts: dict[str, Counter[str]] = {}
    unresolved: dict[str, dict[str, Any]] = {}
    # Do not call ``store.list()`` here: list() parses every full session JSON
    # just to obtain metadata, after which load() would parse every file a
    # second time.  Walk each live session file exactly once.
    for path in paths:
        try:
            document = store.load(path.stem)
        except (FileNotFoundError, ValueError):
            continue
        title = document.get("title") or "(без названия)"
        for sentence in document.get("sentences") or []:
            for token in sentence.get("tokens") or []:
                lexeme_id = token.get("lemma_id")
                if not lexeme_id:
                    continue
                titles.setdefault(lexeme_id, set()).add(title)
                gloss = _head(token.get("lexical_gloss") or "")
                if not gloss:
                    _stem, gloss = host_stem(token.get("morphemes") or [])
                    gloss = _head(gloss or "")
                if gloss:
                    sense_counts.setdefault(lexeme_id, Counter())[gloss] += 1
                if lexeme_id in analyzer.lexemes:
                    continue
                lemma = nfc(token.get("lemma") or token.get("normalized")
                            or token.get("surface") or "")
                if not looks_like_a_lemma(lemma):
                    continue
                entry = unresolved.setdefault(lexeme_id, {
                    "lemma": lemma, "gloss": gloss or token.get("lexical_gloss"),
                    "titles": set(), "candidates": set(),
                })
                if not entry["gloss"] and gloss:
                    entry["gloss"] = gloss
                entry["titles"].add(title)
                entry["candidates"].update(token.get("lexeme_candidates") or ())

    senses = {lexeme_id: [
                {"gloss": gloss, "count": count, "source": "text"}
                for gloss, count in counter.most_common()]
              for lexeme_id, counter in sense_counts.items()}
    glosses = {lexeme_id: counter.most_common(1)[0][0]
               for lexeme_id, counter in sense_counts.items() if counter}

    serial_unresolved = {
        lexeme_id: {
            **{key: value for key, value in info.items()
               if key not in {"titles", "candidates"}},
            "titles": sorted(info.get("titles") or []),
            "candidates": sorted(info.get("candidates") or []),
        }
        for lexeme_id, info in unresolved.items()
    }
    write_json(cache_path, {
        "version": 1,
        "fingerprint": fingerprint,
        "titles": {lexeme_id: sorted(values) for lexeme_id, values in titles.items()},
        "senses": senses,
        "glosses": glosses,
        "unresolved": serial_unresolved,
    })
    return {"titles": titles, "senses": senses, "glosses": glosses,
            "unresolved": unresolved}


def text_senses_by_lexeme(store) -> dict[str, list[dict[str, Any]]]:
    """All lexical meanings actually used in saved texts, with frequencies.

    Unlike ``text_glosses_by_lexeme`` this does not collapse a polysemous
    lexeme to its single most frequent meaning.  It backs the FLEX dictionary
    article where meanings are children of the lemma.
    """
    from collections import Counter
    from maalula_glosser.gloss_shape import lexical_head as _head

    counts: dict[str, Counter[str]] = {}
    for listed in store.list():
        document = store.load(listed["document_id"])
        for sentence in document.get("sentences") or []:
            for token in sentence.get("tokens") or []:
                lexeme_id = token.get("lemma_id")
                if not lexeme_id:
                    continue
                gloss = _head(token.get("lexical_gloss") or "")
                if not gloss:
                    # ``lexical_gloss`` can lag one direct line-3 edit; read
                    # the host stem as a fallback so the dictionary reflects
                    # what the author actually sees in the text.
                    from maalula_glosser.editor.relink import host_stem
                    _stem, gloss = host_stem(token.get("morphemes") or [])
                if gloss:
                    counts.setdefault(lexeme_id, Counter())[gloss] += 1
    return {lexeme_id: [
                {"gloss": gloss, "count": count, "source": "text"}
                for gloss, count in counter.most_common()]
            for lexeme_id, counter in counts.items()}


def active_lexical_glosses_by_lemma(root: Path | str) -> dict[str, dict]:
    """lemma string (full citation and head form) → latest active reading.

    For the string-keyed resolver path (`LexicalGlossResolver`), which has no
    `lexeme_id` to key on for every source it joins.
    """
    path = writable(root, LEXICAL_CORRECTIONS)
    out: dict[str, dict] = {}
    for row in active_readings(path):
        for key in (row.get("lemma_canonical"), row.get("lemma_head")):
            if key:
                out[key] = row
    return out


def _igt_index(root: Path) -> dict[str, dict[str, Any]]:
    """surface (casefolded) → one corpus sentence mentioning it.

    Only the first sentence found is kept — this is a checklist aid, not an
    exhaustive concordance (that stays `find_corpus_sentences`'s job).
    """
    index: dict[str, dict[str, Any]] = {}
    path = root / IGT_EXAMPLES
    if not path.exists():
        return index
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        example = json.loads(line)
        transcription = nfc(example.get("transcription_original") or "")
        if not transcription:
            continue
        translation = nfc(example.get("translation_en")
                          or example.get("translation_original") or "")
        surfaces = [nfc(token.get("surface_canonical") or "")
                    for token in example.get("tokens") or []]
        for position, surface in enumerate(surfaces):
            key = surface.casefold()
            if key and key not in index:
                before, after = kwic(surfaces, position)
                index[key] = {"sentence": transcription,
                              "translation": translation,
                              "context_before": before,
                              "context_after": after}
    return index


def text_occurrences_by_lexeme(store, lexeme_id: str) -> list[dict[str, Any]]:
    """Every saved-text occurrence of one lexeme, with enough context for
    the vocabulary window to be a real concordance rather than only a jump
    list.

    Besides the KWIC snippet we return the complete source segment, its
    translation and FLEx ``segnum`` when present.  ``lexical_gloss`` records
    the sense actually used on this token, so a polysemous lemma can show
    contexts for one sense (``gab`` → ``chez``) without mixing in its sibling
    sense (``gab`` → ``lat``).  Older sessions can have a stale/empty
    ``token.lexical_gloss`` after a direct line-3 edit, therefore the host
    stem is used as the same fallback as ``text_senses_by_lexeme``.
    """
    from maalula_glosser.editor.relink import host_stem

    hits = []
    for listed in store.list():
        document = store.load(listed["document_id"])
        for sentence_index, sentence in enumerate(document.get("sentences") or []):
            # Метка спикера — тоже токен (`session.mark_speaker`), и в окно
            # конкорданса ей попадать нельзя: автор жаловался именно на
            # «MS: …» в строке словника.
            spoken = [token for token in sentence.get("tokens") or []
                      if not is_speaker(token)]
            surfaces = [nfc(token.get("surface") or "") for token in spoken]
            for position, token in enumerate(spoken):
                if token.get("lemma_id") != lexeme_id:
                    continue
                before, after = kwic(surfaces, position)
                morphs = token.get("morphemes") or []
                _stem, stem_gloss = host_stem(morphs)
                lexical_gloss = nfc(token.get("lexical_gloss") or stem_gloss or "")
                flex = sentence.get("flextext") or {}
                hits.append({
                    "source": "text",
                    "surface_canonical": nfc(token.get("surface") or ""),
                    "document_id": document["document_id"],
                    "title": document.get("title") or "(без названия)",
                    "token_id": token.get("token_id"),
                    "sentence_id": sentence.get("sentence_id"),
                    "sentence_index": sentence_index,
                    "line": nfc(sentence.get("source_line") or ""),
                    "translation": nfc(sentence.get("translation") or ""),
                    "segnum": nfc(flex.get("segnum") or ""),
                    "segmentation": join_parts(morphs),
                    "gloss": join_parts(morphs, "gloss"),
                    "lexical_gloss": lexical_gloss,
                    "grammatical_info": token.get("grammatical_info"),
                    "grammatical_infos": list(token.get("grammatical_infos") or []),
                    "lift_entry_id": token.get("lift_entry_id"),
                    "lift_sense_id": token.get("lift_sense_id"),
                    "context_before": before,
                    "context_after": after,
                })
    return hits


def text_titles_by_lexeme(store) -> dict[str, set[str]]:
    """lexeme_id → titles of the saved texts it occurs in, one pass over the
    whole store. Cheap summary for the senses-window filter bar (which text
    to show occurrences from) and the source checkboxes it builds from —
    the per-lexeme detail (`document_id`/`token_id`/line) still comes from
    `text_occurrences_by_lexeme`, fetched lazily only when a lexeme's
    concordance is actually opened."""
    out: dict[str, set[str]] = {}
    # Do not call ``store.list()`` here: list() parses every full session JSON
    # just to obtain metadata, after which load() would parse every file a
    # second time.  The lexicon page can be the hottest read path on a 60k+
    # token FLEX corpus, so walk the live session files once and load each
    # document exactly once.
    paths = sorted(store.dir.glob("*.json")) if store.dir.exists() else []
    for path in paths:
        try:
            document = store.load(path.stem)
        except (FileNotFoundError, ValueError):
            continue
        title = document.get("title") or "(без названия)"
        for sentence in document.get("sentences") or []:
            for token in sentence.get("tokens") or []:
                lexeme_id = token.get("lemma_id")
                if lexeme_id:
                    out.setdefault(lexeme_id, set()).add(title)
    return out


def text_glosses_by_lexeme(store) -> dict[str, str]:
    """lexeme_id -> the meaning the saved texts give it most often (the
    token's `lexical_gloss`, stem part only). For a lexeme with no meaning in
    the dictionary yet: the Словник shows it with this as a suggestion
    instead of hiding it (afhem yafhem bound by hand, 'understand')."""
    from collections import Counter
    from maalula_glosser.gloss_shape import lexical_head as _head
    counts: dict[str, Counter] = {}
    for listed in store.list():
        document = store.load(listed["document_id"])
        for sentence in document.get("sentences") or []:
            for token in sentence.get("tokens") or []:
                lexeme_id = token.get("lemma_id")
                gloss = (token.get("lexical_gloss") or "").split("=")[0].split("-")[0]
                gloss = _head(gloss) if gloss else ""
                if lexeme_id and gloss:
                    counts.setdefault(lexeme_id, Counter())[gloss] += 1
    return {lexeme_id: c.most_common(1)[0][0] for lexeme_id, c in counts.items()}


def unresolved_text_lexemes(store, analyzer) -> dict[str, dict[str, Any]]:
    """lexeme_id -> {"lemma", "gloss", "titles"} for a lexeme that occurs in
    the user's saved texts but is neither in the pre-built corpus lexicon nor
    (checked separately by the caller) already an author-confirmed journal
    entry — freshly imported vocabulary the analyser never saw at all, which
    `session.overlay_user_igt` mints its own `lexnew_...` lemma_id for at
    import time (docs/LEXICAL_REUSABLE_SPEC_RU.md) with nowhere else to be
    discovered from. One pass over the whole store, like
    `text_titles_by_lexeme`, whose job this deliberately does not replace:
    that function still backs the filter bar for lexemes the corpus already
    knows."""
    out: dict[str, dict[str, Any]] = {}
    for listed in store.list():
        document = store.load(listed["document_id"])
        title = document.get("title") or "(без названия)"
        for sentence in document.get("sentences") or []:
            for token in sentence.get("tokens") or []:
                lexeme_id = token.get("lemma_id")
                if not lexeme_id or lexeme_id in analyzer.lexemes:
                    continue
                lemma = nfc(token.get("lemma") or token.get("normalized")
                           or token.get("surface") or "")
                # An Arabic (ᴬ…ᴬ) insert and a bare segmentation string are
                # never a dictionary entry: they used to land in the
                # vocabulary window as junk.
                if not looks_like_a_lemma(lemma):
                    continue
                entry = out.setdefault(lexeme_id, {
                    "lemma": lemma,
                    "gloss": token.get("lexical_gloss"),
                    "titles": set(),
                    "candidates": set(),
                })
                if not entry["gloss"] and token.get("lexical_gloss"):
                    entry["gloss"] = token.get("lexical_gloss")
                entry["titles"].add(title)
                # Не всякое вхождение одной и той же lexnew_-лексемы прошло
                # через резолвер (документ мог быть сохранён до Task 6/8), так
                # что кандидатов может не быть у части токенов — берём
                # объединение по всем.
                entry["candidates"].update(token.get("lexeme_candidates") or ())
    return out


def queue_row(lexeme_id: str, info: dict[str, Any]) -> dict[str, Any]:
    """Строка очереди «новые слова, ещё не в словаре».

    Это не статья словника: формы у неё ещё не заведены, поэтому счётчик
    `None`, а не `0` — «0 форм» читалось как «слово без форм», хотя означало
    «слово ещё не в словарике»."""
    return {"lexeme_id": lexeme_id, "lemma": info["lemma"],
            "gloss": info.get("gloss"), "gloss_language": "en",
            "form_count": None, "excluded_count": 0, "note": None,
            "source": "queue", "needs_citation_form": True,
            "text_titles": sorted(info["titles"]), "part_of_speech": None,
            "candidates": sorted(info.get("candidates") or ())}


def sibling_forms(root: Path | str, analyzer, lexeme_id: str,
                  document: dict[str, Any] | None = None,
                  limit: int = FORM_LIMIT, store=None) -> dict[str, Any]:
    """Every attested form of one lexeme, for the confirm-time checklist and
    for the senses-window concordance.

    Source for `forms` is `data/final/forms.jsonl` (via `analyzer.forms_by_lexeme`),
    not `dictionary_morphology_v1/inflected_forms.jsonl` — the spec's §2 found
    the latter to share only 25 of 3787 lemmas with this lexeme-id namespace.
    Each form is joined to its analysis (segmentation/gloss) where
    `selected_analysis_id` resolves one — true for only ~5% of forms, so most
    rows carry a bare surface and frequency, which is the normal case here,
    not an error.

    The analysis is read from `analyzer.a1` (already repaired at construction
    from `gold_nc_prs_corrections.jsonl` — the cancelled blanket NC→PRF rule),
    never from a fresh re-read of `data/final/analyses.jsonl`: that file still
    carries the old mistake verbatim, which is exactly how a corpus form once
    showed `write.PRF` here when the live analyzer had already settled on
    `write.NC` for the same word (`docs/LEXICAL_REUSABLE_SPEC_RU.md` §12).
    `classify_prf_analysis`/`render_authoritative_gloss` are then the same
    two display-only steps `labels.py::corpus_rows` already applies — nothing
    here touches generation, and an already-decided NC1/NC2 subclass is
    preserved rather than collapsed to bare NC.

    `store`, when given, adds `text_occurrences`: this lexeme's own hits in
    the user's saved texts, each clickable straight to its token (§11 —
    the corpus/dictionary rows below only ever offer to *start* a text).
    """
    from maalula_glosser.morphology.perfect_subclass import (
        classify_prf_analysis, replace_prf_label,
    )
    from maalula_glosser.normalization.gloss import render_authoritative_gloss

    root = Path(root)
    forms = list(getattr(analyzer, "forms_by_lexeme", {}).get(lexeme_id, []))
    lexeme = analyzer.lexemes.get(lexeme_id) or {}

    analyses_by_id: dict[str, dict] = {}
    if any(f.get("selected_analysis_id") for f in forms):
        analyses_by_id = {a["analysis_id"]: a
                          for entries in getattr(analyzer, "a1", {}).values()
                          for a in entries if a.get("analysis_id")}
    igt = _igt_index(root)

    document_surfaces: set[str] = set()
    if document is not None:
        for sentence in document.get("sentences") or []:
            for token in sentence.get("tokens") or []:
                if token.get("lemma_id") == lexeme_id:
                    for key in ("surface", "normalized"):
                        value = nfc(token.get(key) or "").casefold()
                        if value:
                            document_surfaces.add(value)

    rows = []
    for form in forms:
        surface = nfc(form.get("form_canonical") or "")
        if not surface:
            continue
        analysis = analyses_by_id.get(form.get("selected_analysis_id") or "")
        context = igt.get(surface.casefold())
        segmentation = analysis.get("segmentation") if analysis else None
        gloss = analysis.get("gloss_clean") if analysis else None
        if gloss:
            # Same conservative, decisive-cases-only classification _from_a1
            # applies live; an existing NC1/NC2 already in the analysis is
            # left exactly as it is — this only ever adds the subclass, never
            # removes or generalizes one that is already there.
            perfect = classify_prf_analysis(lexeme, segmentation, gloss)
            if perfect:
                gloss = replace_prf_label(gloss, perfect[0])
            gloss = render_authoritative_gloss(gloss)
        # Most rows have no per-form analysis (only ~5% of forms carry a
        # resolved selected_analysis_id — see the module docstring above),
        # but the lexeme's own meaning is usually still known from OTHER
        # confirmed ("green") corpus analyses. Show that lexical gloss even
        # when this exact surface has no segmentation of its own — it is a
        # display value only, never a stand-in for `gloss`/`segmentation`.
        lexical = getattr(analyzer, "lexical", None)
        lexical_gloss, lexical_gloss_status = (
            lexical.resolve(lexeme_id, surface) if lexical is not None
            else (None, "unknown_lexeme"))
        rows.append({
            "source": "corpus",
            "surface_canonical": surface,
            "frequency": form.get("frequency") or 0,
            "segmentation": segmentation,
            "gloss": gloss,
            "lexical_gloss": lexical_gloss,
            "lexical_gloss_status": lexical_gloss_status,
            "grammatical_info": (form.get("grammatical_info")
                                  or getattr(analyzer, "grammatical_info", lambda *_: None)(
                                      lexeme_id, lexical_gloss)),
            "in_document": surface.casefold() in document_surfaces,
            "context": context,
        })

    # Forms an author can actually judge (analysis or corpus sentence, or
    # already in the open text) first; frequency breaks ties. Open question
    # #5 leaves both the ordering and the cutoff for later tuning.
    def sort_key(row):
        judgeable = row["segmentation"] or row["context"] or row["in_document"]
        return (0 if judgeable else 1, -row["frequency"])

    rows.sort(key=sort_key)
    truncated = len(rows) > limit
    rows = rows[:limit]

    return {
        "lexeme_id": lexeme_id,
        "lemma": lexeme.get("lemma_canonical"),
        "forms": rows,
        "text_occurrences": (text_occurrences_by_lexeme(store, lexeme_id)
                             if store is not None else []),
        "truncated": truncated,
        "total_forms": len(forms),
    }


# ---------------------------------------------------------------------------
# Part-of-speech display hint
#
# Not morphology in the sense the module docstring warns about — the journal
# still carries a display gloss only. This is read-only metadata for the
# senses window, inferred from labels a lexeme's OWN confirmed corpus
# analyses already carry, never written back into any gloss or analysis.
# ---------------------------------------------------------------------------

_GLOSS_INVENTORY = "config/gloss_inventory.json"


def _label_categories(root: Path) -> tuple[frozenset, frozenset, frozenset, frozenset]:
    """(verbal TAM labels, nominal-state labels, feminine, masculine) — read
    from the curated `config/gloss_inventory.json`, not hard-coded, so this
    tracks the label set the project actually uses."""
    path = Path(root) / _GLOSS_INVENTORY
    if not path.exists():
        return frozenset(), frozenset(), frozenset(), frozenset()
    data = json.loads(path.read_text(encoding="utf-8"))
    labels = data.get("labels") or {}
    category = {key: (value or {}).get("category") for key, value in labels.items()}
    verbal = frozenset(k for k, c in category.items() if c == "conjugation/aspect")
    # HD (construct/"head" state) is filed under "syntax/object" in the
    # inventory, but it is a nominal-state marker in this project's own usage
    # (docs/GLOSS_EDITOR_SPEC_RU.md: HD replaces the old CNS construct-state
    # label) — added explicitly rather than by category.
    nominal = frozenset(k for k, c in category.items() if c == "nominal state") | {"HD"}
    gendered = frozenset(k for k, c in category.items() if c == "person/gender/number")
    # Confirmed analyses retain internal TAM names (SBJV, PRS) while the editor
    # displays their project labels (PC, NC).  POS inference reads the stored
    # analyses, so aliases whose display target belongs to a category must
    # belong to that category too.
    display_map = data.get("display_map") or {}
    verbal = verbal | frozenset(alias for alias, target in display_map.items()
                                if target in verbal)
    nominal = nominal | frozenset(alias for alias, target in display_map.items()
                                  if target in nominal)
    gendered = gendered | frozenset(alias for alias, target in display_map.items()
                                    if target in gendered)
    feminine = frozenset(k for k in gendered if "F" in k.upper())
    masculine = frozenset(k for k in gendered if "M" in k.upper())
    return verbal, nominal, feminine, masculine


def _analysis_labels(analysis: dict) -> set[str]:
    """Every grammatical label one confirmed analysis attaches to its word.

    A stem morpheme's gloss is `lexical.TAM` (`"go.NC"`); a non-stem
    morpheme's gloss already IS a bare label (`"3MP"`, `"HD"`). Only the part
    after the lexical meaning is a grammatical label.
    """
    labels: set[str] = set()
    for morph in analysis.get("morphemes") or []:
        gloss = morph.get("gloss") or ""
        parts = gloss.split(".")
        if morph.get("type") == "stem":
            labels.update(p for p in parts[1:] if p)
        else:
            labels.update(p for p in parts if p)
    return labels


def analyses_by_lexeme_index(root: Path) -> dict[str, list[dict]]:
    # as the author corrected them: an analysis moved to another lexeme
    # (ʕam=tōkek is atak, not iṭḥan) must not vote for the old one
    from maalula_glosser.gold_corrections import GoldCorrections
    out: dict[str, list[dict]] = {}
    for analysis in GoldCorrections.load(root).corrected(
            iter_jsonl(Path(root) / "data/final/analyses.jsonl")):
        if analysis.get("status") != "confirmed":
            continue
        lemma_id = analysis.get("lemma_id")
        if lemma_id:
            out.setdefault(lemma_id, []).append(analysis)
    return out


_VERBAL_LEXEMES = "config/verbal_lexemes.json"


def curated_verbal_lexemes(root: Path | str) -> dict[str, str]:
    """lexeme_id -> "verb"/"not_verb" (D1, HANDOFF_verbal_lexemes.md). Same
    file and same priority `compositional.py`'s POS vote gives it, so the
    словник shows the part of speech the analyzer actually used."""
    path = Path(root) / _VERBAL_LEXEMES
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("verbal_lexemes") or {}


def infer_part_of_speech(root: Path | str, lexeme_id: str,
                         analyses_by_lexeme: dict[str, list[dict]] | None = None,
                         lemma: str | None = None,
                         curated: dict[str, str] | None = None,
                         ) -> str | None:
    """verb / adjective / noun / None. D1 (HANDOFF_verbal_lexemes.md): the
    curated list, then the citation-form shape (`iqtal yiqtul`), are checked
    first — ahead of the corpus-label heuristic below — so this agrees with
    the part-of-speech vote `compositional.py`'s analyzer makes for the same
    lexeme. Only past those does an exclusively verbal TAM label settle it;
    only otherwise does attested gender variation suggest an adjective; only
    otherwise does a nominal-state label suggest a noun. Anything short of
    that stays `None` rather than guessing.
    """
    root = Path(root)
    if curated is None:
        curated = curated_verbal_lexemes(root)
    decision = curated.get(lexeme_id)
    if decision == "verb":
        return "verb"
    if decision != "not_verb" and citation_is_verbal(lemma):
        return "verb"
    if analyses_by_lexeme is None:
        analyses_by_lexeme = analyses_by_lexeme_index(root)
    verbal, nominal, feminine, masculine = _label_categories(root)
    labels: set[str] = set()
    for analysis in analyses_by_lexeme.get(lexeme_id) or []:
        labels |= _analysis_labels(analysis)
    if not labels:
        return None
    # A curated "not_verb" vetoes only the verb reading; noun/adjective are
    # still free to come from the same labels.
    if decision == "not_verb":
        verbal = frozenset()
    return pos_from_labels(labels, verbal, nominal, feminine, masculine)


def resolved_part_of_speech(root: Path | str, lexeme_id: str,
                            active_row: dict | None,
                            analyses_by_lexeme: dict[str, list[dict]] | None = None,
                            lemma: str | None = None,
                            curated: dict[str, str] | None = None,
                            ) -> str | None:
    """An author's explicit override wins over the heuristic — the senses
    window's own "изменить часть речи" action writes `part_of_speech` onto
    the journal row precisely so it can override a wrong guess."""
    if active_row and active_row.get("part_of_speech"):
        return active_row["part_of_speech"]
    return infer_part_of_speech(root, lexeme_id, analyses_by_lexeme,
                                lemma=lemma, curated=curated)


# ---------------------------------------------------------------------------
# Load in one go: every lexeme the project already has a resolved reading for
# ---------------------------------------------------------------------------

def resolvable_lexemes(analyzer) -> dict[str, tuple[str, str]]:
    """lexeme_id → (gloss, source status), for every lexeme with SOME
    resolved lexical meaning that is not yet an author-confirmed reading —
    i.e. everything `LexicalGloss.resolve` would already show in the editor
    from `PROJECT`/`MASC`/corpus tiers, browsable here before anyone commits
    it to the reusable layer."""
    out: dict[str, tuple[str, str]] = {}
    for lexeme_id in analyzer.lexemes:
        gloss, status = analyzer.lexical.resolve(lexeme_id)
        if gloss and status not in ("author_confirmed", "nonlexical_group",
                                    "unknown_lexeme", "no_sense",
                                    "translation_pending"):
            out[lexeme_id] = (gloss, status)
    return out


def seed_from_corpus(root: Path | str, analyzer,
                     note: str = "bulk-loaded from existing corpus resolution"
                     ) -> list[dict]:
    """Promote every not-yet-confirmed resolvable lexeme into the reusable
    layer in one action — the "load them all in one go" the window offers.

    Each becomes an ordinary journal row (same validation, same retract path
    as a one-by-one confirmation) so nothing here is a special case to
    maintain: undoing one bulk-seeded entry is exactly `retract_lexical_correction`.
    """
    already = active_lexical_glosses(root)
    written = []
    for lexeme_id, (gloss, _status) in resolvable_lexemes(analyzer).items():
        if lexeme_id in already:
            continue
        lexeme = analyzer.lexemes.get(lexeme_id) or {}
        try:
            row = record_lexical_correction(
                root, lexeme_id, lexeme.get("lemma_canonical"), gloss, note=note)
        except ValueError:
            continue  # e.g. an all-caps or otherwise non-lexical stray gloss
        analyzer.lexical.remember(row)
        written.append(row)
    return written
