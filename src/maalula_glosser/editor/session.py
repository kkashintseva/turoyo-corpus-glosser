"""Editor session model.

A session is the whole editable state of one text: three aligned lines per
sentence, the analyzer's alternatives, whatever the human changed, and their
comments. It round-trips through JSON so a document can be reopened exactly as
it was left.

The unit of interaction is the morpheme, not the token, so line 2 (morphemes)
and line 3 (gloss labels) are the same list of cells rendered twice. The
separator is stored *on* the morpheme that follows it, which makes the two
lines structurally incapable of drifting apart and keeps invariant 2 —
segmentation reconstructs the surface — true by construction.
"""
from __future__ import annotations

import copy
import re
import unicodedata
from difflib import SequenceMatcher
from functools import lru_cache
from datetime import datetime, timezone
from typing import Any, Iterable

from maalula_glosser.editor import arabic_marks
from maalula_glosser.editor import lexeme_resolve as LR
from maalula_glosser.bound_labels import typed_bound_options
from maalula_glosser.gloss_shape import has_lexical_part, lexical_head, looks_like_a_lemma
from maalula_glosser.io_utils import stable_id

SCHEMA = "maalula.editor.v1"

# `-`, `=` and `+` are all first-class morpheme boundaries.  `+` behaves like
# `=` for segmentation/clitic logic, but its spelling is significant and must
# round-trip unchanged through the editor, reusable readings and exports.
# `.` is not a boundary — it joins a multi-word lexical gloss (`cool.down`) or
# a category to its stem (`make.PRF`).
SEPARATORS = "-=+"
_SPLIT_RE = re.compile(r"([-=+])")

CONFIDENT, UNCERTAIN, UNRESOLVED = "confident", "uncertain", "unresolved"
STATUSES = (CONFIDENT, UNCERTAIN, UNRESOLVED)
STATUS_BY_CONFIDENCE = {"HIGH": CONFIDENT, "MEDIUM": UNCERTAIN, "LOW": UNRESOLVED}

# What a cell shows when the program produced nothing at all, as opposed to
# producing something it is merely unsure of.
UNRESOLVED_MARK = "??"

AUTO, USER = "auto", "user"

# A technical gloss, not a linguistic one: it marks who is speaking. The tag
# belongs to the transcription and must stay in line 1, but it is not a word of
# the language, so it is kept out of the morpheme and gloss rows on export.
SPEAKER_GLOSS = "#speaker"


def _uppercase_letter(char: str) -> bool:
    """A cased Unicode letter which is actually uppercase."""
    return (unicodedata.category(char).startswith("L")
            and char.upper() == char and char.lower() != char)


def looks_like_speaker_tag(text: str, followed: bool = False) -> bool:
    """Conservative automatic speaker guess: 1–2 capitals + ':' or '.'.

    Position is irrelevant: a change of speaker may occur in the middle of a
    transcription line. Longer labels such as ``Yawse:`` are never guessed,
    but the editor may still mark them explicitly as #speaker. ``followed`` is
    retained for API compatibility; it no longer changes the decision.

    ``A`` is the Arabic-span control and ``E`` is the ordinary affirmative
    'yes'. They are the only one-letter capitals allowed as ordinary text here,
    so A./A:/E./E: are never guessed as speakers.
    """
    text = nfc(text).strip()
    if text in {"A.", "A:", "E.", "E:"}:
        return False
    if len(text) < 2 or text[-1] not in ":.":
        return False
    letters = text[:-1]
    return 1 <= len(letters) <= 2 and all(_uppercase_letter(ch) for ch in letters)


def is_speaker(token: dict[str, Any]) -> bool:
    return any((m.get("gloss") or "") == SPEAKER_GLOSS
               for m in token.get("morphemes") or [])


def mark_speaker(token: dict[str, Any], origin: str = AUTO) -> dict[str, Any]:
    """Collapse a speaker tag to one cell carrying the technical gloss."""
    token["morphemes"] = [{
        "morph_id": token["morphemes"][0]["morph_id"],
        "sep": "", "form": token["surface"], "gloss": SPEAKER_GLOSS,
        "suggested": "", "status": CONFIDENT, "origin": origin,
        "alternatives": [],
    }]
    return token


def _arabic_marker_token_indexes(tokens: list[dict[str, Any]]) -> set[int]:
    """Token indexes covered by a complete A…A Arabic span."""
    surfaces = [nfc(token.get("surface") or "") for token in tokens]
    joined = " ".join(surfaces)
    spans = arabic_marks.spans(joined)
    if not spans:
        return set()
    blocked: set[int] = set()
    cursor = 0
    for index, surface in enumerate(surfaces):
        start, stop = cursor, cursor + len(surface)
        # arabic_marks.spans returns the character offset of the closing A,
        # so treat the pair as an inclusive interval [open, close].
        if any(start <= close and stop > open_ for open_, close in spans):
            blocked.add(index)
        cursor = stop + 1
    return blocked


def mark_automatic_speakers(tokens: list[dict[str, Any]]) -> None:
    """Apply the conservative automatic rule anywhere in a line."""
    arabic_tokens = _arabic_marker_token_indexes(tokens)
    for index, token in enumerate(tokens):
        if index in arabic_tokens:
            continue
        if token.get("locked") or any(m.get("origin") == USER for m in token.get("morphemes") or []):
            continue
        surface = token.get("surface") or ""
        # In this transcription convention a one/two-capital token ending in
        # ':' or '.' is a speaker tag.  Apply that structural fact before a
        # lower-cased lexical analysis can steal Ḥ./M. etc.; A/E were excluded
        # above because they may be ordinary one-letter text.
        if looks_like_speaker_tag(surface):
            mark_speaker(token)



def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


def morphology_text(text: str) -> str:
    """Canonicalize text used by the morphology layer.

    Undertie belongs to transcription, so morphology sees it as a word space.
    Morpheme-boundary spelling is otherwise preserved: in particular ``+`` is
    *not* rewritten to ``=``.  Line 1/source transcription is never passed
    through this function and therefore is left untouched too.
    """
    return nfc(text).replace("‿", " ")


def normalize_morphology(document: dict[str, Any]) -> None:
    """Read old tied cells compatibly, without changing line 1 or identities."""
    def normalize(item):
        for key in ("sep", "form", "gloss", "suggested", "segmentation", "unassigned_gloss"):
            if isinstance(item.get(key), str):
                item[key] = morphology_text(item[key])
        for alternative in item.get("alternatives") or []:
            normalize(alternative)
    for sentence in document.get("sentences") or []:
        for token in sentence.get("tokens") or []:
            for cell in token.get("morphemes") or []:
                normalize(cell)
            for alternative in token.get("alternatives") or []:
                normalize(alternative)


# ---------------------------------------------------------------------------
# splitting a token into morpheme cells
# ---------------------------------------------------------------------------
def split_parts(text: str) -> list[tuple[str, str]]:
    """Split a morphology string into cells.

    ``b=teps-a`` → ``[("", "b"), ("=", "teps"), ("-", "a")]``.
    ``b=u+atro`` → ``[("", "b"), ("=", "u"), ("+", "atro")]``.

    The separator preceding each part is kept with it, including the exact
    distinction between ``=``, ``+`` and ``-``.
    """
    if not text:
        return []
    pieces = _SPLIT_RE.split(morphology_text(text))
    parts = [("", pieces[0])]
    for index in range(1, len(pieces) - 1, 2):
        parts.append((pieces[index], pieces[index + 1]))
    return parts


def join_parts(morphemes: Iterable[dict[str, Any]], key: str = "form") -> str:
    return morphology_text("".join(f"{m.get('sep', '')}{m.get(key) or ''}" for m in morphemes))


def boundary_class(separator: str) -> str:
    """Structural boundary class without changing the stored spelling.

    ``+`` and ``=`` are equivalent for alignment and clitic logic, but the
    exact separator from line 2 remains authoritative for rendering/storage.
    """
    return "=" if separator in {"=", "+"} else separator


def same_boundary_topology(left: Iterable[tuple[str, str]],
                           right: Iterable[tuple[str, str]]) -> bool:
    return [boundary_class(sep) for sep, _ in left] == [
        boundary_class(sep) for sep, _ in right
    ]


def morpheme_status(gloss: str | None, confidence: str | None) -> str:
    """A cell with no gloss at all is unresolved whatever the token score says."""
    if not gloss or not gloss.strip() or set(gloss.strip()) <= {"?", "*"}:
        return UNRESOLVED
    if "?" in gloss or "*" in gloss:
        return UNCERTAIN
    return STATUS_BY_CONFIDENCE.get((confidence or "").upper(), UNRESOLVED)


def build_morphemes(token_id: str, segmentation: str | None, gloss: str | None,
                    fallback: str, confidence: str | None,
                    case=None) -> list[dict[str, Any]]:
    """Line 2 and line 3 as one list of cells.

    When the analyzer produced nothing, or produced a segmentation and a gloss
    whose separator sequences disagree, the token degrades to unresolved cells
    rather than a silently misaligned pair of lines.

    `fallback` is the token as typed minus sentence punctuation, so a form the
    analyzer could not handle still breaks into the boundaries the user marked
    by hand: `b=Amērca` stays two glossable cells instead of one opaque blob.
    """
    # `case` rewrites agreement labels into project orthography (3MS → 3ms).
    # Purely orthographic, so it is safe on analyzer output, unlike the
    # substitutions in the inventory's display_map.
    if case is not None and gloss:
        gloss = case(gloss)
    seg_parts = split_parts(segmentation) if segmentation else []
    gloss_parts = split_parts(gloss) if gloss else []
    aligned = (bool(seg_parts) and len(seg_parts) == len(gloss_parts)
               and same_boundary_topology(seg_parts, gloss_parts))

    if not seg_parts:
        seg_parts, aligned = split_parts(nfc(fallback)) or [("", nfc(fallback))], False
    if not aligned:
        # keep the segmentation the analyzer did produce, but do not pretend to
        # know which label belongs to which piece
        gloss_parts = [(sep, "") for sep, _ in seg_parts]
        if len(seg_parts) == 1 and gloss and len(split_parts(gloss)) == 1:
            gloss_parts = [("", nfc(gloss))]

    cells = []
    for index, ((sep, form), (_, label)) in enumerate(zip(seg_parts, gloss_parts)):
        status = morpheme_status(label, confidence)
        # An unresolved cell always reads `??`, so "the program did not manage
        # this one" is never disguised as an answer. The label it did produce is
        # not thrown away: it becomes the first suggestion, one click from being
        # accepted, and it stays out of the exported gloss until then.
        suggested = ""
        if status == UNRESOLVED and label:
            suggested, label = label, ""
        cells.append({
            "morph_id": stable_id("morph", token_id, index, form),
            "sep": sep,
            "form": form,
            "gloss": label,
            "suggested": suggested,
            "status": status,
            "origin": AUTO,
            "alternatives": [],
        })
    if not aligned and gloss and not any(cell["gloss"] or cell["suggested"] for cell in cells):
        # Archive the unassignable original, not a one-click invalid proposal.
        cells[0]["unassigned_gloss"] = nfc(gloss)
    return cells


def token_alternatives(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Whole-token alternatives, trimmed to what the hover tooltip shows."""
    out = []
    for alternative in record.get("alternatives") or []:
        if not alternative.get("gloss"):
            continue
        out.append({"segmentation": alternative.get("segmentation"),
                    "gloss": alternative.get("gloss"),
                    "tier": alternative.get("tier"),
                    "lemma": alternative.get("lemma"),
                    "lemma_id": alternative.get("lemma_id"),
                    "grammatical_info": alternative.get("grammatical_info"),
                    "lift_entry_id": alternative.get("lift_entry_id"),
                    "lift_sense_id": alternative.get("lift_sense_id"),
                    "count": alternative.get("count"),
                    "examples": (alternative.get("examples") or [])[:3]})
    return out


# ---------------------------------------------------------------------------
# building a document
# ---------------------------------------------------------------------------
def build_token(record: dict[str, Any], sentence_id: str, index: int,
                case=None) -> dict[str, Any]:
    # A sandhi group is written as one word but analysed as several, and each
    # piece gets its own column. The whole group is carried on every piece, so
    # using it as the visible form printed `b‿ō‿blōta` three times across three
    # columns. Each column shows its own piece; the group is kept for provenance
    # and line 1 of the export still comes from the sentence, not from here.
    group = record.get("source_token") or ""
    if record.get("from_sandhi"):
        surface = record.get("token") or group
    else:
        surface = group or record.get("token") or ""
    token_id = stable_id("etok", sentence_id, index, surface)
    return {
        "token_id": token_id,
        "surface": nfc(surface),
        "sandhi_group": nfc(group) if record.get("from_sandhi") else "",
        "normalized": record.get("normalized"),
        "confidence": record.get("confidence"),
        "tier": record.get("tier"),
        "lemma": record.get("lemma"),
        "lemma_id": record.get("lemma_id"),
        "lexical_gloss": record.get("lexical_gloss"),
        "grammatical_info": record.get("grammatical_info"),
        "grammatical_infos": copy.deepcopy(record.get("grammatical_infos") or []),
        "lift_entry_id": record.get("lift_entry_id"),
        "lift_guid": record.get("lift_guid"),
        "lift_sense_id": record.get("lift_sense_id"),
        "lexeme_candidates": copy.deepcopy(record.get("lexeme_candidates") or []),
        "uncertainty": record.get("uncertainty") or [],
        "alternatives": token_alternatives(record),
        # A token the human has touched is never re-glossed behind their back
        # (invariant 3: human decisions outrank generation).
        "locked": False,
        # Snapshot of lines 2-3 kept alive while an automatic re-gloss is
        # highlighted and still refusable with Del.
        "auto_pending": None,
        # `token` is the piece the analyzer actually saw: typed boundaries kept,
        # sentence punctuation stripped. `surface` keeps the punctuation for
        # line 1 and the exported source paragraph.
        "morphemes": build_morphemes(token_id, record.get("segmentation"),
                                     record.get("gloss"),
                                     record.get("token") or surface,
                                     record.get("confidence"), case),
    }


def build_sentence(glossed: dict[str, Any], index: int,
                   document_id: str, translation: str = "",
                   case=None) -> dict[str, Any]:
    sentence_id = stable_id("esent", document_id, index)
    tokens = [build_token(record, sentence_id, position, case)
              for position, record in enumerate(glossed.get("tokens") or [])]
    mark_automatic_speakers(tokens)
    return {
        "sentence_id": sentence_id,
        "source_line": nfc(glossed.get("original") or ""),
        "translation": nfc(translation),
        "tokens": tokens,
        "comments": [],
        # Discussion threads: a separate, unmoderated conversation channel,
        # not an editorial decision — see `discussions` handling in `app.py`
        # and the design doc for why this is not folded into `comments`.
        "discussions": [],
    }


def _column_pieces(token: dict[str, Any], strip) -> list[str]:
    """The written pieces one column stands for, punctuation stripped.

    A piece is what lies between spaces and underties. Columns relate to
    written words in every way the editor allows -- a proclitic fold
    (``m ġarība,``), a PN run (``Brōm Apṭaḷḷa.``), a hand merge, one piece of
    a sandhi group (``be`` of ``be‿Kattaḥ``), or a merge that swallowed a
    sandhi piece plus the next word (``əBrōm Apṭaḷḷa`` out of
    ``bess‿əBrōm Apṭaḷḷa``) -- but always as a run of whole pieces.
    """
    return [core for core in (strip(part) for part in
                              re.split(rf"[\s{UNDERTIE}]+", nfc(token.get("surface") or "")))
            if core]


def _line_pieces(text: str) -> list[dict[str, Any]]:
    """Pieces of a new line 1, in `gloss_sentence` order (same numbering)."""
    from maalula_glosser.gloss_text import tokenize

    out: list[dict[str, Any]] = []
    for word_index, token in enumerate(tokenize(nfc(text).strip())):
        pieces = token["pieces"]
        for piece_index, piece in enumerate(pieces):
            out.append({"core": piece, "word": word_index,
                        "original": token["original"],
                        "whole_word": len(pieces) == 1})
    return out


def rewrite_line(sentence: dict[str, Any], text: str, glosser,
                 case=None) -> dict[str, int]:
    """Replace line 1 without sacrificing human morphology.

    Alignment is by written piece (see `_column_pieces`), not by column, so a
    spelling fix reaches its column even when the line also holds a
    multiword or sandhi column. The old column-level matcher saw such a
    column as an unequal replacement and fell back to changing line 1 only,
    which left ``aсṯar``/``kiṯər`` in the columns of Muše Sanžar line 67
    after the text had been corrected. Per column:

    * every piece it covers is unchanged -> kept as is (surface refreshed);
    * its pieces were respelled one-for-one -> a human column keeps its cells
      (only line 1 changes, as before); an automatic one is re-glossed;
    * all its pieces were deleted -> the column goes, human or not;
    * its pieces were split, merged, partly deleted or moved apart -> an
      automatic column is re-glossed; a human column has no safe new place,
      and then (only then) the whole edit falls back to changing line 1 alone.

    Pieces no surviving column claims are glossed from one sentence-level
    pass over the new line, so a new word after a proclitic, a new name run
    or a new ʕa(m)+verb gets the same contextual treatment as in a fresh
    document. A reading that would straddle a kept column falls back to
    piece-by-piece glossing.
    """
    from maalula_glosser.gloss_text import _strip_punctuation

    def strip(word: str) -> str:
        return _strip_punctuation(nfc(word))

    line = nfc(text).strip()
    old = list(sentence.get("tokens") or [])
    pieces = _line_pieces(line)
    old_pieces: list[tuple[int, str]] = []          # (column, core)
    for column, token in enumerate(old):
        for core in _column_pieces(token, strip):
            old_pieces.append((column, core))
    matcher = SequenceMatcher(None, [core for _c, core in old_pieces],
                              [piece["core"] for piece in pieces], autojunk=False)
    # old piece index -> (new piece index, respelled?)
    mapped: dict[int, tuple[int, bool]] = {}
    deleted: set[int] = set()
    for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
        if opcode == "equal" or (opcode == "replace" and i2 - i1 == j2 - j1):
            for old_index, new_index in zip(range(i1, i2), range(j1, j2)):
                mapped[old_index] = (new_index, opcode != "equal")
        elif opcode == "delete":
            deleted.update(range(i1, i2))

    def human_morphology(token: dict[str, Any]) -> bool:
        return bool(token.get("locked") or token.get("imported_igt")
                    or any(m.get("origin") == USER
                           for m in token.get("morphemes") or []))

    owned: dict[int, list[int]] = {}
    for old_index, (column, _core) in enumerate(old_pieces):
        owned.setdefault(column, []).append(old_index)

    keep: dict[int, list[int]] = {}      # column -> new piece indexes
    respelled_columns: set[int] = set()
    for column, token in enumerate(old):
        own = owned.get(column, [])
        targets = [mapped.get(i) for i in own]
        intact = bool(own) and all(t is not None for t in targets)
        human = human_morphology(token)
        if intact:
            indexes = [t[0] for t in targets]
            contiguous = indexes == list(range(indexes[0], indexes[0] + len(indexes)))
            respelled = any(t[1] for t in targets)
            if contiguous and (not respelled or human):
                keep[column] = indexes
                if respelled:
                    respelled_columns.add(column)
                continue
            if not human:
                continue          # respelled automatic column: re-gloss it
        elif not human:
            continue              # deleted/split/merged automatic column
        elif (own and all(i in deleted for i in own)
              and not any(core == old_pieces[i][1]
                          for i in own for j, (_c, core) in enumerate(old_pieces)
                          if j not in own)):
            # The author deleted exactly this column's words. (If the same
            # spelling occurs elsewhere in the line, which occurrence went is
            # the matcher's guess, so a human column is not risked on it.)
            continue
        # A human column whose pieces were deleted, split, merged or moved
        # apart has no safe new place: change the transcription only.
        sentence["source_line"] = line
        return {"kept": len(old), "added": 0, "dropped": 0,
                "source_only": 1}

    claimed = {index for indexes in keep.values() for index in indexes}
    fresh = [j for j in range(len(pieces)) if j not in claimed]

    # one sentence-level pass for everything that has to be (re)glossed
    fresh_records: dict[int, list[dict[str, Any]]] = {}
    if fresh:
        fresh_set = set(fresh)
        covered: set[int] = set()
        for record in glosser.gloss_sentence(line).get("tokens") or []:
            start, end = record.get("piece_start"), record.get("piece_end")
            if start is None or end is None:
                continue
            inside = set(range(start, end + 1))
            if inside <= fresh_set:
                fresh_records.setdefault(start, []).append(record)
                covered |= inside
        for j in fresh:
            if j in covered:
                continue
            piece = pieces[j]
            record = glosser.gloss_token(piece["core"])
            record["source_token"] = piece["original"]
            record["from_sandhi"] = not piece["whole_word"]
            fresh_records.setdefault(j, []).append(record)

    def surface_of(indexes: list[int]) -> str:
        """Line-1 text for a run of new pieces: whole words keep their
        punctuation; pieces of a sandhi group are joined by the undertie."""
        parts: list[str] = []
        by_word: dict[int, list[int]] = {}
        for j in indexes:
            by_word.setdefault(pieces[j]["word"], []).append(j)
        for word, members in by_word.items():
            if pieces[members[0]]["whole_word"]:
                parts.append(nfc(pieces[members[0]]["original"]))
            else:
                parts.append(UNDERTIE.join(pieces[j]["core"] for j in members))
        return " ".join(parts)

    first_piece: dict[int, list[int]] = {}
    for column, indexes in keep.items():
        first_piece.setdefault(indexes[0], []).append(column)
    tokens: list[dict[str, Any]] = []
    added = kept_count = 0
    for j in range(len(pieces)):
        for column in sorted(first_piece.get(j, [])):
            token = old[column]
            if column in respelled_columns:
                # an untouched column keeps its surface byte for byte (it may
                # legitimately differ from line 1 in punctuation)
                token["surface"] = surface_of(keep[column])
            if token.get("sandhi_group") and not pieces[keep[column][0]]["whole_word"]:
                # the group is provenance for every piece of the written word:
                # a respelling of one piece renames it for its siblings too
                token["sandhi_group"] = nfc(pieces[keep[column][0]]["original"])
            tokens.append(token)
            kept_count += 1
        for position, record in enumerate(fresh_records.get(j, [])):
            tokens.append(build_token(record, f"{sentence.get('sentence_id')}n{j}",
                                      position, case))
            added += 1

    mark_automatic_speakers(tokens)
    sentence["tokens"] = tokens
    sentence["source_line"] = line
    return {"kept": kept_count, "added": added, "dropped": len(old) - kept_count}


def propose_label_options(text: str, glosser) -> tuple[list[list[str]], dict[str, Any] | None]:
    """What the analyzer makes of a segmentation typed by hand.

    Cutting a word up is half the work; the labels are the other half, and the
    program often knows them once it is told where the boundaries are. The
    longest leading run that analyses wins: `mḥammṣ-in=le` is unknown as a
    whole, but `mḥammṣ-in` is `ḥammeṣ.NC-MP`, so the stem and the plural get
    their labels and the object clitic is left open rather than guessed at.

    Returns the labels for as many leading pieces as were accounted for, and
    the analysis they came from — which carries the lemma, and with it the
    dictionary link the token had no way to earn before.
    """
    parts = split_parts(nfc(text).strip())
    proposed: list[str] = ["" for _ in parts]
    best: dict[str, Any] | None = None
    full_match = False
    for count in range(len(parts), 0, -1):
        prefix = "".join(f"{sep}{form}" for sep, form in parts[:count])
        analysis = glosser.gloss_token(prefix)
        segmentation, gloss = analysis.get("segmentation"), analysis.get("gloss")
        if not segmentation or not gloss:
            continue
        # Even a reading that cuts elsewhere is worth keeping for its lemma:
        # `mḥarrc-in` comes back undivided as `ḥarrec`, which is no use as a
        # row of labels but is exactly what the dictionary link hangs on.
        if best is None and analysis.get("lemma"):
            best = analysis
        found = split_parts(segmentation)
        labels = split_parts(gloss)
        # labels are only distributed if the reading agrees with the cuts that
        # were typed, in how many and in which separator; otherwise they would
        # land in the wrong cells
        if len(found) != count or len(labels) != count:
            continue
        if [form for _sep, form in found] != [form for _sep, form in parts[:count]] or \
                not same_boundary_topology(found, parts[:count]) or \
                not same_boundary_topology(labels, parts[:count]):
            continue
        for index, (_sep, label) in enumerate(labels):
            if not proposed[index]:
                proposed[index] = label
        full_match = full_match or (
            count == len(parts) and all(proposed[:count]))
        best = analysis if analysis.get("lemma") else best
        if all(proposed):
            break

    # Whatever is still empty — the pieces past the run that fitted, or all of
    # them — is asked of the inventories directly. A bound form such as `=le`
    # occurs twenty-five times in the corpus and never as a word of its own, so
    # this is the only place its label can come from.
    enclitics = getattr(glosser.analyzer, "enclitics", {})
    suffixes = {nfc(form): gloss for form, gloss, _meta
                in getattr(glosser.analyzer, "suffixes", [])}
    for index, (sep, form) in enumerate(parts):
        if not sep or proposed[index]:
            continue
        if sep in {"=", "+"} and enclitics.get(form):
            proposed[index] = enclitics[form][0][0]
        elif sep == "-" and suffixes.get(form):
            proposed[index] = suffixes[form]

    options = typed_bound_options(
        parts, glosser, proposed, preserve_object_roles=full_match)
    return (options if any(options) else []), best


def propose_labels(text: str, glosser) -> tuple[list[str], dict[str, Any] | None]:
    """Backward-compatible first choices for :func:`propose_label_options`."""
    options, best = propose_label_options(text, glosser)
    return ([choices[0] if choices else "" for choices in options], best)


def resegment(token: dict[str, Any], text: str, glosser=None,
              proposed: list[str] | None = None,
              proposed_options: list[list[str]] | None = None) -> list[dict[str, Any]]:
    """Rebuild a token's cells from a segmentation the human typed.

    Line 2 is not a read-only echo of line 1. The transcription may spell a word
    one way and the morphology another — `ʕal‿ōḏ` is glossed `ʕal=hōḏ`, with the
    h the sandhi swallowed — and the boundaries the analyzer chose are a
    proposal, not a verdict.

    Labels are kept where they still have somewhere to sit: unchanged cell count
    carries them one for one, so fixing a letter never costs a gloss. A real
    re-segmentation carries a label only to a piece spelled the same way, and
    parks the old gloss line on the first cell as a suggestion rather than
    dropping it.
    """
    parts = split_parts(nfc(text).strip())
    if not parts:
        raise SessionError("a token cannot have an empty segmentation")
    if any(not form for _sep, form in parts):
        # a leading `=`, or a doubled one, would make a cell with no morpheme in
        # it — which is exactly the empty `=` column this replaced
        raise SessionError("every morpheme must have something in it")

    old = token.get("morphemes") or []
    if len(parts) == len(old):
        labels = [(cell.get("gloss") or "", cell.get("suggested") or "",
                   cell.get("origin") or AUTO) for cell in old]
    else:
        by_form: dict[str, list[dict[str, Any]]] = {}
        for cell in old:
            by_form.setdefault(cell.get("form") or "", []).append(cell)
        labels = []
        for _sep, form in parts:
            match = (by_form.get(form) or [None])[0]
            if match is not None:
                by_form[form].remove(match)
                labels.append((match.get("gloss") or "", match.get("suggested") or "",
                               match.get("origin") or AUTO))
            else:
                labels.append(("", "", AUTO))
        if labels and not labels[0][0] and not labels[0][1]:
            labels[0] = ("", join_parts(old, key="gloss"), AUTO)

    # Where a cell has come out with nothing, offer what the analyzer makes of
    # the boundaries just typed — as a suggestion, not written in: the human
    # moved the boundary, so the program's answer to it is a proposal. The
    # caller may have asked already; asking is the expensive part.
    if proposed is None and glosser is not None:
        proposed, _analysis = propose_labels(text, glosser)
    if proposed and any(not gloss for gloss, _s, _o in labels):
        for index, label in enumerate(proposed):
            if index < len(labels) and not labels[index][0] and label:
                labels[index] = ("", label, labels[index][2])

    token_id = token.get("token_id") or ""
    cells = []
    for index, ((sep, form), (gloss, suggested, origin)) in enumerate(zip(parts, labels)):
        cells.append({
            "morph_id": stable_id("morph", token_id, index, form),
            "sep": sep,
            "form": form,
            "gloss": gloss,
            "suggested": suggested,
            "status": morpheme_status(gloss, "HIGH" if gloss else "LOW"),
            "origin": origin,
            "alternatives": ([{"gloss": choice} for choice in
                              (proposed_options[index][1:]
                               if proposed_options and index < len(proposed_options)
                               else [])]),
        })
    return cells


UNDERTIE = "‿"

# How much of the line to show either side of a cell in the "same again" list.
CONTEXT_CHARS = 42


def _context(sentence: dict[str, Any], position: int) -> dict[str, str]:
    """The line around one token, trimmed, so a match can be judged on sight."""
    surfaces = [t.get("surface") or "" for t in sentence.get("tokens") or []]
    before = " ".join(surfaces[:position])
    after = " ".join(surfaces[position + 1:])
    if len(before) > CONTEXT_CHARS:
        before = "…" + before[-CONTEXT_CHARS:]
    if len(after) > CONTEXT_CHARS:
        after = after[:CONTEXT_CHARS] + "…"
    return {"before": before, "focus": surfaces[position] if surfaces else "",
            "after": after}


def similar_cells(document: dict[str, Any], morph_id: str) -> dict[str, Any]:
    """Every other cell in the document spelled the same as this one.

    The same word is glossed the same way a hundred times over, and each of
    those hundred used to be its own three clicks. What comes back is a
    proposal, not an action: each row carries its line so the author can see
    that this `ḥ` really is the same `ḥ`, and drop the ones that are not.

    A cell the human has already decided is left out — a batch must not
    overwrite an earlier decision.
    """
    origin_form = None
    matches: list[dict[str, Any]] = []
    for sentence in document.get("sentences") or []:
        for position, token in enumerate(sentence.get("tokens") or []):
            for morpheme in token.get("morphemes") or []:
                if morpheme.get("morph_id") == morph_id:
                    origin_form = morpheme.get("form")
    if not origin_form:
        return {"form": "", "matches": []}
    for sentence in document.get("sentences") or []:
        for position, token in enumerate(sentence.get("tokens") or []):
            for morpheme in token.get("morphemes") or []:
                if morpheme.get("morph_id") == morph_id:
                    continue
                if morpheme.get("form") != origin_form:
                    continue
                if morpheme.get("origin") == USER:
                    continue
                matches.append({
                    "morph_id": morpheme["morph_id"],
                    "gloss": morpheme.get("gloss") or "",
                    "suggested": morpheme.get("suggested") or "",
                    "status": morpheme.get("status"),
                    "sentence_id": sentence.get("sentence_id"),
                    "context": _context(sentence, position),
                })
    return {"form": origin_form, "matches": matches}


def human_surface_key(surface: str) -> str:
    """Identity of a written token for local human-analysis suggestions.

    Case is orthographic: Ayṯul, ayṯul and AYṮUL are the same key. Only external
    punctuation is ignored. Underties are word spaces; spaces and explicit -/= are retained,
    because a merge/split/cliticisation changes token topology and is not safe
    to propagate automatically.
    """
    text = " ".join(morphology_text(surface).split())

    def external_punct(char: str) -> bool:
        return char not in "-=+‿" and unicodedata.category(char).startswith("P")

    while text and external_punct(text[0]):
        text = text[1:]
    while text and external_punct(text[-1]):
        text = text[:-1]
    return text.casefold()


def reading_state(token: dict[str, Any]) -> dict[str, Any]:
    """Only editable reading state; IDs and dictionary metadata are local."""
    return {
        "surface": token.get("surface") or "",
        "cells": [{key: cell.get(key, "") for key in ("sep", "form", "gloss")}
                  for cell in token.get("morphemes") or []],
        "keys": token.get("surface_match_keys") or [],
        "replacement": token.get("surface_replacement"),
    }


def reading_version(token: dict[str, Any]) -> str:
    import hashlib
    import json
    return hashlib.sha256(json.dumps(reading_state(token), sort_keys=True,
                                     ensure_ascii=False).encode()).hexdigest()


def surface_match_keys(token: dict[str, Any]) -> set[str]:
    return {human_surface_key(token.get("surface") or ""),
            *(human_surface_key(key) for key in token.get("surface_match_keys") or [])} - {""}


def remember_surface_edit(token: dict[str, Any], old_surface: str) -> None:
    """Keep the old spelling reachable after a one-token transcription edit."""
    if is_speaker(token):
        return
    structure = lambda text: re.findall(r"[-=+\s]", morphology_text(text))
    if structure(old_surface) != structure(token["surface"]):
        token.pop("surface_match_keys", None)
        token.pop("surface_replacement", None)
        return
    token["surface_match_keys"] = sorted(
        surface_match_keys(token) | {human_surface_key(old_surface)})
    if human_surface_key(old_surface) != human_surface_key(token["surface"]):
        token["surface_replacement"] = token["surface"]


def _surface_parts(text: str) -> tuple[str, str, str]:
    text = nfc(text)
    start, end = 0, len(text)
    def punct(char):
        return char not in "-=+‿" and unicodedata.category(char).startswith("P")
    while start < end and punct(text[start]):
        start += 1
    while end > start and punct(text[end - 1]):
        end -= 1
    return text[:start], text[start:end], text[end:]


def proposed_reading(source: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    proposed = reading_state(source)
    proposed["surface"] = target.get("surface") or ""
    if source.get("surface_replacement") and not is_speaker(source):
        left, _, right = _surface_parts(proposed["surface"])
        _, core, _ = _surface_parts(source["surface_replacement"])
        proposed["surface"] = left + core + right
    return proposed


def similar_readings(document: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    """Offer the entire current reading, including incomplete human glosses."""
    matches = []
    speaker = is_speaker(source)
    keys = surface_match_keys(source)
    for sentence in document.get("sentences") or []:
        tokens = sentence.get("tokens") or []
        arabic = _arabic_marker_token_indexes(tokens)
        for index, token in enumerate(tokens):
            if token.get("token_id") == source.get("token_id") or index in arabic:
                continue
            if speaker:
                same = nfc(token.get("surface") or "") == nfc(source.get("surface") or "")
            else:
                same = not is_speaker(token) and bool(keys & surface_match_keys(token))
            if not same:
                continue
            proposed = proposed_reading(source, token)
            current = reading_state(token)
            if all(current[key] == proposed[key] for key in ("surface", "cells")):
                continue
            matches.append({"token_id": token["token_id"],
                            "version": reading_version(token),
                            "context": _context(sentence, index),
                            "current": current, "proposed": proposed})
    return {"source_id": source["token_id"], "source_version": reading_version(source),
            "form": source.get("surface") or "", "matches": matches}


def apply_reading(sentence: dict[str, Any], target: dict[str, Any],
                  source: dict[str, Any]) -> None:
    proposed = proposed_reading(source, target)
    resurface(sentence, target["token_id"], target.get("surface") or "", proposed["surface"])
    target["surface"] = proposed["surface"]
    cells = []
    for index, cell in enumerate(proposed["cells"]):
        cells.append({**cell, "morph_id": stable_id("morph", target["token_id"], index, cell["form"]),
                      "origin": USER, "suggested": "", "alternatives": [],
                      "status": morpheme_status(cell["gloss"], "HIGH" if cell["gloss"] else "LOW")})
    target["morphemes"] = cells
    target["locked"] = True
    target["alternatives"] = []
    target["auto_pending"] = None
    target["uncertainty"] = []
    target["confidence"] = "HIGH" if all(c["status"] == CONFIDENT for c in cells) else "LOW"
    target["tier"] = "USER"
    # Do not carry a stale lexeme binding across a new human segmentation.
    for key in ("normalized", "lemma", "lemma_id", "lexical_gloss", "grammatical_info",
                "grammatical_infos", "lift_entry_id", "lift_guid", "lift_sense_id",
                "dictionary", "lexeme_candidates"):
        target.pop(key, None)
    if not is_speaker(source):
        target["surface_match_keys"] = sorted(surface_match_keys(target) | surface_match_keys(source))
        if source.get("surface_replacement"):
            target["surface_replacement"] = source["surface_replacement"]


def _token_index(sentence: dict[str, Any], token_id: str) -> int:
    for index, token in enumerate(sentence.get("tokens") or []):
        if token.get("token_id") == token_id:
            return index
    raise SessionError(f"no such token {token_id}")


def merge_tokens(sentence: dict[str, Any], token_id: str, count: int = 2,
                 joiner: str = "=") -> dict[str, Any]:
    """Fold `count` adjacent columns into one, in place; returns the survivor.

    Two things the program cannot decide on its own need this. `la` and `ču`
    attach to some hosts and not others, and no inventory says which — the
    author writes `la=iṯken` by hand. And an Arabic insert is a phrase glossed
    as a unit, `ᴬsənt əlxamsā u ʕəšrīnᴬ` against "the year 1925", which no
    tokenizer will guess.

    `joiner` is `=` or `-` for a clitic, where the cells stay separate and each
    keeps its label. A space asks for the other thing: the whole run collapses
    into one cell, because a space is not a morpheme boundary and a phrase
    glossed as a unit has one label, not five.
    """
    if joiner not in ("=", "-", " "):
        raise SessionError(f"cannot join columns with {joiner!r}")
    tokens = sentence.get("tokens") or []
    start = _token_index(sentence, token_id)
    if count < 2 or start + count > len(tokens):
        raise SessionError("there are not that many columns left to join")
    group = tokens[start:start + count]

    surface = group[0].get("surface") or ""
    for token in group[1:]:
        tie = UNDERTIE if (token.get("sandhi_group")
                           and token["sandhi_group"] == group[0].get("sandhi_group")
                           ) else " "
        surface = f"{surface}{tie}{token.get('surface') or ''}"

    head = dict(group[0])
    head["surface"] = nfc(surface)
    head["sandhi_group"] = ""
    if joiner == " ":
        parts = [join_parts(t["morphemes"], key="gloss") for t in group]
        # a phrase glossed as a unit gets one label, and the author writes it.
        # Stringing the pieces together would say `and` for a run where only the
        # `u` happened to be glossed — an answer where there is none.
        whole = " ".join(parts) if all(part for part in parts) else ""
        head["morphemes"] = [{
            "morph_id": stable_id("morph", head["token_id"], 0, surface),
            "sep": "",
            "form": " ".join(join_parts(t["morphemes"]) for t in group),
            "gloss": whole,
            "suggested": "" if whole else " ".join(p for p in parts if p),
            "status": morpheme_status(whole, "HIGH"),
            "origin": USER,
            "alternatives": [],
        }]
    else:
        cells: list[dict[str, Any]] = []
        for position, token in enumerate(group):
            for index, cell in enumerate(token.get("morphemes") or []):
                copy = dict(cell)
                if position and not index:
                    copy["sep"] = joiner
                copy["morph_id"] = stable_id("morph", head["token_id"],
                                             len(cells), copy.get("form") or "")
                cells.append(copy)
        head["morphemes"] = cells
    # the author decided where this column begins and ends; nothing automatic
    # may take it apart again
    head["locked"] = True
    head["auto_pending"] = None
    head["topology_edited"] = True
    head.pop("surface_match_keys", None)
    head.pop("surface_replacement", None)
    sentence["tokens"] = tokens[:start] + [head] + tokens[start + count:]
    return head


def split_token(sentence: dict[str, Any], token_id: str, glosser,
                case=None) -> list[dict[str, Any]]:
    """Break one column back into a column per written word; returns the pieces.

    The inverse of `merge_tokens`, and the way out of a join the program made
    for itself: a proclitic folded onto the wrong host, or a run wrongly read
    as an Arabic insert. Each word is glossed afresh, so what comes back is
    what the program would have said had it never joined them.
    """
    tokens = sentence.get("tokens") or []
    start = _token_index(sentence, token_id)
    token = tokens[start]
    words = [w for w in re.split(rf"[\s{UNDERTIE}]+", nfc(token.get("surface") or ""))
             if w]
    if len(words) < 2:
        raise SessionError("this column is a single word")
    pieces = []
    for index, word in enumerate(words):
        glossed = glosser.gloss_token(word)
        piece = build_token({**glossed, "source_token": word},
                            f"{token_id}s{index}", index, case)
        piece["sandhi_group"] = token.get("sandhi_group") or ""
        piece["topology_edited"] = True
        piece.pop("surface_match_keys", None)
        piece.pop("surface_replacement", None)
        pieces.append(piece)
    sentence["tokens"] = tokens[:start] + pieces + tokens[start + 1:]
    return pieces


def topology_offer(document: dict[str, Any], before: list[dict[str, Any]],
                   after: list[dict[str, Any]], operation: str, joiner: str = "=") -> dict[str, Any]:
    """Remember one explicit structural edit, not an automatic rewrite rule."""
    import uuid
    document["topology_offer"] = {
        "proposal_id": uuid.uuid4().hex, "operation": operation, "joiner": joiner,
        "pattern": [{"surface": token.get("surface") or "", "speaker": is_speaker(token)}
                    for token in before],
        "source_ids": [token["token_id"] for token in after],
        "source_versions": [reading_version(token) for token in after],
    }
    return similar_topologies(document)


def _topology_sources(document: dict[str, Any]) -> list[dict[str, Any]]:
    proposal = document.get("topology_offer") or {}
    ids = proposal.get("source_ids") or []
    versions = proposal.get("source_versions") or []
    for sentence in document.get("sentences") or []:
        tokens = sentence.get("tokens") or []
        for index in range(len(tokens) - len(ids) + 1):
            group = tokens[index:index + len(ids)]
            if ids and [t["token_id"] for t in group] == ids:
                if [reading_version(t) for t in group] == versions:
                    return group
    raise SessionError("the source sequence changed; repeat the merge or split")


def topology_version(tokens: list[dict[str, Any]]) -> str:
    import hashlib
    return hashlib.sha256("|".join(
        t["token_id"] + ":" + reading_version(t) for t in tokens).encode()).hexdigest()


def similar_topologies(document: dict[str, Any]) -> dict[str, Any]:
    proposal = document.get("topology_offer") or {}
    sources = _topology_sources(document)
    pattern = proposal["pattern"]
    width = len(pattern)
    source_ids = set(proposal["source_ids"])
    matches = []
    for sentence in document.get("sentences") or []:
        tokens = sentence.get("tokens") or []
        index = 0
        while index + width <= len(tokens):
            group = tokens[index:index + width]
            same = not any(t["token_id"] in source_ids for t in group)
            for expected, token in zip(pattern, group):
                if expected["speaker"] or is_speaker(token):
                    same = same and nfc(expected["surface"]) == nfc(token.get("surface") or "")
                else:
                    same = same and human_surface_key(expected["surface"]) == human_surface_key(token.get("surface") or "")
            if same:
                # Greedy non-overlapping windows make a repeated sequence safe
                # to select in one batch (e.g. u u u u).
                context = _context(sentence, index)
                context["focus"] = " ".join(t.get("surface") or "" for t in group)
                context["after"] = _context(sentence, index + width - 1)["after"]
                if proposal["operation"] == "merge":
                    preview_sentence = {"tokens": copy.deepcopy(group)}
                    previews = [merge_tokens(preview_sentence, group[0]["token_id"],
                                             width, proposal["joiner"])]
                else:
                    previews = [{"surface": word} for word in re.split(
                        r"[\s‿]+", group[0].get("surface") or "") if word]
                if len(previews) != len(sources):
                    index += 1
                    continue
                proposed = []
                for source, preview in zip(sources, previews):
                    reading = reading_state(source)
                    reading["surface"] = preview.get("surface") or ""
                    proposed.append(reading)
                matches.append({"token_id": group[0]["token_id"],
                                "token_ids": [t["token_id"] for t in group],
                                "sentence_id": sentence["sentence_id"],
                                "version": topology_version(group), "context": context,
                                "current": [reading_state(t) for t in group],
                                "proposed": proposed})
                index += width
            else:
                index += 1
    return {"proposal_id": proposal["proposal_id"], "operation": proposal["operation"],
            "matches": matches}


def apply_topologies(document: dict[str, Any], proposal_id: str,
                     targets: dict[str, str], glosser, case=None) -> tuple[list, list]:
    """Apply checked disjoint occurrences atomically in the caller's document."""
    proposal = document.get("topology_offer") or {}
    if proposal.get("proposal_id") != proposal_id:
        raise SessionError("a newer merge or split replaced this proposal")
    sources = _topology_sources(document)
    eligible = {m["token_id"]: m for m in similar_topologies(document)["matches"]}
    if any(key not in eligible or eligible[key]["version"] != version
           for key, version in targets.items()):
        raise SessionError("a target sequence changed; reopen the proposal")
    before, after = [], []
    for sentence in document.get("sentences") or []:
        selected = [m for key, m in eligible.items() if key in targets
                    and m["sentence_id"] == sentence["sentence_id"]]
        if not selected:
            continue
        before.append({"sentence_id": sentence["sentence_id"],
                       "tokens": copy.deepcopy(sentence["tokens"])})
        selected.sort(key=lambda m: _token_index(sentence, m["token_id"]), reverse=True)
        for match in selected:
            if proposal["operation"] == "merge":
                replacements = [merge_tokens(sentence, match["token_id"],
                                              len(proposal["pattern"]), proposal["joiner"])]
            else:
                replacements = split_token(sentence, match["token_id"], glosser, case)
            if len(replacements) != len(sources):
                raise SessionError("the target has a different word structure")
            for token, source in zip(replacements, sources):
                # Copy the complete chosen reading, but preserve this occurrence's
                # spelling, punctuation and surface links. Topology is the edit.
                template = copy.deepcopy(source)
                template.pop("surface_replacement", None)
                apply_reading(sentence, token, template)
                token.pop("surface_match_keys", None)
                token.pop("surface_replacement", None)
        after.append({"sentence_id": sentence["sentence_id"],
                      "tokens": copy.deepcopy(sentence["tokens"])})
    return before, after


def resurface(sentence: dict[str, Any], token_id: str,
              old: str, new: str) -> None:
    """Keep line 1 in step with a word the human rewrote, in place.

    `source_line` is what the export prints as line 1 and what the editor shows
    as the running text, so a re-glossed token that only changed `surface` left
    the two disagreeing: the screen said `šōmṭ-in`, the document still said
    `bahar`.

    Exactly one occurrence is touched, found by counting the identical surfaces
    before it, so a line with three `u` in it edits the right one. If the old
    spelling is not in the line — an imported turn whose transcription is not
    token-for-token — the line is left exactly as it was rather than guessed at.
    """
    old, new = nfc(old), nfc(new)
    line = sentence.get("source_line") or ""
    if not old or old == new or not line:
        return
    before = 0
    for token in sentence.get("tokens") or []:
        if token.get("token_id") == token_id:
            break
        if nfc(token.get("surface") or "") == old:
            before += 1
    else:
        return
    # bounded by non-word characters, so `b` does not match inside `bahar`,
    # and a sandhi piece still matches between its ties
    pattern = re.compile(rf"(?<!\w){re.escape(old)}(?!\w)")
    matches = list(pattern.finditer(line))
    if before >= len(matches):
        return
    at = matches[before]
    sentence["source_line"] = line[:at.start()] + new + line[at.end():]


def split_lines(text: str) -> list[str]:
    """One interlinear block per line of input.

    The author decides where a block ends, not a punctuation heuristic: a line
    of the transcription is the unit that gets its own translation and its own
    table in the export. Blank lines are separators, not empty blocks.
    """
    return [line.strip() for line in nfc(text).splitlines() if line.strip()]


def build_document(text: str, glosser, document_id: str | None = None,
                   translations: dict[int, str] | list[str] | None = None,
                   title: str = "", case=None) -> dict[str, Any]:
    """Gloss `text` and wrap the result as an editable session.

    `glosser` is a `TextGlosser`; it is injected so the caller controls which
    morphology engine and project root are in play.

    `translations` is per line, either a list aligned with the lines of `text`
    or a mapping from line index. A translation belongs to one line, so a text
    of twelve lines can carry twelve of them.
    """
    text = nfc(text)
    document_id = document_id or stable_id("edoc", text[:400], now())
    if isinstance(translations, list):
        translations = {index: value for index, value in enumerate(translations)}
    translations = translations or {}
    sentences = [
        build_sentence(glosser.gloss_sentence(line), index, document_id,
                       translations.get(index, ""), case)
        for index, line in enumerate(split_lines(text))
    ]
    stamp = now()
    return {
        "schema": SCHEMA,
        "document_id": document_id,
        "title": title,
        "created": stamp,
        "updated": stamp,
        "source_text": text,
        "sentences": sentences,
    }


def force_regloss_line(sentence: dict[str, Any], glosser, case=None) -> None:
    """Re-analyse every word of this line from scratch, in place.

    Unlike `rewrite_line`, nothing here is preserved: a locked token, a
    human-edited cell, an imported IGT reading — all of it is discarded and
    rebuilt from `source_line` alone. The escape hatch for a line that needs
    a clean slate regardless of what has been done to it since (Alt+R in the
    editor), not a light-touch edit.
    """
    glossed = glosser.gloss_sentence(sentence.get("source_line") or "")
    tokens = [build_token(record, sentence["sentence_id"], position, case)
             for position, record in enumerate(glossed.get("tokens") or [])]
    mark_automatic_speakers(tokens)
    sentence["tokens"] = tokens


def _partial_igt_key(value: str) -> str:
    """Loose spelling key used only to align imported IGT with line-1 tokens.

    Running text can carry underties and allophonic b̥/b̰ while the source IGT
    deliberately restores an unbound form.  Those are alignment noise, not a
    reason to discard a human gloss.  Morphological boundaries and punctuation
    are ignored here; the imported segmentation itself is kept verbatim later.
    """
    value = unicodedata.normalize("NFD", morphology_text(value).casefold())
    value = value.replace("\u0325", "").replace("\u0330", "")
    value = value.replace("-", "").replace("=", "").replace("+", "").replace(UNDERTIE, "")
    value = re.sub(r"[^\wʔʕāēīōūḏ̣ṯḥṣṭẓšžčġḳ]+", "", value, flags=re.UNICODE)
    return unicodedata.normalize("NFC", value)


def _split_igt_cell_parts(text: str) -> list[tuple[str, str]]:
    """`split_parts`, extended across a source IGT cell that names several
    running-text words at once — a sandhi-tied span the docx source still
    glosses one word per column, e.g. `b=aḥ n-ḥaml-ell=l masʔulōy-ṯ-a`
    opposite `want=1P    1-put.PC-INTF=DOM    responsibility-F-FREE`.

    Plain whitespace between the words is not itself a morpheme boundary and
    `split_parts` alone does not know that: it has no separator of its own to
    split on, so the words either side of a run of spaces get glued into one
    garbled `form`/`gloss` (`"aḥ n"`) instead of two clean cells. Splitting on
    whitespace first keeps cells aligned. Word boundaries in morphology are
    spaces; the original undertie is retained only in transcription.
    """
    words = morphology_text(text).split()
    parts: list[tuple[str, str]] = []
    for index, word in enumerate(words):
        word_parts = split_parts(word)
        if index > 0 and word_parts:
            _, form = word_parts[0]
            word_parts = [(" ", form)] + word_parts[1:]
        parts.extend(word_parts)
    return parts


def _imported_user_cells(token: dict[str, Any], imported: dict[str, Any],
                         case=None) -> list[dict[str, Any]] | None:
    """Convert one source IGT column into editor cells without re-analysing it.

    Missing rows return ``None``. A boundary mismatch is preserved literally
    in one uncertain cell, without assigning labels to the wrong morphemes.
    """
    segmentation = nfc(
        imported.get("segmentation")
        or imported.get("isolated_form")
        or imported.get("surface_original")
        or ""
    ).strip()
    gloss = nfc(imported.get("gloss") or imported.get("gloss_original") or "").strip()
    seg_parts = _split_igt_cell_parts(segmentation)
    gloss_parts = _split_igt_cell_parts(gloss)
    if not seg_parts or not gloss_parts:
        return None

    token_id = token.get("token_id") or ""
    if len(seg_parts) != len(gloss_parts) or not same_boundary_topology(
        seg_parts, gloss_parts
    ):
        # The source explicitly says that non-overt morphology need not be
        # glossed, so a table can legitimately contain ``mmaras-č-a`` opposite
        # a single ``practice`` cell.  Preserve that human statement as one
        # editor cell rather than inventing which label belongs to which cut.
        return [{
            "morph_id": stable_id("morph", token_id, 0, segmentation),
            "sep": "",
            "form": morphology_text(segmentation),
            "gloss": morphology_text(gloss),
            "suggested": "",
            "status": morpheme_status(gloss, "MEDIUM"),
            "origin": USER,
            "alternatives": [],
        }]

    cells: list[dict[str, Any]] = []
    for index, ((sep, form), (_gsep, label)) in enumerate(zip(seg_parts, gloss_parts)):
        cells.append({
            "morph_id": stable_id("morph", token_id, index, form),
            "sep": sep,
            "form": nfc(form),
            "gloss": nfc(label),
            "suggested": "",
            "status": morpheme_status(label, "HIGH" if label else "LOW"),
            "origin": USER,
            "alternatives": [],
        })
    return cells


def _partial_igt_alignment(
    imported_tokens: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> list[tuple[str, int | None, int, int, float]]:
    """Sequence-align source IGT columns to one or more running-text tokens.

    Source IGT frequently binds material that running text writes separately:
    ``ču waybin`` → ``ču=w-ayb-in`` and ``l Abu Šalhūb`` →
    ``l=Abu.Šalhūb``.  A one-to-one matcher therefore drops good human work.
    This small dynamic program lets one imported column consume up to four
    adjacent editor tokens while still allowing genuinely omitted material on
    either side.
    """
    imported_keys = [
        _partial_igt_key(
            item.get("surface_original")
            or item.get("segmentation")
            or item.get("isolated_form")
            or ""
        )
        for item in imported_tokens
    ]
    candidate_keys = [_partial_igt_key(token.get("surface") or "")
                      for token in candidates]
    imported_count = len(imported_keys)
    candidate_count = len(candidate_keys)

    @lru_cache(maxsize=None)
    def best(i: int, j: int) -> tuple[float, tuple[tuple[str, int | None, int, int, float], ...]]:
        if i == imported_count:
            tail = tuple(
                ("skip_candidate", None, pos, pos + 1, 0.0)
                for pos in range(j, candidate_count)
            )
            return -1.5 * (candidate_count - j), tail
        if j == candidate_count:
            tail = tuple(
                ("skip_imported", pos, j, j, 0.0)
                for pos in range(i, imported_count)
            )
            return -6.0 * (imported_count - i), tail

        score, path = best(i, j + 1)
        best_score = score - 1.5
        best_path = (("skip_candidate", None, j, j + 1, 0.0),) + path

        score, path = best(i + 1, j)
        score -= 6.0
        if score > best_score:
            best_score = score
            best_path = (("skip_imported", i, j, j, 0.0),) + path

        wanted = imported_keys[i]
        for width in range(1, min(4, candidate_count - j) + 1):
            group = candidates[j:j + width]
            if any(is_speaker(token) or looks_like_speaker_tag(
                    token.get("surface") or "") for token in group):
                continue
            got = "".join(candidate_keys[j:j + width])
            if not wanted or not got:
                continue
            matcher = SequenceMatcher(None, wanted, got)
            ratio = matcher.ratio()
            if ratio < 0.45:
                continue
            if width == 1 and wanted != got and group[0].get("sandhi_group"):
                # A fuzzy restored preposition must not jump to a similar
                # neighbouring name within the same written sandhi group.
                matched = sum(block.size for block in matcher.get_matching_blocks())
                ambiguous = False
                for pos in (j - 1, j + 1):
                    if not 0 <= pos < candidate_count:
                        continue
                    if (candidates[pos].get("sandhi_group") != group[0]["sandhi_group"]
                            or candidate_keys[pos] == got):
                        continue
                    neighbour = SequenceMatcher(None, wanted, candidate_keys[pos])
                    hits = sum(block.size for block in neighbour.get_matching_blocks())
                    if neighbour.ratio() >= 0.75 and hits >= matched - 1:
                        ambiguous = True
                if ambiguous:
                    continue
            # Skipping an unglossed word must never make it profitable to
            # swallow that word into its neighbour's IGT column. Require
            # spelling evidence for EACH source token in a multiword match.
            if width > 1:
                covered = set()
                for block in matcher.get_matching_blocks():
                    covered.update(range(block.b, block.b + block.size))
                offset = 0
                supported = True
                for key in candidate_keys[j:j + width]:
                    hits = sum(pos in covered for pos in range(offset, offset + len(key)))
                    if not key or hits / len(key) < 0.75:
                        supported = False
                    offset += len(key)
                if not supported:
                    continue
            match_score = 8.0 * ratio
            if wanted == got:
                match_score += 2.5
            match_score -= 0.25 * (width - 1)
            score, path = best(i + 1, j + width)
            score += match_score
            if score > best_score:
                best_score = score
                best_path = (
                    ("match", i, j, j + width, ratio),
                ) + path
        return best_score, best_path

    return list(best(0, 0)[1])


def _merge_import_candidates(group: list[dict[str, Any]]) -> dict[str, Any]:
    """Make one editor token from adjacent source tokens without changing text."""
    head = copy.deepcopy(group[0])
    if len(group) == 1:
        return head

    surface = group[0].get("surface") or ""
    for token in group[1:]:
        same_sandhi = (
            token.get("sandhi_group")
            and token.get("sandhi_group") == group[0].get("sandhi_group")
        )
        surface += (UNDERTIE if same_sandhi else " ") + (token.get("surface") or "")
    head["surface"] = nfc(surface)
    # The merged surface already contains every written separator.  Keeping the
    # former sandhi_group as well would make sourcePieces add it a second time.
    head["sandhi_group"] = ""
    for field in ("normalized", "lemma", "lemma_id", "lexical_gloss", "grammatical_info",
                  "grammatical_infos", "lift_entry_id", "lift_guid", "lift_sense_id",
                  "lexeme_candidates"):
        head[field] = None
    head["alternatives"] = []
    head["uncertainty"] = []
    return head


def _arabic_insert_token_ids(tokens: list[dict[str, Any]]) -> set[str]:
    """Token ids covered by an ᴬ…ᴬ span. Computed over the joined line, not
    per word: the middle word of `Aal nādi ṯaqāfiA` carries no marker of its
    own, so a per-word test would let it through as a lexeme."""
    surfaces = [nfc(token.get("surface") or "") for token in tokens]
    marked = arabic_marks.spans(" ".join(surfaces))
    if not marked:
        return set()
    inside: set[str] = set()
    cursor = 0
    for token, surface in zip(tokens, surfaces):
        start, stop = cursor, cursor + len(surface)
        cursor = stop + 1          # the single space the join added
        # `marked` pairs are individual marker *character* offsets, not a
        # half-open range — `span_stop` is where the closing A itself sits,
        # one before the end of its word. Treating it as a range's exclusive
        # end below only comes out right because `arabic_marks._closing_offset`
        # keeps that character strictly inside its own token (its `end > 1`
        # guard never lets it land on the token boundary); a future
        # `arabic_marks` change that returns an offset one past the mark could
        # break this overlap test.
        if any(start < span_stop and span_start < stop
               for span_start, span_stop in marked):
            token_id = token.get("token_id")
            if token_id:
                inside.add(token_id)
    return inside


def _looks_lexical_gloss(gloss: str) -> bool:
    """Does the cell carry a meaning, not only grammatical features. Case is
    not the criterion: the author's own IGT writes features in lowercase
    (`3fs`)."""
    return has_lexical_part(gloss)


# Endings a dictionary form of a non-verb keeps: gender and state
# (blōt-a village-FREE -> blōta, lōġ-ṯ-a language-F-FREE -> lōġṯa,
# Lubnan-ō Lebanese-MS.DEF -> Lubnanō). Number, person and case do not.
CITATION_ENDING_LABELS = frozenset({
    "F", "M", "FS", "MS", "SG", "FREE", "DEF", "INDF", "INDEF", "EMPH"})


def nominal_citation(cells: list[dict[str, Any]], stem: str | None) -> str | None:
    """The dictionary form of a non-verb token: its host stem plus the
    following gender/state endings, without clitics (`=`) and without a
    plural. None for a verbal cell (its dictionary form is not built from
    the token) or when the stem is not among the cells."""
    if not stem:
        return None
    for index, cell in enumerate(cells):
        if nfc(cell.get("form") or "") != stem:
            continue
        if any(label in VERBAL_CITATION_BLOCK
               for label in re.split(r"[.]", cell.get("gloss") or "")):
            return None
        word, last = stem, {label.upper() for label in (cell.get("gloss") or "").split(".")}
        for after in cells[index + 1:]:
            labels = {label.upper() for label in (after.get("gloss") or "").split(".") if label}
            if after.get("sep") != "-" or not labels or not labels <= CITATION_ENDING_LABELS:
                break
            word += nfc(after.get("form") or "")
            last = labels
        # A feminine noun cited without its free-state -a (mawhab-ṯ=e 'his
        # gift' -> mawhabṯa): the dictionary form is the free state.
        if last == {"F"} and word != stem:
            word += "a"
        return word
    return None


VERBAL_CITATION_BLOCK = frozenset({"NC", "NC1", "NC2", "PC", "SC", "PRF", "IPV"})


def _lexical_stem(cells: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    """The stem form and its lexical gloss, with morphology stripped.

    When the lexical cell cannot be chosen (zero or several of them), no
    lemma is invented: the whole segmentation ending up as a headword
    (`apt-aṯ`) is always a bug, never a normal case."""
    lexical = [c for c in cells if _looks_lexical_gloss(c.get("gloss") or "")]
    if len(lexical) != 1:
        return None, None
    return (nfc(lexical[0].get("form") or "") or None,
            lexical_head(lexical[0].get("gloss") or "") or None)


def resolve_reading(surface: str, segmentation: str | None, gloss: str | None,
                    analyzer, confidence: str = "MEDIUM"):
    """Surface/segmentation/gloss strings -> a resolved lexeme, or None.

    The same three-step pipeline `resolve_ambiguous_token` runs on a token's
    chosen alternative (`build_morphemes` -> `_lexical_stem` ->
    `LR.resolve_lexeme`), factored out for callers that only have plain
    strings and no real token (a probe, not a stored token — the `token_id`
    handed to `build_morphemes` is a deterministic placeholder, never a real
    id). `confidence` defaults to `"MEDIUM"` because callers here have no
    token of their own to read a real confidence from; a caller that DOES
    have one (`resolve_ambiguous_token`) passes it directly instead of using
    this default.
    """
    cells = build_morphemes(stable_id("morph_probe", surface, segmentation, gloss),
                            segmentation, gloss, surface, confidence)
    stem_text, stem_gloss = _lexical_stem(cells)
    if analyzer is None:
        return None
    return LR.resolve_lexeme(analyzer, nfc(surface), stem_text)


def resolve_ambiguous_token(token: dict[str, Any], alternative_index: int,
                            analyzer) -> dict[str, Any]:
    """An author picking one of `token["alternatives"]` for a token the
    resolver refused to auto-guess (invariant #7: several competing
    analyses, `lemma_id`/`lemma`/`lexical_gloss` all `None`).

    Mutates `token` in place and returns a small report:
    `{"lexeme_id": str | None, "lemma": str | None, "lexical_gloss": str | None,
      "ambiguous": bool}` — `ambiguous` is true when the chosen reading's stem
    itself resolves to more than one lexeme (`token["lexeme_candidates"]` was
    set instead of `lemma_id`), so a caller can tell "resolved", "still
    ambiguous, but one level down now" and "resolved to nothing (ordinary
    unresolved word)" apart without re-reading the token.

    Two things a reader familiar with `overlay_user_igt`'s import-time branch
    (which this mirrors) will expect and NOT find here, both deliberate:

    - `build_morphemes` is called with `token["confidence"]`, not
      `chosen.get("tier")`. `tier` (`"A1"`/`"D"`, an analysis-quality label)
      and `confidence` (`"HIGH"`/`"MEDIUM"`/`"LOW"`) are different
      vocabularies; `morpheme_status` only understands the latter
      (`STATUS_BY_CONFIDENCE`). Passing `tier` there would not error — it
      would silently fall through to `STATUS_BY_CONFIDENCE.get(..., UNRESOLVED)`
      and degrade every cell in the chosen reading to unresolved.
    - No `lexnew_...` is ever minted here, unlike the import-time code. That
      code is populating a brand-new token for the first time; this function
      is resolving an *already-imported* token whose word may simply not be
      in the dictionary at all — leaving it as an ordinary unresolved word
      afterward is correct, and minting one would be a separate, much bigger
      decision (a new dictionary entry) that a segmentation choice alone
      does not justify.
    """
    alternatives = token.get("alternatives") or []
    if not (0 <= alternative_index < len(alternatives)):
        raise ValueError(
            f"alternative_index {alternative_index} out of range "
            f"(token has {len(alternatives)} alternatives)")
    chosen = alternatives[alternative_index]
    had_lemma_id = token.get("lemma_id") is not None
    new_cells = build_morphemes(token["token_id"], chosen.get("segmentation"),
                                chosen.get("gloss"), token.get("surface") or "",
                                token.get("confidence"))
    if chosen.get("human_surface_match"):
        # Accepting a propagated human reading is itself a human decision. Keep
        # it from being silently re-glossed on the next editor operation.
        for cell in new_cells:
            cell["origin"] = USER
            cell["status"] = morpheme_status(cell.get("gloss"), "HIGH")
        token["locked"] = True
    stem_text, stem_gloss = _lexical_stem(new_cells)
    found = None
    chosen_lexeme_id = chosen.get("lemma_id") or chosen.get("lexeme_id")
    chosen_record = (analyzer.lexeme_record(chosen_lexeme_id)
                     if analyzer is not None and chosen_lexeme_id else {})
    if analyzer is not None and not chosen_record:
        found = LR.resolve_lexeme(analyzer, nfc(token.get("surface") or ""), stem_text)
    token.pop("lexeme_candidates", None)
    ambiguous = False
    if chosen_record:
        token["lemma_id"] = chosen_lexeme_id
        token["lemma"] = chosen_record.get("lemma_canonical")
        token["lexical_gloss"] = chosen.get("lexical_gloss") or stem_gloss
        token["grammatical_info"] = (chosen.get("grammatical_info")
                                     or (chosen_record.get("grammatical_infos") or [None])[0]
                                     if len(chosen_record.get("grammatical_infos") or []) == 1
                                     else chosen.get("grammatical_info"))
        token["grammatical_infos"] = list(chosen_record.get("grammatical_infos") or [])
        token["lift_entry_id"] = chosen.get("lift_entry_id") or chosen_record.get("lift_id")
        token["lift_guid"] = chosen_record.get("lift_guid")
        token["lift_sense_id"] = chosen.get("lift_sense_id")
    elif found is not None and found.candidates:
        token["lexeme_candidates"] = list(found.candidates)
        ambiguous = True
    elif found is not None and found.lexeme_id:
        token["lemma_id"] = found.lexeme_id
        token["lemma"] = found.lemma
        token["lexical_gloss"] = found.gloss or stem_gloss
    elif had_lemma_id:
        # Mirrors overlay_user_igt's had_lemma_id branch: a token that
        # already carried a lemma_id/lemma/lexical_gloss from an earlier
        # resolution (the author reopened the popover and picked a
        # DIFFERENT alternative) must not keep pointing at that stale
        # resolution once the new reading resolves to nothing — otherwise
        # the rebuilt `morphemes` below and the resolved lexeme silently
        # disagree. Unlike overlay_user_igt this does not also clear
        # `normalized`/`tier`: those describe the import-time overlay's own
        # provenance and are never touched by this function's other branch
        # either (the found.lexeme_id branch above leaves them alone too).
        for field in ("lemma_id", "lemma", "lexical_gloss", "grammatical_info",
                      "lift_entry_id", "lift_guid", "lift_sense_id"):
            token[field] = None
        token["grammatical_infos"] = []
    token["morphemes"] = new_cells
    token["uncertainty"] = []
    # Rebind, never `.clear()`: `original_alternatives` in
    # resolve_ambiguous_occurrences is captured as an alias of this same
    # list (`token.get("alternatives") or []`), and `.clear()` would mutate
    # that aliased list too, silently breaking every propagation match.
    token["alternatives"] = []
    return {"lexeme_id": token.get("lemma_id"), "lemma": token.get("lemma"),
            "lexical_gloss": token.get("lexical_gloss"), "ambiguous": ambiguous}


def overlay_user_igt(sentence: dict[str, Any],
                     imported_tokens: list[dict[str, Any]],
                     case=None, analyzer=None) -> dict[str, Any]:
    """Overlay source IGT on an already automatically glossed editor sentence.

    Human morphology is never sent back through the analyser.  Imported
    columns may absorb several adjacent running-text tokens when the source IGT
    binds a clitic or restores an unbound sandhi form.  Anything that cannot be
    aligned with reasonable spelling evidence stays automatic and is reported.
    """
    original_tokens = list(sentence.get("tokens") or [])
    candidates = original_tokens
    alignment = _partial_igt_alignment(imported_tokens, candidates)
    # Computed once per sentence, over the original tokens — not inside the
    # alignment loop, and not per merged group, since a group's membership is
    # only known once the loop below builds it.
    arabic_insert_ids = _arabic_insert_token_ids(original_tokens)

    unmatched_imported: set[int] = set(range(len(imported_tokens)))
    failed_cells: list[int] = []
    applied = 0
    rebuilt: list[dict[str, Any]] = []
    resolved: list[dict[str, Any]] = []
    cursor = 0

    for kind, imported_index, start, stop, ratio in alignment:
        if kind == "skip_imported":
            continue
        if kind == "skip_candidate":
            if start >= cursor:
                rebuilt.extend(copy.deepcopy(candidates[cursor:stop]))
                cursor = stop
            continue
        assert kind == "match" and imported_index is not None
        if start > cursor:
            rebuilt.extend(copy.deepcopy(candidates[cursor:start]))
        group = candidates[start:stop]
        if not group:
            continue
        token = _merge_import_candidates(group)
        imported = imported_tokens[imported_index]
        cells = _imported_user_cells(token, imported, case)
        if cells is None:
            failed_cells.append(imported_index)
            rebuilt.extend(copy.deepcopy(group))
        else:
            token["auto_analysis"] = {
                field: copy.deepcopy(token.get(field)) for field in
                ("normalized", "confidence", "tier", "lemma", "lemma_id",
                 "lexical_gloss", "grammatical_info", "grammatical_infos",
                 "lift_entry_id", "lift_guid", "lift_sense_id", "lexeme_candidates",
                 "uncertainty", "alternatives", "morphemes")
            }
            if len(group) > 1:
                token["auto_analysis"]["source_tokens"] = copy.deepcopy(group)
            # The human IGT re-segmented `blōta` into `blōt-a`: the auto
            # analyser's lemma_id was bound to a stem span that turned out
            # wrong, so it is archived rather than trusted. But most imports
            # only correct the gloss text on a segmentation the analyser
            # already had right — there the lemma_id is still the same word,
            # and losing it is what left confirmed corpus vocabulary
            # untrackable in the senses journal (docs/LEXICAL_REUSABLE_SPEC_RU.md).
            #
            # A re-segmentation is not automatically a fresh, unrecognised
            # word, though: the human's own stem may already name a lexeme
            # the dictionary knows under a different span. The resolver below
            # is keyed off the human's stem, not the analyser's guess, so a
            # unique hit there is better evidence than the discarded span —
            # that is why it is tried before the archive-and-clear branch.
            had_lemma_id = token.get("lemma_id") is not None
            same_segmentation = (had_lemma_id
                                 and join_parts(token["auto_analysis"]["morphemes"])
                                 == join_parts(cells))
            stem_text, stem_gloss = _lexical_stem(cells)
            found = None
            if analyzer is not None and not same_segmentation:
                found = LR.resolve_lexeme(analyzer, nfc(token.get("surface") or ""),
                                          stem_text)
            token.pop("lexeme_candidates", None)
            if found is not None and found.candidates:
                # Выбор между омонимами делает автор (инвариант #7). Список
                # переживает любую ветку: и минт, и сброс переразобранного
                # токена — иначе он терялся именно там, где слово словарю
                # известно лучше всего. `pop` above also drops a stale list
                # from a previous import now that the word resolves clean.
                token["lexeme_candidates"] = list(found.candidates)
            if same_segmentation:
                # Most imports only correct the gloss text on a segmentation
                # the analyser already had right (see the comment above), so
                # the resolver above is never called here — the token already
                # carries the correct lemma_id. That is exactly the ordinary
                # case the senses journal exists for: ask the lexicon directly
                # whether the author's own gloss closes a lexeme still waiting
                # on a meaning.
                if analyzer is not None and stem_gloss:
                    lemma_id = token.get("lemma_id")
                    gloss, status = (analyzer.lexical.resolve(lemma_id)
                                     if lemma_id else (None, "unknown_lexeme"))
                    if status in ("translation_pending", "no_sense"):
                        resolved.append({"lexeme_id": lemma_id,
                                         "lemma": nfc(token.get("lemma") or ""),
                                         "gloss": stem_gloss, "status": status})
            elif found is not None and found.lexeme_id:
                token["lemma_id"] = found.lexeme_id
                # `found.lemma` here is `lemma_canonical` read straight from
                # the lexicon, never a lemma this code synthesises.
                # `looks_like_a_lemma` guards a synthesised lemma (the v4
                # productive candidate, the minted stem, the Arnold headword
                # below) against carrying a segmentation like `apt-aṯ`. It is
                # never applied to a dictionary reading: the lexicon is the
                # authority on what its own citation form looks like, hyphen
                # and all (`ʕemm-`, `l-ukbil` are real bound-form headwords,
                # not segmentation artifacts).
                #
                # This is not inconsistent with the Arnold-headword branch
                # below, which DOES run `looks_like_a_lemma` over
                # `found.lemma`: that headword is text Arnold's entry was
                # parsed out of, and demonstrably carries the same kind of
                # hyphen segmentation artifact this guard exists to catch
                # (unlike `lemma_canonical` here, which is curated by hand and
                # exempt). Same function, two different provenances for
                # "the lemma we were handed" — only one of them is curated.
                token["lemma"] = found.lemma
                token["lexical_gloss"] = found.gloss or stem_gloss
                token["normalized"] = None
                token["tier"] = None
                if stem_gloss and found.status in ("translation_pending", "no_sense"):
                    resolved.append({"lexeme_id": found.lexeme_id,
                                     "lemma": found.lemma, "gloss": stem_gloss,
                                     "status": found.status})
            elif had_lemma_id:
                for field in ("normalized", "tier", "lemma", "lemma_id", "lexical_gloss"):
                    token[field] = None
            else:
                # The analyser never recognised this word at all (it is not
                # in `data/final/lexemes.jsonl`) — with no lemma_id of its own
                # it would vanish from the senses journal the moment it is
                # imported, so a fresh, stable one is minted from the human's
                # own stem gloss instead. Deterministic from the stem text
                # alone, so the same new word converges on one id wherever it
                # recurs; a `lexnew_` prefix keeps it distinct from a real
                # `lex_...` id.
                #
                # An Arabic code-switch insert (ᴬ…ᴬ) is a real part of the
                # text, but it is not a dictionary entry — it never got here
                # as a genuinely new *Aramaic* word, it got here because the
                # analyser (rightly) does not know Arabic. Skip only the mint:
                # the morphology, gloss cells and uncertainty below still
                # apply exactly as for any other token.
                is_arabic_insert = any(item.get("token_id") in arabic_insert_ids
                                       for item in group)
                if stem_text and not is_arabic_insert:
                    token["lemma_id"] = stable_id("lexnew", stem_text.casefold())
                    # Арнольд знает словарную форму даже там, где лексемы у нас
                    # нет: `ničrahhab` → `črahhab`. В словник должна попасть
                    # она, а не основа.
                    headword = (found.lemma if found is not None
                               and found.status == "arnold_headword" else "")
                    token["lemma"] = (headword if headword and looks_like_a_lemma(headword)
                                      else nominal_citation(cells, stem_text) or stem_text)
                    token["lexical_gloss"] = stem_gloss
                token["normalized"] = None
                token["tier"] = None
            token["uncertainty"] = []
            token["alternatives"] = []
            token["morphemes"] = cells
            if ratio < 0.8:
                for cell in cells:
                    if cell["status"] == CONFIDENT:
                        cell["status"] = UNCERTAIN
            token["locked"] = True
            token["auto_pending"] = None
            statuses = {cell["status"] for cell in cells}
            token["confidence"] = ("LOW" if UNRESOLVED in statuses else
                                   "MEDIUM" if UNCERTAIN in statuses else "HIGH")
            if statuses != {CONFIDENT}:
                token["uncertainty"].append({"type": "source_igt_review",
                    "reason": "Incomplete, uncertain or weakly aligned source IGT"})
            token["imported_igt"] = {
                "source": copy.deepcopy(imported.get("source") or {}),
                "gloss_original": imported.get("gloss_original"),
                "alignment_ratio": ratio,
                "source_surfaces": [item.get("surface") for item in group],
            }
            rebuilt.append(token)
            unmatched_imported.discard(imported_index)
            applied += 1
        cursor = max(cursor, stop)

    if cursor < len(candidates):
        rebuilt.extend(copy.deepcopy(candidates[cursor:]))

    # Speaker tags were skipped by the matcher in place, including internal
    # changes of speaker. Never move them to the start of the sentence.
    sentence["tokens"] = rebuilt

    return {
        "applied": applied,
        "automatic_candidates": len(candidates),
        "imported": len(imported_tokens),
        "unmatched_imported": sorted(unmatched_imported),
        "failed_cells": sorted(set(failed_cells)),
        "resolved_lexemes": resolved,
    }


def _split_import_speaker(line: str) -> tuple[str, str]:
    """Return a leading speaker only when it passes the automatic rule."""
    pieces = nfc(line).strip().split(maxsplit=1)
    if len(pieces) == 2 and looks_like_speaker_tag(pieces[0]):
        return pieces[0], pieces[1]
    return "", nfc(line).strip()


def _build_partial_source_sentence(line: str, translation: str, glosser,
                                   index: int, document_id: str,
                                   case=None) -> dict[str, Any]:
    """Gloss only the utterance body so a speaker tag cannot merge with word 1."""
    speaker, body = _split_import_speaker(line)
    if speaker:
        record = glosser.gloss_token(speaker[:-1], sentence_initial=True)
        probe = build_token({**record, "source_token": speaker}, "speaker-probe", 0, case)
        mark_automatic_speakers([probe])
        if not is_speaker(probe):
            speaker, body = "", line
    sentence = build_sentence(
        glosser.gloss_sentence(body or line),
        index,
        document_id,
        translation,
        case,
    )
    sentence["source_line"] = nfc(line).strip()
    if not speaker:
        return sentence

    speaker_token = build_token(
        {
            "source_token": speaker,
            "token": speaker,
            "segmentation": speaker,
            "gloss": SPEAKER_GLOSS,
            "confidence": "HIGH",
            "alternatives": [],
        },
        sentence["sentence_id"],
        -1,
        case,
    )
    mark_speaker(speaker_token)
    sentence["tokens"].insert(0, speaker_token)
    return sentence



def build_partial_igt_document(parsed: dict[str, Any], glosser,
                               document_id: str | None = None,
                               title: str = "", case=None
                               ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build an editor document from a DOCX containing partial human IGT.

    First the running transcription is glossed normally.  Then every source IGT
    table that can be aligned safely replaces the corresponding automatic
    morphology and is marked ``origin=user`` + ``locked=True``.  Transcription
    turns that have no IGT table remain ordinary automatic output.

    The parser may contain an implicit preface segment (glossing rules,
    vocabulary, etc.) ahead of the first time-coded segment. It is dropped
    only when it is short — a genuine untimed preface is a handful of
    example lines, not a real turn count. A source whose recording only
    starts getting timecodes partway through (FS_SHABO_LAST: 22 real
    transcription turns before the first marker) must not lose its opening
    dialogue to this guard.
    """
    segments = list(parsed.get("segments") or [])
    first_timed = next(
        (index for index, segment in enumerate(segments) if segment.get("timecode")),
        None,
    )
    UNTIMED_PREFACE_MAX_TURNS = 5
    if (first_timed is not None and first_timed > 0
            and len(segments[0].get("transcription_turns") or [])
            <= UNTIMED_PREFACE_MAX_TURNS):
        segments = segments[first_timed:]

    line_specs: list[dict[str, Any]] = []
    segment_by_id = {segment.get("segment_id"): segment for segment in segments}
    for segment in segments:
        transcriptions = [
            nfc(value).strip() for value in segment.get("transcription_turns") or []
            if nfc(value).strip()
        ]
        translations = [
            nfc(value).strip() for value in segment.get("translation_turns_en") or []
        ]
        for pair_index, transcription in enumerate(transcriptions):
            line_specs.append({
                "segment_id": segment.get("segment_id"),
                "pair_index": pair_index,
                "timecode": segment.get("timecode"),
                "segment_marker": segment.get("segment_marker"),
                "transcription": transcription,
                "translation": (
                    translations[pair_index]
                    if pair_index < len(translations)
                    else ""
                ),
            })

    if not line_specs:
        raise SessionError("the DOCX contains no transcription turns to import")

    source_text = "\n".join(spec["transcription"] for spec in line_specs)
    document_id = document_id or stable_id("edoc", source_text[:400], now())
    stamp = now()
    document = {
        "schema": SCHEMA,
        "document_id": document_id,
        "title": title or parsed.get("source_file") or "Imported IGT",
        "created": stamp,
        "updated": stamp,
        "source_text": source_text,
        "sentences": [
            _build_partial_source_sentence(
                spec["transcription"],
                spec["translation"],
                glosser,
                index,
                document_id,
                case,
            )
            for index, spec in enumerate(line_specs)
        ],
    }
    for sentence, spec in zip(document["sentences"], line_specs):
        sentence["source_locator"] = {
            "file": parsed.get("source_file"),
            "segment_id": spec["segment_id"],
            "segment_marker": spec["segment_marker"],
            "pair_index": spec["pair_index"],
            "timecode": spec["timecode"],
        }

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    skipped_examples: list[dict[str, Any]] = []
    for example in parsed.get("examples") or []:
        segment_id = example.get("segment_id")
        if segment_id not in segment_by_id:
            continue
        pair_index = (example.get("context_match") or {}).get("pair_index")
        segment = segment_by_id[segment_id]
        if pair_index is None:
            turns = segment.get("transcription_turns") or []
            if len(turns) == 1:
                pair_index = 0
            else:
                skipped_examples.append({
                    "example_id": example.get("example_id"),
                    "source_locator": example.get("source_locator"),
                    "reason": "no_unique_context_turn",
                })
                continue
        grouped.setdefault((segment_id, int(pair_index)), []).extend(
            example.get("tokens") or []
        )

    report: dict[str, Any] = {
        "source_file": parsed.get("source_file"),
        "lines": len(document["sentences"]),
        "segments": len(segments),
        "source_igt_tokens": 0,
        "user_tokens": 0,
        "auto_tokens": 0,
        "unmatched_igt_tokens": [],
        "resolved_lexemes": [],
        "skipped_examples": skipped_examples,
        "parser_skipped_tables": list(parsed.get("skipped_tables") or []),
        "unassigned_translations": [
            {"segment_id": segment.get("segment_id"), **item}
            for segment in segments for item in segment.get("unassigned_translations") or []
        ],
    }

    for sentence, spec in zip(document["sentences"], line_specs):
        imported = grouped.get((spec["segment_id"], spec["pair_index"]), [])
        if not imported:
            continue
        outcome = overlay_user_igt(sentence, imported, case=case,
                                   analyzer=glosser.analyzer)
        report["source_igt_tokens"] += len(imported)
        report["user_tokens"] += outcome["applied"]
        report["resolved_lexemes"].extend(outcome["resolved_lexemes"])
        for imported_index in outcome["unmatched_imported"]:
            item = imported[imported_index]
            report["unmatched_igt_tokens"].append({
                "segment_marker": spec["segment_marker"],
                "pair_index": spec["pair_index"],
                "surface": (
                    item.get("surface_original")
                    or item.get("segmentation")
                    or ""
                ),
                "source": item.get("source") or {},
            })

    language_tokens = 0
    for sentence in document.get("sentences") or []:
        for token in sentence.get("tokens") or []:
            if not is_speaker(token):
                language_tokens += 1
    report["auto_tokens"] = max(0, language_tokens - report["user_tokens"])
    document["import_report"] = report
    document["import_source"] = {
        "kind": "partial_igt_docx",
        "file": parsed.get("source_file"),
    }
    document["updated"] = now()
    validate(document)
    return document, report



# ---------------------------------------------------------------------------
# integrity
# ---------------------------------------------------------------------------
class SessionError(ValueError):
    """A session document that cannot be trusted to render three aligned lines."""


def validate(document: dict[str, Any]) -> dict[str, Any]:
    """Check the structural promises the editor UI relies on.

    Raises rather than repairs: a document that lost its alignment is a bug to
    find, not damage to paper over in the renderer.
    """
    if document.get("schema") != SCHEMA:
        raise SessionError(f"unknown schema {document.get('schema')!r}")
    seen_ids: set[str] = set()
    for sentence in document.get("sentences") or []:
        for token in sentence.get("tokens") or []:
            where = f"{sentence.get('sentence_id')}/{token.get('token_id')}"
            morphemes = token.get("morphemes") or []
            if not morphemes:
                raise SessionError(f"{where}: token has no morpheme cells")
            for morpheme in morphemes:
                morph_id = morpheme.get("morph_id")
                if not morph_id or morph_id in seen_ids:
                    raise SessionError(f"{where}: duplicate or missing morph_id "
                                       f"{morph_id!r}")
                seen_ids.add(morph_id)
                # UNDERTIE joins two running-text words an imported IGT column
                # still glosses one at a time (_split_igt_cell_parts): a
                # legitimate cell boundary, not a morpheme boundary within a
                # single written word, but a separator nonetheless.
                if morpheme.get("sep", "") not in ("", *SEPARATORS, " ", UNDERTIE):
                    raise SessionError(f"{where}: bad separator "
                                       f"{morpheme.get('sep')!r}")
                if morpheme.get("status") not in STATUSES:
                    raise SessionError(f"{where}: bad status "
                                       f"{morpheme.get('status')!r}")
            if morphemes[0].get("sep"):
                raise SessionError(f"{where}: first morpheme carries a separator")
        for comment in sentence.get("comments") or []:
            anchor = comment.get("anchor") or {}
            if not anchor.get("token_id"):
                raise SessionError(f"comment {comment.get('comment_id')} "
                                   "is not anchored to a token")
        for thread in sentence.get("discussions") or []:
            anchor = thread.get("anchor") or {}
            if not anchor.get("token_id"):
                raise SessionError(f"discussion {thread.get('thread_id')} "
                                   "is not anchored to a token")
            if not thread.get("messages"):
                raise SessionError(f"discussion {thread.get('thread_id')} "
                                   "has no messages")
    return document


def segmentation_line(sentence: dict[str, Any]) -> str:
    return " ".join(join_parts(t["morphemes"]) for t in sentence.get("tokens") or [])


def render_gloss(token: dict[str, Any]) -> str:
    """Line 3 for one token, with an explicit `??` per unglossed cell.

    Every cell must be visible: collapsing empty ones would render `b=Amērca`
    as a bare `=` and hide the fact that two labels are missing.
    """
    return morphology_text("".join(
        f"{m.get('sep', '')}{m.get('gloss') or UNRESOLVED_MARK}"
        for m in token.get("morphemes") or [])) or UNRESOLVED_MARK


def gloss_line(sentence: dict[str, Any]) -> str:
    return " ".join(render_gloss(t) for t in sentence.get("tokens") or [])


def finished_reading(token: dict[str, Any]) -> tuple[str, str, str] | None:
    """(surface, segmentation, gloss) of a word the user has finished glossing.

    `None` while anything is still open: a half-glossed word is not a reading
    worth carrying into every other text, and neither is a speaker tag. The `-`, `=` and `+` boundaries all qualify. Their exact spelling is
    preserved in the stored reusable reading; `+` is never normalized to `=`.
    """
    if is_speaker(token):
        return None
    cells = token.get("morphemes") or []
    if not cells or not any(c.get("origin") == USER for c in cells):
        return None
    if any(not (c.get("gloss") or "").strip()
           or "?" in c["gloss"] or "*" in c["gloss"] for c in cells):
        return None
    if any((c.get("sep") or "") not in ("", "-", "=", "+") for c in cells):
        return None
    segmentation = join_parts(cells)
    surface = segmentation.replace("-", "").replace("=", "").replace("+", "")
    if not surface:
        return None
    return surface, segmentation, join_parts(cells, key="gloss")


def counts(document: dict[str, Any]) -> dict[str, int]:
    """How much of the document still needs a human, by cell."""
    tally = {status: 0 for status in STATUSES}
    for sentence in document.get("sentences") or []:
        for token in sentence.get("tokens") or []:
            if is_speaker(token):
                continue
            for morpheme in token.get("morphemes") or []:
                tally[morpheme["status"]] = tally.get(morpheme["status"], 0) + 1
    return tally
