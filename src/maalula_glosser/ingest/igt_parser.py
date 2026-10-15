from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from maalula_glosser.ingest.docx_ordered_reader import extract_editorial_metadata, read_ordered_docx
from maalula_glosser.io_utils import stable_id
from maalula_glosser.normalization import TranscriptionNormalizer, normalize_gloss_with_trace

TIME_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\s*[-–]\s*\d{1,2}:\d{2}(?::\d{2})?\b")
LEADING_POINT_TIME_RE = re.compile(
    r"^\s*\(?(\d{1,2})[:.]\s*(\d{2})\)?\s*:?[ \t]*(.*)$"
)
SEGMENT_RE = re.compile(r"^\s*\d+\.\s+.*" + TIME_RE.pattern)
LATIN_TRANSCRIPTION_RE = re.compile(r"[ʔʕāēīōūḏ̣ṯḥṣṭẓšžčġḳ]")
ENGLISH_HINT_RE = re.compile(
    r"\b(the|and|was|were|is|are|she|they|you|from|with|then|there|this|that|"
    r"right|like|because|when|after|before|people|not|but|have|has|had|will|"
    r"would|can|could|should|may|might|very|more|only|here|outside|village|"
    r"language|our|place|word|means|bread|meat|no|yes)\b",
    re.IGNORECASE,
)
WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
SPEAKER_PREFIX_RE = re.compile(r"^\s*[^:]{1,12}:\s*")
CYRILLIC_LOOKALIKE_RE = re.compile(r"[АВСЕНКМОРТХавсенкмортх]")
RESIDUAL_OLD_TRANSCRIPTION_RE = re.compile(r"[ḳɁḓ]")

# Editorial apparatus a Festschrift-style article surrounds its transcript
# with — a glossary note ("čiṭwīra 'development' (ln. 12), cf. nwaw 870 ..."),
# a participant list, a bibliography entry — reads as neither transcription
# nor its translation to the classifiers below (LATIN_TRANSCRIPTION_RE and
# _looks_like_translation both fire on it, since it quotes a Maalula word
# inside otherwise-English citation prose), so it must be recognised and
# dropped explicitly rather than misfiled as either.
GLOSSARY_ENTRY_RE = re.compile(r"\(ln\.\s*\d")
PARTICIPANT_LIST_RE = re.compile(r"^\s*present\s*:", re.IGNORECASE)
BIBLIOGRAPHY_AUTHOR_RE = re.compile(r"^[A-ZÀ-ÿ][\w.'’-]*,\s*[A-ZÀ-ÿ]")
BIBLIOGRAPHY_YEAR_RE = re.compile(r"\b(?:19|20)\d\d\.")


def _is_editorial_apparatus(text: str) -> bool:
    """Glossary notes, a participant list, and bibliography entries describe
    the transcript; they are never part of it or its translation."""
    if GLOSSARY_ENTRY_RE.search(text) or PARTICIPANT_LIST_RE.match(text):
        return True
    return bool(BIBLIOGRAPHY_AUTHOR_RE.match(text) and BIBLIOGRAPHY_YEAR_RE.search(text))


def _segment_parts(value: str) -> list[str]:
    return [part for part in re.split(r"([-=.])", value) if part != ""]


def _strip_speaker(value: str) -> str:
    return SPEAKER_PREFIX_RE.sub("", value).strip()


def _looks_like_translation(value: str) -> bool:
    """Distinguish English prose from transcription that contains diacritics.

    English translations often mention forms such as ``leḥmō`` or names such as
    ``Calaš``; treating the mere presence of one transcription character as
    decisive makes those paragraphs look like Maalula.  Function words and the
    proportion of plain-ASCII words are stronger signals.
    """
    words = WORD_RE.findall(value)
    english_hits = len(ENGLISH_HINT_RE.findall(value))
    transcription_hits = len(LATIN_TRANSCRIPTION_RE.findall(value))
    ascii_words = sum(1 for word in words if word.isascii() and word.isalpha())
    ascii_ratio = (ascii_words / len(words)) if words else 0.0
    if english_hits >= 2:
        return True
    if english_hits and transcription_hits == 0:
        return True
    if SPEAKER_PREFIX_RE.match(value) and not english_hits and transcription_hits:
        return False
    # A short English gloss can contain one cited Maalula form (``Meat
    # (besra) – bisrō``).  Two or more transcription characters, however, are
    # strong evidence that an otherwise ASCII-heavy sentence is the source.
    return len(words) >= 2 and ascii_ratio >= 0.60 and transcription_hits <= 1


def _context_key(value: str) -> str:
    value = _strip_speaker(value).lower()
    value = value.replace("-", " ").replace("=", " ").replace("‿", " ")
    value = re.sub(r"[^\wʔʕāēīōūḏ̣ṯḥṣṭẓšžčġḳ]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def _form_key(value: str) -> str:
    value = value.lower().replace("-", "").replace("=", "").replace("‿", "")
    value = re.sub(r"[^\wʔʕāēīōūḏ̣ṯḥṣṭẓšžčġḳ]+", "", value, flags=re.UNICODE)
    return value


def _restore_source_label_case(gloss: str, known_labels: set[str] | None) -> str:
    """Restore small-caps IGT labels without applying any label aliases.

    Word stores small-caps text as lower-case characters plus formatting, and
    the ordered DOCX reader intentionally returns text rather than run styling.
    Thus a visible ``love.NC-MP`` can arrive here as ``love.nc-mp``.  For a
    human-authoritative import we restore the spelling of labels the project
    recognises, but never turn one category into another (notably PST→SC or
    NC1/NC2→PRF).
    """
    spellings = {str(label).casefold(): str(label) for label in (known_labels or set())}
    if not spellings:
        return gloss
    parts = re.split(r"([-=.])", gloss)
    out: list[str] = []
    for part in parts:
        if part in {"-", "=", "."} or not part:
            out.append(part)
            continue
        stripped = part.strip()
        match = re.fullmatch(r"(.+?)(\?+)", stripped)
        base = match.group(1) if match else stripped
        uncertainty = match.group(2) if match else ""
        out.append(spellings.get(base.casefold(), base) + uncertainty)
    return "".join(out)


def _context_pairs(transcriptions: list[str], translations: list[str]) -> list[tuple[str, str, int]]:
    """Pair speaker turns by order, retaining unmatched turns."""
    count = max(len(transcriptions), len(translations))
    return [
        (
            transcriptions[index] if index < len(transcriptions) else "",
            translations[index] if index < len(translations) else "",
            index,
        )
        for index in range(count)
    ]


def _ordered_translations(events: list[tuple[str, str, int]]) -> tuple[list[str], list[dict]]:
    """Align paragraph runs without shifting translations over omitted turns.

    Parallel blocks of equal length pair by order. A single translation
    immediately following speech belongs to the last turn, not the first turn
    of the entire timed segment. Other incomplete blocks remain unassigned.
    """
    aligned: list[str] = []
    pending: list[int] = []
    translations: list[tuple[str, int]] = []
    unassigned: list[dict] = []

    def flush():
        if len(pending) == len(translations):
            for index, (value, _position) in zip(pending, translations):
                aligned[index] = value
        elif pending and len(translations) == 1:
            aligned[pending[-1]] = translations[0][0]
        else:
            unassigned.extend({"text": value, "element_index": position,
                               "reason": "ambiguous_translation_alignment"}
                              for value, position in translations)
        pending.clear()
        translations.clear()

    for kind, value, position in events:
        if kind == "source":
            if translations:
                flush()
            pending.append(len(aligned))
            aligned.append("")
        else:
            translations.append((value, position))
    flush()
    return aligned, unassigned


def _best_context(
    forms: list[str],
    transcriptions: list[str],
    translations: list[str],
    pair_usage: Mapping[int, int] | None = None,
) -> dict[str, Any]:
    pairs = _context_pairs(transcriptions, translations)
    # A table cell is usually one morpheme, but not always: a compound
    # number spelled as one cell ("səttīn xamsa u səttīna") must still match
    # word-for-word against a transcription — _form_key on the whole cell
    # concatenates it into one unbroken string ("səttīnxamsausəttīna") that
    # can never appear in space-separated running text. Splitting first
    # leaves an ordinary single-word cell exactly as before.
    form_keys = [key for form in forms for key in
                (_form_key(word) for word in form.split()) if key]
    best: tuple[float, int, int, str, str] | None = None
    usage = pair_usage or {}
    for transcription, translation, pair_index in pairs:
        haystack = _context_key(transcription)
        score = 0.0
        matched = 0
        for key in form_keys:
            if not key:
                continue
            if re.search(rf"(?<!\w){re.escape(key)}(?!\w)", haystack):
                score += 2.0
                matched += 1
            elif key in haystack:
                score += 1.0
                matched += 1
        # Content is primary.  If two repeated turns score equally, hand the
        # next table to the less-used turn instead of piling every identical
        # refrain onto the first/last occurrence.
        candidate = (
            score,
            -int(usage.get(pair_index, 0)),
            -pair_index,
            transcription,
            translation,
        )
        if best is None or candidate > best:
            best = candidate
    if best and best[0] > 0:
        return {
            "transcription": best[3],
            "translation": best[4],
            "pair_index": -best[2],
            "score": best[0],
            "matched_forms": int(min(len(form_keys), best[0])),
            "method": "token_overlap",
        }
    return {
        "transcription": " ".join(transcriptions),
        "translation": " ".join(translations),
        "pair_index": None,
        "score": 0.0,
        "matched_forms": 0,
        "method": "whole_segment_fallback",
    }


def parse_igt(
    path: str | Path,
    normalizer: TranscriptionNormalizer,
    known_labels: set[str] | None = None,
    gloss_aliases: Mapping[str, str] | None = None,
    preserve_source_gloss: bool = False,
) -> dict[str, Any]:
    path = Path(path)
    elements = read_ordered_docx(path)
    examples: list[dict[str, Any]] = []
    segments: dict[str, dict[str, Any]] = {}
    skipped_tables: list[dict[str, Any]] = []
    current_title = ""
    current_timecode = ""
    current_segment_marker = ""
    current_segment_title = ""
    current_segment_element_index: int | None = None
    segment_transcriptions: list[str] = []
    segment_translations: list[str] = []
    segment_events: list[tuple[str, str, int]] = []
    context_usage: dict[int, int] = {}
    table_ordinal = 0

    def save_current_segment() -> str | None:
        """Persist the current text segment even when it has no IGT table.

        The partial editor import needs the opposite halves of a mixed document:
        tables become locked USER morphology, while transcription turns with no
        tables must still survive so TextGlosser can analyse them automatically.
        """
        if not current_segment_marker:
            return None
        segment_id = stable_id("igtseg", path.name, current_segment_marker)
        aligned_translations, unassigned = _ordered_translations(segment_events)
        segments[segment_id] = {
            "segment_id": segment_id,
            "source_text": current_segment_title or current_title,
            "segment_marker": current_segment_marker,
            "timecode": current_timecode or None,
            "transcription_turns": list(segment_transcriptions),
            "translation_turns_en": aligned_translations,
            "unassigned_translations": unassigned,
            "paragraph_order": [{"kind": kind, "text": value, "element_index": position}
                                for kind, value, position in segment_events],
            "source_locator": {
                "file": path.name,
                "segment_marker": current_segment_marker,
                "element_index": current_segment_element_index,
            },
        }
        return segment_id

    for element in elements:
        if element.kind == "paragraph":
            text = element.text.strip()
            if element.style and element.style.lower().startswith("heading") and text:
                current_title = text
            if not text:
                continue
            if _is_editorial_apparatus(text):
                continue
            time_match = TIME_RE.search(text)
            bare_timecode = bool(
                time_match
                and re.fullmatch(
                    rf"\s*[<\[]?\s*{TIME_RE.pattern}\s*[>\]]?\s*",
                    text,
                )
            )
            numbered_timecode = bool(
                SEGMENT_RE.match(text)
                or (time_match and re.match(r"^\s*\d+\.", text))
            )
            if numbered_timecode or bare_timecode:
                save_current_segment()
                current_segment_marker = text
                current_timecode = time_match.group(0) if time_match else ""
                current_segment_title = current_title
                current_segment_element_index = element.index
                segment_transcriptions = []
                segment_translations = []
                segment_events = []
                context_usage = {}
                continue

            # Later parts of Muše use point stamps (``26:09:``, ``(27.50)``)
            # rather than ranges.  A stamp may share its paragraph with the
            # transcription, so peel it off and continue classifying the rest.
            point_match = LEADING_POINT_TIME_RE.match(text)
            if point_match and not time_match:
                save_current_segment()
                point = f"{int(point_match.group(1)):02d}:{point_match.group(2)}"
                current_segment_marker = point
                current_timecode = point
                current_segment_title = current_title
                current_segment_element_index = element.index
                segment_transcriptions = []
                segment_translations = []
                segment_events = []
                context_usage = {}
                text = point_match.group(3).strip()
                if not text or text in {"-", "–"}:
                    continue

            if not current_segment_marker and (
                LATIN_TRANSCRIPTION_RE.search(text)
                or _looks_like_translation(text)
            ):
                current_segment_marker = f"implicit:{current_title or 'untitled'}"
                current_segment_title = current_title
                current_segment_element_index = element.index
                segment_transcriptions = []
                segment_translations = []
                segment_events = []
                context_usage = {}
            if not current_segment_marker:
                continue
            if _looks_like_translation(text):
                segment_translations.append(text)
                segment_events.append(("translation", text, element.index))
            elif LATIN_TRANSCRIPTION_RE.search(text):
                segment_transcriptions.append(text)
                segment_events.append(("source", text, element.index))
            continue

        table_ordinal += 1
        rows = element.table_rows
        source = {"file": path.name, "table": table_ordinal, "element_index": element.index}
        if len(rows) != 2:
            skipped_tables.append({**source, "reason": "not_two_rows", "shape": [len(rows), max((len(row) for row in rows), default=0)]})
            continue
        width = max(len(rows[0]), len(rows[1]))
        forms = rows[0] + [""] * (width - len(rows[0]))
        glosses = rows[1] + [""] * (width - len(rows[1]))
        aligned = [(form.strip(), gloss.strip()) for form, gloss in zip(forms, glosses) if form.strip() or gloss.strip()]
        if not aligned:
            skipped_tables.append({**source, "reason": "empty_table", "shape": [2, width]})
            continue
        morphological_score = sum(1 for form, gloss in aligned if any(mark in form for mark in "-=‿") or any(mark in gloss for mark in "-=."))
        # A table of numbers can be IGT without any morphological punctuation.
        # Its complete form row must be witnessed in the timed transcription;
        # unrelated vocabulary inventories still fail this test.
        phrase = _context_key(" ".join(form for form, _ in aligned))
        in_transcription = bool(current_timecode and phrase and any(
            f" {phrase} " in f" {_context_key(turn)} " for turn in segment_transcriptions))
        if morphological_score == 0 and len(aligned) > 3 and not in_transcription:
            skipped_tables.append({**source, "reason": "probable_inventory_not_igt", "shape": [2, width], "excerpt": element.text[:500]})
            continue
        if not current_segment_marker:
            skipped_tables.append({**source, "reason": "no_numbered_text_segment", "shape": [2, width], "excerpt": element.text[:500]})
            continue

        context = _best_context(
            [form for form, _ in aligned],
            segment_transcriptions,
            _ordered_translations(segment_events)[0],
            context_usage,
        )
        if context["pair_index"] is not None:
            pair_index = int(context["pair_index"])
            context_usage[pair_index] = context_usage.get(pair_index, 0) + 1
        transcription = context["transcription"]
        translation = context["translation"]
        segment_id = save_current_segment()
        assert segment_id is not None
        tokens: list[dict[str, Any]] = []
        unresolved = False
        for column, (form, gloss) in enumerate(aligned, start=1):
            uncertainties: list[dict[str, Any]] = []
            if not form or not gloss:
                unresolved = True
                uncertainties.append({"type": "other", "reason": "Empty aligned form or gloss cell"})
            if "?" in form or "?" in gloss:
                unresolved = True
                uncertainties.append({"type": "other", "reason": "Explicit question mark in source"})
            if RESIDUAL_OLD_TRANSCRIPTION_RE.search(form):
                unresolved = True
                uncertainties.append({"type": "transcription", "reason": "Residual old-transcription character in canonical IGT source"})
            if CYRILLIC_LOOKALIKE_RE.search(form):
                unresolved = True
                uncertainties.append({"type": "transcription", "reason": "Cyrillic lookalike character in IGT form"})

            surface = form.replace("-", "").replace("=", "")
            if preserve_source_gloss:
                # Partial editor import treats the DOCX as the human authority.
                # Restore Word small-caps, but do not collapse NC1/NC2, PST or
                # any other source label through pipeline aliases.
                canonical_gloss = _restore_source_label_case(gloss, known_labels)
                gloss_normalization = []
            else:
                canonical_gloss, gloss_normalization = normalize_gloss_with_trace(
                    gloss, known_labels, gloss_aliases
                )
            tokens.append({
                "token_id": stable_id("igtok", path.name, table_ordinal, column, form, gloss),
                "surface_original": surface,
                "surface_canonical": normalizer.canonical(surface, source_system="canonical"),
                "isolated_form": normalizer.canonical(form, source_system="canonical"),
                "segmentation": normalizer.canonical(form, source_system="canonical"),
                "gloss_original": gloss,
                "gloss": canonical_gloss,
                "gloss_normalization": gloss_normalization,
                "segmentation_parts": _segment_parts(normalizer.canonical(form, source_system="canonical")),
                "lemma_candidates": [],
                "selected_lemma": None,
                "morphemes": [],
                "uncertainties": uncertainties,
                "confidence": 0.6 if uncertainties else 0.92,
                "source": {"file": path.name, "table": table_ordinal, "column": column, "element_index": element.index},
            })
        examples.append({
            "example_id": stable_id("igt", path.name, table_ordinal, current_segment_marker),
            "segment_id": segment_id,
            "source_text": current_title,
            "segment_marker": current_segment_marker,
            "timecode": current_timecode or None,
            "speaker": None,
            "transcription_original": transcription,
            "translation_original": translation,
            "translation_en": translation,
            "context_match": {
                "pair_index": context["pair_index"],
                "score": context["score"],
                "matched_forms": context["matched_forms"],
                "method": context["method"],
            },
            "tokens": tokens,
            "quality_class": None,
            "proposed_quality_class": "noisy_or_unresolved" if unresolved else "gold_working",
            "editorial_notes": [],
            "source_locator": {"file": path.name, "table": table_ordinal, "element_index": element.index},
        })

    save_current_segment()

    return {
        "source_file": path.name,
        # dict insertion order is document order; sorting stable hashes destroys
        # the order needed to rebuild the interview as an editor document.
        "segments": list(segments.values()),
        "examples": examples,
        "skipped_tables": skipped_tables,
        "editorial_metadata": extract_editorial_metadata(path),
    }
