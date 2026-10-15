"""Streaming FLEx ``.flextext`` import for the browser editor.

The legacy project was built around DOCX/Maalula sources.  This module adds a
second, deliberately loss-averse ingest path for FLEx interlinear XML.  It
keeps the raw FLEx items/attributes on every level while also projecting the
parts the editor understands (surface, morpheme cells, gloss, translation).

Nothing here applies Maalula transcription or morphology rules.  A FLEX file
is human annotation and is imported as such.
"""
from __future__ import annotations

import copy
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from maalula_glosser.io_utils import stable_id


def nfc(value: str | None) -> str:
    return unicodedata.normalize("NFC", value or "")


def _tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in list(element) if _tag(child) == name]


def _first(element: ET.Element, name: str) -> ET.Element | None:
    return next((child for child in list(element) if _tag(child) == name), None)


def _items(element: ET.Element) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for child in _children(element, "item"):
        attrs = {str(k): str(v) for k, v in child.attrib.items()}
        out.append({
            "type": attrs.get("type", ""),
            "lang": attrs.get("lang"),
            "value": nfc("".join(child.itertext())),
            "attributes": attrs,
        })
    return out


def _item(items: list[dict[str, Any]], item_type: str, *, lang: str | None = None) -> str:
    for row in items:
        if row.get("type") != item_type:
            continue
        if lang is not None and row.get("lang") != lang:
            continue
        return nfc(row.get("value"))
    return ""


def _item_values(items: list[dict[str, Any]], item_type: str) -> list[str]:
    return [nfc(row.get("value")) for row in items
            if row.get("type") == item_type and nfc(row.get("value"))]


def _boundary_parts(value: str) -> tuple[str, str, str]:
    """Return (leading boundary, core, trailing boundary).

    FLEx exports in this corpus store clitic boundaries in the morph spelling
    itself (``m=``, ``=wa``), while ordinary suffixes may be plain.  The editor
    stores the separator *before* a cell, so a trailing boundary is carried to
    the next morph when possible and retained separately when the word ends
    there (e.g. a standalone proclitic).
    """
    value = nfc(value)
    lead = value[0] if value[:1] in {"-", "=", "+"} else ""
    if lead:
        value = value[1:]
    trail = value[-1] if value[-1:] in {"-", "=", "+"} else ""
    if trail:
        value = value[:-1]
    return lead, value, trail


def _inferred_boundary(morph_type: str) -> str:
    kind = (morph_type or "").casefold()
    if "clitic" in kind:
        return "="
    if kind in {"prefix", "suffix", "particle"}:
        return "-"
    return "-"


def _parse_morph(element: ET.Element, fallback: str) -> dict[str, Any]:
    items = _items(element)
    attrs = {str(k): str(v) for k, v in element.attrib.items()}
    raw_form = _item(items, "txt")
    raw_gloss = _item(items, "gls")
    form_lead, form, form_trail = _boundary_parts(raw_form)
    gloss_lead, gloss, gloss_trail = _boundary_parts(raw_gloss)
    return {
        "id": stable_id("flexm", attrs.get("guid") or fallback),
        "guid": attrs.get("guid"),
        "type": attrs.get("type") or "unknown",
        "raw_form": raw_form,
        "raw_gloss": raw_gloss,
        "form": form,
        "gloss": gloss,
        "lemma": _item(items, "cf"),
        "pos": _item(items, "msa"),
        "leading_boundary": form_lead or gloss_lead,
        "trailing_boundary": form_trail or gloss_trail,
        "items": items,
        "attributes": attrs,
    }


def _prepare_morphs(morphs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    carried = ""
    for index, morph in enumerate(morphs):
        row = copy.deepcopy(morph)
        explicit = row.get("leading_boundary") or ""
        if index == 0:
            sep = explicit
        else:
            sep = explicit or carried or _inferred_boundary(str(row.get("type") or ""))
        # A leading separator on the first morph is source evidence, but the
        # editor cannot have a boundary before the word. Preserve it in FLEX
        # metadata and keep the editable cells structurally valid.
        row["sep"] = "" if index == 0 else sep
        carried = row.get("trailing_boundary") or ""
        out.append(row)
    return out


def _analysis(morphs: list[dict[str, Any]], surface: str) -> dict[str, Any]:
    prepared = _prepare_morphs(morphs)
    if not prepared:
        return {
            "segmentation": "", "source_segmentation": "",
            "gloss": "", "source_gloss": "", "lemma": "", "pos": "",
            "lexical_gloss": "", "lemma_id": None, "morphemes": [],
        }

    segmentation = "".join((m.get("sep") or "") + (m.get("form") or "")
                           for m in prepared)
    gloss = "".join((m.get("sep") or "") + (m.get("gloss") or "")
                    for m in prepared)
    source_seg = "".join(m.get("raw_form") or "" for m in morphs)
    source_gloss = "".join(m.get("raw_gloss") or "" for m in morphs)

    lexical = next((m for m in prepared if (m.get("type") or "").casefold() in {"stem", "root"}
                    and (m.get("form") or m.get("lemma"))), None)
    lexical = lexical or next((m for m in prepared if m.get("form") or m.get("lemma")), None)
    lemma = nfc((lexical or {}).get("lemma") or (lexical or {}).get("form") or surface)
    lexical_gloss = nfc((lexical or {}).get("gloss"))
    pos_values = list(dict.fromkeys(nfc(m.get("pos")) for m in prepared if nfc(m.get("pos"))))
    pos = "; ".join(pos_values)
    lemma_id = (stable_id("flexlex", lemma.casefold(), lexical_gloss.casefold(), pos.casefold())
                if lemma else None)
    return {
        "segmentation": segmentation,
        "source_segmentation": source_seg,
        "gloss": gloss,
        "source_gloss": source_gloss,
        "lemma": lemma,
        "pos": pos,
        "lexical_gloss": lexical_gloss,
        "lemma_id": lemma_id,
        "morphemes": prepared,
    }


def _parse_word(element: ET.Element, fallback: str) -> dict[str, Any]:
    items = _items(element)
    attrs = {str(k): str(v) for k, v in element.attrib.items()}
    punctuation = _item(items, "punct")
    surface = _item(items, "txt") or punctuation
    morph_container = _first(element, "morphemes")
    morphs = [] if morph_container is None else [
        _parse_morph(morph, f"{fallback}:m{index}")
        for index, morph in enumerate(_children(morph_container, "morph"))
    ]
    analysis = _analysis(morphs, surface)
    return {
        "id": stable_id("flexw", attrs.get("guid") or fallback),
        "guid": attrs.get("guid"),
        "surface": nfc(surface),
        "punctuation": bool(punctuation and not _item(items, "txt")),
        "items": items,
        "attributes": attrs,
        "analysis": analysis,
    }


def _parse_phrase(element: ET.Element, text_key: str, pindex: int,
                  sindex: int) -> dict[str, Any]:
    items = _items(element)
    attrs = {str(k): str(v) for k, v in element.attrib.items()}
    key = attrs.get("guid") or f"{text_key}:p{pindex}:s{sindex}"
    words_container = _first(element, "words")
    words = [] if words_container is None else [
        _parse_word(word, f"{key}:w{index}")
        for index, word in enumerate(_children(words_container, "word"))
    ]
    baseline = _item(items, "txt") or " ".join(w["surface"] for w in words if w["surface"])
    translations = _item_values(items, "gls")
    return {
        "id": stable_id("flexseg", key),
        "guid": attrs.get("guid"),
        "segnum": _item(items, "segnum") or f"{pindex + 1}.{sindex + 1}",
        "baseline": nfc(baseline),
        "translation": translations[0] if translations else "",
        "translations": translations,
        "notes": _item_values(items, "note"),
        "words": words,
        "items": items,
        "attributes": attrs,
    }


def _parse_text(element: ET.Element, filename: str, text_index: int) -> dict[str, Any]:
    metadata = _items(element)
    attrs = {str(k): str(v) for k, v in element.attrib.items()}
    title = (_item(metadata, "title", lang="en") or _item(metadata, "title", lang="tru")
             or _item(metadata, "title") or f"Text {text_index + 1}")
    key = attrs.get("guid") or f"{filename}:{text_index}:{title}"
    segments: list[dict[str, Any]] = []
    paragraphs = _first(element, "paragraphs")
    if paragraphs is not None:
        for pindex, paragraph in enumerate(_children(paragraphs, "paragraph")):
            phrases = _first(paragraph, "phrases")
            if phrases is None:
                continue
            for sindex, phrase in enumerate(_children(phrases, "phrase")):
                segments.append(_parse_phrase(phrase, key, pindex, sindex))

    languages: list[dict[str, Any]] = []
    language_container = _first(element, "languages")
    if language_container is not None:
        for language in _children(language_container, "language"):
            languages.append({str(k): str(v) for k, v in language.attrib.items()})
    return {
        "id": stable_id("flextext", key),
        "source_key": key,
        "source_guid": attrs.get("guid"),
        "title": nfc(title),
        "source_filename": filename,
        "metadata": metadata,
        "attributes": attrs,
        "languages": languages,
        "segments": segments,
    }


def iter_flextext(path: Path | str, *, filename: str | None = None) -> Iterator[dict[str, Any]]:
    """Yield each ``<interlinear-text>`` without loading the whole XML string."""
    source = Path(path)
    name = filename or source.name
    index = 0
    try:
        for _event, element in ET.iterparse(source, events=("end",)):
            if _tag(element) != "interlinear-text":
                continue
            yield _parse_text(element, name, index)
            index += 1
            element.clear()
    except ET.ParseError as exc:
        raise ValueError(f"invalid FLEx XML: {exc}") from exc
    if index == 0:
        raise ValueError("no <interlinear-text> blocks found")


def parse_flextext(path: Path | str, *, filename: str | None = None) -> list[dict[str, Any]]:
    return list(iter_flextext(path, filename=filename))


def _editor_cells(word: dict[str, Any], token_id: str) -> list[dict[str, Any]]:
    from maalula_glosser.editor import session as S

    morphs = word.get("analysis", {}).get("morphemes") or []
    if not morphs:
        return [{
            "morph_id": stable_id("morph", token_id, 0, word.get("surface") or ""),
            "sep": "", "form": word.get("surface") or "", "gloss": "",
            "suggested": "", "status": S.UNRESOLVED, "origin": S.AUTO,
            "alternatives": [],
        }]
    cells: list[dict[str, Any]] = []
    for index, morph in enumerate(morphs):
        label = nfc(morph.get("gloss"))
        cells.append({
            "morph_id": stable_id("morph", token_id, index, morph.get("form") or ""),
            "sep": morph.get("sep") or "",
            "form": nfc(morph.get("form")),
            "gloss": label,
            "suggested": "",
            "status": S.CONFIDENT if label else S.UNRESOLVED,
            "origin": S.USER if label else S.AUTO,
            "alternatives": [],
            "flex_morph_type": morph.get("type"),
            "flex_raw_form": morph.get("raw_form"),
            "flex_raw_gloss": morph.get("raw_gloss"),
            "flex_trailing_boundary": morph.get("trailing_boundary") or "",
            "flex_items": copy.deepcopy(morph.get("items") or []),
            "flex_attributes": copy.deepcopy(morph.get("attributes") or {}),
        })
    return cells


def flex_text_to_document(text: dict[str, Any]) -> dict[str, Any]:
    """Project one FLEx text into the existing editable-session schema."""
    from maalula_glosser.editor import session as S

    document_id = stable_id("edoc-flex", text.get("source_key") or text.get("id"))
    sentences: list[dict[str, Any]] = []
    source_lines: list[str] = []
    for sindex, segment in enumerate(text.get("segments") or []):
        sentence_id = stable_id("esent", document_id, sindex)
        tokens: list[dict[str, Any]] = []
        position = 0
        for word in segment.get("words") or []:
            if word.get("punctuation"):
                continue
            surface = nfc(word.get("surface"))
            if not surface:
                continue
            analysis = word.get("analysis") or {}
            has_analysis = bool(analysis.get("morphemes") and
                                (analysis.get("segmentation") or analysis.get("gloss")))
            token_id = stable_id("etok", sentence_id, position, surface)
            cells = _editor_cells(word, token_id)
            token = {
                "token_id": token_id,
                "surface": surface,
                "sandhi_group": "",
                "normalized": surface,
                "confidence": "HIGH" if has_analysis else "LOW",
                "tier": "FLEX" if has_analysis else None,
                "lemma": analysis.get("lemma") or None,
                "lemma_id": analysis.get("lemma_id"),
                "lexical_gloss": analysis.get("lexical_gloss") or None,
                "uncertainty": ([] if has_analysis else ["no analysis in FLEx source"]),
                "alternatives": [],
                "locked": has_analysis,
                "auto_pending": None,
                "morphemes": cells,
                "original_analysis": copy.deepcopy(analysis) if has_analysis else None,
                "imported_igt": {
                    "source": {"kind": "flextext", "filename": text.get("source_filename"),
                               "text_guid": text.get("source_guid"),
                               "segment_guid": segment.get("guid"),
                               "segnum": segment.get("segnum"),
                               "word_guid": word.get("guid")},
                    "gloss_original": analysis.get("source_gloss") or analysis.get("gloss"),
                    "segmentation_original": analysis.get("source_segmentation")
                                             or analysis.get("segmentation"),
                    "alignment_ratio": 1.0,
                    "source_surfaces": [surface],
                } if has_analysis else None,
                "flextext": {
                    "word_id": word.get("id"), "guid": word.get("guid"),
                    "items": copy.deepcopy(word.get("items") or []),
                    "attributes": copy.deepcopy(word.get("attributes") or {}),
                },
            }
            tokens.append(token)
            position += 1
        line = nfc(segment.get("baseline"))
        source_lines.append(line)
        sentences.append({
            "sentence_id": sentence_id,
            "source_line": line,
            "translation": nfc(segment.get("translation")),
            "tokens": tokens,
            "comments": [],
            "discussions": [],
            "flextext": {
                "segment_id": segment.get("id"), "guid": segment.get("guid"),
                "segnum": segment.get("segnum"), "notes": copy.deepcopy(segment.get("notes") or []),
                "items": copy.deepcopy(segment.get("items") or []),
                "attributes": copy.deepcopy(segment.get("attributes") or {}),
            },
        })
    stamp = S.now()
    return {
        "schema": S.SCHEMA,
        "document_id": document_id,
        "title": nfc(text.get("title")),
        "created": stamp,
        "updated": stamp,
        "source_text": "\n".join(source_lines),
        "sentences": sentences,
        "flextext": {
            "text_id": text.get("id"), "source_key": text.get("source_key"),
            "source_guid": text.get("source_guid"),
            "source_filename": text.get("source_filename"),
            "metadata": copy.deepcopy(text.get("metadata") or []),
            "attributes": copy.deepcopy(text.get("attributes") or {}),
            "languages": copy.deepcopy(text.get("languages") or []),
        },
    }


def evidence_from_text(text: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield one corpus-evidence row per analysed, non-punctuation token."""
    for segment in text.get("segments") or []:
        for word in segment.get("words") or []:
            if word.get("punctuation") or not word.get("surface"):
                continue
            analysis = word.get("analysis") or {}
            if not (analysis.get("segmentation") or analysis.get("gloss")):
                continue
            yield {
                "surface": nfc(word.get("surface")),
                "segmentation": nfc(analysis.get("segmentation")),
                "source_segmentation": nfc(analysis.get("source_segmentation")),
                "gloss": nfc(analysis.get("gloss")),
                "source_gloss": nfc(analysis.get("source_gloss")),
                "lemma": nfc(analysis.get("lemma")),
                "lemma_id": analysis.get("lemma_id"),
                "lexical_gloss": nfc(analysis.get("lexical_gloss")),
                "pos": nfc(analysis.get("pos")),
                "morphemes": copy.deepcopy(analysis.get("morphemes") or []),
                "source": {
                    "kind": "flextext", "filename": text.get("source_filename"),
                    "text_id": text.get("id"), "text_title": text.get("title"),
                    "text_guid": text.get("source_guid"), "segment_id": segment.get("id"),
                    "segment_guid": segment.get("guid"), "segnum": segment.get("segnum"),
                    "word_id": word.get("id"), "word_guid": word.get("guid"),
                },
            }
