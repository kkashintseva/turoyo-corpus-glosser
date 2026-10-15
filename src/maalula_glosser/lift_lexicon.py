"""FieldWorks LIFT lexicon import for the FLEX/Turoyo profile.

The LIFT lexicon is authoritative for lexical-entry identity.  Citation forms,
variants, senses, and ``grammatical-info`` stay attached to the exact LIFT
entry that owns them; entries are never merged merely because their spellings
or meanings resemble one another.

The imported corpus remains a separate evidence layer.  It can attest forms,
contexts and new author additions, while LIFT provides stable entry GUIDs and
categories such as ``Preterite`` / ``Infectum``.
"""
from __future__ import annotations

import copy
import json
import shutil
import unicodedata
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

from maalula_glosser.io_utils import iter_jsonl, sha256_file, stable_id, write_json, write_jsonl
from maalula_glosser.state import writable

LIFT_DIR = "data/lift_lexicon"
LIFT_ENTRIES = LIFT_DIR + "/entries.jsonl"
LIFT_MANIFEST = LIFT_DIR + "/manifest.json"
LIFT_SOURCES = LIFT_DIR + "/sources"


def nfc(value: str | None) -> str:
    return unicodedata.normalize("NFC", value or "")


def _tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in list(element) if _tag(child) == name]


def _forms(container: ET.Element | None) -> list[dict[str, str]]:
    if container is None:
        return []
    rows: list[dict[str, str]] = []
    for form in _children(container, "form"):
        text = next((child for child in list(form) if _tag(child) == "text"), None)
        value = nfc("" if text is None else "".join(text.itertext())).strip()
        if value:
            rows.append({"lang": form.attrib.get("lang", ""), "text": value})
    return rows


def _traits(element: ET.Element) -> list[dict[str, str]]:
    return [
        {"name": nfc(child.attrib.get("name")).strip(),
         "value": nfc(child.attrib.get("value")).strip()}
        for child in _children(element, "trait")
    ]


def _fields(element: ET.Element) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field in _children(element, "field"):
        rows.append({
            "type": nfc(field.attrib.get("type")).strip(),
            "forms": _forms(field),
            "traits": _traits(field),
        })
    return rows


def _glosses(sense: ET.Element) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for gloss in _children(sense, "gloss"):
        text = next((child for child in list(gloss) if _tag(child) == "text"), None)
        value = nfc("" if text is None else "".join(text.itertext())).strip()
        if value:
            rows.append({"lang": gloss.attrib.get("lang", ""), "text": value})
    return rows


def _definitions(sense: ET.Element) -> list[dict[str, str]]:
    definition = next((child for child in list(sense) if _tag(child) == "definition"), None)
    return _forms(definition)


def _primary_form(rows: Iterable[dict[str, str]], preferred: str = "tru") -> str:
    rows = list(rows)
    return next((r["text"] for r in rows if r.get("lang") == preferred and r.get("text")),
                rows[0]["text"] if rows else "")


def _high_level_pos(values: Iterable[str]) -> str | None:
    """Compatibility POS only; exact LIFT grammatical-info remains separate."""
    lowered = [nfc(v).strip().casefold() for v in values if nfc(v).strip()]
    if any(any(marker in value for marker in
               ("preterite", "infectum", "imperative", "verb"))
           for value in lowered):
        return "verb"
    if any(value in {"adjective", "comparative"} for value in lowered):
        return "adjective"
    if any(value in {"noun", "substantive", "proper  noun", "proper noun"}
           for value in lowered):
        return "noun"
    return None


def _parse_sense(sense: ET.Element) -> dict[str, Any]:
    grammatical = next((child for child in list(sense)
                        if _tag(child) == "grammatical-info"), None)
    grammatical_info = nfc(grammatical.attrib.get("value") if grammatical is not None else "").strip()
    return {
        "sense_id": sense.attrib.get("id"),
        "order": sense.attrib.get("order"),
        "grammatical_info": grammatical_info or None,
        "grammatical_traits": _traits(grammatical) if grammatical is not None else [],
        "glosses": _glosses(sense),
        "definitions": _definitions(sense),
        "traits": _traits(sense),
        "fields": _fields(sense),
    }


def parse_lift_entry(element: ET.Element) -> dict[str, Any]:
    lift_id = element.attrib.get("id") or ""
    guid = element.attrib.get("guid") or ""
    lexical_unit = next((child for child in list(element) if _tag(child) == "lexical-unit"), None)
    citation = next((child for child in list(element) if _tag(child) == "citation"), None)
    lexical_forms = _forms(lexical_unit)
    citation_forms = _forms(citation)
    lemma = _primary_form(citation_forms) or _primary_form(lexical_forms)

    entry_traits = _traits(element)
    morph_type = next((row["value"] for row in entry_traits
                       if row["name"] == "morph-type"), None)
    variants: list[dict[str, Any]] = []
    for variant in _children(element, "variant"):
        traits = _traits(variant)
        variants.append({
            "forms": _forms(variant),
            "traits": traits,
            "morph_type": next((row["value"] for row in traits
                                if row["name"] == "morph-type"), None),
            "ref": variant.attrib.get("ref"),
        })

    senses = [_parse_sense(sense) for sense in _children(element, "sense")]
    grammatical_infos = list(dict.fromkeys(
        sense["grammatical_info"] for sense in senses if sense.get("grammatical_info")))
    relations = [{"type": rel.attrib.get("type"), "ref": rel.attrib.get("ref"),
                  "order": rel.attrib.get("order"), "traits": _traits(rel),
                  "fields": _fields(rel)}
                 for rel in _children(element, "relation")]
    lexeme_id = stable_id("liftlex", guid or lift_id or lemma)
    return {
        "lexeme_id": lexeme_id,
        "lift_id": lift_id or None,
        "guid": guid or None,
        "lemma_canonical": lemma,
        "lexical_unit": lexical_forms,
        "citation": citation_forms,
        "morph_type": morph_type,
        "variants": variants,
        "senses": senses,
        "grammatical_infos": grammatical_infos,
        "part_of_speech": _high_level_pos(grammatical_infos),
        "traits": entry_traits,
        "fields": _fields(element),
        "relations": relations,
        "date_created": element.attrib.get("dateCreated"),
        "date_modified": element.attrib.get("dateModified"),
        "order": element.attrib.get("order"),
        "status": "lift",
    }


def iter_lift_entries(path: Path | str) -> Iterator[dict[str, Any]]:
    source = Path(path)
    count = 0
    try:
        for _event, element in ET.iterparse(source, events=("end",)):
            if _tag(element) != "entry":
                continue
            yield parse_lift_entry(element)
            count += 1
            element.clear()
    except ET.ParseError as exc:
        raise ValueError(f"invalid LIFT XML: {exc}") from exc
    if count == 0:
        raise ValueError("no <entry> blocks found in LIFT file")


def _sense_glosses(entry: dict[str, Any], lang: str | None = None) -> Iterator[tuple[dict[str, Any], str]]:
    for sense in entry.get("senses") or []:
        for gloss in sense.get("glosses") or []:
            if lang is not None and gloss.get("lang") != lang:
                continue
            text = nfc(gloss.get("text")).strip()
            if text:
                yield sense, text


def preferred_sense_gloss(sense: dict[str, Any], preferred: str = "en") -> str | None:
    glosses = sense.get("glosses") or []
    value = next((nfc(row.get("text")).strip() for row in glosses
                  if row.get("lang") == preferred and nfc(row.get("text")).strip()), "")
    if not value and glosses:
        value = nfc(glosses[0].get("text")).strip()
    return value or None


def entry_forms(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Every form that resolves to this exact LIFT entry, with provenance."""
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for kind, forms in (("citation", entry.get("citation") or []),
                        ("lexical-unit", entry.get("lexical_unit") or [])):
        for form in forms:
            text = nfc(form.get("text")).strip()
            key = (text, form.get("lang") or "")
            if text and key not in seen:
                seen.add(key)
                rows.append({"text": text, "lang": form.get("lang") or "", "kind": kind,
                             "morph_type": entry.get("morph_type")})
    for variant in entry.get("variants") or []:
        for form in variant.get("forms") or []:
            text = nfc(form.get("text")).strip()
            key = (text, form.get("lang") or "")
            if text and key not in seen:
                seen.add(key)
                rows.append({"text": text, "lang": form.get("lang") or "", "kind": "variant",
                             "morph_type": variant.get("morph_type") or entry.get("morph_type")})
    return rows


class LiftLexiconStore:
    def __init__(self, root: Path | str = "."):
        self.root = Path(root).resolve()

    @property
    def entries_path(self) -> Path:
        path = writable(self.root, LIFT_ENTRIES)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def manifest_path(self) -> Path:
        path = writable(self.root, LIFT_MANIFEST)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def archive_source(self, source: Path, original_name: str) -> dict[str, Any]:
        digest = sha256_file(source)
        suffix = "".join(Path(original_name).suffixes) or ".lift"
        target_dir = writable(self.root, LIFT_SOURCES)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{digest}{suffix}"
        if not target.exists():
            shutil.copy2(source, target)
        return {"sha256": digest, "filename": original_name, "archived_as": target.name,
                "bytes": source.stat().st_size}

    def import_file(self, source: Path | str, *, display_name: str | None = None,
                    ranges: Path | str | None = None,
                    ranges_display_name: str | None = None) -> dict[str, Any]:
        source = Path(source).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.suffix.lower() != ".lift":
            raise ValueError("LIFT import expects a .lift file")
        name = display_name or source.name
        archived = self.archive_source(source, name)
        entries = list(iter_lift_entries(source))
        write_jsonl(self.entries_path, entries)
        sense_count = sum(len(entry.get("senses") or []) for entry in entries)
        variant_count = sum(len(entry.get("variants") or []) for entry in entries)
        form_count = sum(len(entry_forms(entry)) for entry in entries)
        grammatical = sorted({value for entry in entries
                              for value in entry.get("grammatical_infos") or []})
        ranges_archive = None
        if ranges:
            ranges_path = Path(ranges).expanduser().resolve()
            if not ranges_path.is_file():
                raise FileNotFoundError(ranges_path)
            ranges_archive = self.archive_source(
                ranges_path, ranges_display_name or ranges_path.name)
        manifest = {
            "version": 1,
            "source_file": name,
            "entries": len(entries),
            "senses": sense_count,
            "variants": variant_count,
            "forms": form_count,
            "grammatical_infos": grammatical,
            "archive": archived,
            "ranges_archive": ranges_archive,
        }
        write_json(self.manifest_path, manifest)
        return manifest

    def manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {"version": 1, "entries": 0, "senses": 0, "variants": 0,
                    "forms": 0, "grammatical_infos": []}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def entries(self) -> list[dict[str, Any]]:
        return list(iter_jsonl(self.entries_path))


class LiftLexiconIndex:
    """Fast exact-form index over the currently imported LIFT file."""

    def __init__(self, root: Path | str = "."):
        self.root = Path(root).resolve()
        self.store = LiftLexiconStore(self.root)
        self.entries: dict[str, dict[str, Any]] = {}
        self.by_form: dict[str, list[str]] = defaultdict(list)
        self.by_lift_id: dict[str, str] = {}
        self.by_guid: dict[str, str] = {}
        for entry in self.store.entries():
            lexeme_id = entry.get("lexeme_id")
            if not lexeme_id:
                continue
            self.entries[lexeme_id] = entry
            if entry.get("lift_id"):
                self.by_lift_id[entry["lift_id"]] = lexeme_id
            if entry.get("guid"):
                self.by_guid[entry["guid"]] = lexeme_id
            for form in entry_forms(entry):
                key = self.form_key(form.get("text"))
                if key and lexeme_id not in self.by_form[key]:
                    self.by_form[key].append(lexeme_id)

    @staticmethod
    def form_key(value: str | None) -> str:
        return nfc(value).strip().casefold()

    def has_entries(self) -> bool:
        return bool(self.entries)

    def entry(self, lexeme_id: str | None) -> dict[str, Any] | None:
        return self.entries.get(lexeme_id or "")

    def entries_for_form(self, form: str | None) -> list[dict[str, Any]]:
        return [self.entries[lid] for lid in self.by_form.get(self.form_key(form), [])
                if lid in self.entries]

    def sense_options(self, form: str | None, *, prefix: str = "",
                      limit: int = 60) -> list[dict[str, Any]]:
        prefix_key = nfc(prefix).strip().casefold()
        rows: list[dict[str, Any]] = []
        for entry in self.entries_for_form(form):
            matched_form = next((f for f in entry_forms(entry)
                                 if self.form_key(f.get("text")) == self.form_key(form)), None)
            senses = entry.get("senses") or []
            if not senses:
                rows.append(self._option(entry, None, matched_form))
                continue
            for sense in senses:
                gloss = preferred_sense_gloss(sense) or ""
                if prefix_key and not (gloss.casefold().startswith(prefix_key)
                                       or nfc(entry.get("lemma_canonical")).casefold().startswith(prefix_key)):
                    continue
                rows.append(self._option(entry, sense, matched_form))
        # Keep separate entries separate even when lemma/gloss strings coincide.
        rows.sort(key=lambda row: (row.get("lemma", "").casefold(),
                                   row.get("grammatical_info") or "",
                                   row.get("gloss") or "",
                                   row.get("lexeme_id") or ""))
        return rows[:max(1, min(int(limit or 60), 200))]

    def _option(self, entry: dict[str, Any], sense: dict[str, Any] | None,
                matched_form: dict[str, Any] | None) -> dict[str, Any]:
        return {
            "lexeme_id": entry.get("lexeme_id"),
            "lift_id": entry.get("lift_id"),
            "guid": entry.get("guid"),
            "lemma": entry.get("lemma_canonical"),
            "surface": (matched_form or {}).get("text"),
            "form_kind": (matched_form or {}).get("kind"),
            "morph_type": (matched_form or {}).get("morph_type") or entry.get("morph_type"),
            "sense_id": (sense or {}).get("sense_id"),
            "gloss": preferred_sense_gloss(sense or {}),
            "glosses": copy.deepcopy((sense or {}).get("glosses") or []),
            "grammatical_info": (sense or {}).get("grammatical_info"),
            "grammatical_infos": list(entry.get("grammatical_infos") or []),
            "part_of_speech": entry.get("part_of_speech"),
            "source": "lift",
        }

    def match(self, form: str | None, *, gloss: str | None = None,
              grammatical_info: str | None = None) -> tuple[dict[str, Any] | None,
                                                            dict[str, Any] | None,
                                                            list[dict[str, Any]]]:
        """Resolve only when evidence identifies one exact LIFT entry.

        The same spelling may belong to unrelated entries.  A matching sense
        gloss and/or grammatical-info can disambiguate; spelling similarity
        alone never merges or guesses between several entries.
        """
        entries = self.entries_for_form(form)
        if not entries:
            return None, None, []
        gloss_key = nfc(gloss).strip().casefold()
        gi_key = nfc(grammatical_info).strip().casefold()
        candidates: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
        for entry in entries:
            senses = entry.get("senses") or [None]
            for sense in senses:
                if sense is None:
                    candidates.append((entry, None))
                    continue
                sense_glosses = {nfc(row.get("text")).strip().casefold()
                                 for row in sense.get("glosses") or []
                                 if nfc(row.get("text")).strip()}
                if gloss_key and gloss_key not in sense_glosses:
                    continue
                if gi_key and nfc(sense.get("grammatical_info")).strip().casefold() != gi_key:
                    continue
                candidates.append((entry, sense))
        if (gloss_key or gi_key) and not candidates:
            return None, None, [self._option(entry, None, None) for entry in entries]
        if not gloss_key and not gi_key:
            candidates = [(entry, None) for entry in entries]
        unique_entries = {entry["lexeme_id"]: entry for entry, _sense in candidates}
        if len(unique_entries) != 1:
            return None, None, [self._option(entry, sense, None)
                                for entry, sense in candidates]
        entry = next(iter(unique_entries.values()))
        matched_senses = [sense for candidate_entry, sense in candidates
                          if candidate_entry.get("lexeme_id") == entry.get("lexeme_id") and sense]
        sense = matched_senses[0] if len(matched_senses) == 1 else None
        return entry, sense, [self._option(entry, s, None) for s in matched_senses] or [self._option(entry, None, None)]


def _host_cell(token: dict[str, Any]) -> dict[str, Any] | None:
    cells = token.get("morphemes") or []
    typed = [cell for cell in cells
             if nfc(cell.get("flex_morph_type")).casefold() in {"stem", "root"}]
    if typed:
        return typed[0]
    try:
        from maalula_glosser.editor.relink import host_stem
        stem, _gloss = host_stem(cells)
    except Exception:
        stem = None
    if stem:
        return next((cell for cell in cells if nfc(cell.get("form")) == nfc(stem)), None)
    return None


def bind_token_to_lift(token: dict[str, Any], index: LiftLexiconIndex) -> tuple[bool, bool]:
    """Attach one token to a LIFT entry only when exact evidence is unique.

    Returns ``(changed, ambiguous)``.  Ambiguous exact-form matches are kept as
    candidates for the author's picker; no root/string-similarity heuristic is
    used.
    """
    cell = _host_cell(token)
    if not cell:
        return False, False
    form = nfc(cell.get("form")).strip("-=+")
    if not form:
        return False, False
    # The lexical head is what FLEX stores as the stem gloss in this project.
    try:
        from maalula_glosser.gloss_shape import lexical_head
        gloss = lexical_head(cell.get("gloss") or token.get("lexical_gloss") or "")
    except Exception:
        gloss = nfc(token.get("lexical_gloss")).strip()
    entry, sense, options = index.match(form, gloss=gloss or None)
    before = (token.get("lemma_id"), token.get("lemma"), token.get("grammatical_info"),
              token.get("lift_sense_id"), tuple(token.get("lexeme_candidates") or ()))
    if entry:
        lexeme_id = entry["lexeme_id"]
        grammatical = ((sense or {}).get("grammatical_info")
                       or ((entry.get("grammatical_infos") or [None])[0]
                           if len(entry.get("grammatical_infos") or []) == 1 else None))
        token["lemma_id"] = lexeme_id
        token["lemma"] = entry.get("lemma_canonical")
        token["grammatical_info"] = grammatical
        token["grammatical_infos"] = list(entry.get("grammatical_infos") or [])
        token["lift_entry_id"] = entry.get("lift_id")
        token["lift_guid"] = entry.get("guid")
        token["lift_sense_id"] = (sense or {}).get("sense_id")
        if sense and preferred_sense_gloss(sense):
            token["lexical_gloss"] = preferred_sense_gloss(sense)
        token.pop("lexeme_candidates", None)
        cell["lexeme_id"] = lexeme_id
        cell["lemma"] = entry.get("lemma_canonical")
        cell["grammatical_info"] = grammatical
        cell["lift_entry_id"] = entry.get("lift_id")
        cell["lift_sense_id"] = (sense or {}).get("sense_id")
        ambiguous = False
    else:
        ids = list(dict.fromkeys(option.get("lexeme_id") for option in options
                                 if option.get("lexeme_id")))
        if ids:
            token["lexeme_candidates"] = ids
        ambiguous = len(ids) > 1
    after = (token.get("lemma_id"), token.get("lemma"), token.get("grammatical_info"),
             token.get("lift_sense_id"), tuple(token.get("lexeme_candidates") or ()))
    return before != after, ambiguous


def backfill_editor_sessions(root: Path | str, *, document_ids: Iterable[str] | None = None,
                             session_store=None) -> dict[str, int]:
    """Link saved corpus tokens to exact LIFT entries after either import.

    This only fills lexical identity/category metadata.  It never changes the
    written surface, segmentation or gloss lines.
    """
    from maalula_glosser.editor.store import SessionStore

    root = Path(root).resolve()
    index = LiftLexiconIndex(root)
    if not index.has_entries():
        return {"documents": 0, "tokens": 0, "linked": 0, "ambiguous": 0}
    store = session_store or SessionStore(root)
    wanted = set(document_ids or [])
    documents = 0
    tokens = linked = ambiguous = 0
    for listed in store.list():
        document_id = listed.get("document_id")
        if wanted and document_id not in wanted:
            continue
        document = store.load(document_id)
        changed = False
        documents += 1
        for sentence in document.get("sentences") or []:
            for token in sentence.get("tokens") or []:
                tokens += 1
                did_change, was_ambiguous = bind_token_to_lift(token, index)
                if did_change:
                    linked += 1
                    changed = True
                if was_ambiguous:
                    ambiguous += 1
        if changed:
            store.save(document)
    return {"documents": documents, "tokens": tokens, "linked": linked,
            "ambiguous": ambiguous}
