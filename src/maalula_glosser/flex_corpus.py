"""Corpus-driven glossing profile built from imported FLEx texts.

This profile is intentionally language-agnostic.  Human FLEx analyses are
exact-form evidence; reusable editor corrections outrank them.  Unknown words
stay unknown instead of falling through to Maalula-specific grammar.

The rest of the original editor is reused unchanged: session history, comments,
discussions, concordances, label inventory, exports, locks, audio/timecodes,
and the reusable morphology/lexical journals all keep their existing formats.
"""
from __future__ import annotations

import copy
import json
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from maalula_glosser.io_utils import (iter_jsonl, sha256_file, stable_id,
                                      write_json, write_jsonl)
from maalula_glosser.lift_lexicon import (LiftLexiconIndex, entry_forms,
                                          preferred_sense_gloss)
from maalula_glosser.reusable import active_readings, surface_key
from maalula_glosser.state import writable


FLEX_DIR = "data/flex_corpus"
FLEX_EVIDENCE = FLEX_DIR + "/evidence"
FLEX_ANALYSES = FLEX_DIR + "/analyses.jsonl"
FLEX_MANIFEST = FLEX_DIR + "/manifest.json"
FLEX_SOURCES = FLEX_DIR + "/sources"
REUSABLE = "data/corrections_v2/reusable_morphology.jsonl"
LEXICAL_REUSABLE = "data/corrections_v2/reusable_lexical_gloss.jsonl"


def nfc(value: str | None) -> str:
    return unicodedata.normalize("NFC", value or "")


def _analysis_signature(row: dict[str, Any]) -> str:
    morphs = [{
        "type": morph.get("type"),
        "form": morph.get("form"),
        "gloss": morph.get("gloss"),
        "lemma": morph.get("lemma"),
        "pos": morph.get("pos"),
        "sep": morph.get("sep"),
    } for morph in row.get("morphemes") or []]
    return json.dumps({
        "segmentation": row.get("segmentation") or "",
        "gloss": row.get("gloss") or "",
        "lemma": row.get("lemma") or "",
        "lemma_id": row.get("lemma_id"),
        "lexical_gloss": row.get("lexical_gloss") or "",
        "pos": row.get("pos") or "",
        "morphemes": morphs,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _lexical_head(gloss: str) -> str:
    """Conservative display meaning for a corpus row.

    Prefer a lexical-looking component; leave grammatical all-caps labels out.
    FLEx already stores a stem gloss separately in our evidence, so this is
    mainly a compatibility fallback for editor corrections.
    """
    for morph in re.split(r"[-=+]", nfc(gloss)):
        parts = [part for part in morph.split(".") if part]
        for part in parts:
            if any(ch.islower() for ch in part):
                return part
    return ""


def _analysis_pairs(segmentation: str | None, gloss: str | None) -> list[tuple[str, str]]:
    """Pair morpheme forms with their glosses when the boundary topology matches.

    Reusable corrections are stored as whole-word strings (``gab-o`` /
    ``chez-DEF``).  The FLEX editor also needs those human-created readings in
    its per-morpheme dictionary, so recover the cells conservatively.  A
    malformed or structurally mismatched correction contributes nothing rather
    than pairing the wrong morphemes.
    """
    segmentation = nfc(segmentation).strip()
    gloss = nfc(gloss).strip()
    if not segmentation or not gloss:
        return []
    seg_boundaries = ["=" if mark in {"=", "+"} else mark
                      for mark in re.findall(r"[-=+]", segmentation)]
    gloss_boundaries = ["=" if mark in {"=", "+"} else mark
                        for mark in re.findall(r"[-=+]", gloss)]
    if seg_boundaries != gloss_boundaries:
        return []
    forms = re.split(r"[-=+]", segmentation)
    glosses = re.split(r"[-=+]", gloss)
    if len(forms) != len(glosses):
        return []
    return [(form.strip(), meaning.strip())
            for form, meaning in zip(forms, glosses)
            if form.strip() and meaning.strip()]


class FlexCorpusStore:
    def __init__(self, root: Path | str = "."):
        self.root = Path(root).resolve()

    @property
    def evidence_dir(self) -> Path:
        path = writable(self.root, FLEX_EVIDENCE)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def analyses_path(self) -> Path:
        path = writable(self.root, FLEX_ANALYSES)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def manifest_path(self) -> Path:
        path = writable(self.root, FLEX_MANIFEST)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def archive_source(self, source: Path, original_name: str) -> dict[str, Any]:
        digest = sha256_file(source)
        suffix = Path(original_name).suffix.lower() or ".flextext"
        target_dir = writable(self.root, FLEX_SOURCES)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{digest}{suffix}"
        if not target.exists():
            shutil.copy2(source, target)
        return {"sha256": digest, "filename": original_name, "archived_as": target.name,
                "bytes": source.stat().st_size}

    def save_text(self, text: dict[str, Any], evidence: Iterable[dict[str, Any]]) -> dict[str, int]:
        rows = list(evidence)
        path = self.evidence_dir / f"{text['id']}.jsonl"
        write_jsonl(path, rows)
        meta = {
            "text_id": text.get("id"), "title": text.get("title"),
            "source_filename": text.get("source_filename"),
            "source_guid": text.get("source_guid"),
            "segments": len(text.get("segments") or []),
            "tokens": sum(1 for seg in text.get("segments") or []
                          for word in seg.get("words") or [] if not word.get("punctuation")),
            "analysed_tokens": len(rows),
        }
        write_json(self.evidence_dir / f"{text['id']}.meta.json", meta)
        return {"tokens": meta["tokens"], "analysed_tokens": meta["analysed_tokens"]}

    def rebuild(self) -> dict[str, int]:
        buckets: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        total_evidence = 0
        text_count = 0
        for path in sorted(self.evidence_dir.glob("*.jsonl")):
            text_count += 1
            for row in iter_jsonl(path):
                surface = nfc(row.get("surface")).strip()
                if not surface:
                    continue
                total_evidence += 1
                key = surface_key(surface)
                signature = _analysis_signature(row)
                found = buckets[key].get(signature)
                if found is None:
                    found = copy.deepcopy(row)
                    found["surface_variants"] = Counter()
                    found["count"] = 0
                    found["examples"] = []
                    buckets[key][signature] = found
                found["count"] += 1
                found["surface_variants"][surface] += 1
                source = row.get("source") or {}
                if len(found["examples"]) < 12:
                    found["examples"].append(source)

        aggregate: list[dict[str, Any]] = []
        for key, options in buckets.items():
            for signature, row in options.items():
                variants: Counter = row.pop("surface_variants")
                display_surface = variants.most_common(1)[0][0] if variants else key
                row["surface"] = display_surface
                row["surface_key"] = key
                row["analysis_id"] = stable_id("flexa", key, signature)
                row["source"] = {"kind": "flextext-corpus"}
                row["surface_variants"] = dict(variants)
                aggregate.append(row)
        aggregate.sort(key=lambda row: (row["surface_key"], -int(row.get("count") or 0),
                                        row["analysis_id"]))
        write_jsonl(self.analyses_path, aggregate)
        manifest = {
            "version": 1,
            "texts": text_count,
            "evidence_tokens": total_evidence,
            "forms": len(buckets),
            "analyses": len(aggregate),
        }
        write_json(self.manifest_path, manifest)
        return manifest

    def manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {"version": 1, "texts": 0, "evidence_tokens": 0, "forms": 0,
                    "analyses": 0}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))


class FlexLexicalGloss:
    def __init__(self, root: Path, builtin: dict[str, str]):
        self.root = root
        self.builtin = dict(builtin)
        self.author: dict[str, dict[str, Any]] = {}
        self.author_excluded: dict[str, set[str]] = {}
        for row in active_readings(writable(root, LEXICAL_REUSABLE)):
            lexeme_id = row.get("lexeme_id")
            if lexeme_id:
                self.author[lexeme_id] = row
                self.author_excluded[lexeme_id] = set(row.get("excluded_surfaces") or ())

    def resolve(self, lexeme_id: str | None,
                surface: str | None = None) -> tuple[str | None, str]:
        if not lexeme_id:
            return None, "unknown_lexeme"
        row = self.author.get(lexeme_id)
        if row and (surface is None or nfc(surface) not in self.author_excluded.get(lexeme_id, ())):
            return row.get("gloss"), "author_confirmed"
        if lexeme_id in self.builtin and self.builtin[lexeme_id]:
            return self.builtin[lexeme_id], "flextext"
        return None, "no_sense"

    def remember(self, row: dict[str, Any]) -> None:
        lexeme_id = row.get("lexeme_id")
        if lexeme_id:
            self.author[lexeme_id] = row
            self.author_excluded[lexeme_id] = set(row.get("excluded_surfaces") or ())


class FlexCorpusAnalyzer:
    """Compatibility façade exposing the editor-facing analyzer attributes."""

    def __init__(self, root: Path | str = "."):
        self.root = Path(root).resolve()
        self.profile = "flex"
        self.a1: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.lexemes: dict[str, dict[str, Any]] = {}
        self.forms_by_surface: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.forms_by_lexeme: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.lexeme_ids_by_root: dict[str, set[str]] = defaultdict(set)
        self.principal_parts_index: dict[str, list[str]] = defaultdict(list)
        self.stems_by_lexeme: dict[str, list[str]] = defaultdict(list)
        self.lexicon_v3: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.service_verbs: dict[str, dict[str, Any]] = {}
        self.verbal_lexeme_ids: set[str] = set()
        self.proclitics: list[dict[str, Any]] = []
        self.prefixes: list[tuple[str, str, dict[str, Any]]] = []
        self.reusable_morphology: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.gloss_labels: set[str] = set()
        bound_label_counts: Counter[str] = Counter()
        stem_label_counts: Counter[str] = Counter()
        stem_label_heads: dict[str, set[str]] = defaultdict(set)
        lexical: dict[str, str] = {}
        # Per-morpheme dictionary used by the gloss-cell picker.  This is
        # deliberately separate from whole-word analyses: ``gab`` may occur
        # inside many different words but still have corpus-attested glosses
        # such as ``lat`` and ``chez``.  Counts are token-weighted from FLEx.
        self.morpheme_gloss_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.morpheme_gloss_types: dict[str, dict[str, Counter[str]]] = defaultdict(
            lambda: defaultdict(Counter))
        self.author_morpheme_glosses: dict[str, set[str]] = defaultdict(set)

        # FieldWorks LIFT is the authoritative lexical-entry layer when it is
        # present.  Crucially, each LIFT entry remains a distinct lexeme even
        # if another entry has the same spelling.  Variants resolve to the
        # owning entry, and exact grammatical-info (Preterite, Infectum, ...)
        # is preserved alongside the coarse compatibility POS.
        self.lift = LiftLexiconIndex(self.root)
        for lexeme_id, entry in self.lift.entries.items():
            self.lexemes[lexeme_id] = {
                "lexeme_id": lexeme_id,
                "lemma_canonical": entry.get("lemma_canonical"),
                "part_of_speech": entry.get("part_of_speech"),
                "root_canonical": None,
                "status": "lift",
                "lift_id": entry.get("lift_id"),
                "lift_guid": entry.get("guid"),
                "morph_type": entry.get("morph_type"),
                "grammatical_infos": list(entry.get("grammatical_infos") or []),
                "senses": copy.deepcopy(entry.get("senses") or []),
                "variants": copy.deepcopy(entry.get("variants") or []),
            }
            if entry.get("part_of_speech") == "verb":
                self.verbal_lexeme_ids.add(lexeme_id)
            for form_row in entry_forms(entry):
                form_surface = nfc(form_row.get("text")).strip()
                if not form_surface:
                    continue
                form = {
                    "form_canonical": form_surface,
                    "frequency": 0,
                    "lexeme_id": lexeme_id,
                    "selected_analysis_id": None,
                    "attested": False,
                    "generated": False,
                    "source": "lift",
                    "form_kind": form_row.get("kind"),
                    "morph_type": form_row.get("morph_type"),
                    "grammatical_infos": list(entry.get("grammatical_infos") or []),
                }
                self.forms_by_surface[form_surface].append(form)
                self.forms_by_lexeme[lexeme_id].append(form)
            # The compatibility resolver is intentionally conservative for a
            # polysemous LIFT entry: only a single-sense entry gets a default
            # lexical meaning.  Sense-specific choices stay separate in the
            # picker and never get collapsed here.
            senses = entry.get("senses") or []
            if len(senses) == 1:
                one_gloss = preferred_sense_gloss(senses[0])
                if one_gloss:
                    lexical.setdefault(lexeme_id, one_gloss)

        for row in iter_jsonl(writable(self.root, FLEX_ANALYSES)):
            surface = nfc(row.get("surface")).strip()
            if not surface:
                continue
            key = row.get("surface_key") or surface_key(surface)
            normalized = copy.deepcopy(row)
            normalized.setdefault("gloss_clean", normalized.get("gloss") or "")
            normalized.setdefault("surface_canonical", surface)
            lexical_morph = next((m for m in normalized.get("morphemes") or []
                                  if (m.get("type") or "").casefold() in {"stem", "root"}
                                  and (m.get("form") or m.get("lemma"))), None)
            lexical_morph = lexical_morph or next(
                (m for m in normalized.get("morphemes") or []
                 if m.get("form") or m.get("lemma")), None)
            stem_form = nfc((lexical_morph or {}).get("form")
                            or (lexical_morph or {}).get("lemma")).strip()
            stem_gloss = nfc((lexical_morph or {}).get("gloss")
                             or normalized.get("lexical_gloss")).strip()
            lift_entry = lift_sense = None
            if self.lift.has_entries() and stem_form:
                lift_entry, lift_sense, lift_candidates = self.lift.match(
                    stem_form, gloss=stem_gloss or None)
                if lift_entry:
                    normalized["source_lemma_id"] = normalized.get("lemma_id")
                    normalized["source_lemma"] = normalized.get("lemma")
                    normalized["lemma_id"] = lift_entry.get("lexeme_id")
                    normalized["lemma"] = lift_entry.get("lemma_canonical")
                    normalized["lift_entry_id"] = lift_entry.get("lift_id")
                    normalized["lift_guid"] = lift_entry.get("guid")
                    normalized["lift_sense_id"] = (lift_sense or {}).get("sense_id")
                    entry_gis = list(lift_entry.get("grammatical_infos") or [])
                    normalized["grammatical_info"] = (
                        (lift_sense or {}).get("grammatical_info")
                        or (entry_gis[0] if len(entry_gis) == 1 else None))
                    if lift_sense and preferred_sense_gloss(lift_sense):
                        normalized["lexical_gloss"] = preferred_sense_gloss(lift_sense)
                elif lift_candidates:
                    normalized["lift_candidates"] = lift_candidates
            # Build a FLEX-native autocomplete vocabulary without importing
            # Maalula's curated inventory.  We intentionally keep this
            # conservative: corpus gloss strings contain both grammatical
            # labels and ordinary English lexical glosses.  Bound-morph labels
            # are admitted when recurrent; stem-internal labels must recur
            # across many lexical heads (or be unmistakable code-like labels).
            row_count = int(normalized.get("count") or 1)
            for morph in normalized.get("morphemes") or []:
                morph_form = nfc(morph.get("form")).strip("-=+")
                gloss = nfc(morph.get("gloss")).strip("-=+")
                kind = nfc(morph.get("type")).casefold()
                if morph_form and gloss:
                    morph_key = surface_key(morph_form)
                    self.morpheme_gloss_counts[morph_key][gloss] += row_count
                    self.morpheme_gloss_types[morph_key][gloss][kind or "unknown"] += row_count
                if not gloss:
                    continue
                parts = [p for p in gloss.split(".") if p]
                if kind in {"stem", "root", "phrase", ""}:
                    if len(parts) > 1:
                        head = parts[0]
                        for label in parts[1:]:
                            stem_label_counts[label] += row_count
                            stem_label_heads[label].add(head)
                    for label in parts:
                        if label.isupper() or re.fullmatch(r"[123]?[mMfF]?[sSpP]", label):
                            self.gloss_labels.add(label)
                else:
                    for label in parts:
                        bound_label_counts[label] += row_count
                        if label.isupper() or re.fullmatch(r"[123]?[mMfF]?[sSpP]", label):
                            self.gloss_labels.add(label)
            self.by_key[key].append(normalized)
            self.a1[surface].append(normalized)
            lemma_id = normalized.get("lemma_id")
            lemma = nfc(normalized.get("lemma"))
            pos = nfc(normalized.get("pos"))
            if lemma_id and lemma:
                lexeme = self.lexemes.setdefault(lemma_id, {
                    "lexeme_id": lemma_id, "lemma_canonical": lemma,
                    "part_of_speech": pos or None, "root_canonical": None,
                    "status": "flextext",
                })
                if not lexeme.get("part_of_speech") and pos:
                    lexeme["part_of_speech"] = pos
                if normalized.get("lexical_gloss"):
                    lexical.setdefault(lemma_id, normalized["lexical_gloss"])
                count = int(normalized.get("count") or 1)
                form = {
                    "form_canonical": surface, "frequency": count,
                    "lexeme_id": lemma_id,
                    "selected_analysis_id": normalized.get("analysis_id"),
                    "attested": True, "generated": False,
                    "source": "flextext",
                    "grammatical_info": normalized.get("grammatical_info"),
                }
                # The same surface can already be a citation/variant row from
                # LIFT.  Keep both provenance rows; downstream views can then
                # show dictionary form and corpus frequency independently.
                self.forms_by_surface[surface].append(form)
                self.forms_by_lexeme[lemma_id].append(form)
                stem = next((m.get("form") for m in normalized.get("morphemes") or []
                             if (m.get("type") or "").casefold() in {"stem", "root"}
                             and m.get("form")), None)
                if stem and stem not in self.stems_by_lexeme[lemma_id]:
                    self.stems_by_lexeme[lemma_id].append(stem)
                low_pos = pos.casefold()
                if low_pos in {"v", "verb", "vb"} or "verb" in low_pos:
                    self.verbal_lexeme_ids.add(lemma_id)

        for label, count in bound_label_counts.items():
            if count >= 10:
                self.gloss_labels.add(label)
        for label, count in stem_label_counts.items():
            if (len(stem_label_heads[label]) >= 20 or
                    (count >= 10 and bound_label_counts.get(label, 0) >= 10)):
                self.gloss_labels.add(label)

        # Author-added lexemes stay visible in the same editor if the generic
        # corpus profile is used; they are never folded into FLEx source data.
        try:
            from maalula_glosser.lexicon_additions import active_additions
            for addition in active_additions(self.root):
                lexeme_id = addition.get("lexeme_id")
                lemma = nfc(addition.get("lemma_canonical"))
                if not lexeme_id or not lemma:
                    continue
                grammatical_info = nfc(addition.get("grammatical_info")).strip() or None
                self.lexemes.setdefault(lexeme_id, {
                    "lexeme_id": lexeme_id, "lemma_canonical": lemma,
                    "part_of_speech": addition.get("part_of_speech"),
                    "root_canonical": addition.get("root_canonical"),
                    "grammatical_infos": [grammatical_info] if grammatical_info else [],
                    "status": "author_addition",
                })
                root_text = nfc(addition.get("root_canonical"))
                if root_text:
                    self.lexeme_ids_by_root[root_text].add(lexeme_id)
                low_gi = (grammatical_info or "").casefold()
                if (addition.get("part_of_speech") == "verb"
                        or any(marker in low_gi for marker in
                               ("preterite", "infectum", "imperative", "verb"))):
                    self.verbal_lexeme_ids.add(lexeme_id)
                for form_row in addition.get("surfaces") or []:
                    form_surface = nfc(form_row.get("form_canonical"))
                    if not form_surface:
                        continue
                    form = {"form_canonical": form_surface,
                            "frequency": int(form_row.get("frequency") or 0),
                            "lexeme_id": lexeme_id, "selected_analysis_id": None,
                            "attested": True, "generated": False,
                            "source": "author",
                            "grammatical_info": grammatical_info}
                    self.forms_by_surface[form_surface].append(form)
                    self.forms_by_lexeme[lexeme_id].append(form)
        except Exception:
            pass

        for row in active_readings(writable(self.root, REUSABLE)):
            self.remember_morphology(row)
        # A lexical sense learned while glossing a new text is not necessarily
        # a reusable WHOLE-WORD morphology correction.  Preserve its stem-form
        # association separately so it still appears in the morpheme picker
        # after the editor is restarted.
        for row in active_readings(writable(self.root, LEXICAL_REUSABLE)):
            stem_form = nfc(row.get("stem_form")).strip("-=+")
            gloss = nfc(row.get("gloss")).strip()
            if stem_form and gloss:
                self.author_morpheme_glosses[surface_key(stem_form)].add(gloss)
        self.lexical = FlexLexicalGloss(self.root, lexical)

    def canonical(self, value: str) -> str:
        return nfc(value)

    def lexeme_record(self, lexeme_id: str | None) -> dict[str, Any]:
        if not lexeme_id:
            return {}
        return self.lexemes.get(lexeme_id) or self.service_verbs.get(lexeme_id) or {}

    def lift_options_for_form(self, form: str, prefix: str = "",
                              limit: int = 60) -> list[dict[str, Any]]:
        return self.lift.sense_options(form, prefix=prefix, limit=limit)

    def grammatical_info(self, lexeme_id: str | None,
                         lexical_gloss: str | None = None) -> str | None:
        entry = self.lift.entry(lexeme_id)
        if not entry:
            record = self.lexeme_record(lexeme_id)
            values = record.get("grammatical_infos") or []
            return values[0] if len(values) == 1 else None
        gloss_key = nfc(lexical_gloss).strip().casefold()
        if gloss_key:
            matched = []
            for sense in entry.get("senses") or []:
                glosses = {nfc(row.get("text")).strip().casefold()
                           for row in sense.get("glosses") or []}
                if gloss_key in glosses and sense.get("grammatical_info"):
                    matched.append(sense["grammatical_info"])
            matched = list(dict.fromkeys(matched))
            if len(matched) == 1:
                return matched[0]
        values = list(entry.get("grammatical_infos") or [])
        return values[0] if len(values) == 1 else None

    def is_verbal(self, lexeme_id: str | None) -> bool:
        return bool(lexeme_id and (lexeme_id in self.verbal_lexeme_ids
                                   or lexeme_id in self.service_verbs))

    def service_verb(self, root: str) -> str:
        root = nfc(root)
        lexeme_id = f"lexsvc_{root}"
        self.service_verbs.setdefault(lexeme_id, {
            "lexeme_id": lexeme_id, "lemma_canonical": f"√{root}",
            "root_canonical": root, "part_of_speech": "verb", "service": True,
        })
        return lexeme_id

    def all_stage_candidates(self, _word: str) -> dict[str, Any]:
        return {"produced": []}

    def remember_morphology(self, row: dict[str, Any]) -> None:
        surface = nfc(row.get("surface_canonical") or row.get("surface_original") or "")
        if not surface:
            return
        key = surface_key(surface)
        current = self.reusable_morphology[key]
        addition_id = row.get("addition_id")
        current[:] = [old for old in current if old.get("addition_id") != addition_id]
        current.append(copy.deepcopy(row))
        # A newly confirmed whole-word reading also teaches the cell picker.
        # Keep it as an author option rather than inflating FLEx corpus counts.
        for form, meaning in _analysis_pairs(row.get("segmentation"), row.get("gloss")):
            self.author_morpheme_glosses[surface_key(form)].add(meaning)

    def remember_morpheme_gloss(self, form: str, gloss: str) -> None:
        form = nfc(form).strip("-=+")
        gloss = nfc(gloss).strip()
        if form and gloss:
            self.author_morpheme_glosses[surface_key(form)].add(gloss)

    def remember_lexeme_surface(self, lexeme_id: str, surface: str,
                                grammatical_info: str | None = None) -> None:
        """Teach this running analyzer one author-confirmed form → entry link.

        Persistence is handled by ``lexicon_additions.extend_addition`` at the
        editor boundary.  This method only updates the in-memory indexes so a
        second new text opened in the same process already recognises the form
        without an expensive full analyzer rebuild.
        """
        lexeme_id = (lexeme_id or "").strip()
        surface = nfc(surface).strip()
        if not lexeme_id or not surface or lexeme_id not in self.lexemes:
            return
        current = self.forms_by_surface[surface]
        if any(row.get("lexeme_id") == lexeme_id for row in current):
            return
        row = {
            "form_canonical": surface,
            "frequency": 1,
            "lexeme_id": lexeme_id,
            "selected_analysis_id": None,
            "attested": True,
            "generated": False,
            "source": "author",
            "grammatical_info": grammatical_info,
        }
        current.append(row)
        self.forms_by_lexeme[lexeme_id].append(row)

    def lexeme_ids_for_surface(self, surface: str) -> list[str]:
        """Distinct lexical entries explicitly known for an exact surface.

        Unlike the whole-word FLEX analysis index this also includes LIFT
        citation/variant forms and author-confirmed new surface→entry links.
        It never guesses by root or spelling similarity.
        """
        surface = nfc(surface).strip()
        if not surface:
            return []
        return list(dict.fromkeys(
            row.get("lexeme_id") for row in self.forms_by_surface.get(surface, [])
            if row.get("lexeme_id") and row.get("lexeme_id") in self.lexemes
        ))

    def morpheme_gloss_options(self, form: str, prefix: str = "",
                               limit: int = 24) -> list[dict[str, Any]]:
        """Known glosses for one morpheme form, author choices first.

        This is the FLEX-profile equivalent of a tiny morpheme dictionary.
        It is intentionally exact-form: it offers what the imported corpus or
        the author has actually used for this morpheme, while the input remains
        free text so a genuinely new analysis can always be entered.
        """
        key = surface_key(nfc(form).strip("-=+"))
        if not key:
            return []
        folded_prefix = nfc(prefix).strip().casefold()
        rows: list[dict[str, Any]] = []
        author = self.author_morpheme_glosses.get(key, set())
        corpus = self.morpheme_gloss_counts.get(key, Counter())
        lift_options = self.lift.sense_options(form, prefix=prefix, limit=100)
        lift_by_gloss: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for option in lift_options:
            lift_gloss = nfc(option.get("gloss")).strip()
            if lift_gloss:
                lift_by_gloss[lift_gloss].append(option)
        all_glosses = set(corpus) | set(author) | set(lift_by_gloss)
        for gloss in all_glosses:
            if folded_prefix and not gloss.casefold().startswith(folded_prefix):
                continue
            count = int(corpus.get(gloss, 0))
            types = self.morpheme_gloss_types.get(key, {}).get(gloss, Counter())
            rows.append({
                "gloss": gloss,
                "count": count,
                "author": gloss in author,
                "types": [kind for kind, _n in types.most_common() if kind],
                "source": ("author" if gloss in author else
                           "lift" if gloss in lift_by_gloss else "flextext"),
                "lift_options": lift_by_gloss.get(gloss, []),
            })
        source_rank = {"author": 0, "lift": 1, "flextext": 2}
        rows.sort(key=lambda row: (source_rank.get(row.get("source"), 3),
                                   -row["count"], row["gloss"].casefold()))
        return rows[:max(1, min(int(limit or 24), 100))]

    def has_confirmed_reading(self, written: str, allow_joined: bool = True) -> str | None:
        written = nfc(written)
        variants = [written]
        if allow_joined:
            variants.append(re.sub(r"[\s‿]+", "", written))
        for spelling in dict.fromkeys(variants):
            if self.reusable_morphology.get(surface_key(spelling)):
                return spelling
        return None

    def lookup(self, surface: str) -> list[dict[str, Any]]:
        key = surface_key(surface)
        corrected = self.reusable_morphology.get(key) or []
        if corrected:
            rows = []
            for row in reversed(corrected):
                rows.append({
                    "surface": surface,
                    "segmentation": row.get("segmentation"),
                    "gloss": row.get("gloss"),
                    "gloss_clean": row.get("gloss"),
                    "lemma_id": row.get("lemma_id"),
                    "lemma": (self.lexeme_record(row.get("lemma_id")).get("lemma_canonical")
                              if row.get("lemma_id") else None),
                    "lexical_gloss": (_lexical_head(row.get("gloss") or "") or None),
                    "count": 10**9, "tier": "A1", "source": {"kind": "author"},
                    "evidence": [{"kind": "AUTHOR_REUSABLE",
                                  "addition_id": row.get("addition_id")}],
                })
            return rows
        rows = list(self.by_key.get(key) or [])
        rows.sort(key=lambda row: (-int(row.get("count") or 0), row.get("analysis_id") or ""))
        return rows


class FlexCorpusTextGlosser:
    """A ``TextGlosser``-compatible exact corpus parser for Turoyo/FLEx."""

    def __init__(self, root: Path | str = "."):
        self.root = Path(root).resolve()
        self.analyzer = FlexCorpusAnalyzer(self.root)
        self._token_cache: dict[tuple[str, bool], dict[str, Any]] = {}
        self.proclitic_forms: frozenset[str] = frozenset()
        self.verbal_prefixes: dict[str, str] = {}
        self.profile = "flex"

    def reload_corpus(self) -> None:
        self.analyzer = FlexCorpusAnalyzer(self.root)
        self._token_cache.clear()

    def remember(self, row: dict[str, Any]) -> None:
        self.analyzer.remember_morphology(row)
        self._token_cache.clear()

    def remember_lexical(self, row: dict[str, Any]) -> None:
        self.analyzer.lexical.remember(row)
        stem_form = row.get("stem_form")
        if stem_form and row.get("gloss"):
            self.analyzer.remember_morpheme_gloss(stem_form, row["gloss"])
        self._token_cache.clear()

    def remember_morpheme_gloss(self, form: str, gloss: str) -> None:
        self.analyzer.remember_morpheme_gloss(form, gloss)
        self._token_cache.clear()

    def gloss_token(self, piece: str, sentence_initial: bool = False,
                    context: dict | None = None, compose: bool = True) -> dict[str, Any]:
        del context, compose
        piece = nfc(piece)
        cache_key = (piece, sentence_initial)
        if cache_key in self._token_cache:
            return copy.deepcopy(self._token_cache[cache_key])
        options = self.analyzer.lookup(piece)
        if not options and sentence_initial and piece[:1].isupper():
            lower = piece[:1].lower() + piece[1:]
            options = self.analyzer.lookup(lower)
        if options:
            best = options[0]
            alternatives = options[1:9]
            ambiguous = len(options) > 1
            lemma_id = best.get("lemma_id")
            lemma = best.get("lemma") or self.analyzer.lexeme_record(lemma_id).get("lemma_canonical")
            lexical, lexical_status = self.analyzer.lexical.resolve(lemma_id, piece)
            lexical = lexical or best.get("lexical_gloss")
            grammatical_info = (best.get("grammatical_info")
                                or self.analyzer.grammatical_info(lemma_id, lexical))
            lexeme_record = self.analyzer.lexeme_record(lemma_id)
            tier = best.get("tier") or "FLEX"
            record = {
                "token": piece, "normalized": piece,
                "segmentation": best.get("segmentation") or piece,
                "gloss": best.get("gloss_clean") or best.get("gloss"),
                "lemma": lemma, "lemma_id": lemma_id,
                "lexical_gloss": lexical,
                "grammatical_info": grammatical_info,
                "grammatical_infos": list(lexeme_record.get("grammatical_infos") or []),
                "lift_entry_id": best.get("lift_entry_id") or lexeme_record.get("lift_id"),
                "lift_guid": best.get("lift_guid") or lexeme_record.get("lift_guid"),
                "lift_sense_id": best.get("lift_sense_id"),
                "lexical_gloss_status": lexical_status,
                "confidence": "MEDIUM" if ambiguous else "HIGH",
                "tier": tier,
                "alternatives": [{
                    "segmentation": row.get("segmentation"),
                    "gloss": row.get("gloss_clean") or row.get("gloss"),
                    "tier": row.get("tier") or "FLEX",
                    "lemma": row.get("lemma") or self.analyzer.lexeme_record(
                        row.get("lemma_id")).get("lemma_canonical"),
                    "lemma_id": row.get("lemma_id"),
                    "lexical_gloss": row.get("lexical_gloss"),
                    "grammatical_info": (row.get("grammatical_info")
                                           or self.analyzer.grammatical_info(
                                               row.get("lemma_id"), row.get("lexical_gloss"))),
                    "lift_entry_id": row.get("lift_entry_id"),
                    "lift_sense_id": row.get("lift_sense_id"),
                    "count": row.get("count"),
                } for row in alternatives],
                "evidence": best.get("evidence") or [{
                    "kind": "FLEXTEXT_EXACT", "count": int(best.get("count") or 1),
                    "examples": best.get("examples") or [],
                }],
                "trace": ["exact form in imported FLEx corpus"],
                "uncertainty": ([f"{len(options)} corpus analyses for this form"]
                                if ambiguous else []),
                "needs_review": ambiguous,
                "analysis_surface": piece,
            }
        else:
            lift_options = self.analyzer.lift_options_for_form(piece, limit=60)
            # Besides forms written explicitly in LIFT, the author may have
            # linked a previously unknown corpus surface to an existing entry.
            # Those links are persisted in lexicon_additions and indexed in
            # ``forms_by_surface``; treat them as lexical evidence on the next
            # occurrence without inventing a morphological analysis.
            entry_ids = list(dict.fromkeys(
                [row.get("lexeme_id") for row in lift_options if row.get("lexeme_id")]
                + self.analyzer.lexeme_ids_for_surface(piece)))
            unique_entry = (self.analyzer.lexeme_record(entry_ids[0])
                            if len(entry_ids) == 1 else {})
            unique_senses = [row for row in lift_options
                             if row.get("lexeme_id") in entry_ids]
            if unique_entry and not unique_senses and unique_entry.get("status") == "lift":
                lift_entry = self.analyzer.lift.entry(entry_ids[0]) or {}
                unique_senses = [{
                    "lexeme_id": entry_ids[0],
                    "sense_id": sense.get("sense_id"),
                    "gloss": preferred_sense_gloss(sense),
                    "grammatical_info": sense.get("grammatical_info"),
                    "lift_id": lift_entry.get("lift_id"),
                } for sense in lift_entry.get("senses") or []]
            unique_gloss = (unique_senses[0].get("gloss")
                            if len(unique_senses) == 1 else None)
            unique_gi = (unique_senses[0].get("grammatical_info")
                         if len({row.get("grammatical_info") for row in unique_senses
                                 if row.get("grammatical_info")}) == 1 and unique_senses
                         else self.analyzer.grammatical_info(
                             entry_ids[0] if len(entry_ids) == 1 else None, unique_gloss))
            exact_lift = any((self.analyzer.lexeme_record(lid) or {}).get("status") == "lift"
                             for lid in entry_ids)
            record = {
                "token": piece, "normalized": piece,
                "segmentation": piece if any(mark in piece for mark in "-=+") else None,
                "gloss": None,
                "lemma": unique_entry.get("lemma_canonical") if unique_entry else None,
                "lemma_id": entry_ids[0] if len(entry_ids) == 1 else None,
                # Candidates signal ENTRY ambiguity only.  A single LIFT entry
                # may still have several senses, but it is already the exact
                # lexical entry and must not look like a homograph conflict.
                "lexeme_candidates": entry_ids if len(entry_ids) > 1 else [],
                "lexical_gloss": unique_gloss,
                "grammatical_info": unique_gi,
                "grammatical_infos": list(unique_entry.get("grammatical_infos") or []),
                "lift_entry_id": unique_entry.get("lift_id") if unique_entry else None,
                "lift_guid": unique_entry.get("lift_guid") if unique_entry else None,
                "lift_sense_id": (unique_senses[0].get("sense_id")
                                  if len(unique_senses) == 1 else None),
                "lexical_gloss_status": (
                    "lift" if unique_entry.get("status") == "lift"
                    else "author_lexeme" if unique_entry
                    else "unknown_lexeme"),
                "confidence": "LOW", "tier": None, "alternatives": [],
                "evidence": ([{"kind": ("LIFT_OR_LINKED_FORM" if exact_lift
                                          else "AUTHOR_LEXEME_FORM"),
                               "entries": entry_ids}] if entry_ids else []),
                "trace": (["exact lexical form linked to a FieldWorks entry; morphology unresolved"]
                          if exact_lift and entry_ids else
                          ["exact lexical form linked by the author; morphology unresolved"]
                          if entry_ids else ["no exact form in imported FLEx corpus"]),
                "uncertainty": (["choose a lexical entry/sense"]
                                if len(entry_ids) > 1 else
                                ["choose a FieldWorks sense"]
                                if unique_entry.get("status") == "lift"
                                   and len(unique_senses) > 1 else
                                ["no corpus analysis found"]),
                "needs_review": True,
                "analysis_surface": piece,
            }
        self._token_cache[cache_key] = copy.deepcopy(record)
        return record

    def gloss_sentence(self, sentence: str, document_id: str | None = None,
                       sentence_id: str | None = None) -> dict[str, Any]:
        from maalula_glosser.gloss_text import tokenize

        sentence = nfc(sentence)
        analyses: list[dict[str, Any]] = []
        piece_counter = 0
        source_cursor = 0
        for word_index, token in enumerate(tokenize(sentence)):
            source_start = sentence.find(token["original"], source_cursor)
            if source_start >= 0:
                source_gap = sentence[source_cursor:source_start]
                source_cursor = source_start + len(token["original"])
            else:
                source_gap = None
            for piece_index, piece in enumerate(token["pieces"]):
                context = None
                if document_id is not None:
                    context = {"document_id": document_id,
                               "sentence_id": sentence_id or "0",
                               "token_index": word_index,
                               "original_token": token["original"]}
                record = self.gloss_token(piece, sentence_initial=(word_index == 0 and piece_index == 0),
                                          context=context)
                record["forced_capital"] = word_index == 0 and piece_index == 0
                record["source_token"] = token["original"]
                record["from_sandhi"] = token["sandhi"]
                record["piece_start"] = record["piece_end"] = piece_counter
                piece_counter += 1
                record["source_gap_before"] = source_gap if piece_index == 0 else ""
                analyses.append(record)
        return {
            "original": sentence,
            "tokens": analyses,
            "segmentation_line": " ".join(a.get("segmentation") or a.get("normalized") or ""
                                            for a in analyses),
            "gloss_line": " ".join(a.get("gloss") or "???" for a in analyses),
            "confidence_counts": {level: sum(1 for a in analyses if a.get("confidence") == level)
                                  for level in ("HIGH", "MEDIUM", "LOW")},
        }

    def gloss_text(self, text: str) -> list[dict[str, Any]]:
        from maalula_glosser.gloss_text import split_sentences
        return [self.gloss_sentence(sentence) for sentence in split_sentences(text)]
