#!/usr/bin/env python3
"""
Export the current Turoyo lexicon to one CSV file.

The export combines:
- imported FieldWorks/LIFT lexical entries;
- local lexical entries added in the editor;
- author-added senses;
- local rename/delete/merge edits.

One CSV row = one current lexical entry.

Run from the project root:
    python3 scripts/export_lexicon_csv.py

Optional:
    python3 scripts/export_lexicon_csv.py --output ~/Desktop/turoyo_lexicon.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import unicodedata
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Iterable


def nfc(value: Any) -> str:
    return unicodedata.normalize("NFC", str(value or "")).strip()


def find_project_root(start: Path | None = None) -> Path:
    """Find the project root by pyproject.toml."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate

    # Also support running the script directly from scripts/.
    script_path = Path(__file__).resolve()
    for candidate in (script_path.parent, *script_path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate

    raise SystemExit(
        "Не удалось найти корень проекта (pyproject.toml). "
        "Запусти скрипт из папки turoyo-corpus-glosser."
    )


def add_src_to_path(root: Path) -> None:
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def unique(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = nfc(value)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def join(values: Iterable[Any]) -> str:
    return " | ".join(unique(values))


def preferred_form(forms: list[dict[str, Any]], preferred_lang: str = "tru") -> str:
    for row in forms:
        if row.get("lang") == preferred_lang and nfc(row.get("text")):
            return nfc(row.get("text"))
    for row in forms:
        if nfc(row.get("text")):
            return nfc(row.get("text"))
    return ""


def lift_sense_strings(entry: dict[str, Any]) -> list[str]:
    """Flatten all LIFT senses, keeping language when it is not English."""
    result: list[str] = []
    for sense in entry.get("senses") or []:
        glosses = sense.get("glosses") or []
        if glosses:
            for gloss in glosses:
                text = nfc(gloss.get("text"))
                if not text:
                    continue
                lang = nfc(gloss.get("lang"))
                result.append(text if not lang or lang == "en" else f"{text} [{lang}]")
        else:
            for definition in sense.get("definitions") or []:
                text = nfc(definition.get("text"))
                if not text:
                    continue
                lang = nfc(definition.get("lang"))
                result.append(text if not lang or lang == "en" else f"{text} [{lang}]")
    return unique(result)


def lift_variant_strings(entry: dict[str, Any]) -> list[str]:
    result: list[str] = []
    lemma = nfc(entry.get("lemma_canonical"))
    for variant in entry.get("variants") or []:
        for form in variant.get("forms") or []:
            text = nfc(form.get("text"))
            if text and text != lemma:
                result.append(text)
    return unique(result)


def lift_grammatical_info(entry: dict[str, Any]) -> list[str]:
    infos = list(entry.get("grammatical_infos") or [])
    if infos:
        return unique(infos)
    # Fallback for data created by older versions.
    for sense in entry.get("senses") or []:
        if sense.get("grammatical_info"):
            infos.append(sense["grammatical_info"])
    return unique(infos)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export the current Turoyo lexicon to CSV."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Project root (normally unnecessary).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Default: data/export/turoyo_lexicon_YYYY-MM-DD.csv",
    )
    args = parser.parse_args()

    root = find_project_root(args.root)
    add_src_to_path(root)

    from maalula_glosser.lift_lexicon import LiftLexiconStore
    from maalula_glosser import lexicon_additions as LA
    from maalula_glosser import lexicon_edits as LE
    from maalula_glosser.editor import senses as SE

    lift_entries = LiftLexiconStore(root).entries()
    additions = LA.active_additions(root)
    edits = LE.active_edits(root)
    local_senses = SE.active_lexical_senses(root)

    deletes = edits.get("deletes") or {}
    merges = edits.get("merges") or {}
    renames = edits.get("renames") or {}

    # Current entries keyed by lexeme_id.
    entries: dict[str, dict[str, Any]] = {}

    # 1. FieldWorks / LIFT
    for entry in lift_entries:
        lexeme_id = nfc(entry.get("lexeme_id"))
        if not lexeme_id:
            continue
        entries[lexeme_id] = {
            "entry_id": lexeme_id,
            "lift_id": nfc(entry.get("lift_id")),
            "guid": nfc(entry.get("guid")),
            "lemma": nfc(entry.get("lemma_canonical"))
                     or preferred_form(entry.get("citation") or [])
                     or preferred_form(entry.get("lexical_unit") or []),
            "grammatical_info": lift_grammatical_info(entry),
            "senses": lift_sense_strings(entry),
            "variants": lift_variant_strings(entry),
            "part_of_speech": nfc(entry.get("part_of_speech")),
            "morph_type": nfc(entry.get("morph_type")),
            "source": "FieldWorks",
        }

    # 2. Local lexical entries / local forms.
    # If a local row has the same lexeme_id as a LIFT entry, treat its surfaces
    # as additional forms of that same entry instead of duplicating the entry.
    for addition in additions:
        lexeme_id = nfc(addition.get("lexeme_id"))
        if not lexeme_id:
            continue

        surfaces = [
            nfc(row.get("form_canonical"))
            for row in (addition.get("surfaces") or [])
            if nfc(row.get("form_canonical"))
        ]

        if lexeme_id in entries:
            entries[lexeme_id]["variants"] = unique(
                list(entries[lexeme_id]["variants"]) + surfaces
            )
            if addition.get("grammatical_info"):
                entries[lexeme_id]["grammatical_info"] = unique(
                    list(entries[lexeme_id]["grammatical_info"])
                    + [addition.get("grammatical_info")]
                )
            continue

        entries[lexeme_id] = {
            "entry_id": lexeme_id,
            "lift_id": "",
            "guid": "",
            "lemma": nfc(addition.get("lemma_canonical")),
            "grammatical_info": unique([addition.get("grammatical_info")]),
            "senses": [],
            "variants": unique(surfaces),
            "part_of_speech": nfc(addition.get("part_of_speech")),
            "morph_type": "",
            "source": "Added",
        }

    # 3. Author-added lexical senses.
    for lexeme_id, senses in local_senses.items():
        lexeme_id = nfc(lexeme_id)
        if not lexeme_id:
            continue

        if lexeme_id not in entries:
            # A very old/local lexeme may exist only in the senses journal.
            first = senses[0] if senses else {}
            entries[lexeme_id] = {
                "entry_id": lexeme_id,
                "lift_id": "",
                "guid": "",
                "lemma": nfc(first.get("lemma_canonical")),
                "grammatical_info": [],
                "senses": [],
                "variants": [],
                "part_of_speech": nfc(first.get("part_of_speech")),
                "morph_type": "",
                "source": "Added",
            }

        for sense in senses:
            gloss = nfc(sense.get("gloss"))
            if gloss:
                entries[lexeme_id]["senses"] = unique(
                    list(entries[lexeme_id]["senses"]) + [gloss]
                )
            stem_form = nfc(sense.get("stem_form"))
            if stem_form and stem_form != entries[lexeme_id]["lemma"]:
                entries[lexeme_id]["variants"] = unique(
                    list(entries[lexeme_id]["variants"]) + [stem_form]
                )

    # 4. Apply active renames.
    for lexeme_id, edit in renames.items():
        lexeme_id = nfc(lexeme_id)
        if lexeme_id in entries and nfc(edit.get("lemma_canonical")):
            entries[lexeme_id]["lemma"] = nfc(edit.get("lemma_canonical"))

    # 5. Apply merges conservatively: move useful local information from the
    # source into the target, then omit the source entry.
    merged_sources: set[str] = set()
    for source_id, edit in merges.items():
        source_id = nfc(source_id)
        target_id = nfc(edit.get("target_lexeme_id"))
        if not source_id or not target_id or source_id == target_id:
            continue
        source = entries.get(source_id)
        target = entries.get(target_id)
        if source and target:
            target["senses"] = unique(target["senses"] + source["senses"])
            target["variants"] = unique(
                target["variants"] + source["variants"] + [source["lemma"]]
            )
            target["grammatical_info"] = unique(
                target["grammatical_info"] + source["grammatical_info"]
            )
        merged_sources.add(source_id)

    # 6. Omit deleted and merged-away entries.
    for lexeme_id in set(deletes) | merged_sources:
        entries.pop(nfc(lexeme_id), None)

    rows = []
    for entry in entries.values():
        lemma = nfc(entry.get("lemma"))
        if not lemma:
            continue

        variants = [
            value
            for value in unique(entry.get("variants") or [])
            if value != lemma
        ]

        rows.append({
            "entry_id": nfc(entry.get("entry_id")),
            "lift_id": nfc(entry.get("lift_id")),
            "guid": nfc(entry.get("guid")),
            "lemma": lemma,
            "grammatical_info": join(entry.get("grammatical_info") or []),
            "part_of_speech": nfc(entry.get("part_of_speech")),
            "morph_type": nfc(entry.get("morph_type")),
            "senses": join(entry.get("senses") or []),
            "variants": join(variants),
            "source": nfc(entry.get("source")),
        })

    rows.sort(
        key=lambda row: (
            row["lemma"].casefold(),
            row["grammatical_info"].casefold(),
            row["entry_id"],
        )
    )

    output = args.output
    if output is None:
        output = root / "data" / "export" / f"turoyo_lexicon_{date.today().isoformat()}.csv"
    elif not output.is_absolute():
        output = (root / output).resolve()

    output.parent.mkdir(parents=True, exist_ok=True)

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

    # utf-8-sig writes a BOM, which makes Excel recognize Unicode reliably.
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    fieldworks_count = sum(1 for row in rows if row["source"] == "FieldWorks")
    added_count = sum(1 for row in rows if row["source"] == "Added")

    print(f"Готово: {output}")
    print(f"Статей: {len(rows)}")
    print(f"FieldWorks: {fieldworks_count}")
    print(f"Added: {added_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
