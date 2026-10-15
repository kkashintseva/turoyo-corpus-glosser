"""Arnold's verbs by (project-canonical) root, from the Arnold working
lexicon `data/lexicon_v3/lexicon.jsonl`: the entries that carry a stem class
(`I`, `II`, `IV`, `I7`…), with their principal parts as a citation form in
the project's transcription (`isbaḳ` -> `isbak`, `aḥək yaḥək` -> `aḥəc yaḥəc`).

Used where the project lexicon (MASC) has no verb of a root at all: the
analyzer's service verb (`lexsvc_<root>`) and the editor's «Леммы…» pass both
take its citation from here, so both name the same verb the same way.
"""
from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any


def nfc(value: str | None) -> str:
    return unicodedata.normalize("NFC", value or "")


@lru_cache(maxsize=4)
def _arnold_verbs(lexicon_path: str, _mtime: float) -> dict[str, list[dict[str, Any]]]:
    from maalula_glosser.normalization.transcription import TranscriptionNormalizer
    normalizer = TranscriptionNormalizer.from_yaml(
        str(Path(lexicon_path).parents[2] / "config" / "transcription_map.yaml"))

    def canon(value: str) -> str:
        value = unicodedata.normalize("NFD", value or "").replace("́", "").replace("̀", "")
        return nfc(normalizer.canonical(nfc(value), source_system="arnold"))

    by_root: dict[str, list[dict[str, Any]]] = {}
    with open(lexicon_path, encoding="utf-8") as handle:
        for line in handle:
            entry = json.loads(line)
            if not entry.get("stem_class") or not entry.get("root"):
                continue
            words = [canon(p.split()[0]) for p in entry.get("principal_parts") or [] if p.split()]
            words = [w for w in words if w and "." not in w and w[:1].islower()]
            perfect = words[0] if words else nfc(entry.get("headword_canonical"))
            imperfect = next((w for w in words[1:] if w.startswith("y")), "")
            citation = f"{perfect} {imperfect}".strip()
            if not citation:
                continue
            german = "; ".join(s.get("de", "") for s in entry.get("senses") or [])[:160]
            by_root.setdefault(canon(re.sub(r"\d+$", "", entry["root"])), []).append(
                {"citation": citation, "stem_class": entry["stem_class"],
                 "page": entry.get("page"), "german": german})
    return by_root


def arnold_verbs(root_or_analyzer) -> dict[str, list[dict[str, Any]]]:
    # an analyzer carries `.root`; a Path has a `.root` too ("/"), so test it first
    root = (root_or_analyzer if isinstance(root_or_analyzer, (str, Path))
            else root_or_analyzer.root)
    path = Path(root) / "data" / "lexicon_v3" / "lexicon.jsonl"
    if not path.exists():
        return {}
    return _arnold_verbs(str(path), path.stat().st_mtime)


