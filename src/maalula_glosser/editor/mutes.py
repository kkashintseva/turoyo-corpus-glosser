"""Per-editor «не предлагать для …»: words for which one editor does not want
the "same reading elsewhere?" panel. Keyed by the editor's display name (the
name they hold the pen under), not by browser, so it follows the person.

A key is a lexeme id when the word has one, else `form:<surface>`; `label` is
only what the UI shows back. Nothing here changes a reading or a rule.
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any

from maalula_glosser.io_utils import atomic_write_text
from maalula_glosser.state import writable

MUTES = "config/editor_mutes.json"


def _name(name: str | None) -> str:
    return unicodedata.normalize("NFC", (name or "").strip()).casefold()


def _load(root: Path) -> dict[str, dict[str, str]]:
    path = writable(root, MUTES)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read(root: Path, name: str | None) -> dict[str, Any]:
    muted = _load(root).get(_name(name), {})
    return {"name": name or "", "muted": [{"key": k, "label": v} for k, v in sorted(muted.items())]}


def set_muted(root: Path, name: str | None, key: str, label: str, muted: bool) -> dict[str, Any]:
    if not key:
        raise ValueError("key is required")
    table = _load(root)
    mine = table.setdefault(_name(name), {})
    if muted:
        mine[key] = label or key
    else:
        mine.pop(key, None)
    atomic_write_text(writable(root, MUTES), json.dumps(table, ensure_ascii=False, indent=2))
    return read(root, name)
