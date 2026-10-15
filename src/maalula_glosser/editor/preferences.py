"""Project-wide editor defaults; never retroactively change stored readings."""
import json
from pathlib import Path
from maalula_glosser.io_utils import atomic_write_text
from maalula_glosser.state import writable

PREFERENCES = "config/editor_preferences.json"


def read(root: Path) -> dict:
    path = writable(root, PREFERENCES)
    value = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return {"reusable_default": value.get("reusable_default", True)}


def save(root: Path, reusable_default: bool) -> dict:
    value = {"reusable_default": reusable_default}
    atomic_write_text(writable(root, PREFERENCES), json.dumps(value, indent=2))
    return value
