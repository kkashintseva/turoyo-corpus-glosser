"""Where editor sessions live on disk.

One JSON file per document under `data/editor_sessions/`, written atomically so
an interrupted save cannot truncate a session. This is the file the user
reopens; it is the whole editable state, nothing derived.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from maalula_glosser.editor import session as S
from maalula_glosser.io_utils import atomic_write_text, read_json
from maalula_glosser.state import writable
import json


class SessionStore:
    def __init__(self, root: Path | str = "."):
        self.root = Path(root)
        self.dir = writable(self.root, "data/editor_sessions")

    def path(self, document_id: str) -> Path:
        # document ids are stable_id output (edoc_<hex>), so they are safe file
        # names; guard anyway against a hand-edited id escaping the directory
        safe = "".join(c for c in document_id if c.isalnum() or c in "_-")
        if not safe:
            raise S.SessionError("empty document id")
        return self.dir / f"{safe}.json"

    def save(self, document: dict[str, Any]) -> Path:
        S.normalize_morphology(document)
        S.validate(document)
        document["updated"] = S.now()
        self.dir.mkdir(parents=True, exist_ok=True)
        target = self.path(document["document_id"])
        atomic_write_text(target, json.dumps(document, ensure_ascii=False, indent=1))
        return target

    def load(self, document_id: str) -> dict[str, Any]:
        data = read_json(self.path(document_id), None)
        if data is None:
            raise FileNotFoundError(document_id)
        S.normalize_morphology(data)
        return S.validate(data)

    def trash(self, document_id: str) -> Path:
        """Move a session out of the live list rather than deleting it: a
        stray click on the wrong row should be recoverable, and a folder is
        cheaper to reason about than an undo stack that outlives the process
        (`history.History` is in memory only, see its own docstring)."""
        source = self.path(document_id)
        if not source.exists():
            raise FileNotFoundError(document_id)
        trash_dir = self.dir / ".trash"
        trash_dir.mkdir(parents=True, exist_ok=True)
        target = trash_dir / source.name
        if target.exists():
            # already-trashed id (rare: only if trashed, restored by hand,
            # then trashed again) — keep both rather than overwrite one
            target = trash_dir / f"{source.stem}.{S.now().replace(':', '-')}{source.suffix}"
        source.rename(target)
        return target

    def list(self) -> list[dict[str, Any]]:
        if not self.dir.exists():
            return []
        rows = []
        for path in sorted(self.dir.glob("*.json")):
            data = read_json(path, None)
            if not isinstance(data, dict):
                continue
            rows.append({"document_id": data.get("document_id"),
                         "title": data.get("title") or "(untitled)",
                         "updated": data.get("updated"),
                         "counts": S.counts(data)})
        rows.sort(key=lambda r: r.get("updated") or "", reverse=True)
        return rows
