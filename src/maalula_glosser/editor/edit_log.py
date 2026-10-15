"""Append-only log of what an edit's target looked like before and after.

Feeds gloss-correction statistics later. This is an event stream, not a
before/after table: for a given morph_id, the auto baseline is the *first*
logged entry whose before["origin"] == "auto"; a second edit to the same
cell has a "user" baseline instead. A token carrying imported_igt starts
life with origin "user" (see session._imported_user_cells) and so has no
auto baseline at all — bucket those separately when aggregating.

Undo and redo don't produce History entries (they would clear the redo
stack), so they are logged directly here rather than through History.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from maalula_glosser.io_utils import append_jsonl
from maalula_glosser.state import writable

from . import history as H
from . import session as S

LOG_NAME = "edit_log.jsonl"


def _path(root: Path | str) -> Path:
    return writable(Path(root), "data/editor_sessions") / LOG_NAME


def sink(root: Path | str) -> Callable[[str, "H.Entry"], None]:
    """An `on_record` callable for `History`, writing one line per entry."""
    path = _path(root)

    def _write(document_id: str, entry: "H.Entry") -> None:
        append_jsonl(path, {
            "ts": S.now(),
            "document_id": document_id,
            "kind": entry.kind,
            **entry.where,
            "before": entry.before,
            "after": entry.after,
        })

    return _write


def log_undo_redo(root: Path | str, action: str, document_id: str,
                   entry: "H.Entry") -> None:
    try:
        append_jsonl(_path(root), {
            "ts": S.now(),
            "document_id": document_id,
            "kind": action,
            "of": entry.kind,
            **entry.where,
        })
    except OSError:
        pass
