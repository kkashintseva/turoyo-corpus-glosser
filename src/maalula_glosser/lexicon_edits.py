"""Author edits to already-established lexemes: merge, delete, rename.

Sibling journal to `lexicon_additions.py`'s `lexicon_additions.jsonl` — same
append-only file with tombstone-retraction convention, same shape
(`writable`/`iter_jsonl`/a stable content-derived id), but a different file
(`lexicon_edits.jsonl`) because these rows describe an edit to an EXISTING
lexeme rather than a brand-new one. Three record types share the file:

- ``merge``: {"type": "merge", "source_lexeme_id", "target_lexeme_id",
  "touched_tokens"}
- ``delete``: {"type": "delete", "lexeme_id", "touched_tokens"}
- ``rename``: {"type": "rename", "lexeme_id", "lemma_canonical"}

`touched_tokens` (merge/delete only) is the PRE-edit snapshot of every
document token the edit rewrote (see `editor/lexeme_resolve.py::_snapshot`/
`attach_lexeme`/`clear_lexeme_tokens`) — stored right here in the
append-only journal so retracting the edit (`app.py`'s
`lexicon_edit_retract`) can restore those documents from this record alone,
with nothing to re-derive. `rename` carries none: it only ever touches an
in-memory `lemma_canonical`, nothing document-level to rewind.

`lexicon_additions.jsonl` rows carry no timestamp field either (recency is
file order, read through `active_readings`'s tombstone loop) — this journal
matches that convention rather than inventing a timestamp nothing else here
has.
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any

from maalula_glosser.io_utils import iter_jsonl, stable_id
from maalula_glosser.state import writable

LEXICON_EDITS = "data/corrections_v2/lexicon_edits.jsonl"
EDIT_TYPES = frozenset(("merge", "delete", "rename"))


def nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value or "")


def _path(root: Path | str) -> Path:
    return writable(Path(root), LEXICON_EDITS)


def _active_rows(path: Path) -> list[dict[str, Any]]:
    """Same tombstone mechanics as `reusable.active_readings`, keyed by
    `edit_id`/`target_edit_id` instead of `addition_id`/`target_addition_id`
    — a distinct id namespace for a distinct file, so a stray retraction row
    from one journal can never accidentally tombstone a row in the other."""
    active: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(iter_jsonl(path)):
        if row.get("scope") == "retracted":
            active.pop(row.get("target_edit_id"), None)
        else:
            key = row.get("edit_id") or f"legacy:{index}"
            active.pop(key, None)
            active[key] = row
    return list(active.values())


def _edit_id(edit: dict[str, Any]) -> str:
    edit_type = edit["type"]
    if edit_type == "merge":
        return stable_id("lexedit", "merge", edit["source_lexeme_id"],
                         edit["target_lexeme_id"])
    if edit_type == "delete":
        return stable_id("lexedit", "delete", edit["lexeme_id"])
    if edit_type == "rename":
        return stable_id("lexedit", "rename", edit["lexeme_id"],
                         edit["lemma_canonical"])
    raise ValueError(f"unknown edit type: {edit_type!r}")


def record_edit(root: Path | str, edit: dict[str, Any]) -> dict[str, Any]:
    """Append one merge/delete/rename record. `edit` carries `type` plus that
    type's own fields (see module docstring); this fills in `edit_id` and
    `scope` and writes the row, following `lexicon_additions.record_addition`'s
    exact append-only convention."""
    edit_type = edit.get("type")
    if edit_type not in EDIT_TYPES:
        raise ValueError(f"type must be one of {sorted(EDIT_TYPES)}")
    row = dict(edit)
    if edit_type == "rename":
        row["lemma_canonical"] = nfc(row.get("lemma_canonical")).strip()
    row["edit_id"] = _edit_id(row)
    row["scope"] = "lexicon_edit"
    path = _path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    # No "unchanged, skip the write" shortcut here (unlike
    # `lexicon_additions.record_addition`): for a rename, `_edit_id` is
    # content-derived (it hashes `lemma_canonical`), so reverting A -> B -> A
    # reuses A's original `edit_id`. A shortcut keyed on "is this exact row
    # already the last one written under this id" would silently no-op that
    # third call — the A-row is still literally present and unretracted in
    # the file, just no longer the newest entry for this lexeme_id, so
    # skipping the append would leave B, not A, as the active reading.
    # Always appending (last-one-wins-by-target, per `active_edits`) is what
    # makes a revert actually take effect.
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def retract_edit(root: Path | str, edit_id: str) -> dict[str, Any]:
    from uuid import uuid4
    path = _path(root)
    if not any(row.get("edit_id") == edit_id for row in _active_rows(path)):
        return {"target_edit_id": edit_id, "already_inactive": True}
    row = {"edit_id": f"retraction_{uuid4().hex}", "scope": "retracted",
           "target_edit_id": edit_id,
           "note": "Undone from the Словник"}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def find_active_edit(root: Path | str, edit_id: str) -> dict[str, Any] | None:
    """Look up one active (non-retracted) edit row by its `edit_id`,
    regardless of type. Used by the retraction endpoint to fetch a
    merge/delete row's `touched_tokens` before tombstoning it — `active_edits`
    itself is keyed by lexeme id, not by `edit_id`, so it can't answer
    this directly."""
    for row in _active_rows(_path(root)):
        if row.get("edit_id") == edit_id:
            return row
    return None


def active_edits(root: Path | str) -> dict[str, dict[str, dict[str, Any]]]:
    """The current, precedence-resolved view of the journal, ready for
    `compositional.py`'s load sequence to apply directly:

    ``{"deletes": {lexeme_id: row}, "merges": {source_lexeme_id: row},
    "renames": {lexeme_id: row}}``

    Precedence: a lexeme that is both renamed/merged-away and later deleted
    has the delete win (it disappears from the vocabulary window entirely,
    so a stale rename or merge for the same id would be meaningless).
    """
    deletes: dict[str, dict[str, Any]] = {}
    merges: dict[str, dict[str, Any]] = {}
    renames: dict[str, dict[str, Any]] = {}
    for row in _active_rows(_path(root)):
        edit_type = row.get("type")
        if edit_type == "delete":
            deletes[row["lexeme_id"]] = row
        elif edit_type == "merge":
            merges[row["source_lexeme_id"]] = row
        elif edit_type == "rename":
            renames[row["lexeme_id"]] = row
    for lexeme_id in deletes:
        renames.pop(lexeme_id, None)
        merges.pop(lexeme_id, None)
    return {"deletes": deletes, "merges": merges, "renames": renames}
