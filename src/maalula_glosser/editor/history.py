"""Undo/redo for editor documents.

In memory, per process: the journal is a convenience for the editing session and
is gone when the server stops. The documents themselves are on disk and are
never at risk — undo replays an edit backwards, it does not restore a file.

Entries record the *change*, not the document. A snapshot per edit would be
correct too, but "The Text" is a 5.7 MB session and twenty snapshots of it would
cost more memory than the whole corpus.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Callable

from . import session

# Deep enough for a working session, bounded so a long session cannot grow
# without limit.
DEPTH = 200


class Entry:
    """One reversible change: what the target looked like before, and after."""

    def __init__(self, kind: str, before: Any, after: Any, **where: Any):
        self.kind = kind
        self.before = before
        self.after = after
        self.where = where

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, **self.where}


class History:
    """Per-document undo and redo stacks.

    `on_record`, if given, is called with (document_id, entry) on every
    `record()` — the one choke point every edit endpoint already passes
    through. It exists to let a caller persist before/after pairs (for gloss-
    correction statistics) without this class knowing anything about files;
    a write failure there must never break undo or the edit itself, so it is
    swallowed here rather than left to the sink or the caller.
    """

    def __init__(self, depth: int = DEPTH,
                 on_record: Callable[[str, "Entry"], None] | None = None):
        self.depth = depth
        self._on_record = on_record
        self._undo: dict[str, deque[Entry]] = {}
        self._redo: dict[str, deque[Entry]] = {}

    def _stack(self, table: dict[str, deque[Entry]], document_id: str) -> deque[Entry]:
        if document_id not in table:
            table[document_id] = deque(maxlen=self.depth)
        return table[document_id]

    def record(self, document_id: str, entry: Entry) -> None:
        """A new edit invalidates the redo branch, as everywhere else."""
        self._stack(self._undo, document_id).append(entry)
        self._stack(self._redo, document_id).clear()
        if self._on_record is not None:
            try:
                self._on_record(document_id, entry)
            except OSError:
                pass

    def can_undo(self, document_id: str) -> bool:
        return bool(self._stack(self._undo, document_id))

    def can_redo(self, document_id: str) -> bool:
        return bool(self._stack(self._redo, document_id))

    def state(self, document_id: str) -> dict[str, bool]:
        return {"can_undo": self.can_undo(document_id),
                "can_redo": self.can_redo(document_id)}

    def pop_undo(self, document_id: str) -> Entry | None:
        stack = self._stack(self._undo, document_id)
        if not stack:
            return None
        entry = stack.pop()
        self._stack(self._redo, document_id).append(entry)
        return entry

    def pop_redo(self, document_id: str) -> Entry | None:
        stack = self._stack(self._redo, document_id)
        if not stack:
            return None
        entry = stack.pop()
        self._stack(self._undo, document_id).append(entry)
        return entry

    def forget(self, document_id: str) -> None:
        self._undo.pop(document_id, None)
        self._redo.pop(document_id, None)


# ---------------------------------------------------------------------------
# applying an entry to a document
# ---------------------------------------------------------------------------
def _sentence(document: dict, sentence_id: str) -> dict | None:
    for sentence in document.get("sentences") or []:
        if sentence.get("sentence_id") == sentence_id:
            return sentence
    return None


def apply(document: dict[str, Any], entry: Entry, *, forward: bool) -> dict[str, Any]:
    """Move the document to the entry's `after` (redo) or `before` (undo) state.

    Returns a description of what changed, so the browser can patch its copy
    instead of reloading a session that may be megabytes long.
    """
    target = entry.after if forward else entry.before
    kind = entry.kind

    if kind == "morpheme":
        for sentence in document.get("sentences") or []:
            for token in sentence.get("tokens") or []:
                for index, morpheme in enumerate(token.get("morphemes") or []):
                    if morpheme.get("morph_id") == entry.where.get("morph_id"):
                        token["morphemes"][index] = dict(target)
                        return {"kind": kind, "morpheme": token["morphemes"][index],
                                "token_id": token.get("token_id")}
        return {"kind": kind}

    if kind == "token":
        for sentence in document.get("sentences") or []:
            for index, token in enumerate(sentence.get("tokens") or []):
                if token.get("token_id") != entry.where.get("token_id"):
                    continue
                # a re-gloss can rewrite the word itself, so line 1 has to walk
                # back with it or the export would print the undone spelling
                session.resurface(sentence, token.get("token_id"),
                                  token.get("surface") or "",
                                  (target or {}).get("surface") or "")
                restored = dict(target)
                # Reuse policy and journal links are independent of text Undo.
                # Undoing segmentation must not silently undo a local exception
                # or lose the handle needed to retract a global reading.
                for key in ("reuse_mode", "reusable_record"):
                    if key in token:
                        restored[key] = token[key]
                sentence["tokens"][index] = restored
                return {"kind": kind, "token": sentence["tokens"][index],
                        "sentence_id": sentence.get("sentence_id")}
        return {"kind": kind}

    if kind == "cells":
        # one batch, one entry: applying a label to sixty places must come back
        # in a single Ctrl+Z, not sixty
        wanted = {cell["morph_id"]: cell for cell in target or []}
        touched = []
        for sentence in document.get("sentences") or []:
            for token in sentence.get("tokens") or []:
                for index, morpheme in enumerate(token.get("morphemes") or []):
                    replacement = wanted.get(morpheme.get("morph_id"))
                    if replacement is None:
                        continue
                    token["morphemes"][index] = dict(replacement)
                    touched.append(token["morphemes"][index])
        return {"kind": kind, "morphemes": touched}

    if kind == "topology_batch":
        for row in target or []:
            apply(document, Entry("tokens", row["tokens"], row["tokens"],
                                  sentence_id=row["sentence_id"]), forward=forward)
        return {"kind": kind, "reload": True}

    if kind == "reading_batch":
        for token in target or []:
            apply(document, Entry("token", token, token, token_id=token["token_id"]),
                  forward=forward)
        return {"kind": kind, "reload": True}

    if kind == "tokens":
        # joining or splitting columns changes how many there are, so the whole
        # row of one block is swapped rather than one token patched
        sentence = _sentence(document, entry.where.get("sentence_id"))
        if sentence is not None:
            current = {token["token_id"]: token for token in sentence["tokens"]}
            restored = []
            for token in target or []:
                item = dict(token)
                previous = current.get(token["token_id"], {})
                for key in ("reuse_mode", "reusable_record"):
                    if key in previous:
                        item[key] = previous[key]
                restored.append(item)
            sentence["tokens"] = restored
            return {"kind": kind, "sentence_id": sentence["sentence_id"],
                    "tokens": sentence["tokens"]}
        return {"kind": kind}

    if kind == "translation":
        sentence = _sentence(document, entry.where.get("sentence_id"))
        if sentence is not None:
            sentence["translation"] = target
            return {"kind": kind, "sentence_id": sentence["sentence_id"],
                    "translation": target}
        return {"kind": kind}

    if kind == "comment":
        # One entry shape covers create (before=None), edit (before and
        # after share a comment_id) and delete (after=None): drop whatever
        # currently has that id, then put `target` back if it isn't None.
        # A presence-only check (as this used to be) cannot tell an edit's
        # unchanged id apart from "already applied" and silently drops the
        # new text.
        sentence = _sentence(document, entry.where.get("sentence_id"))
        if sentence is None:
            return {"kind": kind}
        comments = sentence.setdefault("comments", [])
        comment_id = ((entry.after or entry.before) or {}).get("comment_id")
        index = next((i for i, c in enumerate(comments)
                     if c.get("comment_id") == comment_id), None)
        if target is None:
            if index is not None:
                comments.pop(index)
        elif index is not None:
            # an edit's undo/redo replaces in place — appending instead
            # would relocate the comment to the end of the list on every
            # Ctrl+Z, ahead of comments that were never touched
            comments[index] = dict(target)
        else:
            comments.append(dict(target))
        return {"kind": kind, "sentence_id": sentence["sentence_id"],
                "comments": sentence.get("comments", [])}

    if kind == "line":
        # Adding and removing a line are the same entry read in the two
        # directions: one of `before`/`after` is the line, the other is None.
        sentences = document.setdefault("sentences", [])
        sentence_id = ((entry.after or entry.before) or {}).get("sentence_id")
        at = entry.where.get("index", len(sentences))
        present = any(s.get("sentence_id") == sentence_id for s in sentences)
        if target is None and present:
            document["sentences"] = [s for s in sentences
                                     if s.get("sentence_id") != sentence_id]
        elif target is not None and not present:
            sentences.insert(min(at, len(sentences)), dict(target))
        return {"kind": kind, "reload": True}

    return {"kind": kind}
