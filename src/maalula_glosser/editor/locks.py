"""Which browser holds the pen for a document, one editor at a time.

In memory, per process — the same trade-off `history.py` makes and for the
same reason: this is a fact about the live editing session, not an
editorial artefact that belongs in the saved JSON. A server restart frees
every document; the next person to open one becomes its editor again, which
is fine, since "first to open a free document" is exactly how a lock is
supposed to be won in the first place.

There is deliberately no auto-expiry. A held pen stays held — even past the
holder closing their tab — until someone explicitly takes it over. The UI
compensates by showing when the document was last actually edited
(`document["updated"]`), so a reader can judge for themselves whether the
holder is still around.
"""
from __future__ import annotations

from typing import Any

from maalula_glosser.editor import session as S


class EditLocks:
    def __init__(self) -> None:
        self._by_doc: dict[str, dict[str, Any]] = {}

    def status(self, document_id: str) -> dict[str, Any] | None:
        return self._by_doc.get(document_id)

    def claim(self, document_id: str, client_id: str, name: str) -> dict[str, Any]:
        """Become the editor if the document is free; otherwise change nothing.

        Either way, the caller gets back whoever actually holds it now, so a
        newly opened tab can tell from one call whether it is the editor or a
        reader.
        """
        current = self._by_doc.get(document_id)
        if current is None or current["client_id"] == client_id:
            current = {"client_id": client_id, "name": name, "since": S.now()}
            self._by_doc[document_id] = current
        return current

    def take_over(self, document_id: str, client_id: str, name: str) -> dict[str, Any]:
        """Reassign the pen unconditionally — the only way to displace a holder."""
        current = {"client_id": client_id, "name": name, "since": S.now()}
        self._by_doc[document_id] = current
        return current

    def release(self, document_id: str, client_id: str) -> bool:
        """Free the document, but only for the client that currently holds it."""
        current = self._by_doc.get(document_id)
        if current is not None and current["client_id"] == client_id:
            del self._by_doc[document_id]
            return True
        return False

    def holder_is(self, document_id: str, client_id: str) -> bool:
        current = self._by_doc.get(document_id)
        return current is not None and client_id != "" and current["client_id"] == client_id
