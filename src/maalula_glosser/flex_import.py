"""Shared FLEx import service for the browser editor and CLI.

Import is deliberately separated from the language-specific analyzer.  It
materializes each ``interlinear-text`` as an editable editor session, stores
one evidence shard per FLEx text, archives the original XML byte-for-byte, and
rebuilds the exact-form corpus index once at the end.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from maalula_glosser.editor.store import SessionStore
from maalula_glosser.flex_corpus import FlexCorpusStore
from maalula_glosser.ingest.flextext_parser import (
    evidence_from_text,
    flex_text_to_document,
    iter_flextext,
)


def import_flextext_to_state(
    source: Path | str,
    root: Path | str = ".",
    *,
    display_name: str | None = None,
    session_store: SessionStore | None = None,
    corpus_store: FlexCorpusStore | None = None,
) -> dict[str, Any]:
    source = Path(source).expanduser().resolve()
    root = Path(root).resolve()
    if source.suffix.lower() not in {".flextext", ".xml"}:
        raise ValueError("FLEx import expects a .flextext or .xml file")
    if not source.is_file():
        raise FileNotFoundError(source)

    name = display_name or source.name
    sessions = session_store or SessionStore(root)
    corpus = corpus_store or FlexCorpusStore(root)
    archived = corpus.archive_source(source, name)

    imported: list[dict[str, Any]] = []
    total_segments = total_tokens = total_analysed = 0
    for text in iter_flextext(source, filename=name):
        document = flex_text_to_document(text)
        sessions.save(document)
        stats = corpus.save_text(text, evidence_from_text(text))
        segments = len(text.get("segments") or [])
        total_segments += segments
        total_tokens += stats["tokens"]
        total_analysed += stats["analysed_tokens"]
        imported.append({
            "document_id": document["document_id"],
            "text_id": text.get("id"),
            "title": document.get("title"),
            "segments": segments,
            **stats,
        })

    manifest = corpus.rebuild()
    return {
        "documents": imported,
        "first_document_id": imported[0]["document_id"] if imported else None,
        "report": {
            "source_file": name,
            "texts": len(imported),
            "segments": total_segments,
            "tokens": total_tokens,
            "analysed_tokens": total_analysed,
            "archive": archived,
            "corpus": manifest,
        },
    }
