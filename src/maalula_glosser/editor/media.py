"""Audio-link and timecode metadata for the gloss editor.

The IGT parser already preserves source timecodes in ``source_locator.timecode``.
This module adds the small writable layer the browser needs: a document-level
source-audio URL and an independently editable timecode on each editor line.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from maalula_glosser.editor import session as S
from maalula_glosser.editor.app import create_app as base_create_app
from maalula_glosser.editor.store import SessionStore



class AudioUrlBody(BaseModel):
    url: str = ""


class TimecodeBody(BaseModel):
    text: str = ""


_CLOCK_RE = re.compile(r"^\d{1,4}:\d{2}(?::\d{2})?$")


def _normalise_clock(value: str) -> str:
    value = value.strip()
    if not _CLOCK_RE.fullmatch(value):
        raise ValueError("expected MM:SS or HH:MM:SS")
    numbers = [int(part) for part in value.split(":")]
    if len(numbers) == 2:
        minutes, seconds = numbers
        if seconds >= 60:
            raise ValueError("seconds must be below 60")
        return f"{minutes:02d}:{seconds:02d}"
    hours, minutes, seconds = numbers
    if minutes >= 60 or seconds >= 60:
        raise ValueError("minutes and seconds must be below 60")
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def normalise_timecode(value: str) -> str:
    """Canonicalise an editable point/range while allowing an empty field."""
    value = S.nfc(value).strip()
    if not value:
        return ""
    value = value.strip("<>[] ").replace("–", "-").replace("—", "-")
    parts = re.split(r"\s*-\s*", value)
    if len(parts) not in (1, 2) or any(not part for part in parts):
        raise ValueError("expected MM:SS or MM:SS-MM:SS")
    return "-".join(_normalise_clock(part) for part in parts)


def normalise_audio_url(value: str) -> str:
    """Keep only browser-loadable http(s) URLs; blank detaches the audio."""
    value = S.nfc(value).strip()
    if not value:
        return ""
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("audio URL must be an http(s) URL")
    return value


def _find_sentence(document: dict[str, Any], sentence_id: str) -> dict[str, Any]:
    for sentence in document.get("sentences") or []:
        if sentence.get("sentence_id") == sentence_id:
            return sentence
    raise KeyError(sentence_id)


def install_media_routes(app: FastAPI, root: Path | str = ".") -> FastAPI:
    """Attach persistent audio/timecode metadata routes to an editor app.

    The base app's edit-lock middleware is path-based, so these routes inherit
    the same single-editor protection automatically.
    """
    store = SessionStore(root)

    @app.patch("/api/documents/{document_id}/audio")
    def edit_audio_url(document_id: str, body: AudioUrlBody):
        try:
            url = normalise_audio_url(body.url)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        try:
            document = store.load(document_id)
        except FileNotFoundError:
            raise HTTPException(404, "no such document")
        document["audio_url"] = url
        store.save(document)
        return {"audio_url": url}

    @app.patch("/api/documents/{document_id}/sentences/{sentence_id}/timecode")
    def edit_timecode(document_id: str, sentence_id: str, body: TimecodeBody):
        try:
            timecode = normalise_timecode(body.text)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        try:
            document = store.load(document_id)
        except FileNotFoundError:
            raise HTTPException(404, "no such document")
        try:
            sentence = _find_sentence(document, sentence_id)
        except KeyError:
            raise HTTPException(404, "no such sentence")
        locator = sentence.setdefault("source_locator", {})
        # Deliberately do not rewrite segment_marker.  A source marker can
        # contain a verse/block number; timecode is its own editable field.
        locator["timecode"] = timecode
        store.save(document)
        return {"sentence_id": sentence_id, "timecode": timecode}

    return app


def create_app(root: Path | str = ".", *, warm: bool = False) -> FastAPI:
    app = base_create_app(root, warm=warm)
    return install_media_routes(app, root)
