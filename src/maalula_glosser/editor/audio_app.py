"""Editor audio helpers.

Persistent document audio/timecode metadata comes from ``editor.media``.
Yandex Disk public-share URLs need one extra layer because they are HTML pages:
the documented public-resource API turns a stable share URL into a short-lived
direct download URL.  Resolve that on the server so browser CORS does not get
in the way and expiring downloader URLs are never saved in editor data.
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

from fastapi import HTTPException
from fastapi.responses import RedirectResponse

from maalula_glosser.editor.media import create_app as _create_app

YANDEX_PUBLIC_DOWNLOAD_API = (
    "https://cloud-api.yandex.net/v1/disk/public/resources/download"
)
_YANDEX_SHARE_HOSTS = {
    "yadi.sk",
    "disk.yandex.ru",
    "disk.yandex.ee",
    "disk.yandex.com",
    "disk.yandex.net",
    "disk.yandex.az",
    "disk.yandex.by",
    "disk.yandex.co.il",
    "disk.yandex.com.am",
    "disk.yandex.com.ge",
    "disk.yandex.com.tr",
    "disk.yandex.fr",
    "disk.yandex.kg",
    "disk.yandex.kz",
    "disk.yandex.lt",
    "disk.yandex.lv",
    "disk.yandex.md",
    "disk.yandex.tj",
    "disk.yandex.tm",
    "disk.yandex.uz",
    "disk.360.yandex.ru",
    "disk.360.yandex.ee",
    "disk.360.yandex.com",
    "disk.360.yandex.net",
    "disk.360.yandex.az",
    "disk.360.yandex.by",
    "disk.360.yandex.co.il",
    "disk.360.yandex.com.am",
    "disk.360.yandex.com.ge",
    "disk.360.yandex.com.tr",
    "disk.360.yandex.fr",
    "disk.360.yandex.kg",
    "disk.360.yandex.kz",
    "disk.360.yandex.lt",
    "disk.360.yandex.lv",
    "disk.360.yandex.md",
    "disk.360.yandex.tj",
    "disk.360.yandex.tm",
    "disk.360.yandex.uz",
}
_SHARE_PATH_RE = re.compile(r"^/(d|i)/([^/?#]+)/?$")


def canonical_yandex_public_url(url: str) -> str:
    """Validate a Yandex share URL and normalize locale/360 host variants."""
    parsed = urlsplit((url or "").strip())
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or host not in _YANDEX_SHARE_HOSTS:
        raise ValueError("not a supported Yandex Disk public URL")
    match = _SHARE_PATH_RE.fullmatch(parsed.path)
    if not match:
        raise ValueError("unsupported Yandex Disk public-link shape")
    kind, key = match.groups()
    return f"https://disk.yandex.ru/{kind}/{key}"


def resolve_yandex_download_url(public_url: str, *, timeout: float = 15.0) -> str:
    """Return Yandex's current direct, time-limited download URL."""
    canonical = canonical_yandex_public_url(public_url)
    endpoint = f"{YANDEX_PUBLIC_DOWNLOAD_API}?{urlencode({'public_key': canonical})}"
    request = Request(endpoint, headers={"User-Agent": "turoyo-corpus-glosser/1"})
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed Yandex API host
            payload = json.load(response)
    except HTTPError as exc:
        raise RuntimeError(f"Yandex Disk API returned HTTP {exc.code}") from exc
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Yandex Disk API request failed: {exc}") from exc

    href = payload.get("href") if isinstance(payload, dict) else None
    if not isinstance(href, str) or not href.startswith(("https://", "http://")):
        raise RuntimeError("Yandex Disk API did not return a download URL")
    return href


def create_app(root: Path | str = ".", *, warm: bool = False):
    """Persistent media editor plus a same-origin Yandex-audio redirect."""
    app = _create_app(root, warm=warm)

    @app.get("/api/audio/yandex")
    async def yandex_audio(url: str):
        try:
            canonical_yandex_public_url(url)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            href = await asyncio.to_thread(resolve_yandex_download_url, url)
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return RedirectResponse(href, status_code=307)

    return app
