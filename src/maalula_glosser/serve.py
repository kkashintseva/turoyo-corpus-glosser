"""Entry point for the hosted editor: Railway's port, one worker.

Distinct from `cli.py edit`, which binds 127.0.0.1 for local, offline,
single-user use and never asks for a login. This one is reachable over a
real network, so it refuses to start at all without a login configured —
`create_app`'s own check (`editor/app.py::_configure_auth`) only *enables*
auth when both variables are set and leaves the app as it always was when
neither is; that permissiveness is right for a laptop and every test in this
repo, but wrong for the one process that will actually sit on a public URL.
The redundancy is deliberate: two independent places refusing a login-less
deploy is cheaper than one silent path around it.

See `docs/DEPLOY_RAILWAY_ROADMAP_RU.md`, Stages 2 and 3.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn

from maalula_glosser.editor.audio_app import create_app


def main() -> None:
    if not (os.environ.get("MAALULA_USER") and os.environ.get("MAALULA_PASSWORD")):
        sys.exit(
            "MAALULA_USER and MAALULA_PASSWORD must both be set to serve this "
            "editor on a network address. Set them (Railway → Variables) "
            "before deploying; `cli.py edit` remains available for local, "
            "offline, single-user use with no login."
        )
    root = Path(os.environ.get("MAALULA_ROOT", "/app"))
    port = int(os.environ.get("PORT", "8000"))
    # warm=True: this is the one process a real user's first request hits
    # cold — see editor/app.py::create_app and Stage 4 of the road-map.
    uvicorn.run(create_app(root, warm=True), host="0.0.0.0", port=port, workers=1)


if __name__ == "__main__":
    main()
