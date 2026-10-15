"""Where the editor's writable files live.

On a laptop they live in the checkout, next to the data they correct. On a
server the checkout is rebuilt at every deploy, so they live on a mounted
volume instead — set MAALULA_STATE_DIR to its path — and are seeded from the
checkout the first time the volume sees each file, so the curated readings
that ship with the checkout (`reusable_morphology.jsonl`,
`gloss_inventory.json`) are not silently blank on a fresh volume.

Read-only corpus and lexicon data never goes through this: only files the
editor itself writes.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path


def state_root(root: Path) -> Path:
    """The volume, or the checkout when there is no volume."""
    configured = os.environ.get("MAALULA_STATE_DIR")
    return Path(configured).resolve() if configured else Path(root)


def writable(root: Path, relative: str) -> Path:
    """A path the editor may write to, seeded from the checkout once."""
    base = state_root(root)
    target = base / relative
    # Compared resolved: state_root() resolves a configured volume path, but
    # not the checkout fallback, so an unresolved "." vs. its resolved self
    # would otherwise look like two different places and try to copy a file
    # onto itself.
    if base.resolve() != Path(root).resolve() and not target.exists():
        source = Path(root) / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_file():
            shutil.copy2(source, target)
        elif source.is_dir():
            shutil.copytree(source, target)
    return target
