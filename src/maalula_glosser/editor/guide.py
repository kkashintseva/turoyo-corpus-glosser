"""Editable in-app Turoyo guide stored as Markdown."""
from __future__ import annotations

import html
import re
from pathlib import Path

from maalula_glosser.io_utils import atomic_write_text
from maalula_glosser.state import writable

GUIDE = "docs/USER_GUIDE_RU.md"

_CODE_FENCE = re.compile(r"```\n?(.*?)```", re.S)
_CODE_SPAN = re.compile(r"`([^`]+)`")
_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_HR = re.compile(r"^-{3,}$")
_HEADING = re.compile(r"^(#{1,6})\s+(.+)$")
_UL_ITEM = re.compile(r"^\s*[*-]\s+(.*)$")
_OL_ITEM = re.compile(r"^\s*(\d+)\.\s+(.*)$")


def read(root: Path) -> str:
    path = writable(root, GUIDE)
    return path.read_text(encoding="utf-8") if path.exists() else ""


def save(root: Path, text: str) -> str:
    atomic_write_text(writable(root, GUIDE), text)
    return text


def _inline(fragment: str) -> str:
    spans: list[str] = []

    def stash(match: re.Match) -> str:
        spans.append(match.group(1))
        return f"\x00{len(spans) - 1}\x00"

    fragment = _CODE_SPAN.sub(stash, fragment)
    fragment = _BOLD.sub(r"<strong>\1</strong>", fragment)
    return re.sub(
        r"\x00(\d+)\x00",
        lambda m: f"<code>{spans[int(m.group(1))]}</code>",
        fragment,
    )


def render(text: str) -> str:
    """Render the small safe Markdown subset used by the project guide."""
    escaped = html.escape(text, quote=False)
    code_blocks: list[str] = []

    def stash_block(match: re.Match) -> str:
        code_blocks.append(match.group(1).rstrip("\n"))
        return f"\n\n\x00BLOCK{len(code_blocks) - 1}\x00\n\n"

    escaped = _CODE_FENCE.sub(stash_block, escaped)
    parts: list[str] = []

    for raw_block in re.split(r"\n{2,}", escaped):
        block = raw_block.strip("\n")
        if not block.strip():
            continue

        block_ref = re.match(r"^\x00BLOCK(\d+)\x00$", block.strip())
        if block_ref:
            parts.append(
                f"<pre><code>{code_blocks[int(block_ref.group(1))]}</code></pre>"
            )
            continue

        if _HR.match(block.strip()):
            parts.append("<hr>")
            continue

        heading = _HEADING.match(block) if "\n" not in block else None
        if heading:
            level = len(heading.group(1))
            parts.append(f"<h{level}>{_inline(heading.group(2))}</h{level}>")
            continue

        lines = block.split("\n")
        first_ol = _OL_ITEM.match(lines[0])
        is_ul = bool(_UL_ITEM.match(lines[0]))
        is_ol = bool(first_ol)
        if is_ul or is_ol:
            marker = _UL_ITEM if is_ul else _OL_ITEM
            content_group = 1 if is_ul else 2
            items: list[str] = []
            for line in lines:
                match = marker.match(line)
                if match:
                    items.append(match.group(content_group))
                elif items:
                    items[-1] += " " + line.strip()
            tag = "ul" if is_ul else "ol"
            parts.append(
                f"<{tag}>" + "".join(f"<li>{_inline(item)}</li>" for item in items) + f"</{tag}>"
            )
            continue

        parts.append(f"<p>{_inline(' '.join(line.strip() for line in lines))}</p>")

    return "\n".join(parts)
