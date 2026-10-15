"""Real Word footnotes for python-docx, which exposes no public API for them.

Comments (`sentence["comments"]`, the примечания) export as footnotes rather
than Word comments — see docs/superpowers/specs/
2026-09-12-discussion-threads-and-editor-lock-design.md, §6.1, which also
records the spike this was verified against: python-docx already ships the
`WML_FOOTNOTES` content type and `FOOTNOTES` relationship constants it needs
(they exist for exactly this purpose, just not wrapped in a public
`add_footnote`), and content-types/relationships are handled by python-docx
itself once a part is attached with `relate_to` — the same mechanism
`docx/parts/comments.py` uses internally to add comment support. No zip or
`[Content_Types].xml` surgery is needed.
"""
from __future__ import annotations

from typing import Any, cast
from xml.sax.saxutils import escape

from docx.opc.constants import CONTENT_TYPE as CT
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.opc.packuri import PackURI
from docx.oxml.ns import qn
from docx.oxml.parser import parse_xml
from docx.parts.story import StoryPart

_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

# Pin the marker itself instead of relying on Word's FootnoteReference style:
# templates may define that style at a different size.  OOXML stores font size
# in half-points, so 20 means 10pt.
_REFERENCE_PROPERTIES = (
    '<w:rStyle w:val="FootnoteReference"/>'
    '<w:sz w:val="20"/><w:szCs w:val="20"/>'
    '<w:vertAlign w:val="superscript"/>'
)

# The two footnotes every `footnotes.xml` must carry, at the reserved ids
# `-1` and `0` — Word supplies the actual separator/continuation mark itself;
# these elements just have to exist for the part to be well-formed.
_DEFAULT_FOOTNOTES_XML = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:footnotes xmlns:w="{_W}">
  <w:footnote w:type="separator" w:id="-1"><w:p><w:r><w:separator/></w:r></w:p></w:footnote>
  <w:footnote w:type="continuationSeparator" w:id="0"><w:p><w:r><w:continuationSeparator/></w:r></w:p></w:footnote>
</w:footnotes>
""".encode("utf-8")


class FootnotesPart(StoryPart):
    """Container part for footnotes, created the first time one is added.

    Mirrors `docx.parts.comments.CommentsPart`: python-docx has no public
    footnotes API, but adds comment support itself through exactly this
    pattern (a `StoryPart` subclass plus `document_part.relate_to`).
    """

    @classmethod
    def default(cls, package) -> "FootnotesPart":
        return cls(PackURI("/word/footnotes.xml"), CT.WML_FOOTNOTES,
                   parse_xml(_DEFAULT_FOOTNOTES_XML), package)


def _footnotes_part(document) -> FootnotesPart:
    document_part = document.part
    try:
        return cast(FootnotesPart, document_part.part_related_by(RT.FOOTNOTES))
    except KeyError:
        part = FootnotesPart.default(document_part.package)
        document_part.relate_to(part, RT.FOOTNOTES)
        return part


class FootnoteAnchors:
    """Adds footnotes in document order, several to the same anchor run.

    `anchor_run._r.addnext(...)` always inserts immediately after the
    *original* run, so a second footnote on the same token would land ahead
    of the first one rather than after it. This keeps a cursor per anchor run
    and advances it with each insertion so creation order survives — found by
    the spike behind this module, see the design doc §6.1.
    """

    def __init__(self) -> None:
        self._cursor: dict[int, Any] = {}

    def add(self, document, anchor_run, text: str) -> int:
        part = _footnotes_part(document)
        root = part._element
        real_ids = [int(fn.get(qn("w:id"))) for fn in root.findall(qn("w:footnote"))
                   if int(fn.get(qn("w:id"))) > 0]
        footnote_id = max(real_ids, default=0) + 1

        footnote_xml = (
            f'<w:footnote xmlns:w="{_W}" w:id="{footnote_id}">'
            "<w:p><w:pPr><w:pStyle w:val=\"FootnoteText\"/></w:pPr>"
            f"<w:r><w:rPr>{_REFERENCE_PROPERTIES}</w:rPr>"
            "<w:footnoteRef/></w:r>"
            f'<w:r><w:t xml:space="preserve"> {escape(text)}</w:t></w:r>'
            "</w:p></w:footnote>"
        )
        root.append(parse_xml(footnote_xml.encode("utf-8")))

        ref_xml = (
            f'<w:r xmlns:w="{_W}">'
            f"<w:rPr>{_REFERENCE_PROPERTIES}</w:rPr>"
            f'<w:footnoteReference w:id="{footnote_id}"/></w:r>'
        )
        ref_run = parse_xml(ref_xml.encode("utf-8"))
        cursor = self._cursor.get(id(anchor_run), anchor_run._r)
        cursor.addnext(ref_run)
        self._cursor[id(anchor_run)] = ref_run
        return footnote_id
