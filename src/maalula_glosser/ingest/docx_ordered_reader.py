from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator
from xml.etree import ElementTree as ET

from docx import Document
from docx.document import Document as _Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P


@dataclass
class DocxElement:
    index: int
    kind: str
    text: str
    style: str | None = None
    table_rows: list[list[str]] = field(default_factory=list)
    source: dict = field(default_factory=dict)


def iter_block_items(parent: _Document) -> Iterator[Paragraph | Table]:
    """Yield paragraphs and tables in document-body order."""
    parent_elm = parent.element.body
    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)


def read_ordered_docx(path: str | Path) -> list[DocxElement]:
    path = Path(path)
    document = Document(path)
    result: list[DocxElement] = []
    paragraph_count = 0
    table_count = 0
    for global_index, block in enumerate(iter_block_items(document)):
        if isinstance(block, Paragraph):
            paragraph_count += 1
            result.append(DocxElement(
                index=global_index,
                kind="paragraph",
                text=block.text,
                style=block.style.name if block.style else None,
                source={"paragraph": paragraph_count},
            ))
        else:
            table_count += 1
            rows = [[cell.text for cell in row.cells] for row in block.rows]
            text = "\n".join("\t".join(row) for row in rows)
            result.append(DocxElement(
                index=global_index,
                kind="table",
                text=text,
                table_rows=rows,
                source={"table": table_count},
            ))
    return result


def _extract_text_from_xml(xml_bytes: bytes) -> list[dict]:
    root = ET.fromstring(xml_bytes)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    items: list[dict] = []
    for node in root.findall(".//w:footnote", ns) + root.findall(".//w:endnote", ns) + root.findall(".//w:comment", ns):
        identifier = node.attrib.get(f"{{{ns['w']}}}id")
        texts = [element.text or "" for element in node.findall(".//w:t", ns)]
        items.append({"id": identifier, "text": "".join(texts)})
    return items


def extract_editorial_metadata(path: str | Path) -> dict[str, list[dict]]:
    """Extract footnotes/endnotes/comments as metadata, not running IGT text."""
    path = Path(path)
    output = {"footnotes": [], "endnotes": [], "comments": []}
    names = {
        "word/footnotes.xml": "footnotes",
        "word/endnotes.xml": "endnotes",
        "word/comments.xml": "comments",
    }
    with zipfile.ZipFile(path) as archive:
        for member, key in names.items():
            if member in archive.namelist():
                output[key] = _extract_text_from_xml(archive.read(member))
    return output
