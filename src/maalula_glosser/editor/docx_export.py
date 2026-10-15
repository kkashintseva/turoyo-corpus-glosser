"""Export an editor session to docx, JSON and CSV.

The docx mirrors `data/raw/glossed_all.docx`: the source sentence as a
paragraph, the translation under it, then the interlinear block as a two-row
table — morphemes above, gloss labels below — with every border explicitly set
to `none` and every width left on auto, which is what makes the columns line up
without a visible grid.

Примечания (`sentence["comments"]`) become Word footnotes anchored to the
cell they were made on (see `docx_footnotes.py` — python-docx has no public
footnote API). Discussion threads (`sentence["discussions"]`), being a
conversation rather than an editorial note, take over the role примечания
used to have: one real Word comment per thread, with every reply listed
inside its text.
"""
from __future__ import annotations

import csv
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.shared import Pt

from maalula_glosser.editor import arabic_marks as A
from maalula_glosser.editor import session as S
from maalula_glosser.editor import docx_footnotes as FN

# How many tokens share one table. The reference document breaks a long
# sentence into several narrow tables so each fits the page width.
TOKENS_PER_TABLE = 6

def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


BORDER_SIDES = ("top", "left", "bottom", "right", "insideH", "insideV")


def _strip_borders(table) -> None:
    """Set every border to `none`, the way the reference document does.

    A table style alone is not enough: the source file carries an explicit
    `w:tblBorders` block with `none` on all six sides, and without it Word
    falls back to the style's grid.
    """
    properties = table._tbl.tblPr
    for existing in properties.findall(qn("w:tblBorders")):
        properties.remove(existing)
    borders = properties.makeelement(qn("w:tblBorders"), {})
    for side in BORDER_SIDES:
        element = borders.makeelement(qn(f"w:{side}"), {})
        element.set(qn("w:val"), "none")
        element.set(qn("w:sz"), "0")
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), "auto")
        borders.append(element)
    properties.append(borders)


# The reference document is set in Times New Roman, the interlinear rows at
# 10pt. Both matter: the widths below are measured off that document, so they
# only hold for the font and size they were measured with. Writing them out in
# Calibri 11 — python-docx's default — put wider text into cells sized for
# narrower text, which is why a gloss still wrapped after the widths were fixed.
FONT = "Times New Roman"
BODY_PT = 12
GLOSS_PT = 10
NC_SUBCLASS_PT = 8

# Advance widths of Times New Roman, in ems, read out of the font itself. A
# single figure per character does not work: over the reference document's 5336
# columns the width actually needed ranges from 0.25 em per character to 0.64,
# so one constant is either too narrow for `mōn who` or a quarter too wide for
# everything else. Anything not listed falls back to a capital's width, which
# cannot be too narrow.
ADVANCE_EM = {
    0.0: "̣̥́",
    0.18: "'",
    0.25: ",.",
    0.278: "/:;ijltīḷṭṯ",
    0.333: "!()-I[]fr‘’",
    0.389: "Jsšṣ",
    0.408: '"',
    0.444: "?acezàáèéāčēžəʔʕẓ",
    0.5: "*0123456789bdghknopquvxyġōūḍḏḥḳ–",
    0.541: "~",
    0.556: "FPSŠṢ",
    0.564: "<=>",
    0.6: "‿∼",
    0.611: "ELTZŽṮ",
    0.667: "BCRČ",
    0.722: "ADGHKNOQUVXYwÁĠŌƏḎḤ",
    0.778: "&m",
    0.889: "M",
    0.944: "W",
    1.0: "—",
}
_WIDTH = {char: em for em, chars in ADVANCE_EM.items() for char in chars}
_UNKNOWN_EM = 0.722
# Word draws a small capital at about four fifths of the capital's width
SMALL_CAP = 0.8
# The gap between columns, set on the table rather than assumed: Word's default
# cell margin is 5.76pt a side, and a width computed against a guess at it is a
# width that can be wrong. Half of this is written into `w:tblCellMar`, so what
# is reserved here is what the cell actually takes, and the rest is slack.
CELL_MARGIN = Pt(4)
CELL_PADDING = Pt(12)


def _text_em(text: str, small_caps: bool = False) -> float:
    total = 0.0
    for char in nfc(text):
        if small_caps and char.islower():
            total += _WIDTH.get(char.upper(), _UNKNOWN_EM) * SMALL_CAP
        else:
            total += _WIDTH.get(char, _UNKNOWN_EM)
    return total


def _column_width(morphemes: str, gloss_pieces, size: int):
    """How much room this column needs for its longer row to stay on one line.

    Measured character by character, so a column of `m`s gets the room it needs
    and a column of `i`s does not get room it does not.
    """
    gloss = sum(_text_em(text, small_caps) for text, small_caps in gloss_pieces)
    return Pt(max(_text_em(morphemes), gloss, 1.0) * size) + CELL_PADDING


def _fixed_layout(table) -> None:
    """Use the widths given instead of redistributing them.

    With the layout left on autofit Word recomputes the columns to fill the
    page, which is what squeezed a long gloss until it wrapped despite noWrap.
    """
    properties = table._tbl.tblPr
    for existing in properties.findall(qn("w:tblLayout")):
        properties.remove(existing)
    layout = properties.makeelement(qn("w:tblLayout"), {qn("w:type"): "fixed"})
    properties.append(layout)


def _rows_that_fit(tokens: list, widths: list, usable, cap: int) -> Iterable[list]:
    """Break the block into tables no wider than the page.

    The reference document breaks a long sentence into several narrow tables;
    this decides where by measuring rather than by counting, so one long gloss
    ends a row early instead of pushing the whole line over the margin.
    """
    row: list[int] = []
    used = 0
    for index, width in enumerate(widths):
        if row and (used + width > usable or len(row) >= cap):
            yield [tokens[i] for i in row]
            row, used = [], 0
        row.append(index)
        used += width
    if row:
        yield [tokens[i] for i in row]


def _igt_rows(tokens: list, widths: list, usable, cap: int) -> Iterable[list]:
    """Break IGT at full stops as well as at the available page width."""
    start = 0
    for end, token in enumerate(tokens, start=1):
        if nfc(token.get("surface") or "").rstrip().endswith("."):
            yield from _rows_that_fit(tokens[start:end], widths[start:end],
                                      usable, cap)
            start = end
    if start < len(tokens):
        yield from _rows_that_fit(tokens[start:], widths[start:], usable, cap)


def _use_font(doc) -> None:
    """Set the document in the reference document's face, everywhere.

    On the style, not on every run, so headings and anything added later come
    out in it too — and on `w:eastAsia`/`w:cs` as well, or Word picks its own
    face for the characters it decides are not Latin, which in a transcription
    full of ḥ, ṯ and ʕ is a real risk.
    """
    for name in ("Normal", "Title", "Heading 1", "Heading 2"):
        try:
            style = doc.styles[name]
        except KeyError:
            continue
        style.font.name = FONT
        properties = style.element.get_or_add_rPr().get_or_add_rFonts()
        for attribute in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
            properties.set(qn(attribute), FONT)
    doc.styles["Normal"].font.size = Pt(BODY_PT)


def _cell_margins(table) -> None:
    """Pin the side margins, so the width reserved for them is the width used."""
    properties = table._tbl.tblPr
    for existing in properties.findall(qn("w:tblCellMar")):
        properties.remove(existing)
    margins = properties.makeelement(qn("w:tblCellMar"), {})
    for side, value in (("top", 0), ("left", CELL_MARGIN),
                        ("bottom", 0), ("right", CELL_MARGIN)):
        element = margins.makeelement(qn(f"w:{side}"), {})
        # `dxa` is twentieths of a point; `Pt` hands back EMU, and writing that
        # number out unconverted asks Word for a margin of about seventy inches
        element.set(qn("w:w"), str(value.twips if value else 0))
        element.set(qn("w:type"), "dxa")
        margins.append(element)
    properties.append(margins)


def _no_wrap(cell) -> None:
    """Forbid wrapping inside a cell, so a gloss never breaks across two lines.

    An interlinear column has to stay one visual unit: `bakery-FREE` splitting
    after the hyphen puts FREE under the next word and destroys the alignment
    the format exists for. With autofit on, the column widens instead.
    """
    properties = cell._tc.get_or_add_tcPr()
    for existing in properties.findall(qn("w:noWrap")):
        properties.remove(existing)
    properties.append(properties.makeelement(qn("w:noWrap"), {}))


def _cell_text(cell, pieces, *, bold: bool = False, size: int = GLOSS_PT):
    """Write one cell and hand back a run, so a comment can anchor to it.

    `pieces` is `(text, superscript)` as `arabic_marks` splits it, so the ᴬ…ᴬ
    brackets of an Arabic insert are raised the way the reference document
    raises them.
    """
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(0)
    if isinstance(pieces, str):
        pieces = [(pieces, False)]
    first = None
    for text, superscript in pieces or [("", False)]:
        run = paragraph.add_run(text)
        run.bold = bold
        run.font.size = Pt(size)
        run.font.superscript = superscript or None
        first = first or run
    _no_wrap(cell)
    return first if first is not None else paragraph.add_run("")


def _gloss_runs(gloss: str, is_label,
                is_full_agreement=None) -> list[tuple[str, bool]]:
    """Split a gloss line into `(text, small_caps)` pieces.

    Grammatical labels are set in small capitals and lexical material is not.
    Only labels that encode person, gender *and* number together are the
    exception: `3ms` stays ordinary lower case, while `MS`, `3M` and `1S`
    remain small capitals like other grammatical labels.
    """
    from maalula_glosser.editor.labels import AMBIGUOUS_WITH_LEXICAL

    pieces: list[tuple[str, bool]] = []
    for morpheme in re.split(r"([-=+])", nfc(gloss)):
        if not morpheme:
            continue
        if morpheme in "-=+":
            if pieces:
                pieces[-1] = (pieces[-1][0] + morpheme, pieces[-1][1])
            else:
                pieces.append((morpheme, False))
            continue
        parts = morpheme.split(".")
        # `do` sharing a morpheme with a category is the English verb, not the
        # direct-object marker, so it is lexical and stays in plain type
        dotted = len(parts) > 1
        for index, part in enumerate(parts):
            lexical_homograph = dotted and part.upper() in AMBIGUOUS_WITH_LEXICAL
            label = bool(part) and is_label(part) and not lexical_homograph
            plain_lowercase = (
                label
                and is_full_agreement
                and is_full_agreement(part)
            )
            text = (part.lower() if plain_lowercase else part) + (
                "." if index < len(parts) - 1 else "")
            if not text:
                continue
            pieces.append((text, label and not plain_lowercase))
    return pieces


def _cell_gloss(cell, gloss: str, is_label, is_full_agreement=None,
                size: int = GLOSS_PT):
    """Write a gloss row with the label-specific DOCX typography.

    Small caps only render lowercase letters as small capitals, so a label is
    lower-cased in the run. Full person/gender/number bundles stay plain lower
    case, and the subclass digit in NC1/NC2 is reduced to 8pt. The stored
    spelling is untouched; these are typesetting changes only.
    """
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(0)
    for text, small_caps in _gloss_runs(gloss, is_label, is_full_agreement):
        shown = text.lower() if small_caps else text
        nc_subclass = re.fullmatch(r"(nc)([12])([.=-]*)", shown)
        if small_caps and nc_subclass:
            nc_run = paragraph.add_run(nc_subclass.group(1))
            nc_run.font.size = Pt(size)
            nc_run.font.small_caps = True
            digit_run = paragraph.add_run(nc_subclass.group(2))
            digit_run.font.size = Pt(NC_SUBCLASS_PT)
            if nc_subclass.group(3):
                boundary_run = paragraph.add_run(nc_subclass.group(3))
                boundary_run.font.size = Pt(size)
            continue
        run = paragraph.add_run(shown)
        run.font.size = Pt(size)
        run.font.small_caps = small_caps or None
    _no_wrap(cell)
    return paragraph


def _cell_metadata(cell, text: str, size: int = 7):
    """Compact lexical-entry metadata row used only when a token has it."""
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(0)
    run = paragraph.add_run(text or "")
    run.font.size = Pt(size)
    run.italic = True
    _no_wrap(cell)
    return paragraph


def _token_columns(token: dict[str, Any]) -> tuple[str, str]:
    """One token as its two IGT lines: morphemes and labels, boundaries included."""
    return S.join_parts(token["morphemes"]), S.render_gloss(token)


def build_docx(document: dict[str, Any], target: Path,
               tokens_per_table: int = TOKENS_PER_TABLE,
               inventory=None) -> Path:
    # Which components are grammatical labels. Only a complete person + gender
    # + number bundle is kept in ordinary lower case by the DOCX convention.
    from maalula_glosser.editor.labels import _label_parts

    known = inventory.known() if inventory is not None else None
    full_agreement_re = re.compile(r"^[123][MF][SP]$", re.IGNORECASE)

    def is_label(component: str) -> bool:
        return bool(_label_parts(component, known))

    def is_full_agreement(component: str) -> bool:
        canonical = (inventory.canonical(component)
                     if inventory is not None else component)
        return bool(canonical and full_agreement_re.fullmatch(canonical))

    doc = Document()
    _use_font(doc)
    section = doc.sections[0]
    usable = section.page_width - section.left_margin - section.right_margin
    title = document.get("title") or "Glossed text"
    doc.add_heading(title, level=1)
    footnotes = FN.FootnoteAnchors()

    for sentence in document.get("sentences") or []:
        source = doc.add_paragraph()
        for text, superscript in A.runs(sentence.get("source_line") or ""):
            source_run = source.add_run(text)
            source_run.bold = True
            source_run.font.superscript = superscript or None

        translation = (sentence.get("translation") or "").strip()
        if translation:
            note = doc.add_paragraph()
            note_run = note.add_run(translation)
            note_run.italic = True

        # comment anchors, filled in as the cells are written
        anchors: dict[str, Any] = {}
        # A speaker tag belongs to the transcription, which is the paragraph
        # above; it is not a word of the language, so it gets no morpheme and no
        # gloss column.
        tokens = [t for t in (sentence.get("tokens") or []) if not S.is_speaker(t)]
        # an insert runs across columns, so the whole row decides which A is a
        # bracket and which is an ordinary capital
        columns = A.runs_per_word([_token_columns(t)[0] for t in tokens])
        widths = [_column_width(_token_columns(t)[0],
                                _gloss_runs(_token_columns(t)[1], is_label,
                                            is_full_agreement),
                                size=GLOSS_PT) for t in tokens]
        position = 0
        for group in _igt_rows(tokens, widths, usable, tokens_per_table):
            # DOCX export intentionally keeps the classic two-line IGT layout:
            # morphemes on the first row and glosses on the second. Exact
            # FieldWorks grammatical-info remains in the editor/lexicon and in
            # JSON/CSV exports, but is not rendered as a third DOCX row.
            table = doc.add_table(rows=2, cols=len(group))
            table.alignment = WD_TABLE_ALIGNMENT.LEFT
            table.autofit = False
            _fixed_layout(table)
            _cell_margins(table)
            _strip_borders(table)
            for index, token in enumerate(group):
                _morphemes, glosses = _token_columns(token)
                width = widths[position]
                table.columns[index].width = width
                run = _cell_text(table.rows[0].cells[index], columns[position])
                _cell_gloss(table.rows[1].cells[index], glosses, is_label,
                            is_full_agreement)
                for row in table.rows:
                    row.cells[index].width = width
                anchors[token["token_id"]] = run
                position += 1
            doc.add_paragraph()

        for comment in sentence.get("comments") or []:
            anchor = anchors.get((comment.get("anchor") or {}).get("token_id"))
            if anchor is None:
                # an orphaned comment must still reach the file rather than be
                # dropped silently
                orphan = doc.add_paragraph()
                orphan_run = orphan.add_run(comment.get("text") or "")
                orphan_run.font.size = Pt(8)
                continue
            footnotes.add(doc, anchor, comment.get("text") or "")

        for thread in sentence.get("discussions") or []:
            anchor = anchors.get((thread.get("anchor") or {}).get("token_id"))
            messages = thread.get("messages") or []
            if not messages:
                continue
            if anchor is None:
                # an orphaned thread must still reach the file rather than be
                # dropped silently — one line per reply, same as an orphaned
                # comment falls back to a plain paragraph
                orphan = doc.add_paragraph()
                orphan_run = orphan.add_run(
                    " / ".join(f"{m.get('author') or ''}: {m.get('text') or ''}"
                              for m in messages))
                orphan_run.font.size = Pt(8)
                continue
            comment = doc.add_comment(runs=[anchor], text="",
                                      author=messages[0].get("author") or "",
                                      initials="")
            for message in messages:
                comment.add_paragraph(
                    f"{message.get('author') or ''} ({message.get('created') or ''}): "
                    f"{message.get('text') or ''}")

    target.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(target))
    return target


def build_csv(document: dict[str, Any], target: Path) -> Path:
    """One row per morpheme — the flat view for spreadsheets and diffing."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sentence_id", "token_id", "surface", "morph_id",
                         "sep", "form", "gloss", "status", "origin",
                         "lemma", "lemma_id", "grammatical_info",
                         "lift_entry_id", "lift_sense_id",
                         "confidence", "comments", "discussions"])
        for sentence in document.get("sentences") or []:
            by_token: dict[str, list[str]] = {}
            for comment in sentence.get("comments") or []:
                key = (comment.get("anchor") or {}).get("token_id") or ""
                by_token.setdefault(key, []).append(comment.get("text") or "")
            discussions_by_token: dict[str, list[str]] = {}
            for thread in sentence.get("discussions") or []:
                key = (thread.get("anchor") or {}).get("token_id") or ""
                for message in thread.get("messages") or []:
                    discussions_by_token.setdefault(key, []).append(
                        f"{message.get('author') or ''}: {message.get('text') or ''}")
            for token in sentence.get("tokens") or []:
                notes = " | ".join(by_token.get(token["token_id"], []))
                discussion_notes = " | ".join(
                    discussions_by_token.get(token["token_id"], []))
                for morpheme in token.get("morphemes") or []:
                    writer.writerow([
                        sentence.get("sentence_id"), token["token_id"],
                        token.get("surface"), morpheme.get("morph_id"),
                        morpheme.get("sep", ""), morpheme.get("form", ""),
                        morpheme.get("gloss") or S.UNRESOLVED_MARK,
                        morpheme.get("status"), morpheme.get("origin"),
                        token.get("lemma") or "", token.get("lemma_id") or "",
                        token.get("grammatical_info") or "",
                        token.get("lift_entry_id") or "", token.get("lift_sense_id") or "",
                        token.get("confidence") or "", notes, discussion_notes,
                    ])
    return target


def build_json(document: dict[str, Any], target: Path) -> Path:
    """The session itself, so the export can be reopened in the editor."""
    S.validate(document)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, ensure_ascii=False, indent=1),
                      encoding="utf-8")
    return target


def export_all(document: dict[str, Any], directory: Path,
               stem: str | None = None, inventory=None) -> dict[str, Path]:
    stem = stem or (document.get("title") or document["document_id"])
    safe = "".join(c for c in stem if c.isalnum() or c in " _-").strip() or "gloss"
    return {
        "docx": build_docx(document, directory / f"{safe}.docx",
                           inventory=inventory),
        "json": build_json(document, directory / f"{safe}.json"),
        "csv": build_csv(document, directory / f"{safe}.csv"),
    }
