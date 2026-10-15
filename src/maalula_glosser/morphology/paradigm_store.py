"""Exact paradigm store: the normative cells indexed for lookup and round-trip.

This is the "exact irregular / normative cell" tier. Every surface a normative
table lists resolves here to the exact (lexeme, features, gloss) it was recorded
under — no generation, no guessing. An exact cell always outranks a generic
class downstream.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Optional


def _nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value or "")


def strip_optional(surface: str) -> str:
    """`ni-ḥḥ(i)` also occurs as `ni-ḥḥi` and `ni-ḥḥ`; index all realisations."""
    return surface.replace("(", "").replace(")", "")


def optional_realisations(surface: str) -> list[str]:
    """Expand a single `(x)` optional segment into with/without forms."""
    if "(" not in surface:
        return [surface]
    with_seg = surface.replace("(", "").replace(")", "")
    without = surface[:surface.index("(")] + surface[surface.index(")") + 1:]
    out = [with_seg]
    if without != with_seg:
        out.append(without)
    return out


_PARAGRAPH = re.compile(r"\n\s*\n")
# a line that ends in a slash goes on in the next one: battay=n/ batt=e
_LINE = re.compile(r"(?<!/)\n")


def _forms(text: str) -> list[str]:
    """The forms one table line lists: `ḥā/iḥḥa`, with a line break as a
    soft break of the same slash list (source_ingest.expand_slash_variants)."""
    return [part.strip() for part in text.replace("\n", "/").split("/")
            if part.strip()]


def split_paired_cell(cell: dict) -> list[dict]:
    """A table cell that lists two forms with a gloss each is two cells.

    The tables of ṯōle, zalle, kʕōle and batte put the plain form and the form
    with its subject suffix into one cell, the glosses below each other:
    `n-ōṯy-in` / `ni-ṯy-illaḥ` over `1-come.prs-mp` / `1-come.prs-1mp`. The
    ingest keeps the cell verbatim, so both forms carried both glosses, line
    break included (ni-ṯy-illaḥ `1-come.NC-mp 1-come.NC-1mp`), and the second
    form took the features of the first. Paired by order here: the n-th gloss
    belongs to the n-th form. A cell with one gloss for several forms
    (`ḥā/iḥḥa`) is left as it is.
    """
    gloss = cell.get("gloss_original") or ""
    if "\n" not in gloss:
        return [cell]
    glosses = [part.strip().rstrip("/").strip()
               for part in _PARAGRAPH.split(gloss) if part.strip()]
    raw = _nfc(cell.get("raw_surface") or "")
    groups = [part for part in _PARAGRAPH.split(raw) if part.strip()]
    if len(groups) != len(glosses):
        groups = [part for part in _LINE.split(raw) if part.strip()]
    if len(glosses) < 2 or len(groups) != len(glosses):
        return [cell]
    return [{**cell, "cell_id": f"{cell['cell_id']}#{index}",
             "raw_surface": group.strip(), "surface_variants": _forms(group),
             "gloss_original": one, "split_from": cell["cell_id"]}
            for index, (group, one) in enumerate(zip(groups, glosses), 1)]


class ParadigmStore:
    def __init__(self, cells_path: Path):
        self.cells: list[dict] = [
            part for line in cells_path.read_text().splitlines() if line.strip()
            for part in split_paired_cell(json.loads(line))]
        # surface (clitics/hyphens removed) -> cells
        self.by_surface: dict[str, list[dict]] = defaultdict(list)
        # lexeme -> cells
        self.by_lexeme: dict[str, list[dict]] = defaultdict(list)
        for cell in self.cells:
            lex = cell.get("lexeme_heading")
            if lex:
                self.by_lexeme[_nfc(lex)].append(cell)
            for variant in cell.get("surface_variants", []):
                for realisation in optional_realisations(_nfc(variant)):
                    key = self._surface_key(realisation)
                    self.by_surface[key].append(cell)
                    # also index the segmented form directly
                    self.by_surface[_nfc(realisation)].append(cell)

    @staticmethod
    def _surface_key(segmented: str) -> str:
        """The attested token: drop internal morpheme boundaries, keep letters."""
        return _nfc(segmented).replace("-", "").replace("=", "")

    def lookup_surface(self, surface: str) -> list[dict]:
        """Exact cells whose surface (ignoring boundaries) equals this token."""
        key = self._surface_key(_nfc(surface))
        seen, out = set(), []
        for cell in self.by_surface.get(key, []):
            if cell["cell_id"] not in seen:
                seen.add(cell["cell_id"])
                out.append(cell)
        return out

    def generate(self, lexeme: str, tam: Optional[str] = None,
                 person: Optional[int] = None, gender: Optional[str] = None,
                 number: Optional[str] = None) -> list[dict]:
        """Exact surfaces for a lexeme filtered by whichever features are given."""
        out = []
        for cell in self.by_lexeme.get(_nfc(lexeme), []):
            f = cell["features"]
            if tam and f.get("tam") != tam and tam not in (f.get("tam_alt") or []):
                continue
            if person and f.get("person") != person:
                continue
            if gender and f.get("gender") != gender:
                continue
            if number and f.get("number") != number:
                continue
            out.append(cell)
        return out

    def __len__(self) -> int:
        return len(self.cells)


def default_store(root: Path) -> ParadigmStore:
    return ParadigmStore(root / "data" / "grammar_kb" / "morphology_v4"
                         / "paradigm_cells.jsonl")
