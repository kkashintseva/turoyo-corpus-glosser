"""The author's corrections to the trusted corpus analyses.

`data/final/analyses.jsonl` and the publication overlay are built artifacts
and are never edited by hand. A reading the author has reviewed and found
wrong -- a typo in the gloss, the label of another clitic, a boundary that
does not match its gloss, a row attached to the wrong lexeme -- is recorded in
`data/corrections_v2/gold_analysis_corrections.jsonl`, one decision per row,
and applied wherever those analyses are read: the A1 tier, the analysis
prior, the meaning a lexeme takes from its corpus analyses.

A row, by its `action`:

``replace`` (the default)
    `surface_canonical`, `segmentation`, `gloss` name the analysis exactly as
    it is stored (`gloss` is its `gloss_clean`); `analysis_id` narrows the row
    to one analysis where several are stored alike. `segmentation_corrected`,
    `gloss_corrected`, `lemma_id_corrected` say what it becomes.
``drop``
    The named analysis is no reading of the word.
``add``
    A reading the corpus lacks: `surface_canonical`, `segmentation`, `gloss`,
    optionally `lemma_id`. With ``"prefer": true`` it leads.
``prefer``
    The named analysis leads the other readings of its word.

Every row carries `basis` (who decided and when) and `source` (the table the
analysis came from). Glosses are written as the author writes them (SC, NC,
PC); the display convention leaves them as they are.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Iterable, Iterator

from maalula_glosser.gloss_shape import has_lexical_part
from maalula_glosser.io_utils import iter_jsonl
from maalula_glosser.normalization.gloss import lexical_do

CORRECTIONS = "data/corrections_v2/gold_analysis_corrections.jsonl"

# Above the publication overlay (authority_rank 100): the author's ruling on
# which reading of a word comes first is newer than either source.
PREFERRED_RANK = 200

_BOUNDARY = re.compile(r"[-=+]")


def _nfc(value: str | None) -> str:
    return unicodedata.normalize("NFC", value or "")


def rebuild_morphemes(segmentation: str, gloss: str,
                      previous: Iterable[dict] = ()) -> list[dict]:
    """The morpheme list of an analysis whose line 2 or line 3 was corrected.

    The cells keep the types they had when their number is unchanged (a
    relabelled clitic is still a clitic). Otherwise the cell with the meaning
    is the stem -- past a proclitic, whose gloss is a word too (m=yōm=l
    from=day=HD) -- what `=` joins to it is a clitic, the rest are affixes.
    """
    previous = list(previous)
    forms = _BOUNDARY.split(segmentation or "")
    glosses = _BOUNDARY.split(gloss or "")
    if len(forms) != len(glosses):
        return previous
    if len(previous) == len(forms):
        types = [morph.get("type") for morph in previous]
    else:
        joints = _BOUNDARY.findall(segmentation)
        lexical = [index for index, cell in enumerate(glosses)
                   if has_lexical_part(cell)]
        stem = next((index for index in lexical
                     if not (index < lexical[-1] and joints[index] in {"=", "+"})),
                    lexical[0] if lexical else 0)
        types = []
        for index in range(len(forms)):
            if index == stem:
                types.append("stem")
            elif index < stem:
                types.append("clitic" if joints[index] in {"=", "+"} else "prefix")
            else:
                types.append("clitic" if joints[index - 1] in {"=", "+"} else "suffix")
    return [{"form": form, "gloss": cell, "type": kind}
            for form, cell, kind in zip(forms, glosses, types)]


def with_lexical_do(analysis: dict) -> dict:
    """`DO.NC-MP` on a form of išw is the verb `do`: the import read the
    author's `do.nc-mp` as the label DO (direct object), which never carries
    a conjugation label of its own."""
    gloss = analysis.get("gloss_clean") or ""
    fixed = lexical_do(gloss)
    if fixed == gloss:
        return analysis
    return {**analysis, "gloss_clean": fixed,
            "morphemes": [{**morph, "gloss": lexical_do(morph.get("gloss") or "")}
                          for morph in analysis.get("morphemes") or []]}


class GoldCorrections:
    """The corrections file, indexed for the three places that read analyses."""

    def __init__(self, rows: Iterable[dict] = ()):
        self.by_key: dict[tuple[str, str, str], dict] = {}
        self.by_id: dict[str, dict] = {}
        self.dropped: set[tuple[str, str, str]] = set()
        self.preferred: set[tuple[str, str, str]] = set()
        self.additions: list[dict] = []
        for row in rows:
            action = row.get("action") or "replace"
            surface = _nfc(row.get("surface_canonical"))
            key = (surface, row.get("segmentation") or "", row.get("gloss") or "")
            if not surface or not key[1] or not key[2]:
                continue
            if action == "add":
                self.additions.append(row)
                if row.get("prefer"):
                    self.preferred.add(key)
            elif action == "drop":
                self.dropped.add(key)
            elif action == "prefer":
                self.preferred.add(key)
            elif row.get("analysis_id"):
                self.by_id[row["analysis_id"]] = row
            else:
                self.by_key[key] = row

    @classmethod
    def load(cls, root: Path | str) -> "GoldCorrections":
        return cls(iter_jsonl(Path(root) / CORRECTIONS))

    def __bool__(self) -> bool:
        return bool(self.by_key or self.by_id or self.dropped
                    or self.preferred or self.additions)

    def correct(self, analysis: dict) -> dict | None:
        """The analysis as the author corrected it; the same object when no
        row names it; ``None`` when it was dropped."""
        surface = _nfc(analysis.get("surface_canonical"))
        key = (surface, analysis.get("segmentation") or "",
               analysis.get("gloss_clean") or "")
        if key in self.dropped:
            return None
        row = self.by_id.get(analysis.get("analysis_id") or "") or self.by_key.get(key)
        if row and (row.get("segmentation"), row.get("gloss")) != key[1:]:
            row = None                 # an id whose analysis has since changed
        if row:
            segmentation = row.get("segmentation_corrected") or key[1]
            gloss = row.get("gloss_corrected") or key[2]
            analysis = {
                **analysis,
                "segmentation": segmentation, "gloss_clean": gloss,
                "lemma_id": row.get("lemma_id_corrected") or analysis.get("lemma_id"),
                "morphemes": rebuild_morphemes(segmentation, gloss,
                                               analysis.get("morphemes") or ()),
                "correction_id": row.get("correction_id"),
            }
            key = (surface, segmentation, gloss)
        if key in self.preferred:
            analysis = {**analysis, "authority_rank": PREFERRED_RANK}
        return analysis

    def correct_reading(self, surface: str, segmentation: str,
                        gloss: str) -> tuple[str, str] | None:
        """A token's (segmentation, gloss) as corrected, for sources that keep
        no analysis id (the training examples behind the analysis prior);
        ``None`` when the reading was dropped."""
        key = (_nfc(surface), segmentation, gloss)
        if key in self.dropped:
            return None
        row = self.by_key.get(key)
        if not row:
            return segmentation, gloss
        return (row.get("segmentation_corrected") or segmentation,
                row.get("gloss_corrected") or gloss)

    def corrected(self, analyses: Iterable[dict]) -> Iterator[dict]:
        """Stored analyses as they are to be read: corrected, without the
        dropped ones, with the verb `do` in its own letters."""
        for analysis in analyses:
            analysis = self.correct(analysis)
            if analysis is not None:
                yield with_lexical_do(analysis)

    def added(self) -> Iterator[dict]:
        """The readings the corpus lacks, shaped like its confirmed analyses."""
        for row in self.additions:
            surface = _nfc(row["surface_canonical"])
            analysis = {
                "analysis_id": f"gold_{row.get('correction_id')}",
                "status": "confirmed",
                "surface_canonical": surface,
                "segmentation": row["segmentation"],
                "gloss_clean": row["gloss"],
                "lemma_id": row.get("lemma_id"),
                "morphemes": rebuild_morphemes(row["segmentation"], row["gloss"]),
                "source": {"file": CORRECTIONS,
                           "correction_id": row.get("correction_id")},
                "correction_id": row.get("correction_id"),
            }
            if (surface, row["segmentation"], row["gloss"]) in self.preferred:
                analysis["authority_rank"] = PREFERRED_RANK
            yield analysis
