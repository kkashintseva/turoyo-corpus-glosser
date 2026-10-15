"""Part of speech from a lexeme's citation form and its own confirmed
analyses; and, separately, whether a *gloss string* carries a finite-verb
conjugation/aspect label (HANDOFF_verbal_lexemes.md, D1/D5).

Shared by compositional.py and editor/senses.py (no editor import here:
that would close the compositional -> editor -> compositional cycle).
"""
from __future__ import annotations

import re
import unicodedata

from maalula_glosser.normalization.transcription import fold_allophones

# Labels only a noun phrase head carries. One of them on the lexeme's own
# analyses outweighs a stray NC/SC label (mxarm-a vineyard.NC-F beside
# mxarm-a vineyard-FREE): a verb never takes FREE, CST or HD.
HARD_NOMINAL = frozenset({"FREE", "CST", "HD"})

# Conjugation/aspect labels that mean "this cell is a finite verb reading"
# for the purpose of D5 lexeme binding. PST/PRF/IPV are deliberately not
# included: D5 (HANDOFF_verbal_lexemes.md) names exactly NC/NC1/NC2/PC/SC.
VERBAL_TRIGGER = frozenset({"NC", "NC1", "NC2", "PC", "SC"})


def _nfc(value: str | None) -> str:
    return unicodedata.normalize("NFC", value or "")


def citation_is_verbal(lemma: str | None) -> bool:
    """Candidate A (D1): a citation form shaped like a verb's principal
    parts -- `iqtal yiqtul`, `aff/affi yaff/yaffi`, `ammen yʔammen` -- where
    a later part (after `/`-alternates are split out) starts with `y`.
    A single-word citation (`imeṯ`, `aḥref`, `ḥassel`) is not verbal by
    shape alone; those go through the curated list instead (D1).
    """
    parts = _nfc(lemma).replace("/", " ").split()
    return len(parts) >= 2 and any(part.startswith("y") for part in parts[1:])


def pos_from_labels(labels, verbal, nominal, feminine, masculine):
    """One part-of-speech vote from the gloss labels seen on a lexeme's own
    confirmed analyses. A verb never carries FREE/CST/HD (only a noun
    phrase head does), so those labels veto an otherwise-verbal reading.
    """
    upper = {label.upper() for label in labels}
    if labels & verbal and not upper & HARD_NOMINAL:
        return "verb"
    if labels & feminine and labels & masculine:
        return "adjective"
    if labels & nominal:
        return "noun"
    return None


def verbal_trigger(display_map) -> frozenset:
    """VERBAL_TRIGGER plus its other spellings from gloss_inventory's
    `display_map` (PRS/PRES -> NC, SBJV/SUBJ/INFC -> PC, "NC 1"/"NC 2" ->
    NC1/NC2): the same labels under another spelling, not a wider rule
    (plan section 4, decision 4).
    """
    trigger = set(VERBAL_TRIGGER)
    for alias, target in (display_map or {}).items():
        canonical = _nfc(str(target)).upper().replace(" ", "")
        if canonical in VERBAL_TRIGGER:
            trigger.add(_nfc(str(alias)))
    return frozenset(trigger)


_GLOSS_CELL_SPLIT = re.compile(r"[-=]")


def gloss_is_verbal(gloss: str | None, trigger: frozenset = VERBAL_TRIGGER) -> bool:
    """True if any cell of `gloss` (split on `-`/`=`, then on `.`) carries a
    label from `trigger`. Works across attached clitics: both
    `u=sabek.NC=3MP` and `qwl=lun.SC` are verbal, independent of how many
    proclitics or enclitics surround the lexical cell.
    """
    if not gloss:
        return False
    for cell in _GLOSS_CELL_SPLIT.split(gloss):
        for label in cell.split("."):
            if label in trigger:
                return True
    return False


# ---------------------------------------------------------------------------
# derived stem ("порода") from shape: which of a root's verbs a stem is.
# Used by the analyzer's D5 rebinding and by the editor's «Леммы…» pass.
# ---------------------------------------------------------------------------
_VOWELS = set("aeiouāēīōūəAEIOUĀĒĪŌŪ")


def _geminate(text: str) -> bool:
    return any(x == y and x not in _VOWELS for x, y in zip(text, text[1:]))


def citation_class(citation: str, root: str) -> str:
    """G / D / L / C / N / Gt / tD / tL / St from a citation form."""
    words = _nfc(citation).split()
    perf = (words[0] if words else "").split("/")[0].strip("()")
    impf = next((w.split("/")[0] for w in words[1:] if w.startswith("y")), "")
    c1 = root[:1]
    if perf.startswith(("sč", "isč")):
        return "St"
    if perf.startswith("č") and c1 != "č":
        rest = perf[1:]
        return "tL" if rest[1:2] == "ō" else "tD" if _geminate(rest) else "tD"
    if len(perf) > 2 and perf[0] == "i" and perf[2] == "č" and root[1:2] != "č":
        return "Gt"                                               # iġčar, ifčham
    if perf.startswith(("inə", "in")) and c1 != "n":
        return "N"
    if perf.startswith("a") and impf.startswith("ya") and c1 not in ("ʔ", ""):
        return "C"
    body = perf.lstrip("i")
    if len(body) > 1 and body[1] == "ō" and body[0] not in _VOWELS:
        return "L"
    if _geminate(body):
        return "D"
    return "G"


def stem_classes(stem: str, root: str, gloss: str) -> set[str] | None:
    """What the token's own stem says about its derived stem, or None when it
    says nothing decisive. Deliberately few, unambiguous cues:
    m-a-C1… (maḥcyin, mafəhmin) -> C; č-C… -> tD/tL (čġayyar, məčrōžaʕ);
    C1-č-… -> Gt (ščaġl); (m)inə- -> N; C1-ō-… without m- in NC -> G
    participle (zōrʕin, xōlkin)."""
    s = _nfc(fold_allophones(stem or "")).lower()
    c1 = root[:1]
    participle = s.startswith("m") and c1 != "m"
    if participle:
        s = s[1:].lstrip("əi")
    s = s.lstrip("ə")
    if s.startswith("č") and c1 != "č":
        rest = s[1:]
        if rest[1:2] == "ō":
            return {"tL"}
        return {"tD"} if _geminate(rest) else {"tD", "tL", "St"}
    if len(s) > 2 and s[0] == c1 and s[1] == "č" and root[1:2] != "č":
        return {"Gt"}
    if s.startswith(("inə", "nə", "in")) and c1 != "n" and len(s) > 3:
        return {"N"}
    if s.startswith("a" + c1) and c1 and c1 != "ʔ":
        return {"C"}
    if "NC2" in gloss:
        return None                         # CaCCīC is the G stem's second participle
    if len(s) > 1 and s[0] == c1 and s[1:2] == "ō":
        return {"L", "tL"} if participle else ({"G"} if "NC" in gloss else None)
    return None


