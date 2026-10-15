"""Conservative Turoyo verb segmentation suggestions.

This module is deliberately separate from the exact FLEx corpus parser.
Imported FLEx/human analyses are authoritative and are never offered for
reanalysis.  For newly typed tokens, LOW/AUTO placeholders from the ordinary
glosser do not block this parser.  A rule never writes to a document by itself:
it returns candidates which the editor can show and a human can accept.

The first rule set follows ``instructions & etc.txt`` supplied with the
project: Infectum / intransitive-preterite endings, -wa past forms,
L-preterite person endings, selected object paradigms, imperative plural and
verbal preverbs.  It does *not* attempt noun/adjective morphology.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import re
import unicodedata
from typing import Any, Iterable, Iterator, Literal


BOUNDARIES = "-=+"

# Vocalism alternates heavily across the verbal paradigm.  For corpus-evidence
# matching, consonants remain exact and ordered while vowels may differ.  This
# is *not* a general fuzzy matcher: š != s, ḥ != h, b != v, and y/w are kept.
_ASCII_VOWELS = frozenset("aeiou")
_EXTRA_VOWELS = frozenset({"ə", "ǝ", "ә", "ı", "ɨ", "æ", "œ", "ø", "ɐ", "ɑ"})

_VERBAL_GI_MARKERS = (
    "infectum", "preterite", "imperative", "verb", "detransitive",
)
_VERBAL_GLOSS_MARKERS = (
    ".infc", ".pret", ".intr.pret", ".detr.infc", ".detr.pret", ".ipv",
)


def nfc(value: Any) -> str:
    return unicodedata.normalize("NFC", str(value or ""))


def _clusters(text: str) -> Iterator[str]:
    current = ""
    for char in unicodedata.normalize("NFD", nfc(text).casefold()):
        if unicodedata.combining(char):
            current += char
            continue
        if current:
            yield current
        current = char
    if current:
        yield current


def _is_vowel_cluster(cluster: str) -> bool:
    base = cluster[0].casefold()
    return base in _ASCII_VOWELS or base in _EXTRA_VOWELS


def consonant_skeleton(text: str) -> str:
    """Exact consonant sequence, ignoring only vowels and punctuation."""
    out: list[str] = []
    for cluster in _clusters(text):
        base = cluster[0]
        if _is_vowel_cluster(cluster):
            continue
        if unicodedata.category(base).startswith("L"):
            out.append(unicodedata.normalize("NFC", cluster))
    return "".join(out)


def _clean_form(text: str) -> str:
    return nfc(text).strip().strip(".,;:!?…()[]{}<>«»“”\"'‘’` ")


def _clean_morph_form(text: str) -> str:
    return _clean_form(text).strip(BOUNDARIES)


def _safe_gloss(text: str) -> str:
    """Return a one-cell lexical gloss, or empty if it is structurally unsafe."""
    value = nfc(text).strip().strip(BOUNDARIES)
    if not value or any(mark in value for mark in BOUNDARIES):
        return ""
    return value


def _looks_verbal_gloss(value: str) -> bool:
    low = nfc(value).casefold()
    return any(marker in low for marker in _VERBAL_GLOSS_MARKERS)


def _looks_verbal_gi(value: str) -> bool:
    low = nfc(value).casefold()
    return any(marker in low for marker in _VERBAL_GI_MARKERS)


@dataclass(frozen=True, slots=True)
class StemEvidence:
    stem: str
    gloss: str
    count: int = 1
    lemma: str | None = None
    lemma_id: str | None = None
    grammatical_info: str | None = None
    source: str = "flextext"

    @property
    def skeleton(self) -> str:
        return consonant_skeleton(self.stem)


@dataclass(frozen=True, slots=True)
class RulePiece:
    form: str
    gloss: str
    sep: str = "-"


@dataclass(frozen=True, slots=True)
class VerbRule:
    name: str
    label: str
    suffix: str
    pieces: tuple[RulePiece, ...]
    priority: int
    family: str


@dataclass(frozen=True, slots=True)
class VerbSuggestion:
    suggestion_id: str
    surface: str
    segmentation: str
    gloss: str
    rule: str
    rule_label: str
    family: str
    confidence: Literal["HIGH", "MEDIUM", "LOW"]
    score: int
    evidence_match: Literal["exact_stem", "same_consonants", "none"]
    evidence_stem: str | None
    evidence_gloss: str | None
    evidence_count: int
    evidence_lemma: str | None
    evidence_lemma_id: str | None
    evidence_grammatical_info: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "suggestion_id": self.suggestion_id,
            "surface": self.surface,
            "segmentation": self.segmentation,
            "gloss": self.gloss,
            "rule": self.rule,
            "rule_label": self.rule_label,
            "family": self.family,
            "confidence": self.confidence,
            "score": self.score,
            "evidence": None if self.evidence_match == "none" else {
                "match": self.evidence_match,
                "stem": self.evidence_stem,
                "gloss": self.evidence_gloss,
                "count": self.evidence_count,
                "lemma": self.evidence_lemma,
                "lemma_id": self.evidence_lemma_id,
                "grammatical_info": self.evidence_grammatical_info,
            },
        }


def _rule(
    name: str,
    label: str,
    suffix: str,
    pieces: Iterable[tuple[str, str] | tuple[str, str, str]],
    priority: int,
    family: str,
) -> VerbRule:
    normalized: list[RulePiece] = []
    for piece in pieces:
        if len(piece) == 2:
            form, gloss = piece
            sep = "-"
        else:
            form, gloss, sep = piece
        normalized.append(RulePiece(form, gloss, sep))
    return VerbRule(name, label, suffix, tuple(normalized), priority, family)


# Long/specific endings are tried before short endings.  The source document
# explicitly gives both -utu and -i-tu analyses for 2pl; both are candidates.
PAST_RULES: tuple[VerbRule, ...] = (
    _rule("past_f_1s", "F + PST + 1s", "owayno",
          (("o", "f"), ("way", "pst"), ("no", "1s")), 980, "past"),
    _rule("past_p_1p", "P + PST + 1p", "iwayna",
          (("i", "p"), ("way", "pst"), ("na", "1p")), 980, "past"),
    _rule("past_1s", "PST + 1s", "wayno",
          (("way", "pst"), ("no", "1s")), 970, "past"),
    _rule("past_2p", "2p + PST", "utwa",
          (("ut", "2p"), ("wa", "pst")), 960, "past"),
    _rule("past_f", "F + PST", "owa",
          (("o", "f"), ("wa", "pst")), 950, "past"),
    _rule("past_2s", "2s + PST", "atwa",
          (("at", "2s"), ("wa", "pst")), 950, "past"),
    _rule("past_p", "P + PST", "iwa",
          (("i", "p"), ("wa", "pst")), 950, "past"),
    _rule("past", "PST", "wa", (("wa", "pst"),), 900, "past"),
)

BASIC_RULES: tuple[VerbRule, ...] = (
    _rule("base_f_1s", "F + 1s", "ono", (("o", "f"), ("no", "1s")),
          780, "infectum/intransitive-preterite"),
    _rule("base_p_1p", "P + 1p", "ina", (("i", "p"), ("na", "1p")),
          780, "infectum/intransitive-preterite"),
    _rule("base_p_2", "P + 2", "itu", (("i", "p"), ("tu", "2")),
          775, "infectum/intransitive-preterite"),
    _rule("base_2p", "2p", "utu", (("utu", "2p"),),
          770, "infectum/intransitive-preterite"),
    _rule("base_2s", "2s", "at", (("at", "2s"),),
          730, "infectum/intransitive-preterite"),
    _rule("base_1s", "1s", "no", (("no", "1s"),),
          730, "infectum/intransitive-preterite"),
    _rule("base_f", "F", "o", (("o", "f"),),
          690, "infectum/intransitive-preterite"),
    _rule("base_p", "P", "i", (("i", "p"),),
          690, "infectum/intransitive-preterite"),
    # III-y examples in the supplied paradigm have ḥoz-ən / bax-ən.
    _rule("base_p_en", "P (-ən allomorph)", "ən", (("ən", "p"),),
          650, "infectum/intransitive-preterite"),
    _rule("base_p_cyr_en", "P (-әn allomorph)", "әn", (("әn", "p"),),
          650, "infectum/intransitive-preterite"),
)

L_PRET_RULES: tuple[VerbRule, ...] = (
    _rule("lpret_2ms", "L-preterite A.2ms", "lux", (("lux", "a.2ms"),),
          845, "l-preterite"),
    _rule("lpret_2fs", "L-preterite A.2fs", "lax", (("lax", "a.2fs"),),
          845, "l-preterite"),
    _rule("lpret_1p", "L-preterite A.1p", "lan", (("lan", "a.1p"),),
          845, "l-preterite"),
    _rule("lpret_3ms", "L-preterite A.3ms", "le", (("le", "a.3ms"),),
          830, "l-preterite"),
    _rule("lpret_3fs", "L-preterite A.3fs", "la", (("la", "a.3fs"),),
          830, "l-preterite"),
    _rule("lpret_1s", "L-preterite A.1s", "li", (("li", "a.1s"),),
          830, "l-preterite"),
    _rule("lpret_2p", "L-preterite A.2p", "xu", (("xu", "a.2p"),),
          815, "l-preterite"),
    # 3p is class-dependent in the supplied tables.  Keep the attested
    # allomorphs as lower-priority candidates; without corpus evidence they
    # remain explicitly LOW rather than masquerading as a secure analysis.
    _rule("lpret_3p_he", "L-preterite A.3p (-ḥe)", "ḥe", (("ḥe", "a.3p"),),
          760, "l-preterite"),
    _rule("lpret_3p_me", "L-preterite A.3p (-me)", "me", (("me", "a.3p"),),
          760, "l-preterite"),
    _rule("lpret_3p_lle", "L-preterite A.3p (-lle)", "lle", (("lle", "a.3p"),),
          750, "l-preterite"),
)

L_PRET_PAST_RULES: tuple[VerbRule, ...] = (
    _rule("lpret_past_2ms", "L-preterite PST + A.2ms", "waylux",
          (("way", "pst"), ("lux", "a.2ms")), 1010, "l-preterite-past"),
    _rule("lpret_past_2fs", "L-preterite PST + A.2fs", "waylax",
          (("way", "pst"), ("lax", "a.2fs")), 1010, "l-preterite-past"),
    _rule("lpret_past_1p", "L-preterite PST + A.1p", "waylan",
          (("way", "pst"), ("lan", "a.1p")), 1010, "l-preterite-past"),
    _rule("lpret_past_3ms", "L-preterite PST + A.3ms", "wayle",
          (("way", "pst"), ("le", "a.3ms")), 1000, "l-preterite-past"),
    _rule("lpret_past_3fs", "L-preterite PST + A.3fs", "wayla",
          (("way", "pst"), ("la", "a.3fs")), 1000, "l-preterite-past"),
    _rule("lpret_past_1s", "L-preterite PST + A.1s", "wayli",
          (("way", "pst"), ("li", "a.1s")), 1000, "l-preterite-past"),
    # The 3p/2p forms in the source are internally inconsistent (wal-xu / wa-lle),
    # so they are intentionally not generalized here until confirmed.
)

INFECTUM_OBJECT_RULES: tuple[VerbRule, ...] = (
    _rule("infc_f_1s_o2ms", "F + 1s + O.2ms", "allux",
          (("a", "f"), ("l", "1s"), ("lux", "o.2ms")), 1180, "infectum-object"),
    _rule("infc_f_1s_o2fs", "F + 1s + O.2fs", "allax",
          (("a", "f"), ("l", "1s"), ("lax", "o.2fs")), 1180, "infectum-object"),
    _rule("infc_f_1s_o3ms", "F + 1s + O.3ms", "alle",
          (("a", "f"), ("l", "1s"), ("le", "o.3ms")), 1175, "infectum-object"),
    _rule("infc_f_1s_o3fs", "F + 1s + O.3fs", "alla",
          (("a", "f"), ("l", "1s"), ("la", "o.3fs")), 1175, "infectum-object"),
    _rule("infc_f_1s_o3p", "F + 1s + O.3p", "anne",
          (("a", "f"), ("n", "1s"), ("ne", "o.3p")), 1170, "infectum-object"),
    _rule("infc_f_1s_o2p", "F + 1s + O.2p", "anxu",
          (("a", "f"), ("n", "1s"), ("xu", "o.2p")), 1170, "infectum-object"),
    _rule("infc_1s_o3p", "1s + O.3p", "nanne",
          (("na", "1s"), ("nne", "o.3p")), 1160, "infectum-object"),
    _rule("infc_1s_o2p", "1s + O.2p", "nanxu",
          (("na", "1s"), ("nxu", "o.2p")), 1160, "infectum-object"),
    _rule("infc_1s_o2ms", "1s + O.2ms", "nux",
          (("n", "1s"), ("ux", "o.2ms")), 1140, "infectum-object"),
    _rule("infc_1s_o2fs", "1s + O.2fs", "nax",
          (("n", "1s"), ("ax", "o.2fs")), 1140, "infectum-object"),
    _rule("infc_1s_o3ms", "1s + O.3ms", "ne",
          (("n", "1s"), ("e", "o.3ms")), 1110, "infectum-object"),
    _rule("infc_1s_o3fs", "1s + O.3fs", "na",
          (("n", "1s"), ("a", "o.3fs")), 1110, "infectum-object"),
    _rule("infc_f_o3p", "F + O.3p", "alle",
          (("a", "f"), ("lle", "o.3p")), 1090, "infectum-object"),
    _rule("infc_f_o2p", "F + O.2p", "alxu",
          (("a", "f"), ("lxu", "o.2p")), 1090, "infectum-object"),
    _rule("infc_p_o3p", "P + O.3p", "ənne",
          (("ən", "p"), ("ne", "o.3p")), 1090, "infectum-object"),
    _rule("infc_p_o2p", "P + O.2p", "ənxu",
          (("ən", "p"), ("xu", "o.2p")), 1090, "infectum-object"),
    _rule("infc_p_o3p_cyr", "P + O.3p", "әnne",
          (("әn", "p"), ("ne", "o.3p")), 1090, "infectum-object"),
    _rule("infc_p_o2p_cyr", "P + O.2p", "әnxu",
          (("әn", "p"), ("xu", "o.2p")), 1090, "infectum-object"),
    _rule("infc_o3p", "O.3p", "še", (("še", "o.3p"),),
          1030, "infectum-object"),
    _rule("infc_o2p", "O.2p", "xu", (("xu", "o.2p"),),
          1010, "infectum-object"),
)

# Explicit L-preterite object patterns supplied in the paradigm.  This is not
# expanded to unattested person/object combinations yet.
L_PRET_OBJECT_RULES: tuple[VerbRule, ...] = (
    _rule("lpret_a3fs_o2ms", "A.3fs + O.2ms", "lalux",
          (("la", "a.3fs"), ("lux", "o.2ms")), 1230, "l-preterite-object"),
    _rule("lpret_a3ms_o3ms", "A.3ms + O.3ms", "lele",
          (("le", "a.3ms"), ("le", "o.3ms")), 1220, "l-preterite-object"),
    # Feminine object forms.
    _rule("lpret_f_a2p", "F + A.2p", "alxu",
          (("a", "f"), ("lxu", "a.2p")), 1200, "l-preterite-object"),
    _rule("lpret_f_a3p", "F + A.3p", "alle",
          (("a", "f"), ("lle", "a.3p")), 1200, "l-preterite-object"),
    _rule("lpret_f_a2ms", "F + A.2ms", "olux",
          (("o", "f"), ("lux", "a.2ms")), 1170, "l-preterite-object"),
    _rule("lpret_f_a2fs", "F + A.2fs", "olax",
          (("o", "f"), ("lax", "a.2fs")), 1170, "l-preterite-object"),
    _rule("lpret_f_a1p", "F + A.1p", "olan",
          (("o", "f"), ("lan", "a.1p")), 1170, "l-preterite-object"),
    _rule("lpret_f_a3ms", "F + A.3ms", "ole",
          (("o", "f"), ("le", "a.3ms")), 1160, "l-preterite-object"),
    _rule("lpret_f_a3fs", "F + A.3fs", "ola",
          (("o", "f"), ("la", "a.3fs")), 1160, "l-preterite-object"),
    # Plural object forms.
    _rule("lpret_p_a2p", "P + A.2p", "ənxu",
          (("ə", "p"), ("nxu", "a.2p")), 1200, "l-preterite-object"),
    _rule("lpret_p_a3p", "P + A.3p", "ənne",
          (("ə", "p"), ("nne", "a.3p")), 1200, "l-preterite-object"),
    _rule("lpret_p_a2ms", "P + A.2ms", "ilux",
          (("i", "p"), ("lux", "a.2ms")), 1170, "l-preterite-object"),
    _rule("lpret_p_a2fs", "P + A.2fs", "ilax",
          (("i", "p"), ("lax", "a.2fs")), 1170, "l-preterite-object"),
    _rule("lpret_p_a1p", "P + A.1p", "ilan",
          (("i", "p"), ("lan", "a.1p")), 1170, "l-preterite-object"),
    _rule("lpret_p_a3ms", "P + A.3ms", "ile",
          (("i", "p"), ("le", "a.3ms")), 1160, "l-preterite-object"),
    _rule("lpret_p_a3fs", "P + A.3fs", "ila",
          (("i", "p"), ("la", "a.3fs")), 1160, "l-preterite-object"),
)

IMPERATIVE_RULES: tuple[VerbRule, ...] = (
    _rule("imperative_plural", "Imperative plural", "u", (("u", "pl"),),
          610, "imperative"),
)

CORE_RULES: tuple[VerbRule, ...] = tuple(sorted(
    (*L_PRET_OBJECT_RULES, *INFECTUM_OBJECT_RULES, *L_PRET_PAST_RULES,
     *PAST_RULES, *L_PRET_RULES, *BASIC_RULES, *IMPERATIVE_RULES),
    key=lambda rule: (-len(rule.suffix), -rule.priority, rule.name),
))

# Preverbs from the supplied instructions. They form an independent outer
# morphological layer. The parser may therefore suggest a preverb boundary
# even when the verbal host itself cannot yet be segmented. Such proposals
# remain LOW/suggestion-only until accepted by the user.
PREVERBS: tuple[tuple[str, str], ...] = tuple(sorted((
    ("kalko", "prst"),
    ("ṭro", "deo"),
    ("gəd", "prs/fut"),
    ("gәd", "prs/fut"),
    ("kə", "prs/fut"),
    ("kә", "prs/fut"),
    ("ko", "prs/fut"),
    ("g", "prs/fut"),
    ("k", "prs/fut"),
), key=lambda row: -len(row[0])))



# ---------------------------------------------------------------------------
# Possessive suffixes and enclitic demonstratives
# ---------------------------------------------------------------------------
#
# These are suggestion-only rules. They do not require the host to be present
# in the lexicon: the whole point is to help with previously unanalysed forms.
#
# Longest matching suffix wins, so e.g. bayt-ayḏe is not additionally offered
# as baytayḏ-e.

POSSESSION_AND_DEM_RULES: tuple[VerbRule, ...] = tuple(sorted((

    # Alienable possession
    _rule(
        "alienable_3ms",
        "Alienable possession 3ms",
        "ayḏe",
        (("ayḏe", "al.3ms"),),
        940,
        "alienable-possession",
    ),
    _rule(
        "alienable_3fs",
        "Alienable possession 3fs",
        "ayḏa",
        (("ayḏa", "al.3fs"),),
        940,
        "alienable-possession",
    ),
    _rule(
        "alienable_2ms",
        "Alienable possession 2ms",
        "ayḏux",
        (("ayḏux", "al.2ms"),),
        950,
        "alienable-possession",
    ),
    _rule(
        "alienable_2fs",
        "Alienable possession 2fs",
        "ayḏax",
        (("ayḏax", "al.2fs"),),
        950,
        "alienable-possession",
    ),
    _rule(
        "alienable_1s",
        "Alienable possession 1s",
        "ayḏi",
        (("ayḏi", "al.1s"),),
        940,
        "alienable-possession",
    ),
    _rule(
        "alienable_3p",
        "Alienable possession 3p",
        "aṯṯe",
        (("aṯṯe", "al.3p"),),
        940,
        "alienable-possession",
    ),
    _rule(
        "alienable_2p",
        "Alienable possession 2p",
        "aṯxu",
        (("aṯxu", "al.2p"),),
        940,
        "alienable-possession",
    ),
    _rule(
        "alienable_1p",
        "Alienable possession 1p",
        "ayḏan",
        (("ayḏan", "al.1p"),),
        950,
        "alienable-possession",
    ),

    # Postclitic =ste
    _rule(
        "postclitic_top_ste",
        "Top postclitic",
        "ste",
        (("ste", "top", "="),),
        960,
        "postclitic",
    ),

    # Enclitic demonstratives: = is significant and must be preserved.
    _rule(
        "dem_prox_ms",
        "Proximate demonstrative ms",
        "ano",
        (("ano", "prox.m.s", "="),),
        910,
        "demonstrative",
    ),
    _rule(
        "dem_dist_ms",
        "Distal demonstrative ms",
        "awo",
        (("awo", "dist.m.s", "="),),
        910,
        "demonstrative",
    ),
    _rule(
        "dem_prox_fs",
        "Proximate demonstrative fs",
        "aṯe",
        (("aṯe", "prox.f.s", "="),),
        910,
        "demonstrative",
    ),
    _rule(
        "dem_dist_fs",
        "Distal demonstrative fs",
        "ayo",
        (("ayo", "dist.f.s", "="),),
        910,
        "demonstrative",
    ),
    _rule(
        "dem_prox_p",
        "Proximate demonstrative p",
        "ani",
        (("ani", "prox.p", "="),),
        910,
        "demonstrative",
    ),
    _rule(
        "dem_dist_p",
        "Distal demonstrative p",
        "anәk",
        (("anәk", "dist.p", "="),),
        920,
        "demonstrative",
    ),

    # Inalienable possession
    _rule(
        "inalienable_3p",
        "Inalienable possession 3p",
        "ayye",
        (("ayye", "inal.3p"),),
        900,
        "inalienable-possession",
    ),
    _rule(
        "inalienable_2p",
        "Inalienable possession 2p",
        "ayxu",
        (("ayxu", "inal.2p"),),
        900,
        "inalienable-possession",
    ),
    _rule(
        "inalienable_2ms",
        "Inalienable possession 2ms",
        "ux",
        (("ux", "inal.2ms"),),
        850,
        "inalienable-possession",
    ),
    _rule(
        "inalienable_2fs",
        "Inalienable possession 2fs",
        "ax",
        (("ax", "inal.2fs"),),
        850,
        "inalienable-possession",
    ),
    _rule(
        "inalienable_1p",
        "Inalienable possession 1p",
        "an",
        (("an", "inal.1p"),),
        830,
        "inalienable-possession",
    ),
_rule(
        "inalienable_3fs",
        "Inalienable possession 3fs",
        "a",
        (("a", "inal.3fs"),),
        800,
        "inalienable-possession",
    ),
    _rule(
        "inalienable_1s",
        "Inalienable possession 1s",
        "i",
        (("i", "inal.1s"),),
        800,
        "inalienable-possession",
    ),

), key=lambda rule: (-len(rule.suffix), -rule.priority, rule.name)))


def token_parser_eligibility(token: dict[str, Any]) -> tuple[bool, str]:
    """Server-side guard for the suggestion-only verb parser.

    The crucial distinction is provenance, not whether the ordinary glosser
    has already put an AUTO placeholder/suggestion into the cell.  Everything
    that comes from an imported FLEx word is source data and is therefore
    completely out of scope for this parser.  Human edits are equally
    authoritative.  A newly typed token, however, may carry LOW/AUTO cells
    created by the normal editor pipeline; those cells must *not* make the
    token ineligible, otherwise the verb parser never gets a chance to help.
    """
    # FLEx provenance by itself is NOT an analysis.  An imported word for which
    # FLEx supplied no morphology has ``flextext`` metadata but
    # ``original_analysis is None`` and must remain eligible: this is precisely
    # where the parser is useful.  What is protected is existing source
    # analysis, not the mere fact that the spelling came from FLEx.
    if token.get("original_analysis"):
        return False, "У слова уже есть исходный анализ FLEx; парсер его не трогает."
    if token.get("imported_igt"):
        return False, "У слова уже есть импортированный анализ; парсер его не трогает."
    if token.get("locked"):
        return False, "Разбор уже был подтверждён или изменён вручную."

    cells = token.get("morphemes") or []
    if any((cell.get("origin") == "user") for cell in cells):
        return False, "Разбор уже был изменён вручную."

    # A committed, non-LOW automatic analysis is already doing useful work.
    # LOW/AUTO guesses and `suggested` values are explicitly allowed: they are
    # not human/source analyses and the verb parser only offers alternatives.
    confidence = nfc(token.get("confidence")).upper()
    committed_auto = any(
        nfc(cell.get("gloss")).strip() and cell.get("origin") != "user"
        for cell in cells
    )
    if committed_auto and confidence in {"HIGH", "MEDIUM"}:
        return False, "У нового слова уже есть уверенный автоматический анализ."

    return True, ""


class VerbEvidenceIndex:
    """Read-only index of verbal stems attested in the current FLEx corpus."""

    def __init__(self, evidence: Iterable[StemEvidence]):
        merged: dict[tuple[str, str, str | None], StemEvidence] = {}
        for row in evidence:
            stem = _clean_morph_form(row.stem)
            gloss = _safe_gloss(row.gloss)
            if not stem or not gloss:
                continue
            key = (stem.casefold(), gloss.casefold(), row.lemma_id)
            old = merged.get(key)
            if old is None:
                merged[key] = StemEvidence(
                    stem=stem,
                    gloss=gloss,
                    count=max(1, int(row.count or 1)),
                    lemma=row.lemma,
                    lemma_id=row.lemma_id,
                    grammatical_info=row.grammatical_info,
                    source=row.source,
                )
            else:
                merged[key] = StemEvidence(
                    stem=old.stem,
                    gloss=old.gloss,
                    count=old.count + max(1, int(row.count or 1)),
                    lemma=old.lemma or row.lemma,
                    lemma_id=old.lemma_id or row.lemma_id,
                    grammatical_info=old.grammatical_info or row.grammatical_info,
                    source=old.source,
                )

        self.rows = tuple(merged.values())
        self.exact: dict[str, list[StemEvidence]] = defaultdict(list)
        self.by_skeleton: dict[str, list[StemEvidence]] = defaultdict(list)
        for row in self.rows:
            self.exact[row.stem.casefold()].append(row)
            skeleton = row.skeleton
            if skeleton:
                self.by_skeleton[skeleton].append(row)

    @classmethod
    def from_analyzer(cls, analyzer: Any) -> "VerbEvidenceIndex":
        evidence: list[StemEvidence] = []
        seen_rows: set[int] = set()

        for rows in getattr(analyzer, "by_key", {}).values():
            for row in rows or []:
                # The same row normally lives in one surface bucket, but keep
                # this robust if a future index aliases it under several keys.
                identity = id(row)
                if identity in seen_rows:
                    continue
                seen_rows.add(identity)

                lemma_id = row.get("lemma_id")
                gi = nfc(row.get("grammatical_info")) or None
                pos = nfc(row.get("pos"))
                glosses = [nfc(row.get("gloss_clean") or row.get("gloss")),
                           nfc(row.get("lexical_gloss"))]
                morphs = row.get("morphemes") or []
                glosses.extend(nfc(m.get("gloss")) for m in morphs)
                is_verbal = bool(
                    (lemma_id and getattr(analyzer, "is_verbal", lambda _x: False)(lemma_id))
                    or pos.casefold() in {"v", "vb", "verb"}
                    or "verb" in pos.casefold()
                    or _looks_verbal_gi(gi or "")
                    or any(_looks_verbal_gloss(value) for value in glosses if value)
                )
                if not is_verbal:
                    continue

                lexical_morph = next((m for m in morphs
                                      if nfc(m.get("type")).casefold() in {"stem", "root"}
                                      and _clean_morph_form(m.get("form") or m.get("lemma"))), None)
                lexical_morph = lexical_morph or next(
                    (m for m in morphs if _clean_morph_form(m.get("form") or m.get("lemma"))),
                    None)
                stem = _clean_morph_form((lexical_morph or {}).get("form")
                                         or (lexical_morph or {}).get("lemma"))
                base_gloss = _safe_gloss(row.get("lexical_gloss"))
                if not base_gloss:
                    base_gloss = _safe_gloss((lexical_morph or {}).get("gloss"))
                if not base_gloss:
                    # If the whole gloss has one lexical head followed by bound
                    # labels, use only that first cell; never copy a whole
                    # already-segmented gloss as a lexical morpheme.
                    whole = nfc(row.get("gloss_clean") or row.get("gloss"))
                    first = re.split(r"[-=+]", whole, maxsplit=1)[0]
                    base_gloss = _safe_gloss(first)
                if stem and base_gloss:
                    evidence.append(StemEvidence(
                        stem=stem,
                        gloss=base_gloss,
                        count=max(1, int(row.get("count") or 1)),
                        lemma=nfc(row.get("lemma")) or None,
                        lemma_id=lemma_id,
                        grammatical_info=gi,
                    ))

        # Analyzer-level stem lists can contribute a spelling even if a
        # particular source row lacked POS.  They are used only for lexemes the
        # analyzer already independently knows are verbal, and only when a
        # lexical meaning can be resolved safely.
        for lemma_id in getattr(analyzer, "verbal_lexeme_ids", set()) or set():
            record = getattr(analyzer, "lexeme_record", lambda _x: {})(lemma_id) or {}
            gloss = ""
            lexical = getattr(analyzer, "lexical", None)
            if lexical is not None:
                try:
                    gloss, _status = lexical.resolve(lemma_id, "")
                except Exception:
                    gloss = ""
            if not gloss:
                senses = record.get("senses") or []
                if len(senses) == 1:
                    gloss_rows = senses[0].get("glosses") or []
                    if gloss_rows:
                        gloss = gloss_rows[0].get("text") or ""
            gloss = _safe_gloss(gloss)
            if not gloss:
                continue
            for stem in getattr(analyzer, "stems_by_lexeme", {}).get(lemma_id, ()):
                if _clean_morph_form(stem):
                    evidence.append(StemEvidence(
                        stem=_clean_morph_form(stem), gloss=gloss,
                        lemma=record.get("lemma_canonical"), lemma_id=lemma_id,
                        grammatical_info=(record.get("grammatical_infos") or [None])[0],
                        source="flextext",
                    ))

        return cls(evidence)

    def matches(self, stem: str, *, limit: int = 12) -> list[tuple[StemEvidence, str, int]]:
        stem = _clean_morph_form(stem)
        if not stem:
            return []
        out: list[tuple[StemEvidence, str, int]] = []
        for row in self.exact.get(stem.casefold(), ()):
            out.append((row, "exact_stem", 130))

        skeleton = consonant_skeleton(stem)
        if len(skeleton) >= 2:
            exact_ids = {(row.stem.casefold(), row.gloss.casefold(), row.lemma_id)
                         for row, _kind, _bonus in out}
            for row in self.by_skeleton.get(skeleton, ()):
                key = (row.stem.casefold(), row.gloss.casefold(), row.lemma_id)
                if key in exact_ids:
                    continue
                out.append((row, "same_consonants", 85))

        out.sort(key=lambda item: (-item[2], -item[0].count,
                                   len(item[0].stem), item[0].stem))
        return out[:limit]


class TuroyoVerbParser:
    """Generate, but never automatically apply, Turoyo verb analyses."""

    def __init__(self, analyzer: Any = None, *, evidence: Iterable[StemEvidence] | None = None):
        self.analyzer = analyzer
        self.evidence = (VerbEvidenceIndex(evidence)
                         if evidence is not None else
                         VerbEvidenceIndex.from_analyzer(analyzer))

    @staticmethod
    def eligible(token: dict[str, Any]) -> tuple[bool, str]:
        return token_parser_eligibility(token)

    @staticmethod
    def _join(stem: str, pieces: Iterable[RulePiece]) -> tuple[str, str]:
        segmentation = stem
        gloss = "{BASE}"
        for piece in pieces:
            segmentation += f"{piece.sep}{piece.form}"
            gloss += f"{piece.sep}{piece.gloss}"
        return segmentation, gloss

    @staticmethod
    def _id(surface: str, segmentation: str, gloss: str, rule: str,
            evidence: StemEvidence | None) -> str:
        raw = "\x1f".join((surface, segmentation, gloss, rule,
                            evidence.stem if evidence else "",
                            evidence.gloss if evidence else "",
                            (evidence.lemma_id or "") if evidence else ""))
        return "vsg_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    def _core_candidates(self, surface: str) -> list[VerbSuggestion]:
        surface = _clean_form(surface)
        if not surface or any(mark in surface for mark in BOUNDARIES):
            # Existing explicit segmentation is a human/source statement; this
            # parser does not reinterpret it.
            return []

        suggestions: list[VerbSuggestion] = []
        for rule in CORE_RULES:
            if len(surface) <= len(rule.suffix) or not surface.casefold().endswith(rule.suffix.casefold()):
                continue
            stem = surface[:-len(rule.suffix)]
            if not stem:
                continue
            segmentation, gloss_template = self._join(stem, rule.pieces)
            evidence_rows = self.evidence.matches(stem)
            for evidence, match_kind, evidence_bonus in evidence_rows:
                gloss = gloss_template.replace("{BASE}", evidence.gloss, 1)
                score = rule.priority + evidence_bonus + min(30, evidence.count)
                confidence: Literal["HIGH", "MEDIUM", "LOW"] = (
                    "HIGH" if match_kind == "exact_stem" else "MEDIUM")
                suggestions.append(VerbSuggestion(
                    suggestion_id=self._id(surface, segmentation, gloss, rule.name, evidence),
                    surface=surface,
                    segmentation=segmentation,
                    gloss=gloss,
                    rule=rule.name,
                    rule_label=rule.label,
                    family=rule.family,
                    confidence=confidence,
                    score=score,
                    evidence_match=match_kind,  # type: ignore[arg-type]
                    evidence_stem=evidence.stem,
                    evidence_gloss=evidence.gloss,
                    evidence_count=evidence.count,
                    evidence_lemma=evidence.lemma,
                    evidence_lemma_id=evidence.lemma_id,
                    evidence_grammatical_info=evidence.grammatical_info,
                ))

            # Corpus evidence is a ranking/interpretation aid, not a gate.  A
            # genuinely new verb may have no attested stem in FLEx yet.  In that
            # case expose the rule-based segmentation with an unknown lexical
            # gloss instead of silently returning nothing.  Require at least a
            # two-consonant stem to suppress the noisiest one-letter accidents.
            if not evidence_rows and len(consonant_skeleton(stem)) >= 2:
                gloss = gloss_template.replace("{BASE}", "?", 1)
                suggestions.append(VerbSuggestion(
                    suggestion_id=self._id(surface, segmentation, gloss, rule.name, None),
                    surface=surface,
                    segmentation=segmentation,
                    gloss=gloss,
                    rule=rule.name,
                    rule_label=rule.label,
                    family=rule.family,
                    confidence="LOW",
                    score=rule.priority - 180,
                    evidence_match="none",
                    evidence_stem=None,
                    evidence_gloss=None,
                    evidence_count=0,
                    evidence_lemma=None,
                    evidence_lemma_id=None,
                    evidence_grammatical_info=None,
                ))
        return suggestions


    def _possession_and_dem_candidates(
        self,
        surface: str,
        *,
        anaphoric: bool = False,
    ) -> list[VerbSuggestion]:
        """Possessive suffixes and enclitic demonstratives.

        `anaphoric=True` means the noun is immediately followed by d=...
        In that environment only -e/-a change from inal.3ms/fs to
        anp.3ms/fs, exactly as specified in the project rules.
        """
        surface = _clean_form(surface)

        if not surface or any(mark in surface for mark in BOUNDARIES):
            return []

        matching = [
            rule
            for rule in POSSESSION_AND_DEM_RULES
            if len(surface) > len(rule.suffix)
            and surface.casefold().endswith(rule.suffix.casefold())
        ]

        if not matching:
            return []

        # Do not analyse bayt-ayḏe additionally as baytayḏ-e,
        # abbot=ani as abbotan-i, etc.
        longest = max(len(rule.suffix) for rule in matching)
        matching = [
            rule for rule in matching
            if len(rule.suffix) == longest
        ]

        out: list[VerbSuggestion] = []

        for rule in matching:
            stem = surface[:-len(rule.suffix)]

            if not stem:
                continue

            # Short -a/-e/-i/-an are extremely common string endings.
            # Require a plausible lexical host rather than splitting e.g.
            # a one-letter remainder.
            minimum_consonants = (
                2
                if rule.name in {
                    "inalienable_3ms",
                    "inalienable_3fs",
                    "inalienable_1s",
                    "inalienable_1p",
                }
                else 1
            )

            if len(consonant_skeleton(stem)) < minimum_consonants:
                continue

            pieces = rule.pieces

            if anaphoric and rule.name == "inalienable_3ms":
                pieces = (RulePiece("e", "anp.3ms", "-"),)

            elif anaphoric and rule.name == "inalienable_3fs":
                pieces = (RulePiece("a", "anp.3fs", "-"),)

            segmentation, gloss_template = self._join(stem, pieces)
            gloss = gloss_template.replace("{BASE}", "?", 1)

            rule_name = rule.name
            rule_label = rule.label
            family = rule.family

            if anaphoric and rule.name == "inalienable_3ms":
                rule_name = "anaphoric_3ms"
                rule_label = "Anaphoric suffix 3ms before d="
                family = "anaphoric"

            elif anaphoric and rule.name == "inalienable_3fs":
                rule_name = "anaphoric_3fs"
                rule_label = "Anaphoric suffix 3fs before d="
                family = "anaphoric"

            out.append(
                VerbSuggestion(
                    suggestion_id=self._id(
                        surface,
                        segmentation,
                        gloss,
                        rule_name,
                        None,
                    ),
                    surface=surface,
                    segmentation=segmentation,
                    gloss=gloss,
                    rule=rule_name,
                    rule_label=rule_label,
                    family=family,
                    confidence="LOW",
                    score=rule.priority,
                    evidence_match="none",
                    evidence_stem=None,
                    evidence_gloss=None,
                    evidence_count=0,
                    evidence_lemma=None,
                    evidence_lemma_id=None,
                    evidence_grammatical_info=None,
                )
            )

        return out

    def _with_preverbs(self, surface: str) -> list[VerbSuggestion]:
        """Separate a verbal preverb even when the host is not yet analysable.

        The supplied Turoyo rules treat these as genuine clitic boundaries:

            k=/kə=/ko=   prs/fut
            g=/gəd=      prs/fut
            ṭro=         deo
            kalko=       prst

        Earlier versions required the material after the preverb to receive a
        complete CORE_RULES analysis first.  That made forms such as
        kəmʕawaḷan invisible to the parser.  Here preverb recognition is an
        independent first layer.

        Only the longest matching preverb is used. Thus kə- is preferred over
        the formally possible shorter k-, and gəd- over g-.
        """
        surface = _clean_form(surface)

        if not surface or any(mark in surface for mark in BOUNDARIES):
            return []

        matching = [
            (prefix, prefix_gloss)
            for prefix, prefix_gloss in PREVERBS
            if len(surface) > len(prefix)
            and surface.casefold().startswith(prefix.casefold())
        ]

        if not matching:
            return []

        # Critical for kə=/k=, ko=/k=, gəd=/g= etc.
        longest = max(len(prefix) for prefix, _gloss in matching)
        matching = [
            (prefix, gloss)
            for prefix, gloss in matching
            if len(prefix) == longest
        ]

        out: list[VerbSuggestion] = []

        for prefix, prefix_gloss in matching:
            host = surface[len(prefix):]

            # First try to analyse the host morphologically.
            host_candidates = self._core_candidates(host)

            if host_candidates:
                for candidate in host_candidates:
                    segmentation = f"{prefix}={candidate.segmentation}"
                    gloss = f"{prefix_gloss}={candidate.gloss}"

                    evidence = (
                        StemEvidence(
                            stem=candidate.evidence_stem or "",
                            gloss=candidate.evidence_gloss or "",
                            count=candidate.evidence_count,
                            lemma=candidate.evidence_lemma,
                            lemma_id=candidate.evidence_lemma_id,
                            grammatical_info=candidate.evidence_grammatical_info,
                        )
                        if candidate.evidence_match != "none"
                        else None
                    )

                    out.append(
                        VerbSuggestion(
                            suggestion_id=self._id(
                                surface,
                                segmentation,
                                gloss,
                                f"preverb:{prefix}:{candidate.rule}",
                                evidence,
                            ),
                            surface=surface,
                            segmentation=segmentation,
                            gloss=gloss,
                            rule=f"preverb:{prefix}:{candidate.rule}",
                            rule_label=(
                                f"{prefix}= ({prefix_gloss}) + "
                                f"{candidate.rule_label}"
                            ),
                            family=f"preverb+{candidate.family}",
                            confidence=candidate.confidence,
                            score=candidate.score + 90 + len(prefix),
                            evidence_match=candidate.evidence_match,
                            evidence_stem=candidate.evidence_stem,
                            evidence_gloss=candidate.evidence_gloss,
                            evidence_count=candidate.evidence_count,
                            evidence_lemma=candidate.evidence_lemma,
                            evidence_lemma_id=candidate.evidence_lemma_id,
                            evidence_grammatical_info=(
                                candidate.evidence_grammatical_info
                            ),
                        )
                    )

            else:
                # The preverb itself is still a useful and linguistically
                # motivated segmentation even if the host morphology is not
                # known yet.  This remains LOW and suggestion-only.
                segmentation = f"{prefix}={host}"
                gloss = f"{prefix_gloss}=?"

                out.append(
                    VerbSuggestion(
                        suggestion_id=self._id(
                            surface,
                            segmentation,
                            gloss,
                            f"preverb:{prefix}:host_unknown",
                            None,
                        ),
                        surface=surface,
                        segmentation=segmentation,
                        gloss=gloss,
                        rule=f"preverb:{prefix}:host_unknown",
                        rule_label=(
                            f"{prefix}= ({prefix_gloss}); "
                            "основа пока не разобрана"
                        ),
                        family="preverb",
                        confidence="LOW",
                        score=620 + len(prefix),
                        evidence_match="none",
                        evidence_stem=None,
                        evidence_gloss=None,
                        evidence_count=0,
                        evidence_lemma=None,
                        evidence_lemma_id=None,
                        evidence_grammatical_info=None,
                    )
                )

        return out


    # === PRODUCTIVE ENCLITIC DEMONSTRATIVES ===
    def _demonstrative_candidates(
        self,
        surface: str,
    ) -> list[VerbSuggestion]:
        """Productive enclitic demonstratives.

        X=ano   prox.m.s
        X=awo   dist.m.s
        X=aṯe   prox.f.s
        X=ayo   dist.f.s
        X=ani   prox.p
        X=anәk  dist.p

        This layer is independent of the verbal paradigms and does not
        require FLEx evidence. Existing explicit segmentation is never
        reanalysed.
        """
        surface = _clean_form(surface)

        if not surface:
            return []

        # Existing + and - boundaries belong to the host and must be preserved.
        # Example: ba+dḳoṭani -> ba+dḳoṭ=ani.
        # Only an existing = means that the enclitic has already been segmented.
        if "=" in surface:
            return []

        rules = (
            ("anәk", "dist.p",   "dem_dist_p",  "Distal demonstrative p"),
            ("anək", "dist.p",   "dem_dist_p",  "Distal demonstrative p"),
            ("ano",  "prox.m.s", "dem_prox_ms", "Proximate demonstrative ms"),
            ("awo",  "dist.m.s", "dem_dist_ms", "Distal demonstrative ms"),
            ("aṯe",  "prox.f.s", "dem_prox_fs", "Proximate demonstrative fs"),
            ("ayo",  "dist.f.s", "dem_dist_fs", "Distal demonstrative fs"),
            ("ani",  "prox.p",   "dem_prox_p",  "Proximate demonstrative p"),
        )

        out = []

        for suffix, gloss_value, rule_name, rule_label in rules:
            if not surface.casefold().endswith(suffix.casefold()):
                continue

            if len(surface) <= len(suffix):
                continue

            host = surface[:-len(suffix)]

            # Prevent standalone demonstratives such as hano/hawo/hani
            # from becoming h=ano, h=awo, h=ani.
            if len(consonant_skeleton(host)) < 2:
                continue

            segmentation = f"{host}={surface[-len(suffix):]}"
            gloss = f"?={gloss_value}"

            out.append(
                VerbSuggestion(
                    suggestion_id=self._id(
                        surface,
                        segmentation,
                        gloss,
                        rule_name,
                        None,
                    ),
                    surface=surface,
                    segmentation=segmentation,
                    gloss=gloss,
                    rule=rule_name,
                    rule_label=rule_label,
                    family="demonstrative",
                    confidence="LOW",

                    # High enough to outrank accidental short suffix analyses
                    # such as final -i.
                    score=1100,

                    evidence_match="none",
                    evidence_stem=None,
                    evidence_gloss=None,
                    evidence_count=0,
                    evidence_lemma=None,
                    evidence_lemma_id=None,
                    evidence_grammatical_info=None,
                )
            )

        return out

    def suggest(
        self,
        surface: str,
        *,
        limit: int = 12,
        anaphoric: bool = False,
    ) -> list[VerbSuggestion]:
        surface = _clean_form(surface)
        candidates = (
            self._core_candidates(surface)
            + self._with_preverbs(surface)
            + self._possession_and_dem_candidates(
                surface,
                anaphoric=anaphoric,
            )
        )

        candidates += self._demonstrative_candidates(surface)

        # Several evidence rows can yield the same linguistic proposal. Keep
        # the strongest one but do not collapse genuinely different analyses of
        # the same surface (e.g. gurš-a-l-le vs gurš-a-lle).
        best: dict[tuple[str, str, str], VerbSuggestion] = {}
        for row in candidates:
            key = (row.segmentation, row.gloss, row.rule)
            previous = best.get(key)
            if previous is None or row.score > previous.score:
                best[key] = row

        rows = sorted(best.values(), key=lambda row: (
            -row.score,
            {"exact_stem": 0, "same_consonants": 1, "none": 2}.get(row.evidence_match, 3),
            -len(row.segmentation),
            row.segmentation,
            row.gloss,
        ))
        # FINAL-O-NOT-FILTER
        result = rows[:max(0, limit)]

        # A bare final -o is overwhelmingly nominal/adjectival in the material.
        # Do not offer the verbal feminine -f interpretation merely because
        # the surface happens to end in o.
        #
        # This does NOT suppress internal -o- in forms such as nəšq-o-wa.
        if surface.casefold().endswith("o"):
            result = [
                row for row in result
                if not (
                    row.segmentation.casefold().endswith("-o")
                    and row.gloss.casefold().endswith("-f")
                )
            ]

        return result

    def suggestions_for_token(self, token: dict[str, Any], *, limit: int = 12) -> dict[str, Any]:
        eligible, reason = self.eligible(token)
        if not eligible:
            return {"eligible": False, "reason": reason, "suggestions": []}
        surface = nfc(token.get("surface") or token.get("normalized") or "").strip()
        rows = self.suggest(surface, limit=limit)
        return {
            "eligible": True,
            "reason": "" if rows else "Для этой формы ни одно из заданных глагольных правил не подошло.",
            "suggestions": [row.as_dict() for row in rows],
        }
