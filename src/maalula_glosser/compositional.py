"""Minimally working compositional morphological analyzer (V2).

Practical goal: correctly gloss most tokens of running text, do very well on
frequent forms, never present a doubtful analysis as reliable, and route the
rest to a review queue.

Evidence tiers (highest first):

    A1  human-trusted corpus analysis (data/final/analyses.jsonl)
    A0  the project's own glossing standard, i.e. the closed lexical classes of
        Glossing_updated.docx (data/grammar_kb/closed_class_v2.jsonl). Ranked
        below A1 on purpose: it supplies readings the corpus never recorded,
        but a human annotation of running text still wins.
    A2  explicit Maalula form in Arnold (data/dictionary_morphology_v1)
    A3  exact paradigm cell for the same lexeme/root/class (grammar KB)
    B1  compositional reconstruction, same lexeme
    B2  same-root / same-class analogy
    C   cross-lexeme template only (never auto-confirmable)

Confidence levels: HIGH (one strong, fully reconstructing analysis — safe to
apply automatically), MEDIUM (probable, weak ambiguity — show as a suggestion
with alternatives), LOW (weak analogy or several equal candidates — leave
unresolved for review).

Stem-internal TAM morphology (vowel melody, gemination, weak-root alternation)
stays in analysis metadata and is never split into an artificial IGT segment.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

import json as _json

from maalula_glosser.gold_corrections import GoldCorrections, with_lexical_do
from maalula_glosser.io_utils import iter_jsonl
from maalula_glosser.reusable import (
    morphology_readings, morphology_surfaces, surface_key,
)
from maalula_glosser.normalization.gloss import lexical_do, render_authoritative_gloss
from maalula_glosser.morphology.arnold_tables import normalise_gloss as _label_case
from maalula_glosser.morphology.morphophonology import devoice_b_after_consonant
from maalula_glosser.morphology.perfect_subclass import (
    classify_prf_analysis, is_unambiguous_strong_stem_i, replace_prf_label,
)
from maalula_glosser.normalization.transcription import (
    TranscriptionNormalizer,
    fold_allophones,
)
from maalula_glosser.gloss_shape import (
    has_lexical_part, is_grammatical_label, lexical_head as _lexical_head)
from maalula_glosser.pos import (
    citation_class, citation_is_verbal, gloss_is_verbal, pos_from_labels, stem_classes,
)


def load_rules() -> dict:
    """Linguistic rules, kept in data/grammar_kb/rules_v3.json with their sources.

    They live in data rather than in this module so the author can read and
    change them without touching code, and so every rule carries the evidence
    it rests on.
    """
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        path = parent / "data" / "grammar_kb" / "rules_v3.json"
        if path.exists():
            return _json.loads(path.read_text(encoding="utf-8"))
    return {}


RULES = load_rules()


# Authoritative TAM map from a paradigm-table conjugation header to the tense
# gloss. NC is the present/nominal conjugation (PRS); the perfect subclasses
# NC1/NC2 gloss PRF and never reach this table (their cells carry PRF directly).
# Kept identical to normalization.gloss.DEFAULT_GLOSS_ALIASES and
# grammar_kb.EXPONENT_TABLE_CATEGORIES; tests/unit/test_tam_consistency.py
# pins all three together.
CAT_TO_TAM = {"SC": "PST", "NC": "PRS", "PC": "SBJV", "IPV": "IPV"}
# Preposition senses a proclitic (b=, ʕa=, l=, m=) can realise. Used to rank the
# readings of a fused proclitic+pronoun by the morpheme's overall distribution
# rather than the tiny per-fused-form token count.
PREP_SENSES = {"loc", "in", "ins", "with", "on", "for", "to", "into", "by",
               "about", "at", "from"}
# a finite-verb TAM tag in a gloss (display sc/nc/pc or internal PST/PRS/SBJV)
_HAS_TAM = re.compile(r"\.(SC|NC|PC|PRF|IPV|NC1|NC2|PST|PRS|SBJV)\b", re.I)

# Lexical predicates whose final -a is not the nominal FREE exponent.  Keep
# this deliberately tiny and explicit: `be.loc` is attested with pronominal
# suffixes (ipp=e / be.loc=3MS), and the author confirms that ippa is ipp=a,
# not *ipp-a / be.loc-FREE.
NON_NOMINAL_FREE_GLOSSES = frozenset({"be.loc"})


# A0 sits below A1: the glossing standard supplies readings the corpus never
# recorded, but where a human annotator did gloss a form in running text that
# annotation wins. Ranking A0 first cost 1.0pp frequency-weighted joint exact,
# almost all of it on ōṯ, minn and yīb, where the corpus reading is the right
# one in context.
# D is the Arnold-derived working lexicon: forms cited in the dictionary, with
# glosses read mechanically off Arnold's abbreviations. Deliberately last and
# never better than LOW — the lexical slot is an untranslated German sense.
TIER_ORDER = {"A1": 0, "A0": 1, "A2": 2, "A3": 3, "B1": 4, "B2": 5, "C": 6,
              "D": 7}
TIER_CONFIDENCE = {"A0": "HIGH", "A1": "HIGH", "A2": "HIGH", "A3": "HIGH",
                   "B1": "MEDIUM", "B2": "MEDIUM", "C": "LOW", "D": "LOW"}

# Typographic markers that leak out of the dictionary as if they were senses.
# "*" in Arnold flags a footnote, not a meaning, and must never reach a gloss.
_JUNK_GLOSS = set(RULES.get("junk_gloss_markers", {}).get(
    "markers", ["*", "**", "-", "--", "—", "?", "??", "…", "..."]))


def usable_gloss(value: str | None) -> bool:
    """False for empty glosses and for bare typographic markers."""
    if not value or not value.strip():
        return False
    return all(part.strip() not in _JUNK_GLOSS
               for part in re.split(r"[-=]", value.strip()))



def prf_confidence(gloss: str | None, confidence: str) -> str:
    """PRF is an unresolved NC1/NC2 fallback and is therefore always MEDIUM."""
    if re.search(r"\.PRF\b", gloss or "", re.IGNORECASE):
        return "MEDIUM"
    return confidence


def nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value or "")


_TRANSCRIPTION_LETTER = re.compile(r"[āēīōūəḥṯḏṭṣẓšžčġʕʔḳ]")


def shifted_paradigm_cell(form: str, gloss: str) -> bool:
    """A cell of the paradigm tables read one column off: the form is a row
    or column label and the "gloss" is the form of the next cell.

    `f | ḥāy/ iḥḥay`, `come.ipv.ms | ḥ-ōn/ iḥḥ-un`, `3fs | šaḳl-aṯ`, and from
    the comparative tables one verb glossed by the other (`išḳal | išmeʕ`,
    `ġappayn | ʕlayn`). Such a cell answered for real words: f, s, p, šakəl,
    ġappayn came out as that pair, and iškal, ġapp, m had it among their
    readings. A gloss is English words and labels: it has no blank, no slash
    between forms and no letter of the transcription.
    """
    gloss = nfc(gloss).strip()
    form = nfc(form).strip()
    return bool("." in form or is_grammatical_label(form)
                or re.search(r"[\s/]", gloss)
                or _TRANSCRIPTION_LETTER.search(gloss))


def old_runtime_gloss(gloss: str) -> str:
    """Policy-v4 spelling used only to match leak-free exclusion triples."""
    return re.sub(
        r"(?<=[./])(SC|NC|PC)\b",
        lambda match: {"SC": "PST", "NC": "PRS", "PC": "SBJV"}[
            match.group(1).upper()],
        gloss,
        flags=re.IGNORECASE,
    )


VOWELS = set("aeiouāēīōūəáéíóúàèìòù")


def _cluster(text: str) -> bool:
    """True when `text` opens with two consonants in a row."""
    return len(text) > 1 and text[0] not in VOWELS and text[1] not in VOWELS


# A third-person prefix is glossed `3M` on its own and `3` once the form is
# plural — `yi-nfuḏ 3M-arrive.SBJV` beside `yi-šw-un 3-treat.SBJV-3MP`. The
# number is carried by the suffix, not by the shape of the prefix, which is why
# both `y-aḥəḏ̣r-un` and `yi-šw-un` are glossed `3`.
THIRD_SINGULAR, THIRD_PLURAL = "3M", "3"
_PLURAL_EXPONENT = re.compile(r"^3?[MF]P$")
_PERSON_PREFIX = re.compile(r"^([123])([MF]?)-")

# `-ičxen` is the second person feminine *plural*. Twelve word forms in the same
# KB file spell it out — ḥ-ičxen, zli-čxen, afk-ičxen, all 2FP — as does Arnold
# (Gr. S. 144) and the hand-read object-suffix digest in data/raw. The exponent
# row alone says 2fs, because the SC table is transcribed off a grid and the
# plural column slipped a row, exactly as the person prefixes acquired a number.
# Left standing it did double damage: a 2FP form could not be split at all, and
# a 2FS one was split and labelled wrongly. Repaired here rather than in the
# JSONL, which is regenerated from Glossing_updated.docx.
_EXPONENT_REPAIRS = {"ičxen": ("2FP", "P")}

# Vowel length alternates across the nominal paradigm — santūka beside
# santuk-ō — so a stem is matched on a length-insensitive skeleton.
# Vowel length, and the ḳ/k and č/ć pairs the sources spell either way.
_LENGTH = str.maketrans("āēīōūḳć", "aeioukč")


def skeleton(value: str) -> str:
    return nfc(value).translate(_LENGTH).lower()


# Arnold's grammar, §2.2. When stress shifts to a following syllable a long
# vowel shortens and ē, ō change quality: xēfa but xifō, yōma but yumō, bīra
# but birō. Where ō goes back to *ā it shortens to a instead: mōra but marō.
# Umlaut raises ē, e, ō, o to ī, i, ū, u before a suffix containing i.
_ALTERNATION = str.maketrans("āēīōūeo", "aiiuuiu")


def alternation_key(value: str) -> str:
    """Stem key that survives the documented vowel alternations."""
    return nfc(value).translate(_ALTERNATION).lower()


def alternation_keys(value: str) -> set[str]:
    """Both readings of a stem whose ō may go back to *ā (mōra / marō)."""
    keys = {alternation_key(value)}
    if "ō" in value or "o" in value:
        keys.add(alternation_key(nfc(value).replace("ō", "a").replace("o", "a")))
    return keys


# Glossing_updated, §Существительное and §Прилагательное. Longest first, so
# -ōyin is tried before -in. Several endings are genuinely ambiguous (-ōy is
# both -FS.INDF and -MP.DEF, -a is both -FREE and -MS.DEF); every reading is
# emitted and the token goes to review rather than one being picked.
_NOMINAL = RULES.get("nominal_suffixes", {})
NOMINAL_SUFFIXES: list[tuple[str, str]] = [
    (ending, label) for ending, label in _NOMINAL.get("simple", [])]
COMPOUND_SUFFIXES: list[tuple[str, list[str], str]] = [
    (ending, parts, label) for ending, parts, label in _NOMINAL.get("compound", [])]
POSSESSIVE_A = (_NOMINAL.get("possessive_a", {}).get("ending", "a"),
                _NOMINAL.get("possessive_a", {}).get("label", "=3FS"))

# Productive possessive endings of nouns.  These are deliberately kept local
# to the nominal rescue path below: they do not enter the general clitic
# inventory and therefore cannot be attached to verbs or closed-class hosts.
# Longest first, so =xun is never mistaken for =un, etc.
NOMINAL_POSSESSIVE_SUFFIXES: list[tuple[str, str]] = [
    ("xun", "=2MP"),
    ("xen", "=2FP"),
    ("un", "=3MP"),
    ("en", "=3FP"),
    ("ax", "=2MS"),
    ("iš", "=2FS"),
    ("aḥ", "=1P"),
    ("e", "=3MS"),
    # =a is already handled by _possessive_a as a deliberately low-ranked
    # ambiguity.  Keeping it out of this rescue prevents teps=a from beating
    # the citation reading teps-a / ...-FREE.
]

# Possessed plural allomorphs.  The zero 1S ending is intentionally absent:
# without an overt exponent it cannot be distinguished productively from an
# ordinary plural form and must remain an exact lexical/corpus analysis.
NOMINAL_PLURAL_POSSESSIVES: list[tuple[str, str, str, str]] = [
    # surface ending, plural segment, clitic segment, gloss ending
    ("ayxun", "ay", "xun", "-MP=2MP"),
    ("ayxen", "ay", "xen", "-MP=2FP"),
    ("ayhun", "ay", "hun", "-MP=3MP"),
    ("ayhen", "ay", "hen", "-MP=3FP"),
    ("aynaḥ", "ay", "naḥ", "-MP=1P"),
    ("ōye", "ōy", "e", "-MP=3MS"),
    ("ōya", "ōy", "a", "-MP=3FS"),
    ("ōx", "ō", "x", "-MP=2MS"),
    ("ōš", "ō", "š", "-MP=2FS"),
]

# The house paradigm lexicalises -yōṯ- as one plural morpheme. Elsewhere the
# productive feminine plural before FREE or a possessive clitic is -ō/-yō- +
# -ṯ- (P-F), as in bhim-ō-ṯ=e and imcan-yō-ṯ=e.
LEXICALIZED_YOT_PLURAL_LEMMAS = frozenset({"payṯa"})


def verbal_stem(stem: str) -> str | None:
    """Say whether a stem shape belongs to a verb, and on what evidence.

    -in is the adjectival INDF plural and the verbal plural at once, and the
    stem template settles it. Counted over the corpus, of the stems standing
    before -in:

        ō in the first syllable   37 verbal, 0 adjectival   (CōCeC, CōCC)
        m- prefix                 53 verbal, 1 adjectival
        neither                   19 verbal, 5 adjectival

    So ō and m- are decisive, and even the residue is verbal four times out of
    five. An adjectival reading is only offered where neither mark is present.
    """
    head = stem.split("-")[-1]
    tests = {"long_o_in_first_syllable": "ō" in head[:4],
             "m_prefix": head[:1] == "m" and len(head) > 2}
    for mark in RULES.get("verbal_stem_templates", {}).get("marks", []):
        if tests.get(mark["test"]):
            return mark["reason"]
    return None


_NC_PERSON_PREFIXES: tuple[tuple[str, str | None], ...] = (
    # ni-/či- are the surface shapes already attested in the project;
    # n-/č- are the corresponding bare prefix shapes.
    ("ni", "1"),
    ("či", "2"),
    ("n", "1"),
    ("č", "2"),
    ("", None),
)



# NC_PARADIGM_CONTEXT_V2
_NC_GUTTURAL_LOWERING = frozenset({"ʕ", "ḥ", "r"})


def _nc_e(root: str) -> str:
    """Canonical lowered NC vowel; matching also accepts e before final ʕ/ḥ/r."""
    return "a" if len(root) == 3 and root[2] in _NC_GUTTURAL_LOWERING else "e"


def _nc_shape_matches(surface_stem: str, model_stem: str) -> bool:
    """Allow predictable epenthetic ə without changing the visible segmentation."""
    return (
        surface_stem == model_stem
        or surface_stem.replace("ə", "") == model_stem.replace("ə", "")
    )


def _nc_model_variants(
        root: str,
        model_stem: str,
        display_class: str,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Licensed spellings for an otherwise exact NC-family surface model.

    The alternations live *inside* root/template matching; none is a general
    transcription rewrite.

    * underlying w may surface as y (including geminated ww -> yy);
    * root-initial ʔ may be zero before a/ō;
    * NC (not NC1/NC2) allows both e and a before final ʕ/ḥ/r;
    * a root b after a consonant may be written p (ṭalp-il=la, ṭalp-ō=la for
      √ṭlb, xaṯp-il=la for √xṯb). Line 2 keeps the written p, the lexeme is
      the verb of the root with b.
    """
    variants: dict[str, tuple[str, ...]] = {model_stem: ()}

    if "w" in root and "w" in model_stem:
        variants.setdefault(model_stem.replace("w", "y"), ("w>y",))

    if display_class == "NC" and root[2] in _NC_GUTTURAL_LOWERING:
        c3 = root[2]
        for form, rules in list(variants.items()):
            if form.endswith(f"a{c3}"):
                variants.setdefault(form[:-2] + f"e{c3}",
                                    (*rules, "e~a/_ʕ,ḥ,r"))
            elif form.endswith(f"e{c3}"):
                variants.setdefault(form[:-2] + f"a{c3}",
                                    (*rules, "e~a/_ʕ,ḥ,r"))

    # Apply initial-ʔ loss to every already licensed vowel/allomorph variant.
    for form, rules in list(variants.items()):
        if root.startswith("ʔ") and form.startswith(("ʔa", "ʔō")):
            variants.setdefault(form[1:], (*rules, "ʔ>Ø/_a,ō"))

    # Last, so a stem spelled with the b of its root matches first.
    if "b" in root:
        for form, rules in list(variants.items()):
            variants.setdefault(devoice_b_after_consonant(form),
                                (*rules, "b>p/C_"))

    return tuple(variants.items())


def nc_surface_models(
        root: str,
        context: str,
        *,
        person_prefixed: bool = False,
) -> tuple[tuple[str, str, str], ...]:
    """Strict NC/NC1/NC2 models separated by surface context.

    context:
      bare            no clitic, no F/PL ending
      free_f          no clitic, feminine -a
      plural_free     no clitic, plural -in/-an
      clitic_m        SG.M + indivisible l-object clitic
      clitic_f        SG.F -ō + indivisible l-object clitic
      clitic_plural   PL -il/-al + indivisible l-object clitic

    Short qtil/qattil are never bare no-clitic forms.
    -ō is F only in clitic_f.
    """
    if len(root) != 3:
        return ()

    c1, c2, c3 = root
    e = _nc_e(root)

    if context == "bare":
        # NC1: initial i- is epenthetic in the unprefixed 3SG.M only.
        nc1 = f"{c1}{c2}e{c3}" if person_prefixed else f"i{c1}{c2}e{c3}"
        return (
            ("NC",  f"{c1}ō{c2}{e}{c3}",       "qōtel"),
            ("NC",  f"m{c1}a{c2}{c2}{e}{c3}",  "mqattel"),
            ("NC",  f"ma{c1}{c2}{e}{c3}",       "maqtel"),
            ("NC1", nc1,                         "qtel" if person_prefixed else "iqtel"),
            ("NC2", f"{c1}a{c2}{c2}e{c3}",      "qattel"),
        )

    if context == "free_f":
        return (
            ("NC",  f"{c1}ō{c2}{c3}",           "qōtl"),
            ("NC",  f"m{c1}a{c2}{c3}",          "mqatl"),
            ("NC",  f"m{c1}a{c2}{c2}{c3}",      "mqattl"),
            ("NC",  f"ma{c1}{c2}{c3}",          "maqtl"),
            ("NC1", f"{c1}{c2}ī{c3}",           "qtīl"),
            ("NC2", f"{c1}a{c2}{c2}ī{c3}",      "qattīl"),
        )

    if context == "plural_free":
        return (
            ("NC",  f"{c1}ō{c2}{c3}",           "qōtl"),
            ("NC",  f"m{c1}a{c2}{c3}",          "mqatl"),
            ("NC",  f"m{c1}a{c2}{c2}{c3}",      "mqattl"),
            ("NC",  f"ma{c1}{c2}{c3}",          "maqtl"),
            ("NC1", f"{c1}{c2}ī{c3}",           "qtīl"),
            ("NC2", f"{c1}a{c2}{c2}ī{c3}",      "qattīl"),
        )

    if context == "clitic_m":
        return (
            ("NC",  f"{c1}a{c2}{e}{c3}",        "qatel"),
            ("NC",  f"m{c1}a{c2}{c2}{e}{c3}",   "mqattel"),
            ("NC",  f"ma{c1}{c2}{e}{c3}",       "maqtel"),
            # Long Arnold object stems plus short attested variants.
            ("NC1", f"{c1}{c2}ī{c3}",           "qtīl"),
            ("NC1", f"{c1}{c2}i{c3}",           "qtil"),
            ("NC2", f"{c1}a{c2}{c2}ī{c3}",      "qattīl"),
            ("NC2", f"{c1}a{c2}{c2}i{c3}",      "qattil"),
        )

    if context == "clitic_f":
        # ONLY this context licenses -ō as a productive F exponent.
        return (
            ("NC",  f"{c1}a{c2}{c3}",           "qatl"),
            ("NC",  f"m{c1}a{c2}{c3}",          "mqatl"),
            ("NC",  f"m{c1}a{c2}{c2}{c3}",      "mqattl"),
            ("NC",  f"ma{c1}{c2}{c3}",          "maqtl"),
            ("NC1", f"{c1}{c2}i{c3}",           "qtil"),
            ("NC1", f"{c1}{c2}ī{c3}",           "qtīl"),
            ("NC2", f"{c1}a{c2}{c2}i{c3}",      "qattil"),
            ("NC2", f"{c1}a{c2}{c2}ī{c3}",      "qattīl"),
        )

    if context == "clitic_plural":
        return (
            ("NC",  f"{c1}a{c2}{c3}",           "qatl"),
            ("NC",  f"m{c1}a{c2}{c3}",          "mqatl"),
            ("NC",  f"m{c1}a{c2}{c2}{c3}",      "mqattl"),
            ("NC",  f"ma{c1}{c2}{c3}",          "maqtl"),
            ("NC1", f"{c1}{c2}i{c3}",           "qtil"),
            ("NC1", f"{c1}{c2}ī{c3}",           "qtīl"),
            ("NC2", f"{c1}a{c2}{c2}i{c3}",      "qattil"),
            ("NC2", f"{c1}a{c2}{c2}ī{c3}",      "qattīl"),
        )

    return ()


# The stem class an NC template belongs to. A participle with m- is a derived
# stem's (mšattar II, maḥref IV), one without is the ground stem's: tafš-il=le
# is itfaš, mtafš-il=le is taffeš, never the other way round.
_NC_MODEL_CLASS = {
    "qōtel": "G", "qōtl": "G", "qatel": "G", "qatl": "G",
    "mqattel": "D", "mqattl": "D", "mqatl": "D",
    "maqtel": "C", "maqtl": "C",
}


# the object marker (=l DOM, the nota accusativi =l DO) with an object clitic
# directly after it: ...=DO=3MP
_OBJECT_AFTER_DOM = re.compile(r"(?:^|[-=.])(?:DOM|DO)=[123]")

_PNG_LABEL = re.compile(r"^[123][mfMF]?[sSpP]$")


def normalize_png_labels(gloss: str) -> str:
    """Uppercase person/number/gender labels such as 3ms → 3MS.

    The corpus writes these uppercase 971 times and lowercase never, but the
    source tables in Glossing_updated print some of them lowercase (=3ms, -3fs),
    so glosses assembled from the KB came out inconsistent with glosses taken
    from the corpus — sometimes within one sentence.
    """
    return re.sub(r"[^-=.]+",
                  lambda m: m.group(0).upper() if _PNG_LABEL.match(m.group(0))
                  else m.group(0), gloss or "")


def _gloss_lexical_head(gloss: str | None) -> str | None:
    """The lexical meaning inside a whole gloss line: `say.SC=3ms` -> `say`,
    `NEG=know.NC-MP` -> `know`. None when no cell carries a meaning."""
    for piece in re.split(r"[-=]", gloss or ""):
        if piece and has_lexical_part(piece):
            return _lexical_head(piece) or None
    return None


def gloss_placeholder(lemma: str | None) -> str | None:
    """Reduce a citation form to something usable as a one-word gloss.

    A third of the lexicon (1266 of 3816) cites two principal parts, e.g.
    "ižčmaʕ yižčmaʕ". Dropping that straight into a gloss inserts a space, which
    breaks both the IGT columns and the segmentation/gloss alignment check.
    """
    if not lemma:
        return None
    # some lemma fields hold an editorial note rather than a word, e.g.
    # "[proper name]" — that is a description of the entry, not a gloss
    if lemma.lstrip().startswith(("[", "(")):
        return None
    first = lemma.split()[0].split("/")[0].strip(" []()\u2019\u2018\"'")
    if not first or not any(c.isalpha() for c in first):
        return None
    return first


def boundary_hints(token: str) -> tuple[str, frozenset[int]]:
    """Strip ``-``/``=`` from a pre-segmented token, keeping the cut offsets.

    A hyphen typed by the user is treated as a hint that a morpheme (or clitic)
    boundary is probably there, so ``blōt-a`` yields ("blōta", {5}).
    """
    bare: list[str] = []
    cuts: set[int] = set()
    for char in token:
        if char in "-=":
            if bare:
                cuts.add(len(bare))
        else:
            bare.append(char)
    return "".join(bare), frozenset(cuts)


def _sep_seq(value: str) -> list[str]:
    return re.findall(r"[-=]", value or "")


def aligned(segmentation: str, gloss: str) -> bool:
    """Separator sequences must match exactly, component counts too."""
    if _sep_seq(segmentation) != _sep_seq(gloss):
        return False
    return len(re.split(r"[-=]", segmentation)) == len(re.split(r"[-=]", gloss))


def nonempty_segments(segmentation: str) -> bool:
    """No internal empty cell; an A0 attachment marker may end the entry."""
    if not segmentation or segmentation[0] in "-=":
        return False
    pieces = re.split(r"[-=]", segmentation)
    return all(pieces[:-1]) and (bool(pieces[-1]) or segmentation[-1] in "-=")


# Licensed boundary morphophonology used when checking reconstruction.
LICENSED_RULES = [
    (rule["name"], re.compile(rule["pattern"]), rule["replacement"])
    for rule in RULES.get("licensed_morphophonology", {}).get("rules", [])
]


def reconstruct(segmentation: str) -> str:
    """Concatenate morphemes back into the isolated surface form."""
    return re.sub(r"[-=]", "", nfc(segmentation))


def reconstructs_to(segmentation: str, surface: str) -> tuple[bool, list[str]]:
    """True when the segmentation yields the surface, directly or via a rule."""
    rebuilt = reconstruct(segmentation)
    if rebuilt == surface:
        return True, []
    if rebuilt.startswith("ʔ") and rebuilt[1:] == surface:
        return True, ["rule initial_glottal_zero"]
    for name, pattern, replacement in LICENSED_RULES:
        if reconstruct(pattern.sub(replacement, nfc(segmentation))) == surface:
            return True, [f"rule {name}"]
    return False, []


@dataclass
class Candidate:
    surface: str
    segmentation: str | None
    gloss: str | None
    lemma_id: str | None
    lemma: str | None
    lexical_gloss: str | None
    features: dict[str, Any] = field(default_factory=dict)
    tier: str = "C"
    confidence: str = "LOW"
    score: float = 0.0
    reconstruction_ok: bool = False
    reconstruction_trace: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    uncertainties: list[dict[str, Any]] = field(default_factory=list)
    lexical_gloss_status: str = "resolved"

    def key(self) -> tuple:
        return (self.segmentation, self.gloss, self.lemma_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "segmentation": self.segmentation,
            "gloss": (render_authoritative_gloss(self.gloss)
                      if self.gloss else self.gloss),
            "lemma_id": self.lemma_id, "lemma": self.lemma,
            "lexical_gloss": self.lexical_gloss,
            "lexical_gloss_status": self.lexical_gloss_status,
            "features": self.features, "tier": self.tier,
            "confidence": self.confidence, "score": round(self.score, 4),
            "reconstruction_ok": self.reconstruction_ok,
            "reconstruction_trace": self.reconstruction_trace,
            "evidence": self.evidence, "uncertainties": self.uncertainties,
        }


def _looks_lexical_head(head: str) -> bool:
    # isupper() stays local rather than joining STRUCT_LABELS: the uppercaser
    # in morphology/integration.py needs "do"/"io" to stay lexical for šwy,
    # so the shared sets can't also carry DO/IO as grammatical.
    return (bool(head) and usable_gloss(head) and has_lexical_part(head)
            and not head.isupper())


def lexical_head_from_analysis(analysis: dict) -> str | None:
    """The confirmed lexical meaning one analysis carries, or ``None``.

    Tries the morpheme(s) tagged ``stem`` first, matching how this analysis
    is meant to be read. Some confirmed analyses put the person/TAM exponent
    on the ``stem`` slot instead (the actual root then sits on a ``clitic``
    morpheme, e.g. a hollow-root imperfective like ``yarəḥmenne``), so when
    no stem-typed morpheme yields a lexical head, every morpheme is tried —
    this only ever adds coverage, it never overrides a stem-typed hit.
    """
    morphemes = analysis.get("morphemes") or []
    stems = [m for m in morphemes if m.get("type") == "stem"]
    for group in (stems, morphemes):
        for morph in group:
            head = _lexical_head(morph.get("gloss") or "")
            if _looks_lexical_head(head):
                return head.lower()
    return None


class LexicalGloss:
    """lemma_id → concise English gloss, with an explicit status."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.by_lexeme: dict[str, str] = {}
        self.status: dict[str, str] = {}
        # Author-confirmed reusable lexical gloss, one row per lexeme, always
        # consulted before any built artifact below — the editor's own path
        # (`docs/LEXICAL_REUSABLE_SPEC_RU.md` §4A). Local import: `editor/`
        # depends on this module already, so a top-level import would cycle.
        from maalula_glosser.editor.senses import active_lexical_glosses
        self.author: dict[str, dict] = active_lexical_glosses(self.root)
        self.author_excluded: dict[str, set[str]] = {
            lemma_id: set(row.get("excluded_surfaces") or ())
            for lemma_id, row in self.author.items()}
        # Corpus buckets such as [loanword] are not lexical identities. Their
        # members have unrelated meanings; no sense can be inherited globally.
        lexemes = list(iter_jsonl(root / "data/final/lexemes.jsonl"))
        self.group_ids = {
            row["lexeme_id"] for row in lexemes
            if (row.get("lemma_canonical") or "").strip().startswith("[")
        }
        # 0. the author's ruling on what one lexeme means: the rows of
        # stem_glosses.jsonl that name their lexeme. It precedes the corpus
        # head, which is only what the first analysis in the file happens to
        # say (iṭḥan `grind` was `play`, from one token of another verb), and
        # the sense join, which hands a derived stem the meaning of the basic
        # one (zappen `sell` was `buy`).
        stem_rows = [row for row in
                     iter_jsonl(root / "data/lexicon_v3/stem_glosses.jsonl")
                     if row.get("stem") and usable_gloss(row.get("gloss"))]
        for row in stem_rows:
            lemma_id = row.get("lexeme_id")
            if (lemma_id and row.get("authority") == "author_gold"
                    and lemma_id not in self.group_ids):
                self.by_lexeme[lemma_id] = row["gloss"]
                self.status[lemma_id] = "exact_stem_gloss"
        # 1. gloss used by a trusted corpus analysis (best evidence). Includes
        # the authoritative publication-IGT overlay: a lexeme whose only green
        # evidence lives there must count too, not just data/final/analyses.
        # Read as the author corrected them (`gold_corrections`): a gloss with
        # a typo, or an analysis attached to another verb's lexeme, must not
        # name the lexeme.
        corrections = GoldCorrections.load(root)
        for analysis_path in (root / "data" / "final" / "analyses.jsonl",
                              root / "data" / "grammar_kb"
                              / "authoritative_igt_overlay.jsonl"):
            for analysis in corrections.corrected(iter_jsonl(analysis_path)):
                if analysis.get("status") != "confirmed":
                    continue
                lemma_id = analysis.get("lemma_id")
                if not lemma_id or lemma_id in self.by_lexeme or lemma_id in self.group_ids:
                    continue
                lexical = lexical_head_from_analysis(analysis)
                if lexical:
                    self.by_lexeme[lemma_id] = lexical
                    self.status[lemma_id] = "corpus_analysis"
        # An exact, sourced stem translation precedes a cross-file sense join.
        # This changes lexical display only, not POS or the morphological class.
        # (a row that names its lexeme speaks for that lexeme alone: mōra `man`
        # is not its homonym of another root)
        stems = {nfc(row["stem"]): row["gloss"] for row in stem_rows
                 if row.get("authority") in ("author_gold", "arnold_german")
                 and not row.get("lexeme_id")}
        for lexeme in lexemes:
            lemma_id = lexeme["lexeme_id"]
            gloss = stems.get(nfc(lexeme.get("lemma_canonical") or ""))
            if gloss and lemma_id not in self.by_lexeme and lemma_id not in self.group_ids:
                self.by_lexeme[lemma_id] = gloss
                self.status[lemma_id] = "exact_stem_gloss"
        # 2. translated Arnold sense
        translations = {t["sense_id"]: t for t in
                        iter_jsonl(root / "data" / "senses_v1" / "translations.jsonl")}
        for link in iter_jsonl(root / "data" / "senses_v2" / "lexeme_sense_links_v2.jsonl"):
            lexeme_id = link["lexeme_id"]
            if lexeme_id in self.by_lexeme or not link.get("usable_for_maalula"):
                continue
            translation = translations.get(link["sense_id"])
            if translation:
                self.by_lexeme[lexeme_id] = translation["gloss_en"]
                self.status[lexeme_id] = "translated_sense"
        # 3. untranslated German sense → translation_pending
        self.pending_de: dict[str, str] = {}
        senses = {s["sense_id"]: s for s in
                  iter_jsonl(root / "data" / "senses_v1" / "senses.jsonl")}
        for link in iter_jsonl(root / "data" / "senses_v2" / "lexeme_sense_links_v2.jsonl"):
            lexeme_id = link["lexeme_id"]
            if lexeme_id in self.by_lexeme or not link.get("usable_for_maalula"):
                continue
            sense = senses.get(link["sense_id"])
            if sense and sense.get("definition_de_raw"):
                self.pending_de.setdefault(lexeme_id, sense["definition_de_raw"][:60])
                self.status[lexeme_id] = "translation_pending"

    def resolve(self, lemma_id: str | None,
               surface: str | None = None) -> tuple[str | None, str]:
        """`surface`, when given, lets a form the author explicitly excluded
        from the confirmed reading fall through to the built-in tiers instead
        — the checklist's unticked rows (`docs/LEXICAL_REUSABLE_SPEC_RU.md` §3)."""
        if not lemma_id:
            return None, "unknown_lexeme"
        if lemma_id in self.group_ids:
            return None, "nonlexical_group"
        author_row = self.author.get(lemma_id)
        if author_row and (surface is None
                          or nfc(surface) not in self.author_excluded.get(lemma_id, ())):
            return author_row["gloss"], "author_confirmed"
        if lemma_id in self.by_lexeme:
            return self.by_lexeme[lemma_id], self.status.get(lemma_id, "resolved")
        if lemma_id in self.pending_de:
            return None, "translation_pending"
        return None, "no_sense"

    def remember(self, row: dict[str, Any]) -> None:
        """Register one freshly confirmed lexical reading without a rebuild.

        Mirrors `gloss_text.TextGlosser.remember` for the morphology journal:
        the corrections layer is read once at construction, so a confirmation
        would otherwise only take effect after a restart.
        """
        lexeme_id = row.get("lexeme_id")
        if lexeme_id:
            self.author[lexeme_id] = row
            self.author_excluded[lexeme_id] = set(row.get("excluded_surfaces") or ())


class CompositionalAnalyzer:
    def __init__(self, root: Path | str | None = None,
                 exclude_analyses: frozenset[tuple[str, str, str]] | None = None):
        """``exclude_analyses`` drops (surface, segmentation, gloss) triples from
        the A1 inventory.

        Used only by the evaluation harness, to measure the analyzer without the
        held-out gold it would otherwise recognise verbatim. Runtime behaviour is
        unaffected: the default is to use everything available.
        """
        self.exclude_analyses = exclude_analyses or frozenset()
        from maalula_glosser.paths import ProjectPaths
        paths = ProjectPaths.discover(root)
        self.root = paths.root
        self.normalizer = TranscriptionNormalizer.from_yaml(
            self.root / "config" / "transcription_map.yaml")
        self.lexical = LexicalGloss(self.root)

        self.lexemes = {lx["lexeme_id"]: lx for lx in iter_jsonl(paths.final / "lexemes.jsonl")}
        self.form_freq: dict[str, int] = {}
        self.forms_by_surface: dict[str, list[dict]] = defaultdict(list)
        # By lexeme rather than surface: the roster the lexical-gloss checklist
        # reads to list every form of one lexeme (`editor/senses.py::sibling_forms`).
        self.forms_by_lexeme: dict[str, list[dict]] = defaultdict(list)
        for form in iter_jsonl(paths.final / "forms.jsonl"):
            surface = nfc(form["form_canonical"])
            self.forms_by_surface[surface].append(form)
            self.form_freq[surface] = self.form_freq.get(surface, 0) + (form.get("frequency") or 0)
            lexeme_id = form.get("lexeme_id")
            if lexeme_id:
                self.forms_by_lexeme[lexeme_id].append(form)

        # Слова, добавленные автором из своих текстов: обычные лексемы с
        # формами, а не отдельный список. Читаются из append-only слоя и
        # только дополняют — запись из data/final не перекрывается даже в
        # памяти (инвариант #4).
        from maalula_glosser.lexicon_additions import active_additions
        for addition in active_additions(self.root):
            lexeme_id = addition["lexeme_id"]
            if lexeme_id in self.lexemes:
                continue
            self.lexemes[lexeme_id] = {
                "lexeme_id": lexeme_id,
                "lemma_canonical": addition["lemma_canonical"],
                "part_of_speech": addition.get("part_of_speech"),
                "root_canonical": addition.get("root_canonical"),
                "status": "author_addition",
            }
            for surface_row in addition.get("surfaces") or []:
                surface = nfc(surface_row["form_canonical"])
                frequency = int(surface_row.get("frequency") or 0)
                form = {"form_canonical": surface, "frequency": frequency,
                        "lexeme_id": lexeme_id, "selected_analysis_id": None,
                        "attested": True, "generated": False}
                self.forms_by_surface[surface].append(form)
                self.forms_by_lexeme[lexeme_id].append(form)
                self.form_freq[surface] = self.form_freq.get(surface, 0) + frequency

        # Author edits to an already-established lexeme (merge/delete/rename).
        # Applied on top of base + additions, in memory only — `data/final`
        # is never rewritten (invariant #4). `forms_by_surface` and
        # `forms_by_lexeme` share the very same form dicts built above (each
        # form was appended to both), so mutating a form's `lexeme_id` here
        # (merge) or dropping it from `forms_by_lexeme` (delete) is visible
        # in both structures without a second pass over `forms_by_surface`.
        from maalula_glosser.lexicon_edits import active_edits
        edits = active_edits(self.root)
        for lexeme_id, row in edits["renames"].items():
            if lexeme_id in self.lexemes:
                self.lexemes[lexeme_id]["lemma_canonical"] = row["lemma_canonical"]
        for source_id, row in edits["merges"].items():
            if source_id not in self.lexemes:
                continue
            target_id = row["target_lexeme_id"]
            self.lexemes.pop(source_id, None)
            moved = self.forms_by_lexeme.pop(source_id, [])
            for form in moved:
                form["lexeme_id"] = target_id
            self.forms_by_lexeme[target_id].extend(moved)
        for lexeme_id in edits["deletes"]:
            if lexeme_id not in self.lexemes:
                continue
            self.lexemes.pop(lexeme_id, None)
            removed = self.forms_by_lexeme.pop(lexeme_id, [])
            for form in removed:
                surface = nfc(form.get("form_canonical") or "")
                bucket = self.forms_by_surface.get(surface)
                if bucket and form in bucket:
                    bucket.remove(form)
                    if not bucket:
                        # An empty bucket left behind here breaks every
                        # other reader that assumes `forms_by_surface[key]`
                        # is non-empty whenever the key is present at all
                        # (e.g. `forms[0]` a few hundred lines below).
                        del self.forms_by_surface[surface]
                self.form_freq[surface] = max(
                    0, self.form_freq.get(surface, 0) - (form.get("frequency") or 0))
                if self.form_freq.get(surface) == 0:
                    del self.form_freq[surface]

        # ---- evidence inventories -------------------------------------
        # Corpus analyses annotated under the cancelled blanket NC→PRF rule are
        # rewritten to the current NC→PRS on the way in. data/final is never
        # touched; the repairs live in the append-only corrections layer.
        gloss_repairs: dict[tuple[str, str, str], str] = {}
        for row in iter_jsonl(self.root / "data" / "corrections_v2"
                              / "gold_nc_prs_corrections.jsonl"):
            gloss_repairs[(nfc(row["surface_canonical"]), row["segmentation"],
                           row["gloss_original"])] = row["gloss_corrected"]
        self.gloss_repairs = gloss_repairs
        # The author's own corrections to single analyses (a typo, the label of
        # another clitic, a boundary), applied before anything else reads them.
        self.gold_corrections = GoldCorrections.load(self.root)

        self._enclitics: dict[str, list[tuple[str, int]]] | None = None
        self._proclitic_canonical: dict[str, str] | None = None
        self._principal_parts: dict[str, set[str]] | None = None
        self.a1: dict[str, list[dict]] = defaultdict(list)   # trusted corpus
        for analysis in iter_jsonl(paths.final / "analyses.jsonl"):
            if analysis.get("status") != "confirmed":
                continue
            surface = nfc(analysis.get("surface_canonical") or "")
            if surface and analysis.get("segmentation") and analysis.get("gloss_clean"):
                key = (surface, analysis["segmentation"], analysis["gloss_clean"])
                if key in self.exclude_analyses:
                    continue
                corrected = self.gold_corrections.correct(analysis)
                if corrected is None:
                    continue
                if corrected is analysis:
                    repaired = gloss_repairs.get(key)
                    if repaired:
                        corrected = {**analysis, "gloss_clean": repaired,
                                     "nc_prs_repaired": True}
                elif (surface, corrected["segmentation"],
                      corrected["gloss_clean"]) in self.exclude_analyses:
                    continue
                self.a1[surface].append(with_lexical_do(corrected))

        # Newer publication IGT overlays older corpus analyses without
        # rewriting either source. Every entry retains a table/column locator.
        # The overlay is generated by scripts/build_authoritative_igt_overlay.py.
        for analysis in iter_jsonl(
                self.root / "data" / "grammar_kb"
                / "authoritative_igt_overlay.jsonl"):
            if analysis.get("status") != "confirmed":
                continue
            surface = nfc(analysis.get("surface_canonical") or "")
            key = (surface, analysis.get("segmentation"),
                   analysis.get("gloss_clean"))
            old_key = (surface, analysis.get("segmentation"),
                       old_runtime_gloss(analysis.get("gloss_clean") or ""))
            if not surface or key in self.exclude_analyses \
                    or old_key in self.exclude_analyses:
                continue
            corrected = self.gold_corrections.correct(analysis)
            if corrected is None:
                continue
            self.a1[surface].append(
                {**with_lexical_do(corrected), "authoritative_overlay": True})

        # Readings the author named that neither source has (yōm=l day=HD
        # beside yōməl when).
        for analysis in self.gold_corrections.added():
            surface = analysis["surface_canonical"]
            key = (surface, analysis["segmentation"], analysis["gloss_clean"])
            if key not in self.exclude_analyses:
                self.a1[surface].append(analysis)

        # Confirmed corpus morphology already carries useful POS evidence even
        # though data/final/lexemes.jsonl leaves part_of_speech empty for the
        # base lexicon.  This mirrors editor/senses.py::infer_part_of_speech,
        # but lives here too so morphology can use it without importing the
        # editor package (which would create a compositional -> editor.session
        # -> gloss_text -> compositional cycle).
        gloss_inventory = _json.loads(
            (self.root / "config" / "gloss_inventory.json").read_text(encoding="utf-8")
        )
        gloss_labels = gloss_inventory.get("labels") or {}
        categories = {key: (value or {}).get("category")
                      for key, value in gloss_labels.items()}
        verbal_labels = {key for key, category in categories.items()
                         if category == "conjugation/aspect"}
        nominal_labels = {key for key, category in categories.items()
                          if category == "nominal state"} | {"HD"}
        gendered_labels = {key for key, category in categories.items()
                           if category == "person/gender/number"}
        display_map = gloss_inventory.get("display_map") or {}
        verbal_labels |= {alias for alias, target in display_map.items()
                          if target in verbal_labels}
        nominal_labels |= {alias for alias, target in display_map.items()
                           if target in nominal_labels}
        gendered_labels |= {alias for alias, target in display_map.items()
                            if target in gendered_labels}
        feminine_labels = {label for label in gendered_labels if "F" in label.upper()}
        masculine_labels = {label for label in gendered_labels if "M" in label.upper()}

        self.a1_by_lexeme: dict[str, list[dict]] = defaultdict(list)
        for analyses in self.a1.values():
            for analysis in analyses:
                lemma_id = analysis.get("lemma_id")
                if lemma_id:
                    self.a1_by_lexeme[lemma_id].append(analysis)

        self.lexeme_part_of_speech: dict[str, str] = {
            lexeme_id: str(lexeme.get("part_of_speech") or "").lower()
            for lexeme_id, lexeme in self.lexemes.items()
            if str(lexeme.get("part_of_speech") or "").lower()
            in {"verb", "noun", "adjective"}
        }
        self.lexeme_pos_source: dict[str, str] = {
            lexeme_id: "explicit" for lexeme_id in self.lexeme_part_of_speech
        }
        for lexeme_id, row in self.lexical.author.items():
            pos = str(row.get("part_of_speech") or "").lower()
            if lexeme_id in self.lexemes and pos in {"verb", "noun", "adjective"}:
                self.lexeme_part_of_speech[lexeme_id] = pos
                self.lexeme_pos_source[lexeme_id] = "explicit"

        # D1 (HANDOFF_verbal_lexemes.md): a curated, hand-revisable list of
        # lexeme_id -> "verb"/"not_verb", checked ahead of the citation-shape
        # and corpus-label votes below. "not_verb" does not assign a POS by
        # itself; it only stops the corpus vote further down from calling
        # that lexeme a verb (data, not code -- config/verbal_lexemes.json).
        curated_pos: dict = {}
        curated_not_verb: set[str] = set()
        verbal_lexemes_path = self.root / "config" / "verbal_lexemes.json"
        if verbal_lexemes_path.exists():
            curated_pos = (
                _json.loads(verbal_lexemes_path.read_text(encoding="utf-8")).get(
                    "verbal_lexemes"
                )
                or {}
            )
        for lexeme_id, decision in curated_pos.items():
            if lexeme_id not in self.lexemes:
                continue
            if decision == "verb" and lexeme_id not in self.lexeme_part_of_speech:
                self.lexeme_part_of_speech[lexeme_id] = "verb"
                self.lexeme_pos_source[lexeme_id] = "curated_list"
            elif decision == "not_verb":
                curated_not_verb.add(lexeme_id)

        # D1, candidate A: a citation form shaped like a verb's principal
        # parts (`iqtal yiqtul`) is a verb ahead of corpus evidence, so
        # `kall ykall` / `rapp yrapp` stay verbs even where a same-root noun
        # also has corpus attestations (those get reattached by hand).
        for lexeme_id, lexeme in self.lexemes.items():
            if lexeme_id in self.lexeme_part_of_speech or lexeme_id in curated_not_verb:
                continue
            if citation_is_verbal(lexeme.get("lemma_canonical")):
                self.lexeme_part_of_speech[lexeme_id] = "verb"
                self.lexeme_pos_source[lexeme_id] = "citation_shape"

        for lexeme_id, analyses in self.a1_by_lexeme.items():
            if lexeme_id in self.lexeme_part_of_speech:
                continue
            labels: set[str] = set()
            for analysis in analyses:
                for morph in analysis.get("morphemes") or []:
                    parts = (morph.get("gloss") or "").split(".")
                    labels.update(part for part in
                                  (parts[1:] if morph.get("type") == "stem" else parts)
                                  if part)
            # A curated "not_verb" vetoes only the verb reading of the vote;
            # noun/adjective are still free to come from the same labels.
            vote_verbal = verbal_labels if lexeme_id not in curated_not_verb else set()
            inferred = pos_from_labels(labels, vote_verbal, nominal_labels,
                                        feminine_labels, masculine_labels)
            if inferred:
                self.lexeme_part_of_speech[lexeme_id] = inferred
                self.lexeme_pos_source[lexeme_id] = "confirmed_corpus_glosses"

        # Analysis-frequency prior for homonyms, counted over TRAIN documents
        # only (never dev/test), so the commonest reading of an ambiguous
        # high-frequency form wins instead of an arbitrary one.
        self.analysis_prior: dict[tuple[str, str, str], int] = defaultdict(int)
        split_path = self.root / "data" / "evaluation" / "split_examples.jsonl"
        train_ids = {row["example_id"] for row in iter_jsonl(split_path)
                     if row.get("split") == "train"}
        if train_ids:
            for example in iter_jsonl(paths.final / "igt_examples.jsonl"):
                if example.get("example_id") not in train_ids:
                    continue
                for token in example.get("tokens") or []:
                    surface = nfc(token.get("surface_canonical") or "")
                    segmentation = token.get("segmentation")
                    gloss = token.get("gloss")
                    if surface and segmentation and gloss:
                        reading = self.gold_corrections.correct_reading(
                            surface, segmentation, gloss)
                        if reading is None:
                            continue
                        if reading == (segmentation, gloss):
                            gloss = gloss_repairs.get(
                                (surface, segmentation, gloss), gloss)
                        else:
                            segmentation, gloss = reading
                        self.analysis_prior[
                            (surface, segmentation, lexical_do(gloss))] += 1

        # Morpheme-level preposition sense prior. A fused proclitic+pronoun such
        # as bē has only a handful of memorised tokens, and ranking its readings
        # by that tiny fused count makes a rare sense (with=3MS ×5) beat the
        # b= morpheme's real distribution (loc/in/ins). Aggregate, per proclitic
        # morpheme, how often each preposition sense is attested across ALL its
        # uses, so the dominant sense leads and the rest stay as alternatives.
        self.prep_sense_prior: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int))
        for (surface, segmentation, gloss), count in self.analysis_prior.items():
            seg_head = re.split(r"[=-]", segmentation, maxsplit=1)[0]
            gloss_head = re.split(r"[=-]", gloss, maxsplit=1)[0]
            if ("=" in segmentation and gloss_head.lower() in PREP_SENSES
                    and len(seg_head) <= 3):
                self.prep_sense_prior[seg_head][gloss_head] += count

        self.a3: dict[str, list[dict]] = defaultdict(list)   # paradigm cells
        for cell in iter_jsonl(self.root / "data" / "grammar_kb" / "paradigm_cells.jsonl"):
            if cell.get("cell_type") != "word_form":
                continue
            surface = nfc(cell.get("form_canonical") or "")
            if surface and cell.get("segmentation") and cell.get("gloss"):
                if shifted_paradigm_cell(surface, cell["gloss"]):
                    continue
                if "DO." in cell["gloss"]:
                    cell = {**cell, "gloss": lexical_do(cell["gloss"])}
                self.a3[surface].append(cell)

        self.a2: dict[str, list[dict]] = defaultdict(list)   # Arnold Maalula forms
        # Exact B/J attestations are not A2 Maalula evidence.  Keep them in a
        # separate index, however: when the Maalula exponent inventory
        # independently segments the same surface, the cross-dialect match is
        # positive confirmation rather than a reason to reject the analysis.
        self.cross_dialect: dict[str, list[dict]] = defaultdict(list)

        # A consonantal root is not a lexeme identifier.  Several unrelated
        # lexemes may share one root (krṣ has koreṣča, korṣa, akreṣ, karreṣ,
        # etc.), so choosing the first lexeme seen for a root can silently bind
        # an inflected form to the wrong word.  Resolve dictionary records by
        # their exact Arnold entry first, then by the cited lemma, and use a root
        # only when it identifies exactly one project lexeme.
        lexemes_by_root: dict[str, set[str]] = defaultdict(set)
        lexemes_by_lemma: dict[str, set[str]] = defaultdict(set)
        for lexeme_id, lexeme in self.lexemes.items():
            root = nfc(lexeme.get("root_canonical") or "")
            if root:
                lexemes_by_root[root].add(lexeme_id)
            values = [lexeme.get("lemma_canonical") or "",
                      *(lexeme.get("principal_parts") or [])]
            for value in values:
                lemma_head = gloss_placeholder(nfc(value))
                if lemma_head:
                    lexemes_by_lemma[lemma_head].add(lexeme_id)

        def cited_noun(match: dict) -> bool:
            """A noun the dictionary pipeline took for the verb of its example.

            `inflected_forms` decides which word of a cited example a label
            such as "prät. 3 pl." belongs to by its likeness to the root. In
            xaṯpunnlə xṯōba "they wrote the contract" the noun kept all three
            radicals and the verb had devoiced one, so xṯōba went on record as
            that preterite (likewise payṯa in applēle hanna payṯa, šoġlun in
            ščaġlušš šoġlun), and the corpus form of the same spelling came
            out book.SC. Such a pick is marked `parsed`, not `explicit`. When
            the corpus knows the spelling as a form of a lexeme that is no
            verb and is cited in the free state (-a), the word is that noun
            and the row is no evidence of a verb form.
            """
            if match.get("confidence") != "parsed" or not match.get("tam_canonical"):
                return False
            lexeme_id = match.get("lexeme_id")
            if self.lexeme_part_of_speech.get(lexeme_id) == "verb":
                return False
            citation = gloss_placeholder(nfc(
                (self.lexemes.get(lexeme_id) or {}).get("lemma_canonical") or "")) or ""
            return citation.endswith("a")

        lexeme_by_source_entry: dict[str, str] = {}
        ambiguous_source_entries: set[str] = set()
        cited_nouns: set[str] = set()       # their dictionary records
        for match in iter_jsonl(self.root / "data" / "dictionary_morphology_v1" / "masc_matches.jsonl"):
            source_entry_id = match.get("source_entry_id")
            lexeme_id = match.get("lexeme_id")
            if source_entry_id and lexeme_id:
                previous = lexeme_by_source_entry.get(source_entry_id)
                if previous and previous != lexeme_id:
                    ambiguous_source_entries.add(source_entry_id)
                else:
                    lexeme_by_source_entry[source_entry_id] = lexeme_id
            if not match.get("usable_for_maalula"):
                continue
            if cited_noun(match):
                cited_nouns.add(match["inflected_form_id"])
                continue
            self.a2[nfc(match["arnold_surface_canonical"])].append(match)
            self.a2[nfc(match["form_canonical"])].append(match)
        for source_entry_id in ambiguous_source_entries:
            lexeme_by_source_entry.pop(source_entry_id, None)
        # dictionary entry -> the one project lexeme its forms were matched to
        self.lexeme_by_source_entry = lexeme_by_source_entry

        def dictionary_lexeme_id(record: dict) -> str | None:
            source_entry_id = record.get("source_entry_id")
            if source_entry_id in lexeme_by_source_entry:
                return lexeme_by_source_entry[source_entry_id]
            lemma_head = gloss_placeholder(nfc(record.get("lemma_canonical") or ""))
            lemma_ids = lexemes_by_lemma.get(lemma_head or "", set())
            if len(lemma_ids) == 1:
                return next(iter(lemma_ids))
            root_ids = lexemes_by_root.get(nfc(record.get("root") or ""), set())
            if len(root_ids) == 1:
                only_id = next(iter(root_ids))
                # A verb form of this root is not lent to the one same-root
                # lexeme unless that lexeme is itself a verb (D1/D5): this is
                # exactly the msabakča/sōba/kappōna bug -- an Arnold TAM form
                # matched, by spelling, to a same-root noun's citation form.
                if (record.get("tam_canonical")
                        and self.lexeme_part_of_speech.get(only_id) != "verb"):
                    return None
                return only_id
            return None
        # every other explicitly cited form, linked to a lexeme by root
        matched_ids = {m["inflected_form_id"] for ms in self.a2.values() for m in ms}
        for record in iter_jsonl(self.root / "data" / "dictionary_morphology_v1" / "inflected_forms.jsonl"):
            if not record.get("selected"):
                continue
            surface = nfc(record["surface_canonical"])
            if not surface:
                continue
            normalized = {
                "arnold_surface_canonical": surface,
                "form_canonical": surface,
                "lexeme_id": dictionary_lexeme_id(record),
                "lemma_canonical": record.get("lemma_canonical"),
                "root": record.get("root"),
                "inflected_form_id": record["inflected_form_id"],
                "tam_canonical": record.get("tam_canonical"),
                "features": record.get("features") or {},
                "object_features": record.get("object_features"),
                "effective_dialect": record.get("effective_dialect"),
                "source_page": record.get("source_page"),
                "source_span": record.get("source_span"),
            }
            if record["inflected_form_id"] in cited_nouns:
                continue                    # the record itself is the wrong word
            if record.get("usable_for_maalula"):
                if record["inflected_form_id"] not in matched_ids:
                    self.a2[surface].append(normalized)
            elif record.get("effective_dialect") in {"B", "J"}:
                self.cross_dialect[surface].append(normalized)

        # NC1/NC2 are the two perfect stem shapes of the base/G conjugation.
        # A root can also own derived II/III/IV lexemes, so same-root lookup must
        # prefer an explicitly class-I/G lexical binding when the dictionary has
        # one.  If no G link is known, the reverse parser keeps the existing
        # same-root fallback rather than inventing a lexeme.
        self.g_lexeme_ids_by_root: dict[str, set[str]] = defaultdict(set)

        # Strong G lexemes can be recognized directly from the project lexeme
        # record (citation i- + the linked root skeleton).
        for lexeme_id, lexeme in self.lexemes.items():
            if is_unambiguous_strong_stem_i(lexeme):
                root = nfc(lexeme.get("root_canonical") or "")
                if root:
                    self.g_lexeme_ids_by_root[root].add(lexeme_id)

        # Arnold class-I links also cover weak roots, which the deliberately
        # conservative strong-stem helper above rejects.
        for matches in self.a2.values():
            for match in matches:
                cls = str(match.get("arnold_stem_class")
                          or match.get("stem_class") or "").upper()
                lexeme_id = match.get("lexeme_id")
                if cls != "I" or not lexeme_id or lexeme_id not in self.lexemes:
                    continue
                root = nfc((self.lexemes.get(lexeme_id) or {}).get("root_canonical") or "")
                if root:
                    self.g_lexeme_ids_by_root[root].add(lexeme_id)

        # Some class-I dictionary entries have no exact MASC surface row, but
        # their source entry is still linked to a project lexeme.  Preserve that
        # class evidence too instead of falling back to arbitrary same-root IDs.
        for row in iter_jsonl(self.root / "data" / "dictionary_morphology_v1"
                              / "roots_and_classes.jsonl"):
            if not any(str(stem.get("class") or "").upper() == "I"
                       for stem in row.get("stems") or ()):
                continue
            lexeme_id = lexeme_by_source_entry.get(row.get("entry_id"))
            if not lexeme_id or lexeme_id not in self.lexemes:
                continue
            root = nfc((self.lexemes.get(lexeme_id) or {}).get("root_canonical")
                       or row.get("root") or "")
            if root:
                self.g_lexeme_ids_by_root[root].add(lexeme_id)

        # ---- morphology: exponents and clitics -------------------------
        self.suffixes: list[tuple[str, str, dict]] = []
        self.prefixes: list[tuple[str, str, dict]] = []
        cat_to_tam = CAT_TO_TAM
        for cell in iter_jsonl(self.root / "data" / "grammar_kb" / "paradigm_cells.jsonl"):
            if cell.get("cell_type") != "exponent":
                continue
            form = (cell.get("form_original") or "").strip("()")
            gloss = normalize_png_labels((cell.get("gloss") or "").strip())
            if not form or not gloss:
                continue
            tam = cat_to_tam.get(cell.get("source_category"))
            feats = {**(cell.get("features") or {}), "tam": tam}
            if cell.get("exponent_position") == "suffix":
                for variant in form.lstrip("-").split("/"):
                    if not variant:
                        continue
                    if variant in _EXPONENT_REPAIRS:
                        gloss, number = _EXPONENT_REPAIRS[variant]
                        feats = {**feats, "number": number}
                    self.suffixes.append((variant, gloss.lstrip("-"), feats))
            elif cell.get("exponent_position") == "prefix":
                # The person prefix does not carry number. The exponent table
                # reads its rows off a grid, so the singular row kept `yi-` and
                # the plural row `y-`, as if the two shapes were a number
                # contrast; they are one prefix in two phonological shapes,
                # chosen by what follows (see the realisation rule below), and
                # number is spelled by the suffix. Left in place, the spurious
                # feature hid `yi-` from every plural form, so a cluster-initial
                # plural like `yi-ščaġl-un` could not be cut at all.
                feats = {**feats, "number": None}
                self.prefixes.append((form.rstrip("-"), gloss.rstrip("-"), feats))
        self.suffixes.sort(key=lambda t: -len(t[0]))
        self.prefixes.sort(key=lambda t: -len(t[0]))

        self.clitics: list[dict] = []
        for clitic in iter_jsonl(self.root / "data" / "grammar_kb" / "clitic_paradigms.jsonl"):
            for variant in clitic.get("variants") or []:
                variant = variant.strip("=")
                if variant:
                    self.clitics.append({
                        "form": variant,
                        "gloss": normalize_png_labels(
                            (clitic.get("gloss") or "").lstrip("=")),
                        "function": clitic.get("function"), "paradigm": clitic.get("paradigm"),
                        "features": clitic.get("features") or {},
                    })
        self.clitics.sort(key=lambda c: -len(c["form"]))

        # ---- A0: the project's own glossing standard --------------------
        # Closed lexical classes transcribed from Glossing_updated.docx —
        # prepositions, pseudoverbs, particles, reflexives/reciprocals and
        # possessive pronouns, none of which the V1 extractor picked up.
        self.a0: dict[str, list[dict]] = defaultdict(list)
        self.proclitics: list[dict] = []
        for entry in iter_jsonl(self.root / "data" / "grammar_kb" / "closed_class_v2.jsonl"):
            form = nfc(entry.get("form") or "")
            if not form or not usable_gloss(entry.get("gloss")):
                continue
            self.a0[form].append(entry)
            if entry.get("attachment") == "proclitic":
                self.proclitics.append(entry)
        self.proclitics.sort(key=lambda e: -len(e["form"]))

        # Author-confirmed readings for individual surfaces. These are promoted,
        # never substituted: the competing attested analyses stay in the
        # candidate list so the choice remains visible and reversible.
        # Token corrections carry a scope. A scoped correction names the exact
        # (document_id, sentence_id, token_index, original_token) it applies to
        # and changes only that occurrence. A legacy correction predates the
        # coordinates: it is offered but flagged and routed to review, never a
        # silent global override of every identical surface. scope=="moved"
        # entries are tombstones for readings relocated to the reusable layer.
        self.token_corrections_scoped: dict[tuple, list[dict]] = defaultdict(list)
        self.token_corrections_legacy: dict[str, list[dict]] = defaultdict(list)
        for row in iter_jsonl(self.root / "data" / "corrections_v2" / "token_corrections.jsonl"):
            surface = nfc(row.get("surface_canonical") or "")
            scope = row.get("scope")
            if not (surface and usable_gloss(row.get("gloss"))) or scope == "moved":
                continue
            key = self._scope_key(row)
            if key is not None and scope != "legacy":
                self.token_corrections_scoped[key].append(row)
            else:
                self.token_corrections_legacy[surface].append(row)

        # Reusable morphology additions apply to a surface globally, like a
        # lexeme sense. A separate, deliberate layer with its own evidence.
        self.reusable_morphology: dict[str, list[dict]] = defaultdict(list)
        # The host part of a confirmed proclitic reading, under the host's own
        # spelling: ʕa=Falaṣṭīn "to=Palestine" also teaches Falaṣṭīn
        # "Palestine", so m Falaṣṭīn / Falaṣṭīn alone / b=Falaṣṭīn follow.
        # Consulted only where no direct row exists (`_from_reusable_morphology`).
        self.reusable_hosts: dict[str, list[dict]] = defaultdict(list)
        for row in morphology_readings(self.root):
            surface = nfc(row.get("surface_canonical") or "")
            # Older editors silently globalised ordinary occurrence edits.
            # Keep the append-only history, but require explicit reapproval
            # before those rows can override unrelated occurrences.
            if (row.get("note") == "finished in the text editor"
                    and not row.get("explicit_reuse")):
                continue
            if surface and usable_gloss(row.get("gloss")):
                self.remember_morphology(row)

        # Exact normative/irregular paradigm cells (morphology_v4). An
        # independent source (paradigms_final.docx), not held-out corpus, so
        # using it is not leakage. It supplies the weak/irregular verb forms the
        # compositional path never generates.
        self._paradigm_v4 = None
        v4_cells = self.root / "data" / "grammar_kb" / "morphology_v4" / "paradigm_cells.jsonl"
        if v4_cells.exists():
            from maalula_glosser.morphology.paradigm_store import ParadigmStore
            self._paradigm_v4 = ParadigmStore(v4_cells)

        # The M column of Arnold's own paradigms, hand-corrected: the third-weak
        # verb the productive engine cannot reach, its object suffixes, and the
        # pronouns. Read here rather than lazily so a malformed table fails at
        # startup, where it can be seen, and not mid-document.
        from maalula_glosser.morphology import arnold_tables
        self._arnold_tables: dict[str, list[tuple[str, str, str, str]]] = {}
        if arnold_tables.ENABLED and (self.root / arnold_tables.TABLES).is_dir():
            self._arnold_tables = arnold_tables.readings(self.root)

        # ---- D: the Arnold-derived working lexicon ----------------------
        self.lexicon_v3: dict[str, list[dict]] = defaultdict(list)
        for entry in iter_jsonl(self.root / "data" / "lexicon_v3" / "lexicon.jsonl"):
            german = "; ".join(s["de"] for s in entry.get("senses") or [])[:120]
            head = entry.get("headword") or ""
            for form in entry.get("forms") or []:
                surface = nfc(form.get("canonical") or "")
                if surface:
                    self.lexicon_v3[surface].append({**form, "headword": head,
                                                     "german": german,
                                                     "page": entry.get("page"),
                                                     "root": entry.get("root"),
                                                     "lexicon_id": entry.get("lexicon_id")})
            head_surface = nfc(entry.get("headword_canonical") or "")
            if head_surface:
                self.lexicon_v3[head_surface].append(
                    {"canonical": head_surface, "proposed_segmentation": head_surface,
                     "proposed_gloss": "?", "headword": head, "german": german,
                     "page": entry.get("page"), "root": entry.get("root"),
                     "lexicon_id": entry.get("lexicon_id"),
                     "arnold_label": None, "gloss_authority": "low_derived"})

        # Lexical glosses for stems, keyed on the length-insensitive skeleton.
        # These outrank the lemma placeholder the nominal paradigm would print.
        # Two authorities live in the one file and are not interchangeable:
        #   author_gold    the author confirmed this reading
        #   arnold_german  Arnold's German sense rendered in English here
        # The author always wins, whatever order the append-only file is in.
        self.stem_glosses: dict[str, str] = {}
        self.stem_gloss_authority: dict[str, str] = {}
        self.entry_glosses: dict[str, str] = {}
        for row in iter_jsonl(self.root / "data" / "lexicon_v3" / "stem_glosses.jsonl"):
            if not (row.get("stem") and row.get("gloss")):
                continue
            if row.get("lexeme_id"):
                # the meaning of one lexeme (`LexicalGloss`), reached through
                # its id: its citation form is no stem to cut other words by
                continue
            authority = row.get("authority") or "author_gold"
            if row.get("lexicon_id"):
                self.entry_glosses[row["lexicon_id"]] = row["gloss"]
            for key in (skeleton(nfc(row["stem"])),
                        skeleton(nfc(row["stem"]).rstrip("a"))):
                if key in self.stem_glosses and not (
                        authority == "author_gold"
                        and self.stem_gloss_authority.get(key) != "author_gold"):
                    continue
                self.stem_glosses[key] = row["gloss"]
                self.stem_gloss_authority[key] = authority

        # length-insensitive index of stems that some source vouches for, used
        # to anchor nominal-paradigm segmentation, plus a looser one keyed on
        # the documented vowel alternations. The loose index is consulted only
        # when the exact one misses, so it can add matches but never displace.
        self.stem_index: dict[str, tuple[str, str]] = {}
        self.stem_index_alt: dict[str, tuple[str, str]] = {}
        # An author-confirmed stem gloss is itself sufficient evidence that the
        # stem exists. A gloss merely carried over from Arnold's German is not:
        # it says what a known word means, never that an unseen cut is a word.
        for key, gloss in self.stem_glosses.items():
            if self.stem_gloss_authority.get(key) == "author_gold":
                self.stem_index[key] = (gloss, "author-confirmed stem gloss")
        for surface_form, forms in self.forms_by_surface.items():
            lemma_id = forms[0].get("lexeme_id")
            if lemma_id in self.lexical.group_ids:
                continue
            gloss, _status = self.lexical.resolve(lemma_id)
            name = gloss or gloss_placeholder(
                (self.lexemes.get(lemma_id) or {}).get("lemma_canonical"))
            if not name:
                continue
            for base in (surface_form, surface_form.rstrip("a")):
                if len(base) >= 3:
                    self.stem_index.setdefault(skeleton(base), (name, "corpus form"))
                    for key in alternation_keys(base):
                        self.stem_index_alt.setdefault(key, (name, "corpus form, vowel alternation"))
        for surface_form, entries in self.lexicon_v3.items():
            name = gloss_placeholder(entries[0].get("headword"))
            if not name:
                continue
            for base in (surface_form, surface_form.rstrip("a")):
                if len(base) >= 3:
                    self.stem_index.setdefault(skeleton(base), (name, "Arnold headword"))
                    for key in alternation_keys(base):
                        self.stem_index_alt.setdefault(key, (name, "Arnold headword, vowel alternation"))

        # A dictionary (Arnold) verb form is no longer a POS source on its
        # own (D1/D5, HANDOFF_verbal_lexemes.md): `masc_matches` links a MASC
        # form to a same-written Arnold TAM form by spelling, and that link
        # is exactly what put NC/PRF labels on nouns of the same root
        # (msabakča, kappōna, sōba, ...). A lexeme's verb status now comes
        # only from `part_of_speech`/the author journal above, the curated
        # list and citation shape (D1), or its own confirmed corpus glosses.
        self.verbal_lexeme_ids: set[str] = {
            lexeme_id for lexeme_id, pos in self.lexeme_part_of_speech.items()
            if pos == "verb"
        }

        # Roots are already part of the project lexeme records, but until now
        # the productive analyzer only used pre-stored stems. Retain the root
        # index so an NC surface template can license an unseen inflected stem
        # without inventing a lexical root.
        self.lexeme_ids_by_root: dict[str, set[str]] = defaultdict(set)
        for lexeme_id, lexeme in self.lexemes.items():
            root = nfc(lexeme.get("root_canonical") or "")
            if root:
                self.lexeme_ids_by_root[root].add(lexeme_id)

        # morphology_v4 is independent evidence that a root is verbal.
        from maalula_glosser.morphology.lexeme_profiles import ProfileStore
        self.verbal_profile_roots: set[str] = set()
        profile_store = ProfileStore.load(self.root)
        self.explicit_nc_stems = [
            (profile.lexeme_id, display, profile.slot(display + "_STEM"))
            for profile in profile_store.profiles
            if profile.lexeme_id in self.lexemes
            for display in ("NC", "NC1", "NC2")
            if profile.slot(display + "_STEM")
        ]
        for profile in profile_store.profiles:
            if str(profile.pos).upper() != "VERB":
                continue
            if profile.lexeme_id:
                if (profile.lexeme_id in self.lexemes
                        and profile.lexeme_id not in self.lexeme_part_of_speech):
                    self.lexeme_part_of_speech[profile.lexeme_id] = "verb"
                    self.lexeme_pos_source[profile.lexeme_id] = "morphology_v4_profile"
                if self.lexeme_part_of_speech.get(profile.lexeme_id) == "verb":
                    self.verbal_lexeme_ids.add(profile.lexeme_id)
            if profile.root:
                profile_root = nfc("".join(profile.root))
                self.verbal_profile_roots.add(profile_root)
                if (profile.lexeme_id in self.lexemes
                        and (profile.derivational_stem == "G"
                             or profile.root_class == "STRONG_G")):
                    self.g_lexeme_ids_by_root[profile_root].add(profile.lexeme_id)

        # D2: a root known to be verbal (v4 VERB profile, an Arnold TAM form,
        # an Arnold verb entry) whose project lexemes are all non-verbs gets a
        # service verb `lexsvc_<root>` instead of lending its NC reading to a
        # noun of the same root (sabeklun -> msabakča 'basket'). Service verbs
        # live OUTSIDE `self.lexemes`: principal parts, stems and minting read
        # that dict and must not see them. The citation comes from Arnold when
        # he has exactly one verb of the root, else `√root`.
        from maalula_glosser.arnold_verbs import arnold_verbs
        self.arnold_verbs = arnold_verbs(self.root)
        self.service_verbs: dict[str, dict[str, Any]] = {}
        self.verbal_roots: set[str] = set(self.verbal_profile_roots) | set(self.arnold_verbs)
        for matches in self.a2.values():
            for match in matches:
                if match.get("tam_canonical") and match.get("root"):
                    self.verbal_roots.add(nfc(match["root"]))

        # known stems per lexeme (for compositional matching)
        self.stems_by_lexeme: dict[str, set[str]] = defaultdict(set)
        for surface, matches in self.a2.items():
            for match in matches:
                self.stems_by_lexeme[match["lexeme_id"]].add(surface)
        for surface, analyses in self.a1.items():
            for analysis in analyses:
                if analysis.get("lemma_id"):
                    self.stems_by_lexeme[analysis["lemma_id"]].add(surface)

        # `stems_by_lexeme` is built from `a1`/`a2` (analyses), independent
        # of `self.lexemes`/forms above — a merge/delete edit applied there
        # does not touch it on its own. Left alone, `resolve_lexeme`'s
        # "analysed stem" tier would still hand back a merged-away or
        # deleted id: `_lemma_of` then reads `self.lexemes.get(lexeme_id)`,
        # which is gone, so the resolve silently degrades to an empty lemma
        # rather than the merge target — exactly the re-minting the merge
        # spec says must not happen. Re-key here, the same way forms were
        # re-keyed above.
        for source_id, row in edits["merges"].items():
            target_id = row["target_lexeme_id"]
            moved_stems = self.stems_by_lexeme.pop(source_id, None)
            if moved_stems:
                self.stems_by_lexeme[target_id] |= moved_stems
        for lexeme_id in edits["deletes"]:
            self.stems_by_lexeme.pop(lexeme_id, None)
        # The corpus analyses themselves carry the id as well: the trusted-
        # corpus stage builds its candidate straight from `analysis
        # ["lemma_id"]`, so without this a merged-away id comes back out of
        # `analyze` with an empty lemma (`činya` -> lex_… / None after a
        # merge). Follow merge chains to the final target.
        merged = {source: row["target_lexeme_id"]
                  for source, row in edits["merges"].items()}

        def final_target(lexeme_id: str) -> str:
            seen = set()
            while lexeme_id in merged and lexeme_id not in seen:
                seen.add(lexeme_id)
                lexeme_id = merged[lexeme_id]
            return lexeme_id

        if merged:
            for analyses in self.a1.values():
                for analysis in analyses:
                    if analysis.get("lemma_id") in merged:
                        analysis["lemma_id"] = final_target(analysis["lemma_id"])
            for matches in self.a2.values():
                for match in matches:
                    if match.get("lexeme_id") in merged:
                        match["lexeme_id"] = final_target(match["lexeme_id"])
            for source in [s for s in self.a1_by_lexeme if s in merged]:
                self.a1_by_lexeme[final_target(source)].extend(
                    self.a1_by_lexeme.pop(source))
        # `verbal_lexeme_ids` above is a narrower safety gate, not a resolve
        # tier: it is only ever consulted (search "verbal_lexeme_ids" in this
        # file) against a `lexeme_id` that already came out of a resolve —
        # and every resolve tier a merged/deleted id could reach
        # (`forms_by_surface`, `stems_by_lexeme`, `principal_parts_index`) is
        # now cleaned above, so a stale id lingering in `verbal_lexeme_ids`
        # itself is never looked up again. Left as-is deliberately.

    # ------------------------------------------------------------------
    def canonical(self, value: str, source: str = "canonical") -> str:
        return nfc(self.normalizer.canonical(value, source_system=source))

    def _lexical(self, lemma_id: str | None,
                surface: str | None = None) -> tuple[str | None, str]:
        return self.lexical.resolve(lemma_id, surface)

    def _build_gloss(self, lexical: str | None, tam: str | None,
                     features: dict, suffix_gloss: str | None,
                     placeholder: str | None = None) -> str:
        # A pending translation must not put "??" into the gloss: fall back to
        # the lemma as a visible placeholder and flag it via lexical_gloss_status.
        stem = lexical or gloss_placeholder(placeholder) or "UNK"
        parts = [stem + (f".{tam}" if tam else "")]
        if suffix_gloss:
            parts.append(suffix_gloss)
        return "-".join(parts)

    # ---- A-tier exact lookups ----------------------------------------
    # A piece that only ever occurs bound: `=le` is 3MS twenty-five times over
    # in the confirmed corpus, but no *word* is spelled `le`, so looking it up
    # the way a token is looked up finds nothing. Harvested from the
    # segmentations instead, which is where the knowledge actually lives.
    _ENCLITIC_MAX = 6

    @property
    def enclitics(self) -> dict[str, list[tuple[str, int]]]:
        """Bound form → its labels in the confirmed corpus, commonest first."""
        if self._enclitics is None:
            counts: dict[str, dict[str, int]] = {}
            hosts: dict[str, set[str]] = {}
            for surface, analyses in self.a1.items():
                for analysis in analyses:
                    segmentation = nfc(analysis.get("segmentation") or "")
                    gloss = nfc(analysis.get("gloss_clean") or "")
                    pieces = re.split(r"[-=]", segmentation)
                    labels = re.split(r"[-=]", gloss)
                    separators = re.findall(r"[-=]", segmentation)
                    if len(pieces) != len(labels) or not separators:
                        continue
                    for index, separator in enumerate(separators):
                        if separator != "=":
                            continue
                        form, label = pieces[index + 1], labels[index + 1]
                        if not form or not label or len(form) > self._ENCLITIC_MAX:
                            continue
                        # a grammatical label, not a name that followed a `=`:
                        # `=Siryōn Siryōn` is a place glossed as itself, and a
                        # label never repeats the letters it stands for
                        if not any(c.isdigit() or c.isupper() for c in label):
                            continue
                        if label.casefold() == form.casefold():
                            continue
                        counts.setdefault(form, {})
                        counts[form][label] = counts[form].get(label, 0) + 1
                        hosts.setdefault(form, set()).add(surface)
            self._enclitics = {
                form: sorted(labels.items(), key=lambda kv: -kv[1])
                # bound to more than one host, or it is part of that one word
                for form, labels in counts.items() if len(hosts[form]) > 1
            }
        return self._enclitics

    _PRINCIPAL_PART = re.compile(r"[\s/]+")
    _PAREN_GROUP = re.compile(r"\(([^()]+)\)")
    _MAX_PAREN_GROUPS = 3  # measured max is 1; capped so a future entry can't blow up the key set

    def _paren_variants(self, part: str) -> set[str]:
        """`app(i)` names one optional segment: the cartesian product of
        include/exclude per group, resolved before any schwa-stripping (a
        part with both a paren and a schwa needs the paren settled first, or
        the schwa pass has nothing to bite on). A stray unmatched `(` or `)`
        -- `(čṭamman` is one half of a group that a split on `/`/whitespace
        cut in two -- is stripped the same pass, and `(?)` is an editorial
        uncertainty marker, not a spelling, so any variant still carrying
        `?` is junk and dropped.

        Групп больше `_MAX_PAREN_GROUPS` — раскрытия нет вовсе, остаётся
        сырое написание: в словаре максимум одна группа на часть, и если
        когда-нибудь появится запись с четырьмя, лучше потерять её ключи,
        чем раздуть индекс на 16 вариантов. Потеря при этом честная:
        такой ключ просто не найдётся, и слово уйдёт в очередь к автору.
        """
        groups = list(self._PAREN_GROUP.finditer(part))
        if not groups or len(groups) > self._MAX_PAREN_GROUPS:
            variants = {part}
        else:
            variants = set()
            for combo in range(2 ** len(groups)):
                pieces = []
                cursor = 0
                for bit, group in enumerate(groups):
                    pieces.append(part[cursor:group.start()])
                    if combo & (1 << bit):
                        pieces.append(group.group(1))
                    cursor = group.end()
                pieces.append(part[cursor:])
                variants.add("".join(pieces))
        cleaned = {v.replace("(", "").replace(")", "") for v in variants}
        return {v for v in cleaned if v and "?" not in v}

    @property
    def principal_parts_index(self) -> dict[str, set[str]]:
        """Часть словарной формы → лексемы, у которых она есть.

        `abət/apt yabət` даёт ключи `abət`, `apt`, `yabət`; каждый ключ
        индексируется ещё и без швы, потому что шва здесь предсказуема
        (`awəʕ` ↔ основа `awʕ` в тексте). Ключ с несколькими владельцами
        оставляется как есть: решение об омонимах принимает автор, а не
        первый совпавший ключ (инвариант #7).

        Скобки помечают необязательный сегмент (`app(i) yapp(i)` находится
        и по `app`, и по `appi`); сырой ключ со скобками остаётся тоже, как
        безвредный балласт.

        **Единственный владелец здесь — не повод привязывать лексему.**
        Индекс говорит только о том, что написано в словарных формах, и
        ничего не знает о засвидетельствованных формах: `ʕayni` однозначно
        ведёт сюда к глаголу `ʕayn(i) yʕayn(i)`, хотя то же написание 5 раз
        засвидетельствовано как форма существительного `ʕayna` — и таких
        расхождений в словаре 26. Поэтому спрашивать индекс можно только
        после того, как формы (`forms_by_surface`, `stems_by_lexeme`) не
        дали ответа: порядок ярусов в `editor/lexeme_resolve.py` и есть
        защита, а не этот индекс.
        """
        if self._principal_parts is None:
            index: dict[str, set[str]] = defaultdict(set)
            for lexeme_id, lexeme in self.lexemes.items():
                lemma = nfc(lexeme.get("lemma_canonical") or "")
                if not lemma or lemma.startswith("["):
                    continue
                for part in self._PRINCIPAL_PART.split(lemma):
                    if not part:
                        continue
                    index[part].add(lexeme_id)
                    for variant in self._paren_variants(part):
                        index[variant].add(lexeme_id)
                        index[variant.replace("ə", "")].add(lexeme_id)
            self._principal_parts = dict(index)
        return self._principal_parts

    @property
    def proclitic_canonical_forms(self) -> dict[str, str]:
        """A proclitic's schwa-bearing spelling -> its bare form, wherever the
        lexicon independently attests both under the identical sense (`lə=`
        alongside `l=`, both glossed "to="): the schwa is predictable before
        whatever follows, not a separate morpheme, so the gloss shows the
        isolated form even though line 1 keeps whatever was actually written.

        Derived from `self.proclitics` (`data/grammar_kb/closed_class_v2.jsonl`)
        rather than hand-listed, so it tracks whatever the lexicon actually
        attests and nothing else: `bib=`/`bil=` differ from `b=` by a whole
        consonant, not a schwa, and correctly never collapse.
        """
        if self._proclitic_canonical is None:
            by_gloss: dict[str, set[str]] = {}
            for entry in self.proclitics:
                by_gloss.setdefault(entry["gloss"], set()).add(entry["form"])
            canonical: dict[str, str] = {}
            for forms in by_gloss.values():
                for form in forms:
                    if form.endswith("ə") and form[:-1] in forms:
                        canonical[form] = form[:-1]
                    elif form.startswith("ə") and form[1:] in forms:
                        canonical[form] = form[1:]
            self._proclitic_canonical = canonical
        return self._proclitic_canonical

    def _from_a1(self, surface: str) -> list[Candidate]:
        out = []
        lookup_surfaces = [surface]
        if surface.startswith("ʔ") and len(surface) > 1:
            lookup_surfaces.append(surface[1:])
        for lookup_surface in lookup_surfaces:
            for analysis in self.a1.get(lookup_surface, []):
                if not usable_gloss(analysis.get("gloss_clean")):
                    continue
                lemma_id = analysis.get("lemma_id")
                lexeme = self.lexemes.get(lemma_id) or {}
                root = nfc(lexeme.get("root_canonical") or "")
                # A ʔ-prefixed spelling may reuse an otherwise identical A1 row
                # only when the linked lexical root itself begins in ʔ.
                if lookup_surface != surface and not root.startswith("ʔ"):
                    continue
                lexical, status = self._lexical(lemma_id, lookup_surface)
                if analysis.get("authoritative_overlay"):
                    # Publication IGT is a resolved display analysis even before a
                    # dictionary lexeme link has been assigned.
                    status = "authoritative_igt"
                # Homonym disambiguation: prefer the reading most often attested in
                # the training corpus for this exact surface.
                attested = self.analysis_prior.get(
                    (lookup_surface, analysis["segmentation"], analysis["gloss_clean"]), 0)
                authority_score = analysis.get("authority_rank", 0)
                # A fused proclitic+pronoun (b=ē): rank its preposition sense by the
                # b= morpheme's overall distribution, not the handful of fused tokens,
                # so loc/in/ins lead and 'with' does not win on a small sample.
                seg = analysis["segmentation"]
                gloss_head = re.split(r"[=-]", analysis["gloss_clean"], maxsplit=1)[0]
                seg_head = re.split(r"[=-]", seg, maxsplit=1)[0]
                senses = self.prep_sense_prior.get(seg_head)
                if senses and "=" in seg and gloss_head.lower() in PREP_SENSES:
                    attested = senses.get(gloss_head, 0) / (sum(senses.values()) or 1)
                gloss = analysis["gloss_clean"]
                source_segmentation = analysis["segmentation"]
                # An SC row is not relabelled NC because another row of the same
                # word was NC: iṯken is SC in every text the author glossed, and
                # ōmar is SC by his ruling (02.10.2026: "это исключение, NC
                # mōmar"). That rule fired on these two words only.

                segmentation = source_segmentation
                # Visible segmentation follows the transcription. Older trusted
                # rows can themselves carry a restored initial ʔ; drop it when
                # the current surface does not spell it. Conversely, if the
                # user did spell ʔ and we reached an otherwise-identical A1 row
                # through the unprefixed alias, retain the written character.
                if (not surface.startswith("ʔ") and segmentation.startswith("ʔ")
                        and reconstruct(segmentation[1:]) == surface):
                    segmentation = segmentation[1:]
                elif (lookup_surface != surface and surface.startswith("ʔ")
                        and not segmentation.startswith("ʔ")
                        and segmentation.startswith(("a", "ā", "e", "ē", "i", "ī",
                                                     "o", "ō", "u", "ū"))):
                    segmentation = "ʔ" + segmentation

                # Same display rule for the plural allomorph: if an old row has
                # canonical -il/-al but this token literally spells -in/-an,
                # keep the written allomorph in line 2. This is deliberately
                # boundary-local and only fires when it exactly rebuilds surface.
                for canonical, written in (("-il", "-in"), ("-al", "-an")):
                    if canonical in segmentation:
                        trial = segmentation.replace(canonical, written, 1)
                        if reconstruct(trial) == surface:
                            segmentation = trial
                            break

                features = {"train_attestations": attested}
                evidence = [{
                    "kind": ("authoritative_publication_igt"
                             if analysis.get("authoritative_overlay")
                             else "trusted_corpus_analysis"),
                    "analysis_id": analysis.get("analysis_id"),
                    "source": analysis.get("source"),
                    **({"correction_id": analysis["correction_id"]}
                       if analysis.get("correction_id") else {}),
                }]
                lexeme = self.lexemes.get(lemma_id) or {}
                perfect = classify_prf_analysis(
                    lexeme, source_segmentation, gloss)
                if perfect:
                    subclass, stem = perfect
                    gloss = replace_prf_label(gloss, subclass)
                    features["tam"] = "PRF"
                    features["perfect_subclass"] = subclass
                    evidence.append({
                        "kind": "strong_stem_i_perfect_subclass",
                        "subclass": subclass,
                        "root": (lexeme or {}).get("root_canonical"),
                        "stem": stem,
                        "basis": "written_consonant_skeleton",
                    })
                out.append(Candidate(
                    surface=surface, segmentation=segmentation,
                    gloss=gloss, lemma_id=lemma_id,
                    lemma=lexeme.get("lemma_canonical"),
                    lexical_gloss=lexical, lexical_gloss_status=status,
                    features=features,
                    tier="A1", score=1.0 + attested + authority_score,
                    reconstruction_ok=reconstructs_to(segmentation, surface)[0],
                    reconstruction_trace=["A1 trusted corpus analysis"]
                    + reconstructs_to(segmentation, surface)[1],
                    evidence=evidence))
        return out
    def _from_a3(self, surface: str) -> list[Candidate]:
        out = []
        for cell in self.a3.get(surface, []):
            out.append(Candidate(
                surface=surface, segmentation=cell["segmentation"], gloss=cell["gloss"],
                lemma_id=None, lemma=cell.get("lexeme_or_root"),
                lexical_gloss=None, lexical_gloss_status="paradigm_cell",
                features=cell.get("features") or {}, tier="A3", score=0.95,
                reconstruction_ok=reconstructs_to(cell["segmentation"], surface)[0],
                reconstruction_trace=["A3 paradigm cell"],
                evidence=[{"kind": "paradigm_cell", "kb_cell_id": cell.get("kb_cell_id"),
                           "source": cell.get("source")}]))
        return out

    def _from_a0(self, surface: str) -> list[Candidate]:
        """Closed-class entries taken from the project's glossing standard."""
        out = []
        for entry in self.a0.get(surface, []):
            segmentation = entry.get("segmentation") or surface
            gloss = entry["gloss"]
            # A host entry is written "before=" / "with=" to show that it takes
            # a pronominal enclitic, but the "=" is notation, not a segment: the
            # separator is supplied by whatever attaches. Leaving it in made the
            # gloss carry one more separator than the segmentation, so every
            # composed form failed the alignment check.
            if entry.get("attachment") == "host":
                gloss = gloss.rstrip("=")
            # the lexical base is the first segment: mar-ōy=lə has lemma mar,
            # the MP allomorph -ōy and the construct clitic =lə are morphology
            lemma = re.split(r"[-=]", segmentation)[0] or entry["form"]
            out.append(Candidate(
                surface=surface, segmentation=segmentation, gloss=gloss,
                lemma_id=None, lemma=lemma,
                lexical_gloss=entry["gloss"], lexical_gloss_status="glossing_standard",
                features={"category": entry.get("category"),
                          "attachment": entry.get("attachment")},
                tier="A0", score=1.0,
                reconstruction_ok=reconstructs_to(segmentation, surface)[0],
                reconstruction_trace=[f"A0 glossing standard, {entry.get('section')}"],
                evidence=[{"kind": "glossing_standard", "entry_id": entry["entry_id"],
                           "section": entry.get("section"), "source": entry.get("source")}]))
        return out

    def _from_attested_form(self, surface: str) -> list[Candidate]:
        """A corpus-attested surface whose own morphology is not analysed.

        This is intentionally pure: unlike `_host_candidates`, it never falls
        through to Arnold.  The production stage called "attested form" must
        mean exactly that, so the following Arnold stage remains observable.
        """
        out = []
        for form in self.forms_by_surface.get(surface, [])[:1]:
            lemma_id = form.get("lexeme_id")
            lexical, status = self._lexical(lemma_id, surface)
            lemma_form = (self.lexemes.get(lemma_id) or {}).get("lemma_canonical")
            if not lexical:
                lexical = self.stem_glosses.get(skeleton(nfc(lemma_form or surface)))
                if lexical:
                    status = ("stem_gloss_" +
                              (self.stem_gloss_authority.get(
                                  skeleton(nfc(lemma_form or surface))) or "author_gold"))
            # The corpus only says this spelling belongs to the lexeme. When
            # the spelling *is* the citation form (barrōyṯ, xwōṯ) that is a
            # complete reading. When it is not (yraḥmell of raḥḥem, wazne of
            # wazna, mōrəl of mōra) the word is inflected and its morphology
            # simply was never analysed: a pointer to the lemma, not an
            # analysis. It must lose to any segmented reading (the possessive
            # wazn=e, a v4 object form) and never pass for MEDIUM.
            inflected = not self._is_citation_spelling(surface, lemma_form)
            out.append(Candidate(
                surface=surface, segmentation=surface,
                gloss=lexical or gloss_placeholder(lemma_form) or surface,
                lemma_id=lemma_id, lemma=lemma_form,
                lexical_gloss=lexical, lexical_gloss_status=status,
                features={"part_of_speech": self.lexeme_part_of_speech.get(lemma_id),
                          "part_of_speech_source": self.lexeme_pos_source.get(lemma_id),
                          "unanalysed_inflected": inflected},
                tier="C" if inflected else "B1",
                score=0.45 if inflected else 0.6, reconstruction_ok=True,
                reconstruction_trace=["attested form, no analysis on record"]
                + (["inflected spelling of the lemma: morphology not analysed"]
                   if inflected else []),
                evidence=[{"kind": "attested_form", "form_id": form.get("form_id")}]))
        return out

    def _from_h_dropped_host(self, surface: str) -> list[Candidate]:
        """P + demonstrative with its h lost: banna = b=hanna, mnōxa = mn=hōxa.

        Added after the staged search (see `analyze`) because such a surface
        is often itself a hit for something else (banna is also bann=a
        'I want her'), and the early stop would otherwise never let the
        demonstrative reading compete. In Muše Sanžar every banna before a
        noun (4 of 4) is b=hanna, so the demonstrative ranks first; the
        other reading stays as an alternative.
        """
        if not any(surface.startswith(e["form"]) for e in self.proclitics):
            return []
        out = []
        for entry in self.proclitics:
            form = entry["form"]
            rest = surface[len(form):]
            if not surface.startswith(form) or len(rest) < 3 \
                    or rest[0].lower() not in VOWELS:
                continue
            known = self._from_a0("h" + rest) or self._from_a1("h" + rest)
            for host in known[:1]:
                if not host.segmentation or not usable_gloss(host.gloss):
                    continue
                canonical = self.proclitic_canonical_forms.get(form, form)
                segmentation = f"{canonical}={host.segmentation}"
                gloss = f"{entry['gloss'].rstrip('=')}={host.gloss}"
                ok, trace = reconstructs_to(segmentation, surface)
                if not ok or not aligned(segmentation, gloss):
                    continue
                out.append(Candidate(
                    surface=surface, segmentation=segmentation, gloss=gloss,
                    lemma_id=host.lemma_id, lemma=host.lemma,
                    lexical_gloss=host.lexical_gloss,
                    lexical_gloss_status=host.lexical_gloss_status,
                    features={**host.features, "proclitic": entry["gloss"]},
                    tier="B1", score=0.95, reconstruction_ok=True,
                    reconstruction_trace=host.reconstruction_trace
                    + [f"proclitic {form}= before h-{rest}"] + trace,
                    evidence=host.evidence + [{"kind": "h_loss_after_proclitic"}]))
        return out

    def _restore_lost_h(self, candidate: Candidate, surface: str) -> Candidate:
        """``P=Vx`` -> ``P=hVx`` when ``hVx`` is a known h-initial word.

        Only for a host that starts with a vowel right after a proclitic of
        the closed-class inventory, and only when the h-form is itself an
        A0/A1 word (hanna, hōṯe, hōḏ, hōxa, hačč, hinn): anna/ōṯe/ōxa are
        what the proclitic left of them, not lexemes of their own. The
        author's journal is never rewritten.
        """
        if any(str(e.get("kind", "")).startswith("reusable_morphology")
               for e in candidate.evidence):
            return candidate
        segmentation = nfc(candidate.segmentation or "")
        parts = re.split(r"([-=])", segmentation)
        forms = {nfc(e["form"]).lower() for e in self.proclitics}
        if len(parts) < 3 or parts[1] != "=" or parts[0].lower() not in forms:
            return candidate
        host_segmentation = "".join(parts[2:])
        host_word = reconstruct(host_segmentation)
        # a pronominal clitic (b=ā "in=3fs") is not a demonstrative: hosts
        # of three letters and more only (anna, ōṯe, ōxa)
        if len(host_word) < 3 or host_word[0].lower() not in VOWELS:
            return candidate
        h_word = "h" + host_word
        known = self._from_a0(h_word) or self._from_a1(h_word)
        if not known:
            return candidate
        new_segmentation = f"{parts[0]}=h{host_segmentation}"
        ok, trace = reconstructs_to(new_segmentation, surface)
        if not ok:
            return candidate
        head = known[0]
        lemma, lemma_id = candidate.lemma, candidate.lemma_id
        if not (lemma or "").lower().startswith("h"):
            lemma, lemma_id = head.lemma, head.lemma_id
        return replace(candidate, segmentation=new_segmentation,
                       lemma=lemma, lemma_id=lemma_id,
                       reconstruction_ok=True,
                       reconstruction_trace=candidate.reconstruction_trace + trace
                       + [f"h of {h_word} restored after the proclitic"])

    @staticmethod
    def _is_citation_spelling(surface: str, lemma_form: str | None) -> bool:
        """Is ``surface`` one of the written citation forms of the lemma?

        Lemmas are stored as `amar yīmar`, `hačč/hačči`, `ayṯ(i) yayṯ(i)`,
        `cān/cōn/(i)ncōn`: any of the space/slash separated forms counts, with
        an optional parenthesised letter either present or absent.
        """
        if not lemma_form:
            return True          # nothing to compare with: do not demote

        def plain(value: str) -> str:
            # spelling noise, not morphology: ə, written/unwritten gemination
            # (xet ~ xett, ḥḥōḏ ~ ḥōḏ), vowel length, allophone marks
            value = unicodedata.normalize("NFD", fold_allophones(nfc(value)).lower())
            value = value.replace("\u0304", "").replace("ə", "")
            value = re.sub(r"(.)\1+", r"\1", unicodedata.normalize("NFC", value))
            return value.replace("-", "")

        target = plain(surface)
        for form in re.split(r"[\s/]+", nfc(lemma_form).lower()):
            if not form:
                continue
            variants = {re.sub(r"\(([^)]*)\)", r"\1", form),
                        re.sub(r"\([^)]*\)", "", form)}
            if target in {plain(v) for v in variants}:
                return True
        return False

    def _host_candidates(self, host: str) -> list[Candidate]:
        """Analyses of a proclitic's host, falling back to a bare attested form.

        A noun like lēlya is attested with no translated sense; refusing it as a
        host would lose the whole analysis, so it is admitted with the lemma as
        a visible placeholder and the missing gloss flagged, never as "??".
        """
        # The author's own reading of the host outranks every built source,
        # exactly as it does for the host standing alone (stage 1 of
        # `_candidate_stages`): a confirmed `wazn=e` must also be the host
        # of b=wazn=e, a confirmed `Lubnān` "Lebanon" the host of m=Lubnān.
        # Only a reading that spells this host literally: a row that maps a
        # different spelling onto it (xwōṯ confirmed as xwōṯ=l before a word
        # that swallowed the l) describes that context, not the host.
        confirmed = [c for c in self._from_reusable_morphology(host)
                     if c.segmentation and reconstructs_to(c.segmentation, host)[0]]
        if confirmed:
            return confirmed
        base = self._from_a0(host) or self._from_a1(host) or self._from_a2(host)
        if base:
            return base
        attested = self._from_attested_form(host)
        if attested:
            return attested
        # A form cited only in Arnold can still be the host of a proclitic or a
        # construct clitic: b=ḥeṣna, l=ṭuflō. Tried after the corpus, so it
        # never displaces better evidence.
        return self._from_lexicon_v3(host)

    # The transcriber is not always consistent with the emphatics or the k/ḳ
    # contrast: the same word turns up as ḥesna and ḥeṣna, kuṭṭaʕōyət and
    # ḳuṭṭaʕōyət. These are real phonemic distinctions, so a variant match is
    # only ever a last resort and is always flagged.
    SPELLING_VARIANTS = [tuple(pair) for pair in
                         RULES.get("spelling_variants", {}).get("pairs", [])]

    def _from_spelling_variant(self, surface: str) -> list[Candidate]:
        """Retry the lookup under a plausible respelling of the token."""
        seen = set()
        out = []
        for old, new in self.SPELLING_VARIANTS:
            if old not in surface:
                continue
            variant = surface.replace(old, new)
            if variant == surface or variant in seen:
                continue
            seen.add(variant)
            found = (self._from_a1(variant) or self._from_a0(variant)
                     or self._from_a2(variant) or self._host_candidates(variant))
            for candidate in found[:1]:
                if not candidate.segmentation or not usable_gloss(candidate.gloss):
                    continue
                # the segmentation must spell the token as it was written, not
                # as the variant it was read through: ʕiṣər is not ʕisər
                segmentation = candidate.segmentation
                if reconstruct(segmentation) == variant:
                    rebuilt, position = [], 0
                    for piece in re.split(r"([-=])", segmentation):
                        if piece in "-=":
                            rebuilt.append(piece)
                            continue
                        rebuilt.append(surface[position:position + len(piece)])
                        position += len(piece)
                    segmentation = "".join(rebuilt)
                    if reconstruct(segmentation) != surface:
                        continue
                out.append(Candidate(
                    surface=surface, segmentation=segmentation,
                    gloss=candidate.gloss, lemma_id=candidate.lemma_id,
                    lemma=candidate.lemma, lexical_gloss=candidate.lexical_gloss,
                    lexical_gloss_status=candidate.lexical_gloss_status,
                    features={**candidate.features, "spelling_variant": variant},
                    tier="D", score=0.2, reconstruction_ok=False,
                    reconstruction_trace=candidate.reconstruction_trace
                    + [f"read as {variant}, a spelling variant of {surface}"],
                    evidence=candidate.evidence
                    + [{"kind": "spelling_variant", "read_as": variant}],
                    uncertainties=[{"type": "spelling_variant_assumed",
                                    "reason": f"token spelled {surface}, "
                                              f"analysed as {variant}"}]))
        return out

    def _from_prosthetic_schwa(self, surface: str) -> list[Candidate]:
        """Retry the lookup with a word-initial ə dropped.

        `ənḥammel` is `nḥammel` said with nothing before it to carry the
        first consonant: the ə breaks up an otherwise unpronounceable
        n-ḥ- onset, but it is a phonetic prop, not a segment the lexicon
        or a human import ever contributes. Retried last, only once every
        stage above has already failed on the word exactly as written —
        a real lexical ə- must never be shadowed by this guess.
        """
        return self._initial_schwa_host_candidates(surface)[:1]

    @staticmethod
    def _initial_schwa_variant(surface: str) -> str | None:
        """The existing conservative əCC- condition, shared by host parsing."""
        if (len(surface) >= 3 and surface.startswith("ə")
                and surface[1] not in VOWELS and surface[2] not in VOWELS):
            return surface[1:]
        return None

    def _initial_schwa_host_candidates(self, surface: str) -> list[Candidate]:
        """Analyse a host once without a phonetic initial schwa.

        This is the host-level continuation of c355f43/23a9451.  It calls only
        bounded host/nominal/construct helpers, so a proclitic cannot recurse
        through the public analyzer indefinitely.
        """
        variant = self._initial_schwa_variant(surface)
        if variant is None:
            return []
        base = self._host_candidates(variant)
        nominal = self._from_nominal_suffix(variant)
        free = self._free_state_a(base, variant) if base else []
        # In Muše Sanjar the confirmed default is blōt-a village-FREE.  A more
        # weakly inferred F-FREE analysis may remain available, but it must not
        # precede that corpus-backed reading.
        composed = free + nominal + base
        if not composed:
            composed = self._from_construct_clitic(variant)
        out = []
        seen = set()
        for candidate in composed:
            if not candidate.segmentation or not usable_gloss(candidate.gloss):
                continue
            key = candidate.key()
            if key in seen:
                continue
            seen.add(key)
            out.append(Candidate(
                surface=surface,
                segmentation=candidate.segmentation,
                gloss=candidate.gloss,
                lemma_id=candidate.lemma_id,
                lemma=candidate.lemma,
                lexical_gloss=candidate.lexical_gloss,
                lexical_gloss_status=candidate.lexical_gloss_status,
                features={**candidate.features, "prosthetic_schwa": True},
                tier="D", score=0.2, reconstruction_ok=False,
                reconstruction_trace=candidate.reconstruction_trace
                + [f"read host as {variant}, dropping a prosthetic ə"],
                evidence=candidate.evidence
                + [{"kind": "prosthetic_schwa", "read_as": variant}],
                uncertainties=list(candidate.uncertainties)
                + [{"type": "prosthetic_schwa_assumed",
                    "reason": f"host spelled {surface}, analysed as {variant}"}]))
        return out

    def _from_nominal_suffix(self, surface: str) -> list[Candidate]:
        """Segment a documented nominal or adjectival ending off a known stem.

        Only fires when the stem is independently attested, so it cannot invent
        a noun. The stem is matched ignoring vowel length, because the paradigm
        alternates it: santūka is the free form behind santuk-ō chest-P.
        """
        out, seen = [], set()
        compound = [(e, "-".join(parts), lab) for e, parts, lab in COMPOUND_SUFFIXES]
        for ending, label in [(e, lab) for e, _p, lab in compound] + NOMINAL_SUFFIXES:
            if not surface.endswith(ending):
                continue
            stem = surface[: -len(ending)]
            if len(stem) < 3:
                continue
            entry = self.stem_index.get(skeleton(stem))
            if not entry:
                for key in alternation_keys(stem):
                    entry = self.stem_index_alt.get(key)
                    if entry:
                        break
            if not entry:
                continue
            # -in is syncretic: adjectival -MP.INDF or verbal present .PRS-MP.
            # There is no part-of-speech field in the corpus or the lexicon, so
            # the stem shape is the only signal. A verbal template (ō in the
            # first syllable, m- prefix) favours the verbal reading and its
            # absence favours the adjectival one, but as a score, not a veto:
            # both readings are offered so the disfavoured one stays visible for
            # review instead of being silently blocked (e.g. ġōly-in, an
            # adjective on a verbal-looking stem). The template-favoured reading
            # keeps its rank; the other is scored just below it.
            verbal = verbal_stem(stem) if ending == "in" else None
            score = 0.5
            if ending == "in" and label in (".PRS-MP", "-MP.INDF"):
                favoured = (label == ".PRS-MP") == (verbal is not None)
                if not favoured:
                    score = 0.45
            gloss_stem, source = entry
            confirmed = self.stem_glosses.get(skeleton(stem))
            if confirmed:
                gloss_stem = confirmed
                source = ("author-confirmed stem gloss"
                          if self.stem_gloss_authority.get(skeleton(stem))
                          == "author_gold" else "stem gloss from Arnold's German")
            gloss = f"{gloss_stem}{label}"
            written = next((parts for e, parts, lab in compound
                            if e == ending and lab == label), ending)
            segmentation = f"{stem}-{written}"
            if (segmentation, gloss) in seen or not aligned(segmentation, gloss):
                continue
            if not reconstructs_to(segmentation, surface)[0]:
                continue
            seen.add((segmentation, gloss))
            out.append(Candidate(
                surface=surface, segmentation=segmentation, gloss=gloss,
                lemma_id=None, lemma=gloss_stem,
                lexical_gloss=None,
                lexical_gloss_status="nominal_paradigm",
                features={"ending": ending, "label": label},
                tier="B2", score=score, reconstruction_ok=True,
                reconstruction_trace=[f"nominal ending -{ending} = {label}, "
                                      f"stem matched against {source}"]
                + ([verbal] if ending == "in" and verbal else []),
                evidence=[{"kind": "nominal_paradigm",
                           "source": "Glossing_updated §Существительное/§Прилагательное",
                           "stem_evidence": source}]))
        return out

    def _root_licensed_feminine(
            self,
            base: list[Candidate],
            surface: str,
    ) -> list[Candidate]:
        """Refine a known nominal lexeme by its canonical root.

        Productive ṯ/č is F only when the corresponding consonant does not
        belong to the lexical root. Thus sažər-ṯ-a (√sžr) is licensed,
        whereas xronṯ-a (√xrnṯ) remains internally unsplit. General feminine
        plurals are P-F: -ō-ṯ- / -yō-ṯ-.
        """
        patterns = (
            ("yōṯa", "yō-ṯ-a", "-P-F-FREE", "ṯ"),
            ("ōṯa", "ō-ṯ-a", "-P-F-FREE", "ṯ"),
            ("ṯa", "ṯ-a", "-F-FREE", "ṯ"),
            ("ča", "č-a", "-F-FREE", "č"),
        )

        out: list[Candidate] = []
        for candidate in base[:3]:
            if candidate.tier == "A1" or not candidate.segmentation:
                continue
            if "-" in candidate.segmentation or "=" in candidate.segmentation:
                continue
            if _HAS_TAM.search(candidate.gloss or ""):
                continue

            lexeme = self.lexemes.get(candidate.lemma_id) or {}
            # Corpus-bound candidates carry the canonical lexeme root. Arnold
            # D-candidates have no lemma_id but do carry the dictionary root in
            # features. Both are lexical evidence for the same decision.
            root = nfc(
                lexeme.get("root_canonical")
                or candidate.features.get("root")
                or ""
            )
            if not root:
                continue

            for ending, written, gloss_tail, marker in patterns:
                if not surface.endswith(ending):
                    continue
                if marker in root:
                    continue
                stem = surface[:-len(ending)]
                if len(stem) < 2:
                    continue

                lexical = (
                    candidate.lexical_gloss
                    or candidate.gloss
                    or gloss_placeholder(candidate.lemma)
                )
                if not usable_gloss(lexical):
                    continue

                segmentation = f"{stem}-{written}"
                gloss = f"{lexical}{gloss_tail}"
                if not aligned(segmentation, gloss):
                    continue
                ok, rule_trace = reconstructs_to(segmentation, surface)
                if not ok:
                    continue

                out.append(Candidate(
                    surface=surface,
                    segmentation=segmentation,
                    gloss=gloss,
                    lemma_id=candidate.lemma_id,
                    lemma=candidate.lemma,
                    lexical_gloss=candidate.lexical_gloss,
                    lexical_gloss_status=candidate.lexical_gloss_status,
                    features={
                        **candidate.features,
                        "root": root,
                        "root_licensed_feminine": True,
                    },
                    tier=candidate.tier,
                    score=candidate.score + 0.03,
                    reconstruction_ok=candidate.reconstruction_ok,
                    reconstruction_trace=candidate.reconstruction_trace + [
                        f"root {root} excludes ṯ/č; analyse the surface marker as F"
                    ] + rule_trace,
                    evidence=candidate.evidence + [{
                        "kind": "root_licensed_feminine",
                        "root": root,
                    }],
                    uncertainties=list(candidate.uncertainties),
                ))
                break
        return out

    def _from_same_lexeme_enclitic(
            self, surface: str, base: list[Candidate]
    ) -> list[Candidate]:
        """Extend the documented be.loc pattern to an attested sibling form.

        This is the safe piece worth importing from the old lexicon-v3 idea:
        do not strip an ending because it merely *looks* familiar.  Require
        (1) the whole target surface to be attested, (2) the same lexeme to have
        a confirmed sibling analysis with an overt pronominal enclitic, and
        (3) the target ending to be a documented member of that same overt PNG
        series.  The sibling supplies the host segmentation and lexical gloss;
        the grammar KB supplies only the new PNG label.
        """
        attested_ids = {
            candidate.lemma_id for candidate in base
            if candidate.lemma_id and any(
                evidence.get("kind") == "attested_form"
                for evidence in candidate.evidence
            )
        }
        if not attested_ids:
            return []

        png_by_form: dict[str, list[tuple[str, dict]]] = defaultdict(list)
        for clitic in self.clitics:
            if clitic.get("paradigm") != "object_accusative":
                continue
            form = clitic.get("form") or ""
            gloss = (clitic.get("gloss") or "").lstrip("=")
            if form and gloss:
                png_by_form[form].append((gloss, clitic.get("features") or {}))

        base_by_id = {candidate.lemma_id: candidate for candidate in base
                      if candidate.lemma_id in attested_ids}
        out: list[Candidate] = []
        seen: set[tuple] = set()
        for lemma_id in attested_ids:
            anchor = base_by_id[lemma_id]
            for sibling in self.a1_by_lexeme.get(lemma_id, []):
                segmentation = sibling.get("segmentation") or ""
                gloss = sibling.get("gloss_clean") or ""
                if segmentation.count("=") != 1 or gloss.count("=") != 1:
                    continue
                host_seg, sibling_clitic = segmentation.rsplit("=", 1)
                host_gloss, sibling_label = gloss.rsplit("=", 1)
                # Shape alone cannot establish the clitic's function. This
                # cross-series analogy is licensed only for the documented
                # predicate, not arbitrary nominal or verbal suffix paradigms.
                if host_gloss not in NON_NOMINAL_FREE_GLOSSES:
                    continue
                if sibling_label not in {
                        label for label, _ in png_by_form.get(sibling_clitic, [])}:
                    continue
                if (not aligned(segmentation, gloss)
                        or not reconstructs_to(
                            segmentation, sibling.get("surface_canonical") or "")[0]):
                    continue
                host_surface = reconstruct(host_seg)
                if not host_surface or not surface.startswith(host_surface):
                    continue
                target_clitic = surface[len(host_surface):]
                if not target_clitic or target_clitic not in png_by_form:
                    continue
                for target_label, target_features in png_by_form[target_clitic]:
                    target_segmentation = f"{host_seg}={target_clitic}"
                    target_gloss = f"{host_gloss}={target_label}"
                    if not aligned(target_segmentation, target_gloss):
                        continue
                    ok, trace = reconstructs_to(target_segmentation, surface)
                    if not ok:
                        continue
                    key = (target_segmentation, target_gloss, lemma_id)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(Candidate(
                        surface=surface, segmentation=target_segmentation,
                        gloss=target_gloss, lemma_id=lemma_id,
                        lemma=anchor.lemma, lexical_gloss=anchor.lexical_gloss,
                        lexical_gloss_status=anchor.lexical_gloss_status,
                        features={**anchor.features, **target_features,
                                  "same_lexeme_enclitic": True},
                        tier="B1", score=0.64, reconstruction_ok=True,
                        reconstruction_trace=anchor.reconstruction_trace + [
                            "same lexeme has a confirmed overt pronominal enclitic series",
                            f"sibling {segmentation} licenses ={target_clitic}",
                        ] + trace,
                        evidence=anchor.evidence + [{
                            "kind": "same_lexeme_enclitic_paradigm",
                            "analysis_id": sibling.get("analysis_id"),
                            "sibling_segmentation": segmentation,
                        }]))
        return out

    # Object-clitic series a confirmed verbal cell may license siblings in.
    # Only the l-series for now: the accusative series is made of one- and
    # two-letter endings (=a, =e, =u, =un, =ax) that collide with subject
    # agreement and with the nominal free state / possessives, so it needs its
    # own review before it is switched on here.
    VERBAL_ENCLITIC_SERIES = ("l_object",)
    # Bare =l is three-way ambiguous (1S.IO, 3MP.IO and the DOM marker, see
    # test_nc_plural_terminal_l_prefers_dom).  A sibling cannot settle which,
    # so =l never licenses anything; as a target it is still offered, DOM
    # first, exactly as _from_nc_root_plural_l_object ranks it.
    VERBAL_ENCLITIC_AMBIGUOUS_FORMS = frozenset({"l"})

    def _from_verbal_enclitic_sibling(self, surface: str) -> list[Candidate]:
        """License verb + object clitic from a confirmed cell of the same host.

        The verbal counterpart of ``_nominal_possessive_sibling_base``. A
        confirmed amel=le 'say.SC=3MS' (author journal or trusted corpus) fixes
        the host ``amel``, its segmentation, its TAM gloss and its lexeme;
        amel=lun, amel=la, ... are that same host with another member of the
        same clitic series. Without this, amellun fell through to the
        productive NC root model, which reads amel as ʔamel (ʔ>Ø) and bound
        the word to √ʔml čʔammal.

        Deliberately conservative, like the nominal rule:
        * the sibling reading is exact, has exactly one '=', and its clitic is
          a documented member of the SAME series as the target clitic (only
          the l-series is enabled, see VERBAL_ENCLITIC_SERIES; the host before
          =l- differs from the host before the accusative series anyway,
          amel=le vs amr=e), and bare =l is never used on either side;
        * the host gloss carries a TAM label and the reading has a lexeme link;
        * the sibling reconstructs its own surface and the host rebuilds the
          target's host;
        * the author's journal is consulted first and the corpus only when the
          journal has nothing: the corpus keeps several readings of one cell
          side by side (amelle is on record as say.PST=3MS, say.PST=3MS.IO and
          tell.PST=3MS.IO at once), and an author-confirmed row is exactly the
          decision between them;
        * within one source every usable sibling must agree on lexeme, host
          segmentation, host gloss and object role (plain object vs .IO), or
          nothing is produced;
        * the target keeps the sibling's role: a plain =3MS sibling licenses
          plain =3MP, a =3MS.IO sibling licenses =3MP.IO; a bare =l target is
          offered as DOM first and as the syncretic 1S/3MP pronouns after.
        Exact evidence for the target itself is found in earlier stages and
        always wins; here the candidate only has to outrank the productive NC
        root reading of the same stage (B1 0.86 at most), hence 0.90.
        """
        series: dict[str, dict[str, list[dict]]] = {}
        for clitic in self.clitics:
            paradigm = clitic.get("paradigm")
            if paradigm in self.VERBAL_ENCLITIC_SERIES and clitic.get("form"):
                series.setdefault(paradigm, {}).setdefault(
                    clitic["form"], []).append(clitic)

        out: list[Candidate] = []
        seen: set[tuple] = set()
        for paradigm, by_form in series.items():
            for target_form, target_clitics in by_form.items():
                if not surface.endswith(target_form):
                    continue
                host_surface = surface[:-len(target_form)]
                if len(host_surface) < 2:
                    continue

                matches: dict[tuple, dict] = {}
                for source in ("reusable_morphology", "trusted_corpus_analysis"):
                    if matches:
                        break
                    for sibling_form, sibling_clitics in by_form.items():
                        if (sibling_form == target_form
                                or sibling_form in self.VERBAL_ENCLITIC_AMBIGUOUS_FORMS):
                            continue
                        sibling_surface = host_surface + sibling_form
                        if source == "reusable_morphology":
                            readings = [
                                (nfc(c.segmentation or ""), nfc(c.gloss or ""),
                                 c.lemma_id, source)
                                for c in self._from_reusable_morphology(sibling_surface)
                            ]
                        else:
                            readings = [
                                (nfc(a.get("segmentation") or ""),
                                 nfc(a.get("gloss_clean") or ""),
                                 a.get("lemma_id"), source)
                                for a in self.a1.get(sibling_surface, [])
                            ]
                        roles = {}
                        for clitic in sibling_clitics:
                            for label, role, _bonus in self._nc_object_readings(clitic):
                                roles[label.casefold()] = role
                        for segmentation, gloss, lemma_id, origin in readings:
                            if not lemma_id:
                                continue
                            if segmentation.count("=") != 1 or gloss.count("=") != 1:
                                continue
                            host_seg, seen_form = segmentation.rsplit("=", 1)
                            host_gloss, seen_label = gloss.rsplit("=", 1)
                            if seen_form != sibling_form:
                                continue
                            role = roles.get(seen_label.casefold())
                            if role is None:
                                continue
                            if not _HAS_TAM.search(host_gloss) or not usable_gloss(host_gloss):
                                continue
                            if not aligned(segmentation, gloss):
                                continue
                            if not reconstructs_to(segmentation, sibling_surface)[0]:
                                continue
                            if not reconstructs_to(host_seg, host_surface)[0]:
                                continue
                            key = (lemma_id, host_seg, host_gloss, role)
                            matches.setdefault(key, {
                                "source": origin, "surface": sibling_surface,
                                "segmentation": segmentation, "gloss": gloss})

                # Conflicting siblings are not resolved by this productive rule.
                if len(matches) != 1:
                    continue
                (lemma_id, host_seg, host_gloss, role), sibling = next(iter(matches.items()))
                lexeme = self.lexemes.get(lemma_id) or {}
                lexical, status = self._lexical(lemma_id, surface)
                ambiguous = target_form in self.VERBAL_ENCLITIC_AMBIGUOUS_FORMS
                targets: list[tuple[str, str, float]] = []
                if ambiguous:
                    # Terminal =l is first of all the differential object
                    # marker, as _from_nc_root_plural_l_object already ranks
                    # it; the syncretic 1S/3MP pronouns follow as alternatives.
                    targets.append(("DOM", "DOM", 0.91))
                for clitic in target_clitics:
                    for label, target_role, _bonus in self._nc_object_readings(clitic):
                        if target_role != role:
                            continue
                        targets.append((label, target_role, 0.88 if ambiguous else 0.90))
                for label, target_role, score in targets:
                    segmentation = f"{host_seg}={target_form}"
                    gloss = f"{host_gloss}={label}"
                    if not aligned(segmentation, gloss):
                        continue
                    ok, trace = reconstructs_to(segmentation, surface)
                    if not ok:
                        continue
                    key = (segmentation, gloss, lemma_id)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(Candidate(
                        surface=surface, segmentation=segmentation, gloss=gloss,
                        lemma_id=lemma_id,
                        lemma=lexeme.get("lemma_canonical"),
                        lexical_gloss=lexical, lexical_gloss_status=status,
                        features={"verbal_enclitic_sibling": True,
                                  "object": label, "object_role": target_role,
                                  "clitic_series": paradigm},
                        tier="B1", score=score, reconstruction_ok=True,
                        reconstruction_trace=[
                            f"confirmed sibling {sibling['segmentation']} / "
                            f"{sibling['gloss']} ({sibling['source']})",
                            f"same host, same {paradigm} series: ={target_form}",
                        ] + trace,
                        evidence=[{
                            "kind": "verbal_enclitic_sibling",
                            "source": sibling["source"],
                            "source_surface": sibling["surface"],
                            "source_segmentation": sibling["segmentation"],
                            "source_gloss": sibling["gloss"],
                        }]))
        return out

    def _free_state_a(self, base: list[Candidate], surface: str) -> list[Candidate]:
        """Offer the free-state reading of a noun that came back unsegmented.

        Glossing_updated makes -a the free-state ending, and the corpus
        segments it: safr-a voyage-FREE, ḏ̣aġṭ-a pressure-FREE. A source that
        stores the citation form whole leaves it unsegmented, which was the
        single largest group of segmentation misses.

        Only bare nominal glosses qualify — a gloss already carrying a category
        label or a TAM is not a noun in its free state.
        """
        if not surface.endswith("a") or len(surface) < 4:
            return []
        out = []
        # a name is not a noun in its free state: Turcīya stays whole
        if surface[:1].isupper():
            return []
        for candidate in base[:2]:
            # A1 is a human annotation of running text: if an annotator left
            # the form whole, that decision stands. The rule is for the weaker
            # sources, which store citation forms and cannot know.
            if candidate.tier == "A1" or not candidate.segmentation:
                continue
            # Tried on composed analyses too (b=baḥra as b=baḥr-a) and it cost
            # 1.0pp: after a proclitic the last component is often not a noun
            # in its free state. Restricted to a form that came back whole.
            if "-" in candidate.segmentation or "=" in candidate.segmentation:
                continue
            gloss = candidate.gloss or ""
            # A dot inside a lexical gloss is not morphology:
            # grape.molasses must still yield teps-a / grape.molasses-FREE.
            # Reject only an actual finite-TAM label here. Structural labels
            # are already blocked by the uppercase check below.
            if not gloss or _HAS_TAM.search(gloss) or any(c.isupper() for c in gloss):
                continue
            # If the lexeme is independently classified as verbal, a final -a
            # cannot be promoted to nominal FREE merely because the lexical
            # gloss happens to be lower-case.  Dictionary TAM fills only
            # otherwise-unclassified lexemes, so confirmed noun/adjective
            # morphology has already won any source conflict.
            if (candidate.lemma_id
                    and self.lexeme_part_of_speech.get(candidate.lemma_id) == "verb"):
                continue
            lexical_head = (candidate.lexical_gloss or gloss).split("=", 1)[0]
            if lexical_head in NON_NOMINAL_FREE_GLOSSES:
                continue
            segmentation = f"{surface[:-1]}-a"
            composed = f"{gloss}-FREE"
            if not aligned(segmentation, composed) or \
                    not reconstructs_to(segmentation, surface)[0]:
                continue
            out.append(Candidate(
                surface=surface, segmentation=segmentation, gloss=composed,
                lemma_id=candidate.lemma_id, lemma=candidate.lemma,
                lexical_gloss=candidate.lexical_gloss,
                lexical_gloss_status=candidate.lexical_gloss_status,
                features={**candidate.features, "free_state": True},
                # ranked just above the unsegmented reading: the corpus writes
                # these segmented, and leaving the citation form whole was the
                # single largest group of segmentation misses
                tier=candidate.tier, score=candidate.score + 0.01,
                # inherit the base's caveats: re-reading the ending does not
                # make a dictionary-derived or respelled analysis any surer,
                # and claiming reconstruction here would hide that the base
                # analysed a different string
                reconstruction_ok=candidate.reconstruction_ok,
                reconstruction_trace=candidate.reconstruction_trace
                + ["final -a read as the free state"],
                evidence=candidate.evidence
                + [{"kind": "free_state_a",
                    "source": "Glossing_updated §Свободная форма"}],
                uncertainties=list(candidate.uncertainties)))
        return out

    def _nominal_free_base(self, stem: str,
                           allow_final_devoicing: bool = False) -> Candidate | None:
        """Return a securely attested ``stem-a / lexical-FREE`` noun base.

        This is the gate for productive possessive segmentation.  It requires
        an exact citation form in the ordinary form inventory, rejects lexemes
        independently attested as finite verbs, and reuses the existing
        free-state rule rather than guessing a POS from an Arnold translation.
        """
        citation_stems = [stem]
        # Before =xun/=xen a final voiced stop may surface voiceless
        # (tarb-a but tarp=xun).  This is the only extra lookup licensed here;
        # the visible segmentation still preserves the actually written tarp-.
        if allow_final_devoicing and stem.endswith("p"):
            citation_stems.append(f"{stem[:-1]}b")

        for citation_stem in citation_stems:
            citation = f"{citation_stem}a"
            forms = self.forms_by_surface.get(citation, [])
            dictionary_only = False
            if forms:
                allowed_ids = {
                    form.get("lexeme_id") for form in forms
                    if form.get("lexeme_id")
                    and form.get("lexeme_id") not in self.verbal_lexeme_ids
                }
                if not allowed_ids:
                    continue
            else:
                # No corpus form at all: an exact Arnold headword can still
                # license the citation, at a deliberately weaker tier --
                # keṭəʕṯa (Arnold-only, root ḳṭʕ1) must still be able to
                # license keṭəʕ-ṯ=e. _host_candidates() already falls
                # through to _from_lexicon_v3() for an exact key match; the
                # checks below (TAM gloss, uppercase, -FREE ending) are the
                # same ones a corpus citation must also pass, so this never
                # licenses something that merely looks nominal.
                if citation not in self.lexicon_v3:
                    continue
                allowed_ids = None
                dictionary_only = True
            bases = self._host_candidates(citation)
            # A corpus or standard source may already carry the desired FREE
            # segmentation.  Reuse it directly; _free_state_a deliberately does
            # not rewrite A1 annotations.  The citation may already contain
            # internal nominal morphology: šaġəl-ṯ-a must license šaġəl-ṯ=e,
            # rather than being flattened to *šaġəlṯ-a.
            free = [
                candidate for candidate in bases
                if candidate.segmentation
                and candidate.segmentation.endswith("-a")
                and (candidate.gloss or "").endswith("-FREE")
                and reconstruct(candidate.segmentation[:-2]) == citation_stem
            ]
            # A whole citation can itself expose productive feminine morphology
            # by its lexical root (ṭaləʕṯa √ṭlʕ -> ṭaləʕ-ṯ-a). Feed that refined
            # citation into the possessive transfer below.
            free += self._root_licensed_feminine(bases, citation)
            free += self._free_state_a(bases, citation)
            for candidate in free:
                if (not candidate.segmentation
                        or not candidate.segmentation.endswith("-a")
                        or reconstruct(candidate.segmentation[:-2]) != citation_stem
                        or not (candidate.gloss or "").endswith("-FREE")):
                    continue
                if (allowed_ids is not None and candidate.lemma_id
                        and candidate.lemma_id not in allowed_ids):
                    continue
                if dictionary_only:
                    candidate = replace(
                        candidate,
                        features={**candidate.features,
                                  "dictionary_only_citation": True})
                return candidate
        return None

    def _nominal_possessive_sibling_base(
            self,
            stem: str,
            target_ending: str,
            direct_ids: set[str],
    ) -> Candidate | None:
        """License a nominal possessive cell from a confirmed sibling reading.

        Example: a confirmed zitt=e / grandfather=3MS reading licenses
        zitt=ah / grandfather=1P for the same stem, because =e and =ah are
        both members of the same nominal possessive paradigm. The sibling
        may come from either trusted source the rest of the analyzer already
        treats as A1-equivalent: the corpus index (self.a1) or the
        author-confirmed reusable-morphology journal (self.reusable_morphology
        / self._from_reusable_morphology) -- a form promoted into the journal
        (e.g. a human IGT reading with no corpus lemma link) never reaches
        self.a1, so both sources must be checked or this rule silently misses
        exactly the case it exists for.

        This is deliberately conservative: the sibling must be an exact
        reading, reconstruct exactly, belong to a non-verbal lexeme, and all
        usable siblings must agree on lexical identity and host gloss.
        """
        matches: dict[tuple[str, str, str], Candidate] = {}

        # The confirmed reading of the *same* cell counts too: a confirmed
        # žitt=aḥ licenses the stem žitt, and that stem must then license
        # žittaḥ itself wherever it turns up as a host (l=žitt=aḥ,
        # b=žitt=aḥ), not only its siblings (žitt=e). Standalone žittaḥ is
        # unaffected -- the journal / A1 stage settles it long before this
        # productive rule is consulted.
        for sibling_ending, sibling_label in NOMINAL_POSSESSIVE_SUFFIXES:
            sibling_surface = f"{stem}{sibling_ending}"

            sibling_readings: list[tuple[str, str, str | None]] = []
            for analysis in self.a1.get(sibling_surface, []):
                sibling_readings.append((
                    nfc(analysis.get("segmentation") or ""),
                    nfc(analysis.get("gloss_clean") or ""),
                    analysis.get("lemma_id"),
                ))
            # Author-confirmed readings (e.g. a promoted human IGT line that
            # was never linked to a corpus lemma_id) live in the journal, not
            # in self.a1 -- this is exactly how zitt=ah reaches this rule.
            for candidate in self._from_reusable_morphology(sibling_surface):
                sibling_readings.append((
                    nfc(candidate.segmentation or ""),
                    nfc(candidate.gloss or ""),
                    candidate.lemma_id,
                ))

            for segmentation, gloss, lemma_id in sibling_readings:
                if not lemma_id or lemma_id in self.verbal_lexeme_ids:
                    continue
                if direct_ids and lemma_id not in direct_ids:
                    continue
                if segmentation.count("=") != 1 or gloss.count("=") != 1:
                    continue

                host_seg, seen_ending = segmentation.rsplit("=", 1)
                host_gloss, seen_label = gloss.rsplit("=", 1)

                if seen_ending != sibling_ending:
                    continue
                if seen_label != sibling_label.lstrip("="):
                    continue
                if reconstruct(host_seg) != stem:
                    continue
                if not usable_gloss(host_gloss) or _HAS_TAM.search(host_gloss):
                    continue
                if not aligned(segmentation, gloss):
                    continue
                if not reconstructs_to(segmentation, sibling_surface)[0]:
                    continue

                lexeme = self.lexemes.get(lemma_id) or {}
                candidate = Candidate(
                    surface=stem,
                    segmentation=host_seg,
                    gloss=host_gloss,
                    lemma_id=lemma_id,
                    lemma=lexeme.get("lemma_canonical") or stem,
                    lexical_gloss=host_gloss,
                    lexical_gloss_status="trusted_corpus_possessive_sibling",
                    features={
                        "possessive_sibling_base": True,
                        "source_ending": sibling_ending,
                        "source_label": seen_label,
                    },
                    tier="B1",
                    score=0.62,
                    reconstruction_ok=True,
                    reconstruction_trace=[
                        f"confirmed sibling {segmentation} / {gloss}",
                        "same nominal possessive paradigm",
                    ],
                    evidence=[{
                        "kind": "nominal_possessive_sibling",
                        "source_surface": sibling_surface,
                        "source_segmentation": segmentation,
                        "source_gloss": gloss,
                    }],
                )
                matches[(lemma_id, host_seg, host_gloss)] = candidate

        # Conflicting siblings must not be resolved by this productive rule.
        if len(matches) != 1:
            return None
        return next(iter(matches.values()))

    def _from_nominal_possessive(self, surface: str) -> list[Candidate]:
        """Productively segment the finite, overt noun-possessive paradigm.

        The rule is intentionally narrow.  It fires only when the corresponding
        free citation ``stem+a`` is an attested non-verbal lexeme.  If the
        possessed surface is itself in the form inventory, it must point to the
        same lexeme as the citation.  Thus lawn=e can rescue the weak whole-form
        lawne reading, while an unrelated verb ending in -e is untouched.
        """
        out: list[Candidate] = []
        direct_ids = {
            form.get("lexeme_id") for form in self.forms_by_surface.get(surface, [])
            if form.get("lexeme_id")
        }

        def add_candidate(stem: str, segmentation: str, gloss_ending: str,
                          features: dict[str, Any], trace: str,
                          allow_final_devoicing: bool = False,
                          split_feminine_plural: bool = False) -> None:
            base = self._nominal_free_base(
                stem, allow_final_devoicing=allow_final_devoicing)
            sibling_based = False
            if base is None and gloss_ending.startswith("="):
                base = self._nominal_possessive_sibling_base(
                    stem, str(features.get("ending") or ""), direct_ids)
                sibling_based = base is not None
            if base is None:
                return
            if direct_ids and base.lemma_id and base.lemma_id not in direct_ids:
                return
            # An Arnold-only citation (no corpus form at all, e.g. keṭəʕṯa)
            # is weaker evidence than either a corpus citation or a confirmed
            # sibling -- score it below both, never displacing them.
            dictionary_only = bool(base.features.get("dictionary_only_citation"))
            stem_gloss = (
                (base.gloss or "")
                if sibling_based
                else (base.gloss or "").removesuffix("-FREE")
            )
            if not stem_gloss:
                return
            internal_trace: list[str] = []
            if (split_feminine_plural
                    and (base.lemma or "") not in LEXICALIZED_YOT_PLURAL_LEMMAS):
                for surface_tail, plural_segment in (("yōṯ", "yō"),
                                                     ("ōṯ", "ō")):
                    if not stem.endswith(surface_tail):
                        continue
                    lexical_stem = stem[:-len(surface_tail)]
                    if len(lexical_stem) < 3:
                        break
                    clitic_tail = segmentation[len(stem):]
                    segmentation = (
                        f"{lexical_stem}-{plural_segment}-ṯ{clitic_tail}")
                    lexical_gloss = base.lexical_gloss or re.sub(
                        r"(?:-(?:P|F))+$", "", stem_gloss)
                    stem_gloss = f"{lexical_gloss}-P-F"
                    features = {**features, "number": "plural",
                                "gender": "feminine"}
                    internal_trace.append(
                        f"feminine plural -{plural_segment}-ṯ- = -P-F-")
                    break

            # Preserve internal host morphology from either source:
            # citation shagal-t-a -> shagal-t=e, or a confirmed sibling
            # zitt=e -> zitt=ah.
            if (sibling_based and base.segmentation
                    and reconstruct(base.segmentation) == stem
                    and segmentation.startswith(stem)):
                segmentation = base.segmentation + segmentation[len(stem):]
            elif (base.segmentation and base.segmentation.endswith("-a")
                    and reconstruct(base.segmentation[:-2]) == stem
                    and segmentation.startswith(stem)):
                segmentation = base.segmentation[:-2] + segmentation[len(stem):]

            gloss = f"{stem_gloss}{gloss_ending}"
            if not aligned(segmentation, gloss):
                return
            ok, rule_trace = reconstructs_to(segmentation, surface)
            if not ok:
                return
            out.append(Candidate(
                surface=surface, segmentation=segmentation, gloss=gloss,
                lemma_id=base.lemma_id, lemma=base.lemma,
                lexical_gloss=base.lexical_gloss,
                lexical_gloss_status=base.lexical_gloss_status,
                features={**base.features, **features, "possessive": True},
                # Same evidence tier as a weak attested-form fallback, but one
                # point above its 0.60 score.  No A-tier or analysed candidate is
                # displaced, and the global stage order remains unchanged.
                # Arnold-only citations (no corpus form) sit below both a
                # normal citation and a confirmed sibling.
                tier="B1",
                score=(0.62 if sibling_based
                       else 0.58 if dictionary_only
                       else 0.61),
                reconstruction_ok=True,
                reconstruction_trace=base.reconstruction_trace
                + internal_trace + [trace] + rule_trace,
                evidence=base.evidence + [{
                    "kind": "nominal_possessive_paradigm",
                    "source": "Glossing_updated §Поссессивные",
                }]))

        # Possessed plurals must be tried first because several end in the same
        # short strings as singular possessives.
        for ending, plural_segment, clitic_segment, gloss_ending in \
                NOMINAL_PLURAL_POSSESSIVES:
            if not surface.endswith(ending) or len(surface) <= len(ending) + 2:
                continue
            stem = surface[:-len(ending)]
            segmentation = f"{stem}-{plural_segment}={clitic_segment}"
            add_candidate(
                stem, segmentation, gloss_ending,
                {"number": "plural", "ending": ending},
                f"possessed plural ending -{plural_segment}={clitic_segment}")

        for ending, label in NOMINAL_POSSESSIVE_SUFFIXES:
            if not surface.endswith(ending) or len(surface) <= len(ending) + 2:
                continue
            stem = surface[:-len(ending)]
            segmentation = f"{stem}={ending}"
            add_candidate(
                stem, segmentation, label,
                {"ending": ending, "label": label.lstrip("=")},
                f"nominal possessive ={ending} = {label}",
                allow_final_devoicing=ending in {"xun", "xen"},
                split_feminine_plural=True)

        # Multiple paths can converge under vowel-length normalisation.  Keep a
        # single copy of each visible analysis without changing global ranking.
        unique: dict[tuple, Candidate] = {}
        for candidate in out:
            unique.setdefault(candidate.key(), candidate)
        return list(unique.values())

    @staticmethod
    def _weak_unsegmented_nominal_slot(candidates: list[Candidate]) -> bool:
        """Whether a local nominal rescue may compete with settled evidence.

        Adding any alternative lowers HIGH to MEDIUM in ``analyze``.  Therefore
        possessive rescue is offered only against weak whole-form fallbacks,
        never beside A-tier evidence or an already segmented analysis.
        """
        return bool(candidates) and all(
            candidate.tier in {"B1", "B2", "C", "D"}
            and candidate.segmentation
            and "-" not in candidate.segmentation
            and "=" not in candidate.segmentation
            for candidate in candidates
        )

    def _possessive_a(self, surface: str) -> list[Candidate]:
        """Offer the =3FS reading of a final -a whenever the stem is known."""
        ending, label = POSSESSIVE_A
        if not surface.endswith(ending):
            return []
        stem = surface[: -len(ending)]
        if len(stem) < 3:
            return []
        entry = self.stem_index.get(skeleton(stem))
        if not entry:
            return []
        gloss_stem = self.stem_glosses.get(skeleton(stem)) or entry[0]
        segmentation, gloss = f"{stem}={ending}", f"{gloss_stem}{label}"
        if not aligned(segmentation, gloss) or \
                not reconstructs_to(segmentation, surface)[0]:
            return []
        return [Candidate(
            surface=surface, segmentation=segmentation, gloss=gloss,
            lemma_id=None, lemma=gloss_stem, lexical_gloss=None,
            lexical_gloss_status="nominal_paradigm",
            features={"ending": ending, "label": label},
            # the lowest tier and the lowest score, so it can never become the
            # best reading: its job is to make the ambiguity visible and keep
            # the token out of HIGH, not to displace real evidence. Tier C
            # would not do — C sorts above D, and it stole the dictionary
            # reading of every noun ending in -a.
            tier="D", score=0.01, reconstruction_ok=True,
            reconstruction_trace=["final -a read as the 3FS possessive"],
            evidence=[{"kind": "possessive_a",
                       "source": "Glossing_updated §Поссессивные"}])]

    @staticmethod
    def _scope_key(row: dict) -> tuple | None:
        """The (document, sentence, token, surface) a scoped correction binds to,
        or None when the row carries no document coordinates."""
        doc = row.get("document_id")
        sent = row.get("sentence_id")
        idx = row.get("token_index")
        if doc is None or sent is None or idx is None:
            return None
        return (str(doc), str(sent), int(idx),
                nfc(row.get("original_token") or row.get("surface_canonical") or ""))

    def _correction_candidate(self, surface: str, row: dict, *, legacy: bool,
                              kind: str) -> Candidate:
        segmentation = row.get("segmentation") or surface
        features = {"author_confirmed": True}
        uncertainties = []
        if legacy:
            features["legacy_scope"] = True
            uncertainties.append({
                "type": "legacy_scope",
                "reason": "author-confirmed for a specific context but without "
                          "document coordinates; applies to this surface but is "
                          "flagged for review rather than overriding globally"})
        # A stored lemma string is never trusted directly (the citation form is
        # only ever read from the live lexicon) — and the lexeme may since have
        # been merged or deleted via lexicon_edits, in which case self.lexemes
        # no longer has it and this falls back to no lemma rather than a
        # dangling reference.
        lemma_id = row.get("lemma_id")
        lemma = self.lexemes.get(lemma_id, {}).get("lemma_canonical") if lemma_id else None
        if lemma is None:
            lemma_id = None
        return Candidate(
            surface=surface, segmentation=segmentation, gloss=row["gloss"],
            lemma_id=lemma_id, lemma=lemma,
            # the meaning of the stem, not the whole gloss line: `say`, not
            # `say.SC=3ms` (it is what the «Значение» dialog offers to edit)
            lexical_gloss=_gloss_lexical_head(row["gloss"]) or row["gloss"],
            lexical_gloss_status="author_confirmed",
            features=features, tier="A1", score=99.0,
            reconstruction_ok=reconstructs_to(segmentation, surface)[0],
            reconstruction_trace=[f"author-confirmed {kind}"],
            uncertainties=uncertainties,
            evidence=[{"kind": kind,
                       "correction_id": row.get("correction_id")
                       or row.get("addition_id"),
                       "scope": row.get("scope"),
                       "context": row.get("context")}])

    def remember_morphology(self, row: dict) -> None:
        """Index an author's source and reconstructed spellings identically."""
        keys = {surface_key(self.canonical(s)) for s in morphology_surfaces(row)}
        for key in keys - {""}:
            current = self.reusable_morphology[key]
            current[:] = [old for old in current
                          if old.get("addition_id") != row.get("addition_id")]
            current.append(row)
        hosts = self.__dict__.setdefault("reusable_hosts", defaultdict(list))
        for host_row in self._reusable_host_rows(row):
            key = surface_key(self.canonical(host_row["surface_canonical"]))
            current = hosts[key]
            current[:] = [old for old in current
                          if old.get("addition_id") != host_row.get("addition_id")]
            current.append(host_row)

    def has_confirmed_reading(self, written: str,
                              allow_joined: bool = True) -> str | None:
        """Cheap test: does the journal hold a row for this phrase?

        Returns the spelling to analyse -- the phrase as written, or the
        same with its spaces/underties closed up (the editor often records
        `b ʕÁrabeṯ` as surface `bʕÁrabeṯ`) -- or None. Used by the sentence
        path to try a multiword phrase as one unit
        (`gloss_text.apply_confirmed_spans`) without analysing every window.
        """
        written = nfc(written)
        joined = re.sub(r"[\s‿]+", "", written) if allow_joined else written
        for spelling in dict.fromkeys((written, joined)):
            if self.reusable_morphology.get(surface_key(self.canonical(spelling))):
                return spelling
        return None

    def _reusable_host_rows(self, row: dict) -> list[dict]:
        """What a confirmed ``P=host`` reading says about ``host`` alone.

        Peels up to two proclitics of the closed-class inventory off the
        front of the confirmed segmentation (w=l=ḥarīm-a gives l=ḥarīm-a and
        ḥarīm-a). The host must still be a word: at least three letters and a
        lexical gloss part -- b=ē "LOC=3ms" teaches nothing about ``ē``.
        """
        segmentation = nfc(row.get("segmentation") or "")
        gloss = nfc(row.get("gloss") or "")
        forms = {nfc(entry["form"]).lower()
                 for entry in getattr(self, "proclitics", [])}
        out: list[dict] = []
        for _depth in range(2):
            seg_parts = re.split(r"([-=])", segmentation)
            gloss_parts = re.split(r"([-=])", gloss)
            if (len(seg_parts) < 3 or len(seg_parts) != len(gloss_parts)
                    or seg_parts[1] != "=" or gloss_parts[1] != "="
                    or seg_parts[0].lower() not in forms):
                break
            segmentation = "".join(seg_parts[2:])
            gloss = "".join(gloss_parts[2:])
            host = reconstruct(segmentation)
            if len(host) < 3 or not has_lexical_part(gloss):
                break
            out.append({**row,
                        "addition_id": f"{row.get('addition_id')}#host{_depth + 1}",
                        "surface_canonical": host, "surface_original": host,
                        "original_surfaces": [],
                        "segmentation": segmentation, "gloss": gloss,
                        "host_of": row.get("addition_id")})
        return out

    def _from_reusable_morphology(self, surface: str) -> list[Candidate]:
        """Reusable lexeme-level readings, applied to the surface globally.

        Newest first: the file is append-only, so a form corrected twice has
        both readings on record, and the one the author wrote last is the one
        they meant. The earlier reading stays as an alternative.
        """
        out = []
        key = surface_key(self.canonical(surface))
        # A row confirmed under one word-initial-ʔ spelling must still be
        # found from the other: the corpus and the editor both waver between
        # ʔamelle and amelle for the same word, and only one spelling may
        # ever have been explicitly recorded as an alias. Mirrors _from_a1's
        # own alias lookup, which already does this for the corpus tier.
        lookup_keys = [key]
        if surface.startswith("ʔ") and len(surface) > 1:
            lookup_keys.append(surface_key(self.canonical(surface[1:])))
        elif not surface.startswith("ʔ"):
            lookup_keys.append(surface_key(self.canonical("ʔ" + surface)))
        seen_rows: set[str] = set()
        index = self.reusable_morphology
        if not any(self.reusable_morphology.get(k) for k in lookup_keys):
            # no row for this very spelling: fall back to what a confirmed
            # proclitic reading taught about it as a host
            index = getattr(self, "reusable_hosts", {})
        for lookup_key in lookup_keys:
            for row in reversed(index.get(lookup_key, [])):
                row_id = row.get("addition_id") or id(row)
                if row_id in seen_rows:
                    continue
                seen_rows.add(row_id)
                candidate = self._correction_candidate(
                    surface, row, legacy=False, kind="reusable_morphology")
                segmentation = candidate.segmentation or ""
                # Visible segmentation follows what was actually typed: drop a
                # stored initial ʔ the current surface does not spell, or
                # restore one the current surface does spell but the stored
                # segmentation (recorded from the other alias) omits.
                if (not surface.startswith("ʔ") and segmentation.startswith("ʔ")
                        and reconstruct(segmentation[1:]) == surface):
                    segmentation = segmentation[1:]
                elif (surface.startswith("ʔ") and not segmentation.startswith("ʔ")
                        and reconstruct("ʔ" + segmentation) == surface):
                    segmentation = "ʔ" + segmentation
                if segmentation != candidate.segmentation:
                    candidate = replace(candidate, segmentation=segmentation)
                reconstructed = reconstruct(candidate.segmentation or "")
                target = row.get("surface_canonical") or ""
                # The segmentation reconstructs the isolated form. A separately
                # recorded author mapping licenses its different source spelling.
                if (surface_key(self.canonical(reconstructed)) == surface_key(self.canonical(target))
                        and lookup_key in {surface_key(self.canonical(s)) for s in morphology_surfaces(row)}
                        and aligned(candidate.segmentation or "", candidate.gloss or "")):
                    candidate.reconstruction_ok = True
                    if reconstructed != surface:
                        candidate.features["reconstructed_surface"] = reconstructed
                        candidate.reconstruction_trace.append(
                            f"author-confirmed surface mapping: {surface} -> {reconstructed}")
                elif reconstructs_to(candidate.segmentation or "", surface)[0]:
                    # The ʔ-normalization above already made this reconstruct
                    # the literal query surface directly.
                    candidate.reconstruction_ok = True
                out.append(candidate)
        return out

    # The normative tables print grammatical labels lower-case. Uppercase the
    # TAM, the PNG bundles and the closed structural labels, but never a lexical
    # word: šwy's gloss "do" stays "do" (the verb), not "DO" (the object marker,
    # which does not occur in these subject tables).
    _V4_TAM = {"pst", "prs", "prf", "sbjv", "ipv", "infc", "prfc"}
    _V4_STRUCT = {"intf", "neg", "hd", "free", "cns", "indf", "def", "exist",
                  "dom", "num", "aff", "emp", "prsv", "recp", "refl"}
    _V4_PNG = re.compile(r"^[123]?(m|f|c)?(s|p)$|^(mp|fp|ms|fs)$", re.I)

    def _v4_gloss(self, gloss_original: str) -> str:
        out = []
        for part in re.split(r"([-=.])", nfc(gloss_original)):
            low = part.lower()
            if low in self._V4_TAM or low in self._V4_STRUCT \
                    or self._V4_PNG.match(part):
                out.append(part.upper())
            else:
                out.append(part)
        return "".join(out)

    def _from_paradigm_v4(self, surface: str) -> list[Candidate]:
        """Exact normative/irregular paradigm cells for this surface.

        The weak and irregular verbs (ṯō-le, zīl-in, kaʕy-ōla, n-allex) and the
        pseudoverbs (batt=ax) that the compositional path never generates. Only
        an exact cell whose segmentation rebuilds the token is offered.
        """
        if self._paradigm_v4 is None:
            return []
        out, seen = [], set()
        for cell in self._paradigm_v4.lookup_surface(surface):
            gloss = self._v4_gloss(cell["gloss_original"])
            for segmentation in cell["surface_variants"]:
                segmentation = nfc(segmentation)
                if "(" in segmentation:
                    segmentation = segmentation.replace("(", "").replace(")", "")
                key = (segmentation, gloss)
                if key in seen or not usable_gloss(gloss):
                    continue
                if not aligned(segmentation, gloss) \
                        or not reconstructs_to(segmentation, surface)[0]:
                    continue
                seen.add(key)
                out.append(Candidate(
                    surface=surface, segmentation=segmentation, gloss=gloss,
                    lemma_id=None, lemma=cell.get("lexeme_heading"),
                    lexical_gloss=None, lexical_gloss_status="paradigm_v4",
                    features={**(cell.get("features") or {}),
                              "lexeme": cell.get("lexeme_heading")},
                    # MEDIUM, not HIGH: the normative paradigm and the corpus can
                    # disagree on a form (mišw-an PRS in the table, PRF in the
                    # corpus), so an exact cell fills a gap and is offered for
                    # review but never overrides the corpus at HIGH.
                    tier="B1", score=0.82, reconstruction_ok=True,
                    reconstruction_trace=["exact normative/irregular paradigm cell"],
                    evidence=[{"kind": "EXACT_NORMATIVE_CELL",
                               "cell_id": cell.get("cell_id"),
                               "source": cell.get("source"),
                               "locator": cell.get("locator")}]))
        return out

    def _from_arnold_tables(self, surface: str) -> list[Candidate]:
        """A cell of Arnold's own M paradigms, read exactly as printed.

        The third-weak verb and its object suffixes, which nothing else in the
        pipeline reaches: `lexeme_profiles` admits strong triliterals only, and
        the corpus has the odd IIIy form but no paradigm. Only what the tables
        print — a cell is a fact about one word, not a rule about a class.

        The cell's own mark rides along in the evidence, so a reading resting on
        a form the project could not confirm says so where a human will see it.
        """
        from maalula_glosser.morphology import arnold_tables

        out = []
        for segmentation, gloss, status, where in self._arnold_tables.get(surface, []):
            if not usable_gloss(gloss) or not aligned(segmentation, gloss):
                continue
            ok, trace = reconstructs_to(segmentation, surface)
            if not ok:
                continue
            out.append(Candidate(
                surface=surface, segmentation=segmentation, gloss=gloss,
                lemma_id=None, lemma=None, lexical_gloss=None,
                lexical_gloss_status="arnold_table",
                features={"source": "arnold_table"},
                tier=arnold_tables.TIER,
                score=arnold_tables.SCORE.get(status, 0.72),
                reconstruction_ok=True,
                reconstruction_trace=[f"Arnold paradigm cell ({where})"] + trace,
                evidence=[{"kind": "ARNOLD_PARADIGM_CELL",
                           "locator": where, "checked": status}]))
        return out

    def _from_token_correction(self, surface: str,
                               context: dict | None = None) -> list[Candidate]:
        """Author-confirmed readings scoped to a token.

        A scoped correction only fires when ``context`` names the same
        (document_id, sentence_id, token_index) it was recorded for, so two
        identical surfaces in one document can be corrected differently and the
        same surface in another document is untouched. A legacy correction (no
        coordinates) is still offered for its surface but flagged legacy_scope
        and forced to review, never a silent global override.
        """
        out = []
        if context:
            key = self._scope_key({
                "document_id": context.get("document_id"),
                "sentence_id": context.get("sentence_id"),
                "token_index": context.get("token_index"),
                "original_token": context.get("original_token") or surface,
                "surface_canonical": surface})
            for row in self.token_corrections_scoped.get(key, []):
                out.append(self._correction_candidate(
                    surface, row, legacy=False, kind="token_correction"))
        out += [self._correction_candidate(surface, row, legacy=True,
                                           kind="token_correction")
                for row in self.token_corrections_legacy.get(surface, [])]
        return out

    def _from_capitalised_direct(self, surface: str) -> list[Candidate]:
        """Exact trusted lookup for a token differing only in initial case.

        This belongs with direct evidence, before productive composition.  With
        the old stage order, sentence-initial ``Waybin`` was compositionally
        misread as ``Wayb-in / on-MP.INDF`` and the exact lower-case corpus
        analysis ``w-ayb-in / PST-be-MP`` was never reached.
        """
        if not surface[:1].isupper():
            return []
        lowered = surface[:1].lower() + surface[1:]
        return (self._from_a0(lowered) + self._from_a1(lowered)
                + self._from_a3(lowered) + self._from_a2(lowered))

    def _from_lexicon_v3(self, surface: str) -> list[Candidate]:
        """Forms cited in Arnold, glossed from his own abbreviations.

        Last resort. Where the entry's German sense has been rendered in
        English it fills the lexical slot; otherwise the slot holds the
        dictionary headword, which the missing-gloss rule keeps out of HIGH.
        Either way the German travels in the evidence for review.
        """
        out = []
        for entry in self.lexicon_v3.get(surface, [])[:2]:
            segmentation = entry.get("proposed_segmentation") or surface
            english = self.entry_glosses.get(entry.get("lexicon_id"))
            placeholder = english or gloss_placeholder(entry.get("headword")) or surface
            gloss = (entry.get("proposed_gloss") or "?").replace("?", placeholder, 1)
            if not reconstructs_to(segmentation, surface)[0] or \
                    not aligned(segmentation, gloss) or not usable_gloss(gloss):
                continue
            out.append(Candidate(
                surface=surface, segmentation=segmentation, gloss=gloss,
                lemma_id=None, lemma=entry.get("headword"),
                lexical_gloss=english,
                lexical_gloss_status=("arnold_sense_translated" if english
                                      else "arnold_sense_untranslated"),
                features={"arnold_label": entry.get("arnold_label"),
                          "root": entry.get("root")},
                tier="D", score=0.3, reconstruction_ok=True,
                reconstruction_trace=["Arnold working lexicon, gloss read off "
                                      "the dictionary abbreviation"],
                evidence=[{"kind": "arnold_lexicon_v3", "page": entry.get("page"),
                           "label": entry.get("arnold_label"),
                           "german": entry.get("german"),
                           "authority": "low_derived"}],
                uncertainties=[] if english else
                [{"type": "derived_from_dictionary",
                  "reason": f"German sense not translated: "
                            f"{entry.get('german', '')[:70]}"}]))
        return out

    def _from_construct_clitic(self, surface: str) -> list[Candidate]:
        """Peel the construct-head clitic =l / =lə off a known host.

        Glossing_updated, "Клитика сопряжённой формы": =l =HD marks the head of
        a noun phrase before an immediately following definite dependent, as in
        ġann=l Mar Sarces "the garden of Mar Sarces". Requires a host the
        lexicon already knows, so it cannot invent a split.
        """
        out = []
        for cut, shown in (("il", "l"), ("əl", "l"), ("lə", "lə"), ("l", "l")):
            if not surface.endswith(cut) or len(surface) - len(cut) < 3:
                continue
            literal = surface[: -len(cut)]
            # ġannəl is ġann + epenthetic ə + l: look the host up without the
            # ə. The segmentation keeps the isolated clitic =l; exact surface
            # reconstruction records schwa insertion as a licensed rule.
            for host in dict.fromkeys((literal.rstrip("ə"), literal)):
                if len(host) < 3:
                    continue
                # The free-state -a drops in the construct: ešn-a year.F-FREE
                # stands behind b=ešn=əl "in the year of". Look the free form up
                # and use its stem, without the -FREE segment.
                found = self._host_candidates(host)
                free_state = False
                if not found:
                    citation = host + "a"
                    citation_candidates = self._host_candidates(citation)
                    # A corpus/standard entry may already segment citation -a.
                    # A root-licensed feminine goes first: the construct of
                    # kīmča is kīm-č=l, of ʕaylṯa ʕayl-ṯ=l -- the -ṯ/-č stays
                    # a morpheme when only the free-state -a drops.
                    found = [candidate for candidate
                             in self._root_licensed_feminine(citation_candidates, citation)
                             if (candidate.gloss or "").endswith("FREE")]
                    found += [candidate for candidate in citation_candidates
                              if (candidate.gloss or "").endswith("FREE")
                              # mōr-a is glossed person-MS in the standard;
                              # its -a drops in the construct just the same
                              or ((candidate.segmentation or "").endswith("-a")
                                  and (candidate.gloss or "").endswith("-MS"))]
                    # Arnold stores many citation nouns whole. Apply the same
                    # documented free-state analysis used by the ordinary
                    # nominal path before dropping -a in the construct.
                    if not found:
                        found = self._free_state_a(citation_candidates, citation)
                    free_state = bool(found)
                for candidate in found[:1]:
                    if not candidate.segmentation or not usable_gloss(candidate.gloss):
                        continue
                    if free_state:
                        stem_seg = candidate.segmentation.rsplit("-", 1)[0]
                        stem_gloss = candidate.gloss.rsplit("-", 1)[0]
                        if not stem_seg or not stem_gloss or \
                                not candidate.gloss.endswith(("FREE", "-MS")):
                            continue
                        candidate = Candidate(
                            surface=host, segmentation=stem_seg, gloss=stem_gloss,
                            lemma_id=candidate.lemma_id, lemma=candidate.lemma,
                            lexical_gloss=candidate.lexical_gloss,
                            lexical_gloss_status=candidate.lexical_gloss_status,
                            features=candidate.features, tier=candidate.tier,
                            reconstruction_ok=True,
                            reconstruction_trace=candidate.reconstruction_trace
                            + ["free-state -a drops in the construct"],
                            evidence=candidate.evidence)
                    seg = f"{candidate.segmentation}={shown}"
                    gloss = f"{candidate.gloss}=HD"
                    ok, trace = reconstructs_to(seg, surface)
                    if not ok or not aligned(seg, gloss):
                        continue
                    out.append(Candidate(
                        surface=surface, segmentation=seg, gloss=gloss,
                        lemma_id=candidate.lemma_id, lemma=candidate.lemma,
                        lexical_gloss=candidate.lexical_gloss,
                        lexical_gloss_status=candidate.lexical_gloss_status,
                        features={**candidate.features, "construct": True},
                        tier="B1", score=0.65, reconstruction_ok=True,
                        reconstruction_trace=candidate.reconstruction_trace
                        + [f"construct clitic ={cut} = HD"] + trace,
                        evidence=candidate.evidence
                        + [{"kind": "construct_head_clitic", "form": cut}]))
        return out

    def _as_proper_noun(self, surface: str, token: str) -> list[Candidate]:
        """Capitalised and still unknown after every real analyser → PN fallback.

        Capitalisation alone never pre-empts lexical, corpus, Arnold, v4,
        compositional or spelling evidence. If all of those fail, PN is still
        offered (including sentence-initially), never at HIGH.
        """
        if not token[:1].isupper() or len(surface) < 2:
            return []
        lowered = surface[:1].lower() + surface[1:]
        if lowered != surface and (self.a0.get(lowered) or self.a1.get(lowered)
                                   or self.a2.get(lowered)):
            return []
        # being listed as a form with no gloss of its own is not evidence that
        # the capitalised token is that word
        if lowered != surface and self.forms_by_surface.get(lowered):
            if any((c.gloss or "").lower() != lowered.lower()
                   for c in self._host_candidates(lowered)):
                return []
        return [Candidate(
            surface=surface, segmentation=surface, gloss="PN",
            lemma_id=None, lemma=token, lexical_gloss=None,
            lexical_gloss_status="proper_noun_guess",
            features={"category": "proper_noun"},
            tier="C", score=0.4, reconstruction_ok=True,
            reconstruction_trace=["capitalised token absent from the lexicon"],
            evidence=[{"kind": "proper_noun_heuristic"}],
            uncertainties=[{"type": "proper_noun_guess",
                            "reason": "glossed pn only because it is capitalised "
                                      "and unknown"}])]

    def _from_past_prefix(self, surface: str) -> list[Candidate]:
        """w- PST, restricted to the copula/pseudoverb slot.

        The corpus uses w- with ōb, ōṯ and ayb only — 83 tokens, 6 distinct
        forms, all of them "be" or "EXIST". It is not a general past prefix, so
        applying it freely would split every word that begins with w.
        """
        if not surface.startswith("w") or len(surface) < 4:
            return []
        host = surface[1:]
        if not any(entry.get("category") == "pseudoverb" and host.startswith(entry["form"])
                   for entries in self.a0.values() for entry in entries):
            return []
        base = (self._from_a0(host) or self._from_a1(host) or self._from_a2(host)
                or self._from_compositional(host))
        out = []
        for candidate in base[:2]:
            if not candidate.segmentation or not usable_gloss(candidate.gloss):
                continue
            seg = f"w-{candidate.segmentation}"
            gloss = f"PST-{candidate.gloss}"
            ok, rule_trace = reconstructs_to(seg, surface)
            if not ok or not aligned(seg, gloss):
                continue
            out.append(Candidate(
                surface=surface, segmentation=seg, gloss=gloss,
                lemma_id=candidate.lemma_id, lemma=candidate.lemma,
                lexical_gloss=candidate.lexical_gloss,
                lexical_gloss_status=candidate.lexical_gloss_status,
                features={**candidate.features, "tam": "PST"},
                tier="B1", score=0.78, reconstruction_ok=True,
                reconstruction_trace=candidate.reconstruction_trace
                + ["prefix w- = PST (pseudoverb slot)"] + rule_trace,
                evidence=candidate.evidence + [{"kind": "past_prefix", "form": "w-"}]))
        return out

    def _from_assimilated_proclitic(self, surface: str) -> list[Candidate]:
        """Proclitics that leave no letter of their own on the surface.

        Two licensed processes, both confirmed by the author and mirrored in
        Arnold:

        * ``l=`` assimilates completely to a following consonant, so the only
          trace is gemination: ṭ-ṭuflō is l-ṭuflō, r-roḥla is l-roḥla. Arnold
          p.330 √hnn1 documents the mirror image — the demonstrative assimilating
          to a preceding l-, giving lanna from l- + hanna.
        * ``b=`` devoices to p before a voiceless consonant and is written p-.
          There is no separate p- proclitic in the inventory, so a p- that is
          followed by a known host is a b=.

        The segmentation keeps the surface exactly as written (ṭ=ṭuflō, not
        l=ṭuflō) so it still reconstructs; only the gloss names the underlying
        preposition.
        """
        if surface in self.forms_by_surface or surface in self.a0 \
                or surface in self.a1 or surface in self.a2:
            return []
        if len(surface) < 4:
            return []
        first, rest = surface[0], surface[1:]
        candidates: list[tuple[str, str]] = []
        if first not in VOWELS and rest[:1] == first:
            candidates.append(("to=", "l assimilated to the following consonant"))
        if first == "p" and rest[:1] not in VOWELS:
            candidates.append(("in=", "b devoiced to p before a voiceless consonant"))
        if not candidates:
            return []

        out = []
        for gloss_prefix, why in candidates:
            for host in (rest, "h" + rest):
                # h- of the demonstrative is lost after a proclitic (lanna),
                # so the host may need it put back before lookup
                base = self._host_candidates(host) or self._from_construct_clitic(host)
                for candidate in base[:1]:
                    if not candidate.segmentation or not usable_gloss(candidate.gloss):
                        continue
                    seg = f"{first}={candidate.segmentation}"
                    gloss = f"{gloss_prefix.rstrip('=')}={candidate.gloss}"
                    if not reconstructs_to(seg, surface)[0] or not aligned(seg, gloss):
                        continue
                    out.append(Candidate(
                        surface=surface, segmentation=seg, gloss=gloss,
                        lemma_id=candidate.lemma_id, lemma=candidate.lemma,
                        lexical_gloss=candidate.lexical_gloss,
                        lexical_gloss_status=candidate.lexical_gloss_status,
                        features={**candidate.features, "assimilated_proclitic": True},
                        tier="B1", score=0.55, reconstruction_ok=True,
                        reconstruction_trace=candidate.reconstruction_trace + [why],
                        evidence=candidate.evidence
                        + [{"kind": "assimilated_proclitic", "rule": why}]))
        return out

    def _from_proclitic(self, surface: str, depth: int = 2) -> list[Candidate]:
        """Peel proclitics off the front of an attested host.

        Guarded twice, because b=, l=, m= and w= are single letters: a word that
        is itself attested is never re-split, and the remainder must be a form
        the lexicon already knows. Up to ``depth`` proclitics may stack, which
        running text needs — w-l-ḥarīma is and= plus to= plus the noun.
        """
        if surface in self.forms_by_surface or surface in self.a0 \
                or surface in self.a1 or surface in self.a2:
            return []
        out = []
        for entry in self.proclitics:
            form = entry["form"]
            host = surface[len(form):]
            if len(host) < 3 or not surface.startswith(form):
                continue
            base = self._host_candidates(host)
            prosthetic = self._initial_schwa_host_candidates(host)
            if prosthetic:
                # The literal ə-host remains available after these candidates;
                # the underlying segmentation is the useful first suggestion.
                base = prosthetic + base
            # A bounded nominal-possessive host (žesm=e, mʕalmōn=e) is never
            # itself an attested whole form -- that is exactly why it needs
            # the productive possessive rule -- so it must be offered even
            # when _host_candidates(host) came back empty. Unlike the
            # unattested-fragment guard below, this is not a spurious split:
            # _from_nominal_possessive already requires the free citation
            # stem+a (or a confirmed sibling) to be securely attested.
            possessive_host = self._from_nominal_possessive(host)
            if possessive_host:
                seen_possessive = {(c.segmentation, c.gloss) for c in base}
                for cand in possessive_host:
                    if (cand.segmentation, cand.gloss) not in seen_possessive:
                        base.append(cand)
                        seen_possessive.add((cand.segmentation, cand.gloss))
            # The whole-form lookup returns an attested host unsegmented, so
            # b=baḥra came back b=baḥra "in=sea" instead of b=baḥr-a
            # "loc=sea-FREE": the noun's free state was lost the moment it hosted
            # a proclitic, though standalone baḥra segments fine. Offer the
            # host's nominal segmentation too, ranked first as it is standalone.
            # Only when the host is itself an attested form (base non-empty):
            # segmenting a deep-stripped fragment invents spurious splits like
            # mʕalman-ō -> m=ʕal=man-ō.
            if base:
                nominal = (
                    self._root_licensed_feminine(base, host)
                    + self._from_nominal_suffix(host)
                    + self._free_state_a(base, host)
                )
                seen_host = {(c.segmentation, c.gloss) for c in base}
                for cand in nominal:
                    if (cand.segmentation, cand.gloss) not in seen_host:
                        base.append(cand)
                        seen_host.add((cand.segmentation, cand.gloss))
                # A trusted (A-tier) host analysis that is already segmented
                # outranks everything productive: m=ġappay=naḥ 'from=with=1P'
                # must not lose to a guessed m=ġapp-ay=naḥ 'from=have-MP=1P'.
                # Among the rest, a bounded possessive host is the most
                # specific reading (it reconstructs the whole host from a
                # securely attested citation or sibling); rank it next.
                base.sort(key=lambda c: (
                    0 if (c.tier in {"A0", "A1", "A2", "A3"}
                          and ("=" in (c.segmentation or "")
                               or "-" in (c.segmentation or ""))) else 1,
                    0 if c.features.get("possessive") else 1,
                    0 if (c.features.get("free_state")
                          or "-" in (c.segmentation or "")) else 1))
            # b=ešn=əl "in the year of", b=šenn=əl "on the rock of": a proclitic
            # and the construct clitic occur in one token, and neither path
            # alone could see it.
            # An inflected-but-unanalysed host (b̥=sōḥṯil: sōḥṯil is only
            # "attested, of sōḥṯa") is no analysis: the construct reading of
            # the host competes, first.
            if not base or all(c.features.get("unanalysed_inflected") for c in base):
                base = self._from_construct_clitic(host) + base
            if not base and depth > 1:
                base = self._from_proclitic(host, depth - 1)
            prep = entry["gloss"].rstrip("=").lower()
            for candidate in base[:2]:
                if not candidate.segmentation or not usable_gloss(candidate.gloss):
                    continue
                # PP is progressive morphology, not a generic reading of ʕa=.
                # It is licensed only on an NC/PRS verbal host; nominal/PN,
                # NC1/NC2, SC, PC and IPV hosts must never receive PP=.
                if prep == "pp":
                    tam = str(candidate.features.get("tam") or "").upper()
                    is_nc = (
                        tam == "PRS"
                        or bool(re.search(
                            r"(?:^|[.=\-])NC(?=$|[.=\-])",
                            candidate.gloss or "", re.IGNORECASE))
                    )
                    if not is_nc:
                        continue
                # A preposition proclitic takes a nominal complement, never a
                # finite verb: m=inəfṯaḥ 'from=[he opened]' is illicit and was
                # preempting the m- present participle (minəfṯaḥ 'get.open.NC').
                if prep in PREP_SENSES and _HAS_TAM.search(candidate.gloss or ""):
                    continue
                # nor another bare preposition: m=ʕan "from=about"
                if (prep in PREP_SENSES
                        and (candidate.gloss or "").lower() in PREP_SENSES):
                    continue
                if (prep in PREP_SENSES and candidate.lemma_id in self.verbal_lexeme_ids
                        and any(e.get("kind") == "attested_form" for e in candidate.evidence)):
                    # An unanalysed finite-verb form is not a nominal host
                    # merely because its display gloss omitted the TAM label.
                    continue
                seg_literal = f"{form}={candidate.segmentation}"
                gloss = f"{entry['gloss'].rstrip('=')}={candidate.gloss}"
                ok, rule_trace = reconstructs_to(seg_literal, surface)
                prosthetic = bool(candidate.features.get("prosthetic_schwa"))
                if (not ok and not prosthetic) or not aligned(seg_literal, gloss):
                    continue
                # The gloss shows the isolated clitic (l=), never the schwa
                # that only ever appears because of what follows it (lə=) -
                # line 1 keeps the literal spelling regardless, so nothing
                # here can lose what was actually written.
                canonical = self.proclitic_canonical_forms.get(form, form)
                seg = f"{canonical}={candidate.segmentation}"
                out.append(Candidate(
                    surface=surface, segmentation=seg, gloss=gloss,
                    lemma_id=candidate.lemma_id, lemma=candidate.lemma,
                    lexical_gloss=candidate.lexical_gloss,
                    lexical_gloss_status=candidate.lexical_gloss_status,
                    features={**candidate.features, "proclitic": entry["gloss"]},
                    tier="B1",
                    # an author-confirmed host is the strongest host there is
                    score=0.9 if candidate.features.get("author_confirmed") else 0.75,
                    reconstruction_ok=(canonical == form and not prosthetic),
                    reconstruction_trace=candidate.reconstruction_trace
                    + [f"proclitic {form}= = {entry['gloss']}"] + rule_trace,
                    evidence=candidate.evidence
                    + [{"kind": "glossing_standard_proclitic",
                        "entry_id": entry["entry_id"], "section": entry.get("section")}]))
        return out

    @staticmethod
    def _feature_fit(feats: dict, features: dict) -> bool:
        for key in ("person", "number", "gender"):
            want, have = features.get(key), feats.get(key)
            if want and have and want != have:
                return False
        return True

    @staticmethod
    def _tam_compatible(exponent_tam: str | None, form_tam: str | None) -> bool:
        """NC, NC1 and NC2 share one set of person/number exponents.

        The KB records every cell of that shared table under source_category
        NC (→ PRS), so a strict equality test hid the -in -MP suffix from every
        PRF form and glosses came out a segment short.
        """
        if not exponent_tam or not form_tam or exponent_tam == form_tam:
            return True
        nc_family = {"PRS", "PRF"}
        return exponent_tam in nc_family and form_tam in nc_family

    def _segment_with_exponent(self, surface: str, tam: str | None,
                               features: dict) -> tuple[str | None, str, str | None, list[str]]:
        """Split known person prefix and PNG suffix off the stem.

        Returns (prefix_gloss, stem, suffix_gloss, trace). Stem-internal TAM
        morphology is never split — only overt affixes are.
        """
        trace: list[str] = []
        stem = surface
        prefix_gloss = None
        for variant, gloss, feats in self.prefixes:
            if not self._tam_compatible(feats.get("tam"), tam):
                continue
            if not self._feature_fit(feats, features):
                continue

            # A person prefix keeps its fill vowel before a cluster of two
            # consonants and loses it everywhere else: yi-nfuḏ, yi-ščaġl-un
            # against y-allex, y-īb, y-lufẓ-un, y-ʔaḏḏen. Twenty-nine of the
            # thirty prefixed forms in the confirmed corpus follow it; the odd
            # one out is `yi-šu`. The earlier reading of the rule — short only
            # before a vowel — accounted for the vowel-initial stems and left
            # `y-lufẓ-un` and `y-ʔaḏḏen` looking like exceptions.
            #
            # Derived from the prefix inventory rather than hard-coded per
            # person, so every licensed Ci- prefix is treated alike while its
            # TAM/PNG features still have to match the dictionary record. Both
            # shapes are offered where the cut itself is in doubt; the ranking
            # decides, and the loser stays visible as an alternative.
            realizations: list[tuple[str, str | None]] = [(variant, None)]
            if len(variant) > 1 and variant.endswith("i"):
                reduced = variant[:-1]
                rest = stem[len(reduced):] if stem.startswith(reduced) else ""
                if len(stem) > len(reduced) + 1 and rest and not _cluster(rest):
                    realizations.append((
                        reduced,
                        f"prefix {variant}- surfaces as {reduced}- "
                        "outside a two-consonant cluster",
                    ))

            for realized, allomorph_trace in realizations:
                if len(stem) <= len(realized) + 1 or not stem.startswith(realized):
                    continue
                prefix_gloss = gloss
                trace.append(f"prefix {realized}- = {gloss}")
                if allomorph_trace:
                    trace.append(allomorph_trace)
                stem = stem[len(realized):]
                break
            if prefix_gloss is not None:
                break
        suffix_gloss = None
        for variant, gloss, feats in self.suffixes:
            if not self._tam_compatible(feats.get("tam"), tam):
                continue
            if not self._feature_fit(feats, features):
                continue
            if len(stem) > len(variant) + 1 and stem.endswith(variant):
                suffix_gloss = gloss
                trace.append(f"suffix -{variant} = {gloss}")
                stem = stem[: -len(variant)]
                break
        # The third person is `3M` alone and `3` once something makes the form
        # plural. Decided here, after the suffix is known, because the prefix is
        # matched first and cannot see it — which is how a plural `y-` form came
        # to be labelled by whichever inventory row happened to match.
        if prefix_gloss in (THIRD_SINGULAR, THIRD_PLURAL):
            plural = bool(suffix_gloss and _PLURAL_EXPONENT.match(suffix_gloss))
            wanted = THIRD_PLURAL if plural else THIRD_SINGULAR
            if wanted != prefix_gloss:
                trace.append(f"third person is {wanted} "
                             f"({'plural' if plural else 'singular'} form)")
                prefix_gloss = wanted
        return prefix_gloss, stem, suffix_gloss, trace

    def _from_a2(self, surface: str) -> list[Candidate]:
        out = []
        for match in self.a2.get(surface, []):
            lemma_id = match.get("lexeme_id")
            lexical, status = self._lexical(lemma_id, surface)
            tam = match.get("tam_canonical")
            feats = dict(match.get("features") or {})
            prefix_gloss, stem, suffix_gloss, trace = self._segment_with_exponent(
                surface, tam, feats)
            stem_at = surface.find(stem)
            prefix_str = surface[:stem_at] if prefix_gloss else ""
            suffix_str = surface[stem_at + len(stem):] if suffix_gloss else ""
            parts = ([prefix_str] if prefix_gloss else []) + [stem] + (
                [suffix_str] if suffix_gloss else [])
            segmentation = "-".join(parts) if len(parts) > 1 else surface
            lemma_form = ((self.lexemes.get(lemma_id) or {}).get("lemma_canonical")
                          or match.get("lemma_canonical"))
            gloss = self._build_gloss(lexical, tam, feats, suffix_gloss, lemma_form)
            if prefix_gloss:
                gloss = f"{prefix_gloss}-{gloss}"
            if not aligned(segmentation, gloss):
                segmentation, gloss = surface, self._build_gloss(
                    lexical, tam, feats, None, lemma_form)
            candidate = Candidate(
                surface=surface, segmentation=segmentation, gloss=gloss,
                lemma_id=lemma_id,
                lemma=(self.lexemes.get(lemma_id) or {}).get("lemma_canonical"),
                lexical_gloss=lexical, lexical_gloss_status=status,
                features={**feats, "tam": tam,
                          "object": match.get("object_features"),
                          "root": match.get("root")},
                tier="A2", score=0.9,
                reconstruction_ok=reconstructs_to(segmentation, surface)[0],
                reconstruction_trace=["A2 explicit Arnold Maalula form"] + trace,
                evidence=[{"kind": "arnold_inflected_form",
                           "inflected_form_id": match.get("inflected_form_id"),
                           "page": match.get("source_page"),
                           "span": match.get("source_span"),
                           "dialect": match.get("effective_dialect")}])
            if status == "translation_pending":
                candidate.uncertainties.append({
                    "type": "lexical_gloss_pending",
                    "reason": "Arnold sense not yet translated to English"})
            # A verb form of the dictionary tied, by its spelling in the
            # corpus, to a lexeme on record as a noun or an adjective (ʕaynay
            # "look!" to ʕayna "eye", yḥulpun to ḥalba "milk"): the form may
            # be right, the lexeme is not, and the reading is never settled.
            if tam and self.lexeme_part_of_speech.get(lemma_id) in ("noun", "adjective"):
                candidate.uncertainties.append({
                    "type": "verbal_label_on_a_noun",
                    "reason": "a dictionary verb form tied to a lexeme on record "
                              "as " + self.lexeme_part_of_speech[lemma_id]})
            out.append(candidate)
        return out

    def _from_cross_dialect(self, surface: str) -> list[Candidate]:
        """Confirm a productive Maalula split with an exact Arnold B/J form.

        B/J evidence must never enter A2 directly.  It is admitted only when
        the ordinary Maalula exponent inventory independently removes an overt
        prefix or suffix and the resulting segmentation reconstructs exactly.
        Thus ``yakreṣ`` may become ``y-akreṣ`` at B1, while an arbitrary
        unsegmented B/J citation remains excluded.
        """
        out = []
        for match in self.cross_dialect.get(surface, []):
            lemma_id = match.get("lexeme_id")
            lexical, status = self._lexical(lemma_id, surface)
            tam = match.get("tam_canonical")
            feats = dict(match.get("features") or {})
            prefix_gloss, stem, suffix_gloss, trace = self._segment_with_exponent(
                surface, tam, feats)
            # The cross-dialect record confirms a generated analysis; it must
            # not create a whole-form analysis merely because Arnold cites it.
            if not trace:
                continue
            # Prefer a Maalula stem gloss over the dictionary lemma placeholder.
            # This prevents a B/J confirmation from degrading an independently
            # analysable form such as ḥōfr-in ``engrave.NC-MP`` to
            # ``iḥfar.NC-MP`` merely because the lexeme link has no translated
            # sense.  An unresolved cross-dialect form may still be generated,
            # but at B2 so it cannot outrank a resolved Maalula B1 candidate.
            if not lexical:
                stem_key = skeleton(stem)
                lexical = self.stem_glosses.get(stem_key)
                if lexical:
                    authority = self.stem_gloss_authority.get(stem_key) or "author_gold"
                    status = f"stem_gloss_{authority}"
            stem_at = surface.find(stem)
            prefix_str = surface[:stem_at] if prefix_gloss else ""
            suffix_str = surface[stem_at + len(stem):] if suffix_gloss else ""
            parts = ([prefix_str] if prefix_gloss else []) + [stem] + (
                [suffix_str] if suffix_gloss else [])
            segmentation = "-".join(parts)
            lemma_form = ((self.lexemes.get(lemma_id) or {}).get("lemma_canonical")
                          or match.get("lemma_canonical"))
            gloss = self._build_gloss(lexical, tam, feats, suffix_gloss, lemma_form)
            if prefix_gloss:
                gloss = f"{prefix_gloss}-{gloss}"
            ok, rule_trace = reconstructs_to(segmentation, surface)
            if not aligned(segmentation, gloss) or not ok:
                continue
            candidate = Candidate(
                surface=surface, segmentation=segmentation, gloss=gloss,
                lemma_id=lemma_id, lemma=lemma_form,
                lexical_gloss=lexical, lexical_gloss_status=status,
                features={**feats, "tam": tam,
                          "object": match.get("object_features"),
                          "cross_dialect_confirmation": True},
                tier=("B1" if lexical else "B2"),
                score=(0.84 if lexical else 0.49), reconstruction_ok=True,
                reconstruction_trace=[
                    "productive Maalula exponent segmentation",
                    f"exact Arnold {match.get('effective_dialect')} attestation "
                    "confirms the generated form",
                ] + trace + rule_trace,
                evidence=[{"kind": "cross_dialect_inflected_form",
                           "inflected_form_id": match.get("inflected_form_id"),
                           "page": match.get("source_page"),
                           "span": match.get("source_span"),
                           "dialect": match.get("effective_dialect")}])
            if status in {"translation_pending", "no_sense", "unknown_lexeme"}:
                candidate.uncertainties.append({
                    "type": "lexical_gloss_missing",
                    "reason": "cross-dialect morphology is confirmed but the "
                              "English lexical gloss is unavailable"})
            out.append(candidate)
        return out

    # ---- B-tier compositional ----------------------------------------
    def _strip_clitics(self, surface: str) -> list[tuple[str, list[dict], list[str]]]:
        """Peel up to two enclitics off the end. Returns (host, clitics, trace)."""
        results = [(surface, [], [])]
        for _ in range(2):
            grown = []
            for host, taken, trace in results:
                for clitic in self.clitics:
                    form = clitic["form"]
                    if len(host) > len(form) + 1 and host.endswith(form):
                        # =le is an indivisible IO clitic where licensed: do not
                        # re-split a host that already ends in the l-object form.
                        grown.append((host[: -len(form)], taken + [clitic],
                                      trace + [f"clitic ={form} = {clitic['gloss']}"]))
            results = results + grown
            if not grown:
                break
        return results

    def _from_subject_plural(self, surface: str) -> list[Candidate]:
        """Plural agreement on an attested prefix-conjugation singular.

        `-un` and `-an` are the subject exponents of the prefix conjugation and
        also the third-person object suffixes, so `yi-nfuḏ-un` and
        `yi-nfuḏ=un` both reconstruct. Only the object reading was ever built —
        the enclitic inventory is consulted, the exponent table is not, unless
        the whole plural form happens to be in the dictionary — so agreement on
        the verb came out as an object every time.

        Deliberately narrow: only the prefix conjugation, whose exponents are
        the ones that collide, and only over a host the corpus already glosses
        as a finite singular of the same TAM and person. Nothing is invented;
        the plural cell is read off an attested form. The object reading stays
        as an alternative, because on a subjunctive it is a real analysis.
        """
        out: list[Candidate] = []
        for variant, exponent, feats in self.suffixes:
            if feats.get("tam") != "SBJV" or not _PLURAL_EXPONENT.match(exponent):
                continue
            if len(surface) <= len(variant) + 1 or not surface.endswith(variant):
                continue
            host = surface[: -len(variant)]
            for base in (self._from_a1(host) or self._from_a2(host))[:2]:
                if not base.segmentation or not base.gloss:
                    continue
                person = _PERSON_PREFIX.match(base.gloss)
                if not person or int(person.group(1)) != feats.get("person"):
                    continue
                if ".SBJV" not in base.gloss.upper():
                    continue
                # The prefix loses its gender label once the form is plural,
                # by the same rule that decides it after an exponent split.
                # `či-` is the exception: it spells the third feminine
                # singular only — the third plural is `y-`/`yi-` for both
                # genders — so a 3F host has no plural to grow into. In the
                # corpus every prefix that carries a plural suffix is n-, y-,
                # yi- or č-; či- never does.
                label = person.group(1) + (person.group(2) or "")
                if label == "3F":
                    continue
                if label in (THIRD_SINGULAR, THIRD_PLURAL):
                    label = THIRD_PLURAL
                seg = f"{base.segmentation}-{variant}"
                gloss = f"{label}-{base.gloss[person.end():]}-{exponent}"
                ok, rule_trace = reconstructs_to(seg, surface)
                if not ok or not aligned(seg, gloss):
                    continue
                out.append(Candidate(
                    surface=surface, segmentation=seg, gloss=gloss,
                    lemma_id=base.lemma_id, lemma=base.lemma,
                    lexical_gloss=base.lexical_gloss,
                    lexical_gloss_status=base.lexical_gloss_status,
                    features={**base.features, **{k: v for k, v in feats.items()
                                                  if k != "tam"}},
                    tier="B1", score=0.78,
                    reconstruction_ok=True,
                    reconstruction_trace=base.reconstruction_trace
                    + [f"subject suffix -{variant} = {exponent}"] + rule_trace,
                    evidence=base.evidence + [{"kind": "subject_agreement",
                                               "exponent": variant}]))
        return out

    def _from_compositional(self, surface: str) -> list[Candidate]:
        out = []
        for host, clitics, trace in self._strip_clitics(surface):
            if host == surface or not clitics:
                continue
            # A0 belongs here too: most of the closed-class inventory exists
            # precisely to carry pronominal enclitics (ʕemm=, komm=, ḥōl=,
            # ġapp=), and leaving it out made 20-odd entries inert.
            if not (self.a0.get(host) or self.a1.get(host) or self.a2.get(host)):
                continue
            # A cumulative word entry such as minnay "from.3P" must not act as
            # a host: attaching =hen to it would mark person twice
            # (from.3P=3FP). When something attaches, only the host/proclitic
            # entry qualifies.
            a0_hosts = [c for c in self._from_a0(host)
                        if c.features.get("attachment") in ("host", "proclitic")]
            base = (a0_hosts or self._from_a1(host) or self._from_a2(host))
            for b in base[:2]:
                if not b.segmentation:
                    continue
                # Arnold cites the 3 sg m preterite before a noun with its -il
                # (ḳaṭʕil, laḳṭil; Gr. § 5.12 S. 286: "Nach KK- lautet das
                # Suffix -il"): that word is the verb with the DOM, kaṭʕ=l.
                cited_with_dom = (
                    b.tier == "A2" and b.features.get("tam") == "PST"
                    and host.endswith("il") and b.segmentation == host)
                seg = b.segmentation
                gl = b.gloss or ""
                for clitic in reversed(clitics):
                    seg = f"{seg}={clitic['form']}"
                    gl = f"{gl}={clitic['gloss']}"
                ok, rule_trace = reconstructs_to(seg, surface)
                if not aligned(seg, gl) or not ok:
                    continue
                # The DOM need not end the word (našk-l-il=lun
                # kiss.NC-DOM-1S=3MP.IO has the subject marker after it), but
                # an object clitic never stands directly after it. Such a
                # reading stays on offer below every other one: kaṭʕillun is
                # kaṭʕ-il=lun NC-MP=3MP before it is kaṭʕil=lun (author,
                # 02.10.2026).
                after_dom = bool(cited_with_dom or _OBJECT_AFTER_DOM.search(gl))
                out.append(Candidate(
                    surface=surface, segmentation=seg, gloss=gl,
                    lemma_id=b.lemma_id, lemma=b.lemma,
                    lexical_gloss=b.lexical_gloss,
                    lexical_gloss_status=b.lexical_gloss_status,
                    features={**b.features, "clitics": [c["gloss"] for c in clitics]},
                    tier="C" if after_dom else "B1",
                    score=0.2 if after_dom else 0.7,
                    reconstruction_ok=True,
                    reconstruction_trace=b.reconstruction_trace + trace + rule_trace
                    + [f"reconstructed {surface}"],
                    evidence=b.evidence + [{"kind": "clitic_composition",
                                            "clitics": [c["form"] for c in clitics]}],
                    uncertainties=([{"type": "object_after_dom",
                                     "reason": "an object clitic directly after the DOM"}]
                                   if after_dom else [])))
        return out


    # ---- D2: service verbs, outside self.lexemes --------------------------
    def service_verb(self, root: str) -> str:
        lexeme_id = f"lexsvc_{root}"
        if lexeme_id not in self.service_verbs:
            entries = self.arnold_verbs.get(root) or []
            self.service_verbs[lexeme_id] = {
                "lexeme_id": lexeme_id,
                "lemma_canonical": entries[0]["citation"] if len(entries) == 1 else f"√{root}",
                "root_canonical": root, "part_of_speech": "verb", "service": True}
        return lexeme_id

    def lexeme_record(self, lexeme_id: str | None) -> dict[str, Any]:
        """A real lexeme (including a promoted `lexsvc_`), else a service verb."""
        return self.lexemes.get(lexeme_id) or self.service_verbs.get(lexeme_id) or {}

    def is_verbal(self, lexeme_id: str | None) -> bool:
        return bool(lexeme_id) and (lexeme_id in self.verbal_lexeme_ids
                                    or (lexeme_id in self.service_verbs
                                        and lexeme_id not in self.lexemes))

    def verb_for_root(self, root: str, stem: str = "", gloss: str = ""
                      ) -> tuple[str | None, list[str]]:
        """D2/D5: the verb of a root. One verbal lexeme -> it; several -> the
        one the stem's shape singles out (maḥcy- is IV aḥəc, not III ḥōc),
        else a service verb with the verbs as candidates; none -> a service
        verb if the root is known to be verbal, else nothing."""
        verbs = sorted(i for i in self.lexeme_ids_by_root.get(root, ())
                       if i in self.verbal_lexeme_ids)
        if len(verbs) == 1:
            return verbs[0], []
        if verbs:
            shape = stem_classes(stem, root, gloss) if stem else None
            if shape:
                fitting = [v for v in verbs
                           if citation_class(self.lexemes[v].get("lemma_canonical") or "", root) in shape]
                if len(fitting) == 1:
                    return fitting[0], []
            return self.service_verb(root), verbs
        if root in self.verbal_roots:
            return self.service_verb(root), []
        return None, []

    def _nc_root_matches(
            self,
            prefixed_stem: str,
            context: str,
            allowed_lexeme_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Match a surface stem only to a real verbal lexeme/root."""
        out: list[dict[str, Any]] = []

        for prefix, person_gloss in _NC_PERSON_PREFIXES:
            if prefix:
                if not prefixed_stem.startswith(prefix):
                    continue
                stem = prefixed_stem[len(prefix):]
            else:
                stem = prefixed_stem

            if len(stem) < 3:
                continue

            # Lexeme-bound exceptional stems bypass root-template inference.
            # The lexical root and the identity of the verb remain untouched.
            for lexeme_id, display, explicit_stem in self.explicit_nc_stems:
                if stem != explicit_stem or (allowed_lexeme_ids and lexeme_id not in allowed_lexeme_ids):
                    continue
                lexeme = self.lexemes[lexeme_id]
                lexical, status = self._lexical(lexeme_id, prefixed_stem)
                out.append({
                    "prefix": prefix, "person_gloss": person_gloss, "stem": stem,
                    "root": lexeme.get("root_canonical") or "",
                    "lexeme_id": lexeme_id, "lemma": lexeme.get("lemma_canonical"),
                    "lexical": lexical, "status": status, "display_class": display,
                    "model": "explicit_" + display + "_stem", "model_surface": stem,
                    "weak_root_rules": ("curated lexeme stem; no root rewriting",),
                    "unique_binding": True,
                })

            for root, root_ids in self.lexeme_ids_by_root.items():
                if len(root) != 3:
                    continue

                ids = set(root_ids)
                if allowed_lexeme_ids:
                    ids &= allowed_lexeme_ids
                if not ids:
                    continue

                verbal_ids = sorted(
                    lexeme_id for lexeme_id in ids
                    if lexeme_id in self.verbal_lexeme_ids
                )
                if not verbal_ids and allowed_lexeme_ids:
                    # D5: the settled reading chose a non-verb of this root
                    # (fṯīḥ-a -> ifṯeḥ); its NC reading goes to the root's verbs.
                    verbal_ids = sorted(i for i in root_ids if i in self.verbal_lexeme_ids)
                if not verbal_ids:
                    # D2: never a same-root noun; a service verb when the root
                    # is known to be verbal and no lexeme set was imposed.
                    if allowed_lexeme_ids or root not in self.verbal_roots:
                        continue
                    verbal_ids = [self.service_verb(root)]

                for display_class, expected, model in nc_surface_models(
                        root, context, person_prefixed=bool(prefix)):
                    matched_rules: tuple[str, ...] | None = None
                    matched_expected: str | None = None
                    for variant, rules in _nc_model_variants(root, expected, display_class):
                        if _nc_shape_matches(stem, variant):
                            matched_expected = variant
                            matched_rules = rules
                            break
                    if matched_rules is None:
                        continue

                    # NC1/NC2 are G-perfect display subclasses.  If the same
                    # root has an explicitly class-I/G lexeme, do not borrow the
                    # meaning of a derived same-root lexeme (e.g. šmʕ II
                    # šammaʕ instead of G išmeʕ 'hear').  With no known G link
                    # we retain the old same-root behaviour.
                    binding_ids = verbal_ids
                    if display_class in {"NC1", "NC2"}:
                        g_ids = sorted(set(verbal_ids)
                                       & self.g_lexeme_ids_by_root.get(root, set()))
                        resolved_g = [lexeme_id for lexeme_id in g_ids
                                      if usable_gloss(self._lexical(
                                          lexeme_id, prefixed_stem)[0])]
                        resolved_any = [lexeme_id for lexeme_id in verbal_ids
                                        if usable_gloss(self._lexical(
                                            lexeme_id, prefixed_stem)[0])]
                        if resolved_g:
                            binding_ids = resolved_g
                        elif resolved_any:
                            binding_ids = resolved_any
                        elif g_ids:
                            binding_ids = g_ids

                    # Among the verbs of the root the template picks those of
                    # its own stem class (see _NC_MODEL_CLASS). With no verb of
                    # that class on record the same-root binding stays, since a
                    # citation does not always show its class.
                    if display_class == "NC":
                        wanted = _NC_MODEL_CLASS.get(model)
                        of_class = [
                            lexeme_id for lexeme_id in binding_ids
                            if citation_class(str(self.lexeme_record(lexeme_id).get(
                                "lemma_canonical") or ""), root) == wanted]
                        if of_class:
                            binding_ids = of_class

                    # Root-initial ʔ may license matching, but line 2 follows
                    # the written surface. The recovered ʔ remains in root/trace.
                    canonical_stem = stem

                    for lexeme_id in binding_ids:
                        lexeme = self.lexeme_record(lexeme_id)
                        lemma = lexeme.get("lemma_canonical")
                        lexical, status = self._lexical(lexeme_id, prefixed_stem)
                        out.append({
                            "prefix": prefix,
                            "person_gloss": person_gloss,
                            "stem": canonical_stem,
                            "root": root,
                            "lexeme_id": lexeme_id,
                            "lemma": lemma,
                            "lexical": lexical,
                            "status": status,
                            "display_class": display_class,
                            "model": model,
                            "model_surface": matched_expected,
                            "weak_root_rules": matched_rules,
                            "unique_binding": len(binding_ids) == 1,
                        })
        return out

    @staticmethod
    def _nc_weak_trace(match: dict[str, Any]) -> list[str]:
        rules = tuple(match.get("weak_root_rules") or ())
        return [f"weak-root allomorphy: {', '.join(rules)}"] if rules else []

    @staticmethod
    def _nc_candidate_gloss(match: dict[str, Any]) -> str | None:
        lexical = match["lexical"] or gloss_placeholder(match["lemma"])
        if not lexical:
            return None
        core = f"{lexical}.{match['display_class']}"
        return f"{match['person_gloss']}-{core}" if match["person_gloss"] else core

    # PRF_TO_NC_RUNTIME_V1
    @staticmethod
    def _has_legacy_prf(candidate: Candidate) -> bool:
        return bool(re.search(r"\.PRF\b", candidate.gloss or "", re.IGNORECASE))

    def _prf_nc_structural_competitors(
            self,
            surface: str,
            candidates: list[Candidate],
    ) -> list[Candidate]:
        """Generate NC1/NC2 structure only when a settled reading still says PRF.

        Old Arnold/A1 links may be weak, so migration asks the new root/template
        recognizer independently. Relabelling happens only if every exact
        structural reading for the same segmentation agrees on NC1 vs NC2.
        """
        if not any(self._has_legacy_prf(c) for c in candidates):
            return []

        out: list[Candidate] = []

        # No-clitic forms: deliberately ignore the legacy lemma link here.
        out += self._from_nc_root_bare(surface, None)
        out += self._from_nc_root_f(surface, None)

        # Exact A1/A2 evidence normally stops the cascade before composition,
        # so explicitly ask the overt-morphology paths during migration.
        out += self._from_nc_root_plural(surface)
        out += self._from_nc_root_plain_clitic(surface)
        out += self._from_nc_root_f_clitic(surface)
        out += self._from_nc_root_plural_l_object(surface)

        seen: set[tuple] = set()
        unique: list[Candidate] = []
        for candidate in out:
            key = candidate.key()
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        return unique

    def _canonicalize_legacy_prf(
            self,
            candidates: list[Candidate],
    ) -> list[Candidate]:
        """Replace display .PRF by NC1/NC2 when structure decides the subclass.

        Tier, score, lemma and lexical source stay exactly as before.
        Ambiguous or unclassifiable PRF remains PRF.
        """
        by_segmentation: dict[str, set[str]] = {}
        structural_by_segmentation: dict[str, list[Candidate]] = defaultdict(list)

        for candidate in candidates:
            segmentation = candidate.segmentation
            gloss = candidate.gloss or ""
            if not segmentation:
                continue
            match = re.search(r"\.(NC1|NC2)\b", gloss, re.IGNORECASE)
            if not match:
                continue
            by_segmentation.setdefault(segmentation, set()).add(
                match.group(1).upper()
            )
            structural_by_segmentation[segmentation].append(candidate)

        out: list[Candidate] = []

        for candidate in candidates:
            if not self._has_legacy_prf(candidate) or not candidate.segmentation:
                out.append(candidate)
                continue

            subclasses = by_segmentation.get(candidate.segmentation, set())
            if len(subclasses) != 1:
                out.append(candidate)
                continue

            subclass = next(iter(subclasses))
            new_gloss = re.sub(
                r"\.PRF\b",
                f".{subclass}",
                candidate.gloss or "",
                count=1,
                flags=re.IGNORECASE,
            )

            # The old A1/A2 candidate keeps its evidence tier, but if the new
            # exact root+template parse has a single usable G lexical binding,
            # inherit that lexical identity too.  This turns a legacy
            # yišmaʕ.PRF-F lemma placeholder into hear.NC2-F without demoting
            # the A2 evidence that licensed the surface form.
            structural = [
                c for c in structural_by_segmentation.get(candidate.segmentation, [])
                if re.search(fr"\.{subclass}\b", c.gloss or "", re.IGNORECASE)
            ]
            lexical_structural = [
                c for c in structural
                if c.lexical_gloss and usable_gloss(c.lexical_gloss)
            ]
            structural_lexical_heads = {
                c.lexical_gloss for c in lexical_structural if c.lexical_gloss
            }
            lexical_source = None
            if len(structural_lexical_heads) == 1 and lexical_structural:
                lexical_source = max(
                    lexical_structural,
                    key=lambda c: (c.tier in {"A1", "A2", "A3"}, c.score),
                )
                new_gloss = lexical_source.gloss or new_gloss

            features = dict(candidate.features)
            features["legacy_tam"] = "PRF"
            features["perfect_subclass"] = subclass
            features["display_conjugation"] = subclass

            out.append(replace(
                candidate,
                gloss=new_gloss,
                lemma_id=(lexical_source.lemma_id if lexical_source else candidate.lemma_id),
                lemma=(lexical_source.lemma if lexical_source else candidate.lemma),
                lexical_gloss=(lexical_source.lexical_gloss
                               if lexical_source else candidate.lexical_gloss),
                lexical_gloss_status=(lexical_source.lexical_gloss_status
                                      if lexical_source else candidate.lexical_gloss_status),
                features=features,
                reconstruction_trace=(
                    list(candidate.reconstruction_trace)
                    + [f"legacy PRF canonicalized to {subclass} "
                       "by exact root+surface-template match"]
                ),
                evidence=(
                    list(candidate.evidence)
                    + [{
                        "kind": "prf_to_nc_subclass",
                        "subclass": subclass,
                        "basis": "exact segmentation + verbal root + NC surface model",
                    }]
                ),
            ))

        return out

    def _from_nc_root_bare(
            self, surface: str, base: list[Candidate] | None = None
    ) -> list[Candidate]:
        """Bare no-clitic NC-family reading; kept as a competitor."""
        allowed_ids = {c.lemma_id for c in (base or []) if c.lemma_id} or None
        out: list[Candidate] = []
        for match in self._nc_root_matches(surface, "bare", allowed_ids):
            gloss = self._nc_candidate_gloss(match)
            if not gloss:
                continue
            prefix, stem = match["prefix"], match["stem"]
            segmentation = f"{prefix}-{stem}" if prefix else stem
            ok, trace = reconstructs_to(segmentation, surface)
            if not ok or not aligned(segmentation, gloss):
                continue
            overt_person = bool(prefix)
            out.append(Candidate(
                surface=surface, segmentation=segmentation, gloss=gloss,
                lemma_id=match["lexeme_id"], lemma=match["lemma"],
                lexical_gloss=match["lexical"], lexical_gloss_status=match["status"],
                features={"root": match["root"],
                          "display_conjugation": match["display_class"],
                          "nc_surface_model": match["model"],
                          "nc_surface_context": "bare"},
                tier="B1" if overt_person or match["model"].startswith("explicit_") else "B2",
                score=(0.86 if match["model"].startswith("explicit_")
                       else 0.76 if overt_person else 0.52),
                reconstruction_ok=True,
                reconstruction_trace=([f"reverse NC {match['model']} √{match['root']}"]
                                      + self._nc_weak_trace(match)
                                      + ["bare no-clitic competitor"] + trace),
                evidence=[{"kind": "root_licensed_nc_bare",
                           "root": match["root"], "model": match["model"]}],
            ))
        return out

    def _from_nc_root_f(
            self, surface: str, base: list[Candidate] | None = None
    ) -> list[Candidate]:
        """No-clitic F: qōtl-a / qtīl-a / qattīl-a; never *qatl-a."""
        if not surface.endswith("a") or len(surface) < 4:
            return []
        stem_surface = surface[:-1]
        allowed_ids = {c.lemma_id for c in (base or []) if c.lemma_id} or None
        out: list[Candidate] = []
        for match in self._nc_root_matches(stem_surface, "free_f", allowed_ids):
            core = self._nc_candidate_gloss(match)
            if not core:
                continue
            prefix, stem = match["prefix"], match["stem"]
            segmentation = f"{prefix}-{stem}-a" if prefix else f"{stem}-a"
            gloss = f"{core}-F"
            ok, trace = reconstructs_to(segmentation, surface)
            if not ok or not aligned(segmentation, gloss):
                continue
            out.append(Candidate(
                surface=surface, segmentation=segmentation, gloss=gloss,
                lemma_id=match["lexeme_id"], lemma=match["lemma"],
                lexical_gloss=match["lexical"], lexical_gloss_status=match["status"],
                features={"root": match["root"],
                          "display_conjugation": match["display_class"],
                          "nc_surface_model": match["model"],
                          "nc_surface_context": "free_f", "gender": "F"},
                tier="B1" if match["unique_binding"] else "B2",
                score=0.82 if match["unique_binding"] else 0.58,
                reconstruction_ok=True,
                reconstruction_trace=([f"reverse NC {match['model']} √{match['root']}"]
                                      + self._nc_weak_trace(match)
                                      + ["no-clitic -a = F"] + trace),
                evidence=[{"kind": "root_licensed_nc_f",
                           "root": match["root"], "model": match["model"]}],
            ))
        return out

    def _from_nc_root_plural(self, surface: str) -> list[Candidate]:
        """No-clitic plural -in/-an; use free/long plural stem vocalism."""
        out: list[Candidate] = []
        for ending, agreement, gender in (("in", "MP", "M"), ("an", "FP", "F")):
            if not surface.endswith(ending) or len(surface) <= len(ending) + 2:
                continue
            stem_surface = surface[:-len(ending)]
            for match in self._nc_root_matches(stem_surface, "plural_free"):
                core = self._nc_candidate_gloss(match)
                if not core:
                    continue
                prefix, stem = match["prefix"], match["stem"]
                segmentation = f"{prefix}-{stem}-{ending}" if prefix else f"{stem}-{ending}"
                gloss = f"{core}-{agreement}"
                ok, trace = reconstructs_to(segmentation, surface)
                if not ok or not aligned(segmentation, gloss):
                    continue
                out.append(Candidate(
                    surface=surface, segmentation=segmentation, gloss=gloss,
                    lemma_id=match["lexeme_id"], lemma=match["lemma"],
                    lexical_gloss=match["lexical"], lexical_gloss_status=match["status"],
                    features={"root": match["root"],
                              "display_conjugation": match["display_class"],
                              "nc_surface_model": match["model"],
                              "nc_surface_context": "plural_free",
                              "number": "plural", "gender": gender},
                    tier="B1" if match["unique_binding"] else "B2",
                    score=0.80 if match["unique_binding"] else 0.57,
                    reconstruction_ok=True,
                    reconstruction_trace=([f"reverse NC {match['model']} √{match['root']}"]
                                          + self._nc_weak_trace(match)
                                          + [f"no-clitic -{ending} = {agreement}"] + trace),
                    evidence=[{"kind": "root_licensed_nc_plural",
                               "root": match["root"], "model": match["model"]}],
                ))
        return out

    @staticmethod
    def _nc_object_readings(clitic: dict) -> list[tuple[str, str, float]]:
        raw = clitic.get("gloss") or ""
        plain = re.sub(r"\.IO$", "", raw, flags=re.IGNORECASE)
        return ([(plain, "OBJECT", 0.02), (raw, "IO", 0.01)]
                if raw.upper().endswith(".IO")
                else [(raw, "OBJECT", 0.02)])

    def _from_nc_root_plain_clitic(self, surface: str) -> list[Candidate]:
        """SG.M + indivisible l-object clitic."""
        out: list[Candidate] = []
        for clitic in self.clitics:
            if clitic.get("paradigm") != "l_object":
                continue
            form = clitic.get("form") or ""
            if not form or not surface.endswith(form):
                continue
            host = surface[:-len(form)]
            if not host or host.endswith(("ō", "il", "al")):
                continue
            for match in self._nc_root_matches(host, "clitic_m"):
                core = self._nc_candidate_gloss(match)
                if not core:
                    continue
                prefix, stem = match["prefix"], match["stem"]
                stem_seg = f"{prefix}-{stem}" if prefix else stem
                segmentation = f"{stem_seg}={form}"
                for label, role, bonus in self._nc_object_readings(clitic):
                    gloss = f"{core}={label}"
                    ok, trace = reconstructs_to(segmentation, surface)
                    if not ok or not aligned(segmentation, gloss):
                        continue
                    out.append(Candidate(
                        surface=surface, segmentation=segmentation, gloss=gloss,
                        lemma_id=match["lexeme_id"], lemma=match["lemma"],
                        lexical_gloss=match["lexical"], lexical_gloss_status=match["status"],
                        features={"root": match["root"],
                                  "display_conjugation": match["display_class"],
                                  "nc_surface_model": match["model"],
                                  "nc_surface_context": "clitic_m",
                                  "object": label, "object_role": role},
                        tier="B1" if match["unique_binding"] else "B2",
                        score=(0.84 if match["unique_binding"] else 0.60) + bonus,
                        reconstruction_ok=True,
                        reconstruction_trace=([f"reverse NC {match['model']} √{match['root']}"]
                                              + self._nc_weak_trace(match)
                                              + [f"indivisible clitic ={form}"] + trace),
                        evidence=[{"kind": "nc_plain_l_object",
                                   "root": match["root"], "clitic": form}],
                    ))
        return out

    def _from_nc_root_f_clitic(self, surface: str) -> list[Candidate]:
        """SG.F -ō + l-object; the only productive path where -ō = F."""
        out: list[Candidate] = []
        for clitic in self.clitics:
            if clitic.get("paradigm") != "l_object":
                continue
            form = clitic.get("form") or ""
            if not form or not surface.endswith(form):
                continue
            host = surface[:-len(form)]
            if not host.endswith("ō"):
                continue
            stem_surface = host[:-1]
            for match in self._nc_root_matches(stem_surface, "clitic_f"):
                core = self._nc_candidate_gloss(match)
                if not core:
                    continue
                prefix, stem = match["prefix"], match["stem"]
                stem_seg = f"{prefix}-{stem}" if prefix else stem
                segmentation = f"{stem_seg}-ō={form}"
                for label, role, bonus in self._nc_object_readings(clitic):
                    gloss = f"{core}-F={label}"
                    ok, trace = reconstructs_to(segmentation, surface)
                    if not ok or not aligned(segmentation, gloss):
                        continue
                    out.append(Candidate(
                        surface=surface, segmentation=segmentation, gloss=gloss,
                        lemma_id=match["lexeme_id"], lemma=match["lemma"],
                        lexical_gloss=match["lexical"], lexical_gloss_status=match["status"],
                        features={"root": match["root"],
                                  "display_conjugation": match["display_class"],
                                  "nc_surface_model": match["model"],
                                  "nc_surface_context": "clitic_f",
                                  "gender": "F", "object": label, "object_role": role},
                        tier="B1" if match["unique_binding"] else "B2",
                        score=(0.86 if match["unique_binding"] else 0.61) + bonus,
                        reconstruction_ok=True,
                        reconstruction_trace=([f"reverse NC {match['model']} √{match['root']}"]
                                              + self._nc_weak_trace(match)
                                              + ["-ō = F only before an overt NC-family clitic",
                                                 f"indivisible clitic ={form}"] + trace),
                        evidence=[{"kind": "nc_f_l_object",
                                   "root": match["root"], "clitic": form}],
                    ))
        return out

    def _from_nc_root_plural_l_object(self, surface: str) -> list[Candidate]:
        """PL before l: -in -> -il and -an -> -al, with object/DOM readings."""
        out: list[Candidate] = []

        # Bare final =l is the differential object marker, not a pronominal
        # l-object clitic.  Keep the syncretic =l pronoun readings below as
        # alternatives, but rank the structurally licensed DOM reading first.
        #
        # Exact A-tier analyses have already had the chance to stop the cascade,
        # so this productive rule cannot override a corpus/author decision.
        if surface.endswith("l") and len(surface) > 4:
            host = surface[:-1]
            for surface_exp, underlying, agreement, gender in (
                ("il", "in", "MP", "M"),
                ("al", "an", "FP", "F"),
            ):
                if not host.endswith(surface_exp):
                    continue
                stem_surface = host[:-len(surface_exp)]
                for match in self._nc_root_matches(stem_surface, "clitic_plural"):
                    core = self._nc_candidate_gloss(match)
                    if not core:
                        continue
                    prefix, stem = match["prefix"], match["stem"]
                    stem_seg = f"{prefix}-{stem}" if prefix else stem
                    segmentation = f"{stem_seg}-{surface_exp}=l"
                    gloss = f"{core}-{agreement}=DOM"
                    ok, trace = reconstructs_to(segmentation, surface)
                    if not ok or not aligned(segmentation, gloss):
                        continue
                    out.append(Candidate(
                        surface=surface, segmentation=segmentation, gloss=gloss,
                        lemma_id=match["lexeme_id"], lemma=match["lemma"],
                        lexical_gloss=match["lexical"], lexical_gloss_status=match["status"],
                        features={"root": match["root"],
                                  "display_conjugation": match["display_class"],
                                  "nc_surface_model": match["model"],
                                  "nc_surface_context": "clitic_plural",
                                  "number": "plural", "gender": gender,
                                  "agreement_underlying": underlying,
                                  "agreement_surface_allomorph": surface_exp,
                                  "object": "DOM", "object_role": "DOM"},
                        tier="B1" if match["unique_binding"] else "B2",
                        score=0.91 if match["unique_binding"] else 0.66,
                        reconstruction_ok=True,
                        reconstruction_trace=([f"reverse NC {match['model']} √{match['root']}"]
                                              + self._nc_weak_trace(match)
                                              + [f"-{underlying} -> -{surface_exp} before l",
                                                 "terminal =l = DOM"] + trace),
                        evidence=[{"kind": "nc_plural_dom",
                                   "root": match["root"], "clitic": "l"}],
                    ))

        for clitic in self.clitics:
            if clitic.get("paradigm") != "l_object":
                continue
            form = clitic.get("form") or ""
            if not form or not surface.endswith(form):
                continue
            host = surface[:-len(form)]
            for surface_exp, underlying, agreement, gender in (
                ("il", "in", "MP", "M"),
                ("al", "an", "FP", "F"),
            ):
                if not host.endswith(surface_exp):
                    continue
                stem_surface = host[:-len(surface_exp)]
                for match in self._nc_root_matches(stem_surface, "clitic_plural"):
                    core = self._nc_candidate_gloss(match)
                    if not core:
                        continue
                    prefix, stem = match["prefix"], match["stem"]
                    stem_seg = f"{prefix}-{stem}" if prefix else stem
                    segmentation = f"{stem_seg}-{surface_exp}={form}"
                    for label, role, bonus in self._nc_object_readings(clitic):
                        gloss = f"{core}-{agreement}={label}"
                        ok, trace = reconstructs_to(segmentation, surface)
                        if not ok or not aligned(segmentation, gloss):
                            continue
                        out.append(Candidate(
                            surface=surface, segmentation=segmentation, gloss=gloss,
                            lemma_id=match["lexeme_id"], lemma=match["lemma"],
                            lexical_gloss=match["lexical"], lexical_gloss_status=match["status"],
                            features={"root": match["root"],
                                      "display_conjugation": match["display_class"],
                                      "nc_surface_model": match["model"],
                                      "nc_surface_context": "clitic_plural",
                                      "number": "plural", "gender": gender,
                                      "agreement_underlying": underlying,
                                      "agreement_surface_allomorph": surface_exp,
                                      "object": label, "object_role": role},
                            tier="B1" if match["unique_binding"] else "B2",
                            score=(0.88 if match["unique_binding"] else 0.63) + bonus,
                            reconstruction_ok=True,
                            reconstruction_trace=([f"reverse NC {match['model']} √{match['root']}"]
                                                  + self._nc_weak_trace(match)
                                                  + [f"-{underlying} -> -{surface_exp} before l",
                                                     f"indivisible clitic ={form}"] + trace),
                            evidence=[{"kind": "nc_plural_l_object",
                                       "root": match["root"], "clitic": form}],
                        ))
        return out

    def _from_nc_mp_object(self, surface: str) -> list[Candidate]:
        """Recover NC-MP followed by a third-person object enclitic.

        The surface ``-ill-`` is not sufficient evidence: ``amr-il=lun`` is a
        confirmed SC-1S form.  This rule therefore requires the same stem with
        independently analysable NC ``-in`` plural morphology.  Exact A-tier
        forms are tried in the earlier stage and keep their own object role.
        """
        if self._from_nc_root_plural_l_object(surface):
            return []

        # An exact corpus or author analysis decides both the subject exponent
        # and the object role.  The productive ambiguity is only a fallback for
        # forms that have no such evidence (mappille, but not amrillun).
        if (self._from_reusable_morphology(surface) or self._from_a0(surface)
                or self._from_a1(surface) or self._from_a3(surface)
                or self._from_a2(surface)):
            return []
        endings = (
            ("illun", "lun", "3MP"),
            ("ille", "le", "3MS"),
            ("illa", "la", "3FS"),
        )
        out = []
        for ending, clitic, label in endings:
            if not surface.endswith(ending) or len(surface) <= len(ending) + 2:
                continue
            stem = surface[:-len(ending)]
            plural_surface = stem + "in"
            plurals = [
                candidate for candidate in self._from_nominal_suffix(plural_surface)
                if (candidate.gloss or "").endswith((".NC-MP", ".PRS-MP"))
                and (candidate.segmentation or "").endswith("-in")
            ]
            if not plurals:
                # A few old corpus rows know the lexeme of every member of an
                # object series but carry no analysis for any of them.  Admit
                # that evidence only when the same verbal lexeme owns the bare
                # ``-ill`` plural and at least two members of the ``-ill+OBJ``
                # series.  This is much narrower than treating every surface
                # ending in ``ille`` as a plural verb, while recovering the
                # fully attested aspill/aspilla/aspille/aspillun series.
                exact_ids = {
                    row.get("lexeme_id")
                    for row in self.forms_by_surface.get(surface, [])
                    if row.get("lexeme_id") in self.verbal_lexeme_ids
                }
                if len(exact_ids) == 1:
                    lexeme_id = next(iter(exact_ids))
                    bare_plural = any(
                        row.get("lexeme_id") == lexeme_id
                        for row in self.forms_by_surface.get(stem + "ill", [])
                    )
                    series = sum(any(
                        row.get("lexeme_id") == lexeme_id
                        for row in self.forms_by_surface.get(stem + member, [])
                    ) for member in ("illa", "ille", "illun"))
                    if bare_plural and series >= 2:
                        lexical, status = self._lexical(lexeme_id, surface)
                        lemma = (self.lexemes.get(lexeme_id) or {}).get(
                            "lemma_canonical")
                        gloss_stem = lexical or gloss_placeholder(lemma)
                        if gloss_stem:
                            plurals = [Candidate(
                                surface=plural_surface,
                                segmentation=f"{stem}-in",
                                gloss=f"{gloss_stem}.NC-MP",
                                lemma_id=lexeme_id,
                                lemma=lemma,
                                lexical_gloss=lexical,
                                lexical_gloss_status=status,
                                features={"number": "plural",
                                          "attested_object_series": True},
                                tier="B1", score=0.7,
                                reconstruction_ok=True,
                                reconstruction_trace=[
                                    "same lexeme attests bare -ill plural and "
                                    "multiple -ill object forms"
                                ],
                                evidence=[{
                                    "kind": "attested_nc_mp_object_series",
                                    "forms": [stem + "ill", stem + "illa",
                                              stem + "ille", stem + "illun"],
                                }],
                            )]
            for plural in plurals[:1]:
                base_seg = plural.segmentation[:-3]
                base_gloss = plural.gloss.rsplit("-", 1)[0]
                base_gloss = re.sub(r"\.PRS$", ".NC", base_gloss)
                segmentation = f"{base_seg}-il={clitic}"
                for object_label, score in ((label, 0.74), (label + ".IO", 0.73)):
                    gloss = f"{base_gloss}-MP={object_label}"
                    if not aligned(segmentation, gloss):
                        continue
                    ok, trace = reconstructs_to(segmentation, surface)
                    if not ok:
                        continue
                    out.append(Candidate(
                        surface=surface, segmentation=segmentation, gloss=gloss,
                        lemma_id=plural.lemma_id, lemma=plural.lemma,
                        lexical_gloss=plural.lexical_gloss,
                        lexical_gloss_status=plural.lexical_gloss_status,
                        features={**plural.features, "number": "plural",
                                  "object": object_label,
                                  "object_role": "AMBIGUOUS"},
                        tier="B1", score=score, reconstruction_ok=True,
                        reconstruction_trace=plural.reconstruction_trace
                        + ["NC -in plural has -il before an object enclitic"] + trace,
                        evidence=plural.evidence + [{
                            "kind": "nc_mp_object",
                            "source": "Arnold Object Suffixes §5.1.2",
                        }],
                        uncertainties=[{
                            "type": "object_role_ambiguous",
                            "reason": "direct and indirect readings are both offered",
                        }]))
        return out

    # ---- public API ---------------------------------------------------
    def _candidate_stages(self, surface: str, token: str,
                          sentence_initial: bool,
                          context: dict | None = None) -> list[tuple[str, Any]]:
        """The ordered evidence stages. `analyze` stops at the first non-empty
        one; the diagnostic in `all_stage_candidates` runs them all."""
        return [
            ("author-confirmed reusable morphology", lambda:
                self._from_reusable_morphology(token)),
            ("token corrections and direct lookups", lambda: (
                self._from_token_correction(surface, context)
                + self._from_a0(surface) + self._from_a1(surface)
                + self._from_a3(surface) + self._from_a2(surface)
                + self._from_capitalised_direct(surface)
                + self._from_paradigm_v4(surface)
                + self._from_arnold_tables(surface))),
            ("composition: clitics, affixes, the nominal paradigm", lambda: (
                self._from_verbal_enclitic_sibling(surface)
                + self._from_cross_dialect(surface)
                + self._from_subject_plural(surface)
                + self._from_nc_root_plural(surface)
                + self._from_nc_root_plain_clitic(surface)
                + self._from_nc_root_f_clitic(surface)
                + self._from_nc_root_plural_l_object(surface)
                + self._from_nc_mp_object(surface)
                + self._from_compositional(surface) + self._from_past_prefix(surface)
                + self._from_proclitic(surface) + self._from_construct_clitic(surface)
                + self._from_assimilated_proclitic(surface)
                + self._from_nominal_suffix(surface))),
            ("an attested form with no analysis on record", lambda:
                self._from_attested_form(surface)),
            ("the Arnold working lexicon", lambda:
                self._from_lexicon_v3(surface)),
            ("a plausible respelling", lambda:
                self._from_spelling_variant(surface)),
            ("a prosthetic schwa breaking up an onset cluster", lambda:
                self._from_prosthetic_schwa(surface)),
            ("a capitalised unknown is a name (last-resort fallback)", lambda:
                self._as_proper_noun(surface, nfc(token).lstrip("-="))),
        ]

    def all_stage_candidates(self, token: str,
                             sentence_initial: bool = False) -> dict[str, Any]:
        """Diagnostic: run every stage (no early stop) and return the union,
        each candidate tagged with the stage that produced it and the index of
        the stage the production `analyze` would actually have stopped at.

        Includes the two post-stage wrappers ``analyze`` always adds — the
        free-state -a reading and the possessive -a reading — so the diagnostic
        sees exactly the candidate pool ``analyze`` ranks, not one short of it.

        Never used in production; it exists to measure how often a correct
        analysis is generated by a stage the early stop never reaches.
        """
        bare, _cuts = boundary_hints(fold_allophones(nfc(token)))
        surface = self.canonical(bare)
        stages = self._candidate_stages(surface, token, sentence_initial)
        stop_index = None
        produced: list[dict[str, Any]] = []
        settled: list[Candidate] = []
        for index, (name, produce) in enumerate(stages):
            found = produce()
            if found and stop_index is None:
                stop_index = index
                settled = found
            for candidate in found:
                if (candidate.segmentation
                        and not nonempty_segments(candidate.segmentation)):
                    continue
                produced.append({"stage_index": index, "stage": name,
                                 "candidate": candidate})
        if settled:
            for candidate in (self._from_nc_root_bare(surface, settled)
                              + self._from_nc_root_f(surface, settled)):
                produced.append({"stage_index": stop_index if stop_index is not None else len(stages),
                                 "stage": "post-stage NC-family competitors",
                                 "candidate": candidate})

        # the same wrappers analyze() applies to whatever stage settled
        wrap_index = stop_index if stop_index is not None else len(stages)
        rescues: list[Candidate] = []
        if not settled or self._weak_unsegmented_nominal_slot(settled):
            rescues += self._from_nominal_possessive(surface)
        if settled:
            rescues += self._from_same_lexeme_enclitic(surface, settled)
            rescues += self._root_licensed_feminine(settled, surface)
            rescues += self._free_state_a(settled, surface) + self._possessive_a(surface)
        for candidate in rescues:
            if (candidate.segmentation
                    and not nonempty_segments(candidate.segmentation)):
                continue
            produced.append({"stage_index": wrap_index,
                             "stage": "post-stage nominal wrappers",
                             "candidate": candidate})
        return {"surface": surface, "stop_index": stop_index, "produced": produced}

    def _demote_service_verb(self, candidate: Candidate) -> Candidate:
        """A reading bound to a service verb is a guess about an unlisted verb:
        tier C at best, so any real reading of the word outranks it."""
        if candidate.lemma_id not in self.service_verbs or candidate.lemma_id in self.lexemes:
            return candidate
        if TIER_ORDER[candidate.tier] < TIER_ORDER["C"]:
            candidate.tier = "C"
        candidate.score = min(candidate.score, 0.3)
        if not any(u.get("type") == "service_verb" for u in candidate.uncertainties):
            candidate.uncertainties = list(candidate.uncertainties) + [{
                "type": "service_verb",
                "reason": f"no verb of √{self.service_verbs[candidate.lemma_id]['root_canonical']} "
                          "in the project lexicon"}]
        return candidate

    def _enforce_verbal_binding(self, candidate: Candidate) -> Candidate:
        if not gloss_is_verbal(candidate.gloss) or self.is_verbal(candidate.lemma_id):
            return candidate
        old = self.lexemes.get(candidate.lemma_id) or {}
        root = nfc(candidate.features.get("root") or old.get("root_canonical") or "")
        if not root:
            return candidate                      # a late guess is worse than none
        stem = ""
        for form, gloss in zip(re.split(r"[-=]", candidate.segmentation or ""),
                               re.split(r"[-=]", candidate.gloss or "")):
            if has_lexical_part(gloss):
                stem = form
                break
        new_id, verbs = self.verb_for_root(root, stem, candidate.gloss or "")
        if not new_id or new_id == candidate.lemma_id:
            return candidate
        record = self.lexeme_record(new_id)
        new_lexical, status = self._lexical(new_id, candidate.surface)
        old_lexical = candidate.lexical_gloss or gloss_placeholder(old.get("lemma_canonical"))
        replacement = new_lexical or gloss_placeholder(record.get("lemma_canonical"))
        gloss = candidate.gloss
        if old_lexical and replacement and old_lexical in (gloss or ""):
            gloss = gloss.replace(old_lexical, replacement, 1)
        rebound = replace(
            candidate, gloss=gloss, lemma_id=new_id, lemma=record.get("lemma_canonical"),
            lexical_gloss=new_lexical, lexical_gloss_status=status,
            features={**candidate.features, "root": root,
                      **({"lexeme_candidates": verbs} if verbs else {})},
            uncertainties=list(candidate.uncertainties) + [{
                "type": "verbal_rebind", "from": candidate.lemma_id,
                "reason": "finite-verb label on a non-verb lexeme"}])
        # The reading itself keeps its tier: only the binding was wrong, and a
        # human-confirmed segmentation must not lose to a weaker parse.
        return rebound

    def analyze(self, token: str, sentence_initial: bool = False,
                context: dict | None = None, *, _normal_only: bool = False,
                _all_normal: bool = False) -> dict[str, Any]:
        original = token
        # A ``-``/``=`` typed by the user marks a probable morpheme or clitic
        # boundary. Strip it to get the surface form to look up, but remember
        # where the cuts were and prefer analyses that agree with them.
        # Allophonic subscripts go before the cuts are measured, not inside
        # ``canonical`` below: folding after would shift every offset and the
        # length guard would throw the author's segmentation away.
        bare, hinted_cuts = boundary_hints(fold_allophones(nfc(token)))
        surface = self.canonical(bare)
        # cut offsets are computed on the pre-normalisation string; only use
        # them when normalisation did not change the length
        if hinted_cuts and len(surface) != len(bare):
            hinted_cuts = frozenset()

        # Sources are tried in order and the search stops at the first stage
        # that yields anything, so weaker evidence never competes with stronger.
        # A conservative rescue that also ran the later stages whenever the
        # settled stage produced nothing that both reconstructs and carries a
        # real gloss was tried and reverted: the all-stage oracle
        # (scripts/diagnose_stages.py) showed early-stop blocks the gold on only
        # 3 held-out and 0 Santuko tokens, and the rescue moved no metric at all.
        stages = self._candidate_stages(surface, token, sentence_initial, context)
        candidates: list[Candidate] = []
        for name, produce in stages:
            if name == "a capitalised unknown is a name (last-resort fallback)":
                continue
            found = produce()
            candidates.extend(found)
            if (found and not _all_normal
                    and not all(c.lemma_id in self.service_verbs for c in found)):
                break

        # NC-family no-clitic readings compete with existing SC/PC/lexical evidence.
        candidates += self._from_nc_root_bare(surface, candidates)
        candidates += self._from_nc_root_f(surface, candidates)

        # Migrate legacy generic PRF only when the strict root+template parser
        # independently proves an NC1/NC2 subclass for the exact segmentation.
        candidates += self._prf_nc_structural_competitors(surface, candidates)
        candidates = self._canonicalize_legacy_prf(candidates)
        candidates = [self._demote_service_verb(c) for c in candidates]

        # A literal əCC spelling may itself be attested and therefore stop the
        # staged search before the older prosthetic-schwa fallback is reached.
        # Let the fallback compete only with a weak, unsegmented whole-form
        # reading, and only when both spellings resolve to the same lexeme.
        if self._weak_unsegmented_nominal_slot(candidates):
            direct_ids = {candidate.lemma_id for candidate in candidates
                          if candidate.lemma_id}
            prosthetic = [candidate for candidate
                          in self._initial_schwa_host_candidates(surface)
                          if candidate.lemma_id in direct_ids]
            for candidate in prosthetic:
                candidate.tier = "B1"
                candidate.score = max(candidate.score, 0.62)
            candidates += prosthetic

        same_lexeme_enclitic = self._from_same_lexeme_enclitic(surface, candidates)

        # Local nominal rescue.  It does not move or reorder any evidence
        # stage: an overt possessive analysis is merely offered alongside a weak
        # whole-form fallback, or on its own when the possessed form is unseen.
        possessive: list[Candidate] = []
        if not candidates or self._weak_unsegmented_nominal_slot(candidates):
            possessive = self._from_nominal_possessive(surface)
        if candidates:
            candidates = (
                candidates
                + same_lexeme_enclitic
                + possessive
                + self._root_licensed_feminine(candidates, surface)
                + self._free_state_a(candidates, surface)
                + self._possessive_a(surface)
            )
        elif possessive:
            candidates = possessive

        # A reading derived from an inflected stub by the wrappers above
        # (meʕl-a from meʕla, wazn=e from wazne) *is* the missing analysis:
        # it gets the stub's original B1 standing back, the bare stub keeps C.
        for candidate in candidates:
            if (candidate.features.get("unanalysed_inflected")
                    and any(mark in (candidate.segmentation or "") for mark in "-=")):
                candidate.features = {k: v for k, v in candidate.features.items()
                                      if k != "unanalysed_inflected"}
                if TIER_ORDER[candidate.tier] > TIER_ORDER["B1"]:
                    candidate.tier = "B1"
                candidate.score = max(candidate.score, 0.58)

        # h- of a demonstrative/deictic is lost after a proclitic (licensed
        # rule h_loss_after_proclitic): ʕalanna, bōṯe, mnōxa. Line 2 writes
        # the host as the author does, with its h and its own lemma:
        # ʕal=hann-a, b=hōṯe, mn=hōxa.
        candidates = [self._restore_lost_h(c, surface) for c in candidates]
        # ... and offered even when an earlier stage stopped the search on
        # something else (banna = bann=a 'I want her' is a direct hit). Not a
        # stage of its own: added after the search, so it never blocks one.
        if not any(re.match(r"^[^-=]+=h", c.segmentation or "") for c in candidates):
            candidates += self._from_h_dropped_host(surface)

        # D5: a finite-verb label on a non-verb lexeme is rebound to a verb of
        # the same root, whoever set the label (parser or reusable rule).
        candidates = [self._enforce_verbal_binding(c) for c in candidates]

        # deduplicate by (segmentation, gloss, lemma), keeping the best tier.
        # Label case is not a difference: the corpus writes to=2MS, the Arnold
        # tables and the journal write to=2ms, and as two keys they made one
        # reading look ambiguous and cost it HIGH (see `distinct` below).
        best: dict[tuple, Candidate] = {}
        for candidate in candidates:
            if candidate.segmentation and not nonempty_segments(candidate.segmentation):
                continue
            key = (candidate.segmentation, _label_case(candidate.gloss or ""),
                   candidate.lemma_id)
            # One reading takes the best of its evidence: the better tier, and
            # within a tier the higher score. Keeping the first row met gave a
            # reading attested twenty times the score of its unattested twin
            # (become.SC from the publication before become.PST of the corpus).
            kept = best.get(key)
            if (kept is None or TIER_ORDER[candidate.tier] < TIER_ORDER[kept.tier]
                    or (candidate.tier == kept.tier and candidate.score > kept.score)):
                best[key] = candidate

        def agrees(candidate: Candidate) -> bool:
            if not hinted_cuts or not candidate.segmentation:
                return False
            # Separators typed in the input are positive boundary hints, not a
            # claim that the token contains no other morpheme boundaries.  Thus
            # b-tepsa licenses b=teps-a: the hinted proclitic boundary is
            # preserved, while the independently motivated free-state -a may
            # still be segmented.
            candidate_cuts = boundary_hints(candidate.segmentation)[1]
            return hinted_cuts.issubset(candidate_cuts)

        ranked = sorted(best.values(),
                        key=lambda c: (not agrees(c), TIER_ORDER[c.tier], -c.score,
                                       not c.reconstruction_ok))
        # An inflected-but-unanalysed corpus form still knows its lexeme. If
        # the winning analysis names the same lemma without a lexicon link
        # (wazn=e from the Arnold/possessive path), hand the link over, so
        # the meaning and the dictionary link come with it.
        stubs = [c for c in ranked if c.features.get("unanalysed_inflected")
                 and c.lemma_id]
        if ranked and stubs and not ranked[0].lemma_id \
                and not ranked[0].features.get("unanalysed_inflected"):
            top = ranked[0]
            for stub in stubs:
                if top.lemma and stub.lemma and nfc(top.lemma) == nfc(stub.lemma):
                    top.lemma_id = stub.lemma_id
                    if not top.lexical_gloss or top.lexical_gloss_status in (
                            "arnold_sense_untranslated", "no_sense",
                            "translation_pending", "unknown_lexeme"):
                        lexical, status = self._lexical(stub.lemma_id, surface)
                        if lexical:
                            top.lexical_gloss, top.lexical_gloss_status = lexical, status
                    break

        confidence = "LOW"
        hint_satisfied = None
        if ranked:
            top = ranked[0]
            distinct = {(c.segmentation, _label_case(c.gloss or "")) for c in ranked}
            confidence = TIER_CONFIDENCE[top.tier]
            if top.gloss == "PN":
                confidence = "MEDIUM"
            if confidence == "HIGH" and len(distinct) > 1:
                confidence = "MEDIUM"          # weak ambiguity → suggestion only
            # A legacy token correction (author-confirmed for a context, but
            # without document coordinates) may win its surface, but it is never
            # presented as settled: it drops to review so the global-vs-local
            # question stays visible.
            if confidence == "HIGH" and top.features.get("legacy_scope"):
                confidence = "MEDIUM"
            if confidence == "HIGH" and any(
                    note.get("type") == "verbal_label_on_a_noun"
                    for note in top.uncertainties):
                confidence = "MEDIUM"
            if not top.reconstruction_ok:
                # a failed reconstruction may only lower confidence; the old
                # branch promoted LOW to MEDIUM, which is backwards
                confidence = {"HIGH": "MEDIUM", "MEDIUM": "LOW"}.get(confidence,
                                                                     confidence)
            # A lemma standing in for a missing English gloss is not something a
            # user can take as is. On a 235-token acceptance text seven of the
            # ten HIGH-confidence errors were exactly this: morphology correct,
            # gloss replaced by the lemma, presented as safe.
            if confidence == "HIGH" and top.lexical_gloss_status in (
                    "translation_pending", "no_sense", "unknown_lexeme"):
                confidence = "MEDIUM"
                top.uncertainties.append({
                    "type": "lexical_gloss_missing",
                    "reason": "morphology is analysed but the lemma stands in "
                              "for an English gloss"})
            if hinted_cuts:
                hint_satisfied = agrees(top)
                if hint_satisfied:
                    # the user's own boundaries were reproduced — that is
                    # stronger evidence than an unaided lookup
                    confidence = "HIGH" if top.reconstruction_ok else confidence
                else:
                    # never silently overrule a boundary the user typed
                    confidence = "MEDIUM" if confidence == "HIGH" else confidence
                    top.uncertainties.append({
                        "type": "segmentation_hint_unmet",
                        "reason": "no analysis reproduces the boundaries you typed"})
            # Re-apply the hard rule last: a satisfied segmentation hint may
            # raise confidence, but a missing English gloss still forbids HIGH.
            # Without this the hint promotion smuggled bib=lēlya "in=lēlya"
            # back to HIGH.
            if confidence == "HIGH" and top.lexical_gloss_status in (
                    "translation_pending", "no_sense", "unknown_lexeme"):
                confidence = "MEDIUM"
            confidence = prf_confidence(top.gloss, confidence)
            top.confidence = confidence
            for other in ranked[1:]:
                other.confidence = prf_confidence(
                    other.gloss, TIER_CONFIDENCE[other.tier]
                )

        result = {
            "token": original,
            "normalized": surface,
            "frequency": self.form_freq.get(surface, 0),
            "confidence": confidence,
            "best": ranked[0].to_dict() if ranked else None,
            "alternatives": [c.to_dict() for c in ranked[1:4]],
            "n_candidates": len(ranked),
            "needs_review": confidence != "HIGH",
            "segmentation_hint": sorted(hinted_cuts) if hinted_cuts else None,
            "segmentation_hint_satisfied": hint_satisfied,
        }

        # Confidence is evaluated before PN. Only weak capitalized results need
        # the expensive full normal cascade, including the lowercase spelling.
        if surface[:1].isupper() and not _all_normal:
            normal = bool(result["best"] and result["best"].get("gloss") != "PN"
                          and result["confidence"] in {"HIGH", "MEDIUM"})
            if not normal:
                spellings = list(dict.fromkeys([token, token[:1].lower() + token[1:]]))
                for spelling in spellings:
                    retry = self.analyze(spelling, sentence_initial, context,
                                         _normal_only=True, _all_normal=True)
                    if (retry.get("best") and retry["best"].get("gloss") != "PN"
                            and retry["confidence"] in {"HIGH", "MEDIUM"}):
                        retry["token"] = original
                        retry["normalized"] = surface
                        return retry
            # Sentence-initial capitalisation is orthographic, not evidence of
            # a name -- every word is capitalised there regardless. The PN
            # guess stays available mid-sentence (where capitalisation is
            # actually informative) but is withheld sentence-initially rather
            # than risk a routine word being relabelled as a name. This gate
            # was asserted by test_proper_noun_guess_is_never_high_and_never_
            # sentence_initial but never actually wired in -- sentence_initial
            # only ever reached the lowercase-retry recursion above, which
            # can't itself produce a PN result (_all_normal=True skips this
            # whole block on the way back in).
            if not normal and not _normal_only and not sentence_initial:
                names = self._as_proper_noun(surface, nfc(token).lstrip("-="))
                if names:
                    name = names[0]
                    name.confidence = TIER_CONFIDENCE[name.tier]
                    result.update(best=name.to_dict(), alternatives=[], n_candidates=1,
                                  confidence=name.confidence, needs_review=True)
        return result

    def analyze_many(self, tokens: Iterable[str]) -> list[dict[str, Any]]:
        return [self.analyze(t) for t in tokens]
