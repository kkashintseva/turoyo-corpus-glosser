from __future__ import annotations

from maalula_glosser.turoyo_verbs import (
    StemEvidence,
    TuroyoVerbParser,
    consonant_skeleton,
    token_parser_eligibility,
)


def parser() -> TuroyoVerbParser:
    return TuroyoVerbParser(evidence=[
        StemEvidence("nəšq", "kiss.infc", count=12),
        StemEvidence("ftəḥ", "open.pret", count=8),
        StemEvidence("goraš", "pull.infc", count=4),
        StemEvidence("gurš", "pull.infc", count=7),
    ])


def test_consonant_evidence_allows_vowel_change_but_not_consonant_change():
    assert consonant_skeleton("nəšq") == consonant_skeleton("nošaq") == "nšq"
    assert consonant_skeleton("nosak") != consonant_skeleton("nəšq")


def test_past_rule_uses_longest_specific_ending_first():
    rows = parser().suggest("nošaqwayno", limit=20)
    target = next(row for row in rows if row.rule == "past_1s")
    assert target.segmentation == "nošaq-way-no"
    assert target.gloss == "kiss.infc-pst-1s"
    assert target.evidence_match == "same_consonants"


def test_l_preterite_and_object_examples_are_available():
    rows = parser().suggest("ftəḥlele", limit=30)
    target = next(row for row in rows if row.rule == "lpret_a3ms_o3ms")
    assert target.segmentation == "ftəḥ-le-le"
    assert target.gloss == "open.pret-a.3ms-o.3ms"


def test_infectum_1s_object_example_is_segmented():
    rows = parser().suggest("gorašnux", limit=30)
    target = next(row for row in rows if row.rule == "infc_1s_o2ms")
    assert target.segmentation == "goraš-n-ux"
    assert target.gloss == "pull.infc-1s-o.2ms"


def test_parser_does_not_guess_without_verbal_stem_evidence():
    assert parser().suggest("xyzwayno") == []


def test_existing_flex_or_user_analysis_is_ineligible():
    imported = {
        "original_analysis": {"segmentation": "ftəḥ-le", "gloss": "open.pret-a.3ms"},
        "locked": True,
        "morphemes": [],
    }
    assert token_parser_eligibility(imported)[0] is False

    manual = {
        "locked": True,
        "morphemes": [{"origin": "user", "form": "foo", "gloss": "bar", "sep": ""}],
    }
    assert token_parser_eligibility(manual)[0] is False

    fresh = {
        "locked": False,
        "morphemes": [{"origin": "auto", "form": "nošaqwayno", "gloss": "", "suggested": "", "sep": ""}],
    }
    assert token_parser_eligibility(fresh) == (True, "")
