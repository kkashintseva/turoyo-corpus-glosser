from __future__ import annotations

from pathlib import Path

from maalula_glosser.flex_corpus import FlexCorpusTextGlosser
from maalula_glosser.flex_import import import_flextext_to_state
from maalula_glosser.ingest.flextext_parser import parse_flextext


FLEX = '''<?xml version="1.0" encoding="UTF-8"?>
<document><interlinear-text guid="t1">
<item type="title" lang="en">Tiny Turoyo</item>
<paragraphs><paragraph><phrases>
<phrase guid="p1"><item type="txt" lang="tru">Kətwa latwa.</item><item type="segnum">1</item><words>
<word guid="w1"><item type="txt" lang="tru">Kətwa</item><morphemes>
<morph type="stem"><item type="txt" lang="tru">kət</item><item type="gls" lang="en">exist</item></morph>
<morph type="enclitic"><item type="txt" lang="tru">=wa</item><item type="gls" lang="en">=pst</item></morph>
</morphemes></word>
<word guid="w2"><item type="txt" lang="tru">latwa</item><morphemes>
<morph type="stem"><item type="txt" lang="tru">lat</item><item type="gls" lang="en">neg.exist</item></morph>
<morph type="suffix"><item type="txt" lang="tru">wa</item><item type="gls" lang="en">pst</item></morph>
</morphemes></word>
<word><item type="punct" lang="tru">.</item></word>
</words><item type="gls" lang="en">There was, there wasn't.</item></phrase>
</phrases></paragraph></paragraphs></interlinear-text></document>'''


def test_flextext_preserves_morphemes_translation_and_raw_items(tmp_path: Path):
    source = tmp_path / "tiny.flextext"
    source.write_text(FLEX, encoding="utf-8")
    [text] = parse_flextext(source)
    segment = text["segments"][0]
    word = segment["words"][0]
    assert text["title"] == "Tiny Turoyo"
    assert segment["translation"] == "There was, there wasn't."
    assert word["analysis"]["segmentation"] == "kət=wa"
    assert word["analysis"]["gloss"] == "exist=pst"
    assert word["analysis"]["morphemes"][1]["raw_form"] == "=wa"
    assert word["analysis"]["morphemes"][1]["raw_gloss"] == "=pst"


def test_import_builds_exact_form_parser_without_maalula_rules(tmp_path: Path, monkeypatch):
    source = tmp_path / "tiny.flextext"
    source.write_text(FLEX, encoding="utf-8")
    monkeypatch.setenv("MAALULA_STATE_DIR", str(tmp_path / "state"))
    result = import_flextext_to_state(source, tmp_path)
    assert result["report"]["texts"] == 1
    assert result["report"]["tokens"] == 2

    glosser = FlexCorpusTextGlosser(tmp_path)
    known = glosser.gloss_token("Kətwa")
    unknown = glosser.gloss_token("never-seen")
    assert known["segmentation"] == "kət=wa"
    assert known["gloss"] == "exist=pst"
    assert known["confidence"] == "HIGH"
    assert unknown["confidence"] == "LOW"
    assert unknown["gloss"] is None

MORPHEME_DICTIONARY_FLEX = '''<document><interlinear-text guid="dict">
<item type="title" lang="en">Morpheme dictionary</item><paragraphs><paragraph><phrases>
<phrase guid="d1"><item type="txt" lang="tru">gab gab gab</item><words>
<word><item type="txt" lang="tru">gab</item><morphemes><morph type="stem"><item type="txt" lang="tru">gab</item><item type="gls" lang="en">lat</item></morph></morphemes></word>
<word><item type="txt" lang="tru">gab</item><morphemes><morph type="stem"><item type="txt" lang="tru">gab</item><item type="gls" lang="en">lat</item></morph></morphemes></word>
<word><item type="txt" lang="tru">gab</item><morphemes><morph type="stem"><item type="txt" lang="tru">gab</item><item type="gls" lang="en">chez</item></morph></morphemes></word>
</words></phrase></phrases></paragraph></paragraphs></interlinear-text></document>'''


def test_flex_profile_builds_morpheme_dictionary_and_learns_new_reusable_values(tmp_path: Path, monkeypatch):
    source = tmp_path / "morphs.flextext"
    source.write_text(MORPHEME_DICTIONARY_FLEX, encoding="utf-8")
    monkeypatch.setenv("MAALULA_STATE_DIR", str(tmp_path / "state"))
    import_flextext_to_state(source, tmp_path)

    glosser = FlexCorpusTextGlosser(tmp_path)
    options = glosser.analyzer.morpheme_gloss_options("gab")
    assert [(row["gloss"], row["count"]) for row in options] == [("lat", 2), ("chez", 1)]

    glosser.remember({
        "addition_id": "rm_new",
        "surface_canonical": "gab-o",
        "segmentation": "gab-o",
        "gloss": "beside-DEF",
    })
    options = glosser.analyzer.morpheme_gloss_options("gab")
    assert options[0]["gloss"] == "beside"
    assert options[0]["author"] is True
    assert {row["gloss"] for row in options} >= {"lat", "chez", "beside"}

PLUS_BOUNDARY_FLEX = '''<document><interlinear-text guid="plus">
<item type="title" lang="en">Plus boundaries</item><paragraphs><paragraph><phrases>
<phrase guid="pplus"><item type="txt" lang="tru">buatro</item><words>
<word><item type="txt" lang="tru">buatro</item><morphemes>
<morph type="proclitic"><item type="txt" lang="tru">b+</item><item type="gls" lang="en">in+</item></morph>
<morph type="clitic"><item type="txt" lang="tru">u+</item><item type="gls" lang="en">and+</item></morph>
<morph type="stem"><item type="txt" lang="tru">atro</item><item type="gls" lang="en">place</item></morph>
</morphemes></word></words></phrase>
</phrases></paragraph></paragraphs></interlinear-text></document>'''


def test_flex_plus_boundaries_are_preserved_as_plus_boundaries(tmp_path: Path):
    source = tmp_path / "plus.flextext"
    source.write_text(PLUS_BOUNDARY_FLEX, encoding="utf-8")
    [text] = parse_flextext(source)
    analysis = text["segments"][0]["words"][0]["analysis"]
    assert analysis["segmentation"] == "b+u+atro"
    assert analysis["gloss"] == "in+and+place"
    assert [m["sep"] for m in analysis["morphemes"]] == ["", "+", "+"]
    # Lossless metadata also keeps the source spelling with plus signs.
    assert analysis["morphemes"][0]["raw_form"] == "b+"
    assert analysis["morphemes"][1]["raw_form"] == "u+"
