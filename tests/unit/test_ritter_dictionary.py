from maalula_glosser.ritter_dictionary import (
    RitterDictionary,
    consonant_skeleton,
    parse_ritter_text,
)


SAMPLE = """\
Helmut Ritter
Turoyo
B: Wörterbuch

ʕ

ʕabdo, pl –e عbdo, pl. –e m Knecht
qabila = qabile qabīla = qabīle f Stamm
qebəl qēbil fähig
šuro šūro m Mauer
suro sūro m Beispiel
žēb/ǧēb jēb m Tasche

a

abar abar Beispiel

tiefgestellte Textangabe bei Ritter angegeben
ar arabisch
"""


def test_consonant_skeleton_ignores_vowels_but_not_consonant_identity():
    assert consonant_skeleton("qəbəl") == "qbl"
    assert consonant_skeleton("qabila") == "qbl"

    assert consonant_skeleton("šuro") == "šr"
    assert consonant_skeleton("suro") == "sr"
    assert consonant_skeleton("šuro") != consonant_skeleton("suro")

    assert consonant_skeleton("ʕabdo") == "ʕbd"
    assert consonant_skeleton("ʕavdo") == "ʕvd"


def test_parser_keeps_raw_entry_and_slash_alternatives():
    entries = parse_ritter_text(SAMPLE)
    pocket = next(entry for entry in entries if entry.headword == "žēb")
    assert pocket.lookup_forms == ("žēb", "ǧēb")
    assert pocket.raw_entry.startswith("žēb/ǧēb")
    assert not any(entry.headword == "ar" for entry in entries)


def test_lookup_returns_exact_first_then_vowel_variants():
    dictionary = RitterDictionary(parse_ritter_text(SAMPLE))

    results = dictionary.lookup("qebəl", limit=10)
    assert results[0].kind == "exact"
    assert results[0].matched_form == "qebəl"

    forms = {(row.kind, row.matched_form) for row in results}
    assert ("vowel_variant", "qabila") in forms


def test_lookup_never_changes_consonants():
    dictionary = RitterDictionary(parse_ritter_text(SAMPLE))

    sh = dictionary.lookup("šero", limit=20)
    assert any(row.matched_form == "šuro" for row in sh)
    assert all(row.skeleton == "šr" for row in sh)

    s = dictionary.lookup("sero", limit=20)
    assert any(row.matched_form == "suro" for row in s)
    assert all(row.skeleton == "sr" for row in s)
    assert not any(row.matched_form == "šuro" for row in s)


def test_one_consonant_words_do_not_expand_by_default():
    dictionary = RitterDictionary(parse_ritter_text(SAMPLE))
    assert dictionary.lookup("e") == []
