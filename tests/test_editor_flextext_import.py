from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from maalula_glosser.editor.app import create_app
from maalula_glosser.editor import session as S


AMBIGUOUS_FLEX = '''<document><interlinear-text guid="amb">
<item type="title" lang="en">Ambiguous corpus</item><paragraphs><paragraph><phrases>
<phrase guid="p1"><item type="txt" lang="tru">latwa latwa latwa</item><words>
<word><item type="txt" lang="tru">latwa</item><morphemes><morph type="stem"><item type="txt" lang="tru">lat</item><item type="gls" lang="en">neg.exist</item></morph><morph type="suffix"><item type="txt" lang="tru">wa</item><item type="gls" lang="en">pst</item></morph></morphemes></word>
<word><item type="txt" lang="tru">latwa</item><morphemes><morph type="stem"><item type="txt" lang="tru">lat</item><item type="gls" lang="en">neg.exist</item></morph><morph type="suffix"><item type="txt" lang="tru">wa</item><item type="gls" lang="en">pst</item></morph></morphemes></word>
<word><item type="txt" lang="tru">latwa</item><morphemes><morph type="stem"><item type="txt" lang="tru">lat</item><item type="gls" lang="en">neg.exist</item></morph><morph type="enclitic"><item type="txt" lang="tru">=wa</item><item type="gls" lang="en">=pst</item></morph></morphemes></word>
</words></phrase></phrases></paragraph></paragraphs></interlinear-text></document>'''


def test_editor_imports_flex_and_keeps_frequency_on_alternative(tmp_path: Path, monkeypatch):
    state = tmp_path / "state"
    source = tmp_path / "amb.flextext"
    source.write_text(AMBIGUOUS_FLEX, encoding="utf-8")
    monkeypatch.setenv("GLOSSER_PROFILE", "flex")
    monkeypatch.setenv("MAALULA_STATE_DIR", str(state))

    client = TestClient(create_app(Path("."), warm=False))
    imported = client.post("/api/documents/import-flextext", json={"path": str(source)})
    assert imported.status_code == 200
    assert imported.json()["report"]["corpus"]["forms"] == 1
    assert imported.json()["report"]["corpus"]["analyses"] == 2

    created = client.post("/api/documents", json={"title": "New", "text": "latwa", "translation": ""})
    assert created.status_code == 200
    document = client.get(f"/api/documents/{created.json()['document_id']}").json()
    token = document["sentences"][0]["tokens"][0]
    assert token["confidence"] == "MEDIUM"
    assert token["morphemes"][0]["form"] == "lat"
    assert token["alternatives"]
    assert token["alternatives"][0]["count"] == 1

    status = client.get("/api/flex-corpus").json()
    assert status["profile"] == "flex"
    assert status["evidence_tokens"] == 3


def test_flex_profile_uses_only_turoyo_flex_label_inventory(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MAALULA_STATE_DIR", str(tmp_path / "state"))
    client = TestClient(create_app(Path("."), warm=False))
    rows = client.get("/api/labels").json()["rows"]
    assert rows
    assert {row["category"] for row in rows} == {"FLEx corpus"}


def test_reusable_human_edit_outranks_flex_corpus(tmp_path: Path, monkeypatch):
    state = tmp_path / "state"
    source = tmp_path / "amb.flextext"
    source.write_text(AMBIGUOUS_FLEX, encoding="utf-8")
    monkeypatch.setenv("GLOSSER_PROFILE", "flex")
    monkeypatch.setenv("MAALULA_STATE_DIR", str(state))
    client = TestClient(create_app(Path("."), warm=False))

    imported = client.post("/api/documents/import-flextext", json={"path": str(source)}).json()
    did = imported["first_document_id"]
    document = client.get(f"/api/documents/{did}").json()
    token = document["sentences"][0]["tokens"][0]
    morph_id = token["morphemes"][-1]["morph_id"]

    client_id = "test-editor"
    claim = client.post(f"/api/documents/{did}/lock/claim",
                        json={"client_id": client_id, "name": "test"})
    assert claim.status_code == 200
    edited = client.patch(
        f"/api/documents/{did}/morphemes/{morph_id}",
        json={"gloss": "newpst", "reusable": True},
        headers={"x-editor-client": client_id},
    )
    assert edited.status_code == 200
    assert edited.json()["reusable"]["recorded"] is True

    created = client.post("/api/documents", json={"title": "After correction", "text": "latwa", "translation": ""})
    new_doc = client.get(f"/api/documents/{created.json()['document_id']}").json()
    new_token = new_doc["sentences"][0]["tokens"][0]
    assert new_token["confidence"] == "HIGH"
    assert new_token["morphemes"][-1]["gloss"] == "newpst"

MORPHEME_PICKER_FLEX = '''<document><interlinear-text guid="picker">
<item type="title" lang="en">Picker corpus</item><paragraphs><paragraph><phrases>
<phrase guid="p1"><item type="txt" lang="tru">gab gab</item><item type="segnum" lang="en">7</item><item type="gls" lang="en">At the house.</item><words>
<word><item type="txt" lang="tru">gab</item><morphemes><morph type="stem"><item type="txt" lang="tru">gab</item><item type="gls" lang="en">lat</item></morph></morphemes></word>
<word><item type="txt" lang="tru">gab</item><morphemes><morph type="stem"><item type="txt" lang="tru">gab</item><item type="gls" lang="en">chez</item></morph></morphemes></word>
</words></phrase></phrases></paragraph></paragraphs></interlinear-text></document>'''


def test_gloss_cell_suggests_all_flex_glosses_for_exact_morpheme(tmp_path: Path, monkeypatch):
    state = tmp_path / "state"
    source = tmp_path / "picker.flextext"
    source.write_text(MORPHEME_PICKER_FLEX, encoding="utf-8")
    monkeypatch.setenv("GLOSSER_PROFILE", "flex")
    monkeypatch.setenv("MAALULA_STATE_DIR", str(state))
    client = TestClient(create_app(Path("."), warm=False))

    client.post("/api/documents/import-flextext", json={"path": str(source)})
    created = client.post("/api/documents", json={"title": "New", "text": "gab", "translation": ""}).json()
    document = client.get(f"/api/documents/{created['document_id']}").json()
    morph_id = document["sentences"][0]["tokens"][0]["morphemes"][0]["morph_id"]

    res = client.get(f"/api/documents/{created['document_id']}/suggest",
                     params={"morph_id": morph_id, "prefix": "lat", "all_morpheme": True})
    assert res.status_code == 200
    options = res.json()["morpheme_options"]
    assert {row["label"] for row in options} == {"lat", "chez"}
    assert all("FLEx" in row["note"] for row in options)

    filtered = client.get(f"/api/documents/{created['document_id']}/suggest",
                          params={"morph_id": morph_id, "prefix": "ch"}).json()
    assert [row["label"] for row in filtered["morpheme_options"]] == ["chez"]


def test_line_two_plus_boundary_is_preserved_and_split_into_cells(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GLOSSER_PROFILE", "flex")
    monkeypatch.setenv("MAALULA_STATE_DIR", str(tmp_path / "state"))
    client = TestClient(create_app(Path("."), warm=False))

    created = client.post(
        "/api/documents",
        json={"title": "Plus", "text": "buatro", "translation": ""},
    ).json()
    did = created["document_id"]
    document = client.get(f"/api/documents/{did}").json()
    token = document["sentences"][0]["tokens"][0]

    client_id = "plus-test-editor"
    claim = client.post(
        f"/api/documents/{did}/lock/claim",
        json={"client_id": client_id, "name": "test"},
    )
    assert claim.status_code == 200

    edited = client.patch(
        f"/api/documents/{did}/tokens/{token['token_id']}/segmentation",
        json={"text": "b=u+atro"},
        headers={"x-editor-client": client_id},
    )
    assert edited.status_code == 200
    cells = edited.json()["token"]["morphemes"]
    assert [(cell["sep"], cell["form"]) for cell in cells] == [
        ("", "b"), ("=", "u"), ("+", "atro")
    ]
    assert S.join_parts(cells) == "b=u+atro"


def test_flex_senses_groups_same_lemma_and_keeps_all_meanings(tmp_path: Path, monkeypatch):
    state = tmp_path / "state"
    source = tmp_path / "picker.flextext"
    source.write_text(MORPHEME_PICKER_FLEX, encoding="utf-8")
    monkeypatch.setenv("GLOSSER_PROFILE", "flex")
    monkeypatch.setenv("MAALULA_STATE_DIR", str(state))
    client = TestClient(create_app(Path("."), warm=False))

    client.post("/api/documents/import-flextext", json={"path": str(source)})
    rows = [row for row in client.get("/api/senses").json()["rows"]
            if row.get("lemma") == "gab"]
    assert len(rows) == 1
    assert {sense["gloss"] for sense in rows[0]["senses"]} == {"lat", "chez"}
    # The old import format minted one technical id per lemma+gloss; the UI
    # now treats them as one lemma article rather than two dictionary entries.
    assert len(rows[0]["lexeme_ids"]) == 2



def test_flex_lexicon_concordance_exposes_full_context_and_sense(tmp_path: Path, monkeypatch):
    state = tmp_path / "state"
    source = tmp_path / "picker.flextext"
    source.write_text(MORPHEME_PICKER_FLEX, encoding="utf-8")
    monkeypatch.setenv("GLOSSER_PROFILE", "flex")
    monkeypatch.setenv("MAALULA_STATE_DIR", str(state))
    client = TestClient(create_app(Path("."), warm=False))

    client.post("/api/documents/import-flextext", json={"path": str(source)})
    row = next(row for row in client.get("/api/senses").json()["rows"]
               if row.get("lemma") == "gab")
    data = client.get(f"/api/senses/{row['lexeme_id']}/concordance").json()

    assert len(data["text_occurrences"]) == 2
    assert {hit["lexical_gloss"] for hit in data["text_occurrences"]} == {"lat", "chez"}
    assert {hit["line"] for hit in data["text_occurrences"]} == {"gab gab"}
    assert {hit["translation"] for hit in data["text_occurrences"]} == {"At the house."}
    assert {hit["segnum"] for hit in data["text_occurrences"]} == {"7"}
    assert {hit["segmentation"] for hit in data["text_occurrences"]} == {"gab"}
    assert {hit["gloss"] for hit in data["text_occurrences"]} == {"lat", "chez"}


def test_senses_ui_lets_lemma_and_sense_open_contexts():
    js = Path("web/editor/static/senses.js").read_text(encoding="utf-8")
    html = Path("web/editor/templates/senses.html").read_text(encoding="utf-8")
    assert 'name.addEventListener("click", () => toggleForms(row, card))' in js
    assert 'toggleForms(row, card, sense.gloss)' in js
    assert 'occurrence.translation' in js
    assert 'occurrence.segnum' in js
    assert "клик по конкретному значению" in html

def test_glossing_unknown_flex_word_adds_lemma_and_sense_to_dictionary(tmp_path: Path, monkeypatch):
    state = tmp_path / "state"
    monkeypatch.setenv("GLOSSER_PROFILE", "flex")
    monkeypatch.setenv("MAALULA_STATE_DIR", str(state))
    client = TestClient(create_app(Path("."), warm=False))

    created = client.post(
        "/api/documents", json={"title": "New text", "text": "foo", "translation": ""}
    ).json()
    did = created["document_id"]
    document = client.get(f"/api/documents/{did}").json()
    token = document["sentences"][0]["tokens"][0]
    morph_id = token["morphemes"][0]["morph_id"]
    client_id = "lexicon-test-editor"
    assert client.post(
        f"/api/documents/{did}/lock/claim",
        json={"client_id": client_id, "name": "test"},
    ).status_code == 200

    edited = client.patch(
        f"/api/documents/{did}/morphemes/{morph_id}",
        json={"gloss": "bar", "reusable": False},
        headers={"x-editor-client": client_id},
    )
    assert edited.status_code == 200
    assert edited.json()["lexical"]["recorded"] is True
    assert edited.json()["token"]["lemma"] == "foo"
    assert edited.json()["token"]["lemma_id"].startswith("lexnew_")

    row = next(row for row in client.get("/api/senses").json()["rows"]
               if row.get("lemma") == "foo")
    assert {sense["gloss"] for sense in row["senses"]} == {"bar"}
    assert row["senses"][0]["author"] is True
    assert row["needs_citation_form"] is False

    # A second contextual meaning is a sibling sense, not a replacement.
    edited2 = client.patch(
        f"/api/documents/{did}/morphemes/{morph_id}",
        json={"gloss": "baz", "reusable": False},
        headers={"x-editor-client": client_id},
    )
    assert edited2.status_code == 200
    row2 = next(row for row in client.get("/api/senses").json()["rows"]
                if row.get("lemma") == "foo")
    assert {sense["gloss"] for sense in row2["senses"]} == {"bar", "baz"}

    # The stem->meaning picker learns the author sense even though morphology
    # itself was not marked reusable, and the association survives a restart.
    restarted = TestClient(create_app(Path("."), warm=False))
    options = restarted.get(
        f"/api/documents/{did}/suggest",
        params={"morph_id": morph_id, "prefix": "", "all_morpheme": True},
    ).json()["morpheme_options"]
    assert {option["label"] for option in options} >= {"bar", "baz"}


def test_flex_dictionary_can_add_and_retract_one_sense_without_touching_siblings(tmp_path: Path, monkeypatch):
    state = tmp_path / "state"
    source = tmp_path / "picker.flextext"
    source.write_text(MORPHEME_PICKER_FLEX, encoding="utf-8")
    monkeypatch.setenv("GLOSSER_PROFILE", "flex")
    monkeypatch.setenv("MAALULA_STATE_DIR", str(state))
    client = TestClient(create_app(Path("."), warm=False))
    client.post("/api/documents/import-flextext", json={"path": str(source)})

    row = next(row for row in client.get("/api/senses").json()["rows"]
               if row.get("lemma") == "gab")
    added = client.post(f"/api/senses/{row['lexeme_id']}", json={"gloss": "side"})
    assert added.status_code == 200

    row = next(row for row in client.get("/api/senses").json()["rows"]
               if row.get("lemma") == "gab")
    assert {sense["gloss"] for sense in row["senses"]} == {"lat", "chez", "side"}
    side = next(sense for sense in row["senses"] if sense["gloss"] == "side")
    assert side["author"] is True
    assert len(side["addition_ids"]) == 1

    # A sense added in the dictionary itself is also a future morpheme choice.
    created = client.post("/api/documents", json={"title": "Probe", "text": "gab"}).json()
    probe = client.get(f"/api/documents/{created['document_id']}").json()
    morph_id = probe["sentences"][0]["tokens"][0]["morphemes"][0]["morph_id"]
    options = client.get(
        f"/api/documents/{created['document_id']}/suggest",
        params={"morph_id": morph_id, "prefix": "si", "all_morpheme": True},
    ).json()["morpheme_options"]
    assert options[0]["label"] == "side"
    assert options[0]["source"] == "author"

    retracted = client.post(
        f"/api/senses/{row['lexeme_id']}/readings/{side['addition_ids'][0]}/retract"
    )
    assert retracted.status_code == 200
    after = next(row for row in client.get("/api/senses").json()["rows"]
                 if row.get("lemma") == "gab")
    # The author-level reusable sense is gone.  The probe text itself still
    # contains ``side``, so the dictionary correctly keeps it as text evidence
    # rather than pretending the saved token never existed.
    side_after = next(sense for sense in after["senses"] if sense["gloss"] == "side")
    assert side_after["author"] is False
    assert side_after["sources"] == ["text"]
    assert {sense["gloss"] for sense in after["senses"]} == {"lat", "chez", "side"}

LIFT_LEXICON = '''<?xml version="1.0" encoding="UTF-8"?>
<lift version="0.13">
  <entry id="obaʕ_infc" guid="11111111-1111-1111-1111-111111111111">
    <lexical-unit><form lang="tru"><text>obaʕ</text></form></lexical-unit>
    <trait name="morph-type" value="stem"/>
    <variant><form lang="tru"><text>әbʕ</text></form></variant>
    <sense id="s-want"><grammatical-info value="Infectum"/><gloss lang="en"><text>want.infc</text></gloss></sense>
    <sense id="s-require"><grammatical-info value="Infectum"/><gloss lang="en"><text>require.infc</text></gloss></sense>
  </entry>
  <entry id="abəʕ_pret" guid="22222222-2222-2222-2222-222222222222">
    <lexical-unit><form lang="tru"><text>abəʕ</text></form></lexical-unit>
    <variant><form lang="tru"><text>ábəʕ</text></form></variant>
    <sense id="s-pret"><grammatical-info value="Intransitive Preterite"/><gloss lang="en"><text>want.intr.pret</text></gloss></sense>
  </entry>
  <entry id="same_pret" guid="33333333-3333-3333-3333-333333333333">
    <lexical-unit><form lang="tru"><text>same</text></form></lexical-unit>
    <sense id="s-same-pret"><grammatical-info value="Preterite"/><gloss lang="en"><text>do.pret</text></gloss></sense>
  </entry>
  <entry id="same_infc" guid="44444444-4444-4444-4444-444444444444">
    <lexical-unit><form lang="tru"><text>same</text></form></lexical-unit>
    <sense id="s-same-infc"><grammatical-info value="Infectum"/><gloss lang="en"><text>do.infc</text></gloss></sense>
  </entry>
</lift>'''


def _lift_client(tmp_path: Path, monkeypatch):
    source = tmp_path / "tiny.lift"
    source.write_text(LIFT_LEXICON, encoding="utf-8")
    monkeypatch.setenv("GLOSSER_PROFILE", "flex")
    monkeypatch.setenv("MAALULA_STATE_DIR", str(tmp_path / "state"))
    client = TestClient(create_app(Path("."), warm=False))
    imported = client.post("/api/documents/import-lift", json={"path": str(source)})
    assert imported.status_code == 200, imported.text
    return client


def test_lift_variant_resolves_to_exact_entry_and_keeps_fieldworks_category(tmp_path: Path, monkeypatch):
    client = _lift_client(tmp_path, monkeypatch)
    created = client.post("/api/documents", json={"title": "LIFT", "text": "әbʕ", "translation": ""}).json()
    document = client.get(f"/api/documents/{created['document_id']}").json()
    token = document["sentences"][0]["tokens"][0]
    assert token["lemma"] == "obaʕ"
    assert token["grammatical_info"] == "Infectum"
    assert token["grammatical_infos"] == ["Infectum"]
    assert token["lift_entry_id"] == "obaʕ_infc"
    assert token["lexeme_candidates"] == []

    search = client.get("/api/lexicon/search", params={"q": "әbʕ"}).json()["results"]
    oba = next(row for row in search if row["lemma"] == "obaʕ")
    assert oba["grammatical_infos"] == ["Infectum"]
    assert {sense["gloss"] for sense in oba["senses"]} == {"want.infc", "require.infc"}


def test_lift_homographic_entries_stay_separate_in_lexicon(tmp_path: Path, monkeypatch):
    client = _lift_client(tmp_path, monkeypatch)
    rows = [row for row in client.get("/api/senses").json()["rows"] if row.get("lemma") == "same"]
    assert len(rows) == 2
    assert {tuple(row["grammatical_infos"]) for row in rows} == {("Preterite",), ("Infectum",)}
    assert all(row["source"] == "lift" for row in rows)
    assert all(len(row["lexeme_ids"]) == 1 for row in rows)


def test_lift_sense_picker_carries_entry_sense_and_category_into_token(tmp_path: Path, monkeypatch):
    client = _lift_client(tmp_path, monkeypatch)
    created = client.post("/api/documents", json={"title": "Pick", "text": "әbʕ", "translation": ""}).json()
    did = created["document_id"]
    document = client.get(f"/api/documents/{did}").json()
    token = document["sentences"][0]["tokens"][0]
    morph_id = token["morphemes"][0]["morph_id"]
    suggestions = client.get(f"/api/documents/{did}/suggest",
                             params={"morph_id": morph_id, "all_morpheme": True}).json()["morpheme_options"]
    lift = [row for row in suggestions if row.get("source") == "lift"]
    assert {row["label"] for row in lift} == {"want.infc", "require.infc"}
    assert {row["grammatical_info"] for row in lift} == {"Infectum"}
    picked = next(row for row in lift if row["label"] == "want.infc")

    editor = "lift-picker"
    assert client.post(f"/api/documents/{did}/lock/claim",
                       json={"client_id": editor, "name": "test"}).status_code == 200
    edited = client.patch(
        f"/api/documents/{did}/morphemes/{morph_id}",
        json={"gloss": picked["label"], "reusable": False,
              "lexeme_id": picked["lexeme_id"], "sense_id": picked["sense_id"],
              "grammatical_info": picked["grammatical_info"]},
        headers={"x-editor-client": editor})
    assert edited.status_code == 200, edited.text
    token = edited.json()["token"]
    assert token["lemma"] == "obaʕ"
    assert token["grammatical_info"] == "Infectum"
    assert token["lift_sense_id"] == "s-want"
    assert token["lexical_gloss"] == "want.infc"


def test_new_author_entry_keeps_explicit_fieldworks_style_category(tmp_path: Path, monkeypatch):
    client = _lift_client(tmp_path, monkeypatch)
    created = client.post("/api/documents", json={"title": "New", "text": "foo", "translation": ""}).json()
    did = created["document_id"]
    token = client.get(f"/api/documents/{did}").json()["sentences"][0]["tokens"][0]
    editor = "new-entry"
    assert client.post(f"/api/documents/{did}/lock/claim",
                       json={"client_id": editor, "name": "test"}).status_code == 200
    res = client.post(
        f"/api/documents/{did}/tokens/{token['token_id']}/new-lemma",
        json={"lemma": "foo", "root": "", "gloss": "do.pret",
              "grammatical_info": "Preterite"},
        headers={"x-editor-client": editor})
    assert res.status_code == 200, res.text
    bound = res.json()["token"]
    assert bound["grammatical_info"] == "Preterite"
    rows = [row for row in client.get("/api/senses").json()["rows"] if row.get("lemma") == "foo"]
    assert len(rows) == 1
    assert rows[0]["grammatical_infos"] == ["Preterite"]

def test_lift_homograph_choice_keeps_contexts_and_categories_separate(tmp_path: Path, monkeypatch):
    client = _lift_client(tmp_path, monkeypatch)

    chosen = []
    for title, text, gloss, expected_gi in (("Pret", "same one", "do.pret", "Preterite"),
                                             ("Infc", "two same", "do.infc", "Infectum")):
        created = client.post("/api/documents", json={"title": title, "text": text, "translation": ""}).json()
        did = created["document_id"]
        document = client.get(f"/api/documents/{did}").json()
        token = next(token for token in document["sentences"][0]["tokens"]
                     if token["surface"] == "same")
        assert token["lemma_id"] is None
        assert len(token["lexeme_candidates"]) == 2
        morph_id = token["morphemes"][0]["morph_id"]
        options = client.get(f"/api/documents/{did}/suggest",
                             params={"morph_id": morph_id, "all_morpheme": True}).json()["morpheme_options"]
        pick = next(row for row in options if row.get("label") == gloss and row.get("source") == "lift")
        editor = "homograph-editor"
        assert client.post(f"/api/documents/{did}/lock/claim",
                           json={"client_id": editor, "name": "test"}).status_code == 200
        edited = client.patch(
            f"/api/documents/{did}/morphemes/{morph_id}",
            json={"gloss": pick["label"], "reusable": False,
                  "lexeme_id": pick["lexeme_id"], "sense_id": pick["sense_id"],
                  "grammatical_info": pick["grammatical_info"]},
            headers={"x-editor-client": editor})
        assert edited.status_code == 200, edited.text
        bound = edited.json()["token"]
        assert bound["grammatical_info"] == expected_gi
        chosen.append((pick["lexeme_id"], title, expected_gi))

    for lexeme_id, expected_title, expected_gi in chosen:
        concordance = client.get(f"/api/senses/{lexeme_id}/concordance").json()
        occurrences = concordance["text_occurrences"]
        assert len(occurrences) == 1
        assert occurrences[0]["title"] == expected_title
        assert occurrences[0]["grammatical_info"] == expected_gi

def test_new_manual_sense_on_lift_entry_appears_in_lexicon(tmp_path: Path, monkeypatch):
    client = _lift_client(tmp_path, monkeypatch)
    created = client.post("/api/documents", json={"title": "New sense", "text": "әbʕ", "translation": ""}).json()
    did = created["document_id"]
    document = client.get(f"/api/documents/{did}").json()
    token = document["sentences"][0]["tokens"][0]
    assert token["lemma"] == "obaʕ"
    morph_id = token["morphemes"][0]["morph_id"]
    editor = "new-lift-sense"
    assert client.post(f"/api/documents/{did}/lock/claim",
                       json={"client_id": editor, "name": "test"}).status_code == 200
    edited = client.patch(
        f"/api/documents/{did}/morphemes/{morph_id}",
        json={"gloss": "wish.custom", "reusable": False,
              "lexeme_id": token["lemma_id"], "grammatical_info": "Infectum"},
        headers={"x-editor-client": editor})
    assert edited.status_code == 200, edited.text
    rows = [row for row in client.get("/api/senses").json()["rows"]
            if row.get("lift_id") == "obaʕ_infc"]
    assert len(rows) == 1
    custom = next(sense for sense in rows[0]["senses"] if sense["gloss"] == "wish.custom")
    assert custom["author"] is True
    assert "author" in custom["sources"]
    assert custom["grammatical_info"] == "Infectum"

