from __future__ import annotations

import csv
import io
from pathlib import Path

import docx
from fastapi.testclient import TestClient

from maalula_glosser.editor.app import create_app


LIFT = '''<?xml version="1.0" encoding="UTF-8"?>
<lift version="0.13">
  <entry id="oba_infc" guid="11111111-1111-1111-1111-111111111111">
    <lexical-unit><form lang="tru"><text>obaʕ</text></form></lexical-unit>
    <variant><form lang="tru"><text>әbʕ</text></form></variant>
    <sense id="want"><grammatical-info value="Infectum"/><gloss lang="en"><text>want.infc</text></gloss></sense>
    <sense id="need"><grammatical-info value="Infectum"/><gloss lang="en"><text>require.infc</text></gloss></sense>
  </entry>
  <entry id="aba_pret" guid="22222222-2222-2222-2222-222222222222">
    <lexical-unit><form lang="tru"><text>abəʕ</text></form></lexical-unit>
    <sense id="wanted"><grammatical-info value="Intransitive Preterite"/><gloss lang="en"><text>want.intr.pret</text></gloss></sense>
  </entry>
  <entry id="same_pret" guid="33333333-3333-3333-3333-333333333333">
    <lexical-unit><form lang="tru"><text>same</text></form></lexical-unit>
    <sense id="samep"><grammatical-info value="Preterite"/><gloss lang="en"><text>do.pret</text></gloss></sense>
  </entry>
  <entry id="same_infc" guid="44444444-4444-4444-4444-444444444444">
    <lexical-unit><form lang="tru"><text>same</text></form></lexical-unit>
    <sense id="samei"><grammatical-info value="Infectum"/><gloss lang="en"><text>do.infc</text></gloss></sense>
  </entry>
</lift>'''


def make_client(tmp_path: Path, monkeypatch):
    source = tmp_path / "dictionary.lift"
    source.write_text(LIFT, encoding="utf-8")
    monkeypatch.setenv("GLOSSER_PROFILE", "flex")
    monkeypatch.setenv("MAALULA_STATE_DIR", str(tmp_path / "state"))
    client = TestClient(create_app(Path("."), warm=False))
    imported = client.post("/api/documents/import-lift", json={"path": str(source)})
    assert imported.status_code == 200, imported.text
    return client, source


def test_lift_reimport_is_idempotent_and_homographs_remain_distinct(tmp_path: Path, monkeypatch):
    client, source = make_client(tmp_path, monkeypatch)
    first = client.get("/api/lift-lexicon").json()
    again = client.post("/api/documents/import-lift", json={"path": str(source)})
    assert again.status_code == 200
    second = client.get("/api/lift-lexicon").json()
    assert first["entries"] == second["entries"] == 4
    rows = [row for row in client.get("/api/senses").json()["rows"]
            if row.get("lemma") == "same"]
    assert len(rows) == 2
    assert {tuple(row["grammatical_infos"]) for row in rows} == {
        ("Preterite",), ("Infectum",)
    }


def test_manual_unknown_form_link_persists_and_reuses_lift_entry(tmp_path: Path, monkeypatch):
    client, _source = make_client(tmp_path, monkeypatch)
    oba = next(row for row in client.get("/api/lexicon/search", params={"q": "obaʕ"}).json()["results"]
               if row["lemma"] == "obaʕ")

    created = client.post("/api/documents", json={"title": "new", "text": "әbʕi", "translation": ""}).json()
    did = created["document_id"]
    token = client.get(f"/api/documents/{did}").json()["sentences"][0]["tokens"][0]
    editor = "fieldworks-link"
    assert client.post(f"/api/documents/{did}/lock/claim",
                       json={"client_id": editor, "name": "test"}).status_code == 200
    linked = client.post(
        f"/api/documents/{did}/tokens/{token['token_id']}/lemma",
        json={"lexeme_id": oba["lexeme_id"]}, headers={"x-editor-client": editor})
    assert linked.status_code == 200, linked.text
    linked_token = linked.json()["token"]
    assert linked_token["lemma"] == "obaʕ"
    assert linked_token["grammatical_info"] == "Infectum"
    assert linked_token["lift_entry_id"] == "oba_infc"

    morph_id = linked_token["morphemes"][0]["morph_id"]
    suggestions = client.get(f"/api/documents/{did}/suggest",
                             params={"morph_id": morph_id, "all_morpheme": True}).json()["morpheme_options"]
    lift = [row for row in suggestions if row.get("source") == "lift"]
    assert {row["label"] for row in lift} >= {"want.infc", "require.infc"}
    assert {row["grammatical_info"] for row in lift} == {"Infectum"}

    # The local overlay survives a fresh app/analyzer process and recognises
    # the new written form without changing the LIFT source itself.
    restarted = TestClient(create_app(Path("."), warm=False))
    second = restarted.post("/api/documents",
                            json={"title": "later", "text": "әbʕi", "translation": ""}).json()
    token2 = restarted.get(f"/api/documents/{second['document_id']}").json()["sentences"][0]["tokens"][0]
    assert token2["lemma"] == "obaʕ"
    assert token2["grammatical_info"] == "Infectum"
    assert token2["lift_entry_id"] == "oba_infc"


def test_category_reaches_context_csv_and_docx(tmp_path: Path, monkeypatch):
    client, _source = make_client(tmp_path, monkeypatch)
    created = client.post("/api/documents", json={"title": "ctx", "text": "әbʕ", "translation": "he wants"}).json()
    did = created["document_id"]
    document = client.get(f"/api/documents/{did}").json()
    token = document["sentences"][0]["tokens"][0]
    assert token["grammatical_info"] == "Infectum"

    concordance = client.get(f"/api/senses/{token['lemma_id']}/concordance").json()
    assert concordance["grammatical_infos"] == ["Infectum"]
    occurrence = next(row for row in concordance["text_occurrences"]
                      if row["token_id"] == token["token_id"])
    assert occurrence["grammatical_info"] == "Infectum"

    csv_response = client.get(f"/api/documents/{did}/export/csv")
    assert csv_response.status_code == 200
    rows = list(csv.DictReader(io.StringIO(csv_response.text)))
    assert rows and rows[0]["grammatical_info"] == "Infectum"
    assert rows[0]["lift_entry_id"] == "oba_infc"

    docx_response = client.get(f"/api/documents/{did}/export/docx")
    assert docx_response.status_code == 200
    out = tmp_path / "out.docx"
    out.write_bytes(docx_response.content)
    table = docx.Document(str(out)).tables[0]
    assert len(table.rows) == 2
    assert "Infectum" not in [cell.text for row in table.rows for cell in row.cells]


def test_local_sense_on_lift_entry_survives_lift_reimport(tmp_path: Path, monkeypatch):
    client, source = make_client(tmp_path, monkeypatch)
    oba = next(row for row in client.get("/api/lexicon/search", params={"q": "obaʕ"}).json()["results"]
               if row["lemma"] == "obaʕ")
    add = client.post(f"/api/senses/{oba['lexeme_id']}",
                      json={"gloss": "wish.infc", "gloss_language": "en"})
    assert add.status_code == 200, add.text
    again = client.post("/api/documents/import-lift", json={"path": str(source)})
    assert again.status_code == 200
    row = next(row for row in client.get("/api/senses").json()["rows"]
               if row.get("lexeme_id") == oba["lexeme_id"])
    assert {sense["gloss"] for sense in row["senses"]} >= {
        "want.infc", "require.infc", "wish.infc"
    }
