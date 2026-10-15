from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from maalula_glosser.editor.app import create_app


VERB_FLEX = '''<document><interlinear-text guid="verbs">
<item type="title" lang="en">Verb evidence</item><paragraphs><paragraph><phrases>
<phrase guid="p1"><item type="txt" lang="tru">nəšqo</item><words>
<word guid="w1"><item type="txt" lang="tru">nəšqo</item><morphemes>
<morph type="stem"><item type="txt" lang="tru">nəšq</item><item type="gls" lang="en">kiss.infc</item></morph>
<morph type="suffix"><item type="txt" lang="tru">o</item><item type="gls" lang="en">f</item></morph>
</morphemes></word>
</words></phrase></phrases></paragraph></paragraphs></interlinear-text></document>'''


def _client(tmp_path: Path, monkeypatch):
    state = tmp_path / "state"
    source = tmp_path / "verbs.flextext"
    source.write_text(VERB_FLEX, encoding="utf-8")
    monkeypatch.setenv("MAALULA_STATE_DIR", str(state))
    client = TestClient(create_app(Path("."), warm=False))
    imported = client.post("/api/documents/import-flextext", json={"path": str(source)})
    assert imported.status_code == 200, imported.text
    return client, imported.json()


def test_imported_flex_analysis_is_never_offered_or_overwritten(tmp_path: Path, monkeypatch):
    client, imported = _client(tmp_path, monkeypatch)
    did = imported["first_document_id"]
    document = client.get(f"/api/documents/{did}").json()
    token = document["sentences"][0]["tokens"][0]
    before = token["morphemes"]
    assert token["original_analysis"]

    suggestions = client.get(
        f"/api/documents/{did}/tokens/{token['token_id']}/verb-suggestions"
    )
    assert suggestions.status_code == 200
    assert suggestions.json()["eligible"] is False
    assert suggestions.json()["suggestions"] == []

    editor = "verb-protection-test"
    assert client.post(
        f"/api/documents/{did}/lock/claim",
        json={"client_id": editor, "name": "test"},
    ).status_code == 200
    refused = client.post(
        f"/api/documents/{did}/tokens/{token['token_id']}/verb-suggestions/accept",
        json={"suggestion_id": "vsg_fake"},
        headers={"x-editor-client": editor},
    )
    assert refused.status_code == 409
    after = client.get(f"/api/documents/{did}").json()["sentences"][0]["tokens"][0]
    assert after["morphemes"] == before
    assert after["original_analysis"] == token["original_analysis"]


def test_new_unanalysed_word_gets_rule_suggestion_and_accept_is_local(tmp_path: Path, monkeypatch):
    client, _imported = _client(tmp_path, monkeypatch)
    created = client.post(
        "/api/documents",
        json={"title": "New", "text": "nošaqwayno", "translation": ""},
    ).json()
    did = created["document_id"]
    document = client.get(f"/api/documents/{did}").json()
    token = document["sentences"][0]["tokens"][0]
    assert token.get("original_analysis") is None
    assert all(not (m.get("gloss") or m.get("suggested")) for m in token["morphemes"])

    response = client.get(
        f"/api/documents/{did}/tokens/{token['token_id']}/verb-suggestions"
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["eligible"] is True
    target = next(row for row in data["suggestions"]
                  if row["segmentation"] == "nošaq-way-no")
    assert target["gloss"] == "kiss.infc-pst-1s"
    assert target["evidence"]["match"] == "same_consonants"

    editor = "verb-parser-editor"
    assert client.post(
        f"/api/documents/{did}/lock/claim",
        json={"client_id": editor, "name": "test"},
    ).status_code == 200
    accepted = client.post(
        f"/api/documents/{did}/tokens/{token['token_id']}/verb-suggestions/accept",
        json={"suggestion_id": target["suggestion_id"]},
        headers={"x-editor-client": editor},
    )
    assert accepted.status_code == 200, accepted.text
    updated = accepted.json()["token"]
    assert updated["locked"] is True
    assert [(m["sep"], m["form"], m["gloss"], m["origin"])
            for m in updated["morphemes"]] == [
        ("", "nošaq", "kiss.infc", "user"),
        ("-", "way", "pst", "user"),
        ("-", "no", "1s", "user"),
    ]
    assert updated.get("original_analysis") is None

    # Acceptance is a local human edit; it must not teach the automatic corpus
    # parser or rewrite a second new occurrence behind the user's back.
    second = client.post(
        "/api/documents",
        json={"title": "Second", "text": "nošaqwayno", "translation": ""},
    ).json()
    second_token = client.get(
        f"/api/documents/{second['document_id']}"
    ).json()["sentences"][0]["tokens"][0]
    assert all(not (m.get("gloss") or m.get("suggested"))
               for m in second_token["morphemes"])


def test_editor_js_contains_explicit_verb_parser_ui():
    js = Path("web/editor/static/editor.js").read_text(encoding="utf-8")
    assert "Глагольный разбор…" in js
    assert "/verb-suggestions/accept" in js
    assert "Применить этот разбор" in js
