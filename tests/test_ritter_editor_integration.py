from pathlib import Path

from fastapi.testclient import TestClient

from maalula_glosser.editor.app import create_app


ROOT = Path(__file__).resolve().parents[1]


def test_ritter_hints_are_exposed_for_editor_token(tmp_path, monkeypatch):
    monkeypatch.setenv("MAALULA_STATE_DIR", str(tmp_path / "state"))
    client = TestClient(create_app(ROOT, warm=False))

    created = client.post(
        "/api/documents",
        json={"title": "Ritter lookup", "text": "qəbəl", "translation": ""},
    )
    assert created.status_code == 200
    document_id = created.json()["document_id"]

    document = client.get(f"/api/documents/{document_id}").json()
    token = document["sentences"][0]["tokens"][0]

    response = client.get(
        f"/api/documents/{document_id}/tokens/{token['token_id']}/ritter?limit=5"
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is True
    assert payload["queries"][0]["skeleton"] == "qbl"
    assert payload["results"]
    assert all(row["skeleton"] == "qbl" for row in payload["results"])
    assert any(row["matched_form"] == "qabəl" for row in payload["results"])

    hover = client.get(
        f"/api/documents/{document_id}/tokens/{token['token_id']}/alternatives"
    )
    assert hover.status_code == 200
    assert hover.json()["ritter"]["results"]
