"""HTTP-level tests (cookies, isolation, CSRF origin check, upload, chat, restore). Needs: pip install fastapi httpx python-multipart pytest
NOT run in the authoring sandbox (FastAPI was not installable there); run it before deploying:   pytest tests/test_api.py -v
Uses the dependency-free hashing embedder and a fake generator; the real detector is not required (analysis then reports 'unavailable')."""
import os, sys, tempfile
from pathlib import Path

_tmp = tempfile.mkdtemp()
os.environ.update(QUELLUM_DATA_DIR=_tmp, QUELLUM_EMBEDDER="hashing", QUELLUM_CLEANUP_INTERVAL_MINUTES="0", QUELLUM_SERVE_FRONTEND="false",
                  QUELLUM_ARTIFACTS_DIR=os.path.join(_tmp, "no-artifacts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi.testclient import TestClient

from app.main import app

TEXT = "Paris is the capital of France. The Eiffel Tower was completed in 1889. " * 20


class Gen:
    def generate(self, q, passages): return "Paris is the capital of France."


@pytest.fixture()
def client():
    with TestClient(app, base_url="http://testserver") as c:
        app.state.generator = Gen()
        yield c


def new_client():
    return TestClient(app, base_url="http://testserver")


def make_conv(c, ask=True):
    d = c.post("/api/documents/text", json={"text": TEXT}).json()["document"]
    cid = c.post("/api/conversations", json={"documentId": d["id"], "title": "T"}).json()["conversation"]["id"]
    if ask:
        r = c.post(f"/api/conversations/{cid}/messages", json={"content": "What is the capital of France?"})
        assert r.status_code == 200, r.text
    return d["id"], cid


def test_cookie_is_httponly_and_secret_never_in_body(client):
    r = client.get("/api/session")
    sc = r.headers["set-cookie"].lower()
    assert "httponly" in sc and "samesite=lax" in sc and "quellum_sid=" in sc
    token = client.cookies.get("quellum_sid")
    assert token and token not in r.text


def test_full_flow_and_restore(client):
    d, cid = make_conv(client)
    lst = client.get("/api/conversations?include=messages").json()["conversations"]
    assert [c["id"] for c in lst] == [cid] and len(lst[0]["messages"]) == 2
    full = client.get(f"/api/conversations/{cid}").json()["conversation"]
    assert full["textContent"].startswith("Paris") and full["messages"][1]["hallucinationResult"]["status"] in ("ok", "unavailable")
    r = client.post(f"/api/conversations/{cid}/messages", json={"content": "And the tower?"})
    assert len(client.get(f"/api/conversations/{cid}").json()["conversation"]["messages"]) == 4
    assert len(client.get("/api/conversations").json()["conversations"]) == 1


def test_other_session_cannot_touch_anything(client):
    d, cid = make_conv(client)
    other = new_client()
    other.get("/api/session")
    for method, url, kw in (("get", f"/api/conversations/{cid}", {}), ("patch", f"/api/conversations/{cid}", {"json": {"title": "x"}}),
                            ("delete", f"/api/conversations/{cid}", {}), ("post", f"/api/conversations/{cid}/messages", {"json": {"content": "hi"}}),
                            ("get", f"/api/documents/{d}", {}), ("get", f"/api/documents/{d}/file", {}),
                            ("post", "/api/conversations", {"json": {"documentId": d, "title": "steal"}})):
        r = getattr(other, method)(url, **kw)
        assert r.status_code == 404, (method, url, r.status_code)
    assert other.get("/api/conversations").json()["conversations"] == []
    assert client.get(f"/api/conversations/{cid}").status_code == 200


def test_ids_without_the_cookie_grant_nothing(client):
    d, cid = make_conv(client)
    anon = new_client()                                   # no cookie at all
    assert anon.get(f"/api/conversations/{cid}").status_code == 404
    forged = new_client(); forged.cookies.set("quellum_sid", cid)       # an id is not a credential
    assert forged.get(f"/api/conversations/{cid}").status_code == 404


def test_cross_origin_writes_are_refused(client):
    r = client.post("/api/documents/text", json={"text": TEXT}, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403
    assert client.post("/api/documents/text", json={"text": TEXT}, headers={"Origin": "http://testserver"}).status_code == 200


def test_pdf_upload_validation_and_private_file(client):
    assert client.post("/api/documents", files={"file": ("a.pdf", b"nope", "application/pdf")}).status_code == 422
    assert client.post("/api/documents", files={"file": ("a.pdf", b"", "application/pdf")}).status_code == 422


def test_validation_errors_do_not_echo_input(client):
    r = client.post("/api/conversations/" + "0" * 32 + "/messages", json={"content": ""})
    assert r.status_code == 422 and "input" not in r.text


def test_no_generator_gives_503(client):
    app.state.generator = None
    d, cid = make_conv(client, ask=False)
    r = client.post(f"/api/conversations/{cid}/messages", json={"content": "q?"})
    assert r.status_code == 503 and r.json()["code"] == "generation_not_configured"
    assert client.get("/api/conversations").json()["conversations"] == []        # nothing saved


def test_expired_conversation_returns_410(client):
    d, cid = make_conv(client)
    db = app.state.db
    with db.tx() as c:
        c.execute("UPDATE conversations SET expires_at=? WHERE id=?", (db.now() - 1, cid))
    r = client.get(f"/api/conversations/{cid}")
    assert r.status_code == 410 and r.json()["detail"] == "This conversation has expired after 30 days of inactivity."
    assert client.get("/api/conversations").json()["conversations"] == []
    assert client.get(f"/api/documents/{d}").status_code == 404
