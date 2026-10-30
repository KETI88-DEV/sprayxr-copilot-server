import sqlite3

from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.db_viewer import create_db_viewer_router
from trainer.demo_api.access_control import AccessControlMiddleware, OperatorCredentials


def _client(monkeypatch, tmp_path) -> TestClient:
    path = tmp_path / "viewer.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE sessions (session_id TEXT, session_status TEXT, payload TEXT)")
    for index in range(23):
        conn.execute(
            "INSERT INTO sessions VALUES (?, ?, ?)",
            (f"S-{index:02d}", "COMPLETED" if index % 2 else "ACTIVE", '{"score": 88}'),
        )
    conn.commit()
    conn.close()
    monkeypatch.setenv("SEQUENCE_DB_PATH", str(path))
    app = FastAPI()
    app.add_middleware(AccessControlMiddleware, credentials=OperatorCredentials(tmp_path / "operator.json"), protect_admin=True)
    app.include_router(create_db_viewer_router())
    return TestClient(app)


def test_db_viewer_has_readable_dashboard_and_navigation(monkeypatch, tmp_path) -> None:
    response = _client(monkeypatch, tmp_path).get("/admin?table=sessions&limit=10&page=2")

    assert response.status_code == 200
    body = response.text
    assert 'class="app-shell"' in body
    assert "Co-Pilot 데이터베이스" in body
    assert "읽기 전용" in body
    assert "전체 23건" in body
    assert "2 / 3 페이지" in body
    assert "S-12" in body
    assert "JSON 보기" in body
    assert "Swagger로 돌아가기" in body


def test_db_viewer_searches_values_and_escapes_query(monkeypatch, tmp_path) -> None:
    client = _client(monkeypatch, tmp_path)

    filtered = client.get("/admin", params={"table": "sessions", "q": "S-07"})
    escaped = client.get("/admin", params={"table": "sessions", "q": "<script>"})

    assert "S-07" in filtered.text
    assert "S-08" not in filtered.text
    assert "<script>" not in escaped.text
    assert "&lt;script&gt;" in escaped.text


def test_public_viewer_rejects_mutating_methods_and_keeps_database_unchanged(monkeypatch, tmp_path) -> None:
    client = _client(monkeypatch, tmp_path)
    for method in ("POST", "PUT", "PATCH", "DELETE"):
        assert client.request(method, "/admin", json={"session_id": "S-00", "session_status": "CHANGED"}).status_code == 405
    with sqlite3.connect(tmp_path / "viewer.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 23
        assert conn.execute("SELECT session_status FROM sessions WHERE session_id='S-00'").fetchone()[0] == "ACTIVE"
