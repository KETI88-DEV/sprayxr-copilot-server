import sqlite3

from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.db_viewer import create_db_viewer_router


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
