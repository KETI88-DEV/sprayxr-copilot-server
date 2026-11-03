import sqlite3
import re

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
    assert 'href="/docs"' in body and "API 문서" in body


def test_db_viewer_searches_values_and_escapes_query(monkeypatch, tmp_path) -> None:
    client = _client(monkeypatch, tmp_path)

    filtered = client.get("/admin", params={"table": "sessions", "q": "S-07"})
    escaped = client.get("/admin", params={"table": "sessions", "q": "<script>"})

    assert "S-07" in filtered.text
    assert "S-08" not in filtered.text
    assert "<script>" not in escaped.text
    assert "&lt;script&gt;" in escaped.text


def test_table_record_popup_returns_nested_parameters_without_mutating_data(monkeypatch, tmp_path) -> None:
    client = _client(monkeypatch, tmp_path)
    assert 'data-detail-url=' in client.get('/admin?table=sessions').text
    response = client.get('/admin/api/record', params={'table':'sessions','rowid':1})
    assert response.status_code == 200
    data = response.json()['sections'][0]['data']
    assert data == {'session_id':'S-00','session_status':'ACTIVE','payload':{'score':88}}
    assert client.post('/admin/api/record?table=sessions&rowid=1').status_code == 405
    assert client.get('/admin/api/record?table=sessions&rowid=999').status_code == 404
    assert client.get('/admin/api/record',params={'table':'sessions"; DROP TABLE sessions;--','rowid':1}).status_code == 404
    assert client.get('/admin?table=sessions').status_code == 200


def test_record_details_preserve_nested_literal_strings_and_handle_legacy_nonfinite_values(monkeypatch, tmp_path) -> None:
    import json
    client = _client(monkeypatch, tmp_path)
    literal='{"question":1}'
    with sqlite3.connect(tmp_path/'viewer.db') as conn:
        conn.execute('UPDATE sessions SET payload=? WHERE rowid=1',(json.dumps({'prompt':literal,'old_value':float('nan')}),))
    response=client.get('/admin/api/record?table=sessions&rowid=1')
    assert response.status_code==200
    data=response.json()['sections'][0]['data']['payload']
    assert data['prompt']==literal
    assert data['old_value']=='NaN'


def test_public_viewer_rejects_mutating_methods_and_keeps_database_unchanged(monkeypatch, tmp_path) -> None:
    client = _client(monkeypatch, tmp_path)
    for method in ("POST", "PUT", "PATCH", "DELETE"):
        assert client.request(method, "/admin", json={"session_id": "S-00", "session_status": "CHANGED"}).status_code == 405
    with sqlite3.connect(tmp_path / "viewer.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 23
        assert conn.execute("SELECT session_status FROM sessions WHERE session_id='S-00'").fetchone()[0] == "ACTIVE"


def test_home_is_default_and_table_navigation_returns_home(monkeypatch, tmp_path) -> None:
    client = _client(monkeypatch, tmp_path)
    response = client.get("/admin")
    assert response.status_code == 200
    assert '<h1>Home</h1>' in response.text
    assert 'aria-current="page" href="/admin"' in response.text
    assert 'data-stat="sessions">23' in response.text
    assert 'href="/admin?table=sessions"' in response.text
    table = client.get("/admin?table=sessions")
    assert "JSON 보기" in table.text
    assert 'href="/admin"' in table.text


def _dashboard_client(monkeypatch, tmp_path) -> TestClient:
    path = tmp_path / "dashboard.db"
    with sqlite3.connect(path) as conn:
        conn.executescript("""
            CREATE TABLE users (user_id TEXT, name TEXT);
            INSERT INTO users VALUES ('W1','전형기'),('W2','<script>');
            CREATE TABLE sessions (session_id TEXT, user_id TEXT, scenario_id TEXT,
                session_status TEXT, started_at TEXT, ended_at TEXT);
            INSERT INTO sessions VALUES
                ('S1','W1','3','ACTIVE','2026-10-29T15:10:00Z',NULL),
                ('S2','W1','3','ACTIVE','2026-10-29T04:00:00Z',NULL),
                ('S3','W2','7','COMPLETED','2026-10-28T18:00:00Z','2026-10-29T05:00:00Z');
            CREATE TABLE sequences (session_key TEXT, sequence_id TEXT, received_at TEXT);
            INSERT INTO sequences VALUES
                ('S2','SEQ-001','2026-10-30T00:00:00Z'),
                ('S2','SEQ-002','2026-10-30T00:01:00Z'),
                ('S3','SEQ-001','2026-10-29T04:30:00Z'),
                ('legacy','SEQ-001','invalid');
            CREATE TABLE query_turns (id INTEGER, session_key TEXT, created_at TEXT);
            INSERT INTO query_turns VALUES (1,'S2','2026-10-29T04:30:00Z'),(2,'legacy','2026-10-30T01:00:00Z');
            CREATE TABLE completions (report_id TEXT,session_id TEXT,report_status TEXT,requested_at TEXT);
            INSERT INTO completions VALUES
                ('R1','S3','complete','2026-10-29T05:10:00Z'),
                ('R2','S2','pending','2026-10-30T00:02:00Z'),
                ('R3','legacy','failed','2026-10-30T01:01:00Z');
        """)
    monkeypatch.setenv("SEQUENCE_DB_PATH", str(path))
    app = FastAPI()
    app.include_router(create_db_viewer_router())
    return TestClient(app)


def test_home_aggregates_saved_records_and_keeps_session_report_states_separate(monkeypatch, tmp_path) -> None:
    response = _dashboard_client(monkeypatch, tmp_path).get("/admin")
    assert response.status_code == 200
    for key, expected in {"users": 2,"sessions": 3,"sequences": 4,"queries": 2,
                          "started": 1,"progress": 1,"ended": 1,
                          "reports_complete": 1,"reports_pending": 1,"reports_failed": 1}.items():
        assert re.search(rf'data-stat="{key}">{expected}(?:<|\s)', response.text), key
    assert "전형기" in response.text
    assert "&lt;script&gt;" in response.text and "<script>" not in response.text
    assert "학습모드 3" in response.text
    assert "API 호출 횟수" in response.text


def test_home_daily_chart_uses_korea_dates_and_includes_days_without_activity(monkeypatch, tmp_path) -> None:
    client = _dashboard_client(monkeypatch, tmp_path)
    # A fixed clock keeps the UTC-to-KST midnight boundary deterministic.
    import trainer.demo_api.db_dashboard as dashboard
    from datetime import datetime, timezone
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 30, 5, tzinfo=timezone.utc).astimezone(tz)
    monkeypatch.setattr(dashboard, "datetime", Clock)
    response = client.get("/admin?days=7&metric=sessions")
    assert response.status_code == 200
    assert 'data-date="2026-10-30" data-value="1"' in response.text
    assert 'data-date="2026-10-29" data-value="2"' in response.text
    assert 'data-date="2026-10-24" data-value="0"' in response.text
    assert 'href="/admin?days=7&amp;metric=sequences"' in response.text
    sequence = client.get("/admin?days=7&metric=sequences")
    assert 'data-date="2026-10-30" data-value="2"' in sequence.text
    assert 'data-date="2026-10-29" data-value="1"' in sequence.text


def test_home_without_database_is_useful_and_does_not_create_database(monkeypatch, tmp_path) -> None:
    path = tmp_path / "missing.db"
    monkeypatch.setenv("SEQUENCE_DB_PATH", str(path))
    app = FastAPI()
    app.include_router(create_db_viewer_router())
    response = TestClient(app).get("/admin")
    assert response.status_code == 200
    assert '<h1>Home</h1>' in response.text
    assert 'data-stat="sessions">0' in response.text
    assert "아직" in response.text
    assert not path.exists()
