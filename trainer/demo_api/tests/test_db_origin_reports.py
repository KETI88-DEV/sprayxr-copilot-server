"""Origin metadata must describe creation evidence and never guess historical data."""
import html
import re
import sqlite3
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlsplit

from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.db_viewer import create_db_viewer_router
from trainer.demo_api.v1_store import V1Store
from trainer.demo_api.report_jobs import ReportJobs, save_artifact


def test_origin_migration_preserves_unknown_history_and_retry_origin(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as conn:
        conn.executescript("""CREATE TABLE sessions(session_id TEXT PRIMARY KEY,user_id TEXT,scenario_id TEXT,
          session_status TEXT,started_at TEXT,ended_at TEXT);
          INSERT INTO sessions VALUES('TEST-NAME','U','3','ACTIVE','old',NULL);""")
    store = V1Store(str(path))
    store.start_session(session_id="NORMAL",user_id="U",scenario_id="3",started_at="t",
                        creation_origin="real",creation_path="/api/v1/sessions/start")
    store.start_session(session_id="NORMAL",user_id="U",scenario_id="3",started_at="t",
                        creation_origin="test",creation_path="/api/v1/test/workflow/1-start")
    store.start_session(session_id="UNMARKED",user_id="U",scenario_id="3",started_at="t")
    store.start_session(session_id="TEST-NAME",user_id="U",scenario_id="3",started_at="old",creation_origin="real")
    store.end_session(session_id="END-ONLY",user_id="U",scenario_id="3",ended_at="t")
    store.close()
    reopened = V1Store(str(path))
    reopened.close()
    with sqlite3.connect(path) as conn:
        rows = dict(conn.execute("SELECT session_id,creation_origin FROM sessions"))
        assert rows == {"TEST-NAME":None,"NORMAL":"real","UNMARKED":None,"END-ONLY":None}
        assert conn.execute("SELECT creation_path FROM sessions WHERE session_id='NORMAL'").fetchone()[0] == "/api/v1/sessions/start"


def _viewer(monkeypatch, tmp_path):
    path = tmp_path / "viewer.db"
    state = V1Store(str(path))
    for sid, origin, mode in (("A-TEST-IN-NAME","real","3"),("B","test","7"),("C",None,"9")):
        state.start_session(session_id=sid,user_id=sid,scenario_id=mode,started_at="2026-11-03T00:00:00Z",creation_origin=origin)
        state.upsert_user(user_id=sid,skill_level="novice",dominant_hand="right",created_at="t")
        state.create_completion(session_id=sid,requested_at="2026-11-03T00:00:00Z",payload={})
    with sqlite3.connect(path) as conn:
        conn.executescript("""CREATE TABLE sequences(session_key TEXT,sequence_id TEXT,received_at TEXT);
          INSERT INTO sequences VALUES('A-TEST-IN-NAME','SEQ1','2026-11-03T00:00:00Z'),('B','SEQ1','2026-11-03T00:00:00Z'),('C','SEQ1','2026-11-03T00:00:00Z'),('ORPHAN','SEQ1','2026-11-03T00:00:00Z');
          CREATE TABLE query_turns(session_key TEXT,created_at TEXT);
          INSERT INTO query_turns VALUES('A-TEST-IN-NAME','2026-11-03T00:00:00Z'),('B','2026-11-03T00:00:00Z'),('C','2026-11-03T00:00:00Z');
          CREATE TABLE api_calls(id INTEGER,session_id TEXT,source TEXT,category TEXT,status_code INTEGER);
          INSERT INTO api_calls VALUES(1,'C','http','start',200);""")
    state.close()
    monkeypatch.setenv("SEQUENCE_DB_PATH",str(path))
    app = FastAPI()
    app.include_router(create_db_viewer_router())
    return TestClient(app), path


def stat(body, key):
    return int(re.search(rf'data-stat="{key}">([\d,]+)<',body).group(1).replace(",",""))


def test_server_entrypoints_record_origin_without_changing_api_body(monkeypatch,tmp_path):
    from trainer.demo_api.app import create_app
    path = tmp_path / "http.db"
    monkeypatch.setenv("SEQUENCE_DB_PATH",str(path))
    monkeypatch.setenv("DEMO_LIVE_MODE","off")
    client = TestClient(create_app())
    response = client.post("/api/v1/sessions/start",json={"session_id":"TEST-NAMED-NORMAL","user_id":"U","scenario_id":"3","started_at":"2026-11-03T00:00:00Z","worker":{"skill_level":"novice"}})
    assert response.status_code == 200
    assert "creation_origin" not in response.json()["data"]
    demo = client.post("/api/v1/test/workflow/1-start").json()["data"]["session_id"]
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT creation_origin,creation_path FROM sessions WHERE session_id='TEST-NAMED-NORMAL'").fetchone() == ("real","/api/v1/sessions/start")
        assert conn.execute("SELECT creation_origin FROM sessions WHERE session_id=?",(demo,)).fetchone()[0] == "test"


def test_filters_cover_dashboard_counts_trends_content_recent_and_raw_links(monkeypatch,tmp_path):
    client, path = _viewer(monkeypatch,tmp_path)
    real = client.get("/admin?origin=real&days=90").text
    for key in ("sessions","users","sequences","queries","reports_pending"):
        assert stat(real,key) == 1
    assert "A-TEST-IN-NAME</a>" in real and "학습모드 3" in real
    assert "학습모드 7" not in real and "학습모드 9" not in real
    assert 'data-date="2026-11-03" data-value="1"' in real
    assert "일반 연동" in real and "수동 테스트" in real and "미분류" in real
    for url in re.findall(r'href="([^"]+)"',real):
        parsed = urlsplit(html.unescape(url))
        if parsed.path.startswith("/admin"):
            assert parse_qs(parsed.query).get("origin") == ["real"], url
    assert 'name="origin" value="real"' in real
    raw = client.get("/admin?table=sequences&origin=real").text
    assert "전체 1건" in raw and "ORPHAN" not in raw
    unknown = client.get("/admin?origin=unknown").text
    assert stat(unknown,"sessions") == 1 and stat(unknown,"sequences") == 2
    assert stat(client.get("/admin").text,"sessions") == 3
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT creation_origin FROM sessions WHERE session_id='C'").fetchone()[0] is None


def test_session_counts_and_navigation_preserve_origin(monkeypatch,tmp_path):
    client, _ = _viewer(monkeypatch,tmp_path)
    body = client.get("/admin/sessions?origin=test&state=unfinished&q=B").text
    assert stat(body,"sessions") == 1 and stat(body,"unfinished") == 1
    assert "B</strong>" in body and "A-TEST-IN-NAME</strong>" not in body
    assert "모의 테스트" in body
    for url in re.findall(r'href="([^"]+)"',body):
        parsed = urlsplit(html.unescape(url))
        if parsed.path.startswith("/admin"):
            assert parse_qs(parsed.query).get("origin") == ["test"], url
    detail = client.get("/admin/sessions/view?session_id=B&origin=test").text
    assert 'href="/admin/sessions?origin=test"' in detail
    assert "모의 테스트" in detail


def test_pending_metadata_distinguishes_fresh_old_unknown_and_completed_unavailable():
    from trainer.demo_api.db_report_status import report_progress
    now = datetime(2026,11,3,1,tzinfo=timezone.utc)
    old = report_progress({"report_status":"pending","requested_at":"2026-10-01T00:00:00Z"},now=now)
    assert old["stale"] is True and "장기 대기" in old["label"]
    assert old["reason"] == "미기록 · 진행 여부 확인 불가"
    assert old["latest_at"] is None and old["elapsed"] == "33일 1시간"
    fresh = report_progress({"report_status":"pending","requested_at":"2026-11-03T00:58:00Z","processing_reason":"queued","processing_updated_at":"2026-11-03T00:58:00Z"},now=now)
    assert fresh["stale"] is False and fresh["elapsed"] == "2분"
    unknown = report_progress({"report_status":"pending","requested_at":"bad"},now=now)
    assert unknown["elapsed"] == "알 수 없음" and not unknown["stale"]
    completed = report_progress({"report_status":"complete","requested_at":"2026-10-01T00:00:00Z","payload":'{"analysis":{"pose_report":{"analysis_status":"unavailable","reason_code":"engine_unavailable"}}}'},now=now)
    assert completed["label"] == "처리 완료" and completed["elapsed"] == "해당 없음"
    assert "engine_unavailable" not in completed["reason"]


def test_report_lifecycle_records_observed_processing_and_no_migration_reprocessing(tmp_path):
    path = tmp_path / "jobs.db"
    state = V1Store(str(path))
    old = state.create_completion(session_id="old",requested_at="2026-10-01T00:00:00Z",payload={})
    name = save_artifact(state.artifact_dir,b"a,b\n1,2\n")
    row = state.create_completion(session_id="new",requested_at="t",payload={"artifacts":{"telemetry":name},"user_id":"U","scenario_id":"3"})
    def analyze(report_id, **kwargs):
        with sqlite3.connect(path) as conn:
            started, reason = conn.execute("SELECT processing_started_at,processing_reason FROM completions WHERE report_id=?",(report_id,)).fetchone()
        assert started and reason == "processing"
        state.finish_completion(report_id,status="complete",analysis={"pose_report":{"analysis_status":"unavailable"}})
    ReportJobs(state,analyze).run(row.report_id)
    state.close()
    V1Store(str(path)).close()
    with sqlite3.connect(path) as conn:
        result = conn.execute("SELECT report_status,processing_finished_at,processing_reason FROM completions WHERE report_id=?",(row.report_id,)).fetchone()
        assert result[0] == "complete" and result[1] and result[2] == "complete"
        assert conn.execute("SELECT report_status,processing_started_at FROM completions WHERE report_id=?",(old.report_id,)).fetchone() == ("pending",None)


def test_legacy_pending_migration_does_not_invent_processing_timestamps(tmp_path):
    path = tmp_path / "legacy-reports.db"
    with sqlite3.connect(path) as conn:
        conn.executescript("""CREATE TABLE completions(report_id TEXT PRIMARY KEY,session_id TEXT,report_status TEXT,requested_at TEXT,payload TEXT);
          INSERT INTO completions VALUES('R-old','S-old','pending','2026-10-01T00:00:00Z','{}');""")
    V1Store(str(path)).close()
    V1Store(str(path)).close()
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT report_status,processing_started_at,processing_updated_at,processing_finished_at,processing_reason FROM completions").fetchone() == ('pending',None,None,None,None)


def test_pending_progress_is_visible_in_dashboard_session_raw_record_and_modal(monkeypatch,tmp_path):
    client,path = _viewer(monkeypatch,tmp_path)
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE completions SET requested_at='2026-10-01T00:00:00Z',processing_updated_at=NULL,processing_reason=NULL WHERE session_id='C'")
        rowid = conn.execute("SELECT rowid FROM completions WHERE session_id='C'").fetchone()[0]
    dashboard = client.get('/admin?origin=unknown').text
    assert stat(dashboard,'reports_stale') == 1 and stat(dashboard,'reports_progress_unknown') == 1
    for body in (client.get('/admin/sessions/view?session_id=C&origin=unknown').text,client.get('/admin?table=completions&origin=unknown').text):
        assert '장기 대기' in body and '대기 경과' in body and '최근 처리' in body
        assert '미기록 · 진행 여부 확인 불가' in body
        assert 'engine_unavailable' not in body
    detail = client.get(f'/admin/api/record?table=completions&rowid={rowid}').json()
    meta = {item['label']:item['value'] for item in detail['meta']}
    assert meta['최근 처리 시각'] == '미기록'
    assert '장기 대기' in meta['처리 상태']


def test_legacy_read_only_schema_can_filter_unclassified_without_running_migration(monkeypatch,tmp_path):
    path = tmp_path/'legacy.db'
    with sqlite3.connect(path) as conn:
        conn.executescript("CREATE TABLE sessions(session_id TEXT,session_status TEXT); INSERT INTO sessions VALUES('ORDINARY','ACTIVE'),('TEST','COMPLETED');")
    monkeypatch.setenv('SEQUENCE_DB_PATH',str(path))
    app = FastAPI()
    app.include_router(create_db_viewer_router())
    client = TestClient(app)
    assert stat(client.get('/admin?origin=real').text,'sessions') == 0
    assert stat(client.get('/admin?origin=unknown').text,'sessions') == 2
    assert stat(client.get('/admin/sessions?origin=test').text,'sessions') == 0
    with sqlite3.connect(path) as conn:
        assert [row[1] for row in conn.execute('PRAGMA table_info(sessions)')] == ['session_id','session_status']
