"""Caller metadata is evidence, independent of trainee identity and authorization."""
import asyncio
import json
import sqlite3

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.api_call_log import ApiCallLoggingMiddleware, ApiCallStore
from trainer.demo_api.db_viewer import create_db_viewer_router


def _capture(*, peer=("192.0.2.31", 1234), headers=(), path="/api/v1/copilot/query"):
    events = []
    class Sink:
        def record(self, event):
            events.append(event)
    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": b'{"data":{}}'})
    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}
    async def send(message):
        pass
    scope = {"type": "http", "path": path, "method": "POST", "headers": list(headers), "client": peer}
    asyncio.run(ApiCallLoggingMiddleware(app, store=Sink())(scope, receive, send))
    return events[0]


@pytest.mark.parametrize("peer,expected", [(("192.0.2.31", 44), "192.0.2.31"), (("2001:db8::1", 44), "2001:db8::1"), (None, None), (("testclient", 44), None)])
def test_client_ip_uses_asgi_peer_and_ignores_untrusted_forwarding(peer, expected):
    event = _capture(peer=peer, headers=[(b"x-forwarded-for", b"203.0.113.99"), (b"forwarded", b"for=203.0.113.98")])
    assert event["client_ip"] == expected


@pytest.mark.parametrize("headers,path,source,evidence", [
    ([(b"x-copilot-source", b"motion-viewer")], "/api/v1/copilot/query", "motion-viewer", "header"),
    ([(b"x-copilot-source", b"assistant-test")], "/api/v1/sessions/start", "assistant-test", "header"),
    ([], "/api/v1/test/workflow/1-start", "test-api", "path"),
    ([(b"referer", b"https://example.test/docs")], "/api/v1/sessions/start", "api-docs", "referer"),
    ([(b"x-copilot-source", b"TSB")], "/api/v1/sessions/start", "simulator-api", "fallback"),
])
def test_bounded_caller_channels_are_separate_from_worker_identity(headers, path, source, evidence):
    event = _capture(headers=headers, path=path)
    assert event["caller_source"] == source
    assert event["source_evidence"] == evidence
    assert "user_id" not in event


def test_origin_and_referer_omit_credentials_query_and_fragment():
    event = _capture(headers=[(b"origin", b"https://user:password@example.test:8120"),
                              (b"referer", b"https://user:password@example.test/docs?token=secret#secret")])
    assert event["caller_origin"] == "https://example.test:8120"
    assert event["caller_referer"] == "https://example.test/docs"
    assert "password" not in json.dumps(event) and "secret" not in json.dumps(event)


def test_existing_api_call_db_migration_is_nullable_idempotent_and_keeps_history(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE api_calls (id INTEGER PRIMARY KEY, session_id TEXT, user_id TEXT, scenario_id TEXT, request_id TEXT, sequence_id TEXT, method TEXT NOT NULL, path TEXT NOT NULL, category TEXT NOT NULL, source TEXT NOT NULL, received_at TEXT NOT NULL, finished_at TEXT, status_code INTEGER, duration_ms REAL, request_json TEXT, response_json TEXT, request_bytes INTEGER, response_bytes INTEGER, request_truncated INTEGER, response_truncated INTEGER, content_type TEXT)")
        conn.execute("INSERT INTO api_calls (id,method,path,category,source,received_at) VALUES (1,'POST','/old','other','http','2026-01-01')")
    for _ in range(2):
        store = ApiCallStore(str(path))
        store.close()
    store = ApiCallStore(str(path))
    assert store.record({"method": "POST", "path": "/new", "client_ip": "192.0.2.31", "caller_source": "assistant-test"}) == 2
    store.close()
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT client_ip,caller_source FROM api_calls ORDER BY id").fetchall() == [(None, None), ("192.0.2.31", "assistant-test")]


@pytest.fixture
def viewer(monkeypatch, tmp_path):
    path = tmp_path / "viewer.db"
    store = ApiCallStore(str(path))
    start = "2026-11-06T01:00:00.100000+00:00"
    finish = "2026-11-06T01:00:01.900000+00:00"
    common = {"method": "POST", "session_id": "S", "user_id": "U", "received_at": start,
              "finished_at": finish, "status_code": 200, "client_ip": "192.0.2.31", "caller_source": "motion-viewer", "source_evidence": "header"}
    store.record(common | {"category": "start", "path": "/api/v1/sessions/start"})
    store.record(common | {"category": "sequence", "sequence_id": "SEQ-1", "path": "/api/v1/telemetry/sequence"})
    store.record(common | {"category": "query", "path": "/api/v1/copilot/query", "request_json": {"prompt": "질문", "task_type": "knowledge_search", "user_id": "U"}, "response_json": {"data": {"answer": "답변"}}})
    store.record(common | {"category": "complete", "path": "/api/v1/sessions/complete", "response_json": {"data": {"report_id": "R1"}}})
    store.close()
    with sqlite3.connect(path) as conn:
        conn.executescript("""
        CREATE TABLE sessions (session_id TEXT,user_id TEXT,scenario_id TEXT,session_status TEXT,created_at TEXT,started_at TEXT,creation_path TEXT,creation_origin TEXT);
        INSERT INTO sessions VALUES ('S','U','3','ACTIVE','2026-11-06T01:00:01+00:00','2026-11-06T01:00:01+00:00','/api/v1/sessions/start','real');
        INSERT INTO sessions VALUES ('OLD','U','3','ACTIVE',NULL,'2026-01-01',NULL,NULL);
        CREATE TABLE users (user_id TEXT,name TEXT,created_at TEXT);
        INSERT INTO users VALUES ('U','훈련자','2026-11-06T01:00:01+00:00');
        CREATE TABLE sequences (session_key TEXT,sequence_id TEXT,received_at TEXT);
        INSERT INTO sequences VALUES ('S','SEQ-1','2026-11-06T01:00:01+00:00');
        INSERT INTO sequences VALUES ('OLD','SEQ-1','2026-01-01');
        CREATE TABLE query_turns (id INTEGER,session_key TEXT,user_id TEXT,prompt TEXT,answer TEXT,task_type TEXT,created_at TEXT);
        INSERT INTO query_turns VALUES (1,'S','U','질문','답변','knowledge_search','2026-11-06T01:00:01+00:00');
        INSERT INTO query_turns VALUES (2,'S','U','다른 질문','답변','knowledge_search','2026-11-06T01:00:01+00:00');
        CREATE TABLE completions (report_id TEXT,session_id TEXT,requested_at TEXT,report_status TEXT,payload TEXT);
        INSERT INTO completions VALUES ('R1','S','2026-11-06T01:00:01+00:00','complete','{}');
        INSERT INTO completions VALUES ('UNLINKED','S','2026-11-06T01:00:01+00:00','complete','{}');
        CREATE TABLE legacy_import_records (source_path TEXT);
        INSERT INTO legacy_import_records VALUES ('dataset.csv');
        """)
    monkeypatch.setenv("SEQUENCE_DB_PATH", str(path))
    app = FastAPI()
    app.include_router(create_db_viewer_router())
    return TestClient(app), path


@pytest.mark.parametrize("table,rowid", [("api_calls", 1), ("sessions", 1), ("sequences", 1), ("query_turns", 1), ("completions", 1), ("users", 1)])
def test_record_details_show_exact_linked_caller_ip_and_channel(viewer, table, rowid):
    client, _ = viewer
    detail = client.get("/admin/api/record", params={"table": table, "rowid": rowid}).json()
    meta = {item["label"]: item["value"] for item in detail["meta"]}
    assert meta["호출 IP"] == "192.0.2.31"
    assert meta["호출 채널"] == "3D VIEWER"


@pytest.mark.parametrize("table,rowid", [("sessions", 2), ("sequences", 2), ("query_turns", 2), ("completions", 2)])
def test_unlinked_rows_never_borrow_another_call_ip(viewer, table, rowid):
    client, _ = viewer
    detail = client.get("/admin/api/record", params={"table": table, "rowid": rowid}).json()
    meta = {item["label"]: item["value"] for item in detail["meta"]}
    assert meta["호출 IP"] == "미기록"
    assert meta["호출 채널"] == "미기록"


def test_file_import_has_no_http_caller(viewer):
    client, _ = viewer
    detail = client.get("/admin/api/record?table=legacy_import_records&rowid=1").json()
    meta = {item["label"]: item["value"] for item in detail["meta"]}
    assert meta["호출 IP"] == "해당 없음"


def test_read_only_surfaces_show_caller_and_session_creation_provenance(viewer):
    client, _ = viewer
    for url in ["/admin?table=api_calls", "/admin?table=sequences", "/admin/sessions", "/admin/sessions/view?session_id=S", "/admin"]:
        body = client.get(url).text
        assert "192.0.2.31" in body, url
        assert "3D VIEWER" in body, url
    detail = client.get("/admin/api/session-event?session_id=S&event_key=call:1").json()
    assert {item["label"]: item["value"] for item in detail["meta"]}["호출 IP"] == "192.0.2.31"


def test_retry_after_storage_does_not_relabel_creation_call(viewer):
    client, path = viewer
    store = ApiCallStore(str(path))
    store.record({"method": "POST", "session_id": "S", "sequence_id": "SEQ-1", "category": "sequence", "path": "/api/v1/telemetry/sequence", "status_code": 200, "client_ip": "203.0.113.5", "caller_source": "assistant-test", "received_at": "2026-11-06T02:00:00Z", "finished_at": "2026-11-06T02:00:01Z"})
    store.close()
    meta = client.get("/admin/api/record?table=sequences&rowid=1").json()["meta"]
    assert {item["label"]: item["value"] for item in meta}["호출 IP"] == "192.0.2.31"


def test_ambiguous_matching_calls_do_not_guess_creator(viewer):
    client, path = viewer
    with sqlite3.connect(path) as conn:
        conn.execute("INSERT INTO api_calls (method,path,source,category,session_id,sequence_id,status_code,received_at,finished_at,client_ip,caller_source) SELECT method,path,source,category,session_id,sequence_id,status_code,received_at,finished_at,'203.0.113.5','assistant-test' FROM api_calls WHERE id=2")
    detail = client.get("/admin/api/record?table=sequences&rowid=1").json()
    assert {item["label"]: item["value"] for item in detail["meta"]}["호출 IP"] == "알 수 없음"


@pytest.mark.parametrize("path,category", [("/api/v1/test/workflow/1-start", "start"),
                                           ("/api/v1/test/workflow/3-complete", "complete"),
                                           ("/api/v1/test/workflow/4-report", "report")])
def test_guided_workflow_call_categories_can_link_created_records(path, category):
    assert _capture(path=path)["category"] == category


def test_general_chat_record_matches_null_public_session_and_same_worker(viewer):
    client, path = viewer
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE api_calls SET session_id=NULL WHERE id=3")
        conn.execute("UPDATE query_turns SET session_key='CHAT:U' WHERE id=1")
    meta = client.get("/admin/api/record?table=query_turns&rowid=1").json()["meta"]
    assert {item["label"]: item["value"] for item in meta}["호출 IP"] == "192.0.2.31"


def test_historical_generic_http_and_invalid_peers_are_not_invented(viewer):
    client, path = viewer
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE api_calls SET client_ip='not-an-ip',caller_source=NULL WHERE id=1")
    meta = client.get("/admin/api/record?table=api_calls&rowid=1").json()["meta"]
    assert {item["label"]: item["value"] for item in meta}["호출 IP"] == "미기록"
    assert {item["label"]: item["value"] for item in meta}["호출 채널"] == "미기록"
