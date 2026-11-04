"""Administrator writes must be authenticated, reversible and scoped to a preview."""
import hashlib
import json
import sqlite3

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.db_viewer import create_db_viewer_router
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.v1_store import V1Store

BASE = "/admin/api/manage"
ORIGIN = {"Origin": "http://testserver"}


@pytest.fixture
def admin(monkeypatch, tmp_path):
    path = tmp_path / "records.db"
    state = V1Store(str(path))
    state.upsert_user(user_id="U1", name="기존 이름", skill_level="novice", dominant_hand="right", created_at="t")
    state.start_session(session_id="S1", user_id="U1", scenario_id="3", started_at="t")
    state.start_session(session_id="KEEP", user_id="U1", scenario_id="3", started_at="t")
    state.create_completion(session_id="S1", requested_at="t", payload={})
    state.close()
    SequenceStore(str(path)).close()
    with sqlite3.connect(path) as conn:
        conn.execute("INSERT INTO sequences VALUES('S1','SEQ1','sha',1,'t','{}')")
        conn.execute("INSERT INTO sequence_analysis_results VALUES('S1','SEQ1','key','{}')")
        conn.execute("INSERT INTO query_turns(session_key,user_id,prompt,answer,task_type,created_at) VALUES('S1','U1','q','a','knowledge_search','t')")
        conn.execute("CREATE TABLE api_calls(id INTEGER PRIMARY KEY,session_id TEXT,path TEXT)")
        conn.execute("INSERT INTO api_calls VALUES(1,'S1','/api/v1/sessions/start')")
        conn.execute("INSERT INTO api_calls VALUES(2,'KEEP','/api/v1/sessions/start')")
    # A provisioned hash file, never a plaintext password in the application DB.
    salt = bytes.fromhex("11" * 16)
    credentials = tmp_path / "db-admin.json"
    credentials.write_text(json.dumps({"username": "admin", "algorithm": "pbkdf2_sha256", "iterations": 600000,
        "salt": salt.hex(), "password_hash": hashlib.pbkdf2_hmac("sha256", b"test-password", salt, 600000).hex(),
        "signing_key": "22" * 32}), encoding="utf-8")
    monkeypatch.setenv("SEQUENCE_DB_PATH", str(path))
    monkeypatch.setenv("DB_ADMIN_CREDENTIALS_PATH", str(credentials))
    app = FastAPI()
    app.include_router(create_db_viewer_router())
    return TestClient(app), path, tmp_path


def login(client):
    response = client.post(BASE + "/login", json={"username": "admin", "password": "test-password"}, headers=ORIGIN)
    assert response.status_code == 200, response.text
    return {**ORIGIN, "X-DB-CSRF": response.json()["csrf_token"]}


def rowid(path, table, condition="1"):
    with sqlite3.connect(path) as conn:
        return conn.execute(f'SELECT rowid FROM "{table}" WHERE {condition}').fetchone()[0]


def test_public_viewer_remains_public_but_management_data_and_writes_require_login(admin):
    client, path, _ = admin
    assert client.get("/admin").status_code == 200
    assert 'href="/admin/manage"' in client.get("/admin").text
    assert client.get("/admin/manage").status_code == 200
    assert client.get(BASE + "/records?table=users").status_code == 401
    for suffix in ("update", "preview-delete", "delete"):
        assert client.post(BASE + "/" + suffix, json={"table": "sessions", "rowid": 1}, headers=ORIGIN).status_code == 401
    assert rowid(path, "sessions", "session_id='S1'")


def test_login_uses_cookie_and_logout_revokes_it(admin):
    client, _, _ = admin
    wrong = client.post(BASE + "/login", json={"username": "admin", "password": "wrong"}, headers=ORIGIN)
    assert wrong.status_code == 401 and "wrong" not in wrong.text
    headers = login(client)
    assert "HttpOnly" in client.post(BASE + "/login", json={"username": "admin", "password": "test-password"}, headers=ORIGIN).headers["set-cookie"]
    headers["X-DB-CSRF"] = client.get(BASE + "/status").json()["csrf_token"]
    token = client.cookies.get("copilot_db_admin")
    assert client.get(BASE + "/status").json()["authenticated"] is True
    assert client.post(BASE + "/logout", headers=headers).status_code == 200
    client.cookies.set("copilot_db_admin", token)
    assert client.get(BASE + "/records?table=users").status_code == 401


def test_writes_require_csrf_and_same_origin_even_with_a_login_cookie(admin):
    client, path, _ = admin
    headers = login(client)
    body = {"table": "sessions", "rowid": rowid(path, "sessions", "session_id='S1'")}
    assert client.post(BASE + "/preview-delete", json=body, headers=ORIGIN).status_code == 403
    assert client.post(BASE + "/preview-delete", json=body, headers={**headers, "Origin": "http://elsewhere.test"}).status_code == 403
    assert client.get(BASE + "/status", headers={"Origin": "http://elsewhere.test"}).status_code == 403
    assert client.post(BASE + "/login", json={"username": "admin", "password": "test-password"}, headers={"Origin": "http://elsewhere.test"}).status_code == 403


def test_update_validates_fields_rejects_stale_version_and_creates_backup_and_audit(admin):
    client, path, directory = admin
    headers = login(client)
    target = {"table": "users", "rowid": rowid(path, "users")}
    detail = client.get(BASE + "/record", params=target).json()
    payload = {**target, "version": detail["version"], "fields": {"name": "수정된 이름", "age": 35}}
    assert client.post(BASE + "/update", json=payload, headers=headers).status_code == 200
    assert client.post(BASE + "/update", json=payload, headers=headers).status_code == 409
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT name,age,user_id FROM users").fetchone() == ("수정된 이름", 35, "U1")
        audit = conn.execute("SELECT action,actor,backup_name FROM admin_change_log").fetchone()
    assert audit[:2] == ("update", "admin")
    backup = directory / "admin_backups" / audit[2]
    with sqlite3.connect(backup) as conn:
        assert conn.execute("SELECT name FROM users").fetchone()[0] == "기존 이름"
    current = client.get(BASE + "/record", params=target).json()["version"]
    for fields in ({"user_id": "CHANGED"}, {"age": -1}, {"skill_level": "other"}):
        assert client.post(BASE + "/update", json={**target, "version": current, "fields": fields}, headers=headers).status_code == 422


def test_session_delete_uses_preview_and_only_deletes_related_records(admin):
    client, path, _ = admin
    headers = login(client)
    target = {"table": "sessions", "rowid": rowid(path, "sessions", "session_id='S1'")}
    preview = client.post(BASE + "/preview-delete", json=target, headers=headers).json()
    assert preview["counts"] == {"sessions": 1, "sequences": 1, "sequence_analysis_results": 1, "query_turns": 1, "completions": 1, "api_calls": 1}
    payload = {**target, "preview_token": preview["preview_token"], "confirmation": "wrong"}
    assert client.post(BASE + "/delete", json=payload, headers=headers).status_code == 422
    payload["confirmation"] = "S1"
    assert client.post(BASE + "/delete", json=payload, headers=headers).status_code == 200
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT session_id FROM sessions").fetchall() == [("KEEP",)]
        assert conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 1
        for table in ("sequences", "sequence_analysis_results", "query_turns", "completions"):
            assert conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0] == 0
        assert conn.execute("SELECT session_id FROM api_calls").fetchall() == [("KEEP",)]
        assert conn.execute("SELECT action FROM admin_change_log").fetchone()[0] == "delete"


def test_delete_preview_expires_if_new_related_records_arrive_and_linked_users_are_protected(admin):
    client, path, _ = admin
    headers = login(client)
    user = {"table": "users", "rowid": rowid(path, "users")}
    assert client.post(BASE + "/preview-delete", json=user, headers=headers).status_code == 409
    target = {"table": "sessions", "rowid": rowid(path, "sessions", "session_id='S1'")}
    preview = client.post(BASE + "/preview-delete", json=target, headers=headers).json()
    with sqlite3.connect(path) as conn:
        conn.execute("INSERT INTO sequences VALUES('S1','SEQ2','new',1,'t','{}')")
    assert client.post(BASE + "/delete", json={**target, "preview_token": preview["preview_token"], "confirmation": "S1"}, headers=headers).status_code == 409
    assert rowid(path, "sessions", "session_id='S1'")


def test_manager_limits_tables_fields_and_escapes_user_content(admin):
    client, path, _ = admin
    headers = login(client)
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE users SET name='<img src=x onerror=alert(1)>'")
    listed = client.get(BASE + "/records?table=users").json()
    assert listed["total"] == 1 and listed["rows"][0]["values"]["name"] == '<img src=x onerror=alert(1)>'
    for table in ("sqlite_master", "admin_change_log", 'users; DROP TABLE users'):
        assert client.get(BASE + "/records", params={"table": table}).status_code == 422
    assert client.post(BASE + "/update", json={"table": "sequences", "rowid": 1, "version": "bad", "fields": {"body": "{}"}}, headers=headers).status_code == 422


def test_login_failure_throttle_and_missing_configuration_fail_closed(admin, monkeypatch):
    client, _, directory = admin
    for _ in range(5):
        assert client.post(BASE + "/login", json={"username": "admin", "password": "wrong"}, headers=ORIGIN).status_code == 401
    assert client.post(BASE + "/login", json={"username": "admin", "password": "test-password"}, headers=ORIGIN).status_code == 429
    monkeypatch.setenv("DB_ADMIN_CREDENTIALS_PATH", str(directory / "missing.json"))
    app = FastAPI()
    app.include_router(create_db_viewer_router())
    assert TestClient(app).post(BASE + "/login", json={"username": "admin", "password": "test-password"}, headers=ORIGIN).status_code == 503


def test_expired_login_cannot_read_or_write(admin, monkeypatch):
    import time
    client, _, _ = admin
    login(client)
    later = time.time() + 3 * 60 * 60
    monkeypatch.setattr("trainer.demo_api.db_admin_auth.time.time", lambda: later)
    assert client.get(BASE + "/status").json()["authenticated"] is False
    assert client.get(BASE + "/records?table=users").status_code == 401


def test_backup_failure_leaves_original_record_unchanged(admin):
    client, path, directory = admin
    headers = login(client)
    (directory / "admin_backups").write_text("not a directory")
    target = {"table": "users", "rowid": rowid(path, "users")}
    current = client.get(BASE + "/record", params=target).json()
    result = client.post(BASE + "/update", json={**target, "version": current["version"], "fields": {"name": "변경되면 안 됨"}}, headers=headers)
    assert result.status_code == 503
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT name FROM users").fetchone()[0] == "기존 이름"


def test_delete_rejects_forged_preview_and_running_report(admin):
    from trainer.demo_api import report_jobs
    client, path, _ = admin
    headers = login(client)
    target = {"table": "sessions", "rowid": rowid(path, "sessions", "session_id='S1'")}
    preview = client.post(BASE + "/preview-delete", json=target, headers=headers).json()
    token = preview["preview_token"]
    forged = token[:-1] + ("0" if token[-1] != "0" else "1")
    payload = {**target, "confirmation": "S1", "preview_token": forged}
    assert client.post(BASE + "/delete", json=payload, headers=headers).status_code == 409
    with sqlite3.connect(path) as conn:
        report_id = conn.execute("SELECT report_id FROM completions").fetchone()[0]
    key = str(path.parent / (path.name + ".reports") / report_id)
    with report_jobs._GUARD:
        report_jobs._ACTIVE.add(key)
    try:
        payload["preview_token"] = token
        assert client.post(BASE + "/delete", json=payload, headers=headers).status_code == 409
        assert rowid(path, "sessions", "session_id='S1'")
    finally:
        with report_jobs._GUARD:
            report_jobs._ACTIVE.discard(key)


def test_delete_preview_detects_a_report_finishing_without_new_rows(admin):
    client, path, _ = admin
    headers = login(client)
    target = {"table": "sessions", "rowid": rowid(path, "sessions", "session_id='S1'")}
    preview = client.post(BASE + "/preview-delete", json=target, headers=headers).json()
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE completions SET report_status='complete' WHERE session_id='S1'")
    result = client.post(BASE + "/delete", json={**target, "preview_token": preview["preview_token"], "confirmation": "S1"}, headers=headers)
    assert result.status_code == 409
    assert rowid(path, "sessions", "session_id='S1'")


def test_deletion_is_committed_before_report_guard_is_released(admin, monkeypatch):
    from contextlib import contextmanager
    from trainer.demo_api import db_admin_store
    client, path, _ = admin
    headers = login(client)
    target = {"table": "sessions", "rowid": rowid(path, "sessions", "session_id='S1'")}
    preview = client.post(BASE + "/preview-delete", json=target, headers=headers).json()
    real_guard = db_admin_store.hold_idle_reports

    @contextmanager
    def observed_guard(directory, ids):
        with real_guard(directory, ids):
            yield
            with sqlite3.connect(path) as observer:
                assert observer.execute("SELECT COUNT(*) FROM sessions WHERE session_id='S1'").fetchone()[0] == 0

    monkeypatch.setattr(db_admin_store, "hold_idle_reports", observed_guard)
    result = client.post(BASE + "/delete", json={**target, "preview_token": preview["preview_token"], "confirmation": "S1"}, headers=headers)
    assert result.status_code == 200


def test_relative_database_path_also_blocks_an_active_report(admin, monkeypatch):
    from trainer.demo_api import report_jobs
    _, path, directory = admin
    monkeypatch.chdir(directory)
    monkeypatch.setenv("SEQUENCE_DB_PATH", path.name)
    app = FastAPI()
    app.include_router(create_db_viewer_router())
    client = TestClient(app)
    headers = login(client)
    target = {"table": "sessions", "rowid": rowid(path, "sessions", "session_id='S1'")}
    preview = client.post(BASE + "/preview-delete", json=target, headers=headers).json()
    with sqlite3.connect(path) as conn:
        report_id = conn.execute("SELECT report_id FROM completions").fetchone()[0]
    key = str(path.parent / (path.name + ".reports") / report_id)
    with report_jobs._GUARD:
        report_jobs._ACTIVE.add(key)
    try:
        assert client.post(BASE + "/delete", json={**target, "preview_token": preview["preview_token"], "confirmation": "S1"}, headers=headers).status_code == 409
    finally:
        with report_jobs._GUARD:
            report_jobs._ACTIVE.discard(key)


def test_management_audit_stays_private_when_the_viewer_is_public(admin):
    client, path, _ = admin
    headers = login(client)
    target = {"table": "users", "rowid": rowid(path, "users")}
    version = client.get(BASE + "/record", params=target).json()["version"]
    assert client.post(BASE + "/update", json={**target, "version": version, "fields": {"name": "New"}}, headers=headers).status_code == 200
    assert client.get(BASE + "/history").json()["rows"][0]["actor"] == "admin"
    client.cookies.clear()
    assert client.get("/admin/api/record?table=admin_change_log&rowid=1").status_code == 404
    assert "admin_change_log" not in client.get("/admin").text
    assert "backup_name" not in client.get("/admin?table=admin_change_log").text
