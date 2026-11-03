import html
import re
import sqlite3
from urllib.parse import parse_qs, urlsplit

import pytest

from trainer.demo_api.db_sessions import (
    SESSION_CSS,
    render_session_detail,
    render_session_list,
    session_event_detail,
)


@pytest.fixture
def conn():
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.executescript("""
        CREATE TABLE users (user_id TEXT PRIMARY KEY, name TEXT);
        INSERT INTO users VALUES ('U1','홍길동'),('U2','<img src=x onerror=alert(1)>');
        CREATE TABLE sessions (session_id TEXT PRIMARY KEY, user_id TEXT, scenario_id TEXT,
            session_status TEXT, started_at TEXT, ended_at TEXT, created_at TEXT,
            target_thickness_um REAL);
        INSERT INTO sessions VALUES
            ('S1','U1','3','ACTIVE','2026-10-29T15:10:00Z',NULL,NULL,120),
            ('S2','U1','7','ACTIVE','2026-10-29T04:00:00Z',NULL,'2026-10-29T03:59:00Z',150),
            ('S3','U2','9','COMPLETED','2026-10-28T18:00:00Z','2026-10-29T05:00:00Z',NULL,100),
            ('S4','U1','3','ACTIVE',NULL,'2026-10-29T06:00:00Z',NULL,100);
        CREATE TABLE sequences (session_key TEXT, sequence_id TEXT, received_at TEXT,
            frames INTEGER, body_sha256 TEXT, body TEXT, PRIMARY KEY(session_key,sequence_id));
        INSERT INTO sequences VALUES
            ('S2','SEQ-1','2026-10-30T00:00:00Z',24,'hash','{"secret":"large-body"}'),
            ('S2','SEQ-2','2026-10-30T00:01:00Z',12,'hash2','{"frames":[]}'),
            ('S3','SEQ-1','2026-10-29T04:30:00Z',10,'hash3','{}');
        CREATE TABLE query_turns (id INTEGER PRIMARY KEY,session_key TEXT,user_id TEXT,
            prompt TEXT,answer TEXT,task_type TEXT,created_at TEXT);
        INSERT INTO query_turns VALUES
            (1,'S2','U1','도장 자세 질문','저장된 답변','pose_recomm','2026-10-30T00:02:00Z'),
            (2,'S3','U2','비공개 질문','다른 세션 답변','pose_recomm','2026-10-29T04:40:00Z');
        CREATE TABLE completions (report_id TEXT PRIMARY KEY,session_id TEXT,
            report_status TEXT,requested_at TEXT,payload TEXT);
        INSERT INTO completions VALUES ('R1','S3','complete','2026-10-29T05:10:00Z','{"score":88}');
        CREATE TABLE api_calls (id INTEGER PRIMARY KEY,session_id TEXT,user_id TEXT,
            scenario_id TEXT,request_id TEXT,sequence_id TEXT,method TEXT,path TEXT,
            category TEXT,source TEXT,received_at TEXT,finished_at TEXT,status_code INTEGER,
            duration_ms REAL,request_json TEXT,response_json TEXT,request_bytes INTEGER,
            response_bytes INTEGER,request_truncated INTEGER,response_truncated INTEGER,
            content_type TEXT);
    """)
    yield db
    db.close()


def add_call(conn, number, session="S2", category="sequence", sequence="SEQ-1", **values):
    row = dict(id=number, session_id=session, user_id="U1", scenario_id="7",
               request_id=f"REQ-{number}", sequence_id=sequence, method="POST",
               path="/api/v1/telemetry/sequence", category=category, source="http",
               received_at="2026-10-30T00:00:00Z", finished_at="2026-10-30T00:00:00.020Z",
               status_code=200, duration_ms=20, request_json='{"sequence_id":"SEQ-1"}',
               response_json='{"success":true}', request_bytes=23,response_bytes=16,
               request_truncated=0,response_truncated=0,content_type="application/json")
    row.update(values)
    conn.execute(f"INSERT INTO api_calls ({','.join(row)}) VALUES ({','.join('?' for _ in row)})", tuple(row.values()))


def event_keys(body):
    urls = re.findall(r'data-detail-url="([^"]+)"', body)
    return [parse_qs(urlsplit(html.unescape(url)).query)["event_key"][0] for url in urls]


def test_list_totals_gray_unfinished_rows_and_ended_timestamp(conn):
    body = render_session_list(conn)
    for key, value in (("sessions",4),("unfinished",2),("ended",2)):
        assert re.search(rf'data-stat="{key}">{value}<', body)
    assert len(re.findall(r'class="session-item unfinished"', body)) == 2
    assert len(re.findall(r'class="session-item ended"', body)) == 2
    assert 'class="session-ended-at"' in body
    assert "2026.10.29 14:00:00 KST" in body
    assert ".session-item.unfinished" in SESSION_CSS
    assert "홍길동" in body and "학습모드 7" in body


def test_search_state_and_pagination_keep_params(conn):
    body = render_session_list(conn, q="홍길동", state="unfinished", page=2, limit=1)
    assert "2 / 2 페이지" in body
    assert len(re.findall(r'class="session-item ', body)) == 1
    assert "S1" in body and "S2</" not in body
    assert "state=unfinished" in body and "limit=1" in body
    assert 'class="session-state">훈련 종료</span>' not in body
    assert "조건에 맞는" in render_session_list(conn, q="missing")
    assert "전체 2건" in render_session_list(conn, state="ended")


def test_list_escape_and_literal_search_wildcards(conn):
    body = render_session_list(conn, q='<script>"')
    assert '<script>' not in body
    assert '&lt;script&gt;&quot;' in body
    assert "조건에 맞는" in render_session_list(conn,q="%")
    body = render_session_list(conn)
    assert '<img src=x' not in body and '&lt;img src=x' in body


def test_timestamps_distinguish_server_creation_training_start_and_missing(conn):
    body = render_session_detail(conn,"S1")
    assert "생성 시각" in body and "미기록" in body
    assert "훈련 시작" in body and "2026.10.30 00:10:00 KST" in body
    new_body = render_session_detail(conn,"S2")
    assert "2026.10.29 12:59:00 KST" in new_body
    assert "2026.10.29 13:00:00 KST" in new_body


def test_actual_retry_counts_are_separate_from_unique_stored_sequences(conn):
    add_call(conn,1)
    add_call(conn,2,response_json='{"duplicate":true}')
    body = render_session_detail(conn,"S2")
    assert 'data-stat="sequences">2<' in body
    assert 'data-stat="queries">1<' in body
    assert 'data-stat="api_calls">2<' in body
    assert "수집 이후" in body and "실제 호출" in body and "기존 저장 기록" in body
    keys=event_keys(body)
    assert "call:1" in keys and "call:2" in keys
    assert "sequence:SEQ-1" not in keys
    assert "sequence:SEQ-2" in keys


def test_session_list_and_kpis_separate_category_attempts_from_saved_counts(conn):
    add_call(conn,1)
    add_call(conn,2,status_code=409)
    add_call(conn,3,category="query",sequence=None,status_code=500)
    add_call(conn,4,category="query",sequence=None)
    add_call(conn,5,category="other",sequence=None)
    body=render_session_list(conn,q="S2")
    assert "저장 시퀀스 2 · 질문 1" in body
    assert "시퀀스 호출 2 · 쿼리 호출 2 (수집 이후)" in body
    detail=render_session_detail(conn,"S2")
    assert 'data-stat="api_calls">5<' in detail
    assert "실제 시퀀스 호출 2건 (수집 이후)" in detail
    assert "실제 쿼리 호출 2건 (수집 이후)" in detail


def test_only_successful_captured_activity_advances_sessions_without_saved_history(conn):
    add_call(conn,1,session="S1",category="query",sequence=None,status_code=422)
    failed=render_session_list(conn,q="S1")
    assert 'class="session-state">시작 · 입력 대기</span>' in failed
    add_call(conn,2,session="S1",category="query",sequence=None,status_code=200)
    accepted=render_session_list(conn,q="S1")
    assert 'class="session-state">진행 중 · 미종료</span>' in accepted
    assert "저장 시퀀스 0 · 질문 0" in accepted
    assert "시퀀스 호출 0 · 쿼리 호출 2 (수집 이후)" in accepted
    conn.execute("DELETE FROM api_calls WHERE session_id='S1'")
    add_call(conn,3,session="S1",status_code=201)
    assert 'class="session-state">진행 중 · 미종료</span>' in render_session_list(conn,q="S1")


def test_call_summary_keeps_unknown_legacy_categories_and_statuses_conservative(conn):
    conn.execute("DROP TABLE api_calls")
    conn.execute("CREATE TABLE api_calls (id INTEGER,session_id TEXT,received_at TEXT)")
    conn.execute("INSERT INTO api_calls VALUES (1,'S1','2026-10-30T00:00:00Z')")
    body=render_session_list(conn,q="S1")
    assert 'class="session-state">시작 · 입력 대기</span>' in body
    assert "시퀀스 호출 0 · 쿼리 호출 0 (수집 이후)" in body
    assert 'data-stat="api_calls">1<' in render_session_detail(conn,"S1")


def test_prompt_only_matches_and_start_calls_without_matching_time_keep_legacy(conn):
    add_call(conn,1,category="query",sequence=None,request_json='{"prompt":"도장 자세 질문"}')
    add_call(conn,2,category="start",sequence=None)
    body=render_session_detail(conn,"S2")
    assert "query:1" in event_keys(body)
    assert "start" in event_keys(body)
    other=render_session_detail(conn,"S3")
    assert "query:2" in event_keys(other) and "end" in event_keys(other)
    assert "completion:R1" in event_keys(other)


def test_failed_captured_retries_do_not_hide_legacy_storage_records(conn):
    add_call(conn,1,status_code=409)
    add_call(conn,2,category="start",sequence=None,status_code=422,
             request_json='{"started_at":"2026-10-29T04:00:00Z"}')
    add_call(conn,3,session="S3",category="end",sequence=None,status_code=409,
             response_json='{"data":{"ended_at":"2026-10-29T05:00:00Z"}}')
    add_call(conn,4,session="S3",category="complete",sequence=None,status_code=422,
             response_json='{"data":{"report_id":"R1"}}')
    keys=event_keys(render_session_detail(conn,"S2"))
    assert "sequence:SEQ-1" in keys and "start" in keys
    other=event_keys(render_session_detail(conn,"S3"))
    assert "end" in other and "completion:R1" in other


def test_successful_start_end_suppress_only_when_recorded_time_matches(conn):
    add_call(conn,1,category="start",sequence=None,
             request_json='{"started_at":"2026-10-29T03:00:00Z"}')
    add_call(conn,2,session="S3",category="end",sequence=None,
             response_json='{"data":{"ended_at":"2026-10-29T06:00:00Z"}}')
    assert "start" in event_keys(render_session_detail(conn,"S2"))
    assert "end" in event_keys(render_session_detail(conn,"S3"))
    add_call(conn,3,category="start",sequence=None,
             response_json='{"data":{"started_at":"2026-10-29T04:00:00Z"}}')
    add_call(conn,4,session="S3",category="end",sequence=None,
             request_json='{"ended_at":"2026-10-29T05:00:00Z"}')
    assert "start" not in event_keys(render_session_detail(conn,"S2"))
    assert "end" not in event_keys(render_session_detail(conn,"S3"))


@pytest.mark.parametrize("response",['{"data":{"report_id":"R1"}}','{"report_id":"R1"}'])
def test_successful_report_envelopes_match_the_exact_report_identifier(conn,response):
    add_call(conn,1,session="S3",category="complete",sequence=None,response_json=response)
    assert "completion:R1" not in event_keys(render_session_detail(conn,"S3"))


def test_repeated_prompt_turns_are_preserved_even_with_successful_query_capture(conn):
    conn.execute("INSERT INTO query_turns VALUES (3,'S2','U1','도장 자세 질문','다른 답변','pose_recomm','2026-10-30T00:03:00Z')")
    add_call(conn,1,category="query",sequence=None,request_json='{"prompt":"도장 자세 질문"}')
    keys=event_keys(render_session_detail(conn,"S2",kind="query"))
    assert "call:1" in keys and "query:1" in keys and "query:3" in keys


def test_legacy_details_do_not_invent_http_status_or_response(conn):
    event=session_event_detail(conn,"S2","sequence:SEQ-1")
    assert event is not None and "기존 저장 기록" in event["notice"]
    assert "HTTP" in event["notice"]
    assert any(section["data"].get("secret")=="large-body" for section in event["sections"] if isinstance(section["data"],dict))
    assert not any(item["label"]=="HTTP 상태" for item in event["meta"])
    completion=session_event_detail(conn,"S3","completion:R1")
    assert completion is not None
    assert any(section["data"]=={"score":88} for section in completion["sections"])


def test_actual_detail_has_payload_metadata_and_truncation_notice(conn):
    add_call(conn,1,request_truncated=1,response_json='{"answer":"응답"}')
    event=session_event_detail(conn,"S2","call:1")
    assert event is not None
    assert any(item["label"]=="HTTP 상태" and str(item["value"])=="200" for item in event["meta"])
    assert any(section["data"]=={"answer":"응답"} for section in event["sections"])
    assert "잘림" in event["notice"]


def test_events_are_strictly_session_isolated_including_unknown_sessions(conn):
    add_call(conn,1,session="S2")
    for key in ("call:1","sequence:SEQ-2","query:1"):
        assert session_event_detail(conn,"S3",key) is None
    assert session_event_detail(conn,"S2","completion:R1") is None
    assert session_event_detail(conn,"missing","call:1") is None
    assert session_event_detail(conn,"S2","call:1 OR 1=1") is None
    assert render_session_detail(conn,"missing") is None


def test_event_urls_encode_session_and_event_identifiers(conn):
    conn.execute("UPDATE sessions SET session_id=? WHERE session_id='S1'",('S/<x>&"',))
    conn.execute("INSERT INTO sequences VALUES (?,?,?,?,?,?)",('S/<x>&"','SEQ/<x>&"','invalid',1,'h','{}'))
    body=render_session_detail(conn,'S/<x>&"')
    assert 'S/<x>' not in body and 'SEQ/<x>' not in body
    assert 'sequence:SEQ/<x>&"' in event_keys(body)
    assert "형식 확인 필요" in body


def test_calls_and_legacy_lists_are_bounded_with_pagination_and_filters(conn):
    for number in range(1,206):
        add_call(conn,number,category="query",sequence=None,request_json='{"prompt":"other"}')
    first=render_session_detail(conn,"S2",kind="query")
    assert len([key for key in event_keys(first) if key.startswith("call:")])==200
    assert "call_page=2" in first and "kind=query" in first
    second=render_session_detail(conn,"S2",kind="query",call_page=2)
    assert len([key for key in event_keys(second) if key.startswith("call:")])==5
    sequence=render_session_detail(conn,"S2",kind="sequence")
    assert not any(key.startswith("call:") for key in event_keys(sequence))
    assert "sequence:SEQ-1" in event_keys(sequence)


def test_summary_does_not_read_large_payload_columns(conn):
    blocked={"body","payload","prompt","answer","request_json","response_json"}
    def authorizer(action,table,column,_db,_trigger):
        if action==sqlite3.SQLITE_READ and column in blocked:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    conn.set_authorizer(authorizer)
    body=render_session_list(conn)
    assert "S2" in body


def test_no_actual_call_history_is_explicit_and_legacy_is_not_counted_as_http(conn):
    conn.execute("DROP TABLE api_calls")
    body=render_session_detail(conn,"S3")
    assert "기존 저장 기록" in body and "수집된 실제 호출이 없습니다" in body
    assert 'data-stat="api_calls">0<' in body
    assert "HTTP 응답" in body
    assert session_event_detail(conn,"S3","call:1") is None


def test_empty_and_minimal_legacy_schemas_are_readable():
    with sqlite3.connect(":memory:") as db:
        db.row_factory=sqlite3.Row
        assert "아직" in render_session_list(db)
        assert "1 / 1 페이지" in render_session_list(db,page=99)
        assert render_session_detail(db,"S1") is None
        db.execute("CREATE TABLE sessions (session_id TEXT,session_status TEXT)")
        db.execute("INSERT INTO sessions VALUES ('old','COMPLETED')")
        assert "old" in render_session_list(db)
        body=render_session_detail(db,"old")
        assert body is not None and "미기록" in body
        assert session_event_detail(db,"old","start") is None


def test_invalid_captured_json_does_not_hide_unmatched_legacy_query(conn):
    add_call(conn,1,category="query",sequence=None,request_json='{"prompt":',response_json='not json',request_truncated=1)
    body=render_session_detail(conn,"S2",kind="query")
    assert "call:1" in event_keys(body) and "query:1" in event_keys(body)
    event=session_event_detail(conn,"S2","call:1")
    assert any(section["data"]=='{"prompt":' for section in event["sections"])


def test_legacy_event_pagination_and_rowid_fallback_do_not_require_new_schema():
    with sqlite3.connect(":memory:") as db:
        db.row_factory=sqlite3.Row
        db.execute("CREATE TABLE sessions (session_id TEXT)")
        db.execute("INSERT INTO sessions VALUES ('old')")
        db.execute("CREATE TABLE query_turns (session_key TEXT,prompt TEXT,answer TEXT)")
        db.executemany("INSERT INTO query_turns VALUES ('old',?,'answer')",[(str(number),) for number in range(205)])
        first=render_session_detail(db,"old",kind="query")
        assert len(event_keys(first))==200 and "call_page=2" in first
        second=render_session_detail(db,"old",kind="query",call_page=2)
        assert len(event_keys(second))==5
        detail=session_event_detail(db,"old","query:1")
        assert detail is not None and detail["sections"][0]["data"]["prompt"]=="0"
