"""Actual HTTP calls remain distinct and ASGI traffic passes through unchanged."""

import asyncio
import json
import sqlite3
import threading

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.api_call_log import (
    ApiCallLoggingMiddleware, ApiCallStore, REQUEST_CAPTURE_LIMIT, RESPONSE_CAPTURE_LIMIT,
)


class MemoryStore:
    def __init__(self):
        self.events = []

    def record(self, event):
        self.events.append(event)
        return len(self.events)


def run_asgi(app, *, path="/api/v1/copilot/query", chunks=(), query=b"", headers=(), method="POST"):
    incoming = [
        {"type": "http.request", "body": chunk, "more_body": index < len(chunks) - 1}
        for index, chunk in enumerate(chunks)
    ] or [{"type": "http.request", "body": b"", "more_body": False}]
    original = list(incoming)
    outgoing = []

    async def receive():
        return incoming.pop(0) if incoming else {"type": "http.disconnect"}

    async def send(message):
        outgoing.append(message)

    asyncio.run(app({"type": "http", "method": method, "path": path,
                     "query_string": query, "headers": list(headers)}, receive, send))
    return original, outgoing


def json_app(response, *, status=200, observed=None):
    async def app(scope, receive, send):
        while True:
            message = await receive()
            if observed is not None:
                observed.append(message)
            if not message.get("more_body"):
                break
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": json.dumps(response).encode()})
    return app


def test_store_persists_exact_call_schema_and_distinct_retries(tmp_path):
    path = tmp_path / "calls.sqlite3"
    store = ApiCallStore(str(path))
    event = {"session_id": "S", "method": "POST", "path": "/api/v1/sessions/start",
             "category": "start", "source": "http", "received_at": "2026-11-03T00:00:00Z",
             "status_code": 200, "request_json": {"session_id": "S", "target": 123.5},
             "response_json": {"data": {"session_id": "S"}}}
    assert store.record(event) == 1
    assert store.record(event) == 2
    store.close()
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM api_calls ORDER BY id").fetchall()
    assert len(rows) == 2
    assert json.loads(rows[0]["request_json"]) == event["request_json"]
    assert rows[0]["session_id"] == "S"
    assert {"finished_at", "duration_ms", "request_bytes", "response_bytes",
            "request_truncated", "response_truncated", "content_type"} <= set(rows[0].keys())


def test_store_keeps_json_scalar_payloads_valid(tmp_path):
    path = tmp_path / "scalars.sqlite3"
    store = ApiCallStore(str(path))
    assert store.record({"method": "POST", "path": "/api/v1/other",
                         "request_json": "hello", "response_json": False}) == 1
    store.close()
    with sqlite3.connect(path) as conn:
        row = conn.execute("SELECT request_json, response_json FROM api_calls").fetchone()
    assert json.loads(row[0]) == "hello"
    assert json.loads(row[1]) is False


def test_store_failure_is_safe_and_does_not_fallback_to_fake_persistence(tmp_path):
    store = ApiCallStore(str(tmp_path / "missing" / "calls.sqlite3"))
    assert store.record({"method": "POST", "path": "/api/v1/test"}) is None
    store.close()


def test_json_body_and_response_are_unchanged_with_all_identifiers():
    body = {"session_id": "S", "user_id": "U", "scenario_id": "3", "sequence_id": "SEQ-1",
            "request_id": "REQ-1", "frames": [{"x": 1.0, "trigger": False}]}
    encoded = json.dumps(body).encode()
    response = {"data": {"accepted": True}, "message": "정상"}
    seen = []
    store = MemoryStore()
    original, sent = run_asgi(ApiCallLoggingMiddleware(json_app(response, observed=seen), store=store),
                              path="/api/v1/copilot/analyze-sequence",
                              chunks=[encoded[:13], encoded[13:]],
                              headers=[(b"content-type", b"application/json"),
                                       (b"authorization", b"Bearer supersecret")])
    assert seen == original
    assert sent[-1]["body"] == json.dumps(response).encode()
    event = store.events[0]
    assert event["request_json"] == body
    assert event["response_json"] == response
    for key in ("session_id", "user_id", "scenario_id", "sequence_id", "request_id"):
        assert event[key] == body[key]
    assert event["category"] == "sequence"
    assert event["source"] == "http"
    assert event["request_bytes"] == len(encoded)
    assert event["duration_ms"] >= 0
    assert "supersecret" not in json.dumps(event)


def test_real_validation_failures_and_retries_are_individual_calls():
    app = FastAPI()

    @app.post("/api/v1/sessions/start")
    async def start(body: dict):
        return {"data": body}

    store = MemoryStore()
    app.add_middleware(ApiCallLoggingMiddleware, store=store)
    client = TestClient(app)
    for _ in range(2):
        assert client.post("/api/v1/sessions/start", json={"session_id": "S"}).status_code == 200
    invalid = client.post("/api/v1/sessions/start", content=b"broken", headers={"content-type": "application/json"})
    assert invalid.status_code == 422
    assert [event["status_code"] for event in store.events] == [200, 200, 422]
    assert store.events[-1]["response_json"] == invalid.json()
    assert store.events[-1]["request_json"]["parse_error"] == "invalid_json"


def test_streaming_chunks_forward_immediately_and_in_order():
    store = MemoryStore()
    expected = [
        {"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]},
        {"type": "http.response.body", "body": b'{"data":', "more_body": True},
        {"type": "http.response.body", "body": b'{"session_id":"S"}}', "more_body": False},
    ]
    sent = []

    async def app(scope, receive, send):
        for message in expected:
            await send(message)
            assert sent[-1] is message
            if message.get("more_body") or message["type"] == "http.response.start":
                assert not store.events

    async def outer_send(message):
        sent.append(message)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    asyncio.run(ApiCallLoggingMiddleware(app, store=store)(
        {"type": "http", "method": "GET", "path": "/api/v1/reports/R", "headers": [], "query_string": b""},
        receive, outer_send))
    assert sent == expected
    assert store.events[0]["session_id"] == "S"
    assert store.events[0]["category"] == "report"


def test_finished_response_is_recorded_before_background_work():
    store = MemoryStore()

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": b'{"data":{"session_id":"S"}}'})
        assert len(store.events) == 1
        # A failure in post-response background analysis cannot turn the delivered
        # 200 response into a second API call or an invented HTTP 500 response.
        raise RuntimeError("background analysis failed")

    with pytest.raises(RuntimeError, match="background analysis failed"):
        run_asgi(ApiCallLoggingMiddleware(app, store=store))
    assert len(store.events) == 1
    assert store.events[0]["status_code"] == 200


@pytest.mark.parametrize("path", ["/api/v1/stream/anomalies/S", "/assets/app.js", "/api/v10/other"])
def test_sse_and_non_api_paths_are_not_logged(path):
    store = MemoryStore()
    run_asgi(ApiCallLoggingMiddleware(json_app({"ok": True}), store=store), path=path)
    assert store.events == []


def test_unexpected_sse_content_type_is_not_logged_or_buffered():
    store = MemoryStore()

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"text/event-stream; charset=utf-8")]})
        await send({"type": "http.response.body", "body": b"data: hello\n\n"})

    _, sent = run_asgi(ApiCallLoggingMiddleware(app, store=store))
    assert sent[-1]["body"] == b"data: hello\n\n"
    assert store.events == []


def test_multipart_tracks_fields_after_large_binary_without_recording_binary():
    boundary = b"spray-xr-boundary"
    binary = b"CSV_PRIVATE_CONTENT" * (REQUEST_CAPTURE_LIMIT // 19 + 10)
    body = (b"--" + boundary + b'\r\nContent-Disposition: form-data; name="telemetry_csv"; filename="frames.csv"\r\n'
            b"Content-Type: text/csv\r\n\r\n" + binary + b"\r\n--" + boundary +
            b'\r\nContent-Disposition: form-data; name="session_id"\r\n\r\nS-UPLOAD\r\n--' + boundary +
            b'\r\nContent-Disposition: form-data; name="user_id"\r\n\r\nU\r\n--' + boundary + b"--\r\n")
    store = MemoryStore()
    chunks = [body[i:i + 1021] for i in range(0, len(body), 1021)]
    run_asgi(ApiCallLoggingMiddleware(json_app({"data": {"report_id": "R"}}), store=store),
             path="/api/v1/sessions/complete", chunks=chunks,
             headers=[(b"content-type", b"multipart/form-data; boundary=" + boundary)])
    event = store.events[0]
    assert event["session_id"] == "S-UPLOAD"
    assert event["user_id"] == "U"
    assert event["category"] == "complete"
    assert event["request_bytes"] == len(body)
    captured = event["request_json"]
    assert captured["fields"]["session_id"] == "S-UPLOAD"
    assert captured["files"][0]["filename"] == "frames.csv"
    assert captured["files"][0]["size_bytes"] == len(binary)
    assert captured["files"][0]["complete"] is True
    assert "CSV_PRIVATE_CONTENT" not in json.dumps(captured)


def test_partial_multipart_never_claims_known_file_size():
    body = (b'--b\r\nContent-Disposition: form-data; name="audio"; filename="a.wav"\r\n'
            b'Content-Type: audio/wav\r\n\r\nPRIVATE_AUDIO_BYTES')
    store = MemoryStore()
    run_asgi(ApiCallLoggingMiddleware(json_app({"data": {"session_id": "S"}}), store=store),
             chunks=[body], headers=[(b"content-type", b"multipart/form-data; boundary=b")])
    capture = store.events[0]["request_json"]
    assert capture["files"][0]["size_bytes"] is None
    assert capture["files"][0]["complete"] is False
    assert store.events[0]["request_truncated"]
    assert "PRIVATE_AUDIO_BYTES" not in json.dumps(capture)


def test_repeated_empty_multipart_parts_are_bounded_without_changing_traffic():
    part = b'--b\r\nContent-Disposition: form-data; name="x"\r\n\r\n\r\n'
    body = part * 1500 + b"--b--\r\n"
    store = MemoryStore()
    seen = []
    original, _ = run_asgi(
        ApiCallLoggingMiddleware(json_app({"data": {"session_id": "S"}}, observed=seen), store=store),
        path="/api/v1/sessions/complete", chunks=[body[i:i + 31] for i in range(0, len(body), 31)],
        headers=[(b"content-type", b"multipart/form-data; boundary=b")],
    )
    assert seen == original
    event = store.events[0]
    assert event["request_bytes"] == len(body)
    assert event["request_truncated"]
    assert len(event["request_json"]["fields"]["x"]) <= 1024


def test_json_size_limits_keep_byte_counts_without_leaking_binary():
    request = b'{"session_id":"S","blob":"' + b"x" * REQUEST_CAPTURE_LIMIT + b'"}'
    response = {"blob": "y" * RESPONSE_CAPTURE_LIMIT}
    store = MemoryStore()
    _, sent = run_asgi(ApiCallLoggingMiddleware(json_app(response), store=store), chunks=[request],
                       headers=[(b"content-type", b"application/json")])
    event = store.events[0]
    assert event["request_bytes"] == len(request)
    assert event["response_bytes"] == len(sent[-1]["body"])
    assert event["request_truncated"] and event["response_truncated"]
    assert event["session_id"] == "S"
    assert len(json.dumps(event["request_json"])) < REQUEST_CAPTURE_LIMIT + 1024
    assert len(json.dumps(event["response_json"])) < RESPONSE_CAPTURE_LIMIT + 1024


def test_many_query_parameters_still_produce_a_record_with_a_capture_limit():
    store = MemoryStore()
    query = b"session_id=S&" + b"&".join(f"key{i}=value".encode() for i in range(1100))
    run_asgi(ApiCallLoggingMiddleware(json_app({}), store=store),
             path="/api/v1/test/sequence/normal", query=query)
    assert len(store.events) == 1
    assert store.events[0]["session_id"] == "S"
    assert store.events[0]["request_truncated"]


def test_query_fields_response_ids_and_known_report_path_are_attributed():
    store = MemoryStore()
    run_asgi(ApiCallLoggingMiddleware(json_app({"data": {"sequence_id": "TEST-normal"}}), store=store),
             path="/api/v1/test/sequence/normal", query=b"session_id=S-TEST&user_id=U")
    assert store.events[0]["session_id"] == "S-TEST"
    assert "user_id" not in store.events[0]
    assert store.events[0]["source"] == "test"
    assert store.events[0]["sequence_id"] == "TEST-normal"
    run_asgi(ApiCallLoggingMiddleware(json_app({"detail": "not found"}, status=404), store=store),
             path="/api/v1/sessions/S-PATH/reports/R")
    assert store.events[1]["session_id"] == "S-PATH"


@pytest.mark.parametrize("path,method,body,response,expected", [
    ("/api/v1/sessions/S-REAL/reports/R", "GET", None, {"data": {"session_id": "S-REAL"}}, "S-REAL"),
    ("/api/v1/sessions/S-REAL/reports/R", "GET", None, {"data": {"session_id": "S-RESPONSE"}}, "S-REAL"),
    ("/api/v1/reports/R", "GET", None, {"data": {"session_id": "S-REAL"}}, "S-REAL"),
    ("/api/v1/copilot/query", "POST", {"session_id": "S-BODY"}, {"data": {"session_id": "S-CANONICAL"}}, "S-CANONICAL"),
    ("/api/v1/copilot/query", "POST", {"session_id": "S-BODY"}, {"data": {}}, "S-BODY"),
    ("/api/v1/copilot/query", "POST", {}, {"data": {}}, None),
])
def test_ignored_query_cannot_override_canonical_path_response_or_body(path, method, body, response, expected):
    store = MemoryStore()
    raw_body = json.dumps(body).encode() if body is not None else b""
    run_asgi(ApiCallLoggingMiddleware(json_app(response), store=store), path=path, method=method,
             chunks=[raw_body], query=b"session_id=OTHER&user_id=IGNORED&request_id=IGNORED",
             headers=[(b"content-type", b"application/json")])
    event = store.events[0]
    assert event.get("session_id") == expected
    assert "user_id" not in event
    assert "request_id" not in event
    assert event["request_json"]["query_parameters"]["session_id"] == "OTHER"


def test_unhandled_app_failure_is_recorded_then_reraised():
    store = MemoryStore()

    async def app(scope, receive, send):
        await receive()
        raise RuntimeError("upstream failed")

    with pytest.raises(RuntimeError, match="upstream failed"):
        run_asgi(ApiCallLoggingMiddleware(app, store=store), chunks=[b'{"session_id":"S"}'],
                 headers=[(b"content-type", b"application/json")])
    assert store.events[0]["status_code"] == 500
    assert store.events[0]["session_id"] == "S"


def test_store_failure_never_changes_successful_response():
    class BrokenStore:
        def record(self, event):
            raise OSError("disk full")

    _, sent = run_asgi(ApiCallLoggingMiddleware(json_app({"data": {"ok": True}}), store=BrokenStore()))
    assert json.loads(sent[-1]["body"]) == {"data": {"ok": True}}


def test_slow_record_does_not_block_independent_stream_progress():
    async def scenario():
        entered = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        progressed = []

        class SlowStore(MemoryStore):
            progress_during_record = False

            def record(self, event):
                loop.call_soon_threadsafe(entered.set)
                self.progress_during_record = release.wait(timeout=1)
                return super().record(event)

        store = SlowStore()

        async def independent_stream():
            await entered.wait()
            progressed.append(b"data: independent stream is still running\n\n")
            release.set()

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            pass

        await asyncio.gather(
            ApiCallLoggingMiddleware(json_app({"ok": True}), store=store)(
                {"type": "http", "method": "POST", "path": "/api/v1/copilot/query",
                 "headers": [], "query_string": b""}, receive, send),
            independent_stream(),
        )
        assert progressed
        assert store.progress_during_record, "SQLite logging blocked the event loop"
        assert len(store.events) == 1

    asyncio.run(scenario())


def test_urlencoded_fields_identify_session_and_raw_binary_is_metadata_only():
    store = MemoryStore()
    run_asgi(ApiCallLoggingMiddleware(json_app({}), store=store),
             chunks=[b"session_id=S-FORM&user_id=U&text=hello+world"],
             headers=[(b"content-type", b"application/x-www-form-urlencoded")])
    assert store.events[0]["session_id"] == "S-FORM"
    run_asgi(ApiCallLoggingMiddleware(json_app({}), store=store), chunks=[b"PRIVATE_PNG_BYTES"],
             headers=[(b"content-type", b"image/png")])
    assert "PRIVATE_PNG_BYTES" not in json.dumps(store.events[-1]["request_json"])
