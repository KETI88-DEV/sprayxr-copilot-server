"""Durable records of actual finite public API calls, independent of domain events.

The ASGI wrappers tee messages without consuming ahead or changing their contents.
Multipart uploads retain fields and file metadata, never the uploaded file bodies.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import parse_qs, unquote

from anyio import CancelScope, to_thread

from trainer.demo_api.request_provenance import PROVENANCE_COLUMNS, capture_provenance

logger = logging.getLogger(__name__)

REQUEST_CAPTURE_LIMIT = 8 * 1024 * 1024
RESPONSE_CAPTURE_LIMIT = 2 * 1024 * 1024
_IDENTIFIERS = ("session_id", "user_id", "scenario_id", "request_id", "sequence_id")
_COLUMNS = (
    *_IDENTIFIERS, "method", "path", "category", "source", "received_at", "finished_at",
    "status_code", "duration_ms", "request_json", "response_json", "request_bytes",
    "response_bytes", "request_truncated", "response_truncated", "content_type",
    *PROVENANCE_COLUMNS,
)
_SCHEMA = """
CREATE TABLE IF NOT EXISTS api_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT, user_id TEXT, scenario_id TEXT, request_id TEXT, sequence_id TEXT,
    method TEXT NOT NULL, path TEXT NOT NULL, category TEXT NOT NULL, source TEXT NOT NULL,
    received_at TEXT NOT NULL, finished_at TEXT, status_code INTEGER, duration_ms REAL,
    request_json TEXT, response_json TEXT, request_bytes INTEGER, response_bytes INTEGER,
    request_truncated INTEGER, response_truncated INTEGER, content_type TEXT
)
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ApiCallStore:
    """Failure-safe SQLite sink. Unavailable storage never becomes fake persistence."""

    def __init__(self, db_path: str) -> None:
        self._guard = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(str(db_path), timeout=0.2, check_same_thread=False)
            self._conn = conn
            with conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute(_SCHEMA)
                # Nullable, idempotent migration: history has no recoverable IP.
                present = {row[1] for row in conn.execute("PRAGMA table_info(api_calls)")}
                for name in PROVENANCE_COLUMNS:
                    if name not in present:
                        conn.execute(f"ALTER TABLE api_calls ADD COLUMN {name} TEXT")
                conn.execute("CREATE INDEX IF NOT EXISTS api_calls_session_id ON api_calls(session_id, id)")
        except Exception:
            logger.warning("API call storage is unavailable", exc_info=True)
            self.close()

    def record(self, event: dict[str, Any]) -> int | None:
        try:
            values = dict(event)
            for key in ("request_json", "response_json"):
                value = values.get(key)
                if value is not None:
                    values[key] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            values.setdefault("category", "other")
            values.setdefault("source", "http")
            values.setdefault("received_at", _now())
            with self._guard:
                if self._conn is None:
                    return None
                with self._conn:
                    cursor = self._conn.execute(
                        f"INSERT INTO api_calls ({','.join(_COLUMNS)}) VALUES ({','.join('?' for _ in _COLUMNS)})",
                        tuple(values.get(key) for key in _COLUMNS),
                    )
                    return int(cursor.lastrowid)
        except Exception:
            logger.warning("API call could not be persisted", exc_info=True)
            return None

    def close(self) -> None:
        with self._guard:
            conn, self._conn = self._conn, None
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    logger.warning("API call storage close failed", exc_info=True)


def _header(headers: list, name: bytes) -> str:
    for key, value in headers:
        if key.lower() == name:
            return value.decode("latin-1")
    return ""


def _query(raw: bytes) -> dict[str, Any]:
    # Bound both parsing work and retained metadata. Keep early identity fields even
    # when an invalid request supplies too many query/form parameters.
    retained = b"&".join(raw[:REQUEST_CAPTURE_LIMIT].split(b"&", 1024)[:1024])
    values = parse_qs(retained.decode("utf-8", errors="replace"), keep_blank_values=True, max_num_fields=1024)
    return {key: items[0] if len(items) == 1 else items for key, items in values.items()}


def _partial_identifiers(raw: bytes) -> dict[str, Any]:
    """Recover only complete top-level scalar fields before a truncated JSON value."""
    text = raw.decode("utf-8", errors="replace")
    decoder = json.JSONDecoder()
    offset = len(text) - len(text.lstrip())
    found: dict[str, Any] = {}
    if text[offset:offset + 1] != "{":
        return found
    offset += 1
    try:
        for _ in range(64):
            if offset >= len(text):
                break
            offset += len(text[offset:]) - len(text[offset:].lstrip())
            key, offset = decoder.raw_decode(text, offset)
            offset += len(text[offset:]) - len(text[offset:].lstrip())
            if text[offset:offset + 1] != ":":
                break
            offset += 1
            offset += len(text[offset:]) - len(text[offset:].lstrip())
            value, offset = decoder.raw_decode(text, offset)
            if key in _IDENTIFIERS and isinstance(value, (str, int)) and not isinstance(value, bool):
                found[key] = value
            offset += len(text[offset:]) - len(text[offset:].lstrip())
            if text[offset:offset + 1] != ",":
                break
            offset += 1
    except (ValueError, TypeError, RecursionError):
        pass
    return found


class _MultipartCapture:
    """Streaming parser: file data is counted and immediately discarded."""

    def __init__(self, content_type: str, limit: int) -> None:
        try:
            from python_multipart.multipart import MultipartParser, parse_options_header
        except ImportError:
            from multipart.multipart import MultipartParser, parse_options_header
        self._parse_options = parse_options_header
        _, options = parse_options_header(content_type)
        self.fields: dict[str, Any] = {}
        self.files: list[dict[str, Any]] = []
        self.truncated = False
        self.complete = False
        self._limit = limit
        self._retained_bytes = 0
        self._part_count = 0
        self._part: dict[str, Any] | None = None
        self._header_name = bytearray()
        self._header_value = bytearray()
        self._headers: dict[bytes, bytes] = {}
        self._parser = MultipartParser(options[b"boundary"], {
            "on_part_begin": self._begin,
            "on_header_field": self._header_field,
            "on_header_value": self._header_content,
            "on_header_end": self._header_end,
            "on_headers_finished": self._headers_finished,
            "on_part_data": self._data,
            "on_part_end": self._end_part,
            "on_end": self._end,
        })

    def _begin(self) -> None:
        self._part_count += 1
        if self._part_count > 1024:
            self.truncated = True
            raise ValueError("multipart capture part limit exceeded")
        self._part = {"bytes": 0, "value": bytearray()}
        self._headers = {}

    def _header_field(self, data: bytes, start: int, end: int) -> None:
        retained = min(end - start, max(0, 4096 - len(self._header_name)))
        self._header_name.extend(data[start:start + retained])
        self.truncated |= retained < end - start

    def _header_content(self, data: bytes, start: int, end: int) -> None:
        retained = min(end - start, max(0, 8192 - len(self._header_value)))
        self._header_value.extend(data[start:start + retained])
        self.truncated |= retained < end - start

    def _header_end(self) -> None:
        name = bytes(self._header_name).lower()
        if name in {b"content-disposition", b"content-type"}:
            self._headers[name] = bytes(self._header_value)
        self._header_name.clear()
        self._header_value.clear()

    def _headers_finished(self) -> None:
        _, options = self._parse_options(self._headers.get(b"content-disposition", b""))
        part = self._part
        if part is None:
            return
        metadata_bytes = (len(options.get(b"name", b"")) + len(options.get(b"filename", b""))
                          + len(self._headers.get(b"content-type", b"")))
        if self._retained_bytes + metadata_bytes > self._limit:
            self.truncated = True
            raise ValueError("multipart capture metadata limit exceeded")
        self._retained_bytes += metadata_bytes
        part["name"] = options.get(b"name", b"").decode("utf-8", errors="replace")
        if b"filename" in options:
            part["file"] = {
                "field": part["name"],
                "filename": options[b"filename"].decode("utf-8", errors="replace"),
                "content_type": self._headers.get(b"content-type", b"application/octet-stream").decode("latin-1"),
                "bytes_received": 0, "size_bytes": None, "complete": False,
            }

    def _data(self, data: bytes, start: int, end: int) -> None:
        part = self._part
        if part is None:
            return
        part["bytes"] += end - start
        if "file" not in part:
            remaining = min(65536 - len(part["value"]), self._limit - self._retained_bytes)
            captured = max(0, min(remaining, end - start))
            part["value"].extend(data[start:start + captured])
            self._retained_bytes += captured
            if captured < end - start:
                self.truncated = True

    def _end_part(self) -> None:
        part, self._part = self._part, None
        if part is None:
            return
        if "file" in part:
            part["file"].update(bytes_received=part["bytes"], size_bytes=part["bytes"], complete=True)
            self.files.append(part["file"])
        elif part.get("name"):
            value = bytes(part["value"]).decode("utf-8", errors="replace")
            name = part["name"]
            if name in self.fields:
                old = self.fields[name]
                if isinstance(old, list):
                    old.append(value)
                else:
                    self.fields[name] = [old, value]
            else:
                self.fields[name] = value

    def _end(self) -> None:
        self.complete = True

    def feed(self, body: bytes) -> None:
        self._parser.write(body)

    def payload(self) -> dict[str, Any]:
        files = list(self.files)
        part = self._part
        if part is not None and "file" in part:
            files.append(part["file"] | {"bytes_received": part["bytes"]})
        return {"fields": self.fields, "files": files}


class _BodyCapture:
    def __init__(self, content_type: str, limit: int) -> None:
        self.content_type = content_type
        self.limit = limit
        self.size = 0
        self.complete = False
        self.buffer = bytearray()
        self.multipart: _MultipartCapture | None = None
        self.capture_error = False
        self.metadata_truncated = False
        if content_type.lower().startswith("multipart/form-data"):
            try:
                self.multipart = _MultipartCapture(content_type, limit)
            except Exception:
                self.capture_error = True

    def feed(self, body: bytes, complete: bool = False) -> None:
        self.size += len(body)
        self.complete = complete
        if self.capture_error:
            return
        try:
            if self.multipart is not None:
                self.multipart.feed(body)
            elif "json" in self.content_type.lower() or self.content_type.lower().startswith("application/x-www-form-urlencoded"):
                remaining = max(0, self.limit - len(self.buffer))
                self.buffer.extend(body[:remaining])
        except Exception:
            self.capture_error = True

    @property
    def truncated(self) -> bool:
        if self.capture_error:
            return True
        if self.metadata_truncated:
            return True
        if self.multipart is not None:
            return self.multipart.truncated or not self.multipart.complete
        return self.size > self.limit or (self.size > 0 and not self.complete)

    def payload(self) -> Any:
        if self.multipart is not None:
            return self.multipart.payload()
        if not self.size:
            return None
        if self.capture_error:
            return {"capture": "unavailable", "content_type": self.content_type}
        raw = bytes(self.buffer)
        if "json" in self.content_type.lower():
            if not self.truncated:
                try:
                    return json.loads(raw)
                except (ValueError, UnicodeError, RecursionError):
                    pass
            return {
                "capture": "truncated" if self.truncated else "invalid_json",
                "parse_error": "incomplete_json" if self.truncated else "invalid_json",
                "bytes_captured": len(raw), "identifiers": _partial_identifiers(raw),
                "preview": raw[:4096].decode("utf-8", errors="replace"),
            }
        if self.content_type.lower().startswith("application/x-www-form-urlencoded"):
            self.metadata_truncated = raw.count(b"&") >= 1024
            return {"fields": _query(raw)}
        return {"capture": "binary_omitted", "content_type": self.content_type,
                "bytes_received": self.size}


def _category(path: str) -> str:
    endpoint = path.rstrip("/").rsplit("/", 1)[-1]
    if path.startswith("/api/v1/test/workflow/"):
        return {"1-start": "start", "2-sequence": "sequence", "3-complete": "complete", "4-report": "report"}.get(endpoint, "other")
    if path.startswith("/api/v1/sessions/") and endpoint in {"start", "end", "complete"}:
        return endpoint
    if "sequence" in path:
        return "sequence"
    if endpoint == "query" or "voice" in path:
        return "query"
    if "/reports/" in path:
        return "report"
    return "other"


def _ids(*payloads: Any) -> dict[str, str]:
    found: dict[str, str] = {}
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        roots = [payload] + [payload.get(key) for key in ("fields", "identifiers", "data", "body", "query_parameters")]
        for root in roots:
            if not isinstance(root, dict):
                continue
            for key in _IDENTIFIERS:
                value = root.get(key)
                if key not in found and isinstance(value, (str, int)) and not isinstance(value, bool) and str(value).strip():
                    found[key] = str(value)
    return found


def _bounded_payload(payload: Any, limit: int) -> tuple[Any, bool]:
    if payload is None:
        return None, False
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if len(serialized.encode("utf-8")) <= limit:
        return payload, False
    return {"capture": "truncated", "identifiers": _ids(payload),
            "preview": serialized[:4096]}, True


class ApiCallLoggingMiddleware:
    def __init__(self, app: Any, *, store: ApiCallStore) -> None:
        self.app = app
        self.store = store

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        path = scope.get("path", "")
        if (scope.get("type") != "http" or not (path == "/api/v1" or path.startswith("/api/v1/"))
                or path.startswith("/api/v1/stream/") or "/assets/" in path):
            await self.app(scope, receive, send)
            return
        received_at, started = _now(), time.perf_counter()
        headers = scope.get("headers", [])
        provenance = capture_provenance(scope)
        content_type = _header(headers, b"content-type")
        request = _BodyCapture(content_type, REQUEST_CAPTURE_LIMIT)
        response = _BodyCapture("", RESPONSE_CAPTURE_LIMIT)
        status_code = None
        streaming = False
        failed = False
        recorded = False

        def record_event(finished_at: str, duration_ms: float) -> None:
            try:
                request_json, response_json = request.payload(), response.payload()
                raw_query = scope.get("query_string", b"")
                query = _query(raw_query)
                query_truncated = len(raw_query) > REQUEST_CAPTURE_LIMIT or raw_query.count(b"&") >= 1024
                # Public handlers consume identity from their body/form or canonical
                # response/path. Only synthetic test handlers accept session_id in
                # the query; arbitrary ignored query keys must not misattribute calls.
                test_identity = ({"session_id": query.get("session_id")}
                                 if path.startswith("/api/v1/test/") else None)
                successful_response = response_json if status_code is not None and 200 <= status_code < 300 else None
                identifiers = _ids(successful_response, request_json, test_identity)
                match = re.match(r"^/api/v1/sessions/([^/]+)/reports(?:/|$)", path)
                if match:
                    identifiers["session_id"] = unquote(match.group(1))
                request_id = _header(headers, b"x-request-id")
                if request_id and "request_id" not in identifiers:
                    identifiers["request_id"] = request_id
                if query:
                    request_json = ({"query_parameters": query} if request_json is None else
                                    {"body": request_json, "query_parameters": query})
                request_json, request_limited = _bounded_payload(request_json, REQUEST_CAPTURE_LIMIT)
                response_json, response_limited = _bounded_payload(response_json, RESPONSE_CAPTURE_LIMIT)
                self.store.record({
                    **provenance,
                    **identifiers, "method": scope.get("method", ""), "path": path,
                    "category": _category(path),
                    "source": "test" if path.startswith("/api/v1/test/") else "http",
                    "received_at": received_at, "finished_at": finished_at,
                    "status_code": status_code if status_code is not None else (500 if failed else None),
                    "duration_ms": duration_ms,
                    "request_json": request_json, "response_json": response_json,
                    "request_bytes": request.size, "response_bytes": response.size,
                    "request_truncated": int(request.truncated or query_truncated or request_limited),
                    "response_truncated": int(response.truncated or response_limited or
                                              (status_code is not None and not response.complete)),
                    "content_type": content_type,
                })
            except Exception:
                logger.warning("API call capture failed", exc_info=True)

        async def finish() -> None:
            nonlocal recorded
            if recorded or streaming:
                return
            recorded = True
            finished_at = _now()
            duration_ms = round((time.perf_counter() - started) * 1000, 3)
            try:
                # Parsing, serialization and SQLite waits run away from the event
                # loop. Await completion before the app begins BackgroundTasks,
                # and retain cleanup if its request cancel scope has been cancelled.
                with CancelScope(shield=True):
                    await to_thread.run_sync(record_event, finished_at, duration_ms)
            except Exception:
                logger.warning("API call recording worker failed", exc_info=True)

        async def capture_receive() -> dict[str, Any]:
            message = await receive()
            if message["type"] == "http.request":
                request.feed(message.get("body", b""), not message.get("more_body", False))
            return message

        async def capture_send(message: dict[str, Any]) -> None:
            nonlocal status_code, streaming, response
            if message["type"] == "http.response.start":
                status_code = message["status"]
                response_type = _header(message.get("headers", []), b"content-type")
                streaming = response_type.lower().startswith("text/event-stream")
                response = _BodyCapture(response_type, RESPONSE_CAPTURE_LIMIT)
            elif message["type"] == "http.response.body" and not streaming:
                response.feed(message.get("body", b""), not message.get("more_body", False))
            await send(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                await finish()

        try:
            await self.app(scope, capture_receive, capture_send)
        except BaseException:
            failed = True
            raise
        finally:
            await finish()
