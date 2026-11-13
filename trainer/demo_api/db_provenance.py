"""Read-only provenance from a uniquely matching, successful creation call.

Never copy an arbitrary session's latest caller to its records. Old data remains
unknown; file imports have no HTTP caller. Timestamps are compared at second
precision because several domain stores intentionally persist UTC seconds.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any

from trainer.demo_api.request_provenance import caller_label, valid_ip


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute('PRAGMA table_info("' + table.replace('"', '""') + '")')}


def _object(value: Any) -> dict:
    try:
        data = json.loads(value) if isinstance(value, str) else value
    except (ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def _body(value: Any) -> dict:
    data = _object(value)
    return _object(data["body"]) if "body" in data else data


def _result(status: str, call: dict | None = None, *, evidence: str = "") -> dict:
    call = call or {}
    missing = "해당 없음" if status == "file" else "알 수 없음" if status == "ambiguous" else "미기록"
    return {"status": status, "client_ip": valid_ip(call.get("client_ip")) or missing,
            "caller_label": caller_label(call) if call else ("파일 수집" if status == "file" else missing),
            "call_id": call.get("id"), "evidence": evidence,
            "source_evidence": call.get("source_evidence"),
            "caller_origin": call.get("caller_origin"), "caller_referer": call.get("caller_referer")}


def record_provenance(conn: sqlite3.Connection, table: str, row: Any) -> dict:
    data = dict(row)
    if table == "api_calls":
        return _result("direct", data, evidence="수집된 실제 API 호출")
    if table == "legacy_import_records":
        return _result("file", evidence="파일 이관 기록 · HTTP 호출 IP 해당 없음")
    stamp_key = {"sessions": "created_at", "users": "created_at", "sequences": "received_at",
                 "query_turns": "created_at", "completions": "requested_at"}.get(table)
    if stamp_key is None or not data.get(stamp_key):
        return _result("unknown")
    columns = _columns(conn, "api_calls")
    required = {"id", "session_id", "method", "category", "status_code", "received_at", "finished_at"}
    if not required <= columns:
        return _result("unknown")
    conditions = ["method='POST'", "typeof(status_code) IN ('integer','real')", "status_code>=200", "status_code<300",
                  "CAST(strftime('%s',received_at) AS INTEGER)<=CAST(strftime('%s',?) AS INTEGER)",
                  "CAST(strftime('%s',finished_at) AS INTEGER)>=CAST(strftime('%s',?) AS INTEGER)"]
    params: list[Any] = [data[stamp_key], data[stamp_key]]
    session_id = data.get("session_id", data.get("session_key"))
    if session_id:
        if table == "query_turns" and session_id == "CHAT:" + str(data.get("user_id") or ""):
            conditions.append("session_id IS NULL")
        else:
            conditions.append("session_id=?")
            params.append(session_id)
    elif table != "users":
        return _result("unknown")
    if data.get("user_id"):
        if "user_id" not in columns:
            return _result("unknown")
        conditions.append("user_id=?")
        params.append(data["user_id"])
    category = {"sessions": "start", "users": "start", "sequences": "sequence", "query_turns": "query", "completions": "complete"}[table]
    conditions.append("category=?")
    params.append(category)
    if table == "sessions" and data.get("creation_path"):
        if "path" not in columns:
            return _result("unknown")
        conditions.append("path=?")
        params.append(data["creation_path"])
    if table == "sequences":
        if "sequence_id" not in columns or not data.get("sequence_id"):
            return _result("unknown")
        conditions.append("sequence_id=?")
        params.append(data["sequence_id"])
    payloads = table in {"query_turns", "completions"}
    # Sequence bodies can be megabytes; list/detail provenance never needs them.
    projected = [name for name in ("id", "client_ip", "caller_source", "source", "path", "source_evidence", "caller_origin", "caller_referer") if name in columns]
    if payloads:
        if not {"request_json", "response_json"} <= columns:
            return _result("unknown")
        projected += ["request_json", "response_json"]
    candidates = conn.execute("SELECT " + ",".join('"' + name + '"' for name in projected) + " FROM api_calls WHERE " + " AND ".join(conditions) + " ORDER BY id LIMIT 9", params).fetchall()
    if len(candidates) > 8:
        return _result("ambiguous", evidence="동일 식별자·저장 시각의 호출이 많아 생성 호출을 특정할 수 없음")
    matches = []
    for candidate in candidates:
        call = dict(candidate)
        if table == "query_turns":
            request, response = _body(call.get("request_json")), _object(_object(call.get("response_json")).get("data"))
            if not all(request.get(key) == data.get(key) for key in ("prompt", "task_type", "user_id")) or response.get("answer") != data.get("answer"):
                continue
        elif table == "completions":
            response = _object(_object(call.get("response_json")).get("data"))
            if not data.get("report_id") or response.get("report_id") != data["report_id"]:
                continue
        matches.append(call)
    if len(matches) == 1:
        return _result("matched", matches[0], evidence="식별자·서버 저장 시각 일치 (초 단위)")
    return _result("ambiguous" if matches else "unknown", evidence="여러 호출이 일치하여 생성 호출을 특정할 수 없음" if matches else "")


def provenance_meta(provenance: dict) -> list[dict]:
    result = [{"label": "호출 채널", "value": provenance["caller_label"]},
              {"label": "호출 IP", "value": provenance["client_ip"]}]
    for key, label in (("call_id", "연결 API 호출 번호"), ("evidence", "호출 연결 근거"),
                       ("source_evidence", "채널 판별 근거"), ("caller_origin", "브라우저 Origin"), ("caller_referer", "브라우저 Referer")):
        if provenance.get(key):
            result.append({"label": label, "value": provenance[key]})
    return result
