"""Read-only origin scoping from server-recorded session creation evidence."""
from __future__ import annotations

import html
import re
import sqlite3
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

ORIGINS = {"all":"전체", "real":"실제 훈련(일반 연동)", "test":"모의 테스트", "unknown":"미분류"}
ORIGIN_NOTE = "생성 경로 기준입니다. 실제 훈련(일반 연동)은 일반 시작 API로 생성되어 수동 테스트도 포함할 수 있습니다. 모의 테스트는 서버의 모의 훈련 경로입니다. 과거 기록·출처 미기록은 이름이나 HTTP 수집 경로로 추정하지 않고 미분류로 표시합니다."


def normalize_origin(value: str) -> str:
    return value if value in ORIGINS else "all"


def columns(conn: sqlite3.Connection, table: str) -> set[str]:
    quoted = '"' + table.replace('"','""') + '"'
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({quoted})")}


def origin_expression(session_columns: set[str], alias: str = "s") -> str:
    if "creation_origin" not in session_columns:
        return "'unknown'"
    return f"CASE WHEN {alias}.creation_origin IN ('real','test') THEN {alias}.creation_origin ELSE 'unknown' END"


def scope_condition(conn: sqlite3.Connection, table: str, origin: str, alias: str = "") -> str:
    """All unlinked records are unknown; api_calls.source is never origin evidence."""
    origin = normalize_origin(origin)
    if origin == "all":
        return "1"
    own_columns, session_columns = columns(conn,table), columns(conn,"sessions")
    prefix = (alias + ".") if alias else ('"' + table.replace('"','""') + '".')
    if table == "sessions":
        return f"({origin_expression(session_columns,alias or 'sessions')})='{origin}'"
    if "session_id" not in session_columns:
        return "1" if origin == "unknown" else "0"
    owner = "session_key" if "session_key" in own_columns else "session_id" if "session_id" in own_columns else None
    match = f"s.session_id={prefix}{owner}" if owner else f"s.user_id={prefix}user_id" if table == "users" and "user_id" in session_columns else None
    if not match:
        return "1" if origin == "unknown" else "0"
    exists = f"EXISTS (SELECT 1 FROM sessions s WHERE {match} AND ({origin_expression(session_columns)})='{origin}')"
    if origin == "unknown":
        exists = f"({exists} OR NOT EXISTS (SELECT 1 FROM sessions s WHERE {match}))"
    return exists


def origin_counts(conn: sqlite3.Connection | None) -> dict[str,int]:
    result = {key:0 for key in ("real","test","unknown")}
    if conn is not None and "session_id" in columns(conn,"sessions"):
        result.update(dict(conn.execute(f"SELECT {origin_expression(columns(conn,'sessions'))},COUNT(*) FROM sessions s GROUP BY 1")))
    return result


def origin_control(conn: sqlite3.Connection | None, origin: str, **params: object) -> str:
    counts = origin_counts(conn)
    options = "".join(f'<option value="{key}"{" selected" if key == origin else ""}>{label}</option>' for key,label in ORIGINS.items())
    counts_text = " · ".join(f'{ORIGINS[key]} {value:,}건' for key,value in counts.items())
    hidden = "".join(f'<input type="hidden" name="{html.escape(key)}" value="{html.escape(str(value))}">' for key,value in params.items())
    return f'<div class="toolbar origin-toolbar"><label for="origin-filter">데이터 출처</label><select id="origin-filter" name="origin" form="origin-form">{options}</select><form id="origin-form" method="get">{hidden}<button>적용</button></form></div><p class="session-note panel-note">세션 전체: {counts_text}<br>{ORIGIN_NOTE}</p>'


def preserve_origin_links(markup: str, origin: str) -> str:
    """Preserve a DB scope through navigation and GET forms, including raw tables."""
    origin = normalize_origin(origin)
    if origin == "all":
        return markup
    def link(match: re.Match) -> str:
        parsed = urlsplit(html.unescape(match.group(1)))
        if parsed.netloc or not (parsed.path == "/admin" or parsed.path.startswith("/admin/")):
            return match.group(0)
        params = dict(parse_qsl(parsed.query,keep_blank_values=True))
        # Origin selector links can explicitly choose another scope.
        params.setdefault("origin",origin)
        return 'href="' + html.escape(urlunsplit(parsed._replace(query=urlencode(params)))) + '"'
    markup = re.sub(r'href="([^"]*)"',link,markup)
    def form(match: re.Match) -> str:
        tag, body = match.group(1), match.group(2)
        if 'id="origin-form"' not in tag and not re.search(r'name=["\']origin["\']',body):
            body = f'<input type="hidden" name="origin" value="{origin}">' + body
        return tag + body + '</form>'
    return re.sub(r'(<form\b[^>]*>)(.*?)</form>',form,markup,flags=re.S)
