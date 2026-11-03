"""Friendly, read-only SQLite dashboard retained at the legacy /admin link."""

from __future__ import annotations

import html
import json
import math
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote, urlencode

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import HTMLResponse, FileResponse

from trainer.demo_api.sequence_store import DEFAULT_DB_PATH
from trainer.demo_api.db_dashboard import DASHBOARD_CSS, render_dashboard
from trainer.demo_api.db_record_detail import DETAIL_CSS, DETAIL_HTML, record_detail, json_safe
from trainer.demo_api.db_sessions import SESSION_CSS, render_session_list, render_session_detail, session_event_detail

_CSS = """
:root{color-scheme:light;--bg:#f4f7fb;--panel:#fff;--ink:#152033;--muted:#68758a;--line:#dde4ee;
--brand:#3157d5;--good:#087f5b;--shadow:0 8px 30px rgba(31,45,76,.08)}*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,Pretendard,"Noto Sans KR",system-ui,sans-serif}
.app-shell{display:grid;grid-template-columns:260px minmax(0,1fr);min-height:100vh}.sidebar{background:#13213d;color:#eef3ff;
padding:24px 18px;position:sticky;top:0;height:100vh;overflow:auto}.brand{font-weight:800;font-size:19px}.brand small{display:block;
color:#9fb2da;font-size:12px;font-weight:500;margin-top:6px}.side-title{font-size:11px;text-transform:uppercase;letter-spacing:.12em;
color:#8299c7;margin:28px 10px 10px}.table-link{display:flex;justify-content:space-between;gap:12px;color:#dbe6ff;text-decoration:none;
padding:11px 12px;border-radius:9px;margin:3px 0;font-size:14px}.table-link:hover,.table-link.active{background:#28406d;color:#fff}
.count{font-variant-numeric:tabular-nums;color:#9fb2da}.main{padding:30px;min-width:0}.topbar{display:flex;justify-content:space-between;
gap:20px;align-items:flex-start;margin-bottom:22px}.eyebrow{color:var(--brand);font-size:12px;font-weight:800;letter-spacing:.08em}
h1{font-size:27px;margin:5px 0 7px;letter-spacing:-.03em}.sub{color:var(--muted);font-size:14px}.actions a{display:inline-block;
background:var(--brand);color:#fff;text-decoration:none;border-radius:9px;padding:10px 14px;font-size:13px;font-weight:700}
.cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin-bottom:18px}.card{background:var(--panel);
border:1px solid var(--line);border-radius:13px;padding:17px;box-shadow:var(--shadow)}.card span{display:block;color:var(--muted);font-size:12px}
.card strong{display:block;font-size:23px;margin-top:6px}.toolbar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;background:var(--panel);
border:1px solid var(--line);border-radius:13px 13px 0 0;padding:14px}.toolbar form{display:flex;gap:8px;flex:1}.toolbar input,
.toolbar select{border:1px solid #cbd5e1;border-radius:8px;padding:9px 11px;background:#fff;min-width:0}.toolbar input{flex:1}
button{border:0;border-radius:8px;background:var(--brand);color:#fff;font-weight:700;padding:9px 14px;cursor:pointer}.reset{color:var(--muted);font-size:13px}
.table-wrap{background:var(--panel);border:1px solid var(--line);border-top:0;overflow:auto;max-height:66vh}table{border-collapse:separate;
border-spacing:0;width:100%;font-size:13px}th{position:sticky;top:0;background:#eef3fb;color:#40506a;text-align:left;padding:11px 12px;
border-bottom:1px solid var(--line);white-space:nowrap;z-index:1}td{padding:10px 12px;border-bottom:1px solid #edf1f6;vertical-align:top;
max-width:420px;overflow-wrap:anywhere}tr:hover td{background:#f8faff}.null{color:#9aa6b6;font-style:italic}.status{display:inline-block;
padding:3px 8px;border-radius:999px;background:#e9f7f1;color:var(--good);font-weight:700;font-size:11px}details summary{cursor:pointer;color:var(--brand)}
pre{white-space:pre-wrap;margin:8px 0 0;background:#f5f7fb;padding:9px;border-radius:7px;max-height:220px;overflow:auto}.empty{text-align:center;
padding:54px;color:var(--muted)}.pagination{display:flex;justify-content:space-between;align-items:center;background:var(--panel);border:1px solid var(--line);
border-top:0;border-radius:0 0 13px 13px;padding:13px 15px;font-size:13px;color:var(--muted)}.pagination a{color:var(--brand);text-decoration:none;font-weight:700}
.readonly{display:inline-flex!important;align-items:center;gap:6px;color:var(--good)!important;font-weight:700;font-size:14px!important}.dot{width:8px;height:8px;border-radius:50%;background:#20c997}
@media(max-width:850px){.app-shell{grid-template-columns:1fr}.sidebar{position:relative;height:auto}.main{padding:20px}.cards{grid-template-columns:1fr}.topbar{display:block}.actions{margin-top:14px}}
"""


def _page(content: str, sidebar: str = "") -> HTMLResponse:
    return HTMLResponse(
        "<!doctype html><html lang='ko'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Co-Pilot DB Viewer</title><style>{_CSS}{DASHBOARD_CSS}{SESSION_CSS}{DETAIL_CSS}</style></head>"
        f'<body><div class="app-shell"><aside class="sidebar">{sidebar}</aside>'
        f'<main class="main">{content}</main></div>{DETAIL_HTML}</body></html>',
        headers={"Cache-Control":"no-store"},
    )


def _cell(value: object) -> str:
    if value is None:
        return '<span class="null">NULL</span>'
    text = str(value)
    if text.upper() in {"ACTIVE", "COMPLETED", "SUCCESS", "PENDING"}:
        return f'<span class="status">{html.escape(text)}</span>'
    if text.lstrip().startswith(("{", "[")):
        if len(text) > 2000:
            return f'<span>JSON · {len(text):,}자</span><br><small>상세 버튼으로 전체 파라미터 보기</small>'
        try:
            pretty = json.dumps(json.loads(text), ensure_ascii=False, indent=2)
            return f"<details><summary>JSON 보기</summary><pre>{html.escape(pretty)}</pre></details>"
        except (json.JSONDecodeError, TypeError):
            pass
    return html.escape(text)


def _sidebar(tables: list[tuple[str, int]], selected: str | None, overview: str = "home") -> str:
    labels = {"sessions":"훈련 세션","users":"작업자","sequences":"시퀀스 기록",
              "query_turns":"질문·답변","completions":"최종 리포트",
              "sequence_analysis_results":"시퀀스 분석 결과","legacy_import_records":"이전 데이터 이관 기록","api_calls":"API 호출 이력"}
    links = "".join(
        f'<a class="table-link{" active" if name == selected else ""}" href="/admin?table={quote(name)}">'
        f"<span class='table-name'>{html.escape(labels.get(name,name))}<small>{html.escape(name)}</small></span><span class='count'>{count:,}</span></a>"
        for name, count in tables
    )
    is_home = selected is None and overview == "home"
    current_attr = ' aria-current="page"' if is_home else ""
    home = f'<a class="table-link home-nav{" active" if is_home else ""}"{current_attr} href="/admin"><span><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><path d="m3 10 9-7 9 7v10H3Z"/><path d="M9 20v-7h6v7"/></svg>Home</span></a>'
    sessions = f'<a class="table-link home-nav{" active" if overview == "sessions" else ""}" href="/admin/sessions"><span>▤ 훈련세션</span></a>'
    return '<div class="brand">AI Co-Pilot<small>훈련 데이터 대시보드</small></div><div class="side-title">Overview</div>' + home + sessions + '<div class="side-title">데이터 조회</div>' + links


@contextmanager
def _read_database():
    path = Path(os.environ.get("SEQUENCE_DB_PATH", DEFAULT_DB_PATH))
    if not path.is_file():
        yield None
        return
    conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True, timeout=3)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN")
        yield conn
    finally:
        conn.close()


def _table_counts(conn):
    if conn is None:
        return []
    names = [str(r[0]) for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    return [(name, int(conn.execute('SELECT COUNT(*) FROM "' + name.replace('"','""') + '"').fetchone()[0])) for name in names]


def create_db_viewer_router() -> APIRouter:
    router = APIRouter()

    @router.get("/admin/assets/db_viewer.js", include_in_schema=False)
    def detail_script():
        return FileResponse(Path(__file__).parent / "static" / "db_viewer.js", media_type="text/javascript", headers={"Cache-Control":"no-cache"})

    @router.get("/admin/api/record", include_in_schema=False)
    def raw_record(table: str = Query(max_length=200), rowid: int = Query(ge=-(2**63), le=2**63-1)):
        with _read_database() as conn:
            data = record_detail(conn, table, rowid) if conn is not None else None
        if data is None:
            raise HTTPException(404, "record not found")
        return data

    @router.get("/admin/sessions", include_in_schema=False, response_class=HTMLResponse)
    def sessions(q: str = Query(default="", max_length=100), state: str = Query(default="all", pattern="^(all|unfinished|ended)$"), page: int = Query(default=1, ge=1), limit: int = Query(default=25, ge=10, le=200)):
        with _read_database() as conn:
            side = _sidebar(_table_counts(conn), None, "sessions")
            content = render_session_list(conn, q=q, state=state, page=page, limit=limit) if conn is not None else '<h1>훈련세션</h1><p class="empty">아직 저장된 훈련세션이 없습니다.</p>'
            return _page(content, side)

    @router.get("/admin/sessions/view", include_in_schema=False, response_class=HTMLResponse)
    def session_view(session_id: str = Query(min_length=1, max_length=256), kind: str = Query(default="all", pattern="^(all|start|end|complete|sequence|query|report|other)$"), call_page: int = Query(default=1,ge=1), q: str = Query(default="",max_length=100), outcome: str = Query(default="all",pattern="^(all|errors)$"), order: str = Query(default="newest",pattern="^(newest|oldest)$")):
        with _read_database() as conn:
            content = render_session_detail(conn, session_id, kind=kind, call_page=call_page, q=q, outcome=outcome, order=order) if conn is not None else None
            if content is None:
                raise HTTPException(404, "session not found")
            return _page(content, _sidebar(_table_counts(conn), None, "sessions"))

    @router.get("/admin/api/session-event", include_in_schema=False)
    def event_view(session_id: str = Query(min_length=1,max_length=256), event_key: str = Query(min_length=1,max_length=400)):
        with _read_database() as conn:
            result = session_event_detail(conn, session_id, event_key) if conn is not None else None
        if result is None:
            raise HTTPException(404, "event not found")
        return json_safe(result)

    @router.get("/admin", include_in_schema=False, response_class=HTMLResponse)
    def database_viewer(
        table: str | None = None,
        limit: int = Query(default=50, ge=10, le=200),
        page: int = Query(default=1, ge=1),
        q: str = Query(default="", max_length=100),
        days: int = Query(default=30, ge=7, le=90),
        metric: str = Query(default="sessions", pattern="^(sessions|sequences|queries)$"),
    ) -> HTMLResponse:
        path = Path(os.environ.get("SEQUENCE_DB_PATH", DEFAULT_DB_PATH))
        if not path.is_file():
            return _page(render_dashboard(None, {}, days=days, metric=metric), _sidebar([], None))
        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN")
            names = [str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
            selected = table if table in names else None
            table_counts: list[tuple[str, int]] = []
            for name in names:
                quoted_name = '"' + name.replace('"', '""') + '"'
                table_counts.append((name, int(conn.execute(f"SELECT COUNT(*) FROM {quoted_name}").fetchone()[0])))
            side = _sidebar(table_counts, selected)
            if selected is None:
                return _page(render_dashboard(conn, dict(table_counts), days=days, metric=metric), side)
            quoted = '"' + selected.replace('"', '""') + '"'
            columns = [str(item[1]) for item in conn.execute(f"PRAGMA table_info({quoted})")]
            where = ""
            params: list[object] = []
            if q and columns:
                safe_columns = [column.replace('"', '""') for column in columns]
                where = " WHERE " + " OR ".join(f'CAST("{column}" AS TEXT) LIKE ?' for column in safe_columns)
                params = [f"%{q}%"] * len(columns)
            total = int(conn.execute(f"SELECT COUNT(*) FROM {quoted}{where}", params).fetchone()[0])
            pages = max(1, math.ceil(total / limit))
            current_page = min(page, pages)
            rows = conn.execute(f"SELECT rowid AS __viewer_rowid, * FROM {quoted}{where} ORDER BY rowid DESC LIMIT ? OFFSET ?", [*params, limit, (current_page - 1) * limit]).fetchall()
            heading = '<th>상세</th>' + "".join(f"<th>{html.escape(column)}</th>" for column in columns)
            def record_row(row):
                url = html.escape('/admin/api/record?' + urlencode({'table':selected,'rowid':row['__viewer_rowid']}))
                return f'<tr data-detail-url="{url}" tabindex="0" aria-label="레코드 {row["__viewer_rowid"]} 상세"><td><button type="button" class="record-button" data-detail-url="{url}">상세</button></td>' + ''.join(f'<td>{_cell(row[column])}</td>' for column in columns) + '</tr>'
            rendered = ''.join(record_row(row) for row in rows)
            if not rendered:
                rendered = f'<tr><td class="empty" colspan="{len(columns)+1}">조건에 맞는 데이터가 없습니다.</td></tr>'
            query = html.escape(q)
            base = f"/admin?table={quote(selected)}&limit={limit}&q={quote(q)}"
            previous = f'<a href="{base}&page={current_page - 1}">← 이전</a>' if current_page > 1 else "<span></span>"
            following = f'<a href="{base}&page={current_page + 1}">다음 →</a>' if current_page < pages else "<span></span>"
            options = ''.join(f'<option value="{size}"{" selected" if size == limit else ""}>{size}개씩</option>' for size in (10,25,50,100,200))
            content = f"""
            <div class="topbar"><div><div class="eyebrow">DATABASE EXPLORER</div><h1>Co-Pilot 데이터베이스</h1>
            <div class="sub">운영 데이터의 최신 상태를 안전하게 조회합니다.</div></div><div class="actions"><a href="/docs">Swagger로 돌아가기</a></div></div>
            <div class="cards"><div class="card"><span>선택 테이블</span><strong>{html.escape(selected)}</strong></div>
            <div class="card"><span>검색 결과</span><strong>전체 {total:,}건</strong></div><div class="card"><span>접근 모드</span>
            <strong class="readonly"><i class="dot"></i>읽기 전용</strong></div></div>
            <div class="toolbar"><form method="get" action="/admin"><input type="hidden" name="table" value="{html.escape(selected)}">
            <input name="q" value="{query}" placeholder="모든 열에서 검색"><select name="limit">{options}</select><button type="submit">검색</button></form>
            <a class="reset" href="/admin?table={quote(selected)}">초기화</a></div><div class="table-wrap"><table><thead><tr>{heading}</tr></thead>
            <tbody>{rendered}</tbody></table></div><div class="pagination">{previous}<span>{current_page} / {pages} 페이지</span>{following}</div>
            """
            return _page(content, side)
        except sqlite3.Error:
            return _page('<h1>Co-Pilot 데이터베이스</h1><p class="empty">DB를 읽을 수 없습니다.</p>')
        finally:
            if conn is not None:
                conn.close()

    return router
