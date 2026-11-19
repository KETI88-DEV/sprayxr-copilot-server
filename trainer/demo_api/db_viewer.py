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
from trainer.demo_api.db_record_detail import DETAIL_CSS, DETAIL_HTML, PRIVATE_TABLES, record_detail, json_safe
from trainer.demo_api.db_sessions import SESSION_CSS, render_session_list, render_session_detail, session_event_detail
from trainer.demo_api.portal_navigation import portal_document
from trainer.demo_api.db_origin import origin_control, preserve_origin_links, scope_condition
from trainer.demo_api.db_report_status import report_progress
from trainer.demo_api.db_provenance import record_provenance
from trainer.demo_api.training_data_catalog import TrainingDataCatalog, create_training_data_router
from trainer.demo_api.training_data_ui import TRAINING_DATA_CSS, render_training_data
from trainer.demo_api.db_sequence_heatmap import attach_heatmap_link, create_sequence_heatmap_router
from trainer.demo_api.db_sequence_heatmap_ui import SEQUENCE_HEATMAP_CSS
from trainer.demo_api.feedback import render_feedback_list
from trainer.demo_api.db_record_time import MISSING_TIME, first_record_time

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

_CSS += """
.sidebar{display:flex;flex-direction:column}.sidebar>.brand,.sidebar>.side-title,.sidebar>.table-link{flex-shrink:0}
.sidebar-admin{margin-top:auto;padding-top:28px;flex-shrink:0}.sidebar-admin a{display:flex;align-items:center;gap:9px;border-top:1px solid #ffffff15;padding:16px 10px 3px;color:#91a5c6;text-decoration:none;font-size:12px;font-weight:500;transition:color .15s}
.sidebar-admin svg{width:16px;height:16px}.sidebar-admin a:hover,.sidebar-admin a.active{color:#e1eaff}.sidebar-admin a:focus-visible{outline:2px solid #9cb7f2;outline-offset:3px}
@media(max-width:850px){.sidebar-admin{margin-top:12px;padding-top:8px}}
.sequence-text-guide{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.6;min-width:280px;max-width:420px;max-height:8em;overflow:auto}
.sequence-guide-legacy{display:block;margin-top:5px;color:var(--muted);font-size:11px}
"""


def _page(content: str, sidebar: str = "", origin: str = "all") -> HTMLResponse:
    content, sidebar = preserve_origin_links(content,origin), preserve_origin_links(sidebar,origin)
    document = (
        "<!doctype html><html lang='ko'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Co-Pilot DB Viewer</title><style>{_CSS}{DASHBOARD_CSS}{SESSION_CSS}{DETAIL_CSS}{TRAINING_DATA_CSS}{SEQUENCE_HEATMAP_CSS}</style></head>"
        f'<body><div class="app-shell"><aside class="sidebar">{sidebar}</aside>'
        f'<main class="main">{content}</main></div>{DETAIL_HTML}</body></html>'
    )
    return HTMLResponse(preserve_origin_links(portal_document(document,"database"),origin), headers={"Cache-Control":"no-store"})


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


def _analysis_text_guide(value: object) -> tuple[str | None, bool]:
    """Read the saved cache field once; never guess a guide from unrelated keys."""
    payload = value
    if isinstance(value, str):
        try:
            payload = json.loads(value)
        except (ValueError, TypeError):
            return (value if value.strip() else None), True
    if isinstance(payload, dict):
        guide = payload.get("text_guide")
        return (guide if isinstance(guide, str) and guide.strip() else None), False
    if isinstance(payload, str):
        return (payload if payload.strip() else None), True
    return None, False


def _guide_cell(guide: str | None, *, legacy: bool = False) -> str:
    if guide is None:
        return '<span class="null">안내 없음</span>'
    escaped = html.escape(guide, quote=True)
    kind = "legacy" if legacy else "text"
    note = '<small class="sequence-guide-legacy">구형 result 문자열</small>' if legacy else ''
    return f'<div class="sequence-text-guide" data-guide-format="{kind}" title="{escaped}">{escaped}</div>{note}'


def _sidebar(tables: list[tuple[str, int]], selected: str | None, overview: str = "home") -> str:
    labels = {"sessions":"훈련 세션","users":"작업자","sequences":"시퀀스 기록",
              "query_turns":"질문·답변","completions":"최종 리포트",
              "sequence_analysis_results":"시퀀스 분석 결과","legacy_import_records":"이전 데이터 이관 기록","api_calls":"API 호출 이력","user_feedback":"사용자 의견 원본"}
    links = "".join(
        f'<a class="table-link{" active" if name == selected else ""}" href="/admin?table={quote(name)}">'
        f"<span class='table-name'>{html.escape(labels.get(name,name))}<small>{html.escape(name)}</small></span><span class='count'>{count:,}</span></a>"
        for name, count in tables
    )
    is_home = selected is None and overview == "home"
    current_attr = ' aria-current="page"' if is_home else ""
    home = f'<a class="table-link home-nav{" active" if is_home else ""}"{current_attr} href="/admin"><span><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><path d="m3 10 9-7 9 7v10H3Z"/><path d="M9 20v-7h6v7"/></svg>Home</span></a>'
    sessions = f'<a class="table-link home-nav{" active" if overview == "sessions" else ""}" href="/admin/sessions"><span>▤ 훈련세션</span></a>'
    training = f'<a class="table-link home-nav{" active" if overview == "training" else ""}"' + (' aria-current="page"' if overview == 'training' else '') + ' href="/admin/training-data"><span><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v7c0 4 16 4 16 0V5M4 12v7c0 4 16 4 16 0v-7"/></svg>학습 데이터</span></a>'
    admin = '<div class="sidebar-admin"><a href="/admin/manage"' + (' class="active" aria-current="page"' if overview == "manage" else '') + '><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><path d="m9 3-.6 2.2-2 .9-2-.7-2 3.4 1.5 1.6v2.3l-1.5 1.6 2 3.4 2-.6 2 .9L9 20h4l.6-2.1 2-.9 2 .6 2-3.4-1.5-1.6v-2.3l1.5-1.6-2-3.4-2 .7-2-.9L13 3Z"/><circle cx="11" cy="11.5" r="3"/></svg><span>관리자 모드</span></a></div>'
    feedback = '<a class="table-link home-nav' + (' active' if overview == 'feedback' else '') + '" href="/admin/feedback"><span><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><path d="M21 15a3 3 0 0 1-3 3H7l-4 3V6a3 3 0 0 1 3-3h12a3 3 0 0 1 3 3Z"/><path d="M8 8h8M8 12h5"/></svg>사용자 의견</span></a>'
    return '<div class="brand">AI Co-Pilot<small>훈련 데이터 대시보드</small></div><div class="side-title">Overview</div>' + home + sessions + training + feedback + '<div class="side-title">데이터 조회</div>' + links + admin


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


def _table_counts(conn, origin: str = "all"):
    if conn is None:
        return []
    names = [str(r[0]) for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name") if str(r[0]) not in PRIVATE_TABLES]
    return [(name, int(conn.execute('SELECT COUNT(*) FROM "' + name.replace('"','""') + '" WHERE ' + scope_condition(conn,name,origin)).fetchone()[0])) for name in names]


def create_db_viewer_router() -> APIRouter:
    router = APIRouter()
    from trainer.demo_api.db_admin_router import create_db_admin_router
    router.include_router(create_db_admin_router())
    router.include_router(create_training_data_router(TrainingDataCatalog()))
    router.include_router(create_sequence_heatmap_router(_read_database))

    @router.get('/admin/feedback', include_in_schema=False, response_class=HTMLResponse)
    def feedback_page(status: str = Query(default='all', pattern='^(all|new|reviewing|done)$'), page: int = Query(default=1, ge=1)):
        with _read_database() as conn:
            return _page(render_feedback_list(conn, status=status, page=page), _sidebar(_table_counts(conn), None, 'feedback'))

    @router.get('/admin/assets/db_sequence_heatmap.js', include_in_schema=False)
    def sequence_heatmap_script():
        return FileResponse(Path(__file__).parent / 'static' / 'db_sequence_heatmap.js', media_type='text/javascript', headers={'Cache-Control': 'no-cache'})

    @router.get('/admin/training-data', include_in_schema=False, response_class=HTMLResponse)
    def training_data_page():
        with _read_database() as conn:
            return _page(render_training_data(), _sidebar(_table_counts(conn), None, 'training'))

    @router.get('/admin/assets/training_data.js', include_in_schema=False)
    def training_data_script():
        return FileResponse(Path(__file__).parent / 'static' / 'training_data.js', media_type='text/javascript', headers={'Cache-Control': 'no-cache'})

    @router.get('/admin/assets/training_verification.js', include_in_schema=False)
    def training_verification_script():
        return FileResponse(Path(__file__).parent / 'static' / 'training_verification.js', media_type='text/javascript', headers={'Cache-Control': 'no-cache'})

    @router.get('/admin/assets/training_verification.css', include_in_schema=False)
    def training_verification_styles():
        return FileResponse(Path(__file__).parent / 'static' / 'training_verification.css', media_type='text/css', headers={'Cache-Control': 'no-cache'})

    @router.get("/admin/assets/db_viewer.js", include_in_schema=False)
    def detail_script():
        return FileResponse(Path(__file__).parent / "static" / "db_viewer.js", media_type="text/javascript", headers={"Cache-Control":"no-cache"})

    @router.get("/admin/api/record", include_in_schema=False)
    def raw_record(table: str = Query(max_length=200), rowid: int = Query(ge=-(2**63), le=2**63-1)):
        with _read_database() as conn:
            data = record_detail(conn, table, rowid) if conn is not None else None
            if data is not None:
                attach_heatmap_link(conn, data, table=table, rowid=rowid)
        if data is None:
            raise HTTPException(404, "record not found")
        return data

    @router.get("/admin/sessions", include_in_schema=False, response_class=HTMLResponse)
    def sessions(q: str = Query(default="", max_length=100), state: str = Query(default="all", pattern="^(all|unfinished|ended)$"), page: int = Query(default=1, ge=1), limit: int = Query(default=25, ge=10, le=200), origin: str = Query(default="all",pattern="^(all|real|test|unknown)$")):
        with _read_database() as conn:
            side = _sidebar(_table_counts(conn,origin), None, "sessions")
            content = render_session_list(conn, q=q, state=state, page=page, limit=limit, origin=origin) if conn is not None else '<h1>훈련세션</h1>'+origin_control(None,origin)+'<p class="empty">아직 저장된 훈련세션이 없습니다.</p>'
            return _page(content, side, origin)

    @router.get("/admin/sessions/view", include_in_schema=False, response_class=HTMLResponse)
    def session_view(session_id: str = Query(min_length=1, max_length=256), kind: str = Query(default="all", pattern="^(all|start|end|complete|sequence|query|report|other)$"), call_page: int = Query(default=1,ge=1), q: str = Query(default="",max_length=100), outcome: str = Query(default="all",pattern="^(all|errors)$"), order: str = Query(default="newest",pattern="^(newest|oldest)$"), origin: str = Query(default="all",pattern="^(all|real|test|unknown)$")):
        with _read_database() as conn:
            content = render_session_detail(conn, session_id, kind=kind, call_page=call_page, q=q, outcome=outcome, order=order, origin=origin) if conn is not None else None
            if content is None:
                raise HTTPException(404, "session not found")
            return _page(content, _sidebar(_table_counts(conn,origin), None, "sessions"), origin)

    @router.get("/admin/api/session-event", include_in_schema=False)
    def event_view(session_id: str = Query(min_length=1,max_length=256), event_key: str = Query(min_length=1,max_length=400)):
        with _read_database() as conn:
            result = session_event_detail(conn, session_id, event_key) if conn is not None else None
            if result is not None:
                attach_heatmap_link(conn, result, session_id=session_id, event_key=event_key)
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
        origin: str = Query(default="all",pattern="^(all|real|test|unknown)$"),
    ) -> HTMLResponse:
        path = Path(os.environ.get("SEQUENCE_DB_PATH", DEFAULT_DB_PATH))
        if not path.is_file():
            return _page(render_dashboard(None, {}, days=days, metric=metric,origin=origin), _sidebar([], None), origin)
        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN")
            names = [str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name") if str(row[0]) not in PRIVATE_TABLES]
            selected = table if table in names else None
            table_counts: list[tuple[str, int]] = []
            for name in names:
                quoted_name = '"' + name.replace('"', '""') + '"'
                table_counts.append((name, int(conn.execute(f"SELECT COUNT(*) FROM {quoted_name} WHERE " + scope_condition(conn,name,origin)).fetchone()[0])))
            side = _sidebar(table_counts, selected)
            if selected is None:
                return _page(render_dashboard(conn, dict(table_counts), days=days, metric=metric,origin=origin), side, origin)
            quoted = '"' + selected.replace('"', '""') + '"'
            columns = [str(item[1]) for item in conn.execute(f"PRAGMA table_info({quoted})")]
            time_column = first_record_time(columns)
            display_columns = [column for column in columns if column != time_column]
            derived_guide = selected == "sequence_analysis_results" and "result" in columns and "text_guide" not in columns
            if derived_guide:
                display_columns.insert(display_columns.index("result"), "text_guide")
            where = " WHERE " + scope_condition(conn,selected,origin)
            params: list[object] = []
            if q and columns:
                safe_columns = [column.replace('"', '""') for column in columns]
                where += " AND (" + " OR ".join(f'CAST("{column}" AS TEXT) LIKE ?' for column in safe_columns) + ")"
                params = [f"%{q}%"] * len(columns)
            total = int(conn.execute(f"SELECT COUNT(*) FROM {quoted}{where}", params).fetchone()[0])
            pages = max(1, math.ceil(total / limit))
            current_page = min(page, pages)
            rows = conn.execute(f"SELECT rowid AS __viewer_rowid, * FROM {quoted}{where} ORDER BY rowid DESC LIMIT ? OFFSET ?", [*params, limit, (current_page - 1) * limit]).fetchall()
            time_heading = '<th>' + ('생성 시각 미기록' if time_column == MISSING_TIME else html.escape(time_column)) + '</th>'
            heading = '<th>상세</th><th>호출 채널</th><th>호출 IP</th>' + "".join(f"<th>{html.escape(column)}</th>" for column in display_columns)
            if selected == 'completions':
                heading = '<th>처리 현황 · 서버 기록 기준</th>' + heading
            heading = time_heading + heading
            def record_row(row):
                url = html.escape('/admin/api/record?' + urlencode({'table':selected,'rowid':row['__viewer_rowid']}))
                provenance = record_provenance(conn, selected, row)
                caller_cells = '<td>' + html.escape(provenance['caller_label']) + '</td><td>' + html.escape(provenance['client_ip']) + '</td>'
                progress_cell = ''
                if selected == 'completions':
                    progress = report_progress(row)
                    progress_cell = '<td>' + html.escape(progress['label']) + '<small style="display:block;line-height:1.7">대기 경과 '+html.escape(progress['elapsed'])+'<br>최근 처리 '+html.escape(progress['latest_at'] or '미기록')+'<br>상태 사유 '+html.escape(progress['reason'])+'</small></td>'
                cells: list[str] = []
                for column in display_columns:
                    if selected == "sequence_analysis_results" and column == "text_guide":
                        guide, legacy = (_analysis_text_guide(row["result"]) if derived_guide else
                                         (row[column] if isinstance(row[column], str) and row[column].strip() else None, False))
                        cells.append('<td data-column="text_guide">' + _guide_cell(guide, legacy=legacy) + '</td>')
                    else:
                        cells.append(f'<td>{_cell(row[column])}</td>')
                time_cell = '<td>' + ('미기록' if time_column == MISSING_TIME else _cell(row[time_column])) + '</td>'
                return f'<tr data-detail-url="{url}" tabindex="0" aria-label="레코드 {row["__viewer_rowid"]} 상세">'+time_cell+progress_cell+f'<td><button type="button" class="record-button" data-detail-url="{url}">상세</button></td>' + caller_cells + ''.join(cells) + '</tr>'
            rendered = ''.join(record_row(row) for row in rows)
            if not rendered:
                rendered = f'<tr><td class="empty" colspan="{len(display_columns)+4+int(selected == "completions")}">조건에 맞는 데이터가 없습니다.</td></tr>'
            query = html.escape(q)
            base = f"/admin?table={quote(selected)}&limit={limit}&q={quote(q)}"
            previous = f'<a href="{base}&page={current_page - 1}">← 이전</a>' if current_page > 1 else "<span></span>"
            following = f'<a href="{base}&page={current_page + 1}">다음 →</a>' if current_page < pages else "<span></span>"
            options = ''.join(f'<option value="{size}"{" selected" if size == limit else ""}>{size}개씩</option>' for size in (10,25,50,100,200))
            content = f"""
            <div class="topbar"><div><div class="eyebrow">DATABASE EXPLORER</div><h1>Co-Pilot 데이터베이스</h1>
            <div class="sub">운영 데이터의 최신 상태를 안전하게 조회합니다.</div></div><div class="actions"><a href="/admin">Home 대시보드</a></div></div>
            {origin_control(conn,origin,table=selected,q=q,limit=limit)}
            <p class="session-note">호출 채널은 요청의 화면·테스트 표식이며 작업자 신원과 별개입니다. 저장 기록은 생성 호출과 명확히 연결될 때만 IP를 표시하며, 과거 IP는 미기록입니다.</p>
            <div class="cards"><div class="card"><span>선택 테이블</span><strong>{html.escape(selected)}</strong></div>
            <div class="card"><span>검색 결과</span><strong>전체 {total:,}건</strong></div><div class="card"><span>접근 모드</span>
            <strong class="readonly"><i class="dot"></i>읽기 전용</strong></div></div>
            <div class="toolbar"><form method="get" action="/admin"><input type="hidden" name="table" value="{html.escape(selected)}">
            <input name="q" value="{query}" placeholder="모든 열에서 검색"><select name="limit">{options}</select><button type="submit">검색</button></form>
            <a class="reset" href="/admin?table={quote(selected)}">초기화</a></div><div class="table-wrap"><table><thead><tr>{heading}</tr></thead>
            <tbody>{rendered}</tbody></table></div><div class="pagination">{previous}<span>{current_page} / {pages} 페이지</span>{following}</div>
            """
            return _page(content, side, origin)
        except sqlite3.Error:
            return _page('<h1>Co-Pilot 데이터베이스</h1><p class="empty">DB를 읽을 수 없습니다.</p>')
        finally:
            if conn is not None:
                conn.close()

    return router
