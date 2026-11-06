"""Read-only session summaries and session-scoped API/legacy event inspection."""

from __future__ import annotations

import html
import json
import math
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

from trainer.demo_api.db_origin import ORIGINS, normalize_origin, origin_control, origin_expression, preserve_origin_links
from trainer.demo_api.db_provenance import record_provenance, provenance_meta
from trainer.demo_api.db_report_status import PROCESSING_FIELDS, report_progress


SESSION_CSS = """
.session-kpis{grid-template-columns:repeat(3,minmax(0,1fr))}.session-kpis.detail-kpis{grid-template-columns:repeat(4,minmax(0,1fr))}
.session-filters{display:flex;gap:8px;flex-wrap:wrap;margin:20px 0 14px}.session-filters a{border:1px solid var(--line);padding:8px 13px;border-radius:8px;color:#65748d;text-decoration:none;font-size:13px;background:#fff}.session-filters a.active{background:var(--brand);border-color:var(--brand);color:#fff}
.session-list{display:grid;gap:10px;margin:14px 0}.session-item{display:grid;grid-template-columns:minmax(160px,1.2fr) minmax(120px,.8fr) minmax(150px,1fr) minmax(170px,1fr);gap:18px;padding:19px 20px;border:1px solid var(--line);border-radius:12px;text-decoration:none;color:var(--ink);background:#fff;transition:border-color .15s}
.session-item.unfinished{background:#eef0f3;border-color:#d9dee6}.session-item.unfinished:hover{background:#e7ebf0;border-color:#9aaacc}.session-item.ended:hover{border-color:#9aaacc;background:#f8faff}.session-item strong{display:block;font-size:14px;overflow-wrap:anywhere}.session-item small{display:block;color:var(--muted);font-size:11px;line-height:1.7;margin-top:5px;overflow-wrap:anywhere}.session-item .session-state{font-size:12px;font-weight:750;color:#63718a}.session-item.ended .session-state{color:#087f5b}.session-item .session-ended-at{display:block;font-size:11px;color:#087f5b;line-height:1.6;margin-top:5px}.session-item .session-counts{font-size:12px;color:#566780;margin-top:9px}
.session-panel{background:#fff;border:1px solid var(--line);border-radius:13px;padding:20px;margin:18px 0}.session-panel h2{font-size:17px;margin:0 0 8px}.session-note{font-size:12px;color:var(--muted);line-height:1.8;margin:8px 0}.session-meta{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px;margin:18px 0 0}.session-meta dt{font-size:11px;color:var(--muted);margin-bottom:7px}.session-meta dd{font-size:13px;margin:0;overflow-wrap:anywhere}.session-meta .session-state{font-weight:750}.session-back{display:inline-block;color:var(--brand);text-decoration:none;font-size:12px;margin-bottom:10px}.session-list-toolbar{border-radius:12px}.session-list-toolbar label{font-size:12px;color:var(--muted);align-self:center;white-space:nowrap}
.session-events{max-height:510px;overflow:auto;border:1px solid var(--line);border-radius:10px;margin-top:14px}.session-event{display:grid;grid-template-columns:156px minmax(0,1fr) 95px;gap:15px;align-items:center;border:0;border-bottom:1px solid #edf1f6;border-radius:0;background:#fff;color:var(--ink);width:100%;padding:13px 15px;text-align:left;font:inherit;font-size:12px;font-weight:400;cursor:pointer}.session-event:last-child{border-bottom:0}.session-event:hover{background:#f5f8fe}.session-event:focus-visible{outline:2px solid var(--brand);outline-offset:-2px}.session-event time{font-size:11px;color:var(--muted);line-height:1.7}.session-event strong{display:block;font-size:12px;overflow-wrap:anywhere}.session-event small{display:block;font-size:11px;color:var(--muted);margin-top:5px;line-height:1.6;overflow-wrap:anywhere}.session-event .event-origin{font-size:10px;color:#68758a;text-align:right}.session-event .event-status{display:block;margin-top:5px;color:#3157d5}.session-event .event-status.error{color:#b64040}.session-events .empty{padding:30px 16px}.session-event-page{display:flex;justify-content:space-between;align-items:center;gap:12px;padding-top:14px;font-size:12px;color:var(--muted)}.session-event-page a{color:var(--brand);text-decoration:none;font-weight:650}.session-call-counts{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:#61718a;margin:12px 0}.session-call-counts strong{color:#40516e}.session-summary-title{font-size:15px;margin:0}.session-search-result{font-size:12px;color:var(--muted);margin:15px 0 0}.session-list-page{border-top:1px solid var(--line);border-radius:12px}
.session-refresh{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:-5px 0 20px;font-size:12px;color:var(--muted)}.session-refresh button{padding:8px 12px;font-size:12px}.session-refresh label{display:flex;gap:6px;align-items:center;cursor:pointer}.session-refresh input{accent-color:var(--brand)}.session-refresh [data-session-refresh-state]{font-size:11px}
.session-report-badge{display:inline-block;margin-top:9px;padding:4px 7px;border-radius:6px;font-size:10px;background:#edf1f6;color:#65748d}.session-report-badge.complete{background:#e4f5ed;color:#087f5b}.session-report-badge.pending{background:#fff2d5;color:#926312}.session-report-badge.failed{background:#ffe8e8;color:#b64040}.session-health{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin:16px 0}.session-health>div{background:#f6f8fc;border-radius:9px;padding:13px}.session-health span{display:block;font-size:12px;color:var(--muted)}.session-health strong{display:block;font-size:22px;margin-top:7px}.session-health small{display:block;font-size:11px;color:var(--muted);line-height:1.7;margin-top:6px}.session-report-counts{display:flex;gap:20px;flex-wrap:wrap;margin:16px 0;font-size:12px;color:#65748d}.session-report-counts strong{color:var(--ink);margin-left:5px;font-size:18px}.session-report-link{display:flex;justify-content:space-between;align-items:center;gap:15px;flex-wrap:wrap;width:100%;text-align:left;color:var(--ink);background:#f6f8fc;border:1px solid var(--line);font-weight:500;padding:13px 15px}.session-report-link:hover{background:#eef3fb}.session-report-link strong{font-size:13px}.session-report-link small{display:block;font-size:11px;color:var(--muted);line-height:1.7;margin-top:5px}.session-detail-search{border-radius:12px}.session-detail-search input[name=q]{min-width:220px}.session-outcomes{margin:12px 0}.session-report-latest{font-size:13px;color:#566780;line-height:1.7}
@media(max-width:1100px){.session-item{grid-template-columns:1.2fr 1fr}.session-meta{grid-template-columns:repeat(2,minmax(0,1fr))}.session-kpis.detail-kpis{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:600px){.session-item{grid-template-columns:1fr;gap:12px;padding:16px}.session-kpis{grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}.session-kpis .card{padding:12px}.session-kpis .card strong{font-size:24px}.session-panel{padding:16px}.session-meta{grid-template-columns:1fr;gap:13px}.session-event{grid-template-columns:1fr 75px;gap:8px;padding:13px 12px}.session-event time{grid-column:1 / -1}.session-list-toolbar form{flex-wrap:wrap}.session-list-toolbar input{flex-basis:100%}.session-list-toolbar select{flex:1}.session-filters{gap:5px}.session-filters a{font-size:11px;padding:7px 9px}}
"""

_KST = timezone(timedelta(hours=9))
_KINDS = {"all": "전체", "start": "세션 시작", "end": "훈련 종료", "sequence": "시퀀스", "query": "질문·답변", "complete": "리포트 요청", "report": "리포트 조회", "other": "기타"}
_STAGES = {"started": "시작 · 입력 대기", "progress": "진행 중 · 미종료", "ended": "훈련 종료"}
_CALL_PAGE_SIZE = 200


def _e(value: object) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def _url(path: str, **params: object) -> str:
    return _e(path + "?" + urlencode(params))


def _time(value: object) -> str:
    if not value:
        return "미기록"
    try:
        return _parse_time(value).astimezone(_KST).strftime("%Y.%m.%d %H:%M:%S KST")
    except (ValueError, OverflowError):
        return "형식 확인 필요"


def _parse_time(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def _duration(started_at: object, ended_at: object, stage: str) -> str:
    try:
        start = _parse_time(started_at) if started_at else None
        end = _parse_time(ended_at) if ended_at else None
    except (ValueError, OverflowError):
        return "시각 확인 필요"
    if end is None:
        return "미기록" if stage == "ended" else "미종료"
    if start is None:
        return "미기록"
    seconds = (end - start).total_seconds()
    if seconds < 0:
        return "시각 확인 필요"
    hours, remaining = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remaining, 60)
    return f"{hours}시간 {minutes:02d}분 {seconds:02d}초" if hours else f"{minutes}분 {seconds:02d}초" if minutes else f"{seconds}초"


def _needle(value: str) -> str:
    return "%" + value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


def _refresh() -> str:
    return f'<div class="session-refresh"><button type="button" data-session-refresh-now>새로고침</button><label><input type="checkbox" data-session-auto-refresh>10초 자동갱신</label><span data-session-refresh-state aria-live="polite">조회 시각 {_time(datetime.now(timezone.utc).isoformat())}</span></div>'


def _mode(value: object) -> str:
    text = str(value or "미지정")
    return f"학습모드 {text}" if text.isdigit() and 1 <= int(text) <= 10 else text


def _schema(conn: sqlite3.Connection) -> dict[str, set[str]]:
    names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    return {name: {row[1] for row in conn.execute(f'PRAGMA table_info("{name}")')} if name in names else set()
            for name in ("sessions", "users", "sequences", "query_turns", "completions", "api_calls")}


def _field(columns: set[str], name: str, alias: str = "s", default: str = "NULL") -> str:
    return f'{alias}."{name}"' if name in columns else default


def _reports_cte(schema: dict[str, set[str]]) -> str:
    columns = schema["completions"]
    if "session_id" not in columns:
        return "reports AS (SELECT NULL AS owner,0 AS n,NULL AS last_at,0 AS complete_n,0 AS pending_n,0 AS failed_n,NULL AS latest_status,NULL AS latest_rowid,NULL AS latest_id,NULL AS latest_at," + ','.join('NULL AS latest_'+name for name in PROCESSING_FIELDS) + " WHERE 0)"
    status = _field(columns, "report_status", "r")
    at = _field(columns, "requested_at", "r")
    report_id = _field(columns, "report_id", "r")
    processing = ','.join(f'{_field(columns,name,"r")} AS {name}' for name in PROCESSING_FIELDS)
    latest_processing = ','.join(f'MAX(CASE WHEN position=1 THEN {name} END) AS latest_{name}' for name in PROCESSING_FIELDS)
    return f"""report_rows AS (SELECT r.session_id AS owner,r.rowid AS record_rowid,
        LOWER(TRIM({status})) AS status,{at} AS at,{report_id} AS report_id,{processing},
        ROW_NUMBER() OVER (PARTITION BY r.session_id ORDER BY julianday({at}) IS NULL,julianday({at}) DESC,r.rowid DESC) AS position
        FROM completions r), reports AS (SELECT owner,COUNT(*) AS n,MAX(datetime(at)) AS last_at,
        SUM(status IN ('complete','completed','success')) AS complete_n,
        SUM(status IN ('pending','processing','running')) AS pending_n,
        SUM(status IN ('failed','failure','error')) AS failed_n,
        MAX(CASE WHEN position=1 THEN status END) AS latest_status,
        MAX(CASE WHEN position=1 THEN record_rowid END) AS latest_rowid,
        MAX(CASE WHEN position=1 THEN report_id END) AS latest_id,
        MAX(CASE WHEN position=1 THEN at END) AS latest_at,{latest_processing} FROM report_rows GROUP BY owner)"""


def _report_state(value: object) -> tuple[str, str]:
    status = str(value or "").strip().lower()
    if status in {"complete", "completed", "success"}:
        return "complete", "처리 완료"
    if status in {"pending", "processing", "running"}:
        return "pending", "분석 대기·진행"
    if status in {"failed", "failure", "error"}:
        return "failed", "처리 실패"
    return "other", f"기타 상태: {status}" if status else "상태 미기록"


def _summary_sql(schema: dict[str, set[str]]) -> str:
    sessions = schema["sessions"]
    def field(name: str) -> str:
        return _field(sessions, name)
    def aggregate(table: str, owner: str, timestamp: str, count: str = "COUNT(*)") -> str:
        columns = schema[table]
        if owner not in columns:
            return "SELECT NULL AS owner, 0 AS n, NULL AS last_at WHERE 0"
        stamp = f'MAX(datetime("{timestamp}"))' if timestamp in columns else "NULL"
        return f'SELECT "{owner}" AS owner, {count} AS n, {stamp} AS last_at FROM "{table}" GROUP BY "{owner}"'
    call_columns = schema["api_calls"]
    if "session_id" in call_columns:
        call_time = 'MAX(datetime("received_at"))' if "received_at" in call_columns else "NULL"
        call_sequence = "SUM(category='sequence')" if "category" in call_columns else "0"
        call_query = "SUM(category='query')" if "category" in call_columns else "0"
        call_accepted = "SUM(category IN ('sequence','query') AND status_code>=200 AND status_code<300)" if {"category", "status_code"} <= call_columns else "0"
        call_summary = f'SELECT session_id AS owner,COUNT(*) AS n,{call_time} AS last_at,{call_sequence} AS sequence_n,{call_query} AS query_n,{call_accepted} AS accepted_n FROM api_calls GROUP BY session_id'
    else:
        call_summary = "SELECT NULL AS owner,0 AS n,NULL AS last_at,0 AS sequence_n,0 AS query_n,0 AS accepted_n WHERE 0"
    seq_count = 'COUNT(DISTINCT "sequence_id")' if "sequence_id" in schema["sequences"] else "COUNT(*)"
    user_join = {"user_id", "name"} <= schema["users"] and "user_id" in sessions
    worker = f"COALESCE(NULLIF(u.name,''),{field('user_id')},'미지정')" if user_join else f"COALESCE({field('user_id')},'미지정')"
    joins = "LEFT JOIN users u ON u.user_id=s.user_id" if user_join else ""
    ended = f"(UPPER(COALESCE({field('session_status')},'')) IN ('ENDED','COMPLETED') OR NULLIF({field('ended_at')},'') IS NOT NULL)"
    last_times = [f"COALESCE(datetime({field(name)}),'')" for name in ("created_at", "started_at", "ended_at")]
    last_times += [f"COALESCE({name}.last_at,'')" for name in ("seq", "qry", "calls", "reports")]
    report_processing = ','.join('reports.latest_'+name for name in PROCESSING_FIELDS)
    return f"""WITH seq AS ({aggregate('sequences','session_key','received_at',seq_count)}),
        qry AS ({aggregate('query_turns','session_key','created_at')}),
        calls AS ({call_summary}),
        {_reports_cte(schema)}, summary AS (
        SELECT s.session_id, {field('user_id')} AS user_id, {worker} AS worker,
            {origin_expression(sessions)} AS creation_origin, {field('creation_path')} AS creation_path,
            {field('scenario_id')} AS scenario_id, {field('session_status')} AS session_status,
            {field('created_at')} AS created_at, {field('started_at')} AS started_at,
            {field('ended_at')} AS ended_at, {field('target_thickness_um')} AS target_thickness_um,
            COALESCE(seq.n,0) AS sequence_count, COALESCE(qry.n,0) AS query_count,
            COALESCE(calls.n,0) AS call_count, COALESCE(reports.n,0) AS report_count,
            COALESCE(reports.complete_n,0) AS report_complete_count,COALESCE(reports.pending_n,0) AS report_pending_count,
            COALESCE(reports.failed_n,0) AS report_failed_count,reports.latest_status AS latest_report_status,
            reports.latest_rowid AS latest_report_rowid,reports.latest_id AS latest_report_id,reports.latest_at AS latest_report_at,
            {report_processing},
            COALESCE(calls.sequence_n,0) AS call_sequence_count, COALESCE(calls.query_n,0) AS call_query_count,
            CASE WHEN {ended} THEN 'ended' WHEN COALESCE(seq.n,0)+COALESCE(qry.n,0)+COALESCE(calls.accepted_n,0)>0 THEN 'progress' ELSE 'started' END AS stage,
            MAX({','.join(last_times)}) AS last_at
        FROM sessions s LEFT JOIN seq ON seq.owner=s.session_id LEFT JOIN qry ON qry.owner=s.session_id
        LEFT JOIN calls ON calls.owner=s.session_id LEFT JOIN reports ON reports.owner=s.session_id {joins}) """


def _summary(conn: sqlite3.Connection, schema: dict[str, set[str]], session_id: str) -> sqlite3.Row | None:
    if "session_id" not in schema["sessions"]:
        return None
    return conn.execute(_summary_sql(schema) + "SELECT * FROM summary WHERE session_id=?", (session_id,)).fetchone()


def _kpi(key: str, label: str, value: int, note: str = "") -> str:
    return f'<div class="card"><span>{label}</span><strong data-stat="{key}">{value:,}</strong><p class="session-note">{note}</p></div>'


def render_session_list(conn: sqlite3.Connection, *, q: str = "", state: str = "all", page: int = 1, limit: int = 25, origin: str = "all") -> str:
    """Render safe, paginated summaries without selecting stored request bodies."""
    schema = _schema(conn)
    origin = normalize_origin(origin)
    state = state if state in {"all", "unfinished", "ended"} else "all"
    limit, page = max(1, min(200, int(limit))), max(1, int(page))
    counts = {"sessions": 0, "unfinished": 0, "ended": 0}
    total, rows = 0, []
    pages = 1
    if "session_id" in schema["sessions"]:
        sql = _summary_sql(schema)
        scope = " WHERE creation_origin='"+origin+"'" if origin != "all" else ""
        counts["sessions"], counts["ended"] = conn.execute(sql + "SELECT COUNT(*),COALESCE(SUM(stage='ended'),0) FROM summary" + scope).fetchone()
        counts["unfinished"] = counts["sessions"] - counts["ended"]
        conditions, params = [], []
        if origin != "all":
            conditions.append("creation_origin=?")
            params.append(origin)
        if state != "all":
            conditions.append("stage='ended'" if state == "ended" else "stage<>'ended'")
        if q:
            learning_mode = re.fullmatch(r"학습\s*모드\s*(10|[1-9])", q.strip())
            if learning_mode:
                conditions.append("scenario_id=?")
                params.append(learning_mode.group(1))
            else:
                conditions.append("(" + " OR ".join(f"COALESCE({col},'') LIKE ? ESCAPE '\\'" for col in ("session_id", "user_id", "worker", "scenario_id")) + ")")
                params.extend([_needle(q)] * 4)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        total = conn.execute(sql + "SELECT COUNT(*) FROM summary" + where, params).fetchone()[0]
        pages = max(1, math.ceil(total / limit))
        page = min(page, pages)
        rows = conn.execute(sql + "SELECT * FROM summary" + where + " ORDER BY last_at DESC,session_id DESC LIMIT ? OFFSET ?", [*params, limit, (page-1)*limit]).fetchall()
    page = min(page, pages)
    cards = "".join(_kpi(key, label, counts[key]) for key, label in (("sessions", "전체 훈련 세션"), ("unfinished", "미종료 세션"), ("ended", "훈련 종료")))
    tabs = "".join(f'<a class="{"active" if key == state else ""}" href="{_url("/admin/sessions",q=q,state=key,limit=limit)}">{label}</a>' for key, label in (("all", "전체"), ("unfinished", "미종료"), ("ended", "훈련 종료")))
    items = []
    for row in rows:
        provenance = record_provenance(conn, "sessions", row)
        ended = row["stage"] == "ended"
        stamp = f'<span class="session-ended-at">{_time(row["ended_at"])}</span>' if ended else '<small>현재 접속 여부와 무관한 저장 상태</small>'
        report_kind, report_label = _report_state(row["latest_report_status"])
        if row['report_count']:
            report_label = _latest_report_progress(row)['label']
        report_badge = f'<span class="session-report-badge {report_kind}">리포트 {_e(report_label)}</span>' if row["report_count"] else '<small>리포트 기록 없음</small>'
        items.append(f'''<a class="session-item {"ended" if ended else "unfinished"}" href="{_url("/admin/sessions/view",session_id=row["session_id"])}">
            <div><strong>{_e(row['session_id'])}</strong><div class="session-counts">저장 시퀀스 {row['sequence_count']:,} · 질문 {row['query_count']:,}</div><small>시퀀스 호출 {row['call_sequence_count']:,} · 쿼리 호출 {row['call_query_count']:,} (수집 이후)</small><small>전체 실제 호출 {row['call_count']:,}건 · 수집 이후</small></div>
            <div><strong>{_e(row['worker'])}</strong><small>{_e(row['user_id'] or '작업자 ID 미기록')}</small><small>{_e(_mode(row['scenario_id']))} · {ORIGINS[row['creation_origin']]}</small><small>생성 호출 {_e(provenance['caller_label'])} · IP {_e(provenance['client_ip'])}</small></div>
            <div><span class="session-state">{_STAGES[row['stage']]}</span>{stamp}{report_badge}</div>
            <div><small>훈련 시작 {_time(row['started_at'])}</small><small>마지막 활동 {_time(row['last_at'])}</small></div></a>''')
    empty = "조건에 맞는 세션이 없습니다." if q or state != "all" else "아직 등록된 훈련 세션이 없습니다."
    rendered = "".join(items) or f'<div class="session-panel empty">{empty}</div>'
    def page_link(number: int, label: str) -> str:
        return f'<a href="{_url("/admin/sessions",q=q,state=state,page=number,limit=limit)}">{label}</a>'
    previous = page_link(page-1, "← 이전") if page > 1 else "<span></span>"
    following = page_link(page+1, "다음 →") if page < pages else "<span></span>"
    options = "".join(f'<option value="{size}"{" selected" if size == limit else ""}>{size}개씩</option>' for size in sorted({10,25,50,100,200,limit}))
    return preserve_origin_links(f'''<div class="topbar"><div><div class="eyebrow">TRAINING SESSIONS</div><h1>훈련 세션</h1><div class="sub">세션별 진행 상태와 저장 기록을 확인하세요.</div></div><span class="readonly"><i class="dot"></i>읽기 전용</span></div>
        {origin_control(conn,origin,q=q,state=state,limit=limit)}{_refresh()}<div class="cards session-kpis">{cards}</div><nav class="session-filters" aria-label="훈련 상태 필터">{tabs}</nav>
        <div class="toolbar session-list-toolbar"><form action="/admin/sessions" method="get"><input type="hidden" name="state" value="{state}"><label for="session-search">세션 검색</label><input id="session-search" name="q" value="{_e(q)}" placeholder="세션 ID · 작업자 이름/ID · 학습모드"><select name="limit" aria-label="페이지당 세션">{options}</select><button type="submit">검색</button></form><a class="reset" href="/admin/sessions">초기화</a></div>
        <p class="session-search-result">검색 결과 전체 {total:,}건 · 모든 시각은 한국 시간 (KST)</p><div class="session-list">{rendered}</div><div class="pagination session-list-page">{previous}<span>{page} / {pages} 페이지</span>{following}</div>
        <p class="session-note">회색 배경은 미종료 세션입니다. 저장 시퀀스는 세션 내 고유 시퀀스 수이며, 실제 API 호출 수는 호출 수집 이후의 재시도·실패를 포함합니다.</p>''',origin)


def _call_match(schema: dict[str, set[str]], category: str, extra: str = "") -> str:
    cols = schema["api_calls"]
    if not {"session_id", "category", "status_code"} <= cols:
        return "0"
    return f"EXISTS (SELECT 1 FROM api_calls c WHERE c.session_id=s.session_id AND c.category='{category}' AND c.status_code>=200 AND c.status_code<300 {extra})"


def _call_json_value(schema: dict[str, set[str]], column: str, *paths: str) -> str:
    if column not in schema["api_calls"]:
        return "NULL"
    values = ",".join(f"json_extract(c.{column},'{path}')" for path in paths)
    value = f"COALESCE({values})" if len(paths) > 1 else values
    return f"(CASE WHEN json_valid(c.{column}) THEN {value} ELSE NULL END)"


def _session_time_match(schema: dict[str, set[str]], field: str) -> str:
    response = _call_json_value(schema, "response_json", f"$.data.{field}", f"$.{field}")
    request = _call_json_value(schema, "request_json", f"$.{field}")
    return f"AND COALESCE({response},{request})=s.{field}"


def _legacy_sql(schema: dict[str, set[str]], kind: str) -> str:
    """A bounded projection of storage events; these never claim HTTP outcomes."""
    parts = []
    session = schema["sessions"]
    def add(category: str, sql: str) -> None:
        if kind in {"all", category}:
            parts.append(sql)
    if "started_at" in session:
        add("start", f"SELECT 'start' AS event_key,'start' AS category,s.started_at AS at,'훈련 시작 기록' AS summary FROM sessions s WHERE s.session_id=? AND NULLIF(s.started_at,'') IS NOT NULL AND NOT {_call_match(schema,'start',_session_time_match(schema,'started_at'))}")
    if "ended_at" in session or "session_status" in session:
        ended = _field(session, "ended_at")
        status = _field(session, "session_status")
        match = _call_match(schema, "end", _session_time_match(schema, "ended_at")) if "ended_at" in session else "0"
        add("end", f"SELECT 'end','end',{ended},'훈련 종료 기록' FROM sessions s WHERE s.session_id=? AND (NULLIF({ended},'') IS NOT NULL OR UPPER(COALESCE({status},'')) IN ('ENDED','COMPLETED')) AND NOT {match}")
    seq = schema["sequences"]
    if {"session_key", "sequence_id"} <= seq:
        at = _field(seq, "received_at", "r")
        match = _call_match(schema, "sequence", "AND c.sequence_id=r.sequence_id") if "sequence_id" in schema["api_calls"] else "0"
        add("sequence", f"SELECT 'sequence:' || r.sequence_id,'sequence',{at},'시퀀스 ' || substr(r.sequence_id,1,120) FROM sequences r JOIN sessions s ON s.session_id=r.session_key WHERE s.session_id=? AND NOT {match}")
    qry = schema["query_turns"]
    if "session_key" in qry:
        row_id = _field(qry, "id", "r", "r.rowid")
        at = _field(qry, "created_at", "r")
        preview = "substr(r.prompt,1,100)" if "prompt" in qry else "'저장된 질문·답변'"
        # Prompt equality cannot identify a particular saved turn, even when a
        # captured request succeeded. Keep every legacy turn in its own section.
        add("query", f"SELECT 'query:' || {row_id},'query',{at},{preview} FROM query_turns r JOIN sessions s ON s.session_id=r.session_key WHERE s.session_id=?")
    reports = schema["completions"]
    if "session_id" in reports:
        row_id = _field(reports, "report_id", "r", "r.rowid")
        at = _field(reports, "requested_at", "r")
        match = "0"
        if "response_json" in schema["api_calls"] and "report_id" in reports:
            report_id = _call_json_value(schema, "response_json", "$.data.report_id", "$.report_id")
            match = _call_match(schema, "complete", f"AND {report_id}=r.report_id")
        add("complete", f"SELECT 'completion:' || {row_id},'complete',{at},'리포트 저장 기록' FROM completions r JOIN sessions s ON s.session_id=r.session_id WHERE s.session_id=? AND NOT {match}")
    if not parts:
        return "SELECT NULL AS event_key,NULL AS category,NULL AS at,NULL AS summary WHERE 0"
    return "WITH legacy_events(event_key,category,at,summary) AS (" + " UNION ALL ".join(parts) + ") SELECT event_key,category,at,summary FROM legacy_events"


def _event_button(session_id: str, key: str, at: object, title: object, summary: object, origin: str, status: object = None) -> str:
    status_html = ""
    if status is not None:
        error = " error" if str(status).isdigit() and int(status) >= 400 else ""
        status_html = f'<span class="event-status{error}">HTTP {_e(status)}</span>'
    return f'<button type="button" class="session-event" data-detail-url="{_url("/admin/api/session-event",session_id=session_id,event_key=key)}"><time>{_time(at)}</time><span><strong>{_e(title)}</strong><small>{_e(summary)}</small></span><span class="event-origin">{origin}{status_html}<small>상세 보기 ↗</small></span></button>'


def _call_health(conn: sqlite3.Connection, schema: dict[str, set[str]], session_id: str) -> str:
    columns = schema["api_calls"]
    status_count, errors, duration_count, average = 0, 0, 0, None
    if "session_id" in columns:
        known_status = "typeof(status_code) IN ('integer','real')" if "status_code" in columns else "0"
        valid_duration = "typeof(duration_ms) IN ('integer','real') AND duration_ms>=0" if "duration_ms" in columns else "0"
        error_sql = f"SUM(CASE WHEN {known_status} AND status_code>=400 THEN 1 ELSE 0 END)" if "status_code" in columns else "0"
        average_sql = f"AVG(CASE WHEN {valid_duration} THEN duration_ms END)" if "duration_ms" in columns else "NULL"
        status_count, errors, duration_count, average = conn.execute(
            f"SELECT SUM(CASE WHEN {known_status} THEN 1 ELSE 0 END),{error_sql},SUM(CASE WHEN {valid_duration} THEN 1 ELSE 0 END),{average_sql} FROM api_calls WHERE session_id=?",
            (session_id,),
        ).fetchone()
    error_value = f'<strong data-stat="api_errors">{errors:,}</strong><small>상태 기록 {status_count:,}건 기준 · HTTP 400 이상</small>' if status_count else '<strong>미기록</strong><small>응답 상태 미기록</small>'
    duration_value = f'<strong data-stat="api_avg_duration">{average:,.1f} ms</strong><small>소요 시간 기록 {duration_count:,}건 기준</small>' if average is not None and math.isfinite(average) else '<strong>미기록</strong><small>소요 시간 미기록</small>'
    return f'<div class="session-health"><div><span>HTTP 오류 · 세션 전체</span>{error_value}</div><div><span>평균 처리 시간 · 세션 전체</span>{duration_value}</div></div>'


def _latest_report_progress(row: sqlite3.Row) -> dict:
    return report_progress({'report_status':row['latest_report_status'],'requested_at':row['latest_report_at'],**{name:row['latest_'+name] for name in PROCESSING_FIELDS}})


def _report_panel(row: sqlite3.Row) -> str:
    counts = "".join(f'<span>{label}<strong data-stat="reports_{key}">{row[column]:,}</strong></span>' for key, column, label in (
        ("complete", "report_complete_count", "처리 완료"), ("pending", "report_pending_count", "분석 대기·진행"), ("failed", "report_failed_count", "처리 실패")))
    other = row["report_count"] - row["report_complete_count"] - row["report_pending_count"] - row["report_failed_count"]
    if other:
        counts += f'<span>기타·미기록<strong>{other:,}</strong></span>'
    state, label = _report_state(row["latest_report_status"])
    progress = _latest_report_progress(row)
    if row["latest_report_rowid"] is not None:
        latest = f'''<button type="button" class="session-report-link" data-detail-url="{_url('/admin/api/record',table='completions',rowid=row['latest_report_rowid'])}"><span><strong>최근 리포트 · {_e(row['latest_report_id'] or 'ID 미기록')}</strong><small>요청 시각 {_time(row['latest_report_at'])} · 저장 기록 상세 보기 ↗</small><small>대기 경과 {_e(progress['elapsed'])} · 최근 처리 {_time(progress['latest_at'])}<br>상태 사유 {_e(progress['reason'])}</small></span><span class="session-report-badge {state}">{_e(progress['label'])}</span></button>'''
    else:
        latest = '<p class="session-report-latest">아직 저장된 리포트가 없습니다.</p>'
    return f'<section class="session-panel"><h2>리포트 처리 상태</h2><p class="session-note">이 세션에 저장된 리포트의 현재 상태 · 세션 전체 {row["report_count"]:,}건</p><div class="session-report-counts">{counts}</div>{latest}<p class="session-note">요청 후 24시간 이상은 장기 대기로 표시하며, 실패나 현재 실행 여부를 확정하지 않습니다. 최근 처리는 서버가 기록한 작업 시각만 표시합니다. 리포트 처리 상태와 분석 엔진의 이용 가능 여부는 별개입니다. 상세 보기는 저장된 DB 기록을 읽습니다.</p></section>'


def render_session_detail(conn: sqlite3.Connection, session_id: str, *, kind: str = "all", call_page: int = 1, q: str = "", outcome: str = "all", order: str = "newest", origin: str = "all") -> str | None:
    """Render one session with independently bounded captured and legacy lists."""
    schema = _schema(conn)
    row = _summary(conn, schema, session_id)
    if row is None:
        return None
    kind = kind if kind in _KINDS else "all"
    outcome = outcome if outcome in {"all", "errors"} else "all"
    order = order if order in {"newest", "oldest"} else "newest"
    direction = "DESC" if order == "newest" else "ASC"
    call_page = max(1, int(call_page))
    call_cols = schema["api_calls"]
    calls, call_total, category_counts = [], 0, {}
    call_where, call_params = " WHERE session_id=?", [session_id]
    if kind != "all":
        call_where += " AND category=?" if "category" in call_cols else " AND 0"
        if "category" in call_cols:
            call_params.append(kind)
    if outcome == "errors":
        call_where += " AND typeof(status_code) IN ('integer','real') AND status_code>=400" if "status_code" in call_cols else " AND 0"
    if q:
        search_columns = [name for name in ("path", "request_id", "sequence_id", "category", "prompt") if name in call_cols]
        call_where += " AND (" + " OR ".join(f'COALESCE("{name}",\'\') LIKE ? ESCAPE \'\\\'' for name in search_columns) + ")" if search_columns else " AND 0"
        call_params.extend([_needle(q)] * len(search_columns))
    if {"id", "session_id"} <= call_cols:
        call_total = conn.execute("SELECT COUNT(*) FROM api_calls" + call_where, call_params).fetchone()[0]
        if "category" in call_cols:
            category_counts = dict(conn.execute("SELECT category,COUNT(*) FROM api_calls WHERE session_id=? GROUP BY category", (session_id,)).fetchall())
    legacy_sql = _legacy_sql(schema, kind) if outcome == "all" else "SELECT NULL AS event_key,NULL AS category,NULL AS at,NULL AS summary WHERE 0"
    legacy_params = [session_id] * legacy_sql.count("?")
    legacy_where = ""
    if q:
        legacy_where = " WHERE (COALESCE(summary,'') LIKE ? ESCAPE '\\' OR COALESCE(event_key,'') LIKE ? ESCAPE '\\' OR COALESCE(category,'') LIKE ? ESCAPE '\\')"
        legacy_params.extend([_needle(q)] * 3)
    legacy_total = conn.execute("SELECT COUNT(*) FROM (" + legacy_sql + ")" + legacy_where, legacy_params).fetchone()[0]
    pages = max(1, math.ceil(max(call_total, legacy_total) / _CALL_PAGE_SIZE))
    call_page = min(call_page, pages)
    offset = (call_page - 1) * _CALL_PAGE_SIZE
    if {"id", "session_id"} <= call_cols:
        names = ("id", "method", "path", "category", "source", "received_at", "status_code", "duration_ms", "sequence_id", "request_id", "client_ip", "caller_source")
        projection = ",".join(f'"{name}"' if name in call_cols else f'NULL AS "{name}"' for name in names)
        ordering = f'julianday(received_at) IS NULL,julianday(received_at) {direction},id {direction}' if "received_at" in call_cols else f"id {direction}"
        calls = conn.execute(f"SELECT {projection} FROM api_calls" + call_where + f" ORDER BY {ordering} LIMIT ? OFFSET ?", [*call_params, _CALL_PAGE_SIZE, offset]).fetchall()
    legacy = conn.execute("SELECT * FROM (" + legacy_sql + ")" + legacy_where + f" ORDER BY julianday(at) IS NULL,julianday(at) {direction},event_key {direction} LIMIT ? OFFSET ?", [*legacy_params, _CALL_PAGE_SIZE, offset]).fetchall()
    call_items = []
    for call in calls:
        provenance = record_provenance(conn, "api_calls", call)
        elapsed = f" · {call['duration_ms']:g} ms" if isinstance(call["duration_ms"], (int, float)) else ""
        summary = _KINDS.get(call["category"], "기타") + elapsed
        if call["sequence_id"]:
            summary += " · " + str(call["sequence_id"])
        if call["request_id"]:
            summary += " · 요청 " + str(call["request_id"])
        if call["source"] == "test":
            summary += " · 테스트"
        summary += " · " + provenance['caller_label'] + " · IP " + provenance['client_ip']
        call_items.append(_event_button(session_id, f"call:{call['id']}", call["received_at"], f"{call['method'] or ''} {call['path'] or '경로 미기록'}", summary, "실제 호출", call["status_code"]))
    legacy_items = [_event_button(session_id, event["event_key"], event["at"], _KINDS.get(event["category"], "저장 기록"), event["summary"], "기존 저장 기록") for event in legacy]
    if not call_items:
        message = "수집된 실제 호출이 없습니다." if row["call_count"] == 0 else "필터 조건에 맞는 실제 호출이 없습니다." if call_total == 0 else "이 페이지의 실제 호출이 없습니다. 이전 페이지에서 확인하세요."
        call_items = [f'<div class="empty">{message}</div>']
    if not legacy_items:
        message = "기존 저장 기록은 HTTP 상태가 없어 오류 필터에서 제외됩니다." if outcome == "errors" else "이 페이지에 표시할 기존 저장 기록이 없습니다."
        legacy_items = [f'<div class="empty">{message}</div>']
    cards = "".join(_kpi(key, label, amount, note) for key, label, amount, note in (
        ("sequences", "저장 시퀀스", row["sequence_count"], f"세션 내 고유 시퀀스 · 실제 시퀀스 호출 {row['call_sequence_count']:,}건 (수집 이후)"),
        ("queries", "저장 질문·답변", row["query_count"], f"DB에 저장된 대화 기록 · 실제 쿼리 호출 {row['call_query_count']:,}건 (수집 이후)"),
        ("api_calls", "실제 API 호출", row["call_count"], "수집 이후 · 재시도·실패 포함"),
        ("reports", "저장 리포트", row["report_count"], "리포트 요청·처리 상태 기록")))
    metadata = (("작업자", f"{row['worker']} ({row['user_id'] or 'ID 미기록'})"), ("훈련 콘텐츠", _mode(row["scenario_id"])),
                ("생성 출처", ORIGINS[row['creation_origin']]), ("생성 경로", row['creation_path'] or '미기록'),
                ("훈련 상태", _STAGES[row["stage"]]), ("생성 시각", _time(row["created_at"])), ("훈련 시작", _time(row["started_at"])),
                ("훈련 종료", _time(row["ended_at"])), ("훈련 소요 시간", _duration(row["started_at"], row["ended_at"], row["stage"])), ("마지막 활동", _time(row["last_at"])),
                ("목표 도막 두께", f"{row['target_thickness_um']:g} μm" if isinstance(row["target_thickness_um"], (int, float)) else "미기록"))
    provenance = record_provenance(conn, "sessions", row)
    metadata += (("생성 호출 채널", provenance['caller_label']), ("생성 호출 IP", provenance['client_ip']))
    meta_html = "".join(f'<div><dt>{label}</dt><dd>{_e(value)}</dd></div>' for label, value in metadata)
    tabs = "".join(f'<a class="{"active" if key == kind else ""}" href="{_url("/admin/sessions/view",session_id=session_id,kind=key,q=q,outcome=outcome,order=order)}">{label}</a>' for key, label in _KINDS.items())
    outcome_tabs = "".join(f'<a class="{"active" if key == outcome else ""}" href="{_url("/admin/sessions/view",session_id=session_id,kind=kind,q=q,outcome=key,order=order)}">{label}</a>' for key, label in (("all", "모든 응답"), ("errors", "HTTP 오류만")))
    order_options = "".join(f'<option value="{key}"{" selected" if key == order else ""}>{label}</option>' for key, label in (("newest", "최신 수신순"), ("oldest", "오래된 수신순")))
    search_form = f'''<div class="toolbar session-detail-search"><form method="get" action="/admin/sessions/view"><input type="hidden" name="session_id" value="{_e(session_id)}"><input type="hidden" name="kind" value="{kind}"><input type="hidden" name="outcome" value="{outcome}"><label for="session-event-search">기록 검색</label><input id="session-event-search" name="q" value="{_e(q)}" placeholder="경로 · 요청 ID · 시퀀스 ID · 종류"><select name="order" aria-label="기록 정렬">{order_options}</select><button type="submit">조회</button></form><a class="reset" href="{_url('/admin/sessions/view',session_id=session_id)}">초기화</a></div><p class="session-note">기존 저장 기록은 표시되는 요약과 질문도 검색합니다. HTTP 오류 필터는 상태 코드가 기록된 실제 호출에만 적용됩니다.</p>'''
    breakdown = "".join(f'<span>{label} <strong>{category_counts.get(key,0):,}</strong></span>' for key, label in _KINDS.items() if key != "all")
    def nav(number: int, label: str) -> str:
        return f'<a href="{_url("/admin/sessions/view",session_id=session_id,kind=kind,call_page=number,q=q,outcome=outcome,order=order)}">{label}</a>'
    pagination = f'<div class="session-event-page">{nav(call_page-1,"← 이전 기록") if call_page>1 else "<span></span>"}<span>{call_page} / {pages} 페이지 · 목록별 최대 200건</span>{nav(call_page+1,"다음 기록 →") if call_page<pages else "<span></span>"}</div>'
    return preserve_origin_links(f'''<a class="session-back" href="/admin/sessions">← 훈련 세션 목록</a><div class="topbar"><div><div class="eyebrow">SESSION DASHBOARD</div><h1>{_e(session_id)}</h1><div class="sub">세션 요약과 API 기록 · 모든 시각은 한국 시간 (KST)</div></div><span class="readonly"><i class="dot"></i>읽기 전용</span></div>
        {_refresh()}<div class="cards session-kpis detail-kpis">{cards}</div><section class="session-panel"><h2>세션 정보</h2><dl class="session-meta">{meta_html}</dl><p class="session-note">생성 시각은 서버에서 새로 생성한 세션부터 기록됩니다. 훈련 소요 시간은 저장된 훈련 시작·종료 시각의 차이이며, 미종료 세션은 계산하지 않습니다.</p></section>
        {_report_panel(row)}<nav class="session-filters" aria-label="API 기록 종류">{tabs}</nav>{search_form}<nav class="session-filters session-outcomes" aria-label="응답 결과 필터">{outcome_tabs}</nav><section class="session-panel"><h2>실제 호출 <small>· 수집 이후 세션 전체 {row['call_count']:,}건</small></h2><p class="session-note">수집 이후 실제로 처리한 요청·응답입니다. 수집 시작 전의 호출은 포함되지 않습니다. 아래 목록의 필터 결과 {call_total:,}건.</p>{_call_health(conn,schema,session_id)}<p class="session-note">종류별 호출 수 · 세션 전체 · 수집 이후</p><div class="session-call-counts">{breakdown}</div><div class="session-events">{''.join(call_items)}</div></section>
        <section class="session-panel"><h2>기존 저장 기록 <small>· 필터 결과 {legacy_total:,}건</small></h2><p class="session-note">DB에 남은 훈련 시작·종료, 시퀀스, 질문·답변, 리포트 기록입니다. 당시 HTTP 응답과 호출 성공 여부는 알 수 없습니다. 실제 호출과 명확히 연결되는 저장 기록은 중복 표시하지 않습니다.</p><div class="session-events">{''.join(legacy_items)}</div>{pagination}</section>''',origin)


def _json(value: object) -> object:
    if value is None:
        return None
    try:
        return json.loads(str(value))
    except (ValueError, TypeError):
        return str(value)


def _metadata(data: dict, labels: tuple[tuple[str, str], ...]) -> list[dict]:
    return [{"label": label, "value": _time(data[key]) if key.endswith("_at") else data[key]}
            for key, label in labels if key in data and data[key] is not None]


def session_event_detail(conn: sqlite3.Connection, session_id: str, event_key: str) -> dict | None:
    """Fetch a single payload only after verifying its owning session."""
    schema = _schema(conn)
    row = _summary(conn, schema, session_id)
    if row is None:
        return None
    if event_key.startswith("call:"):
        identifier = event_key[5:]
        if not identifier.isascii() or not identifier.isdigit() or not {"id", "session_id"} <= schema["api_calls"]:
            return None
        call = conn.execute("SELECT * FROM api_calls WHERE session_id=? AND id=?", (session_id, identifier)).fetchone()
        if call is None:
            return None
        data = dict(call)
        meta = _metadata(data, (("session_id", "세션 ID"), ("request_id", "요청 ID"), ("user_id", "작업자 ID"), ("scenario_id", "학습모드"),
            ("category", "종류"), ("source", "수집 출처"), ("sequence_id", "시퀀스 ID"), ("received_at", "요청 수신"), ("finished_at", "처리 종료"),
            ("status_code", "HTTP 상태"), ("duration_ms", "처리 시간 (ms)"), ("request_bytes", "요청 크기 (bytes)"), ("response_bytes", "응답 크기 (bytes)"), ("content_type", "Content-Type")))
        meta.extend(provenance_meta(record_provenance(conn, "api_calls", data)))
        sections = [{"title": label, "data": _json(data.get(key))} for key, label in (("request_json", "요청 데이터"), ("response_json", "응답 데이터"))]
        notice = "실제 호출 · 수집 이후 기록"
        if data.get("request_truncated") or data.get("response_truncated"):
            notice += " · 크기 제한으로 데이터 잘림: " + "·".join(label for key, label in (("request_truncated", "요청"), ("response_truncated", "응답")) if data.get(key))
        return {"title": f"{data.get('method') or ''} {data.get('path') or 'API 호출'}", "meta": meta, "sections": sections, "notice": notice}
    notice = "기존 저장 기록입니다. 당시 HTTP 상태·요청·응답은 기록되지 않았으며, 저장된 데이터만 표시합니다."
    if event_key in {"start", "end"}:
        is_start = event_key == "start"
        if (is_start and not row["started_at"]) or (not is_start and row["stage"] != "ended"):
            return None
        data = {key: row[key] for key in ("session_id", "user_id", "scenario_id", "session_status", "created_at", "started_at", "ended_at", "target_thickness_um")}
        # End events are not the session's creation call.
        provenance = record_provenance(conn, "sessions", row) if is_start else record_provenance(conn, "unknown", {})
        return {"title": "훈련 시작 저장 기록" if is_start else "훈련 종료 저장 기록", "meta": [{"label": "세션 ID", "value": session_id}] + provenance_meta(provenance), "sections": [{"title": "세션 저장 데이터", "data": data}], "notice": notice}
    prefix, separator, identifier = event_key.partition(":")
    if not separator or not identifier:
        return None
    mapping = {"sequence": ("sequences", "session_key", "sequence_id", "시퀀스 저장 기록"),
               "query": ("query_turns", "session_key", "id", "질문·답변 저장 기록"),
               "completion": ("completions", "session_id", "report_id", "리포트 저장 기록")}
    if prefix not in mapping:
        return None
    table, owner, key, title = mapping[prefix]
    cols = schema[table]
    if owner not in cols or (prefix == "sequence" and key not in cols):
        return None
    key_sql = f'"{key}"' if key in cols else "rowid"
    stored = conn.execute(f'SELECT * FROM "{table}" WHERE "{owner}"=? AND {key_sql}=?', (session_id, identifier)).fetchone()
    if stored is None:
        return None
    data = dict(stored)
    provenance = record_provenance(conn, table, data)
    sections = []
    payload_key = "body" if prefix == "sequence" else "payload" if prefix == "completion" else None
    if payload_key in data:
        sections.append({"title": "저장된 시퀀스 데이터" if prefix == "sequence" else "저장된 리포트 데이터", "data": _json(data.pop(payload_key))})
    sections.insert(0, {"title": "저장 기록", "data": data})
    return {"title": title, "meta": [{"label": "세션 ID", "value": session_id}] + provenance_meta(provenance), "sections": sections, "notice": notice}
