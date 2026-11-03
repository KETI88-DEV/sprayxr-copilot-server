"""Read-only operational overview of persisted records, without loading raw telemetry."""
from __future__ import annotations

import html
import math
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode

KST = timezone(timedelta(hours=9))
METRICS = {
    "sessions": ("훈련 세션", "sessions", "started_at", "훈련 시작일"),
    "sequences": ("저장 시퀀스", "sequences", "received_at", "서버 수신일"),
    "queries": ("저장 질문", "query_turns", "created_at", "답변 생성일"),
}

DASHBOARD_CSS = """
.home-kpis{grid-template-columns:repeat(4,minmax(0,1fr));gap:16px;margin:0 0 22px}.kpi{text-decoration:none;color:var(--ink);position:relative;transition:transform .15s,border-color .15s}
.kpi:hover{transform:translateY(-2px);border-color:#8c9fe9}.kpi .kpi-top{display:flex;justify-content:space-between;align-items:center;font-size:13px;color:#53627a;font-weight:650}
.kpi strong{font-size:34px;letter-spacing:-.04em;font-variant-numeric:tabular-nums;margin:9px 0 5px}.kpi p{font-size:12px;color:var(--muted);margin:0}
.kpi-icon{width:32px;height:32px;display:grid;place-items:center;background:#edf1ff;border-radius:10px;color:#3157d5;font-size:15px;font-weight:800}
.home-meta{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:12px 0 0;font-size:12px;color:var(--muted)}.scope-pill{background:#e9eefb;color:#405995;border-radius:6px;padding:4px 8px;font-weight:650}
.home-actions{display:flex;gap:8px;flex-wrap:wrap}.home-actions a.secondary{background:white;color:var(--brand);border:1px solid var(--line)}
.dashboard-grid{display:grid;grid-template-columns:minmax(0,1.65fr) minmax(280px,1fr);gap:20px;margin:0 0 20px}.dashboard-panel{background:#fff;border:1px solid var(--line);border-radius:14px;padding:22px;min-width:0;box-shadow:var(--shadow)}
.panel-head{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;margin-bottom:18px}.panel-head h2{font-size:16px;margin:0 0 7px;letter-spacing:-.02em}.panel-head p,.panel-note{font-size:12px;color:var(--muted);line-height:1.7;margin:0}
.period-form{display:flex;align-items:center;gap:6px;flex-shrink:0}.period-form select{background:#f7f9fd;border:1px solid var(--line);border-radius:7px;padding:7px;color:var(--ink);font:inherit;font-size:12px}.period-form button{font-size:12px;padding:8px 10px}
.metric-tabs{display:flex;gap:5px;flex-wrap:wrap;margin-bottom:15px}.metric-tabs a{font-size:12px;font-weight:650;padding:7px 12px;border-radius:7px;color:#65748d;text-decoration:none;background:#f2f5fa}.metric-tabs a.active{background:#3157d5;color:#fff}
.trend-meta{display:flex;align-items:baseline;gap:8px;margin-bottom:8px}.trend-meta strong{font-size:25px;font-variant-numeric:tabular-nums}.trend-meta span{color:var(--muted);font-size:12px}
.chart-scroll{overflow-x:auto}.activity-chart{display:block;width:100%;height:auto;min-height:190px;overflow:visible}.chart-data{margin-top:10px;font-size:12px}.chart-data table{margin-top:10px}.chart-data th,.chart-data td{padding:6px 8px}.chart-data .chart-data-scroll{max-height:180px;overflow:auto}
.rank-list{display:grid;gap:17px}.rank-line{display:flex;justify-content:space-between;gap:10px;font-size:12px;margin-bottom:6px}.rank-line span{min-width:0;overflow-wrap:anywhere}.rank-line strong{font-size:12px;color:#40516e;white-space:nowrap}.rank-track{height:9px;background:#edf1f8;border-radius:20px;overflow:hidden}.rank-fill{height:100%;background:linear-gradient(90deg,#5478ed,#91a7f4);border-radius:20px}
.rank-list .other{font-size:12px;color:var(--muted);margin:0}.home-empty{display:flex;align-items:center;justify-content:center;text-align:center;min-height:150px;color:var(--muted);font-size:13px;line-height:1.8;padding:18px}
.status-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.state-box{background:#f8faff;border:1px solid #e7edf7;border-radius:10px;padding:14px 12px}.state-box strong{display:block;font-size:25px;margin:8px 0;font-variant-numeric:tabular-nums}.state-box p{font-size:11px;color:var(--muted);line-height:1.6;margin:0}
.state-badge{display:inline-block;border-radius:6px;padding:4px 7px;font-size:11px;font-weight:700;white-space:nowrap;background:#eff3f8;color:#65748d}.state-badge.started,.state-badge.pending{background:#fff2d5;color:#926312}.state-badge.progress{background:#e7efff;color:#3157d5}.state-badge.ended,.state-badge.complete{background:#e4f5ed;color:#087f5b}.state-badge.failed{background:#ffe8e8;color:#b64040}
.report-list{display:grid;gap:12px}.report-row{display:flex;align-items:center;justify-content:space-between;gap:12px;font-size:13px}.report-row strong{font-size:20px;font-variant-numeric:tabular-nums}.panel-footer{font-size:11px;color:var(--muted);line-height:1.7;margin:14px 0 0}
.recent-panel{padding:0;overflow:hidden}.recent-panel .panel-head{padding:20px 22px;margin:0}.text-link{color:var(--brand);text-decoration:none;font-size:12px;white-space:nowrap;font-weight:650}.recent-scroll{overflow:auto}.recent-table{min-width:800px}.recent-table th{position:static;font-size:12px}.recent-table td{font-size:12px;vertical-align:middle}.recent-table .session-link{color:#3157d5;text-decoration:none;font-weight:700}.recent-table .time-cell{white-space:nowrap;color:#65748d;font-size:11px}.recent-table .identity small{display:block;color:var(--muted);margin-top:4px}.recent-note{padding:12px 22px;margin:0;border-top:1px solid var(--line)}
.home-footnote{margin:20px 0 0;padding:14px 17px;border:1px solid var(--line);background:#edf2fa;border-radius:11px;font-size:12px;color:#61718a;line-height:1.8}.home-footnote strong{color:#40516e}.table-name small{display:block;color:#99afd5;font-size:10px;margin-top:3px}.home-nav{font-weight:750}.home-nav svg{width:17px;height:17px;vertical-align:-3px;margin-right:7px}
@media(min-width:851px) and (max-width:1150px){.dashboard-grid{grid-template-columns:1fr}.home-kpis{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:850px){.dashboard-grid{grid-template-columns:1fr}.home-kpis{grid-template-columns:repeat(2,minmax(0,1fr))}.dashboard-panel{padding:17px}.sidebar .table-link{display:inline-flex;margin-right:5px}.sidebar .table-link .table-name small{display:none}.sidebar .side-title{margin:15px 10px 5px}.sidebar .brand small{margin-bottom:10px}.panel-head{flex-wrap:wrap}}
@media(max-width:480px){.main{padding:15px}.home-kpis{gap:10px}.kpi{padding:14px}.kpi strong{font-size:29px}.activity-chart{min-width:600px}.status-grid{grid-template-columns:1fr}.state-box{display:grid;grid-template-columns:95px 40px 1fr;align-items:center;gap:8px;padding:10px}.state-box strong{margin:0;font-size:22px}.panel-head{gap:8px}.home-actions a{font-size:12px;padding:8px 10px}}
"""


def _escape(value: object) -> str:
    return html.escape(str(value))


def _link(table: str, **params: object) -> str:
    if table == "sessions":
        return _escape("/admin/sessions/view?" + urlencode({"session_id":params["q"]})) if params.get("q") else "/admin/sessions"
    return _escape("/admin?" + urlencode({"table": table, **params}))


def _columns(conn: sqlite3.Connection, name: str, tables: dict[str, int]) -> set[str]:
    if name not in tables:
        return set()
    return {str(row[1]) for row in conn.execute(f'PRAGMA table_info("{name}")')}


def _session_summary(conn: sqlite3.Connection, tables: dict[str, int]) -> tuple[dict[str, int], list[sqlite3.Row]]:
    columns = {name: _columns(conn, name, tables) for name in ("sessions", "users", "sequences", "query_turns")}
    if "session_id" not in columns["sessions"]:
        return {"started": 0, "progress": 0, "ended": 0}, []

    def field(key: str, fallback: str = "NULL") -> str:
        return f's."{key}"' if key in columns["sessions"] else fallback

    def aggregate(table: str, time: str) -> str:
        if "session_key" not in columns[table]:
            return "SELECT NULL AS session_key, 0 AS n, NULL AS last_at WHERE 0"
        timestamp = f'MAX(datetime("{time}"))' if time in columns[table] else "NULL"
        return f'SELECT session_key, COUNT(*) AS n, {timestamp} AS last_at FROM "{table}" GROUP BY session_key'

    has_user = {"user_id", "name"} <= columns["users"] and "user_id" in columns["sessions"]
    join_user = "LEFT JOIN users u ON u.user_id=s.user_id" if has_user else ""
    worker = f"COALESCE(NULLIF(u.name,''),{field('user_id')},'미지정')" if has_user else f"COALESCE({field('user_id')},'미지정')"
    ended = f"(UPPER(COALESCE({field('session_status')},'')) IN ('COMPLETED','ENDED') OR NULLIF({field('ended_at')},'') IS NOT NULL)"
    sql = f"""WITH seq AS ({aggregate('sequences','received_at')}),
        qry AS ({aggregate('query_turns','created_at')}), summary AS (
        SELECT s.session_id, {field('user_id')} AS user_id, {worker} AS worker,
            {field('scenario_id')} AS scenario_id, {field('started_at')} AS started_at,
            COALESCE(seq.n,0) AS sequence_count, COALESCE(qry.n,0) AS query_count,
            CASE WHEN {ended} THEN 'ended'
                 WHEN COALESCE(seq.n,0)+COALESCE(qry.n,0)>0 THEN 'progress' ELSE 'started' END AS stage,
            MAX(COALESCE(datetime({field('started_at')}),''),COALESCE(datetime({field('ended_at')}),''),
                COALESCE(seq.last_at,''),COALESCE(qry.last_at,'')) AS last_at
        FROM sessions s LEFT JOIN seq ON seq.session_key=s.session_id
        LEFT JOIN qry ON qry.session_key=s.session_id {join_user}) """
    counts = {"started": 0, "progress": 0, "ended": 0}
    counts.update({row[0]: row[1] for row in conn.execute(sql + "SELECT stage,COUNT(*) FROM summary GROUP BY stage")})
    recent = conn.execute(sql + "SELECT * FROM summary ORDER BY last_at DESC, session_id DESC LIMIT 8").fetchall()
    return counts, recent


def _scenario_label(value: object) -> str:
    text = str(value or "미지정")
    return f"학습모드 {text}" if text.isdigit() and 1 <= int(text) <= 10 else text


def _time(value: object) -> str:
    if not value:
        return "미기록"
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return (parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed).astimezone(KST).strftime("%m.%d %H:%M")
    except ValueError:
        return "형식 확인 필요"


def _chart(points: list[tuple[str, int]], label: str) -> str:
    width, height, left, right, top, bottom = 760, 245, 44, 13, 16, 33
    plot_h, plot_w = height - top - bottom, width - left - right
    largest = max((value for _, value in points), default=0)
    maximum = max(4, math.ceil(largest / 4) * 4)
    pieces = [f'<div class="chart-scroll"><svg class="activity-chart" viewBox="0 0 {width} {height}" role="img" aria-label="{_escape(label)} 일별 저장 건수">']
    for tick in range(5):
        y = top + plot_h * (1 - tick / 4)
        pieces.append(f'<line x1="{left}" x2="{width-right}" y1="{y}" y2="{y}" stroke="#e7edf6"/><text x="{left-9}" y="{y+4}" text-anchor="end" fill="#8190a5" font-size="11">{maximum*tick//4:,}</text>')
    step = plot_w / max(1, len(points))
    for index, (date, value) in enumerate(points):
        x = left + index * step + step * .18
        h = value / maximum * plot_h
        pieces.append(f'<rect data-date="{date}" data-value="{value}" x="{x:.2f}" y="{top+plot_h-h:.2f}" width="{step*.64:.2f}" height="{h:.2f}" rx="2" fill="#5b7be8"><title>{date}: {value:,}건</title></rect>')
        if index % max(1, math.ceil(len(points) / 7)) == 0 or index == len(points)-1:
            pieces.append(f'<text x="{left+(index+.5)*step:.2f}" y="{height-9}" text-anchor="middle" fill="#8190a5" font-size="11">{date[5:].replace("-",".")}</text>')
    pieces.append('</svg></div>')
    rows = "".join(f'<tr><td>{date}</td><td>{value:,}건</td></tr>' for date, value in reversed(points))
    pieces.append(f'<details class="chart-data"><summary>날짜별 수치 보기</summary><div class="chart-data-scroll"><table><thead><tr><th>날짜 (한국 시간)</th><th>{_escape(label)}</th></tr></thead><tbody>{rows}</tbody></table></div></details>')
    return "".join(pieces)


def render_dashboard(conn: sqlite3.Connection | None, tables: dict[str, int], *, days: int = 30, metric: str = "sessions") -> str:
    now = datetime.now(timezone.utc).astimezone(KST)
    dates = [(now.date() - timedelta(days=offset)).isoformat() for offset in reversed(range(days))]
    counts = {key: tables.get(table, 0) for key, table in (("users","users"),("sessions","sessions"),("sequences","sequences"),("queries","query_turns"))}
    stages = {"started": 0, "progress": 0, "ended": 0}
    reports = {"complete": 0, "pending": 0, "failed": 0, "other": 0}
    recent: list[sqlite3.Row] = []
    scenarios: list[Any] = []
    daily: dict[str, int] = {}
    label, table, time_col, date_basis = METRICS[metric]
    if conn is not None:
        stages, recent = _session_summary(conn, tables)
        if "scenario_id" in _columns(conn, "sessions", tables):
            scenarios = conn.execute("SELECT COALESCE(NULLIF(scenario_id,''),'미지정'), COUNT(*) AS n FROM sessions GROUP BY 1 ORDER BY n DESC,1").fetchall()
        if "report_status" in _columns(conn, "completions", tables):
            for status, amount in conn.execute("SELECT LOWER(TRIM(report_status)), COUNT(*) FROM completions GROUP BY 1"):
                kind = "complete" if status in {"complete","completed","success"} else status if status in {"pending","failed"} else "other"
                reports[kind] += amount
        if time_col in _columns(conn, table, tables):
            daily = dict(conn.execute(f'SELECT date("{time_col}",\'+9 hours\') AS day, COUNT(*) FROM "{table}" WHERE date("{time_col}",\'+9 hours\') BETWEEN ? AND ? GROUP BY day', (dates[0],dates[-1])).fetchall())
    cards = "".join(
        f'<a class="card kpi" href="{_link(table_name)}"><div class="kpi-top">{title}<i class="kpi-icon" aria-hidden="true">{icon}</i></div><strong data-stat="{key}">{counts[key]:,}</strong><p>{note}</p></a>'
        for key, table_name, title, icon, note in (
            ("sessions","sessions","전체 훈련 세션","S","시작된 세션의 누적 기록"),
            ("users","users","등록 작업자","U","DB에 등록된 작업자"),
            ("sequences","sequences","저장 시퀀스","↗","동일 세션·시퀀스의 중복 저장 제외"),
            ("queries","query_turns","저장 질문·답변","Q","세션에 연결되어 저장된 대화"),
        )
    )
    current_attr = ' aria-current="true"'
    tabs = "".join(f'<a class="{"active" if key==metric else ""}" href="/admin?days={days}&amp;metric={key}"{current_attr if key==metric else ""}>{value[0]}</a>' for key,value in METRICS.items())
    choices = sorted({7,30,90,days})
    options = "".join(f'<option value="{d}"{" selected" if d==days else ""}>최근 {d}일</option>' for d in choices)
    chart = _chart([(day,daily.get(day,0)) for day in dates],label)
    if not sum(daily.values()):
        chart = '<p class="panel-note">선택 기간에 저장된 기록이 없습니다. 다른 지표나 기간을 선택해 보세요.</p>' + chart
    max_mode = max((row[1] for row in scenarios), default=1)
    ranking = "".join(f'<div><div class="rank-line"><span>{_escape(_scenario_label(mode))}</span><strong>{amount:,}회</strong></div><div class="rank-track"><div class="rank-fill" style="width:{100*amount/max_mode:.2f}%"></div></div></div>' for mode,amount in scenarios[:8])
    if len(scenarios)>8:
        ranking += f'<p class="other">그 외 {len(scenarios)-8}개 콘텐츠 · {sum(row[1] for row in scenarios[8:]):,}회</p>'
    if not ranking:
        ranking = '<div class="home-empty">아직 콘텐츠별 훈련 기록이 없습니다.</div>'
    stage_cards = "".join(f'<div class="state-box"><span class="state-badge {key}">{title}</span><strong data-stat="{key}">{stages[key]:,}</strong><p>{note}</p></div>' for key,title,note in (
        ("started","시작 · 입력 대기","시퀀스·질문이 아직 없는 세션"),
        ("progress","진행 중","시퀀스 또는 질문이 저장된 미종료 세션"),
        ("ended","훈련 종료","세션 종료 기록이 있는 훈련")))
    report_rows = "".join(f'<div class="report-row"><span class="state-badge {key}">{title}</span><strong data-stat="reports_{key}">{reports[key]:,}</strong></div>' for key,title in (("complete","처리 완료"),("pending","분석 대기·진행"),("failed","처리 실패"),("other","기타 상태")))
    stage_labels = {"started":"시작 · 입력 대기","progress":"진행 중","ended":"훈련 종료"}
    rows = "".join(f'<tr><td><a class="session-link" href="{_link("sessions",q=row["session_id"])}">{_escape(row["session_id"])}</a></td><td class="identity">{_escape(row["worker"])}<small>{_escape(row["user_id"] or "")}</small></td><td>{_escape(_scenario_label(row["scenario_id"]))}</td><td><span class="state-badge {row["stage"]}">{stage_labels[row["stage"]]}</span></td><td>{row["sequence_count"]:,}회 / {row["query_count"]:,}회</td><td class="time-cell">{_time(row["last_at"])}</td></tr>' for row in recent)
    if not rows:
        rows = '<tr><td colspan="6" class="home-empty">아직 등록된 세션이 없습니다.</td></tr>'
    empty_notice = '<p class="home-footnote">DB 파일이 아직 생성되지 않았습니다. 훈련 데이터가 저장되면 통계가 표시됩니다.</p>' if conn is None else ""
    return f"""
    <div class="topbar"><div><div class="eyebrow">CO-PILOT · DATA OVERVIEW</div><h1>Home</h1><div class="sub">전체 훈련 데이터와 처리 현황을 한눈에 확인하세요.</div>
    <div class="home-meta"><span class="scope-pill">전체 누적 데이터</span><span class="readonly"><i class="dot"></i>읽기 전용</span><span>조회 시각 {now:%Y.%m.%d %H:%M:%S} KST</span></div></div>
    <div class="actions home-actions"><a class="secondary" href="/admin?days={days}&amp;metric={metric}">새로고침</a></div></div>
    {empty_notice}<div class="cards home-kpis">{cards}</div>
    <div class="dashboard-grid"><section class="dashboard-panel"><div class="panel-head"><div><h2>일별 데이터 추이</h2><p>최근 {days}일 · 한국 시간 · {date_basis} 기준</p></div>
    <form class="period-form" action="/admin"><input type="hidden" name="metric" value="{metric}"><label for="days" class="sub" style="font-size:12px">기간</label><select id="days" name="days">{options}</select><button>조회</button></form></div>
    <nav class="metric-tabs" aria-label="추이 지표">{tabs}</nav><div class="trend-meta"><strong>{sum(daily.values()):,}<small style="font-size:13px;font-weight:500"> 건</small></strong><span>선택 기간 {label}</span></div>{chart}</section>
    <section class="dashboard-panel"><div class="panel-head"><div><h2>훈련 콘텐츠별 이용 현황</h2><p>전체 세션 기준 · 상위 8개 콘텐츠</p></div></div><div class="rank-list">{ranking}</div></section></div>
    <div class="dashboard-grid"><section class="dashboard-panel"><div class="panel-head"><div><h2>훈련 진행 상태</h2><p>현재 저장된 세션의 진행 단계</p></div></div><div class="status-grid">{stage_cards}</div><p class="panel-footer">진행 중은 미종료 상태를 뜻합니다. 현재 접속 여부를 의미하지 않으며, 마지막 수신 시각은 아래 세션 목록에서 확인합니다.</p></section>
    <section class="dashboard-panel"><div class="panel-head"><div><h2>리포트 처리 현황</h2><p>누적 리포트 {tables.get('completions',0):,}건</p></div><a class="text-link" href="{_link('completions')}">기록 보기 →</a></div><div class="report-list">{report_rows}</div><p class="panel-footer">훈련 종료와 리포트 처리는 별개입니다. 처리 완료라도 영역별 분석 가능 여부는 리포트에서 확인해야 합니다.</p></section></div>
    <section class="dashboard-panel recent-panel"><div class="panel-head"><div><h2>최근 활동 세션</h2><p>마지막 시퀀스·질문 수신 또는 훈련 시작·종료 시각 기준 · 최근 8개</p></div><a class="text-link" href="{_link('sessions')}">전체 세션 보기 →</a></div>
    <div class="recent-scroll"><table class="recent-table"><thead><tr><th>세션 ID · 대시보드</th><th>작업자</th><th>훈련 콘텐츠</th><th>훈련 상태</th><th>시퀀스 / 질문</th><th>마지막 활동 (KST)</th></tr></thead><tbody>{rows}</tbody></table></div></section>
    <p class="home-footnote"><strong>통계 기준 안내</strong><br>전체 수치는 현재 DB에 저장된 데이터 기준이며 실제 훈련과 테스트 데이터를 모두 포함합니다. 저장 건수는 API 호출 횟수와 다릅니다. 중복 재전송과 저장되지 않은 호출·실패는 포함되지 않습니다. 질문 통계는 저장된 대화 기록 기준이며 세션 없이 처리한 질문은 포함되지 않을 수 있습니다. 날짜가 없거나 형식이 잘못된 기록은 전체 건수에는 포함하고 일별 그래프에서는 제외합니다.</p>
    """
