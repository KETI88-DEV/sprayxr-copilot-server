"""Read-only dashboard for generated training-recording JSON artifacts."""

from __future__ import annotations


TRAINING_DATA_CSS = r"""
#training-data-root{--td-blue:#3157d5;--td-green:#087f5b;--td-orange:#ac6c13;--td-red:#b33b46;min-width:0}
#training-data-root *{box-sizing:border-box}
#training-data-root button,#training-data-root input,#training-data-root select{font:inherit}
#training-data-root button,#training-data-root a,#training-data-root input,#training-data-root select,#training-data-root summary{outline-offset:3px}
#training-data-root :focus-visible{outline:3px solid #8faaf5}
#training-data-root .td-header{display:flex;align-items:flex-start;justify-content:space-between;gap:20px;margin-bottom:22px}
#training-data-root .td-eyebrow{color:var(--brand,#3157d5);font-size:11px;font-weight:800;letter-spacing:.12em}
#training-data-root h1{font-size:28px;letter-spacing:-.04em;margin:6px 0 9px}
#training-data-root h2,#training-data-root h3,#training-data-root h4{color:var(--ink,#152033);letter-spacing:-.025em}
#training-data-root .td-description{font-size:13px;line-height:1.8;color:var(--muted,#68758a);margin:0;max-width:780px}
#training-data-root .td-header-meta{font-size:11px;line-height:1.8;color:var(--muted,#68758a);text-align:right;flex-shrink:0;padding-top:3px}
#training-data-root .td-header-meta strong{display:block;color:var(--ink,#152033);font-weight:650}
#training-data-root .td-kpis{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:12px;margin-bottom:18px}
#training-data-root .td-kpi{border:1px solid var(--line,#dde4ee);border-radius:12px;padding:17px 16px;background:var(--panel,#fff);min-width:0;box-shadow:0 3px 12px #243a5e03}
#training-data-root .td-kpi>span{font-size:11px;color:var(--muted,#68758a);display:block}
#training-data-root .td-kpi strong{display:block;font-size:28px;letter-spacing:-.04em;margin-top:9px;font-variant-numeric:tabular-nums;line-height:1.2}
#training-data-root .td-kpi small{font-size:10px;color:var(--muted,#68758a);display:block;margin-top:7px;line-height:1.5}
#training-data-root .td-kpi.td-highlight{background:#edf2ff;border-color:#d5dffb}
#training-data-root .td-kpi.td-highlight strong{color:var(--td-blue)}
#training-data-root .td-kpi.td-good strong{color:var(--td-green)}
#training-data-root .td-kpi.td-review strong{color:var(--td-orange)}
#training-data-root .td-panel{background:var(--panel,#fff);border:1px solid var(--line,#dde4ee);border-radius:12px;min-width:0}
#training-data-root .td-filter-panel{padding:17px 18px;margin-bottom:18px}
#training-data-root .td-filter-form{display:grid;grid-template-columns:minmax(130px,.8fr) minmax(120px,.8fr) minmax(150px,1fr) minmax(170px,1.6fr) auto;gap:12px;align-items:end}
#training-data-root .td-field{display:flex;flex-direction:column;gap:7px;min-width:0}
#training-data-root .td-field label{font-size:11px;font-weight:700;color:#506079}
#training-data-root input,#training-data-root select{border:1px solid #cfd8e5;border-radius:8px;padding:10px 11px;background:#fff;color:var(--ink,#152033);width:100%;min-width:0;font-size:12px;min-height:39px}
#training-data-root input::placeholder{color:#91a0b4}
#training-data-root button{border:1px solid transparent;background:var(--td-blue);color:#fff;border-radius:8px;padding:10px 14px;cursor:pointer;font-size:12px;font-weight:700;line-height:1.4}
#training-data-root button:hover{filter:brightness(.96)}
#training-data-root button:disabled{opacity:.45;cursor:default}
#training-data-root .td-button-secondary{background:#fff;border-color:var(--line,#dde4ee);color:#51617a}
#training-data-root .td-filter-actions{display:flex;gap:7px}
#training-data-root .td-filter-foot{display:flex;gap:12px;justify-content:space-between;align-items:center;margin-top:14px;padding-top:12px;border-top:1px solid #edf1f6;flex-wrap:wrap}
#training-data-root .td-help{font-size:11px;color:var(--muted,#68758a);line-height:1.8;margin:0}
#training-data-root .td-link{border:0;background:transparent;padding:0;color:var(--td-blue);font-size:11px;font-weight:650;cursor:pointer;text-decoration:none}
#training-data-root .td-chart-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin-bottom:20px}
#training-data-root .td-chart-card{padding:18px 19px;min-height:188px}
#training-data-root .td-section-heading{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}
#training-data-root .td-chart-card h2{font-size:13px;margin:0}
#training-data-root .td-chart-card .td-chart-note{font-size:10px;color:var(--muted,#68758a);margin:7px 0 12px}
#training-data-root .td-bars{width:100%;max-height:225px;overflow:auto}
#training-data-root .td-bars svg{display:block;width:100%;min-width:180px}
#training-data-root .td-bars .td-chart-empty{padding:30px 0;text-align:center}
#training-data-root .td-list-heading{display:flex;justify-content:space-between;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:11px}
#training-data-root .td-list-heading h2{font-size:17px;margin:0 0 5px}
#training-data-root .td-exports{display:flex;gap:7px;align-items:center;flex-wrap:wrap}
#training-data-root .td-export{display:inline-flex;align-items:center;gap:6px;font-size:11px;font-weight:650;color:#4a5b76;background:#fff;padding:8px 11px;text-decoration:none;border:1px solid var(--line,#dde4ee);border-radius:7px}
#training-data-root .td-export svg{width:13px;height:13px;flex-shrink:0}
#training-data-root .td-export[aria-disabled=true]{opacity:.45;pointer-events:none}
#training-data-root .td-list-panel{overflow:hidden}
#training-data-root .td-table-scroll{overflow:auto}
#training-data-root table{border-collapse:collapse;width:100%;font-size:12px;table-layout:auto}
#training-data-root th{background:#f1f5fb;color:#5a6b83;text-align:left;font-weight:650;padding:12px 15px;border-bottom:1px solid var(--line,#dde4ee);font-size:10px;letter-spacing:.02em;white-space:nowrap;position:static}
#training-data-root td{padding:13px 15px;border-bottom:1px solid #edf1f6;vertical-align:middle;white-space:nowrap;max-width:340px}
#training-data-root .td-dataset-row{cursor:pointer}
#training-data-root .td-dataset-row:hover td{background:#f7f9ff}
#training-data-root .td-dataset-row:last-child td{border-bottom:0}
#training-data-root .td-dataset-name{background:transparent;border:0;color:var(--ink,#152033);font-weight:700;text-align:left;padding:0;font-size:12px;display:block;max-width:330px;white-space:normal;overflow-wrap:anywhere;line-height:1.6}
#training-data-root .td-dataset-name:hover{color:var(--td-blue);filter:none}
#training-data-root .td-row-note{display:block;color:#8793a6;font-size:10px;margin-top:4px;max-width:330px;overflow:hidden;text-overflow:ellipsis}
#training-data-root .td-date{font-size:11px;font-variant-numeric:tabular-nums;color:#65748b}
#training-data-root .td-muted{color:var(--muted,#68758a)}
#training-data-root .td-badge{display:inline-flex;align-items:center;gap:5px;border-radius:5px;background:#edf1f6;color:#637089;padding:4px 7px;font-size:10px;font-weight:650;white-space:nowrap}
#training-data-root .td-badge::before{content:'';width:5px;height:5px;border-radius:50%;background:currentColor}
#training-data-root .td-badge.complete{background:#e8f6ee;color:var(--td-green)}
#training-data-root .td-badge.needs_review{background:#fff4df;color:var(--td-orange)}
#training-data-root .td-badge.failed{background:#ffebed;color:var(--td-red)}
#training-data-root .td-table-message{text-align:center;padding:42px 20px;white-space:normal;color:var(--muted,#68758a);line-height:1.8;font-size:12px}
#training-data-root .td-table-message strong{display:block;color:#52617a;font-size:13px;margin-bottom:6px}
#training-data-root .td-table-message button{margin-top:14px}
#training-data-root .td-pagination{display:flex;justify-content:space-between;align-items:center;gap:15px;padding:13px 16px;border-top:1px solid var(--line,#dde4ee);font-size:11px;color:var(--muted,#68758a);flex-wrap:wrap}
#training-data-root .td-page-controls{display:flex;align-items:center;gap:10px}
#training-data-root .td-page-controls button{padding:6px 10px;font-size:11px;background:#fff;color:#51617a;border:1px solid var(--line,#dde4ee)}
#training-data-root .td-page-size{display:flex;align-items:center;gap:7px}
#training-data-root .td-page-size select{width:auto;min-height:0;font-size:11px;padding:5px 8px}
#training-data-root .td-footer-note{font-size:11px;color:#8793a6;line-height:1.8;margin:12px 0 0}
#training-data-root [hidden]{display:none!important}
#training-data-root [aria-busy=true] .td-dataset-row{opacity:.5;pointer-events:none}
#training-data-root .td-modal{padding:0;border:1px solid #d4ddea;border-radius:16px;background:#f4f7fb;color:var(--ink,#152033);width:min(1200px,calc(100vw - 44px));max-width:none;max-height:calc(100dvh - 44px);box-shadow:0 30px 100px #12203c35;overflow:hidden}
#training-data-root .td-modal::backdrop{background:#10223b88;backdrop-filter:blur(3px)}
#training-data-root .td-modal-head{display:flex;align-items:flex-start;justify-content:space-between;gap:20px;padding:21px 25px 18px;border-bottom:1px solid var(--line,#dde4ee);background:#fff}
#training-data-root .td-modal-head h2{font-size:20px;line-height:1.5;margin:4px 0;overflow-wrap:anywhere}
#training-data-root .td-modal-head .td-help{font-size:11px}
#training-data-root .td-close{font-size:23px;line-height:1;padding:7px 10px;border:1px solid var(--line,#dde4ee);background:#fff;color:#68758a;flex-shrink:0}
#training-data-root .td-modal-body{padding:23px 25px;overflow:auto;max-height:calc(100dvh - 180px);overscroll-behavior:contain}
#training-data-root .td-detail-section{padding:20px 22px;margin-bottom:17px}
#training-data-root .td-detail-section h3{font-size:16px;margin:0 0 7px}
#training-data-root .td-detail-section h4{font-size:13px;margin:0 0 13px}
#training-data-root .td-meta-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:19px 24px;margin:20px 0 0}
#training-data-root .td-meta-grid dt{font-size:10px;color:var(--muted,#68758a);margin-bottom:7px}
#training-data-root .td-meta-grid dd{font-size:12px;font-weight:550;margin:0;overflow-wrap:anywhere;line-height:1.6}
#training-data-root .td-quality-box{border-left:3px solid #d2dcec;padding:11px 14px;background:#f5f7fb;border-radius:0 7px 7px 0;margin-top:19px;font-size:11px;color:#65748b;line-height:1.8}
#training-data-root .td-quality-box.complete{border-color:#2c9a74;background:#f0f9f5}
#training-data-root .td-quality-box.needs_review{border-color:#d4a04d;background:#fffbf1}
#training-data-root .td-quality-box.failed{border-color:#ca6370;background:#fff4f5}
#training-data-root .td-quality-box ul{margin:8px 0 0;padding-left:18px}
#training-data-root .td-source{font-size:11px;margin-top:17px;border-top:1px solid #edf1f6;padding-top:14px}
#training-data-root summary{cursor:pointer;line-height:1.7;color:#4d6389;font-size:11px}
#training-data-root .td-source dl{display:grid;grid-template-columns:90px minmax(0,1fr);gap:9px;margin:13px 0 0;padding:12px;background:#f6f8fc;border-radius:7px;line-height:1.7}
#training-data-root .td-source dt{color:var(--muted,#68758a)}
#training-data-root .td-source dd{margin:0;overflow-wrap:anywhere;color:#4d5f7a}
#training-data-root .td-map-controls{display:flex;gap:12px;align-items:end;justify-content:space-between;flex-wrap:wrap;margin:19px 0 14px}
#training-data-root .td-map-select{max-width:410px;flex:1 1 220px}
#training-data-root .td-segment{display:inline-flex;border:1px solid var(--line,#dde4ee);border-radius:8px;padding:3px;background:#f4f7fb;gap:3px}
#training-data-root .td-segment button{background:transparent;color:#68758a;padding:7px 11px;font-size:11px}
#training-data-root .td-segment button[aria-pressed=true]{background:#fff;color:var(--td-blue);box-shadow:0 1px 4px #20304e15}
#training-data-root .td-map-tools{display:flex;align-items:center;gap:11px;flex-wrap:wrap}
#training-data-root .td-map-enlarge{font-size:11px;color:var(--td-blue);text-decoration:none;white-space:nowrap}
#training-data-root .td-map-enlarge:hover{text-decoration:underline}
#training-data-root .td-map-scale{max-width:490px;margin:13px auto 0}
#training-data-root .td-map-scale svg{display:block;width:100%;height:auto}
#training-data-root .td-map-figure{margin:0}
#training-data-root .td-map-stage{display:flex;align-items:center;justify-content:center;min-height:180px;border:1px solid #e0e6f0;background:#f0f3f8;border-radius:9px;overflow:hidden;padding:12px}
#training-data-root .td-map-stage img{display:block;max-width:100%;max-height:480px;width:auto;height:auto;object-fit:contain;image-rendering:auto}
#training-data-root .td-map-figure figcaption{font-size:10px;line-height:1.8;color:var(--muted,#68758a);margin-top:9px}
#training-data-root .td-map-notes{margin-top:14px;padding:12px 14px;background:#f6f8fc;border-radius:8px;font-size:10px;color:#68758a;line-height:1.9}
#training-data-root .td-map-notes p{margin:0}
#training-data-root .td-map-message{font-size:12px;color:var(--muted,#68758a);text-align:center;line-height:1.8;padding:25px 10px}
#training-data-root .td-coating-metrics{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:10px;margin:17px 0 22px}
#training-data-root .td-coating-metric{padding:12px;background:#f6f8fc;border-radius:8px;min-width:0}
#training-data-root .td-coating-metric span{display:block;font-size:10px;color:var(--muted,#68758a);line-height:1.6}
#training-data-root .td-coating-metric strong{display:block;font-size:17px;letter-spacing:-.03em;margin-top:7px;font-variant-numeric:tabular-nums}
#training-data-root .td-analysis-grid{display:grid;grid-template-columns:1fr 1fr;gap:22px;margin-top:18px}
#training-data-root .td-analysis-card{min-width:0;overflow:hidden}
#training-data-root .td-analysis-card>p{margin:5px 0 12px}
#training-data-root .td-chart-scroll{overflow:auto;max-height:430px}
#training-data-root .td-chart-scroll svg{display:block;width:100%;min-width:260px}
#training-data-root .td-chart-empty{color:var(--muted,#68758a);font-size:11px;padding:28px 16px;background:#f6f8fc;border-radius:7px;text-align:center;line-height:1.8}
#training-data-root .td-data-table{margin-top:12px;border:1px solid #e5ebf3;border-radius:7px;max-height:240px;overflow:auto}
#training-data-root .td-data-table table{font-size:10px}
#training-data-root .td-data-table th,#training-data-root .td-data-table td{font-size:10px;padding:8px 10px}
#training-data-root .td-data-table th{position:sticky;top:0}
#training-data-root .td-json-actions{display:flex;align-items:center;justify-content:space-between;gap:15px;margin-bottom:14px}
#training-data-root .td-json-actions h3{margin:0}
#training-data-root .td-json-view{border:1px solid #e3e9f1;border-radius:8px;padding:12px 14px;margin-top:10px}
#training-data-root .td-json-tree{font-family:Consolas,monospace;font-size:11px;line-height:1.8;margin-top:10px;max-height:470px;overflow:auto}
#training-data-root .td-json-tree details{padding-left:13px;border-left:1px solid #e3e9f1;margin:4px 0}
#training-data-root .td-json-tree summary{font-family:inherit;overflow-wrap:anywhere;color:#4d6286}
#training-data-root .td-json-leaf{padding-left:14px;overflow-wrap:anywhere;white-space:pre-wrap;color:#62728a}
#training-data-root .td-json-key{color:#3157a9;font-weight:600}
#training-data-root .td-json-null{color:#98a3b5}
#training-data-root .td-json-raw{font-family:Consolas,monospace;font-size:10px;line-height:1.8;white-space:pre;max-height:470px;overflow:auto;padding:13px;background:#f4f7fb;border-radius:6px;margin:12px 0 0;color:#50617d}
#training-data-root .td-detail-loading{text-align:center;color:var(--muted,#68758a);padding:65px 20px;font-size:13px;line-height:1.8}
body.td-modal-open{overflow:hidden}
@media(min-width:1600px){#training-data-root .td-chart-card{min-height:204px}}
@media(max-width:1200px){#training-data-root .td-kpis{grid-template-columns:repeat(3,minmax(0,1fr))}#training-data-root .td-filter-form{grid-template-columns:repeat(3,minmax(0,1fr))}#training-data-root .td-search-field{grid-column:span 2}#training-data-root .td-filter-actions{justify-content:flex-end}#training-data-root .td-coating-metrics{grid-template-columns:repeat(3,minmax(0,1fr))}}
@media(max-width:760px){#training-data-root .td-header{display:block}#training-data-root .td-header-meta{text-align:left;margin-top:12px}#training-data-root .td-header-meta strong{display:inline;margin-left:6px}#training-data-root .td-chart-grid{grid-template-columns:1fr;gap:10px}#training-data-root .td-chart-card{min-height:130px;padding:16px}#training-data-root .td-filter-form{grid-template-columns:1fr 1fr;gap:12px}#training-data-root .td-search-field{grid-column:span 1}#training-data-root .td-filter-actions{grid-column:1 / -1;justify-content:flex-start}#training-data-root .td-kpi{padding:14px 12px}#training-data-root .td-kpi strong{font-size:25px}#training-data-root .td-modal{width:calc(100vw - 20px);max-height:calc(100dvh - 20px);border-radius:12px}#training-data-root .td-modal-head{padding:17px 18px 14px}#training-data-root .td-modal-head h2{font-size:17px}#training-data-root .td-modal-body{padding:15px;max-height:calc(100dvh - 145px)}#training-data-root .td-detail-section{padding:17px 15px}#training-data-root .td-meta-grid{grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}#training-data-root .td-analysis-grid{grid-template-columns:1fr;gap:22px}#training-data-root .td-map-stage{min-height:150px;padding:7px}#training-data-root .td-map-stage img{max-height:350px}#training-data-root .td-coating-metric{padding:10px}#training-data-root .td-coating-metric strong{font-size:16px}}
@media(max-width:430px){#training-data-root .td-kpis{gap:8px}#training-data-root .td-kpi>span{font-size:10px}#training-data-root .td-kpi small{font-size:9px}#training-data-root .td-kpi strong{font-size:23px}#training-data-root .td-filter-panel{padding:15px}#training-data-root .td-filter-form{grid-template-columns:1fr}#training-data-root .td-filter-actions{grid-column:auto}#training-data-root .td-pagination{gap:12px}#training-data-root .td-page-controls{gap:7px}#training-data-root .td-map-controls{gap:10px}#training-data-root .td-segment{width:100%}#training-data-root .td-segment button{flex:1}#training-data-root .td-coating-metrics{gap:7px}}
@media(prefers-reduced-motion:reduce){#training-data-root *{scroll-behavior:auto!important}}
"""


def render_training_data() -> str:
    """Return a shell populated from the generated-artifact API in the browser."""
    return r"""
<section id="training-data-root" aria-labelledby="td-page-title">
  <header class="td-header">
    <div><div class="td-eyebrow">TRAINING DATASET</div><h1 id="td-page-title">학습 데이터</h1>
      <p class="td-description">훈련 기록 CSV와 연결된 높이맵 텍스처를 세션별 JSON으로 변환한 데이터입니다.<br>작업자와 학습모드별 구성을 살펴보고, 각 데이터의 맵 분석과 생성 JSON을 확인합니다.</p>
    </div>
    <div class="td-header-meta">최근 변환<strong id="td-build-time">불러오는 중</strong><span id="td-build-note">생성된 데이터 기준</span></div>
  </header>

  <div class="td-kpis" aria-label="학습 데이터 현황">
    <div class="td-kpi"><span>전체 데이터</span><strong id="td-kpi-total">—</strong><small>CSV 세션 기준</small></div>
    <div class="td-kpi td-highlight"><span>조회 데이터</span><strong id="td-kpi-filtered">—</strong><small>현재 필터에 해당</small></div>
    <div class="td-kpi"><span>작업자</span><strong id="td-kpi-workers">—</strong><small>현재 조회 결과</small></div>
    <div class="td-kpi td-good"><span>품질 완료</span><strong id="td-kpi-complete">—</strong><small>변환 검증 기준</small></div>
    <div class="td-kpi td-review"><span>확인 필요</span><strong id="td-kpi-review">—</strong><small id="td-kpi-review-note">품질 검토 대상</small></div>
    <div class="td-kpi"><span>높이맵 텍스처</span><strong id="td-kpi-maps">—</strong><small>현재 조회 결과</small></div>
  </div>

  <section class="td-panel td-filter-panel" aria-label="학습 데이터 필터">
    <form id="td-filter-form" class="td-filter-form" role="search">
      <div class="td-field"><label for="td-worker">작업자</label><select id="td-worker" name="worker_id"><option value="all">전체 작업자</option></select></div>
      <div class="td-field"><label for="td-process">공정</label><select id="td-process" name="process"><option value="all">전체 공정</option></select></div>
      <div class="td-field"><label for="td-mode">시나리오 / 학습모드</label><select id="td-mode" name="scenario_id"><option value="all">전체 학습모드</option></select></div>
      <div class="td-field td-search-field"><label for="td-search">검색어</label><input id="td-search" name="q" type="search" maxlength="100" autocomplete="off" placeholder="데이터명, 작업자, 원본 파일 검색"></div>
      <div class="td-filter-actions"><button type="submit">조회</button><button type="button" id="td-reset" class="td-button-secondary">초기화</button></div>
    </form>
    <div class="td-filter-foot"><p class="td-help">CSV 훈련 기록 1개(1세션) = 학습 데이터 1건 · 높이맵은 해당 세션에 연결됩니다.</p><button type="button" id="td-refresh" class="td-link">새로고침 ↻</button></div>
  </section>

  <div class="td-chart-grid" aria-label="조회 결과 분포">
    <section class="td-panel td-chart-card"><h2>작업자별 데이터</h2><p class="td-chart-note">현재 조회 결과 · 건</p><div id="td-worker-chart" class="td-bars"><p class="td-chart-empty">불러오는 중</p></div></section>
    <section class="td-panel td-chart-card"><h2>학습모드별 데이터</h2><p class="td-chart-note">현재 조회 결과 · 건</p><div id="td-mode-chart" class="td-bars"><p class="td-chart-empty">불러오는 중</p></div></section>
    <section class="td-panel td-chart-card"><h2>공정별 데이터</h2><p class="td-chart-note">현재 조회 결과 · 건</p><div id="td-process-chart" class="td-bars"><p class="td-chart-empty">불러오는 중</p></div></section>
  </div>

  <section aria-labelledby="td-list-title">
    <div class="td-list-heading"><div><h2 id="td-list-title">데이터 목록</h2><p id="td-table-status" class="td-help" role="status" aria-live="polite">생성된 학습 데이터를 불러오고 있습니다.</p></div>
      <div class="td-exports"><a id="td-export-csv" class="td-export" href="/admin/api/training-data/export.csv">목록 CSV ↓</a><a id="td-export-zip" class="td-export" href="/admin/api/training-data/export.zip">JSON 묶음 ZIP ↓</a></div>
    </div>
    <div id="td-list-panel" class="td-panel td-list-panel" aria-busy="true">
      <div class="td-table-scroll"><table aria-label="생성된 학습 데이터 목록"><thead><tr><th scope="col">데이터명</th><th scope="col">작업자</th><th scope="col">공정</th><th scope="col">학습모드</th><th scope="col">취득 시각</th><th scope="col">변환 시각</th><th scope="col">품질</th></tr></thead><tbody id="td-items"><tr><td colspan="7" class="td-table-message">학습 데이터를 불러오는 중입니다.</td></tr></tbody></table></div>
      <div class="td-pagination"><span id="td-range">—</span><div class="td-page-controls"><button type="button" id="td-prev" disabled aria-label="이전 페이지">← 이전</button><span id="td-page">1 / 1</span><button type="button" id="td-next" disabled aria-label="다음 페이지">다음 →</button></div><label class="td-page-size" for="td-limit">표시<select id="td-limit"><option value="10">10건</option><option value="25" selected>25건</option><option value="50">50건</option><option value="100">100건</option></select></label></div>
    </div>
    <p class="td-footer-note">취득 시각이 원본에 없으면 ‘미기록’으로 표시합니다. 변환 시각은 JSON을 생성한 시각입니다.<br>이 화면은 학습용 데이터의 생성·검증 결과이며, 모델 학습 완료 여부를 의미하지 않습니다.</p>
  </section>

  <dialog id="td-detail" class="td-modal" aria-labelledby="td-detail-title" aria-describedby="td-detail-description">
    <header class="td-modal-head"><div><div class="td-eyebrow">DATASET DETAIL</div><h2 id="td-detail-title">학습 데이터 상세</h2><p id="td-detail-description" class="td-help">원본 출처, 연결 맵과 생성된 JSON을 확인합니다.</p></div><button type="button" id="td-detail-close" class="td-close" aria-label="상세 창 닫기">×</button></header>
    <div id="td-detail-content" class="td-modal-body"><p class="td-detail-loading">상세 데이터를 불러오는 중입니다.</p></div>
  </dialog>
</section>
<script src="/admin/assets/training_data.js" defer></script>
"""
