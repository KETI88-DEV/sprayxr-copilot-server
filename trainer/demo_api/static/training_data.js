/* Generated training artifacts: read-only dashboard and accessible detail viewer. */
(() => {
  'use strict';
  const root = document.getElementById('training-data-root');
  if (!root) return;
  const $ = id => document.getElementById(id);
  const API = '/admin/api/training-data';
  const NS = 'http://www.w3.org/2000/svg';
  const numberFormat = new Intl.NumberFormat('ko-KR', {maximumFractionDigits: 1});
  const query = new URLSearchParams(window.location.search);
  const state = {
    worker: query.get('worker_id') || 'all', process: query.get('process') || 'all', mode: query.get('scenario_id') || 'all',
    q: (query.get('q') || '').slice(0, 100), page: Number.isSafeInteger(Number(query.get('page'))) && Number(query.get('page')) > 0 ? Number(query.get('page')) : 1,
    limit: [10, 25, 50, 100].includes(Number(query.get('limit'))) ? Number(query.get('limit')) : 25,
    pages: 1, listAbort: null, detailAbort: null, returnFocus: null, mapView: 'decoded', data: null,
  };

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }
  function svgEl(tag, attrs = {}, text) {
    const node = document.createElementNS(NS, tag);
    Object.entries(attrs).forEach(([name, value]) => node.setAttribute(name, String(value)));
    if (text !== undefined) node.textContent = String(text);
    return node;
  }
  function finite(value) {
    if (value === null || value === undefined || value === '') return null;
    const result = Number(value);
    return Number.isFinite(result) ? result : null;
  }
  function count(value) { return numberFormat.format(Math.max(0, finite(value) || 0)); }
  function valueText(value, unit = '') {
    const n = finite(value);
    return n === null ? '미기록' : `${numberFormat.format(n)}${unit}`;
  }
  function dateText(value) {
    if (!value) return '미기록';
    const text = String(value), match = text.match(/^(\d{4}-\d{2}-\d{2})(?:$|T|\s)/);
    return match ? match[1] : '미기록';
  }
  function bytesText(value) {
    const n = finite(value);
    if (n === null) return '미기록';
    if (n < 1024) return `${count(n)} B`;
    if (n < 1048576) return `${numberFormat.format(n / 1024)} KB`;
    return `${numberFormat.format(n / 1048576)} MB`;
  }
  function processLabel(value) {
    return {spray: '스프레이 도장', blasting: '블라스팅'}[value] || value || '미지정';
  }
  function qualityInfo(value) {
    const status = String(value || '').toLowerCase();
    if (['complete', 'completed', 'ready', 'success'].includes(status)) return {key: 'complete', label: '완료'};
    if (['needs_review', 'review', 'warning', 'partial'].includes(status)) return {key: 'needs_review', label: '확인 필요'};
    if (['failed', 'error'].includes(status)) return {key: 'failed', label: '변환 실패'};
    return {key: '', label: status ? String(value) : '미확인'};
  }
  function badge(value) {
    const quality = qualityInfo(value);
    return el('span', `td-badge ${quality.key}`, quality.label);
  }
  function safeImageUrl(value) {
    if (typeof value !== 'string' || !value) return null;
    try {
      const url = new URL(value, window.location.href);
      return url.origin === window.location.origin && ['http:', 'https:'].includes(url.protocol) ? url.href : null;
    } catch (_) { return null; }
  }
  function params(paginated = true) {
    const result = new URLSearchParams({worker_id: state.worker, process: state.process, scenario_id: state.mode, q: state.q});
    if (paginated) { result.set('page', String(state.page)); result.set('limit', String(state.limit)); }
    return result;
  }
  function syncUrl() {
    const url = new URL(window.location.href);
    params().forEach((value, key) => {
      if (value === 'all' || value === '' || (key === 'page' && value === '1') || (key === 'limit' && value === '25')) url.searchParams.delete(key);
      else url.searchParams.set(key, value);
    });
    window.history.replaceState(null, '', url);
    $('td-export-csv').href = `${API}/export.csv?${params(false)}`;
    $('td-export-zip').href = `${API}/export.zip?${params(false)}`;
  }
  function setOptions(id, entries, selected, allLabel) {
    const select = $(id);
    select.replaceChildren(new Option(allLabel, 'all'));
    (Array.isArray(entries) ? entries : []).forEach(entry => {
      if (entry && entry.id !== undefined && String(entry.id) !== 'all') select.add(new Option(String(entry.label ?? entry.id), String(entry.id)));
    });
    if (![...select.options].some(option => option.value === selected) && selected !== 'all') select.add(new Option(selected, selected));
    select.value = selected;
  }
  function listMessage(title, note, retry = false) {
    const cell = el('td', 'td-table-message');
    cell.colSpan = 6;
    cell.append(el('strong', '', title));
    if (note) cell.append(el('div', '', note));
    if (retry) {
      const button = el('button', 'td-button-secondary', '다시 불러오기');
      button.type = 'button';
      button.addEventListener('click', loadList);
      cell.append(button);
    }
    const row = el('tr'); row.append(cell); $('td-items').replaceChildren(row);
  }
  function renderBars(id, entries, title, color) {
    const host = $(id);
    const data = (Array.isArray(entries) ? entries : []).filter(entry => entry && (finite(entry.count) || 0) > 0);
    if (!data.length) { host.replaceChildren(el('p', 'td-chart-empty', '조회된 데이터가 없습니다.')); return; }
    const width = 430, rowHeight = 34, labelWidth = 125, chartWidth = 247;
    const svg = svgEl('svg', {viewBox: `0 0 ${width} ${data.length * rowHeight}`, role: 'img', 'aria-label': title});
    svg.append(svgEl('title', {}, title));
    const max = Math.max(...data.map(entry => Number(entry.count)), 1);
    data.forEach((entry, index) => {
      const y = index * rowHeight;
      const label = String(entry.label ?? entry.id ?? '미지정');
      const short = label.length > 13 ? `${label.slice(0, 12)}…` : label;
      const text = svgEl('text', {x: 0, y: y + 21, fill: '#62728b', 'font-size': 12}, short);
      text.append(svgEl('title', {}, label)); svg.append(text);
      svg.append(svgEl('rect', {x: labelWidth, y: y + 8, width: chartWidth, height: 17, rx: 4, fill: '#f0f3f9'}));
      const bar = svgEl('rect', {x: labelWidth, y: y + 8, width: Math.max(2, Number(entry.count) / max * chartWidth), height: 17, rx: 4, fill: color});
      bar.append(svgEl('title', {}, `${label}: ${count(entry.count)}건`)); svg.append(bar);
      svg.append(svgEl('text', {x: width - 1, y: y + 21, fill: '#465b7b', 'text-anchor': 'end', 'font-size': 12, 'font-weight': 650}, count(entry.count)));
    });
    host.replaceChildren(svg);
  }
  function renderRows(items) {
    const rows = (Array.isArray(items) ? items : []).map(item => {
      const row = el('tr', 'td-dataset-row');
      const nameCell = el('td');
      const name = el('button', 'td-dataset-name', item.dataset_name || item.dataset_id || '이름 없음');
      name.type = 'button'; name.setAttribute('aria-haspopup', 'dialog');
      name.addEventListener('click', () => openDetail(item, name));
      nameCell.append(name, el('small', 'td-row-note', `${count(item.frame_count)} 프레임 · ${count(item.zone_count)} 구역 · 맵 ${count(item.heightmap_count)}개`));
      row.append(nameCell, el('td', '', item.worker_name || item.worker_id || '미지정'), el('td', 'td-muted', processLabel(item.process)), el('td', '', item.mode_label || item.scenario_id || '미지정'));
      row.append(el('td', 'td-date', dateText(item.acquired_at)));
      const qualityCell = el('td'); qualityCell.append(badge(item.quality_status)); row.append(qualityCell);
      row.addEventListener('click', event => { if (!event.target.closest('button, a')) name.click(); });
      return row;
    });
    $('td-items').replaceChildren(...rows);
  }
  function renderList(data) {
    state.data = data;
    state.page = Math.max(1, finite(data.page) || state.page);
    state.pages = Math.max(1, finite(data.pages) || 1);
    const summary = data.summary || {};
    const total = finite(data.total) ?? finite(summary.total) ?? 0;
    const filtered = finite(data.filtered_count) ?? 0;
    $('td-kpi-total').textContent = count(total);
    $('td-kpi-filtered').textContent = count(filtered);
    $('td-kpi-workers').textContent = count(summary.worker_count);
    $('td-kpi-complete').textContent = count(summary.complete);
    $('td-kpi-review').textContent = count((finite(summary.needs_review) || 0) + (finite(summary.failed) || 0));
    $('td-kpi-review-note').textContent = summary.failed ? `품질 검토 · 변환 실패 ${count(summary.failed)}건 포함` : '품질 검토 대상';
    $('td-kpi-maps').textContent = count(summary.heightmap_count);
    const firstDate = dateText(summary.acquisition_date_min), lastDate = dateText(summary.acquisition_date_max);
    $('td-acquisition-range').textContent = firstDate === lastDate ? firstDate : `${firstDate} ~ ${lastDate}`;
    $('td-acquisition-note').textContent = `취득일 확인 ${count(summary.dated_count)} / ${count(filtered)}건 · 현재 조회 기준`;
    const options = data.options || {};
    setOptions('td-worker', options.workers, state.worker, '전체 작업자');
    setOptions('td-process', options.processes, state.process, '전체 공정');
    setOptions('td-mode', options.modes, state.mode, '전체 학습모드');
    renderBars('td-worker-chart', summary.by_worker, '현재 조회 결과의 작업자별 데이터 수', '#5275d9');
    renderBars('td-mode-chart', summary.by_mode, '현재 조회 결과의 학습모드별 데이터 수', '#42a69b');
    renderBars('td-process-chart', summary.by_process, '현재 조회 결과의 공정별 데이터 수', '#8d7ac9');
    if (data.available === false) listMessage('학습 데이터가 아직 생성되지 않았습니다.', data.build?.reason || '훈련 기록 CSV와 연결된 높이맵 텍스처의 변환 결과가 준비되면 표시됩니다.');
    else if (!Array.isArray(data.items) || !data.items.length) listMessage(total ? '조건에 맞는 학습 데이터가 없습니다.' : '등록된 학습 데이터가 없습니다.', total ? '작업자, 공정, 학습모드 또는 검색어를 바꿔 보세요.' : '훈련 기록을 변환하면 이곳에서 확인할 수 있습니다.');
    else renderRows(data.items);
    $('td-table-status').textContent = `전체 ${count(total)}건 중 ${count(filtered)}건 조회 · 데이터명을 누르면 상세 내용을 확인합니다.`;
    const start = filtered ? (state.page - 1) * state.limit + 1 : 0;
    const end = Math.min(state.page * state.limit, filtered);
    $('td-range').textContent = `${count(start)}–${count(end)} / ${count(filtered)}건`;
    $('td-page').textContent = `${count(state.page)} / ${count(state.pages)}`;
    $('td-prev').disabled = state.page <= 1;
    $('td-next').disabled = state.page >= state.pages || !filtered;
    for (const id of ['td-export-csv', 'td-export-zip']) $(id).setAttribute('aria-disabled', String(!filtered || data.available === false));
    syncUrl();
  }
  async function fetchJson(url, signal) {
    const response = await fetch(url, {signal, headers: {'Accept': 'application/json'}, cache: 'no-store'});
    if (!response.ok) throw new Error(response.status === 404 ? '데이터를 찾을 수 없습니다.' : `서버 응답을 확인해 주세요. (HTTP ${response.status})`);
    return response.json();
  }
  async function loadList() {
    state.listAbort?.abort();
    const abort = new AbortController(); state.listAbort = abort;
    $('td-list-panel').setAttribute('aria-busy', 'true');
    $('td-table-status').textContent = '학습 데이터를 불러오고 있습니다.';
    $('td-prev').disabled = true; $('td-next').disabled = true;
    syncUrl();
    try {
      const data = await fetchJson(`${API}?${params()}`, abort.signal);
      if (!abort.signal.aborted) renderList(data);
    } catch (error) {
      if (error.name !== 'AbortError' && !abort.signal.aborted) {
        listMessage('학습 데이터를 불러오지 못했습니다.', error.message || '연결 상태를 확인한 후 다시 시도해 주세요.', true);
        $('td-table-status').textContent = '조회에 실패했습니다. 다시 불러오기를 눌러 주세요.';
        ['td-export-csv', 'td-export-zip'].forEach(id => $(id).setAttribute('aria-disabled', 'true'));
      }
    } finally {
      if (state.listAbort === abort) $('td-list-panel').setAttribute('aria-busy', 'false');
    }
  }

  function addMeta(grid, label, value) {
    const group = el('div'); group.append(el('dt', '', label));
    const dd = el('dd');
    if (value instanceof Node) dd.append(value); else dd.textContent = value === undefined || value === null || value === '' ? '미기록' : String(value);
    group.append(dd); grid.append(group);
  }
  function detailSection(title, subtitle) {
    const section = el('section', 'td-panel td-detail-section');
    section.append(el('h3', '', title));
    if (subtitle) section.append(el('p', 'td-help', subtitle));
    return section;
  }
  function downloadLink(id) {
    const link = el('a', 'td-export', 'JSON 다운로드 ↓');
    link.href = `${API}/${encodeURIComponent(id)}/download`;
    return link;
  }
  function renderSummary(data, fallback) {
    const meta = {...(fallback || {}), ...(data.metadata || {})};
    const section = detailSection('데이터 요약', '원본 기록과 생성 결과를 함께 확인합니다.');
    const grid = el('dl', 'td-meta-grid');
    addMeta(grid, '작업자', meta.worker_name || meta.worker_id || '미지정');
    addMeta(grid, '공정', processLabel(meta.process));
    addMeta(grid, '시나리오 / 학습모드', meta.mode_label || meta.scenario_id || '미지정');
    addMeta(grid, '품질', badge(data.quality?.status || meta.quality_status));
    addMeta(grid, '취득일', dateText(meta.acquired_at));
    addMeta(grid, '프레임 / 구역', `${count(data.features?.frame_count ?? meta.frame_count)} / ${count(meta.zone_count ?? data.features?.zones?.length)}`);
    addMeta(grid, '높이맵 / JSON 크기', `${count(meta.heightmap_count ?? data.coating?.maps?.length)}개 / ${bytesText(meta.json_bytes)}`);
    section.append(grid);
    const quality = qualityInfo(data.quality?.status || meta.quality_status);
    const box = el('div', `td-quality-box ${quality.key}`);
    box.append(el('strong', '', `변환 품질 · ${quality.label}`));
    const issues = Array.isArray(data.quality?.issues) ? data.quality.issues : [];
    if (issues.length) {
      const list = el('ul'); issues.forEach(issue => list.append(el('li', '', typeof issue === 'string' ? issue : JSON.stringify(issue)))); box.append(list);
    } else box.append(el('div', '', quality.key === 'complete' ? '생성 시 기록된 품질 이슈가 없습니다. 도장 성능 평가와는 별도의 변환 검증 결과입니다.' : '추가 품질 사유가 기록되지 않았습니다.'));
    section.append(box);
    const source = data.provenance?.source_csv || {};
    const details = el('details', 'td-source'); details.append(el('summary', '', '원본 출처와 추적 정보'));
    const sourceGrid = el('dl');
    [['취득일 근거', data.provenance?.acquisition_date_source?.folder || data.provenance?.acquisition_date_source?.label], ['원본 CSV', source.filename], ['상대 경로', source.relative_path], ['SHA-256', source.sha256], ['원본 크기', bytesText(source.bytes)], ['연결 맵', `${count(data.provenance?.heightmaps?.length ?? meta.heightmap_count)}개`], ['데이터 ID', data.dataset_id || meta.dataset_id], ['JSON 스키마', data.schema_version]].forEach(([label, value]) => sourceGrid.append(el('dt', '', label), el('dd', '', value || '미기록')));
    details.append(sourceGrid); section.append(details);
    return section;
  }

  function chartEmpty(host, text) { host.replaceChildren(el('p', 'td-chart-empty', text)); }
  function chartTable(headers, rows) {
    const wrap = el('div', 'td-data-table'); const table = el('table'); const head = el('thead'); const tr = el('tr');
    headers.forEach(label => { const th = el('th', '', label); th.scope = 'col'; tr.append(th); }); head.append(tr); table.append(head);
    const body = el('tbody');
    rows.forEach(values => { const row = el('tr'); values.forEach(value => row.append(el('td', '', value))); body.append(row); });
    table.append(body); wrap.append(table); return wrap;
  }
  function renderHistogram(host, histogram) {
    const entries = (Array.isArray(histogram) ? histogram : []).filter(bin => bin && finite(bin.low_um) !== null && finite(bin.high_um) !== null && (finite(bin.count) !== null || finite(bin.percent) !== null));
    if (!entries.length) { chartEmpty(host, '이 맵의 두께 히스토그램이 생성되지 않았습니다.'); return; }
    const width = Math.max(400, entries.length * 24 + 65), height = 220, left = 42, top = 15, bottom = 45, chartHeight = height - top - bottom, chartWidth = width - left - 12;
    const usePercent = entries.every(bin => finite(bin.percent) !== null);
    const amounts = entries.map(bin => Math.max(0, finite(usePercent ? bin.percent : bin.count) || 0));
    const max = Math.max(...amounts, 1);
    const svg = svgEl('svg', {viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-label': '도막 두께 구간별 픽셀 분포'});
    svg.append(svgEl('title', {}, '도막 두께 구간별 픽셀 분포'));
    for (let tick = 0; tick <= 4; tick++) {
      const value = max * tick / 4, y = top + chartHeight - value / max * chartHeight;
      svg.append(svgEl('line', {x1: left, x2: width - 12, y1: y, y2: y, stroke: '#e9eef5'}));
      svg.append(svgEl('text', {x: left - 7, y: y + 3, fill: '#8b98aa', 'font-size': 9, 'text-anchor': 'end'}, `${numberFormat.format(value)}${usePercent ? '%' : ''}`));
    }
    const step = chartWidth / entries.length;
    entries.forEach((bin, i) => {
      const barHeight = amounts[i] / max * chartHeight;
      const rect = svgEl('rect', {x: left + i * step + 2, y: top + chartHeight - barHeight, width: Math.max(1, step - 4), height: barHeight, rx: 2, fill: '#5579d6'});
      rect.append(svgEl('title', {}, `${valueText(bin.low_um)}–${valueText(bin.high_um)} µm: ${valueText(bin.percent, '%')} · ${count(bin.count)} 픽셀`)); svg.append(rect);
      if (i % Math.max(1, Math.ceil(entries.length / 6)) === 0) svg.append(svgEl('text', {x: left + i * step + step / 2, y: height - 26, fill: '#7b8a9f', 'font-size': 9, 'text-anchor': 'middle'}, valueText(bin.low_um)));
    });
    svg.append(svgEl('text', {x: width / 2, y: height - 5, fill: '#8b98aa', 'font-size': 10, 'text-anchor': 'middle'}, '도막 두께 (µm)'));
    host.replaceChildren(svg, chartTable(['두께 구간 (µm)', '픽셀 수', '비율'], entries.map(bin => [`${valueText(bin.low_um)}–${valueText(bin.high_um)}`, count(bin.count), valueText(bin.percent, '%')])));
  }
  function quantile(summary, keys) {
    for (const key of keys) { const value = finite(summary[key]); if (value !== null) return value; }
    return null;
  }
  function zoneStats(zones) {
    return (Array.isArray(zones) ? zones : []).map(zone => {
      const summary = zone.summary || zone.thickness || zone;
      const reference = zone.reference || {};
      return {label: String(zone.zone_id ?? zone.zone_name ?? zone.id ?? '미지정'), color: '#5275d9', median: quantile(summary, ['median_um', 'median']), q1: quantile(summary, ['p25_um', 'q1_um', 'quartile_25_um']), q3: quantile(summary, ['p75_um', 'q3_um', 'quartile_75_um']), reference: {count: finite(reference.record_count) || 0, median: finite(reference.median_um), q1: finite(reference.p25_um), q3: finite(reference.p75_um)}};
    }).filter(zone => zone.median !== null);
  }
  function renderZones(host, zones, target) {
    const entries = zoneStats(zones);
    if (!entries.length) { chartEmpty(host, '구역별 두께 중앙값이 기록되지 않았습니다.'); return; }
    const hasReference = entries.some(zone => zone.reference.count > 0 && zone.reference.median !== null);
    const width = 500, left = 90, right = 30, top = 21, rowHeight = hasReference ? 47 : 35, height = top + entries.length * rowHeight + 35;
    const maximum = Math.max(...entries.flatMap(zone => [zone.median, zone.q1 || 0, zone.q3 || 0, zone.reference.median || 0, zone.reference.q1 || 0, zone.reference.q3 || 0]), finite(target) || 0, 1) * 1.1;
    const scale = value => left + Math.max(0, value) / maximum * (width - left - right);
    const svg = svgEl('svg', {viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-label': '구역별 도막 두께 중앙값과 사분위 범위'});
    svg.append(svgEl('title', {}, '구역별 도막 두께: 점은 중앙값, 가로 범위는 25–75 백분위수(IQR)'));
    for (let tick = 0; tick <= 4; tick++) {
      const value = maximum * tick / 4, x = scale(value);
      svg.append(svgEl('line', {x1: x, x2: x, y1: top - 7, y2: height - 30, stroke: '#e8edf5'}));
      svg.append(svgEl('text', {x, y: height - 12, fill: '#8b98aa', 'font-size': 10, 'text-anchor': 'middle'}, numberFormat.format(value)));
    }
    if ((finite(target) || 0) > 0) {
      const x = scale(Number(target));
      const line = svgEl('line', {x1: x, x2: x, y1: top - 7, y2: height - 30, stroke: '#d69c42', 'stroke-dasharray': '4 4'});
      line.append(svgEl('title', {}, `목표 두께 ${valueText(target)} µm`)); svg.append(line);
    }
    let hasIqr = false;
    entries.forEach((zone, index) => {
      const centerY = top + index * rowHeight + rowHeight / 2;
      const hasZoneReference = zone.reference.count > 0 && zone.reference.median !== null;
      const y = centerY - (hasZoneReference ? 6 : 0);
      const label = zone.label.length > 11 ? `${zone.label.slice(0, 10)}…` : zone.label;
      const text = svgEl('text', {x: left - 12, y: centerY + 4, fill: '#65768e', 'font-size': 11, 'text-anchor': 'end'}, label);
      text.append(svgEl('title', {}, zone.label)); svg.append(text);
      if (zone.q1 !== null && zone.q3 !== null && zone.q1 <= zone.q3) {
        hasIqr = true;
        svg.append(svgEl('rect', {x: scale(zone.q1), y: y - 7, width: Math.max(2, scale(zone.q3) - scale(zone.q1)), height: 14, rx: 3, fill: zone.color, opacity: .3}));
        svg.append(svgEl('line', {x1: scale(zone.q1), x2: scale(zone.q3), y1: y, y2: y, stroke: zone.color, 'stroke-width': 2}));
      }
      const dot = svgEl('circle', {cx: scale(zone.median), cy: y, r: 4.5, fill: zone.color, stroke: '#fff', 'stroke-width': 1.5});
      dot.append(svgEl('title', {}, `${zone.label}: 중앙값 ${valueText(zone.median)} µm, IQR ${valueText(zone.q1)}–${valueText(zone.q3)} µm`)); svg.append(dot);
      if (hasZoneReference) {
        const reference = zone.reference, refY = centerY + 8, refColor = '#9465c6';
        if (reference.q1 !== null && reference.q3 !== null && reference.q1 <= reference.q3) {
          svg.append(svgEl('rect', {x: scale(reference.q1), y: refY - 5, width: Math.max(2, scale(reference.q3) - scale(reference.q1)), height: 10, rx: 3, fill: refColor, opacity: .2}));
          svg.append(svgEl('line', {x1: scale(reference.q1), x2: scale(reference.q3), y1: refY, y2: refY, stroke: refColor, 'stroke-width': 2}));
        }
        const x = scale(reference.median);
        const diamond = svgEl('path', {d: `M ${x} ${refY - 5} L ${x + 5} ${refY} L ${x} ${refY + 5} L ${x - 5} ${refY} Z`, fill: refColor, stroke: '#fff', 'stroke-width': 1});
        diamond.append(svgEl('title', {}, `${zone.label} · 전문가 ${count(reference.count)}회 중앙값 분포: 중앙값 ${valueText(reference.median)} µm, IQR ${valueText(reference.q1)}–${valueText(reference.q3)} µm`)); svg.append(diamond);
      }
    });
    const note = el('p', 'td-help', hasIqr ? `● 현재 맵 중앙값 · 가로 범위: 도포 픽셀의 25–75 백분위수 (IQR)${finite(target) ? ' · 점선: 목표 두께' : ''} · 단위 µm` : '● 현재 맵 중앙값 · IQR은 생성 JSON에 기록되지 않았습니다. · 단위 µm');
    const comparisonNote = el('p', 'td-help', '◆ 전문가 N회 중앙값 분포: 같은 공정·시나리오·맵·구역의 회차별 중앙값을 모은 중앙값과 IQR입니다. 픽셀을 합친 분포와는 계산 기준이 다릅니다.');
    const headers = ['구역', '현재 중앙값 (µm)', '현재 25–75% (µm)'];
    if (hasReference) headers.push('전문가 N회 중앙값 분포 (µm)');
    const table = chartTable(headers, entries.map(zone => {
      const row = [zone.label, valueText(zone.median), zone.q1 !== null && zone.q3 !== null ? `${valueText(zone.q1)}–${valueText(zone.q3)}` : '미기록'];
      if (hasReference) row.push(zone.reference.count > 0 && zone.reference.median !== null ? `${count(zone.reference.count)}회 · ${valueText(zone.reference.median)} [${valueText(zone.reference.q1)}–${valueText(zone.reference.q3)}]` : '미기록');
      return row;
    }));
    host.replaceChildren(svg, note, ...(hasReference ? [comparisonNote] : []), table);
  }

  function renderCoating(data) {
    const coating = data.coating || {};
    const maps = Array.isArray(coating.maps) ? coating.maps : [];
    const notApplicable = data.metadata?.process === 'blasting' || (maps.length > 0 && maps.every(map => map.status === 'not_applicable'));
    const section = detailSection(notApplicable ? '텍스처 확인' : '도막 분석', notApplicable ? '블라스팅 기록의 원본 텍스처입니다. 도막 두께로 해석하지 않습니다.' : '연결된 최종 맵의 도막 상태입니다. 두께 통계와 히스토그램은 도포된 픽셀을 기준으로 계산합니다.');
    if (coating.semantics) section.append(el('p', 'td-help', coating.semantics));
    if (!maps.length) {
      section.append(el('p', 'td-chart-empty', coating.reason || '이 데이터에 연결된 도장 맵 또는 도막 분석 결과가 없습니다.'));
      return section;
    }
    const controls = el('div', 'td-map-controls');
    const field = el('div', 'td-field td-map-select');
    const label = el('label', '', '분석 맵'); label.htmlFor = 'td-map-select';
    const select = el('select'); select.id = 'td-map-select';
    maps.forEach((map, index) => select.add(new Option(`맵 ${map.map_num ?? index + 1} · ${map.filename || '파일명 미기록'}`, String(index))));
    field.append(label, select);
    const segment = el('div', 'td-segment'); segment.setAttribute('role', 'group'); segment.setAttribute('aria-label', '도장 맵 표시 방식');
    const decoded = el('button', '', '두께 환산 맵'); decoded.type = 'button';
    const raw = el('button', '', '원본 맵'); raw.type = 'button'; segment.append(decoded, raw);
    const tools = el('div', 'td-map-tools');
    const enlarge = el('a', 'td-map-enlarge', '이미지 크게 보기 ↗'); enlarge.id = 'td-image-enlarge'; enlarge.target = '_blank'; enlarge.rel = 'noopener noreferrer';
    tools.append(segment, enlarge); controls.append(field, tools); section.append(controls);
    const figure = el('figure', 'td-map-figure'); const stage = el('div', 'td-map-stage');
    const image = el('img'); image.id = 'td-map-image'; image.alt = '도장 맵';
    const imageMessage = el('p', 'td-map-message'); stage.append(image, imageMessage);
    const caption = el('figcaption'); const scale = el('div', 'td-map-scale'); scale.id = 'td-map-scale'; figure.append(stage, scale, caption); section.append(figure);
    const basisNote = el('div', 'td-map-notes'); section.append(basisNote);
    const metrics = el('div', 'td-coating-metrics'); section.append(metrics);
    const analysisGrid = el('div', 'td-analysis-grid');
    const histCard = el('div', 'td-analysis-card'); histCard.append(el('h4', '', '도막 두께 히스토그램'), el('p', 'td-help', '도포 픽셀 기준 · 두께 구간별 픽셀 수와 비율'));
    const histogram = el('div', 'td-chart-scroll'); histogram.id = 'td-histogram'; histCard.append(histogram);
    const zoneCard = el('div', 'td-analysis-card'); zoneCard.append(el('h4', '', '구역별 중앙값 · IQR'), el('p', 'td-help', '구역별 도막 두께의 대표값과 분포 범위'));
    const zones = el('div', 'td-chart-scroll'); zones.id = 'td-zone-chart'; zoneCard.append(zones); analysisGrid.append(histCard, zoneCard); section.append(analysisGrid);
    const mapNote = el('p', 'td-help'); mapNote.style.marginTop = '15px'; section.append(mapNote);
    let currentMap = maps[0];
    function showImage() {
      const decodedUrl = safeImageUrl(currentMap.thickness_image_url);
      const rawUrl = safeImageUrl(currentMap.image_url);
      decoded.disabled = !decodedUrl; raw.disabled = !rawUrl;
      if (state.mapView === 'decoded' && !decodedUrl && rawUrl) state.mapView = 'raw';
      if (state.mapView === 'raw' && !rawUrl && decodedUrl) state.mapView = 'decoded';
      decoded.setAttribute('aria-pressed', String(state.mapView === 'decoded'));
      raw.setAttribute('aria-pressed', String(state.mapView === 'raw'));
      const url = state.mapView === 'decoded' ? decodedUrl : rawUrl;
      enlarge.hidden = !url;
      if (url) enlarge.href = url; else enlarge.removeAttribute('href');
      image.hidden = !url; imageMessage.hidden = Boolean(url);
      imageMessage.textContent = '표시할 맵 이미지가 없습니다. 생성 JSON의 분석 수치는 아래에서 확인할 수 있습니다.';
      image.onload = () => { image.hidden = false; imageMessage.hidden = true; };
      image.onerror = () => { image.hidden = true; imageMessage.hidden = false; imageMessage.textContent = '맵 이미지를 불러오지 못했습니다. 아래 분석 수치와 JSON은 확인할 수 있습니다.'; };
      if (url) {
        // Assign the same-origin path, keeping generated artifact URLs inspectable.
        const parsed = new URL(url); image.setAttribute('src', `${parsed.pathname}${parsed.search}${parsed.hash}`);
      } else image.removeAttribute('src');
      image.alt = `맵 ${currentMap.map_num ?? Number(select.value) + 1} ${state.mapView === 'decoded' ? '도막 두께 환산 이미지' : '원본 인코딩 이미지'}`;
      caption.textContent = currentMap.status === 'not_applicable' ? '원본 텍스처 · 블라스팅 기록의 저장 이미지입니다. 도막 두께 분석을 적용하지 않습니다.' : state.mapView === 'decoded' ? '두께 환산 맵 · 검은 영역은 미도포 영역입니다. 표시 상한 이상의 두께는 같은 색으로 보이며, 통계와 히스토그램은 원래 두께를 사용합니다.' : '원본 맵 · 저장된 픽셀 인코딩 이미지입니다. 원본 색상 자체를 실제 도막 두께로 해석하지 마세요.';
      const colors = (Array.isArray(currentMap.preview_colors) ? currentMap.preview_colors : []).filter(color => /^#[a-f\d]{6}$/i.test(color));
      const max = finite(currentMap.preview_max_um);
      scale.hidden = state.mapView !== 'decoded' || colors.length < 2 || max === null || max <= 0;
      scale.replaceChildren();
      if (!scale.hidden) {
        const legend = svgEl('svg', {viewBox: '0 0 490 46', role: 'img', 'aria-label': `도막 두께 색상 범위: 0에서 ${valueText(max)} 마이크로미터 이상`});
        const defs = svgEl('defs'); const gradient = svgEl('linearGradient', {id: 'td-thickness-ramp', x1: '0%', x2: '100%'});
        colors.forEach((color, index) => gradient.append(svgEl('stop', {offset: `${index / (colors.length - 1) * 100}%`, 'stop-color': color})));
        defs.append(gradient); legend.append(defs, svgEl('rect', {x: 1, y: 1, width: 488, height: 12, rx: 3, fill: 'url(#td-thickness-ramp)'}));
        legend.append(svgEl('text', {x: 1, y: 31, fill: '#7a899e', 'font-size': 10}, '0 µm'));
        legend.append(svgEl('text', {x: 245, y: 31, fill: '#7a899e', 'font-size': 10, 'text-anchor': 'middle'}, `${valueText(max / 2)} µm`));
        legend.append(svgEl('text', {x: 489, y: 31, fill: '#7a899e', 'font-size': 10, 'text-anchor': 'end'}, `${valueText(max)} µm 이상`));
        scale.append(legend);
      }
    }
    function showMap() {
      currentMap = maps[Number(select.value)] || maps[0];
      showImage();
      const summary = currentMap.summary || {};
      basisNote.replaceChildren();
      const skipThickness = currentMap.status === 'not_applicable' || notApplicable;
      metrics.hidden = analysisGrid.hidden = mapNote.hidden = skipThickness;
      if (skipThickness) {
        basisNote.append(el('p', '', currentMap.reason || '블라스팅 텍스처에는 도막 두께 통계와 히스토그램을 적용하지 않습니다.'));
        return;
      }
      const domainKnown = currentMap.coverage_domain === 'official_zone_masks';
      if (currentMap.coverage_domain) basisNote.append(el('p', '', domainKnown ? '분석 영역: 공식 구역 마스크 · 도포율은 해당 영역의 전체 픽셀 대비 도포 픽셀 비율입니다.' : '공식 구역 영역을 확정하지 못해 도포율을 계산하지 않았습니다. 두께 분포는 이미지의 도포 픽셀을 기준으로 표시합니다.'));
      if (currentMap.validity_convention === 'legacy_rg_nonzero') basisNote.append(el('p', '', '이전 형식 맵: B 채널이 모두 0이므로 R 또는 G 채널이 0이 아닌 픽셀을 도포된 영역으로 해석했습니다.'));
      if (currentMap.status === 'not_applicable') basisNote.append(el('p', '', currentMap.reason || '이 맵에는 도막 두께 분석을 적용하지 않습니다.'));
      if (finite(summary.target_thickness_um) === null) basisNote.append(el('p', '', '목표 두께가 원본에 명시되지 않아 목표 대비 적합 비율은 계산하지 않았습니다.'));
      else basisNote.append(el('p', '', `목표 범위 비율: 목표 두께 ±${valueText(summary.target_tolerance_percent ?? 10)}% 범위에 포함되는 도포 픽셀 비율입니다.`));
      const reasons = {
        official_zone_catalog_unavailable: '공식 구역 정보를 불러오지 못했습니다.', scenario_model_unavailable: '시나리오의 공식 모델 정보가 없습니다.',
        official_zones_unavailable: '해당 맵의 공식 구역 정보가 없습니다.', official_zone_mask_unavailable: '일부 공식 구역 마스크를 사용할 수 없습니다.',
        official_target_domain_empty: '공식 도장 대상 영역이 비어 있습니다.', heightmap_unavailable: '원본 도장 맵을 읽을 수 없습니다.',
      };
      if (currentMap.reason && currentMap.status !== 'not_applicable') basisNote.append(el('p', '', reasons[currentMap.reason] || currentMap.reason));
      metrics.replaceChildren(...[
        ['도포율', valueText(summary.coverage_percent, '%')], ['평균 두께', valueText(summary.mean_um, ' µm')], ['중앙값', valueText(summary.median_um, ' µm')],
        ['목표 두께', valueText(summary.target_thickness_um, ' µm')], [`목표 ±${valueText(summary.target_tolerance_percent ?? 10)}% 비율`, valueText(summary.within_target_percent, '%')], ['두께 변동계수', valueText(summary.cv_percent, '%')],
      ].map(([labelText, value]) => { const metric = el('div', 'td-coating-metric'); metric.append(el('span', '', labelText), el('strong', '', value)); return metric; }));
      renderHistogram(histogram, currentMap.histogram);
      renderZones(zones, currentMap.zones, summary.target_thickness_um);
      mapNote.textContent = `분석 대상 ${valueText(summary.target_pixels)} 픽셀 · 도포 ${valueText(summary.painted_pixels)} 픽셀 · 도포 픽셀의 10–90 백분위 두께 ${valueText(summary.p10_um)}–${valueText(summary.p90_um)} µm`;
      const reference = currentMap.reference;
      if (reference && finite(reference.record_count) > 0 && finite(reference.median_um) !== null) mapNote.append(el('span', '', ` · 전문가 ${count(reference.record_count)}회 중앙값 분포: ${valueText(reference.median_um)} µm (IQR ${valueText(reference.p25_um)}–${valueText(reference.p75_um)} µm, 회차별 중앙값 기준)`));
    }
    decoded.addEventListener('click', () => { state.mapView = 'decoded'; showImage(); });
    raw.addEventListener('click', () => { state.mapView = 'raw'; showImage(); });
    select.addEventListener('change', showMap);
    state.mapView = 'decoded'; showMap();
    return section;
  }

  function jsonBranch(key, value) {
    if (value !== null && typeof value === 'object') {
      const details = el('details');
      const entries = Object.entries(value);
      const summary = el('summary', '', `${key} · ${Array.isArray(value) ? '배열' : '객체'} (${count(entries.length)}개)`);
      details.append(summary);
      let rendered = false;
      details.addEventListener('toggle', () => {
        if (!details.open || rendered) return;
        rendered = true;
        // Populate only this branch when expanded, avoiding a large eager DOM.
        const children = el('div'); entries.forEach(([childKey, child]) => children.append(jsonBranch(childKey, child))); details.append(children);
      });
      return details;
    }
    const leaf = el('div', 'td-json-leaf'); leaf.append(el('span', 'td-json-key', `${key}: `));
    const text = JSON.stringify(value);
    leaf.append(el('span', value === null ? 'td-json-null' : '', text === undefined ? String(value) : text));
    return leaf;
  }
  function renderJson(data, fallbackId) {
    const section = el('section', 'td-panel td-detail-section');
    const actions = el('div', 'td-json-actions'); actions.append(el('h3', '', '생성 JSON'), downloadLink(data.dataset_id || fallbackId)); section.append(actions);
    section.append(el('p', 'td-help', '요약 화면과 동일한 실제 생성 결과입니다. 트리에서 항목을 펼치거나 원본 JSON을 다운로드할 수 있습니다.'));
    const treeDetails = el('details', 'td-json-view'); treeDetails.append(el('summary', '', 'JSON 트리 펼치기'));
    const tree = el('div', 'td-json-tree'); treeDetails.append(tree);
    let treeReady = false;
    treeDetails.addEventListener('toggle', () => {
      if (!treeDetails.open || treeReady) return;
      treeReady = true; Object.entries(data).forEach(([key, value]) => tree.append(jsonBranch(key, value)));
    });
    const rawDetails = el('details', 'td-json-view'); rawDetails.append(el('summary', '', '원본 JSON 보기'));
    let rawReady = false;
    rawDetails.addEventListener('toggle', () => {
      if (!rawDetails.open || rawReady) return;
      rawReady = true; rawDetails.append(el('pre', 'td-json-raw', JSON.stringify(data, null, 2)));
    });
    section.append(treeDetails, rawDetails); return section;
  }
  async function openDetail(item, trigger) {
    state.detailAbort?.abort();
    const abort = new AbortController(); state.detailAbort = abort;
    state.returnFocus = trigger || document.activeElement;
    $('td-detail-title').textContent = item.dataset_name || item.dataset_id || '학습 데이터 상세';
    const body = $('td-detail-content'); body.replaceChildren(el('p', 'td-detail-loading', '상세 데이터를 불러오는 중입니다.')); body.scrollTop = 0;
    const dialog = $('td-detail');
    if (!dialog.open) {
      document.body.classList.add('td-modal-open');
      if (typeof dialog.showModal === 'function') dialog.showModal(); else dialog.setAttribute('open', '');
    }
    $('td-detail-close').focus();
    try {
      const data = await fetchJson(`${API}/${encodeURIComponent(item.dataset_id)}`, abort.signal);
      if (abort.signal.aborted || !dialog.open) return;
      $('td-detail-title').textContent = data.metadata?.dataset_name || item.dataset_name || data.dataset_id || '학습 데이터 상세';
      body.replaceChildren(renderSummary(data, item), renderCoating(data), renderJson(data, item.dataset_id));
      body.scrollTop = 0;
    } catch (error) {
      if (error.name === 'AbortError' || abort.signal.aborted) return;
      const message = el('div', 'td-detail-loading'); message.setAttribute('role', 'alert');
      message.append(el('p', '', '상세 데이터를 불러오지 못했습니다.'), el('p', 'td-help', error.message));
      const retry = el('button', 'td-button-secondary', '다시 불러오기'); retry.type = 'button'; retry.addEventListener('click', () => openDetail(item, trigger)); message.append(retry); body.replaceChildren(message);
    }
  }
  function closeDetail() {
    const dialog = $('td-detail');
    if (typeof dialog.close === 'function') dialog.close(); else { dialog.removeAttribute('open'); afterClose(); }
  }
  function afterClose() {
    state.detailAbort?.abort(); document.body.classList.remove('td-modal-open');
    if (state.returnFocus?.isConnected) state.returnFocus.focus();
  }

  $('td-search').value = state.q; $('td-limit').value = String(state.limit);
  $('td-filter-form').addEventListener('submit', event => {
    event.preventDefault(); state.worker = $('td-worker').value; state.process = $('td-process').value; state.mode = $('td-mode').value; state.q = $('td-search').value.trim(); state.page = 1; loadList();
  });
  ['td-worker', 'td-process', 'td-mode'].forEach(id => $(id).addEventListener('change', () => $('td-filter-form').requestSubmit()));
  $('td-reset').addEventListener('click', () => {
    state.worker = state.process = state.mode = 'all'; state.q = ''; state.page = 1;
    $('td-worker').value = $('td-process').value = $('td-mode').value = 'all'; $('td-search').value = ''; loadList();
  });
  $('td-refresh').addEventListener('click', loadList);
  $('td-limit').addEventListener('change', () => { state.limit = Number($('td-limit').value); state.page = 1; loadList(); });
  $('td-prev').addEventListener('click', () => { if (state.page > 1) { state.page--; loadList(); } });
  $('td-next').addEventListener('click', () => { if (state.page < state.pages) { state.page++; loadList(); } });
  $('td-detail-close').addEventListener('click', closeDetail);
  $('td-detail').addEventListener('close', afterClose);
  $('td-detail').addEventListener('click', event => {
    if (event.target !== $('td-detail')) return;
    const rect = $('td-detail').getBoundingClientRect();
    if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) closeDetail();
  });
  loadList();
})();
