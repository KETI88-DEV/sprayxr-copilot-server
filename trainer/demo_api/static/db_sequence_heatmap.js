(() => {
  'use strict';
  const NS = 'http://www.w3.org/2000/svg';
  const format = new Intl.NumberFormat('ko-KR', {maximumFractionDigits: 1});
  let serial = 0;
  const finite = value => typeof value === 'number' && Number.isFinite(value) ? value : null;
  const valueText = (value, unit = '') => finite(value) === null ? '미기록' : `${format.format(value)}${unit}`;
  const countText = value => finite(value) === null ? '미기록' : new Intl.NumberFormat('ko-KR', {maximumFractionDigits: 0}).format(value);
  const el = (tag, cls, text) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = String(text);
    return node;
  };
  const svgEl = (tag, attributes = {}, text) => {
    const node = document.createElementNS(NS, tag);
    Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, String(value)));
    if (text !== undefined) node.textContent = String(text);
    return node;
  };
  const reasonLabels = {
    invalid_base64: '저장된 Base64 이미지가 올바른 형식이 아닙니다.', malformed_base64: '저장된 Base64 이미지가 올바른 형식이 아닙니다.',
    invalid_image: '저장된 이미지를 읽을 수 없습니다.', invalid_image_data: '저장된 이미지를 읽을 수 없습니다.',
    image_decode_failed: '저장된 Base64 이미지를 해석하지 못했습니다.', heatmap_decode_failed: '저장된 Base64 히트맵을 해석하지 못했습니다.',
    heatmap_unavailable: '저장된 히트맵 이미지가 없습니다.', heightmap_unavailable: '원본 도장 맵을 읽을 수 없습니다.',
    heatmaps_unavailable: '이 시퀀스에 저장된 히트맵이 없습니다.', no_heatmaps: '이 시퀀스에 저장된 히트맵이 없습니다.',
    official_zone_catalog_unavailable: '공식 구역 정보를 불러오지 못했습니다.', scenario_model_unavailable: '시나리오의 공식 모델 정보가 없습니다.',
    official_zones_unavailable: '해당 맵의 공식 구역 정보가 없습니다.', official_zone_mask_unavailable: '일부 공식 구역 마스크를 사용할 수 없습니다.',
    official_target_domain_empty: '크롭과 공식 도장 대상 영역이 겹치지 않습니다.', crop_outside_target: '크롭과 공식 도장 대상 영역이 겹치지 않습니다.',
    no_painted_pixels: '이 크롭의 분석 영역에 도포된 픽셀이 없습니다.', empty_painted_domain: '이 크롭의 분석 영역에 도포된 픽셀이 없습니다.',
    target_thickness_unavailable: '세션에 목표 두께가 기록되지 않았습니다.', session_target_unavailable: '세션에 목표 두께가 기록되지 않았습니다.',
    not_applicable: '블라스팅 텍스처에는 도막 두께 분석을 적용하지 않습니다.',
    unsupported_encoding: '지원하지 않는 이미지 인코딩이므로 도막 두께를 해석하지 않았습니다.',
    invalid_crop: '저장된 크롭 좌표 또는 크기가 올바르지 않습니다.', invalid_dimensions: '저장된 이미지 크기가 올바르지 않습니다.',
    unsupported_heatmap_encoding: '지원하는 PNG Base64 이미지 인코딩이 아닙니다.', missing_heatmap_data: '저장된 히트맵 이미지 데이터가 없습니다.',
    encoded_heatmap_exceeds_limit: '저장된 Base64 이미지가 표시 가능한 크기를 초과했습니다.', decoded_heatmap_exceeds_limit: '해석한 이미지가 표시 가능한 크기를 초과했습니다.',
    invalid_heatmap_base64: '저장된 Base64 이미지가 올바른 형식이 아닙니다.', invalid_heatmap_png: '저장된 PNG 이미지가 손상되었거나 올바른 형식이 아닙니다.',
    image_dimensions_exceed_limit: '저장된 이미지의 가로 또는 세로 크기가 표시 한도를 초과했습니다.', image_dimensions_mismatch: '기록된 이미지 크기와 실제 PNG 크기가 달라 공식 대상 영역을 확정하지 못했습니다.',
    invalid_crop_geometry: '크롭 위치 또는 크기가 올바르지 않아 공식 대상 영역을 확정하지 못했습니다.', invalid_map_number: '저장된 맵 번호가 올바르지 않아 공식 구역을 확인할 수 없습니다.',
    invalid_heatmap_item: '저장된 히트맵 항목이 올바른 형식이 아닙니다.', physical_thickness_not_applicable: '이 공정의 원본 이미지는 도막 두께로 해석하지 않습니다.',
    unsupported_heatmap_format: '저장된 맵이 지원하는 v2B8 도막 형식이 아니므로 원본 이미지만 표시합니다.', unsupported_physical_png_mode: '이 PNG 색상 형식은 도막 두께로 해석할 수 없습니다.',
    unsupported_thickness_scale: '저장된 두께 환산 배율을 해석할 수 없습니다.', official_target_domain_empty_in_crop: '크롭과 공식 도장 대상 영역이 겹치지 않습니다.',
    invalid_target_thickness_ignored: '세션 목표 두께가 올바르지 않아 기본 목표로 계산했습니다.',
  };
  function reasonText(reason) {
    if (!reason) return '';
    const text = typeof reason === 'string' ? reason : String(reason.message || reason.reason || reason.code || '분석 정보를 확인할 수 없습니다.');
    if (reasonLabels[text]) return reasonLabels[text];
    if (text.startsWith('coverage_unknown:')) return '공식 대상 영역을 확정하지 못해 도포율을 계산하지 않았습니다. 두께 분포는 크롭 이미지의 도포 픽셀 기준입니다.';
    if (text.startsWith('official_zone_mask_unavailable:')) return `공식 구역 마스크를 사용할 수 없습니다: ${text.slice(text.indexOf(':') + 1).trim().replace(/^zone\s+/i, '구역 ')}`;
    if (/base64/i.test(text) && /invalid|malformed|decode|error|fail/i.test(text)) return '저장된 Base64 이미지가 손상되었거나 올바른 형식이 아닙니다.';
    return text;
  }
  function sameOriginUrl(value) {
    if (typeof value !== 'string' || !value.trim()) return null;
    try {
      const parsed = new URL(value, location.href);
      return parsed.origin === location.origin && ['http:', 'https:'].includes(parsed.protocol) && !parsed.username && !parsed.password ? parsed : null;
    } catch { return null; }
  }
  function tableDetails(headers, rows) {
    const details = el('details', 'sh-data'); details.append(el('summary', '', '수치 표 보기'));
    const wrap = el('div', 'sh-table'), table = el('table'), head = el('thead'), heading = el('tr'), body = el('tbody');
    headers.forEach(text => { const cell = el('th', '', text); cell.scope = 'col'; heading.append(cell); });
    head.append(heading); table.append(head, body);
    rows.forEach(values => { const row = el('tr'); values.forEach(text => row.append(el('td', '', text))); body.append(row); });
    wrap.append(table); details.append(wrap); return details;
  }
  // 6구간 개수는 서버가 순서대로 준다. 정상 구간은 셋째 칸이다.
  const bandCounts = summary => { const values = Object.values(summary.six_band_counts || {}).map(value => finite(value) ?? 0); return values.length === 6 ? values : [0, 0, 0, 0, 0, 0]; };
  function renderBands(host, summary, labels) {
    const counts = bandCounts(summary), total = counts.reduce((sum, count) => sum + count, 0), names = Array.isArray(labels) && labels.length === 6 ? labels : counts.map((_, index) => `${index + 1}구간`);
    if (!total) { host.replaceChildren(el('p', 'sh-empty', '이 크롭에 도포 표본이 없어 6구간을 계산하지 않았습니다.')); return; }
    host.replaceChildren(tableDetails(['목표 대비 두께', '픽셀 수', '비율'], counts.map((count, index) => [names[index], countText(count), valueText(count / total * 100, '%')])));
    host.firstChild.open = true;
  }
  function renderHistogram(host, histogram) {
    const bins = (Array.isArray(histogram) ? histogram : []).filter(bin => bin && finite(bin.low_um) !== null && finite(bin.high_um) !== null && bin.high_um >= bin.low_um && (finite(bin.count) !== null || finite(bin.percent) !== null));
    if (!bins.length) { host.replaceChildren(el('p', 'sh-empty', '이 크롭의 두께 히스토그램이 없습니다.')); return; }
    const width = Math.max(400, bins.length * 24 + 65), height = 220, left = 43, top = 15, chartHeight = 160, chartWidth = width - left - 12;
    const usePercent = bins.every(bin => finite(bin.percent) !== null);
    const amounts = bins.map(bin => Math.max(0, (usePercent ? bin.percent : bin.count) ?? 0));
    const max = Math.max(...amounts, 1), step = chartWidth / bins.length;
    const svg = svgEl('svg', {viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-label': '크롭 내 도포 픽셀의 도막 두께 분포'});
    svg.append(svgEl('title', {}, '도막 두께 구간별 도포 픽셀 분포'));
    for (let tick = 0; tick <= 4; tick++) {
      const value = max * tick / 4, y = top + chartHeight - value / max * chartHeight;
      svg.append(svgEl('line', {x1: left, x2: width - 12, y1: y, y2: y, stroke: '#e9eef5'}), svgEl('text', {x: left - 7, y: y + 3, fill: '#8291a6', 'font-size': 9, 'text-anchor': 'end'}, `${format.format(value)}${usePercent ? '%' : ''}`));
    }
    bins.forEach((bin, index) => {
      const barHeight = amounts[index] / max * chartHeight;
      const rect = svgEl('rect', {x: left + index * step + 2, y: top + chartHeight - barHeight, width: Math.max(1, step - 4), height: barHeight, rx: 2, fill: '#5579d6'});
      rect.append(svgEl('title', {}, `${valueText(bin.low_um)}–${valueText(bin.high_um)} µm · ${countText(bin.count)} 픽셀 · ${valueText(bin.percent, '%')}`));
      svg.append(rect);
      if (index % Math.max(1, Math.ceil(bins.length / 6)) === 0) svg.append(svgEl('text', {x: left + index * step + step / 2, y: height - 26, fill: '#7b8a9f', 'font-size': 9, 'text-anchor': 'middle'}, valueText(bin.low_um)));
    });
    svg.append(svgEl('text', {x: width / 2, y: height - 5, fill: '#8291a6', 'font-size': 10, 'text-anchor': 'middle'}, '도막 두께 (µm)'));
    host.replaceChildren(svg, tableDetails(['두께 구간 (µm)', '픽셀 수', '비율'], bins.map(bin => [`${valueText(bin.low_um)}–${valueText(bin.high_um)}`, countText(bin.count), valueText(bin.percent, '%')])));
  }
  function renderZones(host, zones, target) {
    const entries = (Array.isArray(zones) ? zones : []).filter(zone => zone && zone.summary && typeof zone.summary === 'object').map(zone => ({
      label: String(zone.zone_id ?? '미지정'), color: /^#[\da-f]{6}$/i.test(zone.hex_color || '') ? zone.hex_color : '#5275d9', summary: zone.summary,
    }));
    const plotted = entries.filter(zone => finite(zone.summary.median_um) !== null);
    if (!plotted.length) { host.replaceChildren(el('p', 'sh-empty', '이 크롭에 구역별 두께 중앙값이 기록되지 않았습니다.')); return; }
    const width = 500, left = 93, right = 30, top = 18, rowHeight = 35, height = top + plotted.length * rowHeight + 40;
    const maximum = Math.max(1, finite(target) ?? 0, ...plotted.flatMap(zone => [zone.summary.median_um, finite(zone.summary.p25_um) ?? 0, finite(zone.summary.p75_um) ?? 0])) * 1.1;
    const scale = value => left + Math.max(0, value) / maximum * (width - left - right);
    const svg = svgEl('svg', {viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-label': '크롭 내 구역별 도막 두께 중앙값과 25–75 백분위 범위'});
    svg.append(svgEl('title', {}, '점: 중앙값 · 가로 범위: 25–75 백분위수 (IQR) · 단위 µm'));
    for (let tick = 0; tick <= 4; tick++) {
      const value = maximum * tick / 4, x = scale(value);
      svg.append(svgEl('line', {x1: x, x2: x, y1: top, y2: height - 34, stroke: '#e9eef5'}), svgEl('text', {x, y: height - 14, fill: '#8291a6', 'font-size': 10, 'text-anchor': 'middle'}, format.format(value)));
    }
    if (finite(target) !== null && target >= 0) {
      const x = scale(target), line = svgEl('line', {x1: x, x2: x, y1: top, y2: height - 34, stroke: '#d69c42', 'stroke-dasharray': '4 4'});
      line.append(svgEl('title', {}, `세션 목표 두께 ${valueText(target)} µm`)); svg.append(line);
    }
    plotted.forEach((zone, index) => {
      const y = top + index * rowHeight + rowHeight / 2, summary = zone.summary;
      const text = svgEl('text', {x: left - 12, y: y + 4, fill: '#65768e', 'font-size': 11, 'text-anchor': 'end'}, zone.label.length > 11 ? `${zone.label.slice(0, 10)}…` : zone.label);
      text.append(svgEl('title', {}, zone.label)); svg.append(text);
      const q1 = finite(summary.p25_um), q3 = finite(summary.p75_um);
      if (q1 !== null && q3 !== null && q1 <= q3) {
        svg.append(svgEl('rect', {x: scale(q1), y: y - 7, width: Math.max(2, scale(q3) - scale(q1)), height: 14, rx: 3, fill: zone.color, opacity: .25}), svgEl('line', {x1: scale(q1), x2: scale(q3), y1: y, y2: y, stroke: zone.color, 'stroke-width': 2}));
      }
      const dot = svgEl('circle', {cx: scale(summary.median_um), cy: y, r: 4.5, fill: zone.color, stroke: '#fff', 'stroke-width': 1.5});
      dot.append(svgEl('title', {}, `${zone.label} · 중앙값 ${valueText(summary.median_um)} µm · IQR ${valueText(q1)}–${valueText(q3)} µm`)); svg.append(dot);
    });
    host.replaceChildren(svg, el('p', 'sh-help', `● 중앙값 · 가로 범위: 도포 픽셀의 25–75 백분위수 (IQR)${finite(target) !== null ? ' · 점선: 세션 목표 두께' : ''} · 단위 µm`), tableDetails(['구역', '중앙값 (µm)', '25–75% (µm)', '크롭 내 도포율', '크롭 내 미도장'], entries.map(zone => [zone.label, valueText(zone.summary.median_um), `${valueText(zone.summary.p25_um)}–${valueText(zone.summary.p75_um)}`, valueText(zone.summary.coverage_percent, '%'), finite(zone.summary.sampled_unpainted_fraction) === null ? '미기록' : valueText(zone.summary.sampled_unpainted_fraction * 100, '%')])));
  }
  function mapLabel(map, offset) {
    const number = map.map_num ?? offset + 1;
    const label = typeof map.label === 'string' ? map.label.trim().replace(/\bheight\s*map\b/gi, '높이맵').replace(/\bheat\s*map\b/gi, '히트맵').replace(/\bmap\b/gi, '맵').replace(/\bzone\b/gi, '구역').replace(/\bcrop\b/gi, '크롭') : '';
    return label && (label === `맵 ${number}` || label.startsWith(`맵 ${number} ·`)) ? label : label ? `맵 ${number} · ${label}` : `맵 ${number}`;
  }
  function render(host, url, {signal} = {}) {
    if (!host || signal?.aborted) return () => {};
    host.classList.add('seq-heatmap');
    const id = `sequence-heatmap-${++serial}`, endpoint = sameOriginUrl(url);
    const heading = el('h3', '', '시퀀스 히트맵 · 도막 분석'); heading.id = `${id}-title`; host.setAttribute('aria-labelledby', heading.id);
    const semantics = el('p', 'sh-help', '저장된 시퀀스 크롭의 원본 이미지와 분석 가능한 도막 수치를 확인합니다.');
    const sourceNote = el('p', 'sh-help'); sourceNote.hidden = true;
    const controls = el('div', 'sh-controls'), field = el('div', 'sh-map-field'), label = el('label', '', '분석 맵'), select = el('select');
    label.htmlFor = select.id = `${id}-map`; select.disabled = true; field.append(label, select);
    const tools = el('div', 'sh-tools'), segment = el('div', 'sh-segment'); segment.setAttribute('role', 'group'); segment.setAttribute('aria-label', '히트맵 표시 방식');
    const decoded = el('button', '', '두께 환산 맵'), raw = el('button', '', '원본 맵');
    [decoded, raw].forEach(button => { button.type = 'button'; button.disabled = true; button.setAttribute('aria-pressed', 'false'); });
    segment.append(decoded, raw);
    const enlarge = el('a', '', '이미지 크게 보기 ↗'); enlarge.target = '_blank'; enlarge.rel = 'noopener noreferrer'; enlarge.hidden = true;
    tools.append(segment, enlarge); controls.append(field, tools);
    const status = el('p', 'sh-status'); status.setAttribute('role', 'status'); status.setAttribute('aria-live', 'polite');
    const body = el('div'); body.hidden = true;
    const figure = el('figure'), stage = el('div', 'sh-stage'), imageMessage = el('p', 'sh-image-message'), scale = el('div', 'sh-scale'), caption = el('figcaption');
    stage.append(imageMessage); figure.append(stage, scale, caption);
    const notes = el('div', 'sh-notes'), metrics = el('div', 'sh-metrics'), graphs = el('div', 'sh-graphs');
    const histCard = el('div', 'sh-chart-card'), histogram = el('div', 'sh-chart-scroll'), zoneCard = el('div', 'sh-chart-card'), zones = el('div', 'sh-chart-scroll'), bandCard = el('div', 'sh-chart-card'), bandHost = el('div', 'sh-chart-scroll');
    histCard.append(el('h4', '', '도막 두께 히스토그램'), el('p', 'sh-help', '크롭 내 도포 픽셀 기준 · 두께 구간별 픽셀 수와 비율'), histogram);
    zoneCard.append(el('h4', '', '구역별 중앙값 · IQR'), el('p', 'sh-help', '크롭에 포함된 구역의 도포 픽셀 기준'), zones); bandCard.append(el('h4', '', '목표 대비 6구간'), el('p', 'sh-help', '크롭 안 도포 표본 기준 · 보정 두께를 목표에 대한 비율로 나눈 구간'), bandHost); graphs.append(histCard, zoneCard, bandCard);
    const footnote = el('p', 'sh-help sh-footnote'); body.append(figure, notes, metrics, graphs, footnote); host.replaceChildren(heading, semantics, sourceNote, controls, status, body);
    let request, requestVersion = 0, imageVersion = 0, stopped = false, data = null, view = 'decoded', image = null, selectedIndex = 0;
    function alive(version) { return !stopped && !signal?.aborted && version === requestVersion; }
    function clearImage() {
      imageVersion++;
      if (image) { image.onload = image.onerror = null; image.removeAttribute('src'); image.remove(); image = null; }
      enlarge.hidden = true; enlarge.removeAttribute('href');
    }
    function dispose() { if (stopped) return; stopped = true; requestVersion++; request?.abort(); clearImage(); signal?.removeEventListener('abort', dispose); }
    signal?.addEventListener('abort', dispose, {once: true});
    function setStatus(message, error = false) { status.replaceChildren(); status.textContent = message; status.className = `sh-status${error ? ' sh-error' : ''}`; status.hidden = !message; }
    function showImage() {
      clearImage();
      if (!data || stopped) return;
      const blasting = data.process === 'blasting', physical = data.process === 'spray' && ['available', 'partial'].includes(data.analysis?.status);
      const rawUrl = sameOriginUrl(data.image_url), decodedUrl = physical ? sameOriginUrl(data.thickness_image_url) : null;
      decoded.disabled = !decodedUrl; raw.disabled = !rawUrl; decoded.hidden = !physical;
      if (view === 'decoded' && !decodedUrl) view = 'raw';
      if (view === 'raw' && !rawUrl && decodedUrl) view = 'decoded';
      decoded.setAttribute('aria-pressed', String(view === 'decoded' && !!decodedUrl)); raw.setAttribute('aria-pressed', String(view === 'raw' && !!rawUrl));
      const imageUrl = view === 'decoded' ? decodedUrl : rawUrl;
      imageMessage.hidden = false;
      imageMessage.textContent = imageUrl ? '히트맵 이미지를 불러오고 있습니다…' : '표시할 맵 이미지가 없습니다. 아래 분석 수치와 원본 파라미터는 계속 확인할 수 있습니다.';
      if (imageUrl) {
        const version = imageVersion, node = el('img'); image = node; node.hidden = true;
        node.alt = `${mapLabel((Array.isArray(data.maps) ? data.maps : []).find(map => map && map.index === selectedIndex) || {}, selectedIndex)} ${view === 'decoded' ? '누적 도막 두께 환산 이미지' : '저장된 원본 이미지'}`;
        node.onload = () => { if (stopped || version !== imageVersion) return; node.hidden = false; imageMessage.hidden = true; enlarge.href = imageUrl.href; enlarge.hidden = false; };
        node.onerror = () => { if (stopped || version !== imageVersion) return; node.hidden = true; imageMessage.hidden = false; imageMessage.textContent = '이미지를 불러오지 못했습니다. 저장된 Base64 이미지가 손상되었거나 이미지 응답을 읽을 수 없습니다. 분석 수치와 원본 파라미터는 계속 확인할 수 있습니다.'; enlarge.hidden = true; };
        stage.prepend(node); node.src = imageUrl.href;
      }
      caption.textContent = blasting ? '원본 텍스처 · 블라스팅 기록의 저장 이미지입니다. 도막 두께 분석을 적용하지 않습니다.' : view === 'decoded' ? '두께 환산 맵 · 검은 영역은 미도포입니다. 표시 상한 이상의 두께는 같은 색으로 보이며, 통계와 히스토그램은 원래 두께를 사용합니다.' : '원본 맵 · 저장된 픽셀 인코딩 이미지입니다. 원본 색상 자체를 실제 도막 두께로 해석하지 마세요.';
      const colors = (Array.isArray(data.analysis?.preview_colors) ? data.analysis.preview_colors : []).filter(color => typeof color === 'string' && /^#[a-f\d]{6}$/i.test(color));
      const max = finite(data.analysis?.preview_max_um); scale.replaceChildren(); scale.hidden = view !== 'decoded' || !decodedUrl || colors.length < 2 || max === null || max <= 0;
      if (!scale.hidden) {
        const legend = svgEl('svg', {viewBox: '0 0 490 46', role: 'img', 'aria-label': `도막 두께 색상 범위: 0에서 ${valueText(max)} 마이크로미터 이상`});
        const defs = svgEl('defs'), gradientId = `${id}-ramp`, gradient = svgEl('linearGradient', {id: gradientId, x1: '0%', x2: '100%'});
        colors.forEach((color, index) => gradient.append(svgEl('stop', {offset: `${index / (colors.length - 1) * 100}%`, 'stop-color': color})));
        defs.append(gradient); legend.append(defs, svgEl('rect', {x: 1, y: 1, width: 488, height: 12, rx: 3, fill: `url(#${gradientId})`}), svgEl('text', {x: 1, y: 31, fill: '#7a899e', 'font-size': 10}, '0 µm'), svgEl('text', {x: 245, y: 31, fill: '#7a899e', 'font-size': 10, 'text-anchor': 'middle'}, `${valueText(max / 2)} µm`), svgEl('text', {x: 489, y: 31, fill: '#7a899e', 'font-size': 10, 'text-anchor': 'end'}, `${valueText(max)} µm 이상`)); scale.append(legend);
      }
    }
    function showData(result) {
      data = result;
      const maps = Array.isArray(result.maps) ? result.maps.filter(map => map && Number.isInteger(map.index) && map.index >= 0) : [];
      const optionSignature = maps.map(map => `${map.index}:${mapLabel(map, map.index)}`).join('|');
      if (select.dataset.signature !== optionSignature) {
        select.replaceChildren(); maps.forEach(map => { const option = el('option', '', mapLabel(map, map.index)); option.value = String(map.index); select.append(option); }); select.dataset.signature = optionSignature;
      }
      if (Number.isInteger(result.selected_index)) selectedIndex = result.selected_index;
      select.value = String(selectedIndex); select.disabled = maps.length < 2;
      const analysis = result.analysis && typeof result.analysis === 'object' ? result.analysis : {}, summary = analysis.summary && typeof analysis.summary === 'object' ? analysis.summary : {};
      const blasting = result.process === 'blasting', physical = result.process === 'spray' && ['available', 'partial'].includes(analysis.status);
      semantics.textContent = blasting ? '블라스팅 시퀀스에 저장된 크롭 원본 텍스처입니다. 도막 두께로 해석하지 않습니다.' : physical ? '이 시점의 크롭에 저장된 누적 도막 두께입니다. 이전 도장을 포함하며, 이번 시퀀스만의 도장 증가량이나 최종 전체 구역 맵과는 계산 범위가 다릅니다.' : '이 시퀀스에 저장된 영역 크롭의 원본 이미지입니다. 도막 두께를 해석할 수 있는지 확인한 결과와 사유를 표시합니다.';
      sourceNote.textContent = typeof result.source_note === 'string' ? result.source_note : ''; sourceNote.hidden = !sourceNote.textContent;
      notes.replaceChildren();
      const currentMap = maps.find(map => map.index === selectedIndex);
      if (currentMap) {
        const dimensions = [];
        if ([currentMap.x0, currentMap.y0].every(value => finite(value) !== null)) dimensions.push(`크롭 위치 (${countText(currentMap.x0)}, ${countText(currentMap.y0)})`);
        if ([currentMap.width, currentMap.height].every(value => finite(value) !== null)) dimensions.push(`크롭 크기 ${countText(currentMap.width)} × ${countText(currentMap.height)} px`);
        if ([currentMap.img_width, currentMap.img_height].every(value => finite(value) !== null)) dimensions.push(`이미지 ${countText(currentMap.img_width)} × ${countText(currentMap.img_height)} px`);
        if (dimensions.length) notes.append(el('p', '', dimensions.join(' · ')));
      }
      const knownCoverage = analysis.coverage_domain === 'official_zone_masks_in_crop';
      if (physical) {
        notes.append(el('p', '', knownCoverage ? '크롭 내 도포율: 크롭과 공식 도장 대상 영역이 겹치는 부분의 전체 픽셀 대비 도포 픽셀 비율입니다. 전체 구역 도포율을 뜻하지 않습니다.' : '크롭 내 도포율: 공식 도장 대상 영역을 확정하지 못해 미기록으로 표시합니다. 전체 구역의 도포율로 추정하지 않습니다.'));
        notes.append(el('p', '', analysis.distribution_domain === 'painted_image_pixels' ? '두께 분포·평균·중앙값·IQR은 크롭 이미지 전체의 도포 픽셀 기준입니다. 공식 대상 영역이 확인된 구역별 수치는 해당 구역의 도포 픽셀 기준입니다.' : analysis.distribution_domain === 'official_zone_masks_in_crop' ? '두께 분포·평균·중앙값·IQR은 크롭과 공식 도장 대상 영역이 겹치는 부분의 도포 픽셀 기준입니다.' : '이 크롭의 도막 두께 통계를 계산하지 못했습니다. 사유와 원본 파라미터를 확인할 수 있습니다.'));
        notes.append(el('p', '', '두께 환산에는 저장된 v2B8 유효 픽셀만 사용하며 이전 인코딩으로 대체 해석하지 않습니다.'));
        notes.append(el('p', '', `${analysis.basis_label || `목표 ${valueText(summary.target_thickness_um)} µm 기준`} · 두께는 보정값(측정값 − 10 µm)입니다. 크롭 안 표본 기준이며 Zone 전체 등급이 아닙니다.`));
      }
      const reasons = [...new Set([analysis.reason, result.reason, ...(Array.isArray(analysis.warnings) ? analysis.warnings : [])].map(reasonText).filter(Boolean))];
      if (reasons.length) { const list = el('ul'); reasons.forEach(reason => list.append(el('li', '', reason))); notes.append(list); }
      if (blasting && !reasons.length) notes.append(el('p', '', reasonLabels.not_applicable));
      metrics.hidden = graphs.hidden = footnote.hidden = !physical;
      const bands = bandCounts(summary), bandTotal = bands.reduce((sum, count) => sum + count, 0);
      metrics.replaceChildren(...[
        ['크롭 내 도포율', knownCoverage ? valueText(summary.coverage_percent, '%') : '미기록'], ['평균 두께', valueText(summary.mean_um, ' µm')], ['중앙값', valueText(summary.median_um, ' µm')],
        ['목표 두께', valueText(summary.target_thickness_um, ' µm')], ['정상 구간 비율', bandTotal ? valueText(bands[2] / bandTotal * 100, '%') : '미기록'], ['두께 변동계수', valueText(summary.cv_percent, '%')],
      ].map(([name, value]) => { const card = el('div', 'sh-metric'); card.append(el('span', '', name), el('strong', '', value)); return card; }));
      if (physical) { renderHistogram(histogram, analysis.histogram); renderZones(zones, analysis.zones, summary.target_thickness_um); renderBands(bandHost, summary, analysis.band_labels); }
      footnote.textContent = `크롭 내 공식 대상 ${countText(summary.target_pixels)} 픽셀 · 도포 ${countText(summary.painted_pixels)} 픽셀 · 도포 픽셀의 10–90 백분위 두께 ${valueText(summary.p10_um)}–${valueText(summary.p90_um)} µm · 표준편차 ${valueText(summary.std_um, ' µm')}`;
      showImage(); body.hidden = false;
      setStatus(result.available === false ? '이 시퀀스의 히트맵을 표시할 수 없습니다. 저장 상태와 사유를 아래에서 확인할 수 있습니다.' : '');
    }
    async function load(index) {
      request?.abort(); request = new AbortController(); const current = request, version = ++requestVersion;
      selectedIndex = index; clearImage(); data = null; body.hidden = true; decoded.disabled = raw.disabled = true; host.setAttribute('aria-busy', 'true');
      sourceNote.hidden = true;
      setStatus('시퀀스 히트맵과 도막 분석을 불러오고 있습니다…');
      if (!endpoint) { host.removeAttribute('aria-busy'); setStatus('히트맵 주소가 올바르지 않습니다. 원본 파라미터는 아래에서 확인할 수 있습니다.', true); return; }
      const next = new URL(endpoint.href); next.searchParams.set('map_index', String(index));
      try {
        const response = await fetch(next.href, {signal: current.signal, headers: {Accept: 'application/json'}, cache: 'no-store'});
        if (!response.ok) {
          let reason = ''; try { const errorBody = await response.json(); reason = reasonText(errorBody.detail || errorBody.reason); } catch {}
          throw new Error(reason || (response.status === 404 ? '저장된 시퀀스 히트맵을 찾을 수 없습니다.' : '시퀀스 히트맵을 불러오지 못했습니다.'));
        }
        const result = await response.json(); if (!alive(version)) return;
        if (!result || typeof result !== 'object' || Array.isArray(result)) throw new Error('히트맵 분석 응답을 읽을 수 없습니다.');
        showData(result);
      } catch (error) {
        if (!alive(version) || error.name === 'AbortError') return;
        const message = error.name === 'SyntaxError' ? '히트맵 분석 응답을 읽을 수 없습니다.' : error.name === 'TypeError' ? '히트맵 서버에 연결하지 못했습니다.' : error.message || '히트맵을 불러오지 못했습니다.';
        setStatus(`${message} 원본 파라미터는 아래에서 계속 확인할 수 있습니다.`, true);
        const retry = el('button', '', '다시 불러오기'); retry.type = 'button'; retry.addEventListener('click', () => load(selectedIndex)); status.append(retry);
      } finally { if (alive(version)) host.removeAttribute('aria-busy'); }
    }
    select.addEventListener('change', () => { const index = Number(select.value); if (Number.isInteger(index) && index >= 0) load(index); });
    decoded.addEventListener('click', () => { view = 'decoded'; showImage(); }); raw.addEventListener('click', () => { view = 'raw'; showImage(); });
    load(0); return dispose;
  }
  window.CopilotSequenceHeatmap = Object.freeze({render});
})();
