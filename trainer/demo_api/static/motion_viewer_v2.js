// 모션 3D 뷰어 v2. 기존 재생 스크립트가 보내는 motion-viewer-data / motion-viewer-frame 이벤트를 받아
// 결과 한 줄, 지금 Zone 안내, 세 기능 그림(자세·순서·도막), 고정 질문 열을 그린다. 값은 서버가 준 것만 쓴다.
(() => {
  'use strict';
  const ICON = {"user":"<path d=\"M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2\" /><circle cx=\"12\" cy=\"7\" r=\"4\" />","spray-can":"<path d=\"M3 3h.01\" /><path d=\"M7 5h.01\" /><path d=\"M11 7h.01\" /><path d=\"M3 7h.01\" /><path d=\"M7 9h.01\" /><path d=\"M3 11h.01\" /><rect width=\"4\" height=\"4\" x=\"15\" y=\"5\" /><path d=\"m19 9 2 2v10c0 .6-.4 1-1 1h-6c-.6 0-1-.4-1-1V11l2-2\" /><path d=\"m13 14 8-2\" /><path d=\"m13 19 8-2\" />","graduation-cap":"<path d=\"M21.42 10.922a1 1 0 0 0-.019-1.838L12.83 5.18a2 2 0 0 0-1.66 0L2.6 9.08a1 1 0 0 0 0 1.832l8.57 3.908a2 2 0 0 0 1.66 0z\" /><path d=\"M22 10v6\" /><path d=\"M6 12.5V16a6 3 0 0 0 12 0v-3.5\" />","history":"<path d=\"M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8\" /><path d=\"M3 3v5h5\" /><path d=\"M12 7v5l4 2\" />","gauge":"<path d=\"m12 14 4-4\" /><path d=\"M3.34 19a10 10 0 1 1 17.32 0\" />","person-standing":"<circle cx=\"12\" cy=\"5\" r=\"1\" /><path d=\"m9 20 3-6 3 6\" /><path d=\"m6 8 6 2 6-2\" /><path d=\"M12 10v4\" />","route":"<circle cx=\"6\" cy=\"19\" r=\"3\" /><path d=\"M9 19h8.5a3.5 3.5 0 0 0 0-7h-11a3.5 3.5 0 0 1 0-7H15\" /><circle cx=\"18\" cy=\"5\" r=\"3\" />","crosshair":"<circle cx=\"12\" cy=\"12\" r=\"10\" /><line x1=\"22\" x2=\"18\" y1=\"12\" y2=\"12\" /><line x1=\"6\" x2=\"2\" y1=\"12\" y2=\"12\" /><line x1=\"12\" x2=\"12\" y1=\"6\" y2=\"2\" /><line x1=\"12\" x2=\"12\" y1=\"22\" y2=\"18\" />","play":"<path d=\"M5 5a2 2 0 0 1 3.008-1.728l11.997 6.998a2 2 0 0 1 .003 3.458l-12 7A2 2 0 0 1 5 19z\" />","pause":"<rect x=\"14\" y=\"3\" width=\"5\" height=\"18\" rx=\"1\" /><rect x=\"5\" y=\"3\" width=\"5\" height=\"18\" rx=\"1\" />","skip-back":"<path d=\"M17.971 4.285A2 2 0 0 1 21 6v12a2 2 0 0 1-3.029 1.715l-9.997-5.998a2 2 0 0 1-.003-3.432z\" /><path d=\"M3 20V4\" />","focus":"<circle cx=\"12\" cy=\"12\" r=\"3\" /><path d=\"M3 7V5a2 2 0 0 1 2-2h2\" /><path d=\"M17 3h2a2 2 0 0 1 2 2v2\" /><path d=\"M21 17v2a2 2 0 0 1-2 2h-2\" /><path d=\"M7 21H5a2 2 0 0 1-2-2v-2\" />","ruler":"<path d=\"M21.3 15.3a2.4 2.4 0 0 1 0 3.4l-2.6 2.6a2.4 2.4 0 0 1-3.4 0L2.7 8.7a2.41 2.41 0 0 1 0-3.4l2.6-2.6a2.41 2.41 0 0 1 3.4 0Z\" /><path d=\"m14.5 12.5 2-2\" /><path d=\"m11.5 9.5 2-2\" /><path d=\"m8.5 6.5 2-2\" /><path d=\"m17.5 15.5 2-2\" />","axis-3d":"<path d=\"M13.5 10.5 15 9\" /><path d=\"M4 4v15a1 1 0 0 0 1 1h15\" /><path d=\"M4.293 19.707 6 18\" /><path d=\"m9 15 1.5-1.5\" />","arrow-right-left":"<path d=\"m16 3 4 4-4 4\" /><path d=\"M20 7H4\" /><path d=\"m8 21-4-4 4-4\" /><path d=\"M4 17h16\" />","scan":"<path d=\"M3 7V5a2 2 0 0 1 2-2h2\" /><path d=\"M17 3h2a2 2 0 0 1 2 2v2\" /><path d=\"M21 17v2a2 2 0 0 1-2 2h-2\" /><path d=\"M7 21H5a2 2 0 0 1-2-2v-2\" />","layers":"<path d=\"M12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83z\" /><path d=\"M2 12a1 1 0 0 0 .58.91l8.6 3.91a2 2 0 0 0 1.65 0l8.58-3.9A1 1 0 0 0 22 12\" /><path d=\"M2 17a1 1 0 0 0 .58.91l8.6 3.91a2 2 0 0 0 1.65 0l8.58-3.9A1 1 0 0 0 22 17\" />","paint-roller":"<rect width=\"16\" height=\"6\" x=\"2\" y=\"2\" rx=\"2\" /><path d=\"M10 16v-2a2 2 0 0 1 2-2h8a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2h-2\" /><rect width=\"4\" height=\"6\" x=\"8\" y=\"16\" rx=\"1\" />","rotate-ccw":"<path d=\"M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8\" /><path d=\"M3 3v5h5\" />","rotate-cw":"<path d=\"M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8\" /><path d=\"M21 3v5h-5\" />","locate-fixed":"<line x1=\"2\" x2=\"5\" y1=\"12\" y2=\"12\" /><line x1=\"19\" x2=\"22\" y1=\"12\" y2=\"12\" /><line x1=\"12\" x2=\"12\" y1=\"2\" y2=\"5\" /><line x1=\"12\" x2=\"12\" y1=\"19\" y2=\"22\" /><circle cx=\"12\" cy=\"12\" r=\"7\" /><circle cx=\"12\" cy=\"12\" r=\"3\" />","map-pin":"<path d=\"M20 10c0 4.993-5.539 10.193-7.399 11.799a1 1 0 0 1-1.202 0C9.539 20.193 4 14.993 4 10a8 8 0 0 1 16 0\" /><circle cx=\"12\" cy=\"10\" r=\"3\" />","book-open":"<path d=\"M12 5v16\" /><path d=\"M20.001 19A2 2 0 0022 17V5a2 2 0 00-1.999-2L16 3.002A5 5 0 0012 5a5 5 0 00-4-2H4a2 2 0 00-2 2v12a2 2 0 001.999 2H8a5 5 0 014 2 5 5 0 014-2z\" />","zap":"<path d=\"M15.914 4a1.5 1.5 0 00-2.474-1.561l-9 9A1.5 1.5 0 005.5 14h4.002a.5.5 0 01.471.666L8.086 20a1.5 1.5 0 002.475 1.56l9-9A1.5 1.5 0 0018.5 10h-3.997a.5.5 0 01-.472-.667z\" />","file-search":"<path d=\"M6 22a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h8a2.4 2.4 0 0 1 1.704.706l3.588 3.588A2.4 2.4 0 0 1 20 8v12a2 2 0 0 1-2 2z\" /><path d=\"M14 2v5a1 1 0 0 0 1 1h5\" /><circle cx=\"11.5\" cy=\"14.5\" r=\"2.5\" /><path d=\"M13.3 16.3 15 18\" />","send":"<path d=\"M14.536 21.686a.5.5 0 0 0 .937-.024l6.5-19a.496.496 0 0 0-.635-.635l-19 6.5a.5.5 0 0 0-.024.937l7.93 3.18a2 2 0 0 1 1.112 1.11z\" /><path d=\"m21.854 2.147-10.94 10.939\" />","message-square-plus":"<path d=\"M22 17a2 2 0 0 1-2 2H6.828a2 2 0 0 0-1.414.586l-2.202 2.202A.71.71 0 0 1 2 21.286V5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2z\" /><path d=\"M12 8v6\" /><path d=\"M9 11h6\" />","chart-column":"<path d=\"M3 3v16a2 2 0 0 0 2 2h16\" /><path d=\"M18 17V9\" /><path d=\"M13 17V5\" /><path d=\"M8 17v-3\" />","message-circle":"<path d=\"M2.992 16.342a2 2 0 0 1 .094 1.167l-1.065 3.29a1 1 0 0 0 1.236 1.168l3.413-.998a2 2 0 0 1 1.099.092 10 10 0 1 0-4.777-4.719\" />","triangle-alert":"<path d=\"m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3\" /><path d=\"M12 9v4\" /><path d=\"M12 17h.01\" />","languages":"<path d=\"m5 8 6 6\" /><path d=\"m4 14 6-6 2-3\" /><path d=\"M2 5h12\" /><path d=\"M7 2h1\" /><path d=\"m22 22-5-10-5 10\" /><path d=\"M14 18h6\" />","sliders-horizontal":"<path d=\"M10 5H3\" /><path d=\"M12 19H3\" /><path d=\"M14 3v4\" /><path d=\"M16 17v4\" /><path d=\"M21 12h-9\" /><path d=\"M21 19h-5\" /><path d=\"M21 5h-7\" /><path d=\"M8 10v4\" /><path d=\"M8 12H3\" />","info":"<circle cx=\"12\" cy=\"12\" r=\"10\" /><path d=\"M12 16v-4\" /><path d=\"M12 8h.01\" />","x":"<path d=\"M18 6 6 18\" /><path d=\"m6 6 12 12\" />","square":"<rect width=\"18\" height=\"18\" x=\"3\" y=\"3\" rx=\"2\" />","map":"<path d=\"M14.106 5.553a2 2 0 0 0 1.788 0l3.659-1.83A1 1 0 0 1 21 4.619v12.764a1 1 0 0 1-.553.894l-4.553 2.277a2 2 0 0 1-1.788 0l-4.212-2.106a2 2 0 0 0-1.788 0l-3.659 1.83A1 1 0 0 1 3 19.381V6.618a1 1 0 0 1 .553-.894l4.553-2.277a2 2 0 0 1 1.788 0z\" /><path d=\"M15 5.764v15\" /><path d=\"M9 3.236v15\" />","trending-down":"<path d=\"M16 17h6v-6\" /><path d=\"m22 17-8.5-8.5-5 5L2 7\" />","route-off":"<circle cx=\"6\" cy=\"19\" r=\"3\" /><path d=\"M9 19h8.5c.4 0 .9-.1 1.3-.2\" /><path d=\"M5.2 5.2A3.5 3.53 0 0 0 6.5 12H12\" /><path d=\"m2 2 20 20\" /><path d=\"M21 15.3a3.5 3.5 0 0 0-3.3-3.3\" /><path d=\"M15 5h-4.3\" /><circle cx=\"18\" cy=\"5\" r=\"3\" />","arrow-up-to-line":"<path d=\"M5 3h14\" /><path d=\"m18 13-6-6-6 6\" /><path d=\"M12 7v14\" />","arrow-down-to-line":"<path d=\"M12 17V3\" /><path d=\"m6 11 6 6 6-6\" /><path d=\"M19 21H5\" />","bell":"<path d=\"M10.268 21a2 2 0 0 0 3.464 0\" /><path d=\"M3.262 15.326A1 1 0 0 0 4 17h16a1 1 0 0 0 .74-1.673C19.41 13.956 18 12.499 18 8A6 6 0 0 0 6 8c0 4.499-1.411 5.956-2.738 7.326\" />","box":"<path d=\"M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z\" /><path d=\"m3.3 7 8.7 5 8.7-5\" /><path d=\"M12 22V12\" />"};
  const options = window.motionViewerV2Options || {};
  const $ = (id) => document.getElementById(id);
  const svg = (name) => `<svg class="lucide" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">${ICON[name] || ''}</svg>`;
  const hydrate = (root) => { for (const node of root.querySelectorAll('i[data-icon]')) node.outerHTML = svg(node.dataset.icon); };
  const el = (tag, cls, text) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text != null) node.textContent = text;
    return node;
  };
  const iconButton = (cls, icon, label) => {
    const button = el('button', cls);
    button.type = 'button';
    button.innerHTML = svg(icon);
    button.setAttribute('aria-label', label);
    button.title = label;
    return button;
  };
  // Zone 색은 기존 재생 스크립트의 번호별 고정 색(zoneColor)을 그대로 쓴다.
  const colorOf = (zoneId) => (typeof zoneColor === 'function' ? zoneColor(zoneId) : '#94A3B8');
  const zoneText = (zoneId) => `Zone ${Number(zoneId)}`;
  const tagZone = (node, zoneId, suffix = '') => {
    node.classList.toggle('zone-tag-idle', zoneId == null);
    node.style.setProperty('--zone', zoneId == null ? '' : colorOf(zoneId));
    node.textContent = zoneId == null ? suffix || '이동 중' : zoneText(zoneId) + suffix;
  };
  const percent = (value) => (Number.isFinite(Number(value)) ? `${Math.round(Number(value))}%` : '—');
  const DIRECTION = { horizontal: '수평', vertical: '수직', diagonal: '대각', curved_irregular: '곡선·불규칙' };
  // 화면 어디서나 "Zone 8"로 쓴다(기존 스크립트는 "Zone 08").
  window.zoneLabel = (zoneId) => String(Number.isFinite(Number(zoneId)) ? Number(zoneId) : zoneId);

  let record = null;
  let anchor = { map_num: null, zone_id: null, source_index: 0, playhead_seconds: 0, spraying: false };
  let armSeen;

  const zoneKey = (mapNum, zoneId) => `${mapNum}:${zoneId}`;
  const knownZone = (frame) => {
    if (frame?.map_num == null || !(Number(frame.zone_id) > 0)) return null;
    return record?.pose_evaluation?.zones?.[zoneKey(frame.map_num, frame.zone_id)] ? { map_num: String(frame.map_num), zone_id: String(frame.zone_id) } : null;
  };
  // 질문과 화면이 같은 기준을 쓴다: 재생 위치에서 가장 최근에 분사한 Zone. 그 분사의 Zone을 모르면 Zone 없음.
  const anchorAt = (index, seconds) => {
    const frames = record?.frames || [];
    let zone = null;
    for (let i = Math.min(index, frames.length - 1); i >= 0; i--) {
      if (!frames[i]?.spraying) continue;
      zone = knownZone(frames[i]);
      break;
    }
    return { map_num: zone?.map_num ?? null, zone_id: zone?.zone_id ?? null, source_index: index, playhead_seconds: seconds,
      spraying: Boolean(frames[index]?.spraying) };
  };
  const seekToIndex = (index) => {
    const timeline = $('timeline');
    timeline.value = String(Math.max(0, Math.min(Number(timeline.max), index)));
    timeline.dispatchEvent(new Event('input', { bubbles: true }));
  };
  const seekToSeconds = (seconds) => {
    const frames = record?.frames || [];
    let index = frames.findIndex((frame) => frame.t >= seconds);
    if (index < 0) index = frames.length - 1;
    seekToIndex(index);
  };
  const seekToZone = (zoneId) => {
    const index = (record?.frames || []).findIndex((frame) => frame.spraying && String(frame.zone_id) === String(zoneId));
    if (index >= 0) seekToIndex(index);
  };

  // ── 결과 한 줄 ─────────────────────────────────────────────
  function renderScores() {
    const work = record?.pose_evaluation;
    const arm = record?.arm_pose_evaluation;
    const order = record?.reference_order_evaluation;
    $('v2ScoreWork').textContent = work?.status === 'available' ? percent(work.similarity_percent) : '—';
    $('v2ScorePose').textContent = arm === undefined && record ? '…' : arm?.status === 'available' ? percent(arm.average_similarity_percent) : '—';
    $('v2ScoreOrder').textContent = Number.isFinite(Number(order?.adherence_rate)) ? percent(Number(order.adherence_rate) * 100) : '—';
    const zones = Object.values(work?.zones || {}).filter((zone) => Number.isFinite(Number(zone.similarity_percent)));
    const jump = $('v2Worst');
    jump.hidden = !zones.length;
    if (zones.length) {
      const worst = zones.reduce((low, zone) => (Number(zone.similarity_percent) < Number(low.similarity_percent) ? zone : low));
      tagZone($('v2WorstZone'), worst.zone_id);
      $('v2WorstScore').textContent = percent(worst.similarity_percent);
      jump.title = `작업 유사도가 가장 낮은 ${zoneText(worst.zone_id)}(${percent(worst.similarity_percent)})로 이동`;
      jump.setAttribute('aria-label', jump.title);
      jump.onclick = () => seekToZone(worst.zone_id);
    }
  }

  // ── 지금 Zone 한 줄 ────────────────────────────────────────
  function renderZoneNote() {
    const zone = anchor.zone_id == null ? null : record?.pose_evaluation?.zones?.[zoneKey(anchor.map_num, anchor.zone_id)];
    tagZone($('v2ZoneTag'), zone ? anchor.zone_id : null, zone && !anchor.spraying ? ' · 이동 중' : '');
    $('v2ZoneScore').textContent = zone ? percent(zone.similarity_percent) : '';
    $('v2ZoneScore').parentElement.hidden = !zone;
    const corrections = zone?.corrections || [];
    const advice = $('v2ZoneAdvice');
    // 자세 교정이 없어도 목표 대비 도막이 미만·1.5배 이상이면 '차이가 작습니다'는 도막 패널과 반대로 읽힌다. 서버가 낸 등급을 그대로 말한다.
    const coat = (targetOf()?.zones || []).find((row) => String(row.zone_id) === String(anchor.zone_id) && (row.map_num == null || String(row.map_num) === String(anchor.map_num)));
    const coatNote = coat && (coat.grade === 'below' || coat.grade === 'above') && coat.thickness_um != null
      ? `도막 ${targetOf().grade_labels?.[coat.grade] ?? coat.grade} · ${Math.round(Number(coat.thickness_um))} µm` : '';
    advice.textContent = !zone ? '' : corrections.length ? corrections[0].replace(/[.。]\s*$/, '') + (corrections.length > 1 ? ` 외 ${corrections.length - 1}건` : '') : coatNote || '전문가 기록과 차이가 작습니다';
    advice.title = corrections.join('\n');
    const more = $('v2ZoneMore');
    const metrics = (zone?.metrics || []).filter((metric) => metric.available !== false && metric.similarity_percent != null);
    more.hidden = !metrics.length;
    const detail = $('v2ZoneDetail');
    if (!metrics.length) { detail.hidden = true; more.setAttribute('aria-expanded', 'false'); return; }
    if (detail.hidden) return;
    detail.replaceChildren(...metrics.map((metric) => {
      const item = el('div', 'metric');
      const track = el('div', 'metric-track');
      const fill = el('div', 'metric-fill');
      fill.style.width = `${Math.max(0, Math.min(100, Number(metric.similarity_percent)))}%`;
      track.append(fill);
      const value = el('div', 'metric-value');
      const unit = { deg: '°', mm: ' mm', um: ' µm' }[metric.unit] || '';
      const show = (number, label) => label || DIRECTION[number] || (Number.isFinite(Number(number)) ? `${Number(number).toFixed(1)}${unit}` : '—');
      value.append(show(metric.actual, metric.actual_label), el('span', null, ` / 전문가 ${show(metric.reference, metric.reference_label)}`));
      item.append(el('div', 'metric-name', metric.label), track, value);
      return item;
    }));
  }
  $('v2ZoneMore').addEventListener('click', () => {
    const detail = $('v2ZoneDetail');
    detail.hidden = !detail.hidden;
    $('v2ZoneMore').setAttribute('aria-expanded', String(!detail.hidden));
    renderZoneNote();
    dispatchEvent(new Event('resize'));
  });

  // ── 자세: 스켈레톤과 분사별 자세 유사도 ───────────────────
  const ARM_UNAVAILABLE = {
    process_not_supported: '블라스팅 기록은 자세를 채점하지 않습니다',
    no_joint_sequences: '채점할 분사 구간이 없습니다',
  };
  const sequenceAt = (seconds) => {
    const started = (record?.arm_pose_evaluation?.sequences || []).filter((item) => Number.isFinite(Number(item.start_seconds)) && item.start_seconds <= seconds);
    return started.length ? started.reduce((late, item) => (item.start_seconds >= late.start_seconds ? item : late)) : null;
  };
  function renderPose() {
    const arm = record?.arm_pose_evaluation;
    const host = $('v2PoseBars');
    if (arm?.status !== 'available') $('v2PoseScore').textContent = arm === undefined && record ? '…' : '—';
    if (arm?.status !== 'available') {
      const message = !record ? '' : arm === undefined ? '분사마다 채점하는 중' : ARM_UNAVAILABLE[arm.reason_code] || '자세 채점 결과를 불러오지 못했습니다';
      host.replaceChildren(el('p', 'chart-empty', message));
      $('v2PoseNow').textContent = '';
      return;
    }
    const average = el('div', 'pose-avg');
    average.dataset.label = percent(arm.average_similarity_percent);
    average.style.bottom = `calc((100% - 8px) * ${Math.max(0, Math.min(100, Number(arm.average_similarity_percent))) / 100})`;
    host.replaceChildren(average, ...arm.sequences.map((item) => {
      const bar = el('button', 'pose-bar');
      bar.type = 'button';
      bar.dataset.sequence = item.sequence_id;
      bar.style.setProperty('--h', `${Math.max(2, Math.min(100, Number(item.similarity_percent)))}%`);
      bar.style.setProperty('--zone', colorOf(item.zone_id));
      bar.classList.toggle('is-limited', item.confidence === 'limited');
      bar.classList.toggle('has-fix', Boolean(item.correction));
      bar.title = `${item.sequence_id} · ${zoneText(item.zone_id)} · ${percent(item.similarity_percent)}`
        + (item.correction ? `\n${item.correction}` : '') + (item.confidence === 'limited' ? '\n입력 품질 제한으로 참고용' : '');
      bar.setAttribute('aria-label', bar.title.replaceAll('\n', '. '));
      bar.addEventListener('click', () => seekToSeconds(Number(item.start_seconds) + 0.01));
      return bar;
    }));
    renderPoseNow();
  }
  function renderPoseNow() {
    const arm = record?.arm_pose_evaluation;
    if (arm?.status !== 'available') return;
    const current = sequenceAt(anchor.playhead_seconds);
    for (const bar of $('v2PoseBars').querySelectorAll('.pose-bar')) bar.classList.toggle('is-current', bar.dataset.sequence === current?.sequence_id);
    const now = $('v2PoseNow');
    $('v2PoseScore').textContent = current ? percent(current.similarity_percent) : '—';
    if (!current) { now.textContent = '첫 분사 전입니다'; return; }
    const head = el('b', null, current.sequence_id);
    const limited = current.confidence === 'limited';
    now.replaceChildren(head, ` ${current.correction || (limited ? '' : '교정할 관절이 없습니다')}${limited ? ' (입력 품질 제한으로 참고용)' : ''}`);
  }

  // 채점 엔진이 교정을 낸 관절을 스켈레톤 위에 고리로 표시한다(기존 스켈레톤 그림과 같은 투영 함수).
  function markFlaggedJoints(frame) {
    const flagged = (sequenceAt(anchor.playhead_seconds)?.joints || []).map((name) => String(name).replace(/([a-z])([A-Z])/g, '$1 $2'));
    if (!flagged.length || typeof skeletonProjection !== 'function') return;
    const canvas = $('skeletonCanvas');
    const box = canvas.getBoundingClientRect();
    const joints = panelSkeleton(frame).joints;
    const view = skeletonProjection(joints, box.width, box.height);
    if (!view) return;
    const context = canvas.getContext('2d');
    context.strokeStyle = '#f2b84b';
    context.lineWidth = 2;
    for (const name of flagged) {
      const point = joints[name];
      if (!point) continue;
      const [x, y] = view.project(point);
      context.beginPath();
      context.arc(x, y, 10, 0, Math.PI * 2);
      context.stroke();
    }
  }

  // ── 순서: 전문가 순서와 이 기록의 순서를 선으로 잇는다 ─────
  function renderOrder() {
    const order = record?.reference_order_evaluation || {};
    const host = $('v2Order');
    const expert = (order.recommended_flow || []).map(String);
    const actual = (order.selected_flow || []).map(String);
    if (!expert.length) { host.replaceChildren(el('p', 'chart-empty', record ? '이 학습모드에는 전문가 순서 기준이 없습니다' : '')); return; }
    const row = (label, zones, missing = []) => {
      const line = el('div', 'order-row');
      const chips = el('div', 'order-chips');
      for (const zone of [...zones, ...missing]) {
        const chip = el('button', 'order-chip', String(Number(zone)));
        chip.type = 'button';
        chip.dataset.zone = zone;
        chip.style.setProperty('--zone', colorOf(zone));
        const skipped = missing.includes(zone);
        chip.classList.toggle('is-missing', skipped);
        chip.title = skipped ? `${zoneText(zone)} · 칠하지 않음` : `${zoneText(zone)}로 이동`;
        chip.setAttribute('aria-label', chip.title);
        chip.disabled = skipped;
        chip.addEventListener('click', () => seekToZone(zone));
        chips.append(chip);
      }
      line.append(el('span', null, label), chips);
      return line;
    };
    const links = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    links.setAttribute('class', 'order-links');
    links.setAttribute('aria-hidden', 'true');
    host.replaceChildren(row('전문가', expert), links, row('이 기록', actual, expert.filter((zone) => !actual.includes(zone))));
    drawOrderLinks();
    markCurrentZone();
  }
  function drawOrderLinks() {
    const host = $('v2Order');
    const links = host.querySelector('.order-links');
    if (!links) return;
    const [top, bottom] = host.querySelectorAll('.order-chips');
    const box = links.getBoundingClientRect();
    if (!box.width || !box.height) return;
    const center = (chip) => { const rect = chip.getBoundingClientRect(); return rect.left + rect.width / 2 - box.left; };
    links.setAttribute('viewBox', `0 0 ${box.width} ${box.height}`);
    links.replaceChildren();
    [...top.children].forEach((chip, position) => {
      const match = [...bottom.children].findIndex((other) => other.dataset.zone === chip.dataset.zone && !other.classList.contains('is-missing'));
      if (match < 0) return;
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', center(chip)); line.setAttribute('y1', 2);
      line.setAttribute('x2', center(bottom.children[match])); line.setAttribute('y2', box.height - 2);
      // 같은 순번이면 흐리게, 순번이 다르면 그 Zone 색으로 잇는다.
      line.setAttribute('stroke', match === position ? 'rgba(134,150,176,.5)' : colorOf(chip.dataset.zone));
      line.setAttribute('stroke-opacity', match === position ? '1' : '.85');
      links.append(line);
    });
  }
  new ResizeObserver(drawOrderLinks).observe($('v2Order'));

  // ── 도막: 목표 대비(기본, PR #8 기준)와 전문가 대비 ─────────
  // 등급·칸 이름·정상 범위·척도는 서버가 서명한 target의 값만 읽는다. 화면에서 등급을 다시 계산하지 않는다.
  let coatBasis = 'target';
  let coatMap = false;
  const targetOf = () => record?.coating_evaluation?.target || null;
  const targetReady = () => targetOf()?.analysis_status === 'available';
  const targetMissing = () => (targetOf()?.reason_code === 'height_map_unavailable'
    ? '이 기록은 높이맵 원본이 없어 목표 대비 판정이 없습니다' : '이 기록은 목표 대비 판정이 없습니다');
  const tally = (cls, label, number) => { const item = el('span', cls, `${label} `); item.append(el('b', null, String(number))); return item; };
  const shareText = (share) => `${Math.round(share * 100)}%`;

  function setCoating(basis, showMap = false) {
    coatBasis = basis;
    coatMap = basis === 'expert' && showMap;
    renderCoating();
    dispatchEvent(new Event('resize'));
  }

  function renderCoating() {
    const expert = coatBasis === 'expert';
    $('v2CoatBasisTarget').setAttribute('aria-pressed', String(!expert));
    $('v2CoatBasisExpert').setAttribute('aria-pressed', String(expert));
    $('v2CoatViews').hidden = !expert;
    $('v2CoatBarsView').setAttribute('aria-pressed', String(!coatMap));
    $('v2CoatMapView').setAttribute('aria-pressed', String(coatMap));
    $('v2CoatBars').hidden = coatMap;
    $('v2CoatMap').hidden = !coatMap;
    $('v2CoatBars').setAttribute('aria-label', expert ? 'Zone별 도막 두께가 전문가 기준과 얼마나 다른지' : 'Zone별 도막 등급(목표 대비)');
    $('v2CoatBasisLabel').textContent = !expert && targetReady() ? targetOf().basis_label || '' : '';
    if (expert) renderExpertCoating(); else renderTargetCoating();
  }

  function renderTargetCoating() {
    const host = $('v2CoatBars');
    const count = $('v2CoatCount');
    const target = targetOf();
    count.replaceChildren();
    if (!record) { host.replaceChildren(); return; }
    if (!targetReady()) {
      const go = el('button', 'text-btn', '전문가 대비 보기');
      go.type = 'button';
      go.addEventListener('click', () => setCoating('expert'));
      const box = el('div', 'chart-empty coat-empty');
      box.append(el('span', null, targetMissing()), go);
      host.replaceChildren(box);
      return;
    }
    const labels = target.grade_labels || {};
    const goal = Number(target.target_thickness_um);
    const edges = target.band_edges_um || [];
    const band = target.normal_band_um || [];
    const bandLabels = target.band_labels || [];
    // 막대 끝(척도)은 마지막 6구간 경계다. 보기 위한 척도일 뿐 판정과 무관하다.
    const scaleMax = Number(edges[edges.length - 1]) > 0 ? Number(edges[edges.length - 1]) : 1;
    const at = (value) => `${Math.max(0, Math.min(1, Number(value) / scaleMax)) * 100}%`;
    const away = (row) => Math.abs(Number(row.thickness_um) - goal) / goal;
    const rank = (row) => (row.grade === 'below' || row.grade === 'above' ? 0 : row.grade == null ? 1 : 2);
    const rows = [...(target.zones || [])].sort((left, right) => rank(left) - rank(right)
      || (rank(left) === 0 ? away(right) - away(left) : 0)
      || Number(left.map_num) - Number(right.map_num) || Number(left.zone_id) - Number(right.zone_id));
    const grades = ['below', 'normal', 'above', 'none'];
    const tallies = grades.map((grade) => [grade, rows.filter((row) => (row.grade ?? 'none') === grade).length])
      .filter(([grade, number]) => grade !== 'none' || number > 0);
    count.replaceChildren(...tallies.map(([grade, number]) => tally(`is-${grade}`, labels[grade] ?? grade, number)));
    // 미도장 0.5% 이상인 Zone이 있을 때만 그 칸을 둔다(표시 문턱일 뿐 판정이 아니다).
    const showUnpainted = (row) => Number(row.unpainted_fraction) >= 0.005;
    host.classList.toggle('has-unpainted', rows.some(showUnpainted));
    const axis = el('div', 'coat-axis');
    axis.setAttribute('aria-hidden', 'true');
    const ticks = el('span', 'coat-axis-ticks');
    for (const value of band) {
      const label = el('span', null, String(value));
      label.style.left = at(value);
      ticks.append(label);
    }
    axis.append(el('span'), ticks, el('span', null, 'µm'));
    // 좁은 폭에서 눈금 글자가 붙으면(160·240 → 160240) 앞 눈금과 겹치는 것을 숨긴다. 위치 계산은 그대로.
    new ResizeObserver(() => {
      let edge = -Infinity;
      for (const tickLabel of ticks.children) {
        tickLabel.style.visibility = '';
        const box = tickLabel.getBoundingClientRect();
        if (!box.width) continue;
        if (box.left < edge + 4) tickLabel.style.visibility = 'hidden'; else edge = box.right;
      }
    }).observe(ticks);
    host.replaceChildren(axis, ...rows.map((row) => {
      const grade = row.grade ?? 'none';
      const button = el('button', 'coat-row is-target');
      button.type = 'button';
      button.dataset.zone = String(row.zone_id);
      const name = el('span', 'coat-zone', String(Number(row.zone_id)));
      name.style.setProperty('--zone', colorOf(row.zone_id));
      const track = el('span', 'coat-track is-scale');
      const fill = el('span', `coat-fill is-${grade}`);
      fill.style.width = row.thickness_um == null ? '0%' : at(row.thickness_um);
      track.append(fill);
      band.forEach((value, index) => {
        const tick = el('span', `coat-tick${index === 0 ? ' is-goal' : ''}`);
        tick.style.left = at(value);
        track.append(tick);
      });
      const thickness = row.thickness_um == null ? '—' : `${Math.round(Number(row.thickness_um))} µm`;
      const unpainted = showUnpainted(row) ? `미도장 ${shareText(Number(row.unpainted_fraction))}` : '';
      const counts = row.six_band_counts || {};
      const keys = Object.keys(counts);
      const total = keys.reduce((sum, key) => sum + (Number(counts[key]) || 0), 0);
      const strip = el('span', 'coat-bands');
      const shares = keys.map((key, index) => {
        const share = total ? (Number(counts[key]) || 0) / total : 0;
        const part = el('span', `band-${index + 1}`);
        part.style.width = `${share * 100}%`;
        strip.append(part);
        return `${bandLabels[index] ?? key} ${shareText(share)}`;
      });
      button.append(name, track, el('span', 'coat-value', thickness), el('span', 'coat-unpainted', unpainted), strip);
      const summary = [zoneText(row.zone_id), labels[grade] ?? grade,
        `칠한 곳 평균 ${row.thickness_um == null ? '—' : `${Math.round(Number(row.thickness_um))} µm`}`,
        `미도장 ${shareText(Number(row.unpainted_fraction) || 0)}`, target.basis_label].filter(Boolean).join(' · ');
      button.title = total ? `${summary}\n칠한 면적: ${shares.join(', ')}` : summary;
      button.setAttribute('aria-label', button.title.replaceAll('\n', '. '));
      button.addEventListener('click', () => seekToZone(row.zone_id));
      return button;
    }));
    markCurrentZone();
  }

  function renderExpertCoating() {
    const coating = record?.coating_evaluation;
    const host = $('v2CoatBars');
    host.classList.remove('has-unpainted');
    const zones = (coating?.zones || []).filter((zone) => Number(zone.reference_um) > 0 && Number.isFinite(Number(zone.actual_um)))
      .map((zone) => ({ ...zone, deviation: (Number(zone.actual_um) / Number(zone.reference_um) - 1) * 100 }))
      .sort((left, right) => Math.abs(right.deviation) - Math.abs(left.deviation));
    const count = $('v2CoatCount');
    // 목표 대비 판정이 없어 전문가 대비로 연 기록은 그 까닭을 한 줄로 알린다.
    const note = record && !targetReady() ? [el('p', 'coat-note', targetMissing())] : [];
    if (coating?.analysis_status !== 'available' || !zones.length) {
      count.replaceChildren();
      host.replaceChildren(...note, el('p', 'chart-empty', record ? '이 기록에는 전문가와 견줄 도막 자료가 없습니다' : ''));
      return;
    }
    count.replaceChildren(tally('is-thick', '두꺼움', zones.filter((zone) => zone.deviation > 0).length),
      tally('is-thin', '얇음', zones.filter((zone) => zone.deviation < 0).length));
    const scale = Math.max(20, Math.ceil(Math.max(...zones.map((zone) => Math.abs(zone.deviation))) / 10) * 10);
    host.replaceChildren(...note, ...zones.map((zone) => {
      const row = el('button', 'coat-row');
      row.type = 'button';
      row.dataset.zone = String(zone.zone_id);
      const name = el('span', 'coat-zone', String(Number(zone.zone_id)));
      name.style.setProperty('--zone', colorOf(zone.zone_id));
      const track = el('span', 'coat-track');
      const fill = el('span', `coat-fill ${zone.deviation >= 0 ? 'is-thick' : 'is-thin'}`);
      fill.style.width = `${Math.min(50, Math.abs(zone.deviation) / scale * 50)}%`;
      track.append(fill);
      row.append(name, track, el('span', 'coat-value', `${zone.deviation >= 0 ? '+' : '−'}${Math.abs(zone.deviation).toFixed(0)}%`));
      row.title = `${zoneText(zone.zone_id)} · ${Number(zone.actual_um).toFixed(1)} µm / 전문가 ${Number(zone.reference_um).toFixed(1)} µm`;
      row.setAttribute('aria-label', `${row.title} (${zone.deviation >= 0 ? '두꺼움' : '얇음'} ${Math.abs(zone.deviation).toFixed(0)}%)`);
      row.addEventListener('click', () => seekToZone(zone.zone_id));
      return row;
    }));
    markCurrentZone();
  }
  $('v2CoatBarsView').addEventListener('click', () => setCoating('expert', false));
  $('v2CoatMapView').addEventListener('click', () => setCoating('expert', true));
  // 목표 대비로 돌아오면 지도를 닫는다. 전문가 지도는 "전문가 대비 → 전문가 지도"로 지금과 같은 깊이다.
  $('v2CoatBasisTarget').addEventListener('click', () => setCoating('target'));
  $('v2CoatBasisExpert').addEventListener('click', () => setCoating('expert', coatMap));
  new MutationObserver(() => {
    const node = $('coatingZone');
    const match = node.textContent.match(/^Zone 0*(\d+)$/);
    const text = match ? zoneText(match[1]) : node.textContent;
    if (match) node.style.setProperty('--zone', colorOf(match[1]));
    node.classList.toggle('zone-tag-idle', !match);
    if (node.textContent !== text) node.textContent = text;
  }).observe($('coatingZone'), { childList: true, characterData: true, subtree: true });

  // 목록은 이상 행이 맨 위인 순서다. 불러올 때 그려 다시 칠하는 호출은 강조만 하고, 재생·탐색으로 Zone 이 바뀔 때(follow)만 따라간다.
  let coatScrolledByUser = 0;
  for (const type of ['wheel', 'touchstart', 'pointerdown', 'keydown']) $('v2CoatBars').addEventListener(type, () => { coatScrolledByUser = performance.now(); }, { passive: true });
  function markCurrentZone(follow = false) {
    for (const node of document.querySelectorAll('.order-chip, .coat-row')) {
      node.classList.toggle('is-current', anchor.zone_id != null && node.dataset.zone === String(anchor.zone_id) && !node.classList.contains('is-missing'));
    }
    const row = document.querySelector('.coat-row.is-current');
    const box = row?.parentElement;
    if (!follow || !box || box.scrollHeight <= box.clientHeight || performance.now() - coatScrolledByUser < 4000) return;
    const inner = row.getBoundingClientRect();
    const outer = box.getBoundingClientRect();
    if (inner.top < outer.top) box.scrollTop -= outer.top - inner.top;
    else if (inner.bottom > outer.bottom) box.scrollTop += inner.bottom - outer.bottom;
  }

  // ── 장면 크기와 측정값 표기 ───────────────────────────────
  // 기존 카메라는 장면을 화면 크기와 무관하게 약 290px로 맞춘다. 큰 화면에서는 그만큼 당겨 보여 준다.
  let fitZoom = null;
  function fillStage() {
    const canvas = $('canvas');
    if (!fitZoom || typeof zoom !== 'number') return;
    const target = fitZoom * Math.max(.7, Math.min(2.4, Math.min(canvas.clientHeight, canvas.clientWidth * .62) / 440));
    const steps = Math.round(Math.log(target / zoom) / Math.log(1.1));
    for (let i = 0; i < Math.abs(steps); i++) canvas.dispatchEvent(new WheelEvent('wheel', { deltaY: steps > 0 ? 100 : -100, cancelable: true }));
  }
  document.addEventListener('click', (event) => { if (event.target.closest('#cameraHome')) fillStage(); });
  // 답과 같은 단위(mm/s)로 보여 준다. 기존 스크립트는 m/s로 쓴다.
  const speedNode = $('realtimeHudSpeed');
  const inMillimetres = (text) => text.replace(/(\d+(?:\.\d+)?) m\/s/g, (_, value) => `${Math.round(Number(value) * 1000).toLocaleString('ko-KR')} mm/s`);
  new MutationObserver(() => {
    const text = inMillimetres(speedNode.textContent);
    if (text !== speedNode.textContent) speedNode.textContent = text;
    if (speedNode.title !== inMillimetres(speedNode.title)) speedNode.title = inMillimetres(speedNode.title);
  }).observe(speedNode, { childList: true, characterData: true, subtree: true, attributes: true, attributeFilter: ['title'] });

  // ── 재생 단추와 불러오기 실패 ─────────────────────────────
  const play = $('v2Play');
  const syncPlay = () => {
    const playing = $('play').textContent.trim() !== '재생';
    play.innerHTML = svg(playing ? 'pause' : 'play');
    play.setAttribute('aria-label', playing ? '일시정지' : '재생');
    play.title = playing ? '일시정지' : '재생';
  };
  play.addEventListener('click', () => $('play').click());
  new MutationObserver(syncPlay).observe($('play'), { childList: true, characterData: true, subtree: true });
  new MutationObserver(() => {
    const text = $('status').textContent.trim();
    const failed = Boolean(text) && !text.endsWith('준비 완료') && !text.endsWith('중입니다.');
    $('v2StageError').hidden = !failed;
    $('v2StageErrorText').textContent = failed ? `기록을 불러오지 못했습니다. ${text}` : '';
    if (failed) { record = null; armSeen = undefined; renderAll(); chat.setRecord(null); notices.load(null); }
  }).observe($('status'), { childList: true, characterData: true, subtree: true });
  $('v2Reload').addEventListener('click', () => $('session').dispatchEvent(new Event('change')));

  function renderAll() {
    renderScores(); renderZoneNote(); renderPose(); renderOrder(); renderCoating();
  }

  // ── 고정 질문 열 ───────────────────────────────────────────
  const TABS = [
    { id: 'pose', task: 'pose_recomm', label: '자세', icon: 'person-standing', hint: '자세에 대해 질문',
      presets: ['지금 자세는 어때?', '이 Zone에서 어떻게 고쳐?', '오른손을 어떻게 고쳐?', '왼손을 어떻게 고쳐?', '가장 문제가 큰 Zone은?'] },
    { id: 'order', task: 'trajectory_recomm', label: '순서', icon: 'route', hint: '순서에 대해 질문',
      presets: ['권장 Zone 순서는?', '실제 수행 순서는?', '누락한 Zone은?', '순서 준수율은?', '도포 방향은?'] },
    { id: 'coating', task: 'knowledge_search', label: '도막', icon: 'paint-roller', hint: '도막에 대해 질문',
      presets: ['도막이 과하거나 모자란 영역이 있어?', '이 Zone 속도와 이격거리는 어땠어?', '가장 먼저 고쳐야 할 점은?'] },
    { id: 'knowledge', task: 'knowledge_search', label: '지식', icon: 'book-open', hint: '도장 지식을 질문',
      presets: ['저도막이 뭐야?', '적정 이격거리는?', '적정 분사 속도는?', '과도막의 원인은?'] },
  ];
  // 답변 언어: [언어 코드, 국기 파일, 그 언어로 쓴 이름, 한국어 이름]. 기존 뷰어 질문 창과 같은 12개, 같은 순서다.
  const LANGUAGES = [['ko', 'kr', '한국어', '한국어'], ['en', 'us', 'English', '영어'], ['vi', 'vn', 'Tiếng Việt', '베트남어'],
    ['id', 'id', 'Bahasa Indonesia', '인도네시아어'], ['th', 'th', 'ไทย', '태국어'], ['ne', 'np', 'नेपाली', '네팔어'],
    ['my', 'mm', 'မြန်မာ', '미얀마어'], ['uz', 'uz', 'O‘zbekcha', '우즈베크어'], ['si', 'lk', 'සිංහල', '싱할라어'],
    ['ru', 'ru', 'Русский', '러시아어'], ['zh', 'cn', '中文', '중국어'], ['km', 'kh', 'ខ្មែរ', '크메르어']];
  const flagImage = (country) => {
    const image = el('img');
    image.src = `${options.flagBase || '/motion-viewer/assets/flags/'}${country}.svg`;
    image.alt = '';
    return image;
  };
  const MODEL = { quick: '8B', detailed: '27B' };
  const TIMEOUT_SECONDS = { quick: 45, detailed: 120 };
  const TRANSLATED_TIMEOUT_SECONDS = 180;   // 번역은 27B가 한다. 영어를 거치는 언어는 중앙값 24~34초, 가장 긴 답은 164초였다(2026-11-12 시험 서버)

  const chat = (() => {
    const root = $('v2Ask');
    const newConversation = () => `mv2-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 12)}`;
    let conversation = newConversation();
    let tab = TABS[0];
    let mode = 'quick';
    let data = null;
    let busy = null;

    const head = el('div', 'ask-head');
    const anchorBox = el('div', 'anchor');
    anchorBox.title = '질문은 이 재생 위치를 기준으로 답합니다';
    const anchorZone = el('span', 'zone-tag zone-tag-idle', '이동 중');
    const anchorTime = el('span', 'time', '0.0초');
    anchorBox.innerHTML = svg('map-pin');
    anchorBox.append(anchorZone, anchorTime);
    const fresh = iconButton('icon-btn', 'message-square-plus', '새 대화');
    head.append(anchorBox, fresh);

    const tabs = el('div', 'tabs');
    tabs.setAttribute('role', 'radiogroup');
    tabs.setAttribute('aria-label', '질문 종류');
    const log = el('div', 'ask-log');
    log.setAttribute('role', 'log');
    log.setAttribute('aria-label', '질문과 답');
    log.id = 'v2AskLog';
    const presets = el('div', 'presets');
    presets.setAttribute('role', 'group');
    presets.setAttribute('aria-label', '자주 묻는 질문');
    const form = el('form', 'composer');
    const line = el('div', 'composer-row');
    const input = el('input');
    input.setAttribute('aria-label', '질문');
    input.autocomplete = 'off';
    const send = iconButton('icon-btn primary send', 'send', '보내기');
    send.type = 'submit';
    line.append(input, send);
    const tools = el('div', 'composer-tools');
    const modes = el('div', 'seg');
    modes.setAttribute('role', 'group');
    modes.setAttribute('aria-label', '답변 수준');
    // 답변 언어는 국기 12개를 늘 보이게 둔다. 질문은 한국어로 보내고 답만 고른 언어로 받는다.
    let language = LANGUAGES[0];
    const flags = el('div', 'flags');
    flags.setAttribute('role', 'radiogroup');
    flags.setAttribute('aria-label', '답변 언어');
    const languageName = el('span', 'language-name');
    languageName.title = '답변 언어. 질문은 한국어로 보냅니다';
    const chooseLanguage = (index, focus = false) => {
      language = LANGUAGES[index];
      languageName.innerHTML = svg('languages');
      languageName.append(language[2]);
      languageName.lang = language[0];
      [...flags.children].forEach((button, i) => {
        button.setAttribute('aria-checked', String(i === index));
        button.tabIndex = i === index ? 0 : -1;
      });
      if (focus) flags.children[index].focus();
    };
    LANGUAGES.forEach(([code, country, name, korean], index) => {
      const button = el('button', 'flag');
      button.type = 'button';
      button.setAttribute('role', 'radio');
      button.title = code === 'ko' ? name : `${name} · ${korean}`;
      button.setAttribute('aria-label', button.title);
      button.append(flagImage(country));
      button.addEventListener('click', () => chooseLanguage(index));
      button.addEventListener('keydown', (event) => {
        const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[event.key];
        const next = step ? (index + step + LANGUAGES.length) % LANGUAGES.length
          : event.key === 'Home' ? 0 : event.key === 'End' ? LANGUAGES.length - 1 : null;
        if (next == null) return;
        event.preventDefault();
        chooseLanguage(next, true);
      });
      flags.append(button);
    });
    chooseLanguage(0);
    // 기록된 응답만 보여 주는 화면(링크용)에서는 준비된 질문만 쓴다.
    input.disabled = send.disabled = Boolean(options.presetOnly);
    tools.append(modes, languageName);
    form.append(line, flags, tools);
    root.append(head, tabs, log, presets, form);

    const empty = () => {
      if (log.children.length) return;
      log.append(el('p', 'ask-empty', options.notice || '재생을 멈춘 곳을 기준으로 답합니다'));
    };
    const scroll = () => { log.scrollTop = log.scrollHeight; };
    const chip = (icon, text, title) => {
      const node = el('span', 'chip');
      node.innerHTML = svg(icon);
      node.append(text);
      if (title) node.title = title;
      return node;
    };
    const bubble = (cls, text) => {
      log.querySelector('.ask-empty')?.remove();
      const node = el('div', `msg ${cls}`, text);
      log.append(node);
      scroll();
      return node;
    };

    for (const [id, icon, label] of [['quick', 'zap', '빠른 안내'], ['detailed', 'file-search', '상세 분석']]) {
      const button = el('button');
      button.type = 'button';
      button.innerHTML = svg(icon);
      button.append(label);
      button.title = `${label} · 모델이 답할 때 Qwen ${MODEL[id]}`;
      button.setAttribute('aria-pressed', String(id === mode));
      button.addEventListener('click', () => {
        mode = id;
        for (const other of modes.children) other.setAttribute('aria-pressed', String(other === button));
      });
      modes.append(button);
    }

    const selectTab = (next, focus = false) => {
      tab = next;
      for (const button of tabs.children) {
        const selected = button.dataset.tab === tab.id;
        button.setAttribute('aria-checked', String(selected));
        button.tabIndex = selected ? 0 : -1;
      }
      for (const feature of document.querySelectorAll('.feature')) feature.classList.toggle('is-asked', feature.dataset.feature === tab.id);
      input.placeholder = options.presetOnly ? '준비된 질문을 눌러 주세요' : tab.hint;
      presets.replaceChildren(...tab.presets.map((question) => {
        const button = el('button', 'preset', question);
        button.type = 'button';
        button.addEventListener('click', () => ask(question));
        return button;
      }));
      if (focus) input.focus();
    };
    TABS.forEach((item, index) => {
      const button = el('button');
      button.type = 'button';
      button.setAttribute('role', 'radio');
      button.dataset.tab = item.id;
      button.innerHTML = svg(item.icon);
      button.append(item.label);
      button.addEventListener('click', () => selectTab(item));
      button.addEventListener('keydown', (event) => {
        const move = { ArrowRight: 1, ArrowLeft: -1 }[event.key];
        if (!move) return;
        event.preventDefault();
        const next = TABS[(index + move + TABS.length) % TABS.length];
        selectTab(next);
        tabs.children[TABS.indexOf(next)].focus();
      });
      tabs.append(button);
    });
    for (const link of document.querySelectorAll('[data-ask-tab]')) {
      link.addEventListener('click', () => selectTab(TABS.find((item) => item.id === link.dataset.askTab), true));
    }

    const recordLabel = (payload) => `${payload.worker?.display_name || '작업자'} · ${payload.process_label || payload.process} · 학습모드 ${payload.mode_id} · ${payload.selected_session_id}`;
    // 기존 뷰어와 같은 형식의 문맥. 서버는 서명이 맞는 값만 쓴다.
    const context = (task, payload, state) => {
      const shared = {
        frame_direction_context: payload.frame_direction_context ?? null, frame_direction_signature: payload.frame_direction_signature,
        dialogue_context: payload.dialogue_evaluation ?? null, dialogue_signature: payload.query_context_signatures?.dialogue,
        coating_context: payload.coating_evaluation ?? null, coating_signature: payload.query_context_signatures?.coating_status,
        arm_pose_context: payload.arm_pose_evaluation ?? null, arm_pose_signature: payload.query_context_signatures?.arm_pose,
      };
      if (task === 'knowledge_search') return `motion_viewer_playback:${JSON.stringify({ record: recordLabel(payload), playback_context: state, ...shared })}`;
      const pose = payload.pose_evaluation || {};
      const order = payload.reference_order_evaluation || {};
      const analysis = task === 'pose_recomm'
        ? { record: recordLabel(payload), status: pose.status ?? null, similarity_percent: pose.similarity_percent ?? null, zones: pose.zones ?? null, corrections: pose.corrections ?? null }
        : { record: recordLabel(payload), recommended_flow: order.recommended_flow ?? null, selected_flow: order.selected_flow ?? null,
          adherence_rate: order.adherence_rate ?? null, best_reference_similarity: order.best_reference_similarity ?? null,
          position_similarity: order.position_similarity ?? null, position_distribution: order.position_distribution ?? null,
          reference_session_count: order.reference_session_count ?? null, direction_summary: payload.direction_summary || null,
          zone_keys: Object.keys(payload.pose_evaluation?.zones || {}).sort() };
      return `motion_viewer_analysis:${JSON.stringify({ ...shared, analysis_context: analysis, signature: payload.query_context_signatures?.[task], playback_context: state })}`;
    };

    const sourceChip = (result, sentMode) => {
      const source = result.answer_source
        || (result.retrieval_used ? 'documents' : result.pose_context || result.sequence_context || result.coating_context ? 'analysis' : null);
      const took = `서버 처리 ${result.latency_ms} ms`;
      if (source === 'analysis') return chip('chart-column', '채점·분석 결과', `${took}. 모델이 문장을 만들지 않고 채점·분석 값을 그대로 옮긴 답입니다.`);
      if (source === 'documents') return chip('book-open', `문서 검색 · ${MODEL[sentMode]}`, `${took}. 문서를 찾아 Qwen ${MODEL[sentMode]}가 답했습니다.`);
      if (source === 'conversation') return chip('message-circle', `모델 대화 · ${MODEL[sentMode]}`, `${took}. Qwen ${MODEL[sentMode]}가 기록 값을 받아 답했습니다.`);
      return null;
    };

    async function ask(question, answerLanguage = language) {
      const prompt = String(question || '').trim();
      if (!prompt || busy) return;
      if (!data && tab.id !== 'knowledge') {
        bubble('msg-a is-error', '기록을 불러온 뒤에 질문할 수 있습니다.');
        return;
      }
      const sent = { tab, mode, language: answerLanguage, data, state: { map_num: anchor.map_num, zone_id: anchor.zone_id, source_index: anchor.source_index, playhead_seconds: anchor.playhead_seconds },
        armPending: tab.id === 'pose' && data && data.arm_pose_evaluation === undefined };
      const asked = bubble('msg-q', prompt);
      const meta = el('div', 'msg-meta');
      const pin = el('button', 'chip');
      pin.type = 'button';
      pin.innerHTML = svg('map-pin');
      pin.append(`${sent.state.zone_id == null ? 'Zone 없음' : zoneText(sent.state.zone_id)} · ${sent.state.playhead_seconds.toFixed(2)}초`);
      pin.title = '질문한 재생 위치로 이동';
      pin.addEventListener('click', () => seekToSeconds(sent.state.playhead_seconds));
      meta.append(pin);
      asked.append(meta);
      addPin(sent.state.playhead_seconds, prompt);
      input.value = '';

      const waiting = bubble('msg-a');
      const wait = el('span', 'msg-wait', '답을 찾는 중 0초');
      const cancel = iconButton('icon-btn', 'x', '질문 취소');
      waiting.append(wait, cancel);
      const controller = new AbortController();
      const started = performance.now();
      let reason = null;
      const ticker = setInterval(() => { wait.textContent = `답을 찾는 중 ${Math.round((performance.now() - started) / 1000)}초`; }, 1000);
      const limitSeconds = sent.language[0] === 'ko' ? TIMEOUT_SECONDS[sent.mode] : TRANSLATED_TIMEOUT_SECONDS;
      const limit = setTimeout(() => { reason = 'timeout'; controller.abort(); }, limitSeconds * 1000);
      cancel.addEventListener('click', () => { reason = 'cancel'; controller.abort(); });
      busy = controller;
      send.disabled = true;
      const fail = (message, retry = true, inKorean = false) => {
        const node = bubble('msg-a is-error', message);
        if (!input.value && !options.presetOnly) input.value = prompt;
        if (!retry) return;
        const row = el('div', 'msg-meta');
        const resend = (label, answerIn) => {
          const button = el('button', 'chip', label);
          button.type = 'button';
          button.addEventListener('click', () => { if (input.value === prompt) input.value = ''; ask(prompt, answerIn); });
          row.append(button);
        };
        resend('다시 보내기', sent.language);
        if (inKorean) resend('한국어로 보기', LANGUAGES[0]);   // 서버 안내: 번역이 안 되면 다시 보내거나 한국어로 받는다
        node.append(row);
        scroll();
      };
      try {
        const response = await fetch('/api/v1/copilot/query', {
          method: 'POST', signal: controller.signal,
          headers: { 'Content-Type': 'application/json', 'X-Copilot-Source': 'motion-viewer', 'X-Copilot-Conversation': conversation },
          body: JSON.stringify({
            user_id: sent.data?.worker?.worker_id || 'COPILOT-DEMO-USER',
            session_id: sent.data?.selected_session_id ?? null,
            scenario_id: sent.data?.mode_id != null ? String(sent.data.mode_id) : null,
            task_type: sent.tab.task, prompt, response_mode: sent.mode,
            language: 'ko', response_language: sent.language[0], use_rag: 'Y',
            context: sent.data ? context(sent.tab.task, sent.data, sent.state) : null,
          }),
        });
        let payload = null;
        try { payload = await response.json(); } catch { /* 본문이 JSON이 아니다 */ }
        waiting.remove();
        if (!payload) { fail(`서버가 답을 주지 못했습니다(HTTP ${response.status}). 잠시 뒤 다시 보내 주세요.`); return; }
        if (response.status === 503 && sent.language[0] !== 'ko') { fail(`답을 ${sent.language[3]}로 옮기지 못했습니다.`, true, true); return; }
        if (!response.ok || payload.status !== 'SUCCESS') { fail(`질문을 처리하지 못했습니다. ${payload.message || `HTTP ${response.status}`}`, true, sent.language[0] !== 'ko'); return; }
        const result = { ...payload.data, answer_source: response.headers.get('X-Copilot-Answer-Source') };
        const answer = bubble('msg-a', result.answer);
        answer.lang = sent.language[0];
        const row = el('div', 'msg-meta');
        row.lang = 'ko';
        const source = sourceChip(result, sent.mode);
        if (source) row.append(source);
        if (result.citations?.length) row.append(chip('book-open', `출처 ${result.citations.length}`, result.citations.join('\n')));
        if (sent.language[0] !== 'ko') {
          const said = el('span', 'chip');
          said.append(flagImage(sent.language[1]), sent.language[2]);
          said.title = `답변 언어 · ${sent.language[3]}`;
          row.append(said);
        }
        if (result.context_status === 'not_available') {
          const none = chip('info', '기록 자료 없음', '이 질문에 쓸 기록 분석 자료가 없었습니다.');
          none.classList.add('is-warn');
          row.append(none);
        }
        if (result.degraded) { const cut = chip('triangle-alert', '줄인 답', '시간 안에 생성이 끝나지 않아 줄인 답입니다.'); cut.classList.add('is-warn'); row.append(cut); }
        if (sent.armPending) { const early = chip('info', '자세 채점 전', '자세 채점이 끝나기 전에 보낸 질문이라 팔 자세 결과가 답에 없을 수 있습니다.'); early.classList.add('is-warn'); row.append(early); }
        if (row.children.length) answer.append(row);
        scroll();
      } catch (error) {
        waiting.remove();
        if (reason === 'cancel') { if (!input.value && !options.presetOnly) input.value = prompt; return; }
        fail(reason === 'timeout' ? `${limitSeconds}초 안에 답이 오지 않아 멈췄습니다.` : '서버에 연결하지 못했습니다. 네트워크를 확인해 주세요.', true, reason === 'timeout' && sent.language[0] !== 'ko');
      } finally {
        clearInterval(ticker);
        clearTimeout(limit);
        busy = null;
        send.disabled = Boolean(options.presetOnly);
      }
    }
    form.addEventListener('submit', (event) => { event.preventDefault(); ask(input.value); });

    const pins = $('v2Pins');
    function addPin(seconds, prompt) {
      const duration = Number(data?.duration_seconds) || 0;
      if (!duration) return;
      const pin = el('button', 'pin');
      pin.type = 'button';
      pin.style.left = `${Math.max(0, Math.min(100, seconds / duration * 100))}%`;
      pin.title = `${seconds.toFixed(1)}초에 질문: ${prompt}`;
      pin.setAttribute('aria-label', pin.title);
      pin.addEventListener('click', () => seekToSeconds(seconds));
      pins.append(pin);
      while (pins.children.length > 12) pins.firstChild.remove();
    }
    const reset = () => {
      busy?.abort();
      conversation = newConversation();
      log.replaceChildren();
      pins.replaceChildren();
      empty();
    };
    fresh.addEventListener('click', reset);
    selectTab(tab);
    empty();

    return {
      setRecord(payload) {
        const changed = (payload && recordLabel(payload)) !== (data && recordLabel(data));
        data = payload;
        if (changed) reset();   // 다른 기록의 대화가 섞이지 않게 새 대화로 시작한다
      },
      setAnchor(state) {
        tagZone(anchorZone, state.zone_id, state.zone_id != null && !state.spraying ? ' · 이동 중' : '');
        anchorTime.textContent = `${state.playhead_seconds.toFixed(2)}초`;
      },
    };
  })();

  // ── 재생 알림 ─────────────────────────────────────────────
  // 서버가 공개 이상 알림과 같은 함수로 판정한 알림을 재생 시각에 맞춰 띄운다. 고르기·짧은 표기는 motion_viewer_alerts.js.
  const notices = (() => {
    const A = window.motionViewerAlerts;
    const toastHost = $('v2Toast'), marks = $('v2AlertMarks'), button = $('v2AlertsButton'), list = $('v2AlertList');
    const items = $('v2AlertItems'), note = $('v2AlertNote'), count = $('v2AlertCount');
    const NOTE = { process_not_supported: '블라스팅 기록은 순서를 판정하지 않습니다', sequence_reference_unavailable: '전문가 순서 기준이 없습니다',
      sequence_reference_context_unavailable: '전문가 순서 기준이 없습니다', no_sequences: '순서를 판정할 분사 구간이 없습니다' };
    let all = [], last = 0, seeking = true, lastKey = null, lastAt = -Infinity, current = null;
    const tone = (alert) => `tone-${A.KIND[alert.anomaly_type]?.tone || 'warn'}`;
    const describe = (alert) => `${A.KIND[alert.anomaly_type]?.label || '알림'} · ${zoneText(alert.zone_id)} · ${Number(alert.at_seconds).toFixed(1)}초. ${alert.anomaly_type === 'direction_deviation' ? '' : alert.message || ''}`;   // 방향 문장은 영어 키가 섞여 있어 사용자 확인 전까지 쓰지 않는다
    const face = (alert) => {   // [종류 아이콘][Zone 네모+번호][짧은 표기]
      const icon = el('span', `alert-icon ${tone(alert)}`);
      icon.innerHTML = svg(A.KIND[alert.anomaly_type]?.icon || 'triangle-alert');
      const zone = el('span', 'zone-tag');
      tagZone(zone, alert.zone_id);
      return [icon, zone, el('span', 'alert-word', A.short(alert))];
    };
    const labelled = (node, alert) => { node.title = describe(alert); node.setAttribute('aria-label', node.title); return node; };
    function go(alert) {
      current = alert.id;
      if (alert.anomaly_type.startsWith('coating_')) setCoating('target');   // 도막 패널의 그 행이 강조되게
      seekToIndex(A.seekIndex(record?.frames || [], Number(alert.seek_seconds), Number(alert.at_seconds)));
      paintList();
    }
    function toast(alert) {
      const node = labelled(el('button', 'toast-item'), alert);
      node.type = 'button';
      node.append(...face(alert));
      node.addEventListener('click', () => go(alert));
      toastHost.prepend(node);
      while (toastHost.children.length > 2) toastHost.lastChild.remove();
      const expire = () => (node.matches(':hover, :focus-within') ? setTimeout(expire, 1000) : node.remove());
      setTimeout(expire, 5000);
    }
    function paintMarks() {
      const duration = Number(record?.duration_seconds) || 0;
      marks.replaceChildren(...(duration ? all : []).map((alert) => {
        const mark = labelled(el('button', `alert-mark ${tone(alert)}`), alert);
        mark.type = 'button';
        mark.tabIndex = -1;   // 키보드는 목록으로 같은 일을 한다
        mark.style.left = `${Math.max(0, Math.min(100, alert.at_seconds / duration * 100))}%`;
        mark.addEventListener('click', () => go(alert));
        return mark;
      }));
    }
    function paintAhead() {
      for (const row of items.querySelectorAll('.alert-row')) row.classList.toggle('is-ahead', Number(row.dataset.at) > anchor.playhead_seconds);
    }
    function paintList() {
      const sequence = record?.sequence_alerts;
      const pending = Boolean(record) && sequence === undefined;
      button.setAttribute('aria-busy', String(pending));
      button.title = pending ? '알림 판정 중' : `알림 ${all.length}건`;
      button.setAttribute('aria-label', button.title);
      count.hidden = pending || !all.length;
      count.textContent = String(all.length);
      const reason = sequence && sequence.status !== 'available' ? NOTE[sequence.reason_code] || '순서 판정을 불러오지 못했습니다' : '';
      const empty = !all.length ? (pending ? '알림 판정 중' : '이 기록에는 알림이 없습니다') : '';
      note.textContent = [reason, empty].filter(Boolean).join('. ');
      note.hidden = !note.textContent;
      items.replaceChildren(...all.map((alert) => {
        const row = labelled(el('button', 'alert-row'), alert);
        row.type = 'button';
        row.dataset.at = String(alert.at_seconds);
        row.append(...face(alert), el('span', 'alert-time', `${Number(alert.at_seconds).toFixed(1)}초`));
        if (alert.id === current) row.setAttribute('aria-current', 'true');
        row.addEventListener('click', () => go(alert));
        const item = el('li');
        item.append(row);
        return item;
      }));
      paintAhead();
    }
    const open = (value) => {
      list.hidden = !value;
      button.setAttribute('aria-expanded', String(value));
      if (value) {
        list.style.maxHeight = `${Math.round(($('v2Toast').parentElement.clientHeight || 400) * 0.5)}px`;   // 장면 높이의 절반
        paintList(); items.querySelector('button')?.focus();
      }
    };
    button.addEventListener('click', () => open(list.hidden));
    list.addEventListener('keydown', (event) => {
      const rows = [...items.querySelectorAll('button')];
      if ((event.key !== 'ArrowDown' && event.key !== 'ArrowUp') || !rows.length) return;
      event.preventDefault();
      const step = event.key === 'ArrowDown' ? 1 : -1;
      rows[(rows.indexOf(document.activeElement) + step + rows.length) % rows.length].focus();
    });
    document.addEventListener('click', (event) => { if (!list.hidden && !event.target.closest('.alerts')) open(false); });
    document.addEventListener('focusin', (event) => { if (!list.hidden && !event.target.closest('.alerts')) open(false); });
    document.addEventListener('keydown', (event) => { if (event.key === 'Escape' && !list.hidden) { open(false); button.focus(); } });
    // 탐색은 알림 없이 다시 맞춘다. 캡처 단계라 기존 재생 스크립트의 처리(그리기)보다 먼저 표시가 선다.
    document.addEventListener('input', (event) => { if (event.target.id === 'timeline') seeking = true; }, true);
    document.addEventListener('click', (event) => { if (event.target.closest('#restart')) seeking = true; }, true);

    const sync = () => { all = A.merge(record); last = anchor.playhead_seconds; seeking = true; lastKey = null; paintMarks(); paintList(); };
    return {
      load(payload) {
        current = null;
        toastHost.replaceChildren();
        sync();
        if (!payload) return;
        const query = new URLSearchParams({ worker_id: payload.worker?.worker_id, process: payload.process,
          mode_id: payload.mode_id, session_id: payload.selected_session_id });
        fetch('/motion-viewer/api/alerts?' + query)
          .then((response) => (response.ok ? response.json() : Promise.reject(new Error(String(response.status)))))
          .then((result) => { payload.sequence_alerts = result.sequence_alerts; })
          .catch(() => { payload.sequence_alerts = { status: 'unavailable', reason_code: 'request_failed', alerts: [] }; })
          .finally(() => { if (record === payload) sync(); });   // 늦게 온 다른 기록의 응답은 버린다
      },
      frame(seconds) {
        const jumped = A.isJump(last, seconds, Number($('speed').value) || 1, seeking);
        if (jumped) lastKey = null;
        for (const alert of A.crossed(all, last, seconds, jumped)) {
          const key = `${alert.anomaly_type}:${alert.zone_id}`;
          if (key === lastKey && alert.at_seconds - lastAt <= 3) continue;   // 같은 종류·Zone 이 3초 안에 연달아 오면 한 번만(목록·표지에는 다 있다)
          lastKey = key;
          lastAt = alert.at_seconds;
          toast(alert);
        }
        last = seconds;
        seeking = false;
        if (!list.hidden) paintAhead();
      },
    };
  })();

  // ── 이벤트 ────────────────────────────────────────────────
  addEventListener('motion-viewer-data', (event) => {
    record = event.detail;
    // 기록마다 목표 대비로 연다. 목표 대비 판정이 없는 기록은 전문가 대비(막대)로 바로 연다.
    coatBasis = targetReady() ? 'target' : 'expert';
    coatMap = false;
    armSeen = record.arm_pose_evaluation;
    anchor = anchorAt(0, 0);
    renderAll();
    chat.setRecord(record);
    chat.setAnchor(anchor);
    notices.load(record);
    fitZoom = typeof zoom === 'number' ? zoom : null;
    requestAnimationFrame(() => { dispatchEvent(new Event('resize')); fillStage(); });
  });
  addEventListener('motion-viewer-frame', (event) => {
    if (!record) return;
    const { sourceIndex, playheadSeconds } = event.detail;
    const index = Number.isInteger(sourceIndex) && sourceIndex >= 0 ? sourceIndex : anchor.source_index;
    const seconds = Number.isFinite(Number(playheadSeconds)) && playheadSeconds >= 0 ? Number(playheadSeconds) : anchor.playhead_seconds;
    const previous = anchor;
    anchor = anchorAt(index, seconds);
    if (record.arm_pose_evaluation !== armSeen) {   // 자세 채점은 기록보다 늦게 도착한다
      armSeen = record.arm_pose_evaluation;
      renderScores();
      renderPose();
    }
    if (previous.zone_id !== anchor.zone_id || previous.map_num !== anchor.map_num || previous.spraying !== anchor.spraying) {
      renderZoneNote();
      markCurrentZone(true);
    }
    renderPoseNow();
    markFlaggedJoints(event.detail.frame);
    chat.setAnchor(anchor);
    notices.frame(seconds);
  });

  hydrate(document);
  syncPlay();
  renderAll();
  requestAnimationFrame(() => dispatchEvent(new Event('resize')));   // 스켈레톤·도막 캔버스가 새 배치의 크기로 다시 그려지게 한다
})();
