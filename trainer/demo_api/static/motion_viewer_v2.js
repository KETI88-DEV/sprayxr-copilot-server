// 모션 3D 뷰어 v2. 기존 재생 스크립트가 보내는 motion-viewer-data / motion-viewer-frame 이벤트를 받아
// 결과 한 줄, HUD 교정 칸, 세 기능 그림(자세·순서·도막), 고정 질문 열을 그린다. 값은 서버가 준 것만 쓴다.
(() => {
  'use strict';
  const ICON = {"user":"<path d=\"M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2\" /><circle cx=\"12\" cy=\"7\" r=\"4\" />","spray-can":"<path d=\"M3 3h.01\" /><path d=\"M7 5h.01\" /><path d=\"M11 7h.01\" /><path d=\"M3 7h.01\" /><path d=\"M7 9h.01\" /><path d=\"M3 11h.01\" /><rect width=\"4\" height=\"4\" x=\"15\" y=\"5\" /><path d=\"m19 9 2 2v10c0 .6-.4 1-1 1h-6c-.6 0-1-.4-1-1V11l2-2\" /><path d=\"m13 14 8-2\" /><path d=\"m13 19 8-2\" />","graduation-cap":"<path d=\"M21.42 10.922a1 1 0 0 0-.019-1.838L12.83 5.18a2 2 0 0 0-1.66 0L2.6 9.08a1 1 0 0 0 0 1.832l8.57 3.908a2 2 0 0 0 1.66 0z\" /><path d=\"M22 10v6\" /><path d=\"M6 12.5V16a6 3 0 0 0 12 0v-3.5\" />","history":"<path d=\"M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8\" /><path d=\"M3 3v5h5\" /><path d=\"M12 7v5l4 2\" />","gauge":"<path d=\"m12 14 4-4\" /><path d=\"M3.34 19a10 10 0 1 1 17.32 0\" />","person-standing":"<circle cx=\"12\" cy=\"5\" r=\"1\" /><path d=\"m9 20 3-6 3 6\" /><path d=\"m6 8 6 2 6-2\" /><path d=\"M12 10v4\" />","route":"<circle cx=\"6\" cy=\"19\" r=\"3\" /><path d=\"M9 19h8.5a3.5 3.5 0 0 0 0-7h-11a3.5 3.5 0 0 1 0-7H15\" /><circle cx=\"18\" cy=\"5\" r=\"3\" />","crosshair":"<circle cx=\"12\" cy=\"12\" r=\"10\" /><line x1=\"22\" x2=\"18\" y1=\"12\" y2=\"12\" /><line x1=\"6\" x2=\"2\" y1=\"12\" y2=\"12\" /><line x1=\"12\" x2=\"12\" y1=\"6\" y2=\"2\" /><line x1=\"12\" x2=\"12\" y1=\"22\" y2=\"18\" />","play":"<path d=\"M5 5a2 2 0 0 1 3.008-1.728l11.997 6.998a2 2 0 0 1 .003 3.458l-12 7A2 2 0 0 1 5 19z\" />","pause":"<rect x=\"14\" y=\"3\" width=\"5\" height=\"18\" rx=\"1\" /><rect x=\"5\" y=\"3\" width=\"5\" height=\"18\" rx=\"1\" />","skip-back":"<path d=\"M17.971 4.285A2 2 0 0 1 21 6v12a2 2 0 0 1-3.029 1.715l-9.997-5.998a2 2 0 0 1-.003-3.432z\" /><path d=\"M3 20V4\" />","focus":"<circle cx=\"12\" cy=\"12\" r=\"3\" /><path d=\"M3 7V5a2 2 0 0 1 2-2h2\" /><path d=\"M17 3h2a2 2 0 0 1 2 2v2\" /><path d=\"M21 17v2a2 2 0 0 1-2 2h-2\" /><path d=\"M7 21H5a2 2 0 0 1-2-2v-2\" />","ruler":"<path d=\"M21.3 15.3a2.4 2.4 0 0 1 0 3.4l-2.6 2.6a2.4 2.4 0 0 1-3.4 0L2.7 8.7a2.41 2.41 0 0 1 0-3.4l2.6-2.6a2.41 2.41 0 0 1 3.4 0Z\" /><path d=\"m14.5 12.5 2-2\" /><path d=\"m11.5 9.5 2-2\" /><path d=\"m8.5 6.5 2-2\" /><path d=\"m17.5 15.5 2-2\" />","axis-3d":"<path d=\"M13.5 10.5 15 9\" /><path d=\"M4 4v15a1 1 0 0 0 1 1h15\" /><path d=\"M4.293 19.707 6 18\" /><path d=\"m9 15 1.5-1.5\" />","arrow-right-left":"<path d=\"m16 3 4 4-4 4\" /><path d=\"M20 7H4\" /><path d=\"m8 21-4-4 4-4\" /><path d=\"M4 17h16\" />","scan":"<path d=\"M3 7V5a2 2 0 0 1 2-2h2\" /><path d=\"M17 3h2a2 2 0 0 1 2 2v2\" /><path d=\"M21 17v2a2 2 0 0 1-2 2h-2\" /><path d=\"M7 21H5a2 2 0 0 1-2-2v-2\" />","layers":"<path d=\"M12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83z\" /><path d=\"M2 12a1 1 0 0 0 .58.91l8.6 3.91a2 2 0 0 0 1.65 0l8.58-3.9A1 1 0 0 0 22 12\" /><path d=\"M2 17a1 1 0 0 0 .58.91l8.6 3.91a2 2 0 0 0 1.65 0l8.58-3.9A1 1 0 0 0 22 17\" />","paint-roller":"<rect width=\"16\" height=\"6\" x=\"2\" y=\"2\" rx=\"2\" /><path d=\"M10 16v-2a2 2 0 0 1 2-2h8a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2h-2\" /><rect width=\"4\" height=\"6\" x=\"8\" y=\"16\" rx=\"1\" />","rotate-ccw":"<path d=\"M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8\" /><path d=\"M3 3v5h5\" />","rotate-cw":"<path d=\"M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8\" /><path d=\"M21 3v5h-5\" />","locate-fixed":"<line x1=\"2\" x2=\"5\" y1=\"12\" y2=\"12\" /><line x1=\"19\" x2=\"22\" y1=\"12\" y2=\"12\" /><line x1=\"12\" x2=\"12\" y1=\"2\" y2=\"5\" /><line x1=\"12\" x2=\"12\" y1=\"19\" y2=\"22\" /><circle cx=\"12\" cy=\"12\" r=\"7\" /><circle cx=\"12\" cy=\"12\" r=\"3\" />","map-pin":"<path d=\"M20 10c0 4.993-5.539 10.193-7.399 11.799a1 1 0 0 1-1.202 0C9.539 20.193 4 14.993 4 10a8 8 0 0 1 16 0\" /><circle cx=\"12\" cy=\"10\" r=\"3\" />","book-open":"<path d=\"M12 5v16\" /><path d=\"M20.001 19A2 2 0 0022 17V5a2 2 0 00-1.999-2L16 3.002A5 5 0 0012 5a5 5 0 00-4-2H4a2 2 0 00-2 2v12a2 2 0 001.999 2H8a5 5 0 014 2 5 5 0 014-2z\" />","zap":"<path d=\"M15.914 4a1.5 1.5 0 00-2.474-1.561l-9 9A1.5 1.5 0 005.5 14h4.002a.5.5 0 01.471.666L8.086 20a1.5 1.5 0 002.475 1.56l9-9A1.5 1.5 0 0018.5 10h-3.997a.5.5 0 01-.472-.667z\" />","file-search":"<path d=\"M6 22a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h8a2.4 2.4 0 0 1 1.704.706l3.588 3.588A2.4 2.4 0 0 1 20 8v12a2 2 0 0 1-2 2z\" /><path d=\"M14 2v5a1 1 0 0 0 1 1h5\" /><circle cx=\"11.5\" cy=\"14.5\" r=\"2.5\" /><path d=\"M13.3 16.3 15 18\" />","send":"<path d=\"M14.536 21.686a.5.5 0 0 0 .937-.024l6.5-19a.496.496 0 0 0-.635-.635l-19 6.5a.5.5 0 0 0-.024.937l7.93 3.18a2 2 0 0 1 1.112 1.11z\" /><path d=\"m21.854 2.147-10.94 10.939\" />","message-square-plus":"<path d=\"M22 17a2 2 0 0 1-2 2H6.828a2 2 0 0 0-1.414.586l-2.202 2.202A.71.71 0 0 1 2 21.286V5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2z\" /><path d=\"M12 8v6\" /><path d=\"M9 11h6\" />","chart-column":"<path d=\"M3 3v16a2 2 0 0 0 2 2h16\" /><path d=\"M18 17V9\" /><path d=\"M13 17V5\" /><path d=\"M8 17v-3\" />","message-circle":"<path d=\"M2.992 16.342a2 2 0 0 1 .094 1.167l-1.065 3.29a1 1 0 0 0 1.236 1.168l3.413-.998a2 2 0 0 1 1.099.092 10 10 0 1 0-4.777-4.719\" />","triangle-alert":"<path d=\"m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3\" /><path d=\"M12 9v4\" /><path d=\"M12 17h.01\" />","languages":"<path d=\"m5 8 6 6\" /><path d=\"m4 14 6-6 2-3\" /><path d=\"M2 5h12\" /><path d=\"M7 2h1\" /><path d=\"m22 22-5-10-5 10\" /><path d=\"M14 18h6\" />","chevron-down":"<path d=\"m6 9 6 6 6-6\" />","info":"<circle cx=\"12\" cy=\"12\" r=\"10\" /><path d=\"M12 16v-4\" /><path d=\"M12 8h.01\" />","x":"<path d=\"M18 6 6 18\" /><path d=\"m6 6 12 12\" />","square":"<rect width=\"18\" height=\"18\" x=\"3\" y=\"3\" rx=\"2\" />","map":"<path d=\"M14.106 5.553a2 2 0 0 0 1.788 0l3.659-1.83A1 1 0 0 1 21 4.619v12.764a1 1 0 0 1-.553.894l-4.553 2.277a2 2 0 0 1-1.788 0l-4.212-2.106a2 2 0 0 0-1.788 0l-3.659 1.83A1 1 0 0 1 3 19.381V6.618a1 1 0 0 1 .553-.894l4.553-2.277a2 2 0 0 1 1.788 0z\" /><path d=\"M15 5.764v15\" /><path d=\"M9 3.236v15\" />","trending-down":"<path d=\"M16 17h6v-6\" /><path d=\"m22 17-8.5-8.5-5 5L2 7\" />","route-off":"<circle cx=\"6\" cy=\"19\" r=\"3\" /><path d=\"M9 19h8.5c.4 0 .9-.1 1.3-.2\" /><path d=\"M5.2 5.2A3.5 3.53 0 0 0 6.5 12H12\" /><path d=\"m2 2 20 20\" /><path d=\"M21 15.3a3.5 3.5 0 0 0-3.3-3.3\" /><path d=\"M15 5h-4.3\" /><circle cx=\"18\" cy=\"5\" r=\"3\" />","arrow-up-to-line":"<path d=\"M5 3h14\" /><path d=\"m18 13-6-6-6 6\" /><path d=\"M12 7v14\" />","arrow-down-to-line":"<path d=\"M12 17V3\" /><path d=\"m6 11 6 6 6-6\" /><path d=\"M19 21H5\" />","loader-circle":"<path d=\"M21 12a9 9 0 1 1-6.219-8.56\" />","bell":"<path d=\"M10.268 21a2 2 0 0 0 3.464 0\" /><path d=\"M3.262 15.326A1 1 0 0 0 4 17h16a1 1 0 0 0 .74-1.673C19.41 13.956 18 12.499 18 8A6 6 0 0 0 6 8c0 4.499-1.411 5.956-2.738 7.326\" />","chevron-left":"<path d=\"m15 18-6-6 6-6\" />","chevron-right":"<path d=\"m9 18 6-6-6-6\" />","grip-vertical":"<circle cx=\"9\" cy=\"12\" r=\"1\" /><circle cx=\"9\" cy=\"5\" r=\"1\" /><circle cx=\"9\" cy=\"19\" r=\"1\" /><circle cx=\"15\" cy=\"12\" r=\"1\" /><circle cx=\"15\" cy=\"5\" r=\"1\" /><circle cx=\"15\" cy=\"19\" r=\"1\" />","thumbs-down":"<path d=\"M9 18.12 10 14H4.17a2 2 0 0 1-1.92-2.56l2.33-8A2 2 0 0 1 6.5 2H20a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-2.76a2 2 0 0 0-1.79 1.11L12 22a3.13 3.13 0 0 1-3-3.88Z\" /><path d=\"M17 14V2\" />","message-square-text":"<path d=\"M22 17a2 2 0 0 1-2 2H6.828a2 2 0 0 0-1.414.586l-2.202 2.202A.71.71 0 0 1 2 21.286V5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2z\" /><path d=\"M7 11h10\" /><path d=\"M7 15h6\" /><path d=\"M7 7h8\" />","check":"<path d=\"M20 6 9 17l-5-5\" />","maximize-2":"<path d=\"M15 3h6v6\" /><path d=\"m21 3-7 7\" /><path d=\"m3 21 7-7\" /><path d=\"M9 21H3v-6\" />","plus":"<path d=\"M5 12h14\" /><path d=\"M12 5v14\" />","list-checks":"<path d=\"M13 5h8\" /><path d=\"M13 12h8\" /><path d=\"M13 19h8\" /><path d=\"m3 17 2 2 4-4\" /><path d=\"m3 7 2 2 4-4\" />","box":"<path d=\"M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z\" /><path d=\"m3.3 7 8.7 5 8.7-5\" /><path d=\"M12 22V12\" />"};
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
  const DIRECTION = window.motionViewerAlerts.DIRECTION;
  const L = window.motionViewerV2Layout;
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

  // ── HUD: 점·값은 기존 스크립트가 쓴다. 이동 문구만 바꾸고, 교정은 오른쪽 끝 단추로 HUD 아래에 편다 ──
  // 기존 스크립트(v1과 같이 쓴다)는 매 프레임 '분사 중'/'이동 중'을 쓴다. Zone 이름이 보이면 분사 중이라 그 낱말은 숨긴다.
  const activityNode = $('realtimeHudActivity');
  new MutationObserver(() => {
    const text = activityNode.textContent.trim();
    if (text === '이동 중') activityNode.textContent = 'Zone 사이 이동';
    activityNode.hidden = text === '분사 중';
  }).observe(activityNode, { childList: true, characterData: true, subtree: true });

  const usableMetric = (metric) => metric.available !== false && metric.similarity_percent != null;
  const metricShow = (metric, number, label) => {
    const unit = { deg: '°', mm: ' mm', um: ' µm' }[metric.unit] || '';
    return label || DIRECTION[number] || (Number.isFinite(Number(number)) ? `${Number(number).toFixed(1)}${unit}` : '—');
  };
  const zoneAt = (state) => (state.zone_id == null ? null : record?.pose_evaluation?.zones?.[zoneKey(state.map_num, state.zone_id)] || null);
  let hudOpen = false;
  function renderHudFix() {
    const zone = zoneAt(anchor);
    const corrections = zone?.corrections || [];
    const metrics = (zone?.metrics || []).filter(usableMetric);
    const button = $('v2HudFix');
    const panel = $('v2HudPanel');
    button.hidden = !corrections.length && !metrics.length;
    panel.hidden = button.hidden || !hudOpen;
    button.setAttribute('aria-expanded', String(!panel.hidden));
    $('v2HudFixCount').textContent = corrections.length ? String(corrections.length) : '';
    button.title = button.hidden ? '' : `${zoneText(anchor.zone_id)} ${corrections.length ? `교정 ${corrections.length}건과 ` : ''}항목별 비교`;
    button.setAttribute('aria-label', button.title || '교정');
    if (panel.hidden) return;
    // 자세 교정이 없어도 도막 등급이 below·above 면 '차이가 작습니다'는 도막 패널과 반대로 읽힌다. 서버가 낸 등급 글자를 그대로 말한다.
    const coat = (targetOf()?.zones || []).find((row) => String(row.zone_id) === String(anchor.zone_id) && (row.map_num == null || String(row.map_num) === String(anchor.map_num)));
    const coatNote = coat && (coat.grade === 'below' || coat.grade === 'above') && coat.thickness_um != null
      ? `도막 ${targetOf().grade_labels?.[coat.grade] ?? coat.grade} · ${Math.round(Number(coat.thickness_um))} µm` : '';
    const lines = corrections.length ? corrections : [coatNote || '전문가 기록과 차이가 작습니다'];
    $('v2HudFixList').replaceChildren(...lines.map((line) => el('li', null, line)));
    $('v2ZoneDetail').replaceChildren(...metrics.map((metric) => {
      const item = el('div', 'metric');
      const track = el('div', 'metric-track');
      const fill = el('div', 'metric-fill');
      fill.style.width = `${Math.max(0, Math.min(100, Number(metric.similarity_percent)))}%`;
      track.append(fill);
      const value = el('div', 'metric-value');
      value.append(metricShow(metric, metric.actual, metric.actual_label), el('span', null, ` / 전문가 ${metricShow(metric, metric.reference, metric.reference_label)}`));
      item.append(el('div', 'metric-name', metric.label), track, value);
      return item;
    }));
  }
  $('v2HudFix').addEventListener('click', () => { hudOpen = !hudOpen; renderHudFix(); });
  // 장면 칸의 크기가 바뀌면 캔버스 해상도를 다시 맞춘다. 창 크기 이벤트만 믿으면 아래 칸이 접히거나 펼쳐질 때
  // 캔버스가 예전 높이로 남아, 오른쪽 아래 좌표축이 캔버스 밖에 그려진다.
  // 크기가 바뀌면(창·분석 칸 접기·손잡이 끌기·화면 나누기) 잠시 뒤 구조물이 잘리지 않게 줌도 다시 맞춘다.
  let refit = 0;
  new ResizeObserver(() => { resize(); clearTimeout(refit); refit = setTimeout(() => fillStage(), 150); }).observe($('canvas').parentElement);

  // ── 자세: 스켈레톤과 분사별 자세 유사도 ───────────────────
  const ARM_UNAVAILABLE = {
    process_not_supported: '블라스팅 기록은 자세를 채점하지 않습니다',
    no_joint_sequences: '채점할 분사 구간이 없습니다',
  };
  const sequenceAt = (seconds) => {
    const started = (record?.arm_pose_evaluation?.sequences || []).filter((item) => Number.isFinite(Number(item.start_seconds)) && item.start_seconds <= seconds);
    return started.length ? started.reduce((late, item) => (item.start_seconds >= late.start_seconds ? item : late)) : null;
  };
  // 분사별 자세 유사도 그림(점과 선). 자세 칸과 채팅 참고 카드가 같이 쓴다. 칸의 지금 크기로 다시 그린다.
  // host.poseSpec = { sequences, average, currentId, onPick }. onPick 이 있으면 점마다 누르는 단추를 얹는다.
  const SVG_NS = 'http://www.w3.org/2000/svg';
  const svgNode = (tag, attrs, text) => {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
    if (text != null) node.textContent = text;
    return node;
  };
  const sequenceLabel = (item) => `${item.sequence_id} · ${zoneText(item.zone_id)} · ${percent(item.similarity_percent)}`
    + (item.correction ? `. ${item.correction}` : '') + (item.confidence === 'limited' ? '. 입력 품질 제한으로 참고용' : '');
  function drawPoseChart(host) {
    const spec = host.poseSpec;
    const width = host.clientWidth, height = host.clientHeight;
    if (!spec || !width || !height) return;
    const { sequences, average, currentId, onPick } = spec;
    const box = { left: 30, right: width - 64, top: 20, bottom: height - 11 };
    const points = L.posePoints(sequences.map((item) => Number(item.similarity_percent)), box);
    const chart = svgNode('svg', { class: 'pose-svg', viewBox: `0 0 ${width} ${height}`, 'aria-hidden': 'true' });
    for (const value of [50, 75, 100]) {
      const y = L.poseY(value, box);
      chart.append(svgNode('line', { class: 'pc-grid', x1: box.left, x2: box.right, y1: y, y2: y }),
        svgNode('text', { class: 'pc-tick', x: box.left - 6, y: y + 4, 'text-anchor': 'end' }, String(value)));
    }
    if (Number.isFinite(Number(average))) {
      const y = L.poseY(average, box);
      chart.append(svgNode('line', { class: 'pc-avg', x1: box.left, x2: box.right, y1: y, y2: y }),
        svgNode('text', { class: 'pc-avg-label', x: box.right + 8, y: y + 4 }, `평균 ${Math.round(Number(average))}%`));
    }
    chart.append(svgNode('polyline', { class: 'pc-line', points: points.map((point) => `${point.x},${point.y}`).join(' ') }));
    // Zone 색은 바닥의 3px 띠로만
    const slot = points.length > 1 ? (box.right - box.left) / (points.length - 1) : 12;
    sequences.forEach((item, index) => chart.append(svgNode('rect', { x: points[index].x - slot / 2 + 1, y: height - 3, width: Math.max(2, slot - 2), height: 3, rx: 1, fill: colorOf(item.zone_id) })));
    const current = sequences.findIndex((item) => item.sequence_id === currentId);
    sequences.forEach((item, index) => {
      if (index === current) return;
      const cls = `pc-dot${item.correction ? ' is-fix' : ''}${item.confidence === 'limited' ? ' is-limited' : ''}`;
      chart.append(svgNode('circle', { class: cls, cx: points[index].x, cy: points[index].y, r: item.confidence === 'limited' ? 3.5 : 4 }));
    });
    if (current >= 0) {
      const item = sequences[current], point = points[current];
      chart.append(svgNode('line', { class: 'pc-now-line', x1: point.x, x2: point.x, y1: point.y + 7, y2: height - 6 }),
        svgNode('circle', { class: `pc-now${item.confidence === 'limited' ? ' is-limited' : ''}`, cx: point.x, cy: point.y, r: 6 }),
        svgNode('text', { class: 'pc-now-label', x: point.x, y: point.y - 11, 'text-anchor': 'middle' }, percent(item.similarity_percent)));
    }
    const focused = host.contains(document.activeElement) ? [...host.querySelectorAll('.pose-pt')].indexOf(document.activeElement) : -1;
    host.replaceChildren(chart);
    if (!onPick) return;
    // 점마다 누르는 단추. 키보드는 한 칸만 Tab 으로 들어오고 화살표로 옮긴다.
    const stops = sequences.map((item, index) => {
      const button = el('button', 'pose-pt');
      button.type = 'button';
      button.style.left = `${points[index].x}px`;
      button.style.top = `${points[index].y}px`;
      button.title = sequenceLabel(item);
      button.setAttribute('aria-label', button.title);
      if (index === current) button.setAttribute('aria-current', 'true');
      button.tabIndex = index === Math.max(0, current) ? 0 : -1;
      button.addEventListener('click', () => onPick(item));
      button.addEventListener('keydown', (event) => {
        const next = { ArrowRight: index + 1, ArrowLeft: index - 1, Home: 0, End: sequences.length - 1 }[event.key];
        if (next == null) return;
        event.preventDefault();
        stops[Math.max(0, Math.min(sequences.length - 1, next))].focus();
      });
      return button;
    });
    host.append(...stops);
    if (focused >= 0) stops[focused]?.focus();
  }
  const chartSizes = new ResizeObserver((entries) => { for (const { target } of entries) drawPoseChart(target); });
  chartSizes.observe($('v2PoseChart'));

  function renderPose() {
    const arm = record?.arm_pose_evaluation;
    const host = $('v2PoseChart');
    if (arm?.status !== 'available') {
      const message = !record ? '' : arm === undefined ? '분사마다 채점하는 중' : ARM_UNAVAILABLE[arm.reason_code] || '자세 채점 결과를 불러오지 못했습니다';
      host.poseSpec = null;
      host.replaceChildren(el('p', 'chart-empty', message));
      $('v2PoseNow').textContent = '';
      return;
    }
    host.poseSpec = { sequences: arm.sequences || [], average: arm.average_similarity_percent, currentId: null,
      onPick: (item) => seekToSeconds(Number(item.start_seconds) + 0.01) };
    renderPoseNow(true);
  }
  function renderPoseNow(force = false) {
    const host = $('v2PoseChart');
    if (!host.poseSpec) return;
    const current = sequenceAt(anchor.playhead_seconds);
    if (force || host.poseSpec.currentId !== (current?.sequence_id ?? null)) {
      host.poseSpec.currentId = current?.sequence_id ?? null;
      drawPoseChart(host);
    }
    const now = $('v2PoseNow');
    if (!current) { now.textContent = '첫 분사 전입니다'; return; }
    const limited = current.confidence === 'limited';
    now.replaceChildren(el('b', null, current.sequence_id), ` · ${zoneText(current.zone_id)} · ${percent(current.similarity_percent)}`
      + (limited ? ' · 참고용(입력 품질 제한)' : '') + ` · ${current.correction || (limited ? '' : '교정할 관절이 없습니다')}`.replace(/ · $/, ''));
    now.title = now.textContent;
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

  // ── 순서: 위 = 전문가 권장, 아래 = 이 기록. 같은 자리의 Zone은 흐리게 두고, 자리가 다른 Zone만 그 색 곡선으로 잇는다 ──
  function renderOrder() {
    const order = record?.reference_order_evaluation || {};
    const host = $('v2Order');
    const sum = $('v2OrderSum');
    const expert = (order.recommended_flow || []).map(String);
    const actual = (order.selected_flow || []).map(String);
    if (!expert.length) { sum.replaceChildren(); host.replaceChildren(el('p', 'chart-empty', record ? '이 학습모드에는 전문가 순서 기준이 없습니다' : '')); return; }
    const moves = L.orderMoves(expert, actual);
    const count = (text, number, tail) => [text, el('b', null, String(number)), tail];
    const line = el('span');   // 머리 줄은 flex 라 글 조각을 한 칸에 묶는다(사이 간격이 끼지 않게)
    line.append(...count('권장 순서와 자리가 다른 Zone ', moves.moved, '곳'), ...(moves.missing ? count(' · 칠하지 않은 Zone ', moves.missing, '곳') : []));
    sum.replaceChildren(line);
    const row = (label, items) => {
      const line = el('div', 'order-row');
      const chips = el('div', 'order-chips');
      for (const item of items) {
        const chip = el('button', `order-chip is-${item.state}`, String(Number(item.zone)));
        chip.type = 'button';
        chip.dataset.zone = item.zone;
        chip.dataset.state = item.state;
        if (item.actualIndex != null) chip.dataset.to = String(item.actualIndex);
        chip.style.setProperty('--zone', colorOf(item.zone));
        const skipped = item.state === 'skipped';
        chip.title = skipped ? `${zoneText(item.zone)} · 칠하지 않음` : `${zoneText(item.zone)}로 이동`;
        chip.setAttribute('aria-label', chip.title);
        chip.disabled = skipped;
        chip.addEventListener('click', () => seekToZone(item.zone));
        chips.append(chip);
      }
      line.append(el('span', null, label), chips);
      return line;
    };
    const links = document.createElementNS(SVG_NS, 'svg');
    links.setAttribute('class', 'order-links');
    links.setAttribute('aria-hidden', 'true');
    // 이 기록 줄 끝에는 칠하지 않은 Zone을 점선 칩으로 덧붙인다
    const skipped = moves.expert.filter((item) => item.state === 'missing').map((item) => ({ zone: item.zone, state: 'skipped' }));
    host.replaceChildren(row('권장', moves.expert), links, row('이 기록', [...moves.actual, ...skipped]));
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
    const y1 = 2, y2 = box.height - 2, middle = box.height / 2;
    for (const chip of top.querySelectorAll('[data-state="moved"]')) {
      const target = bottom.children[Number(chip.dataset.to)];
      if (!target) continue;
      const x1 = center(chip), x2 = center(target);
      links.append(svgNode('path', { d: `M${x1},${y1} C${x1},${middle} ${x2},${middle} ${x2},${y2}`, stroke: colorOf(chip.dataset.zone) }));
    }
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
    $('v2CoatBarsView').setAttribute('aria-pressed', String(!coatMap));
    $('v2CoatMapView').setAttribute('aria-pressed', String(coatMap));
    coatLayout();
    $('v2CoatBars').setAttribute('aria-label', expert ? 'Zone별 도막 두께가 전문가 기준과 얼마나 다른지' : 'Zone별 도막 등급(목표 대비)');
    $('v2CoatBasisLabel').textContent = !expert && targetReady() ? targetOf().basis_label || '' : '';
    if (expert) renderExpertCoating(); else renderTargetCoating();
  }

  /** 전문가 대비를 한 칸 보기로 볼 때는 편차(왼쪽)와 전문가 지도(오른쪽)를 함께 둔다. 세 칸 보기는 편차·지도 전환. */
  function coatLayout() {
    const expert = coatBasis === 'expert';
    const side = expert && layout.count === 1;
    $('v2FeatureCoating').classList.toggle('is-side', side);
    $('v2CoatViews').hidden = !expert || side;
    $('v2CoatBars').hidden = expert && coatMap && !side;
    $('v2CoatMap').hidden = !expert || !(coatMap || side);
    $('v2CoatMapSeg').hidden = $('v2CoatMap').hidden;
  }

  // 전문가 지도 범례: 가는 색 띠 + 최소·중앙·최대(µm 보정이 있을 때). 지도는 기존 도막 스크립트가 그리고, 범례를 다시 쓸 때 따라 그린다.
  function renderMapLegend() {
    const host = $('v2MapLegend');
    const number = Number(($('coatingZone').textContent.match(/(\d+)/) || [])[1]);
    const references = record?.coating_reference || {};
    const key = Number(anchor.zone_id) === number && anchor.map_num != null && references[`${anchor.map_num}:${anchor.zone_id}`]
      ? `${anchor.map_num}:${anchor.zone_id}` : Object.keys(references).find((item) => Number(item.split(':')[1]) === number);
    const reference = key ? references[key] : null;
    const absolute = $('coatingScaleAbsolute').getAttribute('aria-pressed') === 'true';
    let stops = reference?.physical_calibration === 'available' ? L.legendStops(reference.thickness_um, absolute ? reference.color_scale_um : null) : [];
    const ticks = [...$('coatingLegend').children].map((tick) => tick.textContent);
    if (!stops.length && ticks.length > 1) stops = [{ label: ticks[0], at: 0 }, { label: ticks[ticks.length - 1], at: 1 }];
    const bar = el('i');
    const labels = el('div', 'map-legend-labels');
    labels.append(...stops.map((stop) => {
      const label = el('span', stop.label.startsWith('중앙') ? 'is-mid' : null, stop.label);
      label.style.left = `${stop.at * 100}%`;
      label.style.transform = `translateX(-${stop.at * 100}%)`;   // 양 끝은 띠 안쪽으로 붙인다
      return label;
    }));
    const note = el('p', null, `${absolute ? '모든 Zone 공통 µm' : 'Zone 안 상대 색'} · 전문가 최종 도막(훈련자 진행 도막 아님)`);
    host.title = `${note.textContent}. ${$('coatingStatus').textContent}`;
    host.replaceChildren(bar, labels, note);
  }
  new MutationObserver(renderMapLegend).observe($('coatingLegend'), { childList: true, characterData: true, subtree: true });

  // 목표 대비(C1): Zone마다 한 줄, 문제 Zone 먼저. Zone | 칠한 면적 6구간 100% 막대 | 칠한 곳 평균 µm + 서버 등급 글자.
  // 등급·구간 이름·경계는 서버 target 값을 그대로 쓴다. 범례는 머리 한 번.
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
    const normal = target.normal_band_um || [];
    const edges = target.band_edges_um || [];
    const bandLabels = target.band_labels || [];
    host.classList.remove('has-unpainted');
    // 머리 줄: 6구간 범례(경계 µm는 title)
    const legend = el('span', 'band-legend');
    legend.append(el('span', 't', '칠한 면적'));
    bandLabels.forEach((label, index) => {
      const item = el('span');
      item.append(el('i', `band-${index + 1}`), label);
      if (edges.length === bandLabels.length + 1) item.title = `${edges[index]}~${edges[index + 1]} µm`;
      legend.append(item);
    });
    count.replaceChildren(legend);
    const showUnpainted = (row) => Number(row.unpainted_fraction) >= 0.005;   // 표시 문턱일 뿐 판정이 아니다
    host.replaceChildren(...L.coatOrder(target.zones || []).map((row) => {
      const grade = row.grade ?? 'none';
      const button = el('button', 'coat-row is-c1');
      button.type = 'button';
      button.dataset.zone = String(row.zone_id);
      const name = el('span', 'coat-zone', String(Number(row.zone_id)));
      name.style.setProperty('--zone', colorOf(row.zone_id));
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
      const value = el('span', 'coat-value');
      value.append(el('b', null, row.thickness_um == null ? '—' : String(Math.round(Number(row.thickness_um)))), ' µm ', el('em', `g-${grade}`, labels[grade] ?? ''));
      if (showUnpainted(row)) value.append(el('small', null, `미도장 ${shareText(Number(row.unpainted_fraction))}`));
      button.append(name, strip, value);
      const summary = [zoneText(row.zone_id), labels[grade] ?? grade,
        `칠한 곳 평균 ${row.thickness_um == null ? '—' : `${Math.round(Number(row.thickness_um))} µm`}`,
        `미도장 ${shareText(Number(row.unpainted_fraction) || 0)}`, normal.length === 2 ? `정상 ${normal[0]}~${normal[1]} µm` : '', target.basis_label].filter(Boolean).join(' · ');
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
  let coatScrolledByUser = 0, coatAutoTop = null, coatCatchUp = 0;
  let alertedZone = null;   // 누른 알림의 Zone(현재 Zone 과 다를 수 있다). 다시 그려도 남도록 여기서 칠한다.
  for (const type of ['wheel', 'touchstart', 'pointerdown', 'keydown']) $('v2CoatBars').addEventListener(type, () => { coatScrolledByUser = performance.now(); }, { passive: true });
  $('v2CoatBars').addEventListener('scroll', (event) => {   // 스크롤바 끌기 등 우리가 한 것이 아닌 스크롤
    if (event.target.scrollTop !== coatAutoTop) coatScrolledByUser = performance.now();
  }, { passive: true });
  // 도막 목록이 칸보다 길면 아래를 흐리게 하고 '아래로 N개 더'를 붙인다(끝까지 내리면 걷는다).
  const coatBars = $('v2CoatBars');
  const hintMore = () => {
    let more = coatBars.querySelector(':scope > .more');
    if (!more) { more = el('div', 'more'); more.setAttribute('aria-hidden', 'true'); coatBars.append(more); return; }   // 붙이면 다시 불린다
    const bottom = coatBars.getBoundingClientRect().bottom;
    const below = [...coatBars.querySelectorAll(':scope > .coat-row')].filter((row) => row.getBoundingClientRect().top >= bottom - 12).length;
    coatBars.classList.toggle('has-more', below > 0);
    more.hidden = !below;
    if (below && more.dataset.count !== String(below)) { more.dataset.count = String(below); more.innerHTML = svg('chevron-down'); more.append(`아래로 ${below}개 더`); }
  };
  coatBars.addEventListener('scroll', hintMore, { passive: true });
  new ResizeObserver(hintMore).observe(coatBars);
  new MutationObserver(hintMore).observe(coatBars, { childList: true });
  function markCurrentZone(follow = false) {
    for (const node of document.querySelectorAll('.order-chip, .coat-row')) {
      node.classList.toggle('is-current', anchor.zone_id != null && node.dataset.zone === String(anchor.zone_id) && !node.classList.contains('is-skipped'));
      node.classList.toggle('is-alerted', alertedZone != null && node.dataset.zone === alertedZone);
    }
    // 알림으로 옮겼으면 알림 Zone 행을 보인다(건너뛴 Zone 은 현재 Zone 과 다르다, 카나리 V1′).
    const row = document.querySelector('.coat-row.is-alerted') || document.querySelector('.coat-row.is-current');
    const box = row?.parentElement;
    if (!follow || !box || box.scrollHeight <= box.clientHeight) return;
    const wait = 4000 - (performance.now() - coatScrolledByUser);
    if (wait > 0) {   // 사용자 스크롤이 끝나면 한 번 따라잡는다
      clearTimeout(coatCatchUp);
      coatCatchUp = setTimeout(() => markCurrentZone(true), wait + 50);
      return;
    }
    const inner = row.getBoundingClientRect();
    const outer = box.getBoundingClientRect();
    if (inner.top < outer.top) box.scrollTop -= outer.top - inner.top;
    else if (inner.bottom > outer.bottom) box.scrollTop += inner.bottom - outer.bottom;
    coatAutoTop = box.scrollTop;
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
    renderScores(); renderHudFix(); renderPose(); renderOrder(); renderCoating();
  }

  // ── 화면 나누기: 경계선 손잡이로 뷰어·채팅을 접고 폭을 바꾼다. 낮거나 좁은 뷰어는 분석 칸을 탭으로 하나만 보인다 ──
  const stored = (key) => { try { return localStorage.getItem(`motionViewerV2.${key}`); } catch { return null; } };
  const remember = (key, value) => { try { localStorage.setItem(`motionViewerV2.${key}`, value); } catch { /* 저장소가 막혀도 화면은 그대로 쓴다 */ } };
  const layout = (() => {
    const grid = $('v2Grid');
    const main = document.querySelector('.v2-main');
    const features = $('v2Features');
    const tabs = $('v2FeatureTabs');
    const tools = $('v2FeatureTools');
    const fold = $('v2Fold');
    const tabOf = (id) => tabs.querySelector(`[data-feature-tab="${id}"]`);
    let mode = ['both', 'chat-off', 'viewer-off'].includes(stored('mode')) ? stored('mode') : 'both';
    const narrow = matchMedia('(max-width: 900px)');
    const effective = () => (narrow.matches ? 'both' : mode);
    let chatWidth = Number(stored('chatWidth'));
    let count = 3;
    let panel = { active: 'pose', folded: stored('fold') === '1' };
    // 칸마다 머리줄의 도구(기준 전환·돌려 보기·질문 단추). 한 칸 보기에서는 탭 줄 오른쪽으로 옮긴다.
    const toolsOf = Object.fromEntries([...features.querySelectorAll('.feature')].map((feature) => {
      const header = feature.querySelector('header');
      return [feature.dataset.feature, { header, nodes: [...header.children].filter((node) => node.tagName !== 'H2') }];
    }));
    const applyWidth = () => grid.style.setProperty('--chat-w', `${L.clampChatWidth(chatWidth || Number.NaN, grid.clientWidth)}px`);
    function applyFeatures() {
      count = L.panelCount(main.clientWidth, innerHeight);
      const one = count === 1;
      tabs.hidden = !one;
      features.dataset.count = String(count);
      features.hidden = panel.folded;
      for (const feature of features.querySelectorAll('.feature')) {
        const id = feature.dataset.feature;
        feature.classList.toggle('is-off', one && id !== panel.active);
        const tab = tabOf(id);
        tab.setAttribute('aria-selected', String(one && !panel.folded && id === panel.active));
        tab.tabIndex = id === panel.active ? 0 : -1;
        const home = toolsOf[id];
        if (one && !panel.folded && id === panel.active) tools.append(...home.nodes); else home.header.append(...home.nodes);
      }
      // 전문가 지도의 2D·3D·상대·µm 도 한 칸 보기에서는 탭 줄 도구 칸(질문 단추 앞)으로
      const mapSeg = $('v2CoatMapSeg');
      const coatAsk = toolsOf.coating.nodes[toolsOf.coating.nodes.length - 1];
      if (one && coatAsk.parentElement === tools) coatAsk.before(mapSeg); else if (mapSeg.parentElement !== $('v2CoatMapTools')) $('v2CoatMapTools').append(mapSeg);
      coatLayout();
      // 접기 단추: 한 칸 보기는 탭 줄 끝, 세 칸 보기는 재생 막대 끝
      const box = $('v2FoldBox');
      const place = one ? tabs : document.querySelector('.transport');
      if (box.parentElement !== place) place.append(box);
      fold.setAttribute('aria-expanded', String(!panel.folded));
      fold.title = panel.folded ? '분석 칸 펴기' : '분석 칸 접기';
      fold.setAttribute('aria-label', fold.title);
    }
    function setPanel(next) {
      const changed = next.folded !== panel.folded || next.active !== panel.active;
      panel = next;
      remember('fold', panel.folded ? '1' : '0');
      applyFeatures();
      if (changed) dispatchEvent(new Event('resize'));   // 다시 보인 칸(순서 선·도막 지도)이 새 폭으로 그려지게 한다
    }
    /** 질문 탭을 바꾸면 한 칸 보기의 그 칸을 앞에 둔다. 사용자가 접어 둔 것은 그대로 둔다. */
    function showFeature(id) {
      if (!L.FEATURES.includes(id) || id === panel.active) return;
      setPanel({ ...panel, active: id });
    }
    function setMode(next, { focus = false } = {}) {
      const before = effective();
      mode = next;
      const now = effective();
      grid.classList.toggle('is-chat-off', now === 'chat-off');
      grid.classList.toggle('is-viewer-off', now === 'viewer-off');
      $('v2RailChat').hidden = now !== 'chat-off';
      $('v2RailViewer').hidden = now !== 'viewer-off';
      $('v2Split').hidden = now !== 'both';
      if (now !== 'viewer-off') feedback.closePanel();
      // 뷰어를 접으면 재생을 멈춘다. 알림·재생 위치가 보이지 않는 채로 흘러가지 않게
      if (now === 'viewer-off' && before !== 'viewer-off' && $('play').textContent.trim() !== '재생') $('v2Play').click();
      remember('mode', mode);
      applyWidth();
      requestAnimationFrame(() => { dispatchEvent(new Event('resize')); fillStage(); });
      if (focus) ({ both: $('v2Resize'), 'chat-off': $('v2OpenChat'), 'viewer-off': $('v2OpenViewer') })[now]?.focus();
    }
    $('v2CollapseViewer').addEventListener('click', () => setMode('viewer-off', { focus: true }));
    $('v2CollapseChat').addEventListener('click', () => setMode('chat-off', { focus: true }));
    $('v2OpenViewer').addEventListener('click', () => setMode('both', { focus: true }));
    $('v2OpenChat').addEventListener('click', () => setMode('both', { focus: true }));
    $('v2OpenChatIcon').addEventListener('click', () => setMode('both', { focus: true }));
    // 끌어서 폭 조절. 손잡이는 키보드로도 움직인다(왼쪽 = 채팅 넓게).
    const grip = $('v2Resize');
    grip.addEventListener('pointerdown', (event) => {
      grip.setPointerCapture(event.pointerId);
      grid.classList.add('is-resizing');
    });
    grip.addEventListener('pointermove', (event) => {
      if (!grip.hasPointerCapture(event.pointerId)) return;
      chatWidth = L.clampChatWidth(grid.getBoundingClientRect().right - event.clientX, grid.clientWidth);
      applyWidth();
    });
    const endDrag = (event) => {
      if (!grip.hasPointerCapture(event.pointerId)) return;
      grip.releasePointerCapture(event.pointerId);
      grid.classList.remove('is-resizing');
      remember('chatWidth', String(chatWidth));
      dispatchEvent(new Event('resize'));
    };
    grip.addEventListener('pointerup', endDrag);
    grip.addEventListener('pointercancel', endDrag);
    grip.addEventListener('keydown', (event) => {
      const step = { ArrowLeft: 32, ArrowRight: -32 }[event.key];
      if (!step && event.key !== 'Home') return;
      event.preventDefault();
      const now = L.clampChatWidth(chatWidth || Number.NaN, grid.clientWidth);
      chatWidth = event.key === 'Home' ? Number.NaN : L.clampChatWidth(now + step, grid.clientWidth);
      applyWidth();
      remember('chatWidth', Number.isFinite(chatWidth) ? String(chatWidth) : '');
      dispatchEvent(new Event('resize'));
    });
    for (const tab of tabs.querySelectorAll('[role="tab"]')) {
      tab.addEventListener('click', () => setPanel(L.tabPress(panel, tab.dataset.featureTab)));
      tab.addEventListener('keydown', (event) => {
        const move = { ArrowRight: 1, ArrowLeft: -1 }[event.key];
        if (!move) return;
        event.preventDefault();
        const ids = L.FEATURES;
        const next = ids[(ids.indexOf(tab.dataset.featureTab) + move + ids.length) % ids.length];
        if (!panel.folded) setPanel({ ...panel, active: next });
        tabOf(next).focus();
      });
    }
    fold.addEventListener('click', () => setPanel({ ...panel, folded: !panel.folded }));
    new ResizeObserver(applyFeatures).observe(main);
    addEventListener('resize', () => { applyWidth(); applyFeatures(); });
    narrow.addEventListener('change', () => setMode(mode));
    return {
      setMode, showFeature, applyFeatures,
      get mode() { return effective(); },
      get count() { return count; },
      start() { setMode(mode); applyFeatures(); },
    };
  })();

  // ── 참고 카드(B안): 채팅만 볼 때 답 아래에 질문한 순간의 기록 값으로 그린 카드를 둔다. 화면을 찍지 않는다 ──
  /** 질문한 순간의 값을 떠 둔다. 자세 = 분사 그림 + Zone 수치, 도막 = Zone 수치, 순서·지식 = 없음. */
  function captureAttachments(tabId, state) {
    const plan = L.attachmentsFor(tabId);
    if (!plan.length) return null;
    const card = { zoneId: state.zone_id, chart: null, metrics: [] };
    if (plan.includes('poseChart')) {
      const arm = record?.arm_pose_evaluation;
      const sequences = (arm?.status === 'available' ? arm.sequences || [] : []).filter((item) => Number.isFinite(Number(item.similarity_percent)))
        .map(({ sequence_id, zone_id, similarity_percent, confidence, correction }) => ({ sequence_id, zone_id, similarity_percent, confidence, correction }));
      const current = sequenceAt(state.playhead_seconds);
      if (sequences.length) card.chart = { sequences, average: arm.average_similarity_percent, currentId: current?.sequence_id ?? null, current };
    }
    if (plan.includes('zoneMetrics')) card.metrics = (zoneAt(state)?.metrics || []).filter(usableMetric).map((metric) => ({ ...metric }));
    return card.chart || card.metrics.length ? card : null;
  }
  function renderAttachments(card) {
    const box = el('div', 'refcard');
    const head = el('div', 'refcard-head');
    if (card.zoneId != null) { const tag = el('span', 'zone-tag'); tagZone(tag, card.zoneId); head.append(tag); }
    if (card.chart?.current) {
      const value = el('b', null, percent(card.chart.current.similarity_percent));
      head.append(`${card.chart.current.sequence_id} 자세 유사도 `, value);
    }
    head.append(el('span', 'sub', '질문한 재생 위치 기준'));
    box.append(head);
    if (card.chart) {
      const plot = el('div', 'refcard-chart');
      plot.setAttribute('role', 'img');
      plot.setAttribute('aria-label', `분사별 자세 유사도 ${card.chart.sequences.length}개, 평균 ${percent(card.chart.average)}`);
      plot.poseSpec = card.chart;
      chartSizes.observe(plot);   // 카드가 보이는 순간(채팅만 보기) 그 폭으로 그린다
      const legend = el('div', 'refcard-legend');
      legend.innerHTML = '<span><i class="mk is-fix"></i>교정 있음</span><span><i class="mk is-limited"></i>참고용</span><span><i class="mk is-now"></i>질문한 분사</span>';
      box.append(plot, legend);
    }
    if (card.metrics.length) {
      const table = el('table', 'refcard-table');
      table.innerHTML = '<thead><tr><th scope="col"></th><th scope="col">이 기록</th><th scope="col">전문가</th><th scope="col">유사도</th></tr></thead>';
      if (card.zoneId != null) table.querySelector('th').textContent = zoneText(card.zoneId);
      const body = el('tbody');
      for (const metric of card.metrics) {
        const row = el('tr');
        const name = el('th', null, metric.label);
        name.scope = 'row';
        row.append(name, el('td', null, metricShow(metric, metric.actual, metric.actual_label)),
          el('td', 'ref', metricShow(metric, metric.reference, metric.reference_label)), el('td', null, percent(metric.similarity_percent)));
        body.append(row);
      }
      table.append(body);
      box.append(table);
    }
    return box;
  }

  // ── 답변 피드백: 뷰어가 열려 있으면 답 안의 작은 칸, 채팅만 볼 때는 오른쪽 피드백 칸 ──
  const uuid = () => {
    if (globalThis.crypto?.randomUUID) return crypto.randomUUID();
    const bytes = new Uint8Array(16);
    crypto.getRandomValues(bytes);
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    const hex = [...bytes].map((b) => b.toString(16).padStart(2, '0')).join('');
    return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
  };
  const feedback = (() => {
    const panel = $('v2Feedback');
    const form = $('v2FeedbackForm');
    const text = $('v2FeedbackText');
    const correct = $('v2FeedbackCorrect');
    const error = $('v2FeedbackError');
    const sendButton = $('v2FeedbackSend');
    const LABEL = { wrong: '무엇이 틀렸나요', memo: '메모' };
    let target = null;   // { node, info, kind, submissionId, epoch }
    let inline = null;
    let sentCount = 0;
    let epoch = 0;       // 새 대화마다 올린다. 보내는 중에 대화가 바뀌면 결과를 버린다

    const markTarget = (node) => {
      for (const other of document.querySelectorAll('.msg-a.is-target')) other.classList.remove('is-target');
      node?.classList.add('is-target');
    };
    const pressActs = (node, kind) => {
      for (const button of document.querySelectorAll('.fb-acts button')) {
        button.setAttribute('aria-pressed', String(button.closest('.msg-a') === node && button.dataset.kind === kind));
      }
    };
    function setKind(kind) {
      target.kind = kind;
      for (const button of form.querySelectorAll('.fb-kind [role="radio"]')) {
        const on = button.dataset.kind === kind;
        button.setAttribute('aria-checked', String(on));
        button.setAttribute('aria-pressed', String(on));
        button.tabIndex = on ? 0 : -1;
      }
      $('v2FeedbackLabel').textContent = LABEL[kind];
      pressActs(target.node, kind);
    }
    function closeInline() {
      inline?.remove();
      inline = null;
    }
    function closePanel() {
      panel.hidden = true;
      $('v2Grid').classList.remove('has-feedback');
      if (!inline) { markTarget(null); pressActs(null); }
    }
    function openPanel(node, info, kind, carried = '') {
      closeInline();
      target = { node, info, kind, submissionId: uuid(), epoch };
      panel.hidden = false;
      $('v2Grid').classList.add('has-feedback');
      $('v2FeedbackEmpty').hidden = true;
      form.hidden = false;
      error.hidden = true;
      tagZone($('v2FeedbackZone'), info.zone_id);
      $('v2FeedbackQuestion').textContent = info.question;
      $('v2FeedbackQuestion').title = info.question;
      $('v2FeedbackTime').textContent = Number.isFinite(info.playhead_seconds) ? L.clock(info.playhead_seconds) : '';
      text.value = carried;
      correct.value = '';
      $('v2FeedbackMore').open = false;
      setKind(kind);
      markTarget(node);
      text.focus();
    }
    function openInline(node, info, kind) {
      closeInline();
      target = { node, info, kind, submissionId: uuid(), epoch };
      markTarget(node);
      pressActs(node, kind);
      const box = el('form', 'fb-inline');
      const id = `fbInline${Date.now().toString(36)}`;
      const label = el('label', null, LABEL[kind]);
      label.htmlFor = id;
      const area = el('textarea');
      area.id = id;
      area.rows = 3;
      area.maxLength = 4000;
      const message = el('p', 'fb-error');
      message.setAttribute('role', 'alert');
      message.hidden = true;
      const row = el('div', 'fb-row');
      const widen = iconButton('icon-btn', 'maximize-2', '넓게 쓰기');
      widen.title = '넓게 쓰기';
      const spacer = el('span', 'grow');
      const cancel = el('button', 'btn', '취소');
      cancel.type = 'button';
      const submit = el('button', 'btn primary');
      submit.type = 'submit';
      submit.innerHTML = svg('send');
      submit.append('보내기');
      row.append(widen, spacer, cancel, submit);
      box.append(label, area, message, row);
      widen.addEventListener('click', () => {
        const carried = area.value;
        layout.setMode('viewer-off');
        openPanel(node, info, target.kind, carried);
      });
      cancel.addEventListener('click', () => { closeInline(); markTarget(null); pressActs(null); });
      box.addEventListener('submit', (event) => { event.preventDefault(); submit.disabled = true; deliver(area.value, '', message).finally(() => { submit.disabled = false; }); });
      node.append(box);
      inline = box;
      area.focus();
    }
    function open(node, info, kind) {
      if (layout.mode === 'viewer-off') openPanel(node, info, kind);
      else openInline(node, info, kind);
    }
    async function deliver(message, correctText, errorNode) {
      const current = target;
      if (!current) return;
      const body = L.feedbackPayload({ submissionId: current.submissionId, kind: current.kind, text: message, correct: correctText, answer: current.info });
      errorNode.hidden = true;
      if (!body) { errorNode.textContent = '메모를 적어 주세요.'; errorNode.hidden = false; return; }
      let response = null;
      let payload = null;
      try {
        response = await fetch('/api/feedback', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
        try { payload = await response.json(); } catch { /* 본문 없음 */ }
      } catch {
        errorNode.textContent = '보내지 못했습니다. 네트워크를 확인하고 다시 보내 주세요.';
        errorNode.hidden = false;
        return;
      }
      if (!response.ok) {
        errorNode.textContent = payload?.detail && typeof payload.detail === 'string' ? payload.detail : `보내지 못했습니다(HTTP ${response.status}).`;
        errorNode.hidden = false;
        return;
      }
      sent(current);
    }
    function sent(done) {
      if (done.epoch !== epoch) return;
      const acts = done.node.querySelector('.fb-acts');
      const mark = el('span', 'fb-done');
      mark.innerHTML = svg('check');
      mark.append('보냄');
      acts?.replaceWith(mark);
      const item = el('li');
      item.innerHTML = svg(done.kind === 'wrong' ? 'thumbs-down' : 'message-square-text');
      const question = el('span', 'q', done.info.question);
      question.title = done.info.question;
      item.append(question, el('span', 'st', '접수'));
      $('v2FeedbackSentList').prepend(item);
      sentCount += 1;
      $('v2FeedbackSentCount').textContent = String(sentCount);
      $('v2FeedbackSent').hidden = false;
      if (target !== done) return;   // 보내는 사이 다른 답을 골랐으면 그 칸은 그대로 둔다
      closeInline();
      markTarget(null);
      form.hidden = true;
      $('v2FeedbackEmpty').hidden = false;
      target = null;
    }
    form.addEventListener('submit', (event) => {
      event.preventDefault();
      sendButton.disabled = true;
      deliver(text.value, correct.value, error).finally(() => { sendButton.disabled = false; });
    });
    for (const button of form.querySelectorAll('.fb-kind [role="radio"]')) {
      button.addEventListener('click', () => setKind(button.dataset.kind));
      button.addEventListener('keydown', (event) => {
        if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
        event.preventDefault();
        const other = form.querySelector(`.fb-kind [data-kind="${button.dataset.kind === 'wrong' ? 'memo' : 'wrong'}"]`);
        setKind(other.dataset.kind);
        other.focus();
      });
    }
    $('v2FeedbackClose').addEventListener('click', closePanel);
    return {
      closePanel,
      /** 성공한 답 아래에 잘못됨·메모 단추를 단다. */
      attach(node, row, info) {
        const acts = el('span', 'fb-acts');
        for (const [kind, icon, label] of [['wrong', 'thumbs-down', '잘못된 답으로 알리기'], ['memo', 'message-square-text', '이 답에 메모 남기기']]) {
          const button = iconButton('fb-act', icon, label);
          button.title = kind === 'wrong' ? '잘못됨' : '메모';
          button.dataset.kind = kind;
          button.setAttribute('aria-pressed', 'false');
          button.addEventListener('click', () => open(node, info, kind));
          acts.append(button);
        }
        row.append(acts);
      },
      reset() {
        epoch += 1;
        closeInline();
        closePanel();
        form.hidden = true;
        $('v2FeedbackEmpty').hidden = false;
        target = null;
        sentCount = 0;
        $('v2FeedbackSentList').replaceChildren();
        $('v2FeedbackSentCount').textContent = '';
        $('v2FeedbackSent').hidden = true;
      },
    };
  })();

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
    const anchorZone = el('span', 'zone-tag zone-tag-idle');
    const anchorTime = el('span', 'time', '0:00');
    anchorBox.innerHTML = svg('map-pin');
    anchorBox.append(anchorZone, anchorTime);
    const fresh = iconButton('icon-btn', 'message-square-plus', '새 대화');
    head.append(anchorBox, fresh);

    const tabs = el('div', 'tabs segtabs');
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
    // 답변 언어: 단추 하나(국기 + 그 언어 이름)가 12개 언어 목록을 연다. 질문은 한국어로 보내고 답만 고른 언어로 받는다.
    let language = LANGUAGES[0];
    const languageBox = el('div', 'lang');
    const languageButton = el('button', 'lang-btn');
    languageButton.type = 'button';
    languageButton.setAttribute('aria-haspopup', 'listbox');
    languageButton.setAttribute('aria-expanded', 'false');
    const languageList = el('div', 'lang-list');
    languageList.id = 'v2LangList';
    languageList.setAttribute('role', 'listbox');
    languageList.setAttribute('aria-label', '답변 언어');
    languageList.hidden = true;
    languageButton.setAttribute('aria-controls', languageList.id);
    const languageOptions = LANGUAGES.map(([code, country, name, korean], index) => {
      const option = el('button', 'lang-option');
      option.type = 'button';
      option.setAttribute('role', 'option');
      option.lang = code;
      option.append(flagImage(country), el('span', null, name), el('span', 'ko', code === 'ko' ? '' : korean));
      option.insertAdjacentHTML('beforeend', svg('check'));
      option.addEventListener('click', () => { chooseLanguage(index); openLanguages(false); languageButton.focus(); });
      option.addEventListener('keydown', (event) => {
        const next = { ArrowDown: index + 1, ArrowUp: index - 1, Home: 0, End: LANGUAGES.length - 1 }[event.key];
        if (next == null) return;
        event.preventDefault();
        languageOptions[(next + LANGUAGES.length) % LANGUAGES.length].focus();
      });
      return option;
    });
    languageList.append(...languageOptions);
    const chooseLanguage = (index) => {
      language = LANGUAGES[index];
      languageButton.replaceChildren(flagImage(language[1]), el('span', null, language[2]));
      languageButton.insertAdjacentHTML('beforeend', svg('chevron-down'));
      languageButton.lang = language[0];
      languageButton.title = '답변 언어. 질문은 한국어로 보냅니다';
      languageButton.setAttribute('aria-label', `답변 언어: ${language[3]}`);
      languageOptions.forEach((option, i) => { option.setAttribute('aria-selected', String(i === index)); option.tabIndex = i === index ? 0 : -1; });
    };
    function openLanguages(open) {
      languageList.hidden = !open;
      languageButton.setAttribute('aria-expanded', String(open));
      if (open) languageOptions[LANGUAGES.indexOf(language)].focus();
    }
    languageButton.addEventListener('click', () => openLanguages(languageList.hidden));
    languageButton.addEventListener('keydown', (event) => { if (event.key === 'ArrowDown' || event.key === 'ArrowUp') { event.preventDefault(); openLanguages(true); } });
    languageBox.addEventListener('keydown', (event) => {
      if (event.key !== 'Escape' || languageList.hidden) return;
      event.stopPropagation();
      openLanguages(false);
      languageButton.focus();
    });
    languageBox.addEventListener('focusout', (event) => { if (!languageBox.contains(event.relatedTarget)) openLanguages(false); });
    document.addEventListener('click', (event) => { if (!languageList.hidden && !languageBox.contains(event.target)) openLanguages(false); });
    languageBox.append(languageButton, languageList);
    chooseLanguage(0);
    // 기록된 응답만 보여 주는 화면(링크용)에서는 준비된 질문만 쓴다.
    input.disabled = send.disabled = Boolean(options.presetOnly);
    tools.append(modes, languageBox);
    form.append(line, tools);
    root.append(head, tabs, log, presets, form);

    // 추천 질문은 빈 대화에서는 안내 아래에, 대화가 시작되면 입력 칸 위 한 줄(가로로 넘김)에 둔다.
    const empty = () => {
      if (log.children.length) return;
      const box = el('div', 'ask-empty');
      box.append(el('p', null, options.notice || '재생을 멈춘 곳을 기준으로 답합니다'), presets);
      log.append(box);
      presets.classList.remove('is-row');
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
      if (log.querySelector('.ask-empty')) {
        log.querySelector('.ask-empty').remove();
        presets.classList.add('is-row');
        form.before(presets);
      }
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
      layout.showFeature(tab.id);   // 칸이 접혀 있으면 질문하는 칸을 연다
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
      let refs = null;
      try { refs = sent.data ? captureAttachments(sent.tab.id, sent.state) : null; } catch { refs = null; }   // 첨부를 못 떠도 질문은 보낸다
      const asked = bubble('msg-q', prompt);
      const meta = el('div', 'msg-meta');
      const pin = el('button', 'chip');
      pin.type = 'button';
      pin.innerHTML = svg('map-pin');
      pin.append(`${sent.state.zone_id == null ? 'Zone 없음' : zoneText(sent.state.zone_id)} · ${L.clock(sent.state.playhead_seconds)}`);
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
        if (refs) answer.append(renderAttachments(refs));
        feedback.attach(answer, row, {
          conversation_id: conversation, question: prompt, answer: String(result.answer || ''), task: sent.tab.task, response_mode: sent.mode,
          answer_source: result.answer_source || null, zone_id: sent.state.zone_id, map_num: sent.state.map_num,
          playhead_seconds: sent.state.playhead_seconds, record: sent.data ? recordLabel(sent.data) : null,
          session_id: sent.data?.selected_session_id ?? null, worker_id: sent.data?.worker?.worker_id ?? null,
        });
        answer.append(row);
        $('v2RailCount').textContent = String(log.querySelectorAll('.msg-q').length);
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
      for (const plot of log.querySelectorAll('.refcard-chart')) chartSizes.unobserve(plot);
      log.replaceChildren();
      pins.replaceChildren();
      feedback.reset();
      $('v2RailCount').textContent = '';
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
        tagZone(anchorZone, state.zone_id);   // Zone 사이 이동은 HUD 만 말한다. 이 칸은 답의 기준 Zone 만 보인다
        anchorZone.hidden = state.zone_id == null;
        anchorTime.textContent = L.clock(state.playhead_seconds);
        // 접힌 뷰어 줄에는 Zone 번호만 둔다(56px 칸)
        const rail = $('v2RailZone');
        rail.classList.toggle('zone-tag-idle', state.zone_id == null);
        rail.style.setProperty('--zone', state.zone_id == null ? '' : colorOf(state.zone_id));
        rail.textContent = state.zone_id == null ? '—' : String(Number(state.zone_id));
        rail.title = state.zone_id == null ? '이동 중' : `${zoneText(state.zone_id)} · ${state.playhead_seconds.toFixed(1)}초`;
      },
    };
  })();

  // ── 재생 알림 ─────────────────────────────────────────────
  // 서버가 공개 이상 알림과 같은 함수로 판정한 알림을 재생 시각에 맞춰 띄운다. 고르기·짧은 표기는 motion_viewer_alerts.js.
  const notices = (() => {
    const A = window.motionViewerAlerts;
    const toastHost = $('v2Toast'), button = $('v2AlertsButton'), list = $('v2AlertList');
    const items = $('v2AlertItems'), note = $('v2AlertNote'), kinds = $('v2AlertKinds'), live = $('v2AlertLive');
    let kindsKey = null;
    let wasPending = false, all = [], last = 0, seeking = true, lastKey = null, lastAt = -Infinity, current = null, focused = null;
    const tone = (alert) => `tone-${A.KIND[alert.anomaly_type]?.tone || 'warn'}`;
    const face = (alert) => {   // [종류 아이콘][Zone 네모+번호][짧은 표기]
      const icon = el('span', `alert-icon ${tone(alert)}`);
      icon.innerHTML = svg(A.KIND[alert.anomaly_type]?.icon || 'triangle-alert');
      const zone = el('span', 'zone-tag');
      tagZone(zone, alert.zone_id);
      return [icon, zone, el('span', 'alert-word', A.short(alert))];
    };
    const labelled = (node, alert) => { node.title = A.describe(alert); node.setAttribute('aria-label', node.title); return node; };
    // 순서 칩·도막 행에 알림 Zone 을 현재 Zone 과 따로 표시한다. 탐색·처음으로·알림 시각 통과·다른 기록에서 지운다.
    function focusZone(zone, alert = null) {
      focused = zone == null ? null : alert;
      alertedZone = zone;
      markCurrentZone(zone != null);
    }
    function go(alert) {
      current = alert.id;
      if (alert.anomaly_type.startsWith('coating_')) setCoating('target');   // 도막 패널의 그 행이 강조되게
      const frames = record?.frames || [];
      const index = A.seekIndex(frames, Number(alert.seek_seconds), Number(alert.at_seconds), alert);
      seekToIndex(index);   // timeline input 이 이전 강조를 지운 뒤에 새로 건다
      focusZone(A.alertZone(frames[index], alert), alert);
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
    function paintAhead() {
      for (const row of items.querySelectorAll('.alert-row')) row.classList.toggle('is-ahead', Number(row.dataset.at) > anchor.playhead_seconds);
    }
    function paintList() {
      const { pending, text } = A.listStatus(record, all.length);
      button.setAttribute('aria-busy', String(pending));
      list.setAttribute('aria-busy', String(pending));
      // 단추 = 종류별 개수(아이콘 + 수, 넓으면 이름). 개수가 바뀔 때만 다시 그린다.
      // 클릭마다 바꾸면 클릭한 svg 가 떨어져 나가 바깥 클릭 처리가 목록을 바로 닫는다.
      const counts = L.countByKind(all, Object.keys(A.KIND));
      const key = `${pending}|${counts.map((item) => `${item.type}:${item.count}`).join(',')}`;
      if (key !== kindsKey) {
        kindsKey = key;
        kinds.replaceChildren(...counts.map(({ type, count }) => {
          const item = el('span', `kind tone-${A.KIND[type].tone}`);
          item.innerHTML = svg(A.KIND[type].icon);
          item.append(el('span', 'nm', A.KIND[type].label), String(count));
          return item;
        }));
        if (counts.length) {   // 휴대폰 폭에서는 종류별 수 대신 종 + 합계 하나(CSS 가 고른다)
          const total = el('span', 'kinds-total');
          total.innerHTML = svg('bell');
          total.append(String(all.length));
          kinds.append(total);
        }
        if (pending || !counts.length) kinds.insertAdjacentHTML('beforeend', svg(pending ? 'loader-circle' : 'bell'));
      }
      const spoken = counts.map(({ type, count }) => `${A.KIND[type].label} ${count}`).join(', ');
      button.title = `알림 ${all.length}건${spoken ? ` (${spoken})` : ''}${pending ? ' · 순서·방향 알림을 불러오는 중' : ''}`;
      button.setAttribute('aria-label', button.title);
      note.textContent = text;
      note.hidden = !note.textContent;
      // 목록이 닫혀 있어도 늦게 온 순서·방향 알림이 다 왔다는 것을 화면 읽기 프로그램에 한 번 알린다
      live.textContent = wasPending && !pending ? `순서·방향 알림을 불러왔습니다. 알림 ${all.length}건` : '';
      wasPending = pending;
      const keep = items.contains(document.activeElement) ? document.activeElement.dataset.id : null;   // 늦게 온 알림으로 다시 그려도 초점 유지
      items.replaceChildren(...all.map((alert) => {
        const row = labelled(el('button', 'alert-row'), alert);
        row.type = 'button';
        row.dataset.at = String(alert.at_seconds);
        row.dataset.id = String(alert.id);
        row.append(...face(alert), el('span', 'alert-time', L.clock(alert.at_seconds)));
        if (alert.id === current) row.setAttribute('aria-current', 'true');
        row.addEventListener('click', () => go(alert));
        const item = el('li');
        item.append(row);
        return item;
      }));
      if (keep != null) [...items.querySelectorAll('.alert-row')].find((row) => row.dataset.id === keep)?.focus();
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
    document.addEventListener('input', (event) => { if (event.target.id === 'timeline') { seeking = true; if (focused) focusZone(null); } }, true);
    document.addEventListener('click', (event) => { if (event.target.closest('#restart')) { seeking = true; if (focused) focusZone(null); } }, true);

    const sync = () => { all = A.merge(record); last = anchor.playhead_seconds; seeking = true; lastKey = null; paintList(); };
    return {
      load(payload) {
        current = null;
        focusZone(null);
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
        if (focused && !jumped && seconds >= Number(focused.at_seconds)) focusZone(null);   // 재생이 알림 시각을 지났다
        last = seconds;
        seeking = false;
        if (!list.hidden) paintAhead();
      },
    };
  })();

  // 재생 시각(m:ss)과 막대 위치. 막대의 지난 쪽·남은 쪽과 손잡이는 --p 하나로 그린다.
  function renderClock(seconds) {
    const duration = Number(record?.duration_seconds) || 0;
    $('v2TimeNow').textContent = L.clock(seconds);
    $('v2TimeTotal').textContent = L.clock(duration);
    $('v2Scrub').style.setProperty('--p', `${duration ? Math.max(0, Math.min(100, seconds / duration * 100)) : 0}%`);
  }

  // ── 이벤트 ────────────────────────────────────────────────
  addEventListener('motion-viewer-data', (event) => {
    record = event.detail;
    // 기록마다 목표 대비로 연다. 목표 대비 판정이 없는 기록은 전문가 대비(막대)로 바로 연다.
    coatBasis = targetReady() ? 'target' : 'expert';
    coatMap = false;
    armSeen = record.arm_pose_evaluation;
    anchor = anchorAt(0, 0);
    hudOpen = false;
    renderAll();
    renderClock(0);
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
    if (previous.zone_id !== anchor.zone_id || previous.map_num !== anchor.map_num) renderHudFix();
    if (previous.zone_id !== anchor.zone_id || previous.map_num !== anchor.map_num) markCurrentZone(true);   // 분사만 바뀔 때는 따라가지 않는다
    renderPoseNow();
    renderClock(seconds);
    markFlaggedJoints(event.detail.frame);
    chat.setAnchor(anchor);
    notices.frame(seconds);
  });

  hydrate(document);
  syncPlay();
  layout.start();
  renderAll();
  requestAnimationFrame(() => dispatchEvent(new Event('resize')));   // 스켈레톤·도막 캔버스가 새 배치의 크기로 다시 그려지게 한다
})();
