// Read-only, explicitly opened preview. Full playback remains in the motion viewer.
(() => {
  'use strict';
  const ID = /^[a-f0-9]{32}$/;
  const MAX_BYTES = 24 * 1024 * 1024;
  const COLORS = [null, '#FF0000', '#00FF00', '#0000FF', '#00FFFF', '#FF00FF', '#FFFF00', '#FF8000', '#FF0080', '#00FF80', '#80FF00', '#0080FF', '#8000FF', '#FF8080', '#80FF80', '#8080FF', '#800000', '#008000', '#000080', '#80FFFF', '#FF80FF', '#FFFF80', '#008080', '#800080', '#808000', '#FFBF80', '#FF80BF', '#BFFF80', '#80FFBF', '#BF80FF', '#80BFFF', '#FF40BF', '#FFBF40', '#40FFBF', '#BFFF40', '#40BFFF', '#BF40FF', '#40BFBF', '#BF40BF', '#BFBF40'];
  const point = value => Array.isArray(value) && value.length === 3 && value.every(v => typeof v === 'number' && Number.isFinite(v) && Math.abs(v) <= 100000);
  const zone = item => item && /^\d+$/.test(String(item.map_num)) && /^[1-9]\d?$/.test(String(item.zone_id)) ? `${item.map_num}:${item.zone_id}` : null;
  const color = key => COLORS[Number(String(key || '').split(':')[1])] || '#94A3B8';
  const element = (tag, className, text) => {
    const node = document.createElement(tag); if (className) node.className = className; if (text != null) node.textContent = text; return node;
  };
  async function readJson(url, signal) {
    const response = await fetch(url, { signal, credentials: 'same-origin', cache: 'no-store', headers: { Accept: 'application/json', 'X-Copilot-Source': 'training-inspection' } });
    if (signal.aborted) return null;
    if (!response.ok) throw new Error(`미리보기 자료를 불러오지 못했습니다 (HTTP ${response.status}).`);
    if (Number(response.headers.get('Content-Length')) > MAX_BYTES) throw new Error('자료가 미리보기 표시 한도를 넘습니다. 전체 3D 모션뷰어에서 확인해 주세요.');
    let text;
    if (response.body?.getReader) {
      const reader = response.body.getReader(), decoder = new TextDecoder('utf-8'); let size = 0; const chunks = [];
      try {
        while (!signal.aborted) {
          const next = await reader.read(); if (next.done) break; size += next.value.byteLength;
          if (size > MAX_BYTES) { await reader.cancel(); throw new Error('자료가 미리보기 표시 한도를 넘습니다. 전체 3D 모션뷰어에서 확인해 주세요.'); }
          chunks.push(decoder.decode(next.value, { stream: true }));
        }
        if (signal.aborted) { await reader.cancel(); return null; }
        chunks.push(decoder.decode()); text = chunks.join('');
      } finally { reader.releaseLock(); }
    } else {
      text = await response.text(); if (text.length > MAX_BYTES) throw new Error('자료가 미리보기 표시 한도를 넘습니다.');
    }
    if (signal.aborted) return null;
    const data = JSON.parse(text);
    if (!data || typeof data !== 'object' || Array.isArray(data)) throw new Error('미리보기 자료 형식을 확인해 주세요.');
    return data;
  }
  function create(host, options = {}) {
    if (!host?.append || !ID.test(options.datasetId || '')) throw new Error('올바른 학습데이터를 선택해 주세요.');
    const datasetId = options.datasetId, controller = new AbortController(), events = [];
    const root = element('section', 'tt-scene'), toolbar = element('div', 'tt-scene-toolbar');
    const field = element('label', 'tt-scene-zone-label', 'Zone 보기 '), select = element('select', 'tt-scene-zone');
    select.setAttribute('aria-label', '3D 미리보기 Zone'); select.disabled = true;
    const reset = element('button', 'td-button-secondary', '초기 시점'); reset.type = 'button';
    const canvas = element('canvas', 'tt-scene-canvas'); canvas.style.width = '100%'; canvas.style.height = '280px'; canvas.style.display = 'block'; canvas.style.touchAction = 'none';
    canvas.tabIndex = 0; canvas.setAttribute('role', 'img'); canvas.setAttribute('aria-label', '선택 훈련 기록의 도장 경로와 기준 자료의 Zone 경계. 끌어서 회전합니다.');
    const status = element('p', 'tt-scene-status'), note = element('p', 'td-help tt-scene-source', '색 경로: 선택 기록의 원본 CSV · Zone 경계·흰 점: 기준 CSV에서 만든 참고 형상. JSON만으로 복원한 형상이 아닙니다. 끌어서 회전할 수 있습니다.');
    status.setAttribute('role', 'status'); status.setAttribute('aria-live', 'polite');
    field.append(select); toolbar.append(field, reset); root.append(toolbar, canvas, status, note); host.append(root);
    const ctx = canvas.getContext('2d');
    let destroyed = false, loaded = false, zoneKey = options.zoneKey || 'whole', observer, drag = null;
    let paths = [], surfaces = [], cloud = [], yaw = -.55, pitch = .1, initialYaw = -.55, initialPitch = .1;
    let width = 600, height = 280, center = [0, 0, 0], radius = 1;
    const emit = (state, message) => {
      if (destroyed) return; status.textContent = message; root.setAttribute('aria-busy', String(state === 'loading'));
      if (typeof options.onStatus === 'function') options.onStatus({ status: state, message });
    };
    const on = (node, name, fn) => { node.addEventListener(name, fn); events.push([node, name, fn]); };
    const visible = key => Boolean(key) && (zoneKey === 'whole' || key === zoneKey);
    function fit() {
      const points = [];
      for (const path of paths) if (visible(path.key)) points.push(path.point);
      for (const surface of surfaces) if (visible(surface.key)) for (const vertex of surface.vertices) if (point(vertex)) points.push(vertex);
      if (!points.length && zoneKey === 'whole') points.push(...cloud);
      if (!points.length) { center = [0, 0, 0]; radius = 1; return; }
      const low = [Infinity, Infinity, Infinity], high = [-Infinity, -Infinity, -Infinity];
      for (const p of points) for (let axis = 0; axis < 3; axis++) { low[axis] = Math.min(low[axis], p[axis]); high[axis] = Math.max(high[axis], p[axis]); }
      center = low.map((value, axis) => (value + high[axis]) / 2); radius = Math.max(.05, Math.hypot(...high.map((value, axis) => value - low[axis])) / 2);
    }
    // Same handedness as motion_viewer.py transform/project: no X/Z reflection.
    function project(p) {
      const x = p[0] - center[0], y = p[1] - center[1], z = p[2] - center[2];
      const x1 = x * Math.cos(yaw) - z * Math.sin(yaw), z1 = x * Math.sin(yaw) + z * Math.cos(yaw);
      const y1 = y * Math.cos(pitch) - z1 * Math.sin(pitch), z2 = y * Math.sin(pitch) + z1 * Math.cos(pitch);
      const distance = radius * 4.5, perspective = distance / Math.max(radius * .25, distance + z2), scale = Math.min(width, height) * .36 / radius;
      return [width / 2 + x1 * scale * perspective, height / 2 - y1 * scale * perspective];
    }
    function line(a, b, stroke, lineWidth) {
      const start = project(a), end = project(b); ctx.strokeStyle = stroke; ctx.lineWidth = lineWidth;
      ctx.beginPath(); ctx.moveTo(...start); ctx.lineTo(...end); ctx.stroke();
    }
    function draw() {
      if (destroyed || !ctx) return; ctx.clearRect(0, 0, width, height); ctx.fillStyle = '#101926'; ctx.fillRect(0, 0, width, height);
      if (!loaded) return;
      if (zoneKey === 'whole') { ctx.fillStyle = 'rgba(236,244,255,.28)'; for (const p of cloud) { const q = project(p); ctx.fillRect(q[0] - 1, q[1] - 1, 2, 2); } }
      let drawn = 0;
      for (const surface of surfaces) if (visible(surface.key)) {
        for (const edge of surface.edges) { line(surface.vertices[edge[0]], surface.vertices[edge[1]], color(surface.key) + '88', 1); drawn++; }
      }
      let previous = null;
      for (const sample of paths) {
        if (!visible(sample.key)) { previous = null; continue; }
        if (previous && previous.key === sample.key && previous.segment === sample.segment && sample.segment != null) { line(previous.point, sample.point, color(sample.key), 2.5); drawn++; }
        else { const q = project(sample.point); ctx.fillStyle = color(sample.key); ctx.fillRect(q[0] - 1.5, q[1] - 1.5, 3, 3); drawn++; }
        previous = sample;
      }
      if (!drawn) { ctx.fillStyle = '#d0dbea'; ctx.font = '14px system-ui'; ctx.fillText('선택한 Zone의 3D 위치 자료가 없습니다.', 18, height / 2); }
    }
    function resize() {
      if (destroyed) return; const rect = canvas.getBoundingClientRect(); width = Math.max(1, rect.width || 600); height = Math.max(1, rect.height || 280);
      const ratio = Math.min(2, Math.max(1, window.devicePixelRatio || 1)); canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
      ctx?.setTransform(ratio, 0, 0, ratio, 0, 0); draw();
    }
    function selectZone(key) {
      if (destroyed) return; zoneKey = key === 'whole' || /^\d+:[1-9]\d?$/.test(String(key)) ? key : 'whole';
      if (loaded) { if (!Array.from(select.children).some(item => item.value === zoneKey)) { const option = element('option', '', `맵 ${zoneKey.split(':')[0]} · Zone ${zoneKey.split(':')[1]}`); option.value = zoneKey; select.append(option); } select.value = zoneKey; fit(); draw(); }
    }
    function destroy() {
      if (destroyed) return; destroyed = true; controller.abort(); observer?.disconnect();
      for (const [node, name, fn] of events) node.removeEventListener(name, fn); events.length = 0; paths = []; surfaces = []; cloud = []; drag = null; root.remove();
    }
    on(select, 'change', () => {
      if (destroyed) return; selectZone(select.value);
      if (typeof options.onZoneChange === 'function') options.onZoneChange(zoneKey);
    });
    on(reset, 'click', () => { yaw = initialYaw; pitch = initialPitch; fit(); draw(); });
    on(canvas, 'pointerdown', event => { if (event.button !== 0) return; drag = { id: event.pointerId, x: event.clientX, y: event.clientY }; canvas.setPointerCapture?.(event.pointerId); });
    on(canvas, 'pointermove', event => {
      if (!drag || drag.id !== event.pointerId) return; yaw += (event.clientX - drag.x) * .008; pitch = Math.max(-1.4, Math.min(1.4, pitch + (event.clientY - drag.y) * .008));
      drag.x = event.clientX; drag.y = event.clientY; draw();
    });
    const endDrag = event => { if (drag?.id !== event.pointerId) return; try { canvas.releasePointerCapture?.(event.pointerId); } catch (_) {} drag = null; };
    on(canvas, 'pointerup', endDrag); on(canvas, 'pointercancel', endDrag); on(canvas, 'lostpointercapture', () => { drag = null; });
    on(canvas, 'keydown', event => { if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home'].includes(event.key)) return; event.preventDefault(); if (event.key === 'Home') { yaw = initialYaw; pitch = initialPitch; } else { yaw += event.key === 'ArrowLeft' ? -.15 : event.key === 'ArrowRight' ? .15 : 0; pitch = Math.max(-1.4, Math.min(1.4, pitch + (event.key === 'ArrowUp' ? -.1 : event.key === 'ArrowDown' ? .1 : 0))); } draw(); });
    if (typeof ResizeObserver !== 'undefined') { observer = new ResizeObserver(resize); observer.observe(canvas); } else on(window, 'resize', resize);
    resize(); emit('loading', '선택 기록의 원본 CSV와 기준 Zone 형상을 불러오는 중입니다.');
    async function load() {
      try {
        if (!ctx) throw new Error('이 브라우저에서는 3D 미리보기를 표시할 수 없습니다.');
        const selected = await readJson(`/motion-viewer/api/selection?dataset_id=${datasetId}`, controller.signal); if (destroyed) return;
        if (!selected || selected.dataset_id !== datasetId || !/^[A-Za-z0-9_-]+$/.test(selected.worker_id || '') || !['spray', 'blasting', 'grinding'].includes(selected.process) || !Number.isInteger(selected.mode_id) || selected.mode_id < 1 || selected.mode_id > 10 || typeof selected.session_id !== 'string' || !selected.session_id || selected.session_id.length > 100) throw new Error('선택한 기록의 재생 연결 정보를 확인해 주세요.');
        // frame_count is fixed at router construction; this endpoint has no per-request frame limit.
        const params = new URLSearchParams({ worker_id: selected.worker_id, process: selected.process, mode_id: String(selected.mode_id), session_id: selected.session_id });
        const data = await readJson(`/motion-viewer/api/motion?${params}`, controller.signal); if (destroyed) return;
        if (!data || data.dataset_id !== datasetId || (data.worker?.selection_worker_id || data.worker?.worker_id) !== selected.worker_id || data.process !== selected.process || data.mode_id !== selected.mode_id || data.selected_session_id !== selected.session_id) throw new Error('불러온 3D 자료와 선택 기록이 일치하지 않습니다.');
        const raw = Array.isArray(data.paint_trace) ? data.paint_trace : [], stride = Math.max(1, Math.ceil(raw.length / 12000));
        for (let index = 0; index < raw.length; index += stride) {
          const sample = raw[index], key = zone(sample);
          if (point(sample?.point) && key) paths.push({ point: sample.point.slice(), key, segment: Number.isInteger(sample.segment) ? sample.segment : null });
          else paths.push({ point: null, key: null, segment: null });
        }
        paths = paths.filter((sample, index) => sample.point || (index > 0 && paths[index - 1]?.point));
        let vertexBudget = 60000, edgeBudget = 12000;
        for (const surface of (Array.isArray(data.zone_surfaces) ? data.zone_surfaces : []).slice(0, 80)) {
          const key = zone(surface); if (!key || !Array.isArray(surface.vertices) || surface.vertices.length > vertexBudget) continue;
          const vertices = surface.vertices, edges = [];
          for (const edge of (Array.isArray(surface.boundary_edges) ? surface.boundary_edges : []).slice(0, edgeBudget)) {
            if (Array.isArray(edge) && edge.length === 2 && edge.every(i => Number.isInteger(i) && i >= 0 && i < vertices.length && point(vertices[i]))) edges.push(edge.slice());
          }
          vertexBudget -= vertices.length; edgeBudget -= edges.length; surfaces.push({ key, vertices, edges });
        }
        const rawCloud = Array.isArray(data.reference_surface?.points) ? data.reference_surface.points : [], cloudStep = Math.max(1, Math.ceil(rawCloud.length / 4000));
        for (let i = 0; i < rawCloud.length; i += cloudStep) if (point(rawCloud[i]?.point)) cloud.push(rawCloud[i].point.slice());
        const forward = (Array.isArray(data.frames) ? data.frames : []).find(frame => point(frame?.camera_forward))?.camera_forward;
        if (forward && Math.hypot(...forward) > .01) { initialYaw = Math.atan2(forward[0], forward[2]); initialPitch = Math.asin(Math.max(-.95, Math.min(.95, forward[1] / Math.hypot(...forward)))); }
        yaw = initialYaw; pitch = initialPitch;
        select.replaceChildren(); const whole = element('option', '', '전체 Zone'); whole.value = 'whole'; select.append(whole);
        const keys = [...new Set([...paths.map(item => item.key), ...surfaces.map(item => item.key)].filter(Boolean))].sort((a, b) => Number(a.split(':')[0]) - Number(b.split(':')[0]) || Number(a.split(':')[1]) - Number(b.split(':')[1]));
        for (const key of keys) { const option = element('option', '', `맵 ${key.split(':')[0]} · Zone ${key.split(':')[1]}`); option.value = key; select.append(option); }
        loaded = true; select.disabled = false; selectZone(zoneKey); emit('ready', `선택 기록의 경로와 기준 Zone 경계를 표시했습니다. ${keys.length}개 Zone · 정지 미리보기`);
      } catch (error) { if (!destroyed && error.name !== 'AbortError') emit('error', error.message || '3D 미리보기를 불러오지 못했습니다.'); }
    }
    void load();
    return Object.freeze({ selectZone, destroy });
  }
  window.TrainingTransformScene = Object.freeze({ create });
})();
