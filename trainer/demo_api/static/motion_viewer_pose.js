(() => {
  'use strict';
  const byId = id => document.getElementById(id);
  const template = byId('poseUiTemplate');
  const pathCanvas = byId('canvas');
  if (!template || !pathCanvas || !window.MotionViewerCamera) return;

  const fragment = template.content.cloneNode(true);
  const gizmo = fragment.querySelector('[data-pose-slot="gizmo"]');
  const summary = fragment.querySelector('[data-pose-slot="summary"]');
  document.querySelector('.stage-viewport').append(gizmo);
  document.querySelector('.evaluation-side').prepend(summary);

  let camera = window.MotionViewerCamera.create({target: [0, 0, 0], yaw: -.55, pitch: .1, distance: 175});
  let cameraDrag = null;
  const pressedKeys = new Set();
  const subtract = (a, b) => a.map((value, axis) => value - b[axis]);
  const addVector = (a, b) => a.map((value, axis) => value + b[axis]);
  const scaleVector = (a, scalar) => a.map(value => value * scalar);
  const vectorLength = a => Math.hypot(...a);
  const normalize = (a, fallback = [0, 0, 1]) => {
    const size = vectorLength(a);
    return size > 1e-6 ? scaleVector(a, 1 / size) : fallback;
  };
  const crossVector = (a, b) => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
  const angleDegrees = (a, b) => {
    const left = normalize(a), right = normalize(b);
    const cosine = Math.max(-1, Math.min(1, left.reduce((sum, value, axis) => sum + value * right[axis], 0)));
    return Math.acos(cosine) * 180 / Math.PI;
  };

  function trustedUpperBody(frame) {
    const tracked = frame?.tracked || {};
    const head = tracked.head;
    if (!Array.isArray(head)) return null;
    const forward = normalize(frame.camera_forward || [0, 0, 1]);
    const horizontalForward = normalize([forward[0], 0, forward[2]]);
    const right = normalize(crossVector([0, 1, 0], horizontalForward), [1, 0, 0]);
    const neck = addVector(head, [0, -0.17, 0]);
    const chest = addVector(head, [0, -0.36, 0]);
    return {
      head, neck, chest,
      leftShoulder: addVector(chest, scaleVector(right, -0.19)),
      rightShoulder: addVector(chest, scaleVector(right, 0.19)),
      leftHand: tracked.left_hand || null,
      rightHand: tracked.right_hand || null,
    };
  }

  function drawTrustedUpperBody(frame, center, width, height) {
    const body = trustedUpperBody(frame);
    if (!body) return;
    const inferred = 'rgba(148,210,230,.52)';
    line(body.head, body.neck, inferred, 4, center, width, height);
    line(body.neck, body.chest, inferred, 5, center, width, height);
    line(body.leftShoulder, body.rightShoulder, inferred, 5, center, width, height);
    if (body.leftHand) line(body.leftShoulder, body.leftHand, 'rgba(34,211,238,.60)', 3, center, width, height);
    if (body.rightHand) line(body.rightShoulder, body.rightHand, 'rgba(34,211,238,.70)', 3, center, width, height);
    point(body.head, 13, '#f5f8ff', center, width, height);
    point(body.neck, 4, 'rgba(186,230,245,.72)', center, width, height);
    point(body.chest, 6, 'rgba(186,230,245,.72)', center, width, height);
    point(body.leftShoulder, 5, inferred, center, width, height);
    point(body.rightShoulder, 5, inferred, center, width, height);
    if (body.rightHand) point(body.rightHand, 7, '#22d3ee', center, width, height);
    if (body.leftHand) point(body.leftHand, 6, '#8be9fd', center, width, height);
  }

  function drawOrientationGuides(frame, center, width, height) {
    const tracked = frame?.tracked || {};
    if (tracked.head && frame.camera_forward) {
      arrow(tracked.head, addVector(tracked.head, scaleVector(normalize(frame.camera_forward), .42)), '#22d3ee', center, width, height);
    }
    if (tracked.right_hand && frame.tool_forward) {
      const end = addVector(tracked.right_hand, scaleVector(normalize(frame.tool_forward), .42));
      arrow(tracked.right_hand, end, '#fb923c', center, width, height);
      if (frame.ray_point) {
        const hitDirection = subtract(frame.ray_point, tracked.right_hand);
        arrow(tracked.right_hand, frame.ray_point, 'rgba(250,204,21,.82)', center, width, height);
        const shown = project(end, center, width, height);
        ctx.fillStyle = '#fde68a';
        ctx.font = '700 10px system-ui';
        ctx.fillText(`분사축 편차 ${angleDegrees(frame.tool_forward, hitDirection).toFixed(0)}°`, shown[0] + 7, shown[1] - 6);
      }
    }
  }

  function renderPoseSummary(frame) {
    const evaluation = data?.pose_evaluation || {};
    byId('posePosture').textContent = frame?.posture?.label || '자세 판정 보류';
    const zone = Number(frame?.zone_id) > 0 ? `Zone ${zoneLabel(frame.zone_id)}` : '이동 구간';
    byId('poseZoneSummary').textContent = zone;
    byId('poseEstimateNote').textContent = 'CSV 조인트 기반 자세 추정 · 채점 분리';
    byId('poseScore').textContent = evaluation.similarity_score == null ? '분석 대기' : evaluation.similarity_score.toFixed(3);
    byId('poseScoreNote').textContent = evaluation.reference_session_id
      ? `대표 전문가 기록 ${evaluation.reference_session_id}과 정규화된 전신 관절 동작을 비교합니다.`
      : '현재 Zone·분사 구간을 같은 조건의 전문가 기록과 비교합니다.';
    byId('poseAction').textContent = evaluation.corrections?.[0]?.message || evaluation.message || '분석 결과가 없습니다.';
    byId('posePredicted').textContent = evaluation.predicted_score == null || evaluation.similarity_score == null
      ? '—'
      : `${evaluation.similarity_score.toFixed(3)} → ${evaluation.predicted_score.toFixed(3)}`;
  }

  function renderZoneTimeline(payload) {
    const host = byId('zoneTimeline');
    const duration = Math.max(.001, Number(payload?.duration_seconds) || 0);
    const timelineBucketSeconds = .5;
    const buckets = new Map();
    for (const sample of payload?.paint_trace || []) {
      if (!(Number(sample.zone_id) > 0)) continue;
      const bucketIndex = Math.floor(Number(sample.t || 0) / timelineBucketSeconds);
      const key = `${sample.map_num ?? ''}:${sample.zone_id ?? ''}`;
      if (!buckets.has(bucketIndex)) buckets.set(bucketIndex, {index: bucketIndex, zoneCounts: new Map()});
      const bucket = buckets.get(bucketIndex);
      const current = bucket.zoneCounts.get(key) || {count: 0, zone: sample.zone_id};
      current.count += 1;
      bucket.zoneCounts.set(key, current);
    }
    const segments = [];
    for (const bucket of [...buckets.values()].sort((left, right) => left.index - right.index)) {
      const [key, winner] = [...bucket.zoneCounts.entries()].sort((left, right) => right[1].count - left[1].count)[0];
      const start = bucket.index * timelineBucketSeconds;
      const end = Math.min(duration, start + timelineBucketSeconds);
      const last = segments[segments.length - 1];
      if (!last || last.key !== key || start - last.end > timelineBucketSeconds * 1.1) {
        segments.push({key, start, end, zone: winner.zone});
      }
      else last.end = end;
    }
    host.replaceChildren(...segments.map(segment => {
      const element = document.createElement('span');
      element.className = 'zone-timeline-segment';
      element.style.left = `${Math.max(0, segment.start / duration * 100)}%`;
      element.style.width = `${Math.max(.7, (segment.end - segment.start) / duration * 100)}%`;
      element.style.background = zoneColor(segment.zone);
      element.textContent = `Z${zoneLabel(segment.zone)}`;
      element.title = `Zone ${zoneLabel(segment.zone)} · ${segment.start.toFixed(2)}~${segment.end.toFixed(2)}초`;
      return element;
    }));
  }

  function drawPoseOverlay({frame, center, w, h, playheadSeconds, durationSeconds}) {
    drawTrustedUpperBody(frame, center, w, h);
    drawOrientationGuides(frame, center, w, h);
    renderPoseSummary(frame);
    const zone = Number(frame?.zone_id) > 0 ? `현재 Zone ${zoneLabel(frame.zone_id)}` : '이동 구간';
    byId('playbackZone').textContent = `${zone} · ${frame?.posture?.label || '자세 판정 보류'}`;
    byId('playbackTime').textContent = `${Number(playheadSeconds || 0).toFixed(2)} / ${Number(durationSeconds || 0).toFixed(2)}초`;
  }
  window.motionViewerPoseOverlay = {draw: drawPoseOverlay};
  window.renderZoneTimeline = renderZoneTimeline;
  addEventListener('motion-viewer-data', event => renderZoneTimeline(event.detail));

  const syncCameraGlobals = state => {
    viewCenter = [...state.target]; yaw = state.yaw; pitch = state.pitch; zoom = state.distance; panX = 0; panY = 0;
  };
  const visiblePoints = payload => {
    const points = [];
    for (const item of payload?.reference_surface?.points || []) if (Array.isArray(item.point)) points.push(item.point);
    for (const surface of payload?.zone_surfaces || []) points.push(...(surface.vertices || []));
    for (const frame of payload?.frames || []) for (const item of Object.values(frame.tracked || {})) if (Array.isArray(item)) points.push(item);
    return points;
  };
  const originalResetView = resetViewToCamera;
  resetViewToCamera = payload => {
    originalResetView(payload);
    camera = window.MotionViewerCamera.create({target: viewCenter || [0, 0, 0], yaw, pitch, distance: zoom});
    syncCameraGlobals(camera.fit(visiblePoints(payload)));
  };
  const applyCamera = state => { syncCameraGlobals(state); draw(); };
  const refit = () => applyCamera(camera.focus(visiblePoints(data)));
  byId('cameraHome').onclick = () => applyCamera(camera.home());
  byId('cameraAxisX').onclick = () => applyCamera(camera.axisView('x'));
  byId('cameraAxisY').onclick = () => applyCamera(camera.axisView('y'));
  byId('cameraAxisZ').onclick = () => applyCamera(camera.axisView('z'));
  pathCanvas.onpointerdown = event => {
    if (![0, 1, 2].includes(event.button)) return;
    event.preventDefault(); pathCanvas.setPointerCapture(event.pointerId);
    cameraDrag = {mode: event.button === 1 ? 'pan' : event.button === 2 ? 'look' : 'orbit', x: event.clientX, y: event.clientY};
    pathCanvas.style.cursor = cameraDrag.mode === 'pan' ? 'move' : 'grabbing';
  };
  pathCanvas.onpointermove = event => {
    if (!cameraDrag) return;
    const dx = event.clientX - cameraDrag.x, dy = event.clientY - cameraDrag.y;
    cameraDrag.x = event.clientX; cameraDrag.y = event.clientY;
    if (cameraDrag.mode === 'pan') applyCamera(camera.pan(dx, dy, pathCanvas.clientHeight));
    else if (cameraDrag.mode === 'look') applyCamera(camera.look(dx, dy));
    else applyCamera(camera.orbit(dx, dy));
  };
  pathCanvas.onpointerup = event => {
    if (pathCanvas.hasPointerCapture(event.pointerId)) pathCanvas.releasePointerCapture(event.pointerId);
    cameraDrag = null; pathCanvas.style.cursor = 'grab';
  };
  pathCanvas.onpointercancel = () => { cameraDrag = null; pathCanvas.style.cursor = 'grab'; };
  pathCanvas.oncontextmenu = event => event.preventDefault();
  pathCanvas.onauxclick = event => event.preventDefault();
  pathCanvas.onwheel = event => { event.preventDefault(); applyCamera(camera.dolly(event.deltaY < 0 ? 1 : -1)); };
  addEventListener('keydown', event => {
    if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return;
    pressedKeys.add(event.code);
    if (event.code === 'KeyF') { event.preventDefault(); refit(); }
  });
  addEventListener('keyup', event => pressedKeys.delete(event.code));
  function flyTick() {
    if (cameraDrag?.mode === 'look') {
      const forward = Number(pressedKeys.has('KeyW')) - Number(pressedKeys.has('KeyS'));
      const right = Number(pressedKeys.has('KeyD')) - Number(pressedKeys.has('KeyA'));
      const up = Number(pressedKeys.has('KeyE')) - Number(pressedKeys.has('KeyQ'));
      if (forward || right || up) applyCamera(camera.fly(forward, right, up));
    }
    requestAnimationFrame(flyTick);
  }
  resize();
  if (typeof data !== 'undefined' && data?.frames?.length) { resetViewToCamera(data); renderZoneTimeline(data); }
  requestAnimationFrame(flyTick);
})();
