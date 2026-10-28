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
  let lastZoneEvaluationKey = null;
  let lastPlaybackZone = null;
  let lastRealtimeZone = null;
  let lastRealtimeTelemetry = null;
  let lastRealtimeGuidance = null;
  let lastTelemetryTime = null;
  let telemetrySamples = [];
  let realtimeGuidance = [];
  const HUD_WINDOW_SECONDS = .5;
  const HUD_UV_CELL_SIZE = 32;

  function currentZoneFromFrame(frame) {
    if (!(frame?.spraying && Number(frame?.zone_id) > 0 && frame?.map_num != null)) return null;
    return {
      key: `${frame.map_num}:${frame.zone_id}`,
      mapNum: frame.map_num,
      zoneId: frame.zone_id,
    };
  }

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

  function finiteNumber(value) {
    if (value == null || value === '') return null;
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  }

  function mean(samples, key) {
    const values = samples.map(sample => sample[key]).filter(value => value != null);
    return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
  }

  function telemetryForFrame(frame) {
    const time = finiteNumber(frame?.t);
    if (frame?.spraying && time != null && time !== lastTelemetryTime) {
      if (lastTelemetryTime != null && time < lastTelemetryTime) telemetrySamples = [];
      const tracked = frame?.tracked || {};
      const hitDirection = tracked.right_hand && frame.ray_point
        ? subtract(frame.ray_point, tracked.right_hand)
        : null;
      telemetrySamples.push({
        t: time,
        speed: finiteNumber(frame.speed_mms),
        distance: finiteNumber(frame.distance_mm),
        axis: frame.tool_forward && hitDirection ? angleDegrees(frame.tool_forward, hitDirection) : null,
      });
      telemetrySamples = telemetrySamples.filter(sample => sample.t >= time - HUD_WINDOW_SECONDS);
      lastTelemetryTime = time;
      lastRealtimeTelemetry = {
        speed: mean(telemetrySamples, 'speed'),
        distance: mean(telemetrySamples, 'distance'),
        axis: mean(telemetrySamples, 'axis'),
      };
    }
    return lastRealtimeTelemetry;
  }

  function comparisonState(actual, reference, lowLabel, highLabel) {
    if (actual == null) return {tone: 'idle', label: '대기'};
    if (!(reference > 0)) return {tone: 'ok', label: '측정'};
    const ratio = actual / reference;
    if (ratio < .8) return {tone: 'warn', label: lowLabel};
    if (ratio > 1.2) return {tone: 'warn', label: highLabel};
    return {tone: 'ok', label: '적정'};
  }

  function axisState(value) {
    if (value == null) return {tone: 'idle', label: '대기'};
    if (value <= 10) return {tone: 'ok', label: '적정'};
    if (value <= 20) return {tone: 'warn', label: '주의'};
    return {tone: 'alert', label: '교정'};
  }

  function paintDirection(delta) {
    const length = Math.hypot(delta[0], delta[1]);
    if (length <= 1e-6) return null;
    const verticalRatio = Math.abs(delta[1]) / length;
    if (verticalRatio >= .7) return 'vertical';
    if (verticalRatio <= .3) return 'horizontal';
    return 'diagonal';
  }

  function directionLabel(value) {
    return ({horizontal: '수평', vertical: '수직', diagonal: '대각선', irregular: '비정형'})[value] || '분석 중';
  }

  function buildRealtimeGuidance(payload) {
    const referenceCells = new Map((payload?.zone_surfaces || []).map(surface => [
      `${surface.map_num}:${surface.zone_id}`,
      Number(surface.reference_uv_cell_count) || 0,
    ]));
    const states = new Map();
    const trace = payload?.paint_trace || [];
    let traceIndex = 0;
    let latest = null;
    return (payload?.frames || []).map(frame => {
      while (traceIndex < trace.length && Number(trace[traceIndex].t || 0) <= Number(frame.t || 0) + 1e-6) {
        const sample = trace[traceIndex++];
        const key = `${sample.map_num}:${sample.zone_id}`;
        const uv = sample.uv;
        const {segment} = sample;
        if (!(Number(sample.zone_id) > 0) || !Array.isArray(uv)) continue;
        const state = states.get(key) || {cells: new Set(), recent: [], previousUv: null, segment: null};
        if (state.segment !== segment) {
          state.previousUv = null;
          state.recent = [];
        }
        state.segment = segment;
        state.cells.add(`${Math.round(Number(uv[0]) / HUD_UV_CELL_SIZE)}:${Math.round(Number(uv[1]) / HUD_UV_CELL_SIZE)}`);
        if (state.previousUv) {
          const direction = paintDirection([Number(uv[0]) - state.previousUv[0], Number(uv[1]) - state.previousUv[1]]);
          if (direction) state.recent.push(direction);
          if (state.recent.length > 18) state.recent.shift();
        }
        state.previousUv = [Number(uv[0]), Number(uv[1])];
        states.set(key, state);
        const counts = state.recent.reduce((result, direction) => ({...result, [direction]: (result[direction] || 0) + 1}), {});
        const ranked = Object.entries(counts).sort((left, right) => right[1] - left[1]);
        const dominant = ranked[0]?.[0] || null;
        const confidence = dominant ? ranked[0][1] / state.recent.length : 0;
        const denominator = referenceCells.get(key) || 0;
        latest = {
          key,
          direction: confidence >= .55 ? dominant : state.recent.length ? 'irregular' : null,
          coverage: denominator ? Math.min(100, state.cells.size / denominator * 100) : null,
        };
      }
      return latest ? {...latest} : null;
    });
  }

  function directionState(actual, expected) {
    if (!actual) return {tone: 'idle', label: '분석 중'};
    if (!expected) return {tone: 'ok', label: '측정'};
    return actual === expected
      ? {tone: 'ok', label: '일치'}
      : {tone: actual === 'irregular' ? 'alert' : 'warn', label: '교정'};
  }

  function coverageState(value) {
    if (value == null) return {tone: 'idle', label: '대기'};
    if (value >= 80) return {tone: 'ok', label: '충분'};
    return {tone: 'warn', label: value < 20 ? '시작' : '진행 중'};
  }

  function setHudMetric(valueId, stateId, value, state, formatter, reference = null) {
    const valueNode = byId(valueId);
    const stateNode = byId(stateId);
    valueNode.textContent = value == null ? '—' : formatter(value);
    valueNode.title = reference > 0 ? `전문가 중앙값 ${formatter(reference)}` : '';
    stateNode.textContent = state.label;
    stateNode.className = `realtime-hud-state ${state.tone}`;
  }

  function setHudTextMetric(valueId, stateId, value, state) {
    const valueNode = byId(valueId);
    const stateNode = byId(stateId);
    valueNode.textContent = value || '—';
    stateNode.textContent = state.label;
    stateNode.className = `realtime-hud-state ${state.tone}`;
  }

  function updateRealtimeHud(frame, sourceIndex) {
    const currentZone = currentZoneFromFrame(frame);
    if (currentZone) lastRealtimeZone = currentZone;
    const zone = lastRealtimeZone;
    const telemetry = telemetryForFrame(frame) || {};
    const preparedGuidance = realtimeGuidance[sourceIndex] || null;
    if (preparedGuidance) lastRealtimeGuidance = preparedGuidance;
    const guidance = lastRealtimeGuidance || {};
    const reference = zone ? data?.zone_statistics?.[zone.key] || {} : {};
    const hud = byId('realtimeHud');
    const zoneNode = byId('realtimeHudZone');
    const activityNode = byId('realtimeHudActivity');
    if (zone) {
      const color = zoneColor(zone.zoneId);
      zoneNode.textContent = `Zone ${zoneLabel(zone.zoneId)}`;
      zoneNode.style.color = color;
      hud.style.borderLeft = `3px solid ${color}`;
    } else {
      zoneNode.textContent = 'Zone 대기';
      zoneNode.style.color = '';
      hud.style.borderLeft = '';
    }
    activityNode.textContent = currentZone ? '분사 중' : '이동 중';
    activityNode.style.color = currentZone ? '#22c55e' : '#94a3b8';
    hud.style.opacity = currentZone ? '1' : '.72';
    const speedReference = finiteNumber(reference.median_speed_mms);
    const distanceReference = finiteNumber(reference.median_distance_mm);
    setHudMetric(
      'realtimeHudSpeed', 'realtimeHudSpeedState', telemetry.speed,
      comparisonState(telemetry.speed, speedReference, '느림', '빠름'),
      value => `${(value / 1000).toFixed(2)} m/s`, speedReference,
    );
    setHudMetric(
      'realtimeHudDistance', 'realtimeHudDistanceState', telemetry.distance,
      comparisonState(telemetry.distance, distanceReference, '가까움', '멂'),
      value => `${Math.round(value)} mm`, distanceReference,
    );
    setHudMetric(
      'realtimeHudAxis', 'realtimeHudAxisState', telemetry.axis,
      axisState(telemetry.axis), value => `${value.toFixed(1)}°`,
    );
    const expectedDirection = reference.expected_direction || null;
    setHudTextMetric(
      'realtimeHudDirection', 'realtimeHudDirectionState', directionLabel(guidance.direction),
      directionState(guidance.direction, expectedDirection),
    );
    setHudMetric(
      'realtimeHudCoverage', 'realtimeHudCoverageState', guidance.coverage,
      coverageState(guidance.coverage), value => `${Math.round(value)}%`,
    );
  }

  function metricValue(metric, side) {
    const label = metric?.[`${side}_label`];
    if (label) return label;
    const value = metric?.[side];
    if (value == null) return '평가 불가';
    const unit = metric?.unit === 'deg' ? '°' : metric?.unit === 'um' ? ' µm' : metric?.unit || '';
    return `${Number(value).toFixed(1)}${unit}`;
  }

  function renderZoneMetric(metric) {
    const row = document.createElement('div');
    row.className = 'pose-zone-metric';
    const name = document.createElement('span');
    name.className = 'pose-zone-metric-name';
    name.textContent = metric.label;
    const values = document.createElement('div');
    values.className = 'pose-zone-metric-values';
    for (const [key, label] of [['actual', '현재 :'], ['reference', '전문가 :']]) {
      const line = document.createElement('span');
      line.className = 'pose-zone-value';
      const caption = document.createElement('span');
      caption.textContent = label;
      const value = document.createElement('b');
      value.textContent = metricValue(metric, key);
      line.append(caption, value);
      values.append(line);
    }
    const progress = document.createElement('div');
    progress.className = 'pose-zone-progress';
    const track = document.createElement('div');
    track.className = 'pose-zone-progress-track';
    const fill = document.createElement('div');
    fill.className = 'pose-zone-progress-fill';
    const score = metric.similarity_percent;
    fill.style.width = `${score == null ? 0 : Math.max(0, Math.min(100, score))}%`;
    track.append(fill);
    const value = document.createElement('span');
    value.textContent = score == null ? '—' : `${score}%`;
    progress.append(track, value);
    row.append(name, values, progress);
    return row;
  }

  function renderCorrections(corrections, emptyMessage = '전문가 정상 범위입니다. 현재 동작을 유지하세요.') {
    const list = byId('poseActionList');
    const messages = (corrections || []).map(item => typeof item === 'string' ? item : item?.message).filter(Boolean).slice(0, 3);
    if (!messages.length) messages.push(emptyMessage);
    list.replaceChildren(...messages.map((message, index) => {
      const item = document.createElement('li');
      if (index === 0) item.id = 'poseAction';
      item.textContent = message;
      return item;
    }));
  }

  function renderZoneEvaluation(frame, evaluation) {
    const currentKey = currentZoneFromFrame(frame)?.key || null;
    if (currentKey && evaluation?.zones?.[currentKey]) lastZoneEvaluationKey = currentKey;
    const zoneEvaluation = lastZoneEvaluationKey ? evaluation?.zones?.[lastZoneEvaluationKey] : null;
    const card = byId('poseZoneCard');
    const metrics = byId('poseZoneMetrics');
    if (!zoneEvaluation) {
      byId('poseZoneTitle').textContent = '현재 Zone 유사도';
      byId('poseZoneScore').textContent = '—%';
      byId('poseActionTitle').textContent = '현재 Zone 자세 개선 안내';
      metrics.replaceChildren(Object.assign(document.createElement('small'), {textContent: '현재 Zone 평가 자료가 없습니다.'}));
      renderCorrections([], '분사 Zone에 진입하면 해당 Zone의 자세 개선 안내를 표시합니다.');
      return null;
    }
    byId('poseZoneTitle').textContent = `Zone ${zoneLabel(zoneEvaluation.zone_id)} 유사도`;
    byId('poseZoneScore').textContent = zoneEvaluation.similarity_percent == null ? '—%' : `${zoneEvaluation.similarity_percent}%`;
    byId('poseActionTitle').textContent = `Zone ${zoneLabel(zoneEvaluation.zone_id)} 자세 개선 안내`;
    card.style.borderColor = zoneColor(zoneEvaluation.zone_id);
    metrics.replaceChildren(...(zoneEvaluation.metrics || []).map(renderZoneMetric));
    renderCorrections(zoneEvaluation.corrections);
    return zoneEvaluation;
  }

  function renderPoseSummary(frame) {
    const evaluation = data?.pose_evaluation || {};
    byId('posePosture').textContent = frame?.posture?.label || '자세 판정 보류';
    byId('poseEstimateNote').textContent = '시간 제외 · 동일 Zone 전문가 기준';
    byId('poseScore').textContent = evaluation.similarity_percent == null ? '분석 대기' : `${evaluation.similarity_percent}%`;
    byId('poseScoreNote').textContent = evaluation.similarity_percent == null
      ? '현재 Zone·분사 구간을 같은 조건의 전문가 기록과 비교합니다.'
      : `Zone 평균 ${evaluation.zone_similarity_percent}% · 순서 ${evaluation.order_similarity_percent}% · 전문가 ${evaluation.reference_session_count}회 기준`;
    const zoneEvaluation = renderZoneEvaluation(frame, evaluation);
    byId('poseZoneSummary').textContent = zoneEvaluation
      ? `Zone ${zoneLabel(zoneEvaluation.zone_id)}`
      : 'Zone 대기 중';
    const current = zoneEvaluation?.similarity_percent ?? evaluation.similarity_percent;
    const predicted = zoneEvaluation?.predicted_similarity_percent ?? evaluation.predicted_similarity_percent;
    byId('posePredicted').textContent = current == null || predicted == null ? '—' : `${current}% → ${predicted}%`;
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

  function drawPoseOverlay({frame, sourceIndex, center, w, h, playheadSeconds, durationSeconds}) {
    drawTrustedUpperBody(frame, center, w, h);
    drawOrientationGuides(frame, center, w, h);
    updateRealtimeHud(frame, sourceIndex);
    renderPoseSummary(frame);
    const currentZone = currentZoneFromFrame(frame);
    if (currentZone) lastPlaybackZone = currentZone;
    const activity = currentZone ? `현재 Zone ${zoneLabel(currentZone.zoneId)}` : '이동 구간';
    byId('playbackZone').textContent = `${activity} · ${frame?.posture?.label || '자세 판정 보류'}`;
    const badge = byId('playbackZoneBadge');
    if (lastPlaybackZone) {
      const color = zoneColor(lastPlaybackZone.zoneId);
      badge.hidden = false;
      badge.textContent = `Zone ${zoneLabel(lastPlaybackZone.zoneId)}`;
      badge.style.color = color;
      badge.style.borderColor = color;
      badge.style.background = colorWithAlpha(color, .16);
    } else {
      badge.hidden = true;
    }
    byId('playbackTime').textContent = `${Number(playheadSeconds || 0).toFixed(2)} / ${Number(durationSeconds || 0).toFixed(2)}초`;
  }
  window.motionViewerPoseOverlay = {draw: drawPoseOverlay};
  window.renderZoneTimeline = renderZoneTimeline;
  addEventListener('motion-viewer-data', event => {
    lastZoneEvaluationKey = null;
    lastPlaybackZone = null;
    lastRealtimeZone = null;
    lastRealtimeTelemetry = null;
    lastRealtimeGuidance = null;
    lastTelemetryTime = null;
    telemetrySamples = [];
    realtimeGuidance = buildRealtimeGuidance(event.detail);
    const badge = byId('playbackZoneBadge');
    if (badge) badge.hidden = true;
    renderZoneTimeline(event.detail);
  });

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
