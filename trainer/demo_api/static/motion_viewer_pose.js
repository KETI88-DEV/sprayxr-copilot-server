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

  // 큰 장면 상체: 머리는 실측 HMD, 어깨·가슴은 패널과 같은 몸 비율 표(BODY)로 놓는다. 팔은 패널과 같은 두 마디 IK 로
  // 오른팔은 건 손잡이, 왼팔은 왼손 추적기까지 뻗는다(팔꿈치 아래·바깥).
  function trustedUpperBody(frame) {
    const tracked = frame?.tracked || {};
    const head = tracked.head;
    if (!Array.isArray(head)) return null;
    const forward = unit(frame.camera_forward || [0, 0, 1], [0, 0, 1]);
    const right = unit(cross([0, 1, 0], unit([forward[0], 0, forward[2]], [0, 0, 1])), [1, 0, 0]);
    const neck = add(head, [0, -BODY.headAboveShoulders, 0]);
    const leftShoulder = add(neck, mul(right, -BODY.shoulderWidth / 2));
    const rightShoulder = add(neck, mul(right, BODY.shoulderWidth / 2));
    const gun = tracked.right_hand || null;
    const leftHand = tracked.left_hand || null;
    const tool = frame.tool_forward;
    const grip = gun && tool && length(tool) > 1e-5 ? gunGrip(gun, tool) : gun;
    const arm = (shoulder, target, outward) => target
      ? twoBoneIK(shoulder, target, BODY.upperArm, BODY.foreArm, add([0, -1, 0], mul(right, outward)))
      : null;
    return {
      head, neck, chest: add(neck, [0, -BODY.upperChest, 0]), leftShoulder, rightShoulder, gun, leftHand, grip, forward, right,
      leftArm: arm(leftShoulder, leftHand, -.5),
      rightArm: arm(rightShoulder, grip, .5),
    };
  }

  // 큰 장면 몸의 살 두께(m, [시작, 끝]): 자세 비교 실험대 body_proportions FLESH_RADIUS_ADULT_M × .85
  // (뷰어 BODY 팔 길이가 성인 위팔의 약 .85 배다). 골반 아래는 재지 않으므로 가슴에서 끊는다(2026-11-12 사용자 선택 A안).
  const FLESH = {
    neck: [.04, .038], neckSpan: [.02, .06], chest: [.095, .085], chestTop: .075, chestHalf: [.1, .08], shoulderBridge: .055,
    upperArm: [.044, .04], foreArm: [.038, .029], palm: [.027, .022], palmReach: [-.01, .07],
    head: {rx: .082, ry: .108, back: .045, lift: .015}, visor: [.03, .07, .036],
  };
  // 건(m): 손잡이 점(gunGrip)에서 아래·뒤로 15° 기운 손잡이, 그 앞 몸통, 연장대, 추적기 앞 노즐. 반폭은 [축, 둘째, 셋째].
  const GUN = {
    handle: [.055, .016, .014], handleTilt: Math.PI / 12, body: [.075, .024, .018], bodyAhead: .035,
    rodFrom: .11, rodRadius: .008, nozzle: [.07, .105], nozzleRadius: [.012, .016],
  };
  // 두 단계 명암 [밝은 색, 어두운 색]. 팔·몸통의 밝은 색은 LIMB_COLORS 에서 읽는다.
  const SHADOW = {arm: '#0f8ea6', core: '#9fb0c6'};
  const GUN_TONE = ['#aab6c8', '#5f6b7d'], NOZZLE_TONE = ['#fb923c', '#c2410c'], VISOR_TONE = ['#3a4458', '#161b26'];
  const OUTLINE = 'rgba(10,14,22,.55)';
  const BODY_LAYER_ALPHA = .88;
  const LIGHT_SCREEN = [-.5, -.86].map(value => value / Math.hypot(.5, .86));
  const LIGHT_CAMERA = [-.45, .75, -.5].map(value => value / Math.hypot(.45, .75, .5));
  const EYE = [0, 0, -4.5]; // 코어 project() 의 원근(4.5 / (4.5 + z))이 가정하는 카메라 공간 눈 위치
  let bodyLayer = null, lastBodyRect = null;

  // 두 원과 공통 외접선으로 캡슐 윤곽을 지금 경로에 더한다. 반지름(px) = r × zoom × 원근.
  function capsulePath(g, a, b, ra, rb, center, w, h) {
    const p = project(a, center, w, h), q = project(b, center, w, h);
    const pr = Math.max(1, ra * zoom * p[2]), qr = Math.max(1, rb * zoom * q[2]);
    const dx = q[0] - p[0], dy = q[1] - p[1], d = Math.hypot(dx, dy);
    if (d <= Math.abs(pr - qr) + .5) {
      // 축이 화면을 향해 한 원이 다른 원을 품으면 큰 원 하나다.
      const [c, r] = pr >= qr ? [p, pr] : [q, qr];
      g.moveTo(c[0] + r, c[1]);
      g.arc(c[0], c[1], r, 0, Math.PI * 2);
    } else {
      const th = Math.atan2(dy, dx), al = Math.acos((pr - qr) / d);
      g.moveTo(p[0] + pr * Math.cos(th + al), p[1] + pr * Math.sin(th + al));
      g.arc(p[0], p[1], pr, th + al, th + Math.PI * 2 - al);
      g.arc(q[0], q[1], qr, th - al, th + al);
      g.closePath();
    }
    return {p, q, d, r: Math.max(pr, qr)};
  }

  // 두 단계 명암: 화면 빛 방향 n 쪽 절반은 밝은 색, 반대쪽은 어두운 색.
  function twoTone(g, cx, cy, n, radius, tone) {
    const gradient = g.createLinearGradient(cx - n[0] * radius, cy - n[1] * radius, cx + n[0] * radius, cy + n[1] * radius);
    for (const [offset, color] of [[0, tone[1]], [.4, tone[1]], [.52, tone[0]], [1, tone[0]]]) gradient.addColorStop(offset, color);
    return gradient;
  }

  function capsule(g, a, b, radii, tone, center, w, h, alpha = 1) {
    g.save();
    g.globalAlpha = alpha;
    g.beginPath();
    const {p, q, d, r} = capsulePath(g, a, b, radii[0], radii[1], center, w, h);
    let n = d > .5 ? [(p[1] - q[1]) / d, (q[0] - p[0]) / d] : LIGHT_SCREEN;
    if (dot(n, LIGHT_SCREEN) < 0) n = mul(n, -1);
    g.fillStyle = twoTone(g, (p[0] + q[0]) / 2, (p[1] + q[1]) / 2, n, r, tone);
    g.fill();
    g.strokeStyle = OUTLINE;
    g.lineWidth = 1;
    g.stroke();
    g.restore();
  }

  // 여러 캡슐을 한 덩어리로: 2px 윤곽을 먼저 긋고 위를 채워 안쪽 이음선을 지운다.
  function union(g, parts, tone, center, w, h) {
    g.beginPath();
    const ends = parts.map(([a, b, ra, rb]) => capsulePath(g, a, b, ra, rb, center, w, h));
    g.strokeStyle = OUTLINE;
    g.lineWidth = 2;
    g.stroke();
    const xs = ends.flatMap(end => [end.p[0], end.q[0]]), ys = ends.flatMap(end => [end.p[1], end.q[1]]);
    const [minX, maxX, minY, maxY] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const radius = Math.max(maxX - minX, maxY - minY) / 2 + Math.max(...ends.map(end => end.r));
    g.fillStyle = twoTone(g, (minX + maxX) / 2, (minY + maxY) / 2, LIGHT_SCREEN, radius, tone);
    g.fill();
  }

  // 축 셋·반폭 셋의 상자. 카메라를 향한 면만 그리고, 빛을 받는 면은 밝은 색이다.
  function box(g, c, axes, half, tone, center, w, h) {
    g.save();
    g.strokeStyle = OUTLINE;
    g.lineWidth = 1;
    g.lineJoin = 'round';
    for (let k = 0; k < 3; k++) for (const sign of [-1, 1]) {
      const normal = transform(mul(axes[k], sign), [0, 0, 0]), face = add(c, mul(axes[k], sign * half[k]));
      if (dot(normal, sub(transform(face, center), EYE)) >= 0) continue;
      const i = (k + 1) % 3, j = (k + 2) % 3;
      g.beginPath();
      for (const [s, t] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
        const corner = project(add(face, add(mul(axes[i], s * half[i]), mul(axes[j], t * half[j]))), center, w, h);
        g.lineTo(corner[0], corner[1]);
      }
      g.closePath();
      g.fillStyle = dot(unit(normal), LIGHT_CAMERA) > .25 ? tone[0] : tone[1];
      g.fill();
      g.stroke();
    }
    g.restore();
  }

  function headEllipse(g, c, tone, center, w, h) {
    const q = project(c, center, w, h), top = project(add(c, [0, FLESH.head.ry, 0]), center, w, h);
    const rx = FLESH.head.rx * zoom * q[2], ry = Math.max(rx, Math.hypot(top[0] - q[0], top[1] - q[1]));
    g.beginPath();
    g.ellipse(q[0], q[1], rx, ry, Math.atan2(top[1] - q[1], top[0] - q[0]) + Math.PI / 2, 0, Math.PI * 2);
    g.fillStyle = twoTone(g, q[0], q[1], LIGHT_SCREEN, ry, tone);
    g.fill();
    g.strokeStyle = OUTLINE;
    g.lineWidth = 1;
    g.stroke();
  }

  // 축 f, 세계 위에서 f 성분을 뺀 위 u, 옆 s = f × u.
  function axesAlong(forward) {
    const f = unit(forward, [0, 0, 1]);
    const u = unit(sub([0, 1, 0], mul(f, f[1])), [0, 0, 1]);
    return {f, u, s: cross(f, u)};
  }

  // 그릴 항목 목록. z 는 항목 점들의 카메라 깊이 평균이라 큰 것(먼 것)부터 그린다.
  function bodyItems(body, frame, center, w, h) {
    const items = [];
    const push = (points, paint) => items.push({z: points.reduce((sum, p) => sum + transform(p, center)[2], 0) / points.length, points, paint});
    const cap = (a, b, radii, tone, alpha) => push([a, b], g => capsule(g, a, b, radii, tone, center, w, h, alpha));
    const arm = [LIMB_COLORS.arm, SHADOW.arm], core = [LIMB_COLORS.core, SHADOW.core];
    // 몸통: 좌우 두 캡슐(정면에서 넓고 옆에서 얇다)과 어깨선을 한 덩어리로.
    const top = add(body.neck, [0, -FLESH.chestTop, 0]);
    const torso = [
      ...[-1, 1].map(sign => [add(top, mul(body.right, sign * FLESH.chestHalf[0])), add(body.chest, mul(body.right, sign * FLESH.chestHalf[1])), ...FLESH.chest]),
      [body.leftShoulder, body.rightShoulder, FLESH.shoulderBridge, FLESH.shoulderBridge],
    ];
    push(torso.flatMap(part => part.slice(0, 2)), g => union(g, torso, core, center, w, h));
    // 목과 머리는 한 항목이라 머리가 늘 목 위에 그려진다. 머리 중심은 HMD 에서 시선 반대로 물러나 조금 위다.
    const headCenter = add(add(body.head, mul(body.forward, -FLESH.head.back)), [0, FLESH.head.lift, 0]);
    push([headCenter], g => {
      capsule(g, add(body.neck, [0, -FLESH.neckSpan[0], 0]), add(headCenter, [0, -FLESH.neckSpan[1], 0]), FLESH.neck, core, center, w, h);
      headEllipse(g, headCenter, core, center, w, h);
    });
    // HMD 바이저: 시선 축을 따라 놓은 납작한 상자라 어디를 보는지 모양으로 보인다.
    const view = axesAlong(body.forward);
    push([add(body.head, mul(view.f, FLESH.visor[0]))], g => box(g, body.head, [view.f, view.s, view.u], FLESH.visor, VISOR_TONE, center, w, h));
    for (const [shoulder, limb] of [[body.leftShoulder, body.leftArm], [body.rightShoulder, body.rightArm]]) {
      if (!limb) continue;
      cap(shoulder, limb.elbow, FLESH.upperArm, arm);
      cap(limb.elbow, limb.hand, FLESH.foreArm, arm);
      const along = unit(sub(limb.hand, limb.elbow));
      cap(add(limb.hand, mul(along, FLESH.palmReach[0])), add(limb.hand, mul(along, FLESH.palmReach[1])), FLESH.palm, arm);
    }
    const tool = frame.tool_forward;
    if (body.gun && tool && length(tool) > 1e-5) {
      const {f, u, s} = axesAlong(tool);
      const handleAxis = unit(add(mul(u, -Math.cos(GUN.handleTilt)), mul(f, -Math.sin(GUN.handleTilt))));
      const handle = add(body.grip, mul(handleAxis, GUN.handle[0]));
      push([handle], g => box(g, handle, [handleAxis, f, s], GUN.handle, GUN_TONE, center, w, h));
      const gunBody = add(body.grip, mul(f, GUN.bodyAhead));
      push([gunBody], g => box(g, gunBody, [f, u, s], GUN.body, GUN_TONE, center, w, h));
      const [nozzleFrom, nozzleTip] = GUN.nozzle.map(k => add(body.gun, mul(f, k)));
      cap(add(body.grip, mul(f, GUN.rodFrom)), nozzleFrom, [GUN.rodRadius, GUN.rodRadius], GUN_TONE);
      cap(nozzleFrom, nozzleTip, GUN.nozzleRadius, NOZZLE_TONE);
    }
    return items;
  }

  function drawTrustedUpperBody(frame, center, width, height) {
    const body = trustedUpperBody(frame);
    if (!body) return;
    // 몸은 화면 밖 한 장에 불투명으로 그린 뒤 한 번에 얹는다. 뒤의 분사 흔적은 비치고, 팔·몸통이 겹친 안쪽은 비치지 않는다.
    bodyLayer ||= document.createElement('canvas');
    const {width: pixelWidth, height: pixelHeight} = ctx.canvas;
    if (bodyLayer.width !== pixelWidth || bodyLayer.height !== pixelHeight) Object.assign(bodyLayer, {width: pixelWidth, height: pixelHeight});
    const items = bodyItems(body, frame, center, width, height).sort((a, b) => b.z - a.z);
    // 지우기·합성은 몸이 든 화면 상자(장치 px)만 한다. 여유 = 가장 굵은 살 반지름보다 넉넉한 .2 m.
    const shown = items.flatMap(item => item.points).map(p => project(p, center, width, height));
    const scale = pixelWidth / width, pad = .2 * zoom * Math.max(...shown.map(q => q[2])) + 4;
    const [x0, y0] = [0, 1].map(k => Math.max(0, Math.floor((Math.min(...shown.map(q => q[k])) - pad) * scale)));
    const [x1, y1] = [[0, pixelWidth], [1, pixelHeight]].map(([k, limit]) => Math.min(limit, Math.ceil((Math.max(...shown.map(q => q[k])) + pad) * scale)));
    const g = bodyLayer.getContext('2d');
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.clearRect(...(lastBodyRect || [0, 0, pixelWidth, pixelHeight]));
    lastBodyRect = x1 > x0 && y1 > y0 ? [x0, y0, x1 - x0, y1 - y0] : null;
    if (!lastBodyRect) return;
    g.setTransform(ctx.getTransform());
    for (const item of items) item.paint(g);
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalAlpha = BODY_LAYER_ALPHA;
    ctx.drawImage(bodyLayer, ...lastBodyRect, ...lastBodyRect);
    ctx.restore();
    // 실측 추적기(건·왼손)는 맨 위에 고리로 그려 IK 손과의 차이가 보이게 둔다.
    for (const tracker of [body.gun, body.leftHand]) {
      if (!tracker) continue;
      const shown = project(tracker, center, width, height);
      ctx.strokeStyle = LIMB_COLORS.arm;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(shown[0], shown[1], 8 * shown[2], 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  // 시선과 분사 축은 바이저·건 모양이 보여 준다. 여기서는 노즐 끝에서 착탄점까지의 분사선만 긋는다.
  function drawOrientationGuides(frame, center, width, height) {
    const gun = frame?.tracked?.right_hand;
    if (!gun || !frame.tool_forward || !frame.ray_point) return;
    const axis = normalize(frame.tool_forward);
    const tip = addVector(gun, scaleVector(axis, GUN.nozzle[1]));
    arrow(tip, frame.ray_point, 'rgba(250,204,21,.82)', center, width, height);
    if (!document.querySelector('.v2-shell')) {
      const shown = project(addVector(tip, scaleVector(axis, .1)), center, width, height);
      ctx.fillStyle = '#fde68a';
      ctx.font = '600 10px Pretendard, "Noto Sans KR", system-ui';
      ctx.fillText(`분사축 편차 ${angleDegrees(frame.tool_forward, subtract(frame.ray_point, gun)).toFixed(0)}°`, shown[0] + 7, shown[1] - 6);
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
      byId('poseZoneTitle').textContent = '현재 Zone 작업 유사도';
      byId('poseZoneScore').textContent = '—%';
      byId('poseActionTitle').textContent = '현재 Zone 작업 개선 안내';
      metrics.replaceChildren(Object.assign(document.createElement('small'), {textContent: '현재 Zone 평가 자료가 없습니다.'}));
      renderCorrections([], '분사 Zone에 진입하면 해당 Zone의 작업 개선 안내를 표시합니다.');
      return null;
    }
    byId('poseZoneTitle').textContent = `Zone ${zoneLabel(zoneEvaluation.zone_id)} 작업 유사도`;
    byId('poseZoneScore').textContent = zoneEvaluation.similarity_percent == null ? '—%' : `${zoneEvaluation.similarity_percent}%`;
    byId('poseActionTitle').textContent = `Zone ${zoneLabel(zoneEvaluation.zone_id)} 작업 개선 안내`;
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

  // 자세 유사도는 분사마다 팔 관절 엔진으로 채점해 수 초 걸린다. 모션과 따로 받고, 채팅도 같은 객체에서 읽는다.
  const ARM_POSE_UNAVAILABLE = {
    process_not_supported: '블라스팅 기록은 자세 유사도를 채점하지 않습니다.',
    no_joint_sequences: '이 기록에는 채점할 분사 구간이 없습니다.',
    request_failed: '자세 채점 결과를 불러오지 못했습니다.',
  };
  function loadArmPose(payload) {
    const query = new URLSearchParams({worker_id: payload.worker?.worker_id, process: payload.process,
      mode_id: payload.mode_id, session_id: payload.selected_session_id});
    fetch('/motion-viewer/api/arm-pose?' + query)
      .then(response => response.ok ? response.json() : Promise.reject(new Error(String(response.status))))
      .then(result => {
        payload.arm_pose_evaluation = result.arm_pose_evaluation;
        (payload.query_context_signatures ||= {}).arm_pose = result.signature;
      })
      .catch(() => { payload.arm_pose_evaluation = {status: 'unavailable', reason_code: 'request_failed'}; })
      .finally(() => { if (data === payload) draw(); });
  }

  function renderArmPose(seconds) {
    const evaluation = data?.arm_pose_evaluation;
    const current = byId('armPoseCurrent');
    if (evaluation?.status !== 'available') {
      byId('armPoseScore').textContent = '—%';
      byId('armPoseNote').textContent = evaluation
        ? ARM_POSE_UNAVAILABLE[evaluation.reason_code] || '이 기록은 자세 유사도를 채점할 수 없습니다.'
        : '분사마다 팔 관절 자세를 채점하는 중입니다.';
      current.hidden = true;
      return;
    }
    byId('armPoseScore').textContent = `${evaluation.average_similarity_percent}%`;
    byId('armPoseNote').textContent = `분사 ${evaluation.evaluated_sequence_count}회 평균 · 숙련자 팔 관절 기준`;
    current.hidden = false;
    // 재생 위치 직전에 시작한 분사. 분사가 끝난 이동 구간에서는 그 분사를 '직전'으로 보인다.
    const sequence = (evaluation.sequences || []).filter(item => item.start_seconds != null && item.start_seconds <= seconds).at(-1);
    if (!sequence) {
      byId('armPoseSequence').textContent = '분사 전';
      byId('armPoseSequenceScore').textContent = '';
      byId('armPoseCorrection').textContent = '분사가 시작되면 그 분사의 자세 유사도를 표시합니다.';
      return;
    }
    const live = sequence.end_seconds == null || seconds <= sequence.end_seconds;
    const limited = sequence.confidence === 'limited';
    byId('armPoseSequence').textContent = `${live ? '현재' : '직전'} 분사 ${sequence.sequence_id}`;
    byId('armPoseSequenceScore').textContent = `${sequence.similarity_percent}%`;
    byId('armPoseCorrection').textContent = [sequence.correction || (limited ? '' : '교정 관절 결과는 없습니다.'),
      limited ? '입력 품질 제한이 있어 참고용입니다.' : ''].filter(Boolean).join(' ');
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

  // 오른쪽 아래 모서리의 작은 좌표축. 장면의 transform 을 그대로 써서 카메라 회전과 함께 돈다
  function drawAxisGizmo(width, height) {
    // 좁은 화면에서는 우하단에 뜨는 질문하기 버튼 위로 올린다
    const ox = width - 40, oy = height - (width < 520 ? 96 : 40), length = 22;
    ctx.save();
    ctx.lineWidth = 2; ctx.lineCap = 'round'; ctx.font = '700 11px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    for (const {axis, dx, dy, color} of window.MotionViewerCamera.axisGizmo(transform)) {
      ctx.strokeStyle = color; ctx.fillStyle = color;
      ctx.beginPath(); ctx.moveTo(ox, oy); ctx.lineTo(ox + dx * length, oy + dy * length); ctx.stroke();
      ctx.fillText(axis.toUpperCase(), ox + dx * (length + 9), oy + dy * (length + 9));
    }
    ctx.fillStyle = '#c9d3e6'; ctx.beginPath(); ctx.arc(ox, oy, 2.5, 0, Math.PI * 2); ctx.fill();
    ctx.restore();
  }

  function drawPoseOverlay({frame, sourceIndex, center, w, h, playheadSeconds, durationSeconds}) {
    drawTrustedUpperBody(frame, center, w, h);
    drawOrientationGuides(frame, center, w, h);
    drawAxisGizmo(w, h);
    updateRealtimeHud(frame, sourceIndex);
    renderPoseSummary(frame);
    renderArmPose(Number(playheadSeconds) || 0);
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
    loadArmPose(event.detail);
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
  for (const axis of ['x', 'y', 'z']) byId(`cameraAxis${axis.toUpperCase()}`).style.color = window.MotionViewerCamera.AXIS_COLORS[axis];
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
    if (window.MotionViewerCamera.isTyping(event.target)) return;
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
