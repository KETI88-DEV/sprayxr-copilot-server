// 3D 뷰. compare()가 낸 `view` 블록만 그린다 — 지표를 브라우저에서 다시 계산하지
// 않고, 화면에 보이는 숫자는 서버의 표시 문자열(`miss_display` 등)만 쓴다(app.js 머리말).
//
// 무엇을 그리나(서버 응답 계약):
//   - 뼈(`layers.query.joints`/`bones`): 시뮬레이터 아바타 추정. 조용한 저채도 색·반투명.
//   - 대표 프레임 실측 마커(`fit.anchors[].measured_marker`): 불투명 구체 — 실측.
//   - 패스 평균 마커(`corrections[].from`): 속 빈 고리. 대표 프레임 마커와 회색 실선으로 잇는다.
//   - 보정 오프셋 선(joint_position→predicted_marker)과 어긋남 점선(predicted→measured).
//   - 교정 화살표: `from_frame`→`to_frame`(없으면 `from`→`to`). 번호 배지는 `n`.
//   - 검증된 벽면 메시(`mesh.available`일 때만). 양면(DoubleSide) 반투명.
//
// 강조는 `view.weights`에서만 나온다 — 부위 이름으로 색·모양 규칙을 분기하지 않는다.
//
// three.js는 cdnjs에서 버전 고정으로 이 스크립트보다 먼저 실린다(index.html).
// 못 실리면 `createSkeletonView`가 아무것도 던지지 않고 안내만 남긴다.

const POINT_LABELS = { right: "오른손", head: "머리", left: "왼손", hit: "착탄점" };

// 층별 기본색. σ 3단(ok/warn/bad)과 겹치지 않는 색. style.css의 --layer-*와 같은 값.
const LAYER_COLOR = { query: 0x5b9bff, reference: 0xa684e0, corrected: 0x34c2c2 };
const MESH_COLOR = 0x9aa5b1;
// 추정 뼈는 실측 마커(층 색 100%)보다 한 단계 조용해야 한다 — 채도를 뺀 밝은 회청색.
const BONE_COLOR = 0xc8d0dc;
const LINK_COLOR = 0x8a93a3;

const SIGMA_OK = 0x2f855a;
const SIGMA_WARN = 0xb7791f;
const SIGMA_BAD = 0xc53030;
const SIGMA_UNKNOWN = 0x718096;

// 크기 상수(m). 화살표 굵기는 교정 크기와 분리한다 — 크기는 길이와 ③의 수치로 전달한다.
const MEASURED_MARKER_RADIUS = 0.035;
const RING_OUTER = 0.05;
const RING_INNER = 0.04;
const HEAD_SPHERE_RADIUS = 0.09;
const BONE_RADIUS = 0.005;
const ARROW_SHAFT_RADIUS = 0.006;
const ARROW_HEAD_RADIUS = 0.016;
const HIT_MARKER_RADIUS = 0.018;

const BADGE_MIN_PX = 26;
const FRAMING_TARGET = { pose: 0.75, tracks: 0.75, wall: 0.9 };

/** 강조 함수 — 색·굵기·마커·라벨 전부 이 하나의 함수다. */
function emphasis(weight) {
  return {
    opacity: 0.25 + 0.75 * weight,
    width: 1 + 2 * weight,
    showMarker: weight > 0,
    showLabel: weight > 0,
  };
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function scaleColor(hex, factor) {
  const r = Math.round(((hex >> 16) & 255) * factor);
  const g = Math.round(((hex >> 8) & 255) * factor);
  const b = Math.round((hex & 255) * factor);
  return (r << 16) | (g << 8) | b;
}

function sigmaColor(sigma) {
  // 근거가 없으면 회색이다 — 없는 σ를 0σ(초록)로 그리면 "근거가 강하다"는 거짓 인상을 준다.
  if (typeof sigma !== "number") return SIGMA_UNKNOWN;
  if (sigma <= 1) return SIGMA_OK;
  if (sigma <= 2) return SIGMA_WARN;
  return SIGMA_BAD;
}

function isVec3(value) {
  return Array.isArray(value) && value.length === 3 && value.every((x) => typeof x === "number");
}

function disposeObject3D(object) {
  object.traverse((child) => {
    if (child.geometry) child.geometry.dispose();
    if (child.material) {
      if (Array.isArray(child.material)) child.material.forEach((m) => m.dispose());
      else child.material.dispose();
    }
  });
}

function buildDom(container) {
  container.replaceChildren();
  container.classList.add("view3d");

  const toolbar = document.createElement("div");
  toolbar.className = "view3d-toolbar";
  const title = document.createElement("h2");
  title.className = "view3d-title";
  title.textContent = "3D 뷰";
  const controls = document.createElement("div");
  controls.className = "view3d-controls";
  controls.innerHTML = `
    <label><input id="view3dLayerQuery" type="checkbox" checked> 질의(스켈레톤)</label>
    <label><input id="view3dLayerReference" type="checkbox" checked> 기준</label>
    <label><input id="view3dLayerCorrected" type="checkbox"> 교정 후(예상)</label>
    <label><input id="view3dLayerWall" type="checkbox" checked> 벽면</label>
    <span class="view3d-controls-sep" aria-hidden="true"></span>
    <button id="view3dPresetFit" type="button">자세 맞춤</button>
    <button id="view3dPresetFront" type="button">벽 정면</button>
    <button id="view3dPresetSide" type="button">벽 측면</button>
    <button id="view3dPresetTracks" type="button">궤적 포함</button>
    <button id="view3dPresetWall" type="button">벽 포함</button>
  `;
  toolbar.append(title, controls);

  const host = document.createElement("div");
  host.className = "view3d-host";
  const canvas = document.createElement("canvas");
  canvas.className = "view3d-canvas";
  const labelsHost = document.createElement("div");
  labelsHost.className = "view3d-labels";
  host.append(canvas, labelsHost);

  const legend = document.createElement("div");
  legend.className = "view3d-legend";

  const notes = document.createElement("ul");
  notes.className = "view3d-notes";
  const wallNote = document.createElement("p");
  wallNote.className = "view3d-wall-note";
  const errors = document.createElement("p");
  errors.className = "view3d-errors";
  const note = document.createElement("p");
  note.className = "view3d-note";
  const correctedNote = document.createElement("p");
  correctedNote.className = "view3d-note view3d-corrected-note";

  container.append(toolbar, host, legend, notes, wallNote, errors, note, correctedNote);
  return { canvas, host, labelsHost, legend, notes, wallNote, errors, note, correctedNote, controls, toolbar };
}

/** `note`에 문구와 상태 톤을 함께 건다. 빈 문구는 CSS `:empty`가 숨긴다. */
function setNote(el, text, tone) {
  el.textContent = text ?? "";
  if (tone) el.dataset.tone = tone;
  else el.removeAttribute("data-tone");
}

function showUnavailable(container, message) {
  const dom = buildDom(container);
  setNote(dom.note, message, "error");
  dom.toolbar.hidden = true;
  dom.host.hidden = true;
  dom.legend.hidden = true;
}

function legendItem(swatchClass, text) {
  const item = document.createElement("span");
  item.className = "legend-item";
  if (swatchClass) {
    const swatch = document.createElement("i");
    swatch.className = `swatch ${swatchClass}`;
    item.append(swatch);
  }
  item.append(document.createTextNode(text));
  return item;
}

function renderLegend(legend, view) {
  legend.replaceChildren();
  const title = document.createElement("span");
  title.className = "legend-title";
  title.textContent = "화살표 색 = 근거 σ";
  legend.append(
    title,
    legendItem("ok", "≤1σ"),
    legendItem("warn", "≤2σ"),
    legendItem("bad", ">2σ"),
  );
  const fit = view?.fit ?? null;
  if (fit) {
    if (typeof fit.offset_line_label === "string") {
      const item = legendItem("line-solid", fit.offset_line_label);
      item.dataset.role = "offset-line";
      legend.append(item);
    }
    if (typeof fit.residual_label === "string") {
      const item = legendItem("line-dashed", `어긋남 = ${fit.residual_label}`);
      item.dataset.role = "miss-line";
      legend.append(item);
    }
  }
  for (const text of view?.legend_notes ?? []) {
    if (typeof text === "string") legend.append(legendItem(null, text));
  }
}

function mount(THREE, container) {
  const dom = buildDom(container);
  renderLegend(dom.legend, null);
  setNote(dom.note, "패스를 고르고 평가를 누르면 여기에 뜹니다.", "idle");

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(50, 1, 0.02, 200);
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ canvas: dom.canvas, antialias: true, preserveDrawingBuffer: true });
  } catch (err) {
    showUnavailable(container, "이 브라우저에서 WebGL을 쓸 수 없어 3D 뷰를 그리지 않습니다.");
    return { render() {}, clear() {}, loading() {} };
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  scene.background = new THREE.Color(0x070a0f);

  scene.add(new THREE.AmbientLight(0xffffff, 0.7));
  const key = new THREE.DirectionalLight(0xffffff, 0.9);
  key.position.set(2, 3, 2);
  scene.add(key);

  const layerGroups = {
    query: new THREE.Group(),
    reference: new THREE.Group(),
    corrected: new THREE.Group(),
  };
  layerGroups.corrected.visible = false;
  for (const group of Object.values(layerGroups)) scene.add(group);
  const correctionsGroup = new THREE.Group();
  scene.add(correctionsGroup);
  let meshObject = null;
  // 카메라를 향해 도는 고리(패스 평균 마커).
  const billboards = [];

  const labelRegistry = [];
  function clearLabels() {
    for (const entry of labelRegistry) entry.el.remove();
    labelRegistry.length = 0;
  }
  function registerLabel(position, text, extraClass, extra) {
    const el = document.createElement("div");
    el.className = extraClass ? `view3d-label ${extraClass}` : "view3d-label";
    el.textContent = text;
    dom.labelsHost.appendChild(el);
    const entry = { position: position.clone(), el, ...(extra ?? {}) };
    labelRegistry.push(entry);
    return entry;
  }

  // 카메라 상태(구면좌표) — 드래그 회전·휠 확대가 이 값만 바꾼다.
  const camState = { target: new THREE.Vector3(0, 1, 0), radius: 3, theta: Math.PI / 4, phi: Math.PI / 2.5 };
  // 자동 맞춤 상태. 사용자가 드래그·휠로 카메라를 만지면 창 크기가 바뀌어도 다시 맞추지 않는다.
  const framing = { mode: "pose", auto: true, ratio: null };
  // 마지막으로 그린 view. resize()가 첫 호출부터 읽으므로 그보다 먼저 선언한다.
  let lastView = null;

  function applyCamera() {
    const sinPhi = Math.sin(camState.phi);
    camera.position.set(
      camState.target.x + camState.radius * sinPhi * Math.sin(camState.theta),
      camState.target.y + camState.radius * Math.cos(camState.phi),
      camState.target.z + camState.radius * sinPhi * Math.cos(camState.theta),
    );
    camera.up.set(0, 1, 0);
    camera.lookAt(camState.target);
    camera.updateMatrixWorld();
    container.__view3dCameraDebug = { theta: camState.theta, phi: camState.phi, radius: camState.radius };
  }
  applyCamera();

  let dragging = false;
  let lastX = 0;
  let lastY = 0;
  dom.canvas.style.touchAction = "none";
  dom.canvas.addEventListener("pointerdown", (event) => {
    dragging = true;
    lastX = event.clientX;
    lastY = event.clientY;
    dom.canvas.setPointerCapture(event.pointerId);
  });
  dom.canvas.addEventListener("pointerup", () => { dragging = false; });
  dom.canvas.addEventListener("pointercancel", () => { dragging = false; });
  dom.canvas.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    const dx = event.clientX - lastX;
    const dy = event.clientY - lastY;
    lastX = event.clientX;
    lastY = event.clientY;
    camState.theta -= dx * 0.008;
    camState.phi = clamp(camState.phi - dy * 0.008, 0.05, Math.PI - 0.05);
    framing.auto = false;
    applyCamera();
  });
  dom.canvas.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      const factor = event.deltaY > 0 ? 1.1 : 1 / 1.1;
      camState.radius = clamp(camState.radius * factor, 0.05, 60);
      framing.auto = false;
      applyCamera();
    },
    { passive: false },
  );

  function hostSize() {
    return { w: dom.host.clientWidth || 1, h: dom.host.clientHeight || 1 };
  }

  function resize() {
    const { w, h } = hostSize();
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    if (lastView && framing.auto) frameTo(framing.mode);
  }
  const resizeObserver = new ResizeObserver(() => resize());
  resizeObserver.observe(dom.host);
  resize();

  function toPixel(vector) {
    const { w, h } = hostSize();
    const p = vector.clone().project(camera);
    return { x: ((p.x + 1) / 2) * w, y: ((1 - p.y) / 2) * h, z: p.z };
  }

  /** 월드 반경 r인 구(원추 밑면)가 화면에서 차지하는 지름(px). 카메라 오른쪽 축으로 잰다. */
  function screenDiameter(center, radius) {
    const right = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0).normalize();
    const a = toPixel(center);
    const b = toPixel(center.clone().addScaledVector(right, radius));
    return 2 * Math.hypot(b.x - a.x, b.y - a.y);
  }

  const debugLive = {
    badgeCenters: [],
    arrowConeScreenDiameters: [],
    markerScreenDiameters: {},
    handMarkerScreenDiameter: null,
    arrowStarts: [],
  };
  let coneRegistry = [];
  let markerRegistry = [];

  function layoutLabels() {
    // 1) 글자 라벨(miss 거리 등)을 먼저 놓고 그 사각형을 장애물로 모은다.
    // 2) 번호 배지는 앞 배지(26px 안)와 글자 라벨 사각형을 피해, 자기 화살표의 화면
    //    방향으로 26px씩 민다. 배지가 거리 라벨을 덮으면 숫자를 읽을 수 없다.
    const obstacles = [];
    const visible = (entry) => {
      const projected = entry.position.clone().project(camera);
      const shown = projected.z <= 1 && projected.z >= -1;
      entry.el.style.display = shown ? "" : "none";
      return shown;
    };
    for (const entry of labelRegistry) {
      if (entry.kind === "badge" || !visible(entry)) continue;
      const px = toPixel(entry.position);
      entry.el.style.left = `${px.x}px`;
      entry.el.style.top = `${px.y}px`;
      const w = entry.el.offsetWidth;
      const h = entry.el.offsetHeight;
      // .view3d-label의 transform: translate(-50%, -140%)
      obstacles.push({ x0: px.x - w / 2, x1: px.x + w / 2, y0: px.y - 1.4 * h, y1: px.y - 0.4 * h });
    }
    const hitsObstacle = (x, y) => obstacles.some((r) => {
      const nx = clamp(x, r.x0, r.x1);
      const ny = clamp(y, r.y0, r.y1);
      return Math.hypot(nx - x, ny - y) < 14;
    });
    const placedBadges = [];
    let obstacleOverlaps = 0;
    for (const entry of labelRegistry) {
      if (entry.kind !== "badge" || !visible(entry)) continue;
      const px = toPixel(entry.position);
      let x = px.x;
      let y = px.y;
      const a = toPixel(entry.from);
      const b = toPixel(entry.to);
      let dx = b.x - a.x;
      let dy = b.y - a.y;
      const len = Math.hypot(dx, dy);
      if (len < 1e-6) { dx = 0; dy = -1; } else { dx /= len; dy /= len; }
      for (let step = 0; step < 10; step += 1) {
        const clash = placedBadges.some((p) => Math.hypot(p.x - x, p.y - y) < BADGE_MIN_PX) || hitsObstacle(x, y);
        if (!clash) break;
        x += dx * BADGE_MIN_PX;
        y += dy * BADGE_MIN_PX;
      }
      if (hitsObstacle(x, y)) obstacleOverlaps += 1;
      placedBadges.push({ n: entry.n, x, y });
      entry.el.style.left = `${x}px`;
      entry.el.style.top = `${y}px`;
    }
    debugLive.badgeObstacleOverlaps = obstacleOverlaps;
    debugLive.badgeCenters = placedBadges.map((p) => ({ n: p.n, x: Math.round(p.x * 10) / 10, y: Math.round(p.y * 10) / 10 }));
    debugLive.arrowConeScreenDiameters = coneRegistry.map((c) => screenDiameter(c.center, c.radius));
    const diameters = {};
    for (const m of markerRegistry) diameters[m.point] = screenDiameter(m.center, MEASURED_MARKER_RADIUS);
    debugLive.markerScreenDiameters = diameters;
    debugLive.handMarkerScreenDiameter = typeof diameters.right === "number" ? diameters.right : null;
  }

  function frame() {
    requestAnimationFrame(frame);
    for (const ring of billboards) ring.quaternion.copy(camera.quaternion);
    renderer.render(scene, camera);
    layoutLabels();
  }
  requestAnimationFrame(frame);

  function buildLayerGroup(layerData, weights, layerColor, labelPointNames) {
    const group = new THREE.Group();
    for (const name of Object.keys(layerData ?? {})) {
      if (name === "joints" || name === "bones") continue;
      const weight = typeof weights?.[name] === "number" ? weights[name] : 0;
      const emph = emphasis(weight);
      const points = (layerData[name] ?? [])
        .filter((p) => isVec3(p))
        .map((p) => new THREE.Vector3(p[0], p[1], p[2]));
      if (points.length === 0) continue;

      const lineMaterial = new THREE.LineBasicMaterial({
        color: layerColor,
        transparent: true,
        opacity: emph.opacity,
      });
      group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), lineMaterial));

      if (emph.showMarker) {
        const markerGeometry = new THREE.SphereGeometry(0.004 * emph.width, 10, 6);
        const markerMaterial = new THREE.MeshBasicMaterial({
          color: layerColor,
          transparent: true,
          opacity: emph.opacity,
        });
        for (const p of points) {
          const marker = new THREE.Mesh(markerGeometry, markerMaterial);
          marker.position.copy(p);
          group.add(marker);
        }
        if (emph.showLabel && labelPointNames) {
          registerLabel(points[points.length - 1], POINT_LABELS[name] ?? name, "view3d-label-point");
        }
      }
    }
    return group;
  }

  /** 추정 뼈. 강조 대상이 아니라 문맥이다 — weights로 강조를 따지지 않고 저채도 한 색. */
  function buildSkeletonGroup(joints, bones, faded) {
    const group = new THREE.Group();
    if (!joints || !bones || Object.keys(joints).length === 0) return group;
    const opacity = faded ? 0.35 : 0.6;
    // 실선(LineBasicMaterial)은 WebGL에서 굵기를 키울 수 없어 머리카락처럼 보인다 —
    // 얇은 실린더로 실제 두께를 준다. 조명과 무관한 색을 위해 MeshBasicMaterial.
    const boneMaterial = new THREE.MeshBasicMaterial({ color: BONE_COLOR, transparent: true, opacity, depthWrite: false });
    for (const [parent, child] of bones) {
      const a = joints[parent];
      const b = joints[child];
      if (!isVec3(a) || !isVec3(b)) continue;
      const start = new THREE.Vector3(...a);
      const end = new THREE.Vector3(...b);
      const dir = new THREE.Vector3().subVectors(end, start);
      const length = dir.length();
      if (length < 1e-6) continue;
      dir.normalize();
      const bone = new THREE.Mesh(new THREE.CylinderGeometry(BONE_RADIUS, BONE_RADIUS, length, 6), boneMaterial);
      bone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
      bone.position.copy(start).addScaledVector(dir, length / 2);
      bone.userData.kind = "bone";
      bone.userData.name = `${parent}>${child}`;
      bone.userData.length = length;
      group.add(bone);
    }
    const jointGeometry = new THREE.SphereGeometry(BONE_RADIUS * 2, 8, 6);
    for (const position of Object.values(joints)) {
      if (!isVec3(position)) continue;
      const marker = new THREE.Mesh(jointGeometry, boneMaterial);
      marker.position.set(position[0], position[1], position[2]);
      group.add(marker);
    }
    if (isVec3(joints.Head)) {
      const headMaterial = new THREE.MeshBasicMaterial({ color: BONE_COLOR, transparent: true, opacity: opacity * 0.4, depthWrite: false });
      const head = new THREE.Mesh(new THREE.SphereGeometry(HEAD_SPHERE_RADIUS, 20, 14), headMaterial);
      head.position.set(...joints.Head);
      group.add(head);
    }
    return group;
  }

  /** 대표 프레임 실측 마커(구체)·보정 오프셋 선·어긋남 점선·miss 라벨. */
  function buildAnchorGroup(fit) {
    const group = new THREE.Group();
    markerRegistry = [];
    const counts = { measuredSpheres: 0, offsetLines: 0, missLines: 0, missLabels: [] };
    if (!fit) return { group, counts };
    const holdout = fit.mounts === "pass_fit_holdout";
    const markerMaterial = new THREE.MeshStandardMaterial({ color: LAYER_COLOR.query, roughness: 0.5 });
    const markerGeometry = new THREE.SphereGeometry(MEASURED_MARKER_RADIUS, 20, 14);
    const offsetMaterial = new THREE.LineBasicMaterial({ color: BONE_COLOR, transparent: true, opacity: 0.5 });
    const missMaterial = new THREE.LineDashedMaterial({
      color: 0xe8ecf3,
      dashSize: holdout ? 0.01 : 0.03,
      gapSize: holdout ? 0.02 : 0.015,
    });
    for (const anchor of fit.anchors ?? []) {
      if (isVec3(anchor.measured_marker)) {
        const marker = new THREE.Mesh(markerGeometry, markerMaterial);
        marker.position.set(...anchor.measured_marker);
        marker.userData.kind = "measured-marker";
        group.add(marker);
        counts.measuredSpheres += 1;
        markerRegistry.push({ point: anchor.point, center: marker.position.clone() });
      }
      if (isVec3(anchor.joint_position) && isVec3(anchor.predicted_marker)) {
        const geometry = new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(...anchor.joint_position),
          new THREE.Vector3(...anchor.predicted_marker),
        ]);
        group.add(new THREE.Line(geometry, offsetMaterial));
        counts.offsetLines += 1;
      }
      if (isVec3(anchor.predicted_marker) && isVec3(anchor.measured_marker)) {
        const a = new THREE.Vector3(...anchor.predicted_marker);
        const b = new THREE.Vector3(...anchor.measured_marker);
        const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints([a, b]), missMaterial);
        line.computeLineDistances();
        group.add(line);
        counts.missLines += 1;
        if (typeof anchor.miss_display === "string") {
          const mid = a.clone().add(b).multiplyScalar(0.5);
          registerLabel(mid, anchor.miss_display, "view3d-label-miss");
          counts.missLabels.push(anchor.miss_display);
        }
      }
    }
    return { group, counts };
  }

  /** 패스 평균 마커(속 빈 고리) — 교정의 `from`이 패스 평균 위치다. 대표 프레임 마커와 잇는다. */
  function buildPassMeanGroup(corrections, fit) {
    const group = new THREE.Group();
    billboards.length = 0;
    let rings = 0;
    const seen = new Set();
    const ringMaterial = new THREE.MeshBasicMaterial({ color: scaleColor(LAYER_COLOR.query, 0.6), side: THREE.DoubleSide });
    const linkMaterial = new THREE.LineBasicMaterial({ color: LINK_COLOR });
    const measuredByPoint = new Map((fit?.anchors ?? []).map((a) => [a.point, a.measured_marker]));
    for (const correction of corrections) {
      if (!isVec3(correction.from) || seen.has(correction.point)) continue;
      seen.add(correction.point);
      const ring = new THREE.Mesh(new THREE.RingGeometry(RING_INNER, RING_OUTER, 32), ringMaterial);
      ring.position.set(...correction.from);
      ring.userData.kind = "pass-mean-ring";
      group.add(ring);
      billboards.push(ring);
      rings += 1;
      const measured = measuredByPoint.get(correction.point);
      if (isVec3(measured)) {
        const geometry = new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(...correction.from),
          new THREE.Vector3(...measured),
        ]);
        group.add(new THREE.Line(geometry, linkMaterial));
      }
    }
    return { group, rings };
  }

  function buildArrow(from, to, color) {
    const start = new THREE.Vector3(...from);
    const end = new THREE.Vector3(...to);
    const dir = new THREE.Vector3().subVectors(end, start);
    const length = dir.length();
    if (length < 1e-6) return null;
    dir.normalize();

    const headLength = Math.min(0.04, 0.35 * length);
    const shaftLength = Math.max(length - headLength, 0);

    const material = new THREE.MeshStandardMaterial({ color, roughness: 0.45, metalness: 0.05 });
    const quat = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);

    const group = new THREE.Group();
    if (shaftLength > 1e-6) {
      const shaft = new THREE.Mesh(new THREE.CylinderGeometry(ARROW_SHAFT_RADIUS, ARROW_SHAFT_RADIUS, shaftLength, 10), material);
      shaft.quaternion.copy(quat);
      shaft.position.copy(start).addScaledVector(dir, shaftLength / 2);
      group.add(shaft);
    }
    const cone = new THREE.Mesh(new THREE.ConeGeometry(ARROW_HEAD_RADIUS, headLength, 16), material);
    cone.quaternion.copy(quat);
    cone.position.copy(start).addScaledVector(dir, shaftLength + headLength / 2);
    group.add(cone);
    const coneBase = start.clone().addScaledVector(dir, shaftLength);
    return { group, coneBase };
  }

  function buildMesh(meshBlock) {
    if (!meshBlock || meshBlock.available !== true) return null;
    const vertices = meshBlock.vertices ?? [];
    const triangles = meshBlock.triangles ?? [];
    if (vertices.length === 0 || triangles.length === 0) return null;
    const positions = new Float32Array(vertices.length * 3);
    vertices.forEach((v, i) => {
      positions[i * 3] = v[0];
      positions[i * 3 + 1] = v[1];
      positions[i * 3 + 2] = v[2];
    });
    const indices = new Uint32Array(triangles.length * 3);
    triangles.forEach((t, i) => {
      indices[i * 3] = t[0];
      indices[i * 3 + 1] = t[1];
      indices[i * 3 + 2] = t[2];
    });
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    geometry.computeVertexNormals();
    // 정렬 변환이 반사(det<0)라 삼각형 감김이 뒤집혀 있다 — 양면으로 그려야 법선 방향과
    // 무관하게 보인다. 반투명 벽이 뒤의 몸·마커를 깊이로 가리지 않게 depthWrite를 끈다.
    const material = new THREE.MeshStandardMaterial({
      color: MESH_COLOR,
      transparent: true,
      opacity: 0.32,
      side: THREE.DoubleSide,
      roughness: 0.9,
      depthWrite: false,
    });
    const group = new THREE.Group();
    const surface = new THREE.Mesh(geometry, material);
    group.add(surface);
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(geometry, 25),
      new THREE.LineBasicMaterial({ color: 0x6b7686, transparent: true, opacity: 0.7 }),
    );
    group.add(edges);
    group.userData.triangleCount = triangles.length;
    return group;
  }

  /** 벽면 줄. 문구는 서버(reason_text·handedness_note)의 것을 그대로 쓴다. */
  function wallNoteText(meshBlock) {
    if (meshBlock === null || meshBlock === undefined) {
      return "이 패스는 레벨 정보가 없어 벽면을 그리지 않습니다.";
    }
    if (meshBlock.available !== true) {
      return meshBlock.reason_text ?? "이 레벨은 벽면을 그리지 않습니다.";
    }
    if (typeof meshBlock.handedness_note === "string") {
      const evidence = meshBlock.handedness_evidence;
      const pair = (entry) => (entry ? `P ${String(entry.P)} / 거울 ${String(entry.mirror)}` : "—");
      const detail = evidence
        ? ` 건 가림 ${pair(evidence.gun_occlusion)} · HMD 가림 ${pair(evidence.hmd_occlusion)}`
        : "";
      return `${meshBlock.handedness_note}${detail}`;
    }
    return "";
  }

  /** 맞춤 대상 점. pose: 스켈레톤·실측/예측 마커·교정 화살표·대표 프레임 착탄점.
   * 궤적·기준층·교정후층·메시는 뺀다(벽은 자세보다 훨씬 커서 넣으면 자세가 쪼그라든다). */
  function framingPoints(view, mode) {
    const points = [];
    const push = (p) => { if (isVec3(p)) points.push(p); };
    const joints = view.layers?.query?.joints ?? {};
    const hasSkeleton = Object.keys(joints).length > 0;
    if (hasSkeleton) {
      for (const p of Object.values(joints)) push(p);
      for (const anchor of view.fit?.anchors ?? []) {
        push(anchor.measured_marker);
        push(anchor.predicted_marker);
      }
    } else {
      for (const name of ["head", "right", "left"]) for (const p of view.layers?.query?.[name] ?? []) push(p);
    }
    for (const correction of view.corrections ?? []) {
      if (isVec3(correction.from_frame) && isVec3(correction.to_frame)) {
        push(correction.from_frame);
        push(correction.to_frame);
      } else {
        push(correction.from);
        push(correction.to);
      }
    }
    push(view.composite?.hit);
    if (mode === "tracks" || mode === "wall") {
      for (const name of ["head", "right", "left"]) for (const p of view.layers?.query?.[name] ?? []) push(p);
    }
    if (mode === "wall") {
      for (const p of view.layers?.query?.hit ?? []) push(p);
      if (view.mesh?.available === true) for (const p of view.mesh.vertices ?? []) push(p);
    }
    return points;
  }

  function boundsOf(points) {
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    for (const p of points) {
      for (let i = 0; i < 3; i += 1) {
        if (p[i] < min[i]) min[i] = p[i];
        if (p[i] > max[i]) max[i] = p[i];
      }
    }
    return { min, max };
  }

  /** 카메라 거리 d에서 AABB 8꼭짓점을 투영했을 때 max(높이비, 폭비). 뒤로 넘어가면 Infinity. */
  function projectedRatio(corners, target, offsetDir, distance) {
    const position = target.clone().addScaledVector(offsetDir, distance);
    const forward = offsetDir.clone().negate();
    const right = new THREE.Vector3().crossVectors(forward, new THREE.Vector3(0, 1, 0));
    if (right.lengthSq() < 1e-12) right.set(1, 0, 0);
    right.normalize();
    const up = new THREE.Vector3().crossVectors(right, forward).normalize();
    const tanV = Math.tan((camera.fov * Math.PI) / 360);
    const tanH = tanV * camera.aspect;
    let xmin = Infinity; let xmax = -Infinity; let ymin = Infinity; let ymax = -Infinity;
    for (const corner of corners) {
      const v = corner.clone().sub(position);
      const z = v.dot(forward);
      if (z <= camera.near) return Infinity;
      const x = v.dot(right) / (z * tanH);
      const y = v.dot(up) / (z * tanV);
      xmin = Math.min(xmin, x); xmax = Math.max(xmax, x);
      ymin = Math.min(ymin, y); ymax = Math.max(ymax, y);
    }
    return Math.max((xmax - xmin) / 2, (ymax - ymin) / 2);
  }

  /** AABB 중심을 보고, 세로·가로 FOV 모두에서 목표 비율이 되는 거리를 이분법으로 찾는다. */
  function frameTo(mode) {
    if (!lastView) return;
    const points = framingPoints(lastView, mode);
    if (points.length === 0) return;
    const { min, max } = boundsOf(points);
    const target = new THREE.Vector3((min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2);
    const corners = [];
    for (const x of [min[0], max[0]]) for (const y of [min[1], max[1]]) for (const z of [min[2], max[2]]) {
      corners.push(new THREE.Vector3(x, y, z));
    }
    const sinPhi = Math.sin(camState.phi);
    const offsetDir = new THREE.Vector3(
      sinPhi * Math.sin(camState.theta),
      Math.cos(camState.phi),
      sinPhi * Math.cos(camState.theta),
    ).normalize();
    const goal = FRAMING_TARGET[mode] ?? 0.75;
    let lo = 0.01;
    let hi = 100;
    for (let i = 0; i < 60; i += 1) {
      const mid = (lo + hi) / 2;
      if (projectedRatio(corners, target, offsetDir, mid) > goal) lo = mid;
      else hi = mid;
    }
    camState.target.copy(target);
    camState.radius = hi;
    framing.mode = mode;
    framing.auto = true;
    framing.ratio = projectedRatio(corners, target, offsetDir, hi);
    applyCamera();
    if (container.__view3dDebug) {
      container.__view3dDebug.framingRatio = framing.ratio;
      container.__view3dDebug.framingMode = mode;
    }
  }

  /** 작업자 쪽(착탄점→골반 수평 방향)에서 옆으로 약 50° 비켜 약간 위에서 본다. */
  function defaultAngles(view) {
    const joints = view.layers?.query?.joints ?? {};
    const body = isVec3(joints.Pelvis) ? joints.Pelvis : (view.corrections ?? []).map((c) => c.from).find(isVec3);
    const hit = view.composite?.hit;
    if (isVec3(body) && isVec3(hit)) {
      const dx = body[0] - hit[0];
      const dz = body[2] - hit[2];
      if (Math.hypot(dx, dz) > 1e-6) {
        return { theta: Math.atan2(dx, dz) + 0.9, phi: Math.PI / 2.5 };
      }
    }
    return { theta: Math.PI / 4, phi: Math.PI / 2.5 };
  }

  /** 벽 법선의 최선 추정(카메라 프리셋 편의, 측정값 아님). 작업자 쪽을 향하게 부호를 맞춘다. */
  function estimateWallNormal(view) {
    let normal = null;
    const mesh = view.mesh;
    const hit = view.composite?.hit;
    if (mesh && mesh.available === true && mesh.triangles?.length > 0 && isVec3(hit)) {
      // 대표 착탄점에 가장 가까운 삼각형 무게중심의 법선.
      let best = null;
      let bestDist = Infinity;
      const h = new THREE.Vector3(...hit);
      for (const [i0, i1, i2] of mesh.triangles) {
        const a = new THREE.Vector3(...mesh.vertices[i0]);
        const b = new THREE.Vector3(...mesh.vertices[i1]);
        const c = new THREE.Vector3(...mesh.vertices[i2]);
        const d = a.clone().add(b).add(c).multiplyScalar(1 / 3).distanceTo(h);
        if (d < bestDist) { bestDist = d; best = [a, b, c]; }
      }
      if (best) {
        const n = new THREE.Vector3().subVectors(best[1], best[0]).cross(new THREE.Vector3().subVectors(best[2], best[0]));
        if (n.lengthSq() > 1e-12) normal = n.normalize();
      }
    }
    if (!normal) {
      const wallCorrection = (view.corrections ?? []).find((c) => c.axis === "wall" && isVec3(c.unit));
      if (wallCorrection) normal = new THREE.Vector3(...wallCorrection.unit).normalize();
    }
    if (!normal) return null;
    const joints = view.layers?.query?.joints ?? {};
    if (isVec3(joints.Pelvis) && isVec3(hit)) {
      const toBody = new THREE.Vector3(...joints.Pelvis).sub(new THREE.Vector3(...hit));
      if (normal.dot(toBody) < 0) normal.negate();
    }
    return normal;
  }

  const checkboxes = {
    query: dom.controls.querySelector("#view3dLayerQuery"),
    reference: dom.controls.querySelector("#view3dLayerReference"),
    corrected: dom.controls.querySelector("#view3dLayerCorrected"),
    wall: dom.controls.querySelector("#view3dLayerWall"),
  };
  checkboxes.query.addEventListener("change", () => { layerGroups.query.visible = checkboxes.query.checked; });
  checkboxes.reference.addEventListener("change", () => { layerGroups.reference.visible = checkboxes.reference.checked; });
  checkboxes.corrected.addEventListener("change", () => {
    layerGroups.corrected.visible = checkboxes.corrected.checked;
    dom.correctedNote.textContent = checkboxes.corrected.checked
      ? "교정 후 층은 관측이 아니라 발부된 교정을 따랐을 때의 예상입니다."
      : "";
  });
  checkboxes.wall.addEventListener("change", () => {
    if (meshObject) meshObject.visible = checkboxes.wall.checked;
  });

  function applyPreset(offset) {
    if (!lastView) return;
    const normal = estimateWallNormal(lastView);
    if (!normal) {
      setNote(dom.note, "벽 방향을 추정할 수 없어 프리셋을 적용하지 않았습니다.", "empty");
      return;
    }
    camState.theta = Math.atan2(normal.x, normal.z) + offset;
    camState.phi = Math.PI / 2.2;
    frameTo("pose");
  }
  dom.controls.querySelector("#view3dPresetFit").addEventListener("click", () => {
    if (!lastView) return;
    Object.assign(camState, defaultAngles(lastView));
    frameTo("pose");
  });
  dom.controls.querySelector("#view3dPresetFront").addEventListener("click", () => applyPreset(0));
  dom.controls.querySelector("#view3dPresetSide").addEventListener("click", () => applyPreset(Math.PI / 2));
  dom.controls.querySelector("#view3dPresetTracks").addEventListener("click", () => frameTo("tracks"));
  dom.controls.querySelector("#view3dPresetWall").addEventListener("click", () => frameTo("wall"));

  function loading(message) {
    setNote(dom.note, message ?? "비교를 계산하는 중…", "loading");
    dom.host.classList.add("is-loading");
  }

  function resetScene() {
    for (const group of Object.values(layerGroups)) {
      disposeObject3D(group);
      group.clear();
    }
    disposeObject3D(correctionsGroup);
    correctionsGroup.clear();
    clearLabels();
    billboards.length = 0;
    coneRegistry = [];
    markerRegistry = [];
    if (meshObject) {
      scene.remove(meshObject);
      disposeObject3D(meshObject);
      meshObject = null;
    }
  }

  function render(payload) {
    dom.host.classList.remove("is-loading");
    const view = payload?.view;
    if (!view) {
      clear("표시할 뷰 데이터가 없습니다.", "empty");
      return;
    }
    lastView = view;
    resetScene();

    const weights = view.weights ?? {};
    const fit = view.fit ?? null;
    const joints = view.layers?.query?.joints ?? {};
    const hasSkeleton = Object.keys(joints).length > 0;

    layerGroups.query.add(buildLayerGroup(view.layers?.query, weights, LAYER_COLOR.query, !hasSkeleton));
    layerGroups.reference.add(buildLayerGroup(view.layers?.reference, weights, LAYER_COLOR.reference, false));
    layerGroups.corrected.add(buildLayerGroup(view.layers?.corrected, weights, LAYER_COLOR.corrected, false));
    layerGroups.query.add(buildSkeletonGroup(joints, view.layers?.query?.bones, fit?.faded === true));
    const anchors = buildAnchorGroup(fit);
    layerGroups.query.add(anchors.group);
    const passMean = buildPassMeanGroup(view.corrections ?? [], fit);
    layerGroups.query.add(passMean.group);
    if (isVec3(view.composite?.hit)) {
      const hitMarker = new THREE.Mesh(
        new THREE.SphereGeometry(HIT_MARKER_RADIUS, 14, 10),
        new THREE.MeshBasicMaterial({ color: LAYER_COLOR.query }),
      );
      hitMarker.position.set(...view.composite.hit);
      layerGroups.query.add(hitMarker);
    }
    layerGroups.query.visible = checkboxes.query.checked;
    layerGroups.reference.visible = checkboxes.reference.checked;
    layerGroups.corrected.visible = checkboxes.corrected.checked;

    // 화살표: 대표 프레임 실측 마커에서 나간다(from_frame). 번호는 서버의 n.
    const arrowStarts = [];
    for (const correction of view.corrections ?? []) {
      const useFrame = isVec3(correction.from_frame) && isVec3(correction.to_frame);
      const from = useFrame ? correction.from_frame : correction.from;
      const to = useFrame ? correction.to_frame : correction.to;
      if (!isVec3(from) || !isVec3(to)) continue;
      const arrow = buildArrow(from, to, sigmaColor(correction.sigma));
      if (!arrow) continue;
      correctionsGroup.add(arrow.group);
      coneRegistry.push({ n: correction.n, center: arrow.coneBase, radius: ARROW_HEAD_RADIUS });
      arrowStarts.push({ n: correction.n, from, source: useFrame ? "from_frame" : "from" });
      const start = new THREE.Vector3(...from);
      const end = new THREE.Vector3(...to);
      const mid = start.clone().add(end).multiplyScalar(0.5);
      registerLabel(mid, typeof correction.n === "number" ? String(correction.n) : "—", "view3d-label-arrow", {
        kind: "badge", n: correction.n, from: start, to: end,
      });
    }

    const mesh = view.mesh ?? null;
    meshObject = buildMesh(mesh);
    if (meshObject) {
      meshObject.visible = checkboxes.wall.checked;
      scene.add(meshObject);
      checkboxes.wall.disabled = false;
    } else {
      checkboxes.wall.disabled = true;
    }

    renderLegend(dom.legend, view);
    dom.notes.replaceChildren();
    if (fit) {
      for (const text of (fit.notes ?? []).slice(0, 3)) {
        const item = document.createElement("li");
        item.textContent = text;
        dom.notes.append(item);
      }
    } else if (typeof view.fit_note === "string") {
      const item = document.createElement("li");
      item.textContent = view.fit_note;
      dom.notes.append(item);
    }
    const wallText = wallNoteText(mesh);
    setNote(dom.wallNote, wallText, wallText && mesh?.available !== true ? "empty" : undefined);
    const errors = view.display_errors ?? [];
    setNote(
      dom.errors,
      errors.length > 0 ? `일부 표시 블록을 만들지 못했습니다: ${errors.map((e) => e.block).join(", ")}` : "",
      errors.length > 0 ? "error" : undefined,
    );
    setNote(dom.note, "", undefined);
    if (!checkboxes.corrected.checked) dom.correctedNote.textContent = "";

    let ringCount = 0;
    layerGroups.query.traverse((child) => { if (child.userData.kind === "pass-mean-ring") ringCount += 1; });
    const boneLengths = {};
    layerGroups.query.traverse((child) => {
      if (child.userData.kind === "bone") boneLengths[child.userData.name] = child.userData.length;
    });

    container.__view3dDebug = debugLive;
    Object.assign(debugLive, {
      arrowCount: correctionsGroup.children.length,
      arrowStarts,
      meshTriangleCount: meshObject ? meshObject.userData.triangleCount : 0,
      labelCount: labelRegistry.length,
      labels: labelRegistry.map((entry) => entry.el.textContent),
      jointCount: Object.keys(joints).length,
      markerKinds: { measuredSpheres: anchors.counts.measuredSpheres, passMeanRings: ringCount },
      offsetLines: anchors.counts.offsetLines,
      missLines: anchors.counts.missLines,
      missLabels: anchors.counts.missLabels,
      boneLengths,
      fit,
      framingRatio: null,
    });

    Object.assign(camState, defaultAngles(view));
    frameTo("pose");
    layoutLabels();
  }

  function clear(message, tone) {
    dom.host.classList.remove("is-loading");
    resetScene();
    lastView = null;
    setNote(dom.note, message ?? "패스를 고르고 평가를 누르면 여기에 뜹니다.", tone ?? "idle");
    dom.notes.replaceChildren();
    setNote(dom.wallNote, "", undefined);
    setNote(dom.errors, "", undefined);
    renderLegend(dom.legend, null);
    dom.correctedNote.textContent = "";
    container.__view3dDebug = null;
  }

  return { render, clear, loading };
}

/** 3D 뷰를 `container`에 붙인다. three.js가 없거나 초기화가 실패하면 안내만 남기고 던지지 않는다. */
export function createSkeletonView(container) {
  if (!container) return { render() {}, clear() {}, loading() {} };
  if (typeof THREE === "undefined") {
    showUnavailable(container, "three.js를 불러오지 못해 3D 뷰 없이 숫자 패널만 씁니다(오프라인이거나 CDN 차단).");
    return { render() {}, clear() {}, loading() {} };
  }
  try {
    return mount(THREE, container);
  } catch (err) {
    console.error("[skeleton_view] 3D 뷰 초기화 실패", err);
    showUnavailable(container, "3D 뷰를 초기화하지 못했습니다 — 숫자 패널은 그대로 동작합니다.");
    return { render() {}, clear() {}, loading() {} };
  }
}
