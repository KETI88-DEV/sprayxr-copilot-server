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
// 뼈는 장면 안에서 읽혀야 한다(2라운드: 1~2px 회색 선이 벽 메시 위에서 가장 약한 요소였다). 실측 마커(층 색
// 구체 0.035m)와는 채도로 위계를 둔다 — 뼈는 무채색, 실측은 층 색.
const BONE_RADIUS = 0.009;
const BONE_OPACITY = 0.9;
const BONE_OPACITY_FADED = 0.45;
// 자리표시 다리(측정 아님): 끊어 그린 실린더. 실선 뼈와 한눈에 구분되게.
const PLACEHOLDER_DASH = 0.05;
const PLACEHOLDER_GAP = 0.035;
// 계산된 층(교정 후·예상)의 선. 실측 궤적의 실선과 한눈에 구분되게 끊어 그린다.
const COMPUTED_DASH = 0.02;
const COMPUTED_GAP = 0.02;
const ARROW_SHAFT_RADIUS = 0.006;
const ARROW_HEAD_RADIUS = 0.016;
const HIT_MARKER_RADIUS = 0.018;

const BADGE_MIN_PX = 26;
// 글자 라벨을 몸통 화면 위치에서 바깥으로 미는 첫 거리(px).
const LABEL_PUSH_PX = 16;
// 맞춤 목표 = 맞춤 대상이 캔버스 반폭·반높이에서 차지할 비율. pose(기본·'자세 맞춤')는 벽도 궤적도
// 다리도 빼고 실측 마커·교정 화살표·상체 관절만 담으므로 캔버스를 거의 채워도 잘리지 않는다. 남기는
// 14%는 마커 구체 반지름·화살표 촉·글자 라벨 몫이다(0.75는 그만큼을 그냥 비워 뒀다).
// 프리셋별 목표 비율 = 맞춤 대상 상자의 **긴 쪽**이 캔버스에서 차지하는 비율.
// torso("상체 확대")만 1.0이다 — 상체를 캔버스에 꽉 채우려고 일부러 다리·교정 화살표·착탄점을 맞춤
// 대상에서 뺐고, 그래서 그것들이 화면 밖으로 나갈 수 있다. 나간 것의 이름은 캔버스 아래에 적는다.
const FRAMING_TARGET = { pose: 0.86, tracks: 0.75, wall: 0.9, torso: 0.98 };

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
    <label><input id="view3dLayerLegs" type="checkbox" checked> 다리(IK)</label>
    <label><input id="view3dLayerReference" type="checkbox" checked> 기준</label>
    <label><input id="view3dLayerCorrected" type="checkbox"> 교정 후(예상)</label>
    <label><input id="view3dLayerWall" type="checkbox" checked> 벽면</label>
    <span class="view3d-controls-sep" aria-hidden="true"></span>
    <button id="view3dPresetFit" type="button">자세 맞춤</button>
    <button id="view3dPresetTorso" type="button">상체 확대</button>
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
  // 벽면을 그리지 않는 사유는 캔버스 왼쪽 위에 겹쳐 띄운다 — 캔버스 아래 마지막 줄에 두면 첫눈에 안 보였다.
  const wallNote = document.createElement("p");
  wallNote.className = "view3d-wall-note";
  host.append(canvas, labelsHost, wallNote);

  const legend = document.createElement("div");
  legend.className = "view3d-legend";

  const notes = document.createElement("ul");
  notes.className = "view3d-notes";
  const notesMore = document.createElement("details");
  notesMore.className = "view3d-notes-more";
  notesMore.hidden = true;
  const errors = document.createElement("p");
  errors.className = "view3d-errors";
  const note = document.createElement("p");
  note.className = "view3d-note";
  const correctedNote = document.createElement("p");
  correctedNote.className = "view3d-note view3d-corrected-note";
  // 상체 확대 프리셋이 화면 밖으로 내보낸 맞춤 대상의 이름. 자른 사실을 화면이 스스로 말한다.
  const framingNote = document.createElement("p");
  framingNote.className = "view3d-note view3d-framing-note";

  // wallNote는 위에서 host(캔버스 위 겹침)에 넣었다 — 여기서 다시 넣으면 host 밖으로 옮겨진다.
  container.append(toolbar, host, legend, notes, notesMore, errors, note, correctedNote, framingNote);
  return { canvas, host, labelsHost, legend, notes, notesMore, wallNote, errors, note, correctedNote, framingNote, controls, toolbar };
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
  let skeletonLegs = null;
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
    // 검증이 **자기 손으로** 투영할 수 있도록 카메라의 날 상태만 내놓는다(계산된 측정값이 아니다).
    container.__view3dCameraDebug = {
      theta: camState.theta,
      phi: camState.phi,
      radius: camState.radius,
      target: [camState.target.x, camState.target.y, camState.target.z],
      fov: camera.fov,
      near: camera.near,
    };
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
    scheduleFramingNote();
  });
  dom.canvas.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      const factor = event.deltaY > 0 ? 1.1 : 1 / 1.1;
      camState.radius = clamp(camState.radius * factor, 0.05, 60);
      framing.auto = false;
      applyCamera();
      scheduleFramingNote();
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

  // --- 검증 계측(헤드리스 브라우저가 읽는다) ---------------------------------------------
  // 몸통 마디: 라벨을 밀어낼 장애물이자, 라벨-몸통 겹침을 세는 대상.
  // 몸통 **줄기**(골반-척추-가슴-머리)만. `torsoLabelOverlaps`가 세는 대상이 정확히 이 셋이다 —
  // 팔까지 포함한 넓은 판정은 `limbScreenRects()`/`boneLabelOverlaps`가 따로 낸다.
  const TORSO_BONES = [["Pelvis", "Spine"], ["Spine", "Chest"], ["Chest", "Head"]];
  const TORSO_HALF_PX = 11;
  // 화살표 기둥 반두께(px). 기둥은 몸통보다 얇으니 몸통(11px)보다 좁게 잡는다.
  const ARROW_HALF_PX = 8;
  // 팔·어깨 마디 반두께(px). 몸통 줄기보다 얇게 그려지므로 좁게 잡는다.
  const LIMB_HALF_PX = 8;
  // 보정 오프셋 선·어긋남 점선 반두께(px). 1px 가는 선이지만 글자 사이를 지나면 글자가 갈라져 보인다.
  const THIN_LINE_HALF_PX = 5;
  // 배지(.view3d-label-arrow) 반폭(px): 24px 원 + 1px 테두리. 중심거리 26px 검사는 대각선에서
  // 모서리 겹침을 놓친다(검증 라운드: ④↔① 20.4px²) — 사각형으로 겹침을 본다.
  const BADGE_HALF_PX = 13;
  const isVec3Like = (v) => v instanceof THREE.Vector3;

  /** 화면 선분 하나를 사각형 여러 개로 쪼갠다(라벨 회피용 장애물).
   * 표본 수는 화면 길이에서 뽑는다 — 고정 步수(옛 코드: 팔 6步)는 긴 마디에서 사각형 사이에 틈을 남겨
   * 라벨이 그 틈으로 뼈 위에 앉았다(840×1050의 '10.5cm' × LeftUpperArm>LeftForeArm). */
  function segmentRects(p0, p1, half, into) {
    const rects = into ?? [];
    const steps = Math.max(2, Math.ceil(Math.hypot(p1.x - p0.x, p1.y - p0.y) / half));
    for (let i = 0; i <= steps; i += 1) {
      const t = i / steps;
      const x = p0.x + (p1.x - p0.x) * t;
      const y = p0.y + (p1.y - p0.y) * t;
      rects.push({ x0: x - half, x1: x + half, y0: y - half, y1: y + half });
    }
    return rects;
  }

  /** 몸통 **줄기** 마디(골반-척추-가슴-머리)를 화면 사각형으로 쪼갠다. */
  function torsoScreenRects() {
    const joints = lastView?.layers?.query?.joints ?? {};
    const rects = [];
    for (const [a, b] of TORSO_BONES) {
      if (!isVec3(joints[a]) || !isVec3(joints[b])) continue;
      segmentRects(toPixel(new THREE.Vector3(...joints[a])), toPixel(new THREE.Vector3(...joints[b])), TORSO_HALF_PX, rects);
    }
    return rects;
  }

  /** 몸통 줄기를 뺀 **나머지 상체 마디**(어깨·팔·손)를 화면 사각형으로 쪼갠다.
   * 다리는 뺀다 — 다리 표시 방식은 아직 정해지지 않았고, 다리를 장애물로 넣으면 라벨이 위로만 몰린다.
   * 줄기만 장애물로 두면 거리 라벨이 팔뼈를 덮었다(검증 라운드: L3 '10.5cm' × LeftForeArm>LeftHand). */
  function limbScreenRects() {
    const joints = lastView?.layers?.query?.joints ?? {};
    const bones = lastView?.layers?.query?.bones ?? [];
    const legSet = new Set(lastView?.fit?.legs?.joints ?? []);
    const stem = new Set(TORSO_BONES.map(([a, b]) => `${a}>${b}`));
    const rects = [];
    for (const [a, b] of bones) {
      if (legSet.has(a) || legSet.has(b) || stem.has(`${a}>${b}`)) continue;
      if (!isVec3(joints[a]) || !isVec3(joints[b])) continue;
      segmentRects(toPixel(new THREE.Vector3(...joints[a])), toPixel(new THREE.Vector3(...joints[b])), LIMB_HALF_PX, rects);
    }
    return rects;
  }

  /** 교정 화살표 기둥(시작→끝)을 화살표별로 화면 사각형으로 쪼갠다.
   * 반투명 라벨 띠가 기둥을 가로지르면 화살표가 끊겨 보인다(L6의 '보정 오프셋' 라벨).
   * 배지는 **자기** 화살표 위에 앉아야 하므로 `n`으로 자기 기둥을 골라낼 수 있게 나눠 담는다. */
  function arrowScreenRectsByN() {
    const byN = [];
    for (const entry of labelRegistry) {
      if (entry.kind !== "badge" || !isVec3Like(entry.from) || !isVec3Like(entry.to)) continue;
      byN.push({ n: entry.n, rects: segmentRects(toPixel(entry.from), toPixel(entry.to), ARROW_HALF_PX) });
    }
    return byN;
  }

  function arrowScreenRects() {
    const rects = [];
    for (const arrow of arrowScreenRectsByN()) rects.push(...arrow.rects);
    return rects;
  }

  /** 보정 오프셋 선(관절→예측 마커)과 어긋남 점선(예측→실측)을 화면 사각형으로 쪼갠다.
   * 검증 라운드에서 '보정 오프셋 0.57m(적합값)' 글자 **사이로** 바로 그 오프셋 선이 지나갔다 —
   * 회피 장애물에 몸통·화살표 기둥은 있었지만 이 선들이 빠져 있었다. */
  function thinLineScreenRects() {
    const rects = [];
    for (const anchor of lastView?.fit?.anchors ?? []) {
      if (isVec3(anchor.joint_position) && isVec3(anchor.predicted_marker)) {
        segmentRects(
          toPixel(new THREE.Vector3(...anchor.joint_position)),
          toPixel(new THREE.Vector3(...anchor.predicted_marker)),
          THIN_LINE_HALF_PX, rects,
        );
      }
      if (isVec3(anchor.predicted_marker) && isVec3(anchor.measured_marker)) {
        segmentRects(
          toPixel(new THREE.Vector3(...anchor.predicted_marker)),
          toPixel(new THREE.Vector3(...anchor.measured_marker)),
          THIN_LINE_HALF_PX, rects,
        );
      }
    }
    return rects;
  }

  /** 실측 마커 구체가 화면에서 차지하는 사각형. 라벨이 실측 부호를 스치면 안 된다
   * (검증 라운드: L3에서 오프셋 라벨 아래 모서리가 왼손 실측 마커를 약 4px 스쳤다). */
  function measuredMarkerScreenRects() {
    const rects = [];
    for (const marker of markerRegistry) {
      const px = toPixel(marker.center);
      const half = screenDiameter(marker.center, MEASURED_MARKER_RADIUS) / 2;
      rects.push({ x0: px.x - half, x1: px.x + half, y0: px.y - half, y1: px.y + half });
    }
    return rects;
  }

  /** 넓은 정의: 다리를 뺀 **모든** 관절(골반·척추·머리·어깨·팔·손)과 **모든** 실측/예측 마커를
   * 화면에 투영한 바운딩 박스. 세로 양끝은 흔히 손 관절이나 손 마커다. 좁은 정의는
   * `torsoScreenSpan()`이 따로 낸다. */
  function upperBodyScreen() {
    if (!lastView) return null;
    const joints = lastView.layers?.query?.joints ?? {};
    const legSet = new Set(lastView.fit?.legs?.joints ?? []);
    const points = [];
    for (const [name, value] of Object.entries(joints)) if (!legSet.has(name) && isVec3(value)) points.push(value);
    for (const anchor of lastView.fit?.anchors ?? []) if (isVec3(anchor.measured_marker)) points.push(anchor.measured_marker);
    if (points.length === 0) return null;
    let top = Infinity; let bottom = -Infinity; let left = Infinity; let right = -Infinity;
    for (const value of points) {
      const px = toPixel(new THREE.Vector3(...value));
      top = Math.min(top, px.y); bottom = Math.max(bottom, px.y);
      left = Math.min(left, px.x); right = Math.max(right, px.x);
    }
    const { w, h } = hostSize();
    return {
      topPx: top, bottomPx: bottom, heightPx: bottom - top, widthPx: right - left,
      hostHeightPx: h, hostWidthPx: w, heightFraction: (bottom - top) / h,
    };
  }

  /** 좁은 정의: 머리 **실측** 마커(HMD)와 골반 관절, 이 두 점만의 화면 세로 거리와 캔버스 대비 비율.
   * F3 합격선 문구('상체(머리 마커~골반) ≥ 캔버스 40%')가 가리키는 바로 그 두 점이다.
   * 넓은 정의(`upperBodyScreen`)는 손 관절·손 마커가 양끝일 때가 많아 훨씬 큰 값이 나온다 —
   * 그 값에 이 문구의 이름을 붙이면 출처와 다른 라벨이 된다. 두 값을 따로 내보내는 이유다. */
  function torsoScreenSpan() {
    if (!lastView) return null;
    const head = (lastView.fit?.anchors ?? []).find((a) => a.joint === "Head" && isVec3(a.measured_marker));
    const pelvis = lastView.layers?.query?.joints?.Pelvis;
    if (!head || !isVec3(pelvis)) return null;
    const a = toPixel(new THREE.Vector3(...head.measured_marker));
    const b = toPixel(new THREE.Vector3(...pelvis));
    const { h } = hostSize();
    return {
      topPx: Math.min(a.y, b.y), bottomPx: Math.max(a.y, b.y),
      heightPx: Math.abs(b.y - a.y), hostHeightPx: h, heightFraction: Math.abs(b.y - a.y) / h,
      ends: ["marker:Head(HMD)", "joint:Pelvis"],
    };
  }

  /** 추정 머리가 채운 구가 아니라 윤곽선인지(범례 진실성). */
  function estimateHeadIsOutline() {
    let found = null;
    layerGroups.query.traverse((child) => {
      if (child.userData.kind === "estimate-head") found = child.material?.wireframe === true;
    });
    return found;
  }

  /** 계산된 층(교정 후·예상)의 표식이 채운 구가 아니라 윤곽선인지(범례 진실성).
   * 범례 ②는 "실측은 … 층 색으로 채운 구체"라고 선언한다 — 계산값이 같은 부호를 쓰면 범례가 거짓이다.
   * 표식이 하나도 없으면 null(판정 대상 없음). */
  function computedLayerIsOutline() {
    let markers = 0;
    let outlines = 0;
    layerGroups.corrected.traverse((child) => {
      if (child.userData.kind !== "computed-layer-marker") return;
      markers += 1;
      if (child.material?.wireframe === true) outlines += 1;
    });
    return markers === 0 ? null : { markers, outlines, allOutline: markers === outlines };
  }

  /** 스켈레톤 가슴(없으면 골반)의 화면 위치. 스켈레톤이 없으면 null. */
  function bodyScreenCenter() {
    const joints = lastView?.layers?.query?.joints ?? {};
    const anchor = isVec3(joints.Chest) ? joints.Chest : isVec3(joints.Pelvis) ? joints.Pelvis : null;
    return anchor ? toPixel(new THREE.Vector3(...anchor)) : null;
  }

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
    let labelOverlaps = 0;
    // 몸통 줄기는 장애물이다 — 글자 라벨이 몸통 위에 앉으면 뼈도 글자도 못 읽는다(보정 오프셋 라벨).
    const torsoRects = torsoScreenRects();
    // 어깨·팔·손 마디도 장애물이다(줄기만 넣었더니 거리 라벨이 팔뼈를 덮었다).
    const limbRects = limbScreenRects();
    // 교정 화살표 기둥도 장애물이다 — 반투명 라벨 띠가 기둥을 가로지르면 화살표가 끊겨 보인다.
    const arrowsByN = arrowScreenRectsByN();
    const arrowRects = [];
    for (const arrow of arrowsByN) arrowRects.push(...arrow.rects);
    // 보정 오프셋 선·어긋남 점선, 그리고 실측 마커 구체도 장애물이다.
    const thinLineRects = thinLineScreenRects();
    const markerRects = measuredMarkerScreenRects();
    // 글자 라벨은 몸통 화면 위치에서 **바깥쪽**으로 민다 — 몸 한가운데에 거리·오프셋 라벨이 쌓여 팔·몸통을
    // 가렸다(2라운드). 기준점은 스켈레톤 가슴(없으면 라벨 자리 그대로).
    const body = bodyScreenCenter();
    let torsoLabelOverlaps = 0;
    let limbLabelOverlaps = 0;
    let arrowLabelOverlaps = 0;
    let thinLineLabelOverlaps = 0;
    let markerLabelOverlaps = 0;
    let labelsClamped = 0;
    for (const entry of labelRegistry) {
      if (entry.kind === "badge" || !visible(entry)) continue;
      const px = toPixel(entry.position);
      const w = entry.el.offsetWidth;
      const h = entry.el.offsetHeight;
      // .view3d-label의 transform: translate(-50%, -140%) — 사각형은 (x, y) 기준 위쪽에 놓인다.
      let ox = 0;
      let oy = 0;
      if (body) {
        const dx = px.x - body.x;
        const dy = px.y - body.y;
        const len = Math.hypot(dx, dy);
        [ox, oy] = len < 1e-6 ? [1, 0] : [dx / len, dy / len];
      }
      const rect = (xx, yy) => ({ x0: xx - w / 2, x1: xx + w / 2, y0: yy - 1.4 * h, y1: yy - 0.4 * h });
      const hits = (r, list) => list.some((o) => r.x0 < o.x1 && r.x1 > o.x0 && r.y0 < o.y1 && r.y1 > o.y0);
      const area = (r, list) => list.reduce((sum, o) => {
        const dx = Math.min(r.x1, o.x1) - Math.max(r.x0, o.x0);
        const dy = Math.min(r.y1, o.y1) - Math.max(r.y0, o.y0);
        return sum + (dx > 0 && dy > 0 ? dx * dy : 0);
      }, 0);
      const clashes = (r) => hits(r, obstacles) || hits(r, torsoRects) || hits(r, limbRects)
        || hits(r, arrowRects) || hits(r, thinLineRects) || hits(r, markerRects);
      // 가릴 때의 손해가 다르다: 몸통·팔을 덮으면 뼈도 글자도 못 읽고, 화살표 기둥을 덮으면 교정 화살표가
      // 끊겨 보이며, 실측 마커를 덮으면 범례가 가리키는 바로 그 부호가 사라진다. 가는 오프셋 선과
      // 앞 라벨끼리 겹치는 것은 그보다 가볍다.
      const cost = (r) => 4 * area(r, torsoRects) + 3 * area(r, limbRects) + 2 * area(r, arrowRects)
        + 2 * area(r, markerRects) + area(r, thinLineRects) + area(r, obstacles);
      // 캔버스 밖으로 밀려나면 글자가 잘린다 — 후보 자리마다 캔버스 안으로 되돌린 뒤에 점수를 매긴다.
      const { w: hostW, h: hostH } = hostSize();
      const fit = (xx, yy) => ({
        x: clamp(xx, w / 2 + 2, hostW - w / 2 - 2),
        y: clamp(yy, 1.4 * h + 2, hostH - 0.4 * h - 2),
      });
      // 한 방향으로만 밀면 화살표 네 개가 모인 쪽(L6 오른손)에서 빠져나갈 자리를 못 찾는다. 바깥쪽
      // 방향을 첫 후보로 두고 ±30°씩 틀어 가며, 거리도 늘려 가며 훑는다. 전부 겹치면 손해가 가장 작은
      // 자리를 쓴다(겹침 수는 아래에서 그대로 센다).
      const turns = [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6];
      const radii = [LABEL_PUSH_PX, LABEL_PUSH_PX + 26, LABEL_PUSH_PX + 58, LABEL_PUSH_PX + 96, LABEL_PUSH_PX + 140];
      let best = null;
      for (const radius of radii) {
        for (const turn of turns) {
          const angle = (turn * Math.PI) / 6;
          const cos = Math.cos(angle);
          const sin = Math.sin(angle);
          const dx = ox * cos - oy * sin;
          const dy = ox * sin + oy * cos;
          const ux = px.x + dx * radius;
          const uy = px.y + dy * radius;
          const spot = fit(ux, uy);
          const score = cost(rect(spot.x, spot.y));
          const clamped = Math.abs(spot.x - ux) > 0.5 || Math.abs(spot.y - uy) > 0.5;
          if (best === null || score < best.score) best = { ...spot, score, clamped };
          if (score === 0) break;
        }
        if (best !== null && best.score === 0) break;
      }
      let x = best.x;
      let y = best.y;
      if (best.clamped) labelsClamped += 1;
      if (clashes(rect(x, y))) labelOverlaps += 1;
      if (hits(rect(x, y), torsoRects)) torsoLabelOverlaps += 1;
      if (hits(rect(x, y), limbRects)) limbLabelOverlaps += 1;
      if (hits(rect(x, y), arrowRects)) arrowLabelOverlaps += 1;
      if (hits(rect(x, y), thinLineRects)) thinLineLabelOverlaps += 1;
      if (hits(rect(x, y), markerRects)) markerLabelOverlaps += 1;
      entry.el.style.left = `${x}px`;
      entry.el.style.top = `${y}px`;
      obstacles.push(rect(x, y));
    }
    // 이름이 재는 것과 정확히 같아야 한다(검증 라운드 지적): `torsoLabelOverlaps`는 줄기 셋만,
    // `boneLabelOverlaps`는 다리를 뺀 **모든** 뼈다. 배지는 글자 라벨과 따로 센다(아래).
    debugLive.labelOverlaps = labelOverlaps;
    debugLive.torsoLabelOverlaps = torsoLabelOverlaps;
    debugLive.limbLabelOverlaps = limbLabelOverlaps;
    debugLive.boneLabelOverlaps = torsoLabelOverlaps + limbLabelOverlaps;
    debugLive.arrowLabelOverlaps = arrowLabelOverlaps;
    debugLive.offsetLineLabelOverlaps = thinLineLabelOverlaps;
    debugLive.measuredMarkerLabelOverlaps = markerLabelOverlaps;
    debugLive.labelsClamped = labelsClamped;
    debugLive.upperBody = upperBodyScreen();
    debugLive.torsoSpan = torsoScreenSpan();
    debugLive.estimateHeadIsOutline = estimateHeadIsOutline();
    debugLive.correctedLayerIsOutline = computedLayerIsOutline();
    // 검증자가 **자기 손으로** 다시 투영할 수 있게 카메라 행렬과 캔버스 크기를 그대로 내보낸다.
    // (구현이 낸 사각형·계수기를 믿지 않고 서버 페이로드에서 픽셀을 직접 얻는 길.)
    camera.updateMatrixWorld();
    debugLive.camera = {
      projectionMatrix: camera.projectionMatrix.toArray(),
      matrixWorldInverse: camera.matrixWorldInverse.toArray(),
      host: hostSize(),
    };
    // 번호 배지. 옛 배치는 (1) 배지끼리를 중심거리 26px로만 봐서 대각선 모서리 겹침을 놓쳤고
    // (2) 회피 탐색에서 **남의 화살표 기둥**을 아예 안 봤다(배지 ④가 화살표 ①·②의 기둥에 앉았다).
    // 이제 사각형 겹침으로 보고, 자기 기둥만 빼고 남의 기둥도 장애물로 넣는다.
    const badgeRect = (x, y) => ({ x0: x - BADGE_HALF_PX, x1: x + BADGE_HALF_PX, y0: y - BADGE_HALF_PX, y1: y + BADGE_HALF_PX });
    const overlapArea = (r, list) => list.reduce((sum, o) => {
      const dx = Math.min(r.x1, o.x1) - Math.max(r.x0, o.x0);
      const dy = Math.min(r.y1, o.y1) - Math.max(r.y0, o.y0);
      return sum + (dx > 0 && dy > 0 ? dx * dy : 0);
    }, 0);
    const overlaps = (r, list) => list.some((o) => r.x0 < o.x1 && r.x1 > o.x0 && r.y0 < o.y1 && r.y1 > o.y0);
    const { w: badgeHostW, h: badgeHostH } = hostSize();
    const placedBadges = [];
    const placedBadgeRects = [];
    let obstacleOverlaps = 0;
    let badgeBadgeOverlaps = 0;
    let badgeArrowOverlaps = 0;
    for (const entry of labelRegistry) {
      if (entry.kind !== "badge" || !visible(entry)) continue;
      const px = toPixel(entry.position);
      const a = toPixel(entry.from);
      const b = toPixel(entry.to);
      let dx = b.x - a.x;
      let dy = b.y - a.y;
      const len = Math.hypot(dx, dy);
      if (len < 1e-6) { dx = 0; dy = -1; } else { dx /= len; dy /= len; }
      // 자기 기둥은 배지가 앉아야 할 자리다 — 장애물에서 뺀다.
      const otherArrowRects = [];
      for (const arrow of arrowsByN) if (arrow.n !== entry.n) otherArrowRects.push(...arrow.rects);
      const badgeScore = (r) => 8 * overlapArea(r, placedBadgeRects) + 4 * overlapArea(r, otherArrowRects) + overlapArea(r, obstacles);
      // 기둥 방향을 축으로 삼아 앞뒤·좌우로만 훑으면(옛 along/side 격자) 찾는 자리가 기둥을 낀 가는 띠
      // 안으로 갇힌다 — 상체 확대처럼 라벨이 촘촘한 화면에서는 그 띠 안에 0점 자리가 없을 수 있다(검증
      // 라운드 지적: 김경호 L2 1000390_1 #2, 1875×1000, 상체 확대에서 배지 ②가 "6.5cm" 라벨과 226px²
      // 겹쳤다). 라벨 배치와 같은 반경×회전 극좌표 훑기로 바꿔 기둥 주변 360°를 본다. radius=1×BADGE_MIN_PX,
      // turn=0(기둥 방향 그대로)이 첫 후보라 이미 0점을 찾던 자리는 그대로 유지된다.
      const turns = [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6];
      const radii = [1, 2, 3, 4, 5, 6, 7, 8].map((n) => n * BADGE_MIN_PX);
      let best = null;
      for (const radius of radii) {
        for (const turn of turns) {
          const angle = (turn * Math.PI) / 6;
          const cos = Math.cos(angle);
          const sin = Math.sin(angle);
          const rdx = dx * cos - dy * sin;
          const rdy = dx * sin + dy * cos;
          const ux = px.x + rdx * radius;
          const uy = px.y + rdy * radius;
          const x = clamp(ux, BADGE_HALF_PX + 1, badgeHostW - BADGE_HALF_PX - 1);
          const y = clamp(uy, BADGE_HALF_PX + 1, badgeHostH - BADGE_HALF_PX - 1);
          const value = badgeScore(badgeRect(x, y));
          if (best === null || value < best.value) best = { x, y, value };
          if (value === 0) break;
        }
        if (best !== null && best.value === 0) break;
      }
      const rect = badgeRect(best.x, best.y);
      if (overlaps(rect, obstacles)) obstacleOverlaps += 1;
      if (overlaps(rect, placedBadgeRects)) badgeBadgeOverlaps += 1;
      if (overlaps(rect, otherArrowRects)) badgeArrowOverlaps += 1;
      placedBadges.push({ n: entry.n, x: best.x, y: best.y });
      placedBadgeRects.push(rect);
      entry.el.style.left = `${best.x}px`;
      entry.el.style.top = `${best.y}px`;
    }
    // 이름이 재는 것과 같아야 한다: 아래 셋은 **배지** 전용 계수기다(글자 라벨은 위에서 따로 센다).
    debugLive.badgeObstacleOverlaps = obstacleOverlaps;
    debugLive.badgeBadgeOverlaps = badgeBadgeOverlaps;
    debugLive.badgeArrowOverlaps = badgeArrowOverlaps;
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

  /** 층 궤적 하나를 그린다.
   *
   * `measured`는 이 층의 점이 관측인지 계산인지다 — 범례가 "실측 = 층 색으로 채운 구체"라고
   * 선언했으므로, 계산된 층(교정 후·예상)이 같은 부호를 쓰면 범례가 거짓이 된다. 질의·기준 궤적은
   * 같은 세 강체의 실측이라 채운 구를 쓰고, 교정 후 층은 **끊어 그린 선 + 속 빈(윤곽선) 표식**으로
   * 그린다(추정 머리를 윤곽선 구로 바꾼 것과 같은 원칙). */
  function buildLayerGroup(layerData, weights, layerColor, labelPointNames, measured) {
    const group = new THREE.Group();
    for (const name of Object.keys(layerData ?? {})) {
      // 착탄 격자(`hit`)는 호길이 리샘플이라 모서리를 가로지르면 벽에서 뜬다 — 선으로 그리지 않는다.
      // 질의 착탄 궤적은 원본 착탄점(`view.surface_hits`)으로 따로 그린다.
      if (name === "joints" || name === "bones" || name === "hit") continue;
      const weight = typeof weights?.[name] === "number" ? weights[name] : 0;
      const emph = emphasis(weight);
      const points = (layerData[name] ?? [])
        .filter((p) => isVec3(p))
        .map((p) => new THREE.Vector3(p[0], p[1], p[2]));
      if (points.length === 0) continue;

      const lineMaterial = measured === false
        ? new THREE.LineDashedMaterial({
          color: layerColor,
          transparent: true,
          opacity: emph.opacity,
          dashSize: COMPUTED_DASH,
          gapSize: COMPUTED_GAP,
        })
        : new THREE.LineBasicMaterial({
          color: layerColor,
          transparent: true,
          opacity: emph.opacity,
        });
      const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), lineMaterial);
      if (measured === false) line.computeLineDistances();
      line.userData.kind = measured === false ? "computed-layer-line" : "measured-layer-line";
      group.add(line);

      if (emph.showMarker) {
        const markerGeometry = new THREE.SphereGeometry(0.004 * emph.width * (measured === false ? 1.6 : 1), 10, 6);
        const markerMaterial = new THREE.MeshBasicMaterial({
          color: layerColor,
          transparent: true,
          opacity: emph.opacity,
          wireframe: measured === false,
        });
        for (const p of points) {
          const marker = new THREE.Mesh(markerGeometry, markerMaterial);
          marker.position.copy(p);
          marker.userData.kind = measured === false ? "computed-layer-marker" : "measured-layer-marker";
          group.add(marker);
        }
        if (emph.showLabel && labelPointNames) {
          registerLabel(points[points.length - 1], POINT_LABELS[name] ?? name, "view3d-label-point");
        }
      }
    }
    return group;
  }

  /** 추정 뼈. 강조 대상이 아니라 문맥이다 — weights로 강조를 따지지 않고 무채색 한 색.
   * 다리 관절 목록은 서버(`fit.legs.joints`)가 준다. IK 다리는 따로 끄고 켜고, 서버가 준 자리표시 다리
   * (`fit.legs.placeholder_joints`, 측정 아님)는 끊어 그린다 — 부위 이름을 여기서 판단하지 않는다. */
  function boneCylinder(start, end, radius, material, kind, name) {
    const dir = new THREE.Vector3().subVectors(end, start);
    const length = dir.length();
    if (length < 1e-6) return null;
    dir.normalize();
    const mesh = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, length, 8), material);
    mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
    mesh.position.copy(start).addScaledVector(dir, length / 2);
    mesh.userData.kind = kind;
    mesh.userData.name = name;
    mesh.userData.length = length;
    return mesh;
  }

  function buildSkeletonGroup(joints, bones, faded, legJoints, placeholder) {
    const group = new THREE.Group();
    const legs = new THREE.Group();
    const placeholderLegs = new THREE.Group();
    if (!joints || !bones || Object.keys(joints).length === 0) return { group, legs, placeholderLegs };
    const legSet = new Set(Array.isArray(legJoints) ? legJoints : []);
    const opacity = faded ? BONE_OPACITY_FADED : BONE_OPACITY;
    // 조명과 무관한 색을 위해 MeshBasicMaterial. WebGL 선은 굵기를 키울 수 없어 실린더로 그린다.
    const boneMaterial = new THREE.MeshBasicMaterial({ color: BONE_COLOR, transparent: true, opacity, depthWrite: false });
    const legMaterial = new THREE.MeshBasicMaterial({ color: BONE_COLOR, transparent: true, opacity: opacity * 0.55, depthWrite: false });
    for (const [parent, child] of bones) {
      const a = joints[parent];
      const b = joints[child];
      if (!isVec3(a) || !isVec3(b)) continue;
      const isLeg = legSet.has(child);
      const bone = boneCylinder(new THREE.Vector3(...a), new THREE.Vector3(...b), BONE_RADIUS, isLeg ? legMaterial : boneMaterial, "bone", `${parent}>${child}`);
      if (!bone) continue;
      bone.userData.leg = isLeg;
      (isLeg ? legs : group).add(bone);
    }
    const jointGeometry = new THREE.SphereGeometry(BONE_RADIUS * 1.6, 10, 8);
    for (const [name, position] of Object.entries(joints)) {
      if (!isVec3(position)) continue;
      const isLeg = legSet.has(name);
      const marker = new THREE.Mesh(jointGeometry, isLeg ? legMaterial : boneMaterial);
      marker.position.set(position[0], position[1], position[2]);
      (isLeg ? legs : group).add(marker);
    }
    if (isVec3(joints.Head)) {
      // 추정 머리는 **윤곽선**이다. 채운 구로 그리면 화면에서 가장 큰 구가 되고, 범례가 층 색으로
      // 채운 구체를 실측이라 선언하는 것과 부호가 겹쳐 읽힌다(service.LEGEND_NOTES).
      const headMaterial = new THREE.MeshBasicMaterial({ color: BONE_COLOR, transparent: true, opacity: opacity * 0.5, depthWrite: false, wireframe: true });
      const head = new THREE.Mesh(new THREE.SphereGeometry(HEAD_SPHERE_RADIUS, 12, 8), headMaterial);
      head.position.set(...joints.Head);
      head.userData.kind = "estimate-head";
      group.add(head);
    }
    // 자리표시 다리: 서버가 준 관절·뼈를 그대로 끊어 그린다(엉덩이는 골반에서 나간다).
    const placeholderJoints = placeholder?.joints ?? null;
    if (placeholderJoints && Array.isArray(placeholder.bones)) {
      const dashMaterial = new THREE.MeshBasicMaterial({ color: BONE_COLOR, transparent: true, opacity: opacity * 0.8, depthWrite: false });
      const lookup = (name) => placeholderJoints[name] ?? joints[name];
      for (const [parent, child] of placeholder.bones) {
        const a = lookup(parent);
        const b = lookup(child);
        if (!isVec3(a) || !isVec3(b)) continue;
        const start = new THREE.Vector3(...a);
        const end = new THREE.Vector3(...b);
        const dir = new THREE.Vector3().subVectors(end, start);
        const total = dir.length();
        if (total < 1e-6) continue;
        dir.normalize();
        const bone = new THREE.Group();
        bone.userData.kind = "placeholder-bone";
        bone.userData.name = `${parent}>${child}`;
        bone.userData.length = total;
        for (let t = 0; t < total; t += PLACEHOLDER_DASH + PLACEHOLDER_GAP) {
          const s0 = start.clone().addScaledVector(dir, t);
          const s1 = start.clone().addScaledVector(dir, Math.min(total, t + PLACEHOLDER_DASH));
          const dash = boneCylinder(s0, s1, BONE_RADIUS * 0.8, dashMaterial, "placeholder-dash", "");
          if (dash) bone.add(dash);
        }
        placeholderLegs.add(bone);
      }
    }
    group.add(legs);
    group.add(placeholderLegs);
    return { group, legs, placeholderLegs };
  }

  /** 대표 프레임 실측 마커(구체)·보정 오프셋 선·어긋남 점선·miss 라벨. */
  function buildAnchorGroup(fit) {
    const group = new THREE.Group();
    markerRegistry = [];
    const counts = { measuredSpheres: 0, offsetLines: 0, missLines: 0, missLabels: [], offsetLabels: [] };
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
        // 손 관절과 건 마커가 멀리 떨어진 이유를 선 위에서 바로 읽게 한다(서버 문자열, 파지 상한 초과일 때만).
        if (typeof anchor.offset_display === "string") {
          const mid = new THREE.Vector3(...anchor.joint_position).add(new THREE.Vector3(...anchor.predicted_marker)).multiplyScalar(0.5);
          registerLabel(mid, anchor.offset_display, "view3d-label-offset");
          counts.offsetLabels.push(anchor.offset_display);
        }
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
      // 캔버스 위 겹침은 한 줄로 — 세부 사유(reason_detail)는 캔버스 아래 '고지 더 보기'에 둔다.
      return [meshBlock.reason_text ?? "이 레벨은 벽면을 그리지 않습니다.", meshBlock.reason_hint]
        .filter((text) => typeof text === "string" && text.length > 0)
        .join(" ");
    }
    if (typeof meshBlock.handedness_note === "string") {
      const detail = typeof meshBlock.handedness_evidence_display === "string" ? ` ${meshBlock.handedness_evidence_display}` : "";
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
    // torso("상체 확대"): 맞춤 대상을 **상체만**으로 좁힌다 — 다리·교정 화살표·착탄점을 다 뺀다.
    // 다리 이름은 여기서 판단하지 않고 서버(`fit.legs.joints`)가 준 목록을 쓴다.
    if (mode === "torso") {
      for (const p of upperBodyPoints(view)) push(p);
      return points;
    }
    if (hasSkeleton) {
      // 숨긴 다리는 맞춤 대상에서도 뺀다 — 안 보이는 점 때문에 몸이 작아지면 안 된다. 대신 보이는
      // 자리표시 다리는 넣는다.
      // pose(자세 맞춤·기본)에서는 **보이든 말든** 다리를 맞춤 대상에서 뺀다. 무릎~발끝이 세로를 다
      // 먹으면 실측 마커·교정 화살표가 모인 상체가 쪼그라든다. 다리는 그대로 그린다 — 여기서 빼는 것은
      // 카메라 거리 계산뿐이고, 궤적·벽 프리셋은 예전대로 다리를 넣는다.
      const legJoints = Array.isArray(view.fit?.legs?.joints) ? view.fit.legs.joints : [];
      const legsHidden = !checkboxes.legs.checked && legJoints.length > 0;
      const excluded = new Set(mode === "pose" || legsHidden ? legJoints : []);
      for (const [name, p] of Object.entries(joints)) if (!excluded.has(name)) push(p);
      const placeholderJoints = view.fit?.legs?.placeholder_joints;
      if (mode !== "pose" && legsHidden && placeholderJoints) for (const p of Object.values(placeholderJoints)) push(p);
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
    // 착탄점은 벽 위에 있어 몸에서 멀다. pose(자세 맞춤·기본)는 실측 마커·교정 화살표·상체 관절을
    // 우선하므로 넣지 않는다 — 넣으면 벽까지 담느라 상체가 다시 쪼그라든다. 궤적·벽 프리셋은 넣는다.
    if (mode !== "pose") push(view.composite?.hit);
    if (mode === "tracks" || mode === "wall") {
      for (const name of ["head", "right", "left"]) for (const p of view.layers?.query?.[name] ?? []) push(p);
    }
    if (mode === "wall") {
      // 원본 착탄점만 — 리샘플 격자는 벽에서 뜰 수 있다.
      for (const p of view.surface_hits ?? []) push(p);
      if (view.mesh?.available === true) for (const p of view.mesh.vertices ?? []) push(p);
    }
    return points;
  }

  /** 상체 점(월드): 서버가 다리라고 이름 붙인 관절을 뺀 모든 관절 + 실측·예측 마커.
   * 화면에 이미 있는 "상체" 정의(`upperBodyScreen`)와 **같은 점 집합**이다 — 두 곳에서 다르게 정의하면
   * 프리셋이 키운 것과 계측이 재는 것이 달라진다. */
  function upperBodyPoints(view) {
    const joints = view?.layers?.query?.joints ?? {};
    const legSet = new Set(view?.fit?.legs?.joints ?? []);
    const points = [];
    for (const [name, value] of Object.entries(joints)) if (!legSet.has(name) && isVec3(value)) points.push(value);
    for (const anchor of view?.fit?.anchors ?? []) {
      if (isVec3(anchor.measured_marker)) points.push(anchor.measured_marker);
      if (isVec3(anchor.predicted_marker)) points.push(anchor.predicted_marker);
    }
    return points;
  }

  const ANCHOR_NAME = { head: "머리", right: "오른손", left: "왼손" };

  /** 기본(자세 맞춤) 프리셋이 자르지 않는 맞춤 대상에 **이름**을 붙여 돌려준다.
   * 상체 확대가 무엇을 화면 밖으로 내보냈는지 말하려면 점마다 사람이 읽는 이름이 있어야 한다. */
  function namedFitTargets(view) {
    const out = [];
    const add = (name, p) => { if (isVec3(p)) out.push({ name, p }); };
    const joints = view?.layers?.query?.joints ?? {};
    const legSet = new Set(view?.fit?.legs?.joints ?? []);
    for (const [name, value] of Object.entries(joints)) if (!legSet.has(name)) add(`${name} 관절`, value);
    for (const anchor of view?.fit?.anchors ?? []) {
      const who = ANCHOR_NAME[anchor.point] ?? anchor.point;
      add(`${who} 실측 마커`, anchor.measured_marker);
      add(`${who} 예측 마커`, anchor.predicted_marker);
    }
    for (const correction of view?.corrections ?? []) {
      const n = typeof correction.n === "number" ? correction.n : "?";
      const useFrame = isVec3(correction.from_frame) && isVec3(correction.to_frame);
      add(`교정 ${n}의 시작점`, useFrame ? correction.from_frame : correction.from);
      add(`교정 ${n}의 화살촉`, useFrame ? correction.to_frame : correction.to);
    }
    return out;
  }

  const OFFSCREEN_NAMES_SHOWN = 4;

  /** 한글 주격 조사. 앞 글자에 받침이 있으면 "이", 없으면 "가"(숫자·영문은 "가"로 둔다). */
  function subjectParticle(text) {
    const code = text.charCodeAt(text.length - 1);
    if (code < 0xac00 || code > 0xd7a3) return "가";
    return (code - 0xac00) % 28 === 0 ? "가" : "이";
  }

  /** 지금 카메라에서 캔버스 밖에 있는 맞춤 대상의 이름. 프리셋이 무엇을 잘랐는지 화면이 스스로 말한다. */
  function offscreenFitTargets() {
    if (!lastView) return [];
    const { w, h } = hostSize();
    const out = [];
    for (const { name, p } of namedFitTargets(lastView)) {
      const px = toPixel(new THREE.Vector3(...p));
      if (px.z > 1 || px.x < 0 || px.y < 0 || px.x > w || px.y > h) out.push(name);
    }
    return out;
  }

  function describeTorsoFraming(mode) {
    if (mode !== "torso") return "";
    // 프리셋을 누른 그대로면 "상체 확대", 그 뒤 카메라를 만졌으면 프리셋 이름으로 부르지 않는다.
    const head = framing.auto ? "상체 확대" : "지금 화면(상체 확대에서 카메라를 만졌습니다)";
    const outside = offscreenFitTargets();
    if (outside.length === 0) return `${head}: 맞춤 대상이 모두 화면 안에 있습니다.`;
    const shown = outside.slice(0, OFFSCREEN_NAMES_SHOWN).join(" · ");
    const rest = outside.length > OFFSCREEN_NAMES_SHOWN ? ` 외 ${outside.length - OFFSCREEN_NAMES_SHOWN}개` : "";
    return `${head}: ${shown}${rest}${subjectParticle(rest || shown)} 화면 밖에 있습니다 — 기본 뷰는 '자세 맞춤'입니다.`;
  }

  /** 자리표시 발(D3, 바닥 맞춤)의 월드 좌표. 발 관절의 **이름은 서버**(`fit.legs.placeholder_feet`)가 준다 —
   * 화면이 리그 이름을 지어내면 출처와 다른 라벨이 된다. 다리를 그리지 않는 화면에서는 빈 배열. */
  function placeholderFeetPoints(view) {
    const joints = view?.fit?.legs?.placeholder_joints;
    const names = view?.fit?.legs?.placeholder_feet;
    if (!joints || !Array.isArray(names)) return [];
    return names.map((name) => joints[name]).filter(isVec3);
  }

  /** 발이 보이는지 재 볼 프리셋. 툴바 버튼과 같은 카메라를 푼다(라벨도 버튼과 같은 글자다). */
  const PRESET_PROBES = [
    { label: "자세 맞춤", mode: "pose", angles: "default" },
    { label: "상체 확대", mode: "torso", angles: "current" },
    { label: "벽 정면", mode: "pose", angles: "wall", offset: 0 },
    { label: "벽 측면", mode: "pose", angles: "wall", offset: Math.PI / 2 },
    { label: "궤적 포함", mode: "tracks", angles: "current" },
    { label: "벽 포함", mode: "wall", angles: "current" },
  ];

  function solvePreset(probe) {
    let theta = camState.theta;
    let phi = camState.phi;
    if (probe.angles === "default") {
      ({ theta, phi } = defaultAngles(lastView));
    } else if (probe.angles === "wall") {
      const normal = estimateWallNormal(lastView);
      if (!normal) return null;
      theta = Math.atan2(normal.x, normal.z) + probe.offset;
      phi = Math.PI / 2.2;
    }
    return solveFraming(probe.mode, theta, phi);
  }

  /** 발이 **이 프리셋에서** 화면 밖이면, 발이 보이는 프리셋의 이름을 같이 말한다.
   *
   * 자리표시 발을 바닥면에 세운 것(D3)이 기본 뷰('자세 맞춤')에서는 보이지 않는다 — 다리는 맞춤
   * 대상이 아니어서 카메라가 상체에 맞춰 서기 때문이다. 실측(김경호 L1·L3·L6, 캔버스 H=600): 기본
   * 뷰에서 발의 캔버스 y가 976·1063 / 697 / 786으로 전부 밖이었다. 그 사실을 화면이 말하지 않으면
   * 바닥 맞춤 고지가 가리키는 것을 아무도 못 본다. 어느 프리셋에서 보이는지는 **말로 정하지 않고**
   * 각 프리셋의 카메라를 실제로 풀어서(`solveFraming`) 투영해 확인한다. */
  function describeFeetFraming() {
    if (!lastView) return "";
    const feet = placeholderFeetPoints(lastView);
    if (feet.length === 0) return "";
    if (feet.every((p) => onScreenNow(p))) return "";
    const shows = PRESET_PROBES.map((probe) => {
      const solved = solvePreset(probe);
      if (!solved) return null;
      return feet.every((p) => insideFrustum(p, solved)) ? probe.label : null;
    }).filter((name) => name !== null);
    const where = shows.length === 0 ? "" : ` — ${shows.map((n) => `'${n}'`).join(" · ")}에서 보입니다`;
    // 바닥에 맞춘 레벨에서만 "바닥에 맞춘"이라고 부른다. 메시가 없는 레벨(2)이나 배율 한계로 맞추지
    // 않은 패스는 리그 길이로 내린 발이다 — 맞추지 않은 것을 맞췄다고 부르면 출처와 다른 라벨이 된다.
    const what = lastView?.fit?.legs?.placeholder_snapped === true ? "바닥에 맞춘 자리표시 발" : "자리표시 발";
    return `${what}은 ${nowWord()} 화면 밖에 있습니다${where}.`;
  }

  /** 지금 화면이 프리셋 그대로인지에 따라 고지가 자기를 부르는 말. 드래그·휠로 카메라를 만진 뒤에
   * "이 프리셋에서"라고 말하면 프리셋이 방어해 주지 못한다 — 사람이 바꾼 것은 프리셋이 아니라 화면이다. */
  function nowWord() {
    return framing.auto ? "이 프리셋에서" : "지금";
  }

  function describeFraming(mode) {
    return [describeTorsoFraming(mode), describeFeetFraming()].filter((t) => t).join(" ");
  }

  function onScreenNow(p) {
    const { w, h } = hostSize();
    const px = toPixel(new THREE.Vector3(...p));
    return !(px.z > 1 || px.x < 0 || px.y < 0 || px.x > w || px.y > h);
  }

  /** `solveFraming`이 낸 카메라(target·offsetDir·radius)로 점을 투영해 절두체 안인지 본다.
   * `camera`를 실제로 움직이지 않고 **가정한** 프리셋을 재는 길이다. */
  function insideFrustum(p, solved) {
    const basis = cameraBasis(solved.offsetDir);
    const position = solved.target.clone().addScaledVector(solved.offsetDir, solved.radius);
    const v = new THREE.Vector3(...p).sub(position);
    const z = v.dot(basis.forward);
    if (z <= camera.near) return false;
    const tanV = Math.tan((camera.fov * Math.PI) / 360);
    const x = v.dot(basis.right) / (z * tanV * camera.aspect);
    const y = v.dot(basis.up) / (z * tanV);
    return Math.abs(x) <= 1 && Math.abs(y) <= 1;
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

  /** 카메라 거리 d에서 주어진 점들을 투영했을 때 max(높이비, 폭비). 뒤로 넘어가면 Infinity. */
  /** 카메라 오프셋 방향에서 (전방, 오른쪽, 위) 축. `applyCamera`의 lookAt(up=(0,1,0))과 같은 기저다. */
  function cameraBasis(offsetDir) {
    const forward = offsetDir.clone().negate();
    const right = new THREE.Vector3().crossVectors(forward, new THREE.Vector3(0, 1, 0));
    if (right.lengthSq() < 1e-12) right.set(1, 0, 0);
    right.normalize();
    const up = new THREE.Vector3().crossVectors(right, forward).normalize();
    return { forward, right, up };
  }

  function projectedRatio(corners, target, offsetDir, distance) {
    const position = target.clone().addScaledVector(offsetDir, distance);
    const { forward, right, up } = cameraBasis(offsetDir);
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

  /** AABB 중심을 보고, 세로·가로 FOV 모두에서 목표 비율이 되는 거리를 이분법으로 찾는다.
   *
   * 카메라를 **움직이지 않고** 답만 낸다 — `frameTo`가 적용하고, 발이 어느 프리셋에서 보이는지 재는
   * `describeFeetFraming`은 적용하지 않은 채로 쓴다(가정한 프리셋을 재는 데 화면을 흔들면 안 된다). */
  function solveFraming(mode, theta, phi) {
    if (!lastView) return null;
    const points = framingPoints(lastView, mode);
    if (points.length === 0) return null;
    const { min, max } = boundsOf(points);
    const target = new THREE.Vector3((min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2);
    const corners = [];
    for (const x of [min[0], max[0]]) for (const y of [min[1], max[1]]) for (const z of [min[2], max[2]]) {
      corners.push(new THREE.Vector3(x, y, z));
    }
    const sinPhi = Math.sin(phi);
    const offsetDir = new THREE.Vector3(
      sinPhi * Math.sin(theta),
      Math.cos(phi),
      sinPhi * Math.cos(theta),
    ).normalize();
    // pose(자세 맞춤·기본)와 torso(상체 확대)는 **실제 점**에 맞춘다. AABB 8꼭짓점에 맞추면 몸 앞뒤로
    // 부풀린 상자의 가까운 모서리가 크게 잡혀, 화면에 실제로 보이는 것보다 멀리 물러선다(L6에서 상체가
    // 59%만 차지해야 할 거리를 75%로 세어 뒤로 물러났다). torso에서도 같은 일이 일어나 **확대 프리셋이
    // 기본보다 작아졌다** — 실측: L3 상체 비율 0.6692(실제 점) vs 0.4864(꼭짓점), 기본이 0.5827이다.
    // 궤적·벽 프리셋은 점이 많아 예전대로 꼭짓점을 쓴다.
    const fitPoints =
      mode === "pose" || mode === "torso" ? points.map((p) => new THREE.Vector3(...p)) : corners;
    const goal = FRAMING_TARGET[mode] ?? 0.75;
    let lo = 0.01;
    let hi = 100;
    for (let i = 0; i < 60; i += 1) {
      const mid = (lo + hi) / 2;
      if (projectedRatio(fitPoints, target, offsetDir, mid) > goal) lo = mid;
      else hi = mid;
    }
    return { target, offsetDir, radius: hi, ratio: projectedRatio(fitPoints, target, offsetDir, hi) };
  }

  function frameTo(mode) {
    const solved = solveFraming(mode, camState.theta, camState.phi);
    if (!solved) return;
    camState.target.copy(solved.target);
    camState.radius = solved.radius;
    framing.mode = mode;
    framing.auto = true;
    framing.ratio = solved.ratio;
    applyCamera();
    // 자른 사실은 카메라를 적용한 **뒤에** 잰다 — 투영이 새 카메라를 써야 이름이 맞는다.
    writeFramingNote();
  }

  /** 고지를 **지금 카메라로** 다시 쓴다.
   *
   * 예전에는 `frameTo`에서만 썼다. 그래서 사람이 드래그·휠로 카메라를 만지면 문구가 그 자리에 굳어
   * 거짓이 됐다 — 실측(L3 '자세 맞춤', 호스트 1094×600): 캔버스 가운데를 위로 140px 끌고 휠을 12회
   * 내리면 두 발이 화면 안(LeftFoot (482.6,323.7) · RightFoot (425.5,340.9))인데 고지는 그대로
   * "…화면 밖에 있습니다"였다. 3D 뷰를 돌려 보는 것은 데모 관람자가 가장 먼저 하는 동작이다.
   * 그래서 카메라가 바뀌는 **모든** 길(프리셋·드래그·휠·창 크기)이 이 함수를 지난다. */
  function writeFramingNote() {
    if (!lastView) return;
    const text = describeFraming(framing.mode);
    dom.framingNote.textContent = text;
    if (container.__view3dDebug) {
      // 프리셋이 푼 비율은 그 카메라에서만 참이다. 사람이 카메라를 만진 뒤에는 내놓지 않는다 —
      // 지금 화면의 상체 비율은 아래 `upperBody`가 live 카메라로 잰다.
      container.__view3dDebug.framingRatio = framing.auto ? framing.ratio : null;
      container.__view3dDebug.framingMode = framing.mode;
      container.__view3dDebug.framingAuto = framing.auto;
      container.__view3dDebug.framingOffscreen = framing.mode === "torso" ? offscreenFitTargets() : [];
      container.__view3dDebug.framingNote = text;
      container.__view3dDebug.torsoSpan = torsoScreenSpan();
      container.__view3dDebug.upperBody = upperBodyScreen();
      // D3 검증용: 바닥에 맞춘 자리표시 발이 지금 보이는가, 어느 프리셋이 보여 주는가.
      const feet = placeholderFeetPoints(lastView);
      container.__view3dDebug.feetOnScreen = feet.length > 0 && feet.every((q) => onScreenNow(q));
      container.__view3dDebug.feetPresets = feet.length === 0
        ? []
        : PRESET_PROBES.map((probe) => {
            const solved = solvePreset(probe);
            return solved && feet.every((q) => insideFrustum(q, solved)) ? probe.label : null;
          }).filter((n) => n !== null);
    }
  }

  // 드래그는 한 프레임에 여러 번 들어온다. 고지 다시 쓰기를 한 프레임에 한 번으로 모은다.
  let framingNotePending = false;
  function scheduleFramingNote() {
    if (framingNotePending) return;
    framingNotePending = true;
    requestAnimationFrame(() => {
      framingNotePending = false;
      writeFramingNote();
    });
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
    legs: dom.controls.querySelector("#view3dLayerLegs"),
  };
  checkboxes.query.addEventListener("change", () => { layerGroups.query.visible = checkboxes.query.checked; });
  /** 조상까지 모두 보이는 뼈 이름(검증용). three.js는 부모가 숨으면 자식도 그리지 않는다. */
  function visibleBoneNames() {
    const names = [];
    layerGroups.query.traverseVisible((child) => {
      if (child.userData.kind === "bone") names.push(child.userData.name);
    });
    return layerGroups.query.visible ? names : [];
  }
  function visiblePlaceholderBones() {
    const names = [];
    layerGroups.query.traverseVisible((child) => {
      if (child.userData.kind === "placeholder-bone") names.push(child.userData.name);
    });
    return layerGroups.query.visible ? names : [];
  }
  let skeletonPlaceholder = null;
  function applyLegsToggle() {
    if (skeletonLegs) skeletonLegs.visible = checkboxes.legs.checked;
    if (skeletonPlaceholder) skeletonPlaceholder.visible = !checkboxes.legs.checked;
  }
  checkboxes.legs.addEventListener("change", () => {
    applyLegsToggle();
    if (lastView && framing.auto) frameTo(framing.mode);
    if (container.__view3dDebug) {
      container.__view3dDebug.legsVisible = checkboxes.legs.checked;
      container.__view3dDebug.visibleBones = visibleBoneNames();
      container.__view3dDebug.visiblePlaceholderBones = visiblePlaceholderBones();
    }
  });
  checkboxes.reference.addEventListener("change", () => { layerGroups.reference.visible = checkboxes.reference.checked; });
  checkboxes.corrected.addEventListener("change", () => {
    layerGroups.corrected.visible = checkboxes.corrected.checked;
    dom.correctedNote.textContent = checkboxes.corrected.checked
      ? "교정 후 층은 관측이 아니라 발부된 교정을 따랐을 때의 예상입니다 — 끊어 그린 선과 속 빈 표식으로 그립니다."
      : "";
    if (container.__view3dDebug) container.__view3dDebug.correctedLayerIsOutline = computedLayerIsOutline();
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
  dom.controls.querySelector("#view3dPresetTorso").addEventListener("click", () => {
    if (!lastView) return;
    frameTo("torso");
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
    skeletonLegs = null;
    skeletonPlaceholder = null;
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

    // 마지막 인자는 "이 층의 점이 실측인가". 질의·기준은 같은 세 강체의 실측 궤적이고, 교정 후 층은
    // 발부된 교정을 따랐을 때의 **계산값**이다 — 범례가 실측에 준 부호(층 색으로 채운 구체)를 쓰면 안 된다.
    layerGroups.query.add(buildLayerGroup(view.layers?.query, weights, LAYER_COLOR.query, !hasSkeleton, true));
    layerGroups.reference.add(buildLayerGroup(view.layers?.reference, weights, LAYER_COLOR.reference, false, true));
    layerGroups.corrected.add(buildLayerGroup(view.layers?.corrected, weights, LAYER_COLOR.corrected, false, false));
    const placeholder = fit?.legs?.placeholder_joints
      ? { joints: fit.legs.placeholder_joints, bones: fit.legs.placeholder_bones }
      : null;
    const skeletonParts = buildSkeletonGroup(joints, view.layers?.query?.bones, fit?.faded === true, fit?.legs?.joints, placeholder);
    layerGroups.query.add(skeletonParts.group);
    skeletonLegs = skeletonParts.legs;
    skeletonPlaceholder = skeletonParts.placeholderLegs;
    // 다리 기본 표시는 서버 판정을 따른다: `legs_visible_default`면 IK 다리, 아니면 자리표시 다리(있을 때).
    const legsKnown = Array.isArray(fit?.legs?.joints);
    checkboxes.legs.disabled = !legsKnown;
    checkboxes.legs.checked = legsKnown ? fit.legs_visible_default === true : true;
    applyLegsToggle();
    // 질의 착탄 궤적: 원본 착탄점(벽 표면 위 실측).
    const surfaceHits = (view.surface_hits ?? []).filter(isVec3).map((p) => new THREE.Vector3(...p));
    if (surfaceHits.length > 1) {
      const trace = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(surfaceHits),
        new THREE.LineBasicMaterial({ color: LAYER_COLOR.query, transparent: true, opacity: 0.45 }),
      );
      trace.userData.kind = "surface-hit-trace";
      layerGroups.query.add(trace);
    }
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
    dom.notesMore.replaceChildren();
    const moreLines = [];
    if (fit) {
      // 전부 싣는다. 서버가 접으라고 한 고지(`notes_collapsed`, 이름은 `note_keys`)만 한 번 눌러 펼치는 칸에 둔다 —
      // 캔버스 아래 고지가 벽처럼 쌓여 3D를 밀어냈다(2라운드). 짝짓기는 위치가 아니라 이름이다.
      const keys = Array.isArray(fit.note_keys) ? fit.note_keys : [];
      const collapsed = new Set(Array.isArray(fit.notes_collapsed) ? fit.notes_collapsed : []);
      (fit.notes ?? []).forEach((text, i) => {
        if (collapsed.has(keys[i])) {
          moreLines.push(text);
          return;
        }
        const item = document.createElement("li");
        item.textContent = text;
        dom.notes.append(item);
      });
    } else if (typeof view.fit_note === "string") {
      const item = document.createElement("li");
      item.textContent = view.fit_note;
      dom.notes.append(item);
    }
    if (mesh && mesh.available !== true && typeof mesh.reason_detail === "string") moreLines.push(mesh.reason_detail);
    if (moreLines.length > 0) {
      const summary = document.createElement("summary");
      summary.textContent = `고지 ${moreLines.length}건 더 보기`;
      const list = document.createElement("ul");
      for (const text of moreLines) {
        const item = document.createElement("li");
        item.textContent = text;
        list.append(item);
      }
      // 닫힌 <details> 안의 목록도 Chromium은 (0,0) 근처에 사각형을 남긴다 — 닫혀 있을 때는 목록 자체를 숨긴다.
      list.hidden = !dom.notesMore.open;
      dom.notesMore.ontoggle = () => { list.hidden = !dom.notesMore.open; };
      dom.notesMore.append(summary, list);
    }
    dom.notesMore.hidden = moreLines.length === 0;
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
      offsetLabels: anchors.counts.offsetLabels,
      legsVisible: checkboxes.legs.checked,
      legsToggleDisabled: checkboxes.legs.disabled,
      visibleBones: visibleBoneNames(),
      visiblePlaceholderBones: visiblePlaceholderBones(),
      placeholderBoneLengths: (() => {
        const out = {};
        layerGroups.query.traverse((child) => {
          if (child.userData.kind === "placeholder-bone") out[child.userData.name] = child.userData.length;
        });
        return out;
      })(),
      notesCollapsed: moreLines,
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
    dom.notesMore.replaceChildren();
    dom.notesMore.hidden = true;
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
