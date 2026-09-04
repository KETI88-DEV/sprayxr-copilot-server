// 3D 뷰. compare()가 낸 `view` 블록(커밋 a983ddc, task-23-report.md)만 그린다 —
// 지표를 브라우저에서 다시 계산하지 않는다(app.js 머리말과 같은 규칙).
//
// 이 화면의 주인공은 교정이다(태스크 브리프 §1) — 자세 두 개를 나란히 그리는 것이
// 목적이 아니라 발부된 교정이 무엇이고 그것을 따르면 자세가 어떻게 되는지가
// 보여야 한다. 그래서 화살표(`view.corrections`)가 항상 그려지고, 관절 마커·궤적은
// 그 교정이 어디서 나왔는지 보여줄 문맥으로 존재한다.
//
// 강조는 `view.weights`에서만 나온다(브리프 §3) — 부위 이름으로 분기하지 않는다.
// 이 파일 안에서 이름(`head`/`right`/`left`)을 쓰는 곳은 그 점의 궤적·가중치를
// 사전에서 찾는 자리뿐이고, 이름별로 다른 색·모양 규칙을 얹지 않는다.
//
// three.js는 cdnjs에서 버전 고정으로 이 스크립트보다 먼저 실린다(index.html).
// 못 실리면(오프라인 등) `createSkeletonView`가 아무것도 던지지 않고 조용한 안내만
// 남긴다 — 숫자 패널은 이 모듈과 무관하게 그대로 동작해야 한다(브리프 §3, 요구사항 3).
//
// 스키마가 실제로 담고 있는 것: 층마다 head/right/left/hit 네 점(각각 격자점 배열)
// 뿐이다 — 설계 문서(§2)가 그리는 "전신 FK 스켈레톤"은 아직 compare() 응답에 없다
// (`skeleton.py`가 그 계산을 갖고 있지만 `view_block`에 배선되지 않았다). 그래서 이
// 렌더러는 네 점의 궤적·마커만 그린다. 없는 것을 그린 척하지 않는다.

const POINT_LABELS = { right: "오른손", head: "머리", left: "왼손", hit: "착탄점" };
const AXIS_LABELS = { wall: "벽", vertical: "수직", lateral: "진행축" };

// 층별 기본색. σ 3단(ok/warn/bad)과 헷갈리지 않도록 그 팔레트와 겹치지 않는 색을 쓴다.
const LAYER_COLOR = { query: 0x4c8dff, reference: 0xa374db, corrected: 0x35c9c9 };
const MESH_COLOR = 0x9aa5b1;

const SIGMA_OK = 0x2f855a;
const SIGMA_WARN = 0xb7791f;
const SIGMA_BAD = 0xc53030;
const SIGMA_UNKNOWN = 0x718096;

/** 브리프 §3의 강조 함수. 그대로 옮긴다 — 색·굵기·마커·라벨 전부 이 하나의 함수다. */
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

function sigmaColor(sigma) {
  // 근거가 없으면(값이 숫자가 아니면) 회색이다 — 없는 σ를 0σ(초록)로 그리면
  // "근거가 강하다"는 거짓 인상을 준다.
  if (typeof sigma !== "number") return SIGMA_UNKNOWN;
  if (sigma <= 1) return SIGMA_OK;
  if (sigma <= 2) return SIGMA_WARN;
  return SIGMA_BAD;
}

function formatMagnitude(magnitudeCm) {
  // 0도 값이다 — `??`/`typeof` 없이 `||`를 쓰면 진짜 0cm 교정이 "값 없음"으로 보인다.
  return typeof magnitudeCm === "number"
    ? `${magnitudeCm >= 0 ? "+" : ""}${magnitudeCm.toFixed(1)}cm`
    : "—";
}

function correctionLabel(correction) {
  const point = POINT_LABELS[correction.point] ?? correction.point;
  const axis = AXIS_LABELS[correction.axis] ?? correction.axis;
  const mag = formatMagnitude(correction.magnitude_cm);
  const direction = correction.direction ? ` ${correction.direction}` : "";
  return `${point} · ${axis} ${mag}${direction}`;
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

  const head = document.createElement("div");
  head.className = "view3d-head";
  const title = document.createElement("h2");
  title.textContent = "3D 뷰 — 교정";
  const controls = document.createElement("div");
  controls.className = "view3d-controls";
  controls.innerHTML = `
    <label><input id="view3dLayerQuery" type="checkbox" checked> 질의</label>
    <label><input id="view3dLayerReference" type="checkbox" checked> 레퍼런스</label>
    <label><input id="view3dLayerCorrected" type="checkbox"> 교정 후(예상)</label>
    <label><input id="view3dLayerWall" type="checkbox" checked> 벽면</label>
    <button id="view3dPresetFront" type="button">벽 정면</button>
    <button id="view3dPresetSide" type="button">벽 측면</button>
  `;
  head.append(title, controls);

  const host = document.createElement("div");
  host.className = "view3d-host";
  const canvas = document.createElement("canvas");
  canvas.className = "view3d-canvas";
  const labelsHost = document.createElement("div");
  labelsHost.className = "view3d-labels";
  host.append(canvas, labelsHost);

  const legend = document.createElement("div");
  legend.className = "view3d-legend";
  legend.innerHTML = `
    <span class="legend-title">화살표 색 — 근거 σ</span>
    <span class="legend-item"><i class="swatch ok"></i>≤1σ</span>
    <span class="legend-item"><i class="swatch warn"></i>≤2σ</span>
    <span class="legend-item"><i class="swatch bad"></i>&gt;2σ</span>
  `;

  const note = document.createElement("p");
  note.className = "view3d-note";
  const correctedNote = document.createElement("p");
  correctedNote.className = "view3d-note view3d-corrected-note";

  container.append(head, host, legend, note, correctedNote);
  return { canvas, host, labelsHost, note, correctedNote, controls };
}

function showUnavailable(container, message) {
  buildDom(container);
  const note = container.querySelector(".view3d-note");
  note.textContent = message;
  container.querySelector(".view3d-controls").hidden = true;
  container.querySelector(".view3d-host").hidden = true;
  container.querySelector(".view3d-legend").hidden = true;
}

/** 벽 법선의 최선 추정. 카메라 프리셋 편의를 위한 것이지 측정값이 아니다.
 * 검증된 메시가 있으면 첫 삼각형의 법선을 쓰고(그 편이 더 믿을 만하다), 없으면
 * 착탄점들의 퍼짐에서 대략 추정한다. 둘 다 안 되면 null — 프리셋 버튼은 이때
 * 조용히 아무 일도 하지 않는다(측정 못한 방향을 지어내지 않는다). */
function estimateWallNormal(THREE, view) {
  const mesh = view.mesh;
  if (mesh && mesh.available === true && mesh.triangles?.length > 0 && mesh.vertices?.length > 0) {
    const [i0, i1, i2] = mesh.triangles[0];
    const a = new THREE.Vector3(...mesh.vertices[i0]);
    const b = new THREE.Vector3(...mesh.vertices[i1]);
    const c = new THREE.Vector3(...mesh.vertices[i2]);
    const normal = new THREE.Vector3().subVectors(b, a).cross(new THREE.Vector3().subVectors(c, a));
    if (normal.lengthSq() > 1e-12) return normal.normalize();
  }
  const hits = [
    ...(view.layers?.query?.hit ?? []),
    ...(view.layers?.reference?.hit ?? []),
  ].filter((p) => p !== null);
  if (hits.length >= 3) {
    const centroid = hits.reduce((acc, p) => acc.add(new THREE.Vector3(...p)), new THREE.Vector3()).multiplyScalar(1 / hits.length);
    let farthest = null;
    let farthestDist = -1;
    for (const p of hits) {
      const v = new THREE.Vector3(...p);
      const d = v.distanceTo(centroid);
      if (d > farthestDist) { farthestDist = d; farthest = v; }
    }
    if (farthest) {
      const v1 = new THREE.Vector3().subVectors(farthest, centroid);
      // v1과 세계 up이 이루는 평면에서 두 번째 축을 뽑아 법선을 얻는다.
      const up = new THREE.Vector3(0, 1, 0);
      const v2 = new THREE.Vector3().crossVectors(v1, up);
      const normal = new THREE.Vector3().crossVectors(v1, v2);
      if (normal.lengthSq() > 1e-12) return normal.normalize();
    }
  }
  return null;
}

function mount(THREE, container) {
  const dom = buildDom(container);
  dom.note.textContent = "패스를 고르고 비교를 누르면 여기에 뜹니다.";

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(50, 1, 0.02, 100);
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ canvas: dom.canvas, antialias: true });
  } catch (err) {
    // WebGL이 없는 브라우저/샌드박스에서도 여기서 멈추고 숫자 패널은 산다.
    showUnavailable(container, "이 브라우저에서 WebGL을 쓸 수 없어 3D 뷰를 그리지 않습니다.");
    return { render() {}, clear() {} };
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  scene.background = new THREE.Color(0x11151d);

  scene.add(new THREE.AmbientLight(0xffffff, 0.65));
  const key = new THREE.DirectionalLight(0xffffff, 0.9);
  key.position.set(2, 3, 2);
  scene.add(key);

  const grid = new THREE.GridHelper(6, 24, 0x394252, 0x262d3a);
  scene.add(grid);

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

  const labelRegistry = [];
  function clearLabels() {
    for (const entry of labelRegistry) entry.el.remove();
    labelRegistry.length = 0;
  }
  function registerLabel(position, text, extraClass) {
    const el = document.createElement("div");
    el.className = extraClass ? `view3d-label ${extraClass}` : "view3d-label";
    el.textContent = text;
    dom.labelsHost.appendChild(el);
    labelRegistry.push({ position: position.clone(), el });
  }

  // 카메라 상태(구면좌표) — 드래그 회전·휠 확대가 이 값만 바꾼다.
  const camState = { target: new THREE.Vector3(0, 1, 0), radius: 3, theta: Math.PI / 4, phi: Math.PI / 2.6 };
  function applyCamera() {
    const sinPhi = Math.sin(camState.phi);
    camera.position.set(
      camState.target.x + camState.radius * sinPhi * Math.sin(camState.theta),
      camState.target.y + camState.radius * Math.cos(camState.phi),
      camState.target.z + camState.radius * sinPhi * Math.cos(camState.theta),
    );
    camera.up.set(0, 1, 0);
    camera.lookAt(camState.target);
    // 디버그·검증 전용(§5) — 드래그 회전·휠 확대가 실제로 카메라 상태를 바꾸는지
    // 자동으로 확인할 수 있게 구면좌표를 그대로 노출한다. 렌더링에는 영향이 없다.
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
    applyCamera();
  });
  dom.canvas.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      const factor = event.deltaY > 0 ? 1.1 : 1 / 1.1;
      camState.radius = clamp(camState.radius * factor, 0.5, 12);
      applyCamera();
    },
    { passive: false },
  );

  function resize() {
    const width = dom.host.clientWidth || 1;
    const height = dom.host.clientHeight || 1;
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }
  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(dom.host);
  resize();

  function frame() {
    requestAnimationFrame(frame);
    renderer.render(scene, camera);
    const halfW = dom.host.clientWidth / 2;
    const halfH = dom.host.clientHeight / 2;
    for (const entry of labelRegistry) {
      const projected = entry.position.clone().project(camera);
      if (projected.z > 1 || projected.z < -1) {
        entry.el.style.display = "none";
        continue;
      }
      entry.el.style.display = "";
      entry.el.style.left = `${halfW + projected.x * halfW}px`;
      entry.el.style.top = `${halfH - projected.y * halfH}px`;
    }
  }
  requestAnimationFrame(frame);

  function buildLayerGroup(layerData, weights, layerColor, labelPointNames) {
    const group = new THREE.Group();
    for (const name of Object.keys(layerData ?? {})) {
      const weight = typeof weights?.[name] === "number" ? weights[name] : 0;
      const emph = emphasis(weight);
      const points = (layerData[name] ?? [])
        .filter((p) => p !== null)
        .map((p) => new THREE.Vector3(p[0], p[1], p[2]));
      if (points.length === 0) continue;

      const lineMaterial = new THREE.LineBasicMaterial({
        color: layerColor,
        transparent: true,
        opacity: emph.opacity,
      });
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      group.add(new THREE.Line(geometry, lineMaterial));

      if (emph.showMarker) {
        const markerRadius = 0.006 * emph.width;
        const markerGeometry = new THREE.SphereGeometry(markerRadius, 12, 8);
        const markerMaterial = new THREE.MeshStandardMaterial({
          color: layerColor,
          transparent: true,
          opacity: emph.opacity,
          roughness: 0.6,
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

  function buildArrow(from, to, color, magnitudeAbs) {
    const start = new THREE.Vector3(...from);
    const end = new THREE.Vector3(...to);
    const dir = new THREE.Vector3().subVectors(end, start);
    const length = dir.length();
    if (length < 1e-6) return null;
    dir.normalize();

    const mag = magnitudeAbs ?? 0;
    const shaftRadius = clamp(0.004 + 0.0025 * mag, 0.004, 0.028);
    const headRadius = shaftRadius * 2.2;
    const headLength = clamp(0.03 + 0.01 * mag, 0.03, Math.max(0.03, length * 0.6));
    const shaftLength = Math.max(length - headLength, length * 0.4);

    const material = new THREE.MeshStandardMaterial({ color, roughness: 0.45, metalness: 0.05 });
    const quat = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);

    const group = new THREE.Group();
    const shaft = new THREE.Mesh(new THREE.CylinderGeometry(shaftRadius, shaftRadius, shaftLength, 10), material);
    shaft.quaternion.copy(quat);
    shaft.position.copy(start).addScaledVector(dir, shaftLength / 2);
    const cone = new THREE.Mesh(new THREE.ConeGeometry(headRadius, headLength, 12), material);
    cone.quaternion.copy(quat);
    cone.position.copy(start).addScaledVector(dir, shaftLength + headLength / 2);
    group.add(shaft, cone);
    return group;
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
    const material = new THREE.MeshStandardMaterial({
      color: MESH_COLOR,
      transparent: true,
      opacity: 0.35,
      side: THREE.DoubleSide,
      roughness: 0.9,
    });
    return new THREE.Mesh(geometry, material);
  }

  function wallNote(meshBlock) {
    if (meshBlock === null || meshBlock === undefined) {
      return "이 패스는 레벨 정보가 없어 벽면을 그리지 않습니다.";
    }
    if (meshBlock.available !== true) {
      return "이 레벨은 검증된 벽면 정렬이 없어 벽면을 그리지 않습니다.";
    }
    return "";
  }

  let lastView = null;
  let hasFramed = false;
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
    const normal = estimateWallNormal(THREE, lastView);
    if (!normal) {
      dom.note.textContent = "벽 방향을 추정할 수 없어 프리셋을 적용하지 않았습니다.";
      return;
    }
    camState.theta = Math.atan2(normal.x, normal.z) + Math.PI + offset;
    camState.phi = Math.PI / 2.3;
    applyCamera();
  }
  dom.controls.querySelector("#view3dPresetFront").addEventListener("click", () => applyPreset(0));
  dom.controls.querySelector("#view3dPresetSide").addEventListener("click", () => applyPreset(Math.PI / 2));

  function render(payload) {
    const view = payload?.view;
    if (!view) {
      clear("표시할 뷰 데이터가 없습니다.");
      return;
    }
    lastView = view;
    dom.note.textContent = "";

    for (const group of Object.values(layerGroups)) {
      disposeObject3D(group);
      group.clear();
    }
    disposeObject3D(correctionsGroup);
    correctionsGroup.clear();
    clearLabels();
    if (meshObject) {
      scene.remove(meshObject);
      disposeObject3D(meshObject);
      meshObject = null;
    }

    const weights = view.weights ?? {};
    layerGroups.query.add(buildLayerGroup(view.layers?.query, weights, LAYER_COLOR.query, true));
    layerGroups.reference.add(buildLayerGroup(view.layers?.reference, weights, LAYER_COLOR.reference, false));
    layerGroups.corrected.add(buildLayerGroup(view.layers?.corrected, weights, LAYER_COLOR.corrected, false));
    layerGroups.query.visible = checkboxes.query.checked;
    layerGroups.reference.visible = checkboxes.reference.checked;
    layerGroups.corrected.visible = checkboxes.corrected.checked;

    for (const correction of view.corrections ?? []) {
      const color = sigmaColor(correction.sigma);
      const magnitudeAbs = typeof correction.magnitude_cm === "number" ? Math.abs(correction.magnitude_cm) : null;
      const arrow = buildArrow(correction.from, correction.to, color, magnitudeAbs);
      if (arrow) correctionsGroup.add(arrow);
      registerLabel(new THREE.Vector3(...correction.to), correctionLabel(correction), "view3d-label-arrow");
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
    const note = wallNote(mesh);
    dom.note.textContent = note;
    if (!checkboxes.corrected.checked) dom.correctedNote.textContent = "";

    if (!hasFramed) {
      const queryPoints = Object.values(view.layers?.query ?? {})
        .flat()
        .filter((p) => p !== null);
      if (queryPoints.length > 0) {
        const centroid = queryPoints.reduce(
          (acc, p) => [acc[0] + p[0], acc[1] + p[1], acc[2] + p[2]],
          [0, 0, 0],
        ).map((v) => v / queryPoints.length);
        camState.target.set(centroid[0], centroid[1], centroid[2]);
        applyCamera();
      }
      hasFramed = true;
    }

    // 실제로 그려진 객체 수를 그대로 노출한다(디버그·검증 전용 — three.js 씬 그래프를
    // 다시 세는 게 아니라 렌더링이 실제로 만든 그 객체들을 센다). 화면을 눈으로 볼 수
    // 없는 자동 점검(§5)에서 "실제로 몇 개가 그려졌는가"를 씬을 다시 파싱하지 않고
    // 확인하는 용도다. 렌더링 동작 자체에는 아무 영향이 없다.
    const markerCount = ["query", "reference", "corrected"].reduce((sum, key) => {
      let count = 0;
      layerGroups[key].traverse((child) => {
        if (child.geometry && child.geometry.type === "SphereGeometry") count += 1;
      });
      return sum + count;
    }, 0);
    container.__view3dDebug = {
      arrowCount: correctionsGroup.children.length,
      markerCount,
      meshTriangleCount: meshObject ? meshObject.geometry.index.count / 3 : 0,
      labelCount: labelRegistry.length,
      labels: labelRegistry.map((entry) => entry.el.textContent),
    };
  }

  function clear(message) {
    for (const group of Object.values(layerGroups)) {
      disposeObject3D(group);
      group.clear();
    }
    disposeObject3D(correctionsGroup);
    correctionsGroup.clear();
    clearLabels();
    if (meshObject) {
      scene.remove(meshObject);
      disposeObject3D(meshObject);
      meshObject = null;
    }
    lastView = null;
    dom.note.textContent = message ?? "패스를 고르고 비교를 누르면 여기에 뜹니다.";
    dom.correctedNote.textContent = "";
    container.__view3dDebug = null;
  }

  return { render, clear };
}

/** 3D 뷰를 `container`에 붙인다. three.js가 없거나 초기화가 실패하면 조용한 안내만
 * 남기고 던지지 않는다 — 숫자 패널은 이 호출과 무관하게 그대로 동작해야 한다. */
export function createSkeletonView(container) {
  if (!container) return { render() {}, clear() {} };
  if (typeof THREE === "undefined") {
    showUnavailable(container, "three.js를 불러오지 못해 3D 뷰 없이 숫자 패널만 씁니다(오프라인이거나 CDN 차단).");
    return { render() {}, clear() {} };
  }
  try {
    return mount(THREE, container);
  } catch (err) {
    console.error("[skeleton_view] 3D 뷰 초기화 실패", err);
    showUnavailable(container, "3D 뷰를 초기화하지 못했습니다 — 숫자 패널은 그대로 동작합니다.");
    return { render() {}, clear() {} };
  }
}
