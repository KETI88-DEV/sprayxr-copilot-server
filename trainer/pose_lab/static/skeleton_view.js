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
const LAYER_COLOR = { query: 0x5b9bff, reference: 0xa684e0, corrected: 0x34c2c2 };
const MESH_COLOR = 0x9aa5b1;
// 스켈레톤(관절·뼈)은 강조 대상이 아니라 문맥이다 — 질의 층 색과 같은 계열이되
// 더 조용한 회색쪽으로 낮춘다. weights[관절명]이 전부 0.0이라 emphasis()를 그대로
// 쓰면 아무것도 안 보이므로(강조는 교정 채점용 점에만 쓰는 규칙), 스켈레톤은 이
// 상수 하나로 고정한 자신만의 모양을 쓴다.
const SKELETON_COLOR = 0xb9c2d1;

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
    <label><input id="view3dLayerQuery" type="checkbox" checked> 질의(스켈레톤)</label>
    <label><input id="view3dLayerReference" type="checkbox" checked> 기준</label>
    <label><input id="view3dLayerCorrected" type="checkbox"> 교정 후(예상)</label>
    <label><input id="view3dLayerWall" type="checkbox" checked> 벽면</label>
    <button id="view3dPresetFit" type="button">프레임 맞춤</button>
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
    <span class="legend-item">①②③ 번호는 옆 목록(발부된 교정)과 같은 순서다</span>
  `;

  const note = document.createElement("p");
  note.className = "view3d-note";
  const fitNote = document.createElement("p");
  fitNote.className = "view3d-fit-note";
  const correctedNote = document.createElement("p");
  correctedNote.className = "view3d-note view3d-corrected-note";

  container.append(head, host, legend, note, fitNote, correctedNote);
  return { canvas, host, labelsHost, note, fitNote, correctedNote, controls };
}

/** `note`에 문구와 상태 톤을 함께 건다 — 화면 어디서든 "아직 안 함/로딩 중/없음/오류"
 * 네 상태를 구별해야 한다(브리프 재확인 지시). 빈 문구는 `.view3d-note:empty`가
 * 조용히 숨긴다(=사실상 다섯 번째 "정상, 낼 말 없음" 상태). */
function setNote(el, text, tone) {
  el.textContent = text ?? "";
  if (tone) el.dataset.tone = tone;
  else el.removeAttribute("data-tone");
}

function showUnavailable(container, message) {
  buildDom(container);
  const note = container.querySelector(".view3d-note");
  setNote(note, message, "error");
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
  setNote(dom.note, "패스를 고르고 평가를 누르면 여기에 뜹니다.", "idle");

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(50, 1, 0.02, 100);
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ canvas: dom.canvas, antialias: true });
  } catch (err) {
    // WebGL이 없는 브라우저/샌드박스에서도 여기서 멈추고 숫자 패널은 산다.
    showUnavailable(container, "이 브라우저에서 WebGL을 쓸 수 없어 3D 뷰를 그리지 않습니다.");
    return { render() {}, clear() {}, loading() {} };
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  scene.background = new THREE.Color(0x070a0f);

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
      // "joints"(관절명 -> 위치 하나)와 "bones"(이름 쌍 목록)는 격자점 궤적이
      // 아니다 — 전신 스켈레톤(11-A)이고 `buildSkeletonGroup`이 따로 그린다.
      if (name === "joints" || name === "bones") continue;
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

  /** 전신 FK 스켈레톤(11-A) — `view.layers.query.joints`(관절명 -> 위치 하나)와
   * `bones`(부모-자식 이름 쌍)를 선분으로 잇고 관절에 작은 점을 찍는다.
   *
   * `weights`로 강조를 따지지 않는다 — 새 관절은 전부 weights에 0.0으로 실려
   * 있지만(브리프), 그 규칙은 "교정 채점에 쓰는 점"의 강조 규칙이지 구조 표시
   * 규칙이 아니다. 스켈레톤은 항상 같은 조용한 색 하나로, 있으면 그린다. */
  function buildSkeletonGroup(joints, bones) {
    const group = new THREE.Group();
    if (!joints || !bones || Object.keys(joints).length === 0) return group;

    // 실선(LineBasicMaterial)은 WebGL에서 굵기를 키울 수 없어(플랫폼 대부분이
    // linewidth를 무시한다) 아무리 진하게 칠해도 머리카락처럼 가늘게 보인다 —
    // "스켈레톤이 없음"이 다시 "있는데 안 보임"으로 재발한다. 교정 화살표
    // (`buildArrow`)와 같은 방식으로 실린더를 세워 실제 두께를 준다.
    const boneMaterial = new THREE.MeshStandardMaterial({
      color: SKELETON_COLOR,
      transparent: true,
      opacity: 0.88,
      roughness: 0.55,
    });
    const boneRadius = 0.009;
    for (const [parent, child] of bones) {
      const a = joints[parent];
      const b = joints[child];
      if (!a || !b) continue;
      const start = new THREE.Vector3(...a);
      const end = new THREE.Vector3(...b);
      const dir = new THREE.Vector3().subVectors(end, start);
      const length = dir.length();
      if (length < 1e-6) continue;
      dir.normalize();
      const quat = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
      const bone = new THREE.Mesh(new THREE.CylinderGeometry(boneRadius, boneRadius, length, 6), boneMaterial);
      bone.quaternion.copy(quat);
      bone.position.copy(start).addScaledVector(dir, length / 2);
      group.add(bone);
    }

    const jointMaterial = new THREE.MeshStandardMaterial({
      color: SKELETON_COLOR,
      transparent: true,
      opacity: 0.92,
      roughness: 0.55,
    });
    const jointGeometry = new THREE.SphereGeometry(0.015, 8, 6);
    for (const position of Object.values(joints)) {
      const marker = new THREE.Mesh(jointGeometry, jointMaterial);
      marker.position.set(position[0], position[1], position[2]);
      group.add(marker);
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

  /** 벽면의 네 상태(아직 안 함/로딩 중/없음/오류) 중 "없음"에 해당하는 문구와 톤.
   * `mesh: null`(레벨 자체를 모름)과 `mesh.available: false`(레벨은 알지만 검증된
   * 정렬이 없음)를 서로 다른 문장으로 구별한다 — 둘 다 "벽이 안 보인다"로 보이지만
   * 사유가 다르다. 어느 레벨이 검증됐는지는 하드코딩하지 않는다 — 이 문구 자체가
   * `mesh.available`을 그대로 따라간다(다른 레벨의 정렬이 검증되면 이 문구도 그때
   * 자동으로 사라진다).
   */
  function wallNote(meshBlock) {
    if (meshBlock === null || meshBlock === undefined) {
      return "이 패스는 레벨 정보가 없어 벽면을 그리지 않습니다.";
    }
    if (meshBlock.available !== true) {
      return "이 레벨은 아직 검증된 벽면 정렬이 없어 벽면을 그리지 않습니다 — 없음, 오류가 아닙니다.";
    }
    return "";
  }

  function fitNoteText(fit) {
    if (fit === null || fit === undefined) {
      return "이 프레임은 전신 스켈레톤을 맞추지 못해 점과 화살표만 표시합니다(관절 컬럼 없음 또는 앵커 부족).";
    }
    const cm = fit.residual_m * 100;
    return `스켈레톤 피팅 잔차 ${cm.toFixed(1)}cm (${fit.source})`;
  }

  /** 장면이 실제로 채우는 공간(질의·기준·교정후 궤적 + 스켈레톤 관절 + 교정 화살표
   * 끝점)의 바운딩 박스. 벽면 메시는 뺀다 — 벽은 자세보다 훨씬 커서 넣으면 자세가
   * 화면 한 귀퉁이의 점 하나로 쪼그라든다(브리프 §11-B: "벽면은 크니 자세+화살표
   * 기준으로 잡는다"). */
  function sceneBounds(view) {
    // "자세+화살표" 기준(브리프)이다 — 질의 층(스켈레톤 포함)과 교정 화살표만 쓴다.
    // 레퍼런스·교정후 층까지 넣으면(서로 다른 사람이 서 있던 자리라 물리적으로
    // 떨어져 있을 수 있다) 무게중심이 질의 자세 바깥으로 밀려, 정작 봐야 할
    // 스켈레톤이 화면 한쪽에 작게 몰린다 — 실측으로 확인한 결함이었다.
    const points = [];
    const queryLayer = view.layers?.query ?? {};
    for (const [name, value] of Object.entries(queryLayer)) {
      if (name === "bones") continue;
      if (name === "joints") {
        points.push(...Object.values(value));
        continue;
      }
      for (const p of value ?? []) if (p) points.push(p);
    }
    for (const correction of view.corrections ?? []) {
      if (correction.from) points.push(correction.from);
      if (correction.to) points.push(correction.to);
    }
    if (points.length === 0) return null;
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

  /** 카메라를 장면 바운딩 박스에 자동 맞춘다(브리프 §11-B). */
  function frameToBounds() {
    if (!lastView) return;
    const bounds = sceneBounds(lastView);
    if (!bounds) return;
    const center = [0, 1, 2].map((i) => (bounds.min[i] + bounds.max[i]) / 2);
    const size = [0, 1, 2].map((i) => Math.max(0, bounds.max[i] - bounds.min[i]));
    const diagonal = Math.hypot(size[0], size[1], size[2]);
    const half = diagonal / 2 || 0.3;
    // 카메라 FOV 절반의 tan으로 필요한 거리를 구하고, 여유(30%)를 얹는다.
    const fovHalf = (camera.fov * Math.PI) / 360;
    const distance = half / Math.max(Math.sin(fovHalf), 0.05);
    camState.target.set(center[0], center[1], center[2]);
    camState.radius = clamp(distance * 1.3, 0.4, 12);
    applyCamera();
  }

  let lastView = null;
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
      setNote(dom.note, "벽 방향을 추정할 수 없어 프리셋을 적용하지 않았습니다.", "empty");
      return;
    }
    camState.theta = Math.atan2(normal.x, normal.z) + Math.PI + offset;
    camState.phi = Math.PI / 2.3;
    applyCamera();
  }
  dom.controls.querySelector("#view3dPresetFit").addEventListener("click", () => frameToBounds());
  dom.controls.querySelector("#view3dPresetFront").addEventListener("click", () => applyPreset(0));
  dom.controls.querySelector("#view3dPresetSide").addEventListener("click", () => applyPreset(Math.PI / 2));

  /** 요청이 진행 중임을 보인다(4상태 중 "로딩 중"). 이전 장면은 그대로 흐리게 남겨
   * "멈췄다"와 "계산 중이다"를 구별한다 — 화면이 비면 그 둘이 구별되지 않는다. */
  function loading(message) {
    setNote(dom.note, message ?? "비교를 계산하는 중…", "loading");
    dom.host.classList.add("is-loading");
  }

  function render(payload) {
    dom.host.classList.remove("is-loading");
    const view = payload?.view;
    if (!view) {
      clear("표시할 뷰 데이터가 없습니다.", "empty");
      return;
    }
    lastView = view;

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
    const hasSkeleton = Object.keys(view.layers?.query?.joints ?? {}).length > 0;
    // 스켈레톤이 있으면 머리·오른손·왼손 글자 라벨을 끈다 — 관절 구조 자체가
    // 어느 점이 머리인지 몸으로 보여주고, 그 자리는 교정 번호(①②③) 배지와 겹치기
    // 딱 좋은 자리라 브리프의 "라벨 겹침을 없앤다"를 스켈레톤 유무로도 지킨다.
    // 스켈레톤이 없을 때(FK 실패)는 이 글자가 유일한 신원 단서라 그대로 켠다.
    layerGroups.query.add(buildLayerGroup(view.layers?.query, weights, LAYER_COLOR.query, !hasSkeleton));
    layerGroups.reference.add(buildLayerGroup(view.layers?.reference, weights, LAYER_COLOR.reference, false));
    layerGroups.corrected.add(buildLayerGroup(view.layers?.corrected, weights, LAYER_COLOR.corrected, false));
    // 전신 스켈레톤(11-A) — 질의 층에만 있다. 질의 체크박스에 함께 매인다.
    layerGroups.query.add(buildSkeletonGroup(view.layers?.query?.joints, view.layers?.query?.bones));
    layerGroups.query.visible = checkboxes.query.checked;
    layerGroups.reference.visible = checkboxes.reference.checked;
    layerGroups.corrected.visible = checkboxes.corrected.checked;

    // 화살표 라벨은 번호(①②③…)만 단다 — 글자는 옆 "발부된 교정" 목록에서 같은
    // 순서로 읽는다(브리프 §11-B: 겹치는 글자 라벨 대신). 목록과 순서가 어긋나면
    // 안 되므로 정렬을 다시 하지 않고 `view.corrections`가 준 순서를 그대로 쓴다.
    const numerals = ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨"];
    (view.corrections ?? []).forEach((correction, index) => {
      const color = sigmaColor(correction.sigma);
      const magnitudeAbs = typeof correction.magnitude_cm === "number" ? Math.abs(correction.magnitude_cm) : null;
      const arrow = buildArrow(correction.from, correction.to, color, magnitudeAbs);
      if (arrow) correctionsGroup.add(arrow);
      const numeral = numerals[index] ?? `#${index + 1}`;
      registerLabel(new THREE.Vector3(...correction.to), numeral, "view3d-label-arrow");
    });

    const mesh = view.mesh ?? null;
    meshObject = buildMesh(mesh);
    if (meshObject) {
      meshObject.visible = checkboxes.wall.checked;
      scene.add(meshObject);
      checkboxes.wall.disabled = false;
    } else {
      checkboxes.wall.disabled = true;
    }
    const wallText = wallNote(mesh);
    setNote(dom.note, wallText, wallText ? "empty" : undefined);
    setNote(dom.fitNote, fitNoteText(view.fit ?? null), view.fit ? undefined : "empty");
    if (!checkboxes.corrected.checked) dom.correctedNote.textContent = "";

    frameToBounds();

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
      jointCount: Object.keys(view.layers?.query?.joints ?? {}).length,
      fit: view.fit ?? null,
    };
  }

  /** `tone`: "idle" | "empty" | "error". 로딩 중이 아닌 정지 상태를 나타낸다 —
   * `loading()`과 짝이다. */
  function clear(message, tone) {
    dom.host.classList.remove("is-loading");
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
    setNote(dom.note, message ?? "패스를 고르고 평가를 누르면 여기에 뜹니다.", tone ?? "idle");
    dom.fitNote.textContent = "";
    dom.correctedNote.textContent = "";
    container.__view3dDebug = null;
  }

  return { render, clear, loading };
}

/** 3D 뷰를 `container`에 붙인다. three.js가 없거나 초기화가 실패하면 조용한 안내만
 * 남기고 던지지 않는다 — 숫자 패널은 이 호출과 무관하게 그대로 동작해야 한다. */
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
