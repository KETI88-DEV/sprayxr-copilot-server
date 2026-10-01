// 교정 시연 — ① 잘못된 자세 입력 → ② 교정 안내 → ③ 교정된 결과를 사람 몸(3D)으로 보인다.
// 숫자·문장·몸의 관절은 전부 서버(/api/demo)가 만든다. 이 파일은 그리기와 재생만 한다.
// 유사도와 몸은 교정을 0%→100% 따르는 동안 서버가 20단계로 실제 계산한 값(display.score_steps, bodies.steps)이다.
// 좌표: 서버는 유니티 월드(Y 위). 장면은 Z를 뒤집는다(skeleton_view.js의 sceneFromUnity와 같은 규칙).

const THREE = window.THREE;
const COLORS = { target: 0xa684e0, worker: 0x5b9bff, corrected: 0x34c2c2, arrow: 0xffd166, wall: 0x9aa5b1, grid: 0x8a96a8 };
const TIMING = { input: 3500, perCue: 1300, beforeResult: 1200, result: 3500 };
const DEFAULT_RADIUS = 0.03;

const $ = (id) => document.getElementById(id);
const state = { stage: 0, cuesShown: 0, finished: false, runId: 0, scenarios: [], data: null, view: "oblique" };
window.__demo = state;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const reduceMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const toScene = (p) => new THREE.Vector3(p[0], p[1], -p[2]);

function html(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

// ---- 장면 ---------------------------------------------------------------------

const host = $("demoCanvas");
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setClearColor(0x0a0d12, 1);
host.append(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, 1, 0.05, 100);
scene.add(new THREE.HemisphereLight(0xffffff, 0x223344, 1.1));
const sun = new THREE.DirectionalLight(0xffffff, 0.9);
sun.position.set(2, 4, 3);
scene.add(sun);

const orbit = { center: new THREE.Vector3(), radius: 3, azimuth: 0, elevation: 0.35, frames: null };
let world = new THREE.Group();
scene.add(world);

function resize() {
  const width = host.clientWidth || 800;
  const height = host.clientHeight || 600;
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);

function placeCamera() {
  const { center, radius, azimuth, elevation } = orbit;
  camera.position.set(
    center.x + radius * Math.cos(elevation) * Math.cos(azimuth),
    center.y + radius * Math.sin(elevation),
    center.z + radius * Math.cos(elevation) * Math.sin(azimuth),
  );
  camera.lookAt(center);
}

// 끌어서 돌리기, 휠로 확대.
let drag = null;
host.addEventListener("pointerdown", (event) => { drag = { x: event.clientX, y: event.clientY }; host.setPointerCapture(event.pointerId); });
host.addEventListener("pointermove", (event) => {
  if (!drag) return;
  orbit.azimuth += (event.clientX - drag.x) * 0.008;
  orbit.elevation = Math.max(-0.2, Math.min(1.4, orbit.elevation + (event.clientY - drag.y) * 0.006));
  drag = { x: event.clientX, y: event.clientY };
});
host.addEventListener("pointerup", () => { drag = null; });
host.addEventListener("wheel", (event) => {
  event.preventDefault();
  orbit.radius = Math.max(0.8, Math.min(12, orbit.radius * (event.deltaY > 0 ? 1.08 : 1 / 1.08)));
}, { passive: false });

function setView(name) {
  state.view = name;
  for (const button of document.querySelectorAll("[data-view]")) {
    button.setAttribute("aria-pressed", button.dataset.view === name ? "true" : "false");
  }
  if (!orbit.frames) return;
  const { away, along } = orbit.frames;
  // 카메라는 작업자 쪽(벽에서 떨어진 쪽)에 둔다. 옆에서 = 진행 방향에서, 뒤에서 = 벽을 마주 보는 작업자의 등 뒤.
  const direction = name === "side" ? along.clone()
    : name === "back" ? away.clone()
    : away.clone().multiplyScalar(0.8).add(along.clone().multiplyScalar(0.6));
  orbit.azimuth = Math.atan2(direction.z, direction.x);
  orbit.elevation = name === "back" ? 0.25 : 0.3;
}

// ---- 몸 -------------------------------------------------------------------------

function material(color, opacity) {
  return new THREE.MeshStandardMaterial({ color, roughness: 0.6, metalness: 0.0, transparent: opacity < 1, opacity, depthWrite: opacity >= 0.6 });
}

function segment(a, b, r0, r1, mat) {
  const length = a.distanceTo(b);
  if (length < 1e-6) return null;
  const mesh = new THREE.Mesh(new THREE.CylinderGeometry(r1, r0, length, 14, 1), mat);
  mesh.position.copy(a).add(b).multiplyScalar(0.5);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), b.clone().sub(a).normalize());
  return mesh;
}

function sphere(center, radius, mat) {
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(radius, 18, 12), mat);
  mesh.position.copy(center);
  return mesh;
}

/** 살을 입힌 몸 — 두께는 서버가 준 것(flesh)만 쓴다. 없는 마디는 기본 굵기. */
function buildBody(joints, data, color, opacity) {
  const group = new THREE.Group();
  const mat = material(color, opacity);
  const p = (name) => (joints[name] ? toScene(joints[name]) : null);
  const flesh = data.flesh ?? {};
  const segments = flesh.segments ?? {};
  for (const [parent, child] of data.bones) {
    const a = p(parent);
    const b = p(child);
    if (!a || !b) continue;
    const [r0, r1] = segments[child] ?? [DEFAULT_RADIUS, DEFAULT_RADIUS];
    const piece = segment(a, b, r0, r1, mat);
    if (piece) group.add(piece);
    group.add(sphere(b, r1, mat));
    group.add(sphere(a, r0, mat));
  }
  for (const bridge of flesh.bridges ?? []) {
    const a = p(bridge.from);
    const b = p(bridge.to);
    if (a && b) {
      const piece = segment(a, b, bridge.radius, bridge.radius, mat);
      if (piece) group.add(piece);
    }
  }
  for (const palm of flesh.palms ?? []) {
    const wrist = p(palm.to);
    const elbow = p(palm.from);
    if (!wrist || !elbow) continue;
    const tip = wrist.clone().add(wrist.clone().sub(elbow).normalize().multiplyScalar(palm.length));
    const piece = segment(wrist, tip, palm.radius[0], palm.radius[1], mat);
    if (piece) group.add(piece);
  }
  const skull = flesh.head;
  const head = skull ? p(skull.to) : p("Head");
  const neck = skull ? p(skull.from) ?? p(skull.fallback_from) : p("Neck");
  if (head) {
    const lift = head && neck ? head.clone().sub(neck).normalize().multiplyScalar(skull?.lift ?? 0.06) : new THREE.Vector3();
    group.add(sphere(head.clone().add(lift), skull?.radius ?? 0.09, mat));
  }
  return group;
}

function disposeGroup(group) {
  group.traverse((child) => {
    if (child.geometry) child.geometry.dispose();
    if (child.material) child.material.dispose();
  });
}

function lerpJoints(a, b, t) {
  const out = {};
  for (const name of Object.keys(a)) {
    const p = a[name];
    const q = b[name] ?? p;
    out[name] = [p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t, p[2] + (q[2] - p[2]) * t];
  }
  return out;
}

// ---- 무대(벽·바닥) ----------------------------------------------------------------

function buildStage(data) {
  const group = new THREE.Group();
  const hit = toScene(data.context.hit);
  const normal = toScene(data.context.wall_normal);
  normal.y = 0;
  normal.normalize();
  const all = Object.values(data.bodies.worker).concat(Object.values(data.bodies.target)).map(toScene);
  const floor = typeof data.context.floor === "number" ? data.context.floor : Math.min(...all.map((v) => v.y));

  // 벽: 착탄점을 지나는 수직 평면. 사람 쪽(법선 +)을 본다.
  const wall = new THREE.Mesh(
    new THREE.PlaneGeometry(4, Math.max(2.6, hit.y - floor + 1.2)),
    new THREE.MeshStandardMaterial({ color: COLORS.wall, transparent: true, opacity: 0.22, side: THREE.DoubleSide, depthWrite: false }),
  );
  wall.position.set(hit.x, floor + wall.geometry.parameters.height / 2, hit.z);
  wall.lookAt(hit.clone().add(new THREE.Vector3(normal.x, 0, normal.z).add(new THREE.Vector3(0, 0, 0))).setY(wall.position.y));
  group.add(wall);
  // 분사가 닿는 점.
  const ring = new THREE.Mesh(new THREE.RingGeometry(0.035, 0.05, 24), new THREE.MeshBasicMaterial({ color: 0xff7a1a, side: THREE.DoubleSide }));
  ring.position.copy(hit).add(normal.clone().multiplyScalar(0.005));
  ring.lookAt(hit.clone().add(normal));
  group.add(ring);
  // 바닥 격자.
  const grid = new THREE.GridHelper(6, 24, COLORS.grid, COLORS.grid);
  grid.material.transparent = true;
  grid.material.opacity = 0.2;
  const center = all.reduce((acc, v) => acc.add(v), new THREE.Vector3()).multiplyScalar(1 / all.length);
  grid.position.set(center.x, floor + 0.002, center.z);
  group.add(grid);

  const along = new THREE.Vector3().crossVectors(new THREE.Vector3(0, 1, 0), normal).normalize();
  const box = new THREE.Box3().setFromPoints(all);
  const size = box.getSize(new THREE.Vector3());
  orbit.center.copy(box.getCenter(new THREE.Vector3()));
  orbit.radius = Math.max(2.2, Math.max(size.x, size.y, size.z) * 2.4);
  orbit.frames = { away: normal.clone(), along };
  return group;
}

function arrowFor(cue) {
  const from = toScene(cue.from);
  const to = toScene(cue.to);
  const vector = to.clone().sub(from);
  const length = vector.length();
  if (length < 0.005) return null;
  const head = Math.min(0.08, length * 0.35);
  const arrow = new THREE.ArrowHelper(vector.normalize(), from, length, COLORS.arrow, head, head * 0.6);
  arrow.line.material.linewidth = 3;
  // 굵은 축: ArrowHelper의 선은 1px이라 원통을 덧댄다.
  const shaft = segment(from, from.clone().add(vector.clone().multiplyScalar(length - head)), 0.008, 0.008,
    new THREE.MeshBasicMaterial({ color: COLORS.arrow }));
  const group = new THREE.Group();
  group.add(arrow);
  if (shaft) group.add(shaft);
  group.visible = false;
  return { group, from, to };
}

// ---- 그리기 -------------------------------------------------------------------------

const live = { worker: null, ghost: null, target: null, arrows: [], badges: [] };

function render(data) {
  scene.remove(world);
  disposeGroup(world);
  world = new THREE.Group();
  scene.add(world);
  live.arrows = [];
  live.badges = [];
  $("demoBadges").replaceChildren();

  world.add(buildStage(data));
  live.target = buildBody(data.bodies.target, data, COLORS.target, 0.3);
  world.add(live.target);
  live.worker = buildBody(data.bodies.worker, data, COLORS.worker, 0.85);
  world.add(live.worker);
  for (const cue of data.cues) {
    const arrow = arrowFor(cue);
    live.arrows.push(arrow);
    if (arrow) world.add(arrow.group);
    const badge = html("span", "demo-badge", String(cue.n));
    $("demoBadges").append(badge);
    live.badges.push(badge);
  }
  setView(state.view);

  $("demoValues").textContent = `· ${data.display.values}`;
  const list = $("demoCues");
  list.replaceChildren();
  for (const cue of data.cues) {
    const item = html("li");
    item.dataset.n = String(cue.n);
    item.append(html("span", "num-badge", String(cue.n)), html("span", null, cue.text));
    list.append(item);
  }
  $("demoNote").textContent =
    "계산상 예상값입니다 — 작업자가 안내를 정확히 따랐다고 보고 계산했고, 다시 뿌려서 잰 값이 아닙니다. " +
    "유사도는 오른손(건)·머리(HMD)·왼손 세 점의 위치로 잽니다. 목표 자세는 작업자의 몸에서 이 세 점을 숙련 작업자 기준 위치로 " +
    "옮긴 모습이고, 팔꿈치·다리 모양은 추정입니다.";
}

function setWorker(joints, color, opacity) {
  if (live.worker) {
    world.remove(live.worker);
    disposeGroup(live.worker);
  }
  live.worker = buildBody(joints, state.data, color, opacity);
  world.add(live.worker);
}

function placeBadges() {
  const width = host.clientWidth;
  const height = host.clientHeight;
  live.arrows.forEach((arrow, index) => {
    const badge = live.badges[index];
    if (!arrow || !badge) return;
    // 화살표 뿌리 바로 옆 — 끝(고친 위치)을 가리지 않게.
    const point = arrow.from.clone().lerp(arrow.to, 0.15).project(camera);
    badge.style.left = `${((point.x + 1) / 2) * width - 18}px`;
    badge.style.top = `${((1 - point.y) / 2) * height - 18}px`;
  });
}

function loop() {
  placeCamera();
  placeBadges();
  renderer.render(scene, camera);
  requestAnimationFrame(loop);
}

// ---- 재생 ---------------------------------------------------------------------

function setStage(stage) {
  state.stage = stage;
  for (const item of $("demoStages").children) {
    const n = Number(item.dataset.stage);
    item.classList.toggle("is-active", n === stage);
    item.classList.toggle("is-done", n < stage);
  }
}

async function play(data, runId) {
  const alive = () => state.runId === runId;
  const fast = reduceMotion();
  render(data);
  state.finished = false;
  state.cuesShown = 0;

  setStage(1);
  $("demoScore").textContent = data.display.score_before;
  $("demoScoreSub").textContent = "";
  $("demoCaption").textContent =
    `① 입력 — ${data.query.worker} 작업자의 자세(파랑)가 목표 자세(보라)와 다릅니다. 유사도 ${data.display.score_before}.`;
  await sleep(fast ? 0 : TIMING.input);
  if (!alive()) return;

  setStage(2);
  $("demoCaption").textContent = data.cues.length
    ? `② 교정 안내 — ${data.label}: 이 안내가 나갑니다(${data.display.values}).`
    : "② 교정 안내 — 고칠 자세가 없습니다.";
  for (const cue of data.cues) {
    $("demoCues").querySelector(`[data-n="${cue.n}"]`)?.classList.add("is-shown");
    const arrow = live.arrows[cue.n - 1];
    if (arrow) arrow.group.visible = true;
    live.badges[cue.n - 1]?.classList.add("is-shown");
    state.cuesShown = cue.n;
    await sleep(fast ? 0 : TIMING.perCue);
    if (!alive()) return;
  }
  await sleep(fast ? 0 : TIMING.beforeResult);
  if (!alive()) return;

  setStage(3);
  $("demoCaption").textContent = "③ 결과 — 안내대로 고치면 몸이 이렇게 바뀝니다.";
  // 고치기 전 몸을 흐린 잔상으로 남긴다 — 어디서 어디로 옮겼는지가 끝까지 보이게.
  live.ghost = buildBody(data.bodies.worker, data, COLORS.worker, 0.22);
  world.add(live.ghost);
  const steps = data.bodies.steps;
  const scores = data.display.score_steps;
  const start = performance.now();
  await new Promise((resolve) => {
    const frame = (now) => {
      if (!alive()) return resolve();
      // rAF의 시각은 프레임 시작 시각이라 `start`보다 앞설 수 있다 — 0 아래로 내려가지 않게 묶는다.
      const t = fast ? 1 : Math.max(0, Math.min(1, (now - start) / TIMING.result));
      const position = t * (steps.length - 1);
      const index = Math.min(steps.length - 2, Math.floor(position));
      const joints = lerpJoints(steps[index], steps[index + 1], position - index);
      setWorker(joints, COLORS.worker, 0.85);
      $("demoScore").textContent = scores[Math.round(t * (scores.length - 1))];
      if (t < 1) requestAnimationFrame(frame);
      else resolve();
    };
    requestAnimationFrame(frame);
  });
  if (!alive()) return;
  setWorker(steps[steps.length - 1], COLORS.corrected, 0.9);
  $("demoScore").textContent = data.display.score_after;
  $("demoScoreSub").textContent = `${data.display.score_before} → ${data.display.score_after} · 개선율 ${data.display.improvement}`;
  $("demoCaption").textContent =
    `③ 결과 — 안내대로 고치면 유사도가 ${data.display.score_before}에서 ${data.display.score_after}까지 오릅니다. ` +
    `남은 격차를 메운 비율(개선율) ${data.display.improvement}, 교정·채점 기준을 나누어 재면 ${data.display.improvement_split}.`;
  setStage(4);
  state.finished = true;
}

function params() {
  const search = new URLSearchParams(location.search);
  const ref = (prefix) => {
    const worker = search.get(`${prefix}w`);
    if (!worker) return null;
    return {
      worker, date: search.get(`${prefix}d`) ?? "", level: search.get(`${prefix}l`) ?? "",
      session: search.get(`${prefix}s`) ?? "", pass_index: Number(search.get(`${prefix}p`) ?? "0"),
    };
  };
  return { mode: search.get("mode"), scenario: search.get("scenario"), query: ref("q"), reference: ref("r"),
    view: search.get("view"), autoplay: search.get("autoplay") !== "0" };
}

async function run() {
  const runId = ++state.runId;
  const scenario = state.scenarios[Number($("demoScenario").value)];
  if (!scenario) {
    $("demoCaption").textContent = "시연할 예시가 없습니다.";
    return;
  }
  $("demoCaption").textContent = "계산하는 중…";
  const body = { query: scenario.query, reference: scenario.reference };
  const mode = params().mode;
  if (mode) body.mode = mode;
  const response = await fetch("api/demo", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await response.json();
  if (runId !== state.runId) return;
  if (data.status !== "ok") {
    $("demoCaption").textContent = `이 쌍은 시연할 수 없습니다(${data.reason ?? "unknown"}).`;
    return;
  }
  state.data = data;
  await play(data, runId);
}

async function start() {
  resize();
  requestAnimationFrame(loop);
  const p = params();
  const response = await fetch("api/demo/scenarios");
  const body = await response.json();
  state.scenarios = body.scenarios ?? [];
  if (p.query && p.reference) {
    state.scenarios = [{ label: "도장 동작 비교에서 고른 쌍", query: p.query, reference: p.reference }, ...state.scenarios];
  }
  const select = $("demoScenario");
  state.scenarios.forEach((scenario, index) => {
    const option = html("option", null, scenario.label);
    option.value = String(index);
    select.append(option);
  });
  if (p.scenario !== null && state.scenarios[Number(p.scenario)]) select.value = String(Number(p.scenario));
  if (p.view) state.view = p.view;
  select.addEventListener("change", () => run());
  $("demoPlay").addEventListener("click", () => run());
  for (const button of document.querySelectorAll("[data-view]")) button.addEventListener("click", () => setView(button.dataset.view));
  if (p.autoplay) run();
}

start();
