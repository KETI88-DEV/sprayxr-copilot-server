'use strict';
// 뷰어의 "화면 좌우 = 자료 좌우" 시험. 실행(호스트): node --test trainer/demo_api/tests/test_motion_viewer_chirality.js
// 실제 화면 코드를 node:vm 에 올리고 캔버스 경계만 바꿔 끼운 뒤, 캔버스가 받은 arc() 호출(= 화면에 찍힌 관절)을 읽는다.
// 판정: 유니티 좌표(X 오른쪽·Y 위·Z 앞, 왼손계)의 작은 변위 셋이 화면의 (오른쪽, 위, 보는 사람 쪽)으로 옮겨지는
// 야코비안의 행렬식 부호. 실물을 보는 그림이면 음수이고 양수면 거울상이다.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function coreScript() {
  const source = fs.readFileSync(path.join(__dirname, '../motion_viewer.py'), 'utf8');
  const html = source.slice(source.indexOf('_VIEWER_HTML = r"""'));
  const start = html.indexOf('<script>') + '<script>'.length;
  return html.slice(start, html.indexOf('</script>', start));
}

/** 실제 화면 함수만 뽑아 올린다. `patch`는 시험이 실패할 수 있는지 보려는 메모리 안 변이다(파일은 그대로). */
function viewer(patch = (line) => line) {
  const lines = coreScript().split('\n');
  const pick = (marker) => {
    const line = lines.find((item) => item.includes(marker));
    assert.ok(line, `화면 코드에서 ${marker} 를 찾지 못했다`);
    return patch(line);
  };
  // arcs: 원마다 {x, y, r, filled, style, alpha}. paths: 그은 선마다 {points, style, dash, alpha}. 그린 순서 그대로다.
  const arcs = [], paths = [];
  let points = [], shape = null;
  const context2d = {
    clearRect() {}, fillText() {}, globalAlpha: 1, lineWidth: 1, strokeStyle: '', fillStyle: '', dash: [],
    setLineDash(pattern) { this.dash = [...pattern]; },
    beginPath() { points = []; shape = null; },
    moveTo(x, y) { shape = 'line'; points.push([x, y]); },
    lineTo(x, y) { points.push([x, y]); },
    arc(x, y, r) { shape = 'arc'; arcs.push({ x, y, r, filled: null, style: null, alpha: this.globalAlpha }); },
    fill() { if (shape === 'arc') Object.assign(arcs.at(-1), { filled: true, style: this.fillStyle }); },
    stroke() {
      if (shape === 'arc') Object.assign(arcs.at(-1), { filled: false, style: this.strokeStyle });
      else paths.push({ points: [...points], style: this.strokeStyle, dash: [...this.dash], alpha: this.globalAlpha });
    },
  };
  const skeletonCanvas = { getBoundingClientRect: () => ({ width: 300, height: 300 }), getContext: () => context2d };
  const sandbox = vm.createContext({ skeletonCanvas, skeletonCtx: context2d, $: () => ({ textContent: '' }) });
  vm.runInContext([
    'var yaw=0,pitch=0,zoom=175,panX=0,panY=0,skeletonYaw=0,data={bones:[]};',
    pick('function add(a,b)'), pick('const BODY='), pick('function gunGrip('), pick('function twoBoneIK('), pick('function transform('), pick('function project('),
    pick('function uprightSkeleton('), pick('function withUpperChest('), pick('function panelSkeleton('), pick('function skeletonProjection('), pick('function drawSkeletonView('),
  ].join('\n'), sandbox);
  /** 관절 이름 → 그 관절이 찍힌 원. 그리기 순서는 깊이 순이라 순서가 아니라 같은 투영 함수가 낸 자리로 찾는다. */
  const byName = (frame) => {
    const { joints } = sandbox.panelSkeleton(frame);
    const view = sandbox.skeletonProjection(joints, 300, 300);
    return Object.fromEntries(Object.keys(frame.joints).map((name) => {
      const [x, y] = view.project(joints[name]);
      const arc = arcs.find((item) => Math.abs(item.x - x) < 1e-9 && Math.abs(item.y - y) < 1e-9);
      assert.ok(arc, `${name} 관절 원이 그려지지 않았다`);
      return [name, [arc.x, arc.y, arc.r, arc.filled, arc.style]];
    }));
  };
  return {
    sandbox, skeletonCanvas, arcs, paths,
    /** 스켈레톤 패널이 관절마다 찍은 [화면 x, 화면 y, 반지름, 채움 여부, 색]. */
    panel(joints, angle = 0, frame = { joints }) {
      sandbox.skeletonYaw = angle;
      arcs.length = 0; paths.length = 0;
      sandbox.drawSkeletonView(frame);
      return byName(frame);
    },
    /** 마지막 그리기에서 그은 선(바닥 격자·앞 표시·뼈)의 점선 무늬. */
    dashes: () => paths.map((path) => path.dash),
  };
}

const det3 = (m) => m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
  - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
  + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
const add = (a, b) => a.map((value, axis) => value + b[axis]);
const AXES = [[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]];

/** 패널: 탐침 관절 넷(기준점 + 유니티 X·Y·Z 변위)을 실제 그리기 함수에 넣고 찍힌 원에서 야코비안을 읽는다. */
function panelJacobian(view, joints, angle) {
  const base = add(joints.Pelvis, [0.01, 0.3, 0.01]);
  const drawn = view.panel({ ...joints, P0: base, PX: add(base, AXES[0]), PY: add(base, AXES[1]), PZ: add(base, AXES[2]) }, angle);
  // 화면 (오른쪽, 위, 보는 사람 쪽) = (x, −y, 반지름: 클수록 가깝다)
  const screen = (name) => [drawn[name][0], -drawn[name][1], drawn[name][2]];
  const origin = screen('P0');
  const columns = ['PX', 'PY', 'PZ'].map((name) => screen(name).map((value, axis) => value - origin[axis]));
  return [0, 1, 2].map((row) => columns.map((column) => column[row]));
}

/** 큰 3D 장면: project() 가 내는 [x, y, 원근 배율(클수록 가깝다)]. */
function sceneJacobian(view, yaw, pitch) {
  view.sandbox.yaw = yaw; view.sandbox.pitch = pitch;
  const screen = (point) => { const [x, y, near] = view.sandbox.project(point, [0, 0, 0], 800, 600); return [x, -y, near]; };
  const origin = screen([0.1, 0.2, 0.3]);
  const columns = AXES.map((axis) => screen(add([0.1, 0.2, 0.3], axis)).map((value, index) => value - origin[index]));
  return [0, 1, 2].map((row) => columns.map((column) => column[row]));
}

// 합성 자세: +Z 를 보고 선 사람. 몸 오른쪽 = +X. 오른손을 앞으로 뻗었고 Front 는 몸 정면 앞의 점이다.
const STANDING = {
  Pelvis: [0, 1.0, 0], Head: [0, 1.6, 0], 'Left Shoulder': [-0.2, 1.45, 0], 'Right Shoulder': [0.2, 1.45, 0],
  'Left Hand': [-0.25, 1.0, 0.05], 'Right Hand': [0.25, 1.3, 0.45], Front: [0, 1.2, 0.3],
};

/** 패널이 "정면에서 본 그림"이려면 지켜야 하는 것 가운데 어긴 항목.
 *  얼굴 없는 막대 인형은 보는 사람을 마주 본 모습으로 읽힌다. 그래서 몸 오른쪽이 화면 왼쪽에 와야
 *  오른손잡이가 오른손잡이로 보인다(2026-11-12 등 뒤 그림은 "왼팔이 점선"으로 읽혔다). */
function panelViolations(view) {
  const broken = [];
  if ([0, 45, 90, 135, 180, -90].some((angle) => !(det3(panelJacobian(view, STANDING, angle)) < 0))) broken.push('거울상');
  const front = view.panel(STANDING, 0);
  if (!(front['Right Hand'][0] < front.Pelvis[0])) broken.push('오른손이 화면 오른쪽');
  if (!(front['Right Hand'][2] > 5)) broken.push('앞으로 뻗은 손이 작게 그려짐');
  // 행렬식만으로는 "깊이 부호만 뒤집기"를 거르지 못한다. 회전 방향은 이 항목이 지킨다.
  // "우"는 사람이 자기 오른쪽으로 도는 것이다. 정면에서 보면 몸 앞의 점이 화면 왼쪽으로 간다.
  const turned = view.panel(STANDING, 15);
  if (!(turned.Front[0] < turned.Pelvis[0])) broken.push('우 15°에서 몸 앞이 화면 오른쪽');
  return broken;
}

test('큰 3D 장면(transform/project)은 어느 카메라 각에서도 거울상이 아니다', () => {
  const view = viewer();
  for (const yaw of [-2.5, -0.55, 0, 1.2, 3.0]) for (const pitch of [-1.2, 0.1, 1.2]) {
    const determinant = det3(sceneJacobian(view, yaw, pitch));
    assert.ok(determinant < 0, `yaw ${yaw} pitch ${pitch}: det ${determinant}`);
  }
});

test('스켈레톤 패널 0°는 정면에서 본 그림이고 "우" 회전은 몸을 자기 오른쪽으로 돌린다', () => {
  assert.deepEqual(panelViolations(viewer()), []);
});

test('이 시험은 틀린 식을 거른다: 처음 식(거울상), 등 뒤 그림, 깊이 부호만 뒤집기', () => {
  const mirror = (line) => line.replace('x=-dot(relative,across)', 'x=dot(relative,across)');
  const back = (line) => mirror(line).replace('cross(across,up)', 'cross(up,across)');
  const depthOnly = (line) => back(line).replace('d/span*.22', '-d/span*.22');
  assert.ok(panelViolations(viewer(mirror)).includes('거울상'));
  assert.ok(panelViolations(viewer(back)).includes('오른손이 화면 오른쪽'));
  assert.ok(panelViolations(viewer(depthOnly)).includes('오른손이 화면 오른쪽'));
});

// ── 오른팔 다시 세우기 ─────────────────────────────────────────
// 합성 프레임: 관절은 +Z 를 보고 선 리그 좌표이고 오른손 관절이 몸 왼쪽에 놓여 있다. 실측 HMD 는 월드에서 +X 를
// 보고(오른쪽 = −Z) 건은 HMD 의 오른쪽 앞 아래에 있다. 관절과 실측의 좌표틀이 다른 예전 형식 기록과 같은 모양이다.
const ARM_BONES = [['Right Shoulder', 'Right Upper Arm'], ['Right Upper Arm', 'Right Fore Arm'], ['Right Fore Arm', 'Right Hand'], ['Left Shoulder', 'Left Hand']];
const REBUILT = ['Right Fore Arm', 'Right Hand'];
const gunFrame = () => ({
  joints: {
    Pelvis: [0, 1.0, 0], Head: [0, 1.6, 0], 'Left Shoulder': [-0.2, 1.45, 0], 'Right Shoulder': [0.2, 1.45, 0],
    'Left Hand': [-0.3, 1.1, 0.3], 'Right Upper Arm': [0.25, 1.42, 0], 'Right Fore Arm': [0.05, 1.3, 0.2], 'Right Hand': [-0.15, 1.2, 0.35],
  },
  tracked: { head: [3, 1.65, 5], right_hand: [3.55, 1.3, 4.7] },
  camera_forward: [1, 0, 0], tool_forward: [1, 0, 0],
});
const distance = (a, b) => Math.hypot(...a.map((value, axis) => value - b[axis]));
const near = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-9, `${actual} ≠ ${expected}`);

test('패널은 오른쪽 전완·손만 실측 건 쪽으로 다시 세우고 다른 관절과 같은 모양으로 그린다', () => {
  const view = viewer();
  view.sandbox.data.bones = ARM_BONES;
  const frame = gunFrame(), before = JSON.stringify(frame);
  const drawn = view.panel(frame.joints, 0, frame);
  assert.ok(drawn['Right Hand'][0] < drawn.Pelvis[0], '정면 그림에서 다시 세운 오른손은 골반보다 화면 왼쪽이다');
  assert.equal(JSON.stringify(frame), before, '프레임 자료(frame.joints)는 그대로다');
  // 점선 팔은 끊긴 팔로 읽혔다(2026-11-12). 다시 세운 팔도 채운 원과 실선으로 그린다.
  for (const [name, arc] of Object.entries(drawn)) assert.equal(arc[3], true, name);
  assert.ok(view.dashes().every((pattern) => pattern.length === 0));
});

test('다시 세운 팔은 마디 길이를 지키고 나머지 관절은 CSV 그대로다', () => {
  const view = viewer(), frame = gunFrame();
  const csv = view.sandbox.uprightSkeleton(frame.joints, 0);
  const upper = distance(csv['Right Upper Arm'], csv['Right Fore Arm']), fore = distance(csv['Right Fore Arm'], csv['Right Hand']);
  const lengths = (joints) => [distance(joints['Right Upper Arm'], joints['Right Fore Arm']), distance(joints['Right Fore Arm'], joints['Right Hand'])];
  const { joints, estimated } = view.sandbox.panelSkeleton(frame, 0);
  assert.deepEqual([...estimated], REBUILT);
  for (const name of Object.keys(csv)) if (!REBUILT.includes(name)) assert.deepEqual([...joints[name]], [...csv[name]], name);
  // 손잡이 = 건 − 0.30m × 분사 축 = HMD 에서 오른쪽 0.3 · 아래 0.35 · 앞 0.25m. 패널(정면 그림)에서는 Head 관절에서
  // 그만큼이다: 몸 오른쪽 = −x, 앞(보는 사람 쪽) = +z.
  joints['Right Hand'].forEach((value, axis) => near(value, [-0.3, 0.25, 0.25][axis]));
  lengths(joints).forEach((value, index) => near(value, [upper, fore][index]));
  assert.ok(joints['Right Fore Arm'][1] < (joints['Right Upper Arm'][1] + joints['Right Hand'][1]) / 2, '팔꿈치는 아래로 굽는다');
  // 180° 돌리면 다시 세운 손도 같이 돈다.
  near(view.sandbox.panelSkeleton(frame, 180).joints['Right Hand'][0], -joints['Right Hand'][0]);
  // 손잡이가 팔 길이 밖이면 늘리지 않고 손잡이 쪽으로 곧게 편다.
  const far = gunFrame();
  far.tracked.right_hand = [5.5, 1.3, 4.7];
  const straight = view.sandbox.panelSkeleton(far, 0).joints;
  lengths(straight).forEach((value, index) => near(value, [upper, fore][index]));
  near(distance(straight['Right Upper Arm'], straight['Right Hand']), upper + fore);
});

test('건 자료가 없으면 CSV 관절을 그대로 실선으로 그린다', () => {
  const view = viewer();
  view.sandbox.data.bones = ARM_BONES;
  const whole = gunFrame(), csv = view.sandbox.uprightSkeleton(whole.joints, 0);
  const missing = [
    (frame) => delete frame.tracked.right_hand, (frame) => delete frame.tracked.head, (frame) => delete frame.tracked,
    (frame) => delete frame.tool_forward, (frame) => delete frame.camera_forward,
    (frame) => { frame.tool_forward = [0, 0, 0]; }, (frame) => { frame.camera_forward = [0, 1, 0]; },
    (frame) => delete frame.joints['Right Fore Arm'],
  ];
  for (const remove of missing) {
    const frame = gunFrame();
    remove(frame);
    const { joints, estimated } = view.sandbox.panelSkeleton(frame, 0);
    assert.equal(estimated.length, 0);
    for (const name of Object.keys(frame.joints)) assert.deepEqual([...joints[name]], [...csv[name]], name);
    const drawn = view.panel(frame.joints, 0, frame);
    assert.ok(Object.values(drawn).every((arc) => arc[3] === true));
    assert.ok(view.dashes().every((pattern) => pattern.length === 0));
  }
  assert.ok(view.panel(whole.joints, 0, { joints: whole.joints })['Right Hand'][0] > 150, 'CSV 오른손 관절은 몸 왼쪽(정면 그림의 화면 오른쪽)에 그려진다');
});

test('v2 교정 고리도 같은 함수로 관절 위치를 구한다', () => {
  const script = fs.readFileSync(path.join(__dirname, '../static/motion_viewer_v2.js'), 'utf8');
  assert.ok(script.includes('panelSkeleton(frame).joints'));
  assert.ok(!script.includes('uprightSkeleton('));
});

// ── 시인성: 팔·다리 색, 바닥 격자, 교정 고리 ───────────────────────
// 서 있는 사람(+Z 를 봄, 몸 오른쪽 = +X)의 전신. 뼈 목록은 서버 _BONES 와 같은 부모→자식 순서다.
const BODY = {
  Pelvis: [0, 1.0, 0], Spine: [0, 1.15, 0], Chest: [0, 1.4, 0], Head: [0, 1.65, 0],
  'Left Shoulder': [-0.08, 1.45, 0], 'Left Upper Arm': [-0.2, 1.45, 0], 'Left Fore Arm': [-0.22, 1.2, 0], 'Left Hand': [-0.22, 0.95, 0],
  'Right Shoulder': [0.08, 1.45, 0], 'Right Upper Arm': [0.2, 1.45, 0], 'Right Fore Arm': [0.25, 1.3, 0.25], 'Right Hand': [0.3, 1.3, 0.5],
  'Left Thigh': [-0.1, 0.95, 0], 'Left Calf': [-0.1, 0.5, 0], 'Left Foot': [-0.1, 0.08, 0], 'Left Toe': [-0.1, 0.02, 0.15],
  'Right Thigh': [0.1, 0.95, 0], 'Right Calf': [0.1, 0.5, 0], 'Right Foot': [0.1, 0.08, 0], 'Right Toe': [0.1, 0.02, 0.15],
};
const BODY_BONES = [['Pelvis', 'Spine'], ['Spine', 'Chest'], ['Chest', 'Head'], ['Chest', 'Left Shoulder'], ['Left Shoulder', 'Left Upper Arm'],
  ['Left Upper Arm', 'Left Fore Arm'], ['Left Fore Arm', 'Left Hand'], ['Chest', 'Right Shoulder'], ['Right Shoulder', 'Right Upper Arm'],
  ['Right Upper Arm', 'Right Fore Arm'], ['Right Fore Arm', 'Right Hand'], ['Pelvis', 'Left Thigh'], ['Left Thigh', 'Left Calf'],
  ['Left Calf', 'Left Foot'], ['Left Foot', 'Left Toe'], ['Pelvis', 'Right Thigh'], ['Right Thigh', 'Right Calf'], ['Right Calf', 'Right Foot'], ['Right Foot', 'Right Toe']];

/** 뼈 [a, b] 를 그은 선의 색. 끝점이 두 관절 원의 중심과 같은 선을 찾는다. */
function boneStyle(view, drawn, a, b) {
  const same = (point, joint) => Math.abs(point[0] - drawn[joint][0]) < 1e-9 && Math.abs(point[1] - drawn[joint][1]) < 1e-9;
  const path = view.paths.find((item) => item.points.length === 2 && same(item.points[0], a) && same(item.points[1], b));
  assert.ok(path, `${a}–${b} 뼈가 그려지지 않았다`);
  return path.style;
}

// 책임자 결정(2026-11-12): 좌우가 아니라 팔과 다리로 나눈다. 팔은 청록, 다리는 회색, 몸통은 밝은 중립색.
test('두 팔은 청록, 두 다리는 회색, 몸통은 둘 다와 다른 색이다', () => {
  const view = viewer();
  view.sandbox.data.bones = BODY_BONES;
  for (const angle of [0, 90, 180]) {
    const drawn = view.panel(BODY, angle);
    const arm = boneStyle(view, drawn, 'Right Fore Arm', 'Right Hand'), leg = boneStyle(view, drawn, 'Right Calf', 'Right Foot');
    const core = boneStyle(view, drawn, 'Spine', 'Chest');
    assert.equal(arm, '#22d3ee', `${angle}° 팔은 청록`);
    assert.equal(boneStyle(view, drawn, 'Left Fore Arm', 'Left Hand'), arm, '왼팔도 같은 청록');
    assert.equal(boneStyle(view, drawn, 'Left Calf', 'Left Foot'), leg, '두 다리는 같은 색');
    assert.notEqual(leg, arm, `${angle}° 다리는 팔과 다른 색`);
    assert.ok(core !== arm && core !== leg, `${angle}° 몸통 색`);
    assert.equal(drawn['Left Hand'][4], drawn['Right Hand'][4], '두 손 관절 원은 같은 색');
  }
});

test('바닥 격자는 가장 낮은 발 밑에 그려지고 몸과 함께 돈다', () => {
  const view = viewer();
  view.sandbox.data.bones = BODY_BONES;
  const grid = (angle) => {
    const drawn = view.panel(BODY, angle);
    const joints = new Set(Object.values(drawn).map(([x, y]) => `${x},${y}`));
    const floor = view.paths.filter((path) => path.points.every(([x, y]) => !joints.has(`${x},${y}`)));
    return { drawn, lines: floor.filter((path) => path.points.length === 2), mark: floor.find((path) => path.points.length === 3) };
  };
  const front = grid(0);
  assert.ok(front.lines.length >= 10, `격자 선 ${front.lines.length}개`);
  // 격자는 발보다 아래(화면에서 더 아래쪽)까지 깔린다.
  const lowest = Math.max(...front.lines.flatMap((path) => path.points.map(([, y]) => y)));
  assert.ok(lowest > Math.max(front.drawn['Left Foot'][1], front.drawn['Right Foot'][1]));
  // 위에서 내려다본다: 0°에서 보는 사람 쪽 선(z = +0.8, 다섯째)이 등 뒤 선(z = −0.8, 첫째)보다 화면 아래다.
  assert.ok(front.lines[4].points[0][1] > front.lines[0].points[0][1] + 10, '보는 사람 쪽 바닥이 화면 아래');
  // 첫 선(몸 좌우 방향 선)은 0°에서 화면 가로, 우 90°에서는 세로 쪽으로 눕는다.
  const slope = ({ points: [[x1, y1], [x2, y2]] }) => Math.abs(y2 - y1) / Math.max(1e-9, Math.abs(x2 - x1));
  const side = grid(90);
  assert.ok(slope(front.lines[0]) < 0.05, `0° 기울기 ${slope(front.lines[0])}`);
  assert.ok(slope(side.lines[0]) > 1, `90° 기울기 ${slope(side.lines[0])}`);
  // 몸 앞 표시: 0°에는 보는 사람 쪽(골반 아래보다 화면 아래), 우 90°에는 화면 왼쪽.
  const tip = (mark) => mark.points[1];
  assert.ok(front.mark && tip(front.mark)[1] > front.drawn.Pelvis[1] && Math.abs(tip(front.mark)[0] - front.drawn.Pelvis[0]) < 1e-6);
  assert.ok(side.mark && tip(side.mark)[0] < side.drawn.Pelvis[0] - 20);
  assert.ok(view.dashes().every((pattern) => pattern.length === 0));
});

test('가까운 관절을 나중에 그린다(0°에는 앞으로 뻗은 오른손, 180°에는 왼손)', () => {
  const view = viewer();
  view.sandbox.data.bones = BODY_BONES;
  const order = (drawn, name) => view.arcs.findIndex((arc) => arc.x === drawn[name][0] && arc.y === drawn[name][1]);
  const front = view.panel(BODY, 0);
  assert.ok(order(front, 'Right Hand') > order(front, 'Left Hand'));
  const back = view.panel(BODY, 180);
  assert.ok(order(back, 'Right Hand') < order(back, 'Left Hand'));
});

// WCAG 상대 휘도 대비. 색을 그린 불투명도로 패널 배경(v2 토큰 --stage) 위에 섞은 뒤 잰다.
const rgb = (hex) => [1, 3, 5].map((index) => parseInt(hex.slice(index, index + 2), 16));
const BACKGROUND = rgb(fs.readFileSync(path.join(__dirname, '../static/motion_viewer_v2.css'), 'utf8').match(/--stage: (#[0-9a-f]{6})/)[1]);
const luminance = (color) => color.map((value) => value / 255).map((value) => (value <= 0.03928 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4))
  .reduce((sum, value, index) => sum + value * [0.2126, 0.7152, 0.0722][index], 0);
const contrast = (hex, alpha) => {
  const mixed = rgb(hex).map((value, index) => value * alpha + BACKGROUND[index] * (1 - alpha));
  const [a, b] = [luminance(mixed), luminance(BACKGROUND)].sort((x, y) => y - x);
  return (a + 0.05) / (b + 0.05);
};

test('가장 옅게 그린 뼈·관절도 패널 배경과 3:1 이상 대비된다', () => {
  const view = viewer();
  view.sandbox.data.bones = BODY_BONES;
  for (const angle of [0, 45, 90, 135, 180, 270]) {
    const drawn = view.panel(BODY, angle);
    const joints = new Set(Object.values(drawn).map(([x, y]) => `${x},${y}`));
    const body = [...view.arcs, ...view.paths.filter((path) => path.points.some(([x, y]) => joints.has(`${x},${y}`)))];
    assert.ok(body.length > 30);
    for (const mark of body) assert.ok(contrast(mark.style, mark.alpha) >= 3, `${angle}° ${mark.style} × ${mark.alpha}: ${contrast(mark.style, mark.alpha).toFixed(2)}`);
  }
});

test('v2 교정 고리는 같은 투영 함수로 관절 원과 같은 중심에 그린다', () => {
  const view = viewer();
  view.sandbox.data.bones = BODY_BONES;
  const script = fs.readFileSync(path.join(__dirname, '../static/motion_viewer_v2.js'), 'utf8');
  const start = script.indexOf('  function markFlaggedJoints(frame) {');
  const body = script.slice(start, script.indexOf('\n  }\n', start) + 4);
  const mark = vm.runInContext(`(function ($, sequenceAt, anchor) {\n${body}\nreturn markFlaggedJoints; })`, view.sandbox)(
    () => view.skeletonCanvas, () => ({ joints: ['RightHand', 'LeftForeArm'] }), { playhead_seconds: 0 });
  for (const angle of [0, 45, 135]) {
    const drawn = view.panel(BODY, angle);
    const before = view.arcs.length;
    mark({ joints: BODY });
    const rings = view.arcs.slice(before);
    assert.equal(rings.length, 2);
    assert.ok(rings.every((ring) => ring.filled === false));
    for (const [ring, name] of [[rings[0], 'Right Hand'], [rings[1], 'Left Fore Arm']]) {
      assert.ok(Math.abs(ring.x - drawn[name][0]) < 1e-9 && Math.abs(ring.y - drawn[name][1]) < 1e-9, `${angle}° ${name}`);
    }
  }
});

test('큰 장면 바닥은 실측 머리에서 CSV 눈높이를 뺀 높이에, 장면 범위보다 넓게 깔린다', () => {
  const view = viewer();
  vm.runInContext([coreScript().split('\n').find((line) => line.startsWith('let floorSource')), coreScript().split('\n').find((line) => line.startsWith('function sceneFloor('))].join('\n'), view.sandbox);
  const frame = (headY) => ({ tracked: { head: [1, headY, 2] }, joints: { Head: [0, 1.7, 0], 'Left Foot': [0, 0.1, 0], 'Right Toe': [0, 0.05, 0] } });
  // 머리가 4.0~5.0m 를 오르내린다(리프트). 바닥 = 가장 낮게 선 때의 발 = 4.0 − (1.7 − 0.05).
  view.sandbox.data = { frames: [4.0, 4.5, 5.0].map(frame), reference_surface: { points: [{ point: [3, 4.2, -1] }] }, paint_trace: [] };
  const grid = vm.runInContext('sceneFloor()', view.sandbox);
  near(grid.y, 4.0 - 1.65);
  assert.ok(grid.lo[0] <= 1 - 1 && grid.hi[0] >= 3 + 1 && grid.lo[1] <= -1 - 1 && grid.hi[1] >= 2 + 1, JSON.stringify(grid));
  // 관절이 없으면 장면에서 가장 낮은 점.
  view.sandbox.data = { frames: [{ tracked: { head: [0, 4.5, 0] } }], reference_surface: { points: [{ point: [0, 3.2, 0] }] }, paint_trace: [] };
  near(vm.runInContext('sceneFloor()', view.sandbox).y, 3.2);
  view.sandbox.data = { frames: [{}] };
  assert.equal(vm.runInContext('sceneFloor()', view.sandbox), null);
});

// ── 몸통 길이: 예전 형식 리그의 빠진 가슴 위 마디 ─────────────────
// 합성 리그: +Z 를 보고 곧게 선 사람(몸 오른쪽 = +X). 마디 값은 두 리그의 바인드 오프셋(m)이다. 예전 형식은 Chest 바로 옆에
// 어깨가 붙고 머리가 0.137m 위에 있다. 2026-10 리그는 Chest 위 0.2106m 에 어깨 선이 있고, 팔·다리·Spine→Chest 는 같다.
function rig({ pelvisToSpine, hipDrop, shoulderRise, head }) {
  const at = (from, offset) => add(from, offset), joints = { Pelvis: [0, 1, 0] };
  joints.Spine = at(joints.Pelvis, [0, pelvisToSpine, 0]);
  joints.Chest = at(joints.Spine, [0, 0.105315, 0]);
  joints.Head = at(joints.Chest, head);
  for (const [side, sign] of [['Left', -1], ['Right', 1]]) {
    joints[`${side} Shoulder`] = at(joints.Chest, [sign * 0.028354, shoulderRise, 0]);
    joints[`${side} Upper Arm`] = at(joints[`${side} Shoulder`], [sign * 0.179695, 0, 0]);
    joints[`${side} Fore Arm`] = at(joints[`${side} Upper Arm`], [0, -0.243033, 0]);
    joints[`${side} Hand`] = at(joints[`${side} Fore Arm`], [0, -0.243033, 0]);
    joints[`${side} Thigh`] = at(joints.Pelvis, [sign * 0.096502, -hipDrop, 0]);
    joints[`${side} Calf`] = at(joints[`${side} Thigh`], [0, -0.422307, 0]);
    joints[`${side} Foot`] = at(joints[`${side} Calf`], [0, -0.410989, 0]);
    joints[`${side} Toe`] = at(joints[`${side} Foot`], [0, -0.067168, 0.133748]);
  }
  return joints;
}
const LEGACY = () => rig({ pelvisToSpine: 0.105314, hipDrop: 0.093129, shoulderRise: 0, head: [0, 0.136773, 0] });
const CURRENT = () => rig({ pelvisToSpine: 0.198451, hipDrop: 0, shoulderRise: 0.2106, head: [0, 0.340962, 0.042165] });
/** 몸 전체를 앞으로 30° 숙이고 옆으로 20° 돌린다(곧게 선 자세에서만 맞는 식을 거른다). */
function leaned(joints) {
  const a = Math.PI / 6, b = Math.PI / 9;
  const turn = ([x, y, z]) => { const [y1, z1] = [y * Math.cos(a) - z * Math.sin(a), y * Math.sin(a) + z * Math.cos(a)]; return [x * Math.cos(b) + z1 * Math.sin(b), y1, -x * Math.sin(b) + z1 * Math.cos(b)]; };
  return Object.fromEntries(Object.entries(joints).map(([name, point]) => [name, turn(point)]));
}
const mid = (a, b) => a.map((value, axis) => (value + b[axis]) / 2);
/** [몸통/다리, 어깨 폭/몸통]. 몸통 = 엉덩이 선(두 Thigh 중점) → 어깨 선(두 Upper Arm 중점), 다리 = 허벅지 + 종아리. */
const proportions = (j) => {
  const torso = distance(mid(j['Left Thigh'], j['Right Thigh']), mid(j['Left Upper Arm'], j['Right Upper Arm']));
  const leg = distance(j['Right Thigh'], j['Right Calf']) + distance(j['Right Calf'], j['Right Foot']);
  return [torso / leg, distance(j['Left Upper Arm'], j['Right Upper Arm']) / torso];
};

test('그려진 뼈 길이 비율은 2026-10 리그 비율과 같다(몸통/다리, 어깨 폭/몸통)', () => {
  const view = viewer();
  view.sandbox.data.bones = BODY_BONES;
  const expected = proportions(CURRENT());
  for (const angle of [0, 90, 180]) for (const joints of [LEGACY(), leaned(LEGACY())]) {
    const drawn = view.sandbox.panelSkeleton({ joints }, angle).joints;
    proportions(drawn).forEach((value, index) => assert.ok(Math.abs(value / expected[index] - 1) < 0.002, `${angle}° ${index}: ${value} ≠ ${expected[index]}`));
    near(distance(drawn.Chest, drawn.Head), 0.136773 + 0.2106);
  }
  // 고치기 전 예전 형식 비율은 몸통이 0.21m 짧아 이 시험에 걸린다.
  assert.ok(proportions(LEGACY())[0] < expected[0] * 0.7);
  // 이미 길이가 맞는 최근 형식은 그대로 그린다.
  for (const joints of [CURRENT(), leaned(CURRENT()), BODY]) {
    assert.equal(view.sandbox.withUpperChest(joints), joints);
    const drawn = view.sandbox.panelSkeleton({ joints }, 45).joints, plain = view.sandbox.uprightSkeleton(joints, 45);
    for (const name of Object.keys(joints)) assert.deepEqual([...drawn[name]], [...plain[name]], name);
  }
});

test('관절 방향(부모→자식 단위 벡터)은 자료와 같고 어깨·머리는 가슴 위로 함께 올라간다', () => {
  const view = viewer();
  for (const joints of [LEGACY(), leaned(LEGACY())]) {
    const before = JSON.stringify(joints), raised = view.sandbox.withUpperChest(joints);
    assert.equal(JSON.stringify(joints), before, '자료(frame.joints)는 그대로다');
    const direction = (j, a, b) => { const d = j[b].map((value, axis) => value - j[a][axis]), size = Math.hypot(...d); return d.map((value) => value / size); };
    // Chest→어깨는 빠진 마디를 지나 이어지는 뼈라 방향이 바뀌는 것이 맞다(옆 → 위).
    for (const [a, b] of BODY_BONES) if (!(a === 'Chest' && b.endsWith('Shoulder'))) {
      direction(raised, a, b).forEach((value, axis) => near(value, direction(joints, a, b)[axis]));
      if (!(a === 'Chest' && b === 'Head')) near(distance(raised[a], raised[b]), distance(joints[a], joints[b]));
    }
    // 올린 양은 척추(Chest→Head) 방향으로 0.2106m 이고 어깨 두 갈래·머리가 같은 벡터로 움직인다.
    const lift = raised.Head.map((value, axis) => value - joints.Head[axis]), spine = direction(joints, 'Chest', 'Head');
    lift.forEach((value, axis) => near(value, spine[axis] * 0.2106));
    for (const name of Object.keys(joints)) {
      const moved = /^(Head|(Left|Right) (Shoulder|Upper Arm|Fore Arm|Hand))$/.test(name);
      raised[name].forEach((value, axis) => near(value - joints[name][axis], moved ? lift[axis] : 0));
    }
  }
});

test('예전 형식 몸에서도 오른팔은 올린 어깨에서 건 손잡이로 다시 세운다', () => {
  const view = viewer(), joints = LEGACY(), frame = { ...gunFrame(), joints };
  const { joints: drawn, estimated } = view.sandbox.panelSkeleton(frame, 0);
  assert.deepEqual([...estimated], REBUILT);
  // 손잡이는 Head 관절 기준 오른쪽 0.3 · 아래 0.35 · 앞 0.25m(정면 그림: 몸 오른쪽 = −x). 머리가 올라간 만큼 손도 올라간다.
  drawn['Right Hand'].forEach((value, axis) => near(value - drawn.Head[axis], [-0.3, -0.35, 0.25][axis]));
  near(distance(drawn['Right Upper Arm'], drawn['Right Fore Arm']), 0.243033);
  near(distance(drawn['Right Fore Arm'], drawn['Right Hand']), 0.243033);
  // 어깨는 올린 자리 그대로다(머리와 어깨 사이 거리는 자료와 같다).
  near(distance(drawn.Head, drawn['Right Upper Arm']), distance(joints.Head, joints['Right Upper Arm']));
});
