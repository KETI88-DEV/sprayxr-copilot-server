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
  const arcs = [];
  const context2d = {
    clearRect() {}, beginPath() {}, moveTo() {}, lineTo() {}, stroke() {}, fill() {}, fillText() {},
    arc(x, y, r) { arcs.push([x, y, r]); },
  };
  const sandbox = vm.createContext({
    skeletonCanvas: { getBoundingClientRect: () => ({ width: 300, height: 300 }) },
    skeletonCtx: context2d,
    $: () => ({ textContent: '' }),
  });
  vm.runInContext([
    'var yaw=0,pitch=0,zoom=175,panX=0,panY=0,skeletonYaw=0,data={bones:[]};',
    pick('function add(a,b)'), pick('function transform('), pick('function project('),
    pick('function uprightSkeleton('), pick('function drawSkeletonView('),
  ].join('\n'), sandbox);
  return {
    sandbox,
    /** 스켈레톤 패널이 관절마다 찍은 [화면 x, 화면 y, 반지름]. 관절 순서 = 넣은 순서. */
    panel(joints, angle = 0) {
      sandbox.skeletonYaw = angle;
      arcs.length = 0;
      sandbox.drawSkeletonView({ joints });
      return Object.fromEntries(Object.keys(joints).map((name, index) => [name, arcs[index]]));
    },
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

/** 패널이 "등 뒤에서 본 그림"이려면 지켜야 하는 것 가운데 어긴 항목. */
function panelViolations(view) {
  const broken = [];
  if ([0, 45, 90, 135, 180, -90].some((angle) => !(det3(panelJacobian(view, STANDING, angle)) < 0))) broken.push('거울상');
  const back = view.panel(STANDING, 0);
  if (!(back['Right Hand'][0] > back.Pelvis[0])) broken.push('오른손이 화면 왼쪽');
  if (!(back['Right Hand'][2] < 5)) broken.push('앞으로 뻗은 손이 크게 그려짐');
  // 행렬식만으로는 "깊이 부호만 뒤집기"를 거르지 못한다. 회전 방향은 이 항목이 지킨다.
  const turned = view.panel(STANDING, 15);
  if (!(turned.Front[0] > turned.Pelvis[0])) broken.push('우 15°에서 몸 앞이 화면 왼쪽');
  return broken;
}

test('큰 3D 장면(transform/project)은 어느 카메라 각에서도 거울상이 아니다', () => {
  const view = viewer();
  for (const yaw of [-2.5, -0.55, 0, 1.2, 3.0]) for (const pitch of [-1.2, 0.1, 1.2]) {
    const determinant = det3(sceneJacobian(view, yaw, pitch));
    assert.ok(determinant < 0, `yaw ${yaw} pitch ${pitch}: det ${determinant}`);
  }
});

test('스켈레톤 패널 0°는 등 뒤에서 본 그림이고 "우" 회전은 몸을 자기 오른쪽으로 돌린다', () => {
  assert.deepEqual(panelViolations(viewer()), []);
});

test('이 시험은 틀린 식을 거른다: 예전 식, 가로 뒤집기, 깊이 부호만 뒤집기', () => {
  const old = (line) => line.replace('cross(up,across)', 'cross(across,up)');
  const flipped = (line) => old(line).replace('w/2+(p[0]-centerX)', 'w/2-(p[0]-centerX)');
  const depthOnly = (line) => old(line).replace('p[2]/span*.22', '-p[2]/span*.22');
  assert.ok(panelViolations(viewer(old)).includes('거울상'));
  assert.ok(panelViolations(viewer(flipped)).includes('오른손이 화면 왼쪽'));
  assert.deepEqual(panelViolations(viewer(depthOnly)), ['우 15°에서 몸 앞이 화면 왼쪽']);
});
