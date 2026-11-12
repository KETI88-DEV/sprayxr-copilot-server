'use strict';
// 큰 3D 장면의 상체와 스켈레톤 패널이 같은 몸 비율 표·같은 두 마디 IK·같은 좌우 색을 쓰는지 본다.
// 실행(호스트): node --test trainer/demo_api/tests/test_motion_viewer_body.js
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const CORE = (() => {
  const source = fs.readFileSync(path.join(__dirname, '../motion_viewer.py'), 'utf8');
  const html = source.slice(source.indexOf('_VIEWER_HTML = r"""'));
  const start = html.indexOf('<script>') + '<script>'.length;
  return html.slice(start, html.indexOf('</script>', start));
})();
const POSE = fs.readFileSync(path.join(__dirname, '../static/motion_viewer_pose.js'), 'utf8');
const UPPER_BODY = POSE.slice(POSE.indexOf('  function trustedUpperBody('), POSE.indexOf('  function drawOrientationGuides('));

/** 코어의 벡터·표·IK 줄과 포즈 스크립트의 상체 함수를 올린다. line/point/ctx 는 그린 것을 적는다. */
function scene() {
  const lines = CORE.split('\n');
  const pick = (marker) => {
    const line = lines.find((item) => item.includes(marker));
    assert.ok(line, `코어 스크립트에서 ${marker} 를 찾지 못했다`);
    return line;
  };
  const drawn = { lines: [], points: [], rings: [] };
  let style = '';
  const sandbox = vm.createContext({
    project: (p) => [p[0], p[1], 1],
    line: (a, b, color, width) => drawn.lines.push({ a, b, color, width }),
    point: (p, r, color) => drawn.points.push({ p, r, color }),
    ctx: {
      set strokeStyle(value) { style = value; }, get strokeStyle() { return style; }, lineWidth: 1,
      beginPath() {}, stroke() {}, arc(x, y, r) { drawn.rings.push({ p: [x, y], r, color: style }); },
    },
  });
  vm.runInContext([pick('function add(a,b)'), pick('const BODY='), pick('function gunGrip('), pick('function twoBoneIK('), UPPER_BODY,
    'this.trustedUpperBody=trustedUpperBody;this.drawTrustedUpperBody=drawTrustedUpperBody;'].join('\n'), sandbox);
  return { sandbox, drawn };
}

const sub = (a, b) => a.map((value, axis) => value - b[axis]);
const add = (a, b) => a.map((value, axis) => value + b[axis]);
const distance = (a, b) => Math.hypot(...sub(a, b));
const near = (actual, expected, message) => assert.ok(Math.abs(actual - expected) < 1e-6, `${message ?? ''} ${actual} ≠ ${expected}`);
const nearPoint = (actual, expected, message) => actual.forEach((value, axis) => near(value, expected[axis], message));

// +Z 를 보는 사람(몸 오른쪽 = +X, 유니티 Y 위). 손잡이 = 건 − 0.30 × 분사 축(+Z).
const HEAD = [1, 1.7, 2];
function frame({ gun = [1.3, 1.3, 2.55], left = [0.7, 1.25, 2.2], tool = [0, 0, 1] } = {}) {
  return { tracked: { head: HEAD, right_hand: gun, left_hand: left }, camera_forward: [0, -0.3, 0.95], tool_forward: tool };
}

test('큰 장면 상체는 패널과 같은 표의 어깨 폭·상완·전완 길이로 오른손은 건 손잡이, 왼손은 왼손 추적기에 닿는다', () => {
  const { sandbox } = scene();
  const body = sandbox.trustedUpperBody(frame());
  const table = sandbox.BODY ?? vm.runInContext('BODY', sandbox);
  near(distance(body.leftShoulder, body.rightShoulder), table.shoulderWidth, '어깨 관절 폭');
  near(table.shoulderWidth, 0.416);
  near(table.upperArm, 0.243); near(table.foreArm, 0.243);
  nearPoint(sub(HEAD, body.neck), [0, table.headAboveShoulders, 0], '머리는 어깨 선 위');
  near(body.rightShoulder[0] - body.leftShoulder[0], table.shoulderWidth, '오른 어깨는 HMD 수평 오른쪽(+X)');
  for (const [shoulder, arm] of [[body.leftShoulder, body.leftArm], [body.rightShoulder, body.rightArm]]) {
    near(distance(shoulder, arm.elbow), table.upperArm, '상완');
    near(distance(arm.elbow, arm.hand), table.foreArm, '전완');
  }
  nearPoint(body.rightArm.hand, [1.3, 1.3, 2.25], '오른손 = 건 손잡이');
  nearPoint(body.leftArm.hand, [0.7, 1.25, 2.2], '왼손 = 왼손 추적기');
  // 팔꿈치는 어깨–손 가운데보다 아래, 몸 바깥쪽에 있다.
  const middle = (arm, shoulder) => add(shoulder, sub(arm.hand, shoulder).map((value) => value / 2));
  assert.ok(body.rightArm.elbow[1] < middle(body.rightArm, body.rightShoulder)[1] && body.rightArm.elbow[0] > middle(body.rightArm, body.rightShoulder)[0]);
  assert.ok(body.leftArm.elbow[1] < middle(body.leftArm, body.leftShoulder)[1] && body.leftArm.elbow[0] < middle(body.leftArm, body.leftShoulder)[0]);
});

test('닿지 않으면 팔을 목표 쪽으로 곧게 편다', () => {
  const { sandbox } = scene();
  const body = sandbox.trustedUpperBody(frame({ gun: [2.5, 1.4, 3.5], left: [-0.5, 1.4, 2.5] }));
  const table = vm.runInContext('BODY', sandbox);
  for (const [shoulder, arm, target] of [[body.rightShoulder, body.rightArm, [2.5, 1.4, 3.2]], [body.leftShoulder, body.leftArm, [-0.5, 1.4, 2.5]]]) {
    near(distance(shoulder, arm.hand), table.upperArm + table.foreArm, '곧게 편 팔 길이');
    const toward = sub(target, shoulder).map((value) => value / distance(target, shoulder));
    nearPoint(sub(arm.hand, shoulder).map((value) => value / (table.upperArm + table.foreArm)), toward, '목표 방향');
    nearPoint(arm.elbow, add(shoulder, toward.map((value) => value * table.upperArm)), '팔꿈치도 같은 직선 위');
  }
});

test('팔·몸통 색은 패널과 같고, 실측 추적기는 따로 고리로 그린다', () => {
  const { sandbox, drawn } = scene();
  sandbox.drawTrustedUpperBody(frame(), [0, 0, 0], 100, 100);
  const body = sandbox.trustedUpperBody(frame());
  const colorOf = (a, b) => drawn.lines.find((item) => distance(item.a, a) < 1e-9 && distance(item.b, b) < 1e-9)?.color;
  assert.equal(colorOf(body.rightShoulder, body.rightArm.elbow), '#22d3ee');
  assert.equal(colorOf(body.rightArm.elbow, body.rightArm.hand), '#22d3ee');
  assert.equal(colorOf(body.leftShoulder, body.leftArm.elbow), '#ff9f7a');
  assert.equal(colorOf(body.leftArm.elbow, body.leftArm.hand), '#ff9f7a');
  assert.equal(colorOf(body.head, body.neck), '#e6edf7');
  assert.equal(colorOf(body.neck, body.chest), '#e6edf7');
  // 고리 = 건(오른쪽 색)과 왼손 추적기(왼쪽 색). 건 고리는 IK 손(손잡이)과 떨어져 있다.
  assert.deepEqual(drawn.rings.map((ring) => ring.color).sort(), ['#22d3ee', '#ff9f7a']);
  assert.ok(drawn.rings.some((ring) => distance(ring.p, [1.3, 1.3]) < 1e-9));
  assert.ok(drawn.rings.some((ring) => distance(ring.p, [0.7, 1.25]) < 1e-9));
});

test('건·왼손 자료가 없으면 그 팔을 그리지 않고, 분사 축이 없으면 건 위치로 뻗는다', () => {
  const { sandbox } = scene();
  const none = sandbox.trustedUpperBody({ tracked: { head: HEAD }, camera_forward: [0, 0, 1] });
  assert.equal(none.rightArm, null); assert.equal(none.leftArm, null);
  const noTool = sandbox.trustedUpperBody({ tracked: { head: HEAD, right_hand: [1.25, 1.3, 2.4] }, camera_forward: [0, 0, 1] });
  nearPoint(noTool.rightArm.hand, [1.25, 1.3, 2.4]);
  assert.equal(sandbox.trustedUpperBody({ tracked: {} }), null);
});

test('패널과 큰 장면은 같은 표·IK·색 상수를 쓰고 값을 따로 적지 않는다', () => {
  const code = CORE.split('\n').filter((line) => !line.startsWith('//'));
  const table = code.find((line) => line.startsWith('const BODY='));
  for (const literal of ['.2106', '.1368', '.416', '.243', '#ff9f7a', '#e6edf7', '#22d3ee']) {
    assert.ok(table.includes(literal), `표에 ${literal} 이 있다`);
    assert.equal(code.filter((line) => line !== table && line.includes(literal)).length, 0, `코어에서 ${literal} 은 표 한 곳에만 있다`);
    if (literal !== '#22d3ee') assert.ok(!POSE.includes(literal), `포즈 스크립트에 ${literal} 을 다시 적지 않는다`);
  }
  assert.ok(!/#22d3ee/.test(UPPER_BODY), '큰 장면 상체 색은 SIDE_COLORS 에서 읽는다');
  assert.ok(!/\.3\b|0\.30/.test(UPPER_BODY), '손잡이 거리는 gunGrip 에서 읽는다');
  const panel = CORE.split('\n').find((line) => line.startsWith('function panelSkeleton('));
  assert.ok(panel.includes('twoBoneIK(') && panel.includes('gunGrip(') && !panel.includes('along='), '패널도 같은 IK 함수를 쓴다');
  assert.ok(UPPER_BODY.includes('twoBoneIK(') && UPPER_BODY.includes('gunGrip(') && UPPER_BODY.includes('BODY.'));
  assert.equal(CORE.split('along=').length - 1, 1, '두 마디 IK 식은 한 곳뿐이다');
  const drawPanel = CORE.split('\n').find((line) => line.startsWith('function drawSkeletonView('));
  assert.ok(drawPanel.includes('SIDE_COLORS.left') && drawPanel.includes('SIDE_COLORS.right') && drawPanel.includes('SIDE_COLORS.middle'));
});
