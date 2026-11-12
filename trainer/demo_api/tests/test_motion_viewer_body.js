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

// ── 회색 무대: 큰 3D 장면·스켈레톤 패널 배경과 격자 ─────────────
const V2_CSS = fs.readFileSync(path.join(__dirname, '../static/motion_viewer_v2.css'), 'utf8');
const LEGACY_CSS = fs.readFileSync(path.join(__dirname, '../motion_viewer.py'), 'utf8').match(/:root\{--bg:#07101f;[^}]*\}/)[0];
const rgb = (hex) => [1, 3, 5].map((index) => parseInt(hex.slice(index, index + 2), 16));
const token = (css, name) => css.match(new RegExp(`--${name}: ?(#[0-9a-f]{6})`))[1];
const saturation = (hex) => {
  const channels = rgb(hex).map((value) => value / 255), high = Math.max(...channels), low = Math.min(...channels), light = (high + low) / 2;
  return high === low ? 0 : (high - low) / (1 - Math.abs(2 * light - 1));
};
const luminance = (color) => color.map((value) => value / 255).map((value) => (value <= 0.03928 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4))
  .reduce((sum, value, index) => sum + value * [0.2126, 0.7152, 0.0722][index], 0);
/** CSS 색(#hex 또는 rgba)을 배경 위에 섞은 뒤 잰 WCAG 대비. */
const contrast = (style, background) => {
  const rgba = style.startsWith('#') ? [...rgb(style), 1] : style.match(/[\d.]+/g).map(Number);
  const mixed = rgba.slice(0, 3).map((value, index) => value * (rgba[3] ?? 1) + background[index] * (1 - (rgba[3] ?? 1)));
  const [a, b] = [luminance(mixed), luminance(background)].sort((x, y) => y - x);
  return (a + 0.05) / (b + 0.05);
};

test('큰 장면·패널 배경 토큰은 채도 낮은 회색이고 두 화면이 같은 값을 쓴다', () => {
  for (const name of ['stage-top', 'stage']) {
    const value = token(V2_CSS, name);
    assert.ok(saturation(value) < 0.1, `--${name} ${value} 채도 ${saturation(value).toFixed(3)}`);
    assert.equal(token(LEGACY_CSS, name), value, `기존 화면 --${name}`);
  }
  assert.ok(luminance(rgb(token(V2_CSS, 'stage-top'))) > luminance(rgb(token(V2_CSS, 'stage'))), '위가 조금 밝다');
  assert.match(V2_CSS, /\.stage-viewport \{[^}]*background: linear-gradient\(180deg, var\(--stage-top\), var\(--stage\)\)/);
  assert.match(V2_CSS, /\.skeleton-screen \{[^}]*background: var\(--stage\)/);
  const legacy = fs.readFileSync(path.join(__dirname, '../motion_viewer.py'), 'utf8');
  assert.ok(legacy.includes('.stage-viewport{position:relative;min-height:0;overflow:hidden;background:linear-gradient(180deg,var(--stage-top),var(--stage))}'));
  assert.ok(/\.skeleton-screen\{[^}]*background:var\(--stage\)\}/.test(legacy));
  for (const navy of ['#1b2f4c', '#111f35', '#172a44', '150,182,224', '150, 182, 224']) {
    assert.ok(!V2_CSS.includes(navy) && !CORE.includes(navy), `${navy} 가 남아 있다`);
  }
});

test('회색 배경에서 몸 색·격자 주선은 3:1 이상 대비된다(밝은 위쪽 기준)', () => {
  const top = rgb(token(V2_CSS, 'stage-top')), panel = rgb(token(V2_CSS, 'stage'));
  const sides = CORE.split('\n').find((line) => line.startsWith('const BODY=')).match(/#[0-9a-f]{6}/g);
  for (const color of [...sides, '#f5f8ff']) assert.ok(contrast(color, top) >= 3, `몸 ${color} ${contrast(color, top).toFixed(2)}`);
  const floor = CORE.split('\n').find((line) => line.startsWith('function drawFloorGrid(')).match(/rgba\([^)]*\)/g);
  assert.ok(contrast(floor[0], top) >= 3, `큰 장면 격자 주선 ${floor[0]} ${contrast(floor[0], top).toFixed(2)}`);
  const panelGrid = CORE.split('\n').find((line) => line.startsWith('function drawSkeletonView(')).match(/rgba\([^)]*\)/g);
  for (const style of panelGrid) assert.ok(contrast(style, panel) >= 3, `패널 ${style} ${contrast(style, panel).toFixed(2)}`);
});
