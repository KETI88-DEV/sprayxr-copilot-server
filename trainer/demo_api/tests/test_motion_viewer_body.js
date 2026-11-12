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

/**
 * 코어의 벡터·표·IK 줄과 포즈 스크립트의 상체 함수를 올린다. 그리기 맥락은 경로(beginPath)마다 원·타원·채움·윤곽을 적는다(shapes, 그린 순서).
 * real 이면 코어의 실제 transform/project 를 yaw·pitch 로 쓰고, 아니면 투영은 (x, y) × zoom 이다(깊이 없음).
 */
function scene({ real = false, yaw = 0, pitch = 0 } = {}) {
  const lines = CORE.split('\n');
  const pick = (marker) => {
    const line = lines.find((item) => item.includes(marker));
    assert.ok(line, `코어 스크립트에서 ${marker} 를 찾지 못했다`);
    return line;
  };
  const shapes = [], alphas = [];
  let shape = null;
  const ctx = {
    canvas: { width: 100, height: 100 }, globalAlpha: 1, lineWidth: 1, strokeStyle: '', fillStyle: '',
    save() { alphas.push(this.globalAlpha); }, restore() { this.globalAlpha = alphas.pop(); },
    setTransform() {}, getTransform() { return null; }, clearRect() {}, drawImage() {}, moveTo() {}, lineTo() {}, closePath() {},
    createLinearGradient() { return { stops: [], addColorStop(offset, color) { this.stops.push([offset, color]); } }; },
    beginPath() { shape = { arcs: [], ellipses: [], fill: null, stroke: null }; shapes.push(shape); },
    arc(x, y, r) { shape.arcs.push({ p: [x, y], r }); },
    ellipse(x, y, rx, ry) { shape.ellipses.push({ p: [x, y], rx, ry }); },
    fill() { shape.fill = this.fillStyle; shape.alpha = this.globalAlpha; },
    stroke() { shape.stroke = this.strokeStyle; },
  };
  const sandbox = vm.createContext({ ctx, document: { createElement: () => ({ width: 0, height: 0, getContext: () => ctx }) } });
  const projection = real
    ? [`var yaw=${yaw},pitch=${pitch},zoom=100,panX=0,panY=0;`, pick('function transform('), pick('function project(')]
    : ['var zoom=100;function transform(p,c){return p.map((v,i)=>v-c[i])}function project(p){return [p[0]*zoom,p[1]*zoom,1]}'];
  vm.runInContext([...projection, pick('function add(a,b)'), pick('const BODY='), pick('function gunGrip('), pick('function twoBoneIK('), UPPER_BODY,
    'this.trustedUpperBody=trustedUpperBody;this.drawTrustedUpperBody=drawTrustedUpperBody;'].join('\n'), sandbox);
  return { sandbox, shapes };
}

/** 채움의 밝은 색(그라디언트면 끝 색). */
const light = (shape) => (typeof shape.fill === 'string' ? shape.fill : shape.fill?.stops.at(-1)[1]);
/** 두 끝 원의 중심이 a, b 의 화면 자리인 캡슐 경로. */
const capsuleAt = (shapes, project, a, b) => shapes.find((shape) => shape.arcs.length === 2 && shape.fill
  && distance(shape.arcs[0].p, project(a)) < 1e-6 && distance(shape.arcs[1].p, project(b)) < 1e-6);

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

test('팔 캡슐은 어깨–팔꿈치–손을 잇고 패널과 같은 색이며, 실측 추적기는 따로 고리로 그린다', () => {
  const { sandbox, shapes } = scene();
  sandbox.drawTrustedUpperBody(frame(), [0, 0, 0], 100, 100);
  const body = sandbox.trustedUpperBody(frame());
  const flat = (p) => [p[0] * 100, p[1] * 100];
  for (const [shoulder, arm] of [[body.leftShoulder, body.leftArm], [body.rightShoulder, body.rightArm]]) {
    const upper = capsuleAt(shapes, flat, shoulder, arm.elbow), fore = capsuleAt(shapes, flat, arm.elbow, arm.hand);
    assert.ok(upper && fore, '위팔·아래팔 캡슐이 관절 사이를 잇는다');
    assert.equal(light(upper), '#22d3ee'); assert.equal(light(fore), '#22d3ee');
    // 굵기(px) = 살 반지름(m) × zoom: 위팔 어깨 .044 → 팔꿈치 .04, 아래팔은 그보다 가늘다.
    near(upper.arcs[0].r, 4.4, '위팔 어깨 쪽'); near(upper.arcs[1].r, 4, '위팔 팔꿈치 쪽');
    assert.ok(fore.arcs[0].r < upper.arcs[1].r && fore.arcs[1].r < fore.arcs[0].r, '아래팔은 손목으로 갈수록 가늘다');
  }
  const torso = shapes.find((shape) => shape.arcs.length === 6);
  assert.equal(light(torso), '#e6edf7', '몸통은 좌우 캡슐과 어깨선을 한 덩어리로 칠한다');
  const head = shapes.find((shape) => shape.ellipses.length === 1);
  assert.equal(light(head), '#e6edf7');
  assert.ok(head.ellipses[0].ry > head.ellipses[0].rx, '머리는 세로로 긴 타원');
  // 고리 = 건과 왼손 추적기(둘 다 팔 색, 채우지 않음). 건 고리는 IK 손(손잡이)과 떨어져 있다.
  const rings = shapes.filter((shape) => !shape.fill && shape.stroke === '#22d3ee');
  assert.equal(rings.length, 2);
  assert.ok(rings.some((ring) => distance(ring.arcs[0].p, [130, 130]) < 1e-9));
  assert.ok(rings.some((ring) => distance(ring.arcs[0].p, [70, 125]) < 1e-9));
  assert.equal(shapes.at(-1), rings.at(-1), '고리는 몸 위에 맨 마지막으로 그린다');
});

test('건 연장대·노즐은 분사 축을 따라 손잡이에서 추적기 앞으로 뻗는다', () => {
  const { sandbox, shapes } = scene();
  const tool = [0.6, -0.48, 0.64], gun = [1.3, 1.3, 2.55];
  const input = frame({ tool, gun });
  sandbox.drawTrustedUpperBody(input, [0, 0, 0], 100, 100);
  const { grip } = sandbox.trustedUpperBody(input);
  const flat = (p) => [p[0] * 100, p[1] * 100];
  const along = (from, k) => add(from, tool.map((value) => value * k));
  nearPoint(grip, along(gun, -0.3), '손잡이는 추적기에서 분사 축 반대로');
  const rod = capsuleAt(shapes, flat, along(grip, 0.11), along(gun, 0.07));
  const nozzle = capsuleAt(shapes, flat, along(gun, 0.07), along(gun, 0.105));
  assert.ok(rod, '연장대: 손잡이 앞 → 추적기 앞 .07 m');
  assert.ok(nozzle, '노즐: 추적기 앞 .07 → .105 m');
  assert.equal(light(nozzle), '#fb923c');
  assert.ok(nozzle.arcs[1].r > nozzle.arcs[0].r, '노즐은 끝으로 갈수록 넓다');
  // 건 몸통·손잡이 상자는 면(사각형)으로 그린다.
  assert.ok(shapes.filter((shape) => light(shape) === '#aab6c8' || light(shape) === '#5f6b7d').length >= 3);
  const noTool = scene();
  noTool.sandbox.drawTrustedUpperBody(frame({ tool: [0, 0, 0] }), [0, 0, 0], 100, 100);
  assert.ok(!noTool.shapes.some((shape) => light(shape) === '#fb923c'), '분사 축이 없으면 건을 그리지 않는다');
});

test('실제 투영에서 거울상이 아니다: 뒤에서 보면 건 든 오른팔이 머리 오른쪽, 앞에서 보면 왼쪽', () => {
  for (const [yaw, side] of [[0, 1], [Math.PI, -1]]) {
    const { sandbox, shapes } = scene({ real: true, yaw });
    sandbox.drawTrustedUpperBody(frame(), [1, 1.5, 2], 400, 400);
    const body = sandbox.trustedUpperBody(frame());
    const project = (p) => vm.runInContext(`project(${JSON.stringify(p)},[1,1.5,2],400,400)`, sandbox).slice(0, 2);
    const right = capsuleAt(shapes, project, body.rightArm.elbow, body.rightArm.hand);
    const left = capsuleAt(shapes, project, body.leftArm.elbow, body.leftArm.hand);
    const head = shapes.find((shape) => shape.ellipses.length === 1).ellipses[0].p;
    assert.ok(right && left);
    assert.ok(side * (right.arcs[1].p[0] - head[0]) > 0, `yaw ${yaw}: 오른손`);
    assert.ok(side * (left.arcs[1].p[0] - head[0]) < 0, `yaw ${yaw}: 왼손`);
  }
});

test('먼 것부터 그린다: 손이 몸 앞에 있으면 앞에서 볼 때 아래팔이 몸통 위, 뒤에서 볼 때 몸통 아래', () => {
  for (const [yaw, armOnTop] of [[Math.PI, true], [0, false]]) {
    const { sandbox, shapes } = scene({ real: true, yaw });
    sandbox.drawTrustedUpperBody(frame(), [1, 1.5, 2], 400, 400);
    const body = sandbox.trustedUpperBody(frame());
    const project = (p) => vm.runInContext(`project(${JSON.stringify(p)},[1,1.5,2],400,400)`, sandbox).slice(0, 2);
    const fore = shapes.indexOf(capsuleAt(shapes, project, body.rightArm.elbow, body.rightArm.hand));
    const torso = shapes.findIndex((shape) => shape.arcs.length === 6);
    assert.ok(fore >= 0 && torso >= 0);
    assert.equal(fore > torso, armOnTop, `yaw ${yaw}`);
  }
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
  for (const literal of ['.2106', '.1368', '.416', '.243', '#9aa3ad', '#e6edf7', '#22d3ee']) {
    assert.ok(table.includes(literal), `표에 ${literal} 이 있다`);
    assert.equal(code.filter((line) => line !== table && line.includes(literal)).length, 0, `코어에서 ${literal} 은 표 한 곳에만 있다`);
    if (literal !== '#22d3ee') assert.ok(!POSE.includes(literal), `포즈 스크립트에 ${literal} 을 다시 적지 않는다`);
  }
  assert.ok(!/#22d3ee/.test(UPPER_BODY), '큰 장면 상체 색은 LIMB_COLORS 에서 읽는다');
  assert.ok(!/\.3\b|0\.30/.test(UPPER_BODY), '손잡이 거리는 gunGrip 에서 읽는다');
  const panel = CORE.split('\n').find((line) => line.startsWith('function panelSkeleton('));
  assert.ok(panel.includes('twoBoneIK(') && panel.includes('gunGrip(') && !panel.includes('along='), '패널도 같은 IK 함수를 쓴다');
  assert.ok(UPPER_BODY.includes('twoBoneIK(') && UPPER_BODY.includes('gunGrip(') && UPPER_BODY.includes('BODY.'));
  assert.equal(CORE.split('along=').length - 1, 1, '두 마디 IK 식은 한 곳뿐이다');
  const drawPanel = CORE.split('\n').find((line) => line.startsWith('function drawSkeletonView('));
  assert.ok(drawPanel.includes('LIMB_COLORS.arm') && drawPanel.includes('LIMB_COLORS.leg') && drawPanel.includes('LIMB_COLORS.core'));
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

// 책임자 선택(2026-11-12): 중립 회색이 남색 화면 사이에서 따뜻해 보여 푸른 기가 도는 회색(B)으로 바꿨다.
test('큰 장면·패널 배경 토큰은 푸른 기가 도는 회색이고 두 화면이 같은 값을 쓴다', () => {
  for (const name of ['stage-top', 'stage']) {
    const value = token(V2_CSS, name), [r, g, b] = rgb(value);
    assert.ok(saturation(value) < 0.3, `--${name} ${value} 채도 ${saturation(value).toFixed(3)}`);
    assert.ok(b > g && g > r, `--${name} ${value} 는 파랑 쪽으로 기운다`);
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
