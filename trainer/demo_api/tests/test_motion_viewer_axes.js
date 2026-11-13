'use strict';
// 우하단 좌표축 표시가 카메라 회전과 맞고, X·Y·Z 버튼과 같은 색 상수를 쓰는지 본다.
// 실행(호스트): node --test trainer/demo_api/tests/test_motion_viewer_axes.js
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const root = path.join(__dirname, '..');
const read = (file) => fs.readFileSync(path.join(root, file), 'utf8');
const coreTransform = read('motion_viewer.py').split('\n').find((line) => line.startsWith('function transform('));
assert.ok(coreTransform, '코어 transform 을 찾지 못했다');

const sandbox = vm.createContext({ window: {}, yaw: 0, pitch: 0 });
vm.runInContext(coreTransform, sandbox);
vm.runInContext(read('static/motion_viewer_camera.js'), sandbox);
const { create, AXIS_COLORS, axisGizmo } = sandbox.window.MotionViewerCamera;

/** 카메라를 view 로 돌린 뒤 축마다 {dx, dy(아래 양수), depth(클수록 멀다)}. */
function axesAt(view) {
  const camera = create();
  const state = view ? camera.axisView(view) : camera.state();
  sandbox.yaw = state.yaw; sandbox.pitch = state.pitch;
  return Object.fromEntries(axisGizmo(sandbox.transform).map((item) => [item.axis, item]));
}
const near = (value, expected) => assert.ok(Math.abs(value - expected) < 1e-3, `${value} != ${expected}`);

test('처음 시점에서 Y 는 위, X 는 오른쪽 위쪽으로 비껴 보인다', () => {
  const a = axesAt(null);
  assert.ok(a.y.dy < -0.9 && Math.abs(a.y.dx) < 0.2, 'Y 가 위');
  assert.ok(a.x.dx > 0.7, 'X 가 오른쪽');
});

test('Z 시점: X 오른쪽, Y 위, Z 는 화면 안쪽(점)', () => {
  const a = axesAt('z');
  near(a.x.dx, 1); near(a.x.dy, 0); near(a.y.dx, 0); near(a.y.dy, -1);
  near(a.z.dx, 0); near(a.z.dy, 0); assert.ok(a.z.depth > 0.99);
});

test('X 시점: Z 는 왼쪽, Y 위, X 는 화면 안쪽(점)', () => {
  const a = axesAt('x');
  near(a.z.dx, -1); near(a.z.dy, 0); near(a.y.dx, 0); near(a.y.dy, -1);
  near(a.x.dx, 0); near(a.x.dy, 0);
});

test('Y 시점(위에서): X 오른쪽, Z 위쪽, Y 는 보는 사람 쪽(점)', () => {
  const a = axesAt('y');
  near(a.x.dx, 1); near(a.z.dy, -1);
  assert.ok(Math.hypot(a.y.dx, a.y.dy) < 0.01 && a.y.depth < 0, 'Y 가 보는 쪽');
});

test('먼 축부터 그려지도록 깊이 내림차순이다', () => {
  const depths = axisGizmo(sandbox.transform).map((item) => item.depth);
  assert.deepEqual([...depths], [...depths].sort((p, q) => q - p));
});

test('X·Y·Z 버튼과 축 표시는 한 색 상수(AXIS_COLORS)를 쓰고 어디에도 값이 복제되지 않는다', () => {
  const pose = read('static/motion_viewer_pose.js');
  assert.match(pose, /axisGizmo\(transform\)/);
  assert.match(pose, /\.style\.color = window\.MotionViewerCamera\.AXIS_COLORS\[axis\]/);
  const others = ['static/motion_viewer_pose.js', 'static/motion_viewer_pose.css', 'static/motion_viewer_v2.css', 'static/motion_viewer_v2.html', 'motion_viewer.py'];
  for (const file of others) for (const color of Object.values(AXIS_COLORS)) assert.ok(!read(file).toLowerCase().includes(color), `${file} 에 ${color} 복제`);
  assert.ok(!/#cameraAxis[XYZ]\s*\{[^}]*color/.test(read('static/motion_viewer_pose.css')));
});

test('축 색은 3D 공간 배경(위 #2e3646, 아래 #1c2230) 위에서 대비 3:1 이상이다', () => {
  const lum = (hex) => {
    const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255).map((v) => (v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4));
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  };
  for (const bg of ['#2e3646', '#1c2230']) for (const [axis, color] of Object.entries(AXIS_COLORS)) {
    const [hi, lo] = [lum(color), lum(bg)].sort((p, q) => q - p);
    assert.ok((hi + 0.05) / (lo + 0.05) >= 3, `${axis} ${color} on ${bg}`);
  }
});

test('처음 시점 버튼은 집이 아니라 전체 보기 아이콘과 이름이다', () => {
  assert.match(read('static/motion_viewer_v2.html'), /id="cameraHome"[^>]*aria-label="전체 보기"[^>]*><i data-icon="focus">/);
  assert.ok(!read('motion_viewer.py').includes('>⌂<'));
  assert.match(read('motion_viewer.py'), /id="cameraHome" aria-label="전체 보기" title="전체 보기"><svg/);
});

test('글을 쓰는 칸(입력·선택·여러 줄 글·편집 가능 영역)에서는 장면 단축키를 쓰지 않는다', () => {
  const { isTyping } = sandbox.window.MotionViewerCamera;
  for (const tagName of ['INPUT', 'SELECT', 'TEXTAREA']) assert.equal(isTyping({ tagName }), true, tagName);
  assert.equal(isTyping({ tagName: 'DIV', isContentEditable: true }), true);
  assert.equal(isTyping({ tagName: 'CANVAS' }), false);
  assert.equal(isTyping(null), false);
});
