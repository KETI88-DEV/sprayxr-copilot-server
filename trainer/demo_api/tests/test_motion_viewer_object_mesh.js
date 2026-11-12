'use strict';
// 큰 3D 장면의 대상 구조물 레이어 시험. 실행(호스트): node --test trainer/demo_api/tests/test_motion_viewer_object_mesh.js
// 실제 화면 코드(motion_viewer.py 의 코어 스크립트)에서 함수만 뽑아 node:vm 에 올리고, 캔버스가 받은 경로를 읽는다.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function coreLines() {
  const source = fs.readFileSync(path.join(__dirname, '../motion_viewer.py'), 'utf8');
  const html = source.slice(source.indexOf('_VIEWER_HTML = r"""'));
  const start = html.indexOf('<script>') + '<script>'.length;
  return html.slice(start, html.indexOf('</script>', start)).split('\n');
}

function viewer() {
  const lines = coreLines();
  const pick = (marker) => {
    const line = lines.find((item) => item.startsWith(marker));
    assert.ok(line, `화면 코드에서 ${marker} 를 찾지 못했다`);
    return line;
  };
  // calls: fill·stroke 마다 {kind, style, subpaths}. subpath 는 화면 점 목록이다.
  const calls = [];
  let subpaths = [];
  const ctx = {
    lineWidth: 1, lineCap: 'butt', strokeStyle: '', fillStyle: '',
    beginPath() { subpaths = []; },
    moveTo(x, y) { subpaths.push([[x, y]]); },
    lineTo(x, y) { subpaths.at(-1).push([x, y]); },
    closePath() {},
    fill() { calls.push({ kind: 'fill', style: this.fillStyle, subpaths: subpaths.map((item) => [...item]) }); },
    stroke() { calls.push({ kind: 'stroke', style: this.strokeStyle, subpaths: subpaths.map((item) => [...item]) }); },
  };
  const sandbox = vm.createContext({ ctx });
  vm.runInContext([
    'var yaw=.4,pitch=.2,zoom=175,panX=0,panY=0;',
    pick('function add(a,b)'), pick('function transform('), pick('function project('),
    pick('let outlineSource='), pick('function meshOutline('), pick('function drawObjectMesh('),
  ].join('\n'), sandbox);
  return {
    sandbox, calls,
    draw(mesh, status = 'verified') { calls.length = 0; sandbox.drawObjectMesh(mesh, status, [1, 4, 0.5], 800, 600); return calls; },
  };
}

// 상자 하나. 면마다 정점을 따로 둔다(FBX 처럼 모서리에서 정점이 갈라진다). 중심을 z≠0 에 두어 Z 반전이 드러나게 한다.
function box([x0, y0, z0], [x1, y1, z1]) {
  const corner = (i) => [i & 1 ? x1 : x0, i & 2 ? y1 : y0, i & 4 ? z1 : z0];
  const faces = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]];
  const vertices = [], triangles = [];
  for (const face of faces) {
    const base = vertices.length;
    vertices.push(...face.map(corner));
    triangles.push([base, base + 1, base + 2], [base, base + 2, base + 3]);
  }
  return { available: true, vertices, triangles };
}

const bounds = (points) => [0, 1, 2].map((axis) => [Math.min(...points.map((p) => p[axis])), Math.max(...points.map((p) => p[axis]))]);
const near = (a, b) => Math.abs(a[0] - b[0]) < 1e-9 && Math.abs(a[1] - b[1]) < 1e-9;

test('검증된 구조물은 면 하나와 바깥·꺾인 모서리만 그린다(삼각형 대각선은 긋지 않는다)', () => {
  const view = viewer();
  const mesh = box([1, 3.5, 0.2], [1.4, 5.2, 1.4]);
  const calls = view.draw(mesh);
  assert.deepEqual(calls.map((call) => call.kind), ['fill', 'stroke']);
  assert.equal(calls[0].subpaths.length, mesh.triangles.length);
  assert.equal(calls[1].subpaths.length, 12, '상자 모서리는 12개다');
  // 단순화한 윤곽이 바깥 크기를 그대로 지킨다.
  const outline = view.sandbox.meshOutline(mesh).flat();
  assert.deepEqual(bounds(outline), bounds(mesh.vertices));
});

test('평면 사각형은 테두리 넷만 긋는다', () => {
  const view = viewer();
  const quad = { available: true, vertices: [[0, 4, 0], [1, 4, 0], [1, 5, 0], [0, 5, 0]], triangles: [[0, 1, 2], [0, 2, 3]] };
  assert.equal(view.draw(quad)[1].subpaths.length, 4);
});

test('면은 화면 감김을 한 방향으로 맞춰 칠한다(앞·뒤 면이 겹쳐도 nonzero 칠하기에 구멍이 나지 않는다)', () => {
  const view = viewer();
  const [fill] = view.draw(box([1, 3.5, 0.2], [1.4, 5.2, 1.4]));
  const signs = fill.subpaths.map(([a, b, c]) => Math.sign((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])));
  assert.deepEqual([...new Set(signs.filter(Boolean))], [1]);
});

test('모델이 없거나 점 대체(ray_point_fallback)면 아무것도 그리지 않는다', () => {
  const view = viewer();
  const mesh = box([0, 0, 0], [1, 1, 1]);
  assert.equal(view.draw(null).length, 0);
  assert.equal(view.draw({ ...mesh, available: false }).length, 0);
  assert.equal(view.draw(mesh, 'ray_point_fallback').length, 0);
  assert.equal(view.draw({ available: true, vertices: [], triangles: [] }).length, 0);
});

test('구조물은 Zone 면과 같은 project() 자리에 그려진다(Z 를 뒤집은 거울상이 아니다)', () => {
  const view = viewer();
  const mesh = box([1, 3.5, 0.2], [1.4, 5.2, 1.4]);
  const projected = mesh.vertices.map((vertex) => view.sandbox.project(vertex, [1, 4, 0.5], 800, 600));
  const mirrored = mesh.vertices.map(([x, y, z]) => view.sandbox.project([x, y, -z], [1, 4, 0.5], 800, 600));
  for (const call of view.draw(mesh)) {
    for (const point of call.subpaths.flat()) {
      assert.ok(projected.some((item) => near(item, point)), `그린 점 ${point} 이 자료 정점의 투영이 아니다`);
    }
  }
  const drawn = view.calls.flatMap((call) => call.subpaths.flat());
  assert.ok(!mirrored.every((item) => drawn.some((point) => near(item, point))), '거울상 정점과 겹치면 시험이 판별하지 못한다');
});
