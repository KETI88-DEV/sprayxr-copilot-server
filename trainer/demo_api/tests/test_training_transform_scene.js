'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const tick = () => new Promise(resolve => setImmediate(resolve));
const ID = 'a'.repeat(32);

class Element {
  constructor(tag) { this.tagName = tag; this.children = []; this.events = {}; this.style = {}; this.attrs = {}; this.value = ''; this.textContent = ''; }
  append(...nodes) { for (const node of nodes) { node.parent = this; this.children.push(node); } }
  replaceChildren(...nodes) { this.children = []; this.append(...nodes); }
  remove() { if (this.parent) this.parent.children = this.parent.children.filter(node => node !== this); }
  setAttribute(key, value) { this.attrs[key] = String(value); }
  addEventListener(key, fn) { (this.events[key] ||= []).push(fn); }
  removeEventListener(key, fn) { this.events[key] = (this.events[key] || []).filter(value => value !== fn); }
  fire(key, event = {}) { for (const fn of this.events[key] || []) fn(event); }
  getBoundingClientRect() { return { width: 600, height: 280 }; }
  setPointerCapture() {}
  releasePointerCapture() {}
}
function fixture() {
  return { dataset_id: ID, worker: { worker_id: 'worker' }, process: 'spray', mode_id: 7, selected_session_id: '7_100_1', frames: [],
    paint_trace: [
      { point: [0, 0, 0], map_num: '1', zone_id: '1', segment: 0 },
      { point: [1, 0, 0], map_num: '1', zone_id: '1', segment: 0 },
      { point: [0, 1, 0], map_num: '1', zone_id: '2', segment: 1 },
      { point: [1, 1, 0], map_num: '1', zone_id: '2', segment: 1 },
    ], zone_surfaces: [{ map_num: '1', zone_id: '1', vertices: [[0, 0, 0], [1, 0, 0], [1, 1, 0]], boundary_edges: [[0, 1], [1, 2]], triangles: [[0, 1, 2]] }],
    reference_surface: { points: [] } };
}
function setup(fetcher) {
  const calls = [], lines = [], elements = [], observers = [], events = {}, statuses = [];
  const context2d = { setTransform() {}, clearRect() { lines.length = 0; }, fillRect() {}, fillText() {}, beginPath() { this.points = []; },
    moveTo(x, y) { this.points.push([x, y]); }, lineTo(x, y) { this.points.push([x, y]); }, closePath() {}, fill() {},
    stroke() { lines.push({ color: this.strokeStyle, points: [...this.points] }); }, arc() {} };
  const document = { createElement(tag) { const node = new Element(tag); if (tag === 'canvas') node.getContext = () => context2d; elements.push(node); return node; } };
  const window = { devicePixelRatio: 1, addEventListener(name, fn) { (events[name] ||= []).push(fn); }, removeEventListener(name, fn) { events[name] = (events[name] || []).filter(value => value !== fn); } };
  const sandbox = { window, document, AbortController, URLSearchParams, TextDecoder, Uint8Array, Number, Math,
    ResizeObserver: class { constructor(fn) { this.fn = fn; observers.push(this); } observe() {} disconnect() { this.disconnected = true; } },
    fetch: async (url, options) => { calls.push({ url, options }); return fetcher ? fetcher(url, options) : new Response(JSON.stringify(url.includes('/selection?') ? { dataset_id: ID, worker_id: 'worker', process: 'spray', mode_id: 7, session_id: '7_100_1' } : fixture())); } };
  const filename = path.join(__dirname, '../static/training_transform_scene.js');
  if (fs.existsSync(filename)) vm.runInNewContext(fs.readFileSync(filename, 'utf8'), sandbox);
  return { ...sandbox, calls, lines, elements, observers, events, statuses, host: new Element('host'), api: window.TrainingTransformScene };
}
async function settle() { for (let i = 0; i < 8; i++) await tick(); }

test('loading the script does not fetch; explicit create resolves the exact record and draws its Zone paths', async () => {
  const app = setup(); assert.ok(app.api, 'the preview API must exist'); assert.equal(app.calls.length, 0);
  const scene = app.api.create(app.host, { datasetId: ID, zoneKey: 'whole', onStatus: status => app.statuses.push(status) });
  await settle(); assert.equal(app.calls.length, 2);
  assert.equal(app.calls[0].url, `/motion-viewer/api/selection?dataset_id=${ID}`);
  const url = new URL(app.calls[1].url, 'http://localhost');
  assert.equal(url.searchParams.get('session_id'), '7_100_1'); assert.equal(url.searchParams.has('max_frames'), false);
  assert.equal(app.statuses.at(-1).status, 'ready');
  assert.ok(app.lines.some(line => line.color === '#FF0000')); assert.ok(app.lines.some(line => line.color === '#00FF00'));
  scene.selectZone('1:2'); assert.ok(app.lines.some(line => line.color === '#00FF00'));
  assert.ok(!app.lines.some(line => line.color === '#FF0000'), 'Zone selection must exclude other Zone geometry');
  scene.destroy();
});

test('a different dataset cannot be displayed as the requested record', async () => {
  const app = setup(url => new Response(JSON.stringify(url.includes('/selection?') ? { dataset_id: 'b'.repeat(32), worker_id: 'worker', process: 'spray', mode_id: 7, session_id: '7_100_1' } : fixture())));
  assert.ok(app.api, 'the preview API must exist');
  const scene = app.api.create(app.host, { datasetId: ID, onStatus: status => app.statuses.push(status) }); await settle();
  assert.equal(app.calls.length, 1); assert.equal(app.statuses.at(-1).status, 'error'); assert.equal(app.lines.length, 0); scene.destroy();
});

test('destroy aborts pending work and prevents late rendering, statuses and leaked resize listeners', async () => {
  let finish; const app = setup(() => new Promise(resolve => { finish = resolve; })); assert.ok(app.api, 'the preview API must exist');
  const scene = app.api.create(app.host, { datasetId: ID, onStatus: status => app.statuses.push(status) });
  const count = app.statuses.length; scene.destroy();
  assert.equal(app.calls[0].options.signal.aborted, true); assert.equal(app.host.children.length, 0);
  finish(new Response(JSON.stringify({ dataset_id: ID, worker_id: 'worker', process: 'spray', mode_id: 7, session_id: '7_100_1' }))); await settle();
  assert.equal(app.calls.length, 1); assert.equal(app.statuses.length, count); assert.ok(app.observers[0].disconnected);
});

test('oversized response is rejected before decoding or a second request', async () => {
  let read = false; const app = setup(() => ({ ok: true, headers: { get: key => key === 'Content-Length' ? '999999999' : null }, body: { getReader() { read = true; throw new Error('must not read'); } } }));
  assert.ok(app.api, 'the preview API must exist');
  const scene = app.api.create(app.host, { datasetId: ID, onStatus: status => app.statuses.push(status) }); await settle();
  assert.equal(read, false); assert.equal(app.statuses.at(-1).status, 'error'); assert.equal(app.calls.length, 1); scene.destroy();
});

test('invalid points and discontinuous segments never produce invalid coordinates or invented connecting strokes', async () => {
  const payload = fixture(); payload.paint_trace = [
    { point: [0, 0, 0], map_num: 1, zone_id: '1', segment: 0 }, { point: [1, 0, 0], map_num: 1, zone_id: '1', segment: 0 },
    { point: [null, 1, 0], map_num: 1, zone_id: '1', segment: 0 }, { point: [100, 1, 0], map_num: 1, zone_id: '1', segment: 1 },
    { point: [101, 1, 0], map_num: 1, zone_id: '1', segment: 1 },
  ]; payload.zone_surfaces = []; const app = setup(url => new Response(JSON.stringify(url.includes('/selection?') ? { dataset_id: ID, worker_id: 'worker', process: 'spray', mode_id: 7, session_id: '7_100_1' } : payload)));
  assert.ok(app.api, 'the preview API must exist'); const scene = app.api.create(app.host, { datasetId: ID }); await settle();
  const paths = app.lines.filter(line => line.color === '#FF0000'); assert.equal(paths.length, 2);
  assert.ok(paths.every(line => line.points.length === 2 && line.points.flat().every(Number.isFinite))); scene.destroy();
});

test('drag rotation and reset redraw on demand without an animation loop', async () => {
  const app = setup(); assert.ok(app.api, 'the preview API must exist');
  const scene = app.api.create(app.host, { datasetId: ID, zoneKey: '1:2' }); await settle();
  const canvas = app.elements.find(node => node.tagName === 'canvas'); const before = JSON.stringify(app.lines);
  canvas.fire('pointerdown', { button: 0, pointerId: 1, clientX: 100, clientY: 100 });
  canvas.fire('pointermove', { pointerId: 1, clientX: 220, clientY: 110 }); canvas.fire('pointerup', { pointerId: 1 });
  assert.notEqual(JSON.stringify(app.lines), before);
  app.elements.find(node => node.tagName === 'button').fire('click'); assert.equal(JSON.stringify(app.lines), before); scene.destroy();
});

test('a correct selection cannot disguise motion from another training session', async () => {
  const payload = fixture(); payload.selected_session_id = '7_101_1';
  const app = setup(url => new Response(JSON.stringify(url.includes('/selection?') ? { dataset_id: ID, worker_id: 'worker', process: 'spray', mode_id: 7, session_id: '7_100_1' } : payload)));
  const scene = app.api.create(app.host, { datasetId: ID, onStatus: status => app.statuses.push(status) }); await settle();
  assert.equal(app.statuses.at(-1).status, 'error'); assert.equal(app.lines.length, 0); scene.destroy();
});

test('unknown-worker group selection accepts the original worker ID preserved by the motion endpoint', async () => {
  const payload = fixture(); payload.worker = { worker_id: 'original-123', selection_worker_id: '__unknown_workers__' };
  const app = setup(url => new Response(JSON.stringify(url.includes('/selection?') ? { dataset_id: ID, worker_id: '__unknown_workers__', process: 'spray', mode_id: 7, session_id: '7_100_1' } : payload)));
  const scene = app.api.create(app.host, { datasetId: ID, onStatus: status => app.statuses.push(status) }); await settle();
  assert.equal(app.statuses.at(-1).status, 'ready'); assert.ok(app.lines.some(line => line.color === '#FF0000'));
  scene.destroy();
});

test('a worker selection-group hint cannot authorize another dataset or training session', async () => {
  for (const changed of [{ dataset_id: 'b'.repeat(32) }, { selected_session_id: '7_999_1' }]) {
    const payload = { ...fixture(), ...changed, worker: { worker_id: 'original-123', selection_worker_id: '__unknown_workers__' } };
    const app = setup(url => new Response(JSON.stringify(url.includes('/selection?') ? { dataset_id: ID, worker_id: '__unknown_workers__', process: 'spray', mode_id: 7, session_id: '7_100_1' } : payload)));
    const scene = app.api.create(app.host, { datasetId: ID, onStatus: status => app.statuses.push(status) }); await settle();
    assert.equal(app.statuses.at(-1).status, 'error'); assert.equal(app.lines.length, 0); scene.destroy();
  }
});

test('a response without Content-Length is bounded while streaming and its reader is released', async () => {
  let cancelled = false, released = false, reads = 0;
  const app = setup(() => ({ ok: true, headers: { get() { return null; } }, body: { getReader() { return {
    async read() { reads++; return { done: false, value: new Uint8Array(13 * 1024 * 1024) }; },
    async cancel() { cancelled = true; }, releaseLock() { released = true; },
  }; } } }));
  const scene = app.api.create(app.host, { datasetId: ID, onStatus: status => app.statuses.push(status) }); await settle();
  assert.equal(app.statuses.at(-1).status, 'error'); assert.equal(reads, 2); assert.equal(cancelled, true); assert.equal(released, true); scene.destroy();
});

test('the preview retains the existing viewer handedness for X, Y and depth rather than reflecting Z', async () => {
  const payload = fixture(), points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]];
  payload.paint_trace = points.map(p => ({ point: p, map_num: 1, zone_id: '1', segment: 0 }));
  payload.zone_surfaces = []; payload.frames = [{ camera_forward: [0, 0, 1] }];
  const app = setup(url => new Response(JSON.stringify(url.includes('/selection?') ? { dataset_id: ID, worker_id: 'worker', process: 'spray', mode_id: 7, session_id: '7_100_1' } : payload)));
  const scene = app.api.create(app.host, { datasetId: ID }); await settle();
  const lines = app.lines.filter(line => line.color === '#FF0000'); assert.equal(lines.length, 3);
  // Hand-calculated bounds: centre (.5,.5,.5), radius sqrt(3)/2. Normalize to
  // the existing viewer's fixed 4.5 perspective distance before comparing.
  const radius = Math.sqrt(3) / 2, core = fs.readFileSync(path.join(__dirname, '../motion_viewer.py'), 'utf8');
  const sandbox = vm.createContext({ yaw: 0, pitch: 0, zoom: 280 * .36, panX: 0, panY: 0 });
  vm.runInContext(core.split('\n').filter(line => line.startsWith('function transform(') || line.startsWith('function project(')).join('\n'), sandbox);
  for (let index = 0; index < lines.length; index++) for (let endpoint = 0; endpoint < 2; endpoint++) {
    const normalized = points[index + endpoint].map(value => (value - .5) / radius);
    const expected = sandbox.project(normalized, [0, 0, 0], 600, 280);
    for (let axis = 0; axis < 2; axis++) assert.ok(Math.abs(lines[index].points[endpoint][axis] - expected[axis]) < 1e-8);
  }
  scene.destroy();
});

test('only a live internal Zone choice notifies the surrounding CSV/JSON view without public select loops', async () => {
  const changes = [], app = setup();
  const scene = app.api.create(app.host, { datasetId: ID, onZoneChange: key => changes.push(key) }); await settle();
  const select = app.elements.find(node => node.tagName === 'select');
  scene.selectZone('1:1'); assert.deepEqual(changes, [], 'external synchronization must not notify back');
  select.value = '1:2'; select.fire('change'); assert.deepEqual(changes, ['1:2']);
  assert.ok(app.lines.some(line => line.color === '#00FF00')); assert.ok(!app.lines.some(line => line.color === '#FF0000'));
  scene.destroy(); select.value = '1:1'; select.fire('change'); scene.selectZone('whole');
  assert.deepEqual(changes, ['1:2'], 'a removed preview must never update its previous parent');
});
