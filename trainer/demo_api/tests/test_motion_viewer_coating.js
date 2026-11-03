'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function viewer() {
  const events = {}, fills = [];
  const context = {
    setTransform() {}, fillText() {}, strokeRect() {}, beginPath() {}, moveTo() {},
    lineTo() {}, closePath() {}, stroke() {},
    fillRect() { fills.push(this.fillStyle); },
    fill() { fills.push(this.fillStyle); },
  };
  const nodes = Object.fromEntries([
    'coatingCanvas', 'coatingZone', 'coatingStatus', 'coatingView2d', 'coatingView3d',
    'coatingScaleRelative', 'coatingScaleAbsolute', 'coatingLegend', 'coatingScaleNote',
    'coatingGeometryNote',
  ].map(id => [id, {
    textContent: '', innerHTML: '', disabled: false, dataset: {},
    classList: {toggle() {}}, setAttribute() {},
    replaceChildren(...children) { this.textContent = children.map(child => child.textContent).join(' '); },
  }]));
  Object.assign(nodes.coatingCanvas, {
    getContext: () => context,
    getBoundingClientRect: () => ({width: 400, height: 250}), parentElement: {},
  });
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../static/motion_viewer_coating.js'), 'utf8'), {
    document: {getElementById: id => nodes[id], createElement: () => ({textContent: ''})},
    window: {devicePixelRatio: 1},
    addEventListener: (event, callback) => { events[event] = callback; },
    ResizeObserver: class { observe() {} },
  });
  return {
    nodes,
    data: data => events['motion-viewer-data']({detail: data}),
    frame: (zone, spraying = true) => events['motion-viewer-frame']({detail: {frame: {
      map_num: '1', zone_id: zone, spraying,
    }}}),
    colors: () => fills.slice(-2),
  };
}

function reference(cells, physical = true) {
  return {
    source: 'expert_final_heightmap', session_count: 2,
    cells: cells.map((value, x) => [x, 0, value]),
    bounds: {min_x: 0, max_x: cells.length - 1, min_y: 0, max_y: 0},
    physical_calibration: physical ? 'available' : 'unavailable',
    cell_unit: physical ? 'um' : null,
    color_scale_um: physical ? {minimum: 0, maximum: 1000, clamp: true} : null,
  };
}

test('absolute colors compare the same thickness across zones in 2D and 3D', () => {
  const view = viewer();
  view.data({coating_reference: {'1:1': reference([100, 200]), '1:2': reference([200, 1000])}});
  view.frame('1');
  view.nodes.coatingScaleAbsolute.onclick();
  const first2d = view.colors()[1];
  // 200 µm is the maximum of Zone 1 and the minimum of Zone 2. Its absolute
  // color must still be the same point (20%) on the common 0–1000 µm scale.
  assert.equal(first2d, 'rgb(27,156,186)');
  assert.match(view.nodes.coatingLegend.textContent, /1000.*µm/);
  assert.match(view.nodes.coatingScaleNote.textContent, /공통/);
  view.frame('2');
  assert.equal(view.colors()[0], first2d);
  view.nodes.coatingView3d.onclick();
  const second3d = view.colors();
  view.frame('1');
  assert.equal(view.colors()[0], second3d[1]);
  assert.match(view.nodes.coatingGeometryNote.textContent, /높이.*강조/);
});

test('relative mode remains selectable and identifies its per-zone normalization', () => {
  const view = viewer();
  view.data({coating_reference: {'1:1': reference([100, 200, 300]), '1:2': reference([100, 200, 1000])}});
  view.frame('1');
  view.nodes.coatingScaleRelative.onclick();
  const firstMiddle = view.colors()[0];
  view.frame('2');
  assert.notEqual(view.colors()[0], firstMiddle);
  assert.match(view.nodes.coatingLegend.textContent, /0%.*50%.*100%/);
  assert.match(view.nodes.coatingScaleNote.textContent, /Zone.*상대/);
});

test('held zone survives non-spray gaps and switches on the next valid spray zone', () => {
  const view = viewer();
  view.data({coating_reference: {'1:1': reference([100]), '1:2': reference([200])}});
  view.frame('1');
  view.frame('-1', false);
  assert.match(view.nodes.coatingZone.textContent, /01/);
  view.frame('2');
  assert.match(view.nodes.coatingZone.textContent, /02/);
});

test('unknown calibration and empty cells give truthful fallback and no NaN legend', () => {
  const view = viewer();
  view.data({coating_reference: {'1:1': reference([10, null, 'bad', 20], false), '1:2': reference([])}});
  view.frame('1');
  assert.equal(view.nodes.coatingScaleAbsolute.disabled, true);
  assert.match(view.nodes.coatingScaleNote.textContent, /물리.*보정.*없/);
  assert.match(view.nodes.coatingStatus.textContent, /전문가 최종/);
  view.frame('2');
  assert.match(view.nodes.coatingStatus.textContent, /셀.*없/);
  assert.doesNotMatch(view.nodes.coatingLegend.textContent, /NaN|Infinity/);
});
