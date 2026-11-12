'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

// 화면 요소 흉내: 글자·툴팁·aria 를 자식까지 모아 읽는다.
function element() {
  return {
    className: '', textContent: '', title: '', hidden: false, children: [], attributes: {}, dataset: {},
    style: {setProperty(key, value) { this[key] = value; }},
    classList: {toggle() {}},
    append(...children) { this.children.push(...children); },
    replaceChildren(...children) { this.children = children; this.textContent = ''; },
    setAttribute(key, value) { this.attributes[key] = value; },
  };
}
function text(node) {
  return [node.textContent, node.title, node.attributes?.['aria-label'], ...(node.children || []).map(text)].filter(Boolean).join(' ');
}

function viewer(layout = null, {basis = false} = {}) {
  const events = {}, fills = [];
  const frames = [], resizeEvents = [], properties = new Map(), layoutEvents = {};
  const right = layout && {
    clientHeight: layout.height,
    style: {
      setProperty: (key, value) => properties.set(key, value),
      getPropertyValue: key => properties.get(key) || '',
      removeProperty: key => properties.delete(key),
    },
  };
  const screenHeight = () => Number.parseFloat(properties.get('--skeleton-view-height') || '310');
  const screen = layout && {getBoundingClientRect: () => ({height: screenHeight()})};
  const skeleton = layout && {
    open: true,
    getBoundingClientRect: () => ({height: screenHeight() + layout.chrome}),
    addEventListener: (name, callback) => { layoutEvents[name] = callback; },
  };
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
  if (basis) {
    for (const id of ['coatingBasisTarget', 'coatingBasisExpert', 'coatingTargetPanel']) nodes[id] = element();
  }
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../static/motion_viewer_coating.js'), 'utf8'), {
    document: {
      getElementById: id => nodes[id], createElement: () => (basis ? element() : {textContent: ''}),
      querySelector: selector => ({'.right': right, '.skeleton-card': skeleton, '.skeleton-screen': screen})[selector] || null,
    },
    window: {
      devicePixelRatio: 1,
      matchMedia: () => ({matches: layout.width > 1200}),
      dispatchEvent: event => { resizeEvents.push(event.type); events[event.type]?.(event); },
    },
    getComputedStyle: () => ({paddingTop: '12px', paddingBottom: '12px', rowGap: '10px'}),
    requestAnimationFrame: callback => frames.push(callback),
    Event: class { constructor(type) { this.type = type; } },
    addEventListener: (event, callback) => { events[event] = callback; },
    ResizeObserver: class { observe() {} },
  });
  return {
    nodes,
    right, skeleton, properties, resizeEvents,
    flushFrames: () => { while (frames.length) frames.shift()(); },
    resize: () => events.resize?.(),
    data: data => events['motion-viewer-data']({detail: data}),
    frame: (zone, spraying = true) => events['motion-viewer-frame']({detail: {frame: {
      map_num: '1', zone_id: zone, spraying,
    }}}),
    colors: () => fills.slice(-2),
    fills,
    panel: () => text(nodes.coatingTargetPanel),
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

test('desktop gives the coating card space and enlarges the skeleton when height allows', () => {
  const layout = {width: 1440, height: 768, chrome: 162};
  const view = viewer(layout);
  assert.equal(view.properties.get('--skeleton-view-height'), '212px');
  view.flushFrames();
  assert.deepEqual(view.resizeEvents, ['resize']);
  view.resize();
  view.flushFrames();
  assert.equal(view.resizeEvents.length, 1); // No resize recursion at a stable size.
  view.right.clientHeight = 848;
  view.resize();
  assert.equal(view.properties.get('--skeleton-view-height'), '292px');
  view.right.clientHeight = 1000;
  view.resize();
  assert.equal(view.properties.get('--skeleton-view-height'), '310px');
  view.right.clientHeight = 500;
  view.resize();
  assert.equal(view.properties.get('--skeleton-view-height'), '160px');
});

test('auto-height narrow layouts and collapsed skeletons remove the desktop override', () => {
  const layout = {width: 1440, height: 768, chrome: 162};
  const view = viewer(layout);
  view.skeleton.open = false;
  view.resize();
  assert.equal(view.properties.has('--skeleton-view-height'), false);
  view.skeleton.open = true;
  view.resize();
  assert.equal(view.properties.get('--skeleton-view-height'), '212px');
  layout.width = 1200;
  view.resize();
  assert.equal(view.properties.has('--skeleton-view-height'), false);
  view.flushFrames();
  assert.equal(view.resizeEvents.length, 1); // Changes coalesce into one redraw.
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

function target(overrides = {}) {
  return {
    analysis_status: 'available', reason_code: null, target_thickness_um: 160, target_basis: 'default_target',
    basis_label: '기본 목표 160µm 기준', normal_band_um: [160, 240], band_edges_um: [144, 160, 240, 320, 400],
    band_labels: ['90% 미만', '90~100%', '100~150%', '150~200%', '200~250%', '250% 이상'],
    grade_labels: {below: '목표 미만', normal: '정상', above: '1.5배 이상', none: '미도장'},
    zones: [{zone_id: '1', map_num: 1, grade: 'below', thickness_um: 98, unpainted_fraction: 0.12, normal_band_um: [160, 240],
      six_band_counts: {lt_0_9t: 70, from_0_9t_to_lt_1_0t: 10, from_1_0t_to_lt_1_5t: 20, from_1_5t_to_lt_2_0t: 0,
        from_2_0t_to_lt_2_5t: 0, gte_2_5t: 0}}],
    ...overrides,
  };
}

test('target basis is the default when the basis buttons exist', () => {
  const view = viewer(null, {basis: true});
  view.data({coating_reference: {'1:1': reference([100, 200])}, coating_evaluation: {target: target()}});
  view.frame('1');
  const panel = view.panel();
  assert.match(panel, /목표 미만/);
  assert.match(panel, /98/);
  assert.match(panel, /12%/);
  assert.match(panel, /기본 목표 160µm 기준/);
  assert.match(panel, /정상 160~240 µm/);
  assert.equal(view.nodes.coatingTargetPanel.hidden, false);
  assert.equal(view.nodes.coatingBasisTarget.attributes['aria-pressed'], 'true');
  assert.equal(view.fills.length, 0); // 목표 대비에서는 전문가 지도를 그리지 않는다.
});

test('grade and band labels come from the server', () => {
  const view = viewer(null, {basis: true});
  const labels = ['가짜 칸', '둘', '셋', '넷', '다섯', '여섯'];
  view.data({coating_evaluation: {target: target({band_labels: labels,
    grade_labels: {below: '가짜 표기', normal: '정상', above: '1.5배 이상', none: '미도장'}})}});
  view.frame('1');
  assert.match(view.panel(), /가짜 표기/);
  assert.match(view.panel(), /가짜 칸/);
});

test('expert basis is one click away and shows the expert map', () => {
  const view = viewer(null, {basis: true});
  view.data({coating_reference: {'1:1': reference([100, 200])}, coating_evaluation: {target: target()}});
  view.frame('1');
  view.nodes.coatingBasisExpert.onclick();
  assert.match(view.nodes.coatingStatus.textContent, /전문가 최종/);
  assert.equal(view.nodes.coatingTargetPanel.hidden, true);
  assert.equal(view.nodes.coatingBasisExpert.attributes['aria-pressed'], 'true');
  assert.ok(view.fills.length > 0);
  view.nodes.coatingBasisTarget.onclick();
  assert.equal(view.nodes.coatingTargetPanel.hidden, false);
});

test('a record without a height map says why and opens on the expert basis', () => {
  const view = viewer(null, {basis: true});
  view.data({coating_reference: {'1:1': reference([100, 200])},
    coating_evaluation: {target: {analysis_status: 'not_available', reason_code: 'height_map_unavailable'}}});
  view.frame('1');
  assert.match(view.nodes.coatingStatus.textContent, /전문가 최종/);
  assert.match(view.panel(), /높이맵 원본이 없어/);
  view.nodes.coatingBasisTarget.onclick();
  assert.match(view.panel(), /전문가 대비/);
  const other = viewer(null, {basis: true});
  other.data({coating_evaluation: {target: {analysis_status: 'not_available', reason_code: 'zone_reference_unavailable'}}});
  assert.match(other.panel(), /목표 대비 판정이 없습니다/);
  assert.doesNotMatch(other.panel(), /높이맵/);
});

test('a zone without a target row says so', () => {
  const view = viewer(null, {basis: true});
  view.data({coating_evaluation: {target: target()}});
  view.frame('2');
  assert.match(view.panel(), /이 Zone의 목표 대비 판정이 없습니다/);
});
