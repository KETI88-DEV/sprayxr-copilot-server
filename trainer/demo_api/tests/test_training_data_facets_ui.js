'use strict';
// Node executes the real browser asset; only its DOM and network boundaries are replaced.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const sourcePath = process.env.TRAINING_DATA_ASSET || path.join(__dirname, '../static/training_data.js');
const source = fs.readFileSync(sourcePath, 'utf8');

class Element {
  constructor(tag = 'div', id = '') {
    this.tagName = tag.toUpperCase(); this.id = id; this.children = []; this.listeners = {};
    this.attrs = {}; this.dataset = {}; this._value = ''; this._text = ''; this.disabled = false;
    this.className = ''; this.open = false; this.isConnected = true; this.style = {};
    const classes = new Set();
    this.classList = {add: (...names) => names.forEach(name => classes.add(name)), remove: (...names) => names.forEach(name => classes.delete(name)), contains: name => classes.has(name)};
  }
  get options() { return this.children.filter(child => child.tagName === 'OPTION'); }
  get selectedOptions() { return this.options.filter(option => option.value === this.value); }
  set value(value) {
    const text = String(value);
    this._value = this.tagName === 'SELECT' && !this.options.some(option => option.value === text) ? '' : text;
  }
  get value() { return this._value; }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return this._text + this.children.map(child => child.textContent ?? String(child)).join(''); }
  get label() { return this._label ?? this.textContent; }
  set label(value) { this._label = String(value); }
  get text() { return this.textContent; }
  set text(value) { this.textContent = value; }
  append(...children) {
    for (const child of children) { child.parentNode = this; this.children.push(child); }
    if (this.tagName === 'SELECT' && !this.options.some(option => option.value === this._value)) this._value = this.options[0]?.value ?? '';
  }
  replaceChildren(...children) { this.children = []; this._text = ''; this._value = ''; this.append(...children); }
  add(option) { this.append(option); }
  setAttribute(name, value) { this.attrs[name] = String(value); if (name === 'disabled') this.disabled = true; }
  getAttribute(name) { return this.attrs[name] ?? null; }
  removeAttribute(name) { delete this.attrs[name]; if (name === 'disabled') this.disabled = false; }
  addEventListener(name, handler) { (this.listeners[name] ||= []).push(handler); }
  emit(name, extra = {}) {
    const event = {type: name, target: this, preventDefault() { this.defaultPrevented = true; }, ...extra};
    // Native dispatchEvent/requestSubmit do not wait for async listener results.
    for (const handler of this.listeners[name] || []) handler(event);
    return event;
  }
  requestSubmit() { return this.emit('submit'); }
  focus() { if (this.ownerDocument) this.ownerDocument.activeElement = this; }
  showModal() { this.open = true; }
  close() { this.open = false; return this.emit('close'); }
}

class Option extends Element {
  constructor(label, value) { super('option'); this.textContent = label; this.value = value; }
}

const option = (id, label = id) => ({id, label});
const universe = {
  workers: [option('w1', '작업자 1'), option('w2', '작업자 2')],
  processes: [option('spray', '스프레이 도장'), option('blasting', '블라스팅')],
  modes: [option('1', '학습모드 1'), option('2', '학습모드 2')],
  qualities: [option('needs_review', '확인 필요'), option('complete', '완료')],
};
function listing(options = universe, filtered = 8, extras = {}) {
  return {available: true, total: 20, filtered_count: filtered, page: 1, pages: 1,
    summary: {worker_count: 2, complete: filtered, needs_review: 0, heightmap_count: filtered, by_worker: [], by_mode: [], by_process: []},
    options, items: [], ...extras};
}
const settle = async () => { for (let index = 0; index < 3; index++) await new Promise(resolve => setImmediate(resolve)); };

function setup(search = '') {
  const elements = new Map();
  const selectIds = new Set(['td-worker', 'td-process', 'td-mode', 'td-quality', 'td-limit']);
  const document = {
    activeElement: null,
    getElementById(id) {
      if (!elements.has(id)) {
        const node = new Element(selectIds.has(id) ? 'select' : id === 'td-search' ? 'input' : id === 'td-filter-form' ? 'form' : 'div', id);
        node.ownerDocument = document; elements.set(id, node);
      }
      return elements.get(id);
    },
    createElement: tag => new Element(tag), createElementNS: (_namespace, tag) => new Element(tag),
  };
  document.body = new Element('body');
  for (const id of ['td-worker', 'td-process', 'td-mode', 'td-quality']) document.getElementById(id).add(new Option('전체', 'all'));
  for (const entry of universe.qualities) document.getElementById('td-quality').add(new Option(entry.label, entry.id));
  for (const limit of [10, 25, 50, 100]) document.getElementById('td-limit').add(new Option(String(limit), String(limit)));
  document.getElementById('td-limit').value = '25';
  let location = new URL(`http://copilot.test/admin?${search.replace(/^\?/, '')}`);
  const window = {
    get location() { return location; },
    history: {replaceState(_state, _title, next) { location = new URL(String(next), location); }},
  };
  const requests = [];
  const fetch = (url, options) => new Promise((resolve, reject) => {
    // Intentionally ignore abort at the transport boundary: late responses must still be rejected by the real asset.
    requests.push({url, options, resolve: data => resolve({ok: true, status: 200, json: async () => data}), reject});
  });
  vm.runInNewContext(source, {document, window, fetch, Option, Node: Element, AbortController, URL, URLSearchParams, Intl, console, setTimeout, clearTimeout}, {filename: sourcePath});
  return {
    document, requests, get: id => document.getElementById(id), location: () => location, presentation: window.TrainingDataPresentation,
    values: id => document.getElementById(id).options.map(entry => entry.value),
    async respond(index, data) { requests[index].resolve(data); await settle(); },
    async reject(index, error = new Error('late network error')) { requests[index].reject(error); await settle(); },
    async change(id, value) { const node = document.getElementById(id); node.value = value; assert.equal(node.value, value, 'test must select an existing option'); await node.emit('change'); },
  };
}

test('reusable coating presentation isolates inspection IDs and leaves library controls intact',async()=>{
  const app=setup();assert.equal(typeof app.presentation?.coating,'function');assert.equal(typeof app.presentation?.json,'function');
  const data={dataset_id:'a'.repeat(32),metadata:{process:'blasting'},coating:{maps:[{map_num:1,status:'not_applicable',image_url:'/admin/api/training-data/a/images/0'}]}};
  const library=app.presentation.coating(data),inspection=app.presentation.coating(data,{idPrefix:'tv-inspection',isolated:true});
  const flatten=node=>node.children.flatMap(child=>[child,...flatten(child)]);
  assert.ok(flatten(library).some(node=>node.id==='td-map-select'));
  assert.ok(flatten(inspection).some(node=>node.id==='tv-inspection-map-select'));
  assert.ok(!flatten(inspection).some(node=>node.id?.startsWith('td-')));
  assert.match(inspection.textContent,/블라스팅.*도막 두께/);
});

test('a worker change renders all four server facets and sends the current search with every selected filter', async () => {
  const app = setup(); await app.respond(0, listing());
  app.get('td-search').value = ' nozzle ';
  await app.change('td-worker', 'w1');
  const request = new URL(app.requests[1].url, app.location());
  assert.equal(request.searchParams.get('worker_id'), 'w1');
  assert.equal(request.searchParams.get('process'), 'all');
  assert.equal(request.searchParams.get('scenario_id'), 'all');
  assert.equal(request.searchParams.get('quality_status'), 'all');
  assert.equal(request.searchParams.get('q'), 'nozzle');
  assert.equal(request.searchParams.get('page'), '1');
  const facets = {workers: universe.workers, processes: [universe.processes[0]], modes: [universe.modes[0]], qualities: [universe.qualities[1]]};
  await app.respond(1, listing(facets, 2));
  assert.deepEqual(app.values('td-worker'), ['all', 'w1', 'w2'], 'worker facet must permit changing its own selection');
  assert.deepEqual(app.values('td-process'), ['all', 'spray']);
  assert.deepEqual(app.values('td-mode'), ['all', '1']);
  assert.deepEqual(app.values('td-quality'), ['all', 'complete'], 'quality options must come from options.qualities, not a fixed pair');
  assert.equal(app.get('td-worker').value, 'w1');
});

test('a quality change narrows the other three facets but retains its own alternatives supplied by the server', async () => {
  const app = setup(); await app.respond(0, listing());
  await app.change('td-quality', 'needs_review');
  const facets = {workers: [universe.workers[1]], processes: [universe.processes[1]], modes: [universe.modes[1]], qualities: universe.qualities};
  await app.respond(1, listing(facets, 1));
  assert.deepEqual(app.values('td-worker'), ['all', 'w2']);
  assert.deepEqual(app.values('td-process'), ['all', 'blasting']);
  assert.deepEqual(app.values('td-mode'), ['all', '2']);
  assert.deepEqual(app.values('td-quality'), ['all', 'needs_review', 'complete']);
  assert.equal(app.get('td-quality').value, 'needs_review');
});

test('an empty facet never invents unrelated normal options', async () => {
  const app = setup();
  await app.respond(0, listing({workers: [], processes: [], modes: [], qualities: []}, 0));
  for (const id of ['td-worker', 'td-process', 'td-mode', 'td-quality']) assert.deepEqual(app.values(id), ['all']);
});

test('direct URL selections absent from available facets remain selected as disabled zero-result options', async t => {
  const app = setup('view=training&worker_id=missing-worker&process=grinding&scenario_id=9&quality_status=complete&q=needle&page=4&limit=50');
  await app.respond(0, listing({workers: [universe.workers[0]], processes: [universe.processes[0]], modes: [universe.modes[0]], qualities: [universe.qualities[0]]}, 0));
  for (const [id, expected] of [['td-worker', 'missing-worker'], ['td-process', 'grinding'], ['td-mode', '9'], ['td-quality', 'complete']]) {
    await t.test(id, () => {
      const control = app.get(id), selected = control.options.find(entry => entry.value === expected);
      assert.equal(control.value, expected);
      assert.ok(selected, 'selected filter must not be silently discarded');
      assert.equal(selected.disabled, true, 'unavailable selection must not appear as a normal available option');
      assert.match(selected.label, /0\s*건/, 'zero-result selection must be explained in its label');
    });
  }
});

test('direct URL filters and export links preserve valid selections and remove only retired filter keys', async () => {
  const app = setup('view=training&worker_id=w1&process=spray&scenario_id=1&quality_status=complete&q=%EB%85%B8%EC%A6%90&page=4&limit=50&acquired_at=2025-01-01&zone_mapping=present');
  const first = new URL(app.requests[0].url, app.location());
  assert.equal(first.searchParams.get('worker_id'), 'w1'); assert.equal(first.searchParams.get('quality_status'), 'complete');
  assert.equal(first.searchParams.get('q'), '노즐'); assert.equal(first.searchParams.get('page'), '4'); assert.equal(first.searchParams.get('limit'), '50');
  await app.respond(0, listing(universe, 150, {page: 4, pages: 4}));
  assert.equal(app.get('td-worker').value, 'w1'); assert.equal(app.get('td-quality').value, 'complete');
  assert.equal(app.get('td-search').value, '노즐');
  assert.equal(app.location().searchParams.get('view'), 'training');
  assert.equal(app.location().searchParams.has('acquired_at'), false); assert.equal(app.location().searchParams.has('zone_mapping'), false);
  for (const id of ['td-export-csv', 'td-export-zip']) {
    const url = new URL(app.get(id).href, app.location());
    assert.equal(url.searchParams.get('worker_id'), 'w1'); assert.equal(url.searchParams.get('process'), 'spray');
    assert.equal(url.searchParams.get('scenario_id'), '1'); assert.equal(url.searchParams.get('quality_status'), 'complete');
    assert.equal(url.searchParams.get('q'), '노즐'); assert.equal(url.searchParams.has('page'), false); assert.equal(url.searchParams.has('limit'), false);
  }
});

test('reset clears all four facets and search, restarts pagination, and restores complete options', async () => {
  const app = setup('worker_id=w1&process=spray&scenario_id=1&quality_status=complete&q=old&page=4&limit=50');
  await app.respond(0, listing(universe, 150, {page: 4, pages: 4}));
  await app.get('td-reset').emit('click');
  const request = new URL(app.requests[1].url, app.location());
  for (const key of ['worker_id', 'process', 'scenario_id', 'quality_status']) assert.equal(request.searchParams.get(key), 'all');
  assert.equal(request.searchParams.get('q'), ''); assert.equal(request.searchParams.get('page'), '1'); assert.equal(request.searchParams.get('limit'), '50');
  await app.respond(1, listing(universe));
  for (const id of ['td-worker', 'td-process', 'td-mode', 'td-quality']) assert.equal(app.get(id).value, 'all');
  assert.equal(app.get('td-search').value, '');
  assert.deepEqual(app.values('td-worker'), ['all', 'w1', 'w2']); assert.deepEqual(app.values('td-quality'), ['all', 'needs_review', 'complete']);
  assert.equal(app.location().searchParams.has('worker_id'), false); assert.equal(app.location().searchParams.has('q'), false);
});

test('all four facet controls stay disabled until the latest response, even when an older request finishes first', async () => {
  const app = setup();
  const ids = ['td-worker', 'td-process', 'td-mode', 'td-quality'];
  for (const id of ids) assert.equal(app.get(id).disabled, true, `${id} must not expose stale choices during initial load`);
  assert.match(app.get('td-table-status').textContent, /불러오/);
  await app.respond(0, listing());
  for (const id of ids) assert.equal(app.get(id).disabled, false);
  await app.change('td-worker', 'w1');
  for (const id of ids) assert.equal(app.get(id).disabled, true);
  await app.get('td-refresh').emit('click');
  assert.equal(app.requests[1].options.signal.aborted, true);
  await app.respond(1, listing(universe, 111));
  for (const id of ids) assert.equal(app.get(id).disabled, true, 'stale finally must not re-enable dependent choices');
  assert.equal(app.get('td-list-panel').getAttribute('aria-busy'), 'true');
  await app.respond(2, listing(universe, 2));
  for (const id of ids) assert.equal(app.get(id).disabled, false);
  assert.equal(app.get('td-list-panel').getAttribute('aria-busy'), 'false');
  assert.equal(app.get('td-worker').value, 'w1');
});

test('a latest request error releases all four controls and retains filter values for retry', async () => {
  const app = setup(); await app.respond(0, listing());
  await app.change('td-quality', 'needs_review');
  for (const id of ['td-worker', 'td-process', 'td-mode', 'td-quality']) assert.equal(app.get(id).disabled, true);
  await app.reject(1, new Error('temporary network error'));
  for (const id of ['td-worker', 'td-process', 'td-mode', 'td-quality']) assert.equal(app.get(id).disabled, false);
  assert.equal(app.get('td-quality').value, 'needs_review');
  assert.equal(app.get('td-list-panel').getAttribute('aria-busy'), 'false');
  assert.match(app.get('td-table-status').textContent, /실패/);
  await app.get('td-refresh').emit('click');
  assert.equal(new URL(app.requests[2].url, app.location()).searchParams.get('quality_status'), 'needs_review');
  await app.respond(2, listing());
  assert.equal(app.get('td-quality').value, 'needs_review');
});

test('a latest failure clears stale normal choices, labels preserved selections as unknown, and restores options after retry', async () => {
  const app = setup(); await app.respond(0, listing());
  await app.change('td-quality', 'needs_review');
  await app.reject(1, new Error('temporary network error'));
  for (const id of ['td-worker', 'td-process', 'td-mode']) {
    assert.deepEqual(app.values(id), ['all'], `${id} must not offer unverified choices from the previous context`);
    assert.equal(app.get(id).disabled, false, 'the all option must remain usable after failure');
  }
  const quality = app.get('td-quality');
  assert.deepEqual(app.values('td-quality'), ['all', 'needs_review']);
  assert.equal(quality.value, 'needs_review'); assert.equal(quality.disabled, false);
  const selected = quality.options.find(entry => entry.value === 'needs_review');
  assert.equal(selected.disabled, true);
  assert.match(selected.label, /선택지.*조회.*실패/, 'failed facet retrieval must be distinguished from known zero results');
  assert.doesNotMatch(selected.label, /0\s*건/, 'a network error is not evidence of zero matching records');
  await app.get('td-refresh').emit('click');
  assert.equal(new URL(app.requests[2].url, app.location()).searchParams.get('quality_status'), 'needs_review');
  await app.respond(2, listing({workers: [universe.workers[1]], processes: [universe.processes[1]], modes: [universe.modes[1]], qualities: universe.qualities}, 1));
  assert.deepEqual(app.values('td-worker'), ['all', 'w2']);
  assert.deepEqual(app.values('td-process'), ['all', 'blasting']); assert.deepEqual(app.values('td-mode'), ['all', '2']);
  const recovered = quality.options.find(entry => entry.value === 'needs_review');
  assert.equal(quality.value, 'needs_review'); assert.equal(recovered.disabled, false);
  assert.equal(recovered.label, '확인 필요');
});

test('a search submitted during initial loading preserves all four direct URL filters while updating only search and page', async () => {
  const app = setup('worker_id=w1&process=spray&scenario_id=1&quality_status=complete&q=old&page=3&limit=50');
  app.get('td-search').value = ' new question ';
  await app.get('td-filter-form').emit('submit');
  const pendingSearch = new URL(app.requests[1].url, app.location());
  for (const [key, value] of [['worker_id', 'w1'], ['process', 'spray'], ['scenario_id', '1'], ['quality_status', 'complete']]) {
    assert.equal(pendingSearch.searchParams.get(key), value, `pending search must preserve ${key} from URL state`);
  }
  assert.equal(pendingSearch.searchParams.get('q'), 'new question');
  assert.equal(pendingSearch.searchParams.get('page'), '1'); assert.equal(pendingSearch.searchParams.get('limit'), '50');
  assert.equal(app.requests[0].options.signal.aborted, true);
  await app.respond(0, listing(universe, 111));
  assert.equal(app.get('td-list-panel').getAttribute('aria-busy'), 'true');
  await app.respond(1, listing(universe, 2));
  for (const [id, value] of [['td-worker', 'w1'], ['td-process', 'spray'], ['td-mode', '1'], ['td-quality', 'complete']]) assert.equal(app.get(id).value, value);
  assert.equal(app.get('td-search').value.trim(), 'new question');
  assert.equal(app.location().searchParams.get('q'), 'new question');
});

test('a late aborted success cannot overwrite facets or clear the newer request loading state', async () => {
  const app = setup(); await app.respond(0, listing());
  await app.change('td-worker', 'w1');
  // Reset remains usable while the dependent selects are disabled.
  await app.get('td-reset').emit('click');
  assert.equal(app.requests[1].options.signal.aborted, true);
  await app.respond(1, listing({workers: [universe.workers[1]], processes: [universe.processes[1]], modes: [universe.modes[1]], qualities: [universe.qualities[0]]}, 111));
  assert.equal(app.get('td-list-panel').getAttribute('aria-busy'), 'true');
  assert.doesNotMatch(app.get('td-table-status').textContent, /111/);
  await app.respond(2, listing({workers: [universe.workers[0]], processes: [universe.processes[0]], modes: [universe.modes[0]], qualities: universe.qualities}, 7));
  assert.equal(app.get('td-list-panel').getAttribute('aria-busy'), 'false');
  assert.equal(app.get('td-worker').value, 'all'); assert.equal(app.get('td-process').value, 'all');
  assert.deepEqual(app.values('td-mode'), ['all', '1']); assert.match(app.get('td-table-status').textContent, /7건/);
});

test('an aborted error arriving after the latest response cannot replace successful results', async () => {
  const app = setup(); await app.respond(0, listing());
  await app.change('td-worker', 'w1'); await app.get('td-refresh').emit('click');
  await app.respond(2, listing(universe, 9));
  const successfulStatus = app.get('td-table-status').textContent;
  await app.reject(1);
  assert.equal(app.get('td-table-status').textContent, successfulStatus);
  assert.equal(app.get('td-list-panel').getAttribute('aria-busy'), 'false');
  assert.equal(app.get('td-worker').value, 'w1'); assert.equal(app.get('td-process').value, 'all');
});
