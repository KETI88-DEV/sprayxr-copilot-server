const assert = require('node:assert/strict');
const { create } = require('../static/motion_viewer_cache.js');
const fs = require('node:fs'), path = require('node:path'), vm = require('node:vm');
const tick = () => new Promise(resolve => setImmediate(resolve));

(async () => {
  const updates = [], requests = [];
  let finish, mainReads = 0, polls = 0;
  const cache = create({ fetchJson: async (url, options = {}) => {
    if (url.includes('/cache-status?')) return { valid: true };
    if (url.includes('/load-progress?')) { polls++; return { status: 'running', stage: 'csv_frames', completed: 2, total: 5, unit: 'rows' }; }
    mainReads++; requests.push(options);
    return new Promise(resolve => { finish = resolve; });
  } });
  const first = cache.load('/motion', { onProgress: value => updates.push(value) });
  await tick();
  assert.match(requests[0].headers['X-Motion-Load-ID'], /^[a-f0-9-]{32,36}$/);
  assert.ok(updates.some(value => value.stage === 'csv_frames' && value.completed === 2 && value.total === 5));
  finish({ frames: [], viewer_cache: { token: 'valid' } }); await first;
  const before = polls, restored = [];
  await cache.load('/motion', { onProgress: value => restored.push(value) });
  assert.equal(mainReads, 1);
  assert.equal(polls, before, 'actual window cache reuse must not create or poll a new server load job');
  assert.ok(restored.some(value => value.stage === 'cache_restore'));

  let complete, sharedSignal, aborted = false;
  const shared = create({ fetchJson: async (url, options) => {
    sharedSignal = options.signal;
    return new Promise((resolve, reject) => {
      complete = resolve;
      options.signal.addEventListener('abort', () => { aborted = true; reject(new DOMException('cancelled', 'AbortError')); });
    });
  } });
  const left = new AbortController(), right = new AbortController();
  const a = shared.load('/same', { signal: left.signal });
  const b = shared.load('/same', { signal: right.signal });
  await tick(); left.abort();
  await assert.rejects(a, { name: 'AbortError' });
  assert.equal(sharedSignal.aborted, false, 'one consumer leaving must not cancel another consumer');
  complete({ frames: [1] }); assert.deepEqual((await b).frames, [1]);
  const cancel = new AbortController(), abandoned = shared.load('/new', { signal: cancel.signal });
  await tick(); cancel.abort();
  await assert.rejects(abandoned, { name: 'AbortError' });
  assert.equal(aborted, true, 'the final consumer leaving aborts the real HTTP request');

  const source = fs.readFileSync(path.join(__dirname, '../motion_viewer.py'), 'utf8');
  const pick = prefix => source.split('\n').find(line => line.startsWith(prefix));
  const controls = Object.fromEntries(['loadingStage', 'loadingUnits', 'loadingProgress', 'status', 'play', 'timeline',
    'sessionCount', 'duration', 'overlayTitle'].map(key => [key, { textContent: '', dataset: {},
      setAttribute(name, value) { this[name] = value; }, removeAttribute(name) { delete this[name]; } }]));
  const formatting = vm.createContext({ $: key => controls[key] });
  vm.runInContext(pick('function setLoadProgress('), formatting);
  vm.runInContext("setLoadProgress({stage_label:'CSV 처리',completed:2,total:5,unit:'rows'})", formatting);
  assert.equal(controls.loadingProgress.value, 2); assert.equal(controls.loadingProgress.max, 5);
  assert.match(controls.loadingUnits.textContent, /2 \/ 5 행/);
  vm.runInContext("setLoadProgress({stage_label:'CSV 행 수 확인',completed:3,total:null,unit:'rows'})", formatting);
  assert.equal(controls.loadingProgress.value, undefined);
  assert.match(controls.loadingUnits.textContent, /총량 확인 중/);
  assert.doesNotMatch(controls.loadingUnits.textContent, /%|초/);
  vm.runInContext("setLoadProgress({stage_label:'빈 기록 확인',completed:0,total:0,unit:'rows',phase_complete:true})", formatting);
  assert.match(controls.loadingUnits.textContent, /0 \/ 0 행/);

  const waiting = [], rendered = [], labels = [];
  const context = vm.createContext({ $: key => controls[key], AbortController, DOMException,
    loadGeneration: 0, activeLoadController: null, data: null, motionUrl: () => '/selection',
    payloadCache: { load: (url, options) => new Promise(resolve => waiting.push({ resolve, options })) },
    setLoading() {}, setLoadProgress: value => labels.push(value.stage_label), resetViewToCamera() {}, renderZoneOrder() {},
    dispatchEvent() {}, CustomEvent: class {}, draw() { rendered.push(context.data?.selected_session_id); },
    requestAnimationFrame: callback => callback() });
  vm.runInContext(pick('async function loadMotion('), context);
  const previous = vm.runInContext('loadMotion()', context);
  const current = vm.runInContext('loadMotion()', context);
  assert.equal(waiting[0].options.signal.aborted, true);
  waiting[0].options.onProgress({ stage_label: 'late private old job' });
  assert.notEqual(labels.at(-1), 'late private old job', 'late progress must not relabel a newer selection');
  const payload = id => ({ selected_session_id: id, frames: [], duration_seconds: 1, sampled_session_count: 1,
    worker: { display_name: '작업자' }, process_label: '도장', mode_id: 4 });
  waiting[1].resolve(payload('B')); await current;
  waiting[0].resolve(payload('A')); await previous;
  assert.equal(context.data.selected_session_id, 'B');
  assert.deepEqual(rendered, ['B']);
  assert.equal(controls.status.dataset.loadState, 'ready');
  console.log('motion viewer measured progress and cancellation: passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
