// The cache retains real payloads in this page; a visit history is not a cache.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { create } = require('../static/motion_viewer_cache.js');

(async () => {
  let clock = 0, reads = 0, valid = true;
  const fetchJson = async url => {
    if (url.startsWith('/motion-viewer/api/cache-status?')) return { valid };
    reads++;
    return { frames: [{ zone_id: String(reads) }], viewer_cache: { token: 'token-' + reads } };
  };
  const cache = create({ fetchJson, now: () => clock, ttlMs: 100, maxEntries: 2 });
  assert.equal(cache.has('/a'), false);
  const first = await cache.load('/a');
  assert.equal(cache.has('/a'), true);
  first.frames[0].zone_id = 'mutation';
  assert.equal((await cache.load('/a')).frames[0].zone_id, '1');
  assert.equal(reads, 1, 'validated reuse must not reload the expensive payload');
  valid = false;
  assert.equal((await cache.load('/a')).frames[0].zone_id, '2');
  assert.equal(reads, 2, 'changed sources must reload');
  valid = true;
  await cache.load('/b'); await cache.load('/c');
  assert.equal(cache.has('/a'), false, 'evicted payloads have no fast-loading badge');
  clock = 101;
  assert.equal(cache.has('/c'), false, 'expired payloads have no fast-loading badge');
  await cache.load('/c');
  assert.equal(reads, 5);

  let complete, concurrentReads = 0;
  const concurrent = create({ fetchJson: async () => {
    concurrentReads++;
    return new Promise(resolve => { complete = resolve; });
  } });
  const a = concurrent.load('/same'), b = concurrent.load('/same');
  await Promise.resolve();
  complete({ frames: [1], viewer_cache: { token: 'same' } });
  const [left, right] = await Promise.all([a, b]);
  assert.equal(concurrentReads, 1);
  left.frames.push(2);
  assert.deepEqual(right.frames, [1], 'concurrent consumers must receive independent payloads');

  let fail = true, failures = 0;
  const retry = create({ fetchJson: async () => { failures++; if (fail) throw Error('offline'); return { frames: [] }; } });
  await assert.rejects(retry.load('/failed'), /offline/);
  assert.equal(retry.has('/failed'), false);
  fail = false;
  await retry.load('/failed');
  assert.equal(failures, 2, 'a rejected in-flight request must not poison retries');
  assert.equal(retry.has('/failed'), false, 'a payload without a validation token is not reusable');

  const limited = create({ fetchJson, maxBytes: 1 });
  await limited.load('/large');
  assert.equal(limited.has('/large'), false);

  // Exercise the labels actually installed in both viewer shells.
  const source = fs.readFileSync(path.join(__dirname, '../motion_viewer.py'), 'utf8');
  const functions = ['function motionUrl(', 'function refreshSessionLabels('].map(prefix =>
    source.split('\n').find(line => line.startsWith(prefix))).join('\n');
  const controls = { worker: { value: 'W' }, process: { value: 'spray' }, mode: { value: '4' },
    session: { value: 'a', options: [{ value: 'a', style: {}, dataset: {} }, { value: 'b', style: {}, dataset: {} }] } };
  let retained = true;
  const labels = vm.createContext({ URLSearchParams, catalog: {}, $: id => controls[id],
    currentWorker: () => ({ display_name: '작업자' }), currentMode: () => ({}),
    payloadCache: { has: url => retained && url.endsWith('session_id=a') } });
  vm.runInContext(functions + '\nrefreshSessionLabels();', labels);
  assert.match(controls.session.options[0].textContent, /빠른 로딩/);
  assert.doesNotMatch(controls.session.options[0].textContent, /이 창/);
  assert.match(controls.session.options[0].textContent, /^✓ /);
  assert.equal(controls.session.options[0].style.color, '#34d399');
  assert.match(controls.session.options[1].textContent, /^↓ /);
  assert.doesNotMatch(controls.session.options[1].textContent, /빠른 로딩/);
  retained = false;
  vm.runInContext('refreshSessionLabels();', labels);
  assert.doesNotMatch(controls.session.options[0].textContent, /빠른 로딩/);
  assert.match(controls.session.options[0].textContent, /^↓ /);
  console.log('motion viewer payload cache: passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
