'use strict';
// 점검 배너 표시·숨김. 실행(호스트): node --test trainer/demo_api/tests/test_maintenance_banner.js
const test = require('node:test');
const assert = require('node:assert/strict');
const { bannerText, mount } = require('../static/maintenance_banner.js');

const NOW = new Date('2026-11-17T00:00:00Z'); // 한국 09:00
const ACTIVE = { active: true, what: 'vllm', reason: 'vLLM 교체', impact: '상세 분석(27B) 일시 중단', until: '2026-11-17T01:30:00Z' };

test('점검 중이면 영향과 한국 시각 기한을, 아니면 빈 문자열을 준다', () => {
  assert.equal(bannerText(ACTIVE, NOW), '점검 중 · 상세 분석(27B) 일시 중단 · 10:30까지(한국 시각)');
  assert.equal(bannerText({ ...ACTIVE, until: '2026-11-17T16:00:00Z' }, NOW), '점검 중 · 상세 분석(27B) 일시 중단 · 11월 18일 01:00까지(한국 시각)');
  assert.equal(bannerText({ active: false }, NOW), '');
  assert.equal(bannerText({ ...ACTIVE, until: '2026-11-16T23:59:00Z' }, NOW), '');
  assert.equal(bannerText({ ...ACTIVE, until: 'nope' }, NOW), '');
  assert.equal(bannerText(null, NOW), '');
});

function fakePage() {
  const label = { textContent: '' };
  const banner = { hidden: true, offsetHeight: 34, querySelector: () => label };
  const listeners = {};
  const vars = {};
  const doc = {
    visibilityState: 'visible',
    getElementById: (id) => (id === 'copilot-maintenance' ? banner : null),
    addEventListener: (name, fn) => { listeners[name] = fn; },
    documentElement: { style: { setProperty: (k, v) => { vars[k] = v; } } },
  };
  return { doc, banner, label, listeners, vars };
}

test('응답에 따라 배너를 켜고 끄며, 요청 실패는 배너를 건드리지 않는다', async () => {
  const page = fakePage();
  const future = new Date(Date.now() + 3600e3).toISOString();
  const replies = [{ ...ACTIVE, until: future }];
  let tick = null;
  const fetchJson = () => (replies.length ? Promise.resolve(replies.shift()) : Promise.reject(new Error('down')));
  await mount(page.doc, fetchJson, { setIntervalFn: (fn, ms) => { tick = fn; assert.equal(ms, 60000); } });
  assert.equal(page.banner.hidden, false);
  assert.match(page.label.textContent, /^점검 중 · 상세 분석\(27B\) 일시 중단 · /);
  assert.equal(page.vars['--copilot-banner-height'], '34px');

  await tick(); // 실패: 그대로
  assert.equal(page.banner.hidden, false);

  replies.push({ active: false });
  page.doc.visibilityState = 'hidden';
  page.listeners.visibilitychange(); // 안 보일 때는 묻지 않는다
  assert.equal(replies.length, 1);
  page.doc.visibilityState = 'visible';
  await page.listeners.visibilitychange();
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(page.banner.hidden, true);
  assert.equal(page.vars['--copilot-banner-height'], '0px');
});

test('배너 자리가 없는 화면에서는 아무것도 하지 않는다', () => {
  const doc = { getElementById: () => null };
  assert.equal(mount(doc, () => assert.fail('요청하면 안 된다')), null);
});
