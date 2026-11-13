'use strict';
// 왼쪽 위 상태 막대(#realtimeHud)의 칸 폭이 재생 중 값 길이에 따라 흔들리지 않는지 본다.
// 실행(호스트): node --test trainer/demo_api/tests/test_motion_viewer_hud.js
// 칸마다 가장 긴 값을 data-max 로 적어 두고, CSS 가 그 글을 보이지 않는 높이 0 줄로 깔아 칸 폭의 하한으로 쓴다.
// 같은 글꼴로 재므로 글꼴이 늦게 실려도 맞다. 실제 폭은 브라우저 측정(작업 기록)으로 따로 확인했다.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const root = path.join(__dirname, '..');
const v2Html = fs.readFileSync(path.join(root, 'static/motion_viewer_v2.html'), 'utf8');
const v2Css = fs.readFileSync(path.join(root, 'static/motion_viewer_v2.css'), 'utf8');
const legacy = (() => {
  const source = fs.readFileSync(path.join(root, 'motion_viewer.py'), 'utf8');
  const start = source.indexOf('_VIEWER_HTML = r"""');
  return source.slice(start, source.indexOf('"""', start + 20));
})();

/** 문서에서 id 요소의 data-max 값. */
function maxOf(html, id) {
  const tag = html.match(new RegExp(`<[^>]*\\bid="${id}"[^>]*>`));
  assert.ok(tag, `${id} 요소가 없다`);
  const found = tag[0].match(/data-max="([^"]*)"/);
  assert.ok(found, `${id} 에 data-max 가 없다`);
  return found[1];
}

const VALUES = ['realtimeHudZone', 'realtimeHudActivity', 'realtimeHudSpeed', 'realtimeHudDistance', 'realtimeHudAxis', 'realtimeHudDirection', 'realtimeHudCoverage'];
// pose.js 가 쓰는 값의 가장 긴 꼴. 거리는 쉼표 없이 반올림, 방향 낱말 중 가장 긴 것은 '분석 중'.
const LONGEST = {
  realtimeHudZone: 'Zone 대기', realtimeHudActivity: '이동 중', realtimeHudDistance: '9999 mm',
  realtimeHudAxis: '99.9°', realtimeHudDirection: '분석 중', realtimeHudCoverage: '100%',
};

// v2 는 이동 문구를 'Zone 사이 이동'으로 바꿔 쓴다(motion_viewer_v2.js, 2026-11-13 재디자인).
const V2_LONGEST = { realtimeHudActivity: 'Zone 사이 이동' };

test('두 화면의 상태 막대 값 칸마다 가장 긴 값이 적혀 있다', () => {
  for (const [name, html] of [['v2', v2Html], ['기존', legacy]]) {
    for (const id of VALUES) if (LONGEST[id]) assert.equal(maxOf(html, id), name === 'v2' && V2_LONGEST[id] || LONGEST[id], `${name} ${id}`);
  }
  // 속도: v2 는 mm/s(천 단위 쉼표, 실제 기록에서 26,930 까지 나와 5자리), 기존 화면은 m/s 소수 둘째 자리
  assert.equal(maxOf(v2Html, 'realtimeHudSpeed'), '99,999 mm/s');
  assert.equal(maxOf(legacy, 'realtimeHudSpeed'), '9.99 m/s');
  // 기존 화면의 판정 표지는 글자가 있어 가장 긴 판정 글로 맞춘다(v2 는 8px 점이라 이미 고정)
  for (const id of ['Speed', 'Distance', 'Axis', 'Direction', 'Coverage']) assert.equal(maxOf(legacy, `realtimeHud${id}State`), '진행 중');
});

test('data-max 글을 보이지 않는 높이 0 줄로 깔아 칸 폭을 고정하고 숫자는 오른쪽 정렬한다', () => {
  for (const [name, css] of [['v2', v2Css], ['기존', legacy]]) {
    const compact = css.replace(/\s+/g, '');
    assert.ok(/\[data-max\]::after\{content:attr\(data-max\);height:0;overflow:hidden;visibility:hidden;?\}/.test(compact), `${name}: 높이 0 줄 규칙이 없다`);
    assert.ok(/\[data-max\]\{display:inline-flex;flex-direction:column;align-items:flex-end;?\}/.test(compact), `${name}: 오른쪽 정렬 칸 규칙이 없다`);
  }
  assert.ok(/\.hud[^{]*\{[^}]*font-variant-numeric:\s*tabular-nums/.test(v2Css), 'v2 상태 막대 숫자 폭 고정');
  assert.ok(/\.realtime-hud-value\{[^}]*font-variant-numeric:tabular-nums/.test(legacy), '기존 화면 숫자 폭 고정');
});

test('Zone 칸의 색 네모는 글자와 세로 가운데로 맞춘다(격자 칸이 flex 의 flex-end 를 물려받지 않는다)', () => {
  const css = fs.readFileSync(path.join(root, 'static/motion_viewer_v2.css'), 'utf8');
  const rule = css.match(/\.hud-zone b\[data-max\]\s*\{([^}]*)\}/);
  assert.ok(rule, 'Zone 칸 규칙이 있어야 한다');
  assert.match(rule[1], /display:\s*inline-grid/);
  assert.match(rule[1], /align-items:\s*center/);
});
