'use strict';
// v2 재생 알림의 순수 함수 시험. 실행(호스트): node --test trainer/demo_api/tests/test_motion_viewer_alerts.js
const test = require('node:test');
const assert = require('node:assert/strict');
const alerts = require('../static/motion_viewer_alerts.js');

const item = (at, anomaly_type = 'direction_deviation', extra = {}) => ({ id: `${at}:${anomaly_type}`, anomaly_type, at_seconds: at, seek_seconds: at - 1, zone_id: '6', ...extra });

// 실제 재생 루프처럼 프레임마다 isJump → crossed 로 고른다.
function play(list, steps, speed = 1) {
  const shown = [];
  let last = steps[0].t;
  for (const step of steps.slice(1)) {
    const jumped = alerts.isJump(last, step.t, speed, step.seeking);
    shown.push(...alerts.crossed(list, last, step.t, jumped).map((a) => a.id));
    last = step.t;
  }
  return shown;
}
const at = (...times) => times.map((t) => (typeof t === 'number' ? { t } : t));

test('1배속으로 지나가면 그 시점의 알림을 한 번만 고른다', () => {
  const list = [item(41.25)];
  assert.deepEqual(play(list, at(41.0, 41.1, 41.2, 41.3, 41.3, 41.4)), ['41.25:direction_deviation']);
});

test('3배속 한 tick 은 재생이고, 그보다 크게 건너뛰면 탐색이다', () => {
  assert.equal(alerts.isJump(10, 10.3, 3, false), false);
  assert.equal(alerts.isJump(10, 10.5, 1, false), true);
  assert.equal(alerts.isJump(10, 10.1, 1, true), true);
  assert.equal(alerts.isJump(10, 9.9, 1, false), true);
});

test('앞으로 건너뛰면 사이의 알림을 띄우지 않고, 뒤로 감은 뒤 다시 지나가면 다시 뜬다', () => {
  const list = [item(5)];
  assert.deepEqual(play(list, at(1, 8)), []);
  assert.deepEqual(play(list, at(8, 4.95, 5.0, 5.05)), ['5:direction_deviation']);
});

test('끝에서 처음으로 돌아가는 반복은 알림을 띄우지 않는다', () => {
  assert.deepEqual(play([item(0.05)], at(141.4, 0.08)), []);
});

test('seeking 표시가 선 프레임은 알림 없이 맞춘다', () => {
  assert.deepEqual(play([item(5)], at(4.95, { t: 5.02, seeking: true }, 5.1)), []);
});

test('같은 시각의 두 알림은 둘 다 고른다', () => {
  const list = [item(3, 'coating_under_app'), item(3, 'direction_deviation')];
  assert.deepEqual(alerts.crossed(list, 2.95, 3.0, false).map((a) => a.anomaly_type), ['coating_under_app', 'direction_deviation']);
});

test('merge 는 도막과 시퀀스 알림을 시각순으로 합치고, 시퀀스 판정 전에는 도막만 쓴다', () => {
  const coat = [item(9, 'coating_under_app'), item(2, 'coating_over_app')];
  assert.deepEqual(alerts.merge({ coating_alerts: coat }).map((a) => a.at_seconds), [2, 9]);
  const record = { coating_alerts: coat, sequence_alerts: { status: 'available', alerts: [item(5)] } };
  assert.deepEqual(alerts.merge(record).map((a) => a.at_seconds), [2, 5, 9]);
  assert.deepEqual(alerts.merge(null), []);
});

test('short 는 서버 값을 옮길 뿐 문턱으로 판정하지 않는다', () => {
  const under = (thickness) => alerts.short(item(1, 'coating_under_app', { values: { thickness_um: thickness } }));
  assert.equal(under(49.4), '저도막 49 µm');
  assert.equal(under(900), '저도막 900 µm');   // 두께가 커도 낱말은 anomaly_type 에서만 정해진다
  assert.equal(alerts.short(item(1, 'coating_over_app', { values: { thickness_um: 250.3 } })), '과도막 250 µm');
  assert.equal(alerts.short(item(1, 'coating_over_app', { values: {} })), '과도막');
  assert.equal(alerts.short(item(1, 'direction_deviation')), '방향');
  assert.equal(alerts.short(item(1, 'sequence_order_deviation', { reason_code: 'zone_revisit' })), '다시 칠함');
  assert.equal(alerts.short(item(1, 'sequence_order_deviation', { reason_code: 'zone_skipped' })), '건너뜀');
  assert.equal(alerts.short(item(1, 'sequence_order_deviation', { reason_code: 'zone_out_of_order' })), '순서 바뀜');
  assert.equal(alerts.short(item(1, 'sequence_order_deviation', { reason_code: 'unexpected_zone' })), '기준 밖');
});

test('종류마다 아이콘과 의미 색이 하나씩 있다', () => {
  for (const type of ['sequence_order_deviation', 'direction_deviation', 'coating_over_app', 'coating_under_app']) {
    assert.ok(alerts.KIND[type].icon && alerts.KIND[type].tone, type);
  }
});
