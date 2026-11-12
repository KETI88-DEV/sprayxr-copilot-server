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

test('알림 이동은 분사 구간 안의 첫 프레임을, 표본이 구간을 건너뛰었으면 seek 이하의 마지막 프레임을 고른다', () => {
  const frames = [{ t: 0 }, { t: 0.6 }, { t: 1.2 }, { t: 1.8 }, { t: 2.4 }];
  assert.equal(alerts.seekIndex(frames, 1.1, 1.5), 2);   // [1.1, 1.5) 안의 1.2
  assert.equal(alerts.seekIndex(frames, 1.3, 1.5), 2);   // 구간에 프레임이 없다 -> 1.2 (at 보다 앞)
  assert.ok(frames[alerts.seekIndex(frames, 1.3, 1.5)].t < 1.5);
  assert.equal(alerts.seekIndex(frames, -1, 0), 0);
});

test('알림 Zone 을 칠하는 프레임이 구간 안에 있으면 그 첫 프레임으로 옮긴다(이동 중·다른 Zone 프레임 건너뜀)', () => {
  // 카나리 V1(김케티 4_9990001_1): SEQ-015 는 이동 중 → Zone 4 → Zone 5 순서라 구간 첫 프레임은 Zone 5 가 아니다.
  const frames = [
    { t: 34.29, spraying: false, zone_id: 6, map_num: 1 },
    { t: 34.80, spraying: false, zone_id: -1, map_num: null },
    { t: 35.06, spraying: true, zone_id: 4, map_num: null },
    { t: 35.32, spraying: true, zone_id: 4, map_num: 1 },
    { t: 38.92, spraying: false, zone_id: 5, map_num: 1 },
    { t: 39.18, spraying: true, zone_id: 5, map_num: 2 },
    { t: 39.44, spraying: true, zone_id: 5, map_num: 1 },
    { t: 43.50, spraying: true, zone_id: 5, map_num: 1 },
  ];
  const alert = { seek_seconds: 34.857, at_seconds: 43.413, zone_id: '5', map_num: '1' };
  assert.equal(alerts.seekIndex(frames, alert.seek_seconds, alert.at_seconds, alert), 6);
  assert.equal(alerts.seekIndex(frames, alert.seek_seconds, alert.at_seconds, { zone_id: '5', map_num: null }), 5);   // 지도 번호를 모르면 Zone 만 본다
  // 구간 안에 그 Zone 분사가 없으면 예전처럼 구간 첫 프레임(at 보다 앞)
  assert.equal(alerts.seekIndex(frames, alert.seek_seconds, alert.at_seconds, { zone_id: '9', map_num: '1' }), 2);
  assert.equal(alerts.seekIndex(frames, alert.seek_seconds, alert.at_seconds), 2);
});
