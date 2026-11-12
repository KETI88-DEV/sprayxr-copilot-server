// v2 재생 알림의 순수 함수. 알림은 서버가 공개 이상 알림과 같은 함수로 판정해 준 것이다(/motion-viewer/api/alerts,
// 모션 응답의 coating_alerts). 여기서는 재생 시각에 맞춰 고르고 짧게 옮겨 적기만 한다. 판정 규칙을 두지 않는다.
(() => {
  'use strict';
  // 종류별 Lucide 아이콘과 의미 색 토큰(motion_viewer_v2.css). Zone 색과 섞지 않는다.
  const KIND = {
    sequence_order_deviation: { icon: 'route-off', tone: 'warn', label: '순서 이탈' },
    direction_deviation: { icon: 'arrow-right-left', tone: 'turn', label: '방향 이탈' },
    coating_over_app: { icon: 'arrow-up-to-line', tone: 'thick', label: '과도막' },
    coating_under_app: { icon: 'arrow-down-to-line', tone: 'thin', label: '저도막' },
  };
  // 방향 열쇠의 한국어 표시 이름. v2 화면 전체가 이 표 하나를 쓴다.
  const DIRECTION = { horizontal: '수평', vertical: '수직', diagonal: '대각', curved_irregular: '곡선·불규칙' };
  const ORDER_WORD = { zone_revisit: '다시 칠함', zone_skipped: '건너뜀', zone_out_of_order: '순서 바뀜', unexpected_zone: '기준 밖' };

  /** 도막 알림(모션 응답)과 순서·방향 알림(늦게 도착)을 시각순으로. 순서 판정 전이면 도막만. */
  const merge = (record) => [...(record?.coating_alerts || []), ...(record?.sequence_alerts?.alerts || [])]
    .sort((a, b) => a.at_seconds - b.at_seconds);

  /** 재생 tick 은 한 번에 최대 0.1초×배속만 나아간다. 그보다 크게 움직였거나 뒤로 갔으면 탐색이다. */
  const isJump = (last, now, speed, seeking) => Boolean(seeking) || now < last || now - last > 0.1 * speed + 0.05;

  /** 재생으로 지나간 구간 (last, now] 의 알림. 탐색이면 없다. */
  const crossed = (alerts, last, now, jumped) => (jumped || !(last < now) ? []
    : alerts.filter((alert) => alert.at_seconds > last && alert.at_seconds <= now));

  /** 화면의 짧은 표기. 낱말은 anomaly_type·reason_code 에서, 숫자는 서버 values 에서 옮긴다. */
  const short = (alert) => {
    if (alert.anomaly_type === 'sequence_order_deviation') return ORDER_WORD[alert.reason_code] || '순서';
    if (alert.anomaly_type === 'direction_deviation') return '방향';
    const word = KIND[alert.anomaly_type]?.label || '알림';
    const thickness = Number(alert.values?.thickness_um);
    return alert.values?.thickness_um != null && Number.isFinite(thickness) ? `${word} ${Math.round(thickness)} µm` : word;
  };

  /** 목록·토스트·표지의 title·aria 문장. 방향 알림의 서버 문장은 영어 열쇠가 섞여 있어(공개 SSE 문구는 사용자 결정 대기)
   *  쓰지 않고, 서버 values 의 기대·관측 방향을 한국어로 옮긴다. 없는 부분은 빼서 빈 꼬리를 남기지 않는다. */
  const describe = (alert) => {
    const head = `${KIND[alert.anomaly_type]?.label || '알림'} · Zone ${Number(alert.zone_id)} · ${Number(alert.at_seconds).toFixed(1)}초`;
    if (alert.anomaly_type !== 'direction_deviation') return alert.message ? `${head}. ${alert.message}` : head;
    const expected = DIRECTION[alert.values?.expected_direction], observed = DIRECTION[alert.values?.observed_direction];
    return [head, expected && `권장 방향 ${expected}`, observed && `실제 ${observed}`].filter(Boolean).join(' · ');
  };

  /** 알림으로 옮길 프레임. 분사 구간 [seek, at) 안에서 서버가 준 알림 Zone(지도 번호를 알면 그것까지)을 칠하는 첫 프레임,
   *  없으면 구간 첫 프레임, 표본이 그 사이를 건너뛰었으면 seek 이하의 마지막 프레임
   *  (모두 at 보다 앞이라 이동 뒤에 재생하면 알림이 다시 뜬다). 구간 첫 프레임은 이동 중이거나 다른 Zone 일 수 있다. */
  const seekIndex = (frames, seek, at, alert) => {
    const painting = alert?.zone_id == null ? -1 : frames.findIndex((frame) => frame.t >= seek && frame.t < at && frame.spraying
      && String(frame.zone_id) === String(alert.zone_id) && (alert.map_num == null ? frame.map_num != null : String(frame.map_num) === String(alert.map_num)));
    if (painting >= 0) return painting;
    const inside = frames.findIndex((frame) => frame.t >= seek && frame.t < at);
    if (inside >= 0) return inside;
    let before = -1;
    frames.forEach((frame, index) => { if (frame.t <= seek) before = index; });
    return Math.max(0, before);
  };

  const api = { KIND, DIRECTION, merge, isJump, crossed, short, describe, seekIndex };
  globalThis.motionViewerAlerts = api;
  if (typeof module === 'object' && module.exports) module.exports = api;
})();
