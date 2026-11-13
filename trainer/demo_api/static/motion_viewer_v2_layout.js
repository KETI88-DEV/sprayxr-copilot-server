// v2 화면 나누기·분석 칸 접기·그림 좌표·답변 피드백 요청·참고 카드 규칙의 순수 함수. DOM을 만지지 않는다(motion_viewer_v2.js가 부른다).
(() => {
  'use strict';
  const FEATURES = ['pose', 'order', 'coating'];
  const CHAT_MIN = 360;
  const VIEWER_MIN = 560;

  /** 분석 칸 수. 창 높이가 800px 이하이거나 뷰어 열이 1,300px보다 좁으면 한 칸(탭), 아니면 세 칸(2026-11-13 재디자인). */
  const panelCount = (width, height) => (height <= 800 || width < 1300 ? 1 : 3);

  /** 한 칸 보기에서 탭을 누른 뒤의 상태. 접혀 있으면 그 탭으로 펴고, 열린 탭을 다시 누르면 접는다. */
  const tabPress = ({ active, folded }, id) => {
    if (!FEATURES.includes(id)) return { active, folded };
    if (folded) return { active: id, folded: false };
    return id === active ? { active, folded: true } : { active: id, folded: false };
  };

  /** 재생 시각 m:ss. */
  const clock = (seconds) => {
    const whole = Math.max(0, Math.floor(Number(seconds) || 0));
    return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, '0')}`;
  };

  /**
   * 순서 그림의 줄마다 Zone 상태. 같은 순번이면 same, 다른 순번이면 moved, 이 기록에 없으면 missing(권장 줄),
   * 권장 순서에 없으면 extra(이 기록 줄). moved 는 자리가 다른 Zone 수다.
   */
  const orderMoves = (expert, actual) => {
    const top = expert.map((zone, index) => {
      const actualIndex = actual.indexOf(zone);
      return { zone, index, actualIndex, state: actualIndex < 0 ? 'missing' : actualIndex === index ? 'same' : 'moved' };
    });
    const bottom = actual.map((zone, index) => {
      const expertIndex = expert.indexOf(zone);
      return { zone, index, expertIndex, state: expertIndex < 0 ? 'extra' : expertIndex === index ? 'same' : 'moved' };
    });
    return { expert: top, actual: bottom, moved: top.filter((item) => item.state === 'moved').length,
      missing: top.filter((item) => item.state === 'missing').length };
  };

  /** 자세 그림 세로 좌표: 유사도 40~100%를 아래~위로. 범위 밖은 끝에 붙인다. */
  const POSE_LOW = 40, POSE_HIGH = 100;
  const poseY = (value, box) => box.top + (1 - (Math.max(POSE_LOW, Math.min(POSE_HIGH, Number(value))) - POSE_LOW) / (POSE_HIGH - POSE_LOW)) * (box.bottom - box.top);
  /** 분사 순서대로 가로를 고르게 나눈다. 하나뿐이면 가운데. */
  const posePoints = (values, box) => values.map((value, index) => ({
    x: values.length < 2 ? (box.left + box.right) / 2 : box.left + index * (box.right - box.left) / (values.length - 1),
    y: poseY(value, box),
  }));

  /** 알림을 종류별로 센다. order 순서대로, 없는 종류는 뺀다. */
  const countByKind = (alerts, order) => order
    .map((type) => ({ type, count: (alerts || []).filter((alert) => alert.anomaly_type === type).length }))
    .filter((item) => item.count > 0);

  const defaultChatWidth = (total) => Math.min(640, Math.max(420, Math.round(total * 0.32)));
  const clampChatWidth = (width, total) => {
    if (!Number.isFinite(width)) return defaultChatWidth(total);
    return Math.max(CHAT_MIN, Math.min(Math.round(width), total - VIEWER_MIN));
  };

  // 서버 한도는 글자 수다. UTF-16 단위로 자르면 이모지 반쪽이 남아 서버가 422로 거절한다.
  const clip = (value, limit) => (value == null ? null : Array.from(String(value)).slice(0, limit).join(''));
  /**
   * 기존 의견 접수(POST /api/feedback)에 답변 정보를 담는다. 잘못됨=bug, 메모=improvement.
   * 잘못됨은 글이 없어도 보낼 수 있고, 메모는 글이 있어야 한다(없으면 null).
   */
  const feedbackPayload = ({ submissionId, kind, text, correct, answer }) => {
    const wrong = kind === 'wrong';
    const message = String(text || '').trim() || (wrong ? '잘못된 답으로 표시' : '');
    if (!message) return null;
    return {
      submission_id: submissionId, message: clip(message, 4000), kind: wrong ? 'bug' : 'improvement', page: 'motion',
      answer: {
        conversation_id: clip(answer.conversation_id, 80), question: clip(answer.question, 2000), answer: clip(answer.answer, 8000),
        task: answer.task, response_mode: answer.response_mode, answer_source: clip(answer.answer_source, 40),
        zone_id: answer.zone_id == null ? null : clip(answer.zone_id, 8),
        map_num: answer.map_num != null && answer.map_num !== '' && Number.isInteger(Number(answer.map_num)) ? Number(answer.map_num) : null,
        playhead_seconds: Number.isFinite(answer.playhead_seconds) ? answer.playhead_seconds : null,
        record: clip(answer.record, 200), session_id: clip(answer.session_id, 80), worker_id: clip(answer.worker_id, 80),
        correct_answer: clip(String(correct || '').trim(), 4000),
      },
    };
  };

  // 채팅만 볼 때 답 아래 참고 카드(B안). 질문한 순간의 기록 값으로 그린다. 모델이 고르지 않는다.
  const ATTACHMENTS = {
    pose: ['poseChart', 'zoneMetrics'],
    coating: ['zoneMetrics'],
  };
  const attachmentsFor = (tabId) => [...(ATTACHMENTS[tabId] || [])];

  window.motionViewerV2Layout = {
    FEATURES, panelCount, tabPress, clock, orderMoves, poseY, posePoints, countByKind, defaultChatWidth, clampChatWidth, feedbackPayload, attachmentsFor,
  };
})();
