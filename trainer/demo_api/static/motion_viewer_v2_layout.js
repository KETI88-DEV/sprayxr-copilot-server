// v2 화면 나누기·분석 칸 접기·답변 피드백 요청·참고 첨부 규칙의 순수 함수. DOM을 만지지 않는다(motion_viewer_v2.js가 부른다).
(() => {
  'use strict';
  const FEATURES = ['pose', 'order', 'coating'];
  const CHAT_MIN = 360;
  const VIEWER_MIN = 560;

  /** 뷰어 폭에 들어가는 분석 칸 수. 세 칸은 1,300px부터 겹치지 않는다(2026-11-13 1920×1080 목업). */
  const featureCount = (width) => (width >= 1300 ? 3 : width >= 900 ? 2 : 1);

  /** 보이는 칸 목록(먼저 연 것부터)을 칸 수에 맞춘다. 줄면 최근 것을 남기고, 늘면 빠진 칸을 앞(오래된 쪽)에 채운다. */
  const normalizeShown = (shown, count) => {
    const list = [];
    for (const id of shown || []) if (FEATURES.includes(id) && !list.includes(id)) list.push(id);
    const kept = list.slice(-count);
    const missing = FEATURES.filter((id) => !kept.includes(id)).slice(0, Math.max(0, count - kept.length));
    return [...missing, ...kept];
  };

  /** 칸을 연다. 이미 보이면 가장 최근으로 올리고, 아니면 가장 먼저 연 칸과 자리를 바꾼다. */
  const pickShown = (shown, id, count) => {
    if (!FEATURES.includes(id)) return normalizeShown(shown, count);
    return normalizeShown([...(shown || []).filter((item) => item !== id), id], count);
  };

  /** 화면에는 늘 자세·순서·도막 순으로 놓는다. */
  const displayOrder = (shown) => FEATURES.filter((id) => (shown || []).includes(id));

  const defaultChatWidth = (total) => Math.min(640, Math.max(420, Math.round(total * 0.32)));
  const clampChatWidth = (width, total) => {
    if (!Number.isFinite(width)) return defaultChatWidth(total);
    return Math.max(CHAT_MIN, Math.min(Math.round(width), total - VIEWER_MIN));
  };

  const clip = (value, limit) => (value == null ? null : String(value).slice(0, limit));
  /**
   * 기존 의견 접수(POST /api/feedback)에 답변 정보를 담는다. 잘못됨=bug, 메모=improvement.
   * 잘못됨은 글이 없어도 보낼 수 있고, 메모는 글이 있어야 한다(없으면 null).
   */
  const feedbackPayload = ({ submissionId, kind, text, correct, answer }) => {
    const wrong = kind === 'wrong';
    const message = String(text || '').trim() || (wrong ? '잘못된 답으로 표시' : '');
    if (!message) return null;
    return {
      submission_id: submissionId, message: message.slice(0, 4000), kind: wrong ? 'bug' : 'improvement', page: 'motion',
      answer: {
        conversation_id: clip(answer.conversation_id, 80), question: clip(answer.question, 2000), answer: clip(answer.answer, 8000),
        task: answer.task, response_mode: answer.response_mode, answer_source: clip(answer.answer_source, 40),
        zone_id: answer.zone_id == null ? null : clip(answer.zone_id, 8),
        map_num: Number.isInteger(answer.map_num) ? answer.map_num : null,
        playhead_seconds: Number.isFinite(answer.playhead_seconds) ? answer.playhead_seconds : null,
        record: clip(answer.record, 200), session_id: clip(answer.session_id, 80), worker_id: clip(answer.worker_id, 80),
        correct_answer: String(correct || '').trim().slice(0, 4000),
      },
    };
  };

  // 질문 칸별 참고 첨부. 그림은 뷰어가 이미 그린 칸을 답을 받은 순간 그대로 옮긴다. 모델이 고르지 않는다.
  const ATTACHMENTS = {
    pose: ['skeleton', 'poseBars', 'zoneMetrics'],
    order: ['scene', 'order'],
    coating: ['scene', 'coating', 'zoneMetrics'],
  };
  const attachmentsFor = (tabId) => [...(ATTACHMENTS[tabId] || [])];

  window.motionViewerV2Layout = {
    FEATURES, featureCount, normalizeShown, pickShown, displayOrder, defaultChatWidth, clampChatWidth, feedbackPayload, attachmentsFor,
  };
})();
