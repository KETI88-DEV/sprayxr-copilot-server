'use strict';
// v2 화면 나누기·분석 칸 접기·답변 피드백 요청·참고 첨부 규칙(motion_viewer_v2_layout.js)의 순수 함수 시험.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../static/motion_viewer_v2_layout.js'), 'utf8');
const sandbox = { window: {} };
vm.runInNewContext(source, sandbox);
const L = sandbox.window.motionViewerV2Layout;

test('뷰어 폭이 1300 이상이면 세 칸, 900 이상이면 두 칸, 그보다 좁으면 한 칸', () => {
  assert.equal(L.featureCount(1306), 3);
  assert.equal(L.featureCount(1300), 3);
  assert.equal(L.featureCount(1299), 2);
  assert.equal(L.featureCount(900), 2);
  assert.equal(L.featureCount(899), 1);
});

test('두 칸일 때 숨은 칸을 열면 먼저 열어 둔 칸과 자리를 바꾼다', () => {
  let shown = L.normalizeShown([], 2);
  assert.deepEqual([...shown], ['pose', 'order']);
  shown = L.pickShown(shown, 'coating', 2);
  assert.deepEqual([...L.displayOrder(shown)], ['order', 'coating']);
  shown = L.pickShown(shown, 'pose', 2);
  assert.deepEqual([...L.displayOrder(shown)], ['pose', 'coating']);
});

test('이미 보이는 칸을 누르면 칸은 그대로이고 다음에 밀려나지 않는다', () => {
  let shown = ['pose', 'order'];
  shown = L.pickShown(shown, 'pose', 2);
  assert.deepEqual([...L.displayOrder(shown)], ['pose', 'order']);
  shown = L.pickShown(shown, 'coating', 2);
  assert.deepEqual([...L.displayOrder(shown)], ['pose', 'coating']);
});

test('칸 수가 줄면 가장 최근 칸을 남기고, 늘면 빠진 칸을 앞에서부터 채운다', () => {
  assert.deepEqual([...L.normalizeShown(['pose', 'coating'], 1)], ['coating']);
  assert.deepEqual([...L.displayOrder(L.normalizeShown(['coating'], 2))], ['pose', 'coating']);
  assert.deepEqual([...L.displayOrder(L.normalizeShown(['order'], 3))], ['pose', 'order', 'coating']);
  assert.deepEqual([...L.normalizeShown(['nope', 'order', 'order'], 1)], ['order']);
});

test('채팅 폭 기본값은 화면의 32%를 420~640px로 묶고, 끌어도 뷰어 560px와 채팅 360px를 지킨다', () => {
  assert.equal(L.defaultChatWidth(1920), 614);
  assert.equal(L.defaultChatWidth(1366), 437);
  assert.equal(L.defaultChatWidth(2560), 640);
  assert.equal(L.clampChatWidth(200, 1920), 360);
  assert.equal(L.clampChatWidth(1700, 1920), 1360);
  assert.equal(L.clampChatWidth(900, 1920), 900);
  assert.equal(L.clampChatWidth(500, 800), 360);
  assert.equal(L.clampChatWidth(Number.NaN, 1920), 614);
});

const answer = {
  conversation_id: 'mv2-abc', question: '이 Zone에서 어떻게 고쳐?', answer: 'Zone 9에서는 수평 왕복이 기준입니다.',
  task: 'pose_recomm', response_mode: 'quick', answer_source: 'conversation', zone_id: '9', map_num: 1,
  playhead_seconds: 12.4, record: '작업자 · 스프레이 · 학습모드 4 · S1', session_id: 'S1', worker_id: 'W1',
};

test('잘못됨은 bug, 메모는 improvement 로 기존 의견 접수 형식에 담는다', () => {
  const wrong = L.feedbackPayload({ submissionId: 'id-1', kind: 'wrong', text: '  기준 숫자가 다릅니다 ', correct: ' 450 mm ', answer });
  assert.equal(wrong.kind, 'bug');
  assert.equal(wrong.page, 'motion');
  assert.equal(wrong.message, '기준 숫자가 다릅니다');
  assert.equal(wrong.answer.correct_answer, '450 mm');
  assert.equal(wrong.answer.zone_id, '9');
  const memo = L.feedbackPayload({ submissionId: 'id-2', kind: 'memo', text: '좋은 설명', correct: '', answer });
  assert.equal(memo.kind, 'improvement');
  assert.equal(memo.answer.correct_answer, '');
});

test('잘못됨은 글 없이도 보낼 수 있고 메모는 글이 있어야 한다', () => {
  assert.equal(L.feedbackPayload({ submissionId: 'a', kind: 'wrong', text: '  ', correct: '', answer }).message, '잘못된 답으로 표시');
  assert.equal(L.feedbackPayload({ submissionId: 'b', kind: 'memo', text: ' ', correct: '', answer }), null);
});

test('서버 한도보다 긴 질문·답은 잘라 보내고 Zone 번호는 문자열로 보낸다', () => {
  const long = { ...answer, question: '가'.repeat(2500), answer: '나'.repeat(9000), zone_id: 9, record: 'r'.repeat(300) };
  const body = L.feedbackPayload({ submissionId: 'c', kind: 'wrong', text: 'x', correct: '', answer: long });
  assert.equal(body.answer.question.length, 2000);
  assert.equal(body.answer.answer.length, 8000);
  assert.equal(body.answer.record.length, 200);
  assert.equal(body.answer.zone_id, '9');
  const none = L.feedbackPayload({ submissionId: 'd', kind: 'wrong', text: 'x', correct: '', answer: { ...answer, zone_id: null, map_num: null } });
  assert.equal(none.answer.zone_id, null);
  assert.equal(none.answer.map_num, null);
});

test('참고 첨부는 질문 칸별로 정해진 그림만 고르고 지식 질문에는 붙이지 않는다', () => {
  assert.deepEqual([...L.attachmentsFor('pose')], ['skeleton', 'poseBars', 'zoneMetrics']);
  assert.deepEqual([...L.attachmentsFor('order')], ['scene', 'order']);
  assert.deepEqual([...L.attachmentsFor('coating')], ['scene', 'coating', 'zoneMetrics']);
  assert.deepEqual([...L.attachmentsFor('knowledge')], []);
  assert.deepEqual([...L.attachmentsFor('unknown')], []);
});
