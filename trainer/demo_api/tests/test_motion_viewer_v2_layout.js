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

test('창 높이가 800 이하이거나 뷰어 열이 1300보다 좁으면 한 칸, 아니면 세 칸(두 칸 보기는 없다)', () => {
  assert.equal(L.panelCount(1306, 1080), 3);
  assert.equal(L.panelCount(1300, 801), 3);
  assert.equal(L.panelCount(1299, 1080), 1);
  assert.equal(L.panelCount(1600, 800), 1);
  assert.equal(L.panelCount(800, 900), 1);
  assert.equal(L.pickShown, undefined);
  assert.equal(L.normalizeShown, undefined);
});

test('한 칸 보기 탭: 다른 탭은 바꾸고, 열린 탭을 다시 누르면 접고, 접힌 채 누르면 그 탭으로 편다', () => {
  let state = { active: 'pose', folded: false };
  state = L.tabPress(state, 'order');
  assert.deepEqual({ ...state }, { active: 'order', folded: false });
  state = L.tabPress(state, 'order');
  assert.deepEqual({ ...state }, { active: 'order', folded: true });
  state = L.tabPress(state, 'coating');
  assert.deepEqual({ ...state }, { active: 'coating', folded: false });
  assert.deepEqual({ ...L.tabPress(state, 'nope') }, { active: 'coating', folded: false });
});

test('재생 시각은 m:ss 로 쓴다', () => {
  assert.equal(L.clock(0), '0:00');
  assert.equal(L.clock(49.9), '0:49');
  assert.equal(L.clock(75), '1:15');
  assert.equal(L.clock(3600), '60:00');
  assert.equal(L.clock(-3), '0:00');
  assert.equal(L.clock(Number.NaN), '0:00');
});

test('순서: 권장 순서와 같은 자리는 same, 다른 자리는 moved, 이 기록에 없으면 missing, 기준 밖은 extra', () => {
  const result = L.orderMoves(['8', '12', '11', '10'], ['12', '8', '11', '99']);
  assert.deepEqual(result.expert.map((item) => item.state), ['moved', 'moved', 'same', 'missing']);
  assert.deepEqual(result.expert.map((item) => item.actualIndex), [1, 0, 2, -1]);
  assert.deepEqual(result.actual.map((item) => item.state), ['moved', 'moved', 'same', 'extra']);
  assert.equal(result.moved, 2);
  assert.equal(result.missing, 1);
  const same = L.orderMoves(['1', '2'], ['1', '2']);
  assert.equal(same.moved, 0);
  assert.equal(same.missing, 0);
});

test('자세 그림 좌표: 분사 순서대로 고르게 놓고 유사도 40~100%를 위아래로, 범위 밖은 끝에 붙인다', () => {
  const box = { left: 30, right: 230, top: 10, bottom: 70 };
  const points = L.posePoints([40, 100, 70, 20, 130], box);
  assert.deepEqual(points.map((p) => p.x), [30, 80, 130, 180, 230]);
  assert.deepEqual(points.map((p) => p.y), [70, 10, 40, 70, 10]);
  assert.equal(L.posePoints([80], box)[0].x, 130);
  assert.equal(L.poseY(75, box), 35);
});

test('알림 개수는 종류 순서대로 세고 없는 종류는 뺀다', () => {
  const alerts = [{ anomaly_type: 'b' }, { anomaly_type: 'a' }, { anomaly_type: 'b' }, { anomaly_type: 'z' }];
  assert.deepEqual(L.countByKind(alerts, ['a', 'b', 'c']).map((item) => [item.type, item.count]), [['a', 1], ['b', 2]]);
  assert.deepEqual([...L.countByKind([], ['a'])], []);
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

test('참고 카드는 자세 질문에 분사 그림과 Zone 수치, 도막 질문에 Zone 수치만, 순서·지식에는 붙이지 않는다', () => {
  assert.deepEqual([...L.attachmentsFor('pose')], ['poseChart', 'zoneMetrics']);
  assert.deepEqual([...L.attachmentsFor('coating')], ['zoneMetrics']);
  assert.deepEqual([...L.attachmentsFor('order')], []);
  assert.deepEqual([...L.attachmentsFor('knowledge')], []);
  assert.deepEqual([...L.attachmentsFor('unknown')], []);
});

test('재생 상태가 Map 번호를 문자열로 들고 있어도 숫자로 보낸다', () => {
  const body = L.feedbackPayload({ submissionId: 'e', kind: 'wrong', text: 'x', correct: '', answer: { ...answer, map_num: '1' } });
  assert.equal(body.answer.map_num, 1);
  assert.equal(L.feedbackPayload({ submissionId: 'f', kind: 'wrong', text: 'x', correct: '', answer: { ...answer, map_num: 'x' } }).answer.map_num, null);
});

test('자르는 자리가 이모지에 걸려도 반쪽 글자를 남기지 않는다', () => {
  const question = `${'가'.repeat(1999)}😀끝`;
  const body = L.feedbackPayload({ submissionId: 'g', kind: 'wrong', text: 'x', correct: '', answer: { ...answer, question } });
  assert.equal([...body.answer.question].length, 2000);
  assert.ok(!/[\ud800-\udbff]$/.test(body.answer.question));
  assert.ok(body.answer.question.endsWith('😀'));
});
