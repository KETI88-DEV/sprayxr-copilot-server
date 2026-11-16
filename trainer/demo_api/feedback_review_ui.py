"""Operator review screen; data loads only after an existing administrator login."""
REVIEW_HTML = '''<div class="topbar"><div><div class="eyebrow">ANSWER FEEDBACK</div><h1>Query 답변 평가</h1><div class="sub">답변 평가를 검토하고 검증한 예제만 학습 후보로 승인합니다.</div></div></div>
<p id="answer-review-state" role="status" aria-live="polite">관리자 로그인 상태를 확인합니다.</p>
<div class="toolbar"><a href="/admin/manage">관리자 로그인</a><a href="/admin/feedback">사용자 의견</a><label>검토 상태 <select id="answer-review-filter"><option value="pending">미검토</option><option value="approved">승인</option><option value="rejected">제외</option><option value="">전체</option></select></label><a href="/admin/api/feedback/training-candidates.jsonl">승인된 학습 후보 JSONL</a></div>
<p class="sub">승인은 학습 후보 등록입니다. 모델 재학습·평가·배포는 별도로 실행합니다. 도움이 되지 않은 답변은 검증한 교정 답변을 입력해야 승인할 수 있습니다. 세션별 분석 답변은 질문 시점의 완전한 문맥이 저장된 경우에만 학습 후보 파일에 포함됩니다.</p>
<div id="answer-review-items"></div>
<style>.answer-review-card{border:1px solid #cbd6e8;border-radius:12px;padding:20px;margin:14px 0;background:#fff}.answer-review-card pre{font:inherit;white-space:pre-wrap;overflow-wrap:anywhere;padding:12px;background:#f3f6fb;border-radius:8px}.answer-review-card label{display:block;margin-top:10px}.answer-review-card textarea{display:block;width:100%;box-sizing:border-box;border:1px solid #cbd6e8;border-radius:8px;padding:10px;color:#17243b;background:#fff}.answer-review-card select,.answer-review-card button{padding:8px 12px;margin:10px 8px 0 0}.answer-review-meta{color:#66768c;font-size:12px}</style>
<script defer src="/assets/answer_feedback_admin.js"></script>'''
