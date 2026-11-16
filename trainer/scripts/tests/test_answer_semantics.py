"""Question requirements and source scope survive concise generation."""
import unittest


class AnswerSemanticsTests(unittest.TestCase):
    def test_general_cause_and_correction_are_reviewed_but_a_definition_is_not(self):
        from trainer.scripts.answer_semantics import review_required
        for question in ['저도막이 생기는 원인은 무엇인가요?', '기포는 왜 생겨요?',
                         '둘은 어떻게 다른가요?', '거리는 적절했는데?', '아까 설명과 다른데요?',
                         '자세 종류로는 무엇이 있어요?']:
            self.assertTrue(review_required(question), question)
        for question in ['저도막이 뭐야?', '적정 이격거리는?', '안녕']:
            self.assertFalse(review_required(question), question)

    def test_technical_followup_markers_trigger_review_without_keyword_matching(self):
        from trainer.scripts.answer_semantics import review_required
        question = '평평한 면 전체가 얇아지는 것도 같은 원리야?'
        self.assertFalse(review_required(question))
        for marker in ['[문서 후속 질문]', '[후속 질문 해석 — 미검증 대화 참고]']:
            self.assertTrue(review_required(question, instruction=marker))

    def test_duplicate_sources_do_not_crowd_out_a_distinct_relevant_passage(self):
        from trainer.scripts.answer_semantics import select_evidence
        chunks = [{'chunk_id': 'DOC-edge-C0001', 'text': '같은 특수 사례 설명 ' * 10, 'rerank_score': 8.0},
                  {'chunk_id': 'DOC-edge-C0002', 'text': '같은 특수 사례 설명 ' * 10, 'rerank_score': 7.9},
                  {'chunk_id': 'DOC-manual-C0001', 'text': '일반적인 작업 조건과 발생 원인을 설명합니다.', 'rerank_score': 7.8},
                  {'chunk_id': 'DOC-edge-C0003', 'text': '특수 사례에 대한 서로 다른 추가 설명입니다.', 'rerank_score': 7.7},
                  {'chunk_id': 'DOC-other-C0001', 'text': '관련성이 매우 낮은 다른 주제입니다.', 'rerank_score': 0.1}]
        selected = select_evidence(chunks, limit=3)
        self.assertEqual(selected[0]['chunk_id'], 'DOC-edge-C0001')
        self.assertIn('DOC-manual-C0001', [c['chunk_id'] for c in selected])
        self.assertNotIn('DOC-edge-C0002', [c['chunk_id'] for c in selected])
        self.assertNotIn('DOC-other-C0001', [c['chunk_id'] for c in selected])

    def test_review_accepts_only_the_three_defined_verdicts(self):
        from trainer.scripts.answer_semantics import parse_review
        self.assertEqual(parse_review(' OK '), 'ok')
        self.assertEqual(parse_review('REWRITE'), 'rewrite')
        self.assertEqual(parse_review('INSUFFICIENT'), 'insufficient')
        self.assertIsNone(parse_review('OK, 하지만 원인을 더 써 주세요.'))

    def test_review_can_explain_a_scope_error_before_choosing_its_verdict(self):
        from trainer.scripts.answer_semantics import parse_review
        self.assertEqual(parse_review('이유: 추가 조건을 반영하지 않고 같은 장점만 반복함\n판정: REWRITE'), 'rewrite')
        self.assertIsNone(parse_review('이유: 검토\n판정: MAYBE'))

    def test_repair_receives_the_semantic_review_reason(self):
        from trainer.scripts.answer_semantics import review_answer
        outputs = iter(['이유: 추가 조건을 무시함\n판정: REWRITE', '이유: 조건과 기술 근거를 반영함\n판정: OK'])
        repairs = []
        def regenerate(repair, timeout):
            repairs.append(repair)
            return '조건을 반영한 수정 답변'
        result = review_answer('조건은 적절했는데?', '문맥', '조건의 장점만 반복', [],
            complete=lambda *args: next(outputs), regenerate=regenerate, require_evidence=False, budget_seconds=3)
        self.assertEqual(result.text, '조건을 반영한 수정 답변')
        self.assertIn('추가 조건을 무시함', repairs[0])

    def test_review_keeps_dialogue_before_a_long_simulator_context(self):
        from trainer.scripts.answer_semantics import review_messages
        instruction = ('[최근 대화]\n사용자: 처음 원인에 관한 질문\nCo-Pilot: 이전 설명\n'
                       '[시뮬레이터 기본 설정]\n' + '장황한 설정 ' * 1000)
        messages = review_messages('조건은 적절했는데?', instruction, '후보', [], require_evidence=True)
        self.assertIn('처음 원인에 관한 질문', messages[-1]['content'])
        self.assertNotIn('장황한 설정 ' * 100, messages[-1]['content'])

    def test_a_blank_or_invalid_review_cannot_approve_the_candidate(self):
        from trainer.scripts.answer_semantics import review_answer
        for output in ['', 'Maybe OK', '[답변] 괜찮습니다.']:
            result = review_answer('어떤 원인이 있나요?', '질문과 대화', '확인되지 않은 후보', [],
                complete=lambda *args: output, regenerate=lambda *args: self.fail('invalid review must not regenerate'),
                require_evidence=True, budget_seconds=3)
            self.assertFalse(result.cacheable)
            self.assertEqual(result.verdict, 'invalid_review')
            self.assertNotIn('확인되지 않은 후보', result.text)

    def test_review_deadline_stops_before_a_repair_without_time_for_it(self):
        from trainer.scripts.answer_semantics import review_answer
        clock = [0.0]
        def complete(*args):
            clock[0] = 3.0
            return 'REWRITE'
        result = review_answer('왜 생겨요?', '문맥', '정의만 있습니다.', [], complete=complete,
            regenerate=lambda *args: self.fail('deadline must stop repair'), require_evidence=True,
            budget_seconds=3, clock=lambda:clock[0])
        self.assertEqual(result.verdict, 'budget_exhausted')
        self.assertEqual(result.retries, 0)

    def test_an_ok_that_arrives_after_the_budget_is_not_approved(self):
        from trainer.scripts.answer_semantics import review_answer
        clock = [0.0]
        def complete(*args):
            clock[0] = 3.1
            return 'OK'
        result = review_answer('왜 생겨요?', '문맥', '후보 답변', [], complete=complete,
            regenerate=lambda *args:'수정', require_evidence=True, budget_seconds=3, clock=lambda:clock[0])
        self.assertEqual(result.verdict, 'budget_exhausted')
        self.assertFalse(result.cacheable)

    def test_diagnostics_identify_the_passage_without_exposing_full_text(self):
        from trainer.scripts.answer_semantics import evidence_trace
        trace = evidence_trace([{'chunk_id':'DOC-a-C0001', 'text':'source text', 'rerank_score':5.0}])
        self.assertEqual(trace[0]['chunk_id'], 'DOC-a-C0001')
        self.assertEqual(len(trace[0]['sha256']), 64)
        self.assertEqual(trace[0]['characters'], 11)
        self.assertNotIn('text', trace[0])

    def test_retrieval_carries_elliptical_conditions_without_poisoning_a_new_topic(self):
        from trainer.demo_api.query_runtime import retrieval_question
        context = '[최근 대화]\n사용자: 저도막이 생기는 원인은 무엇인가요?\nCo-Pilot: 미검증 답변'
        continued = retrieval_question('거리는 적절했는데?', context=context)
        self.assertIn('거리는 적절했는데?', continued)
        self.assertIn('저도막', continued)
        self.assertNotIn('저도막', retrieval_question('프랑스의 수도는?', context=context))
        self.assertNotIn('저도막', retrieval_question('블라스팅 노즐은 어떻게 관리해?', context=context))
        self.assertNotIn('저도막', retrieval_question('그럼 우주 망원경은 어떻게 작동해?', context=context))

    def test_legacy_followup_preserves_the_problem_and_intermediate_conditions_in_order(self):
        from trainer.demo_api.query_runtime import retrieval_question
        context = ('[문서 후속 질문]\n[최근 대화]\n사용자: 저도막이 생기는 원인은 무엇인가요?\n'
                   'Co-Pilot: 이전의 미검증 설명\n사용자: 도료 유량은 정상이야\nCo-Pilot: 추가 설명')
        query = retrieval_question('그 외 확인할 것은?', context=context)
        self.assertIn('저도막', query)
        self.assertIn('도료 유량은 정상이야', query)
        self.assertIn('그 외 확인할 것은?', query)
        self.assertLess(query.index('저도막'), query.index('도료 유량은 정상이야'))
        switched = retrieval_question('새 질문은?', context='[새 주제 질문]\n' + context)
        self.assertNotIn('저도막', switched)
        self.assertNotIn('유량', switched)


if __name__ == '__main__':
    unittest.main()
