"""Live knowledge generation repairs a semantic mismatch once, without fixed QA."""
from trainer.demo_api.tests.test_query_quick_latency import _pipeline
from trainer.scripts.answer_semantics import REVIEW_REGEX


def run_query(pipeline, **overrides):
    options = dict(instruction='[사용자 질문]\n저도막의 원인은?\n[최근 대화]\n사용자: 이전 질문',
        force_route='fast', answer_question='저도막의 원인은?', use_pipeline_history=False,
        quick_cache_scope='semantic-test', concise_answer=True)
    options.update(overrides)
    return pipeline.run(**options)


def test_inadequate_cause_answer_is_rewritten_once_and_only_repaired_answer_is_cached(monkeypatch):
    pipeline, _, searched, chunks = _pipeline(monkeypatch)
    chunks[:] = [{'chunk_id':'DOC-a-C0001', 'text':'빠른 이동과 부족한 겹침은 낮은 도막 두께를 만들 수 있습니다.', 'rerank_score':8.0}]
    outputs = iter(['[답변] 저도막은 두께가 부족한 현상입니다.\n[근거] DOC-a-C0001', 'REWRITE',
                    '[답변] 이동이 빠르거나 겹침이 부족하면 도료가 충분히 쌓이지 않을 수 있습니다.\n[근거] DOC-a-C0001',
                    '이유: 문단이 요청한 원인을 설명함\n근거: 1\n판정: OK'])
    calls = []
    def generate(messages, **kwargs):
        calls.append((messages, kwargs))
        return next(outputs)
    pipeline._generate_direct = generate
    first = run_query(pipeline)
    assert '이동이 빠르거나' in first['response'] and len(calls) == 4
    assert first['metadata']['answer_review']['verdict'] == 'rewritten'
    assert first['metadata']['answer_review']['retries'] == 1
    second = run_query(pipeline)
    assert second['response'] == first['response'] and len(calls) == 4
    assert second['metadata']['quick_cache_hit'] is True
    assert len(searched) == 2


def test_adequate_definition_keeps_one_generation_and_legacy_calls_are_unchanged(monkeypatch):
    pipeline, generated, _, _ = _pipeline(monkeypatch)
    pipeline.run(instruction='저도막이 뭐야?', answer_question='저도막이 뭐야?', force_route='fast', concise_answer=True)
    assert len(generated) == 1
    pipeline.run(instruction='평가 기본 호출', force_route='fast')
    assert len(generated) == 2


def test_failed_or_insufficient_review_does_not_publish_or_cache_the_rejected_candidate(monkeypatch):
    pipeline, _, _, _ = _pipeline(monkeypatch)
    pipeline.config['llm']['append_citations'] = True
    outputs = iter(['[답변] 특정 부위의 현상입니다.\n[근거] DOC-a-C0001', 'INSUFFICIENT'] * 2)
    calls = []
    def generate(*args, **kwargs):
        calls.append(True)
        return next(outputs)
    pipeline._generate_direct = generate
    for _ in range(2):
        result = run_query(pipeline)
        assert '특정 부위의 현상입니다' not in result['response']
        assert result['metadata']['quick_cache_hit'] is False
        assert result['metadata']['answer_review']['verdict'] == 'insufficient'
        assert '[근거]' not in result['response'] and result['citations'] == []
    assert len(calls) == 4


def test_changed_evidence_invalidates_semantically_validated_cache(monkeypatch):
    pipeline, _, _, chunks = _pipeline(monkeypatch)
    calls = []
    def generate(messages, **kwargs):
        calls.append(kwargs)
        if kwargs.get('answer_regex') == REVIEW_REGEX:
            return '이유: 문단이 요청한 원인을 설명함\n근거: 1\n판정: OK'
        return '[답변] 근거에 적힌 작업 조건이 원인이 될 수 있습니다.\n[근거] DOC-a-C0001'
    pipeline._generate_direct = generate
    run_query(pipeline)
    chunks[0]['text'] = '변경된 작업 자료입니다.'
    run_query(pipeline)
    assert len(calls) == 4


def test_a_rewrite_that_still_does_not_answer_is_withheld_instead_of_cached(monkeypatch):
    pipeline, _, _, _ = _pipeline(monkeypatch)
    outputs = iter(['[답변] 정의만 설명합니다.\n[근거] DOC-a-C0001', 'REWRITE',
                    '[답변] 여전히 정의만 설명합니다.\n[근거] DOC-a-C0001', 'REWRITE'])
    pipeline._generate_direct = lambda *args, **kwargs: next(outputs)
    result = run_query(pipeline)
    assert '정의만 설명' not in result['response']
    assert result['metadata']['answer_review']['verdict'] == 'rejected'


def test_review_timeout_never_certifies_or_caches_an_unchecked_candidate(monkeypatch):
    pipeline, _, _, _ = _pipeline(monkeypatch)
    calls = []
    def generate(messages, **kwargs):
        calls.append(kwargs)
        if kwargs.get('answer_regex') == REVIEW_REGEX:
            raise TimeoutError('judge unavailable')
        return '[답변] 잘못된 정의만 반복합니다.\n[근거] DOC-a-C0001'
    pipeline._generate_direct = generate
    for _ in range(2):
        result = run_query(pipeline)
        assert '잘못된 정의' not in result['response']
        assert result['metadata']['answer_review']['verdict'] == 'unavailable'
        assert not result['metadata']['quick_cache_hit']
    assert len(calls) == 4


def test_previous_dialogue_change_invalidates_a_reviewed_answer(monkeypatch):
    pipeline, _, _, _ = _pipeline(monkeypatch)
    calls = []
    def generate(messages, **kwargs):
        calls.append(kwargs)
        if kwargs.get('answer_regex') == REVIEW_REGEX:
            return '이유: 문단과 조건을 반영함\n근거: 1\n판정: OK'
        return '[답변] 앞서 제시한 조건을 반영한 가능한 원인입니다.\n[근거] DOC-a-C0001'
    pipeline._generate_direct = generate
    run_query(pipeline)
    run_query(pipeline, instruction='[최근 대화]\n사용자: 거리는 적절했습니다.\n[현재 질문]\n저도막의 원인은?')
    assert len(calls) == 4


def test_unreviewed_abstention_is_not_cached_even_with_a_valid_response_shape(monkeypatch):
    pipeline, _, _, _ = _pipeline(monkeypatch)
    calls = []
    def generate(*args, **kwargs):
        calls.append(True)
        return '[답변] 제공된 자료만으로는 질문에 맞는 설명을 충분히 확인하지 못했습니다.'
    pipeline._generate_direct = generate
    for _ in range(2):
        result = run_query(pipeline, answer_question='저도막이 뭐야?')
        assert not result['metadata']['quick_cache_hit']
        assert result['metadata']['answer_review']['cacheable'] is False
    assert len(calls) == 2


def test_a_citation_only_candidate_is_repaired_before_a_judge_can_approve_it(monkeypatch):
    pipeline, _, _, _ = _pipeline(monkeypatch)
    outputs = iter(['[답변]\n[근거] DOC-a-C0001',
                    '[답변] 질문에 요청한 내용을 설명합니다.\n[근거] DOC-a-C0001',
                    '이유: 문단에 근거한 종류를 설명함\n근거: 1\n판정: OK'])
    calls = []
    def generate(messages, **kwargs):
        calls.append(kwargs)
        return next(outputs)
    pipeline._generate_direct = generate
    result = run_query(pipeline, answer_question='어떤 종류가 있나요?')
    assert '요청한 내용을 설명' in result['response']
    assert result['metadata']['answer_review']['verdict'] == 'rewritten'
    assert len(calls) == 3


def test_a_resolved_technical_followup_is_reviewed_without_question_keywords(monkeypatch):
    pipeline, _, _, _ = _pipeline(monkeypatch)
    outputs = iter(['[답변] 평평한 면도 모서리와 같은 원리입니다.\n[근거] DOC-a-C0001', 'INSUFFICIENT'])
    calls = []
    def generate(*args, **kwargs):
        calls.append(True)
        return next(outputs)
    pipeline._generate_direct = generate
    result = run_query(pipeline,
        instruction='[문서 후속 질문]\n[최근 대화]\n사용자: 모서리만 얇아지는 이유?\n[후속 질문 해석 — 미검증 대화 참고]\n평평한 면 전체도 같은 현상인가?',
        answer_question='평평한 면 전체가 얇아지는 것도 같은 원리야?')
    assert '평평한 면도 모서리와 같은 원리' not in result['response']
    assert result['metadata']['answer_review']['verdict'] == 'insufficient'
    assert len(calls) == 2


def test_pipeline_preserves_supported_explanation_and_binds_only_its_real_source(monkeypatch):
    pipeline, _, _, chunks = _pipeline(monkeypatch)
    pipeline.config['llm']['append_citations'] = True
    chunks[:] = [{'chunk_id':'DOC-definition-C0001', 'text':'X의 뜻을 정의합니다.', 'rerank_score':8.0},
                 {'chunk_id':'DOC-causes-C0001', 'text':'A와 B가 X를 일으키는 원인입니다.', 'rerank_score':7.9}]
    outputs = iter(['[답변] 원인은 A와 B입니다.\n[근거] DOC-definition-C0001',
                    '이유: 둘째 문단이 두 원인을 설명함\n근거: 2\n판정: OK'])
    calls = []
    def generate(*args, **kwargs):
        calls.append(True)
        return next(outputs)
    pipeline._generate_direct = generate
    result = run_query(pipeline, answer_question='X의 원인은?')
    assert result['response'] == '[답변] 원인은 A와 B입니다.\n[근거] DOC-causes-C0001'
    assert result['citations'] == ['DOC-causes-C0001']
    assert result['metadata']['answer_review']['verdict'] == 'attribution_corrected'
    assert len(calls) == 2
