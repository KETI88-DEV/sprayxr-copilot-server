"""A review may cite only visible evidence, never a decoder-invented paragraph."""
from itertools import combinations
import re

import pytest

from trainer.scripts import answer_semantics as semantics


def _grammar(count):
    builder = getattr(semantics, 'review_regex', None)
    assert callable(builder), 'Missing decoder grammar bounded by visible evidence count'
    return builder(count)


def _review(support, verdict='OK'):
    return f'이유: 원문의 주장을 반영함\n근거: {support}\n판정: {verdict}'


@pytest.mark.parametrize('count', [0, 1, 2, 3, 4, 5, 6, 8, 12])
def test_review_grammar_allows_exactly_the_visible_canonical_support_sets(count):
    grammar = _grammar(count)
    visible = min(count, 8)
    allowed = {'없음'} | {','.join(map(str, values))
                         for size in range(1, visible + 1)
                         for values in combinations(range(1, visible + 1), size)}
    for support in allowed:
        for verdict in ('OK', 'REWRITE', 'INSUFFICIENT'):
            assert re.fullmatch(grammar, _review(support, verdict)), (count, support, verdict)
    for support in ('0', '9', '', '1,1', '2,1', '1,2,4,1', str(visible + 1)):
        assert re.fullmatch(grammar, _review(support)) is None, (count, support)


def test_three_passages_cannot_generate_the_observed_nonexistent_fourth_support():
    observed = '이유: 겹침 도장의 목적을 정확히 반영  \n근거: 1,2,4,1\n판정: OK'
    assert re.fullmatch(semantics.REVIEW_REGEX, observed)
    assert re.fullmatch(_grammar(3), observed) is None
    assert re.fullmatch(_grammar(3), _review('1,3'))
    assert re.fullmatch(_grammar(3), _review('4')) is None


def test_decoder_requires_ascending_unique_support_but_the_parser_keeps_legacy_compatibility():
    assert re.fullmatch(_grammar(4), _review('1,2,3,4'))
    for support in ('1,1', '3,1', '2,3,2', '4,3,2,1'):
        assert re.fullmatch(_grammar(4), _review(support)) is None
    assert semantics.parse_review('이유: 기존 검토 형식\n판정: OK') == 'ok'
    assert semantics.parse_review(_review('1,2,4,1')) == 'ok'


def test_actual_complete_review_passes_the_visible_count_grammar_to_decoder(monkeypatch):
    from trainer.demo_api.tests.test_query_quick_latency import _pipeline
    pipeline, _, _, chunks = _pipeline(monkeypatch)
    chunks[:] = [
        {'chunk_id': 'DOC-a-C0001', 'text': '도장 겹침은 균일한 두께를 형성합니다.', 'rerank_score': 8.0},
        {'chunk_id': 'DOC-b-C0001', 'text': '분사 이동 간격을 조절하여 누락을 방지합니다.', 'rerank_score': 7.9},
        {'chunk_id': 'DOC-c-C0001', 'text': '에어리스 도장의 패턴 가장자리를 중첩합니다.', 'rerank_score': 7.8},
    ]
    calls = []
    def generate(messages, **options):
        calls.append(options)
        if len(calls) == 1:
            return '[답변] 겹침 도장은 균일한 도막 형성을 돕습니다.\n[근거] DOC-a-C0001'
        return _review('1,3')
    pipeline._generate_direct = generate
    result = pipeline.run(instruction='겹침 도장을 하는 이유는?', answer_question='겹침 도장을 하는 이유는?',
                          force_route='fast', use_pipeline_history=False, concise_answer=True)
    assert len(result['metadata']['answer_trace']['selected_evidence']) == 3
    assert len(calls) == 2
    assert re.fullmatch(calls[1]['answer_regex'], _review('1,3'))
    assert re.fullmatch(calls[1]['answer_regex'], _review('4')) is None
    assert re.fullmatch(calls[1]['answer_regex'], _review('1,1')) is None
    assert result['citations'] == ['DOC-a-C0001', 'DOC-c-C0001']
