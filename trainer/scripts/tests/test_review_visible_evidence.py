"""Every visible curated source is available to its semantic reviewer."""
import re

from trainer.scripts.answer_semantics import evidence_trace,review_answer,review_messages,review_regex


def test_the_sixth_visible_source_can_support_an_answer_and_its_real_citation():
    evidence=[{'chunk_id':f'DOC-source-C{i:04d}','text':f'작업 조건 {i}에 관한 문서입니다.'} for i in range(1,7)]
    evidence[5]['text']='조건 X는 현상 P의 직접적인 원인입니다.'
    grammar=review_regex(6)
    output='이유: 여섯째 문단에 직접 원인이 있음\n근거: 6\n판정: OK'
    assert re.fullmatch(grammar,output)
    messages=review_messages('P의 원인은?','','[답변] X가 원인입니다.',evidence,require_evidence=True)
    assert '[문단 6]' in messages[-1]['content']
    assert '조건 X는 현상 P의 직접적인 원인' in messages[-1]['content']
    checked=review_answer('P의 원인은?','','[답변] X가 원인입니다.',evidence,
        complete=lambda *_:output,regenerate=lambda *_:None,require_evidence=True,budget_seconds=5)
    assert checked.cacheable and checked.text.endswith('[근거] DOC-source-C0006')
    assert len(evidence_trace(evidence))==6


def test_support_numbers_above_the_visible_evidence_count_remain_forbidden():
    assert not re.fullmatch(review_regex(6),'이유: 범위 밖 번호\n근거: 7\n판정: OK')


def test_review_receives_the_whole_already_visible_passage_including_its_late_condition():
    import hashlib
    visible='적용 범위 설명입니다. '*150+'마지막 조건: 작업 조건 Z에서만 현상 P가 발생합니다.'
    evidence=[{'chunk_id':'DOC-visible-C0001','text':visible}]
    assert visible.index('마지막 조건')>1600
    messages=review_messages('P의 원인은?','','[답변] Z 조건에서만 발생합니다.',evidence,require_evidence=True)
    assert visible in messages[-1]['content']
    assert evidence_trace(evidence)[0]['sha256']==hashlib.sha256(visible.encode()).hexdigest()


def test_full_review_input_exceeding_model_context_is_withheld_before_the_rpc(monkeypatch):
    from types import SimpleNamespace
    from trainer.demo_api.tests.test_query_quick_latency import _pipeline
    pipeline,calls,_,chunks=_pipeline(monkeypatch)
    chunks[0].update(text='적용 조건입니다. '*200,rerank_score=8.0)
    pipeline._token_counter=SimpleNamespace(exact=True,count_text=lambda _text:100,
        count_messages=lambda messages,**_:9000 if '답변을 독립적으로 검증' in messages[0]['content'] else 100)
    result=pipeline.run(instruction='원인은?',answer_question='원인은?',force_route='fast',
        compact_analysis=True,use_pipeline_history=False,quick_cache_scope='full-review-budget')
    assert len(calls)==1
    assert not result['metadata']['answer_review']['cacheable']
    assert result['metadata']['answer_review']['contexts'][0]['prompt_tokens']==9000
    assert result['citations']==[]


def test_a_validated_partial_finding_remains_visible_after_its_repair_exhausts_the_budget():
    clock=[0.0]
    def complete(*_args):
        clock[0]=.5
        return '이유: 일부 원인만 자료에서 확인됨\n근거: 1\n판정: INSUFFICIENT'
    def regenerate(*_args):
        clock[0]=5.1
        return '[답변] 미승인 재생성 후보입니다.'
    result=review_answer('현상 P의 원인은?','','[답변] 최초 미승인 후보입니다.',
        [{'chunk_id':'DOC-one-C0001','text':'한 가지 원인'}],complete=complete,regenerate=regenerate,
        require_evidence=True,budget_seconds=5,clock=lambda:clock[0])
    assert result.verdict=='budget_exhausted' and not result.cacheable
    assert '미승인' not in result.text
    assert result.events[0]['verdict']=='insufficient'
    assert result.events[0]['support']==['DOC-one-C0001'] and result.events[0]['support_valid'] is True


def test_curated_passages_cannot_exceed_the_shared_rerank_evidence_limit():
    from types import SimpleNamespace
    from trainer.demo_api.tests.test_context_prompts import _pipeline
    p=_pipeline([{'chunk_id':f'C{i}','text':f'본문 {i}'} for i in range(10)])
    found=p._pin_evidence('원인은?',[f'C{i}' for i in range(10)],[],SimpleNamespace(EXCLUDED_CHUNKS=frozenset()))
    assert len(found)==4
