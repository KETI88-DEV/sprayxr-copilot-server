"""A fresh general answer is reviewed against its own information contract."""
import pytest

from trainer.scripts.answer_semantics import review_answer, review_messages


def test_no_document_review_has_no_document_comparison_requirement():
    messages = review_messages("일반적인 원인은?", "", "[답변] 가능한 원인을 설명합니다.", [],
                               require_evidence=False)
    system, user = [item["content"] for item in messages]
    for rag_requirement in ("질문·원문", "각 문단", "근거와 일치", "어느 부분도 문서", "후보와 원문"):
        assert rag_requirement not in system
    assert "문서 부재만으로 INSUFFICIENT" in system
    assert "근거는 항상 없음" in system
    assert "[참고 문서]" not in user


def test_no_document_review_preserves_general_answer_and_honest_limitation_contracts():
    system = review_messages("질문", "", "후보", [], require_evidence=False)[0]["content"]
    for guard in ("가능한 원인", "문서만", "비공개", "실시간", "현재 실측", "임의 수치", "조건", "이전 AI"):
        assert guard in system
    assert "한계 설명은 OK" in system


def test_document_review_still_requires_direct_support_and_causal_subject_matching():
    system = review_messages("원인은?", "", "후보", [], require_evidence=True)[0]["content"]
    assert "원문의 주어" in system
    assert "각 문단" in system
    assert "어느 부분도 문서" in system
    assert "학습 목표" in system


@pytest.mark.parametrize("candidate", [
    "[답변] 일반적으로 가능한 원인은 A입니다. 현재 작업에서 발생한 원인인지는 확인이 필요합니다.",
    "[답변] 요청한 비공개 측정 기록이 제공되지 않아 확인할 수 없습니다. 기록을 제공해 주세요.",
])
def test_independently_approved_no_document_body_can_be_returned_without_rag(candidate):
    result = review_answer("질문", "", candidate, [], require_evidence=False, budget_seconds=3,
        complete=lambda *_: "이유: 요구와 정보 한계에 부합\n근거: 없음\n판정: OK",
        regenerate=lambda *_: pytest.fail("approved candidate needs no repair"))
    assert result.cacheable and result.text == candidate and "[근거]" not in result.text


def test_source_rebinding_removes_stale_inline_ids_before_public_citation_parsing():
    from trainer.scripts.live_response_contract import parse_knowledge_answer
    evidence = [{"chunk_id":"DOC-wrong-C0001", "text":"다른 조건의 문단"},
                {"chunk_id":"DOC-valid-C0001", "text":"조건 A가 현상 P를 유발한다."}]
    result = review_answer("P의 원인은?", "", "[답변] 원인은 A입니다. (DOC-wrong-C0001)\n[근거] DOC-wrong-C0001",
        evidence, require_evidence=True, budget_seconds=3,
        complete=lambda *_: "이유: 둘째 문단이 원인을 지지함\n근거: 2\n판정: OK",
        regenerate=lambda *_: pytest.fail("supported body needs attribution correction only"))
    published = parse_knowledge_answer(result.text, [item["chunk_id"] for item in evidence])
    assert published.citations == ("DOC-valid-C0001",)
    assert "DOC-wrong" not in published.body and "()" not in published.body
    assert "원인은 A입니다." in published.body


def test_knowledge_fallback_keeps_its_training_domain_without_restricting_general_conversation():
    from trainer.demo_api.dialogue import conversation_messages
    options=dict(recent_turns=[],facts={},response_mode='detailed')
    knowledge=conversation_messages('자세 종류로는 어떤 게 있어요?',knowledge_fallback=True,**options)[0]['content']
    general=conversation_messages('프랑스의 수도는?',knowledge_fallback=False,**options)[0]['content']
    assert '선박 도장·표면 처리 훈련' in knowledge
    assert knowledge.startswith('사용자와 자연스럽게 대화하는 선박 도장·표면 처리 훈련 코치입니다.')
    assert '공식 분류' in knowledge and '모호하면' in knowledge
    assert '선박 도장·표면 처리 훈련' not in general


def test_knowledge_generation_uses_its_own_policy_instead_of_record_playback_instructions():
    from trainer.demo_api.dialogue import conversation_messages
    system=conversation_messages('질문',recent_turns=[],facts={},response_mode='detailed',knowledge_fallback=True)[0]['content']
    for playback_key in ('record_direction','selected_sequence','playback_direction','selected_zone.direction'):
        assert playback_key not in system
    assert 'verified_facts=' in system
    assert '임의' in system and '수치' in system and '[답변]' in system


def test_fresh_knowledge_generation_and_review_receive_the_same_task_context():
    import re
    from trainer.demo_api.tests.test_dialogue_pipeline import pipeline as real_pipeline
    pipeline=real_pipeline();seen=[]
    def generate(messages,**options):
        seen.append(messages)
        if re.fullmatch(options.get('answer_regex',''),'이유: 검토\n근거: 없음\n판정: OK'):
            return '이유: 작업 맥락을 구분함\n근거: 없음\n판정: OK'
        return '[답변] 어떤 도장 작업면과 장비에 관한 자세인지 알려 주시겠어요?'
    pipeline._generate_direct=generate
    pipeline.converse_copilot('자세 종류로는 어떤 게 있어요?',recent_turns=[],facts={},
        response_mode='detailed',knowledge_fallback=True,inference_context_limit=8192)
    assert len(seen)==2
    assert '선박 도장·표면 처리 훈련' in seen[0][0]['content']
    assert '선박 도장·표면 처리 훈련' in seen[1][-1]['content']
    assert '[현재 질문]\n자세 종류로는 어떤 게 있어요?' in seen[1][-1]['content']


def test_transport_timeout_retries_the_same_candidate_once_without_regeneration():
    clock=[0.0];requests=[]
    def complete(messages,timeout):
        requests.append((messages,timeout))
        if len(requests)==1:
            clock[0]=8.0;raise TimeoutError('review transport')
        clock[0]=12.0
        return '이유: 원문과 일치함\n근거: 1\n판정: OK'
    result=review_answer('원인은?','','[답변] 원인은 A입니다.',
        [{'chunk_id':'DOC-one-C0001','text':'A가 원인이다.'}],complete=complete,
        regenerate=lambda *_:pytest.fail('transport failure must not regenerate the answer'),
        require_evidence=True,budget_seconds=25,review_timeout_seconds=8,clock=lambda:clock[0],
        review_transport_retries=1,retryable_review_error=lambda exc:isinstance(exc,TimeoutError))
    assert result.cacheable and result.calls==2 and result.retries==0
    assert requests[0][0]==requests[1][0] and [row[1] for row in requests]==[8,8]


def test_review_transport_retry_stays_within_the_remaining_phase_budget():
    clock=[0.0];limits=[]
    def complete(_messages,timeout):
        limits.append(timeout)
        if len(limits)==1:
            clock[0]=24.8;raise TimeoutError('late transport')
        clock[0]=25.01
        return '이유: 늦은 결과\n근거: 1\n판정: OK'
    result=review_answer('원인은?','','[답변] 원인은 A입니다.',
        [{'chunk_id':'DOC-one-C0001','text':'A가 원인이다.'}],complete=complete,
        regenerate=lambda *_:pytest.fail('no repair budget'),require_evidence=True,budget_seconds=25,
        review_timeout_seconds=8,clock=lambda:clock[0],review_transport_retries=1,
        retryable_review_error=lambda exc:isinstance(exc,TimeoutError))
    assert limits==pytest.approx([8,.2])
    assert not result.cacheable and result.verdict=='budget_exhausted'


@pytest.mark.parametrize('retryable', [False,True])
def test_other_errors_and_a_second_transport_timeout_never_approve_a_candidate(retryable):
    calls=[]
    def complete(*_):calls.append(1);raise TimeoutError('still unavailable')
    result=review_answer('원인은?','','[답변] 미검증 후보입니다.',[],complete=complete,
        regenerate=lambda *_:pytest.fail('error must not trigger semantic regeneration'),require_evidence=False,
        budget_seconds=25,review_transport_retries=1,retryable_review_error=lambda _exc:retryable)
    assert len(calls)==(2 if retryable else 1) and not result.cacheable
    assert '미검증 후보' not in result.text
