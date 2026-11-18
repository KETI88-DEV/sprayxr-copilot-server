"""Document misses preserve meaning while separating model knowledge from sources."""
import pytest
import re

from trainer.demo_api.tests.test_query_evidence_fallback import Pipeline, ask, client
from trainer.demo_api.tests.test_dialogue_pipeline import pipeline as real_pipeline


@pytest.mark.parametrize("mode", ["quick", "detailed"])
@pytest.mark.parametrize("missing,verdict", [(True, "not_required"), (False, "insufficient")])
def test_knowledge_fallback_declares_its_limit_and_cannot_publish_model_citations(mode, missing, verdict):
    class InventedCitation(Pipeline):
        def converse_copilot(self, prompt, **kwargs):
            self.conversations.append({"prompt": prompt, **kwargs})
            return "[답변] 가능한 작업 조건을 확인해야 합니다.\n[근거] DOC-invented-C0001"
    p = InventedCitation(missing=missing, verdict=verdict)
    response = ask(client(p), mode=mode)
    data = response.json()["data"]
    assert response.status_code == 200 and data["degraded"] is False
    assert p.conversations[0]["knowledge_fallback"] is True
    assert "일반" in data["answer"] and "문서" in data["answer"] and "확인되지" in data["answer"]
    assert "DOC-" not in data["answer"] and "[근거]" not in data["answer"]
    assert data["citations"] == [] and data["retrieval_used"] is False
    assert response.headers["X-Copilot-Answer-Source"] == "conversation"


@pytest.mark.parametrize("summary", [True, False])
@pytest.mark.parametrize("expression", ["2**3", "11//4", "a != b"])
@pytest.mark.parametrize("mode", ["quick", "detailed"])
def test_an_expression_in_the_model_answer_survives_query_presentation(summary, expression, mode):
    class Chat(Pipeline):
        def plan_copilot_dialogue(self, *_args, **_kwargs):
            return {"kind": "chat", "intent": "knowledge", "scope": "unspecified"}
        def converse_copilot(self, *_args, **_kwargs):
            return f"`{expression}`는 연산을 나타냅니다."
    response = ask(client(Chat()), prompt=f"`{expression}` 표기는 그대로 유지하고 의미를 설명해 주세요.", summarize=summary, mode=mode)
    data = response.json()["data"]
    assert response.status_code == 200 and data["answer"] == f"`{expression}`는 연산을 나타냅니다."
    assert data["citations"] == [] and data["retrieval_used"] is False


def test_fallback_semantic_repair_does_not_certify_a_second_bad_explanation():
    p = real_pipeline()
    calls = []
    def generate(messages, **options):
        calls.append((messages, options))
        if options.get("answer_regex") and re.fullmatch(options['answer_regex'],'이유: 검토\n근거: 없음\n판정: OK'):
            return "이유: 질문의 대상과 다른 설명\n근거: 없음\n판정: REWRITE"
        return "[답변] 반복 도장으로 두꺼운 도막을 만드는 방법입니다."
    p._generate_direct = generate
    with pytest.raises(ValueError, match="could not be verified: rejected"):
        p.converse_copilot("분사 패스 겹침의 목적과 방법은?", recent_turns=[], facts={}, response_mode="detailed",
                          knowledge_fallback=True, inference_context_limit=8192)
    assert len(calls) == 4  # first generation, review, one repair, second review


def test_fallback_review_accepts_a_general_explanation_without_a_document_claim():
    p = real_pipeline()
    outputs = iter(["[답변] 같은 패스의 경계를 겹쳐 도료가 고르게 쌓이도록 합니다.",
                    "이유: 질문 대상과 방법을 설명함\n근거: 없음\n판정: OK"])
    p._generate_direct = lambda *_args, **_kwargs: next(outputs)
    answer = p.converse_copilot("분사 패스 겹침은 왜 하나요?", recent_turns=[], facts={}, response_mode="quick",
                               knowledge_fallback=True, inference_context_limit=8192)
    assert "경계를 겹쳐" in answer and "[근거]" not in answer


def test_a_citation_only_fallback_cannot_be_approved_by_a_judge():
    p = real_pipeline()
    calls = []
    def generate(_messages, **options):
        calls.append(options)
        if options.get("answer_regex") and re.fullmatch(options['answer_regex'],'이유: 검토\n근거: 없음\n판정: OK'):
            return "이유: 설명이 적절함\n근거: 없음\n판정: OK"
        return "[답변]\n[근거] DOC-invented-C0001"
    p._generate_direct = generate
    with pytest.raises(ValueError, match="could not be verified"):
        p.converse_copilot("현상 P의 원인은?", recent_turns=[], facts={}, response_mode="detailed",
                          knowledge_fallback=True, inference_context_limit=8192)
    assert len(calls) == 2  # malformed first body and one repair; a judge cannot approve either


@pytest.mark.parametrize("prompt", [
    "`2**3` 표기는 유지하지 말고 계산값만 알려 주세요.",
    "`2**3` 표기는 그대로 유지하지 말고 계산값만 알려 주세요.",
    "`2**3` 표기를 보존할 필요 없이 계산값만 알려 주세요.",
    "`2**3`을 그대로 두지 말고 결과만 알려 주세요.",
    "Do not preserve `2**3`; give the result only.",
    "Don't keep `2**3` unchanged; give the result only.",
    "`2**3` 표기를 유지해야 하나요?",
    "`2**3` 표기를 그대로 유지할지 말지 설명해 주세요.",
    "`2**3`의 의미를 설명하되 존댓말은 유지해 주세요.",
    "`2**3`과 `11//4` 중 표기를 유지할 것은 무엇인가요?",
    "'`2**3` 표기는 그대로 유지하고'라는 문장을 번역해 주세요.",
    "What does the literal `2**3` mean?",
    "Preserve `2**3`?",
    "Keep `2**3` unchanged only if needed.",
    "'`2**3` 표기를 유지해 주세요.'라는 문장입니다.",
    "`2**3` 표기를 유지해 주지 마세요.",
    "Preserve `2**3` and do not preserve it in the answer.",
    "Keep `2**3` unchanged and omit the expression from the answer.",
    "`2**3` 표기는 유지해 주시면 안 됩니다. 계산 결과만 알려 주세요.",
])
def test_negated_or_ambiguous_preservation_does_not_override_the_model_answer(prompt):
    class Chat(Pipeline):
        def plan_copilot_dialogue(self, *_args, **_kwargs):
            return {"kind": "chat", "intent": "knowledge", "scope": "unspecified"}
        def converse_copilot(self, *_args, **_kwargs):
            return "8입니다."
    response = ask(client(Chat()), prompt=prompt)
    assert response.status_code == 200 and response.json()["data"]["answer"] == "8입니다."


@pytest.mark.parametrize("prompt", [
    "Please keep `2**3` unchanged and explain its meaning.",
    "Preserve `2**3`.",
    "`2**3` 표기를 보존해 주세요.",
])
def test_a_model_omission_is_not_hidden_by_an_automatic_expression_prefix(prompt):
    class Chat(Pipeline):
        def plan_copilot_dialogue(self, *_args, **_kwargs):
            return {"kind": "chat", "intent": "knowledge", "scope": "unspecified"}
        def converse_copilot(self, *_args, **_kwargs):
            return "연산을 설명합니다."
    response = ask(client(Chat()), prompt=prompt)
    assert response.status_code == 200 and response.json()["data"]["answer"] == "연산을 설명합니다."


@pytest.mark.parametrize("mode", ["quick", "detailed"])
@pytest.mark.parametrize("summary", [True, False])
def test_a_negative_literal_request_retains_the_generated_result_in_both_modes(mode, summary):
    class Chat(Pipeline):
        def plan_copilot_dialogue(self, *_args, **_kwargs):
            return {"kind": "chat", "intent": "knowledge", "scope": "unspecified"}
        def converse_copilot(self, *_args, **_kwargs):
            return "8입니다."
    response = ask(client(Chat()), prompt="`2**3` 표기는 유지해 주시면 안 됩니다. 계산 결과만 알려 주세요.", mode=mode, summarize=summary)
    assert response.status_code == 200 and response.json()["data"]["answer"] == "8입니다."


def test_fresh_knowledge_generation_uses_the_existing_no_document_answer_contract():
    p=real_pipeline();calls=[]
    def generate(_messages,**options):
        calls.append(options)
        if options.get('answer_regex') and re.fullmatch(options['answer_regex'],'이유: 설명 일치\n근거: 없음\n판정: OK'):
            return '이유: 설명 일치\n근거: 없음\n판정: OK'
        return '[답변] 작업 조건이 달라지면 결과도 달라질 수 있습니다.'
    p._generate_direct=generate
    p.converse_copilot('발생 원인을 일반적으로 설명해 주세요.',recent_turns=[],facts={},response_mode='detailed',knowledge_fallback=True)
    assert calls[0].get('answer_regex') is not None
    assert re.fullmatch(calls[0]['answer_regex'],'[답변] 원인을 설명합니다.')
    assert not re.fullmatch(calls[0]['answer_regex'],'[분석] 반대 현상입니다.\n[조치] 조절하세요.')


def test_diagnostic_scaffolds_are_not_approved_in_a_fresh_knowledge_answer():
    p=real_pipeline()
    def generate(_messages,**options):
        if options.get('answer_regex') and re.fullmatch(options['answer_regex'],'이유: 설명 일치\n근거: 없음\n판정: OK'):
            return '이유: 설명 일치\n근거: 없음\n판정: OK'
        return '[분석] 다른 현상을 설명합니다.\n[조치] 조절하세요.'
    p._generate_direct=generate
    with pytest.raises(ValueError,match='could not be verified'):
        p.converse_copilot('발생 원인을 설명해 주세요.',recent_turns=[],facts={},response_mode='detailed',knowledge_fallback=True)
