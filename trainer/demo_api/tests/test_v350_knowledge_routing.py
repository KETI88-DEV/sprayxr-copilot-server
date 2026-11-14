from trainer.demo_api.copilot_router import DIZZY_NOTE, NO_EVIDENCE_ANSWER
from trainer.demo_api.tests.test_evidence_gate import Pipeline, ask

QUICK = {"confidence": {"abstain_below_rerank_score": 4.0, "quick_abstain_below_rerank_score": 2.0}}


def test_dizzy_question_without_evidence_gets_the_fixed_line() -> None:   # T16
    data = ask(Pipeline(top=1.0, config=QUICK), "VR로 연습하다가 어지러우면 몇 분 쉬어야 해요?", mode="quick")
    assert data["answer"] == NO_EVIDENCE_ANSWER + "\n" + DIZZY_NOTE


def test_domain_question_with_weak_evidence_does_not_fall_back_to_chat() -> None:
    pipeline = Pipeline(top=1.5)
    data = ask(pipeline, "가사시간이 뭐지?")
    assert data["answer"] == NO_EVIDENCE_ANSWER and pipeline.conversations == []


def test_public_knowledge_questions_planned_as_chat_use_documents_in_quick_mode() -> None:   # T12, T17, T19
    for prompt in ("가사 시간이 지난 도료 아까운데 그냥 써도 돼요?", "금속 도장 기능사 실기는 몇 점 이상이면 합격이에요?",
                   "에어리스 팁 517은 무슨 뜻이에요?"):
        pipeline = Pipeline(top=6.2, config=QUICK)
        data = ask(pipeline, prompt, mode="quick")
        assert data["retrieval_used"] is True and pipeline.conversations == [], prompt


class ClarifyPipeline(Pipeline):
    def plan_copilot_dialogue(self, prompt, **kwargs):
        return {"kind": "clarify", "intent": "knowledge", "scope": "unspecified"}

    def converse_copilot(self, prompt, **kwargs):
        return "5~10분 쉬세요."


def test_domain_question_that_reaches_chat_drops_invented_minutes() -> None:
    data = ask(ClarifyPipeline(top=6.2, config=QUICK), "VR 어지러우면 몇 분 쉬어?", mode="quick")
    assert "5~10분" not in data["answer"]
