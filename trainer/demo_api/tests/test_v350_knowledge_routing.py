from trainer.demo_api.copilot_router import DIZZY_NOTE, NO_EVIDENCE_ANSWER
from trainer.demo_api.tests.test_evidence_gate import Pipeline, ask

QUICK = {"confidence": {"abstain_below_rerank_score": 4.0, "quick_abstain_below_rerank_score": 2.0}}


def test_dizzy_question_without_evidence_gets_the_fixed_line() -> None:   # T16
    data = ask(Pipeline(top=1.0, config=QUICK), "VR로 연습하다가 어지러우면 몇 분 쉬어야 해요?", mode="quick")
    assert data["answer"] == DIZZY_NOTE + "\n" + NO_EVIDENCE_ANSWER


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


def test_no_record_follow_ups_and_record_style_questions_split_documents_from_session_chat() -> None:
    docs = ("그럼 가사 시간은 얼마야?", "그럼 에어리스 팁 517은 무슨 뜻이야?", "나의 VR 어지럼 기준은?",
            "그 도료 가사시간은 몇 시간이야?", "저의 도장 기능사 합격 점수는?")
    chat = ("칠한 도막 두께가 몇 마이크로미터야?", "분사 속도가 몇으로 나왔어?",
            "각도가 정상이라고 나왔는데 맞아?", "도장 끝났는데 점수가 몇 점이야?",
            "내 속도는 얼마야?", "제 이격거리는 얼마였어?", "내 속도는 뭐였지?", "이번 도막은 얼마나 나왔어?",
            "방금 도막 두께 얼마야?", "내 도막 두께 얼마야?")
    for prompt in docs:
        pipeline = Pipeline(top=6.2, config=QUICK)
        ask(pipeline, prompt, mode="quick")
        assert len(pipeline.runs) == 1 and pipeline.conversations == [], prompt
    for prompt in chat:
        pipeline = Pipeline(top=6.2, config={})
        ask(pipeline, prompt, mode="quick")
        assert pipeline.runs == [], prompt  # 문서로 가지 않는다(대화 또는 기록 답)
