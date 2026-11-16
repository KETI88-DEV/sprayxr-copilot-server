"""근거 확인 단계와 지식 탭 질문의 대화 경로 보정(2026-11-13 2-1 측정에서 도입)."""
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.copilot_router import NO_EVIDENCE_ANSWER, create_copilot_router
from trainer.demo_api.sequence_store import SequenceStore

GATE = {"confidence": {"abstain_below_rerank_score": 4.0}}


class Pipeline:
    """계획 모델은 항상 "chat"으로 본다(용어 질문을 대화로 잘못 보내던 상황). 검색 1위 점수는 top으로 정한다."""

    def __init__(self, *, top, config=GATE):
        self.top, self.config = top, config
        self.runs, self.conversations = [], []

    def plan_copilot_dialogue(self, prompt, **kwargs):
        return {"kind": "chat", "intent": "knowledge", "scope": "unspecified"}

    def converse_copilot(self, prompt, **kwargs):
        self.conversations.append(prompt)
        return "대화 경로의 답입니다."

    def run(self, **kwargs):
        self.runs.append(kwargs)
        return {"response": "[답변] 가사시간은 혼합 후 사용할 수 있는 시간이다.\n[근거] DOC-doc_0001-C0001",
                "rag_chunks": [{"chunk_id": "DOC-doc_0001-C0001", "text": "가사시간", "rerank_score": self.top}]}


def ask(pipeline, prompt, mode="detailed"):
    app = FastAPI()
    app.include_router(create_copilot_router(lambda: pipeline, store=SequenceStore(":memory:")))
    body = {"user_id": "COPILOT-AI-TEST", "task_type": "knowledge_search", "prompt": prompt, "response_mode": mode}
    return TestClient(app).post("/api/v1/copilot/query", json=body).json()["data"]


def test_a_term_question_the_planner_calls_chat_is_answered_from_documents_when_evidence_is_strong():
    pipeline = Pipeline(top=6.2)
    data = ask(pipeline, "가사시간이 뭐지?")
    assert data["retrieval_used"] is True and data["citations"] == ["DOC-doc_0001-C0001"]
    assert "혼합 후" in data["answer"]
    assert len(pipeline.runs) == 1 and pipeline.conversations == []


def test_a_redirected_question_with_weak_evidence_returns_to_the_conversation_path():
    pipeline = Pipeline(top=1.5)
    data = ask(pipeline, "BTS 멤버는 몇 명이야?")
    assert data["answer"] == "대화 경로의 답입니다." and data["retrieval_used"] is False
    assert len(pipeline.runs) == 1 and pipeline.conversations == ["BTS 멤버는 몇 명이야?"]


def test_greetings_still_go_straight_to_the_conversation_path():
    pipeline = Pipeline(top=6.2)
    data = ask(pipeline, "안녕")
    assert data["answer"] == "대화 경로의 답입니다." and pipeline.runs == []


def test_without_a_gate_domain_questions_still_use_documents_and_others_chat():
    pipeline = Pipeline(top=6.2, config={})
    ask(pipeline, "가사시간이 뭐지?")
    assert len(pipeline.runs) == 1 and pipeline.conversations == []
    pipeline = Pipeline(top=6.2, config={})
    data = ask(pipeline, "BTS 멤버는 몇 명이야?")
    assert data["answer"] == "대화 경로의 답입니다." and pipeline.runs == []


QUICK = {"confidence": {"abstain_below_rerank_score": 4.0, "quick_abstain_below_rerank_score": 2.0}}


def test_quick_domain_question_uses_documents_and_its_own_gate():
    assert ask(Pipeline(top=1.5, config=QUICK), "가사시간이 뭐지?", mode="quick")["answer"] == NO_EVIDENCE_ANSWER
    assert ask(Pipeline(top=2.5, config=QUICK), "가사시간이 뭐지?", mode="quick")["citations"] == ["DOC-doc_0001-C0001"]
    assert "혼합 후" in ask(Pipeline(top=1.5), "가사시간이 뭐지?", mode="quick")["answer"]


class KnowledgePipeline(Pipeline):
    """계획 모델이 지식 질문으로 맞게 분류한 경우."""

    def plan_copilot_dialogue(self, prompt, **kwargs):
        return {"kind": "coating_knowledge", "intent": "knowledge", "scope": "unspecified"}


def test_weak_evidence_withholds_the_generated_knowledge_answer():
    pipeline = KnowledgePipeline(top=3.9)
    data = ask(pipeline, "인트럼센트 도료의 팽창 배율은?")
    assert data["answer"] == NO_EVIDENCE_ANSWER
    assert data["citations"] == [] and data["retrieval_used"] is True
    assert pipeline.conversations == []


def test_strong_evidence_keeps_the_knowledge_answer_and_a_missing_gate_changes_nothing():
    assert ask(KnowledgePipeline(top=4.0), "가사시간이 뭐지?")["citations"] == ["DOC-doc_0001-C0001"]
    assert ask(KnowledgePipeline(top=0.1, config={}), "가사시간이 뭐지?")["citations"] == ["DOC-doc_0001-C0001"]
    bad = {"confidence": {"abstain_below_rerank_score": "4.0"}}
    assert ask(KnowledgePipeline(top=0.1, config=bad), "가사시간이 뭐지?")["citations"] == ["DOC-doc_0001-C0001"]


class HedgePipeline(KnowledgePipeline):
    """근거가 약할 때 대화 모델이 일반 지식으로 답하는 경우(v7 평가 Z005: "헤그만 눈금 6 = 6µm")."""

    def __init__(self, *, top, reply, kind="coating_knowledge"):
        super().__init__(top=top)
        self.reply, self.kind = reply, kind

    def plan_copilot_dialogue(self, prompt, **kwargs):
        return {"kind": self.kind, "intent": "knowledge", "scope": "unspecified"}

    def converse_copilot(self, prompt, **kwargs):
        self.conversations.append(prompt)
        return self.reply


def test_a_weak_evidence_definition_question_gets_a_marked_general_answer_without_numbers():
    pipeline = HedgePipeline(top=1.5, reply="[분석] 새깅은 도료가 흘러내려 처지는 현상입니다. 보통 150µm 이상에서 생깁니다.")
    data = ask(pipeline, "새깅이 뭐지?")
    assert data["answer"] == "제공된 자료에서는 근거를 찾지 못해 일반적인 설명으로 답합니다. 새깅은 도료가 흘러내려 처지는 현상입니다."
    assert data["citations"] == []


def test_weak_evidence_technical_questions_that_ask_for_values_or_causes_still_abstain():
    for prompt, kind in (("헤그만 게이지 눈금 6은 입자 크기가 몇 µm를 뜻하나요?", "chat"),  # 도메인 낱말 없이 대화로 분류됨
                         ("크실렌의 LD50 값은 얼마인가요?", "chat"),
                         ("과도막의 원인은?", "coating_knowledge"),
                         ("인트럼센트 도료의 팽창 배율은?", "coating_knowledge"),
                         ("SVR이 뭐야?", "chat"),  # Spark 실측: 고형분률 대신 Support Vector Regression으로 답함
                         ("드라이아이스 블라스팅으로 도막을 제거하는 조건은 무엇인가요?", "chat")):
        pipeline = HedgePipeline(top=1.5, reply="지어낸 답입니다.", kind=kind)
        assert ask(pipeline, prompt)["answer"] == NO_EVIDENCE_ANSWER, prompt
        assert pipeline.conversations == [], prompt


def test_a_general_answer_with_only_numbers_left_abstains():
    pipeline = HedgePipeline(top=1.5, reply="두께 150µm 기준입니다.")
    assert ask(pipeline, "새깅이 뭐지?")["answer"] == NO_EVIDENCE_ANSWER
