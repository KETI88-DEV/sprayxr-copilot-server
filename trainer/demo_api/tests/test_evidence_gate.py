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


def test_without_a_configured_gate_chat_questions_are_not_redirected():
    pipeline = Pipeline(top=6.2, config={})
    data = ask(pipeline, "가사시간이 뭐지?")
    assert data["answer"] == "대화 경로의 답입니다." and pipeline.runs == []


def test_quick_mode_is_not_redirected_or_gated():
    pipeline = Pipeline(top=1.5)
    data = ask(pipeline, "가사시간이 뭐지?", mode="quick")
    assert data["answer"] == "대화 경로의 답입니다." and pipeline.runs == []


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
