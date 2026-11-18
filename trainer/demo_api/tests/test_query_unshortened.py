"""Query keeps complete answer prose; the model mode does not impose a summary."""
from __future__ import annotations

import re

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.copilot_router import create_copilot_router
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.tests.test_voice_router import _Stt, _client, _post
from trainer.demo_api.tests.viewer_record import ask
from trainer.scripts.answer_grammar import quick_answer_regex
from trainer.scripts.agent_pipeline import AgentPipeline

BODY = (
    "도막을 고르게 형성하려면 표면 상태와 도료의 사용 조건을 함께 확인해야 합니다.\n"
    "- 표면에 남은 먼지와 오염을 제거하세요.\n"
    "- 도료 제조사가 지정한 혼합 조건을 확인하세요.\n"
    "마지막으로 작업 결과를 확인하고 필요한 부위를 보완하세요."
)


class _Documents:
    def __init__(self) -> None:
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "response": "[답변] " + BODY + "\n[근거] DOC-a-C0001, DOC-b-C0002",
            "rag_chunks": [
                {"chunk_id": "DOC-a-C0001", "text": BODY},
                {"chunk_id": "DOC-b-C0002", "text": BODY},
            ],
        }


@pytest.mark.parametrize("mode", ["quick", "detailed"])
def test_query_preserves_every_model_sentence_and_list_item(mode):
    pipeline = _Documents()
    app = FastAPI()
    app.include_router(create_copilot_router(lambda: pipeline, store=SequenceStore(":memory:")))
    response = TestClient(app).post("/api/v1/copilot/query", json={
        "user_id": "COPILOT-AI-TEST", "prompt": "도막을 고르게 형성하는 절차를 설명해 주세요.",
        "response_mode": mode, "summarize": False, "task_type": "knowledge_search", "use_rag": "Y",
    })
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["answer"] == BODY
    assert data["citations"] == ["DOC-a-C0001", "DOC-b-C0002"]
    assert 512 <= pipeline.calls[0]["max_tokens"] <= 1024
    instruction = pipeline.calls[0].get("generation_instruction", pipeline.calls[0]["instruction"])
    assert "60자" not in instruction and "2문장" not in instruction and "한 줄로 요약" not in instruction


@pytest.mark.parametrize("mode", ["quick", "detailed"])
def test_voice_query_preserves_the_same_complete_body(mode):
    pipeline = _Documents()
    response = _post(_client(_Stt("도막을 고르게 형성하는 절차를 설명해 주세요."), pipeline), response_mode=mode, summarize="false")
    assert response.status_code == 200
    assert response.json()["data"]["answer"] == BODY


@pytest.mark.parametrize("prompt,tab,required", [
    ("도막이 과하거나 모자란 영역이 있어?", "coating", ("저도막: 8개 —", "과도막: 없음", "Zone 9")),
    ("오른손은 어케 고쳐야돼", "pose", ("오른쪽 손목 교정은 없었습니다.", "20cm")),
    ("이 Zone 속도랑 이격거리 숙련자랑 비교해줘.", "knowledge", ("2,230 mm/s", "251 mm", "이전 장비")),
])
def test_quick_record_answers_keep_the_full_factual_explanation(prompt, tab, required):
    text = ask(prompt, tab=tab)["answer"]
    for phrase in required:
        assert phrase in text, text
    assert len(text) > 60


def test_quick_decode_grammar_preserves_multiline_prose_and_supported_citations():
    assert re.fullmatch(quick_answer_regex(has_evidence=True),
                        "[답변] " + BODY + "\n[근거] DOC-a-C0001, DOC-b-C0002")
    for invalid in ("[답변] </think>", "[답변] [원인] 추가 제목",
                    "[답변] \n[근거] DOC-a-C0001"):
        assert re.fullmatch(quick_answer_regex(has_evidence=True), invalid) is None


@pytest.mark.parametrize("mode", ["quick", "detailed"])
def test_conversation_has_room_for_a_complete_answer(mode):
    pipeline = object.__new__(AgentPipeline)
    completions = []
    def complete(messages, **kwargs):
        completions.append((messages, kwargs))
        return BODY
    pipeline._dialogue_completion = complete
    assert pipeline.converse_copilot("표면 준비 절차를 설명해 주세요.", recent_turns=[], facts={}, response_mode=mode) == BODY
    assert 512 <= completions[0][1]["max_tokens"] <= 1024
    instruction = completions[0][0][0]["content"]
    assert "60자" not in instruction and "두 번째 문장으로 요약" not in instruction
