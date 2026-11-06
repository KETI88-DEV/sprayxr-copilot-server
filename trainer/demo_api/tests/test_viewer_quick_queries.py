"""Viewer quick answers retain facts without redundant session history."""
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.copilot_router import create_copilot_router
from trainer.demo_api.copilot_session_context import viewer_definition_context
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.tests.test_query_quick_latency import _pipeline


def _context(index=0):
    return "motion_viewer_playback:" + json.dumps({
        "record": "전형기 · 스프레이 · 학습모드 4 · 4_1000768_1",
        "playback_context": {"map_num": "1", "zone_id": "3", "source_index": index, "playhead_seconds": index / 3},
    }, ensure_ascii=False)


def test_only_standalone_definitions_can_ignore_viewer_playback_time():
    first = viewer_definition_context("저도막이 뭐야?", _context())
    assert first is not None
    assert first == viewer_definition_context("저도막이 뭐야?", _context(150))
    assert "전형기" in first and "playback_context" not in first


@pytest.mark.parametrize("question", ["지금 저도막이야?", "내 도막두께는?", "저도막이면 어떻게 해?", "그럼 왜?", "적정 이격거리는?"])
def test_context_sensitive_questions_keep_their_complete_context(question):
    assert viewer_definition_context(question, _context()) is None


def test_unrecognized_or_nonviewer_context_is_never_discarded():
    assert viewer_definition_context("저도막이 뭐야?", "현재 도막은 100 µm") is None
    assert viewer_definition_context("저도막이 뭐야?", "motion_viewer_playback:invalid") is None
    payload = json.loads(_context().split(":", 1)[1])
    payload["measurement"] = 100
    assert viewer_definition_context("저도막이 뭐야?", "motion_viewer_playback:" + json.dumps(payload)) is None
    payload = json.loads(_context().split(":", 1)[1])
    payload["playback_context"]["measurement"] = 100
    assert viewer_definition_context("저도막이 뭐야?", "motion_viewer_playback:" + json.dumps(payload)) is None


def test_api_uses_authoritative_history_once_and_definition_repeats_have_stable_instructions():
    class Pipeline:
        def __init__(self): self.calls = []
        def run(self, **kwargs):
            self.calls.append(kwargs)
            return {"response": "[답변] 저도막은 기준보다 얇은 도막입니다.", "rag_chunks": []}
    pipeline = Pipeline()
    store = SequenceStore(":memory:")
    store.record_query_turn(session_key="4_1000768_1", user_id="expert", prompt="예전 질문", answer="이전 답변", task_type="knowledge_search", created_at="2026-11-06T01:00:00Z")
    app = FastAPI()
    app.include_router(create_copilot_router(lambda: pipeline, store=store))
    client = TestClient(app)
    body = {"user_id": "expert", "session_id": "4_1000768_1", "task_type": "knowledge_search", "response_mode": "quick", "prompt": "저도막이 뭐야?"}
    for index in (0, 150):
        response = client.post("/api/v1/copilot/query", json={**body, "context": _context(index)})
        assert response.status_code == 200
        assert response.json()["data"]["session_id"] == body["session_id"]
    assert pipeline.calls[0]["instruction"] == pipeline.calls[1]["instruction"]
    assert "이전 답변" not in pipeline.calls[0]["instruction"]
    assert pipeline.calls[0]["concise_answer"] is True
    assert pipeline.calls[0]["use_pipeline_history"] is False
    assert "정의 질문은 뜻만 한 문장" in pipeline.calls[0]["generation_instruction"]
    # A real follow-up still sees the server-owned dialogue and current playback.
    client.post("/api/v1/copilot/query", json={**body, "prompt": "그럼 왜?", "context": _context(200)})
    assert "저도막은 기준보다 얇은 도막" in pipeline.calls[-1]["instruction"]
    assert "source_index" in pipeline.calls[-1]["instruction"]
    assert pipeline.calls[-1]["generation_instruction"].endswith("그럼 왜?")
    assert pipeline.calls[-1]["generation_instruction"].count("그럼 왜?") == 1
    # Detailed mode keeps existing behavior.
    client.post("/api/v1/copilot/query", json={**body, "response_mode": "detailed", "context": _context()})
    assert "이전 답변" in pipeline.calls[-1]["instruction"] or "최근 대화" in pipeline.calls[-1]["instruction"]
    assert not pipeline.calls[-1].get("concise_answer", False)


def test_concise_generation_keeps_search_facts_budget_and_uses_only_server_owned_history(monkeypatch):
    pipe, generated, searched, chunks = _pipeline(monkeypatch)
    pipe.config["prompt"] = {"fast_system": "근거 없는 측정값은 추측하지 마세요."}
    pipe.session_context.update("s", {"instruction": "중복 질문", "response": "중복 내부 대화"})
    instruction = "[현재 세션 누적 요약] 이격거리 300 mm\n[최근 대화] 실제 앞선 질문"
    kwargs = dict(instruction=instruction, session_id="s", force_route="fast", max_tokens=128,
                  concise_answer=True, use_pipeline_history=False, quick_cache_scope="user")
    pipe.run(**kwargs)
    result = pipe.run(**kwargs)
    assert len(generated) == 1 and result["metadata"]["quick_cache_hit"]
    messages, options = generated[0]
    assert "근거 없는 측정값은 추측하지 마세요." in messages[0]["content"]
    assert "정의" in messages[0]["content"] and "한 문장" in messages[0]["content"]
    assert "중복 내부 대화" not in messages[1]["content"]
    assert "실제 앞선 질문" in messages[1]["content"]
    assert "300 mm" in messages[1]["content"]
    assert options["max_tokens"] == 128 and searched == [instruction, instruction]
    pipe.run(**dict(kwargs, instruction=instruction.replace("300 mm", "350 mm")))
    assert len(generated) == 2


def test_generation_style_does_not_change_the_retrieval_query(monkeypatch):
    pipe, generated, searched, chunks = _pipeline(monkeypatch)
    pipe.run(instruction="기존 검색 문맥", generation_instruction="정의만 한 문장으로 답하세요.", force_route="fast")
    assert searched == ["기존 검색 문맥"]
    assert "정의만 한 문장으로" in generated[0][0][1]["content"]
