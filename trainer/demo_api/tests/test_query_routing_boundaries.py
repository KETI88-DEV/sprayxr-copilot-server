"""A requested action or new topic is not a prior Zone-order lookup."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.copilot_router import create_copilot_router
from trainer.demo_api.reference_order_query import is_current_next_question, is_reference_order_question
from trainer.demo_api.sequence_analysis import SequenceAnalysisEngine
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.tests.test_query_contextual_dialogue import DialoguePipeline
from trainer.demo_api.tests.test_scenario_reference_query import profile

USER_ID = "COPILOT-AI-TEST"
SESSION_ID = "query-boundary"
PREVIOUS_PROMPT = "권장 Zone 순서를 알려줘."
HEADERS = {"X-Copilot-Source": "assistant-test"}


def seed_turn(store: SequenceStore, *, session_id: str | None = None) -> None:
    store.record_query_turn(
        session_key=session_id or f"CHAT:{USER_ID}",
        user_id=USER_ID,
        prompt=PREVIOUS_PROMPT,
        answer="Zone 2 → 3 → 1입니다.",
        task_type="trajectory_recomm",
        created_at="2026-11-24T11:00:00Z",
        answer_source="analysis",
    )


def ask(pipeline: DialoguePipeline, store: SequenceStore, prompt: str, **options):
    session_id = options.pop("session_id", None)
    app = FastAPI()
    app.include_router(create_copilot_router(lambda: pipeline, store=store, **options))
    response = TestClient(app).post("/api/v1/copilot/query", headers=HEADERS, json={
        "user_id": USER_ID, "session_id": session_id, "task_type": "knowledge_search", "prompt": prompt,
    })
    assert response.status_code == 200, response.text
    return response.json()["data"]


@pytest.mark.parametrize("prompt", [
    "다음 구역의 목표 도막 두께는 얼마야?",
    "다음 구역에서 어떤 보호구가 필요해?",
    "다음 구역으로 이동하는 방법은?",
])
def test_referencing_the_next_zone_does_not_itself_request_movement(prompt):
    assert not is_current_next_question(prompt)


@pytest.mark.parametrize("prompt", [
    "권장 순서 말고 지금 도막 두께가 어때?",
    "파이썬 작업 순서를 알려줘.",
])
def test_excluded_and_unrelated_work_orders_are_not_spray_reference_questions(prompt):
    assert not is_reference_order_question(prompt)


def test_measurement_after_explicitly_excluding_order_does_not_return_reference_plan(tmp_path):
    profile(tmp_path, "spray", ["2", "3", "1"])
    store = SequenceStore(":memory:")
    seed_turn(store, session_id=SESSION_ID)
    reports = SimpleNamespace(get_session=lambda _: SimpleNamespace(scenario_id="3", tool_type="spray_gun"),
                              latest_complete_analysis=lambda _: None)
    data = ask(DialoguePipeline(), store, "권장 순서 말고 지금 도막 두께가 어때?",
               sequence_engine=SequenceAnalysisEngine(tmp_path), reports=reports, session_id=SESSION_ID)
    assert data["coating_context"] is not None
    assert data["sequence_context"] is None and "2 → 3 → 1" not in data["answer"]


def test_new_general_topic_sees_previous_turn_in_planning_but_not_in_generation():
    store = SequenceStore(":memory:")
    seed_turn(store, session_id=SESSION_ID)
    pipeline = DialoguePipeline({
        "kind": "chat", "intent": "knowledge", "scope": "unspecified", "context_relation": "new_topic",
    })
    data = ask(pipeline, store, "그건 그렇고 프랑스의 수도는?", session_id=SESSION_ID)
    assert data["answer"] == "프랑스의 수도는 파리입니다."
    assert pipeline.plans[-1][1]["recent_turns"][-1]["prompt"] == PREVIOUS_PROMPT
    assert pipeline.conversations[-1][1]["recent_turns"] == []
    assert "training_summary" not in pipeline.conversations[-1][1]["facts"]


def test_new_technical_topic_does_not_send_previous_zone_answer_to_generation():
    store = SequenceStore(":memory:")
    seed_turn(store)
    pipeline = DialoguePipeline({
        "kind": "coating_knowledge", "intent": "knowledge", "scope": "zone", "context_relation": "new_topic",
    })
    ask(pipeline, store, "그런데 이 Zone의 도막 기포를 어떻게 예방하죠?")
    assert pipeline.plans[-1][1]["recent_turns"][-1]["prompt"] == PREVIOUS_PROMPT
    generated = pipeline.calls[-1]["generation_instruction"]
    assert "기포" in generated
    assert PREVIOUS_PROMPT not in generated and "Zone 2 → 3 → 1입니다." not in generated
