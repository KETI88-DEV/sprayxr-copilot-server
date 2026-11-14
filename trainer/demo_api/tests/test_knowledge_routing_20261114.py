"""지식 질문이 측정값·Zone 순서 경로로 새지 않는다(2026-11-14 운영 시험 q25·q28·q30·q32)."""
from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.copilot_intent import is_general_painting_knowledge
from trainer.demo_api.copilot_router import create_copilot_router
from trainer.demo_api.dialogue import viewer_measurement_answer
from trainer.demo_api.reference_order_query import is_reference_order_question
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.tests.viewer_record import AT_3S, RecordingPipeline, ask, context, payload


@pytest.mark.parametrize("tab", ["knowledge", "coating"])
@pytest.mark.parametrize("prompt", [
    "에어리스 스프레이 분사 속도는 얼마가 적당해?", "분사 속도는 얼마가 적당해?", "적정 이격거리는?",
    "스프레이 도장 전에 표면처리는 어떤 순서로 해?", "표면처리 절차 알려줘",
])
def test_viewer_knowledge_questions_go_to_the_documents(tab: str, prompt: str) -> None:
    result = ask(prompt, tab=tab)
    assert result["source"] == "documents" and result["calls"][-1] == "run", result["answer"]
    assert "전문가 기록은" not in result["answer"]


def test_spray_word_is_not_a_record_reference() -> None:
    # "스프레이 분사"의 "이 분사"를 이 기록의 분사로 읽었다.
    signed = context("knowledge_search", payload(), AT_3S)
    assert viewer_measurement_answer(signed, "에어리스 스프레이 분사 속도는 얼마가 적당해?") is None
    assert viewer_measurement_answer(signed, "이 분사 속도는 어땠어?") is not None
    assert is_general_painting_knowledge("에어리스 스프레이 분사 속도는 얼마가 적당해?")
    assert not is_general_painting_knowledge("이 분사 속도는 얼마였어?")


@pytest.mark.parametrize("prompt", ["스프레이 도장 전에 표면처리는 어떤 순서로 해?", "도장 전처리 순서는?", "도장 작업 절차 순서 알려줘"])
def test_procedure_questions_are_not_zone_order_questions(prompt: str) -> None:
    assert not is_reference_order_question(prompt)


@pytest.mark.parametrize("prompt", ["TR-L01 3번 모드 권장 순서 알려줘", "권장 Zone 순서는?", "스프레이 작업 순서 알려줘"])
def test_zone_order_questions_still_use_the_registered_plan(prompt: str) -> None:
    assert is_reference_order_question(prompt)


class _Plans:
    def reference_plan(self, *, scenario_id: str, process: str = "spray") -> dict[str, Any] | None:
        return {"process": process, "flow": ["7", "4"], "reference_session_count": 6}


@pytest.mark.parametrize("mode", ["quick", "detailed"])
def test_public_procedure_question_does_not_ask_for_a_scenario(mode: str) -> None:
    pipeline = RecordingPipeline()
    app = FastAPI()
    app.include_router(create_copilot_router(lambda: pipeline, store=SequenceStore(":memory:"), sequence_engine=_Plans()))  # type: ignore[arg-type]
    data = TestClient(app).post("/api/v1/copilot/query", json={
        "user_id": "COPILOT-AI-TEST", "task_type": "knowledge_search", "response_mode": mode,
        "prompt": "스프레이 도장 전에 표면처리는 어떤 순서로 해?"}).json()["data"]
    assert "scenario_id" not in data["answer"] and pipeline.calls[-1] == "run"
