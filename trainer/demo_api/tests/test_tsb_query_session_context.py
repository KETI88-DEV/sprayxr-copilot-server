"""TSB text/voice queries consume authoritative training and Replay context."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.copilot_router import create_copilot_router
from trainer.demo_api.reference_order_query import is_current_next_question
from trainer.demo_api.sequence_analysis import SequenceAnalysisEngine
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.tests.test_scenario_reference_query import profile
from trainer.demo_api.tests.test_voice_router import _Stt, _wav
from trainer.demo_api.tests.test_voice_session_routing import Planner, _store
from trainer.demo_api.tests.test_zone_completion_gate import engine, request
from trainer.demo_api.voice_router import create_voice_router


NEXT_QUESTIONS = ["그 다음 순서는 뭐지", "다음 순서가 어떻게 돼?", "다음 순서는 뭐야?", "그다음 순서 알려줘"]


def _client(tmp_path, *, prompt="", pipeline=None, store=None, reports=None, coating=None):
    app = FastAPI()
    dependencies = dict(store=store or SequenceStore(":memory:"), reports=reports,
                        sequence_engine=SequenceAnalysisEngine(tmp_path), coating_engine=coating)
    app.include_router(create_copilot_router(lambda: pipeline, **dependencies))
    app.include_router(create_voice_router(lambda: pipeline, stt_provider=lambda: _Stt(prompt), **dependencies))
    return TestClient(app)


def _ask(client, prompt, *, voice=False, **extra):
    fields = dict(user_id="COPILOT-AI-TEST", session_id="gate-test", scenario_id="3",
                  task_type="knowledge_search", response_mode="quick", **extra)
    headers = {"X-Copilot-Source": "assistant-test"}
    if voice:
        return client.post("/api/v1/copilot/query-voice", data={**fields, "language": "ko"},
                           files={"audio": ("query.wav", _wav(), "audio/wav")}, headers=headers)
    return client.post("/api/v1/copilot/query", json={**fields, "prompt": prompt}, headers=headers)


@pytest.mark.parametrize("prompt", NEXT_QUESTIONS)
def test_vague_next_order_wording_selects_live_progress(prompt):
    assert is_current_next_question(prompt)


@pytest.mark.parametrize("prompt", ["전처리 다음 순서는 뭐야?", "다음 순서의 의미는 무엇인가요?",
                                     "다음 순서를 어떻게 정하나요?", "다음 순서가 중요한 이유는?"])
def test_next_order_procedure_and_explanation_questions_remain_knowledge(prompt):
    assert not is_current_next_question(prompt)


@pytest.mark.parametrize("voice", [False, True])
@pytest.mark.parametrize("coverage,next_zone", [(50, None), (100, "2"), (None, None)])
def test_vague_next_order_uses_the_same_official_gate_for_text_and_voice(tmp_path, voice, coverage, next_zone):
    req = request(valid_count=coverage)
    if coverage is None:
        req.sequence_result.cur_sequence_heatmap = None
    coating = engine(tmp_path)
    pipeline = Planner()
    prompt = "그 다음 순서는 뭐지"
    response = _ask(_client(tmp_path, prompt=prompt, pipeline=pipeline, store=_store(req), coating=coating), prompt, voice=voice)
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["sequence_context"]["scope"] == "current_progress"
    assert data["sequence_context"]["next_zone_id"] == next_zone
    assert data["retrieval_used"] is False and data["citations"] == []
    assert pipeline.plans == [] and pipeline.runs == []
    if coverage is None:
        assert data["context_status"] == "not_available"
    elif coverage == 50:
        assert data["coating_context"]["status"] == "needs_more"
    if voice:
        assert data["transcript"] == prompt


@pytest.mark.parametrize("tool_type,expected", [("spray_gun", ["2", "3", "1"]),
                                                ("blasting_gun", ["1", "2", "3"]),
                                                (None, ["2", "3", "1"])])
def test_started_session_plan_uses_its_process_before_any_sequence(tmp_path, tool_type, expected):
    profile(tmp_path, "spray", ["2", "3", "1"])
    profile(tmp_path, "blasting", ["1", "2", "3"])
    reports = SimpleNamespace(get_session=lambda _: SimpleNamespace(scenario_id="3", tool_type=tool_type))
    response = _ask(_client(tmp_path, reports=reports), "이 피도면의 작업 순서를 구역 단위로 알려주세요.")
    data = response.json()["data"]
    assert data["sequence_context"]["recommended_flow"] == expected
    assert len(data["sequence_context"]["reference_plans"]) == 1
    assert data["sequence_context"]["scope"] == "reference_only"


@pytest.mark.parametrize("voice", [False, True])
def test_default_query_task_uses_completed_replay_actual_order_without_model(tmp_path, voice):
    report = {"sequence_report": {"analysis_status": "available", "recommended_flow": ["1", "2", "3"],
        "first_visit_flow": ["1", "3", "2"], "adherence_rate": 0.7}}
    reports = SimpleNamespace(latest_complete_analysis=lambda _: ("RPT-REPLAY", report), get_session=lambda _: None)
    pipeline = Planner()
    prompt = "지금까지 내가 칠한 구역 순서 알려줘"
    response = _ask(_client(tmp_path, prompt=prompt, pipeline=pipeline, reports=reports), prompt, voice=voice)
    data = response.json()["data"]
    assert data["sequence_context"]["report_id"] == "RPT-REPLAY"
    assert data["sequence_context"]["first_visit_flow"] == ["1", "3", "2"]
    assert "1 → 3 → 2" in data["answer"]
    assert pipeline.plans == [] and pipeline.runs == []


def test_current_problem_question_without_analysis_does_not_invent_no_problems(tmp_path):
    pipeline = Planner()
    response = _ask(_client(tmp_path, pipeline=pipeline), "지금 뭐가 문제야?")
    data = response.json()["data"]
    assert data["context_status"] == "not_available"
    assert "세션을 마치면" in data["answer"]
    assert pipeline.plans == [] and pipeline.runs == []


@pytest.mark.parametrize("source", ["session", "sequence"])
def test_generic_plan_question_does_not_replace_recorded_blasting_with_client_metadata(tmp_path, source):
    profile(tmp_path, "spray", ["2", "3", "1"])
    profile(tmp_path, "blasting", ["1", "2", "3"])
    req = request()
    req.sequence_setting["tool_type"] = "blasting_gun"
    reports = SimpleNamespace(get_session=lambda _: SimpleNamespace(
        scenario_id="3", tool_type="blasting_gun" if source == "session" else "spray_gun"))
    store = _store(req) if source == "sequence" else None
    response = _ask(_client(tmp_path, store=store, reports=reports),
                    "이 피도면의 작업 순서를 구역 단위로 알려주세요.", context='{"process":"spray"}')
    assert response.json()["data"]["sequence_context"]["process"] == "blasting"
    assert response.json()["data"]["sequence_context"]["recommended_flow"] == ["1", "2", "3"]


def test_explicit_alternative_plan_question_can_select_spray_without_changing_session(tmp_path):
    profile(tmp_path, "spray", ["2", "3", "1"])
    profile(tmp_path, "blasting", ["1", "2", "3"])
    session = SimpleNamespace(scenario_id="3", tool_type="blasting_gun")
    reports = SimpleNamespace(get_session=lambda _: session)
    response = _ask(_client(tmp_path, reports=reports), "스프레이 작업 순서를 알려주세요.")
    assert response.json()["data"]["sequence_context"]["recommended_flow"] == ["2", "3", "1"]
    assert session.tool_type == "blasting_gun"


@pytest.mark.parametrize("prompt", ["실제 도장에서 저도막이 생기는 원인은?", "순서가 중요한 이유는 뭐야?",
                                     "지금 구역의 도막이 부족한 원인은?", "실제 구역의 이격거리는 얼마야?"])
def test_technical_explanations_are_not_replaced_by_actual_order_lookup(tmp_path, prompt):
    pipeline = Planner()
    data = _ask(_client(tmp_path, pipeline=pipeline), prompt).json()["data"]
    assert data["sequence_context"] is None
    assert pipeline.runs
