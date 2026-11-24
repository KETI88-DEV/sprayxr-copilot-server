"""시뮬레이터 음성 질의의 세션 질문은 판정 모델 없이 세션 자료로 답한다(2026-11-24 음성 질의 시험의 오답)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api import coating_query
from trainer.demo_api.copilot_intent import RECORD_OBJECT
from trainer.demo_api.copilot_router import create_copilot_router
from trainer.demo_api.reference_order_query import actual_order_intent, current_next_answer, is_current_completion_question
from trainer.demo_api.sequence_analysis import SequenceAnalysisEngine
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.tests.test_pose_question_scope import _graded, _saved
from trainer.demo_api.tests.test_query_evidence_fallback import Pipeline
from trainer.demo_api.tests.test_zone_completion_gate import engine, request, thin_mean


@pytest.mark.parametrize("prompt", ["지금 구역 다 칠했어?", "2번 구역 완료됐어?", "지금 구역 끝났어?", "이제 넘어가도 돼?"])
def test_completion_questions_are_progress_questions(prompt):
    assert is_current_completion_question(prompt)


@pytest.mark.parametrize("prompt", ["구역 완료 기준이 뭐야?", "도장이 끝났을 때 주의할 점은?", "다 됐어?", "도막 두께가 다 됐는지 어떻게 알아?"])
def test_knowledge_or_vague_completion_wording_is_not_a_progress_question(prompt):
    assert not is_current_completion_question(prompt)


def _store(*bodies):
    store = SequenceStore(":memory:")
    for body in bodies:
        store.record(session_key=body.session_id, sequence_id=body.sequence_id, body_sha256=body.sequence_id,
                     frames=0, received_at="now", body=body.model_dump_json())
    return store


@pytest.mark.parametrize(("req", "verdict"), [(thin_mean(), "1구역은 아직 완료되지 않았습니다."), (request(), "1구역은 완료 기준을 넘었습니다.")])
def test_completion_question_states_the_gate_verdict_first(tmp_path: Path, req, verdict):
    query = SimpleNamespace(session_id=req.session_id, context=None, scenario_id=None, prompt="지금 구역 다 칠했어?")
    text, _, _ = current_next_answer(query, _store(req), sequence_engine=SequenceAnalysisEngine(tmp_path), coating_engine=engine(tmp_path))
    assert text.startswith(verdict)


def test_completion_question_about_another_zone_says_only_the_current_zone_is_judged(tmp_path: Path):
    req = thin_mean()
    query = SimpleNamespace(session_id=req.session_id, context=None, scenario_id=None, prompt="2번 구역 완료됐어?")
    text, _, _ = current_next_answer(query, _store(req), sequence_engine=SequenceAnalysisEngine(tmp_path), coating_engine=engine(tmp_path))
    assert text.startswith("지금 구역은 1구역입니다. 훈련 중에는 지금 구역의 완료만 판정합니다. 1구역은 아직 완료되지 않았습니다.")


def test_before_any_sequence_the_next_zone_is_the_registered_first_zone():
    plans = SimpleNamespace(reference_plan=lambda **kwargs: {"flow": ["7", "4", "8"]} if kwargs == {"scenario_id": "3", "process": "spray"} else None)
    reports = SimpleNamespace(get_session=lambda _: SimpleNamespace(scenario_id="3", tool_type="spray_gun"))
    query = SimpleNamespace(session_id="empty", context=None, scenario_id=None, prompt="다음 구역은 어디야?")
    text, scope, payload = current_next_answer(query, SequenceStore(":memory:"), reports, sequence_engine=plans)
    assert text == "아직 분사 기록이 없습니다. 7구역에서 시작하세요(권장 순서 7 → 4 → 8)."
    assert (scope["next_zone_id"], scope["reason_code"], payload) == ("7", "no_sequence_yet", None)


@pytest.mark.parametrize(("prompt", "intent"), [
    ("지금까지 내가 칠한 구역 순서 알려줘", "order_actual"), ("내가 칠한 순서는?", "order_actual"),
    ("순서 준수율은 몇 퍼센트야?", "order_adherence"), ("아직 안 칠한 구역은 어디야?", "order_missing"),
    ("빠진 구역 있어?", "order_missing"), ("권장 구역 순서 알려줘", None)])
def test_actual_order_wording(prompt, intent):
    assert actual_order_intent(prompt) == intent


class Planner(Pipeline):
    def __init__(self):
        super().__init__()
        self.plans = []

    def plan_copilot_dialogue(self, prompt, **kwargs):
        self.plans.append(prompt)
        return super().plan_copilot_dialogue(prompt, **kwargs)


def _ask(pipeline, prompt, task_type, store=None, **extra):
    app = FastAPI()
    app.include_router(create_copilot_router(lambda: pipeline, store=store or SequenceStore(":memory:"), **extra))
    return TestClient(app).post("/api/v1/copilot/query", json={
        "user_id": "COPILOT-AI-TEST", "session_id": "gate-test", "scenario_id": "3", "task_type": task_type,
        "prompt": prompt, "response_mode": "quick"})


@pytest.mark.parametrize("prompt", ["순서 준수율은 몇 퍼센트야?", "아직 안 칠한 구역은 어디야?", "빠진 구역 있어?"])
def test_actual_order_questions_skip_the_planner(prompt):
    pipeline = Planner()
    response = _ask(pipeline, prompt, "trajectory_recomm")
    assert response.status_code == 200
    assert pipeline.plans == [] and pipeline.runs == []


def test_session_review_before_completion_does_not_invent_a_verdict():
    pipeline = Planner()
    data = _ask(pipeline, "잘한 점과 고칠 점 알려줘", "knowledge_search").json()["data"]
    assert data["answer"].startswith("세션을 마치면 잘한 점과 고칠 점을 정리해 알려 드립니다.")
    assert data["context_status"] == "not_available" and pipeline.plans == [] and pipeline.runs == []


def test_general_fixing_question_is_still_a_knowledge_question():
    pipeline = Planner()
    _ask(pipeline, "저도막은 어떻게 고칠 수 있어?", "knowledge_search")
    assert pipeline.runs


def test_unscored_body_part_on_the_pose_tab_is_declined_without_the_planner():
    pipeline = Planner()
    data = _ask(pipeline, "어깨는 어떻게 해?", "pose_recomm").json()["data"]
    assert "채점 엔진은 양팔의 위팔·팔꿈치·손목 위치만" in data["answer"] and pipeline.plans == []


def test_knowledge_answer_without_a_record_pointer_gets_no_session_numbers():
    body = json.loads(request().model_dump_json())
    body["sequence_result"].update(avg_speed_mms=465.93, avg_distance_mm=545.08, avg_thickness_um=9.64)
    store = SequenceStore(":memory:")
    store.record(session_key="gate-test", sequence_id="SEQ-001", body_sha256="x", frames=0, received_at="now", body=json.dumps(body))
    pipeline = Planner()
    _ask(pipeline, "스프레이 건과 피도면 적정 거리는 몇 mm야?", "knowledge_search", store)
    assert pipeline.runs and "545.08" not in json.dumps(pipeline.runs, ensure_ascii=False)


def test_numbered_zone_wording_selects_one_zone():
    rows = (_graded("3", "below", 98.0, 0.12, 88, adjustments=(("distance_mm", "decrease", 2),)),
            _graded("5", "above", 301.0, 0.03, 0, 0, 0, 97))
    answer = coating_query.answer_distribution_cause(_saved(*rows), detailed=True, prompt="3번 구역 도막이 부족한 이유가 뭐야?")
    assert "Zone 3" in answer and "Zone 5" not in answer
    assert coating_query._selected(list(_saved(*rows)["zones"]), "3번 구역 도막 두께 어때?") == [_saved(*rows)["zones"][0]]
    assert RECORD_OBJECT.search("3번 구역 도막 두께 어때?")


def test_next_zone_skips_zones_already_painted_in_the_session():
    from trainer.demo_api.coating_progress import CoatingProgressEngine
    profile = {"aggregate": {"macro_orders": [{"order": [{"zone_id": z} for z in ("7", "4", "8", "5", "6")]}]}}
    pick = CoatingProgressEngine._next_zone
    assert pick(profile, 1, "4") == "8"
    assert pick(profile, 1, "4", {"8", "7"}) == "5"
    assert pick(profile, 1, "6", {"4"}) == "7"  # 끝 구역이면 앞에서 아직 안 칠한 구역
    assert pick(profile, 1, "5", {"7", "4", "8", "6"}) is None


def test_overcoated_zone_is_complete_and_sends_the_trainee_on(tmp_path: Path):
    # 2026-11-24 사용자 결정: 하한(160µm) 이상 면적 95%면 완료. 과도막도 더 칠하지 않고 다음으로 간다.
    progress = engine(tmp_path).evaluate(request(value=300))
    assert (progress.status, progress.recommendation) == ("complete", "move_next_zone")
    assert progress.coverage_percent == 100.0
    assert "과도막" in progress.message and "더 칠하지 마세요" in progress.message


def test_session_summary_has_no_averaged_sequence_thickness():
    from trainer.demo_api.copilot_session_context import summarize_session
    bodies = [{"session_id": "s", "scenario_id": "3", "zone_id": "2", "sequence_id": f"SEQ-00{i}",
               "sequence_result": {"avg_speed_mms": 400.0, "avg_distance_mm": 300.0, "avg_thickness_um": t}} for i, t in ((1, 20.0), (2, 30.0))]
    averages = json.loads(summarize_session(bodies))["session_averages"]
    assert "avg_thickness_um" not in averages and averages["avg_speed_mms"] == "400 mm/s"


@pytest.mark.parametrize("prompt", ["스프레이 건과 피도면 적정 거리는 몇 mm야?", "도장할 때 피도면과의 거리는 얼마가 적당해?"])
def test_gun_to_surface_distance_wording_uses_the_distance_type(prompt):
    from trainer.demo_api.context_prompts import matched_prompt
    assert matched_prompt(prompt)["이름"] == "적정 분사 거리"


def test_charged_line_distance_keeps_its_own_type():
    from trainer.demo_api.context_prompts import matched_prompt
    found = matched_prompt("충전전로와의 적정 거리는?")
    assert found is None or found["이름"] != "적정 분사 거리"
