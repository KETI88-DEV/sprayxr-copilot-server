"""Reference work plans need a mode, not trainee telemetry or a report."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.copilot_router import create_copilot_router
from trainer.demo_api.sequence_analysis import SequenceAnalysisEngine
from trainer.demo_api.sequence_store import SequenceStore

URL = "/api/v1/copilot/query"
QUESTION = "이 피도면의 작업 순서를 구역 단위로 알려주세요."


def profile(root: Path, process: str, flow: list[str], mode: int = 3) -> None:
    path = root / "generated" / "profiles" / process / "learning" / f"mode_{mode:02d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"aggregate": {"macro_orders": [{
        "order": [{"map_num": 1, "zone_id": zone} for zone in flow], "session_count": 3,
    }], "zones": {}}}), encoding="utf-8")


class Pipeline:
    def __init__(self, decision=None):
        self.decision = decision or {"kind": "chat", "intent": "knowledge", "scope": "unspecified"}
        self.calls = []

    def plan_copilot_dialogue(self, *args, **kwargs):
        self.calls.append("plan")
        return self.decision

    def converse_copilot(self, *args, **kwargs):
        self.calls.append("converse")
        return "일반 대화 답변입니다."

    def run(self, **kwargs):
        self.calls.append("run")
        return {"response": "[답변] 일반 지식 답변입니다.", "rag_chunks": []}


def client(root, *, pipeline=None, store=None, reports=None, **dependencies):
    app = FastAPI()
    app.include_router(create_copilot_router(
        (lambda: pipeline) if pipeline is not None else None,
        store=store or SequenceStore(":memory:"), reports=reports,
        sequence_engine=SequenceAnalysisEngine(root),
        **dependencies,
    ))
    return TestClient(app)


def query(c, **overrides):
    body = {"user_id": "COPILOT-DEMO-USER", "task_type": "knowledge_search", "scenario_id": "3", "prompt": QUESTION}
    body.update(overrides)
    response = c.post(URL, json=body)
    assert response.status_code == 200, response.text
    return response.json()["data"]


@pytest.mark.parametrize("task", ["knowledge_search", "trajectory_recomm"])
def test_mode_only_reference_question_bypasses_empty_facts_conversation(tmp_path, task):
    profile(tmp_path, "spray", ["1", "3", "2"])
    pipeline = Pipeline()
    data = query(client(tmp_path, pipeline=pipeline), task_type=task, context="스프레이 도장")
    assert "Zone 1 → 3 → 2" in data["answer"]
    assert "현재 등록된 기준" in data["answer"]
    assert "학습 모드 3" in data["answer"]
    assert "스프레이" in data["answer"]
    assert data["context_status"] == "available"
    assert data["retrieval_used"] is False and data["citations"] == []
    context = data["sequence_context"]
    assert context["scope"] == "reference_only" and context["can_compare"] is False
    assert context["recommended_flow"] == ["1", "3", "2"]
    for key in ("adherence_rate", "position_similarity", "best_reference_similarity", "evaluated_visit_count", "on_track_visit_count", "first_visit_flow"):
        assert context[key] is None
    assert pipeline.calls == []


def test_unknown_process_shows_both_registered_plans_without_selecting_spray(tmp_path):
    profile(tmp_path, "spray", ["1", "3", "2"])
    profile(tmp_path, "blasting", ["2", "1", "3"])
    data = query(client(tmp_path))
    assert "스프레이" in data["answer"] and "블라스팅" in data["answer"]
    assert "1 → 3 → 2" in data["answer"] and "2 → 1 → 3" in data["answer"]
    assert data["sequence_context"]["process"] is None
    assert data["sequence_context"]["recommended_flow"] is None
    assert {p["process"] for p in data["sequence_context"]["reference_plans"]} == {"spray", "blasting"}


def test_unknown_process_with_only_blasting_labels_the_available_process(tmp_path):
    profile(tmp_path, "blasting", ["3", "1", "2"])
    data = query(client(tmp_path))
    assert "블라스팅" in data["answer"] and "3 → 1 → 2" in data["answer"]
    assert data["sequence_context"]["process"] == "blasting"


def test_started_session_without_sequences_resolves_saved_mode(tmp_path):
    profile(tmp_path, "spray", ["1", "3", "2"])
    reports = SimpleNamespace(get_session=lambda session_id: SimpleNamespace(scenario_id="3"))
    data = query(client(tmp_path, reports=reports), scenario_id=None, session_id="STARTED-ONLY", context="spray")
    assert data["scenario_id"] == "3"
    assert data["sequence_context"]["recommended_flow"] == ["1", "3", "2"]


def test_explicit_mode_and_prompt_process_override_stored_sequence(tmp_path):
    profile(tmp_path, "spray", ["2", "1"], mode=2)
    profile(tmp_path, "blasting", ["3", "1", "2"])
    store = SequenceStore(":memory:")
    body = {"scenario_id": "3", "sequence_setting": {"tool_type": "blasting_gun"}}
    store.record(session_key="SAVED", sequence_id="SEQ-1", body_sha256="x", frames=0,
                 received_at="2026-11-11T00:00:00Z", body=json.dumps(body))
    data = query(client(tmp_path, store=store), scenario_id="2", session_id="SAVED", prompt="스프레이 작업 순서를 구역 단위로 알려줘")
    assert data["scenario_id"] == "2"
    assert data["sequence_context"]["process"] == "spray"
    assert data["sequence_context"]["recommended_flow"] == ["2", "1"]


def test_saved_sequence_supplies_mode_and_blasting_process(tmp_path):
    profile(tmp_path, "spray", ["1", "3", "2"])
    profile(tmp_path, "blasting", ["3", "1", "2"])
    store = SequenceStore(":memory:")
    store.record(session_key="SAVED", sequence_id="SEQ-1", body_sha256="x", frames=0,
                 received_at="2026-11-11T00:00:00Z", body=json.dumps({
                     "scenario_id": "3", "sequence_setting": {"tool_type": "blasting_gun"}}))
    data = query(client(tmp_path, store=store), scenario_id=None, session_id="SAVED")
    assert data["scenario_id"] == "3" and data["sequence_context"]["process"] == "blasting"
    assert "3 → 1 → 2" in data["answer"]


@pytest.mark.parametrize("scenario,reason,needed", [(None, "scenario_id_missing", "모드"), ("11", "sequence_reference_unavailable", "기준")])
def test_missing_mode_or_reference_explains_the_actual_requirement(tmp_path, scenario, reason, needed):
    profile(tmp_path, "spray", ["1", "2"])
    data = query(client(tmp_path), scenario_id=scenario, context="스프레이")
    assert data["context_status"] == "not_available"
    assert data["sequence_context"]["reason_code"] == reason
    assert needed in data["answer"]
    assert "시퀀스를 전송" not in data["answer"] and "리포트" not in data["answer"]


def test_semantic_order_reference_lookup_works_across_default_tab(tmp_path):
    profile(tmp_path, "spray", ["1", "3", "2"])
    pipeline = Pipeline({"kind": "lookup", "intent": "order_reference", "scope": "record"})
    data = query(client(tmp_path, pipeline=pipeline), prompt="이 피도면을 어디부터 시작하면 좋을까요?", context="spray")
    assert "1 → 3 → 2" in data["answer"]
    assert pipeline.calls == ["plan"]


def test_semantic_tab_classifier_can_request_a_reference_plan_without_a_session(tmp_path):
    profile(tmp_path, "spray", ["1", "3", "2"])

    class Classifier:
        def classify_copilot_intent(self, task_type, prompt):
            return "order_reference"

    data = query(client(tmp_path, pipeline=Classifier()), task_type="trajectory_recomm",
                 prompt="이 피도면을 어디부터 시작하면 좋을까요?", context="spray")
    assert "1 → 3 → 2" in data["answer"]


def test_bad_semantic_reference_label_does_not_replace_an_explicit_actual_question(tmp_path):
    profile(tmp_path, "spray", ["1", "3", "2"])
    pipeline = Pipeline({"kind": "lookup", "intent": "order_reference", "scope": "record"})
    data = query(client(tmp_path, pipeline=pipeline), task_type="trajectory_recomm",
                 session_id="EMPTY", prompt="실제 수행 순서를 알려줘")
    assert data["context_status"] == "not_available"
    assert data["sequence_context"].get("scope") != "reference_only"


@pytest.mark.parametrize("prompt", ["실제 수행 순서는?", "내가 누락한 구역은?", "순서 준수율은?"])
def test_actual_history_questions_still_require_actual_data(tmp_path, prompt):
    profile(tmp_path, "spray", ["1", "3", "2"])
    data = query(client(tmp_path), task_type="trajectory_recomm", session_id="EMPTY", prompt=prompt)
    assert data["context_status"] == "not_available"
    assert data["sequence_context"].get("scope") != "reference_only"
    assert "1 → 3 → 2" not in data["answer"]


@pytest.mark.parametrize("prompt", ["안녕", "다음 구역이란 무엇인가요?"])
def test_general_or_realtime_questions_are_not_replaced_with_a_reference_plan(tmp_path, prompt):
    profile(tmp_path, "spray", ["1", "3", "2"])
    data = query(client(tmp_path, pipeline=Pipeline()), prompt=prompt)
    assert data["sequence_context"] is None
    assert data["answer"] == "일반 대화 답변입니다."


def test_painting_order_term_question_is_answered_as_knowledge_not_a_reference_plan(tmp_path):
    profile(tmp_path, "spray", ["1", "3", "2"])
    data = query(client(tmp_path, pipeline=Pipeline()), prompt="도장 작업 순서란 무엇인가요?")
    assert data["sequence_context"] is None
    assert data["answer"] == "일반 지식 답변입니다."


def test_detailed_reference_answer_is_short_and_has_reference_basis(tmp_path):
    profile(tmp_path, "spray", ["1", "3", "2"])
    data = query(client(tmp_path), context=json.dumps({"process": "spray"}), response_mode="detailed")
    assert "1 → 3 → 2" in data["answer"] and "3회" in data["answer"]
    assert data["sequence_context"]["reference_basis"] == "measured_expert_runs"
    assert len(data["answer"]) < 400
    assert "준수율" not in data["answer"] and "다음 Zone" not in data["answer"]


@pytest.mark.parametrize("prompt", ["현재 작업 순서에서 다음 구역은 어디인가요?", "작업 순서상 다음 Zone은 어디야?", "지금 다음 Zone으로 이동해도 되나요?"])
def test_next_zone_without_current_data_never_becomes_a_whole_reference_plan(tmp_path, prompt):
    profile(tmp_path, "spray", ["1", "3", "2"])
    pipeline = Pipeline({"kind": "lookup", "intent": "order_reference", "scope": "record"})
    data = query(client(tmp_path, pipeline=pipeline), prompt=prompt)
    assert data["sequence_context"]["scope"] == "current_progress"
    assert data["sequence_context"]["next_zone_id"] is None
    assert data["context_status"] == "not_available"
    assert "확인할 수 없" in data["answer"]
    assert "1 → 3 → 2" not in data["answer"]
    assert pipeline.calls == []


@pytest.mark.parametrize("mode", ["quick", "detailed"])
def test_verified_csv_only_viewer_keeps_its_rich_analysis_before_registered_reference(tmp_path, mode):
    from trainer.demo_api.viewer_context import sign_context
    profile(tmp_path, "spray", ["9", "8"])
    analysis = {"status": "available", "recommended_flow": ["1", "3", "2"],
                "selected_flow": ["1", "2", "3"], "adherence_rate": .67,
                "reference_session_count": 4, "best_reference_similarity": .75,
                "position_similarity": .6, "zone_keys": ["1:1", "1:2", "1:3"]}
    context = "motion_viewer_analysis:" + json.dumps({"analysis_context": analysis,
        "signature": sign_context("trajectory_recomm", analysis)})
    data = query(client(tmp_path), task_type="trajectory_recomm", scenario_id=None,
                 prompt="권장 Zone 순서는?", context=context, response_mode=mode)
    assert data["sequence_context"]["selected_flow"] == ["1", "2", "3"]
    assert data["sequence_context"]["adherence_rate"] == .67
    assert data["context_status"] == "available"
    assert "1 → 3 → 2" in data["answer"] and "9 → 8" not in data["answer"]
    if mode == "detailed":
        assert "67%" in data["answer"]


def test_invalid_signed_viewer_playback_cannot_be_hidden_by_reference_shortcut(tmp_path):
    from trainer.demo_api.viewer_context import sign_context
    profile(tmp_path, "spray", ["1", "2"])
    analysis = {"status": "available", "recommended_flow": ["1", "2"], "zone_keys": ["1:1"]}
    context = "motion_viewer_analysis:" + json.dumps({"analysis_context": analysis,
        "signature": sign_context("trajectory_recomm", analysis),
        "playback_context": {"playhead_seconds": -1}})
    response = client(tmp_path).post(URL, json={"user_id": "COPILOT-DEMO-USER",
        "task_type": "trajectory_recomm", "scenario_id": "3", "prompt": "권장 Zone 순서는?", "context": context})
    assert response.status_code == 422


def test_forged_viewer_reference_flow_is_never_used(tmp_path):
    profile(tmp_path, "spray", ["1", "2"])
    context = "motion_viewer_analysis:" + json.dumps({"analysis_context": {
        "status": "available", "recommended_flow": ["99", "98"]}, "signature": "forged"})
    data = query(client(tmp_path), task_type="trajectory_recomm", prompt="권장 Zone 순서는?", context=context)
    assert "99 → 98" not in data["answer"]
    assert data["sequence_context"]["scope"] == "reference_only"


@pytest.mark.parametrize("coverage,status,next_zone", [(None, None, None), (50, "needs_more", None), (100, "complete", "2")])
def test_next_zone_uses_the_shared_official_pixel_completion_gate(tmp_path, coverage, status, next_zone):
    from trainer.demo_api.tests.test_zone_completion_gate import engine, request
    progress = engine(tmp_path)
    req = request(valid_count=coverage)
    if coverage is None:
        req.sequence_result.cur_sequence_heatmap = None
    store = SequenceStore(":memory:")
    store.record(session_key=req.session_id, sequence_id=req.sequence_id, body_sha256="next",
                 frames=0, received_at="now", body=req.model_dump_json())
    data = query(client(tmp_path, store=store, coating_engine=progress),
                 session_id=req.session_id, task_type="trajectory_recomm",
                 prompt="현재 작업 순서에서 다음 구역은 어디인가요?")
    assert data["sequence_context"]["scope"] == "current_progress"
    assert data["sequence_context"]["next_zone_id"] == next_zone
    assert data["coating_context"]["status"] == status
    if next_zone is None:
        assert "다음 권장 구역" not in data["answer"]
    else:
        assert "2" in data["answer"]


def test_current_progress_does_not_apply_another_scenario_to_saved_pixels(tmp_path):
    from trainer.demo_api.tests.test_zone_completion_gate import engine, request
    progress = engine(tmp_path)
    req = request()
    store = SequenceStore(":memory:")
    store.record(session_key=req.session_id, sequence_id=req.sequence_id, body_sha256="next",
                 frames=0, received_at="now", body=req.model_dump_json())
    data = query(client(tmp_path, store=store, coating_engine=progress), scenario_id="2",
                 session_id=req.session_id, prompt="작업 순서상 다음 Zone은 어디야?")
    assert data["sequence_context"]["next_zone_id"] is None
    assert data["context_status"] == "not_available"


def test_next_zone_respects_explicit_saved_session_target(tmp_path):
    from trainer.demo_api.tests.test_zone_completion_gate import engine, request
    progress = engine(tmp_path)
    req = request(value=160)
    store = SequenceStore(":memory:")
    store.record(session_key=req.session_id, sequence_id=req.sequence_id, body_sha256="target",
                 frames=0, received_at="now", body=req.model_dump_json())
    reports = SimpleNamespace(get_session=lambda _: SimpleNamespace(target_thickness_um=200))
    data = query(client(tmp_path, store=store, reports=reports, coating_engine=progress),
                 session_id=req.session_id, prompt="작업 순서상 다음 Zone은 어디야?")
    assert data["coating_context"]["target_thickness_um"] == 200
    assert data["sequence_context"]["next_zone_id"] is None


def test_next_zone_replays_saved_absolute_crops_after_engine_restart(tmp_path):
    from trainer.demo_api.tests.test_zone_completion_gate import engine, request
    progress = engine(tmp_path)
    store = SequenceStore(":memory:")
    left, right = request(1, width=5), request(2, width=5, x0=5)
    for req in (left, right):
        store.record(session_key=req.session_id, sequence_id=req.sequence_id, body_sha256=req.sequence_id,
                     frames=0, received_at="now", body=req.model_dump_json())
    data = query(client(tmp_path, store=store, coating_engine=progress),
                 session_id=right.session_id, prompt="작업 순서상 다음 Zone은 어디야?")
    assert data["coating_context"]["coverage_percent"] == 100
    assert data["sequence_context"]["next_zone_id"] == "2"


def test_blasting_next_guidance_keeps_its_existing_order_engine(tmp_path):
    from trainer.demo_api.tests.test_zone_completion_gate import fixture, request
    fixture(tmp_path)
    source = tmp_path / "generated/profiles/spray/learning/mode_03.json"
    destination = tmp_path / "generated/profiles/blasting/learning/mode_03.json"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(source.read_bytes())
    req = request()
    req.sequence_setting["tool_type"] = "blasting_gun"
    store = SequenceStore(":memory:")
    store.record(session_key=req.session_id, sequence_id=req.sequence_id, body_sha256="blasting",
                 frames=0, received_at="now", body=req.model_dump_json())
    data = query(client(tmp_path, store=store), session_id=req.session_id,
                 prompt="작업 순서상 다음 Zone은 어디야?")
    assert "다음 권장 구역은 2" in data["answer"]
    assert data["coating_context"] is None
