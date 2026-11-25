"""Reproduce historical TSB answer-scope issues with synthetic sessions."""
from types import SimpleNamespace

import pytest

from trainer.demo_api.tests.test_pose_question_scope import _graded
from trainer.demo_api.tests.test_tsb_query_session_context import _ask, _client
from trainer.demo_api.tests.test_voice_session_routing import Planner, _store
from trainer.demo_api.tests.test_zone_completion_gate import engine, request


def _session(user_id="COPILOT-AI-TEST"):
    return SimpleNamespace(user_id=user_id, scenario_id="3", tool_type="spray_gun")


@pytest.mark.parametrize("prompt", ["지금 뭐가 문제야?", "지금 전체적으로 어떤지 봐줘"])
def test_current_review_reads_latest_coating_without_inventing_full_report(tmp_path, prompt):
    pipeline = Planner()
    coating = engine(tmp_path)
    sequence = request(value=132)
    sequence.user_id = "COPILOT-AI-TEST"
    reports = SimpleNamespace(get_session=lambda _: _session(), latest_complete_analysis=lambda _: None)
    client = _client(tmp_path, prompt=prompt, pipeline=pipeline, store=_store(sequence), coating=coating, reports=reports)

    response = _ask(client, prompt, voice=True)

    data = response.json()["data"]
    assert response.status_code == 200
    assert data["transcript"] == prompt
    assert data["sequence_context"]["current_zone_id"] == "1"
    assert data["coating_context"]["status"] == "needs_more"
    assert "저도막" in data["answer"]
    assert "최근 분사" in data["answer"]
    assert not data["retrieval_used"] and not pipeline.plans and not pipeline.runs


def test_overcoat_cause_identifies_records_to_check_without_claiming_cause(tmp_path):
    pipeline = Planner()
    reports = SimpleNamespace(
        get_session=lambda _: _session(),
        latest_complete_analysis=lambda _: ("RPT-TEST", {"coating_zones": [_graded("1", "above", 250.0, 0.0)]}),
    )
    client = _client(tmp_path, pipeline=pipeline, reports=reports)
    response = client.post("/api/v1/copilot/query", json={
        "user_id": "COPILOT-AI-TEST", "session_id": "gate-test", "scenario_id": "3",
        "task_type": "error_analysis", "prompt": "과도막이 심한 원인은 뭘까요?",
    }, headers={"X-Copilot-Source": "assistant-test"})

    answer = response.json()["data"]["answer"]
    assert response.status_code == 200
    assert "250µm" in answer and "확정할 수 없" in answer
    assert "이격거리" in answer and "겹침" in answer and "재분사" in answer
    assert not pipeline.plans and not pipeline.runs


def test_current_review_does_not_read_a_report_without_matching_session_scope(tmp_path):
    def forbidden_read(_):
        raise AssertionError("report read before checking the session scope")

    reports = SimpleNamespace(get_session=lambda _: _session("another-user"), latest_complete_analysis=forbidden_read)
    response = _ask(_client(tmp_path, reports=reports), "지금 전체적으로 어떤지 봐줘")
    assert response.status_code == 200
    assert response.headers["X-Copilot-Unavailable-Reason"] == "session_mismatch"
    assert response.json()["data"]["coating_context"] is None


def test_bare_number_needs_active_choice_and_can_answer_zone_clarification(tmp_path):
    pipeline = Planner()
    reports = SimpleNamespace(
        get_session=lambda _: _session(),
        latest_complete_analysis=lambda _: ("RPT-TEST", {"coating_zones": [
            _graded("1", "above", 250.0, 0.0), dict(_graded("2", "normal", 188.4, 0.0), overpaint_percent=44.9),
        ]}),
    )
    client = _client(tmp_path, pipeline=pipeline, reports=reports)
    common = {"user_id": "COPILOT-AI-TEST", "session_id": "gate-test", "scenario_id": "3", "task_type": "error_analysis"}
    headers = {"X-Copilot-Source": "assistant-test", "X-Copilot-Conversation": "zone-choice"}

    unsolicited = client.post("/api/v1/copilot/query", json={**common, "prompt": "1."}, headers=headers)
    assert unsolicited.headers["X-Copilot-Unavailable-Reason"] == "clarification_required"
    assert "과도막" not in unsolicited.json()["data"]["answer"]

    client.post("/api/v1/copilot/query", json={**common, "prompt": "이 구역의 문제점"}, headers=headers)
    selected = client.post("/api/v1/copilot/query", json={**common, "prompt": "2."}, headers=headers)
    answer = selected.json()["data"]["answer"]
    assert selected.json()["data"]["context_status"] == "available"
    assert "44.9%" in answer and "Zone 1" not in answer
    assert not pipeline.plans and not pipeline.runs


@pytest.mark.parametrize("prompt,task_type", [
    ("ㅏㅣㅏㅣㅏㅣ", "error_analysis"), ("000000000000", "knowledge_search"),
])
def test_uninterpretable_input_does_not_reuse_the_prior_training_topic(tmp_path, prompt, task_type):
    pipeline = Planner()
    reports = SimpleNamespace(
        get_session=lambda _: _session(),
        latest_complete_analysis=lambda _: ("RPT-TEST", {"coating_zones": [_graded("1", "above", 250.0, 0.0)]}),
    )
    client = _client(tmp_path, pipeline=pipeline, reports=reports)
    common = {"user_id": "COPILOT-AI-TEST", "session_id": "gate-test", "scenario_id": "3"}
    headers = {"X-Copilot-Source": "assistant-test", "X-Copilot-Conversation": "scope-test"}
    first = client.post("/api/v1/copilot/query", json={
        **common, "prompt": "전체 도막 평가", "task_type": "error_analysis",
    }, headers=headers)
    assert first.status_code == 200 and "과도막" in first.json()["data"]["answer"]

    response = client.post("/api/v1/copilot/query", json={
        **common, "prompt": prompt, "task_type": task_type,
    }, headers=headers)

    assert response.status_code == 200
    assert response.headers["X-Copilot-Unavailable-Reason"] == "clarification_required"
    data = response.json()["data"]
    assert "다시" in data["answer"] and "과도막" not in data["answer"]
    assert data["coating_context"] is None and not data["retrieval_used"]
    assert not pipeline.plans and not pipeline.runs


@pytest.mark.parametrize("prompt,context,expected,available", [
    ("존 투 문제점", None, "44.9%", True),
    ("2구역 문제점", None, "44.9%", True),
    ("2번 구역 문제점", None, "44.9%", True),
    ("이 구역 문제점", "2", "44.9%", True),
    ("이 구역 문제점", None, "구역 번호", False),
    ("두 번째 구역 문제점", None, "구역 번호", False),
    ("9번 구역 문제점", None, "9번 구역", False),
])
def test_zone_problem_answer_does_not_replace_that_zone_with_global_worst(
    tmp_path, prompt, context, expected, available,
):
    pipeline = Planner()
    zone = dict(_graded("2", "normal", 188.4, 0.0), overpaint_percent=44.9)
    reports = SimpleNamespace(
        get_session=lambda _: _session(),
        latest_complete_analysis=lambda _: ("RPT-TEST", {"coating_zones": [_graded("1", "above", 250.0, 0.0), zone]}),
    )
    client = _client(tmp_path, pipeline=pipeline, reports=reports)
    response = client.post("/api/v1/copilot/query", json={
        "user_id": "COPILOT-AI-TEST", "session_id": "gate-test", "scenario_id": "3",
        "task_type": "error_analysis", "prompt": prompt, "context": context,
    }, headers={"X-Copilot-Source": "assistant-test"})

    data = response.json()["data"]
    assert response.status_code == 200
    assert expected in data["answer"] and "Zone 1" not in data["answer"]
    assert data["context_status"] == ("available" if available else "not_available")
    if available:
        assert "정상" in data["answer"] and "부분" in data["answer"]
    else:
        assert response.headers["X-Copilot-Unavailable-Reason"]
    assert not data["retrieval_used"] and not pipeline.plans and not pipeline.runs
