"""Current Zone replies use the latest visible sequence evidence."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.tests.test_query_unavailable_reason import _query as post_query, _unavailable
from trainer.demo_api.tests.test_scenario_reference_query import Pipeline, client, profile


USER_ID = "COPILOT-AI-TEST"
SESSION_ID = "current-zone-unit-test"


def _query(client, prompt, *, session_id=SESSION_ID):
    return post_query(client, prompt, session_id=session_id, scenario_id="3")


def _data(response):
    assert response.status_code == 200
    return response.json()["data"]


def _store(*zones, user_id=USER_ID, scenario_id="3"):
    store = SequenceStore(":memory:")
    for number, zone in enumerate(zones, start=1):
        sequence_id = f"SEQ-{number:03d}"
        store.record(
            session_key=SESSION_ID, sequence_id=sequence_id, body_sha256=sequence_id,
            frames=1, received_at="now", body=json.dumps({
                "session_id": SESSION_ID, "user_id": user_id, "scenario_id": scenario_id,
                "sequence_id": sequence_id, "zone_id": zone,
                "sequence_setting": {"tool_type": "spray_gun"},
            }),
        )
    return store


@pytest.mark.parametrize("prompt", [
    "방금 칠한 데가 몇 번 존이야?",
    "지금 분사한 곳은 몇 구역이야?",
    "지금 작업 중인 곳이 몇 번이죠?",
    "피도면 어디에 뿌리고 있었어요?",
    "현재 Zone은?",
    "지금 구역 번호는?",
])
def test_current_zone_paraphrases_report_latest_measurement(tmp_path, prompt):
    profile(tmp_path, "spray", ["1", "2", "3"])
    pipeline = Pipeline()
    response = _data(_query(client(tmp_path, store=_store("1", "2"), pipeline=pipeline), prompt))
    assert response["context_status"] == "available"
    assert response["sequence_context"]["current_zone_id"] == "2"
    assert response["sequence_context"]["sequence_id"] == "SEQ-002"
    assert "2번" in response["answer"] and "3번" not in response["answer"]
    assert pipeline.calls == []


def test_reference_first_zone_is_not_current_before_any_sequence(tmp_path):
    profile(tmp_path, "spray", ["2", "3", "1"])
    reports = SimpleNamespace(get_session=lambda _: SimpleNamespace(
        user_id=USER_ID, scenario_id="3", tool_type="spray_gun"))
    response = _query(client(tmp_path, reports=reports, pipeline=Pipeline()), "지금 작업 중인 곳이 몇 번이죠?")
    data = _unavailable(response, "sequence_missing", "분사 기록")
    assert data["sequence_context"]["current_zone_id"] is None
    assert "2번" not in data["answer"]


def test_current_zone_requires_a_training_session(tmp_path):
    pipeline = Pipeline()
    response = _query(client(tmp_path, pipeline=pipeline), "피도면 어디에 뿌리고 있었어요?", session_id=None)
    _unavailable(response, "session_required", "세션")
    assert pipeline.calls == []


def test_current_zone_does_not_promote_last_known_zone(tmp_path):
    response = _query(client(tmp_path, store=_store("2", "-1"), pipeline=Pipeline()),
                      "지금 뿌린 데가 어느 구역이지?")
    data = _unavailable(response, "zone_unresolved", "구역")
    assert data["sequence_context"]["current_zone_id"] is None
    assert data["sequence_context"]["sequence_id"] == "SEQ-002"
    assert "2번" not in data["answer"]


def test_explicit_last_known_zone_uses_earlier_located_sequence(tmp_path):
    response = _data(_query(client(tmp_path, store=_store("2", "-1"), pipeline=Pipeline()),
                           "마지막으로 확인된 존은 몇 번이야?"))
    assert response["context_status"] == "available"
    assert response["sequence_context"]["last_known_zone_id"] == "2"
    assert response["sequence_context"]["sequence_id"] == "SEQ-001"
    assert response["sequence_context"].get("current_zone_id") is None
    assert "2번 구역" in response["answer"]


def test_current_zone_does_not_read_another_workers_session(tmp_path):
    reports = SimpleNamespace(get_session=lambda _: SimpleNamespace(user_id="another-user", scenario_id="3"))
    response = _query(client(tmp_path, store=_store("2", user_id="another-user"), reports=reports, pipeline=Pipeline()),
                      "방금 칠한 데가 몇 번 존이야?")
    data = _unavailable(response, "session_mismatch", "일치하지 않습니다")
    assert "2번" not in data["answer"]


@pytest.mark.parametrize("prompt", [
    "지금 칠하는 데 어려운 점을 줄이는 일반적인 방법은?",
    "지금 어디를 칠하면 도료 낭비를 줄일 수 있나요?",
])
def test_current_zone_wording_does_not_take_over_how_to_questions(tmp_path, prompt):
    pipeline = Pipeline()
    data = _data(_query(client(tmp_path, store=_store("2"), pipeline=pipeline), prompt))
    assert (data["sequence_context"] or {}).get("scope") != "current_sequence"
    assert pipeline.calls


def test_progress_question_is_not_just_a_current_zone_number_lookup(tmp_path):
    data = _data(_query(client(tmp_path, store=_store("2"), pipeline=Pipeline()),
                        "지금 도장한 구역이 어느 정도 진행률이야?"))
    assert data["sequence_context"]["scope"] == "current_progress"


@pytest.mark.parametrize("prompt", [
    "매뉴얼에서 '방금 칠한 데가 몇 번 존이야?'는 대체 어떤 뜻인가요?",
    "예를 들어 '지금 작업 중인 곳이 몇 번이죠?'라는 문장의 의미는?",
    "쓰고 남은 폐유는 어디에 버려요?",
])
def test_manual_examples_and_unrelated_actions_do_not_use_current_zone(tmp_path, prompt):
    data = _data(_query(client(tmp_path, store=_store("2"), pipeline=Pipeline()), prompt))
    assert (data["sequence_context"] or {}).get("scope") != "current_sequence"
