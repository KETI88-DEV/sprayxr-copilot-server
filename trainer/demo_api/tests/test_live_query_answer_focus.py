"""Live Query includes the requested fact and any material coating warning."""

from types import SimpleNamespace

import pytest

from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.tests.test_scenario_reference_query import client, profile, query
from trainer.demo_api.tests.test_zone_completion_gate import engine, request


def _live_query(tmp_path, prompt, *, thickness=171, width=10, summarize=False):
    coating = engine(tmp_path)
    sequence = request(value=thickness, width=width)
    store = SequenceStore(":memory:")
    store.record(session_key=sequence.session_id, sequence_id=sequence.sequence_id,
                 body_sha256=sequence.sequence_id, frames=0, received_at="now",
                 body=sequence.model_dump_json())
    return query(client(tmp_path, store=store, coating_engine=coating),
                 session_id=sequence.session_id, user_id=sequence.user_id, scenario_id=sequence.scenario_id,
                 prompt=prompt, summarize=summarize)


def test_next_zone_before_spraying_answers_the_first_recommendation_only(tmp_path):
    profile(tmp_path, "spray", ["2", "3", "1"])
    reports = SimpleNamespace(get_session=lambda _: SimpleNamespace(
        user_id="COPILOT-AI-TEST", scenario_id="3", tool_type="spray_gun"))
    data = query(client(tmp_path, reports=reports), session_id="started", user_id="COPILOT-AI-TEST",
                 prompt="다음 순서는 뭐지?", summarize=False)
    assert data["sequence_context"]["scope"] == "reference_only"
    assert data["sequence_context"]["next_zone_id"] == "2"
    assert data["coating_context"] is None
    assert "2" in data["answer"] and "→" not in data["answer"]


@pytest.mark.parametrize("summarize", [True, False])
def test_progress_only_reads_the_stored_progress_without_unsolicited_next_zone(tmp_path, summarize):
    data = _live_query(tmp_path, "지금 구역 진행률은 몇 퍼센트야?", summarize=summarize)
    assert data["answer"] == "현재 1구역 진행률은 100.00%입니다."
    assert data["sequence_context"]["next_zone_id"] == "2"
    assert data["coating_context"]["completion_percent"] == 100


def test_area_lookup_names_its_target_without_adding_progress_or_movement(tmp_path):
    data = _live_query(tmp_path, "현재 목표 두께 이상인 면적은 얼마야?")
    assert "160µm" in data["answer"] and "100.00%" in data["answer"]
    assert "면적" in data["answer"]
    assert "진행률" not in data["answer"] and "이동" not in data["answer"]
    assert data["coating_context"]["coverage_percent"] == 100


def test_next_zone_does_not_prepend_the_normal_band_to_the_destination(tmp_path):
    data = _live_query(tmp_path, "다음 순서는 뭐지?")
    assert data["answer"] == "2구역으로 이동하세요."
    assert data["sequence_context"]["next_zone_id"] == "2"


def test_completion_only_reports_completion_without_extra_destination(tmp_path):
    data = _live_query(tmp_path, "지금 구역 다 칠했어?")
    assert "1구역" in data["answer"] and "완료" in data["answer"]
    assert "2구역" not in data["answer"] and "정상범위" not in data["answer"]


def test_move_permission_keeps_overpaint_warning(tmp_path):
    data = _live_query(tmp_path, "이 정도면 옆 칸으로 넘어가도 될까?", thickness=250, width=5)
    assert data["sequence_context"]["next_zone_id"] is None
    assert data["coating_context"]["recommendation"] == "stop_current_zone"
    assert "과도막" in data["answer"] and "분사를 멈추고" in data["answer"]
    assert "부족한" in data["answer"] and "정상범위" not in data["answer"]


def test_explicit_progress_coverage_and_movement_request_keeps_all_facts(tmp_path):
    data = _live_query(tmp_path, "지금 구역 진행률과 목표 두께 이상인 면적, 다음 구역 이동 가능 여부를 알려줘.")
    assert all(fragment in data["answer"] for fragment in ("진행률", "면적", "160µm", "100.00%", "2구역"))
