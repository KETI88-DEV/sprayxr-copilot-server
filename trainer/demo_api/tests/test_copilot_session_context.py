from __future__ import annotations

import json

import pytest

from trainer.demo_api.copilot_session_context import summarize_session
from trainer.demo_api.v1_store import SessionRow


def _sequence(sequence_id: str, zone_id: str | None, *, speed: float, distance: float) -> dict[str, object]:
    return {
        "session_id": "3_11000346_1",
        "scenario_id": "3",
        "sequence_id": sequence_id,
        "zone_id": zone_id,
        "sequence_setting": {"tool_type": "spray_gun"},
        "sequence_info": [
            {"index": index, "gun_metrics": {"raytexuv": [100 + index, 200]}}
            for index in range(200)
        ],
        "sequence_result": {
            "avg_speed_mms": speed,
            "avg_distance_mm": distance,
            "avg_thickness_um": 142.5,
            "cur_sequence_heatmap": [{"data": "very-large-base64-payload"}],
        },
    }


def test_session_summary_keeps_accumulated_order_metrics_without_raw_frames() -> None:
    """The LLM receives the whole-session facts, never bulky 90/120 Hz frame payloads."""
    summary = summarize_session([
        _sequence("SEQ-001", "1", speed=420.0, distance=245.0),
        _sequence("SEQ-002", None, speed=460.0, distance=255.0),
        _sequence("SEQ-003", "3", speed=500.0, distance=265.0),
        _sequence("SEQ-004", "3", speed=480.0, distance=260.0),
    ])

    decoded = json.loads(summary)
    assert decoded == {
        "scenario_id": "3",
        "process": "spray",
        "sequence_count": 4,
        "latest_sequence_id": "SEQ-004",
        "current_zone": "3",
        "zone_visit_order": ["1", "3"],
        "zone_sequence_counts": {"1": 1, "3": 2},
        "session_averages": {
            "avg_speed_mms": "465 mm/s",
            "avg_distance_mm": "256.25 mm",
        },
        "recent_sequences": [
            {"sequence_id": "SEQ-002", "zone_id": None, "avg_speed_mms": "460 mm/s", "avg_distance_mm": "255 mm", "avg_thickness_um": "142.5 µm"},
            {"sequence_id": "SEQ-003", "zone_id": "3", "avg_speed_mms": "500 mm/s", "avg_distance_mm": "265 mm", "avg_thickness_um": "142.5 µm"},
            {"sequence_id": "SEQ-004", "zone_id": "3", "avg_speed_mms": "480 mm/s", "avg_distance_mm": "260 mm", "avg_thickness_um": "142.5 µm"},
        ],
    }
    # 단위를 값에 붙여 넘긴다 — 키 이름에서 단위를 추측하게 두면 모델이 mm/s를 m/s로 쓴다(2026-11-11 Spark 27B 8건 중 7건).
    assert "sequence_info" not in summary
    assert "very-large-base64-payload" not in summary


def _session(**settings) -> SessionRow:
    return SessionRow(
        **{
            "session_id": "2_AI-TEST-QUERY-TARGET",
            "user_id": "COPILOT-AI-TEST",
            "scenario_id": "2",
            "session_status": "STARTED",
            "started_at": None,
            "ended_at": None,
            "target_thickness_um": 160.0,
            "target_basis": "session_target",
            "tool_type": "spray_gun",
            **settings,
        },
    )


def test_session_coating_target_lookup_corrects_an_incorrect_premise() -> None:
    from trainer.demo_api import copilot_session_context

    answer, scope = copilot_session_context.session_coating_target_answer(
        "이번 훈련 목표가 200 마이크론 맞지? 녹색 범위 말고 목표값만.",
        _session(),
    )
    assert "160µm" in answer and "200" not in answer
    assert scope == {
        "scope": "session_target",
        "analysis_status": "available",
        "target_thickness_um": 160.0,
        "normal_band_um": [160.0, 200.0],
        "target_basis": "session_target",
    }


def test_session_coating_target_lookup_returns_requested_upper_bound() -> None:
    from trainer.demo_api import copilot_session_context

    answer, scope = copilot_session_context.session_coating_target_answer(
        "다음 존 번호는 물어본 게 아니고, 지금 보내드린 두께 목표의 윗값만 알고 싶어요.",
        _session(target_thickness_min_um=175.0, target_thickness_max_um=225.0),
    )
    assert "상한" in answer and "225µm" in answer and "175" not in answer
    assert scope["target_thickness_um"] == 175.0
    assert scope["normal_band_um"] == [175.0, 225.0]


def test_session_coating_target_lookup_labels_the_default_band() -> None:
    from trainer.demo_api import copilot_session_context

    answer, scope = copilot_session_context.session_coating_target_answer(
        "이 세션의 정상 도막 범위는 몇 마이크로미터인가요?",
        _session(target_basis="default_target"),
    )
    assert "기본" in answer and "160~200µm" in answer
    assert scope["target_basis"] == "default_target"


@pytest.mark.parametrize("prompt", [
    "내가 칠한 도막 두께가 목표에 도달했나요?",
    "도막 목표값은 어떻게 설정하나요?",
    "지금 실제 도막 두께랑 목표 도막 두께를 모두 알려줘.",
    "다음 구역 목표로 이동해도 될까요?",
    "그라인딩 작업에서는 목표 도막 두께가 얼마야?",
    "웜업 중인데 목표까지 몇 번 더 뿌려야 해?",
])
def test_session_coating_target_lookup_leaves_other_intents_to_their_own_routes(prompt) -> None:
    from trainer.demo_api import copilot_session_context

    assert copilot_session_context.session_coating_target_answer(prompt, _session()) is None


def test_session_coating_target_lookup_requires_spray_configuration() -> None:
    from trainer.demo_api import copilot_session_context

    prompt = "설정된 목표 도막 두께가 얼마예요?"
    assert copilot_session_context.session_coating_target_answer(prompt, None) is None
    assert copilot_session_context.session_coating_target_answer(prompt, _session(tool_type="blasting_gun")) is None
