from __future__ import annotations

import json

from trainer.demo_api.copilot_session_context import summarize_session


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
            "avg_thickness_um": "142.5 µm",
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
