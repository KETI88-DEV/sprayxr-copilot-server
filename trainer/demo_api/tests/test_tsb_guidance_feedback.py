"""TSB feedback: guidance identifies area, gives usable actions and speaks Korean Zone names."""
from __future__ import annotations

from pathlib import Path

import pytest

from trainer.demo_api import coating_assessment
from trainer.demo_api.tests.test_session_start_brief_response import Reference, _body, _client
from trainer.demo_api.tests.test_zone_completion_gate import analyze, request


def test_mixed_thickness_reports_overpaint_area_without_locating_unexposed_pixels(tmp_path: Path) -> None:
    # Seventy pixels are normal; thirty exceed 200µm after the single 10µm correction.
    # The painted mean remains normal, so this exercises needs_more rather than the stop-grade branch.
    data = analyze(tmp_path, request(value=171, valid_count=70, rest=240))
    progress = data["coating_progress"]
    assert progress["status"] == "needs_more"
    assert progress["coverage_percent"] == 70
    assert progress["overpaint_percent"] == 30
    assert progress["zone_id"] == "1" and progress["next_zone_id"] is None
    assert data["coating_feedback"]["grade"] == "normal"
    for text in (data["text_guide"], progress["message"]):
        assert "과도막 면적 30.0%" in text
        assert "분사 중지" in text
        assert not any(word in text for word in ("두꺼운 곳", "회피", "피하세요", "상한 초과 30.0%"))
        assert "1구역" in text and "Zone 1" not in text
    assert "160~200µm" in progress["message"]
    assert "1구역" in data["coating_feedback"]["message"]
    assert len(data["text_guide"]) <= 60


def test_unlocated_thickness_without_overpaint_does_not_instruct_pixel_avoidance(tmp_path: Path) -> None:
    data = analyze(tmp_path, request(valid_count=50))
    assert data["coating_progress"]["overpaint_percent"] == 0
    assert "보완" in data["text_guide"]
    for text in (data["text_guide"], data["coating_progress"]["message"]):
        assert not any(word in text for word in ("두꺼운 곳", "회피", "피하세요"))


def test_completed_guidance_uses_korean_zone_names_without_changing_machine_ids(tmp_path: Path) -> None:
    data = analyze(tmp_path, request(valid_count=95))
    progress = data["coating_progress"]
    assert progress["zone_id"] == "1" and progress["next_zone_id"] == "2"
    assert progress["status"] == "complete" and progress["recommendation"] == "move_next_zone"
    for text in (data["text_guide"], progress["message"]):
        assert "1구역" in text and "2구역" in text
        assert "Zone" not in text


def test_session_start_names_the_registered_zone_in_korean_without_changing_ids() -> None:
    client, _ = _client(Reference())
    result = client.post("/api/v1/sessions/start", json=_body(tool_type="spray_gun")).json()
    assert "8구역" in result["message"]
    guidance = result["data"]["first_zone_guidance"]
    assert guidance["zone_id"] == result["data"]["first_zone_id"] == "8"
    assert guidance["recommended_flow"] == ["8", "3"]
    assert "8구역" in guidance["message"] and "Zone" not in guidance["message"]


@pytest.mark.parametrize(("upper", "samples", "over"), [
    (None, [160.0, 200.0, 201.0], 100 / 3),
    (250.0, [160.0, 250.0, 251.0], 100 / 3),
])
def test_overpaint_area_uses_the_exclusive_upper_bound_not_the_target(upper, samples, over) -> None:
    limits = {"max_thickness_um": upper} if upper is not None else {}
    assessment = coating_assessment.assess_readings(samples, 160, **limits)
    assert assessment["normal_band_um"] == [160, 200 if upper is None else upper]
    assert assessment["overpaint_percent"] == pytest.approx(over)
    assert assessment["point_grades"] == ["normal", "normal", "above"]
