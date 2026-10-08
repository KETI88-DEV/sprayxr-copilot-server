from __future__ import annotations

import json
import os
from pathlib import Path

from trainer.demo_api.expert_guidance import ExpertGuidanceEngine
from trainer.demo_api.telemetry_contracts import AnalyzeSequenceRequest


def _write_profile(
    root: Path,
    *,
    process: str = "spray",
    speed: float = 100.0,
    distance: float = 400.0,
) -> Path:
    target = root / "generated" / "profiles" / process / "learning" / "mode_03.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "identity": {"mode_id": 3, "process": process, "content_type": "learning"},
                "quality": {"status": "accepted", "session_count": 2},
                "aggregate": {
                    "zones": {
                        "1:1": {
                            "zone_id": "1",
                            "map_num": 1,
                            "speed_mms": {"count": 2, "median": speed, "q1": speed * 0.9, "q3": speed * 1.1},
                            "distance_mm": {
                                "count": 2,
                                "median": distance,
                                "q1": distance - 20,
                                "q3": distance + 20,
                            },
                        }
                    }
                },
                "sessions": [],
            }
        ),
        encoding="utf-8",
    )
    return target


def _request(
    *,
    session_id: str = "S-1",
    zone_id: str = "1",
    speed: float | None = 100.0,
    distance: float | None = 400.0,
    tool_type: str | None = "spray_gun",
    scenario_id: str = "3",
    mode_id: int | None = None,
    result_overrides: dict[str, object] | None = None,
) -> AnalyzeSequenceRequest:
    result: dict[str, object] = {}
    if speed is not None:
        result["avg_speed_mms"] = speed
    if distance is not None:
        result["avg_distance_mm"] = distance
    result.update(result_overrides or {})
    sequence_setting = {"tool_type": tool_type} if tool_type is not None else {}
    return AnalyzeSequenceRequest(
        session_id=session_id,
        user_id="U-1",
        scenario_id=scenario_id,
        zone_id=zone_id,
        sequence_id=f"Q-{session_id}-{speed}-{distance}",
        sequence_setting=sequence_setting,
        sequence_info=[],
        sequence_result=result,
        mode_id=mode_id,
    )


def test_scenario_id_selects_the_learning_profile_without_mode_id(tmp_path: Path) -> None:
    _write_profile(tmp_path)

    request = _request()
    guide = ExpertGuidanceEngine(tmp_path).evaluate(request)

    assert request.mode_id is None
    assert request.resolved_mode_id == 3
    assert request.process_type == "spray"
    assert guide.speed.status == guide.distance.status == "ok"


def test_tool_type_selects_the_blasting_profile(tmp_path: Path) -> None:
    _write_profile(tmp_path, process="blasting", speed=200.0, distance=600.0)

    guide = ExpertGuidanceEngine(tmp_path).evaluate(
        _request(tool_type="blasting", speed=200.0, distance=600.0)
    )

    assert guide.speed.status == guide.distance.status == "ok"


def test_established_tr_learning_id_selects_the_profile(tmp_path: Path) -> None:
    _write_profile(tmp_path)

    request = _request(scenario_id="TR-L03")
    guide = ExpertGuidanceEngine(tmp_path).evaluate(request)

    assert request.resolved_mode_id == 3
    assert guide.speed.status == guide.distance.status == "ok"


def test_legacy_explicit_mode_without_tool_type_keeps_the_spray_default(tmp_path: Path) -> None:
    _write_profile(tmp_path)

    request = _request(scenario_id="legacy-scene", mode_id=3, tool_type=None)
    guide = ExpertGuidanceEngine(tmp_path).evaluate(request)

    assert request.process_type == "spray"
    assert guide.speed.status == guide.distance.status == "ok"


def test_classifies_speed_and_distance_against_the_zone_profile(tmp_path: Path) -> None:
    _write_profile(tmp_path)
    engine = ExpertGuidanceEngine(tmp_path)

    low_near = engine.evaluate(_request(session_id="LOW", speed=50, distance=300))
    ok = engine.evaluate(_request(session_id="OK", speed=100, distance=400))
    high_far = engine.evaluate(_request(session_id="HIGH", speed=160, distance=520))

    assert low_near.speed.status == "slow"
    assert low_near.distance.status == "near"
    assert ok.speed.status == "ok"
    assert ok.distance.status == "ok"
    assert high_far.speed.status == "fast"
    assert high_far.distance.status == "far"
    assert all(item.available for guide in (low_near, ok, high_far) for item in (guide.speed, guide.distance))


def test_missing_zone_does_not_invent_ok(tmp_path: Path) -> None:
    _write_profile(tmp_path)

    guide = ExpertGuidanceEngine(tmp_path).evaluate(_request(zone_id="39"))

    assert guide.speed.model_dump() == {
        "status": None,
        "message": None,
        "available": False,
        "reason": "reference_unavailable",
    }
    assert guide.distance.reason == "reference_unavailable"


def test_explicit_metric_units_are_normalized_and_conflicts_are_rejected(tmp_path: Path) -> None:
    _write_profile(tmp_path)
    engine = ExpertGuidanceEngine(tmp_path)

    converted = engine.evaluate(
        _request(
            session_id="METRES",
            speed=None,
            distance=None,
            result_overrides={"avg_speed_mps": 0.1, "avg_distance_m": 0.4},
        )
    )
    invalid = engine.evaluate(
        _request(
            session_id="BAD-UNITS",
            speed=100,
            distance=400,
            result_overrides={"avg_speed_mps": 9.9},
        )
    )

    assert converted.speed.status == converted.distance.status == "ok"
    assert invalid.speed.status is None
    assert invalid.speed.reason == "invalid_units"
    assert invalid.distance.status == "ok"


def test_a_single_opposite_sample_does_not_flip_the_stable_guide(tmp_path: Path) -> None:
    _write_profile(tmp_path)
    engine = ExpertGuidanceEngine(tmp_path)

    first = engine.evaluate(_request(session_id="SMOOTH", speed=50))
    one_fast = engine.evaluate(_request(session_id="SMOOTH", speed=180))
    two_fast = engine.evaluate(_request(session_id="SMOOTH", speed=180))
    one_ok = engine.evaluate(_request(session_id="SMOOTH", speed=100))
    two_ok = engine.evaluate(_request(session_id="SMOOTH", speed=100))

    assert first.speed.status == "slow"
    assert one_fast.speed.status == "slow"
    assert two_fast.speed.status == "fast"
    assert one_ok.speed.status == "fast"
    assert two_ok.speed.status == "ok"


def test_profile_reload_is_atomic_and_keeps_the_last_good_version_on_corruption(tmp_path: Path) -> None:
    profile = _write_profile(tmp_path, speed=100)
    engine = ExpertGuidanceEngine(tmp_path)
    assert engine.evaluate(_request(session_id="BEFORE", speed=100)).speed.status == "ok"

    _write_profile(tmp_path, speed=200)
    os.utime(profile, None)
    assert engine.evaluate(_request(session_id="AFTER", speed=100)).speed.status == "slow"

    profile.write_text("{broken", encoding="utf-8")
    os.utime(profile, None)
    fallback = engine.evaluate(_request(session_id="BROKEN", speed=200))
    assert fallback.speed.status == "ok"
    assert fallback.speed.reason is None


def test_first_load_of_a_corrupt_profile_is_unavailable(tmp_path: Path) -> None:
    profile = _write_profile(tmp_path)
    profile.write_text("{broken", encoding="utf-8")

    guide = ExpertGuidanceEngine(tmp_path).evaluate(_request())

    assert guide.speed.available is False
    assert guide.speed.reason == "reference_invalid"
