from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from trainer.zone_engine.assets import ZoneAssetCatalog
from trainer.zone_engine.contracts import EventRequest, parse_payload
from trainer.zone_engine.engine import CorrectedZoneEngine
from trainer.zone_engine.errors import ConflictError, EngineError

from .conftest import event, height_png, start_event


def _engine(path: Path, catalog: ZoneAssetCatalog) -> CorrectedZoneEngine:
    return CorrectedZoneEngine(
        path,
        assets=catalog,
        clock=lambda: datetime(2026, 11, 6, 1, 0, tzinfo=timezone.utc),
    )


def _sequence_payload(
    *,
    captured_at: str = "2026-11-06T00:00:09Z",
    frame_time: str = "2026-11-06T00:00:08Z",
    zone_id: str = "1",
    speed: float = 300.0,
    distance: float = 250.0,
    cumulative: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "sequence_id": f"sequence-{captured_at}",
        "face_id": "face-1",
        "map_num": 1,
        "round_id": "round-1",
        "captured_at": captured_at,
        "sequence_started_at": None,
        "frames": [
            {
                "timestamp": frame_time,
                "relative_time_s": None,
                "zone_id": zone_id,
                "spray_on": True,
                "speed_mm_s": speed,
                "distance_mm": distance,
                "raytexmapnum": 1,
                "raytexuv": None,
            }
        ],
        "cumulative_crop": cumulative,
    }


def _observation_payload(
    *,
    zone_id: str = "1",
    snapshot_id: str = "snapshot-1",
    snapshot_revision: int = 1,
    captured_at: str = "2026-11-06T00:00:10Z",
    image: dict[str, Any] | None = None,
    round_id: str = "round-1",
) -> dict[str, Any]:
    return {
        "face_id": "face-1",
        "map_num": 1,
        "zone_id": zone_id,
        "round_id": round_id,
        "snapshot_id": snapshot_id,
        "snapshot_revision": snapshot_revision,
        "captured_at": captured_at,
        "height_png": image or height_png(),
    }


def _end_payload(at: str = "2026-11-06T00:00:20Z") -> dict[str, Any]:
    return {"ended_at": at}


def _complete_payload(avg: float | None = 138.91) -> dict[str, Any]:
    return {
        "completed_at": "2026-11-06T00:00:21Z",
        "provider_official_summary": {
            "avg_thickness_um": avg,
            "source": "provider_completion_payload",
            "unit": "um",
            "progress_value": None,
            "progress_unit": None,
        },
    }


def test_event_idempotency_conflict_and_restart_persistence(
    tmp_path: Path, asset_catalog: ZoneAssetCatalog
) -> None:
    db = tmp_path / "engine.db"
    engine = _engine(db, asset_catalog)
    start = start_event(asset_catalog, expected_zone_ids=["1"])
    accepted = engine.process_event(start)
    duplicate = engine.process_event(start)
    assert accepted["status"] == "accepted"
    assert accepted["session_revision"] == 0
    assert duplicate["status"] == "duplicate"

    changed = start.model_copy(deep=True)
    changed.payload["user_id"] = "different-user"
    with pytest.raises(ConflictError, match="different canonical JSON"):
        engine.process_event(changed)

    observation = event(
        "event-zone-1", "zone_observation", _observation_payload()
    )
    response = engine.process_event(observation)
    assert response["session_revision"] == 1
    assessment_id = response["assessment_id"]
    before_restart = engine.get_assessment(assessment_id)

    restarted = _engine(db, asset_catalog)
    after_restart = restarted.get_assessment(assessment_id)
    assert after_restart == before_restart
    assert restarted.get_session("session-1")["event_counts"] == {
        "session_start": 1,
        "zone_observation": 1,
    }


def test_behavior_evidence_uses_only_spray_frames_before_capture_and_stays_noncausal(
    tmp_path: Path, asset_catalog: ZoneAssetCatalog
) -> None:
    engine = _engine(tmp_path / "engine.db", asset_catalog)
    engine.process_event(start_event(asset_catalog, target_um=160.0, expected_zone_ids=["1"]))
    payload = _sequence_payload(captured_at="2026-11-06T00:00:12Z")
    payload["frames"] = [
        {
            "timestamp": "2026-11-06T00:00:08Z",
            "relative_time_s": None,
            "zone_id": "1",
            "spray_on": True,
            "speed_mm_s": 300.0,
            "distance_mm": 250.0,
            "raytexmapnum": 1,
            "raytexuv": None,
        },
        {
            "timestamp": "2026-11-06T00:00:09Z",
            "relative_time_s": None,
            "zone_id": "1",
            "spray_on": False,
            "speed_mm_s": 999.0,
            "distance_mm": 999.0,
            "raytexmapnum": 1,
            "raytexuv": None,
        },
        {
            "timestamp": "2026-11-06T00:00:11Z",
            "relative_time_s": None,
            "zone_id": "1",
            "spray_on": True,
            "speed_mm_s": 700.0,
            "distance_mm": 700.0,
            "raytexmapnum": 1,
            "raytexuv": None,
        },
    ]
    engine.process_event(event("event-sequence", "sequence", payload))
    below_png = height_png(rgb=(1, 70, 255))
    response = engine.process_event(
        event(
            "event-zone",
            "zone_observation",
            _observation_payload(image=below_png),
        )
    )
    assessment = engine.get_assessment(response["assessment_id"])
    assert assessment["quality"]["grade"] == "below"
    stats = assessment["metrics"]["behavior_statistics"]
    assert stats["speed_mm_s"]["count"] == 1
    assert stats["speed_mm_s"]["mean"] == 300.0
    assert stats["distance_mm"]["mean"] == 250.0
    assert assessment["coaching"]["causes"]
    for cause in assessment["coaching"]["causes"]:
        assert cause["causal_status"] == "candidate_not_established"
    for action in assessment["coaching"]["actions"]:
        assert action["target_range"] is None
        assert action["expected_effect"] is None
        assert "기대 효과나 최적값은 확인되지 않았습니다" in action["statement"]
    assert all(
        any("가" <= character <= "힣" for character in item["statement"])
        for item in assessment["evidence"]
    )


def test_sequence_rejects_explicit_zone_missing_from_exact_face(
    tmp_path: Path, asset_catalog: ZoneAssetCatalog
) -> None:
    engine = _engine(tmp_path / "engine.db", asset_catalog)
    engine.process_event(start_event(asset_catalog))
    with pytest.raises(EngineError) as caught:
        engine.process_event(
            event(
                "event-sequence-unknown-zone",
                "sequence",
                _sequence_payload(zone_id="999"),
            )
        )
    assert caught.value.reason_code == "zone_id_unknown"


def test_session_family_duplicate_round_has_stable_conflict(
    tmp_path: Path, asset_catalog: ZoneAssetCatalog
) -> None:
    engine = _engine(tmp_path / "engine.db", asset_catalog)
    engine.process_event(start_event(asset_catalog))
    duplicate_round = start_event(
        asset_catalog,
        session_id="session-2",
        event_id="event-start-2",
    )
    with pytest.raises(ConflictError) as caught:
        engine.process_event(duplicate_round)
    assert caught.value.reason_code == "round_index_conflict"


def test_session_start_rejects_non_utc_offset(
    asset_catalog: ZoneAssetCatalog,
) -> None:
    for invalid in (
        "2026-11-06T09:00:00+09:00",
        "2026-11-06T00:00:00-00:00",
    ):
        payload = start_event(asset_catalog).model_dump(mode="json")
        payload["payload"]["started_at"] = invalid
        with pytest.raises(ValueError, match=r"must use UTC"):
            parse_payload(EventRequest.model_validate(payload))


def test_rework_revisions_are_preserved_and_latest_partial_does_not_fallback(
    tmp_path: Path, asset_catalog: ZoneAssetCatalog
) -> None:
    engine = _engine(tmp_path / "engine.db", asset_catalog)
    engine.process_event(start_event(asset_catalog, expected_zone_ids=["2"]))
    first = engine.process_event(
        event(
            "event-zone-full",
            "zone_observation",
            _observation_payload(zone_id="2", snapshot_id="full"),
        )
    )
    partial_image = height_png(width=1, height=1, x0=0, y0=1)
    second = engine.process_event(
        event(
            "event-zone-partial",
            "zone_observation",
            _observation_payload(
                zone_id="2",
                snapshot_id="rework",
                snapshot_revision=1,
                captured_at="2026-11-06T00:00:11Z",
                image=partial_image,
            ),
        )
    )
    assert first["assessment_id"] != second["assessment_id"]
    assert engine.get_assessment(first["assessment_id"])["revision"] == 1
    latest = engine.get_assessment(second["assessment_id"])
    assert latest["revision"] == 2
    assert latest["status"] == "partial"
    assert latest["quality"]["grade"] is None
    assert engine.get_session("session-1")["zone_revision_index"][0]["revision"] == 2


def test_delayed_pre_end_capture_is_allowed_and_post_end_capture_is_rejected(
    tmp_path: Path, asset_catalog: ZoneAssetCatalog
) -> None:
    engine = _engine(tmp_path / "engine.db", asset_catalog)
    engine.process_event(start_event(asset_catalog, expected_zone_ids=["1"]))
    engine.process_event(event("event-end", "session_end", _end_payload("2026-11-06T00:00:10Z")))
    delayed = engine.process_event(
        event(
            "event-delayed",
            "zone_observation",
            _observation_payload(captured_at="2026-11-06T00:00:09Z"),
        )
    )
    assert delayed["status"] == "accepted"
    with pytest.raises(ConflictError, match="post-end capture"):
        engine.process_event(
            event(
                "event-post-end",
                "zone_observation",
                _observation_payload(
                    snapshot_id="late",
                    captured_at="2026-11-06T00:00:11Z",
                ),
            )
        )
    assert engine.get_session("session-1")["last_event_order"] == 3


def test_sequence_cumulative_crop_revision_overwrites_raw_pixels(
    tmp_path: Path, asset_catalog: ZoneAssetCatalog
) -> None:
    engine = _engine(tmp_path / "engine.db", asset_catalog)
    engine.process_event(start_event(asset_catalog))
    first_spec = height_png(rgb=(1, 70, 255), basis="cumulative_crop")
    second_spec = height_png(rgb=(2, 173, 255), basis="cumulative_crop")
    first_crop = {
        "snapshot_id": "cumulative-1",
        "snapshot_revision": 1,
        "height_png": first_spec,
    }
    second_crop = {
        "snapshot_id": "cumulative-1",
        "snapshot_revision": 2,
        "height_png": second_spec,
    }
    engine.process_event(
        event(
            "event-seq-1",
            "sequence",
            _sequence_payload(cumulative=first_crop),
        )
    )
    engine.process_event(
        event(
            "event-seq-2",
            "sequence",
            _sequence_payload(
                captured_at="2026-11-06T00:00:11Z",
                frame_time="2026-11-06T00:00:10Z",
                cumulative=second_crop,
            ),
        )
    )
    state = engine.get_session("session-1")["cumulative_crop_states"][0]
    expected_per_pixel = (2 / 255.0 + 173 / 65025.0) * 20000.0
    assert state["known_pixel_count"] == 16
    assert math.isclose(state["raw_sum_um"], expected_per_pixel * 16, rel_tol=1e-12)
    first_per_pixel = (1 / 255.0 + 70 / 65025.0) * 20000.0
    assert not math.isclose(
        state["raw_sum_um"], (first_per_pixel + expected_per_pixel) * 16, rel_tol=1e-12
    )


def test_session_aggregate_is_pixel_weighted_and_preserves_official_mean(
    tmp_path: Path, asset_catalog: ZoneAssetCatalog
) -> None:
    engine = _engine(tmp_path / "engine.db", asset_catalog)
    engine.process_event(
        start_event(asset_catalog, target_um=80.0, expected_zone_ids=["1", "2"])
    )
    zone_one_rgb = (1, 70, 255)
    zone_two_rgb = (2, 173, 255)
    first = engine.process_event(
        event(
            "event-zone-1",
            "zone_observation",
            _observation_payload(zone_id="1", image=height_png(rgb=zone_one_rgb)),
        )
    )
    second = engine.process_event(
        event(
            "event-zone-2",
            "zone_observation",
            _observation_payload(
                zone_id="2",
                snapshot_id="snapshot-2",
                captured_at="2026-11-06T00:00:11Z",
                image=height_png(rgb=zone_two_rgb),
            ),
        )
    )
    first_mean = engine.get_assessment(first["assessment_id"])["quality"]["painted_mean_um"]
    second_mean = engine.get_assessment(second["assessment_id"])["quality"]["painted_mean_um"]
    engine.process_event(event("event-end", "session_end", _end_payload()))
    complete = engine.process_event(
        event("event-complete", "session_complete", _complete_payload())
    )
    assessment = engine.get_assessment(complete["assessment_id"])
    expected_weighted = (first_mean + 9 * second_mean) / 10
    assert math.isclose(
        assessment["quality"]["painted_mean_um"], expected_weighted, rel_tol=1e-12
    )
    assert not math.isclose(
        assessment["quality"]["painted_mean_um"], (first_mean + second_mean) / 2
    )
    assert assessment["metrics"]["provider_official_summary"]["avg_thickness_um"] == 138.91
    assert len(assessment["zone_assessments"]) == 2
    aggregate_evidence, official_evidence, lineage_evidence = assessment["evidence"][:3]
    assert aggregate_evidence["evidence_id"].startswith("ev-session-aggregate-")
    assert aggregate_evidence["value"] == assessment["quality"]["painted_mean_um"]
    assert aggregate_evidence["unit"] == "um"
    assert "유효 도포 픽셀 10개" in aggregate_evidence["statement"]
    assert official_evidence["evidence_id"].startswith("ev-provider-official-")
    assert official_evidence["value"] == 138.91
    assert "재현값이 아닙니다" in official_evidence["statement"]
    assert lineage_evidence["evidence_type"] == "lineage"
    zone_evidence_ids = {
        evidence["evidence_id"]
        for result in (first, second)
        for evidence in engine.get_assessment(result["assessment_id"])["evidence"]
    }
    session_evidence_by_id = {
        evidence["evidence_id"]: evidence for evidence in assessment["evidence"]
    }
    assert all(
        session_evidence_by_id[evidence_id]["statement"].startswith(
            "면 face-1, map 1, Zone "
        )
        for evidence_id in zone_evidence_ids
    )


def test_session_coaching_scopes_opposing_directions_to_each_zone(
    tmp_path: Path, asset_catalog: ZoneAssetCatalog
) -> None:
    engine = _engine(tmp_path / "engine.db", asset_catalog)
    engine.process_event(
        start_event(asset_catalog, target_um=160.0, expected_zone_ids=["1", "2"])
    )
    sequence = _sequence_payload(captured_at="2026-11-06T00:00:07Z")
    sequence["frames"] = [
        {
            "timestamp": "2026-11-06T00:00:05Z",
            "relative_time_s": None,
            "zone_id": "1",
            "spray_on": True,
            "speed_mm_s": 300.0,
            "distance_mm": 250.0,
            "raytexmapnum": 1,
            "raytexuv": None,
        },
        {
            "timestamp": "2026-11-06T00:00:06Z",
            "relative_time_s": None,
            "zone_id": "2",
            "spray_on": True,
            "speed_mm_s": 400.0,
            "distance_mm": 300.0,
            "raytexmapnum": 1,
            "raytexuv": None,
        },
    ]
    engine.process_event(event("event-sequence", "sequence", sequence))
    below_result = engine.process_event(
        event(
            "event-zone-below",
            "zone_observation",
            _observation_payload(zone_id="1", image=height_png(rgb=(1, 70, 255))),
        )
    )
    above_result = engine.process_event(
        event(
            "event-zone-above",
            "zone_observation",
            _observation_payload(
                zone_id="2",
                snapshot_id="snapshot-2",
                captured_at="2026-11-06T00:00:11Z",
                image=height_png(rgb=(3, 48, 255)),
            ),
        )
    )
    engine.process_event(event("event-end", "session_end", _end_payload()))
    completed = engine.process_event(
        event("event-complete", "session_complete", _complete_payload())
    )
    assessment = engine.get_assessment(completed["assessment_id"])

    actions = assessment["coaching"]["actions"]
    causes = assessment["coaching"]["causes"]
    zone_action_ids = {
        action["action_id"]
        for result in (below_result, above_result)
        for action in engine.get_assessment(result["assessment_id"])["coaching"]["actions"]
    }
    assert {item["action_id"] for item in actions} == zone_action_ids
    assert {item["direction"] for item in actions} == {"decrease", "increase"}
    assert all(item["statement"].startswith("면 face-1, map 1, Zone ") for item in actions)
    assert all(item["statement"].startswith("면 face-1, map 1, Zone ") for item in causes)
    assert all(
        "Zone 1 관측" in item["statement"]
        for item in actions
        if item["direction"] == "decrease"
    )
    assert all(
        "Zone 2 관측" in item["statement"]
        for item in actions
        if item["direction"] == "increase"
    )


def test_missing_target_withholds_grade_and_coaching(
    tmp_path: Path, asset_catalog: ZoneAssetCatalog
) -> None:
    engine = _engine(tmp_path / "engine.db", asset_catalog)
    engine.process_event(
        start_event(asset_catalog, target_um=None, expected_zone_ids=["1"])
    )
    result = engine.process_event(
        event("event-zone", "zone_observation", _observation_payload())
    )
    assessment = engine.get_assessment(result["assessment_id"])
    assert assessment["quality"]["grade"] is None
    assert assessment["quality"]["analysis_status"] == "unavailable"
    assert assessment["coaching"]["causes"] == []
    assert assessment["coaching"]["actions"] == []


def test_current_round_height_sums_aligned_family_rounds_before_single_dv(
    tmp_path: Path, asset_catalog: ZoneAssetCatalog
) -> None:
    engine = _engine(tmp_path / "engine.db", asset_catalog)
    engine.process_event(
        start_event(
            asset_catalog,
            session_id="round-session-1",
            event_id="start-round-1",
            expected_zone_ids=["1"],
            round_index=1,
            round_id="round-1",
            family_id="family-rounds",
        )
    )
    first_rgb = (1, 70, 255)
    first = event(
        "zone-round-1",
        "zone_observation",
        _observation_payload(round_id="round-1", image=height_png(rgb=first_rgb)),
        session_id="round-session-1",
    )
    engine.process_event(first)

    engine.process_event(
        start_event(
            asset_catalog,
            session_id="round-session-2",
            event_id="start-round-2",
            expected_zone_ids=["1"],
            round_index=2,
            round_id="round-2",
            family_id="family-rounds",
        )
    )
    second_rgb = (1, 5, 255)
    result = engine.process_event(
        event(
            "zone-round-2",
            "zone_observation",
            _observation_payload(
                round_id="round-2",
                image=height_png(rgb=second_rgb),
            ),
            session_id="round-session-2",
        )
    )
    assessment = engine.get_assessment(result["assessment_id"])
    raw_one = (first_rgb[0] / 255.0 + first_rgb[1] / 65025.0) * 20000.0
    raw_two = (second_rgb[0] / 255.0 + second_rgb[1] / 65025.0) * 20000.0
    expected = min(raw_one + raw_two - 10.0, 1600.0)
    assert math.isclose(
        assessment["quality"]["painted_mean_um"], expected, rel_tol=1e-12
    )
    assert {"zone-round-1", "zone-round-2"}.issubset(
        assessment["provenance"]["source_event_ids"]
    )
