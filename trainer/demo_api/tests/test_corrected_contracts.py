from __future__ import annotations

import base64
import hashlib

import pytest
from pydantic import ValidationError

from trainer.demo_api.engine_contracts import (
    CONTRACT_VERSION,
    EVALUATION_MODE,
    HeightPng,
    SequenceRequest,
    SessionStartRequest,
    ZoneObservationRequest,
    coating_progress_from_engine_state,
    engine_event_from_request,
)


def _manifest_hash() -> str:
    return "a" * 64


def _start(**updates: object) -> dict[str, object]:
    body: dict[str, object] = {
        "event_id": "evt-start-1",
        "session_id": "session-1",
        "contract_version": CONTRACT_VERSION,
        "user_id": "user-1",
        "scenario_id": "scenario-1",
        "started_at": "2026-11-06T00:00:00Z",
        "evaluation_mode": EVALUATION_MODE,
        "target_thickness_um": 100.0,
        "target_source": "simulator_session_target",
        "model_mapping": {
            "provider_model_key": "provider-model",
            "work_id": "work-1",
            "session_family_id": "family-1",
            "round_index": 1,
            "round_id": "round-1",
            "geometry_model_key": "geometry-1",
        },
        "zone_manifest": {
            "manifest_id": "tsb-zones-20261015-v1",
            "sha256": _manifest_hash(),
            "model_key": "geometry-1",
            "map_nums": [1],
        },
        "expected_zones": [{"face_id": "face-1", "map_num": 1, "zone_ids": ["1"]}],
    }
    body.update(updates)
    return body


def _height() -> dict[str, object]:
    raw = b"synthetic-height-png"
    return {
        "data_base64": base64.b64encode(raw).decode("ascii"),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "encoding": "rg16_b8",
        "basis": "current_round",
        "width": 2,
        "height": 2,
        "origin": "top_left",
        "crop": {
            "x0": 0,
            "y0": 0,
            "width": 2,
            "height": 2,
            "atlas_width": 2,
            "atlas_height": 2,
        },
    }


def test_start_contract_is_strict_and_engine_projection_keeps_version() -> None:
    body = SessionStartRequest.model_validate(_start())
    event = engine_event_from_request(body, source_instance_id="api-instance-1")

    assert event.event_type == "session_start"
    assert event.payload["contract_version"] == CONTRACT_VERSION
    assert event.payload["source_instance_id"] == "api-instance-1"
    assert "event_id" not in event.payload
    assert "session_id" not in event.payload

    with pytest.raises(ValidationError):
        SessionStartRequest.model_validate(_start(legacy_status="complete"))
    with pytest.raises(ValidationError):
        SessionStartRequest.model_validate(_start(evaluation_mode="legacy_v1"))
    with pytest.raises(ValidationError, match=r"must end in Z or \+00:00"):
        SessionStartRequest.model_validate(_start(started_at="2026-11-06T09:00:00+09:00"))
    with pytest.raises(ValidationError, match=r"must end in Z or \+00:00"):
        SessionStartRequest.model_validate(_start(started_at="2026-11-06T00:00:00-00:00"))


def test_relative_sequence_frames_require_explicit_anchor() -> None:
    body = {
        "event_id": "evt-seq-1",
        "session_id": "session-1",
        "contract_version": CONTRACT_VERSION,
        "sequence_id": "SEQ-001",
        "face_id": "face-1",
        "map_num": 1,
        "round_id": "round-1",
        "captured_at": "2026-11-06T00:00:02Z",
        "frames": [{"relative_time_s": 0.0, "spray_on": True}],
    }
    with pytest.raises(ValidationError, match="sequence_started_at"):
        SequenceRequest.model_validate(body)
    body["sequence_started_at"] = "2026-11-06T00:00:01Z"
    assert SequenceRequest.model_validate(body).frames[0].spray_on is True


def test_height_input_rejects_paths_and_content_hash_mismatch() -> None:
    good = _height()
    assert HeightPng.model_validate(good).width == 2

    with pytest.raises(ValidationError):
        HeightPng.model_validate({**good, "path": "/tmp/untrusted.png"})
    with pytest.raises(ValidationError, match="sha256"):
        HeightPng.model_validate({**good, "sha256": "0" * 64})


def test_observation_projection_does_not_send_public_dedup_fields() -> None:
    body = ZoneObservationRequest.model_validate(
        {
            "event_id": "evt-zone-1",
            "session_id": "session-1",
            "contract_version": CONTRACT_VERSION,
            "observation_id": "observation-1",
            "face_id": "face-1",
            "map_num": 1,
            "zone_id": "1",
            "round_id": "round-1",
            "snapshot_id": "snapshot-1",
            "snapshot_revision": 1,
            "captured_at": "2026-11-06T00:00:03Z",
            "height_png": _height(),
            "provider_official_mean_um": 123.4,
        }
    )
    event = engine_event_from_request(body, source_instance_id="api-instance-1")
    assert event.event_type == "zone_observation"
    assert "observation_id" not in event.payload
    assert "provider_official_mean_um" not in event.payload
    assert "contract_version" not in event.payload


def test_coating_progress_maintains_observed_state_without_area_completion() -> None:
    progress = coating_progress_from_engine_state(
        {
            "cumulative_crop_states": [
                {
                    "face_id": "face-1",
                    "map_num": 1,
                    "known_pixel_count": 3,
                    "raw_sum_um": 225.0,
                    "source_event_ids": ["evt-seq-1"],
                    "semantics": "absolute_cumulative_overwrite",
                },
                {
                    "face_id": "face-2",
                    "map_num": 2,
                    "known_pixel_count": 2,
                    "raw_sum_um": 175.0,
                    "source_event_ids": ["evt-seq-2"],
                    "semantics": "absolute_cumulative_overwrite",
                },
            ]
        },
        source_cursor=3,
        confirmed_through_source_cursor=3,
    )

    assert progress.status == "maintained"
    assert progress.known_pixel_count == 5
    assert progress.raw_sum_um == 400.0
    assert progress.includes_current_source_cursor is True
    assert progress.completion_percent is None
    assert progress.target_pixel_count is None
    assert progress.completion_reason_code == "work_target_denominator_unavailable"


def test_coating_progress_reports_pending_when_engine_has_not_confirmed_sequence() -> None:
    progress = coating_progress_from_engine_state(
        None,
        source_cursor=2,
        confirmed_through_source_cursor=1,
    )

    assert progress.status == "pending"
    assert progress.observation_reason_code == "engine_processing"
    assert progress.completion_percent is None
