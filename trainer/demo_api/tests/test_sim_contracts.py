from __future__ import annotations

import pytest
from pydantic import ValidationError

from trainer.demo_api.sim_contracts import (
    EventTelemetry,
    GunMetrics,
    MonitorEvent,
    MonitorSnapshot,
    SessionEventInput,
    SessionEventOutput,
    SessionReviewOutput,
    SessionStartInput,
    SessionStartOutput,
    SimSessionResponse,
)


def test_session_start_input_parses_nested_models() -> None:
    parsed = SessionStartInput.model_validate(
        {
            "session_id": "s1",
            "level_id": "SP0035-L01",
            "worker": {"trainee_id": "t1", "skill_level": "beginner", "handedness": "right"},
            "spec": {"nominal_um": 100.0},
        }
    )
    assert parsed.worker.skill_level == "beginner"
    assert parsed.spec.nominal_um == 100.0


def test_start_input_rejects_extra_field() -> None:
    with pytest.raises(ValidationError):
        _ = SessionStartInput.model_validate(
            {
                "session_id": "s1",
                "level_id": "L",
                "worker": {"trainee_id": "t", "skill_level": "beginner", "handedness": "right"},
                "spec": {"nominal_um": 100.0},
                "unexpected": 1,
            }
        )


def test_event_input_allows_optional_telemetry_and_request() -> None:
    only_request = SessionEventInput.model_validate({"event_id": "e1", "user_request": "블라스팅?"})
    assert only_request.telemetry is None

    with_telemetry = SessionEventInput.model_validate(
        {
            "event_id": "e2",
            "telemetry": {"gun_metrics": {"speed_m_per_s": 0.8, "standoff_mm": 350.0, "angle_deg": 55.0}},
        }
    )
    assert isinstance(with_telemetry.telemetry, EventTelemetry)
    assert isinstance(with_telemetry.telemetry.gun_metrics, GunMetrics)


def test_output_and_monitor_models_round_trip() -> None:
    start_output = SessionStartOutput(
        session_id="s1", recommended_flow=["A-1"], flow_source="rule", guidance="g"
    )
    assert start_output.flow_source == "rule"

    event_output = SessionEventOutput(
        session_id="s1", event_id="e1", kind="ack", message="ok", anomaly=None
    )
    assert event_output.kind == "ack"

    review = SessionReviewOutput(
        session_id="s1", summary="ok", risk_flags=[], action_items=[], citations=[]
    )
    assert review.risk_flags == []

    event = MonitorEvent(
        seq=1, session_id="s1", phase="start", direction="input", timestamp="t", payload={"a": 1}
    )
    snapshot = MonitorSnapshot(events=[event], next_cursor=1, dropped_before=None)
    assert snapshot.events[0].direction == "input"


def test_session_event_accepts_optional_zone_id() -> None:
    payload = SessionEventInput.model_validate({"event_id": "ev-0", "zone_id": "A-2"})
    assert payload.zone_id == "A-2"


def test_session_event_zone_id_defaults_to_none() -> None:
    payload = SessionEventInput.model_validate({"event_id": "ev-0"})
    assert payload.zone_id is None


def test_sim_session_response_allows_null_session() -> None:
    response = SimSessionResponse.model_validate({"session": None})
    assert response.session is None


def test_sim_session_response_parses_full_view() -> None:
    response = SimSessionResponse.model_validate(
        {
            "session": {
                "session_id": "s-1",
                "level_id": "SP0035-L01",
                "phase": "runtime",
                "recommended_flow": ["A-1", "A-2", "A-3"],
                "current_zone_id": "A-2",
                "completed_zone_ids": ["A-1"],
                "last_event_id": "ev-1",
                "anomaly_count": 1,
                "unresolved_zone_id": None,
                "updated_at": "2026-07-28T00:00:00+00:00",
            }
        }
    )
    assert response.session is not None
    assert response.session.current_zone_id == "A-2"


_VALID_SESSION_VIEW: dict[str, object] = {
    "session_id": "s-1",
    "level_id": "SP0035-L01",
    "phase": "runtime",
    "recommended_flow": ["A-1", "A-2", "A-3"],
    "current_zone_id": "A-2",
    "completed_zone_ids": ["A-1"],
    "last_event_id": "ev-1",
    "anomaly_count": 1,
    "unresolved_zone_id": None,
    "updated_at": "2026-07-28T00:00:00+00:00",
}


def test_sim_session_view_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        _ = SimSessionResponse.model_validate(
            {"session": {**_VALID_SESSION_VIEW, "unexpected": 1}}
        )


def test_sim_session_view_accepts_the_same_payload_without_the_unknown_field() -> None:
    response = SimSessionResponse.model_validate({"session": dict(_VALID_SESSION_VIEW)})
    assert response.session is not None
    assert response.session.anomaly_count == 1
