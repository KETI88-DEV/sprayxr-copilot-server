from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.monitor import MonitorRecorder
from trainer.demo_api.simulator import create_simulator_router


def _client() -> tuple[TestClient, MonitorRecorder]:
    recorder = MonitorRecorder()
    app = FastAPI()
    app.include_router(create_simulator_router(recorder))
    return TestClient(app), recorder


def _start_body() -> dict[str, object]:
    return {
        "session_id": "s1",
        "level_id": "SP0035-L01",
        "worker": {"trainee_id": "t", "skill_level": "beginner", "handedness": "right"},
        "spec": {"nominal_um": 100.0},
    }


def _complete_body() -> dict[str, object]:
    return {
        "session_id": "s1",
        "scenario_id": "sc",
        "mode": "live",
        "zone_id": "A-2",
        "zone_type": "panel",
        "metrics": {
            "avg_dft": 100.0,
            "dft_values": [100.0],
            "avg_speed": 0.8,
            "speed_variance": 0.05,
            "avg_distance": 0.3,
            "distance_variance": 0.05,
            "avg_angle": 70.0,
            "angle_consistency": 0.9,
            "sequence_correct": False,
        },
    }


def test_start_returns_recommended_flow_and_records_input_output() -> None:
    client, recorder = _client()

    response = client.post("/api/sim/sessions/start", json=_start_body())

    assert response.status_code == 200
    assert response.json()["recommended_flow"] == ["A-1", "A-2", "A-3"]
    snapshot = recorder.snapshot(cursor=0, limit=10)
    assert [(event.phase, event.direction) for event in snapshot.events] == [
        ("start", "input"),
        ("start", "output"),
    ]
    assert snapshot.events[0].payload["level_id"] == "SP0035-L01"
    assert snapshot.events[1].payload["flow_source"] == "rule"


def test_event_anomaly_round_trip_records_pair() -> None:
    client, recorder = _client()
    body = {
        "event_id": "e1",
        "telemetry": {"gun_metrics": {"speed_m_per_s": 0.8, "standoff_mm": 350.0, "angle_deg": 55.0}},
    }

    response = client.post("/api/sim/sessions/s1/events", json=body)

    assert response.status_code == 200
    assert response.json()["kind"] == "anomaly"
    snapshot = recorder.snapshot(cursor=0, limit=10)
    assert [(event.phase, event.direction) for event in snapshot.events] == [
        ("runtime", "input"),
        ("runtime", "output"),
    ]
    assert snapshot.events[0].session_id == "s1"


def test_invalid_input_returns_422_and_records_error_output() -> None:
    client, recorder = _client()

    response = client.post("/api/sim/sessions/start", json={"session_id": "s1"})

    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"
    snapshot = recorder.snapshot(cursor=0, limit=10)
    assert snapshot.events[0].direction == "input"
    assert snapshot.events[1].payload["error"] == "validation_error"


def test_path_body_session_id_mismatch_rejected() -> None:
    client, recorder = _client()

    response = client.post("/api/sim/sessions/s1/events", json={"session_id": "OTHER", "event_id": "e1"})

    assert response.status_code == 422
    assert response.json()["error"] == "session_id_mismatch"
    snapshot = recorder.snapshot(cursor=0, limit=10)
    assert snapshot.events[1].payload["error"] == "session_id_mismatch"


def test_malformed_json_is_rejected_and_recorded() -> None:
    client, recorder = _client()

    response = client.post(
        "/api/sim/sessions/s1/events",
        content=b"{not json",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422
    assert response.json()["error"] == "invalid_json"
    snapshot = recorder.snapshot(cursor=0, limit=10)
    assert [event.payload["error"] for event in snapshot.events] == ["invalid_json", "invalid_json"]


def test_complete_review_uses_risk_flags_and_records_pair() -> None:
    client, recorder = _client()

    response = client.post("/api/sim/sessions/s1/complete", json=_complete_body())

    assert response.status_code == 200
    assert any(flag["flag"] == "sequence_violation" for flag in response.json()["risk_flags"])
    snapshot = recorder.snapshot(cursor=0, limit=10)
    assert [(event.phase, event.direction) for event in snapshot.events] == [
        ("session_end", "input"),
        ("session_end", "output"),
    ]


def test_full_session_records_two_events_per_call() -> None:
    client, recorder = _client()

    assert client.post("/api/sim/sessions/start", json=_start_body()).status_code == 200
    assert client.post("/api/sim/sessions/s1/events", json={"event_id": "e1"}).status_code == 200
    assert client.post("/api/sim/sessions/s1/complete", json=_complete_body()).status_code == 200

    snapshot = recorder.snapshot(cursor=0, limit=50)
    assert [(event.phase, event.direction) for event in snapshot.events] == [
        ("start", "input"),
        ("start", "output"),
        ("runtime", "input"),
        ("runtime", "output"),
        ("session_end", "input"),
        ("session_end", "output"),
    ]
    assert [event.seq for event in snapshot.events] == [1, 2, 3, 4, 5, 6]
