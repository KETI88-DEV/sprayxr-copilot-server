from __future__ import annotations

from trainer.demo_api.app import create_app
from trainer.demo_api.monitor import MonitorRecorder


def test_app_mounts_simulator_and_monitor_routes_without_touching_existing() -> None:
    app = create_app()

    paths = {getattr(route, "path", "") for route in app.routes}

    assert "/api/sim/sessions/start" in paths
    assert "/api/sim/sessions/{session_id}/events" in paths
    assert "/api/sim/sessions/{session_id}/complete" in paths
    assert "/api/monitor/events" in paths
    assert "/api/monitor/stream" in paths
    # Existing boundary is untouched.
    assert "/api/health" in paths
    assert "/api/sequence/{level_id}" in paths
    assert "/api/chat" in paths


def test_app_exposes_a_single_shared_recorder() -> None:
    app = create_app()

    recorder = app.state.monitor_recorder
    assert isinstance(recorder, MonitorRecorder)
    # Distinct apps must not share recorder state.
    assert create_app().state.monitor_recorder is not recorder


from fastapi.testclient import TestClient

from trainer.demo_api.chat_service import ChatService

START_BODY: dict[str, object] = {
    "session_id": "s-1",
    "level_id": "TR-L01",
    "worker": {"trainee_id": "t-1", "skill_level": "beginner", "handedness": "right"},
    "spec": {"nominal_um": 100.0},
}


def test_sequence_returns_fixture_before_any_session() -> None:
    client = TestClient(create_app(ChatService()))
    body = client.get("/api/sequence/TR-L01").json()
    assert body["current_zone_id"] == "A-1"
    assert body["completed_zone_ids"] == ["A-1"]


def test_sequence_follows_simulator_session() -> None:
    client = TestClient(create_app(ChatService()))
    _ = client.post("/api/sim/sessions/start", json=START_BODY)
    _ = client.post("/api/sim/sessions/s-1/events", json={"event_id": "ev-0", "zone_id": "A-2"})

    body = client.get("/api/sequence/TR-L01").json()
    assert body["current_zone_id"] == "A-2"
    assert body["completed_zone_ids"] == ["A-1"]
    statuses = {zone["zone_id"]: zone["status"] for zone in body["zones"]}
    assert statuses == {"A-1": "completed", "A-2": "current", "A-3": "pending"}


def test_sequence_ignores_session_for_a_different_level() -> None:
    client = TestClient(create_app(ChatService()))
    _ = client.post("/api/sim/sessions/start", json=START_BODY)
    body = client.get("/api/sequence/OTHER-L02").json()
    assert body["level_id"] == "OTHER-L02"
    assert body["completed_zone_ids"] == ["A-1"]


def test_app_exposes_a_single_shared_session_store() -> None:
    app = create_app(ChatService())
    store = app.state.sim_session_store
    assert store.current() is None
    assert create_app(ChatService()).state.sim_session_store is not store
