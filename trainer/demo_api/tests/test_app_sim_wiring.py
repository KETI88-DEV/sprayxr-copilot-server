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
