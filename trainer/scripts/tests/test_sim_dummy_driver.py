from __future__ import annotations

from trainer.scripts.sim_dummy_driver import build_dummy_session


def test_build_dummy_session_shapes_three_phases() -> None:
    start, events, complete = build_dummy_session("s1", events=2, anomaly=True)

    assert start["session_id"] == "s1"
    assert "worker" in start and "spec" in start
    assert len(events) == 2
    assert events[0]["event_id"] == "ev-0"
    assert any("telemetry" in event for event in events)
    assert complete["session_id"] == "s1"
    assert "metrics" in complete


def test_anomaly_flag_injects_out_of_band_gun_metrics() -> None:
    _start, events, _complete = build_dummy_session("s1", events=1, anomaly=True)

    telemetry = events[0]["telemetry"]
    assert isinstance(telemetry, dict)
    gun_metrics = telemetry["gun_metrics"]
    assert isinstance(gun_metrics, dict)
    assert gun_metrics["standoff_mm"] > 320.0 or gun_metrics["angle_deg"] < 60.0


def test_clean_run_keeps_every_event_in_band() -> None:
    _start, events, complete = build_dummy_session("s1", events=4, anomaly=False)

    for event in events:
        telemetry = event.get("telemetry")
        if not isinstance(telemetry, dict):
            continue
        gun_metrics = telemetry["gun_metrics"]
        assert isinstance(gun_metrics, dict)
        assert gun_metrics["standoff_mm"] <= 320.0
        assert gun_metrics["angle_deg"] >= 60.0

    metrics = complete["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["sequence_correct"] is True
