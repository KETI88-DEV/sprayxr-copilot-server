from __future__ import annotations

from trainer.demo_api import simulator_stub as stub
from trainer.demo_api.sim_contracts import (
    EventTelemetry,
    GunMetrics,
    PaintSpec,
    SessionEventInput,
    SessionStartInput,
    WorkerProfile,
)
from trainer.scripts.session_coaching import SessionMetrics, SessionSummary


def _start_input() -> SessionStartInput:
    return SessionStartInput(
        session_id="s1",
        level_id="TR-L01",
        worker=WorkerProfile(trainee_id="t", skill_level="beginner", handedness="right"),
        spec=PaintSpec(nominal_um=100.0),
    )


def _metrics(*, sequence_correct: bool) -> SessionMetrics:
    return SessionMetrics(
        avg_dft=100.0,
        dft_values=[100.0],
        avg_speed=0.8,
        speed_variance=0.05,
        avg_distance=0.3,
        distance_variance=0.05,
        avg_angle=70.0,
        angle_consistency=0.9,
        sequence_correct=sequence_correct,
        under_thickness_count=0,
        over_thickness_count=0,
    )


def test_start_output_uses_default_snapshot_zone_order() -> None:
    output = stub.build_start_output(_start_input())
    assert output.recommended_flow == ["A-1", "A-2", "A-3"]
    assert output.flow_source == "rule"
    assert "TR-L01" in output.guidance


def test_event_output_rag_when_user_request_present() -> None:
    output = stub.build_event_output(
        "s1", SessionEventInput(event_id="e1", user_request="블라스팅이 뭐야")
    )
    assert output.kind == "rag"
    assert output.anomaly is None
    assert "블라스팅이 뭐야" in output.message


def test_event_output_anomaly_when_gun_metrics_out_of_band() -> None:
    telemetry = EventTelemetry(
        gun_metrics=GunMetrics(speed_m_per_s=0.8, standoff_mm=350.0, angle_deg=55.0)
    )
    output = stub.build_event_output("s1", SessionEventInput(event_id="e2", telemetry=telemetry))
    assert output.kind == "anomaly"
    assert output.anomaly is not None
    assert output.anomaly["type"] == "spray_param_out_of_band"
    assert output.anomaly["severity"] == "high"


def test_event_output_anomaly_severity_medium_for_single_breach() -> None:
    telemetry = EventTelemetry(
        gun_metrics=GunMetrics(speed_m_per_s=0.8, standoff_mm=350.0, angle_deg=72.0)
    )
    output = stub.build_event_output("s1", SessionEventInput(event_id="e3", telemetry=telemetry))
    assert output.kind == "anomaly"
    assert output.anomaly is not None
    assert output.anomaly["severity"] == "medium"


def test_event_output_ack_when_in_band_telemetry() -> None:
    telemetry = EventTelemetry(
        gun_metrics=GunMetrics(speed_m_per_s=0.8, standoff_mm=300.0, angle_deg=72.0)
    )
    output = stub.build_event_output("s1", SessionEventInput(event_id="e4", telemetry=telemetry))
    assert output.kind == "ack"
    assert output.anomaly is None


def test_review_output_reuses_compute_risk_flags() -> None:
    summary = SessionSummary(
        session_id="s1",
        scenario_id="sc",
        mode="live",
        zone_id="A-2",
        zone_type="panel",
        metrics=_metrics(sequence_correct=False),
    )
    output = stub.build_review_output(summary)
    assert any(flag["flag"] == "sequence_violation" for flag in output.risk_flags)
    assert output.action_items
    assert output.citations == []


def test_review_output_reports_clean_session_without_flags() -> None:
    summary = SessionSummary(
        session_id="s2",
        scenario_id="sc",
        mode="live",
        zone_id="A-2",
        zone_type="panel",
        metrics=_metrics(sequence_correct=True),
    )
    output = stub.build_review_output(summary)
    assert output.risk_flags == []
    assert output.action_items == [{"action": "현재 작업 품질 유지.", "priority": "low"}]
