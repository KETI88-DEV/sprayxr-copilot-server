from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp_servers.main import app
from trainer.scripts.agent_pipeline import AgentPipeline

client = TestClient(app)


def _dft(body: dict) -> tuple[dict, dict]:
    response = client.post("/dft_judge/evaluate", json=body)
    assert response.status_code == 200
    payload = response.json()
    return payload, payload["result"]


def _anomaly(measurements: list[tuple[float, str]], *, timeseries: dict | None = None, nominal: float | None = 100.0, zone: str = "3") -> tuple[dict, dict]:
    body: dict = {
        "zone_id": zone,
        "dft_measurements": [{"value_um": v, "zone_id": zone, "position": p} for v, p in measurements],
    }
    if nominal is not None:
        body["spec"] = {"nominal_um": nominal, "tolerance_rule": "90/10"}
    if timeseries:
        body["timeseries"] = timeseries
    response = client.post("/anomaly_detection/detect", json=body)
    assert response.status_code == 200
    payload = response.json()
    return payload, payload["result"]


def test_dft_below_target() -> None:
    _, result = _dft({"measurements": [150, 158, 171], "spec": {"nominal": 160}})
    assert result["grade"] == "below" and result["pass_"] is False and result["verdict"] == "fail"
    assert result["mean"] == pytest.approx(159.667, abs=1e-3)
    assert result["nominal"] == 160.0 and result["target_basis"] == "session_target"
    assert result["normal_band_um"] == [160.0, 240.0]
    assert result["six_band_counts"]["from_0_9t_to_lt_1_0t"] == 2
    assert result["six_band_counts"]["from_1_0t_to_lt_1_5t"] == 1
    assert result["out_of_spec_count"] == 2
    assert result["coverage_pct"] == pytest.approx(33.33, abs=0.01)
    assert result["violations"] == [{"point_index": -1, "value": pytest.approx(159.667, abs=1e-3), "threshold": 160.0, "reason": "mean_below_target"}]


def test_dft_trained_call_shape_is_accepted_and_normal() -> None:
    _, result = _dft({"measurements": [100, 102, 98, 101], "spec": {"nominal": 100, "tolerance_rule": "90/10"}})
    assert result["mean"] == pytest.approx(100.25) and result["grade"] == "normal"
    assert result["pass_"] is True and result["violations"] == []


def test_dft_without_spec_uses_the_default_target() -> None:
    _, result = _dft({"measurements": [170]})
    assert result["nominal"] == 160.0 and result["target_basis"] == "default_target" and result["grade"] == "normal"


def test_dft_above_reports_the_over_limit() -> None:
    _, result = _dft({"measurements": [250, 260], "spec": {"nominal": 160}})
    assert result["grade"] == "above" and result["verdict"] == "fail"
    assert result["violations"][0]["reason"] == "mean_at_or_above_over_limit"
    assert result["violations"][0]["threshold"] == 240.0


def test_dft_rejects_a_non_positive_target() -> None:
    assert client.post("/dft_judge/evaluate", json={"measurements": [1], "spec": {"nominal": 0}}).status_code == 400


def test_anomaly_under_thickness() -> None:
    _, result = _anomaly([(100, "center"), (60, "lower"), (100, "upper")])
    assert result["grade"] == "below" and result["anomaly_detected"] is True
    assert result["stats"]["mean_um"] == pytest.approx(86.7, abs=0.05)
    assert [(a["value_um"], a["position"], a["type"]) for a in result["anomalies"]] == [(60, "lower", "저도막")]
    assert result["distribution_pattern"] == "하부 영역에 저도막 측정점 집중 (1건)"
    assert result["root_causes"] == [] and result["cause"] == ""
    assert result["anomaly_type"] == "저도막" and result["zone"] == "3"


def test_anomaly_with_motion_series_names_adjustment_candidates() -> None:
    _, result = _anomaly(
        [(100, "center"), (60, "lower"), (100, "upper")],
        timeseries={"gun_speed_m_per_s": [0.4, 0.5], "standoff_distance_mm": [300, 320]})
    assert {c["factor"] for c in result["root_causes"]} == {"gun_speed", "standoff_distance"}
    assert all(c["evidence_kind"] == "quality_rule_candidate" for c in result["root_causes"])
    assert result["cause"] == "조절 후보: 이동 속도를 낮추는 쪽, 이격거리를 줄이는 쪽. 원인으로 확정된 것은 아닙니다."


def test_anomaly_normal_has_no_anomalies() -> None:
    _, result = _anomaly([(100, "center"), (102, "lower"), (98, "upper")])
    assert result["anomaly_detected"] is False and result["anomalies"] == []
    assert result["distribution_pattern"] == "이상 없음" and result["anomaly_type"] == ""


def test_anomaly_without_spec_uses_the_default_target_not_the_mean() -> None:
    _, result = _anomaly([(100, "center"), (100, "lower")], nominal=None)
    assert result["target_thickness_um"] == 160.0 and result["target_basis"] == "default_target"
    assert result["grade"] == "below"


def test_no_invented_confidence_or_old_labels() -> None:
    _, dft = _dft({"measurements": [100, 102], "spec": {"nominal": 100, "tolerance_rule": "90/10"}})
    _, ano = _anomaly([(100, "center"), (60, "lower")], timeseries={"gun_speed_m_per_s": [0.4, 0.5]}, zone="3")
    for result in (dft, ano):
        text = json.dumps(result, ensure_ascii=False)
        assert "confidence" not in text and "B-3" not in text and "90/10" not in text


def test_pipeline_sections_are_filled_from_the_new_outputs() -> None:
    dft_envelope, _ = _dft({"measurements": [150, 158, 171], "spec": {"nominal": 160}})
    ano_envelope, _ = _anomaly([(100, "center"), (60, "lower"), (100, "upper")])
    sections = AgentPipeline._assemble_tool_sections([
        {"name": "dft_judge", "result": dft_envelope},
        {"name": "anomaly_detection", "result": ano_envelope}])
    assert sections["dft_result"]["verdict"] == "fail" and sections["dft_result"]["nominal_um"] == 160.0
    assert sections["anomaly"]["type"] == "저도막" and sections["anomaly"]["zone"] == "3"
    assert sections["anomaly"]["detected"] is True
