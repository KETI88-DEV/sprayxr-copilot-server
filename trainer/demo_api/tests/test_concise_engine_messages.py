"""Brief engine guidance retains measurements, action and its factual limits."""
from __future__ import annotations

from pathlib import Path

import pytest

from trainer.demo_api import coating_assessment as ca, session_analysis as sa, session_coating as sc
from trainer.demo_api.expert_guidance import ExpertGuidanceEngine
from trainer.demo_api.pose_query import PoseQueryResult
from trainer.demo_api.sequence_analysis import SequenceAnalysisEngine
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.telemetry_contracts import AnalyzeSequenceRequest
from trainer.demo_api.telemetry_router import _coating_feedback, coating_alerts
from trainer.demo_api.tests.test_coating_progress import _engine, _profile, _request
from trainer.demo_api.tests.test_expert_guidance import _request as guidance_request, _write_profile
from trainer.demo_api.tests.test_sequence_analysis import _body, _profile as sequence_profile, _record


@pytest.mark.parametrize("speed,distance,status,action", [
    (50, 300, ("slow", "near"), ("높이", "멀리")),
    (160, 520, ("fast", "far"), ("낮추", "가까이")),
])
def test_metric_guides_are_brief_and_keep_the_required_direction(tmp_path: Path, speed, distance, status, action):
    _write_profile(tmp_path)
    guide = ExpertGuidanceEngine(tmp_path).evaluate(guidance_request(speed=speed, distance=distance))
    for item, expected_status, required_action in zip((guide.speed, guide.distance), status, action):
        assert item.available and item.status == expected_status
        assert required_action in item.message
        assert len(item.message) <= 20


def test_complete_coating_guide_keeps_the_physical_gate_and_next_zone(tmp_path: Path):
    _profile(tmp_path)
    progress = _engine(tmp_path).evaluate(_request(sequence_id="SHORT-COMPLETE", thickness_um=180))
    assert progress.status == "complete" and progress.recommendation == "move_next_zone"
    assert progress.next_zone_id == "2"
    assert "160~200µm" in progress.message and "95%" in progress.message
    assert "공식" in progress.message and "2구역" in progress.message
    assert len(progress.message) <= 72


def test_overcoat_complete_guide_says_stop_painting_and_names_the_next_zone(tmp_path: Path):
    _profile(tmp_path)
    progress = _engine(tmp_path).evaluate(_request(sequence_id="SHORT-OVER", thickness_um=300))
    assert progress.status == "complete" and progress.recommendation == "move_next_zone"
    assert progress.next_zone_id == "2"
    assert "160~200µm" in progress.message and "칠한 곳 평균" in progress.message
    assert "과도막" in progress.message and "더 칠하지 마세요" in progress.message and "2구역" in progress.message
    assert len(progress.message) <= 105


def test_feedback_and_alerts_keep_the_grade_and_noncausal_adjustment(tmp_path: Path):
    _profile(tmp_path)
    body = _request(sequence_id="SHORT-LOW", thickness_um=110).model_dump(mode="json")
    body["sequence_result"].update(avg_speed_mms=300.0, avg_distance_mm=250.0)
    progress = _engine(tmp_path).evaluate(AnalyzeSequenceRequest.model_validate(body))
    feedback = _coating_feedback(progress, progress.assessment)
    assert feedback.grade == "below" and feedback.normal_band_um == [160, 200]
    assert feedback.thickness_um == progress.assessment["thickness_um"]
    assert "칠한 곳 평균" in feedback.message and "µm" in feedback.message and "미도장" in feedback.message
    assert "후보" in feedback.message and "원인 미확정" in feedback.message
    assert len(feedback.message) <= 115
    over = _engine(tmp_path).evaluate(_request(sequence_id="SHORT-ALERT", thickness_um=300))
    alert = coating_alerts(over.assessment)[0]
    assert alert["anomaly_type"] == "coating_over_app"
    assert "과도막" in alert["payload"]["message"] and "µm" in alert["payload"]["message"]
    assert "멈추" in alert["payload"]["message"] or "중지" in alert["payload"]["message"]
    assert len(alert["payload"]["message"]) <= 100


def test_sequence_unknown_gate_does_not_issue_a_move_in_its_brief_guide(tmp_path: Path):
    sequence_profile(tmp_path)
    engine, store = SequenceAnalysisEngine(tmp_path), SequenceStore(":memory:")
    body = _body("SEQ-001", "3")
    _record(store, body)
    decision = engine.evaluate(AnalyzeSequenceRequest.model_validate(body), store)
    assert decision.feedback.reason_code is not None
    assert "현재 3구역" in decision.text_guide and "이동 전" in decision.text_guide
    assert "다음" not in decision.text_guide
    assert len(decision.text_guide) <= 62


def test_sequence_report_keeps_all_scores_in_a_brief_guideline(tmp_path: Path):
    sequence_profile(tmp_path)
    report = SequenceAnalysisEngine(tmp_path).report_bodies(
        [_body("SEQ-001", "1"), _body("SEQ-002", "2", vertical=True), _body("SEQ-003", "3", vertical=True)],
        scenario_id="3")
    assert report["adherence_rate"] == 1 and report["direction_adherence_rate"] == .667
    assert "100.0%" in report["guideline"] and "66.7%" in report["guideline"]
    assert "1→2→3" in report["guideline"] and "3회" in report["guideline"]
    assert len(report["guideline"]) <= 85


def test_pose_session_summary_keeps_scored_and_excluded_counts():
    pose = PoseQueryResult("available", None, "SEQ-001", .7, "single",
        [{"joint": "RightUpperArm", "label": "오른쪽 위팔을 내리세요."}], None)
    text = sa._guideline([pose, pose], total=3, not_scored=1)
    assert "3회 중 2회" in text and "70%" in text and "오른쪽 위팔" in text and "2회" in text
    assert "미판정 1회 제외" in text
    assert len(text) <= 115


def test_coating_session_summary_keeps_boundaries_unpainted_and_mean():
    text = sc._guideline([{"zone_id": "3", "grade": "below"}],
        [{"zone_id": "3", "status": "under_app", "thickness_um": 98, "target_thickness_um": 160}],
        {"target_thickness_um": 160, "target_basis": "default_target", "normal_band_um": [160, 200],
         "painted_mean_um": 98, "unpainted_fraction": .2}, {"analysis_status": "unavailable"})
    assert "160~200µm" in text and "경계 포함" in text and "98µm" in text
    assert "하한 미만" in text and "상한 초과" in text and "칠한 곳 평균" in text
    assert "미도장 20%" in text and "저도막 1개" in text and "3번" in text
    assert len(text) <= 200


def test_shared_adjustment_is_short_without_claiming_a_verified_cause():
    text = ca.adjustment_sentence([{"variable": "speed_mm_s", "direction": "decrease", "priority": 1}])
    assert "이동 속도를 낮추는 쪽" in text and "후보" in text and "원인 미확정" in text
    assert len(text) <= 42
