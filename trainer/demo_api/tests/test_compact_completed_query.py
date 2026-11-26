"""Focused questions about saved reports keep the facts the user requested."""

import pytest

from trainer.demo_api.pose_query import NO_CORRECTION_CAVEAT, answer_order_question, answer_pose_question


@pytest.fixture
def completed_report():
    return "RPT-1", {
        "pose_report": {
            "average_similarity_score": .75,
            "evaluated_sequence_count": 10,
            "zone_summary": [{
                "zone_id": "2", "evaluated_sequence_count": 10,
                "average_similarity_score": .64, "problem_joints": [],
            }],
        },
        "coating_zones": [{"zone_id": "2", "status": "over_app", "thickness_um": 240, "overpaint_percent": 20}],
        "sequence_report": {
            "analysis_status": "available", "recommended_flow": ["2", "3", "1"],
            "first_visit_flow": ["3", "2", "1"], "adherence_rate": .75,
            "evaluated_visit_count": 4, "on_track_visit_count": 3,
            "best_reference_similarity": .80, "position_similarity": .60,
            "reference_session_count": 6,
        },
    }


def test_named_zone_pose_score_does_not_repeat_other_fields(completed_report):
    result = answer_pose_question(completed_report, prompt="Zone 2 자세 유사도는?", context=None)

    assert result.answer == "2번 구역 자세 유사도는 64%입니다."
    assert result.context["similarity_score"] == .64
    assert result.context["evaluated_sequence_count"] == 10
    assert result.context["coating"]["thickness_um"] == 240


def test_explicit_zone_pose_and_coating_keep_both_scores(completed_report):
    result = answer_pose_question(completed_report, prompt="Zone 2 자세 유사도와 도막 상태는?", context=None)

    assert result.answer.startswith("2번 구역 자세 유사도는 64%입니다.")
    assert "과도장" in result.answer and "240µm" in result.answer
    assert "숙련자도 70%" not in result.answer


def test_pose_improvement_keeps_the_no_correction_limit(completed_report):
    result = answer_pose_question(completed_report, prompt="Zone 2 자세 개선점은?", context=None)

    assert NO_CORRECTION_CAVEAT in result.answer
    assert "과도장" not in result.answer


def test_pose_average_keeps_only_the_requested_score_in_the_text(completed_report):
    result = answer_pose_question(completed_report, prompt="전체 자세 유사도는 몇 퍼센트야?", context=None)

    assert result.answer == "세션 전체 자세 유사도는 75%입니다."
    assert result.context["similarity_score"] == .75


def test_completed_reference_order_stops_after_all_the_zone_numbers(completed_report):
    result = answer_order_question(completed_report, prompt="권장 Zone 순서는?", intent="order_reference")

    assert result.answer == "권장 작업 순서는 Zone 2 → 3 → 1입니다."
    assert result.context["reference_session_count"] == 6


def test_completed_order_score_omits_unused_metrics_but_keeps_them_structured(completed_report):
    result = answer_order_question(completed_report, prompt="순서 유사도는 몇 퍼센트야?", intent="order_adherence")

    assert result.answer == "전문가 순서 유사도는 75%입니다."
    assert result.context["adherence_rate"] == .75
    assert result.context["best_reference_similarity"] == .80
    assert result.context["position_similarity"] == .60


def test_completed_order_score_answers_explicit_submetrics(completed_report):
    result = answer_order_question(completed_report, prompt="최근접 전문가 유사도와 순번별 유사도는?", intent="order_adherence")

    assert result.answer == "최근접 전문가 수행 유사도 80% · 순번별 유사도 60%."
