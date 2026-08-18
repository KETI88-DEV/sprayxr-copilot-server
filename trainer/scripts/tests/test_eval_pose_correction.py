#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

eval_pose_correction = importlib.import_module("eval_pose_correction")
pose_similarity = importlib.import_module("pose_similarity")

Vec3 = tuple[float, float, float]
BUCKET = "horizontal-mid-forward"


def _pose(elbow_drop: float = 0.0) -> dict[str, Vec3]:
    return {
        "Pelvis": (0.0, 0.0, 0.9),
        "Spine": (0.0, 0.0, 1.05),
        "Chest": (0.0, 0.0, 1.2),
        "LeftShoulder": (-0.18, 0.0, 1.32),
        "RightShoulder": (0.18, 0.0, 1.32),
        "LeftUpperArm": (-0.22, 0.02, 1.28),
        "RightUpperArm": (0.22, 0.02, 1.28),
        "LeftForeArm": (-0.28, 0.18, 1.10),
        "RightForeArm": (0.28, 0.18, 1.10 - elbow_drop),
        "LeftHand": (-0.30, 0.34, 1.02),
        "RightHand": (0.30, 0.34, 1.02),
    }


def _bucket(elbow_drop: float = 0.0, spread: float = 0.05) -> dict[str, object]:
    normalized = pose_similarity.normalize_pose(_pose(elbow_drop))
    return {
        "buckets": {
            BUCKET: {
                "n": 400,
                "joints": {
                    name: {"n": 400, "mean": list(position), "spread": spread}
                    for name, position in normalized.items()
                    if name in pose_similarity.SCORING_JOINTS
                },
            }
        }
    }


def test_scores_are_taken_from_the_scoring_reference_not_the_correction_one() -> None:
    """교정에 쓴 레퍼런스로 다시 채점하면 향상이 자동 보장된다. 이 분리가 지표의 전부다."""
    query = pose_similarity.normalize_pose(_pose(elbow_drop=0.15))

    outcome = eval_pose_correction.measure_correction(
        query, BUCKET, correction=_bucket(), scoring=_bucket(elbow_drop=0.15)
    )

    assert outcome is not None
    assert outcome.before == pytest.approx(1.0, abs=1e-6)


def test_perfect_correction_does_not_close_the_whole_gap_under_a_held_out_reference() -> None:
    query = pose_similarity.normalize_pose(_pose(elbow_drop=0.15))

    outcome = eval_pose_correction.measure_correction(
        query, BUCKET, correction=_bucket(), scoring=_bucket(elbow_drop=0.03)
    )

    assert outcome is not None
    assert outcome.after < 1.0


def test_no_outcome_when_the_pass_needs_no_correction() -> None:
    query = pose_similarity.normalize_pose(_pose())

    outcome = eval_pose_correction.measure_correction(
        query, BUCKET, correction=_bucket(), scoring=_bucket()
    )

    assert outcome is None


def test_no_outcome_when_either_reference_lacks_the_bucket() -> None:
    query = pose_similarity.normalize_pose(_pose(elbow_drop=0.15))

    assert (
        eval_pose_correction.measure_correction(
            query, "vertical-high-left", correction=_bucket(), scoring=_bucket()
        )
        is None
    )


def test_eligible_keeps_only_cases_with_room_to_improve() -> None:
    """이미 기준에 가까운 패스를 넣으면 개선율 분모가 작아 지표가 요동친다."""
    roomy = eval_pose_correction.Outcome(1, BUCKET, 0.40, 0.70, 1)
    tight = eval_pose_correction.Outcome(2, BUCKET, 0.95, 0.97, 1)

    kept = eval_pose_correction.eligible([roomy, tight])

    assert [item.sequence for item in kept] == [1]


def test_summarize_reports_median_gap_closure() -> None:
    outcomes = [
        eval_pose_correction.Outcome(1, BUCKET, 0.40, 0.70, 1),
        eval_pose_correction.Outcome(2, BUCKET, 0.20, 0.60, 2),
    ]

    summary = eval_pose_correction.summarize(outcomes)

    assert summary["n"] == 2
    assert summary["median_closure"] == pytest.approx(0.5)


def test_summarize_withholds_the_metric_without_eligible_cases() -> None:
    summary = eval_pose_correction.summarize([])

    assert summary["n"] == 0
    assert summary["median_closure"] is None
