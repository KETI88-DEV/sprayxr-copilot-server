#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pose_sequence_report = importlib.import_module("pose_sequence_report")
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


def _normalized(elbow_drop: float = 0.0) -> dict[str, Vec3]:
    """iter_csv_passes가 내는 것과 같은 정규화된 패스 자세."""
    return pose_similarity.normalize_pose(_pose(elbow_drop))


def _reference() -> dict[str, object]:
    normalized = pose_similarity.normalize_pose(_pose())
    return {
        "buckets": {
            BUCKET: {
                "n": 400,
                "joints": {
                    name: {"n": 400, "mean": list(position), "spread": 0.05}
                    for name, position in normalized.items()
                    if name in pose_similarity.SCORING_JOINTS
                },
            }
        }
    }


def test_sequence_stats_scores_each_sequence_separately() -> None:
    entries = [
        (1, BUCKET, _normalized()),
        (1, BUCKET, _normalized()),
        (2, BUCKET, _normalized(elbow_drop=0.15)),
    ]

    stats = pose_sequence_report.sequence_stats(entries, _reference())

    assert [item.sequence for item in stats] == [1, 2]
    assert stats[0].evaluated == 2
    assert stats[0].average_similarity == pytest.approx(1.0, abs=1e-6)
    assert stats[1].average_similarity < 1.0


def test_sequence_stats_counts_passes_the_reference_cannot_score() -> None:
    entries = [(1, BUCKET, _normalized()), (1, "vertical-high-left", _normalized())]

    stats = pose_sequence_report.sequence_stats(entries, _reference())

    assert stats[0].evaluated == 1
    assert stats[0].unavailable == 1


def test_sequence_stats_withholds_the_average_when_nothing_was_scored() -> None:
    entries = [(1, "vertical-high-left", _normalized())]

    stats = pose_sequence_report.sequence_stats(entries, _reference())

    assert stats[0].evaluated == 0
    assert stats[0].average_similarity is None


def test_sequence_stats_reports_which_joints_recur() -> None:
    """시퀀스 통계의 목적은 어느 관절이 반복해서 벗어나는지를 보는 것이다."""
    entries = [(1, BUCKET, _normalized(elbow_drop=0.15)) for _ in range(4)]

    stats = pose_sequence_report.sequence_stats(entries, _reference())

    top = stats[0].problem_joints[0]
    assert top.joint == "RightForeArm"
    assert top.label == "오른쪽 팔꿈치"
    assert top.occurrence_count == 4
    assert top.occurrence_rate == pytest.approx(1.0)
    assert top.average_sigma == pytest.approx(3.0)


def test_sequence_stats_leaves_problem_joints_empty_inside_the_expert_range() -> None:
    entries = [(1, BUCKET, _normalized(elbow_drop=0.05)) for _ in range(4)]

    stats = pose_sequence_report.sequence_stats(entries, _reference())

    assert stats[0].problem_joints == []


def test_as_payload_matches_the_contract_field_names() -> None:
    entries = [(1, BUCKET, _normalized(elbow_drop=0.15))]

    payload = pose_sequence_report.as_payload(
        pose_sequence_report.sequence_stats(entries, _reference())
    )

    assert payload[0]["sequence_id"] == 1
    assert payload[0]["evaluated_pass_count"] == 1
    assert payload[0]["unavailable_pass_count"] == 0
    assert "average_similarity_score" in payload[0]
    assert payload[0]["problem_joints"][0]["occurrence_rate"] == pytest.approx(1.0)
