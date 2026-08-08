#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pose_similarity = importlib.import_module("pose_similarity")

Vec3 = tuple[float, float, float]


def _pose() -> dict[str, Vec3]:
    return {
        "Pelvis": (1.0, 2.0, 0.9),
        "Chest": (1.0, 2.0, 1.2),
        "LeftShoulder": (0.8, 2.0, 1.35),
        "RightShoulder": (1.2, 2.0, 1.35),
        "RightUpperArm": (1.25, 2.0, 1.3),
        "RightForeArm": (1.3, 2.15, 1.1),
        "RightHand": (1.3, 2.3, 1.0),
    }


def _translated_and_yawed(pose: dict[str, Vec3], shift: Vec3, yaw_deg: float) -> dict[str, Vec3]:
    angle = math.radians(yaw_deg)
    cos = math.cos(angle)
    sin = math.sin(angle)
    moved: dict[str, Vec3] = {}
    for name, (x, y, z) in pose.items():
        moved[name] = (
            x * cos - y * sin + shift[0],
            x * sin + y * cos + shift[1],
            z + shift[2],
        )
    return moved


def test_normalize_pose_is_invariant_to_world_translation_and_yaw() -> None:
    """작업자가 부스 어디에 서서 어느 쪽을 보든 같은 자세는 같게 나와야 한다."""
    original = pose_similarity.normalize_pose(_pose())

    moved = pose_similarity.normalize_pose(
        _translated_and_yawed(_pose(), (5.0, -3.0, 0.0), 137.0)
    )

    assert set(moved) == set(original)
    for name, expected in original.items():
        assert all(abs(a - b) < 1e-9 for a, b in zip(moved[name], expected)), name


def test_normalize_pose_puts_pelvis_at_origin() -> None:
    normalized = pose_similarity.normalize_pose(_pose())

    assert all(abs(value) < 1e-9 for value in normalized["Pelvis"])


def _bucket() -> dict[str, object]:
    normalized = pose_similarity.normalize_pose(_pose())
    return {
        "n": 500,
        "joints": {
            name: {"mean": list(position), "spread": 0.05}
            for name, position in normalized.items()
        },
    }


def test_joint_sigmas_are_zero_on_the_reference_pose() -> None:
    sigmas = pose_similarity.joint_sigmas(pose_similarity.normalize_pose(_pose()), _bucket())

    assert all(abs(value) < 1e-9 for value in sigmas.values())


def test_joint_sigmas_score_only_the_declared_scoring_set() -> None:
    """루트는 정규화로 항상 원점이라 σ가 구조적으로 0이다. 채점에 섞이면 유사도가 낙관 편향된다."""
    bucket = _bucket()

    sigmas = pose_similarity.joint_sigmas(pose_similarity.normalize_pose(_pose()), bucket)

    assert pose_similarity.ROOT_JOINT in bucket["joints"]  # type: ignore[operator]
    assert pose_similarity.ROOT_JOINT not in sigmas
    assert set(sigmas) <= set(pose_similarity.SCORING_JOINTS)


def test_joint_sigmas_ignore_joints_outside_the_scoring_set() -> None:
    bucket = _bucket()
    bucket["joints"]["LeftToe"] = {"mean": [9.0, 9.0, 9.0], "spread": 0.05}  # type: ignore[index]
    pose = _pose()
    pose["LeftToe"] = (0.0, 0.0, 0.0)

    sigmas = pose_similarity.joint_sigmas(pose_similarity.normalize_pose(pose), bucket)

    assert "LeftToe" not in sigmas


def test_joint_sigma_is_one_at_exactly_one_spread_away() -> None:
    pose = _pose()
    pose["RightHand"] = (pose["RightHand"][0], pose["RightHand"][1], pose["RightHand"][2] + 0.05)

    sigmas = pose_similarity.joint_sigmas(pose_similarity.normalize_pose(pose), _bucket())

    assert abs(sigmas["RightHand"] - 1.0) < 1e-6


def test_similarity_score_is_one_for_the_reference_pose() -> None:
    assert pose_similarity.similarity_score({"RightHand": 0.0, "Chest": 0.0}) == 1.0


def test_similarity_score_stays_a_zero_to_one_float() -> None:
    score = pose_similarity.similarity_score({"RightHand": 8.0, "Chest": 8.0})

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_similarity_score_falls_as_deviation_grows() -> None:
    near = pose_similarity.similarity_score({"RightHand": 0.5})
    far = pose_similarity.similarity_score({"RightHand": 2.0})

    assert near > far


def _pose_with_lowered_right_elbow() -> dict[str, Vec3]:
    pose = _pose()
    elbow = pose["RightForeArm"]
    pose["RightForeArm"] = (elbow[0], elbow[1], elbow[2] - 0.1)
    return pose


def test_build_corrections_names_the_body_part_and_the_direction() -> None:
    """결정 2: llm_briefing이 신체 기준으로 설명할 수 있어야 한다."""
    corrections = pose_similarity.build_corrections(
        pose_similarity.normalize_pose(_pose_with_lowered_right_elbow()), _bucket()
    )

    top = corrections[0]
    assert top.joint == "RightForeArm"
    assert top.label == "오른쪽 팔꿈치"
    assert "오른쪽 팔꿈치" in top.correction_text
    assert "올리" in top.correction_text


def test_build_corrections_reports_a_negative_pitch_for_a_lowered_limb() -> None:
    corrections = pose_similarity.build_corrections(
        pose_similarity.normalize_pose(_pose_with_lowered_right_elbow()), _bucket()
    )

    assert corrections[0].angle_corrections["pitch"] < 0.0


def test_build_corrections_returns_contract_shaped_fields() -> None:
    corrections = pose_similarity.build_corrections(
        pose_similarity.normalize_pose(_pose_with_lowered_right_elbow()), _bucket()
    )

    top = corrections[0]
    assert abs(top.distance - 0.1) < 1e-6
    assert isinstance(top.angle_corrections, dict)
    assert top.expected_gain > 0.0


def test_build_corrections_is_empty_on_the_reference_pose() -> None:
    corrections = pose_similarity.build_corrections(
        pose_similarity.normalize_pose(_pose()), _bucket()
    )

    assert corrections == []


def test_build_corrections_is_capped_and_ranked_by_expected_gain() -> None:
    pose = _pose_with_lowered_right_elbow()
    pose["RightHand"] = (pose["RightHand"][0] + 0.02, pose["RightHand"][1], pose["RightHand"][2])
    pose["Chest"] = (pose["Chest"][0], pose["Chest"][1], pose["Chest"][2] + 0.3)

    corrections = pose_similarity.build_corrections(
        pose_similarity.normalize_pose(pose), _bucket(), limit=2
    )

    assert len(corrections) == 2
    assert corrections[0].expected_gain >= corrections[1].expected_gain


def test_correction_joints_are_a_strict_subset_of_scoring_joints() -> None:
    """게이밍 방지: 채점 관절 집합이 교정 관절 집합을 진부분집합으로 포함해야 한다."""
    scoring = set(pose_similarity.SCORING_JOINTS)
    correction = set(pose_similarity.CORRECTION_JOINTS)

    assert correction < scoring


def test_normalize_pose_keeps_vertical_axis_untouched() -> None:
    """정규화는 Z축 회전만 한다. 높이 차는 자세 정보이므로 보존해야 한다."""
    pose = _pose()

    normalized = pose_similarity.normalize_pose(pose)

    assert abs(normalized["Chest"][2] - (pose["Chest"][2] - pose["Pelvis"][2])) < 1e-9
