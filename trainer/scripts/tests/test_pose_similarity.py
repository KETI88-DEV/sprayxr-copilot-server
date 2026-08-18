#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import pytest

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


def test_facing_yaw_is_zero_when_the_shoulder_line_lies_on_the_x_axis() -> None:
    assert abs(pose_similarity.facing_yaw(_pose())) < 1e-9


def test_body_heading_labels_the_four_directions_in_the_worker_frame() -> None:
    """정규화 공간에서 +X는 작업자의 오른쪽, +Y는 앞이다(실측: 건 쥔 손 Y가 100% 양수)."""
    yaw = pose_similarity.facing_yaw(_pose())

    assert pose_similarity.body_heading(yaw, (0.0, 1.0)) == "forward"
    assert pose_similarity.body_heading(yaw, (0.0, -1.0)) == "back"
    assert pose_similarity.body_heading(yaw, (1.0, 0.0)) == "right"
    assert pose_similarity.body_heading(yaw, (-1.0, 0.0)) == "left"


def test_body_heading_is_invariant_to_world_yaw() -> None:
    """월드 요가 달라도 같은 몸 기준 방향이어야 한다. 이것이 B안의 핵심이다."""
    turned = _translated_and_yawed(_pose(), (4.0, -2.0, 0.0), 137.0)
    angle = math.radians(137.0)
    rotated = (0.0 * math.cos(angle) - 1.0 * math.sin(angle), 0.0 * math.sin(angle) + 1.0 * math.cos(angle))

    assert pose_similarity.body_heading(pose_similarity.facing_yaw(turned), rotated) == "forward"


def test_compare_poses_reports_zero_distance_for_the_same_pose() -> None:
    result = pose_similarity.compare_poses(_pose(), _pose(), {"RightForeArm": 0.05})

    assert result.similarity == 1.0
    assert result.deltas == []


def test_compare_poses_measures_the_joint_that_moved() -> None:
    """같은 스트로크를 다른 시퀀스에서 했을 때 어느 관절이 얼마나 달랐는지가 결과다."""
    result = pose_similarity.compare_poses(
        _pose(), _pose_with_lowered_right_elbow(drop=0.15), {"RightForeArm": 0.05}
    )

    assert [item.joint for item in result.deltas] == ["RightForeArm"]
    top = result.deltas[0]
    assert top.label == "오른쪽 팔꿈치"
    assert top.distance == pytest.approx(0.15)
    assert top.sigma == pytest.approx(3.0)


def test_compare_poses_reports_the_difference_along_body_axes() -> None:
    """부호가 신체 기준이라 그대로 지시문이 된다. +right/+forward/+up."""
    result = pose_similarity.compare_poses(
        _pose(), _pose_with_lowered_right_elbow(drop=0.15), {"RightForeArm": 0.05}
    )

    right, forward, up = result.deltas[0].body_delta
    assert right == pytest.approx(0.0, abs=1e-9)
    assert forward == pytest.approx(0.0, abs=1e-9)
    assert up == pytest.approx(-0.15)


def test_compare_poses_ranks_by_how_much_each_joint_explains() -> None:
    other = _pose()
    other["RightForeArm"] = (0.28, 0.18, 1.10 - 0.15)
    other["RightHand"] = (0.30, 0.34, 1.02 - 0.05)

    result = pose_similarity.compare_poses(
        _pose(), other, {"RightForeArm": 0.05, "RightHand": 0.05}
    )

    assert [item.joint for item in result.deltas] == ["RightForeArm", "RightHand"]


def test_compare_poses_reports_nothing_measurable_rather_than_a_number() -> None:
    """비교 가능한 관절이 없으면 1.0(완벽)도 0.0(전혀 다름)도 거짓이다."""
    result = pose_similarity.compare_poses(
        _pose(), _pose_with_lowered_right_elbow(drop=0.15), {"LeftHand": 0.05}
    )

    assert result.deltas == []
    assert result.similarity is None


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


def _pose_with_lowered_right_elbow(drop: float = 0.15) -> dict[str, Vec3]:
    """버킷 산포가 0.05이므로 drop=0.15는 σ=3.0, drop=0.05는 σ=1.0(숙련자 통상 범위)이다."""
    pose = _pose()
    elbow = pose["RightForeArm"]
    pose["RightForeArm"] = (elbow[0], elbow[1], elbow[2] - drop)
    return pose


def test_build_corrections_stays_silent_inside_the_expert_range() -> None:
    """숙련자 held-out σ는 p50 0.84 / p95 1.85다. 그 안의 편차를 교정이라 부르면 안 된다."""
    corrections = pose_similarity.build_corrections(
        pose_similarity.normalize_pose(_pose_with_lowered_right_elbow(drop=0.05)), _bucket()
    )

    assert corrections == []


def test_build_corrections_speaks_up_for_a_clear_outlier() -> None:
    corrections = pose_similarity.build_corrections(
        pose_similarity.normalize_pose(_pose_with_lowered_right_elbow(drop=0.15)), _bucket()
    )

    assert [item.joint for item in corrections] == ["RightForeArm"]


def test_correction_threshold_is_derived_from_the_expert_distribution() -> None:
    assert pose_similarity.CORRECTION_SIGMA_THRESHOLD == 2.0


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
    assert abs(top.distance - 0.15) < 1e-6
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


def test_apply_corrections_moves_the_named_joint_onto_the_reference() -> None:
    """교정을 따랐다면 그 관절은 기준 위치에 온다. 향상폭을 재려면 이 상태가 필요하다."""
    normalized = pose_similarity.normalize_pose(_pose_with_lowered_right_elbow(drop=0.15))
    bucket = _bucket()
    corrections = pose_similarity.build_corrections(normalized, bucket)

    fixed = pose_similarity.apply_corrections(normalized, bucket, corrections)

    target = bucket["joints"]["RightForeArm"]["mean"]  # type: ignore[index]
    assert fixed["RightForeArm"] == pytest.approx(tuple(target), abs=1e-9)


def test_apply_corrections_leaves_untouched_joints_alone() -> None:
    normalized = pose_similarity.normalize_pose(_pose_with_lowered_right_elbow(drop=0.15))
    bucket = _bucket()
    corrections = pose_similarity.build_corrections(normalized, bucket)

    fixed = pose_similarity.apply_corrections(normalized, bucket, corrections)

    assert fixed["RightHand"] == normalized["RightHand"]


def test_apply_corrections_can_move_only_part_of_the_way() -> None:
    normalized = pose_similarity.normalize_pose(_pose_with_lowered_right_elbow(drop=0.15))
    bucket = _bucket()
    corrections = pose_similarity.build_corrections(normalized, bucket)

    fixed = pose_similarity.apply_corrections(normalized, bucket, corrections, fraction=0.5)

    assert fixed["RightForeArm"][2] == pytest.approx(
        normalized["RightForeArm"][2] + 0.075, abs=1e-9
    )


def test_gap_closure_is_the_share_of_the_remaining_gap_that_was_closed() -> None:
    assert pose_similarity.gap_closure(0.5, 0.75) == pytest.approx(0.5)
    assert pose_similarity.gap_closure(0.5, 1.0) == pytest.approx(1.0)
    assert pose_similarity.gap_closure(0.5, 0.5) == pytest.approx(0.0)


def test_gap_closure_is_negative_when_the_correction_made_it_worse() -> None:
    assert pose_similarity.gap_closure(0.5, 0.4) < 0.0


def test_gap_closure_is_undefined_without_room_to_improve() -> None:
    """이미 기준과 같으면 개선율을 만들 수 없다. 0으로 적으면 실패처럼 읽힌다."""
    assert pose_similarity.gap_closure(1.0, 1.0) is None


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
