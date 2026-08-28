#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pose_fk_csv = importlib.import_module("pose_fk_csv")
simulator_csv = importlib.import_module("simulator_csv")

Vec3 = tuple[float, float, float]

# 실측 bind offset. 뼈가 -X를 향하고 좌우를 Z로 가른다. BVH는 +Z와 X라 규약이 다르다.
OFFSETS: dict[str, Vec3] = {
    "Pelvis": (0.0, 0.0, 0.0),
    "Spine": (-0.1053, -0.0001, 0.0),
    "Chest": (-0.1053, -0.0001, 0.0),
    "Head": (-0.1368, 0.0, 0.0),
    "LeftShoulder": (0.0, 0.0001, 0.0284),
    "LeftUpperArm": (-0.1797, 0.0, 0.0),
    "LeftForeArm": (-0.2430, 0.0, 0.0),
    "LeftHand": (-0.2430, 0.0, 0.0),
    "RightShoulder": (0.0, 0.0001, -0.0284),
    "RightUpperArm": (-0.1797, 0.0, 0.0),
    "RightForeArm": (-0.2430, 0.0, 0.0),
    "RightHand": (-0.2430, 0.0, 0.0),
    "LeftThigh": (0.0931, -0.0024, 0.0965),
    "LeftCalf": (-0.4223, 0.0, 0.0),
    "LeftFoot": (-0.4110, 0.0, 0.0),
    "LeftToe": (-0.0672, 0.1337, 0.0),
    "RightThigh": (0.0931, -0.0024, -0.0965),
    "RightCalf": (-0.4223, 0.0, 0.0),
    "RightFoot": (-0.4110, 0.0, 0.0),
    "RightToe": (-0.0672, 0.1337, 0.0),
}


def _row(rotations: dict[str, Vec3] | None = None) -> dict[str, str]:
    row: dict[str, str] = {}
    for column in simulator_csv.JOINT_COLUMNS:
        key = simulator_csv.JOINT_KEYS[column]
        offset = OFFSETS.get(key, (0.0, 0.0, 0.0))
        for axis, value in zip("XYZ", offset):
            row[f"{column} Position {axis}"] = str(value)
        angles = (rotations or {}).get(key, (0.0, 0.0, 0.0))
        for axis, value in zip("XYZ", angles):
            row[f"{column} Rotation {axis}"] = str(value)
    return row


def test_bind_offsets_are_read_from_position_columns() -> None:
    offsets = pose_fk_csv.bind_offsets(_row())
    assert math.isclose(offsets["RightForeArm"][0], -0.2430, abs_tol=1e-9)
    assert math.isclose(offsets["RightShoulder"][2], -0.0284, abs_tol=1e-9)


def test_identity_rotation_reproduces_the_bind_pose() -> None:
    row = _row()
    pose = pose_fk_csv.forward_kinematics(row, pose_fk_csv.bind_offsets(row))
    # Pelvis -> Spine -> Chest 는 각 -0.1053 이므로 Chest 는 -0.2106 이다.
    assert math.isclose(pose["Chest"][0], -0.2106, abs_tol=1e-6)
    assert math.isclose(pose["Pelvis"][0], 0.0, abs_tol=1e-9)


def test_fk_preserves_bone_lengths_under_any_rotation() -> None:
    row = _row({"RightUpperArm": (30.0, 15.0, 45.0), "Chest": (10.0, 0.0, 20.0)})
    offsets = pose_fk_csv.bind_offsets(row)
    pose = pose_fk_csv.forward_kinematics(row, offsets)
    for parent, child in simulator_csv.BONES:
        if parent not in pose or child not in pose:
            continue
        expected = math.sqrt(sum(c * c for c in offsets[child]))
        assert math.isclose(math.dist(pose[parent], pose[child]), expected, abs_tol=1e-6), child


def test_euler_matrix_is_zxy() -> None:
    """BVH가 65개 관절 전부에 Zrotation Xrotation Yrotation을 선언했다. Unity 기본과 같다."""
    matrix = pose_fk_csv.euler_matrix((0.0, 0.0, 90.0))
    assert math.isclose(matrix[0][0], 0.0, abs_tol=1e-9)
    assert math.isclose(matrix[1][0], 1.0, abs_tol=1e-9)


def test_rotation_changes_the_pose() -> None:
    straight = _row()
    bent = _row({"RightForeArm": (0.0, 0.0, 90.0)})
    offsets = pose_fk_csv.bind_offsets(straight)
    assert (
        pose_fk_csv.forward_kinematics(bent, offsets)["RightHand"]
        != pose_fk_csv.forward_kinematics(straight, offsets)["RightHand"]
    )
