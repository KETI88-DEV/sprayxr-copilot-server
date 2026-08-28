#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pose_similarity = importlib.import_module("pose_similarity")
simulator_csv = importlib.import_module("simulator_csv")

Vec3 = tuple[float, float, float]


def _pose() -> dict[str, Vec3]:
    """뼈 길이가 명확한 합성 자세. 팔은 X로, 다리는 -Z로 뻗는다."""
    return {
        "Pelvis": (0.0, 0.0, 0.0),
        "Spine": (0.0, 0.0, 0.1),
        "Chest": (0.0, 0.0, 0.2),
        "Neck": (0.0, 0.0, 0.3),
        "Head": (0.0, 0.0, 0.4),
        "LeftShoulder": (0.1, 0.0, 0.2),
        "LeftUpperArm": (0.3, 0.0, 0.2),
        "LeftForeArm": (0.5, 0.0, 0.2),
        "LeftHand": (0.7, 0.0, 0.2),
        "RightShoulder": (-0.1, 0.0, 0.2),
        "RightUpperArm": (-0.3, 0.0, 0.2),
        "RightForeArm": (-0.5, 0.0, 0.2),
        "RightHand": (-0.7, 0.0, 0.2),
        "LeftThigh": (0.1, 0.0, -0.1),
        "LeftCalf": (0.1, 0.0, -0.5),
        "LeftFoot": (0.1, 0.0, -0.9),
        "LeftToe": (0.1, 0.1, -0.9),
        "RightThigh": (-0.1, 0.0, -0.1),
        "RightCalf": (-0.1, 0.0, -0.5),
        "RightFoot": (-0.1, 0.0, -0.9),
        "RightToe": (-0.1, 0.1, -0.9),
    }


def _bucket() -> dict[str, object]:
    """RightForeArm만 크게 벗어난 것으로 보이게 만든 레퍼런스."""
    joints: dict[str, object] = {}
    for name, position in _pose().items():
        target = position
        if name == "RightForeArm":
            target = (position[0], position[1] + 0.3, position[2])
        joints[name] = {"mean": list(target), "spread": 0.02}
    return {"joints": joints}


def test_bone_lengths_reads_every_bone() -> None:
    lengths = pose_similarity.bone_lengths(_pose())
    assert len(lengths) == len(simulator_csv.BONES)
    assert math.isclose(lengths["RightForeArm"], 0.2, abs_tol=1e-9)


def test_rigid_correction_preserves_every_bone_length() -> None:
    """관절을 목표로 그냥 옮기면 골격이 늘어나 물리적으로 불가능한 자세가 된다."""
    pose = pose_similarity.normalize_pose(_pose())
    bucket = _bucket()
    corrections = pose_similarity.build_corrections(pose, bucket)
    assert corrections, "테스트 레퍼런스가 교정을 유발해야 한다"

    fixed = pose_similarity.apply_corrections_rigid(pose, bucket, corrections)
    before = pose_similarity.bone_lengths(pose)
    after = pose_similarity.bone_lengths(fixed)
    for joint, length in before.items():
        assert math.isclose(after[joint], length, abs_tol=1e-6), joint


def test_rigid_correction_carries_descendants() -> None:
    """팔꿈치를 돌리면 손도 따라가야 뼈 길이가 보존된다."""
    pose = pose_similarity.normalize_pose(_pose())
    bucket = _bucket()
    corrections = pose_similarity.build_corrections(pose, bucket)
    fixed = pose_similarity.apply_corrections_rigid(pose, bucket, corrections)
    assert fixed["RightHand"] != pose["RightHand"]


def test_existing_apply_corrections_is_unchanged() -> None:
    """기존 함수는 그대로 둔다. 소비자가 eval 하네스와 MCP다."""
    pose = pose_similarity.normalize_pose(_pose())
    bucket = _bucket()
    corrections = pose_similarity.build_corrections(pose, bucket)
    result = pose_similarity.apply_corrections(pose, bucket, corrections)
    assert isinstance(result, dict)
    assert set(result) == set(pose)
