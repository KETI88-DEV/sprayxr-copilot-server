"""`skeleton.place_oriented`(M6)와 장착 적합(`fit_mounts`)의 성질.

합성 리그: 루트 회전·관절 회전·장착 상수를 알고 있는 상태에서 실측 강체 위치·회전을 만들어
배치가 그것을 되찾는지, 뼈 길이를 바꾸지 않는지, 중력 투영을 실제로 하는지 본다.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from trainer.pose_lab import skeleton
from trainer.pose_lab.skeleton import AnchorMount, FrameSample, Mounts

import importlib

pose_fk_csv = importlib.import_module("pose_fk_csv")
rotation_math = importlib.import_module("rotation_math")
simulator_csv = importlib.import_module("simulator_csv")
world_axes = importlib.import_module("world_axes")

OFFSETS = {
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

MOUNTS = Mounts(
    head=AnchorMount(rotation=rotation_math.euler_matrix("YXZ", (10.0, -20.0, 5.0)), offset=(0.05, 0.12, -0.2)),
    right=AnchorMount(rotation=rotation_math.euler_matrix("YXZ", (-30.0, 40.0, 15.0)), offset=(-0.4, 0.1, 0.5)),
    left=AnchorMount(rotation=rotation_math.euler_matrix("YXZ", (5.0, 70.0, -10.0)), offset=(0.1, -0.05, 0.08)),
)


def _row(rng: random.Random) -> dict[str, str]:
    row: dict[str, str] = {}
    for column in simulator_csv.JOINT_COLUMNS:
        key = simulator_csv.JOINT_KEYS[column]
        for axis, value in zip("XYZ", OFFSETS.get(key, (0.3, -0.2, 0.1))):
            row[f"{column} Position {axis}"] = str(value)
        for axis in "XYZ":
            row[f"{column} Rotation {axis}"] = str(rng.uniform(-40.0, 40.0))
    return row


def _synthetic_frame(rng: random.Random, root: np.ndarray, t: np.ndarray, mounts: Mounts = MOUNTS):
    """루트 회전 `root`(리그 +Z가 월드 UP으로 가도록 이미 선 회전)·이동 `t`에서 실측을 만든다."""
    row = _row(rng)
    offsets = pose_fk_csv.bind_offsets(row)
    pos, rot = pose_fk_csv.forward_kinematics_with_rotations(row, offsets)
    measured = {}
    for point, joint in skeleton.ANCHOR_JOINTS.items():
        mount = mounts.get(point)
        r_meas = root @ np.array(rot[joint]) @ np.array(mount.rotation)
        marker = root @ np.array(pos[joint]) + t + r_meas @ np.array(mount.offset)
        measured[point] = (tuple(float(x) for x in marker), tuple(tuple(float(x) for x in r) for r in r_meas))
    return row, offsets, pos, rot, measured


def _upright_root(yaw_deg: float) -> np.ndarray:
    """리그 +Z → 월드 +Y(UP), 그 뒤 수직축 둘레 yaw."""
    to_up = skeleton.minimal_rotation(np.array(skeleton.RIG_UP), np.array(world_axes.UP, dtype=float))
    yaw = np.array(rotation_math.axis_matrix("Y", yaw_deg))
    return yaw @ to_up


def test_place_oriented_recovers_a_known_upright_placement() -> None:
    rng = random.Random(1)
    root = _upright_root(35.0)
    t = np.array([1.2, 0.4, -3.0])
    _row_, offsets, pos, rot, measured = _synthetic_frame(rng, root, t)
    placed = skeleton.place_oriented(pos, rot, measured, MOUNTS)
    assert placed is not None
    for name, p in pos.items():
        expected = root @ np.array(p) + t
        assert np.allclose(placed.joints[name], expected, atol=1e-9), name
    for fit in placed.anchors:
        assert fit.miss_m < 1e-9
    assert placed.gravity_correction_deg < 1e-6


def test_gravity_projection_is_applied() -> None:
    """투영 뒤 리그 위(A·ẑ)는 정확히 월드 UP이고, 보고하는 보정량은 투영 **전** A0·ẑ와 UP 사이 각이다.

    이 테스트는 자세가 참임을 보호하지 않는다 — 중력 투영은 측정이 아니라 가정의 덮어쓰기다.
    투영을 했다는 사실과 그 크기를 숨기지 않았다는 사실만 고정한다.
    """
    rng = random.Random(2)
    tilt = np.array(rotation_math.axis_matrix("X", 27.0))
    root = tilt @ _upright_root(-60.0)
    _row_, _offsets, pos, rot, measured = _synthetic_frame(rng, root, np.array([0.0, 1.0, 0.0]))
    placed = skeleton.place_oriented(pos, rot, measured, MOUNTS)
    assert placed is not None
    a = np.array(placed.rotation)
    a0 = np.array(placed.rotation_before_gravity)
    up = np.array(world_axes.UP, dtype=float)
    assert skeleton._angle_between_deg(a @ np.array(skeleton.RIG_UP), up) < 1e-6  # noqa: SLF001
    expected = skeleton._angle_between_deg(a0 @ np.array(skeleton.RIG_UP), up)  # noqa: SLF001
    assert placed.gravity_correction_deg == expected
    assert abs(placed.gravity_correction_deg - 27.0) < 1e-6


def test_placement_is_rigid_so_bone_lengths_are_the_rig_lengths() -> None:
    rng = random.Random(3)
    for k in range(20):
        root = np.array(rotation_math.euler_matrix("YXZ", (rng.uniform(-30, 30), rng.uniform(-180, 180), rng.uniform(-30, 30))))
        _row_, offsets, pos, rot, measured = _synthetic_frame(rng, root, np.array([k * 0.1, 0.0, 0.0]))
        placed = skeleton.place_oriented(pos, rot, measured, MOUNTS)
        assert placed is not None
        assert max(skeleton.bone_length_errors(placed.joints, offsets)) < 1e-9


def test_anchor_miss_is_the_distance_between_predicted_and_measured_marker() -> None:
    rng = random.Random(4)
    _row_, _offsets, pos, rot, measured = _synthetic_frame(rng, _upright_root(0.0), np.zeros(3))
    marker, r = measured["right"]
    measured["right"] = ((marker[0] + 0.1, marker[1], marker[2]), r)
    placed = skeleton.place_oriented(pos, rot, measured, MOUNTS)
    assert placed is not None
    for fit in placed.anchors:
        assert fit.measured_marker == measured[fit.point][0]
        assert abs(fit.miss_m - math.dist(fit.predicted_marker, fit.measured_marker)) < 1e-12
    assert max(fit.miss_m for fit in placed.anchors) > 0.03


def test_fewer_than_two_anchors_is_none() -> None:
    rng = random.Random(5)
    _row_, _offsets, pos, rot, measured = _synthetic_frame(rng, _upright_root(0.0), np.zeros(3))
    measured["right"] = None
    measured["left"] = None
    assert skeleton.place_oriented(pos, rot, measured, MOUNTS) is None


def test_two_anchors_still_place() -> None:
    rng = random.Random(6)
    _row_, _offsets, pos, rot, measured = _synthetic_frame(rng, _upright_root(10.0), np.zeros(3))
    measured["left"] = None
    placed = skeleton.place_oriented(pos, rot, measured, MOUNTS)
    assert placed is not None
    assert {fit.point for fit in placed.anchors} == {"head", "right"}
    assert placed.rotation_residual_deg["left"] is None


def test_fit_mounts_recovers_known_constants_from_moving_root() -> None:
    rng = random.Random(7)
    sessions = []
    for s in range(2):
        session = []
        for i in range(60):
            root = _upright_root(rng.uniform(-180, 180))
            tilt = np.array(rotation_math.axis_matrix("X", rng.uniform(-8, 8)))
            _row_, _offsets, pos, rot, measured = _synthetic_frame(rng, tilt @ root, np.array([i * 0.01, 0.0, s]))
            session.append(FrameSample(local_pos=pos, local_rot=rot, measured=measured, group="한길숏건", index=i))
        sessions.append(session)
    table = skeleton.fit_mounts(sessions)
    for point, key in (("head", "head"), ("left", "left"), ("right", "right:한길숏건")):
        truth = MOUNTS.get(point)
        got = table[key]
        assert skeleton.rotation_angle_deg(np.array(got.rotation), np.array(truth.rotation)) < 1e-3, point
        # 오프셋은 중력 투영 A로 풀므로 합성 루트 기울기(±8°)만큼 치우칠 수 있다.
        assert np.linalg.norm(np.array(got.offset) - np.array(truth.offset)) < 0.05, point


def test_minimal_rotation_handles_antiparallel() -> None:
    up = np.array(world_axes.UP, dtype=float)
    r = skeleton.minimal_rotation(-up, up)
    assert np.allclose(r @ -up, up, atol=1e-12)
    assert abs(np.linalg.det(r) - 1.0) < 1e-12


def test_standing_legs_hang_along_gravity_with_rig_bone_lengths() -> None:
    """자리표시 다리: 엉덩이에서 중력 아래로 리그 허벅지·종아리 길이, 발끝은 수평 전방으로 리그 발 길이.
    자세를 말하지 않는 표시이므로 방향은 정확히 −UP이고 길이는 오프셋 노름과 같아야 한다."""
    up = np.array(world_axes.UP, dtype=float)
    joints = {"LeftThigh": (0.3, 1.1, 0.2), "RightThigh": (0.1, 1.05, 0.25)}
    forward = (1.0, 0.7, 0.5)  # 수직 성분은 버린다
    result = skeleton.standing_legs(joints, OFFSETS, forward)
    assert result is not None
    # 메시(바닥)를 주지 않으면 잴 면이 없다.
    assert result.floor_vertical is None and result.feet_below_floor_m is None
    legs = result.joints
    for hip, knee, ankle, toe in skeleton.PLACEHOLDER_LEG_CHAIN:
        assert legs[hip] == joints[hip]
        for a, b, child in ((hip, knee, knee), (knee, ankle, ankle)):
            d = np.array(legs[b]) - np.array(legs[a])
            assert abs(np.linalg.norm(d) - math.hypot(*OFFSETS[child])) < 1e-9
            assert np.linalg.norm(d / np.linalg.norm(d) + up) < 1e-9
        foot = np.array(legs[toe]) - np.array(legs[ankle])
        assert abs(np.linalg.norm(foot) - math.hypot(*OFFSETS[toe])) < 1e-9
        assert abs(float(np.dot(foot, up))) < 1e-12
    # 수평 성분이 없는 전방이면 발끝 방향을 지어내지 않는다.
    no_toe = skeleton.standing_legs(joints, OFFSETS, tuple(up))
    assert no_toe is not None and "LeftToe" not in no_toe.joints and "RightToe" not in no_toe.joints
    assert skeleton.standing_legs({"LeftThigh": (0.0, 0.0, 0.0)}, OFFSETS, forward) is None



def test_standing_legs_never_scale_to_reach_the_floor() -> None:
    """바닥면을 줘도 다리 길이는 리그 그대로다(U5) — 재는 것은 "발이 그 면보다 얼마나 아래인가"뿐이다.

    옛 코드는 발을 그 면에 닿추려고 허벅지·종아리에 한 배율(0.25~2.0)을 곱했다. 레벨 3에서 배율이
    0.35라 작업자가 난쟁이로 보였고, 그 면은 애초에 작업자가 선 지면이 아니었다
    (`skeleton.py` D3 주석). 여기서는 **어떤 바닥 높이를 줘도** 뼈 길이가 안 변하는 것을 고정한다.
    """
    up = np.array(world_axes.UP, dtype=float)
    joints = {"LeftThigh": (0.3, 1.1, 0.2), "RightThigh": (0.1, 1.05, 0.25)}
    forward = (1.0, 0.7, 0.5)
    free = skeleton.standing_legs(joints, OFFSETS, forward)
    assert free is not None
    lowest = min(float(np.dot(np.array(free.joints[ankle]), up)) for _h, _k, ankle, _t in skeleton.PLACEHOLDER_LEG_CHAIN)
    # 발보다 한참 위·한참 아래, 옛 배율 한계를 넘는 자리까지 훑는다.
    for floor in (lowest + 2.0, lowest + 0.5, lowest, lowest - 0.5, lowest - 3.0):
        snapped = skeleton.standing_legs(joints, OFFSETS, forward, floor=floor)
        assert snapped is not None
        assert snapped.joints == free.joints, floor  # 관절이 한 점도 안 움직인다
        assert snapped.floor_vertical == pytest.approx(floor)
        assert snapped.feet_below_floor_m == pytest.approx(floor - lowest, abs=1e-12)
        for hip, knee, ankle, _toe in skeleton.PLACEHOLDER_LEG_CHAIN:
            for a, b, child in ((hip, knee, knee), (knee, ankle, ankle)):
                d = np.array(snapped.joints[b]) - np.array(snapped.joints[a])
                assert abs(np.linalg.norm(d) - math.hypot(*OFFSETS[child])) < 1e-12, (floor, child)
