#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import importlib
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

measured_pose = importlib.import_module("measured_pose")
pose_fk_csv = importlib.import_module("pose_fk_csv")
rotation_math = importlib.import_module("rotation_math")
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


# Ry(45)·Rx(30)·Rz(60)을 손으로 전개한 값. 구현에서 읽어 온 값이 아니다.
#
#   Ry(b)Rx(a) = [[cb, sb*sa, sb*ca], [0, ca, -sa], [-sb, cb*sa, cb*ca]]
#   (그것)·Rz(g) = [[cb*cg + sb*sa*sg, -cb*sg + sb*sa*cg, sb*ca],
#                   [ca*sg,             ca*cg,             -sa  ],
#                   [-sb*cg + cb*sa*sg, sb*sg + cb*sa*cg,  cb*ca]]
#
# a=30, b=45, g=60을 넣으면(ca=√3/2, sa=1/2, cb=sb=√2/2, cg=1/2, sg=√3/2) 아래가 된다.
YXZ_30_45_60: tuple[Vec3, Vec3, Vec3] = (
    (0.65973961, -0.43559574, 0.61237244),
    (0.75000000, 0.43301270, -0.50000000),
    (-0.04736717, 0.78914913, 0.61237244),
)


def test_euler_matrix_is_the_measured_body_convention_yxz() -> None:
    """관절 회전은 실측 강체와 같은 규약(`measured_pose.EULER_ORDER` = YXZ, 부호 +)이다.

    한 축만 도는 각으로는 규약을 고정할 수 없다(단일 축 회전은 순서에 무관하다). 세 각이
    모두 0이 아닌 경우로 잡는다 — 같은 각을 옛 ZXY로 돌리면 [0][0]이 0.047이라 갈린다.
    기대값은 위 주석의 전개식에서 손으로 계산했다.
    """
    matrix = pose_fk_csv.euler_matrix((30.0, 45.0, 60.0))
    for row in range(3):
        for column in range(3):
            assert math.isclose(
                matrix[row][column], YXZ_30_45_60[row][column], abs_tol=1e-8
            ), (row, column)


FK_EXCERPT = Path(__file__).resolve().parent / "fixtures" / "fk_gun_excerpt.csv"


def _angle_deg(a, b) -> float:  # noqa: ANN001
    rel = rotation_math.matmul(rotation_math.transpose(a), b)
    cosine = (rel[0][0] + rel[1][1] + rel[2][2] - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def test_fk_chain_right_hand_tracks_gun_relative_rotation() -> None:
    """FK 사슬의 오른손 회전이 실측 건 회전과 같은 만큼 돈다(실데이터 발췌).

    두 프레임 사이 상대회전각은 상수 장착회전·상수 루트회전과 무관하므로 회전 규약만
    가른다. 픽스처는 전형기 20250901 L3 3_1000444_1 분사 구간 400행(오른손 사슬 7관절
    ·건·카메라 컬럼만)이다. 프레임 쌍 200개(seed 20260915). 규약을 옛 ZXY로 되돌리면
    중앙값이 약 31°로 올라 이 단언이 깨진다(YXZ에서 약 4.5°).
    """
    rows = simulator_csv.read_rows(FK_EXCERPT)
    gun = measured_pose.active_gun(rows)
    assert gun is not None
    offsets = pose_fk_csv.bind_offsets(rows[0])
    rng = random.Random(20260915)
    pairs = [(rng.randrange(len(rows)), rng.randrange(len(rows))) for _ in range(200)]
    hand = {}
    gun_rot = {}
    for index in {i for pair in pairs for i in pair}:
        _positions, world = pose_fk_csv.forward_kinematics_with_rotations(rows[index], offsets)
        hand[index] = world["RightHand"]
        gun_rot[index] = rotation_math.euler_matrix(
            measured_pose.EULER_ORDER,
            tuple(simulator_csv.vec3(rows[index], gun, "Rotation")),
            measured_pose.EULER_SIGN,
        )
    diffs = sorted(
        abs(_angle_deg(hand[i], hand[j]) - _angle_deg(gun_rot[i], gun_rot[j])) for i, j in pairs
    )
    median_deg = diffs[len(diffs) // 2]
    assert median_deg < 15.0, median_deg


def test_bind_offsets_exclude_moving_pelvis() -> None:
    """Pelvis Position은 프레임마다 움직이고 월드 좌표도 아니다 — 오프셋으로 싣지 않는다."""
    row = _row()
    row["Pelvis Position X"] = "0.31"
    assert "Pelvis" not in pose_fk_csv.bind_offsets(row)


def test_forward_kinematics_puts_the_root_at_the_origin_whatever_pelvis_position_says() -> None:
    row = _row()
    row["Pelvis Position X"] = "0.31"
    pose = pose_fk_csv.forward_kinematics(row, pose_fk_csv.bind_offsets(row))
    assert pose["Pelvis"] == (0.0, 0.0, 0.0)


def test_rotation_changes_the_pose() -> None:
    straight = _row()
    bent = _row({"RightForeArm": (0.0, 0.0, 90.0)})
    offsets = pose_fk_csv.bind_offsets(straight)
    assert (
        pose_fk_csv.forward_kinematics(bent, offsets)["RightHand"]
        != pose_fk_csv.forward_kinematics(straight, offsets)["RightHand"]
    )
