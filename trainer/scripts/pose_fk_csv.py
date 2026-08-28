#!/usr/bin/env python3
"""CSV 자체 리그로 도는 FK. 표시 전용이다.

REQ-18의 CSV-FK 검정은 BVH가 준 계층·본 길이·회전순서를 CSV 오일러에 적용했다. 두
리그를 섞은 것이다. CSV bind offset은 뼈가 -X를 향하고 좌우를 Z로 가르는데 BVH는
+Z와 X이고 뼈 길이도 5% 어긋난다. 69.7mm 오차와 규약이 특정되지 않는 증상이 이것으로
설명된다.

여기서는 CSV 자기 데이터만 쓴다. Position 컬럼의 프레임 불변 상수가 bind offset이고
회전순서는 ZXY다. BVH가 65개 관절 전부에 Zrotation Xrotation Yrotation을 균일
선언했고 이는 Unity의 기본 오일러 순서와 같다.

채점은 이 경로를 타지 않는다. 실측 3점에서 하므로 FK가 틀려도 지표는 오염되지 않는다.
"""

from __future__ import annotations

import math
from typing import Final, Mapping

import simulator_csv

Vec3 = tuple[float, float, float]
Matrix = tuple[Vec3, Vec3, Vec3]

ROTATION_ORDER: Final = "ZXY"
IDENTITY: Final[Matrix] = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def bind_offsets(row: dict[str, str]) -> dict[str, Vec3]:
    """Position 컬럼은 프레임 불변 상수다. 월드 좌표가 아니라 부모 기준 마디 벡터다."""
    offsets: dict[str, Vec3] = {}
    for column in simulator_csv.JOINT_COLUMNS:
        try:
            x, y, z = simulator_csv.vec3(row, column, "Position")
        except (KeyError, ValueError):
            continue
        offsets[simulator_csv.JOINT_KEYS[column]] = (x, y, z)
    return offsets


def _axis_matrix(axis: str, degrees: float) -> Matrix:
    radians = math.radians(degrees)
    cosine, sine = math.cos(radians), math.sin(radians)
    if axis == "X":
        return ((1.0, 0.0, 0.0), (0.0, cosine, -sine), (0.0, sine, cosine))
    if axis == "Y":
        return ((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine))
    return ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))


def _multiply(left: Matrix, right: Matrix) -> Matrix:
    rows: list[Vec3] = []
    for i in range(3):
        rows.append(
            (
                sum(left[i][k] * right[k][0] for k in range(3)),
                sum(left[i][k] * right[k][1] for k in range(3)),
                sum(left[i][k] * right[k][2] for k in range(3)),
            )
        )
    return (rows[0], rows[1], rows[2])


def euler_matrix(degrees_xyz: Vec3, order: str = ROTATION_ORDER) -> Matrix:
    """오일러 각을 회전행렬로. 각은 XYZ 순으로 들어오고 적용 순서는 order다."""
    angles = {"X": degrees_xyz[0], "Y": degrees_xyz[1], "Z": degrees_xyz[2]}
    matrix: Matrix = IDENTITY
    for axis in order:
        matrix = _multiply(matrix, _axis_matrix(axis, angles[axis]))
    return matrix


def _apply(matrix: Matrix, vector: Vec3) -> Vec3:
    return (
        sum(matrix[0][j] * vector[j] for j in range(3)),
        sum(matrix[1][j] * vector[j] for j in range(3)),
        sum(matrix[2][j] * vector[j] for j in range(3)),
    )


def forward_kinematics(
    row: dict[str, str], offsets: Mapping[str, Vec3]
) -> dict[str, Vec3]:
    """루트에서 시작해 계층을 따라 내려가며 관절 위치를 만든다."""
    rotations: dict[str, Matrix] = {}
    for column in simulator_csv.JOINT_COLUMNS:
        key = simulator_csv.JOINT_KEYS[column]
        try:
            x, y, z = simulator_csv.vec3(row, column, "Rotation")
        except (KeyError, ValueError):
            x, y, z = 0.0, 0.0, 0.0
        rotations[key] = euler_matrix((x, y, z))

    root = "Pelvis"
    positions: dict[str, Vec3] = {root: (0.0, 0.0, 0.0)}
    world: dict[str, Matrix] = {root: rotations.get(root, IDENTITY)}

    for parent, child in simulator_csv.BONES:
        if parent not in positions or child not in offsets:
            continue
        parent_rotation = world[parent]
        offset = _apply(parent_rotation, offsets[child])
        base = positions[parent]
        positions[child] = (base[0] + offset[0], base[1] + offset[1], base[2] + offset[2])
        world[child] = _multiply(parent_rotation, rotations.get(child, IDENTITY))

    return positions
