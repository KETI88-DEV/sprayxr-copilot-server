#!/usr/bin/env python3
"""CSV 자체 리그로 도는 FK. 표시 전용이다.

REQ-18의 CSV-FK 검정은 BVH가 준 계층·본 길이·회전순서를 CSV 오일러에 적용했다. 두
리그를 섞은 것이다. CSV bind offset은 뼈가 -X를 향하고 좌우를 Z로 가르는데 BVH는
+Z와 X이고 뼈 길이도 5% 어긋난다. 69.7mm 오차와 규약이 특정되지 않는 증상이 이것으로
설명된다.

여기서는 CSV 자기 데이터만 쓴다.

**Position 컬럼.** Pelvis를 뺀 19관절의 Position은 18세션에서 프레임 간 std 0.0000m
(Toe 최대 0.0001m)인 부모 기준 바인드 오프셋이다. Pelvis Position은 프레임마다
변하고(세션 std 0.05~0.31m), 카메라와의 프레임 차분 상관이 |r|≤0.16이라 월드 좌표가
아니다. 의미(아바타 루트 이동인지 등)는 확인하지 못했다. FK는 Pelvis를 원점에 두고
이 값을 쓰지 않는다.

**회전 규약.** 관절 Rotation은 부모 기준 로컬 오일러이고, 곱 순서는 실측 강체와 같은
`measured_pose.EULER_ORDER`(YXZ, R = Ry·Rx·Rz)·`EULER_SIGN`(+1)이다. 사슬은
W_child = W_parent · L_child다. 규약을 리터럴로 여기 다시 적지 않고 `measured_pose`
상수를 읽는다 — 두 곳에 적으면 한쪽만 바뀐다.

근거는 두 프레임 사이의 **상대회전각**이다. 이 각은 관절↔강체 상수 장착회전과 상수
루트회전의 영향을 받지 않으므로 규약만 가른다. 작업자별 5세션씩 30세션(seed
20260915, 세션당 800쌍)에서 FK 사슬 관절과 짝 강체(Head↔카메라, RightHand↔활성 건,
LeftHand↔플래쉬)의 상대회전각 차이 — 세션 중앙값의 중앙값 — 가 YXZ+에서 오른손
6.5°·왼손 6.3°·머리 6.1°였고, 옛 코드의 ZXY(Rz·Rx·Ry)에서는 37.7°·57.5°·32.0°였다
(스크래치패드 sk_p10.py). 옛 판단("BVH가 Zrotation Xrotation Yrotation을 선언했다")은
다른 리그(mixamo BVH)의 선언을 CSV에 옮긴 것이었다.

채점은 이 경로를 타지 않는다. 실측 3점에서 하므로 FK가 틀려도 지표는 오염되지 않는다.
"""

from __future__ import annotations

from typing import Final, Mapping, Sequence

import measured_pose
import rotation_math
import simulator_csv

Vec3 = tuple[float, float, float]
Matrix = tuple[Vec3, Vec3, Vec3]

IDENTITY: Final[Matrix] = rotation_math.IDENTITY
ROOT: Final = "Pelvis"


def bind_offsets(row: dict[str, str]) -> dict[str, Vec3]:
    """부모 기준 바인드 오프셋. **Pelvis는 싣지 않는다.**

    Pelvis를 뺀 19관절의 Position은 프레임 불변인 부모 기준 마디 벡터다. Pelvis의
    Position은 프레임마다 움직이고 월드 좌표도 아니라(모듈 독스트링) 오프셋이 아니다 —
    FK는 루트를 원점에 두므로 이 값이 필요 없고, 실어 두면 다음 사람이 상수로 읽는다.
    """
    offsets: dict[str, Vec3] = {}
    for column in simulator_csv.JOINT_COLUMNS:
        key = simulator_csv.JOINT_KEYS[column]
        if key == ROOT:
            continue
        try:
            x, y, z = simulator_csv.vec3(row, column, "Position")
        except (KeyError, ValueError):
            continue
        offsets[key] = (x, y, z)
    return offsets


def euler_matrix(degrees_xyz: Vec3) -> Matrix:
    """관절 로컬 오일러 → 회전행렬. 규약은 실측 강체와 같은 `measured_pose`의 것이다."""
    return rotation_math.euler_matrix(
        measured_pose.EULER_ORDER, degrees_xyz, measured_pose.EULER_SIGN
    )


def forward_kinematics_with_rotations(
    row: dict[str, str], offsets: Mapping[str, Vec3]
) -> tuple[dict[str, Vec3], dict[str, Matrix]]:
    """관절 위치(Pelvis 원점 리그 좌표)와 관절 월드(리그) 회전 W를 함께 낸다.

    W가 필요한 소비자는 `pose_lab.skeleton.place_oriented`다 — 실측 강체 회전과 FK 관절
    회전을 맞춰 루트 방향을 푼다.
    """
    local: dict[str, Matrix] = {}
    for column in simulator_csv.JOINT_COLUMNS:
        key = simulator_csv.JOINT_KEYS[column]
        try:
            x, y, z = simulator_csv.vec3(row, column, "Rotation")
        except (KeyError, ValueError):
            x, y, z = 0.0, 0.0, 0.0
        local[key] = euler_matrix((x, y, z))

    positions: dict[str, Vec3] = {ROOT: (0.0, 0.0, 0.0)}
    world: dict[str, Matrix] = {ROOT: local.get(ROOT, IDENTITY)}

    for parent, child in simulator_csv.BONES:
        if parent not in positions or child not in offsets:
            continue
        parent_rotation = world[parent]
        offset = rotation_math.matvec(parent_rotation, offsets[child])
        base = positions[parent]
        positions[child] = (base[0] + offset[0], base[1] + offset[1], base[2] + offset[2])
        world[child] = rotation_math.matmul(parent_rotation, local.get(child, IDENTITY))

    return positions, world


def forward_kinematics(
    row: dict[str, str], offsets: Mapping[str, Vec3]
) -> dict[str, Vec3]:
    """루트에서 시작해 계층을 따라 내려가며 관절 위치를 만든다."""
    positions, _world = forward_kinematics_with_rotations(row, offsets)
    return positions


def pass_inputs(
    rows: Sequence[dict[str, str]], *, min_frames: int = 6
) -> tuple[list[dict[str, Vec3]], list[Vec3], list[Vec3]] | None:
    """패스 하나에서 (프레임별 월드 관절, 착탄점, 건 위치). 못 만들면 `None`.

    **채점 경로에 넣을 재료를 만드는 유일한 곳이다.** 실험 두 개와 pose_lab이 각자
    베껴 쓰고 있었는데, 한 곳만 고치면 같은 패스가 도구마다 다른 값을 내게 된다.

    FK가 실패한 프레임은 빠지고, 착탄·건은 **남은 프레임에 맞춰** 골라낸다.
    착탄이 프레임 수와 어긋나면 위상에 엉뚱한 자세가 들어가므로 `None`을 낸다.
    """
    if len(rows) < min_frames:
        return None
    offsets = bind_offsets(rows[0])
    if not offsets:
        return None
    frames: list[dict[str, Vec3]] = []
    keep: list[int] = []
    for index, row in enumerate(rows):
        try:
            frames.append(dict(forward_kinematics(row, offsets)))
        except (KeyError, ValueError):
            continue
        keep.append(index)
    if len(frames) < min_frames:
        return None
    measured = measured_pose.measured_frames(list(rows))
    impacts: list[Vec3] = []
    tools: list[Vec3] = []
    for index in keep:
        if index >= len(measured):
            continue
        frame = measured[index]
        if frame.hit is not None:
            impacts.append(tuple(frame.hit))  # type: ignore[arg-type]
        if frame.right is not None:
            tools.append(tuple(frame.right))  # type: ignore[arg-type]
    if len(impacts) != len(frames) or not tools:
        return None
    return frames, impacts, tools
