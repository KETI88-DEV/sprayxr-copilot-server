#!/usr/bin/env python3
"""BVH 계층·모션을 읽어 조인트 월드 좌표를 낸다.

CSV의 조인트 Position 컬럼은 바인드 포즈 로컬 오프셋이라 자세 비교에 쓸 수 없다.
BVH는 회전 순서(CHANNELS), 계층, 루트 월드 변환을 모두 파일 안에 담고 있으므로
FK만으로 실제 월드 좌표가 나온다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Collection, Sequence

__all__ = [
    "BvhClip",
    "BvhJoint",
    "forward_kinematics",
    "normalize_joint_name",
    "parse_bvh",
    "required_indices",
]

Vec3 = tuple[float, float, float]
Mat3 = tuple[Vec3, Vec3, Vec3]

_IDENTITY: Mat3 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

_EXPORTER_PREFIX = "mixamorig:"
# 벤더 BVH는 왼팔만 LeftUpperArm이고 오른팔은 RightArm이다. 좌우 쌍을 맞춘다.
_NAME_ALIASES: dict[str, str] = {"RightArm": "RightUpperArm", "LeftArm": "LeftUpperArm"}


def normalize_joint_name(name: str) -> str:
    """익스포터 접두어를 떼고 좌우 명명 비대칭을 흡수한다."""
    bare = name[len(_EXPORTER_PREFIX) :] if name.startswith(_EXPORTER_PREFIX) else name
    return _NAME_ALIASES.get(bare, bare)


@dataclass(frozen=True, slots=True)
class BvhJoint:
    """계층의 한 노드. End Site는 채널이 없으므로 담지 않는다."""

    name: str
    parent: int
    offset: Vec3
    channels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BvhClip:
    joints: tuple[BvhJoint, ...]
    frame_count: int
    frame_time: float
    frames: tuple[tuple[float, ...], ...]


def _parse_hierarchy(lines: Sequence[str]) -> tuple[list[BvhJoint], int]:
    """계층 블록을 읽어 (조인트 목록, MOTION 줄 인덱스)를 낸다.

    End Site는 스택에 None으로 쌓아 OFFSET/CHANNELS가 부모에 잘못 붙지 않게 한다.
    """
    joints: list[BvhJoint] = []
    stack: list[int | None] = []
    index = 0
    for index, raw in enumerate(lines):
        stripped = raw.strip()
        if stripped == "MOTION":
            return joints, index
        tokens = stripped.split()
        if not tokens:
            continue
        head = tokens[0]
        if head in ("ROOT", "JOINT"):
            parent = next((item for item in reversed(stack) if item is not None), -1)
            stack.append(len(joints))
            name = normalize_joint_name(" ".join(tokens[1:]))
            joints.append(BvhJoint(name, parent, (0.0, 0.0, 0.0), ()))
        elif head == "End":
            stack.append(None)
        elif head == "OFFSET" and stack and stack[-1] is not None:
            current = joints[stack[-1]]
            values = tuple(float(token) for token in tokens[1:4])
            joints[stack[-1]] = BvhJoint(current.name, current.parent, values, current.channels)
        elif head == "CHANNELS" and stack and stack[-1] is not None:
            current = joints[stack[-1]]
            joints[stack[-1]] = BvhJoint(
                current.name, current.parent, current.offset, tuple(tokens[2:])
            )
        elif head == "}" and stack:
            stack.pop()
    raise ValueError("MOTION section not found")


def parse_bvh(path: str | Path) -> BvhClip:
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    joints, motion_at = _parse_hierarchy(lines)

    frame_count = int(lines[motion_at + 1].split()[-1])
    frame_time = float(lines[motion_at + 2].split()[-1])
    frames = tuple(
        tuple(float(token) for token in line.split())
        for line in lines[motion_at + 3 :]
        if line.strip()
    )
    return BvhClip(tuple(joints), frame_count, frame_time, frames)


def _rotation(axis: str, angle_deg: float) -> Mat3:
    radians = math.radians(angle_deg)
    cos = math.cos(radians)
    sin = math.sin(radians)
    if axis == "X":
        return ((1.0, 0.0, 0.0), (0.0, cos, -sin), (0.0, sin, cos))
    if axis == "Y":
        return ((cos, 0.0, sin), (0.0, 1.0, 0.0), (-sin, 0.0, cos))
    return ((cos, -sin, 0.0), (sin, cos, 0.0), (0.0, 0.0, 1.0))


def _mat_mul(left: Mat3, right: Mat3) -> Mat3:
    return tuple(  # type: ignore[return-value]
        tuple(sum(left[r][k] * right[k][c] for k in range(3)) for c in range(3))
        for r in range(3)
    )


def _apply(matrix: Mat3, vector: Vec3) -> Vec3:
    return (
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    )


def required_indices(clip: BvhClip, joints: Collection[str]) -> frozenset[int]:
    """요청 관절과 그 조상들의 인덱스. 나머지 서브트리는 FK를 건너뛸 수 있다."""
    wanted = set(joints)
    needed: set[int] = set()
    for index in range(len(clip.joints) - 1, -1, -1):
        joint = clip.joints[index]
        if joint.name in wanted or index in needed:
            needed.add(index)
            if joint.parent >= 0:
                needed.add(joint.parent)
    return frozenset(needed)


def forward_kinematics(
    clip: BvhClip, frame_index: int, joints: Collection[str] | None = None
) -> dict[str, Vec3]:
    """한 프레임의 조인트 월드 좌표. 회전 합성 순서는 CHANNELS 선언을 그대로 따른다."""
    row = clip.frames[frame_index]
    needed = None if joints is None else required_indices(clip, joints)
    requested = None if joints is None else frozenset(joints)
    cursor = 0
    positions: list[Vec3] = []
    rotations: list[Mat3] = []
    world: dict[str, Vec3] = {}

    for index, joint in enumerate(clip.joints):
        if needed is not None and index not in needed:
            cursor += len(joint.channels)
            positions.append((0.0, 0.0, 0.0))
            rotations.append(_IDENTITY)
            continue

        local_translation = list(joint.offset)
        local_rotation = _IDENTITY
        for channel in joint.channels:
            value = row[cursor]
            cursor += 1
            if channel == "Xposition":
                local_translation[0] = value
            elif channel == "Yposition":
                local_translation[1] = value
            elif channel == "Zposition":
                local_translation[2] = value
            else:
                local_rotation = _mat_mul(local_rotation, _rotation(channel[0], value))

        offset: Vec3 = (local_translation[0], local_translation[1], local_translation[2])
        if joint.parent < 0:
            position = offset
            rotation = local_rotation
        else:
            parent_position = positions[joint.parent]
            parent_rotation = rotations[joint.parent]
            shifted = _apply(parent_rotation, offset)
            position = (
                parent_position[0] + shifted[0],
                parent_position[1] + shifted[1],
                parent_position[2] + shifted[2],
            )
            rotation = _mat_mul(parent_rotation, local_rotation)

        positions.append(position)
        rotations.append(rotation)
        if requested is None or joint.name in requested:
            world[joint.name] = position
    return world
