#!/usr/bin/env python3
"""숙련자 버킷 자세 분포와 대조해 유사도와 교정 항목을 낸다.

자세는 루트 기준 상대량으로 본다. BVH 월드와 시뮬레이터 월드의 강체 정합이
없어도 비교가 성립하고, 작업자가 부스 어디에 서서 어느 쪽을 보든 같은 자세는
같게 나온다. 수직축은 회전하지 않는다. 높이 차는 제거할 좌표계 차이가 아니라
자세 정보 자체이기 때문이다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Mapping

__all__ = [
    "CORRECTION_JOINTS",
    "Correction",
    "FACING_JOINTS",
    "JOINT_LABELS",
    "ROOT_JOINT",
    "SCORING_JOINTS",
    "build_corrections",
    "joint_sigmas",
    "normalize_pose",
    "similarity_score",
]

Vec3 = tuple[float, float, float]

ROOT_JOINT: Final = "Pelvis"
FACING_JOINTS: Final = ("LeftShoulder", "RightShoulder")
MIN_SPREAD_M: Final = 0.005
MAX_SIGMA: Final = 6.0
MIN_ANGLE_DEG: Final = 3.0
MAX_CORRECTIONS: Final = 6

SCORING_JOINTS: Final = (
    "Spine",
    "Chest",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftUpperArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightUpperArm",
    "RightForeArm",
    "RightHand",
)

# 채점 집합이 교정 집합을 진부분집합으로 포함해야 교정으로 점수를 끌어올리는 게이밍이 막힌다.
CORRECTION_JOINTS: Final = (
    "Chest",
    "LeftUpperArm",
    "LeftForeArm",
    "RightUpperArm",
    "RightForeArm",
    "RightHand",
)

JOINT_LABELS: Final = {
    "Spine": "허리",
    "Chest": "상체",
    "Neck": "목",
    "Head": "머리",
    "LeftShoulder": "왼쪽 어깨",
    "LeftUpperArm": "왼쪽 위팔",
    "LeftForeArm": "왼쪽 팔꿈치",
    "LeftHand": "왼쪽 손목",
    "RightShoulder": "오른쪽 어깨",
    "RightUpperArm": "오른쪽 위팔",
    "RightForeArm": "오른쪽 팔꿈치",
    "RightHand": "오른쪽 손목",
}

# 관절을 움직이는 것은 그 앞 마디다. 팔꿈치가 내려갔으면 위팔 각도를 올려야 한다.
SEGMENT_PARENT: Final = {
    "Chest": "Spine",
    "LeftUpperArm": "LeftShoulder",
    "LeftForeArm": "LeftUpperArm",
    "LeftHand": "LeftForeArm",
    "RightUpperArm": "RightShoulder",
    "RightForeArm": "RightUpperArm",
    "RightHand": "RightForeArm",
}


@dataclass(frozen=True, slots=True)
class Correction:
    joint: str
    label: str
    distance: float
    angle_corrections: dict[str, float]
    correction_text: str
    expected_gain: float


def _facing_yaw(pose: Mapping[str, Vec3]) -> float:
    """어깨선의 방위각. 이 각만큼 되돌려 몸이 항상 같은 쪽을 보게 만든다."""
    left = pose.get(FACING_JOINTS[0])
    right = pose.get(FACING_JOINTS[1])
    if left is None or right is None:
        return 0.0
    return math.atan2(right[1] - left[1], right[0] - left[0])


def normalize_pose(pose: Mapping[str, Vec3]) -> dict[str, Vec3]:
    """루트를 원점으로 옮기고 수직축 둘레로만 회전시켜 정면을 맞춘다."""
    origin = pose.get(ROOT_JOINT, (0.0, 0.0, 0.0))
    angle = -_facing_yaw(pose)
    cos = math.cos(angle)
    sin = math.sin(angle)

    normalized: dict[str, Vec3] = {}
    for name, (x, y, z) in pose.items():
        dx = x - origin[0]
        dy = y - origin[1]
        normalized[name] = (dx * cos - dy * sin, dx * sin + dy * cos, z - origin[2])
    return normalized


def joint_sigmas(
    normalized: Mapping[str, Vec3], bucket: Mapping[str, object]
) -> dict[str, float]:
    """관절별 편차를 숙련자 산포로 나눈 값. 산포 하한은 측정 잡음이 σ를 부풀리는 것을 막는다.

    채점 대상은 아티팩트에 든 관절 전부가 아니라 선언된 SCORING_JOINTS다. 루트는
    정규화로 항상 원점이라 σ가 구조적으로 0이고, 채점에 섞이면 평균에 0이 강제로
    들어가 유사도가 낙관 편향된다.
    """
    joints = bucket.get("joints")
    if not isinstance(joints, Mapping):
        return {}

    sigmas: dict[str, float] = {}
    for name in SCORING_JOINTS:
        summary = joints.get(name)
        current = normalized.get(name)
        if current is None or not isinstance(summary, Mapping):
            continue
        mean = summary["mean"]
        spread = max(float(summary["spread"]), MIN_SPREAD_M)  # type: ignore[arg-type]
        distance = math.dist(current, (float(mean[0]), float(mean[1]), float(mean[2])))  # type: ignore[index]
        sigmas[name] = min(distance / spread, MAX_SIGMA)
    return sigmas


def similarity_score(sigmas: Mapping[str, float]) -> float:
    """0~1 실수. σ가 모두 0이면 1.0이고 평균 제곱 σ가 커질수록 지수적으로 떨어진다."""
    if not sigmas:
        return 0.0
    mean_square = sum(value * value for value in sigmas.values()) / len(sigmas)
    return math.exp(-0.5 * mean_square)


def _segment_angles(positions: Mapping[str, Vec3], joint: str) -> tuple[float, float] | None:
    parent = SEGMENT_PARENT.get(joint)
    if parent is None:
        return None
    head = positions.get(joint)
    tail = positions.get(parent)
    if head is None or tail is None:
        return None
    dx, dy, dz = head[0] - tail[0], head[1] - tail[1], head[2] - tail[2]
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length == 0.0:
        return None
    return math.degrees(math.asin(dz / length)), math.degrees(math.atan2(dy, dx))


def _object_particle(word: str) -> str:
    last = word[-1]
    if "가" <= last <= "힣":
        return "을" if (ord(last) - 0xAC00) % 28 else "를"
    return "을"


def _correction_text(label: str, angles: Mapping[str, float]) -> str:
    particle = _object_particle(label)
    pitch = angles.get("pitch")
    if pitch is not None:
        verb = "올리세요" if pitch < 0.0 else "내리세요"
        return f"{label}{particle} {abs(pitch):.0f}° {verb}"
    return f"{label}{particle} 기준 자세에 맞추세요"


def _reference_positions(bucket: Mapping[str, object]) -> dict[str, Vec3]:
    joints = bucket.get("joints")
    if not isinstance(joints, Mapping):
        return {}
    positions: dict[str, Vec3] = {}
    for name, summary in joints.items():
        if isinstance(summary, Mapping):
            mean = summary["mean"]
            positions[name] = (float(mean[0]), float(mean[1]), float(mean[2]))  # type: ignore[index]
    return positions


def build_corrections(
    normalized: Mapping[str, Vec3],
    bucket: Mapping[str, object],
    limit: int = MAX_CORRECTIONS,
) -> list[Correction]:
    """반사실 유사도 향상폭 내림차순으로 교정 관절을 최대 limit개 고른다."""
    sigmas = joint_sigmas(normalized, bucket)
    if not sigmas:
        return []
    reference = _reference_positions(bucket)
    baseline = similarity_score(sigmas)

    corrections: list[Correction] = []
    for joint in CORRECTION_JOINTS:
        current = normalized.get(joint)
        target = reference.get(joint)
        if current is None or target is None or sigmas.get(joint, 0.0) <= 0.0:
            continue

        counterfactual = dict(sigmas)
        counterfactual[joint] = 0.0
        gain = similarity_score(counterfactual) - baseline
        if gain <= 0.0:
            continue

        angles: dict[str, float] = {}
        now = _segment_angles(normalized, joint)
        then = _segment_angles(reference, joint)
        if now is not None and then is not None:
            for axis, difference in (("pitch", now[0] - then[0]), ("yaw", now[1] - then[1])):
                if abs(difference) >= MIN_ANGLE_DEG:
                    angles[axis] = round(difference, 1)

        label = JOINT_LABELS.get(joint, joint)
        corrections.append(
            Correction(
                joint=joint,
                label=label,
                distance=math.dist(current, target),
                angle_corrections=angles,
                correction_text=_correction_text(label, angles),
                expected_gain=gain,
            )
        )

    corrections.sort(key=lambda item: item.expected_gain, reverse=True)
    return corrections[:limit]
