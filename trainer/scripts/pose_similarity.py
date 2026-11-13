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
from typing import Final, Literal, Mapping, Sequence

# MCP는 `trainer.scripts.pose_similarity`로, pose_lab·스크립트·테스트는 scripts를
# sys.path에 얹고 `pose_similarity`로 부른다. 자기가 불린 이름 공간에서 형제를
# 잡아야 새 프로세스에서도 풀리고 simulator_csv가 두 벌로 갈라지지 않는다.
if __package__:
    from . import simulator_csv
else:
    import simulator_csv

__all__ = [
    "CORRECTION_JOINTS",
    "Correction",
    "JointDelta",
    "PoseComparison",
    "FACING_JOINTS",
    "JOINT_LABELS",
    "ROOT_JOINT",
    "SCORING_JOINTS",
    "body_heading",
    "apply_corrections",
    "build_corrections",
    "compare_poses",
    "gap_closure",
    "facing_yaw",
    "joint_sigmas",
    "normalize_pose",
    "similarity_score",
]

Vec3 = tuple[float, float, float]
BodyHeading = Literal["right", "left", "forward", "back"]

ROOT_JOINT: Final = "Pelvis"
FACING_JOINTS: Final = ("LeftShoulder", "RightShoulder")
MIN_SPREAD_M: Final = 0.005
MAX_SIGMA: Final = 6.0
MIN_ANGLE_DEG: Final = 3.0
MAX_CORRECTIONS: Final = 6

# 숙련자 held-out σ 실측(n=24,660)에서 역산: σ>2.0은 숙련자 관절-패스의 3.7%만 통과하며
# 궤적 축의 교정 필요 기준(1.3~5.1%)과 같은 대역이다. 게이트가 없으면 숙련자 본인도 100% 상한까지 받는다.
CORRECTION_SIGMA_THRESHOLD: Final = 2.0

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
class JointDelta:
    """두 자세 사이의 관절 하나. body_delta는 (오른쪽, 앞, 위) 순의 신체 기준 차이다."""

    joint: str
    label: str
    distance: float
    sigma: float
    body_delta: Vec3


@dataclass(frozen=True, slots=True)
class PoseComparison:
    similarity: float | None
    deltas: list[JointDelta]


@dataclass(frozen=True, slots=True)
class Correction:
    joint: str
    label: str
    distance: float
    angle_corrections: dict[str, float]
    correction_text: str
    expected_gain: float


def facing_yaw(pose: Mapping[str, Vec3]) -> float:
    """어깨선의 방위각. 이 각만큼 되돌려 몸이 항상 같은 쪽을 보게 만든다."""
    left = pose.get(FACING_JOINTS[0])
    right = pose.get(FACING_JOINTS[1])
    if left is None or right is None:
        return 0.0
    return math.atan2(right[1] - left[1], right[0] - left[0])


def body_heading(yaw: float, heading: tuple[float, float]) -> BodyHeading:
    """월드 수평 방향을 작업자 기준 방향으로 옮긴다. 월드 요가 달라도 라벨이 같다.

    정규화 공간의 축 의미는 실측으로 확정했다. 어깨 X 부호로 +X가 오른쪽이고,
    건을 쥔 오른손의 Y가 3세션 전부 100% 양수(중앙 +0.50~+0.60)라 +Y가 앞이다.
    """
    cos = math.cos(-yaw)
    sin = math.sin(-yaw)
    x = heading[0] * cos - heading[1] * sin
    y = heading[0] * sin + heading[1] * cos
    if abs(x) >= abs(y):
        return "right" if x >= 0.0 else "left"
    return "forward" if y >= 0.0 else "back"


def normalize_pose(pose: Mapping[str, Vec3]) -> dict[str, Vec3]:
    """루트를 원점으로 옮기고 수직축 둘레로만 회전시켜 정면을 맞춘다."""
    origin = pose.get(ROOT_JOINT, (0.0, 0.0, 0.0))
    angle = -facing_yaw(pose)
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


def compare_poses(
    left: Mapping[str, Vec3], right: Mapping[str, Vec3], scales: Mapping[str, float]
) -> PoseComparison:
    """두 패스의 자세를 관절별로 견준다.

    척도는 레퍼런스 경로와 같은 숙련자 산포다. 그래야 "이만큼 다르다"가 두 경로에서
    같은 뜻이 된다. 견줄 관절이 하나도 없으면 점수를 만들지 않는다. 1.0도 0.0도 거짓이다.
    """
    first_pose = normalize_pose(left)
    second_pose = normalize_pose(right)

    sigmas: dict[str, float] = {}
    deltas: list[JointDelta] = []
    for joint in SCORING_JOINTS:
        scale = scales.get(joint)
        first = first_pose.get(joint)
        second = second_pose.get(joint)
        if scale is None or first is None or second is None:
            continue
        distance = math.dist(first, second)
        sigmas[joint] = min(distance / max(scale, MIN_SPREAD_M), MAX_SIGMA)
        if sigmas[joint] <= 0.0:
            continue
        deltas.append(
            JointDelta(
                joint=joint,
                label=JOINT_LABELS.get(joint, joint),
                distance=distance,
                sigma=sigmas[joint],
                body_delta=(
                    second[0] - first[0],
                    second[1] - first[1],
                    second[2] - first[2],
                ),
            )
        )

    deltas.sort(key=lambda item: item.sigma, reverse=True)
    return PoseComparison(similarity_score(sigmas) if sigmas else None, deltas)


def apply_corrections(
    normalized: Mapping[str, Vec3],
    bucket: Mapping[str, object],
    corrections: Sequence[Correction],
    fraction: float = 1.0,
) -> dict[str, Vec3]:
    """교정을 따랐을 때의 자세. 지목된 관절만 기준 위치 쪽으로 fraction만큼 옮긴다.

    fraction=1.0은 지시를 완벽히 따른 상한이다. 나머지 관절은 건드리지 않는다. 채점
    관절이 교정 관절보다 많으므로 완벽히 따라도 유사도가 1.0이 되지는 않는다.
    """
    reference = _reference_positions(bucket)
    moved = dict(normalized)
    for correction in corrections:
        current = moved.get(correction.joint)
        target = reference.get(correction.joint)
        if current is None or target is None:
            continue
        moved[correction.joint] = (
            current[0] + (target[0] - current[0]) * fraction,
            current[1] + (target[1] - current[1]) * fraction,
            current[2] + (target[2] - current[2]) * fraction,
        )
    return moved


def gap_closure(before: float, after: float) -> float | None:
    """남아 있던 격차 중 얼마를 메웠는가. (S1-S0)/(1-S0).

    이미 기준과 같으면(S0=1) 메울 격차가 없어 정의되지 않는다. 0으로 적으면 실패처럼
    읽히므로 값을 만들지 않는다.
    """
    room = 1.0 - before
    if room <= 0.0:
        return None
    return (after - before) / room


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
    """숙련자 범위를 벗어난 관절만 반사실 향상폭 내림차순으로 최대 limit개 낸다.

    임계를 넘는 관절이 없으면 빈 목록이다. 상한을 채우려 하지 않는다.
    """
    sigmas = joint_sigmas(normalized, bucket)
    if not sigmas:
        return []
    reference = _reference_positions(bucket)
    baseline = similarity_score(sigmas)

    corrections: list[Correction] = []
    for joint in CORRECTION_JOINTS:
        current = normalized.get(joint)
        target = reference.get(joint)
        if (
            current is None
            or target is None
            or sigmas.get(joint, 0.0) <= CORRECTION_SIGMA_THRESHOLD
        ):
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


def bone_lengths(pose: Mapping[str, Vec3]) -> dict[str, float]:
    """자식 관절별 뼈 길이. 교정 전후로 변하면 안 되는 불변량이다."""
    lengths: dict[str, float] = {}
    for parent, child in simulator_csv.BONES:
        head = pose.get(parent)
        tail = pose.get(child)
        if head is None or tail is None:
            continue
        lengths[child] = math.dist(head, tail)
    return lengths


def _rotate_about(pivot: Vec3, point: Vec3, source: Vec3, target: Vec3) -> Vec3:
    """pivot을 축점으로 source 방향을 target 방향으로 돌렸을 때 point가 가는 곳.

    로드리게스 회전이라 pivot에서의 거리가 보존된다. 그래서 뼈 길이가 안 변한다.
    """

    def _sub(a: Vec3, b: Vec3) -> Vec3:
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    def _norm(v: Vec3) -> Vec3:
        length = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
        return v if length == 0.0 else (v[0] / length, v[1] / length, v[2] / length)

    def _cross(a: Vec3, b: Vec3) -> Vec3:
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )

    def _dot(a: Vec3, b: Vec3) -> float:
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    start = _norm(_sub(source, pivot))
    end = _norm(_sub(target, pivot))
    axis_raw = _cross(start, end)
    sine = math.sqrt(sum(component * component for component in axis_raw))
    cosine = _dot(start, end)

    # Two cases when sine ≈ 0: parallel (cosine ≈ +1, nothing to rotate) or anti-parallel (cosine ≈ -1, need 180° rotation)
    if sine < 1e-12:
        if cosine > 0.0:
            return point
        # Anti-parallel: build perpendicular axis from world axes
        world_axes = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        best_axis = None
        best_dot = 2.0
        for world_axis in world_axes:
            alignment = abs(_dot(start, world_axis))
            if alignment < best_dot:
                best_dot = alignment
                best_axis = world_axis
        axis_raw = _cross(best_axis, start) if best_axis else (0.0, 0.0, 1.0)
        sine = 0.0
        cosine = -1.0

    axis = _norm(axis_raw)
    local = _sub(point, pivot)
    term_cross = _cross(axis, local)
    term_dot = _dot(axis, local)
    rotated = tuple(
        local[i] * cosine + term_cross[i] * sine + axis[i] * term_dot * (1.0 - cosine)
        for i in range(3)
    )
    return (pivot[0] + rotated[0], pivot[1] + rotated[1], pivot[2] + rotated[2])


def apply_corrections_rigid(
    normalized: Mapping[str, Vec3],
    bucket: Mapping[str, object],
    corrections: Sequence[Correction],
    fraction: float = 1.0,
) -> dict[str, Vec3]:
    """교정을 따랐을 때의 자세. 방향만 돌리고 뼈 길이는 보존한다.

    관절을 목표 위치로 그냥 옮기면 골격이 늘어나 사람이 취할 수 없는 자세가 나온다.
    부모를 축점으로 삼아 마디 방향만 목표 쪽으로 돌리고, 아래 매달린 관절을 함께 옮긴다.
    """
    pose = dict(normalized)
    targets = _reference_positions(bucket)

    for correction in corrections:
        joint = correction.joint
        parent = simulator_csv.PARENT_OF.get(joint)
        current = pose.get(joint)
        goal = targets.get(joint)
        if parent is None or current is None or goal is None:
            continue
        pivot = pose.get(parent)
        if pivot is None:
            continue

        blended = (
            current[0] + (goal[0] - current[0]) * fraction,
            current[1] + (goal[1] - current[1]) * fraction,
            current[2] + (goal[2] - current[2]) * fraction,
        )
        for name in {joint} | simulator_csv.descendants(joint):
            point = pose.get(name)
            if point is None:
                continue
            pose[name] = _rotate_about(pivot, point, current, blended)

    return pose
