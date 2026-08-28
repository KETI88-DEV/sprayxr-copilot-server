#!/usr/bin/env python3
"""두 동작을 대응시키고 스트로크 좌표계로 옮긴다.

정렬 신호는 자세에 외생적이어야 한다. 자세 특징 위에서 DTW를 돌리면 DTW가 보고할
거리를 스스로 최소화해 낙관 편향이 생기고, "이 순간 팔꿈치가 틀렸다"는 교정 의미론이
무너진다. 그래서 착탄점 궤적의 정규화 호길이로 정렬한다.

시간 위상을 쓰지 않는 이유는 둘이다. 두 사람의 스트로크 속도가 다르면 시간 위상이
대응을 왜곡하고, 속도 차이는 지워야 할 잡음이 아니라 보고해야 할 오차다. 호길이로
정렬하면 속도가 독립된 축으로 분리된다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Final, Sequence

from measured_pose import MeasuredFrame

Vec3 = tuple[float, float, float]

DEFAULT_SAMPLES: Final = 20

# 월드 Z가 수직이다(설계 §2.5, 실측 범위 -0.149~1.456 m).
WORLD_UP: Final[Vec3] = (0.0, 0.0, 1.0)
# 접선이 정의되지 않을 때 쓰는 축. 벽은 X 1.2 평면이고 착탄점은 Y를 따라 흐른다.
FALLBACK_TANGENT: Final[Vec3] = (0.0, 1.0, 0.0)


@dataclass(frozen=True, slots=True)
class Basis:
    """원점이 착탄점인 정규직교 기저.

    세 축의 뜻이 고정이다 — `normal`은 벽 법선이고 작업자 쪽이 양수, `tangent`는
    스트로크 진행 방향, `up`은 수직(월드 Z)이다. 세 뜻을 다 지키면 손 방향(오른손 기준
    ±)에 따라 삼중항의 손대칭성이 갈리는데, 여기서는 손대칭성보다 축의 의미를 지킨다.
    "5cm 더 띄우고 15cm 올리세요"가 참이 되려면 `normal`과 `up`의 부호가 각각 벽 바깥과
    위여야 하기 때문이다. 어떤 소비자도 이 기저로 외적을 하지 않는다.
    """

    origin: Vec3
    normal: Vec3
    tangent: Vec3
    up: Vec3


def arclengths(points: Sequence[Vec3]) -> list[float]:
    """0.0에서 1.0까지 정규화한 누적 호길이. 제자리 프레임은 늘리지 않는다."""
    if not points:
        return []
    if len(points) == 1:
        return [0.0]
    cumulative = [0.0]
    for index in range(1, len(points)):
        cumulative.append(cumulative[-1] + math.dist(points[index - 1], points[index]))
    total = cumulative[-1]
    if total <= 0.0:
        return [0.0] * len(points)
    return [value / total for value in cumulative]


def _interpolate(left: Vec3, right: Vec3, weight: float) -> Vec3:
    return (
        left[0] + (right[0] - left[0]) * weight,
        left[1] + (right[1] - left[1]) * weight,
        left[2] + (right[2] - left[2]) * weight,
    )


def _weight_at(values: Sequence[float], target: float) -> tuple[int, float]:
    """target이 놓인 구간의 시작 인덱스와 그 구간 안에서의 비율."""
    if target <= values[0]:
        return 0, 0.0
    if target >= values[-1]:
        return len(values) - 2, 1.0
    for index in range(1, len(values)):
        if values[index] >= target:
            span = values[index] - values[index - 1]
            weight = 0.0 if span <= 0.0 else (target - values[index - 1]) / span
            return index - 1, weight
    return len(values) - 2, 1.0


def _point_at(values: Sequence[float], points: Sequence[Vec3], target: float) -> Vec3:
    if len(points) == 1:
        return points[0]
    index, weight = _weight_at(values, target)
    return _interpolate(points[index], points[index + 1], weight)


def _scalar_at(values: Sequence[float], series: Sequence[float], target: float) -> float:
    if len(series) == 1:
        return series[0]
    index, weight = _weight_at(values, target)
    return series[index] + (series[index + 1] - series[index]) * weight


def resample(points: Sequence[Vec3], samples: int = DEFAULT_SAMPLES) -> list[Vec3]:
    """호길이 균등 격자로 다시 뽑는다. 속도가 달라도 같은 경로면 같은 결과다."""
    if samples < 2 or len(points) < 2:
        return list(points)
    values = arclengths(points)
    grid = [index / (samples - 1) for index in range(samples)]
    return [_point_at(values, points, target) for target in grid]


def resample_frames(
    frames: Sequence[MeasuredFrame], samples: int = DEFAULT_SAMPLES
) -> list[MeasuredFrame]:
    """실측 3점을 착탄점 호길이 격자에 맞춰 다시 뽑는다.

    격자 기준은 착탄점이다. 세 점이 같은 s에서 대응해야 "스트로크 30% 지점에서
    건이 벽에 붙는다"가 두 패스에서 같은 뜻이 된다.
    """
    usable = [frame for frame in frames if frame.hit is not None]
    if len(usable) < 2 or samples < 2:
        return list(usable)

    hits: list[Vec3] = []
    for frame in usable:
        assert frame.hit is not None
        hits.append(frame.hit)
    values = arclengths(hits)
    grid = [index / (samples - 1) for index in range(samples)]

    heads = [frame.head for frame in usable]
    rights = [frame.right for frame in usable]
    times = [frame.time_sec for frame in usable]
    lefts = [frame.left for frame in usable]
    left_track = [point for point in lefts if point is not None]
    left_usable = len(left_track) == len(usable)

    output: list[MeasuredFrame] = []
    for order, target in enumerate(grid):
        output.append(
            replace(
                usable[0],
                index=order,
                time_sec=_scalar_at(values, times, target),
                head=_point_at(values, heads, target),
                right=_point_at(values, rights, target),
                left=_point_at(values, left_track, target) if left_usable else None,
                hit=_point_at(values, hits, target),
                uv=usable[0].uv,
            )
        )
    return output


def _normalize(vector: Vec3) -> Vec3:
    length = math.sqrt(sum(component * component for component in vector))
    if length == 0.0:
        return (0.0, 0.0, 0.0)
    return (vector[0] / length, vector[1] / length, vector[2] / length)


def _cross(left: Vec3, right: Vec3) -> Vec3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _dot(left: Vec3, right: Vec3) -> float:
    return sum(left[axis] * right[axis] for axis in range(3))


def _orthogonalized(vector: Vec3, axis: Vec3) -> Vec3:
    """axis 성분을 뺀 뒤 정규화. axis에 평행하면 영벡터가 나온다."""
    projection = _dot(vector, axis)
    return _normalize(
        (
            vector[0] - projection * axis[0],
            vector[1] - projection * axis[1],
            vector[2] - projection * axis[2],
        )
    )


def stroke_basis(hit: Vec3, tangent: Vec3, gun: Vec3) -> Basis:
    """착탄점을 원점으로 하는 기저. 측정 대상은 축을 정의하지 않는다.

    법선을 `건 - 착탄`으로 잡으면 그 프레임에서 건의 좌표가 정의상 `(거리, 0, 0)`이
    된다. 진행 방향 성분과 수직 성분이 구조적으로 0이라 두 동작이 일치해서 0인 것과
    구별되지 않는다. 오른손은 가중 1.0이고 이 축이 존재하는 이유인 점이므로 채점이
    1차원으로 붕괴한다. 설계 §6.1의 "오른손을 벽에서 5cm 더 띄우고 15cm 올리세요"도
    나올 수 없다.

    그래서 축은 측정 대상이 아닌 기하에서만 만든다(설계 §5.2의 "착탄점 근방 궤적의
    주평면 법선").

    - `tangent`(착탄점 궤적의 접선)를 정규화해 진행 방향으로 쓴다.
    - 월드 Z를 접선에 직교화해 수직축으로 쓴다.
    - 법선은 두 축의 외적이다. 건은 이 법선의 **부호를 고르는 데만** 쓴다.

    퇴화 처리는 둘이다. 접선이 영벡터면(제자리 프레임) 벽을 따라가는 축인 월드 Y를
    접선으로 쓴다. 접선이 월드 Z와 평행하면(순수 수직 스트로크) 수직이 이미 접선이라
    직교화가 영벡터가 되므로, 세 번째 축은 벽을 따라가는 수평축(월드 Y 직교화)으로
    떨어뜨린다.
    """
    forward = _normalize(tangent)
    if forward == (0.0, 0.0, 0.0):
        forward = FALLBACK_TANGENT

    up = _orthogonalized(WORLD_UP, forward)
    if up == (0.0, 0.0, 0.0):
        up = _orthogonalized(FALLBACK_TANGENT, forward)

    normal = _normalize(_cross(forward, up))
    to_worker = (gun[0] - hit[0], gun[1] - hit[1], gun[2] - hit[2])
    if _dot(normal, to_worker) < 0.0:
        normal = (-normal[0], -normal[1], -normal[2])

    return Basis(origin=hit, normal=normal, tangent=forward, up=up)


def to_local(basis: Basis, point: Vec3) -> Vec3:
    """스트로크 프레임 좌표. x는 벽에서 띄운 거리, y는 진행 방향, z는 수직이다."""
    delta = (
        point[0] - basis.origin[0],
        point[1] - basis.origin[1],
        point[2] - basis.origin[2],
    )
    return (_dot(delta, basis.normal), _dot(delta, basis.tangent), _dot(delta, basis.up))
