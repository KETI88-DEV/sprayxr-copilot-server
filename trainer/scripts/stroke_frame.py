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

import world_axes
from measured_pose import MeasuredFrame

Vec3 = tuple[float, float, float]

DEFAULT_SAMPLES: Final = 20

# 월드 수직 축과, 접선이 수직축과 평행할 때 위 축을 다시 세우는 법은 world_axes가
# 소유한다(캡처 CSV는 유니티 좌표 — Y가 위). 여기서 다시 정의하지 않는다.
# `stroke_basis`는 `world_axes.frame_around`를 호출 시점에 부른다 — 모듈 상수로
# 결과를 한 번 복사해 두면 이후 규약이 바뀌어도 그 값은 굳어 반응하지 않는다.
#
# 접선 자체가 정의되지 않을 때(제자리 프레임, 착탄점이 안 움직임) 쓰는 대체축이다.
# 수평이어야 한다(유니티 Y-up이므로 X·Z가 수평이다) — 수직 벡터를 대체 접선으로 쓰면
# 퇴화 처리 자체가 다시 퇴화한다. `world_axes.frame_around`가 다루는 퇴화(접선이
# 월드 수직축과 평행해 위 축 직교화가 무너지는 경우)와는 다른 퇴화다: 이 상수는
# "접선이 아예 없다"를 고치고, 그쪽은 "접선은 있는데 수직축과 겹친다"를 고친다.
FALLBACK_TANGENT: Final[Vec3] = (1.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class Basis:
    """원점이 착탄점인 정규직교 기저.

    `normal`은 벽 법선이고 작업자 쪽이 양수다. 이 축은 항상 수평이다 — `up`이 월드
    수직축(`world_axes.UP`)과 접선이 이루는 평면 안에 있으므로 그 둘의 외적인
    `normal`은 월드 수직축에 직교한다. `tangent`는 스트로크 진행 방향이다.

    **`up`은 수직이 아니다.** 월드 수직축을 접선에 직교화한 축이라, 접선이 수평일
    때만 정확히 수직이고 스트로크가 기울수록 수직에서 벗어난다. 순수 수직 스트로크
    에서는 접선이 곧 월드 수직축이라 직교화가 영벡터가 되고, `world_axes.frame_around`
    안의 수평 대체축이 들어와 `up`이 **수평**이 된다. 이름은 수평 스트로크에서의 뜻을
    딴 것이고 그것이 전부다.

    그래서 로컬 성분을 축 이름으로 읽어 말로 옮기면 안 된다. 수직 스트로크에서 순수한
    옆 이탈이 `local[2]`에 실리는데 그것을 "15cm 올리세요"로 적으면 작업자가 옆으로
    틀어진 손을 위로 올리게 된다. 성분의 세상 뜻은 세 축 벡터로 되살려야 하고,
    `measured_score.corrections_text`가 이 기저를 받아 그 일을 한다.

    세 축의 뜻을 다 지키면 손 방향(오른손 기준 ±)에 따라 삼중항의 손대칭성이 갈리는데,
    여기서는 손대칭성보다 축의 의미를 지킨다. 어떤 소비자도 이 기저로 외적을 하지 않는다.
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

    왼손 마스킹은 프레임 단위다(설계 §5.4 "왼손 지표만 제외, 다른 지표는 유지").
    한 프레임이라도 왼손이 없으면 패스 전체의 왼손을 버리는 규칙은 실데이터에서
    상시 발동한다 — 실측 |머리-왼손| 최소가 0.000 m다. 그래서 왼손이 있는 프레임만
    모아 그 위에서 보간하고, 왼손 데이터가 격자점을 감싸지 못하는 구간만 None으로
    남긴다. 외삽하지 않는다.
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
    yaw_values = [value for value, frame in zip(values, usable) if frame.yaw is not None]
    yaw_track = [frame.yaw for frame in usable if frame.yaw is not None]
    left_values = [value for value, frame in zip(values, usable) if frame.left is not None]
    left_track = [frame.left for frame in usable if frame.left is not None]

    output: list[MeasuredFrame] = []
    for order, target in enumerate(grid):
        output.append(
            replace(
                usable[0],
                index=order,
                time_sec=_scalar_at(values, times, target),
                head=_point_at(values, heads, target),
                right=_point_at(values, rights, target),
                left=_left_at(left_values, left_track, target),
                hit=_point_at(values, hits, target),
                uv=usable[0].uv,
                yaw=_yaw_at(yaw_values, yaw_track, target),
            )
        )
    return output


def _yaw_at(
    values: Sequence[float], angles: Sequence[float], target: float
) -> float | None:
    """요 보간. 각이므로 스칼라로 섞지 않고 단위벡터로 섞는다.

    179°와 −179°를 산술 보간하면 0°가 나온다 — 뒤를 보고 있다고 적는 것이다.
    `dataset_spec.md` §4.1이 산술 통계를 금한 것과 같은 이유이며, 여기서는 두 표본
    사이의 보간이라 사인·코사인을 선형 보간한 뒤 `atan2`로 되읽는다.
    """
    if not angles:
        return None
    if len(angles) == 1:
        return angles[0]
    index, weight = _weight_at(values, target)
    left, right = angles[index], angles[index + 1]
    cos = math.cos(left) + (math.cos(right) - math.cos(left)) * weight
    sin = math.sin(left) + (math.sin(right) - math.sin(left)) * weight
    if math.hypot(cos, sin) < 1e-12:
        return left
    return math.atan2(sin, cos)


def _left_at(
    values: Sequence[float], points: Sequence[Vec3], target: float
) -> Vec3 | None:
    """왼손 보간. 관측 구간 밖이면 지어내지 않고 None이다."""
    if len(points) < 2 or target < values[0] or target > values[-1]:
        return None
    return _point_at(values, points, target)


def _normalize(vector: Vec3) -> Vec3:
    length = math.sqrt(sum(component * component for component in vector))
    if length == 0.0:
        return (0.0, 0.0, 0.0)
    return (vector[0] / length, vector[1] / length, vector[2] / length)


def _dot(left: Vec3, right: Vec3) -> float:
    return sum(left[axis] * right[axis] for axis in range(3))


def stroke_basis(hit: Vec3, tangent: Vec3, gun: Vec3) -> Basis:
    """착탄점을 원점으로 하는 기저. 측정 대상은 축을 정의하지 않는다.

    법선을 `건 - 착탄`으로 잡으면 그 프레임에서 건의 좌표가 정의상 `(거리, 0, 0)`이
    된다. 진행 방향 성분과 수직 성분이 구조적으로 0이라 두 동작이 일치해서 0인 것과
    구별되지 않는다. 오른손은 가중 1.0이고 이 축이 존재하는 이유인 점이므로 채점이
    1차원으로 붕괴한다. 설계 §6.1의 "오른손을 벽에서 5cm 더 띄우고 15cm 올리세요"도
    나올 수 없다.

    그래서 축은 측정 대상이 아닌 기하에서만 만든다(설계 §5.2의 "착탄점 근방 궤적의
    주평면 법선"). 정규직교 셋을 만드는 일 자체는 `world_axes.frame_around`가 한다
    — 접선을 정규화해 진행 방향으로 쓰고, 월드 수직축을 그 진행 방향에 직교화해
    수직축으로 쓰고, 두 축의 외적을 셋째 축(부호 미정)으로 낸다. 건은 그 셋째 축을
    법선으로 삼을 때 **부호를 고르는 데만** 쓴다(`world_axes.sign_toward` — 유니티는
    왼손 좌표계라 외적의 방향 감각이 오른손계와 반대다).

    퇴화 처리는 둘이고, 소유자가 다르다. 접선이 영벡터면(제자리 프레임) 수평
    대체축(`FALLBACK_TANGENT`)을 접선으로 쓴다 — 이건 여기서(스트로크 도메인에서)
    해소한다. 접선이 월드 수직축과 평행하면(순수 수직 스트로크) 수직축 직교화가
    영벡터가 되는데, 그 퇴화는 `world_axes.frame_around`가 자기 수평 대체축으로
    다시 세운다. 이때 `up`은 수평이다 — 이름이 아니라 벡터가 그 축의 뜻이다.
    """
    forward = _normalize(tangent)
    if forward == (0.0, 0.0, 0.0):
        forward = FALLBACK_TANGENT

    frame = world_axes.frame_around(forward)

    to_worker = (gun[0] - hit[0], gun[1] - hit[1], gun[2] - hit[2])
    # 왼손 좌표계(유니티)에서는 외적의 방향 감각이 오른손계와 반대다. `sign_toward`가
    # 그 재보정을 한 곳에서 한다 — 여기서 부호 로직을 다시 적지 않는다.
    normal = world_axes.sign_toward(frame.third, to_worker)

    return Basis(origin=hit, normal=normal, tangent=frame.forward, up=frame.up)


def to_local(basis: Basis, point: Vec3) -> Vec3:
    """스트로크 프레임 좌표. x는 벽에서 띄운 거리, y는 진행 방향, z는 수직이다."""
    delta = (
        point[0] - basis.origin[0],
        point[1] - basis.origin[1],
        point[2] - basis.origin[2],
    )
    return (_dot(delta, basis.normal), _dot(delta, basis.tangent), _dot(delta, basis.up))
