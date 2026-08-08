#!/usr/bin/env python3
"""BVH 프레임과 시뮬레이터 CSV 행의 시간 정합.

BVH 헤더의 Frame Time은 공칭값이라 실제 타임베이스가 아니다. 79세션 실측에서
csvRows/bvhFrames가 1.46~10.30으로 흩어져 인덱스·선언시간 어느 쪽으로도 직접
대응하지 않는다.

정렬 잔차는 정합이 맞다는 증거가 못 된다. 거의 상수인 신호는 어떤 단조 경로에서도
잔차가 작게 나오기 때문이다. 손-머리 거리(BVH std가 평균의 11.9%)로 정렬했을 때
잔차는 14.4mm였지만 정합에 쓰지 않은 머리 높이의 상관은 -0.04로 무상관이었다.
그래서 정합을 쓰려는 쪽은 반드시 cross_validate로 독립 신호를 확인해야 한다.

정렬 신호는 자세에 외생적이어야 한다. 자세 특징 위에서 정렬하면 DTW가 보고할
거리를 스스로 최소화해 유사도가 낙관적으로 편향된다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean, median
from typing import Sequence

__all__ = [
    "Alignment",
    "align_session",
    "co_indexed_alignment",
    "cross_validate",
    "dtw_path",
]


@dataclass(frozen=True, slots=True)
class Alignment:
    """frame_for_row[csv_row] = 대응 BVH 프레임. residual은 정렬 후 신호 불일치 중앙값."""

    frame_for_row: tuple[int, ...]
    residual: float


def _downsample(values: Sequence[float], stride: int) -> list[float]:
    return list(values[::stride])


def dtw_path(
    source: Sequence[float], target: Sequence[float], max_points: int = 1000
) -> list[tuple[int, int]]:
    """두 신호의 DTW 대응쌍을 원본 인덱스 공간으로 낸다.

    긴 신호는 max_points까지 균등 솎아 DP 비용을 억제하고 인덱스를 되돌린다.
    """
    if not source or not target:
        return []

    source_stride = max(1, -(-len(source) // max_points))
    target_stride = max(1, -(-len(target) // max_points))
    left = _downsample(source, source_stride)
    right = _downsample(target, target_stride)

    rows = len(left)
    columns = len(right)
    accumulated = [[0.0] * columns for _ in range(rows)]
    accumulated[0][0] = abs(left[0] - right[0])
    for row in range(1, rows):
        accumulated[row][0] = accumulated[row - 1][0] + abs(left[row] - right[0])
    for column in range(1, columns):
        accumulated[0][column] = accumulated[0][column - 1] + abs(left[0] - right[column])
    for row in range(1, rows):
        previous = accumulated[row - 1]
        current = accumulated[row]
        value = left[row]
        for column in range(1, columns):
            current[column] = abs(value - right[column]) + min(
                previous[column], current[column - 1], previous[column - 1]
            )

    row, column = rows - 1, columns - 1
    path = [(row, column)]
    while row > 0 or column > 0:
        if row == 0:
            column -= 1
        elif column == 0:
            row -= 1
        else:
            diagonal = accumulated[row - 1][column - 1]
            up = accumulated[row - 1][column]
            side = accumulated[row][column - 1]
            best = min(diagonal, up, side)
            if best == diagonal:
                row, column = row - 1, column - 1
            elif best == up:
                row -= 1
            else:
                column -= 1
        path.append((row, column))

    path.reverse()
    last_source = len(source) - 1
    last_target = len(target) - 1
    return [
        (min(row * source_stride, last_source), min(column * target_stride, last_target))
        for row, column in path
    ]


def align_session(
    bvh_signal: Sequence[float], csv_signal: Sequence[float], max_points: int = 1000
) -> Alignment:
    """CSV 행마다 대응 BVH 프레임을 정한다. 솎아내기로 비는 행은 직전 값으로 채운다."""
    path = dtw_path(bvh_signal, csv_signal, max_points)
    if not path:
        return Alignment((), float("inf"))

    assigned: list[int | None] = [None] * len(csv_signal)
    for frame, row in path:
        if assigned[row] is None:
            assigned[row] = frame

    frames: list[int] = []
    carried = path[0][0]
    for value in assigned:
        if value is not None:
            carried = value
        frames.append(carried)

    residual = median(abs(bvh_signal[frame] - csv_signal[row]) for frame, row in path)
    return Alignment(tuple(frames), residual)


def co_indexed_alignment(row_count: int) -> Alignment:
    """자세와 CSV가 같은 프레임으로 함께 들어올 때의 대응. 추정이 아니라 구성상 항등이다.

    런타임 입력이 이 형태이므로 DTW도, 교차검증 게이트도 필요 없다. 항등 대응을
    cross_validate에 넣으면 실제로 r=1.0이 나오므로 게이트를 우회하는 것도 아니다.
    """
    return Alignment(tuple(range(row_count)), 0.0)


def cross_validate(
    frame_for_row: Sequence[int], bvh_signal: Sequence[float], csv_signal: Sequence[float]
) -> float:
    """정합에 쓰지 않은 신호쌍의 피어슨 상관. 상수 신호는 검증력이 없으므로 0을 낸다."""
    if not frame_for_row:
        return 0.0
    mapped = [bvh_signal[frame] for frame in frame_for_row]
    reference = list(csv_signal[: len(mapped)])
    if len(reference) < 2:
        return 0.0

    mapped_mean = fmean(mapped)
    reference_mean = fmean(reference)
    covariance = sum(
        (a - mapped_mean) * (b - reference_mean) for a, b in zip(mapped, reference)
    )
    mapped_spread = sum((a - mapped_mean) ** 2 for a in mapped)
    reference_spread = sum((b - reference_mean) ** 2 for b in reference)
    if mapped_spread == 0.0 or reference_spread == 0.0:
        return 0.0
    return covariance / math.sqrt(mapped_spread * reference_spread)
