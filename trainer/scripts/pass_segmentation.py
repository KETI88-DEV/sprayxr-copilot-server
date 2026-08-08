#!/usr/bin/env python3
"""분사 버스트를 개별 패스로 나누고 패스별 궤적 지표를 낸다.

시뮬레이터가 보내는 단위(Trigger ON->OFF 버스트)는 스트로크가 아니다. 전문가 6인
605세션 실측에서 버스트 하나가 평균 4개 패스를 담았고 단일 패스인 버스트는 5.3%뿐이었다.
버스트 전체에 직선 하나를 맞추면 왕복이 전부 이탈로 잡혀 이탈 중앙값이 12.8mm 대신
63.8mm로 5배 부풀려진다. 그래서 지표는 반드시 패스 단위로 낸다.

좌표는 착탄점(surface hit)의 월드 좌표이며 Z가 수직이다(실측 범위 -0.15~1.46m).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal, Sequence

__all__ = [
    "PassMetrics",
    "body_bucket_key",
    "bucket_key",
    "principal_axis",
    "segment_burst",
    "split_pass_spans",
]

Vec3 = tuple[float, float, float]

MIN_BURST_FRAMES: Final = 10
MIN_PASS_FRAMES: Final = 10
MIN_PASS_LENGTH_MM: Final = 50.0
SMOOTH_WINDOW: Final = 5

HORIZONTAL_MAX_DEG: Final = 20.0
VERTICAL_MIN_DEG: Final = 70.0
HEIGHT_LOW_MAX_M: Final = 0.6
HEIGHT_MID_MAX_M: Final = 1.0

ElevationBand = Literal["horizontal", "diagonal", "vertical"]
HeightBand = Literal["low", "mid", "high"]


@dataclass(frozen=True, slots=True)
class PassMetrics:
    """한 패스의 궤적 지표. 인덱스는 입력 시퀀스 기준이며 끝은 포함이다."""

    frame_start: int
    frame_end: int
    length_mm: float
    straightness_rms_mm: float
    elevation_deg: float
    mid_height_m: float
    heading_axis: Literal["x", "y"]
    heading_sign: Literal["+", "-"]
    duration_sec: float
    speed_mm_per_s: float
    bucket: str


def principal_axis(points: Sequence[Vec3]) -> tuple[Vec3, Vec3]:
    """(중심, 단위 주축). 공분산 최대 고유벡터를 거듭제곱법으로 구한다."""
    count = len(points)
    center = (
        sum(p[0] for p in points) / count,
        sum(p[1] for p in points) / count,
        sum(p[2] for p in points) / count,
    )
    cov = [[0.0] * 3 for _ in range(3)]
    for point in points:
        delta = (point[0] - center[0], point[1] - center[1], point[2] - center[2])
        for i in range(3):
            for j in range(3):
                cov[i][j] += delta[i] * delta[j]

    vector = [1.0, 1.0, 1.0]
    for _ in range(50):
        rotated = [sum(cov[i][j] * vector[j] for j in range(3)) for i in range(3)]
        norm = math.sqrt(sum(v * v for v in rotated))
        if norm == 0.0:
            break
        vector = [v / norm for v in rotated]
    return center, (vector[0], vector[1], vector[2])


def _smoothed_projection(points: Sequence[Vec3]) -> list[float]:
    center, axis = principal_axis(points)
    projection = [
        (p[0] - center[0]) * axis[0] + (p[1] - center[1]) * axis[1] + (p[2] - center[2]) * axis[2]
        for p in points
    ]
    if len(projection) < SMOOTH_WINDOW:
        return projection
    return [
        sum(projection[i : i + SMOOTH_WINDOW]) / SMOOTH_WINDOW
        for i in range(len(projection) - SMOOTH_WINDOW + 1)
    ]


def _turning_indices(smoothed: Sequence[float]) -> list[int]:
    """진행 방향이 뒤집히는 지점. 마지막 비영 차분과 비교해 평탄부를 건너뛴다.

    스트로크 끝에서 건이 잠깐 멈추면 차분이 0인 구간이 생긴다. 인접 차분의 곱만
    보면 0 * (-x) = 0이라 반환점을 놓치므로, 부호가 확정된 마지막 방향과 비교한다.
    """
    turns: list[int] = []
    last_sign = 0
    for i in range(1, len(smoothed)):
        delta = smoothed[i] - smoothed[i - 1]
        if delta == 0.0:
            continue
        sign = 1 if delta > 0.0 else -1
        if last_sign != 0 and sign != last_sign:
            turns.append(i - 1)
        last_sign = sign
    return turns


def split_pass_spans(points: Sequence[Vec3]) -> list[tuple[int, int]]:
    """주축 진행의 방향 전환에서 잘라 [start, end] 인덱스 구간을 낸다(끝 포함).

    MIN_PASS_FRAMES 미만인 반환은 잡음으로 보고 흡수한다. 프레임을 버리지 않으므로
    구간들은 항상 버스트 전체를 덮는다.
    """
    if len(points) < MIN_BURST_FRAMES:
        return []
    last = len(points) - 1
    smoothed = _smoothed_projection(points)
    if len(smoothed) < 3:
        return [(0, last)]

    offset = SMOOTH_WINDOW // 2 if len(points) > len(smoothed) else 0
    boundaries = [0]
    for turn in _turning_indices(smoothed):
        index = min(turn + offset, last)
        if index - boundaries[-1] >= MIN_PASS_FRAMES and last - index >= MIN_PASS_FRAMES:
            boundaries.append(index)
    boundaries.append(last)
    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


def _straightness_rms_mm(points: Sequence[Vec3]) -> float:
    center, axis = principal_axis(points)
    total = 0.0
    for point in points:
        delta = (point[0] - center[0], point[1] - center[1], point[2] - center[2])
        along = delta[0] * axis[0] + delta[1] * axis[1] + delta[2] * axis[2]
        total += sum((delta[i] - along * axis[i]) ** 2 for i in range(3))
    return math.sqrt(total / len(points)) * 1000.0


def _path_length_mm(points: Sequence[Vec3]) -> float:
    return sum(math.dist(points[i], points[i + 1]) for i in range(len(points) - 1)) * 1000.0


def elevation_band(elevation_deg: float) -> ElevationBand:
    if elevation_deg <= HORIZONTAL_MAX_DEG:
        return "horizontal"
    if elevation_deg >= VERTICAL_MIN_DEG:
        return "vertical"
    return "diagonal"


def height_band(mid_height_m: float) -> HeightBand:
    if mid_height_m < HEIGHT_LOW_MAX_M:
        return "low"
    if mid_height_m < HEIGHT_MID_MAX_M:
        return "mid"
    return "high"


def bucket_key(elevation_deg: float, mid_height_m: float, heading_axis: str, heading_sign: str) -> str:
    """레퍼런스 분포의 버킷 키. 자세 레퍼런스도 같은 키를 쓴다."""
    return f"{elevation_band(elevation_deg)}-{height_band(mid_height_m)}-{heading_sign}{heading_axis}"


def body_bucket_key(elevation_deg: float, mid_height_m: float, heading: str) -> str:
    """진행 방향을 월드 축이 아니라 작업자 기준으로 적는 버킷 키.

    월드 요가 다른 캡처끼리도 같은 스트로크가 같은 버킷에 들어간다. 방향 성분이
    부호+축이 아니라 단어라 월드 기준 키와 문자열이 겹치지 않는다. 서로 다른 기준의
    레퍼런스와 질의가 잘못 매칭되는 것을 키 공간 분리로 막는다.
    """
    return f"{elevation_band(elevation_deg)}-{height_band(mid_height_m)}-{heading}"


def _metrics_for_span(
    points: Sequence[Vec3], times: Sequence[float], start: int, end: int
) -> PassMetrics | None:
    segment = list(points[start : end + 1])
    displacement_mm = math.dist(segment[0], segment[-1]) * 1000.0
    if displacement_mm < MIN_PASS_LENGTH_MM:
        return None

    _, axis = principal_axis(segment)
    elevation = math.degrees(math.asin(min(1.0, abs(axis[2]))))
    heading = (segment[-1][0] - segment[0][0], segment[-1][1] - segment[0][1])
    axis_name: Literal["x", "y"] = "x" if abs(heading[0]) >= abs(heading[1]) else "y"
    signed = heading[0] if axis_name == "x" else heading[1]
    sign: Literal["+", "-"] = "+" if signed >= 0.0 else "-"
    mid_height = sum(p[2] for p in segment) / len(segment)
    duration = times[end] - times[start]
    path_mm = _path_length_mm(segment)

    return PassMetrics(
        frame_start=start,
        frame_end=end,
        length_mm=displacement_mm,
        straightness_rms_mm=_straightness_rms_mm(segment),
        elevation_deg=elevation,
        mid_height_m=mid_height,
        heading_axis=axis_name,
        heading_sign=sign,
        duration_sec=duration,
        speed_mm_per_s=path_mm / duration if duration > 0.0 else 0.0,
        bucket=bucket_key(elevation, mid_height, axis_name, sign),
    )


def segment_burst(points: Sequence[Vec3], times: Sequence[float]) -> list[PassMetrics]:
    """버스트 하나를 패스로 나눠 지표를 낸다. 길이 미달 패스는 버린다."""
    if len(points) != len(times):
        raise ValueError("points and times must have the same length")
    metrics: list[PassMetrics] = []
    for start, end in split_pass_spans(points):
        item = _metrics_for_span(points, times, start, end)
        if item is not None:
            metrics.append(item)
    return metrics
