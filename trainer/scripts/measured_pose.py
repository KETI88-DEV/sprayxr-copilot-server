#!/usr/bin/env python3
"""CSV 한 행에서 실측 3점을 읽는다. FK를 돌지 않는다.

시뮬레이터는 HMD + 양손 컨트롤러의 3점 추적이고 관절 20개는 그 3점에서 IK로 푼 값이다.
실측인 세 강체는 카메라(머리), 활성 건(오른손), 플래쉬(왼손)이며 셋 다 착탄점과 같은
월드 좌표계에 있다. 그래서 스켈레톤 리그와 월드의 좌표계 분리(REQ-14)도, 회전순서도,
BVH 정합도 이 경로에서는 무관하다.

미사용 건은 좌표로 판별하지 않는다. 주차 좌표 (-694.61, 701.29, 698.25)로 가는 건도
있지만 반도숏건처럼 정상으로 보이는 위치에 std=0으로 고정되는 건도 있다. 판별은 분산이다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Sequence

import simulator_csv

Vec3 = tuple[float, float, float]

GUN_GROUPS: Final[tuple[str, ...]] = ("한길숏건", "한길폴건", "반도숏건", "반도폴건")
HEAD_GROUP: Final = "카메라"
LEFT_GROUP: Final = "플래쉬"

# 세션 안에서 이만큼도 안 움직이면 들지 않은 건이다.
MIN_ACTIVE_SPREAD_M: Final = 0.05

# 팔 길이 상한. 실측 |머리-건| 평균 0.483m, 최대 4.443m(건을 내려놓은 프레임).
MAX_HEAD_HAND_M: Final = 1.0
MIN_HEAD_RIGHT_M: Final = 0.1
MIN_HEAD_LEFT_M: Final = 0.05

# 벤더 export는 " Ray Point Y" / " Ray Point Z"를 앞 공백째로 싣는다. 정식 리더인
# `simulator_csv.read_rows`는 헤더를 strip하므로 같은 컬럼이 두 철자로 들어온다.
# 한쪽만 보면 실데이터에서 착탄점이 통째로 None이 되고, 그러면 모든 패스가 키를
# 만들지 못해 "재지 못했다"로 떨어진다. 두 철자를 다 받는다.
RAY_POINT_SPELLINGS: Final[tuple[tuple[str, ...], ...]] = (
    ("Ray point X", " Ray point X"),
    ("Ray Point Y", " Ray Point Y"),
    ("Ray Point Z", " Ray Point Z"),
)


@dataclass(frozen=True, slots=True)
class MeasuredFrame:
    """한 프레임의 실측량. left/hit/uv는 못 믿으면 None이고 프레임 자체는 남는다."""

    index: int
    time_sec: float
    head: Vec3
    right: Vec3
    left: Vec3 | None
    hit: Vec3 | None
    uv: tuple[float, float] | None
    spraying: bool


def _position(row: dict[str, str], group: str) -> Vec3 | None:
    try:
        x, y, z = simulator_csv.vec3(row, group, "Position")
    except (KeyError, ValueError):
        return None
    return (x, y, z)


def _spread(values: Sequence[float]) -> float:
    return max(values) - min(values) if values else 0.0


def active_gun(rows: Sequence[dict[str, str]]) -> str | None:
    """세션 안에서 가장 많이 움직인 건. 움직임이 임계 미만이면 없다."""
    best: tuple[float, str] | None = None
    for group in GUN_GROUPS:
        samples = [_position(row, group) for row in rows]
        points = [point for point in samples if point is not None]
        if len(points) < 2:
            continue
        spread = max(_spread([point[axis] for point in points]) for axis in range(3))
        if spread < MIN_ACTIVE_SPREAD_M:
            continue
        if best is None or spread > best[0]:
            best = (spread, group)
    return None if best is None else best[1]


def _column(row: dict[str, str], spellings: Sequence[str]) -> str:
    for name in spellings:
        if name in row:
            return row[name]
    raise KeyError(spellings[0])


def _hit(row: dict[str, str]) -> tuple[Vec3, tuple[float, float]] | None:
    try:
        x, y, z = (float(_column(row, names)) for names in RAY_POINT_SPELLINGS)
        uv = (float(row["Ray Texture UV X"]), float(row["Ray Texture UV Y"]))
    except (KeyError, ValueError):
        return None
    return (x, y, z), uv


def measured_frames(rows: Sequence[dict[str, str]]) -> list[MeasuredFrame]:
    """실측 3점 시계열. 오른손을 못 믿는 프레임은 통째로 버린다."""
    gun = active_gun(rows)
    if gun is None:
        return []

    frames: list[MeasuredFrame] = []
    for offset, row in enumerate(rows):
        head = _position(row, HEAD_GROUP)
        right = _position(row, gun)
        if head is None or right is None:
            continue
        span = math.dist(head, right)
        if span > MAX_HEAD_HAND_M or span < MIN_HEAD_RIGHT_M:
            continue

        left = _position(row, LEFT_GROUP)
        if left is not None:
            reach = math.dist(head, left)
            if reach > MAX_HEAD_HAND_M or reach < MIN_HEAD_LEFT_M:
                left = None

        hit_uv = _hit(row)
        try:
            index = int(row["인덱스"])
            time_sec = float(row["시간(초)"])
        except (KeyError, ValueError):
            index, time_sec = offset, 0.0

        frames.append(
            MeasuredFrame(
                index=index,
                time_sec=time_sec,
                head=head,
                right=right,
                left=left,
                hit=None if hit_uv is None else hit_uv[0],
                uv=None if hit_uv is None else hit_uv[1],
                spraying=row.get("분사", "").strip() == "On",
            )
        )
    return frames
