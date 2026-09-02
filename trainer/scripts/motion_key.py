#!/usr/bin/env python3
"""동작 동일성 판정 키. 의도가 같은 두 동작만 견주기 위한 것이다.

현행 pass_segmentation.bucket_key는 elevation에 abs를 써서 수직 부호를 버린다.
그래서 우상향 대각과 우하향 대각이 같은 버킷에 들어간다. 몸 기준 좌우 위치 축도 없어
몸 정면 스트로크와 오른쪽으로 뻗은 스트로크가 섞인다. 의도가 다른 동작을 한 분포에
넣으면 버킷 내 산포가 부풀고 그 산포로 나눈 유사도가 무의미해진다.

기존 키는 수정하지 않는다. 소비자가 셋(build_pass_reference, pass_coaching,
build_pose_reference)이고 문자열이 바뀌면 기존 레퍼런스 아티팩트가 조용히 무효화된다.
그래서 접두어 m1: 으로 문자열 공간을 분리한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal

import pass_segmentation
import world_axes

Vec3 = tuple[float, float, float]

SignedBand = Literal["horizontal", "diag_up", "diag_down", "vertical_up", "vertical_down"]
LateralBand = Literal["left", "center", "right"]
ReachBand = Literal["near", "far"]

KEY_PREFIX: Final = "m1"


@dataclass(frozen=True, slots=True)
class KeyParams:
    """판정 경계. 어느 기준에서 유효한지 알아내는 것이 목적이라 값을 고정하지 않는다."""

    uv_grid: int = 8
    lateral_deg: float = 30.0
    reach_m: float = 0.8
    uv_size: float = 2048.0


def _elevation_deg(displacement: Vec3) -> float:
    h0, h1 = world_axes.horizontal_of(displacement)
    horizontal = math.hypot(h0, h1)
    vertical = world_axes.vertical_of(displacement)
    if horizontal == 0.0 and vertical == 0.0:
        return 0.0
    return math.degrees(math.atan2(abs(vertical), horizontal))


def signed_band(displacement: Vec3) -> SignedBand:
    """기울기 대역. 부호는 주축이 아니라 변위에서 온다.

    주축은 PCA 고유벡터라 부호가 임의다. 변위 segment[-1]-segment[0]의 수직 성분만이
    올라갔는지 내려갔는지를 알려준다. "수직"이 어느 인덱스인지는 `world_axes`가 정한다.
    """
    band = pass_segmentation.elevation_band(_elevation_deg(displacement))
    if band == "horizontal":
        return "horizontal"
    rising = world_axes.vertical_of(displacement) >= 0.0
    if band == "vertical":
        return "vertical_up" if rising else "vertical_down"
    return "diag_up" if rising else "diag_down"


def heading_word(yaw: float, displacement: Vec3) -> str:
    """작업자 기준 수평 진행 방향. 수직 스트로크에는 정의되지 않는다."""
    if signed_band(displacement).startswith("vertical"):
        return "none"
    h0, h1 = world_axes.horizontal_of(displacement)
    forward = math.cos(yaw) * h0 + math.sin(yaw) * h1
    right = -math.sin(yaw) * h0 + math.cos(yaw) * h1
    if abs(forward) >= abs(right):
        return "forward" if forward >= 0.0 else "back"
    return "right" if right >= 0.0 else "left"


def uv_cell(uv: tuple[float, float], params: KeyParams) -> tuple[int, int]:
    """UV를 성긴 격자로 양자화한다.

    학습모드는 레벨당 맵이 1개라 Ray Texture Map Number로는 구역이 하나다. region
    라벨은 미저작(REQ-06)이므로 이름 붙은 구역 대신 격자 셀을 직접 쓴다.
    """
    limit = params.uv_grid - 1
    cell: list[int] = []
    for value in uv:
        scaled = int(value / params.uv_size * params.uv_grid)
        cell.append(max(0, min(limit, scaled)))
    return (cell[0], cell[1])


def body_placement(
    head: Vec3, yaw: float, hit: Vec3, params: KeyParams
) -> tuple[LateralBand, ReachBand, str]:
    """머리 기준 좌표계에서 본 착탄점의 위치. 머리는 실측이라 IK를 타지 않는다."""
    delta: Vec3 = (hit[0] - head[0], hit[1] - head[1], hit[2] - head[2])
    d0, d1 = world_axes.horizontal_of(delta)
    forward = math.cos(yaw) * d0 + math.sin(yaw) * d1
    right = -math.sin(yaw) * d0 + math.cos(yaw) * d1

    azimuth = math.degrees(math.atan2(right, forward))
    if abs(azimuth) <= params.lateral_deg:
        lateral: LateralBand = "center"
    else:
        lateral = "right" if azimuth > 0.0 else "left"

    span = math.hypot(forward, right)
    reach: ReachBand = "near" if span <= params.reach_m else "far"
    return lateral, reach, pass_segmentation.height_band(world_axes.vertical_of(hit))


def motion_key(
    displacement: Vec3,
    yaw: float,
    head: Vec3,
    hit: Vec3,
    uv: tuple[float, float] | None,
    params: KeyParams,
) -> str:
    """4축 결합 키. uv가 없으면 대상 지점 축을 na로 둔다."""
    band = signed_band(displacement)
    heading = heading_word(yaw, displacement)
    lateral, reach, height = body_placement(head, yaw, hit, params)
    cell = "na" if uv is None else "{}x{}".format(*uv_cell(uv, params))
    return f"{KEY_PREFIX}:{band}:{heading}:{cell}:{lateral}-{reach}-{height}"
