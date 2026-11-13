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
from typing import Final, Literal, Sequence

import pass_segmentation
import stroke_frame
import world_axes

Vec3 = tuple[float, float, float]

SignedBand = Literal["horizontal", "diag_up", "diag_down", "vertical_up", "vertical_down"]
LateralBand = Literal["left", "center", "right"]
ReachBand = Literal["near", "far"]

# m3(사전등록 `docs/measurements/2026-09-22-same-level-pairs-d1-prereg.md` §3.1, D1): 칸 성분의 UV를
# 원 프레임 호장 중점에서 읽는다(`midpoint_uv`). 다른 성분은 m1과 같다. 문자열 모양이 같아
# 접두어만으로 m1 시절 산포 아티팩트·캐시와 갈린다.
KEY_PREFIX: Final = "m3"


@dataclass(frozen=True, slots=True)
class KeyParams:
    """판정 경계. 어느 기준에서 유효한지 알아내는 것이 목적이라 값을 고정하지 않는다."""

    uv_grid: int = 8
    lateral_deg: float = 30.0
    reach_m: float = 0.8
    uv_size: float = 2048.0
    # Task 6: height_band의 실효 버킷이 1.000(전부 "high")이던 결함의 수정 자리.
    # 원인은 절대 높이(world_axes.vertical_of(hit))에 Z-up 시절 임계(0.6/1.0m)를 쓴
    # 것 — 실측 범위 3.551–5.672m가 전부 그 임계를 넘는다. height_relative=True면
    # 머리 기준 상대 높이(vertical_of(delta), delta=hit-head)를 쓴다 — 평행이동
    # 불변이고 수평 성분(lateral/reach)과 기준이 같아진다. 기본값은 기존과 비트
    # 단위로 같다: height_relative=False, 임계는 pass_segmentation의 기존 상수.
    height_relative: bool = False
    height_low_max_m: float = pass_segmentation.HEIGHT_LOW_MAX_M
    height_mid_max_m: float = pass_segmentation.HEIGHT_MID_MAX_M


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


def _height_band(value_m: float, low_max_m: float, mid_max_m: float) -> str:
    """`pass_segmentation.height_band`와 같은 3단 계단이지만 임계를 인자로 받는다.

    절대 임계(0.6/1.0m)는 Z-up 시절 값이라 상대 좌표(머리 기준, 음수 가능)에는
    맞지 않는다 — 호출자가 기준에 맞는 임계를 골라 넘긴다.
    """
    if value_m < low_max_m:
        return "low"
    if value_m < mid_max_m:
        return "mid"
    return "high"


def local_offsets(head: Vec3, yaw: float, hit: Vec3) -> tuple[float, float, Vec3]:
    """머리 기준 전/우 오프셋(m)과 원 델타(`hit - head`).

    `body_placement`와 Task 6 보정 스크립트(임계값을 고르기 전에 원 분포를 보는 쪽)가
    같은 기하를 공유하는 지점이다 — 회전식을 두 곳에 따로 적으면 어긋날 위험이 있다.
    """
    delta: Vec3 = (hit[0] - head[0], hit[1] - head[1], hit[2] - head[2])
    d0, d1 = world_axes.horizontal_of(delta)
    forward = math.cos(yaw) * d0 + math.sin(yaw) * d1
    right = -math.sin(yaw) * d0 + math.cos(yaw) * d1
    return forward, right, delta


def azimuth_deg(head: Vec3, yaw: float, hit: Vec3) -> float:
    """몸 기준 좌우 방위각(도). `body_placement`의 lateral 밴드가 쓰는 값과 같다."""
    forward, right, _ = local_offsets(head, yaw, hit)
    return math.degrees(math.atan2(right, forward))


def body_placement(
    head: Vec3, yaw: float, hit: Vec3, params: KeyParams
) -> tuple[LateralBand, ReachBand, str]:
    """머리 기준 좌표계에서 본 착탄점의 위치. 머리는 실측이라 IK를 타지 않는다."""
    forward, right, delta = local_offsets(head, yaw, hit)

    azimuth = math.degrees(math.atan2(right, forward))
    if abs(azimuth) <= params.lateral_deg:
        lateral: LateralBand = "center"
    else:
        lateral = "right" if azimuth > 0.0 else "left"

    span = math.hypot(forward, right)
    reach: ReachBand = "near" if span <= params.reach_m else "far"
    # 수평(lateral/reach)은 이미 머리 기준(delta)이다. 수직만 절대(hit)를 쓰면
    # 몸 기준과 월드 기준이 섞인 좌표가 된다(Task 5에서 확인) — height_relative가
    # 켜지면 수직도 delta로 통일해 함수 전체가 평행이동 불변이 되게 한다.
    vertical_value = (
        world_axes.vertical_of(delta) if params.height_relative else world_axes.vertical_of(hit)
    )
    height = _height_band(vertical_value, params.height_low_max_m, params.height_mid_max_m)
    return lateral, reach, height


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


def midpoint_uv(
    hits: Sequence[Vec3], uvs: Sequence[tuple[float, float] | None]
) -> tuple[float, float] | None:
    """원 프레임(리샘플 전) 착탄 궤적에서 누적 호장이 절반에 가장 가까운 프레임의 UV.

    리샘플 격자(`stroke_frame.resample_frames`)는 모든 점에 **첫 프레임의** UV를 복사한다 —
    그 격자의 가운데 UV를 읽으면 시작점 칸이 나온다(m1 키의 결함 D1). 그 프레임에 UV가
    없으면 UV가 있는 가장 가까운 원 프레임을 쓴다(프레임 순번 거리, 같으면 앞쪽).
    UV가 하나도 없으면 None. 정의는 m2 사전등록(`a13f58d`)의 `cell`과 같다.
    """
    if not hits or len(hits) != len(uvs):
        return None
    values = stroke_frame.arclengths(list(hits))
    middle = min(range(len(values)), key=lambda index: abs(values[index] - 0.5))
    with_uv = [index for index, uv in enumerate(uvs) if uv is not None]
    if not with_uv:
        return None
    chosen = min(with_uv, key=lambda index: (abs(index - middle), index))
    return uvs[chosen]
