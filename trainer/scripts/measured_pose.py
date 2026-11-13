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

import rotation_math
import simulator_csv
import world_axes

Vec3 = tuple[float, float, float]

GUN_GROUPS: Final[tuple[str, ...]] = ("한길숏건", "한길폴건", "반도숏건", "반도폴건")
HEAD_GROUP: Final = "카메라"
LEFT_GROUP: Final = "플래쉬"

# --- 회전 규약 ---------------------------------------------------------------
#
# 벤더 문서가 없어 `Rotation X/Y/Z`가 어느 오일러 규약인지 확인된 적이 없다(설계 §10).
# 축을 눈으로 고를 수도 없다 — 한 세션에서 RotY는 30.2~154.4로 좁아 피치처럼 보이지만
# RotX·RotZ는 둘 다 0~360을 채운다. 그래서 규약을 **데이터가 고르게 했다**.
#
# 판정은 건이 했다. 분사 광선은 건에서 나가 `Ray Point`에 닿으므로 건의 로컬 전방축은
# `hit − gun`과 거의 정확히 일치해야 한다. 6인 × L1–L10 · 60세션 · 63,280 프레임에서
# (오일러 순서 6 × 부호 2 × 로컬축 6) 72개 후보를 3차원 각으로 재면 아래 하나만
# 중앙값 4.0°이고 나머지는 전부 20.7° 이상이다. 세계 좌표 성분을 Y↔Z로 바꿔 읽으면
# 47.9°로 무너지므로, 이 행렬이 내는 벡터는 **이미 월드 좌표**다.
#
# 같은 규약에서 HMD의 로컬 전방축은 +Z다(착탄 방위와의 순환 상관 0.785, 나머지 축은
# 0.46 이하). 자세한 판정은 `.superpowers/pose-lab-yaw-and-angle-study.md` §1에 있다.
EULER_ORDER: Final = "YXZ"
EULER_SIGN: Final = 1.0
HEAD_FORWARD: Final[Vec3] = (0.0, 0.0, 1.0)
GUN_FORWARD: Final[Vec3] = (0.0, 1.0, 0.0)

# 수평 성분이 이보다 짧으면 방위가 정의되지 않는다(정확히 위/아래를 본 프레임).
# 그 문턱은 world_axes가 소유한다 — `facing_yaw`가 `world_axes.yaw_of`를 그대로 쓴다.

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
    """한 프레임의 실측량. left/hit/uv는 못 믿으면 None이고 프레임 자체는 남는다.

    `yaw`도 실측이다. HMD 회전이 CSV에 들어 있으므로 머리 위치·건 위치와 같은 자격의
    측정량이고, IK를 타지 않는다. 회전 컬럼을 못 읽는 행에서는 None이다.
    """

    index: int
    time_sec: float
    head: Vec3
    right: Vec3
    left: Vec3 | None
    hit: Vec3 | None
    uv: tuple[float, float] | None
    spraying: bool
    yaw: float | None = None


def forward_vector(rotation: Vec3, local_axis: Vec3) -> Vec3:
    """강체의 로컬 축을 월드로 옮긴다. 규약은 위에서 데이터가 고른 것 하나뿐이다."""
    return rotation_math.matvec(
        rotation_math.euler_matrix(EULER_ORDER, rotation, EULER_SIGN), local_axis
    )


def facing_yaw(rotation: Vec3) -> float | None:
    """HMD 회전에서 작업자 정면 방위(라디안). 수평면에서의 방위각이다.

    `motion_key.heading_word` / `body_placement`가 `world_axes.horizontal_of`로 뽑은
    같은 두 성분을 `cos(yaw)·h0 + sin(yaw)·h1`로 쓰므로 이 값이 그대로 그 요다 —
    `world_axes.yaw_of`를 그대로 반환해 그 대응을 강제한다.

    정확히 위/아래를 본 프레임에서는 수평 성분이 사라져 방위가 정의되지 않는다. 그때
    0을 넣으면 월드 수평축 하나를 정면이라고 주장하는 것이 되므로 `yaw_of`가 None을
    낸다.
    """
    forward = forward_vector(rotation, HEAD_FORWARD)
    return world_axes.yaw_of(forward)


def _position(row: dict[str, str], group: str) -> Vec3 | None:
    try:
        x, y, z = simulator_csv.vec3(row, group, "Position")
    except (KeyError, ValueError):
        return None
    return (x, y, z)


def _rotation(row: dict[str, str], group: str) -> Vec3 | None:
    try:
        x, y, z = simulator_csv.vec3(row, group, "Rotation")
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


def _frame_from_row(offset: int, row: dict[str, str], gun: str) -> MeasuredFrame | None:
    """행 하나 -> 프레임 하나(못 믿으면 None). `measured_frames`/
    `measured_frames_with_rows`가 같은 필터·구성 규칙을 공유하려고 뽑아 둔다 — 둘을
    따로 짜면 조건이 갈릴 수 있다."""
    head = _position(row, HEAD_GROUP)
    right = _position(row, gun)
    if head is None or right is None:
        return None
    span = math.dist(head, right)
    if span > MAX_HEAD_HAND_M or span < MIN_HEAD_RIGHT_M:
        return None

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

    return MeasuredFrame(
        index=index,
        time_sec=time_sec,
        head=head,
        right=right,
        left=left,
        hit=None if hit_uv is None else hit_uv[0],
        uv=None if hit_uv is None else hit_uv[1],
        spraying=row.get("분사", "").strip() == "On",
        yaw=None if (rot := _rotation(row, HEAD_GROUP)) is None else facing_yaw(rot),
    )


def measured_frames(rows: Sequence[dict[str, str]]) -> list[MeasuredFrame]:
    """실측 3점 시계열. 오른손을 못 믿는 프레임은 통째로 버린다."""
    gun = active_gun(rows)
    if gun is None:
        return []
    frames: list[MeasuredFrame] = []
    for offset, row in enumerate(rows):
        frame = _frame_from_row(offset, row, gun)
        if frame is not None:
            frames.append(frame)
    return frames


def measured_frames_with_rows(
    rows: Sequence[dict[str, str]]
) -> list[tuple[MeasuredFrame, dict[str, str]]]:
    """`measured_frames`와 같은 필터·순서지만, 프레임마다 원본 CSV 행을 함께 낸다.

    `measured_frames`만 쓰면 그 행을 잃는다 — 리샘플(보간)된 격자점은 어느 원본
    행에도 대응하지 않지만, 전신 FK 스켈레톤(`pose_lab.skeleton`)은 회전 각(원본
    행이 갖고 있다)과 목표 위치(프레임)를 **같은 순간**에서 함께 써야 한다. 그래서
    보간 이전, 원본 행과 1:1인 이 목록이 필요하다.
    """
    gun = active_gun(rows)
    if gun is None:
        return []
    paired: list[tuple[MeasuredFrame, dict[str, str]]] = []
    for offset, row in enumerate(rows):
        frame = _frame_from_row(offset, row, gun)
        if frame is not None:
            paired.append((frame, row))
    return paired
