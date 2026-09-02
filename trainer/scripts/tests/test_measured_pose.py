#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

measured_pose = importlib.import_module("measured_pose")

PARKED = (-694.61, 701.29, 698.25)
STILL = (-2.99, 4.66, -0.27)


def _row(
    index: int,
    head: tuple[float, float, float],
    guns: dict[str, tuple[float, float, float]],
    flash: tuple[float, float, float],
    hit: tuple[float, float, float] | None = None,
) -> dict[str, str]:
    """CSV 한 행. 필요한 컬럼만 채운다."""
    row: dict[str, str] = {
        "인덱스": str(index),
        "시간(초)": f"{index * 0.0111:.4f}",
        "분사": "On",
    }
    for axis, value in zip("XYZ", head):
        row[f"카메라 Position {axis}"] = str(value)
    for group, position in guns.items():
        for axis, value in zip("XYZ", position):
            row[f"{group} Position {axis}"] = str(value)
    for axis, value in zip("XYZ", flash):
        row[f"플래쉬 Position {axis}"] = str(value)
    if hit is None:
        row["Ray point X"] = ""
        row[" Ray Point Y"] = ""
        row[" Ray Point Z"] = ""
        row["Ray Texture UV X"] = ""
        row["Ray Texture UV Y"] = ""
    else:
        row["Ray point X"] = str(hit[0])
        row[" Ray Point Y"] = str(hit[1])
        row[" Ray Point Z"] = str(hit[2])
        row["Ray Texture UV X"] = "1024"
        row["Ray Texture UV Y"] = "512"
    return row


def _rows(count: int = 5) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for i in range(count):
        offset = i * 0.1
        rows.append(
            _row(
                i,
                head=(0.5, 4.7 + offset, 0.8),
                guns={
                    "한길숏건": (0.85, 4.6 + offset, 0.65),
                    "한길폴건": PARKED,
                    "반도숏건": STILL,
                    "반도폴건": PARKED,
                },
                flash=(0.67, 4.5 + offset, 0.79),
                hit=(1.26, 4.66 + offset, 0.66),
            )
        )
    return rows


def test_active_gun_is_chosen_by_variance_not_coordinate() -> None:
    """반도숏건은 주차 좌표가 아닌 정상 위치에 정지한다. 좌표로 판별하면 틀린다."""
    assert measured_pose.active_gun(_rows()) == "한길숏건"


def test_active_gun_is_none_when_nothing_moves() -> None:
    rows = _rows()
    for row in rows:
        for axis, value in zip("XYZ", (0.85, 4.60, 0.65)):
            row[f"한길숏건 Position {axis}"] = str(value)
    assert measured_pose.active_gun(rows) is None


def test_measured_frames_reads_three_points() -> None:
    frames = measured_pose.measured_frames(_rows())
    assert len(frames) == 5
    first = frames[0]
    assert first.head == (0.5, 4.7, 0.8)
    assert first.right == (0.85, 4.6, 0.65)
    assert first.left == (0.67, 4.5, 0.79)
    assert first.hit == (1.26, 4.66, 0.66)
    assert first.uv == (1024.0, 512.0)
    assert first.spraying is True


def test_frame_dropped_when_gun_too_far_from_head() -> None:
    """건을 내려놓은 프레임. 실측 최대 |머리-건| 4.443m는 인체에서 불가능하다."""
    rows = _rows()
    for axis, value in zip("XYZ", (5.0, 4.6, 0.65)):
        rows[2][f"한길숏건 Position {axis}"] = str(value)
    frames = measured_pose.measured_frames(rows)
    assert [frame.index for frame in frames] == [0, 1, 3, 4]


def test_left_hand_dropped_alone_when_overlapping_head() -> None:
    """왼손만 못 믿는 프레임은 왼손만 버리고 나머지 지표는 남긴다.

    i=2의 head는 (0.5, 4.9, 0.8)이다. 플래쉬를 같은 좌표에 두어야 겹침이 된다.
    """
    rows = _rows()
    for axis, value in zip("XYZ", (0.5, 4.9, 0.8)):
        rows[2][f"플래쉬 Position {axis}"] = str(value)
    frames = measured_pose.measured_frames(rows)
    assert [frame.index for frame in frames] == [0, 1, 2, 3, 4]
    assert frames[2].left is None
    assert frames[2].right is not None


def test_missing_ray_leaves_hit_none_but_keeps_frame() -> None:
    rows = _rows()
    rows[3] = _row(
        3,
        head=(0.5, 5.0, 0.8),
        guns={
            "한길숏건": (0.85, 4.9, 0.65),
            "한길폴건": PARKED,
            "반도숏건": STILL,
            "반도폴건": PARKED,
        },
        flash=(0.67, 4.8, 0.79),
        hit=None,
    )
    frames = measured_pose.measured_frames(rows)
    assert frames[3].hit is None
    assert frames[3].uv is None
    assert frames[3].head == (0.5, 5.0, 0.8)


# --- 실측 요 -------------------------------------------------------------------
#
# 규약(YXZ · 부호 +1 · HMD 로컬 +Z)은 실데이터가 골랐다. 판정은
# `.superpowers/pose-lab-yaw-and-angle-study.md` §1에 있고, 여기서는 그 규약이 코드에
# 그대로 박혀 있는지와 퇴화 처리를 잠근다.


def test_facing_yaw_reads_the_convention_the_data_chose() -> None:
    """YXZ에서 전방 = (sin RotY·cos RotX, −sin RotX, cos RotY·cos RotX)이다.

    `world_axes.yaw_of`는 수평(X,Z) 성분으로 방위를 잰다(유니티 Y-up이므로 Y가
    수직이다) — RotY=90 고정·RotX 가변으로 만든 옛 픽스처는 전방[2](cos RotY·cos RotX)
    가 RotY=90에서 늘 0이 되어(cos 90=0) 새 수평 축(X,Z) 중 Z가 상시 0으로 죽는다.
    그래서 RotX=0으로 고정하고 RotY를 돌린다: 전방 = (sin RotY, −sin 0, cos RotY·1) =
    (sin RotY, 0, cos RotY)다. 수평 성분은 (전방[0], 전방[2]) = (sin RotY, cos RotY)이고
    `yaw = atan2(cos RotY, sin RotY)`다. 항등식 atan2(cosθ, sinθ) = 90° − θ(mod 360°)
    로 손으로 유도한다(θ=0→90°=atan2(1,0) ✓, θ=90°→0=atan2(0,1) ✓, θ=180°→−90°=
    atan2(−1,0) ✓로 검산). 이 항등식이 깨지면 규약이 바뀐 것이고, 그때는 몸 기준
    축 전체가 조용히 틀어진다.
    """
    for degrees in (0.0, 30.0, 120.0, 300.0):
        yaw = measured_pose.facing_yaw((0.0, degrees, 0.0))
        assert yaw is not None
        expected = math.radians(90.0) - math.radians(degrees)
        assert math.isclose(math.cos(yaw), math.cos(expected), abs_tol=1e-9)
        assert math.isclose(math.sin(yaw), math.sin(expected), abs_tol=1e-9)


def test_facing_yaw_is_none_when_the_head_points_straight_up() -> None:
    """수평 성분이 사라지면 방위가 없다. 지어내면 월드 수평축 하나를 정면이라 주장하게
    된다.

    RotX=90이면 cos RotX=0이라 전방=(sin RotY·0, −sin 90, cos RotY·0)=(0,−1,0)으로
    RotY와 무관하게 순수 수직(월드 Y)이 된다 — 유니티 Y-up이므로 수평(X,Z) 성분이
    둘 다 0이다.
    """
    assert measured_pose.facing_yaw((90.0, 0.0, 0.0)) is None


def test_gun_forward_axis_is_the_spray_ray() -> None:
    """건의 로컬 +Y가 분사 광선이다 — 이 규약을 고른 실측 판정의 핵심이다."""
    forward = measured_pose.forward_vector((0.0, 0.0, 0.0), measured_pose.GUN_FORWARD)
    assert forward == (0.0, 1.0, 0.0)


def test_frames_carry_the_measured_yaw() -> None:
    """RotX=0, RotY=30이면 전방=(sin30,0,cos30)이라 yaw=atan2(cos30,sin30)=90°−30°=60°다
    (위 `test_facing_yaw_reads_the_convention_the_data_chose`의 항등식과 같다)."""
    rows = _rows()
    for row in rows:
        for axis, value in zip("XYZ", (0.0, 30.0, 0.0)):
            row[f"카메라 Rotation {axis}"] = str(value)
    frames = measured_pose.measured_frames(rows)
    assert all(frame.yaw is not None for frame in frames)
    assert math.isclose(frames[0].yaw or 0.0, math.radians(60.0), abs_tol=1e-9)


def test_yaw_is_none_when_the_rotation_columns_are_absent() -> None:
    """없는 값을 0으로 채우면 몸 기준 축이 조용히 월드 방위각으로 돌아간다."""
    frames = measured_pose.measured_frames(_rows())
    assert all(frame.yaw is None for frame in frames)
