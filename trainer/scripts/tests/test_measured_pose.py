#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import importlib
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
