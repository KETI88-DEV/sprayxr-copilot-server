#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

stroke_frame = importlib.import_module("stroke_frame")
measured_pose = importlib.import_module("measured_pose")


def _line(count: int) -> list[tuple[float, float, float]]:
    return [(1.2, 4.0 + i * (1.0 / (count - 1)), 0.8) for i in range(count)]


def test_arclengths_are_normalized() -> None:
    values = stroke_frame.arclengths(_line(5))
    assert values[0] == 0.0
    assert values[-1] == 1.0
    assert all(a < b for a, b in zip(values, values[1:]))


def test_arclength_ignores_dwell_frames() -> None:
    """제자리에 머문 프레임은 호길이를 늘리지 않는다. 시간 위상과 다른 점이다."""
    points = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    assert stroke_frame.arclengths(points) == [0.0, 0.0, 1.0]


def test_resample_returns_requested_count_on_the_path() -> None:
    points = stroke_frame.resample(_line(11), 5)
    assert len(points) == 5
    assert points[0] == (1.2, 4.0, 0.8)
    assert math.isclose(points[-1][1], 5.0, abs_tol=1e-9)
    assert math.isclose(points[2][1], 4.5, abs_tol=1e-6)


def test_same_path_at_different_speed_resamples_identically() -> None:
    """속도만 다른 두 궤적은 호길이 정렬 후 같아야 한다. 시간 정렬이면 어긋난다."""
    slow = stroke_frame.resample(_line(41), 20)
    fast = stroke_frame.resample(_line(11), 20)
    assert len(slow) == 20
    for left, right in zip(slow, fast):
        for axis in range(3):
            assert math.isclose(left[axis], right[axis], abs_tol=1e-6)


def test_stroke_basis_is_orthonormal() -> None:
    basis = stroke_frame.stroke_basis(
        hit=(1.2, 4.5, 0.8), tangent=(0.0, 1.0, 0.0), gun=(0.6, 4.5, 0.8)
    )
    axes = (basis.normal, basis.tangent, basis.up)
    for axis in axes:
        assert math.isclose(math.dist(axis, (0.0, 0.0, 0.0)), 1.0, abs_tol=1e-9)
    for i in range(3):
        for j in range(i + 1, 3):
            dot = sum(axes[i][k] * axes[j][k] for k in range(3))
            assert math.isclose(dot, 0.0, abs_tol=1e-9)


def test_normal_points_from_hit_toward_the_worker() -> None:
    """건 거리를 양수로 읽으려면 법선이 작업자 쪽을 향해야 한다."""
    basis = stroke_frame.stroke_basis(
        hit=(1.2, 4.5, 0.8), tangent=(0.0, 1.0, 0.0), gun=(0.6, 4.5, 0.8)
    )
    local = stroke_frame.to_local(basis, (0.6, 4.5, 0.8))
    assert math.isclose(local[0], 0.6, abs_tol=1e-9)


def test_to_local_puts_hit_at_origin() -> None:
    basis = stroke_frame.stroke_basis(
        hit=(1.2, 4.5, 0.8), tangent=(0.0, 1.0, 0.0), gun=(0.6, 4.5, 0.8)
    )
    for axis in stroke_frame.to_local(basis, (1.2, 4.5, 0.8)):
        assert math.isclose(axis, 0.0, abs_tol=1e-9)


def test_resample_frames_keeps_three_points_aligned() -> None:
    frames = [
        measured_pose.MeasuredFrame(
            index=i,
            time_sec=i * 0.0111,
            head=(0.5, 4.0 + i * 0.1, 0.8),
            right=(0.85, 4.0 + i * 0.1, 0.65),
            left=(0.67, 4.0 + i * 0.1, 0.79),
            hit=(1.2, 4.0 + i * 0.1, 0.8),
            uv=(1024.0, 512.0),
            spraying=True,
        )
        for i in range(11)
    ]
    resampled = stroke_frame.resample_frames(frames, 5)
    assert len(resampled) == 5
    assert resampled[-1].hit is not None
    assert math.isclose(resampled[-1].hit[1], 5.0, abs_tol=1e-6)
    assert math.isclose(resampled[2].head[1], 4.5, abs_tol=1e-6)
