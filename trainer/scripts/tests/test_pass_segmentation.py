#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pass_segmentation = importlib.import_module("pass_segmentation")

Vec3 = tuple[float, float, float]


def _line(start: Vec3, end: Vec3, count: int) -> list[Vec3]:
    """start에서 end까지 등간격 직선. count는 점 개수."""
    step = count - 1
    return [
        (
            start[0] + (end[0] - start[0]) * i / step,
            start[1] + (end[1] - start[1]) * i / step,
            start[2] + (end[2] - start[2]) * i / step,
        )
        for i in range(count)
    ]


def _times(count: int, dt: float = 1.0 / 90.0) -> list[float]:
    return [i * dt for i in range(count)]


def test_straight_horizontal_burst_is_one_pass() -> None:
    points = _line((0.0, 0.0, 0.8), (0.0, 0.6, 0.8), 40)
    metrics = pass_segmentation.segment_burst(points, _times(40))

    assert len(metrics) == 1
    only = metrics[0]
    assert only.elevation_deg == pytest.approx(0.0, abs=1e-6)
    assert only.straightness_rms_mm == pytest.approx(0.0, abs=1e-6)
    assert only.length_mm == pytest.approx(600.0, rel=1e-6)
    assert only.heading_axis == "y"
    assert only.heading_sign == "+"
    assert only.bucket == "horizontal-mid-+y"


def test_vertical_pass_reports_ninety_degrees() -> None:
    points = _line((0.0, 0.0, 0.2), (0.0, 0.0, 0.9), 40)
    metrics = pass_segmentation.segment_burst(points, _times(40))

    assert len(metrics) == 1
    assert metrics[0].elevation_deg == pytest.approx(90.0, abs=1e-6)
    assert pass_segmentation.elevation_band(metrics[0].elevation_deg) == "vertical"


def test_serpentine_burst_splits_into_three_passes() -> None:
    points = (
        _line((0.0, 0.0, 0.70), (0.0, 0.5, 0.70), 30)
        + _line((0.0, 0.5, 0.75), (0.0, 0.0, 0.75), 30)
        + _line((0.0, 0.0, 0.80), (0.0, 0.5, 0.80), 30)
    )
    metrics = pass_segmentation.segment_burst(points, _times(len(points)))

    assert len(metrics) == 3
    assert [m.heading_sign for m in metrics] == ["+", "-", "+"]
    assert all(m.heading_axis == "y" for m in metrics)
    assert all(pass_segmentation.elevation_band(m.elevation_deg) == "horizontal" for m in metrics)


def test_single_line_is_not_oversplit_by_noise() -> None:
    """미세한 지그재그가 있어도 진행이 단조로우면 한 패스로 남는다."""
    base = _line((0.0, 0.0, 0.8), (0.0, 0.6, 0.8), 60)
    points: list[Vec3] = [
        (x + (0.002 if i % 2 else -0.002), y, z) for i, (x, y, z) in enumerate(base)
    ]
    metrics = pass_segmentation.segment_burst(points, _times(60))

    assert len(metrics) == 1
    assert metrics[0].straightness_rms_mm > 0.0


def test_short_displacement_pass_is_dropped() -> None:
    points = _line((0.0, 0.0, 0.8), (0.0, 0.02, 0.8), 40)
    assert pass_segmentation.segment_burst(points, _times(40)) == []


def test_burst_shorter_than_minimum_yields_no_spans() -> None:
    points = _line((0.0, 0.0, 0.8), (0.0, 0.6, 0.8), 9)
    assert pass_segmentation.split_pass_spans(points) == []
    assert pass_segmentation.segment_burst(points, _times(9)) == []


def test_speed_uses_path_length_over_duration() -> None:
    points = _line((0.0, 0.0, 0.8), (0.0, 0.9, 0.8), 91)
    times = [i * 0.01 for i in range(91)]
    metrics = pass_segmentation.segment_burst(points, times)

    assert len(metrics) == 1
    assert metrics[0].duration_sec == pytest.approx(0.9)
    assert metrics[0].speed_mm_per_s == pytest.approx(1000.0, rel=1e-6)


def test_negative_heading_sign_and_x_axis() -> None:
    points = _line((0.5, 0.0, 0.3), (0.0, 0.0, 0.3), 40)
    metrics = pass_segmentation.segment_burst(points, _times(40))

    assert len(metrics) == 1
    assert metrics[0].heading_axis == "x"
    assert metrics[0].heading_sign == "-"
    assert metrics[0].bucket == "horizontal-low--x"


def test_height_bands_follow_measured_thresholds() -> None:
    assert pass_segmentation.height_band(0.59) == "low"
    assert pass_segmentation.height_band(0.60) == "mid"
    assert pass_segmentation.height_band(0.99) == "mid"
    assert pass_segmentation.height_band(1.00) == "high"


def test_elevation_bands_follow_measured_thresholds() -> None:
    assert pass_segmentation.elevation_band(20.0) == "horizontal"
    assert pass_segmentation.elevation_band(20.1) == "diagonal"
    assert pass_segmentation.elevation_band(69.9) == "diagonal"
    assert pass_segmentation.elevation_band(70.0) == "vertical"


def test_bucket_key_is_stable_for_pose_reference_reuse() -> None:
    assert pass_segmentation.bucket_key(5.0, 0.8, "y", "+") == "horizontal-mid-+y"
    assert pass_segmentation.bucket_key(85.0, 1.2, "x", "-") == "vertical-high--x"


def test_curved_pass_has_larger_deviation_than_straight_pass() -> None:
    straight = _line((0.0, 0.0, 0.8), (0.0, 0.6, 0.8), 40)
    arc: list[Vec3] = [
        (0.0, y, 0.8 + 0.05 * math.sin(math.pi * i / 39))
        for i, (_, y, _) in enumerate(straight)
    ]
    straight_metrics = pass_segmentation.segment_burst(straight, _times(40))
    arc_metrics = pass_segmentation.segment_burst(arc, _times(40))

    assert len(arc_metrics) == 1
    assert arc_metrics[0].straightness_rms_mm > straight_metrics[0].straightness_rms_mm + 10.0


def test_mismatched_times_length_raises() -> None:
    points = _line((0.0, 0.0, 0.8), (0.0, 0.6, 0.8), 40)
    with pytest.raises(ValueError):
        pass_segmentation.segment_burst(points, _times(39))


def test_principal_axis_recovers_direction() -> None:
    points = _line((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 20)
    _, axis = pass_segmentation.principal_axis(points)
    assert abs(axis[1]) == pytest.approx(1.0, abs=1e-6)
    assert abs(axis[0]) == pytest.approx(0.0, abs=1e-6)
    assert abs(axis[2]) == pytest.approx(0.0, abs=1e-6)


def test_body_bucket_key_composes_elevation_height_and_body_direction() -> None:
    assert pass_segmentation.body_bucket_key(5.0, 0.8, "forward") == "horizontal-mid-forward"
    assert pass_segmentation.body_bucket_key(80.0, 1.2, "right") == "vertical-high-right"
    assert pass_segmentation.body_bucket_key(45.0, 0.3, "back") == "diagonal-low-back"


def test_body_bucket_keys_never_collide_with_world_frame_keys() -> None:
    """월드 기준 레퍼런스를 몸 기준 질의로 채점하는 사고를 키 공간 분리로 막는다."""
    world = {
        pass_segmentation.bucket_key(elevation, height, axis, sign)
        for elevation in (5.0, 45.0, 80.0)
        for height in (0.3, 0.8, 1.2)
        for axis in ("x", "y")
        for sign in ("+", "-")
    }
    body = {
        pass_segmentation.body_bucket_key(elevation, height, heading)
        for elevation in (5.0, 45.0, 80.0)
        for height in (0.3, 0.8, 1.2)
        for heading in ("right", "left", "forward", "back")
    }

    assert not (world & body)
