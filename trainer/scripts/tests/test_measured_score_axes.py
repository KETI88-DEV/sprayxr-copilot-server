#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

measured_score = importlib.import_module("measured_score")

WORLD = measured_score.WORLD_FRAME


def _track(point: tuple[float, float, float]) -> list[tuple[float, float, float]]:
    return [point] * 5


def _tracks(right, head=(0.3, 0.0, 0.1), left=(0.3, 0.0, 0.1)):
    return {"right": _track(right), "head": _track(head), "left": _track(left)}


def test_uniform_spreads_reduces_to_the_old_scalar_behaviour() -> None:
    """축별 산포가 모두 같으면 이전 스칼라 채점과 같은 sigma가 나와야 한다."""
    spreads = measured_score.uniform_spreads(0.05)
    assert set(spreads) == {"right", "head", "left"}
    assert set(spreads["right"]) == set(measured_score.AXIS_ORDER)
    result = measured_score.score_track(
        _tracks((0.4, 0.0, 0.1)), _tracks((0.3, 0.0, 0.1)),
        spreads, measured_score.DEFAULT_WEIGHTS,
    )
    right = next(d for d in result.deltas if d.point == "right")
    assert math.isclose(right.sigma, 2.0, abs_tol=1e-6)


def test_axis_sigmas_are_reported_per_axis() -> None:
    spreads = {name: {"wall": 0.02, "vertical": 0.20, "lateral": 0.20}
               for name in measured_score.POINT_ORDER}
    result = measured_score.score_track(
        _tracks((0.34, 0.0, 0.1)), _tracks((0.3, 0.0, 0.1)),
        spreads, measured_score.DEFAULT_WEIGHTS,
    )
    right = next(d for d in result.deltas if d.point == "right")
    assert set(right.axis_sigmas) == set(measured_score.AXIS_ORDER)
    assert math.isclose(right.axis_sigmas["wall"], 2.0, abs_tol=1e-6)
    assert right.axis_sigmas["vertical"] < 0.5


def test_tight_axis_makes_a_small_deviation_significant() -> None:
    """벽 산포가 좁으면 작은 벽 편차가 큰 진행 방향 편차보다 이상하다."""
    spreads = {name: {"wall": 0.02, "vertical": 0.20, "lateral": 0.30}
               for name in measured_score.POINT_ORDER}
    result = measured_score.score_track(
        _tracks((0.35, 0.17, 0.1)), _tracks((0.3, 0.0, 0.1)),
        spreads, measured_score.DEFAULT_WEIGHTS,
    )
    finding = measured_score.right_hand_finding(result.deltas, WORLD)
    assert finding is not None
    assert finding.axis == "wall", f"raw 크기가 아니라 sigma로 골라야 한다: {finding}"


def test_loose_axis_is_not_flagged_even_when_raw_magnitude_is_large() -> None:
    """진행 방향 산포가 넓으면 17cm는 정상이다."""
    spreads = {name: {"wall": 0.30, "vertical": 0.30, "lateral": 0.30}
               for name in measured_score.POINT_ORDER}
    result = measured_score.score_track(
        _tracks((0.3, 0.17, 0.1)), _tracks((0.3, 0.0, 0.1)),
        spreads, measured_score.DEFAULT_WEIGHTS,
    )
    assert measured_score.right_hand_finding(result.deltas, WORLD) is None


def test_spread_floor_still_applies_per_axis() -> None:
    spreads = {name: {"wall": 0.0, "vertical": 0.0, "lateral": 0.0}
               for name in measured_score.POINT_ORDER}
    result = measured_score.score_track(
        _tracks((0.31, 0.0, 0.1)), _tracks((0.3, 0.0, 0.1)),
        spreads, measured_score.DEFAULT_WEIGHTS,
    )
    right = next(d for d in result.deltas if d.point == "right")
    assert math.isclose(right.axis_sigmas["wall"], 0.5, abs_tol=1e-6)


def test_sigma_is_capped() -> None:
    result = measured_score.score_track(
        _tracks((1.3, 0.0, 0.1)), _tracks((0.3, 0.0, 0.1)),
        measured_score.uniform_spreads(0.01), measured_score.DEFAULT_WEIGHTS,
    )
    right = next(d for d in result.deltas if d.point == "right")
    assert right.sigma <= measured_score.MAX_SIGMA


def test_no_correction_sentence_contains_an_angle() -> None:
    spreads = {name: {"wall": 0.02, "vertical": 0.02, "lateral": 0.02}
               for name in measured_score.POINT_ORDER}
    result = measured_score.score_track(
        _tracks((0.4, 0.1, 0.3)), _tracks((0.3, 0.0, 0.1)),
        spreads, measured_score.DEFAULT_WEIGHTS,
    )
    for line in measured_score.corrections_text(result.deltas, WORLD):
        assert "도" not in line


def test_head_and_left_still_produce_no_sentence() -> None:
    spreads = {name: {"wall": 0.02, "vertical": 0.02, "lateral": 0.02}
               for name in measured_score.POINT_ORDER}
    query = _tracks((0.3, 0.0, 0.1), head=(0.9, 0.0, 0.1), left=(0.9, 0.0, 0.1))
    result = measured_score.score_track(
        query, _tracks((0.3, 0.0, 0.1)), spreads, measured_score.DEFAULT_WEIGHTS
    )
    assert measured_score.corrections_text(result.deltas, WORLD) == []
