#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

measured_score = importlib.import_module("measured_score")
pose_similarity = importlib.import_module("pose_similarity")

SPREADS = {"right": 0.05, "head": 0.05, "left": 0.05}


def _track() -> dict[str, list[tuple[float, float, float]]]:
    return {name: [(0.3, 0.0, 0.1)] * 5 for name in ("right", "head", "left")}


def test_identical_tracks_score_one() -> None:
    result = measured_score.score_track(
        _track(), _track(), SPREADS, measured_score.DEFAULT_WEIGHTS
    )
    assert math.isclose(result.score, 1.0, abs_tol=1e-9)
    assert result.samples == 5


def test_right_hand_dominates_the_score() -> None:
    """오른손이 건이라 도막 품질을 직접 결정한다. 가중이 실제로 반영돼야 한다."""
    right_off = _track()
    right_off["right"] = [(0.4, 0.0, 0.1)] * 5
    left_off = _track()
    left_off["left"] = [(0.4, 0.0, 0.1)] * 5

    right_score = measured_score.score_track(
        right_off, _track(), SPREADS, measured_score.DEFAULT_WEIGHTS
    ).score
    left_score = measured_score.score_track(
        left_off, _track(), SPREADS, measured_score.DEFAULT_WEIGHTS
    ).score
    assert right_score < left_score


def test_missing_point_is_skipped_not_scored_zero() -> None:
    """못 잰 것을 0점으로 넣으면 평균이 근거 없이 내려간다."""
    query = _track()
    query["left"] = []
    result = measured_score.score_track(
        query, _track(), SPREADS, measured_score.DEFAULT_WEIGHTS
    )
    assert math.isclose(result.score, 1.0, abs_tol=1e-9)
    assert [delta.point for delta in result.deltas] == ["right", "head"]


def test_correction_text_uses_measured_quantities_not_angles() -> None:
    """REQ-19 게이트 통과 전까지 각도는 근거보다 높은 정밀도다.

    문장 전체를 고정한다. 조각만 확인하면 조사·어미가 깨진 비문이 통과한다.
    """
    deltas = [
        measured_score.PointDelta(point="right", sigma=3.0, local_delta=(0.05, 0.0, 0.15))
    ]
    assert measured_score.corrections_text(deltas) == [
        "오른손을 벽에서 5cm 더 띄우고 15cm 올리세요."
    ]


def test_correction_picks_the_particle_from_the_label() -> None:
    """받침 없는 라벨에는 를이 붙는다. "머리을"은 비문이다."""
    head = measured_score.PointDelta(
        point="head", sigma=3.0, local_delta=(0.0, -0.12, 0.0)
    )
    left = measured_score.PointDelta(
        point="left", sigma=3.0, local_delta=(0.0, 0.0, -0.08)
    )
    assert measured_score.corrections_text([head]) == [
        "머리를 진행 방향 뒤로 12cm 옮기세요."
    ]
    assert measured_score.corrections_text([left]) == ["왼손을 8cm 내리세요."]


def test_correction_skips_points_inside_expert_range() -> None:
    deltas = [
        measured_score.PointDelta(point="head", sigma=1.0, local_delta=(0.01, 0.0, 0.0))
    ]
    assert measured_score.corrections_text(deltas) == []


def test_apply_measured_improves_the_score() -> None:
    query = _track()
    query["right"] = [(0.45, 0.0, 0.1)] * 5

    before = measured_score.score_track(
        query, _track(), SPREADS, measured_score.DEFAULT_WEIGHTS
    )
    fixed = measured_score.apply_measured(query, before.deltas)
    after = measured_score.score_track(
        fixed, _track(), SPREADS, measured_score.DEFAULT_WEIGHTS
    )
    assert after.score > before.score


def test_gap_closure_reuses_the_existing_metric() -> None:
    closure = pose_similarity.gap_closure(0.4, 0.7)
    assert closure is not None
    assert math.isclose(closure, 0.5, abs_tol=1e-9)
