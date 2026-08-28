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
stroke_frame = importlib.import_module("stroke_frame")

SPREADS = {"right": 0.05, "head": 0.05, "left": 0.05}

# 수평 스트로크: 접선이 월드 Y라 up이 정확히 월드 Z다.
HORIZONTAL = stroke_frame.stroke_basis(
    hit=(1.2, 4.5, 0.8), tangent=(0.0, 1.0, 0.0), gun=(0.6, 4.5, 0.8)
)
# 수직 스트로크: 접선이 월드 Z라 up의 직교화가 퇴화하고 대체축(월드 Y)이 들어온다.
# 즉 여기서 up은 수평이고 수직은 tangent다.
VERTICAL = stroke_frame.stroke_basis(
    hit=(1.2, 4.5, 0.8), tangent=(0.0, 0.0, 1.0), gun=(0.6, 4.5, 0.9)
)


def _delta(basis: object, world: tuple[float, float, float]) -> object:
    """월드 오프셋을 그 기저의 로컬 성분으로 적는다. 채점 경로가 만드는 값과 같은 형식이다."""
    local = tuple(
        sum(world[axis] * getattr(basis, name)[axis] for axis in range(3))
        for name in ("normal", "tangent", "up")
    )
    return measured_score.PointDelta(point="right", sigma=3.0, local_delta=local)


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


def test_height_wording_survives_on_a_horizontal_stroke() -> None:
    """수평 스트로크에서 순수 수직 이탈은 계속 올리/내리여야 한다.

    수직 문구를 수직 스트로크에서 그냥 막아 버리는 식의 회피를 막는 못이다.
    """
    line = measured_score.corrections_text(
        [_delta(HORIZONTAL, (0.0, 0.0, 0.15))], HORIZONTAL
    )
    assert line == ["오른손을 15cm 올리세요."]


def test_sideways_offset_on_a_vertical_stroke_is_not_called_height() -> None:
    """수직 스트로크에서 up 축은 수평이다.

    성분을 축 이름으로 읽으면 옆으로 15cm 틀어진 손에 "15cm 올리세요"가 나간다.
    작업자가 그대로 따르면 이탈이 커진다.
    """
    lines = measured_score.corrections_text(
        [_delta(VERTICAL, (0.0, 0.15, 0.0))], VERTICAL
    )
    assert lines == ["오른손을 벽을 따라 15cm 옮기세요."]
    assert "올리" not in lines[0] and "내리" not in lines[0]


def test_vertical_offset_on_a_vertical_stroke_is_still_called_height() -> None:
    """수직 스트로크에서도 진짜 수직 이탈은 올리/내리로 나가야 한다.

    그때 수직은 tangent에 실린다. 진행 방향으로 적으면 높이 지시가 사라진다.
    """
    lines = measured_score.corrections_text(
        [_delta(VERTICAL, (0.0, 0.0, -0.15))], VERTICAL
    )
    assert lines == ["오른손을 15cm 내리세요."]


def test_corrections_never_carry_an_angle() -> None:
    """REQ-19 게이트 통과 전까지 각도를 붙이지 않는다. 새 문구에도 적용된다."""
    for basis, world in (
        (HORIZONTAL, (0.05, 0.0, 0.15)),
        (VERTICAL, (0.0, 0.15, 0.0)),
        (VERTICAL, (0.07, 0.0, -0.15)),
    ):
        for line in measured_score.corrections_text([_delta(basis, world)], basis):
            assert "도" not in line


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
