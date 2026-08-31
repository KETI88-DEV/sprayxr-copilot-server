#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

improvement_study = importlib.import_module("improvement_study")
measured_score = importlib.import_module("measured_score")


def test_summary_reports_median_quartiles_and_threshold_share() -> None:
    summary = improvement_study.summarize([0.1, 0.2, 0.5, 0.75, 0.8, 0.9], threshold=0.70)
    assert summary["n"] == 6
    assert math.isclose(summary["median"], 0.625, abs_tol=1e-9)
    assert math.isclose(summary["at_or_above"], 0.5, abs_tol=1e-9)


def test_none_values_are_excluded_not_zeroed() -> None:
    """계산되지 않은 split을 0으로 넣으면 분포가 근거 없이 내려간다."""
    summary = improvement_study.summarize([0.8, None, 0.9], threshold=0.70)
    assert summary["n"] == 2
    assert math.isclose(summary["at_or_above"], 1.0, abs_tol=1e-9)


def test_empty_input_reports_none_not_zero() -> None:
    summary = improvement_study.summarize([], threshold=0.70)
    assert summary["n"] == 0
    assert summary["median"] is None
    assert summary["at_or_above"] is None


def test_quartiles_are_reported() -> None:
    summary = improvement_study.summarize([0.1, 0.2, 0.3, 0.4], threshold=0.70)
    assert summary["q1"] is not None
    assert summary["q3"] is not None
    assert summary["q1"] <= summary["median"] <= summary["q3"]


def _unknown_outcome() -> object:
    """발동 여부를 재지 않은 결과 하나. 민감도 실행이 내는 모양이다."""
    return improvement_study.Outcome(
        query=None, reference=None, key="m1:horizontal:right:1x2:center-near-high",
        status="ok", reason=None, same=0.2, split=0.1, split_reason=None,
        granularity="uv2", reference_n=30, split_n_a=15, split_n_b=15,
        gate="fail", gate_reason=None, normalization="measured",
        fired=None, split_fired=None, ceiling=0.5,
        before_same_score=0.3, before_split_score=0.3,
    )


def test_share_fired_is_none_when_nothing_was_measured() -> None:
    """아무것도 재지 않았을 때의 0.0은 '하나도 안 발동했다'로 읽힌다. 그건 거짓이다.

    민감도 실행은 compute_extras=False라 발동 여부를 아예 계산하지 않는다. 그때
    전체 건수로 나누면 0/149 = 0.0이 나오고, 옆에 unknown=149가 적혀 있어도 그
    0.0은 측정값처럼 읽힌다.
    """
    unknown = [_unknown_outcome() for _ in range(3)]
    block = improvement_study._fire_rate_block([], [], unknown, lambda o: o.same)
    assert block["unknown"] == 3
    assert block["share_fired"] is None


def test_share_fired_divides_by_what_was_actually_measured() -> None:
    """분모는 전체가 아니라 발동 여부를 아는 건수다."""
    fired = [_unknown_outcome()]
    not_fired = [_unknown_outcome()]
    unknown = [_unknown_outcome() for _ in range(8)]
    block = improvement_study._fire_rate_block(fired, not_fired, unknown, lambda o: o.same)
    assert math.isclose(block["share_fired"], 0.5, abs_tol=1e-9)


def test_ceiling_leaves_the_left_hand_and_the_head_lateral_untouched() -> None:
    """천장은 '지시할 수 있는 것을 전부 완벽히 따랐을 때'다.

    왼손은 지시 대상이 아니고 머리 lateral도 아니다. 그 둘을 0으로 치면 천장이
    실제보다 높게 나오고, 미달 폭이 실제보다 작아 보인다.
    """
    sigmas = {
        "right": {"wall": 3.0, "vertical": 3.0, "lateral": 3.0},
        "head": {"wall": 2.5, "vertical": 2.5, "lateral": 2.5},
        "left": {"wall": 4.0, "vertical": 4.0, "lateral": 4.0},
    }
    weights = {"right": 1.0, "head": 0.4, "left": 0.15}
    residual = improvement_study.ceiling_residual_sigma(sigmas, weights)
    assert residual["left"] == pytest.approx(4.0 * math.sqrt(3))
    assert residual["head"] == pytest.approx(2.5)
    assert residual["right"] == pytest.approx(0.0)


def test_correctable_head_axes_match_the_scorer() -> None:
    """천장이 쓰는 '지시 가능한 축'과 실제 지시 규칙이 갈라지면 천장이 틀린다."""
    assert improvement_study.CORRECTABLE["head"] == measured_score.STANCE_AXES
