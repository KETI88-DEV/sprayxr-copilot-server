#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import importlib
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

spread_stats = importlib.import_module("spread_stats")


def test_summarize_matches_direct_statistics() -> None:
    values = [0.10, 0.12, 0.09, 0.15, 0.11]
    mean, spread = spread_stats.summarize(spread_stats.accumulate(values))
    assert math.isclose(mean, st.mean(values), abs_tol=1e-12)
    assert math.isclose(spread, st.pstdev(values), abs_tol=1e-9)


def test_merged_parts_equal_one_pass_over_everything() -> None:
    """부분합을 합친 결과가 전수 계산과 같아야 날짜 제외가 성립한다."""
    left = [0.10, 0.12, 0.09]
    right = [0.15, 0.11, 0.13, 0.14]
    merged = spread_stats.merge(
        [spread_stats.accumulate(left), spread_stats.accumulate(right)]
    )
    direct = spread_stats.accumulate(left + right)
    assert merged.n == direct.n
    assert math.isclose(merged.total, direct.total, abs_tol=1e-12)
    assert math.isclose(merged.total_sq, direct.total_sq, abs_tol=1e-12)
    for a, b in zip(spread_stats.summarize(merged), spread_stats.summarize(direct)):
        assert math.isclose(a, b, abs_tol=1e-12)


def test_excluding_a_part_equals_building_without_it() -> None:
    a = spread_stats.accumulate([0.10, 0.12])
    b = spread_stats.accumulate([0.15, 0.11, 0.13])
    c = spread_stats.accumulate([0.20, 0.21])
    assert b.n == 3  # b는 제외 대상이며 결과에 들어가면 안 된다
    without_b = spread_stats.merge([a, c])
    direct = spread_stats.accumulate([0.10, 0.12, 0.20, 0.21])
    assert without_b.n == direct.n
    for x, y in zip(spread_stats.summarize(without_b), spread_stats.summarize(direct)):
        assert math.isclose(x, y, abs_tol=1e-9)


def test_single_sample_has_no_spread() -> None:
    mean, spread = spread_stats.summarize(spread_stats.accumulate([0.42]))
    assert math.isclose(mean, 0.42, abs_tol=1e-12)
    assert spread == 0.0


def test_empty_accumulator_is_zero() -> None:
    accum = spread_stats.merge([])
    assert accum.n == 0
    assert spread_stats.summarize(accum) == (0.0, 0.0)


def test_json_round_trip() -> None:
    accum = spread_stats.accumulate([0.10, 0.12, 0.09])
    assert spread_stats.from_json(spread_stats.to_json(accum)) == accum


def test_negative_variance_from_rounding_is_clamped() -> None:
    """제곱합 방식은 부동소수 오차로 음수 분산이 나올 수 있다. 0으로 막는다."""
    accum = spread_stats.AxisAccum(n=3, total=3.0, total_sq=2.9999999999)
    _mean, spread = spread_stats.summarize(accum)
    assert spread >= 0.0
