#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

spread_stats = importlib.import_module("spread_stats")

from trainer.pose_lab.spread_reference import SpreadReference

KEY8 = "m1:horizontal:right:1x2:center-near-high"
KEY_OFF = "m1:horizontal:right:na:center-near-high"


def _bucket(by_date):
    dates = sorted(by_date)
    merged: dict[str, list[float]] = {}
    for date in dates:
        for axis, values in by_date[date].items():
            merged.setdefault(axis, []).extend(values)
    return {
        "n": max(len(values) for values in merged.values()),
        "dates": dates,
        "by_date": {
            date: {
                "right": {
                    axis: spread_stats.to_json(spread_stats.accumulate(values))
                    for axis, values in axes.items()
                }
            }
            for date, axes in by_date.items()
        },
        "points": {
            "right": {
                axis: {
                    "mean": sum(values) / len(values),
                    "spread": spread_stats.summarize(spread_stats.accumulate(values))[1],
                }
                for axis, values in merged.items()
            }
        },
    }


def _payload(uv8=None, uv_off=None):
    return {
        "version": 1, "worker": "전형기", "levels": ["10"], "sessions": 5, "n_min": 20,
        "granularities": {
            "uv8": {KEY8: uv8} if uv8 else {},
            "uv4": {}, "uv2": {},
            "uv_off": {KEY_OFF: uv_off} if uv_off else {},
        },
    }


def _thick(date_count=3, per_date=10, base=0.10):
    return {
        f"2025090{index + 1}": {"wall": [base + 0.01 * step for step in range(per_date)]}
        for index in range(date_count)
    }


KEYS = {"uv8": KEY8, "uv_off": KEY_OFF}


def test_resolve_uses_the_finest_layer_that_meets_n_min() -> None:
    resolved = SpreadReference(_payload(uv8=_bucket(_thick()))).resolve(KEYS)
    assert resolved is not None
    assert resolved.granularity == "uv8"
    assert resolved.n >= 20


def test_resolve_falls_back_when_the_fine_layer_is_thin() -> None:
    thin = _bucket({"20250901": {"wall": [0.10, 0.11, 0.12]}})
    resolved = SpreadReference(_payload(uv8=thin, uv_off=_bucket(_thick()))).resolve(KEYS)
    assert resolved is not None
    assert resolved.granularity == "uv_off", "얇으면 굵은 층으로 내려가야 한다"


def test_resolve_returns_none_when_every_layer_is_thin() -> None:
    thin = _bucket({"20250901": {"wall": [0.10, 0.11]}})
    assert SpreadReference(_payload(uv8=thin)).resolve(KEYS) is None


def test_excluding_a_date_changes_the_statistics() -> None:
    """질의가 전형기 자신이면 그 날짜를 빼야 한다. 자기가 자기 평균에 가까운 건 당연하다."""
    by_date = _thick()
    by_date["20250903"] = {"wall": [0.90 + 0.01 * step for step in range(10)]}
    reference = SpreadReference(_payload(uv8=_bucket(by_date)))
    full = reference.resolve(KEYS)
    without = reference.resolve(KEYS, exclude_dates=["20250903"])
    assert full is not None and without is not None
    assert "20250903" not in without.dates
    assert not math.isclose(
        full.means["right"]["wall"], without.means["right"]["wall"], abs_tol=1e-9
    )


def test_excluded_reconstruction_matches_a_build_without_that_date() -> None:
    by_date = _thick()
    without = SpreadReference(_payload(uv8=_bucket(by_date))).resolve(
        KEYS, exclude_dates=["20250901"]
    )
    trimmed = {date: axes for date, axes in by_date.items() if date != "20250901"}
    baseline = SpreadReference(_payload(uv8=_bucket(trimmed))).resolve(KEYS)
    assert without is not None and baseline is not None
    assert math.isclose(
        without.spreads["right"]["wall"], baseline.spreads["right"]["wall"], abs_tol=1e-9
    )


def test_excluding_every_date_yields_none() -> None:
    by_date = _thick()
    reference = SpreadReference(_payload(uv8=_bucket(by_date)))
    assert reference.resolve(KEYS, exclude_dates=list(by_date)) is None


def test_split_dates_partitions_without_overlap() -> None:
    split = SpreadReference(_payload(uv8=_bucket(_thick(date_count=4)))).split_dates("uv8", KEY8)
    assert split is not None
    first, second = split
    assert first and second
    assert not set(first) & set(second)


def test_split_dates_is_none_with_a_single_date() -> None:
    reference = SpreadReference(_payload(uv8=_bucket({"20250901": {"wall": [0.1] * 25}})))
    assert reference.split_dates("uv8", KEY8) is None


def test_worker_is_exposed_so_callers_know_whose_dates_to_exclude() -> None:
    assert SpreadReference(_payload(uv8=_bucket(_thick()))).worker == "전형기"
