#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pass_coaching = importlib.import_module("pass_coaching")
pass_segmentation = importlib.import_module("pass_segmentation")

BUCKET = "horizontal-mid-+y"


def _summary(mean: float, std: float, median: float) -> dict[str, Any]:
    spread = max(std, 1.0)
    return {
        "n": 200,
        "mean": mean,
        "std": std,
        "p10": median - 2 * spread,
        "p25": median - spread,
        "p50": median,
        "p75": median + spread,
        "p90": median + 2 * spread,
        "p99": median + 4 * spread,
    }


def _reference() -> dict[str, Any]:
    return {
        "level": "10",
        "status": "ok",
        "reference_quality": "multi_worker",
        "worker_count": 6,
        "buckets": {
            BUCKET: {
                "straightness_rms_mm": _summary(23.0, 10.0, 23.0),
                "elevation_deg": _summary(8.0, 4.0, 8.0),
                "speed_mm_per_s": _summary(2800.0, 500.0, 2800.0),
            }
        },
    }


def _metrics(
    straightness: float = 23.0,
    elevation: float = 8.0,
    speed: float = 2800.0,
    bucket: str = BUCKET,
) -> Any:
    return pass_segmentation.PassMetrics(
        frame_start=0,
        frame_end=40,
        length_mm=350.0,
        straightness_rms_mm=straightness,
        elevation_deg=elevation,
        mid_height_m=0.8,
        heading_axis="y",
        heading_sign="+",
        duration_sec=0.5,
        speed_mm_per_s=speed,
        bucket=bucket,
    )


def test_similarity_is_full_when_all_sigmas_are_zero() -> None:
    assert pass_coaching.similarity_pct([0.0, 0.0, 0.0]) == pytest.approx(100.0)


def test_similarity_decreases_as_sigma_grows() -> None:
    tight = pass_coaching.similarity_pct([0.5])
    loose = pass_coaching.similarity_pct([2.0])
    assert tight > loose
    assert pass_coaching.similarity_pct([]) == 0.0


def test_pass_at_reference_mean_scores_full_similarity() -> None:
    result = pass_coaching.assess_pass(_metrics(), _reference())

    assert result.matched is True
    assert result.similarity_pct == pytest.approx(100.0)
    assert result.corrections == []
    assert result.reference_quality == "multi_worker"
    assert result.worker_count == 6


def test_unknown_bucket_is_reported_as_unmatched() -> None:
    result = pass_coaching.assess_pass(_metrics(bucket="vertical-high--x"), _reference())

    assert result.matched is False
    assert result.scores == []
    assert result.corrections == []


def test_worst_metric_ranks_first_in_corrections() -> None:
    result = pass_coaching.assess_pass(
        _metrics(straightness=93.0, elevation=10.0, speed=2900.0), _reference()
    )

    assert result.corrections
    assert result.corrections[0].metric == "straightness_rms_mm"
    assert result.corrections[0].sigma == pytest.approx(6.0)
    gains = [c.expected_gain_pct for c in result.corrections]
    assert gains == sorted(gains, reverse=True)


def test_correction_direction_follows_target() -> None:
    too_fast = pass_coaching.assess_pass(_metrics(speed=4000.0), _reference())
    too_slow = pass_coaching.assess_pass(_metrics(speed=1500.0), _reference())

    assert too_fast.corrections[0].metric == "speed_mm_per_s"
    assert too_fast.corrections[0].direction == "decrease"
    assert too_slow.corrections[0].direction == "increase"


def test_expected_gain_matches_measured_gain_after_correction() -> None:
    """예측 Δ가 실제로 그 지표를 고쳤을 때의 Δ와 일치해야 평가에 쓸 수 있다."""
    reference = _reference()
    before = pass_coaching.assess_pass(_metrics(straightness=60.0, speed=3500.0), reference)
    top = before.corrections[0]

    corrected = _metrics(straightness=60.0, speed=3500.0)
    fixed_value = reference["buckets"][BUCKET][top.metric]["mean"]
    after = pass_coaching.assess_pass(
        _metrics(
            straightness=fixed_value if top.metric == "straightness_rms_mm" else corrected.straightness_rms_mm,
            speed=fixed_value if top.metric == "speed_mm_per_s" else corrected.speed_mm_per_s,
        ),
        reference,
    )

    measured_gain = after.similarity_pct - before.similarity_pct
    assert measured_gain == pytest.approx(top.expected_gain_pct, rel=1e-6)


def test_sigma_is_clamped_for_extreme_outliers() -> None:
    result = pass_coaching.assess_pass(_metrics(straightness=10_000.0), _reference())
    straightness = next(s for s in result.scores if s.metric == "straightness_rms_mm")
    assert straightness.sigma == pytest.approx(6.0)


def test_zero_std_metric_yields_zero_sigma() -> None:
    reference = _reference()
    reference["buckets"][BUCKET]["elevation_deg"]["std"] = 0.0
    result = pass_coaching.assess_pass(_metrics(elevation=45.0), reference)
    elevation = next(s for s in result.scores if s.metric == "elevation_deg")
    assert elevation.sigma == 0.0


def test_percentile_rank_is_monotone() -> None:
    reference = _reference()
    low = pass_coaching.assess_pass(_metrics(straightness=3.0), reference).scores[0].percentile
    mid = pass_coaching.assess_pass(_metrics(straightness=23.0), reference).scores[0].percentile
    high = pass_coaching.assess_pass(_metrics(straightness=63.0), reference).scores[0].percentile

    assert low < mid < high
    assert mid == pytest.approx(50.0)


def test_load_reference_returns_none_for_missing_or_insufficient(tmp_path: Path) -> None:
    path = tmp_path / "ref.json"
    path.write_text(
        json.dumps({"10": {"status": "insufficient_data"}}), encoding="utf-8"
    )
    assert pass_coaching.load_reference(path, "10") is None
    assert pass_coaching.load_reference(path, "99") is None


def test_load_reference_returns_entry_for_ok_level(tmp_path: Path) -> None:
    path = tmp_path / "ref.json"
    path.write_text(json.dumps({"10": _reference()}), encoding="utf-8")
    entry = pass_coaching.load_reference(path, "10")
    assert entry is not None
    assert entry["reference_quality"] == "multi_worker"
