#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

compare_passes = importlib.import_module("compare_passes")

Vec3 = tuple[float, float, float]


def _sample(sequence: int, bucket: str) -> object:
    return compare_passes.PassSample(sequence, bucket, {"RightHand": (0.0, 0.0, 1.0)})


def test_pairs_require_the_same_bucket() -> None:
    pairs = compare_passes.same_bucket_pairs(
        [_sample(1, "horizontal-mid-right"), _sample(2, "vertical-high-left")]
    )

    assert pairs == []


def test_pairs_require_different_sequences() -> None:
    """같은 시퀀스 안의 두 스트로크는 같은 상황이라 비교 가치가 낮다."""
    pairs = compare_passes.same_bucket_pairs(
        [_sample(1, "horizontal-mid-right"), _sample(1, "horizontal-mid-right")]
    )

    assert pairs == []


def test_pairs_match_the_same_stroke_across_sequences() -> None:
    left = _sample(1, "horizontal-mid-right")
    right = _sample(4, "horizontal-mid-right")

    pairs = compare_passes.same_bucket_pairs([left, right])

    assert pairs == [(left, right)]


def test_pairs_are_capped_so_the_report_stays_readable() -> None:
    samples = [_sample(index, "horizontal-mid-right") for index in range(10)]

    pairs = compare_passes.same_bucket_pairs(samples, limit=3)

    assert len(pairs) == 3
    assert all(a.sequence != b.sequence for a, b in pairs)
