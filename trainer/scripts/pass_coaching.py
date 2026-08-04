#!/usr/bin/env python3
"""패스 궤적 지표를 숙련자 버킷 분포와 대조해 σ 점수와 교정 항목을 낸다.

유사도는 반사실(counterfactual)로 정의한다. 지표 하나를 버킷 중앙값으로 되돌렸을 때
유사도가 얼마나 오르는지를 계산해 그 값으로 교정 우선순위를 매긴다. 그래서 "이 항목을
고치면 유사도가 X%p 오른다"를 교정 전에 말할 수 있고, 교정 후 실측치와 대조할 수 있다.

레퍼런스는 처방이다. 여기서 나오는 판정은 "숙련자 분포와 얼마나 다른가"이지
"틀렸다"가 아니다. 위반 판정은 커버리지 불변식(재도포/미도포/wet edge)이 맡는다.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Sequence

from pass_segmentation import PassMetrics

__all__ = [
    "Correction",
    "MetricScore",
    "PassAssessment",
    "assess_pass",
    "load_reference",
    "similarity_pct",
]

COACHABLE_METRICS: Final = ("straightness_rms_mm", "elevation_deg", "speed_mm_per_s")
PERCENTILE_POINTS: Final = (10, 25, 50, 75, 90, 99)
MIN_STD: Final = 1e-6
MAX_SIGMA: Final = 6.0


@dataclass(frozen=True, slots=True)
class MetricScore:
    metric: str
    value: float
    target: float
    sigma: float
    percentile: float


@dataclass(frozen=True, slots=True)
class Correction:
    metric: str
    current: float
    target: float
    sigma: float
    direction: Literal["increase", "decrease"]
    expected_gain_pct: float


@dataclass(frozen=True, slots=True)
class PassAssessment:
    bucket: str
    matched: bool
    similarity_pct: float
    scores: list[MetricScore]
    corrections: list[Correction]
    reference_quality: str
    worker_count: int


def load_reference(path: str | Path, level: str) -> dict[str, object] | None:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    entry = payload.get(level)
    if not isinstance(entry, dict) or entry.get("status") != "ok":
        return None
    return entry


def similarity_pct(sigmas: Sequence[float]) -> float:
    """σ가 모두 0이면 100%. 평균 제곱 σ가 커질수록 지수적으로 감소한다."""
    if not sigmas:
        return 0.0
    mean_square = sum(s * s for s in sigmas) / len(sigmas)
    return 100.0 * math.exp(-0.5 * mean_square)


def _percentile_rank(value: float, summary: dict[str, float]) -> float:
    """저장된 분위점 사이를 선형 보간해 백분위를 추정한다."""
    points = [(float(summary[f"p{q}"]), float(q)) for q in PERCENTILE_POINTS]
    if value <= points[0][0]:
        return points[0][1]
    for (low_value, low_q), (high_value, high_q) in zip(points, points[1:]):
        if value <= high_value:
            if high_value == low_value:
                return high_q
            ratio = (value - low_value) / (high_value - low_value)
            return low_q + (high_q - low_q) * ratio
    return 100.0


def _sigma(value: float, summary: dict[str, float]) -> float:
    std = float(summary["std"])
    if std < MIN_STD:
        return 0.0
    raw = (value - float(summary["mean"])) / std
    return max(-MAX_SIGMA, min(MAX_SIGMA, raw))


def assess_pass(
    metrics: PassMetrics,
    reference: dict[str, object],
    coachable: Sequence[str] = COACHABLE_METRICS,
) -> PassAssessment:
    buckets = reference.get("buckets")
    quality = str(reference.get("reference_quality", "unknown"))
    worker_count = int(reference.get("worker_count", 0))
    bucket_entry = buckets.get(metrics.bucket) if isinstance(buckets, dict) else None

    if not isinstance(bucket_entry, dict):
        return PassAssessment(
            bucket=metrics.bucket,
            matched=False,
            similarity_pct=0.0,
            scores=[],
            corrections=[],
            reference_quality=quality,
            worker_count=worker_count,
        )

    scores: list[MetricScore] = []
    for name in coachable:
        summary = bucket_entry.get(name)
        if not isinstance(summary, dict):
            continue
        value = float(getattr(metrics, name))
        scores.append(
            MetricScore(
                metric=name,
                value=value,
                target=float(summary["p50"]),
                sigma=_sigma(value, summary),
                percentile=_percentile_rank(value, summary),
            )
        )

    sigmas = [score.sigma for score in scores]
    baseline = similarity_pct(sigmas)

    corrections: list[Correction] = []
    for index, score in enumerate(scores):
        counterfactual = list(sigmas)
        counterfactual[index] = 0.0
        gain = similarity_pct(counterfactual) - baseline
        if gain <= 0.0:
            continue
        corrections.append(
            Correction(
                metric=score.metric,
                current=score.value,
                target=score.target,
                sigma=score.sigma,
                direction="decrease" if score.value > score.target else "increase",
                expected_gain_pct=gain,
            )
        )
    corrections.sort(key=lambda item: item.expected_gain_pct, reverse=True)

    return PassAssessment(
        bucket=metrics.bucket,
        matched=True,
        similarity_pct=baseline,
        scores=scores,
        corrections=corrections,
        reference_quality=quality,
        worker_count=worker_count,
    )
