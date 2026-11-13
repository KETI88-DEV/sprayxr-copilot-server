#!/usr/bin/env python3
"""날짜별 부분합을 들고 임의의 부분집합에 대해 평균·산포를 재구성한다.

질의가 전형기 자신의 패스면 그 날짜를 레퍼런스에서 빼야 한다. 자기가 자기 평균에
가까운 것은 당연하기 때문이다. 개선율의 분리 채점도 날짜를 두 그룹으로 갈라야 한다.

전수 재빌드 없이 하려면 합과 제곱합을 날짜별로 들고 있으면 된다. 둘 다 가산적이라
부분집합의 통계가 덧셈만으로 나온다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True, slots=True)
class AxisAccum:
    """한 축의 부분합. n=0이면 빈 누적이다."""

    n: int = 0
    total: float = 0.0
    total_sq: float = 0.0


def accumulate(values: Iterable[float]) -> AxisAccum:
    count = 0
    total = 0.0
    total_sq = 0.0
    for value in values:
        count += 1
        total += value
        total_sq += value * value
    return AxisAccum(n=count, total=total, total_sq=total_sq)


def merge(parts: Iterable[AxisAccum]) -> AxisAccum:
    count = 0
    total = 0.0
    total_sq = 0.0
    for part in parts:
        count += part.n
        total += part.total
        total_sq += part.total_sq
    return AxisAccum(n=count, total=total, total_sq=total_sq)


def summarize(accum: AxisAccum) -> tuple[float, float]:
    """(평균, 모표준편차). 표본이 2개 미만이면 산포는 0이다."""
    if accum.n <= 0:
        return (0.0, 0.0)
    mean = accum.total / accum.n
    if accum.n < 2:
        return (mean, 0.0)
    variance = accum.total_sq / accum.n - mean * mean
    # 제곱합 방식은 부동소수 오차로 음수가 나올 수 있다.
    return (mean, math.sqrt(variance) if variance > 0.0 else 0.0)


def to_json(accum: AxisAccum) -> dict[str, float]:
    return {"n": accum.n, "total": accum.total, "total_sq": accum.total_sq}


def from_json(raw: Mapping[str, object]) -> AxisAccum:
    return AxisAccum(
        n=int(raw.get("n", 0)),  # type: ignore[arg-type]
        total=float(raw.get("total", 0.0)),  # type: ignore[arg-type]
        total_sq=float(raw.get("total_sq", 0.0)),  # type: ignore[arg-type]
    )
