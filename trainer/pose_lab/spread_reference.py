#!/usr/bin/env python3
"""전형기 산포 레퍼런스를 읽고, 질의 날짜를 제외하고, 입도를 폴백한다.

`build_spread_reference`가 우8/우4/우2/우off 네 층으로 구운 아티팩트를 읽는다. 실측 결과
(작업 3) 우8은 패스의 25.0%만 n>=20을 채운다 — 나머지는 층을 낮춰야 한다. 그래서
`resolve`는 가장 가는 층부터 훑다가 처음으로 n_min을 채우는 층에서 멈춘다. 어느 층도
못 채우면 `None`이다 — 이것은 정상적이고 흔한 결과이지 오류가 아니다. `Resolved.granularity`는
실제로 답을 낸 층의 이름이지 질의가 요청한 층이 아니다.

날짜 제외는 재빌드가 아니라 뺄셈이다. `spread_stats.AxisAccum`이 가산적이므로 버킷의
`by_date` 중 제외할 날짜를 뺀 나머지만 합쳐 다시 `summarize`하면 된다. 이 뺄셈 뒤에는
n이 n_min 밑으로 떨어질 수 있다 — 그래서 n_min 검사는 제외 이후의 n에, 사다리의 매 층마다
적용한다. 그러지 않으면 날짜를 뺀 의미가 없어진다(그 날짜가 만든 문턱을 그 날짜로 넘는 셈).
"""

from __future__ import annotations

import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

spread_stats = importlib.import_module("spread_stats")
build_spread_reference = importlib.import_module("build_spread_reference")

GRANULARITIES = build_spread_reference.GRANULARITIES
N_MIN = build_spread_reference.N_MIN


@dataclass(frozen=True, slots=True)
class Resolved:
    """`resolve`(또는 `spreads_for`가 뒷받침하는 A/B 분리)가 실제로 답을 낸 층의 결과.

    `granularity`는 질의가 요청한 층이 아니라 답이 나온 층이다 — 숫자가 나오지 않은
    층의 이름표를 달면 이 프로젝트가 반복해 밟은 결함(잘못된 이름표 아래 숫자)이 된다.
    """

    granularity: str
    key: str
    n: int
    dates: list[str]
    spreads: dict[str, dict[str, float]]
    means: dict[str, dict[str, float]]


class SpreadReference:
    def __init__(self, payload: Mapping[str, object]) -> None:
        self._payload = payload

    @classmethod
    def load(cls, path: Path) -> "SpreadReference":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    @property
    def worker(self) -> str:
        return str(self._payload["worker"])

    def _bucket(self, granularity: str, key: str) -> Mapping[str, object] | None:
        granularities = self._payload.get("granularities")
        if not isinstance(granularities, Mapping):
            return None
        layer = granularities.get(granularity)
        if not isinstance(layer, Mapping):
            return None
        bucket = layer.get(key)
        if not isinstance(bucket, Mapping):
            return None
        return bucket

    @staticmethod
    def _included_dates(bucket: Mapping[str, object], exclude_dates: Sequence[str]) -> list[str]:
        by_date = bucket.get("by_date")
        if not isinstance(by_date, Mapping):
            return []
        excluded = set(exclude_dates)
        return sorted(date for date in by_date if date not in excluded)

    @staticmethod
    def _merge_dates(
        bucket: Mapping[str, object], dates: Sequence[str]
    ) -> tuple[int, dict[str, dict[str, float]], dict[str, dict[str, float]]] | None:
        by_date = bucket.get("by_date")
        if not isinstance(by_date, Mapping) or not dates:
            return None

        accums: dict[str, dict[str, list]] = {}
        for date in dates:
            day = by_date.get(date)
            if not isinstance(day, Mapping):
                continue
            for point, axes in day.items():
                slot = accums.setdefault(point, {})
                for axis, raw in axes.items():
                    slot.setdefault(axis, []).append(spread_stats.from_json(raw))

        if not accums:
            return None

        means: dict[str, dict[str, float]] = {}
        spreads: dict[str, dict[str, float]] = {}
        counts: list[int] = []
        for point, axes in accums.items():
            means[point] = {}
            spreads[point] = {}
            for axis, parts in axes.items():
                merged = spread_stats.merge(parts)
                mean, spread = spread_stats.summarize(merged)
                means[point][axis] = mean
                spreads[point][axis] = spread
                counts.append(merged.n)

        return (max(counts) if counts else 0), spreads, means

    def resolve(self, keys: Mapping[str, str], exclude_dates: Sequence[str] = ()) -> Resolved | None:
        for granularity, _grid in GRANULARITIES:
            key = keys.get(granularity)
            if key is None:
                continue
            bucket = self._bucket(granularity, key)
            if bucket is None:
                continue
            dates = self._included_dates(bucket, exclude_dates)
            if not dates:
                continue
            merged = self._merge_dates(bucket, dates)
            if merged is None:
                continue
            n, spreads, means = merged
            if n < N_MIN:
                continue
            return Resolved(
                granularity=granularity, key=key, n=n, dates=dates,
                spreads=spreads, means=means,
            )
        return None

    def split_dates(self, granularity: str, key: str) -> tuple[list[str], list[str]] | None:
        bucket = self._bucket(granularity, key)
        if bucket is None:
            return None
        by_date = bucket.get("by_date")
        if not isinstance(by_date, Mapping):
            return None
        dates = sorted(by_date)
        first, second = dates[0::2], dates[1::2]
        if not first or not second:
            return None
        return first, second

    def spreads_for(
        self, granularity: str, key: str, dates: Sequence[str]
    ) -> dict[str, dict[str, float]] | None:
        bucket = self._bucket(granularity, key)
        if bucket is None:
            return None
        merged = self._merge_dates(bucket, dates)
        if merged is None:
            return None
        _, spreads, _ = merged
        return spreads
