#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

builder = importlib.import_module("build_spread_reference")
spread_stats = importlib.import_module("spread_stats")
measured_score = importlib.import_module("measured_score")

KEY = "m1:horizontal:right:1x2:center-near-high"


def _buckets(by_date):
    return {"uv8": {KEY: by_date}, "uv4": {}, "uv2": {}, "uv_off": {}}


def test_granularity_ladder_is_fine_to_coarse_ordered() -> None:
    assert [name for name, _grid in builder.GRANULARITIES] == ["uv8", "uv4", "uv2", "uv_off"]


def test_n_min_is_inherited_not_reinvented() -> None:
    """이전 연구에서 사전등록한 값이다. 이번 수치를 보고 정하지 않는다."""
    assert builder.N_MIN == 20


def test_artifact_has_every_layer_and_carries_dates() -> None:
    payload = builder.build_payload(
        worker="전형기", levels=["10"], sessions=3,
        buckets=_buckets({"20250901": {"right": {"wall": spread_stats.accumulate([0.10, 0.12])}}}),
    )
    assert payload["worker"] == "전형기"
    assert payload["n_min"] == 20
    assert set(payload["granularities"]) == {"uv8", "uv4", "uv2", "uv_off"}
    bucket = payload["granularities"]["uv8"][KEY]
    assert bucket["dates"] == ["20250901"]
    assert bucket["n"] == 2
    assert "by_date" in bucket
    assert bucket["points"]["right"]["wall"]["spread"] >= 0.0


def test_payload_round_trips_through_json() -> None:
    payload = builder.build_payload(
        worker="전형기", levels=["10"], sessions=1,
        buckets=_buckets({"20250901": {"right": {"wall": spread_stats.accumulate([0.1, 0.2])}}}),
    )
    restored = json.loads(json.dumps(payload, ensure_ascii=False))
    assert restored["granularities"]["uv8"][KEY]["n"] == 2


def test_two_dates_merge_into_one_bucket() -> None:
    payload = builder.build_payload(
        worker="전형기", levels=["10"], sessions=2,
        buckets=_buckets({
            "20250901": {"right": {"wall": spread_stats.accumulate([0.10, 0.12])}},
            "20250904": {"right": {"wall": spread_stats.accumulate([0.14])}},
        }),
    )
    bucket = payload["granularities"]["uv8"][KEY]
    assert bucket["dates"] == ["20250901", "20250904"]
    assert bucket["n"] == 3


def test_pass_axis_values_uses_the_correction_axes() -> None:
    """산포 축은 교정 축과 같아야 sigma로 축을 고를 수 있다."""
    class _Frame:
        normal = (1.0, 0.0, 0.0)
        tangent = (0.0, 1.0, 0.0)
        up = (0.0, 0.0, 1.0)

    tracks = {"right": [(0.3, 0.1, 0.2)] * 4, "head": [(0.5, 0.0, 0.0)] * 4, "left": []}
    values = builder.pass_axis_values(tracks, _Frame())
    assert set(values["right"]) == set(measured_score.AXIS_ORDER)
    assert "left" not in values


def test_pass_axis_values_keeps_axes_below_the_correction_threshold() -> None:
    """산포는 '문장으로 낼 만큼 크다'는 질문이 아니라 σ의 재료다. `_correction_candidates`를
    썼다면 `MIN_CORRECTION_CM`(1cm) 미만인 wall 성분이 후보에서 통째로 빠졌을 것이고,
    그러면 그 패스는 wall 축 산포에 기여하지 못해 산포가 큰 패스 쪽으로만 부풀었을
    것이다. `_axis_magnitudes_cm`은 임계 없이 세 축을 전부 낸다."""
    class _Frame:
        normal = (1.0, 0.0, 0.0)
        tangent = (0.0, 1.0, 0.0)
        up = (0.0, 0.0, 1.0)

    tiny_wall = 0.005  # 0.5cm, measured_score.MIN_CORRECTION_CM(1.0cm) 미만이지만 0은 아니다.
    tracks = {"right": [(tiny_wall, 0.1, 0.2)] * 4, "head": [], "left": []}
    values = builder.pass_axis_values(tracks, _Frame())
    assert set(values["right"]) == set(measured_score.AXIS_ORDER)
    assert math.isclose(values["right"]["wall"], tiny_wall, abs_tol=1e-12)


def test_key_without_uv_replaces_the_cell_component() -> None:
    assert builder.key_without_uv(KEY) == "m1:horizontal:right:na:center-near-high"


def test_thin_bucket_is_kept_in_the_artifact() -> None:
    """n<20이어도 아티팩트에는 남긴다. 폴백 판정은 읽는 쪽이 한다."""
    payload = builder.build_payload(
        worker="전형기", levels=["10"], sessions=1,
        buckets=_buckets({"20250901": {"right": {"wall": spread_stats.accumulate([0.1])}}}),
    )
    assert payload["granularities"]["uv8"][KEY]["n"] == 1
