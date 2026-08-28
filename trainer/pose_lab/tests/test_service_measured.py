#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false
"""실측 산포 배선(compare(reference_spreads=...))의 회귀.

레퍼런스 아티팩트는 `trainer/pose_lab/tests/test_spread_reference.py`가 이미 도맡아
검증했다(resolve/split_dates/spreads_for). 여기서는 그 결과가 `service.compare`에
실제로 배선됐는지, 즉 (1) normalization이 measured로 뒤집히고 게이트가 실제로
판정되는지, (2) 응답이 어느 층·표본 몇 개인지 말하는지, (3) split이 진짜 disjoint
날짜에서 나오는지(또는 정직하게 null인지)만 본다.

`FakeSource`는 `test_app.py`에서 그대로 가져다 쓴다 — 복제하지 않는다.
"""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

spread_stats = importlib.import_module("spread_stats")

from fastapi.testclient import TestClient

from trainer.pose_lab import service
from trainer.pose_lab.app import SPREAD_REFERENCE_ENV, create_app
from trainer.pose_lab.service import PassRef
from trainer.pose_lab.spread_reference import SpreadReference
from trainer.pose_lab.tests.test_app import REFERENCE, FakeSource

_QUERY_DATE = "20250901"
# 레퍼런스 워커와 다른 작업자 — 자기 배제 로직이 끼어들지 않는 "보통" 질의.
_QUERY = {"worker": "A", "date": _QUERY_DATE, "level": "10", "session": "s1", "pass_index": 0}
# 레퍼런스 워커와 같은 작업자 — 질의 날짜가 레퍼런스에서 빠져야 하는 경우.
_EXPERT_WORKER = "전형기"
_EXPERT_QUERY = {
    "worker": _EXPERT_WORKER, "date": _QUERY_DATE, "level": "10", "session": "s1",
    "pass_index": 0,
}


def _keys_for(query: dict[str, object]) -> dict[str, str]:
    """`compare()`가 내부에서 뽑을 것과 같은 우8/우off 키를 미리 계산한다.

    `service.motion_key.KeyParams()` 기본값(uv_grid=8)이 `compare()`가 쓰는 기본
    `params`와 같으므로, 이 키가 곧 응답의 `query_key`(우8 층)와 같다.
    """
    frames = service.pass_frames(FakeSource().load(PassRef(**query)))
    inputs = service.key_inputs(frames)
    assert inputs is not None
    key8 = service.key_from(inputs, service.motion_key.KeyParams(uv_grid=8))
    key_off = service.build_spread_reference.key_without_uv(key8)
    return {"uv8": key8, "uv_off": key_off}


def _axis_values(base: float, count: int) -> list[float]:
    # 간격을 0.02로 벌린다 — 0.001이면 표준편차가 0.003 m 근처로 너무 얇아서 실제
    # 이탈(수십 cm)과 나누면 MAX_SIGMA(6.0)에서 양쪽(스케일 전/후)이 다 잘려 √2 배율이
    # 관측되지 않는다.
    return [base + 0.02 * step for step in range(count)]


def _full_bucket(dates: list[str], per_date: int = 10) -> dict[str, object]:
    """모든 날짜에 같은 표본을 반복해서 싣는다. 세 점(right/head/left) 모두 채운다.

    반복해도 평균·산포(모표준편차)는 바뀌지 않는다 — 합/표본수 비가 그대로이기
    때문이다. 그래서 이 버킷의 어느 날짜 부분집합(A, B, 전체)을 골라도 산포가 정확히
    같은 값이 나오고, `test_split_never_exceeds_same`이 부동소수 오차 안에서
    등식으로(따라서 결정론적으로 `<=`를 만족하며) 성립한다.
    """
    per_point_axes = {
        "right": {
            "wall": _axis_values(0.05, per_date),
            "vertical": _axis_values(0.06, per_date),
            "lateral": _axis_values(0.045, per_date),
        },
        "head": {
            "wall": _axis_values(0.05, per_date),
            "vertical": _axis_values(0.06, per_date),
            "lateral": _axis_values(0.045, per_date),
        },
        "left": {
            "wall": _axis_values(0.05, per_date),
            "vertical": _axis_values(0.06, per_date),
            "lateral": _axis_values(0.045, per_date),
        },
    }
    by_date = {
        date: {
            point: {
                axis: spread_stats.to_json(spread_stats.accumulate(values))
                for axis, values in axes.items()
            }
            for point, axes in per_point_axes.items()
        }
        for date in dates
    }
    points_out: dict[str, dict[str, dict[str, float]]] = {}
    counts: list[int] = []
    for point, axes in per_point_axes.items():
        points_out[point] = {}
        for axis, values in axes.items():
            merged = values * len(dates)
            counts.append(len(merged))
            points_out[point][axis] = {
                "mean": sum(merged) / len(merged),
                "spread": spread_stats.summarize(spread_stats.accumulate(merged))[1],
            }
    return {
        "n": max(counts),
        "dates": sorted(dates),
        "by_date": by_date,
        "points": points_out,
    }


def _payload(worker: str, key8: str, key_off: str, dates: list[str], per_date: int = 10) -> dict[str, object]:
    return {
        "version": 1, "worker": worker, "levels": ["10"], "sessions": len(dates), "n_min": 20,
        "granularities": {
            "uv8": {key8: _full_bucket(dates, per_date)},
            "uv4": {}, "uv2": {},
            "uv_off": {key_off: _full_bucket(dates, per_date)},
        },
    }


def _compare_with_reference() -> dict[str, object]:
    dates = ["20250901", "20250902", "20250903", "20250904"]
    keys = _keys_for(_QUERY)
    reference_spreads = SpreadReference(
        _payload(_EXPERT_WORKER, keys["uv8"], keys["uv_off"], dates)
    )
    return service.compare(
        FakeSource(),
        PassRef(**_QUERY),
        PassRef(**REFERENCE),
        reference_spreads=reference_spreads,
    )


def _compare_with_single_date_reference() -> dict[str, object]:
    keys = _keys_for(_QUERY)
    reference_spreads = SpreadReference(
        _payload(_EXPERT_WORKER, keys["uv8"], keys["uv_off"], [_QUERY_DATE], per_date=25)
    )
    return service.compare(
        FakeSource(),
        PassRef(**_QUERY),
        PassRef(**REFERENCE),
        reference_spreads=reference_spreads,
    )


def _compare_expert_against_own_reference() -> dict[str, object]:
    dates = ["20250901", "20250902", "20250903", "20250904", "20250905"]
    keys = _keys_for(_EXPERT_QUERY)
    reference_spreads = SpreadReference(
        _payload(_EXPERT_WORKER, keys["uv8"], keys["uv_off"], dates)
    )
    return service.compare(
        FakeSource(),
        PassRef(**_EXPERT_QUERY),
        PassRef(**REFERENCE),
        reference_spreads=reference_spreads,
    )


def _compare_without_reference() -> dict[str, object]:
    return service.compare(FakeSource(), PassRef(**_QUERY), PassRef(**REFERENCE))


# --- 브리핑 문서의 테스트(자구 그대로) ---------------------------------------------


def test_measured_spreads_flip_normalization_and_enable_the_gate() -> None:
    body = _compare_with_reference()
    assert body["measured"]["normalization"] == "measured"
    assert body["measured"]["gate"] in {"pass", "fail"}
    assert body["measured"].get("gate_reason") is None


def test_response_says_which_layer_and_how_many_samples() -> None:
    body = _compare_with_reference()
    assert body["measured"]["granularity"] in {"uv8", "uv4", "uv2", "uv_off"}
    assert isinstance(body["measured"]["reference_n"], int)
    assert body["measured"]["reference_n"] >= 20
    assert body["measured"]["reference_dates"]


def test_split_is_a_real_number_or_an_explained_null() -> None:
    body = _compare_with_reference()
    split = body["measured"]["improvement"]["split"]
    if split is None:
        assert body["measured"]["split_reason"] == "insufficient_dates"
    else:
        assert isinstance(split, float)


def test_split_never_exceeds_same() -> None:
    """교정에 쓴 분포로 다시 채점하면 향상이 정의상 보장된다. split이 그보다 크면 버그다."""
    body = _compare_with_reference()
    improvement = body["measured"]["improvement"]
    if improvement["split"] is not None:
        assert improvement["split"] <= improvement["same"] + 1e-9


def test_insufficient_dates_are_reported_not_invented() -> None:
    body = _compare_with_single_date_reference()
    assert body["measured"]["improvement"]["split"] is None
    assert body["measured"]["split_reason"] == "insufficient_dates"


def test_query_own_date_is_excluded_from_the_reference() -> None:
    body = _compare_expert_against_own_reference()
    assert _QUERY_DATE not in body["measured"]["reference_dates"]


def test_without_a_reference_it_stays_provisional() -> None:
    body = _compare_without_reference()
    assert body["measured"]["normalization"] == "provisional"
    assert body["measured"]["gate"] is None
    assert body["measured"]["gate_reason"] == "no_reference_distribution"


def test_joint_axis_still_borrows_nothing() -> None:
    body = _compare_with_reference()
    assert body["joint"]["status"] == "not_wired"
    assert body["joint"]["score"] is None
    assert body["joint"]["improvement"] is None


# --- 조정자 지시: √2 스케일 -------------------------------------------------------


def test_reference_spreads_reach_the_scorer_scaled_by_sqrt2() -> None:
    """레퍼런스는 버킷 평균이 아니라 한 패스다. 질의-레퍼런스 차의 분산은 2σ²이므로
    σ에 √2를 곱해야 score_track이 보는 산포가 맞는다(서비스 배선, `measured_score`
    아님). 같은 산포를 직접 넘긴 경로(스케일 없음)와 레퍼런스를 거친 경로(스케일 있음)
    의 σ를 견주어 √2배 관계를 못 박는다."""
    dates = ["20250901", "20250902"]
    keys = _keys_for(_QUERY)
    reference_spreads = SpreadReference(
        _payload(_EXPERT_WORKER, keys["uv8"], keys["uv_off"], dates)
    )
    resolved = reference_spreads.resolve({"uv8": keys["uv8"]})
    assert resolved is not None

    via_reference = service.compare(
        FakeSource(), PassRef(**_QUERY), PassRef(**REFERENCE),
        reference_spreads=reference_spreads,
    )
    via_raw = service.compare(
        FakeSource(), PassRef(**_QUERY), PassRef(**REFERENCE),
        spreads=resolved.spreads,
    )

    scaled = {d["point"]: d["sigma"] for d in via_reference["measured"]["deltas"]}
    raw = {d["point"]: d["sigma"] for d in via_raw["measured"]["deltas"]}
    checked = 0
    for point, raw_sigma in raw.items():
        scaled_sigma = scaled[point]
        # 바닥(SIGMA_FLOOR_M)이나 상한(MAX_SIGMA)에 걸리면 배율이 깨끗하게 안 보인다.
        if raw_sigma <= 0.0 or math.isclose(raw_sigma, service.measured_score.MAX_SIGMA):
            continue
        assert math.isclose(raw_sigma, scaled_sigma * math.sqrt(2.0), rel_tol=1e-6)
        checked += 1
    assert checked > 0, "적어도 한 점은 바닥/상한 없이 배율을 검증해야 이 테스트가 뜻을 갖는다"


def test_create_app_reads_the_spread_reference_env_var(tmp_path, monkeypatch) -> None:
    """`POSE_LAB_SPREAD_REFERENCE`가 있으면 HTTP 경로도 실측 산포로 돈다."""
    import json

    dates = ["20250901", "20250902"]
    keys = _keys_for(_QUERY)
    payload = _payload(_EXPERT_WORKER, keys["uv8"], keys["uv_off"], dates)
    artifact_path = tmp_path / "spread_reference.json"
    artifact_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv(SPREAD_REFERENCE_ENV, str(artifact_path))

    client = TestClient(create_app(source=FakeSource()))
    response = client.post("/api/compare", json={"query": _QUERY, "reference": REFERENCE})
    body = response.json()
    assert body["measured"]["normalization"] == "measured"


def test_create_app_without_the_env_var_stays_provisional(monkeypatch) -> None:
    monkeypatch.delenv(SPREAD_REFERENCE_ENV, raising=False)
    client = TestClient(create_app(source=FakeSource()))
    response = client.post("/api/compare", json={"query": _QUERY, "reference": REFERENCE})
    body = response.json()
    assert body["measured"]["normalization"] == "provisional"
