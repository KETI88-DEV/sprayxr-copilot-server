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

import pytest
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
    _frames, inputs = service.pass_key_inputs(FakeSource().load(PassRef(**query)))
    assert inputs is not None
    key8 = service.key_from(inputs, service.motion_key.KeyParams(uv_grid=8))
    key_off = service.build_spread_reference.key_without_uv(key8)
    return {"uv8": key8, "uv_off": key_off}


_AXIS_CENTERS = {"wall": 0.05, "vertical": 0.06, "lateral": 0.045}
# 대부분의 테스트가 쓰는 균일 표준편차. 바닥(SIGMA_FLOOR_M=0.02 m)과 상한(σ=MAX_SIGMA=6.0)
# 양쪽에서 멀어야 (스케일 전/후) σ의 √2 배율이 클리핑 없이 그대로 보인다.
_UNIFORM_STD = 0.06


def _axis_values(center: float, std: float, count: int) -> list[float]:
    """평균이 정확히 `center`로 고정된 등간격 표본.

    등간격 수열의 모분산은 `d²(n²-1)/12`이므로 원하는 `std`가 나오도록 간격 `d`를
    역산한다 — 평균은 `d`(따라서 std)와 무관하게 항상 `center`다. 표본을 만드는
    쪽에서 평균과 산포를 독립으로 조절할 수 있어야 "산포만 다르고 평균은 같은" A/B
    날짜를 쉽게 만들 수 있다.

    `count < 2`면 분산이 정의되지 않으므로(`spread_stats.summarize`) 간격을 역산하지
    않고 그냥 `center`를 `count`번 반복한다 — F2(반쪽이 표본 2개 미만인 경우)를 만드는
    픽스처가 이 경로를 쓴다.
    """
    if count < 2:
        return [center] * count
    d = std / math.sqrt((count * count - 1) / 12)
    return [center + d * (i - (count - 1) / 2) for i in range(count)]


def _bucket(
    std_by_date: dict[str, float],
    per_date: int = 10,
    count_by_date: dict[str, int] | None = None,
) -> dict[str, object]:
    """날짜마다 다른 표준편차·표본 수를 줄 수 있는 버킷. 세 점(right/head/left) 모두 채운다.

    모든 날짜에 같은 std·표본 수를 주면(`_axis_values`가 결정론이므로) 날짜마다 완전히
    같은 표본이 나온다 — 그래서 이 버킷의 어느 날짜 부분집합을 골라도 평균·산포가
    정확히 같다. `test_split_is_not_backfilled_from_same`은 일부러 날짜마다 다른 std를,
    `test_split_reports_insufficient_samples_for_a_thin_half`는 날짜마다 다른 표본
    수(`count_by_date`)를 줘서 이 성질이 깨지게 만든다.
    """
    counts_for = count_by_date or {date: per_date for date in std_by_date}
    dates = sorted(std_by_date)
    per_date_axes = {
        date: {
            axis: _axis_values(center, std_by_date[date], counts_for[date])
            for axis, center in _AXIS_CENTERS.items()
        }
        for date in dates
    }
    by_date = {
        date: {
            point: {
                axis: spread_stats.to_json(spread_stats.accumulate(values))
                for axis, values in axes.items()
            }
            for point in ("right", "head", "left")
        }
        for date, axes in per_date_axes.items()
    }
    points_out: dict[str, dict[str, dict[str, float]]] = {}
    counts: list[int] = []
    for point in ("right", "head", "left"):
        points_out[point] = {}
        for axis in _AXIS_CENTERS:
            merged: list[float] = []
            for date in dates:
                merged.extend(per_date_axes[date][axis])
            counts.append(len(merged))
            points_out[point][axis] = {
                "mean": sum(merged) / len(merged) if merged else 0.0,
                "spread": spread_stats.summarize(spread_stats.accumulate(merged))[1],
            }
    return {"n": max(counts) if counts else 0, "dates": dates, "by_date": by_date, "points": points_out}


def _payload_from_bucket(worker: str, key8: str, key_off: str, bucket: dict[str, object]) -> dict[str, object]:
    return {
        "version": 1, "worker": worker, "levels": ["10"], "sessions": len(bucket["dates"]), "n_min": 20,
        "granularities": {
            "uv8": {key8: bucket}, "uv4": {}, "uv2": {},
            "uv_off": {key_off: bucket},
        },
    }


def _uniform_payload(
    worker: str, key8: str, key_off: str, dates: list[str], per_date: int = 10
) -> dict[str, object]:
    return _payload_from_bucket(
        worker, key8, key_off, _bucket({date: _UNIFORM_STD for date in dates}, per_date)
    )


# `SpreadReference.split_dates`는 정렬된 날짜를 홀/짝 인덱스로 가른다 — 4개 날짜면
# [0,2]가 A(교정 선택), [1,3]가 B(채점)다. 아래 두 상수가 그 순서를 그대로 반영한다.
_SPLIT_DATES_A = ["20250901", "20250903"]
_SPLIT_DATES_B = ["20250902", "20250904"]
# A(교정 선택)는 CORRECTION_SIGMA_THRESHOLD(2.0)를 넘길 만큼 좁아야 교정이 실제로
# 발동한다. B(채점)는 A보다 훨씬 좁게 잡아, 둘을 섞은 전체 레퍼런스(F, `same`이 쓰는
# 잣대)보다도 더 엄격한 잣대가 되게 한다 — 그래야 같은 교정이라도 B로 재면 F로 잰
# 것(same)보다 낮게 나오는, 자명하지 않은(그리고 조작되지 않은) 부등식이 나온다.
# 값은 `docker exec ... python3`으로 (std_a, std_b) 격자를 스캔해, 바닥(0.02 m)과
# 상한(σ=6.0)을 피하면서 split < same이 되는 조합을 확인해 골랐다 — `service.py`의
# 임계 상수(`CORRECTION_SIGMA_THRESHOLD`/`SIGMA_FLOOR_M`/`MAX_SIGMA`)는 손대지 않았다.
_SPLIT_STD_A = 0.09
_SPLIT_STD_B = 0.03


def _compare_with_reference() -> dict[str, object]:
    """A(0901/0903)는 넓혀(σ가 임계를 겨우 넘는 정도), B(0902/0904)는 훨씬 좁힌
    레퍼런스. `split`(B로만 채점)과 `same`(F로만 채점)이 서로 다른 잣대에서 나오므로
    두 값이 우연히 같아지지 않는다.
    """
    keys = _keys_for(_QUERY)
    std_by_date = {date: _SPLIT_STD_A for date in _SPLIT_DATES_A}
    std_by_date.update({date: _SPLIT_STD_B for date in _SPLIT_DATES_B})
    reference_spreads = SpreadReference(
        _payload_from_bucket(_EXPERT_WORKER, keys["uv8"], keys["uv_off"], _bucket(std_by_date))
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
        _uniform_payload(_EXPERT_WORKER, keys["uv8"], keys["uv_off"], [_QUERY_DATE], per_date=25)
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
        _uniform_payload(_EXPERT_WORKER, keys["uv8"], keys["uv_off"], dates)
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


def test_split_is_not_backfilled_from_same() -> None:
    """`_compare_with_reference()`의 A/B는 표준편차가 서로 다르다(`_SPLIT_STD_A` !=
    `_SPLIT_STD_B`) — 그래서 진짜 분리 채점이면 `split`이 `same`과 우연히도 같아질
    수 없다. `improvement.split`을 `after.score`(same이 쓰는 값)로 되메우는 결함이
    돌아오면 이 assert가 그것을 잡는다 — `split <= same`만으로는 `split == same`(즉
    되메움)을 못 잡는다는 것이 바로 이 테스트가 있는 이유다."""
    body = _compare_with_reference()
    improvement = body["measured"]["improvement"]
    assert improvement["split"] is not None
    assert improvement["split"] != pytest.approx(improvement["same"])


def test_insufficient_dates_are_reported_not_invented() -> None:
    body = _compare_with_single_date_reference()
    assert body["measured"]["improvement"]["split"] is None
    assert body["measured"]["split_reason"] == "insufficient_dates"


def test_query_own_date_is_excluded_from_the_reference() -> None:
    body = _compare_expert_against_own_reference()
    assert _QUERY_DATE not in body["measured"]["reference_dates"]


# --- F1: split의 A/B 반쪽도 질의 날짜를 빼야 한다 --------------------------------
#
# `resolve()`는 exclude_dates를 받지만 `split_dates()`/`spreads_for()`는 Task 4의
# 확정된 계약상 그 인자가 없다. 그래서 필터는 배선 지점(`compare()`)에서 건다 — 브리핑은
# `split_dates(granularity, key)`를 인자 없이 쓰라고 문자 그대로 적었지만, 조정자가
# 그 문구를 이 회차에서 오버라이드했다.


def test_query_own_date_is_excluded_from_the_split_halves_too() -> None:
    """레퍼런스 워커가 질의 자신이면 그 날짜가 F(reference_dates)에서만이 아니라
    split의 A/B 어느 쪽에서도 빠져야 한다. 빠지지 않으면 `reference_dates`가 말하는
    날짜와 실제로 split을 만든 날짜가 갈라진다 — "출처가 아닌 값에 붙은 이름표"다.
    """
    body = _compare_expert_against_own_reference()
    measured = body["measured"]
    assert _QUERY_DATE not in measured["split_dates_a"]
    assert _QUERY_DATE not in measured["split_dates_b"]
    # 5개 날짜(0901~0905) 중 하나(질의 자신)를 뺀 나머지가 A/B로 갈리므로, 이 픽스처는
    # split이 실제로 계산되는 경우다 — 필터링이 조용히 아무것도 안 하는 케이스가 아니다.
    assert measured["split_reason"] is None
    assert measured["improvement"]["split"] is not None
    assert measured["split_n_a"] is not None and measured["split_n_a"] >= 2
    assert measured["split_n_b"] is not None and measured["split_n_b"] >= 2


# --- F2: 반쪽이 표본 2개 미만이면 보고하되 게이트하지 않는다 ------------------------


def test_split_reports_insufficient_samples_for_a_thin_half() -> None:
    """`N_MIN`(20)은 resolve()가 쓰는 사전등록 문턱이지 반쪽의 문턱이 아니다 — F2 판정.
    유일한 문턱은 수학적 필요조건(표본 2개 미만이면 분산이 정의되지 않는다)이다.

    A(0901/0903)에 표본 1개짜리 날짜를 몰아넣고 질의 자신이 0901이면, 제외 후 A에는
    0903(표본 1개)만 남는다 — 분산이 없다. B(0902/0904)는 표본이 넉넉해 정상이다.
    F(전체 31표본, 0901 제외)는 `N_MIN`을 넉넉히 넘으므로 normalization은 여전히
    measured다 — 반쪽이 얇다고 전체 측정 자체가 잠정으로 떨어지지 않는다.
    """
    dates = ["20250901", "20250902", "20250903", "20250904"]
    std_by_date = {date: _UNIFORM_STD for date in dates}
    count_by_date = {"20250901": 1, "20250902": 15, "20250903": 1, "20250904": 15}
    keys = _keys_for(_EXPERT_QUERY)
    reference_spreads = SpreadReference(
        _payload_from_bucket(
            _EXPERT_WORKER, keys["uv8"], keys["uv_off"],
            _bucket(std_by_date, count_by_date=count_by_date),
        )
    )
    body = service.compare(
        FakeSource(), PassRef(**_EXPERT_QUERY), PassRef(**REFERENCE),
        reference_spreads=reference_spreads,
    )
    measured = body["measured"]
    assert measured["normalization"] == "measured"
    assert measured["improvement"]["split"] is None
    assert measured["split_reason"] == "insufficient_samples"
    # 반쪽은 버려지지 않고 실제로 걸러진 날짜·표본 수를 그대로 낸다.
    assert measured["split_dates_a"] == ["20250903"]
    assert measured["split_n_a"] == 1
    assert measured["split_dates_b"] == ["20250902", "20250904"]
    assert measured["split_n_b"] == 30


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
        _uniform_payload(_EXPERT_WORKER, keys["uv8"], keys["uv_off"], dates)
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
    payload = _uniform_payload(_EXPERT_WORKER, keys["uv8"], keys["uv_off"], dates)
    artifact_path = tmp_path / "spread_reference.json"
    artifact_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv(SPREAD_REFERENCE_ENV, str(artifact_path))

    client = TestClient(create_app(source=FakeSource()))
    response = client.post("/api/compare", json={"query": _QUERY, "reference": REFERENCE})
    body = response.json()
    assert body["measured"]["normalization"] == "measured"


def test_create_app_refuses_an_artifact_built_with_an_old_key_prefix(tmp_path, monkeypatch, caplog) -> None:
    """D1(m1→m3) 뒤 옛 접두어 아티팩트를 주면 경고하고 잠정 산포로 돈다 — 섞어 쓰지 않는다.

    대조군: 옛 접두어 버킷을 넣지 않은 같은 페이로드는 실측으로 돈다. 현행 키 버킷은 두 쪽이
    같으니 결과를 가른 것은 접두어 검사뿐이다.
    """
    import json

    dates = ["20250901", "20250902"]
    keys = _keys_for(_QUERY)
    prefix = service.motion_key.KEY_PREFIX
    assert keys["uv8"].startswith(f"{prefix}:")
    assert prefix != "m1"
    results = {}
    for label in ("current", "mixed"):
        artifact_path = tmp_path / f"spread_reference_{label}.json"
        payload = _uniform_payload(_EXPERT_WORKER, keys["uv8"], keys["uv_off"], dates)
        if label == "mixed":
            # 현행 키 버킷은 그대로 두고 옛 접두어 버킷을 한 벌 더 넣는다. 검사가 없으면 현행 키로
            # 조회가 성공해 실측으로 돈다 — 옛 산출물이 섞인 아티팩트가 조용히 통과하는 경우다.
            for layer in payload["granularities"].values():
                for key in list(layer):
                    layer["m1" + key[len(prefix):]] = layer[key]
        artifact_path.write_text(json.dumps(payload), encoding="utf-8")
        monkeypatch.setenv(SPREAD_REFERENCE_ENV, str(artifact_path))
        caplog.clear()
        client = TestClient(create_app(source=FakeSource()))
        body = client.post("/api/compare", json={"query": _QUERY, "reference": REFERENCE}).json()
        results[label] = (body["measured"]["normalization"], caplog.text)
    assert results["current"][0] == "measured"
    assert results["mixed"][0] == "provisional", "옛 접두어 버킷이 섞인 아티팩트가 조용히 통과했다"
    assert "접두어" in results["mixed"][1]


def test_create_app_without_the_env_var_stays_provisional(monkeypatch) -> None:
    monkeypatch.delenv(SPREAD_REFERENCE_ENV, raising=False)
    client = TestClient(create_app(source=FakeSource()))
    response = client.post("/api/compare", json={"query": _QUERY, "reference": REFERENCE})
    body = response.json()
    assert body["measured"]["normalization"] == "provisional"


# --- F3: axis_sigmas가 응답에 실린다 -----------------------------------------------


def test_deltas_carry_per_axis_sigma_consistent_with_the_combined_sigma() -> None:
    """Task 6은 점당 막대 옆에 축별(wall/vertical/lateral) σ가 필요하다.
    `measured_score.PointDelta.axis_sigmas`는 이미 계산돼 있었지만 응답에 실리지
    않았다. 세 축 이름이 모두 있는지, 그리고 결합 σ가 그 세 값을 실제로 어떻게
    합치는지(제곱합의 제곱근, `MAX_SIGMA`에서 클립)까지 확인한다 — 근사가 아니라
    `measured_score.score_track`이 실제로 쓰는 식 그대로다.
    """
    body = _compare_with_reference()
    axis_order = set(service.measured_score.AXIS_ORDER)
    deltas = body["measured"]["deltas"]
    assert deltas
    for delta in deltas:
        axis_sigmas = delta["axis_sigmas"]
        assert set(axis_sigmas) == axis_order
        combined = math.sqrt(sum(value * value for value in axis_sigmas.values()))
        expected = min(service.measured_score.MAX_SIGMA, combined)
        assert math.isclose(delta["sigma"], expected, rel_tol=1e-9, abs_tol=1e-9)


# --- F4: 경로는 있는데 파일이 없으면 조용히 넘어가지 않는다 ------------------------


def test_missing_artifact_path_logs_a_warning_naming_the_path(tmp_path, monkeypatch, caplog) -> None:
    missing = tmp_path / "does_not_exist.json"
    monkeypatch.setenv(SPREAD_REFERENCE_ENV, str(missing))
    import logging as _logging

    with caplog.at_level(_logging.WARNING):
        client = TestClient(create_app(source=FakeSource()))
        response = client.post("/api/compare", json={"query": _QUERY, "reference": REFERENCE})
    assert response.json()["measured"]["normalization"] == "provisional"
    assert any(str(missing) in record.message for record in caplog.records)


def test_unreadable_artifact_path_logs_a_warning_naming_the_path(tmp_path, monkeypatch, caplog) -> None:
    """파일이 있어도 JSON이 아니면(깨진 아티팩트) 폴백은 유지하되 조용히 넘어가지 않는다."""
    broken = tmp_path / "spread_reference.json"
    broken.write_text("이건 JSON이 아니다", encoding="utf-8")
    monkeypatch.setenv(SPREAD_REFERENCE_ENV, str(broken))
    import logging as _logging

    with caplog.at_level(_logging.WARNING):
        client = TestClient(create_app(source=FakeSource()))
        response = client.post("/api/compare", json={"query": _QUERY, "reference": REFERENCE})
    assert response.json()["measured"]["normalization"] == "provisional"
    assert any(str(broken) in record.message for record in caplog.records)


# --- 정리: split_reason이 normalization과 모순되지 않는다 --------------------------


def test_split_reason_does_not_contradict_a_measured_normalization() -> None:
    """호출자가 `spreads=`를 직접 넘기면 normalization은 measured지만 날짜별 구조가
    없어 split은 여전히 못 낸다. 그 사유가 gate_reason이 쓰는
    "no_reference_distribution"(=측정이 아니다)이면 같은 응답 안에서 "측정이다"와
    "측정이 아니다"가 동시에 적히는 모순이 생긴다.
    """
    body = service.compare(
        FakeSource(), PassRef(**_QUERY), PassRef(**REFERENCE),
        spreads=service.measured_score.uniform_spreads(0.06),
    )
    measured = body["measured"]
    assert measured["normalization"] == "measured"
    assert measured["improvement"]["split"] is None
    assert measured["split_reason"] != "no_reference_distribution"
    assert measured["split_reason"] == "no_split_reference"


# --- 정리: _reference_keys와 build_from_root의 키-굽기가 갈라지면 시끄럽게 -----------


@pytest.fixture
def _keybake_root(tmp_path):
    from trainer.pose_lab.tests.test_csv_source import _session_rows, _write

    nested = tmp_path / "전형기" / "전형기_20250901" / "전형기_20250901(Sequences)"
    _write(nested / "10_1000451_1.csv", _session_rows(shift=0.3))
    return tmp_path


def test_reference_keys_matches_build_from_root_for_the_same_pass(_keybake_root) -> None:
    """`service._reference_keys`와 `build_spread_reference.build_from_root`는 같은
    사다리(우8/우4/우2/우off) 키를 만드는 루프를 따로 갖고 있다(중복 제거는 이번
    범위가 아니다). 둘이 갈라지면 `resolve()`가 조용히 `None`을 내고 응답은
    `provisional`로 떨어진다 — 화면은 계속 뜨고 라벨만 조용히 틀려지는, 가장 나쁜
    실패다. 실제 CSV 트리에서 만든 아티팩트가 굽는 키 집합과 `_reference_keys`가
    같은 패스에 대해 내는 키를 직접 맞대본다.
    """
    from trainer.pose_lab import csv_source

    payload = service.build_spread_reference.build_from_root(_keybake_root, "전형기", ["10"])
    granularities = payload["granularities"]

    source = csv_source.CsvPassSource(_keybake_root)
    session = next(item for item in source.sessions() if item.worker == "전형기")
    entries = source.index_of(session)
    inputs = None
    for entry in entries:
        ref = PassRef(
            worker=session.worker, date=session.date, level=session.level,
            session=session.session, pass_index=entry.pass_index,
        )
        _frames, candidate = service.pass_key_inputs(source.load(ref))
        if candidate is not None:
            inputs = candidate
            break
    assert inputs is not None, "빌드가 잰 세션에는 최소 하나의 유효한 패스가 있어야 한다"

    keys = service._reference_keys(inputs)
    for layer in ("uv8", "uv4", "uv2", "uv_off"):
        assert keys[layer] in granularities[layer], (
            f"{layer}: _reference_keys가 낸 {keys[layer]!r}가 build_from_root가 구운 "
            f"키 집합 {sorted(granularities[layer])!r}에 없다 — 두 키-굽기 루프가 갈라졌다"
        )
