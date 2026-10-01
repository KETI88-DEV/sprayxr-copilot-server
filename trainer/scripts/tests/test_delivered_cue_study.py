"""전달 안내 한 건 전수 측정의 순수 부분 — 집계가 심은 값을 그대로 내는지 본다.

사전등록: `docs/measurements/2026-10-01-delivered-cue-improvement-prereg.md`
코퍼스는 읽지 않는다(`run_shard`는 CSV가 있어야 돈다).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import delivered_cue_study as dcs  # noqa: E402


def _ok(worker: str, score: float, after: float, after_all: float, split: float | None,
        split_all: float | None, issued: int = 2) -> dict[str, object]:
    return {
        "status": "ok", "worker": worker, "score": score,
        "score_after": after, "score_after_all": after_all,
        "same": (after - score) / (1 - score), "same_all": (after_all - score) / (1 - score),
        "split": split, "split_all": split_all, "issued": issued,
        "delivered_point": "right", "delivered_axis": "vertical",
    }


def test_describe_drops_none_instead_of_counting_it_as_zero() -> None:
    """깨뜨리는 법: None을 0으로 채우면 평균이 0.4로 내려가 실패한다."""
    out = dcs.describe([0.4, None, 0.8])
    assert out["n"] == 2
    assert out["mean"] == pytest.approx(0.6)
    assert out["at_or_above"] == pytest.approx(0.5)


def test_describe_counts_worsening() -> None:
    out = dcs.describe([-0.2, 0.1, 0.9, 0.95])
    assert out["below_zero"] == pytest.approx(0.25)
    assert out["at_or_above"] == pytest.approx(0.5)
    assert out["median"] == pytest.approx(0.5)


def test_aggregate_keeps_the_two_definitions_apart() -> None:
    """1건과 전부는 각자의 열에서만 읽는다.

    깨뜨리는 법: `DEFINITIONS`의 접미사를 둘 다 ""로 두면 all_issued 평균이 1건과 같아져 실패한다.
    """
    records = [
        _ok("A", 0.5, 0.6, 0.9, 0.2, 0.8),
        _ok("B", 0.4, 0.55, 0.85, 0.3, 0.7),
        {"status": "no_expert_reference", "worker": "C"},
    ]
    report = dcs.aggregate(records, expert_worker="E")
    block = report["issued_only"]
    assert block["delivered"]["split"]["mean"] == pytest.approx(0.25)
    assert block["all_issued"]["split"]["mean"] == pytest.approx(0.75)
    assert block["delivered"]["rise"]["mean"] == pytest.approx(0.125)
    assert block["all_issued"]["rise"]["mean"] == pytest.approx(0.425)
    assert report["status_counts"] == {"ok": 2, "no_expert_reference": 1}


def test_pairs_without_a_cue_stay_out_of_the_main_population() -> None:
    """안내가 안 나간 쌍(S1 = S0)은 주 모집단에서 빠지고 `all_ok`에만 들어간다.

    깨뜨리는 법: `issued` 거르기를 지우면 주 평균이 0.2/2 = 0.1로 끌려 실패한다.
    """
    records = [_ok("A", 0.5, 0.6, 0.9, 0.2, 0.8), _ok("B", 0.5, 0.5, 0.5, 0.0, 0.0, issued=0)]
    report = dcs.aggregate(records, expert_worker="E")
    assert report["issued_only"]["delivered"]["split"]["mean"] == pytest.approx(0.2)
    assert report["all_ok"]["delivered"]["split"]["mean"] == pytest.approx(0.1)


def test_expert_self_passes_are_reported_separately() -> None:
    records = [_ok("E", 0.5, 0.6, 0.9, 0.9, 0.9), _ok("A", 0.5, 0.6, 0.9, 0.1, 0.8)]
    report = dcs.aggregate(records, expert_worker="E")
    assert report["excluding_expert_self"]["delivered"]["split"]["mean"] == pytest.approx(0.1)


def test_shard_assignment_is_stable_and_covers_every_shard() -> None:
    """파이썬 `hash`(실행마다 다름)를 쓰면 재실행 때 이미 쓴 줄과 샤드가 어긋난다."""
    keys = [f"W|2025|1|S{i}" for i in range(400)]
    first = [dcs.shard_of(k, 16) for k in keys]
    assert first == [dcs.shard_of(k, 16) for k in keys]
    assert set(first) == set(range(16))
