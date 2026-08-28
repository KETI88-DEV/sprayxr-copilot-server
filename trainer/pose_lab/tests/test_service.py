#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import math

from trainer.pose_lab import service

pose_similarity = service.pose_similarity


def test_gate_verdict_follows_the_preregistered_threshold() -> None:
    """REQ-19 correction_target_gate. 납품 후 수치를 보고 기준을 옮기지 않는다."""
    assert service.gate_verdict(spread=0.06, error=0.144) == "pass"
    assert service.gate_verdict(spread=0.149, error=0.144) == "fail"
    assert service.gate_verdict(spread=0.072, error=0.144) == "fail"


def test_gate_is_none_without_an_error_scale() -> None:
    assert service.gate_verdict(spread=0.06, error=0.0) is None


def test_improvement_reports_split_and_same_reference() -> None:
    """교정에 쓴 분포로 다시 채점하면 향상이 자동 보장된다. 둘 다 보여야 한다."""
    result = service.improvement(before=0.4, after_split=0.55, after_same=0.9)
    assert math.isclose(result["split"], 0.25, abs_tol=1e-9)
    assert math.isclose(result["same"], 0.8333333, abs_tol=1e-6)
    assert result["split"] != result["same"]


def test_improvement_leaves_split_none_when_not_computed() -> None:
    """held-out 레퍼런스가 없으면 분리 채점 값을 지어내지 않는다."""
    result = service.improvement(before=0.4, after_split=None, after_same=0.9)
    assert result["split"] is None
    assert math.isclose(result["same"], 0.8333333, abs_tol=1e-6)


def test_undefined_gap_stays_none_and_does_not_become_zero() -> None:
    """이미 기준과 같으면 메울 격차가 없어 개선율이 정의되지 않는다.

    0.000으로 적으면 "메우지 못했다"로 읽힌다. pose_similarity.gap_closure는 그래서
    None을 내는데, 그것을 뒤에서 0.0으로 바꾸면 의도적인 침묵이 무너진다.
    """
    assert pose_similarity.gap_closure(1.0, 1.0) is None
    result = service.improvement(before=1.0, after_split=1.0, after_same=1.0)
    assert result["split"] is None
    assert result["same"] is None
