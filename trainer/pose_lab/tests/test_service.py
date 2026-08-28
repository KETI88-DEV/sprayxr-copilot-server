#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import math

from trainer.pose_lab import service


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
