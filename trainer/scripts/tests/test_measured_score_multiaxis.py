#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

measured_score = importlib.import_module("measured_score")
stroke_frame = importlib.import_module("stroke_frame")

# 수평 스트로크: 접선이 월드 Y라 up이 정확히 월드 Z이고 normal이 월드 X다.
HORIZONTAL = stroke_frame.stroke_basis(
    hit=(1.2, 4.5, 0.8), tangent=(0.0, 1.0, 0.0), gun=(0.6, 4.5, 0.8)
)
T = measured_score.CORRECTION_SIGMA_THRESHOLD


def _local(basis: object, world: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(
        sum(world[i] * getattr(basis, name)[i] for i in range(3))
        for name in ("normal", "tangent", "up")
    )


def _right(world: tuple[float, float, float], axis_sigmas: dict[str, float] | None) -> object:
    return measured_score.PointDelta(
        point="right",
        sigma=3.0,
        local_delta=_local(HORIZONTAL, world),
        samples=30,
        axis_sigmas=axis_sigmas or {},
    )


def test_every_axis_over_the_threshold_is_issued() -> None:
    """축별 sigma가 임계를 넘는 축은 전부 나간다. 하나만 내면 개선 여지를 버린다."""
    delta = _right(
        (0.08, 0.05, -0.06),  # wall +8cm, lateral +5cm, vertical -6cm
        {"wall": 3.5, "vertical": 2.6, "lateral": 2.2},
    )
    issued = measured_score.right_axes(delta, HORIZONTAL)
    assert {c.axis for c in issued} == {"wall", "vertical", "lateral"}
    assert all(c.point == "right" for c in issued)


def test_axes_below_the_threshold_are_not_issued() -> None:
    """sigma가 임계 아래인 축은 이상하지 않다. 지시하면 지어낸 교정이다."""
    delta = _right(
        (0.08, 0.05, -0.06),
        {"wall": 3.5, "vertical": 1.1, "lateral": 0.4},
    )
    issued = measured_score.right_axes(delta, HORIZONTAL)
    assert {c.axis for c in issued} == {"wall"}


def test_axes_below_the_minimum_centimetre_are_not_issued() -> None:
    """sigma가 커도 5mm를 옮기라는 지시는 사람이 따를 수 없다."""
    delta = _right(
        (0.08, 0.005, 0.0),  # lateral 0.5cm — MIN_CORRECTION_CM(1.0) 미만
        {"wall": 3.5, "vertical": 3.0, "lateral": 3.0},
    )
    issued = measured_score.right_axes(delta, HORIZONTAL)
    assert "lateral" not in {c.axis for c in issued}
    assert "wall" in {c.axis for c in issued}


def test_the_sentence_axis_is_issued_even_when_its_own_sigma_is_low() -> None:
    """단조 보장. 결합 sigma가 임계를 넘는데 개별 축이 아무도 못 넘는 경우가 있다.

    이 보장이 없으면 오늘 나가던 교정이 새 코드에서 사라진다.
    """
    delta = _right(
        (0.08, 0.0, 0.0),
        {"wall": 1.4, "vertical": 1.3, "lateral": 1.2},  # 아무도 2.0 미만
    )
    issued = measured_score.right_axes(delta, HORIZONTAL)
    assert [c.axis for c in issued] == ["wall"]


def test_no_axis_sigmas_falls_back_to_the_single_sentence_axis() -> None:
    """축별 sigma 없이 만든 구조체는 예전 동작 그대로다(옛 호출자 보호)."""
    delta = _right((0.08, 0.05, -0.06), None)
    issued = measured_score.right_axes(delta, HORIZONTAL)
    assert len(issued) == 1


def test_below_the_point_gate_nothing_is_issued() -> None:
    """점 게이트는 그대로다. 오른손이 이상하지 않으면 아무 말도 하지 않는다."""
    delta = measured_score.PointDelta(
        point="right",
        sigma=T,  # 초과가 아니라 같음 — 열리지 않아야 한다
        local_delta=_local(HORIZONTAL, (0.08, 0.05, -0.06)),
        samples=30,
        axis_sigmas={"wall": 3.5, "vertical": 2.6, "lateral": 2.2},
    )
    assert measured_score.right_axes(delta, HORIZONTAL) == []


def test_magnitude_and_unit_agree_so_the_sentence_cannot_drift() -> None:
    """cm와 단위벡터가 같은 곳에서 나온다. 어긋나면 문장과 시뮬레이션이 갈라진다.

    브리프 초안은 `HORIZONTAL.normal`이 월드 +X라고 가정했지만 이 hit/gun 조합에서
    실제로는 월드 -X다(작업자가 -X쪽에 있으므로 `stroke_basis`가 법선을 그쪽으로
    뒤집는다 — `stroke_frame.stroke_basis` 참고). 그래서 `magnitude_cm`의 부호도
    -8.0이 맞다. 부호 자체를 어느 쪽으로 고정할지는 이 테스트의 관심사가 아니다 —
    검증할 불변은 "magnitude_cm/100 × unit이 원래 월드 이동을 정확히 되살린다"는
    것이므로 그 재구성을 직접 확인한다.
    """
    delta = _right((0.08, 0.0, 0.0), {"wall": 3.5, "vertical": 0.1, "lateral": 0.1})
    (wall,) = measured_score.right_axes(delta, HORIZONTAL)
    assert wall.axis == "wall"
    # HORIZONTAL의 normal은 월드 -X다(위 설명 참고).
    assert abs(wall.unit[0]) > 0.99
    reconstructed_world_x = wall.magnitude_cm / 100.0 * wall.unit[0]
    assert abs(reconstructed_world_x - 0.08) < 0.005


def _head(world: tuple[float, float, float], axis_sigmas: dict[str, float]) -> object:
    return measured_score.PointDelta(
        point="head",
        sigma=3.0,
        local_delta=_local(HORIZONTAL, world),
        samples=30,
        axis_sigmas=axis_sigmas,
    )


def test_head_height_becomes_a_stance_sentence() -> None:
    delta = _head((0.0, 0.0, -0.12), {"wall": 0.2, "vertical": 3.1, "lateral": 0.1})
    stance = measured_score.head_stance(delta, HORIZONTAL)
    assert stance is not None
    assert stance.point == "head" and stance.axis == "vertical"
    text = measured_score.stance_text(stance)
    assert "낮춰" in text


def test_head_wall_distance_becomes_a_step_back_sentence() -> None:
    delta = _head((0.18, 0.0, 0.0), {"wall": 3.4, "vertical": 0.2, "lateral": 0.1})
    stance = measured_score.head_stance(delta, HORIZONTAL)
    assert stance is not None and stance.axis == "wall"
    assert "걸음" in measured_score.stance_text(stance)


def test_only_one_stance_sentence_even_when_both_axes_fire() -> None:
    """자세 문장도 하나다. 두 가지를 동시에 시키면 아무것도 안 한다."""
    delta = _head((0.18, 0.0, -0.12), {"wall": 3.4, "vertical": 2.9, "lateral": 0.1})
    stance = measured_score.head_stance(delta, HORIZONTAL)
    assert stance is not None and stance.axis == "wall"  # sigma가 큰 쪽


def test_head_lateral_is_never_issued() -> None:
    """작업자 요가 실측이 아니라 좌우를 판정할 기준이 없다(service.ASSUMED_YAW)."""
    delta = _head((0.0, 0.20, 0.0), {"wall": 0.1, "vertical": 0.1, "lateral": 5.0})
    assert measured_score.head_stance(delta, HORIZONTAL) is None


def test_head_below_the_threshold_says_nothing() -> None:
    delta = _head((0.0, 0.0, -0.12), {"wall": 0.2, "vertical": 1.9, "lateral": 0.1})
    assert measured_score.head_stance(delta, HORIZONTAL) is None


def test_stance_text_never_claims_a_posture_we_cannot_observe() -> None:
    """HMD로는 앉음과 쪼그림과 무릎 굽힘을 구별할 수 없다. 그러니 '앉아'라고 쓰지 않는다."""
    for world, sigmas in (
        ((0.0, 0.0, -0.12), {"vertical": 3.1}),
        ((0.0, 0.0, 0.12), {"vertical": 3.1}),
        ((0.18, 0.0, 0.0), {"wall": 3.4}),
        ((-0.18, 0.0, 0.0), {"wall": 3.4}),
    ):
        stance = measured_score.head_stance(_head(world, sigmas), HORIZONTAL)
        assert stance is not None
        assert "앉아" not in measured_score.stance_text(stance)


def test_issued_axes_are_sorted_by_sigma_descending() -> None:
    """읽는 사람이 위에서부터 심각한 순으로 본다."""
    right = _right((0.08, 0.05, -0.06), {"wall": 2.2, "vertical": 3.9, "lateral": 2.6})
    head = _head((0.18, 0.0, 0.0), {"wall": 3.4, "vertical": 0.2, "lateral": 0.1})
    issued = measured_score.issued_axes([right, head], HORIZONTAL)
    sigmas = [c.sigma for c in issued]
    assert sigmas == sorted(sigmas, reverse=True)


def test_left_hand_is_never_issued() -> None:
    """왼손은 채점에만 남는다. 작업 결과와의 인과가 약하다."""
    left = measured_score.PointDelta(
        point="left",
        sigma=5.0,
        local_delta=_local(HORIZONTAL, (0.30, 0.30, 0.30)),
        samples=30,
        axis_sigmas={"wall": 5.0, "vertical": 5.0, "lateral": 5.0},
    )
    assert measured_score.issued_axes([left], HORIZONTAL) == []
    assert measured_score.issued_shifts([left], HORIZONTAL) == {}


def test_shifts_contain_only_points_that_were_issued() -> None:
    """지시하지 않은 점은 키가 없다. 없으면 apply_measured가 옮길 방법이 없다."""
    right = _right((0.08, 0.0, 0.0), {"wall": 3.5, "vertical": 0.1, "lateral": 0.1})
    quiet_head = _head((0.0, 0.0, -0.01), {"wall": 0.1, "vertical": 0.1, "lateral": 0.1})
    shifts = measured_score.issued_shifts([right, quiet_head], HORIZONTAL)
    assert set(shifts) == {"right"}


def test_apply_measured_moves_exactly_the_issued_axes_and_nothing_else() -> None:
    """불변식. 발부한 축의 합만큼만 옮긴다."""
    right = _right((0.08, 0.0, -0.06), {"wall": 3.5, "vertical": 2.6, "lateral": 0.1})
    query = {
        "right": [(0.0, 0.0, 0.0)] * 3,
        "head": [(0.0, 0.0, 0.0)] * 3,
        "left": [(0.0, 0.0, 0.0)] * 3,
    }
    fixed = measured_score.apply_measured(query, [right], HORIZONTAL)
    assert fixed["head"] == query["head"]
    assert fixed["left"] == query["left"]
    expected = measured_score.issued_shifts([right], HORIZONTAL)["right"]
    moved = tuple(fixed["right"][0][i] - query["right"][0][i] for i in range(3))
    for got, want in zip(moved, expected):
        assert abs(got - want) < 1e-9


def test_head_moves_only_when_a_stance_sentence_was_issued() -> None:
    """자세 문장이 나가면 머리도 옮긴다 — 말한 만큼만."""
    head = _head((0.18, 0.0, 0.0), {"wall": 3.4, "vertical": 0.2, "lateral": 0.1})
    query = {name: [(0.0, 0.0, 0.0)] * 3 for name in ("right", "head", "left")}
    fixed = measured_score.apply_measured(query, [head], HORIZONTAL)
    assert fixed["head"] != query["head"]
    assert fixed["left"] == query["left"]
    assert fixed["right"] == query["right"]
