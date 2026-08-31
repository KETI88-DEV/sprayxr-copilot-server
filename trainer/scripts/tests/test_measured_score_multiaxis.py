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
    """자세 문장도 하나이고, 고르는 기준은 크기가 아니라 sigma다.

    이 픽스처는 sigma와 크기를 **일부러 어긋나게** 둔다. wall이 18cm로 더 크지만
    vertical이 sigma 3.4로 더 이상하다. wall은 AXIS_ORDER에서도 앞이다. 따라서
    vertical이 나와야만 "sigma로 골랐다"가 증명된다 — 세 기준이 같은 답을 내는
    픽스처로는 아무것도 증명하지 못한다(초안이 그랬다).
    """
    delta = _head((0.18, 0.0, -0.12), {"wall": 2.9, "vertical": 3.4, "lateral": 0.1})
    stance = measured_score.head_stance(delta, HORIZONTAL)
    assert stance is not None
    assert stance.axis == "vertical"  # sigma가 큰 쪽. 크기로 고르면 wall이 나온다.


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


def _shift_by_hand(
    corrections: list[object], frame: object
) -> tuple[float, float, float]:
    """`issued_shifts`를 전혀 부르지 않고 같은 계산을 손으로 다시 한다.

    `magnitude_cm/100 × unit`을 월드에서 직접 더한 뒤 기저 세 축과 내적한다.
    테스트가 `issued_shifts`를 불러 기대값을 만들면 "issued_shifts가 issued_shifts와
    일치한다"는 동어반복이 된다 — 그러면 `+=`가 `=`로 바뀌어 다축 합산 자체가
    사라져도 아무 테스트도 실패하지 않는다(실제로 그런 일이 있었다).
    """
    world = [0.0, 0.0, 0.0]
    for correction in corrections:
        metres = correction.magnitude_cm / 100.0  # type: ignore[attr-defined]
        for i in range(3):
            world[i] += correction.unit[i] * metres  # type: ignore[attr-defined]
    return tuple(
        sum(world[i] * axis_vec[i] for i in range(3))
        for axis_vec in (frame.normal, frame.tangent, frame.up)  # type: ignore[attr-defined]
    )


def test_issued_shifts_sums_multiple_axes_for_the_same_point() -> None:
    """다축 합산이 진짜인지 고정한다.

    `issued_shifts`의 `acc[i] +=`를 `acc[i] =`로 바꾸면(마지막 축이 나머지를
    조용히 덮어씀) 이전 스위트가 전부 그대로 통과했다 — 다축 합산은 이 계획
    전체의 존재 이유인데 그것을 지워도 아무도 모른다는 뜻이다. 이 테스트의
    기대값은 `_shift_by_hand`로 `issued_shifts`와 독립적으로 계산한다.

    두 축(wall·vertical)은 서로 직교해 절대 평행하지 않고, 둘 다 8cm 넘게
    커서 한쪽이 없어지면 합이 눈에 띄게 달라진다 — 축이 거의 상쇄되거나 한쪽이
    지배적이면 `=`로 바꿔도 우연히 통과할 수 있다.
    """
    right = _right((0.08, 0.0, -0.09), {"wall": 3.5, "vertical": 3.0, "lateral": 0.2})
    corrections = measured_score.issued_axes([right], HORIZONTAL)
    assert {c.axis for c in corrections} == {"wall", "vertical"}
    assert all(abs(c.magnitude_cm) > 5.0 for c in corrections)  # 둘 다 충분히 크다
    wall = next(c for c in corrections if c.axis == "wall")
    vertical = next(c for c in corrections if c.axis == "vertical")
    dot = sum(wall.unit[i] * vertical.unit[i] for i in range(3))
    assert abs(dot) < 1e-9  # 정말로 직교한다(평행이 아니다)

    expected = _shift_by_hand(corrections, HORIZONTAL)
    got = measured_score.issued_shifts([right], HORIZONTAL)["right"]
    for g, e in zip(got, expected):
        assert abs(g - e) < 1e-9


def test_apply_measured_moves_exactly_the_issued_axes_and_nothing_else() -> None:
    """불변식. 발부한 축의 합만큼만 옮긴다.

    기대값은 `issued_shifts`를 부르지 않고 `_shift_by_hand`로 독립적으로 계산한다
    — `apply_measured`가 `issued_shifts`와 "일치"하는지가 아니라 물리적으로 맞는
    이동을 내는지를 봐야 하기 때문이다.
    """
    right = _right((0.08, 0.0, -0.06), {"wall": 3.5, "vertical": 2.6, "lateral": 0.1})
    query = {
        "right": [(0.0, 0.0, 0.0)] * 3,
        "head": [(0.0, 0.0, 0.0)] * 3,
        "left": [(0.0, 0.0, 0.0)] * 3,
    }
    corrections = measured_score.issued_axes([right], HORIZONTAL)
    assert len(corrections) >= 1
    expected = _shift_by_hand(corrections, HORIZONTAL)

    fixed = measured_score.apply_measured(query, [right], HORIZONTAL)
    assert fixed["head"] == query["head"]
    assert fixed["left"] == query["left"]
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


def test_inclusion_rule_is_governed_by_the_pre_registered_constants(monkeypatch) -> None:
    """포함 규칙이 사전등록 상수를 **실제로** 참조하는지 고정한다.

    스펙 §5는 임계를 움직이지 않겠다고 약속하는데, 그 약속은 상수가 동작을 지배할
    때만 뜻이 있다. 최종 리뷰가 MIN_CORRECTION_CM을 1.0에서 4.0으로 바꿔도 스위트가
    전부 통과함을 확인했다 — 즉 약속을 지키는지 감시하는 장치가 없었다. 여기서
    두 상수를 각각 움직여 발부 집합이 따라 움직이는 것을 확인한다. 리터럴을 새로
    박으면 이 테스트가 깨진다.
    """
    # 점 sigma를 축 sigma들보다 높게 둔다. 그래야 임계를 올렸을 때 점 게이트가 먼저
    # 닫혀 축 게이트를 못 보는 일이 없다 — 처음 쓴 픽스처(점 sigma 3.0)가 정확히
    # 그랬고, 이 테스트가 그것을 잡아냈다.
    delta = measured_score.PointDelta(
        point="right",
        sigma=5.0,
        local_delta=_local(HORIZONTAL, (0.08, 0.025, 0.0)),  # wall 8cm, lateral 2.5cm
        samples=30,
        axis_sigmas={"wall": 3.5, "vertical": 0.1, "lateral": 3.0},
    )
    assert {c.axis for c in measured_score.right_axes(delta, HORIZONTAL)} == {"wall", "lateral"}

    # 최소 cm를 올리면 2.5cm짜리 lateral이 빠져야 한다.
    monkeypatch.setattr(measured_score, "MIN_CORRECTION_CM", 4.0)
    assert {c.axis for c in measured_score.right_axes(delta, HORIZONTAL)} == {"wall"}
    monkeypatch.undo()

    # sigma 임계를 lateral(3.0)과 wall(3.5) 사이에 두면 lateral만 빠져야 한다.
    # 점 sigma가 5.0이라 점 게이트는 열린 채다.
    monkeypatch.setattr(measured_score, "CORRECTION_SIGMA_THRESHOLD", 3.2)
    assert {c.axis for c in measured_score.right_axes(delta, HORIZONTAL)} == {"wall"}
