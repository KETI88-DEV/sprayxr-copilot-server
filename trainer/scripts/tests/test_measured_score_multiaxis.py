#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

measured_score = importlib.import_module("measured_score")
stroke_frame = importlib.import_module("stroke_frame")

# test_measured_score.py의 HORIZONTAL과 같은 값으로 맞춘다: 접선이 월드 X(유니티
# Y-up에서 수평)라 up의 직교화가 forward와 이미 직교한 world_axes.UP 그대로, 즉
# 정확히 월드 Y가 된다. normal=cross(X,Y)=Z이고 gun을 hit에서 -Z로 0.6m 뗀 수평
# 위치로 두면 sign_toward가 normal을 -Z로 맞춘다 — 그래서 HORIZONTAL.normal=(0,0,-1),
# .tangent=(1,0,0), .up=(0,1,0)이다.
HORIZONTAL = stroke_frame.stroke_basis(
    hit=(1.2, 4.5, 0.8), tangent=(1.0, 0.0, 0.0), gun=(1.2, 4.5, 0.2)
)
T = measured_score.CORRECTION_SIGMA_THRESHOLD


def _local(basis: object, world: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(
        sum(world[i] * getattr(basis, name)[i] for i in range(3))
        for name in ("normal", "tangent", "up")
    )


# 이 파일의 모든 `_right`/`_head` 호출은 "world" 인자에 (wall, lateral, vertical)
# 의미의 cm 값을 실어 왔다. **주의**: 옛 HORIZONTAL은 항등 기저가 아니었다 — 옛
# HORIZONTAL(tangent=(0,1,0), gun=(0.6,4.5,0.8))의 법선을 손으로 다시 계산하면(옛
# `stroke_basis`: forward=(0,1,0), up=orthogonalized((0,0,1),forward)=(0,0,1),
# normal=normalize(cross(forward,up))=(1,0,0), to_worker=gun-hit=(-0.6,0,0),
# dot(normal,to_worker)=-0.6<0 → sign_toward가 뒤집는다) **normal=(-1,0,0)**이었다
# — 이 파일이 리팩터 전부터 이미 그렇게 문서화하고 있었다(옛
# test_magnitude_and_unit_agree_so_the_sentence_cannot_drift의 "실제로는 월드
# -X다"). tangent=(0,1,0)·up=(0,0,1)은 부호가 그대로였으므로, 옛
# `_local(OLD_HORIZONTAL, w) = (-w[0], w[1], w[2])`다 — 즉
# **wall = -w_old[0]**, lateral = w_old[1], vertical = w_old[2]였다(1라운드에서
# wall = +w_old[0]로 잘못 가정했다 — 리뷰 A4).
#
# 새 HORIZONTAL은 (normal=-Z, tangent=X, up=Y)이므로
# `_local(HORIZONTAL, w) = (-w[2], w[0], w[1])`이다 — 즉 wall=-w_new[2],
# lateral=w_new[0], vertical=w_new[1]. 같은 (wall, lateral, vertical) 뜻을
# 유지하려면: w_new[0]=lateral=w_old[1], w_new[1]=vertical=w_old[2],
# w_new[2]=-wall=-(-w_old[0])=w_old[0]. 즉 **w_new = (w_old[1], w_old[2],
# w_old[0])** — w_old를 그대로 한 칸 순환시킨 값이다(1라운드의
# `(w_old[1], w_old[2], -w_old[0])`에서 셋째 성분의 부호가 틀렸었다). 아래 각
# 호출의 주석은 옛 (wall, lateral, vertical) 의도를 그대로 남기고, world 튜플은
# 이 수정된 순환으로 다시 지었다.


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
        # wall +8cm, lateral +5cm, vertical -6cm — w = (w_old[1], w_old[2], w_old[0])
        (0.05, -0.06, 0.08),
        {"wall": 3.5, "vertical": 2.6, "lateral": 2.2},
    )
    issued = measured_score.right_axes(delta, HORIZONTAL)
    assert {c.axis for c in issued} == {"wall", "vertical", "lateral"}
    assert all(c.point == "right" for c in issued)


def test_axes_below_the_threshold_are_not_issued() -> None:
    """sigma가 임계 아래인 축은 이상하지 않다. 지시하면 지어낸 교정이다."""
    delta = _right(
        (0.05, -0.06, 0.08),  # wall +8cm, lateral +5cm, vertical -6cm
        {"wall": 3.5, "vertical": 1.1, "lateral": 0.4},
    )
    issued = measured_score.right_axes(delta, HORIZONTAL)
    assert {c.axis for c in issued} == {"wall"}


def test_axes_below_the_minimum_centimetre_are_not_issued() -> None:
    """sigma가 커도 5mm를 옮기라는 지시는 사람이 따를 수 없다."""
    delta = _right(
        # wall 8cm, lateral 0.5cm(MIN_CORRECTION_CM 미만), vertical 0
        (0.005, 0.0, 0.08),
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
        (0.0, 0.0, 0.08),  # wall 8cm, lateral·vertical 0
        {"wall": 1.4, "vertical": 1.3, "lateral": 1.2},  # 아무도 2.0 미만
    )
    issued = measured_score.right_axes(delta, HORIZONTAL)
    assert [c.axis for c in issued] == ["wall"]


def test_no_axis_sigmas_falls_back_to_the_single_sentence_axis() -> None:
    """축별 sigma 없이 만든 구조체는 예전 동작 그대로다(옛 호출자 보호)."""
    delta = _right((0.05, -0.06, 0.08), None)  # wall 8cm, lateral 5cm, vertical -6cm
    issued = measured_score.right_axes(delta, HORIZONTAL)
    assert len(issued) == 1


def test_below_the_point_gate_nothing_is_issued() -> None:
    """점 게이트는 그대로다. 오른손이 이상하지 않으면 아무 말도 하지 않는다."""
    delta = measured_score.PointDelta(
        point="right",
        sigma=T,  # 초과가 아니라 같음 — 열리지 않아야 한다
        # wall +8cm, lateral +5cm, vertical -6cm
        local_delta=_local(HORIZONTAL, (0.05, -0.06, 0.08)),
        samples=30,
        axis_sigmas={"wall": 3.5, "vertical": 2.6, "lateral": 2.2},
    )
    assert measured_score.right_axes(delta, HORIZONTAL) == []


def test_magnitude_and_unit_agree_so_the_sentence_cannot_drift() -> None:
    """cm와 단위벡터가 같은 곳에서 나온다. 어긋나면 문장과 시뮬레이션이 갈라진다.

    HORIZONTAL.normal은 월드 -Z다: tangent=X, up(=world_axes.UP)=Y이므로
    normal=cross(X,Y)=Z이고, 이 hit/gun 조합은 gun이 hit보다 -Z쪽에 있어
    `sign_toward`가 법선을 -Z로 맞춘다(`stroke_frame.stroke_basis` 참고).

    옛 파일(리팩터 전)은 이 픽스처에서 `wall`이 -8.0으로 나온다고 이미 적어
    뒀었다 — 옛 HORIZONTAL.normal도 월드 -X였기 때문이다(파일 상단 주석 참고).
    world 인자는 그 -8.0을 그대로 재현하도록 고른다: 새 좌표에서
    `w = (w_old[1], w_old[2], w_old[0]) = (0, 0, 0.08)`(옛 w_old=(0.08,0,0),
    "wall -8cm"). `_local(HORIZONTAL,(0,0,0.08)) = (-0.08·(-1), ·, ·) = (-0.08,…)`
    이므로 `magnitude_cm`는 -8.0이 맞다. 부호 자체를 어느 쪽으로 고정할지는 이
    테스트의 관심사가 아니다 — 검증할 불변은 "magnitude_cm/100 × unit이 원래 월드
    이동을 정확히 되살린다"는 것이므로 그 재구성을 직접 확인한다.
    """
    delta = _right((0.0, 0.0, 0.08), {"wall": 3.5, "vertical": 0.1, "lateral": 0.1})
    (wall,) = measured_score.right_axes(delta, HORIZONTAL)
    assert wall.axis == "wall"
    # HORIZONTAL의 normal은 월드 -Z다(위 설명 참고).
    assert abs(wall.unit[2]) > 0.99
    reconstructed_world_z = wall.magnitude_cm / 100.0 * wall.unit[2]
    assert abs(reconstructed_world_z - 0.08) < 0.005


def _head(world: tuple[float, float, float], axis_sigmas: dict[str, float]) -> object:
    return measured_score.PointDelta(
        point="head",
        sigma=3.0,
        local_delta=_local(HORIZONTAL, world),
        samples=30,
        axis_sigmas=axis_sigmas,
    )


def test_head_height_becomes_a_stance_sentence() -> None:
    delta = _head((0.0, -0.12, 0.0), {"wall": 0.2, "vertical": 3.1, "lateral": 0.1})
    stance = measured_score.head_stance(delta, HORIZONTAL)
    assert stance is not None
    assert stance.point == "head" and stance.axis == "vertical"
    text = measured_score.stance_text(stance)
    assert "낮춰" in text


def test_head_wall_distance_becomes_a_step_closer_sentence() -> None:
    """w_old=(wall=0.18, 0, 0) → w_new=(w_old[1], w_old[2], w_old[0])=(0,0,0.18).

    local[0] = dot(w_new, HORIZONTAL.normal=(0,0,-1)) = 0.18·(-1) = -0.18 → wall이
    음수이므로 `STANCE_PHRASES[("wall", False)]` = "벽에 한 걸음 더 붙어 서십시오."
    다. 이 함수 이름은 1라운드에서 "step_back"이었다 — 그때 셋째 성분 부호를
    잘못 유도해(리뷰 A4, 파일 상단 주석) 실제로는 +18cm/"떨어져"가 나왔는데, 그
    오류를 "걸음"만 확인하는 약한 assert(옛 이름 그대로)가 가려 아무것도 실패하지
    않았다(리뷰 A7). 이름을 손으로 다시 유도한 진짜 부호("붙어")에 맞춘다.

    `STANCE_PHRASES`의 두 wall 문구를 서로 바꿔치기(swap)하면 이 assert가 실제로
    깨지는지 확인했다 — "걸음"만 보는 옛 assert는 swap에도 통과했다(리뷰 A7이
    지적한 결함 그 자체).
    """
    delta = _head((0.0, 0.0, 0.18), {"wall": 3.4, "vertical": 0.2, "lateral": 0.1})
    stance = measured_score.head_stance(delta, HORIZONTAL)
    assert stance is not None and stance.axis == "wall"
    assert stance.magnitude_cm < 0.0
    text = measured_score.stance_text(stance)
    assert "붙어" in text
    assert "떨어져" not in text


def test_only_one_stance_sentence_even_when_both_axes_fire() -> None:
    """자세 문장도 하나이고, 고르는 기준은 크기가 아니라 sigma다.

    이 픽스처는 sigma와 크기를 **일부러 어긋나게** 둔다. wall이 18cm로 더 크지만
    vertical이 sigma 3.4로 더 이상하다. wall은 AXIS_ORDER에서도 앞이다. 따라서
    vertical이 나와야만 "sigma로 골랐다"가 증명된다 — 세 기준이 같은 답을 내는
    픽스처로는 아무것도 증명하지 못한다(초안이 그랬다).
    """
    delta = _head((0.0, -0.12, 0.18), {"wall": 2.9, "vertical": 3.4, "lateral": 0.1})
    stance = measured_score.head_stance(delta, HORIZONTAL)
    assert stance is not None
    assert stance.axis == "vertical"  # sigma가 큰 쪽. 크기로 고르면 wall이 나온다.


def test_head_lateral_is_never_issued() -> None:
    """작업자 요가 실측이 아니라 좌우를 판정할 기준이 없다(service.ASSUMED_YAW)."""
    delta = _head((0.20, 0.0, 0.0), {"wall": 0.1, "vertical": 0.1, "lateral": 5.0})
    assert measured_score.head_stance(delta, HORIZONTAL) is None


def test_head_below_the_threshold_says_nothing() -> None:
    delta = _head((0.0, -0.12, 0.0), {"wall": 0.2, "vertical": 1.9, "lateral": 0.1})
    assert measured_score.head_stance(delta, HORIZONTAL) is None


def test_stance_text_never_claims_a_posture_we_cannot_observe() -> None:
    """HMD로는 앉음과 쪼그림과 무릎 굽힘을 구별할 수 없다. 그러니 '앉아'라고 쓰지 않는다."""
    # 각 world는 (w_old[1], w_old[2], w_old[0]) 순환이다: vertical -12/+12cm,
    # wall +18/-18cm(옛 wall = -w_old[0]이므로 w_old[0]=+0.18은 wall=-18cm다).
    for world, sigmas in (
        ((0.0, -0.12, 0.0), {"vertical": 3.1}),
        ((0.0, 0.12, 0.0), {"vertical": 3.1}),
        ((0.0, 0.0, 0.18), {"wall": 3.4}),
        ((0.0, 0.0, -0.18), {"wall": 3.4}),
    ):
        stance = measured_score.head_stance(_head(world, sigmas), HORIZONTAL)
        assert stance is not None
        assert "앉아" not in measured_score.stance_text(stance)


def test_issued_axes_are_sorted_by_sigma_descending() -> None:
    """읽는 사람이 위에서부터 심각한 순으로 본다."""
    right = _right((0.05, -0.06, 0.08), {"wall": 2.2, "vertical": 3.9, "lateral": 2.6})
    head = _head((0.0, 0.0, 0.18), {"wall": 3.4, "vertical": 0.2, "lateral": 0.1})
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
    right = _right((0.0, 0.0, 0.08), {"wall": 3.5, "vertical": 0.1, "lateral": 0.1})
    quiet_head = _head((0.0, -0.01, 0.0), {"wall": 0.1, "vertical": 0.1, "lateral": 0.1})
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
    right = _right((0.0, -0.09, 0.08), {"wall": 3.5, "vertical": 3.0, "lateral": 0.2})
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
    right = _right((0.0, -0.06, 0.08), {"wall": 3.5, "vertical": 2.6, "lateral": 0.1})
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
    head = _head((0.0, 0.0, 0.18), {"wall": 3.4, "vertical": 0.2, "lateral": 0.1})
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
        local_delta=_local(HORIZONTAL, (0.025, 0.0, 0.08)),  # wall 8cm, lateral 2.5cm
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
