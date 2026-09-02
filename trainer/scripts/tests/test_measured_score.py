#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

measured_score = importlib.import_module("measured_score")
pose_similarity = importlib.import_module("pose_similarity")
stroke_frame = importlib.import_module("stroke_frame")

# 축별 산포가 모두 같으므로 예전 스칼라(0.05m) 채점과 같은 sigma를 낸다 —
# `uniform_spreads`가 그 하위호환을 보장한다(직교 기저 변환이 벡터 노름을 보존).
SPREADS = measured_score.uniform_spreads(0.05)

# 수평 스트로크: 접선이 월드 X(유니티 Y-up에서 수평)라 up의 직교화가 forward와
# 이미 직교한 world_axes.UP 그대로, 즉 정확히 월드 Y가 된다. normal=cross(X,Y)=Z이고
# gun을 hit에서 -Z로 0.6m 뗀 수평 위치(작업자가 벽 앞에 선다)로 두면 sign_toward가
# normal을 -Z로 맞춘다.
HORIZONTAL = stroke_frame.stroke_basis(
    hit=(1.2, 4.5, 0.8), tangent=(1.0, 0.0, 0.0), gun=(1.2, 4.5, 0.2)
)
# 수직 스트로크: 접선이 월드 Y(=world_axes.UP)라 up의 직교화가 퇴화하고 대체축
# (FALLBACK_TANGENT=(1,0,0), 수평)이 들어온다. 즉 여기서 up은 수평(X)이고 수직은
# tangent(Y)다. normal=cross(Y,X)=-Z이고, gun을 hit에서 Y·Z로만 옮기면(X는 그대로)
# sign_toward가 그 부호를 정한다.
VERTICAL = stroke_frame.stroke_basis(
    hit=(1.2, 4.5, 0.8), tangent=(0.0, 1.0, 0.0), gun=(1.2, 4.4, 0.2)
)


def _delta(basis: object, world: tuple[float, float, float]) -> object:
    """월드 오프셋을 그 기저의 로컬 성분으로 적는다. 채점 경로가 만드는 값과 같은 형식이다."""
    local = tuple(
        sum(world[axis] * getattr(basis, name)[axis] for axis in range(3))
        for name in ("normal", "tangent", "up")
    )
    return measured_score.PointDelta(point="right", sigma=3.0, local_delta=local)


def _track() -> dict[str, list[tuple[float, float, float]]]:
    return {name: [(0.3, 0.0, 0.1)] * 5 for name in ("right", "head", "left")}


def test_identical_tracks_score_one() -> None:
    result = measured_score.score_track(
        _track(), _track(), SPREADS, measured_score.DEFAULT_WEIGHTS
    )
    assert math.isclose(result.score, 1.0, abs_tol=1e-9)
    assert result.samples == 5


def test_right_hand_dominates_the_score() -> None:
    """오른손이 건이라 도막 품질을 직접 결정한다. 가중이 실제로 반영돼야 한다."""
    right_off = _track()
    right_off["right"] = [(0.4, 0.0, 0.1)] * 5
    left_off = _track()
    left_off["left"] = [(0.4, 0.0, 0.1)] * 5

    right_score = measured_score.score_track(
        right_off, _track(), SPREADS, measured_score.DEFAULT_WEIGHTS
    ).score
    left_score = measured_score.score_track(
        left_off, _track(), SPREADS, measured_score.DEFAULT_WEIGHTS
    ).score
    assert right_score < left_score


def test_missing_point_is_skipped_not_scored_zero() -> None:
    """못 잰 것을 0점으로 넣으면 평균이 근거 없이 내려간다."""
    query = _track()
    query["left"] = []
    result = measured_score.score_track(
        query, _track(), SPREADS, measured_score.DEFAULT_WEIGHTS
    )
    assert math.isclose(result.score, 1.0, abs_tol=1e-9)
    assert [delta.point for delta in result.deltas] == ["right", "head"]


def test_correction_text_uses_measured_quantities_not_angles() -> None:
    """REQ-19 게이트 통과 전까지 각도는 근거보다 높은 정밀도다.

    문장 전체를 고정한다. 조각만 확인하면 조사·어미가 깨진 비문이 통과한다. 벽 성분
    (5cm)과 수직 성분(15cm)이 동시에 임계를 넘지만, 더 큰 쪽(수직) 하나만 남아야
    한다 — 사람은 한 번에 하나만 고칠 수 있다.
    """
    deltas = [
        measured_score.PointDelta(point="right", sigma=3.0, local_delta=(0.05, 0.0, 0.15))
    ]
    assert measured_score.corrections_text(deltas) == ["오른손을 15cm 올리세요."]


def test_head_and_left_hand_never_produce_a_correction_sentence() -> None:
    """머리·왼손은 σ가 임계를 크게 넘어도 지시를 내지 않는다.

    오른손이 건을 쥐고 도막 품질을 직접 결정한다(§7). 머리는 작업자가 의도적으로
    조정하는 대상이 아니고 왼손(플래시)은 작업 결과와의 인과가 약하다 — 점수 가중은
    그대로 남지만(변경 없음) 지시는 오른손에만 낸다.
    """
    head = measured_score.PointDelta(
        point="head", sigma=5.0, local_delta=(0.0, -0.12, 0.0)
    )
    left = measured_score.PointDelta(
        point="left", sigma=5.0, local_delta=(0.0, 0.0, -0.08)
    )
    assert measured_score.corrections_text([head]) == []
    assert measured_score.corrections_text([left]) == []
    # 오른손이 섞여 있어도 머리·왼손 몫은 여전히 안 나오고, 오른손 문장 하나만 남는다.
    right = measured_score.PointDelta(
        point="right", sigma=5.0, local_delta=(0.0, 0.0, 0.2)
    )
    assert measured_score.corrections_text([head, right, left]) == [
        "오른손을 20cm 올리세요."
    ]


def test_right_hand_within_expert_range_produces_no_sentence() -> None:
    """오른손도 σ가 임계 이하면 아무것도 내지 않는다. 기존 규약을 유지한다."""
    right = measured_score.PointDelta(
        point="right", sigma=1.0, local_delta=(0.2, 0.0, 0.2)
    )
    assert measured_score.corrections_text([right]) == []


def test_largest_component_is_chosen_when_two_are_close() -> None:
    """벽 성분과 수직 성분이 근접해도 더 큰 쪽만 골라야 한다.

    반대로도 확인한다 — 근접값에서 우연히 항상 같은 축이 뽑히는 구현(정렬 순서에
    기댄 버그)을 잡는 못이다.
    """
    vertical_wins = measured_score.PointDelta(
        point="right", sigma=3.0, local_delta=(0.084, 0.0, 0.086)
    )
    lines = measured_score.corrections_text([vertical_wins])
    assert lines == ["오른손을 9cm 올리세요."]

    wall_wins = measured_score.PointDelta(
        point="right", sigma=3.0, local_delta=(0.096, 0.0, 0.094)
    )
    lines = measured_score.corrections_text([wall_wins])
    assert lines == ["오른손을 벽에서 10cm 더 띄우세요."]


def test_height_wording_survives_on_a_horizontal_stroke() -> None:
    """수평 스트로크에서 순수 수직 이탈은 계속 올리/내리여야 한다.

    수직 문구를 수직 스트로크에서 그냥 막아 버리는 식의 회피를 막는 못이다.

    HORIZONTAL.up은 월드 Y이므로 순수 수직 이탈은 월드 Y 성분 0.15여야 한다(월드 Z가
    아니다 — Z는 HORIZONTAL.normal이 서 있는 벽 축이다). local = (dot(world,normal=
    (0,0,-1)), dot(world,tangent=(1,0,0)), dot(world,up=(0,1,0))) = (0, 0, 0.15)다.
    """
    line = measured_score.corrections_text(
        [_delta(HORIZONTAL, (0.0, 0.15, 0.0))], HORIZONTAL
    )
    assert line == ["오른손을 15cm 올리세요."]


def test_sideways_offset_on_a_vertical_stroke_is_not_called_height() -> None:
    """수직 스트로크에서 up 축은 수평이다.

    성분을 축 이름으로 읽으면 옆으로 15cm 틀어진 손에 "15cm 올리세요"가 나간다.
    작업자가 그대로 따르면 이탈이 커진다.

    VERTICAL.tangent가 월드 Y(진행=수직)이므로 "옆" 이탈은 VERTICAL.up이 서 있는
    월드 X여야 한다(월드 Y가 아니다 — Y로 옮기면 그 자체가 진짜 수직 이탈이 된다).
    local = (dot(world,normal=(0,0,-1)), dot(world,tangent=(0,1,0)),
    dot(world,up=(1,0,0))) = (0, 0, 0.15) — normal·tangent 성분이 0이라 wall·vertical
    후보는 모두 1cm 문턱 아래로 걸러지고, tangent에 수평 성분이 없어(along=None)
    남은 평면 잔차(=local[2]가 만든 세계 X 성분) 하나가 부호 없는 lateral로 남는다.
    """
    lines = measured_score.corrections_text(
        [_delta(VERTICAL, (0.15, 0.0, 0.0))], VERTICAL
    )
    assert lines == ["오른손을 벽을 따라 15cm 옮기세요."]
    assert "올리" not in lines[0] and "내리" not in lines[0]


def test_vertical_offset_on_a_vertical_stroke_is_still_called_height() -> None:
    """수직 스트로크에서도 진짜 수직 이탈은 올리/내리로 나가야 한다.

    그때 수직은 tangent(월드 Y)에 실린다. 진행 방향으로 적으면 높이 지시가 사라진다.
    local = (dot(world,normal=(0,0,-1)), dot(world,tangent=(0,1,0)),
    dot(world,up=(1,0,0))) = (0, -0.15, 0)다.
    """
    lines = measured_score.corrections_text(
        [_delta(VERTICAL, (0.0, -0.15, 0.0))], VERTICAL
    )
    assert lines == ["오른손을 15cm 내리세요."]


def test_corrections_never_carry_an_angle() -> None:
    """REQ-19 게이트 통과 전까지 각도를 붙이지 않는다. 새 문구에도 적용된다.

    세 조합 모두 위 세 테스트가 이미 축 대응을 확인한 조합을 그대로 가져와 벽·수직을
    함께 어긋나게 만든다 — 이 테스트 자체는 "도"가 어디에도 없는지만 본다.
    """
    for basis, world in (
        (HORIZONTAL, (0.0, 0.15, 0.05)),
        (VERTICAL, (0.15, 0.0, 0.0)),
        (VERTICAL, (0.0, -0.15, 0.07)),
    ):
        for line in measured_score.corrections_text([_delta(basis, world)], basis):
            assert "도" not in line


def test_right_hand_finding_carries_sample_count_and_matches_the_sentence() -> None:
    """구조체는 문장과 같은 축·방향·부호를 내고, 문장에는 없는 표본 수를 얹는다.

    표본 3개에서 나온 이탈과 24개에서 나온 이탈은 같은 주장이 아니다 — 다운스트림
    VLM이 그 차이를 봐야 하므로 `samples`가 그대로 전달돼야 한다.
    """
    delta = measured_score.PointDelta(
        point="right", sigma=3.4, local_delta=(0.0, 0.0, 0.18), samples=17
    )
    finding = measured_score.right_hand_finding([delta])
    assert finding is not None
    assert finding.axis == "vertical"
    assert finding.direction == "올리"
    assert finding.samples == 17
    assert finding.sigma == 3.4
    assert math.isclose(finding.magnitude_cm, 18.0, abs_tol=1e-9)

    lines = measured_score.corrections_text([delta])
    assert lines == ["오른손을 18cm 올리세요."]
    assert finding.direction in lines[0]
    assert f"{abs(finding.magnitude_cm):.0f}cm" in lines[0]


def test_right_hand_finding_is_none_below_threshold_or_without_right_hand() -> None:
    below = measured_score.PointDelta(
        point="right", sigma=1.5, local_delta=(0.2, 0.0, 0.2)
    )
    only_head = measured_score.PointDelta(
        point="head", sigma=5.0, local_delta=(0.0, 0.2, 0.0)
    )
    assert measured_score.right_hand_finding([below]) is None
    assert measured_score.right_hand_finding([only_head]) is None
    assert measured_score.right_hand_finding([]) is None


def test_correction_skips_points_inside_expert_range() -> None:
    deltas = [
        measured_score.PointDelta(point="head", sigma=1.0, local_delta=(0.01, 0.0, 0.0))
    ]
    assert measured_score.corrections_text(deltas) == []


def test_apply_measured_improves_the_score() -> None:
    query = _track()
    query["right"] = [(0.45, 0.0, 0.1)] * 5

    before = measured_score.score_track(
        query, _track(), SPREADS, measured_score.DEFAULT_WEIGHTS
    )
    fixed = measured_score.apply_measured(query, before.deltas)
    after = measured_score.score_track(
        fixed, _track(), SPREADS, measured_score.DEFAULT_WEIGHTS
    )
    assert after.score > before.score


def test_gap_closure_reuses_the_existing_metric() -> None:
    closure = pose_similarity.gap_closure(0.4, 0.7)
    assert closure is not None
    assert math.isclose(closure, 0.5, abs_tol=1e-9)


# --- 교정 시뮬레이션이 실제로 낸 지시와 같은가 -------------------------------------


def _old_all_points_apply(
    query: dict[str, list[tuple[float, float, float]]],
    deltas: list[object],
) -> dict[str, list[tuple[float, float, float]]]:
    """fa9c4b0 이전의 시뮬레이션. 임계를 넘은 **모든** 점을 3성분 전부 옮겼다.

    회귀 기준으로만 쓴다. 새 시뮬레이션이 이보다 관대하면(점수가 더 높으면) 내주지
    않은 교정을 개선율에 계상하고 있다는 뜻이다.
    """
    moves = {
        delta.point: delta.local_delta
        for delta in deltas
        if delta.sigma > measured_score.CORRECTION_SIGMA_THRESHOLD
    }
    output: dict[str, list[tuple[float, float, float]]] = {}
    for name, points in query.items():
        shift = moves.get(name)
        if shift is None:
            output[name] = list(points)
            continue
        output[name] = [
            (point[0] + shift[0], point[1] + shift[1], point[2] + shift[2])
            for point in points
        ]
    return output


def _three_point_case() -> tuple[
    dict[str, list[tuple[float, float, float]]],
    dict[str, list[tuple[float, float, float]]],
]:
    """세 점 모두 임계를 넘게 어긋난 질의/레퍼런스. 실측 축 채점 경로와 같은 형식이다."""
    reference = {name: [(0.30, 0.00, 0.10)] * 5 for name in ("right", "head", "left")}
    query = {
        "right": [(0.30 + 0.09, 0.00 + 0.05, 0.10 + 0.14)] * 5,
        "head": [(0.30 + 0.22, 0.00 - 0.11, 0.10 + 0.07)] * 5,
        "left": [(0.30 - 0.18, 0.00 + 0.13, 0.10 - 0.09)] * 5,
    }
    return query, reference


def test_apply_measured_never_moves_the_left_hand() -> None:
    """왼손은 어떤 경우에도 제자리다. 지시하지 않는 점이기 때문이다.

    fa9c4b0이 문장을 좁혔는데 시뮬레이션은 임계를 넘은 세 점을 모두 옮기고 있었다.
    그러면 작업자에게 말한 것과 개선율이 재는 것이 달라진다. 다축 교정 이후에도
    왼손만은 계속 지시 대상이 아니므로 이 단언은 그대로 성립해야 한다.
    """
    query, reference = _three_point_case()
    before = measured_score.score_track(
        query, reference, SPREADS, measured_score.DEFAULT_WEIGHTS
    )
    sigmas = {delta.point: delta.sigma for delta in before.deltas}
    assert min(sigmas.values()) > measured_score.CORRECTION_SIGMA_THRESHOLD

    fixed = measured_score.apply_measured(query, before.deltas, HORIZONTAL)
    assert fixed["left"] == query["left"]
    assert fixed["right"] != query["right"]


def test_apply_measured_moves_the_head_only_along_an_issued_stance_axis() -> None:
    """머리는 자세 문장이 나갈 때만, 그 축으로만, 말한 만큼만 움직인다.

    기대값은 `issued_shifts`를 부르지 않는다 — `head_stance`가 낸 `AxisCorrection`
    하나에서 손으로 다시 만든다: unit × magnitude_cm/100을 월드 벡터로 만들고
    HORIZONTAL의 normal/tangent/up과 내적해 스트로크 성분으로 되돌린다.
    `apply_measured`가 `issued_shifts`와 "일치"하는지가 아니라 물리적으로 맞는
    이동을 내는지를 봐야 한다.
    """
    query, reference = _three_point_case()
    before = measured_score.score_track(
        query, reference, SPREADS, measured_score.DEFAULT_WEIGHTS
    )
    head = next(d for d in before.deltas if d.point == "head")
    stance = measured_score.head_stance(head, HORIZONTAL)
    fixed = measured_score.apply_measured(query, before.deltas, HORIZONTAL)
    if stance is None:
        assert fixed["head"] == query["head"]
        return

    metres = stance.magnitude_cm / 100.0
    world = tuple(stance.unit[i] * metres for i in range(3))
    expected = (
        measured_score._dot(world, HORIZONTAL.normal),
        measured_score._dot(world, HORIZONTAL.tangent),
        measured_score._dot(world, HORIZONTAL.up),
    )
    moved = tuple(fixed["head"][0][i] - query["head"][0][i] for i in range(3))
    for got, want in zip(moved, expected):
        assert abs(got - want) < 1e-9


def test_apply_measured_moves_every_axis_that_clears_the_threshold() -> None:
    """다축 교정이 진짜다: sigma가 넘는 축 전부가, 그만큼만 옮겨진다.

    옛 이름("성분 하나만 옮긴다")은 Task 1이 이미 버린 계약을 여전히 주장하고
    있었다. 게다가 옛 픽스처(`_three_point_case`)의 오른손은 wall 1.8 / vertical
    2.8 / lateral 1.0으로 마침 축 하나만 임계(2.0)를 넘어서 그 주장이 다축 경로를
    한 번도 태우지 않은 채 우연히 통과하고 있었다. 여기서는 wall·vertical 두
    축이 실제로 넘도록 만들어 `score_track` → `right_axes` → `apply_measured`가
    끊기지 않고 이어지는지를 본다. 기대값은 `issued_shifts`를 부르지 않고
    `right_axes`가 낸 `AxisCorrection`들에서 손으로 다시 계산한다.
    """
    reference = {"right": [(0.30, 0.00, 0.10)] * 5}
    query = {"right": [(0.15, -0.03, -0.03)] * 5}
    before = measured_score.score_track(
        query, reference, SPREADS, measured_score.DEFAULT_WEIGHTS
    )
    right = next(d for d in before.deltas if d.point == "right")
    corrections = measured_score.right_axes(right, HORIZONTAL)
    issued = {c.axis for c in corrections}
    assert issued == {"wall", "vertical"}  # lateral(sigma 0.6)은 임계 아래로 남는다

    world = [0.0, 0.0, 0.0]
    for c in corrections:
        metres = c.magnitude_cm / 100.0
        for i in range(3):
            world[i] += c.unit[i] * metres
    expected = (
        measured_score._dot(tuple(world), HORIZONTAL.normal),
        measured_score._dot(tuple(world), HORIZONTAL.tangent),
        measured_score._dot(tuple(world), HORIZONTAL.up),
    )

    fixed = measured_score.apply_measured(query, before.deltas, HORIZONTAL)
    moved = tuple(fixed["right"][0][i] - query["right"][0][i] for i in range(3))
    for got, want in zip(moved, expected):
        assert abs(got - want) < 1e-9


def test_apply_measured_is_never_more_generous_than_the_old_all_points_sim() -> None:
    """덜 고치므로 교정 후 점수가 옛 시뮬레이션보다 높을 수 없다."""
    query, reference = _three_point_case()
    before = measured_score.score_track(
        query, reference, SPREADS, measured_score.DEFAULT_WEIGHTS
    )
    narrow = measured_score.score_track(
        measured_score.apply_measured(query, before.deltas, HORIZONTAL),
        reference,
        SPREADS,
        measured_score.DEFAULT_WEIGHTS,
    )
    wide = measured_score.score_track(
        _old_all_points_apply(query, list(before.deltas)),
        reference,
        SPREADS,
        measured_score.DEFAULT_WEIGHTS,
    )
    assert narrow.score <= wide.score
    assert narrow.score > before.score


def test_apply_measured_does_nothing_when_nothing_is_issued() -> None:
    """점들이 진짜로 어긋나 있어도, 어느 축도 임계를 넘지 않으면 아무것도 옮기지 않는다.

    전부 0인 픽스처는 게이팅이 심하게 고장나도 통과한다 — 애초에 옮길 것이 없기
    때문이다. 여기서는 세 점 모두 몇 센티미터씩 실제로 어긋나 있지만(2~4cm) 산포
    (5cm)에 비춰 이상하지 않은 크기라 어느 축도 발부되지 않는다.
    """
    reference = {name: [(0.30, 0.00, 0.10)] * 5 for name in ("right", "head", "left")}
    query = {
        "right": [(0.33, 0.02, 0.13)] * 5,
        "head": [(0.34, -0.03, 0.09)] * 5,
        "left": [(0.27, 0.04, 0.08)] * 5,
    }
    before = measured_score.score_track(
        query, reference, SPREADS, measured_score.DEFAULT_WEIGHTS
    )
    assert measured_score.issued_axes(before.deltas, HORIZONTAL) == []
    fixed = measured_score.apply_measured(query, before.deltas, HORIZONTAL)
    assert fixed == {name: list(points) for name, points in query.items()}


def test_candidates_agree_with_unfiltered_magnitudes() -> None:
    """후보에 든 축은 임계 이상이고, 그 크기가 필터 없는 값과 정확히 같다."""
    frame = stroke_frame.Basis(
        origin=(0.0, 0.0, 0.0),
        normal=(0.6, 0.8, 0.0),
        tangent=(-0.8, 0.6, 0.0),
        up=(0.0, 0.0, 1.0),
    )
    for local in [
        (0.05, 0.03, 0.07),
        (-0.12, 0.004, -0.02),
        (0.002, -0.001, 0.0005),
        (0.3, -0.25, 0.11),
    ]:
        magnitudes = measured_score._axis_magnitudes_cm(local, frame)
        candidates = measured_score._correction_candidates(local, frame)
        for axis, value, _word, _unit in candidates:
            assert abs(value) >= measured_score.MIN_CORRECTION_CM
            assert value == pytest.approx(magnitudes[axis])
        named = {axis for axis, _v, _w, _u in candidates}
        for axis in measured_score.AXIS_ORDER:
            if axis not in named:
                assert abs(magnitudes[axis]) < measured_score.MIN_CORRECTION_CM


def test_vertical_normal_keeps_zero_wall_magnitude_but_no_wall_candidate() -> None:
    """법선에 수평 성분이 없으면 크기는 0.0으로 남고 후보에서는 빠진다.

    수평은 (index 0, index 2)다(유니티 Y-up). normal=(0,1,0)(월드 Y=수직)이면
    horizontal_of(normal)=(0,0)이라 `_flattened`가 None을 낸다.
    """
    frame = stroke_frame.Basis(
        origin=(0.0, 0.0, 0.0),
        normal=(0.0, 1.0, 0.0),
        tangent=(1.0, 0.0, 0.0),
        up=(0.0, 0.0, 1.0),
    )
    local = (0.2, 0.15, -0.1)
    magnitudes = measured_score._axis_magnitudes_cm(local, frame)
    assert magnitudes["wall"] == 0.0
    assert "wall" not in {
        axis for axis, _v, _w, _u in measured_score._correction_candidates(local, frame)
    }


def test_vertical_stroke_lateral_is_unsigned_in_both_paths() -> None:
    """수평 진행이 없으면 lateral은 두 경로 모두 부호 없는 크기다.

    진행축(tangent)에 수평 성분이 없어야 하므로 tangent=(0,1,0)(월드 Y=수직)을
    쓴다 — horizontal_of(tangent)=(0,0)이라 `_flattened`가 None을 내 along 분기가
    막힌다.
    """
    frame = stroke_frame.Basis(
        origin=(0.0, 0.0, 0.0),
        normal=(1.0, 0.0, 0.0),
        tangent=(0.0, 1.0, 0.0),
        up=(0.0, 0.0, 1.0),
    )
    local = (0.02, -0.18, 0.09)
    magnitudes = measured_score._axis_magnitudes_cm(local, frame)
    assert magnitudes["lateral"] > 0.0
    lateral = [
        c for c in measured_score._correction_candidates(local, frame) if c[0] == "lateral"
    ]
    assert lateral and lateral[0][1] == pytest.approx(magnitudes["lateral"])
    assert lateral[0][2] == ""
