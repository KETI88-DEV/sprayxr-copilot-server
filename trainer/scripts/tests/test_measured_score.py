#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

measured_score = importlib.import_module("measured_score")
pose_similarity = importlib.import_module("pose_similarity")
stroke_frame = importlib.import_module("stroke_frame")

SPREADS = {"right": 0.05, "head": 0.05, "left": 0.05}

# 수평 스트로크: 접선이 월드 Y라 up이 정확히 월드 Z다.
HORIZONTAL = stroke_frame.stroke_basis(
    hit=(1.2, 4.5, 0.8), tangent=(0.0, 1.0, 0.0), gun=(0.6, 4.5, 0.8)
)
# 수직 스트로크: 접선이 월드 Z라 up의 직교화가 퇴화하고 대체축(월드 Y)이 들어온다.
# 즉 여기서 up은 수평이고 수직은 tangent다.
VERTICAL = stroke_frame.stroke_basis(
    hit=(1.2, 4.5, 0.8), tangent=(0.0, 0.0, 1.0), gun=(0.6, 4.5, 0.9)
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
    """
    line = measured_score.corrections_text(
        [_delta(HORIZONTAL, (0.0, 0.0, 0.15))], HORIZONTAL
    )
    assert line == ["오른손을 15cm 올리세요."]


def test_sideways_offset_on_a_vertical_stroke_is_not_called_height() -> None:
    """수직 스트로크에서 up 축은 수평이다.

    성분을 축 이름으로 읽으면 옆으로 15cm 틀어진 손에 "15cm 올리세요"가 나간다.
    작업자가 그대로 따르면 이탈이 커진다.
    """
    lines = measured_score.corrections_text(
        [_delta(VERTICAL, (0.0, 0.15, 0.0))], VERTICAL
    )
    assert lines == ["오른손을 벽을 따라 15cm 옮기세요."]
    assert "올리" not in lines[0] and "내리" not in lines[0]


def test_vertical_offset_on_a_vertical_stroke_is_still_called_height() -> None:
    """수직 스트로크에서도 진짜 수직 이탈은 올리/내리로 나가야 한다.

    그때 수직은 tangent에 실린다. 진행 방향으로 적으면 높이 지시가 사라진다.
    """
    lines = measured_score.corrections_text(
        [_delta(VERTICAL, (0.0, 0.0, -0.15))], VERTICAL
    )
    assert lines == ["오른손을 15cm 내리세요."]


def test_corrections_never_carry_an_angle() -> None:
    """REQ-19 게이트 통과 전까지 각도를 붙이지 않는다. 새 문구에도 적용된다."""
    for basis, world in (
        (HORIZONTAL, (0.05, 0.0, 0.15)),
        (VERTICAL, (0.0, 0.15, 0.0)),
        (VERTICAL, (0.07, 0.0, -0.15)),
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


def test_apply_measured_leaves_head_and_left_where_they_are() -> None:
    """교정문은 오른손 하나만 낸다. 시뮬레이션이 머리·왼손까지 옮기면 개선율이 부푼다.

    fa9c4b0이 문장을 오른손 하나로 좁혔는데 시뮬레이션은 임계를 넘은 세 점을 모두
    옮기고 있었다. 그러면 작업자에게는 오른손만 말하고 개선율은 머리·왼손까지 고쳤을
    때의 값을 낸다. 개선율은 goal_year2의 납품 지표라 그 차이가 계약의 뜻을 바꾼다.
    """
    query, reference = _three_point_case()
    before = measured_score.score_track(
        query, reference, SPREADS, measured_score.DEFAULT_WEIGHTS
    )
    sigmas = {delta.point: delta.sigma for delta in before.deltas}
    assert min(sigmas.values()) > measured_score.CORRECTION_SIGMA_THRESHOLD

    fixed = measured_score.apply_measured(query, before.deltas, HORIZONTAL)
    assert fixed["head"] == query["head"]
    assert fixed["left"] == query["left"]
    assert fixed["right"] != query["right"]


def test_apply_measured_moves_only_the_component_the_sentence_names() -> None:
    """문장이 고른 성분 하나만 옮긴다. 나머지 두 성분은 그대로다.

    문장은 후보 셋 중 절댓값이 가장 큰 하나만 말한다. 시뮬레이션이 세 성분을 다 옮기면
    말하지 않은 두 축의 교정까지 개선율에 계상된다.
    """
    query, reference = _three_point_case()
    before = measured_score.score_track(
        query, reference, SPREADS, measured_score.DEFAULT_WEIGHTS
    )
    finding = measured_score.right_hand_finding(before.deltas, HORIZONTAL)
    assert finding is not None

    fixed = measured_score.apply_measured(query, before.deltas, HORIZONTAL)
    moved = (
        fixed["right"][0][0] - query["right"][0][0],
        fixed["right"][0][1] - query["right"][0][1],
        fixed["right"][0][2] - query["right"][0][2],
    )
    world = measured_score._world_delta(moved, HORIZONTAL)

    # HORIZONTAL은 접선이 월드 Y, up이 월드 Z, normal이 월드 X다.
    axis_index = {"wall": 0, "lateral": 1, "vertical": 2}[finding.axis]
    assert math.isclose(
        world[axis_index] * 100.0, finding.magnitude_cm, abs_tol=1e-6
    )
    for other in range(3):
        if other != axis_index:
            assert math.isclose(world[other], 0.0, abs_tol=1e-9)


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


def test_apply_measured_does_nothing_when_no_sentence_is_issued() -> None:
    """오른손이 임계 아래면 문장이 없다. 없는 지시를 개선율에 계상하지 않는다."""
    reference = {name: [(0.30, 0.00, 0.10)] * 5 for name in ("right", "head", "left")}
    query = dict(reference)
    query["head"] = [(0.55, 0.00, 0.10)] * 5
    before = measured_score.score_track(
        query, reference, SPREADS, measured_score.DEFAULT_WEIGHTS
    )
    assert measured_score.right_hand_finding(before.deltas, HORIZONTAL) is None
    fixed = measured_score.apply_measured(query, before.deltas, HORIZONTAL)
    assert fixed == {name: list(points) for name, points in query.items()}
