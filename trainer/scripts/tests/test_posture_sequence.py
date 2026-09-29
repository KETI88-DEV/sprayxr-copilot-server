#!/usr/bin/env python3
"""자세 시퀀스 검사.

핵심은 둘이다 — **바닥 상수가 기울기를 바꾸지 않는다**는 것과, **레퍼런스가 없을 때
점수가 `0.0`이 아니라 `null`**이라는 것.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
for path in (str(_REPO_ROOT), str(_SCRIPTS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

posture_sequence = importlib.import_module("posture_sequence")
world_axes = importlib.import_module("world_axes")


def _up(height: float, x: float = 0.0) -> tuple[float, float, float]:
    """월드 수직 성분이 `height`인 점. 축은 `world_axes`가 정한다."""
    out = [x, 0.0, 0.0]
    out[world_axes.UP_INDEX] = height
    return (out[0], out[1], out[2])


def _session(pairs, session_id: str = "S1"):
    session = posture_sequence.PostureSession(session_id=session_id)
    for index, (impact, head) in enumerate(pairs):
        session.add(
            posture_sequence.PostureSample(
                sequence_id=f"SEQ-{index}",
                head_height_m=head,
                impact_height_m=impact,
                frames=9,
                gate_status="pass",
            )
        )
    return session


def test_summarize_reduces_a_sequence_to_one_posture_point() -> None:
    """시퀀스 안에서 몸이 고정이므로 한 점으로 줄이는 것은 충실한 요약이다."""
    heads = [_up(4.20), _up(4.21), _up(4.20)]
    impacts = [_up(4.80), _up(4.90), _up(5.00)]
    result = posture_sequence.summarize(heads, impacts)
    assert result is not None
    head, impact = result
    assert head == pytest.approx(4.20)
    assert impact == pytest.approx(4.90)


def test_a_missed_sequence_has_no_impact_height() -> None:
    """착탄점이 없으면 `None`이다. 0.0으로 채우면 '바닥을 칠했다'가 된다.

    실물 SEQ-001이 그 경우다(`raypoint: null`, `sequence_result: {}`).
    """
    result = posture_sequence.summarize([_up(4.2)], [])
    assert result == (pytest.approx(4.2), None)


def test_summarize_reads_the_vertical_axis_at_call_time(monkeypatch) -> None:
    """축을 리터럴 인덱스로 박으면 규약이 바뀌어도 반응하지 않는다.

    깨뜨리는 법: `_vertical`을 `point[1]`로 바꾸면 이 검사가 실패한다.
    """
    point = (7.0, 8.0, 9.0)
    before = posture_sequence.summarize([point])
    monkeypatch.setattr(world_axes, "UP_INDEX", 2)
    after = posture_sequence.summarize([point])
    assert before is not None and after is not None
    assert before[0] != after[0]
    assert after[0] == pytest.approx(9.0)


def test_tracking_slope_matches_the_measured_expert_value() -> None:
    """착탄이 1 m 오를 때 몸이 0.654 m 오르는 관계를 넣으면 그 값이 나와야 한다.

    세션 427개 실측 중앙값이 +0.654(r 0.864)다.
    """
    pairs = [(4.0 + 0.2 * i, 3.9 + 0.654 * 0.2 * i) for i in range(8)]
    slope = _session(pairs).tracking_slope()
    assert slope is not None
    assert slope[0] == pytest.approx(0.654, abs=1e-6)
    assert slope[1] == pytest.approx(1.0, abs=1e-6)
    assert slope[2] == 8


def test_a_constant_floor_offset_does_not_change_the_slope() -> None:
    """바닥은 레벨당 상수다. 상수를 빼도 기울기는 그대로여야 한다 — 그래서 바닥을 몰라도 잰다.

    이 성질이 깨지면 바닥을 알기 전에는 기울기를 낼 수 없게 된다.
    """
    pairs = [(4.0 + 0.2 * i, 3.9 + 0.5 * 0.2 * i) for i in range(8)]
    raw = _session(pairs).tracking_slope()
    shifted = _session([(x, y - 3.557) for x, y in pairs]).tracking_slope()
    assert raw is not None and shifted is not None
    assert raw[0] == pytest.approx(shifted[0])


def test_a_worker_who_never_moves_has_a_zero_slope() -> None:
    """몸을 그대로 두고 팔로만 처리하면 기울기가 0이다 — 이 항이 잡으려는 실패 모드다."""
    pairs = [(4.0 + 0.2 * i, 4.10) for i in range(8)]
    slope = _session(pairs).tracking_slope()
    assert slope is None or slope[0] == pytest.approx(0.0, abs=1e-9)


def test_slope_needs_enough_samples() -> None:
    """표본이 모자라면 값을 만들지 않는다. 두 점으로도 직선은 그어지지만 그건 측정이 아니다."""
    assert _session([(4.0, 3.9), (4.5, 4.2)]).tracking_slope() is None


def test_sequences_with_no_impact_are_excluded_from_the_slope() -> None:
    """미명중 시퀀스는 자세 점은 있지만 착탄 높이가 없다 — 회귀에서 빠져야 한다."""
    session = _session([(4.0 + 0.2 * i, 3.9 + 0.1 * i) for i in range(6)])
    session.add(
        posture_sequence.PostureSample(
            sequence_id="MISS",
            head_height_m=9.9,
            impact_height_m=None,
            frames=7,
            gate_status="reject",
        )
    )
    slope = session.tracking_slope()
    assert slope is not None
    assert slope[2] == 6, "착탄 없는 시퀀스가 회귀에 섞였다"


def test_posture_band_separates_stance_from_locomotion() -> None:
    """학습모드 77,708패스가 전부 바닥 기준 0.346~1.883 m 안이고 1.9 m 초과가 0건이다.

    실전모드는 절대 높이 4.094~7.388 m를 오간다 — 그건 자세가 아니라 발판이다.
    """
    floor = 3.557
    assert posture_sequence.in_posture_band(floor + 1.10, floor) is True
    assert posture_sequence.in_posture_band(floor + 2.60, floor) is False


def test_posture_band_declines_to_judge_without_a_floor() -> None:
    """모르는 것과 통과는 다르다. 모르면서 통과로 적으면 이동 구간을 자세로 읽는다."""
    assert posture_sequence.in_posture_band(4.6, None) is None
    assert posture_sequence.in_posture_band(4.6, 99.0) is None


def test_frozen_joints_accumulate_across_sequences() -> None:
    """R4c가 쓸 집합이다 — 두 번 얼면 그 관절이 안 오는 것이다."""
    session = posture_sequence.PostureSession("S")
    sample = posture_sequence.PostureSample("SEQ-1", 4.2, 4.8, 7, "degrade")
    session.add(sample, frozen=("Head",))
    session.add(sample, frozen=("Head", "Spine"))
    assert session.frozen_joints == {"Head", "Spine"}


def test_the_store_evicts_the_least_recently_used_session() -> None:
    """런타임 서비스라 경계가 있어야 한다. 버린 사실도 세어 둔다."""
    store = posture_sequence.SessionStore(max_sessions=2)
    store.get_or_create("A")
    store.get_or_create("B")
    store.get("A")  # A를 최근 사용으로 올린다
    store.get_or_create("C")
    assert len(store) == 2
    assert store.get("B") is None, "가장 오래 안 쓴 것이 아니라 다른 것을 버렸다"
    assert store.get("A") is not None
    assert store.evicted == 1


def test_samples_are_bounded_per_session() -> None:
    session = posture_sequence.PostureSession("S")
    sample = posture_sequence.PostureSample("SEQ", 4.2, 4.8, 7, "pass")
    for _ in range(posture_sequence.MAX_SAMPLES_PER_SESSION + 5):
        session.add(sample)
    assert len(session.samples) == posture_sequence.MAX_SAMPLES_PER_SESSION


def test_report_withholds_the_score_until_references_are_rebuilt() -> None:
    """레퍼런스가 없을 때 점수는 **null이지 0.0이 아니다.**

    0.0은 '많이 다르다'로 읽히고, 그건 우리가 재지 않은 것에 대한 주장이다.
    키 집합은 레퍼런스가 준비된 세계와 같고 값만 다르다.
    """
    report = posture_sequence.session_report(
        _session([(4.0 + 0.2 * i, 3.9 + 0.13 * i) for i in range(6)])
    )
    assert report["similarity_score"] is None
    assert report["reference_quality"] == "unavailable"
    assert report["reason_code"] == "reference_rebuild_pending"
    assert report["tracking_slope"] is not None, "재지 않고 미루기만 하면 안 된다"
    assert report["sequences"] == 6
