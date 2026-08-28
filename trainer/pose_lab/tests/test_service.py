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
    """교정에 쓴 분포로 다시 채점하면 향상이 자동 보장된다. 둘 다 보여야 한다.

    `split`은 자기 안에서 일관돼야 한다 — S0·S1을 같은(B) 잣대로 재야 잣대를 바꾼
    것만으로 향상이 생기지 않는다. `before_split`이 그 B 위에서 낸 교정 전 점수다.
    """
    result = service.improvement(
        before_same=0.4, after_split=0.55, after_same=0.9, before_split=0.3
    )
    assert math.isclose(result["split"], 0.3571428571, abs_tol=1e-9)
    assert math.isclose(result["same"], 0.8333333, abs_tol=1e-6)
    assert result["split"] != result["same"]


def test_improvement_leaves_split_none_when_not_computed() -> None:
    """held-out 레퍼런스가 없으면 분리 채점 값을 지어내지 않는다."""
    result = service.improvement(before_same=0.4, after_split=None, after_same=0.9)
    assert result["split"] is None
    assert math.isclose(result["same"], 0.8333333, abs_tol=1e-6)


def test_improvement_leaves_split_none_without_its_own_baseline() -> None:
    """`after_split`만 있고 `before_split`이 없으면(짝이 안 맞으면) 지어내지 않는다.

    `same`의 S0(`before_same`)를 대신 꿔다 쓰면 서로 다른 잣대로 S0·S1을 재는
    결함이 되돌아온다 — 반드시 자기 짝이 있어야 split을 낸다.
    """
    result = service.improvement(before_same=0.4, after_split=0.55, after_same=0.9)
    assert result["split"] is None


def test_undefined_gap_stays_none_and_does_not_become_zero() -> None:
    """이미 기준과 같으면 메울 격차가 없어 개선율이 정의되지 않는다.

    0.000으로 적으면 "메우지 못했다"로 읽힌다. pose_similarity.gap_closure는 그래서
    None을 내는데, 그것을 뒤에서 0.0으로 바꾸면 의도적인 침묵이 무너진다.
    """
    assert pose_similarity.gap_closure(1.0, 1.0) is None
    result = service.improvement(
        before_same=1.0, after_split=1.0, after_same=1.0, before_split=1.0
    )
    assert result["split"] is None
    assert result["same"] is None


def _vertical_frames(count: int) -> list[object]:
    """수직으로 올라가는 스트로크. 세 점의 상대 배치는 프레임마다 같다."""
    frames: list[object] = []
    for index in range(count):
        hit = (1.2, 4.5, 0.6 + 0.05 * index)
        frames.append(
            service.measured_pose.MeasuredFrame(
                index=index,
                time_sec=index * 0.0111,
                head=(0.6, 4.6, hit[2]),
                right=(0.7, 4.6, hit[2]),
                left=(0.65, 4.4, hit[2]),
                hit=hit,
                uv=(1024.0, 512.0),
                spraying=True,
            )
        )
    return frames


def test_last_sample_keeps_the_previous_tangent() -> None:
    """마지막 격자점에는 다음 프레임이 없다.

    월드 축으로 떨어뜨리면 그 표본의 진행·수직 축만 나머지와 다른 방향을 가리키고,
    자세가 전혀 변하지 않은 궤적에서도 마지막 표본이 다른 좌표를 낸다. 그 차이가
    평균 이탈에 섞여 들어간다.
    """
    tracks = service._local_tracks(_vertical_frames(5))
    right = tracks["right"]
    assert len(right) == 5
    last, previous = right[-1], right[-2]
    assert last is not None and previous is not None
    for axis in range(3):
        assert math.isclose(last[axis], previous[axis], abs_tol=1e-9)
