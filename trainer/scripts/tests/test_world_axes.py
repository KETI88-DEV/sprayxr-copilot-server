#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

world_axes = importlib.import_module("world_axes")


def test_unity_convention_is_y_up_left_handed() -> None:
    """유니티다. Y가 위, 수평면은 (X, Z), 왼손 좌표계."""
    assert world_axes.UP_INDEX == 1
    assert world_axes.UP == (0.0, 1.0, 0.0)
    assert world_axes.HORIZONTAL_INDICES == (0, 2)
    assert world_axes.HANDEDNESS == "left"


def test_vertical_and_horizontal_split_the_vector() -> None:
    assert world_axes.vertical_of((1.0, 2.0, 3.0)) == 2.0
    assert world_axes.horizontal_of((1.0, 2.0, 3.0)) == (1.0, 3.0)


def test_flatten_removes_only_the_vertical_component() -> None:
    assert world_axes.flatten((1.0, 2.0, 3.0)) == (1.0, 0.0, 3.0)


def test_yaw_is_measured_in_the_horizontal_plane() -> None:
    """수평 두 성분의 atan2다. 전방이 +X면 0, +Z면 90도."""
    assert world_axes.yaw_of((1.0, 0.0, 0.0)) == 0.0
    assert math.isclose(world_axes.yaw_of((0.0, 0.0, 1.0)), math.pi / 2)
    # 수직 성분은 요에 영향을 주지 않는다.
    assert math.isclose(world_axes.yaw_of((1.0, 5.0, 0.0)), 0.0)


def test_yaw_is_none_when_there_is_no_horizontal_component() -> None:
    """정확히 위나 아래를 보면 방위가 정의되지 않는다. 0으로 지어내지 않는다."""
    assert world_axes.yaw_of((0.0, 1.0, 0.0)) is None
    assert world_axes.yaw_of((0.0, -1.0, 0.0)) is None


def test_sign_toward_picks_the_direction_facing_the_reference_point() -> None:
    """왼손 좌표계라 외적의 방향 감각이 오른손계와 반대다. 부호는 기준점으로 정한다."""
    assert world_axes.sign_toward((1.0, 0.0, 0.0), (5.0, 0.0, 0.0)) == (1.0, 0.0, 0.0)
    assert world_axes.sign_toward((1.0, 0.0, 0.0), (-5.0, 0.0, 0.0)) == (-1.0, 0.0, 0.0)


def test_patching_up_index_alone_moves_every_reader() -> None:
    """UP_INDEX가 유일한 손잡이어야 한다.

    UP과 HORIZONTAL_INDICES가 UP_INDEX와 별도인 리터럴이면, UP_INDEX만 몽키패치했을
    때 그 둘과 vertical_of/horizontal_of/flatten/yaw_of가 따라오지 않는다 — 이
    모듈이 막으려던 결함의 축소판이다. 여기서는 UP_INDEX 하나만 바꾸고 나머지 전부가
    새 축을 따라가는지 검사한다.
    """
    original = world_axes.UP_INDEX
    try:
        world_axes.UP_INDEX = 2  # 수직축을 Z로 옮긴다.
        assert world_axes.UP == (0.0, 0.0, 1.0)
        assert world_axes.HORIZONTAL_INDICES == (0, 1)
        assert world_axes.vertical_of((1.0, 2.0, 3.0)) == 3.0
        assert world_axes.horizontal_of((1.0, 2.0, 3.0)) == (1.0, 2.0)
        assert world_axes.flatten((1.0, 2.0, 3.0)) == (1.0, 2.0, 0.0)
        assert math.isclose(world_axes.yaw_of((1.0, 0.0, 0.0)), 0.0)
        assert math.isclose(world_axes.yaw_of((0.0, 1.0, 0.0)), math.pi / 2)
    finally:
        world_axes.UP_INDEX = original


def test_frame_around_is_orthonormal_for_a_horizontal_forward() -> None:
    frame = world_axes.frame_around((0.0, 0.0, 1.0))
    axes = (frame.forward, frame.up, frame.third)
    for axis in axes:
        assert math.isclose(math.dist(axis, (0.0, 0.0, 0.0)), 1.0, abs_tol=1e-9)
    for i in range(3):
        for j in range(i + 1, 3):
            dot = sum(axes[i][k] * axes[j][k] for k in range(3))
            assert math.isclose(dot, 0.0, abs_tol=1e-9)
    # 접선이 이미 수평이라 위 축 직교화는 무너지지 않는다 — 월드 수직축 그대로다.
    assert frame.up == (0.0, 1.0, 0.0)


def test_frame_around_falls_back_to_a_horizontal_up_when_forward_is_vertical() -> None:
    """접선이 월드 수직축과 평행하면(순수 수직 방향) 직교화가 영벡터다.

    수평 대체축으로 up을 다시 세운다 — 이름은 'up'이지만 이 경로에서 값은 수평이다.
    """
    frame = world_axes.frame_around((0.0, 1.0, 0.0))
    assert frame.up == (1.0, 0.0, 0.0)


def test_frame_around_rejects_a_zero_forward() -> None:
    """영벡터를 대체값 없이 조용히 삼키지 않는다 — 호출자가 자기 도메인의 퇴화를
    먼저 해소해야 한다(예: 제자리 프레임의 대체 접선)."""
    with pytest.raises(ValueError):
        world_axes.frame_around((0.0, 0.0, 0.0))


def test_frame_around_fallback_moves_with_up_index() -> None:
    """퇴화 대체축도 UP_INDEX 하나로 따라와야 한다 — 고정된 (1,0,0) 리터럴이 아니다."""
    original = world_axes.UP_INDEX
    try:
        world_axes.UP_INDEX = 2  # 수직축을 Z로 옮긴다. 수평 축은 이제 (X, Y)다.
        frame = world_axes.frame_around((0.0, 0.0, 1.0))  # forward ∥ 새 수직축.
        assert frame.up == (1.0, 0.0, 0.0)
    finally:
        world_axes.UP_INDEX = original


def test_every_module_follows_the_convention_not_a_literal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UP_INDEX를 바꾸면 네 소비 모듈의 출력이 **전부** 따라 움직여야 한다.

    `world_axes`는 `UP`과 `HORIZONTAL_INDICES`를 모듈 `__getattr__`로 `UP_INDEX`
    하나에서 매번 다시 계산한다(PEP 562) — 그래서 손잡이는 `UP_INDEX` 단 하나다.
    이 테스트는 그 손잡이 하나만 돌려서 네 모듈(`measured_score`, `stroke_frame`,
    `measured_pose`, `motion_key`)이 전부 반응하는지 본다.

    입력은 세 축 모두에 0이 아닌 뚜렷한 성분을 갖도록 골랐고, Y-up과 Z-up이 실제로
    다른 답을 낸다는 것을 이 파일을 쓰면서 직접 실행해 확인했다 — 우연히 같은 답이
    나오는 벡터를 썼다면 이 테스트는 아무것도 증명하지 않으면서 통과했을 것이다.

    하나라도 안 움직이면 그 모듈이 리터럴을 다시 박았거나 임포트 시점에 값을
    복사한 것이다. 그 상태는 지금 우연히 옳은 답을 낼 수 있어서 눈으로는 안 보인다.
    """
    measured_score = importlib.import_module("measured_score")
    stroke_frame = importlib.import_module("stroke_frame")
    measured_pose = importlib.import_module("measured_pose")
    motion_key = importlib.import_module("motion_key")

    def sample() -> dict[str, object]:
        basis = stroke_frame.stroke_basis(
            hit=(1.2, 4.5, 0.8), tangent=(0.6, 0.0, 0.8), gun=(0.6, 4.9, -0.3)
        )
        return {
            "axis": measured_score._axis_components((0.11, -0.04, 0.07), basis),
            "basis": (basis.normal, basis.tangent, basis.up),
            "yaw": measured_pose.facing_yaw((17.0, 32.0, -11.0)),
            "heading": motion_key.heading_word(0.3, (0.4, -0.6, 0.5)),
        }

    before = sample()
    monkeypatch.setattr(world_axes, "UP_INDEX", 2)  # 유일한 손잡이만 돌린다.
    after = sample()

    stuck = [k for k in before if before[k] == after[k]]
    assert not stuck, f"규약을 따르지 않는 항목: {stuck}"
