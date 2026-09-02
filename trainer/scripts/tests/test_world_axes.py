#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

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
