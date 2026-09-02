#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

motion_key = importlib.import_module("motion_key")
pass_segmentation = importlib.import_module("pass_segmentation")

PARAMS = motion_key.KeyParams()


def test_up_and_down_diagonals_get_different_bands() -> None:
    """현행 pass_segmentation의 결함 회귀. abs가 수직 부호를 버려 둘이 같아졌다.

    수직은 world_axes.vertical_of(=index 1, 유니티 Y-up)에서 온다. 수평은
    (index 0, index 2)의 노름이다. (1.0, ±1.0, 0.0)이면 수평=hypot(1,0)=1, 수직=±1
    이라 elevation=atan2(1,1)=45°=diagonal이고, 부호는 수직 성분의 부호를 그대로
    따른다.
    """
    up = motion_key.signed_band((1.0, 1.0, 0.0))
    down = motion_key.signed_band((1.0, -1.0, 0.0))
    assert up == "diag_up"
    assert down == "diag_down"


def test_existing_bucket_still_collapses_them() -> None:
    """기존 함수는 그대로 두었음을 명시한다. 바꾸면 레퍼런스 아티팩트가 무효화된다."""
    degrees = math.degrees(math.asin(1.0 / math.sqrt(2.0)))
    assert pass_segmentation.elevation_band(degrees) == "diagonal"


def test_horizontal_keeps_no_vertical_sign() -> None:
    assert motion_key.signed_band((1.0, 0.2, 0.05)) == "horizontal"


def test_vertical_splits_by_direction() -> None:
    """수직(index 1)이 크고 수평(index 0, 2)이 작으면 elevation≈90°=vertical이다."""
    assert motion_key.signed_band((0.02, 1.0, 0.0)) == "vertical_up"
    assert motion_key.signed_band((0.02, -1.0, 0.0)) == "vertical_down"


def test_vertical_heading_is_none() -> None:
    """수직 스트로크는 수평 진행이 정의되지 않는다."""
    assert motion_key.heading_word(0.0, (0.02, 1.0, 0.0)) == "none"


def test_heading_is_relative_to_body_not_world() -> None:
    """작업자가 어디를 보고 서 있든 몸 기준 오른쪽으로 긋는 스트로크는 right다.

    수평 성분은 world_axes.horizontal_of = (d[0], d[2])다. 첫 번째 픽스처는 옛
    (0,1,0)(옛 수평축 Y) 대신 (0,0,1)을 쓴다 — Y는 이제 수직이라 (0,1,0)은 수직
    스트로크로 분류돼 heading_word가 "none"을 낸다. yaw=0에서 forward=h0=0,
    right=h1=1이라 "right"다. 두 번째 픽스처는 d[1]=d[2]=0이라 수평 성분이 옛/새
    규약에서 값이 같아(둘 다 (−1,0)) 그대로 둔다.
    """
    assert motion_key.heading_word(0.0, (0.0, 0.0, 1.0)) == "right"
    assert motion_key.heading_word(math.pi / 2, (-1.0, 0.0, 0.0)) == "right"


def test_uv_cell_quantizes() -> None:
    assert motion_key.uv_cell((0.0, 0.0), PARAMS) == (0, 0)
    assert motion_key.uv_cell((2047.0, 2047.0), PARAMS) == (7, 7)
    assert motion_key.uv_cell((1024.0, 512.0), PARAMS) == (4, 2)


def test_body_placement_separates_front_from_reached_right() -> None:
    """같은 대각선이라도 몸 정면과 오른쪽으로 뻗은 것은 다른 자세다.

    head는 사람 키에 해당하는 값을 Y(수직)에 싣는다 — `head=(0.5, 4.7, 0.8)`은
    1라운드에서 body_placement의 수평 성분(=delta[0],delta[2])만 보고 "그대로
    둔다"고 판단했지만, 그러면 head의 절대 좌표 자체가 Y=4.7m인 사람이 된다(리뷰
    A6). head=(0.5, 0.8, 4.7)로 Y·Z를 맞바꾼다: X=0.5, Y=0.8(키), Z=4.7(벽을
    따라가는 임의 위치). front의 hit는 head와 같은 높이·Z에서 X만 +0.7 뗀
    (1.2, 0.8, 4.7) — 수평 성분 (delta[0],delta[2])=(0.7,0)은 그대로다. reached는
    "오른쪽으로 0.1, 앞으로(수평 Z) 0.8" 뻗은 위치이므로
    hit=head+(0.1, 0.0, 0.8)=(0.6, 0.8, 5.5)다.
    """
    head = (0.5, 0.8, 4.7)
    front = motion_key.body_placement(head, 0.0, (1.2, 0.8, 4.7), PARAMS)
    reached = motion_key.body_placement(head, 0.0, (0.6, 0.8, 5.5), PARAMS)
    assert front[0] == "center"
    assert reached[0] == "right"


def test_body_placement_reports_reach_and_height() -> None:
    """head는 사람 키에 해당하는 값을 Y(수직)에 싣는다: head=(0.5, 1.4, 4.7).
    hit도 낮은 높이를 Y에 싣고(0.3), Z(수평)는 head와 같게 두어 옆으로는 안
    뻗지 않되 X로 0.7 앞으로 뻗는다: hit=(1.2, 0.3, 4.7). 수평 성분은
    (delta[0], delta[2]) = (0.7, 0.0)으로 옛 (0.7,0.0)과 같아 center·near가 그대로
    나오고, height는 `world_axes.vertical_of(hit)=hit[1]=0.3<0.6`이라 low다.
    """
    head = (0.5, 1.4, 4.7)
    lateral, reach, height = motion_key.body_placement(head, 0.0, (1.2, 0.3, 4.7), PARAMS)
    assert lateral == "center"
    assert reach == "near"
    assert height == "low"


def test_motion_key_is_in_its_own_namespace() -> None:
    """기존 키와 문자열이 겹치면 다른 기준의 레퍼런스가 잘못 매칭된다.

    "diag_up"이 나오려면 수직(index 1)과 수평(index 0,2 노름)이 같은 크기여야
    한다(45°) — `test_up_and_down_diagonals_get_different_bands`와 같은 이유로
    (1.0, 1.0, 0.0)을 쓴다. head·hit도 `test_body_placement_separates_front_
    from_reached_right`의 "front" 쌍과 같은 이유로 Y(키)·Z(벽 위치)를 맞바꾼다 —
    Y=4.7m인 사람을 만들지 않는다(리뷰 A6).
    """
    key = motion_key.motion_key(
        displacement=(1.0, 1.0, 0.0),
        yaw=0.0,
        head=(0.5, 0.8, 4.7),
        hit=(1.2, 0.8, 4.7),
        uv=(1024.0, 512.0),
        params=PARAMS,
    )
    assert key.startswith("m1:")
    assert "diag_up" in key
    assert key != pass_segmentation.body_bucket_key(45.0, 0.8, "right")
