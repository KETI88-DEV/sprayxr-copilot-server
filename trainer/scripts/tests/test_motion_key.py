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
    """현행 pass_segmentation의 결함 회귀. abs가 수직 부호를 버려 둘이 같아졌다."""
    up = motion_key.signed_band((1.0, 0.0, 1.0))
    down = motion_key.signed_band((1.0, 0.0, -1.0))
    assert up == "diag_up"
    assert down == "diag_down"


def test_existing_bucket_still_collapses_them() -> None:
    """기존 함수는 그대로 두었음을 명시한다. 바꾸면 레퍼런스 아티팩트가 무효화된다."""
    degrees = math.degrees(math.asin(1.0 / math.sqrt(2.0)))
    assert pass_segmentation.elevation_band(degrees) == "diagonal"


def test_horizontal_keeps_no_vertical_sign() -> None:
    assert motion_key.signed_band((1.0, 0.2, 0.05)) == "horizontal"


def test_vertical_splits_by_direction() -> None:
    assert motion_key.signed_band((0.02, 0.0, 1.0)) == "vertical_up"
    assert motion_key.signed_band((0.02, 0.0, -1.0)) == "vertical_down"


def test_vertical_heading_is_none() -> None:
    """수직 스트로크는 수평 진행이 정의되지 않는다."""
    assert motion_key.heading_word(0.0, (0.02, 0.0, 1.0)) == "none"


def test_heading_is_relative_to_body_not_world() -> None:
    """작업자가 어디를 보고 서 있든 몸 기준 오른쪽으로 긋는 스트로크는 right다."""
    assert motion_key.heading_word(0.0, (0.0, 1.0, 0.0)) == "right"
    assert motion_key.heading_word(math.pi / 2, (-1.0, 0.0, 0.0)) == "right"


def test_uv_cell_quantizes() -> None:
    assert motion_key.uv_cell((0.0, 0.0), PARAMS) == (0, 0)
    assert motion_key.uv_cell((2047.0, 2047.0), PARAMS) == (7, 7)
    assert motion_key.uv_cell((1024.0, 512.0), PARAMS) == (4, 2)


def test_body_placement_separates_front_from_reached_right() -> None:
    """같은 대각선이라도 몸 정면과 오른쪽으로 뻗은 것은 다른 자세다."""
    head = (0.5, 4.7, 0.8)
    front = motion_key.body_placement(head, 0.0, (1.2, 4.7, 0.8), PARAMS)
    reached = motion_key.body_placement(head, 0.0, (0.6, 5.5, 0.8), PARAMS)
    assert front[0] == "center"
    assert reached[0] == "right"


def test_body_placement_reports_reach_and_height() -> None:
    head = (0.5, 4.7, 1.4)
    lateral, reach, height = motion_key.body_placement(head, 0.0, (1.2, 4.7, 0.3), PARAMS)
    assert lateral == "center"
    assert reach == "near"
    assert height == "low"


def test_motion_key_is_in_its_own_namespace() -> None:
    """기존 키와 문자열이 겹치면 다른 기준의 레퍼런스가 잘못 매칭된다."""
    key = motion_key.motion_key(
        displacement=(1.0, 0.0, 1.0),
        yaw=0.0,
        head=(0.5, 4.7, 0.8),
        hit=(1.2, 4.7, 0.8),
        uv=(1024.0, 512.0),
        params=PARAMS,
    )
    assert key.startswith("m1:")
    assert "diag_up" in key
    assert key != pass_segmentation.body_bucket_key(45.0, 0.8, "right")
