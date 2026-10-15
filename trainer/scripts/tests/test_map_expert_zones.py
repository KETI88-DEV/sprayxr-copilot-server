"""전문가 CSV 구역 붙이기: 착탄 UV → 라벨맵 구역, 처음 칠한 순서."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from trainer.scripts.map_expert_zones import first_visit_flow, learning_scenario, recommended_flow
from trainer.zone_reference import ZoneReferenceCatalog


def test_impact_uv_reads_the_zone_with_a_bottom_left_origin() -> None:
    """UV는 좌하단 원점 픽셀이다. 깨뜨리는 법: v를 뒤집지 않으면 다른 구역(또는 밖)을 읽는다."""
    catalog = ZoneReferenceCatalog.default()
    mask = np.asarray(catalog.zone_mask("Learn03", 1, "5")) > 0  # 좌상단 원점
    rows, columns = np.nonzero(mask)
    y, x = int(rows[len(rows) // 2]), int(columns[len(columns) // 2])

    assert catalog.zone_at("Learn03", 1, x + 0.4, mask.shape[0] - 1 - y + 0.4) == "5"
    assert catalog.zone_at("Learn03", 1, 0, 0) is None  # 구역 밖(검정)
    assert catalog.zone_at("Learn03", 1, 5000, 10) is None  # 텍스처 밖
    assert catalog.zone_at("Learn03", 9, 100, 100) is None  # 라벨맵 없는 면


def test_first_visit_flow_keeps_first_order_and_drops_unjudged() -> None:
    assert first_visit_flow(["9", "9", "-1", "8", "9", "7", "8"]) == ["9", "8", "7"]


def test_only_learning_levels_one_to_ten_are_mapped() -> None:
    assert learning_scenario(Path("3_1000096_1.csv")) == "3"
    assert learning_scenario(Path("10_1000001_1.csv")) == "10"
    assert learning_scenario(Path("1-A_1000063_1.csv")) is None
    assert learning_scenario(Path("11_1_1.csv")) is None


def test_recommended_flow_is_the_most_common_complete_order_with_the_lead_breaking_ties() -> None:
    """가장 많이 나온 순서. 구역이 빠진 순서는 후보가 아니다. 같으면 전형기 반장의 순서.

    깨뜨리는 법: 빠진 순서를 후보로 두면 [1,2](3회)가 뽑히고, 동률을 먼저 나온 순서로 풀면 [2,1,3]이 뽑힌다.
    """
    partial = [("김경호", ["1", "2"])] * 3
    tied = [("김경호", ["2", "1", "3"]), ("손영석", ["2", "1", "3"]), ("전형기", ["1", "2", "3"]), ("임채원", ["1", "2", "3"])]

    assert recommended_flow(partial + tied, 3) == (["1", "2", "3"], 2, "동률 2개 → 전형기 반장")
    assert recommended_flow(partial + tied[:2], 3) == (["2", "1", "3"], 2, "최빈")
    assert recommended_flow(partial, 3) == (["1", "2"], 3, "최빈(전 구역 순서 없음)")
