from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image
import pytest

from trainer.scripts.build_zone_guidance_profiles import (
    _cell_mean_um,
    _uses_blue_validity,
    build_profiles,
)


def _session(root: Path, name: str, rows: list[list[object]]) -> None:
    target = root / "raw" / "blasting" / "learning" / "mode_03" / name
    target.mkdir(parents=True)
    with (target / f"{name}.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["시간(초)", "분사", "Speed (Real)", "Distance (Real)", "Ray Texture Map Number", "Zone ID"]
        )
        writer.writerows(rows)
    (target / "Map01.png").write_bytes(b"height-map")


def test_builds_blasting_zone_profiles_from_nested_expert_csvs(tmp_path: Path) -> None:
    _session(
        tmp_path,
        "3_A_1",
        [
            [0.0, "Off", 999, 999, 1, "1"],
            [0.1, "On", 100, 300, 1, "1"],
            [0.2, "On", 120, 320, 1, "1"],
            [0.3, "On", 200, 400, 1, "2"],
        ],
    )
    _session(
        tmp_path,
        "3_B_1",
        [
            [0.1, "On", 140, 340, 1, "1"],
            [0.2, "On", 220, 420, 1, "2"],
            [0.3, "On", 999, 999, 1, "-1"],
        ],
    )

    outputs = build_profiles(tmp_path, process="blasting")

    assert outputs == [tmp_path / "generated" / "profiles" / "blasting" / "learning" / "mode_03.json"]
    profile = json.loads(outputs[0].read_text(encoding="utf-8"))
    assert profile["identity"] == {
        "content_name": "mode_03",
        "content_type": "learning",
        "mode_id": 3,
        "process": "blasting",
        "zone_id_policy": "legacy_expert",
    }
    assert profile["quality"]["session_count"] == 2
    assert profile["aggregate"]["zones"]["1:1"]["speed_mms"]["median"] == 125.0
    assert profile["aggregate"]["zones"]["1:2"]["distance_mm"]["median"] == 410.0
    assert profile["aggregate"]["macro_orders"][0]["order"] == [
        {"map_num": 1, "zone_id": "1"},
        {"map_num": 1, "zone_id": "2"},
    ]
    assert len(profile["sessions"][0]["heightmaps"]) == 1
    assert "\\" not in profile["sessions"][0]["sources"]["csv_path"]

    first_profile = outputs[0].read_bytes()
    first_audit = (tmp_path / "generated" / "audits" / "blasting_learning.json").read_bytes()
    build_profiles(tmp_path, process="blasting")
    assert outputs[0].read_bytes() == first_profile
    assert (tmp_path / "generated" / "audits" / "blasting_learning.json").read_bytes() == first_audit


def test_spray_mode_two_canonicalizes_reversed_expert_zone_ids(tmp_path: Path) -> None:
    target = tmp_path / "raw" / "spray" / "learning" / "mode_02" / "2_1001_1"
    target.mkdir(parents=True)
    with (target / "2_1001_1.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["시간(초)", "분사", "Speed (Real)", "Distance (Real)", "Ray Texture Map Number", "Zone ID"]
        )
        writer.writerows(
            [
                [0.1, "On", 100, 300, 1, "1"],
                [0.2, "On", 200, 400, 1, "3"],
            ]
        )

    [output] = build_profiles(tmp_path, process="spray")
    profile = json.loads(output.read_text(encoding="utf-8"))

    assert profile["aggregate"]["zones"]["1:3"]["speed_mms"]["median"] == 100
    assert profile["aggregate"]["zones"]["1:1"]["speed_mms"]["median"] == 200
    assert profile["aggregate"]["macro_orders"][0]["order"] == [
        {"map_num": 1, "zone_id": "3"},
        {"map_num": 1, "zone_id": "1"},
    ]


def test_new_worker_preserves_official_source_zone_ids(tmp_path: Path) -> None:
    target = tmp_path / "raw/spray/learning/mode_02/2_2001_1"
    target.mkdir(parents=True)
    with (target / "2_2001_1.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["시간(초)", "분사", "Speed (Real)", "Distance (Real)", "Ray Texture Map Number", "Zone ID"])
        writer.writerows([[0.1, "On", 100, 300, 1, "1"], [0.2, "On", 200, 400, 1, "3"]])
    [output] = build_profiles(tmp_path, process="spray", legacy_expert_zone_ids=False)
    profile = json.loads(output.read_text(encoding="utf-8"))
    assert profile["identity"]["zone_id_policy"] == "source"
    assert profile["aggregate"]["zones"]["1:1"]["speed_mms"]["median"] == 100
    assert profile["aggregate"]["zones"]["1:3"]["speed_mms"]["median"] == 200

def test_spray_profile_learns_zone_coating_targets_from_csv_uv_and_final_heightmap(
    tmp_path: Path,
) -> None:
    target = tmp_path / "raw" / "spray" / "learning" / "mode_03" / "3_A_1"
    target.mkdir(parents=True)
    with (target / "3_A_1.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "시간(초)",
                "분사",
                "Speed (Real)",
                "Distance (Real)",
                "Ray Texture Map Number",
                "Ray Texture UV X",
                "Ray Texture UV Y",
                "Zone ID",
            ]
        )
        writer.writerows(
            [
                [0.1, "On", 100, 300, 1, 0, 0, "1"],
                [0.2, "On", 100, 300, 1, 3, 3, "1"],
            ]
        )
    # PNG top row is high V. Both visited corners carry the same 100um target.
    raw = round(100.0 / 20000.0 * 65025.0)
    image = Image.new("RGBA", (4, 4), (raw // 255, raw % 255, 255, 255))
    image.save(target / "Map01_3_A_1_1_Height.png")

    outputs = build_profiles(
        tmp_path,
        process="spray",
        coating_grid_size=2,
        zone_reference_root=tmp_path / "no-zone-reference",
    )

    profile = json.loads(outputs[0].read_text(encoding="utf-8"))
    coating = profile["aggregate"]["zones"]["1:1"]["coating"]
    assert coating["analysis_status"] == "available"
    assert coating["atlas_width"] == 4
    assert coating["atlas_height"] == 4
    assert coating["grid_size"] == 2
    assert coating["cells"]["0:0"]["sample_count"] == 1
    assert coating["cells"]["0:0"]["q1_um"] == pytest.approx(99.961553)
    assert coating["cells"]["1:0"]["median_um"] == pytest.approx(99.961553)
    assert coating["cells"]["0:0"]["q3_um"] == pytest.approx(99.961553)
    assert profile["quality"]["heightmap_scale_status"] == "rg16_scale_20000"


def test_spray_profile_uses_official_zone_mask_and_blue_validity_channel(
    tmp_path: Path,
) -> None:
    target = tmp_path / "raw" / "spray" / "learning" / "mode_04" / "4_A_1"
    target.mkdir(parents=True)
    with (target / "4_A_1.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "시간(초)", "분사", "Speed (Real)", "Distance (Real)",
                "Ray Texture Map Number", "Ray Texture UV X", "Ray Texture UV Y", "Zone ID",
            ]
        )
        writer.writerow([0.1, "On", 100, 300, 1, 0, 0, "1"])

    raw_100 = round(100.0 / 20000.0 * 65025.0)
    raw_500 = round(500.0 / 20000.0 * 65025.0)
    image = Image.new("RGB", (4, 4), (raw_500 // 255, raw_500 % 255, 255))
    for index in range(4):
        image.putpixel((index, index), (raw_100 // 255, raw_100 % 255, 255))
    image.putpixel((0, 0), (raw_500 // 255, raw_500 % 255, 0))
    image.save(target / "Map01_4_A_1_1_Height.png")

    reference = tmp_path / "zone_reference"
    (reference / "color").mkdir(parents=True)
    (reference / "palette.csv").write_text(
        "zone_id,hex,r,g,b\n1,#FF0000,255,0,0\n2,#00FF00,0,255,0\n",
        encoding="utf-8",
    )
    (reference / "scenario_map.csv").write_text(
        "scenario_id,model_key,note\n4,Learn04,학습\n",
        encoding="utf-8",
    )
    (reference / "zone_rects.csv").write_text(
        "model_key,map_num,zone_id,rect_x,rect_y,rect_w,rect_h,px_x,px_y,px_w,px_h,u_min,v_min,u_max,v_max,area_percent\n"
        "Learn04,01,1,0,0,1,1,0,0,4,4,0,0,1,1,25\n",
        encoding="utf-8",
    )
    label = Image.new("RGB", (4, 4), (0, 255, 0))
    for index in range(4):
        label.putpixel((index, index), (255, 0, 0))
    label.save(reference / "color" / "Learn04_01.png")

    [output] = build_profiles(
        tmp_path,
        process="spray",
        coating_grid_size=2,
        zone_reference_root=reference,
    )

    profile = json.loads(output.read_text(encoding="utf-8"))
    coating = profile["aggregate"]["zones"]["1:1"]["coating"]
    assert coating["mask_source"] == "tsb_color_label_map"
    assert coating["display_semantics"] == "physical_thickness_um"
    assert set(coating["cells"]) == {"0:1", "1:0"}
    assert coating["cells"]["0:1"]["median_um"] == pytest.approx(99.961553)
    assert coating["cells"]["1:0"]["median_um"] == pytest.approx(99.961553)


def test_blasting_profile_does_not_invent_a_coating_thickness_model(tmp_path: Path) -> None:
    _session(tmp_path, "3_A_1", [[0.1, "On", 100, 300, 1, "1"]])

    output = build_profiles(tmp_path, process="blasting")[0]
    profile = json.loads(output.read_text(encoding="utf-8"))

    assert profile["aggregate"]["zones"]["1:1"]["coating"] == {
        "analysis_status": "not_applicable",
        "reason": "coating_not_applicable",
    }


def test_legacy_expert_heightmap_without_blue_channel_uses_nonblack_pixels() -> None:
    image = Image.new("RGB", (2, 2), (0, 200, 0))
    mask = Image.new("L", (2, 2), 255)

    assert _uses_blue_validity(image) is False
    assert _cell_mean_um(
        image,
        0,
        0,
        1,
        zone_mask=mask,
        blue_is_validity=False,
    ) > 0


def test_expert_orders_put_the_decided_flow_first_and_replace_transitions(tmp_path: Path) -> None:
    """반장 한 사람의 raw 세션 순서가 아니라, 전문가 전체에서 정한 권장 순서가 `macro_orders[0]`이 된다(엔진은 [0]을 쓴다).

    깨뜨리는 법: 기준 JSON 을 무시하면 [0]이 raw 순서(3 → 1)로 남는다.
    """
    target = tmp_path / "raw" / "spray" / "learning" / "mode_02" / "2_1001_1"
    target.mkdir(parents=True)
    with (target / "2_1001_1.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["시간(초)", "분사", "Speed (Real)", "Distance (Real)", "Ray Texture Map Number", "Zone ID"])
        writer.writerows([[0.1, "On", 100, 300, 1, "1"], [0.2, "On", 200, 400, 1, "3"]])
    orders = tmp_path / "orders.json"
    orders.write_text(json.dumps({"scenarios": {"2": {
        "file_count": 48, "recommended_flow": ["1", "2", "3"], "flow_file_count": 18, "decision": "최빈",
        "transitions": [{"from": "1", "to": "2", "count": 30}, {"from": "2", "to": "3", "count": 25}],
    }}}), encoding="utf-8")

    [output] = build_profiles(tmp_path, process="spray", expert_orders_path=orders)
    aggregate = json.loads(output.read_text(encoding="utf-8"))["aggregate"]

    assert [item["zone_id"] for item in aggregate["macro_orders"][0]["order"]] == ["1", "2", "3"]
    assert aggregate["macro_orders"][0]["source"] == "expert_zone_orders"
    assert [item["zone_id"] for item in aggregate["macro_orders"][1]["order"]] == ["3", "1"]  # raw 순서는 뒤에 남는다
    assert [(t["from"]["zone_id"], t["to"]["zone_id"], t["session_count"]) for t in aggregate["transitions"]] == [
        ("1", "2", 30), ("2", "3", 25),
    ]


def test_the_shipped_expert_orders_cover_learning_levels_one_to_ten() -> None:
    from trainer.scripts.build_zone_guidance_profiles import EXPERT_ZONE_ORDERS, _expert_orders

    orders = _expert_orders(EXPERT_ZONE_ORDERS)

    assert sorted(orders, key=int) == [str(level) for level in range(1, 11)]
    assert orders["3"]["recommended_flow"] == ["7", "4", "8", "5", "6", "3", "2", "1", "9"]  # 2026-10-16 반장 순서
