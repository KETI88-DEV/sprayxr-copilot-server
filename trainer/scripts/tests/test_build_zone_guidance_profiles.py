from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image
import pytest

from trainer.scripts.build_zone_guidance_profiles import build_profiles


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
    image = Image.new("RGBA", (4, 4), (raw // 255, raw % 255, 0, 255))
    image.save(target / "Map01_3_A_1_1_Height.png")

    outputs = build_profiles(tmp_path, process="spray", coating_grid_size=2)

    profile = json.loads(outputs[0].read_text(encoding="utf-8"))
    coating = profile["aggregate"]["zones"]["1:1"]["coating"]
    assert coating["analysis_status"] == "available"
    assert coating["atlas_width"] == 4
    assert coating["atlas_height"] == 4
    assert coating["grid_size"] == 2
    assert coating["cells"]["0:0"]["sample_count"] == 1
    assert coating["cells"]["0:0"]["q1_um"] == pytest.approx(99.961553)
    assert coating["cells"]["0:0"]["median_um"] == pytest.approx(99.961553)
    assert coating["cells"]["0:0"]["q3_um"] == pytest.approx(99.961553)
    assert profile["quality"]["heightmap_scale_status"] == "rg16_scale_20000"


def test_blasting_profile_does_not_invent_a_coating_thickness_model(tmp_path: Path) -> None:
    _session(tmp_path, "3_A_1", [[0.1, "On", 100, 300, 1, "1"]])

    output = build_profiles(tmp_path, process="blasting")[0]
    profile = json.loads(output.read_text(encoding="utf-8"))

    assert profile["aggregate"]["zones"]["1:1"]["coating"] == {
        "analysis_status": "not_applicable",
        "reason": "coating_not_applicable",
    }
