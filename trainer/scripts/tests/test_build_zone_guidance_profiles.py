from __future__ import annotations

import csv
import json
from pathlib import Path

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
