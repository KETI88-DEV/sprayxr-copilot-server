#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import csv
import importlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

parse_session = importlib.import_module("parse_session")
session_csv = importlib.import_module("session_csv")

OBJECT_COLUMNS = ["한길숏건", "한길폴건", "반도숏건", "반도폴건", "플래쉬", "카메라"]
JOINT_COLUMNS = [
    "Pelvis", "Spine", "Chest",
    "Left Shoulder", "Left Upper Arm", "Left Fore Arm", "Left Hand",
    "Right Shoulder", "Right Upper Arm", "Right Fore Arm", "Right Hand",
    "Left Thigh", "Left Calf", "Left Foot", "Left Toe",
    "Right Thigh", "Right Calf", "Right Foot", "Right Toe",
    "Head",
]
# The vendor export ships " Ray Point Y" / " Ray Point Z" with a leading space.
TAIL_COLUMNS = [
    "Record", "Distance (Input)", "Speed (Input)", "Distance (Real)", "Speed (Real)",
    "Model Input Data", "Ray point X", " Ray Point Y", " Ray Point Z",
    "Ray Texture Map Number", "Ray Texture UV X", "Ray Texture UV Y",
]
MODEL_INPUT = "thick@251.03@527@1600@360@550@0.01111111@72@9.6@29@JOTUN@30@15@10"


def _header() -> list[str]:
    # utf-8-sig on write supplies the BOM the vendor export carries.
    columns = ["인덱스", "시간(초)", "분사"]
    for group in (*OBJECT_COLUMNS, *JOINT_COLUMNS):
        for kind in ("Position", "Rotation"):
            columns.extend(f"{group} {kind} {axis}" for axis in "XYZ")
    columns.extend(TAIL_COLUMNS)
    return columns


ACTIVE_GUN = "한길숏건"


def _row(index: int, *, spray: str, ray: tuple[str, str, str, str, str, str]) -> list[str]:
    values = [str(index), f"{index * 0.01111111:.6f}", spray]
    for offset, group in enumerate((*OBJECT_COLUMNS, *JOINT_COLUMNS)):
        drift = index * 0.5 if group == ACTIVE_GUN else 0.0
        values.extend([f"{offset + axis / 10 + drift:.4f}" for axis in range(3)])
        values.extend([f"{offset * 2 + axis:.4f}" for axis in range(3)])
    values.extend(["Off", "550", "1600", "440.3471", "3299", MODEL_INPUT, *ray])
    return values


def write_csv(tmp_path: Path, rows: list[list[str]], stem: str = "1_1000442_1") -> Path:
    csv_path = tmp_path / f"{stem}.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(_header())
        writer.writerows(rows)
    (tmp_path / f"{stem}.txt").write_text(
        "VarnishID=2\nVarnishColorID=1\nNozzleID=3\nMonth=7\nPressure=360\nThinnerRate=9.6\nModel=Exam\n",
        encoding="utf-8",
    )
    return csv_path


HIT = ("1.4365", "3.5702", "0.5166", "01", "600.9111", "36.71369")
MISS = ("", "", "", "", "", "")


def test_ray_columns_become_surface_hit_on_hit_frames(tmp_path: Path) -> None:
    csv_path = write_csv(tmp_path, [_row(0, spray="On", ray=HIT), _row(1, spray="On", ray=MISS)])

    converted = session_csv.convert_csv(csv_path, session_csv.WorkerProfile("expert", "right"))

    frames = converted.sequences[0]["frames"]
    assert frames[0]["surface_hit"] == {
        "hit_pos_m": [1.4365, 3.5702, 0.5166],
        "region_id": "MAP01",
        "map_number": "01",
        "uv_texel": [600.9111, 36.71369],
    }
    assert frames[1]["surface_hit"] is None


def test_converted_session_passes_parse_session_validation(tmp_path: Path) -> None:
    csv_path = write_csv(tmp_path, [_row(i, spray="On", ray=HIT) for i in range(4)])
    converted = session_csv.convert_csv(csv_path, session_csv.WorkerProfile("expert", "right"))

    sequence_paths = session_csv.write_session(converted, tmp_path / "out")
    result = parse_session.parse_session_file(sequence_paths[0])

    assert result.frame_count == 4
    assert result.summary["skeleton_joint_model"] == "root+19"
    assert result.summary["tracked_object_count"] == 6
    assert result.inferred_active_gun.key == "gun_hangil_short"
    assert result.surface_hit_region_bursts[0]["region_ratios"] == {"MAP01": 1.0}
    assert {field.name for field in result.degraded_fields} == {"rotation.order", "coordinate_frame"}


# Characterization test: _burst_spans was generalized beyond what the test above
# required, so this locks the approach-frame semantics rather than deriving them.
def test_each_spray_burst_becomes_a_sequence_carrying_its_approach_frames(tmp_path: Path) -> None:
    pattern = ["Off", "On", "Off", "Off", "On", "On"]
    csv_path = write_csv(tmp_path, [_row(i, spray=s, ray=HIT) for i, s in enumerate(pattern)])

    converted = session_csv.convert_csv(csv_path, session_csv.WorkerProfile("expert", "right"))

    ranges = [
        (item["sequence"]["frame_range"], item["sequence"]["spray_range"]["start_i"])
        for item in converted.sequences
    ]
    assert ranges == [
        ({"start_i": 0, "end_i": 1}, 1),
        ({"start_i": 2, "end_i": 5}, 4),
    ]
    assert [item["stats"]["spray_frame_count"] for item in converted.sequences] == [1, 2]


def test_final_sequence_absorbs_frames_trailing_the_last_burst(tmp_path: Path) -> None:
    pattern = ["On", "Off", "On", "Off", "Off"]
    csv_path = write_csv(tmp_path, [_row(i, spray=s, ray=HIT) for i, s in enumerate(pattern)])

    converted = session_csv.convert_csv(csv_path, session_csv.WorkerProfile("expert", "right"))

    last = converted.sequences[-1]["sequence"]
    assert last["frame_range"] == {"start_i": 1, "end_i": 4}
    assert last["spray_range"]["end_i"] == 2
    assert sum(len(item["frames"]) for item in converted.sequences) == len(pattern)


def test_model_input_decodes_and_unknown_conventions_stay_null(tmp_path: Path) -> None:
    csv_path = write_csv(tmp_path, [_row(0, spray="On", ray=HIT), _row(1, spray="On", ray=HIT)])

    converted = session_csv.convert_csv(csv_path, session_csv.WorkerProfile("expert", "right"))

    conventions = converted.session_index["conventions"]
    assert conventions["rotation"]["order"] is None
    assert conventions["coordinate_frame"] is None
    assert converted.session_index["skeleton_def"]["hierarchy"] is None
    assert converted.sequences[0]["frames"][0]["model_input"] == {
        "tag": "thick", "viscosity": 251.03, "nozzle": 527.0, "gun_speed": 1600.0,
        "set_pressure": 360.0, "gun_distance": 550.0, "push_time": 0.01111111,
        "svr": 72.0, "thinner_dilution_law": 9.6, "temperature": 29.0,
        "maker_name": "JOTUN", "h1": 30.0, "h2": 15.0, "h3": 10.0,
    }
    assert converted.session_index["settings"]["Pressure"] == "360"


def test_cli_writes_session_and_sequence_files_for_a_csv(tmp_path: Path) -> None:
    csv_path = write_csv(tmp_path, [_row(i, spray="On", ray=HIT) for i in range(3)])
    out_dir = tmp_path / "out"

    completed = subprocess.run(
        [
            sys.executable, "/workspace/trainer/scripts/session_csv.py",
            "--csv", str(csv_path),
            "--out", str(out_dir),
            "--skill-level", "expert",
            "--handedness", "right",
        ],
        check=False, capture_output=True, text=True,
    )

    assert completed.returncode == 0, completed.stderr
    session_index = json.loads((out_dir / "1_1000442_session.json").read_text(encoding="utf-8"))
    assert session_index["worker"]["skill_level"] == "expert"
    assert session_index["source"]["level"] == "1"
    assert (out_dir / "1_1000442_seq001.json").is_file()
