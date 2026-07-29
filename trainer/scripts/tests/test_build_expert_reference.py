#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import importlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

parse_session = importlib.import_module("parse_session")
build_expert_reference = importlib.import_module("build_expert_reference")

SAMPLE_PATH = Path("/workspace/docs/1_1000442_seq002_sample30.json")
SAMPLE_INDEX_PATH = Path("/workspace/docs/1_1000442_session.json")
CHAIN_JOINTS = ["Spine", "Chest", "RightShoulder", "RightUpperArm", "RightForeArm", "RightHand"]


def _load_sample_payloads() -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        json.loads(SAMPLE_PATH.read_text(encoding="utf-8")),
        json.loads(SAMPLE_INDEX_PATH.read_text(encoding="utf-8")),
    )


def _set_chain_pose(frame: dict[str, Any], base: float) -> None:
    joints = frame["skeleton"]["joints_rot_euler_deg"]
    for offset, joint_name in enumerate(CHAIN_JOINTS):
        joints[joint_name] = [base + offset, base + offset + 0.25, base + offset + 0.5]


def _make_synthetic_session(tmp_path: Path, session_name: str, delta: float) -> Path:
    sequence_payload, session_index_payload = _load_sample_payloads()
    session_index_payload["worker"]["skill_level"] = "expert"
    session_index_payload["worker"]["handedness"] = "right"
    session_index_payload["conventions"]["rotation"]["order"] = "ZXY"
    session_index_payload["conventions"]["coordinate_frame"] = "unity-left-handed"

    frames = sequence_payload["frames"][:7]
    for idx, frame in enumerate(frames):
        frame["i"] = idx
        frame["t_sec"] = round(idx * 0.1, 4)
        frame["surface_hit"] = None
        frame["target_region"] = None

    burst_plan = [
        (0, 1, "region_upper", 10.0),
        (3, 4, "region_lower", 20.0),
        (6, 6, None, 0.0),
    ]
    for start, end, region_id, base in burst_plan:
        for frame_idx in range(start, end + 1):
            frame = frames[frame_idx]
            frame["spray_on"] = region_id is not None
            frame["gun_metrics"]["distance_real_mm"] = 450.0 + delta + frame_idx
            frame["gun_metrics"]["speed_real_mms"] = 500.0 + delta + frame_idx
            if region_id is not None:
                frame["surface_hit"] = {
                    "region_id": region_id,
                    "face_id": frame_idx,
                    "hit_pos_m": [0.1 * frame_idx, 0.0, 0.0],
                    "normal": [0.0, 1.0, 0.0],
                }
                frame["target_region"] = region_id
                _set_chain_pose(frame, base + delta + (frame_idx - start) * 3.0)

    sequence_payload["frames"] = frames
    sequence_payload["sequence"]["frame_range"] = {"start_i": 0, "end_i": 6}
    sequence_payload["sequence"]["spray_range"] = {"start_i": 0, "end_i": 4}
    sequence_payload["session_ref"]["session_file"] = f"{session_name}_session.json"
    sequence_payload["stats"]["spray_frame_count"] = 4

    session_path = tmp_path / f"{session_name}_session.json"
    sequence_path = tmp_path / f"{session_name}_sequence.json"
    session_path.write_text(json.dumps(session_index_payload), encoding="utf-8")
    sequence_path.write_text(json.dumps(sequence_payload), encoding="utf-8")
    return sequence_path


def test_resample_upsamples_without_overrunning_the_final_index() -> None:
    vectors = [[float(step), 0.0, 0.0] for step in range(8)]

    resampled = build_expert_reference._resample_vectors(vectors, 26)

    assert len(resampled) == 26
    assert resampled[0] == vectors[0]
    assert resampled[-1] == vectors[-1]


def test_real_sample_cli_emits_insufficient_data(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts"

    completed = subprocess.run(
        [
            sys.executable,
            "/workspace/trainer/scripts/build_expert_reference.py",
            "--sessions",
            str(SAMPLE_PATH),
            "--level",
            "1",
            "--out",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    pose_payload = json.loads((output_dir / "expert_pose_reference.json").read_text(encoding="utf-8"))
    flow_payload = json.loads((output_dir / "expert_flow.json").read_text(encoding="utf-8"))

    assert pose_payload["status"] == "insufficient_data"
    assert flow_payload["status"] == "insufficient_data"
    reasons = pose_payload["reasons"]
    assert any("single session" in reason for reason in reasons)
    assert any("surface_hit" in reason and "null" in reason for reason in reasons)
    assert any("truncated" in reason for reason in reasons)
    assert "insufficient_data" in completed.stdout


def test_build_reference_artifacts_computes_pose_and_flow_from_synthetic_sessions(tmp_path: Path) -> None:
    session_paths = [
        _make_synthetic_session(tmp_path, "expert_a", 0.0),
        _make_synthetic_session(tmp_path, "expert_b", 3.0),
        _make_synthetic_session(tmp_path, "expert_c", 6.0),
    ]

    result = build_expert_reference.build_reference_artifacts(session_paths, level="1", out_dir=tmp_path / "out")

    assert result.pose_payload["status"] == "ok"
    assert result.pose_payload["reference_quality"] == "expert"
    assert result.flow_payload["status"] == "ok"
    assert result.flow_payload["reference_quality"] == "expert"

    pose_region = result.pose_payload["regions"]["region_upper"]
    series = pose_region["joint_rotation_distributions"]["RightShoulder"]
    assert pose_region["sample_count"] == 3
    assert pose_region["normalized_frame_count"] == 2
    assert series[0]["mean"] == pytest.approx([15.0, 15.25, 15.5])
    expected_std = math.sqrt(6.0)
    assert series[0]["std"] == pytest.approx([expected_std, expected_std, expected_std])
    assert series[1]["mean"] == pytest.approx([18.0, 18.25, 18.5])

    flow_edges = {
        (edge["source"], edge["target"]): edge["frequency"]
        for edge in result.flow_payload["edges"]
    }
    assert flow_edges[("region_upper", "region_lower")] == 3

    written_pose = json.loads((tmp_path / "out" / "expert_pose_reference.json").read_text(encoding="utf-8"))
    written_flow = json.loads((tmp_path / "out" / "expert_flow.json").read_text(encoding="utf-8"))
    assert written_pose["status"] == "ok"
    assert written_flow["status"] == "ok"
