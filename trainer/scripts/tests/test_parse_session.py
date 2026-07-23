#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

parse_session = importlib.import_module("parse_session")

SAMPLE_PATH = Path("/workspace/docs/1_1000442_seq002_sample30.json")


def test_parse_sample_sequence_happy_path() -> None:
    result = parse_session.parse_session_file(SAMPLE_PATH)

    assert result.frame_count == 30
    assert result.summary["skeleton_joint_model"] == "root+19"
    assert result.summary["tracked_object_count"] == 6
    assert len(result.frame_validity) == 30
    assert all(item.is_valid for item in result.frame_validity)


def test_active_gun_and_degraded_fields_detected() -> None:
    result = parse_session.parse_session_file(SAMPLE_PATH)

    assert result.inferred_active_gun.key == "gun_hangil_short"
    assert result.inferred_active_gun.confidence > 0.5
    degraded_names = {field.name for field in result.degraded_fields}
    assert {"surface_hit", "skill_level", "rotation.order", "coordinate_frame"}.issubset(degraded_names)


def test_malformed_missing_joint_raises_validation_error(tmp_path: Path) -> None:
    payload = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    del payload["frames"][0]["skeleton"]["joints_rot_euler_deg"]["Head"]
    bad_path = tmp_path / "bad_sequence.json"
    bad_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(parse_session.SessionValidationError):
        parse_session.parse_session_file(bad_path)


def test_loads_index_and_links_sequence_metadata() -> None:
    result = parse_session.parse_session_file(SAMPLE_PATH)

    assert result.session_index is not None
    assert result.session_index.schema_version == "0.3.0"
    assert result.sequence.sequence.seq_id == 2
