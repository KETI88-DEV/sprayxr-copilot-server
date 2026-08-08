#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

simulator_csv = importlib.import_module("simulator_csv")


def _row(**overrides: float) -> dict[str, str]:
    row: dict[str, str] = {}
    for column in simulator_csv.JOINT_COLUMNS:
        for axis in "XYZ":
            key = f"{column} Position {axis}"
            row[key] = str(overrides.get(key, 0.0))
    return row


def test_joint_world_positions_uses_normalized_joint_names() -> None:
    positions = simulator_csv.joint_world_positions(
        _row(**{"Right Upper Arm Position X": 0.22, "Right Upper Arm Position Z": 1.28})
    )

    assert "RightUpperArm" in positions
    assert "Right Upper Arm" not in positions
    assert positions["RightUpperArm"] == (0.22, 0.0, 1.28)


def test_joint_world_positions_covers_every_declared_joint() -> None:
    positions = simulator_csv.joint_world_positions(_row())

    assert len(positions) == len(simulator_csv.JOINT_COLUMNS)


def test_has_world_joint_motion_rejects_bind_pose_offsets() -> None:
    """현재 납품본은 관절 Position이 상수(바인드 포즈)다. 이것을 월드 좌표로 쓰면 안 된다."""
    rows = [_row(**{"Right Hand Position X": 0.3}) for _ in range(50)]

    assert simulator_csv.has_world_joint_motion(rows, ("RightHand",)) is False


def test_has_world_joint_motion_ignores_float_noise() -> None:
    rows = [
        _row(**{"Right Hand Position X": 0.3 + index * 1e-7}) for index in range(50)
    ]

    assert simulator_csv.has_world_joint_motion(rows, ("RightHand",)) is False


def test_has_world_joint_motion_accepts_a_moving_joint() -> None:
    rows = [_row(**{"Right Hand Position X": 0.3 + index * 0.01}) for index in range(50)]

    assert simulator_csv.has_world_joint_motion(rows, ("RightHand",)) is True


def test_has_world_joint_motion_needs_every_requested_joint_to_move() -> None:
    rows = [_row(**{"Right Hand Position X": 0.3 + index * 0.01}) for index in range(50)]

    assert simulator_csv.has_world_joint_motion(rows, ("RightHand", "Chest")) is False
