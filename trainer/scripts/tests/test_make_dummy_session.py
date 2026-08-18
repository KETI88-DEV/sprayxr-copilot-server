#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import csv
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

make_dummy_session = importlib.import_module("make_dummy_session")
simulator_csv = importlib.import_module("simulator_csv")

Vec3 = tuple[float, float, float]


def _poses(count: int = 40) -> list[dict[str, Vec3]]:
    poses: list[dict[str, Vec3]] = []
    for index in range(count):
        sweep = 0.02 * index
        base = {name: (0.0, 0.0, 0.9) for name in make_dummy_session.EXPORT_JOINTS}
        base["Pelvis"] = (0.0, 0.0, 0.9)
        base["LeftShoulder"] = (-0.18, 0.0, 1.32)
        base["RightShoulder"] = (0.18, 0.0, 1.32)
        base["RightForeArm"] = (0.28 + sweep, 0.18, 1.10)
        base["RightHand"] = (0.30 + sweep, 0.34, 1.02)
        base["Head"] = (0.0, 0.0, 1.45)
        poses.append(base)
    return poses


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_generated_session_carries_every_production_joint_column(tmp_path: Path) -> None:
    path = make_dummy_session.write_session(tmp_path / "3_9000001_1.csv", _poses(), 0.011)

    rows = _read(path)
    for column in simulator_csv.JOINT_COLUMNS:
        for axis in "XYZ":
            assert f"{column} Position {axis}" in rows[0]


def test_generated_joint_columns_are_world_coordinates_not_bind_offsets(
    tmp_path: Path,
) -> None:
    """실납품이 월드 좌표이므로 더미도 그래야 한다. 상수면 가드가 걸러낸다."""
    path = make_dummy_session.write_session(tmp_path / "3_9000001_1.csv", _poses(), 0.011)

    rows = _read(path)
    assert simulator_csv.has_world_joint_motion(rows, ("RightHand", "RightForeArm")) is True


def test_generated_session_is_co_indexed_one_row_per_pose(tmp_path: Path) -> None:
    path = make_dummy_session.write_session(tmp_path / "3_9000001_1.csv", _poses(25), 0.011)

    assert len(_read(path)) == 25


def test_generated_session_splits_into_several_sequences(tmp_path: Path) -> None:
    """같은 스트로크를 다른 시퀀스에서 비교하려면 더미도 버스트가 여러 개여야 한다."""
    build_pose_reference = importlib.import_module("build_pose_reference")
    path = make_dummy_session.write_session(
        tmp_path / "3_9000001_1.csv", _poses(60), 0.011, sequence_frames=12
    )

    bursts = list(build_pose_reference.iter_bursts_with_rows(path))

    assert len(bursts) >= 4


def test_spray_gap_rows_are_marked_off(tmp_path: Path) -> None:
    path = make_dummy_session.write_session(
        tmp_path / "3_9000001_1.csv", _poses(30), 0.011, sequence_frames=10
    )

    spray = [row["분사"] for row in _read(path)]
    assert "Off" in spray
    assert "On" in spray


def test_mutation_shifts_only_the_named_joint(tmp_path: Path) -> None:
    """알려진 편차를 주입해야 비교 결과가 그 값을 되찾는지 확인할 수 있다."""
    mutated = make_dummy_session.mutate(_poses(3), "RightForeArm", (0.0, 0.0, -0.15))

    assert mutated[0]["RightForeArm"][2] == _poses(3)[0]["RightForeArm"][2] - 0.15
    assert mutated[0]["RightHand"] == _poses(3)[0]["RightHand"]


def test_mutation_leaves_the_source_untouched(tmp_path: Path) -> None:
    original = _poses(3)
    make_dummy_session.mutate(original, "RightForeArm", (0.0, 0.0, -0.15))

    assert original[0]["RightForeArm"] == _poses(3)[0]["RightForeArm"]
