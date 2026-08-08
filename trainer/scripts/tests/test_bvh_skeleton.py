#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

bvh_skeleton = importlib.import_module("bvh_skeleton")

HIERARCHY_BVH = """HIERARCHY
ROOT Pelvis
{
\tOFFSET\t0.00\t0.00\t0.00
\tCHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
\tJOINT Spine
\t{
\t\tOFFSET\t 0.000000\t 0.000000\t 0.100000
\t\tCHANNELS 3 Zrotation Xrotation Yrotation
\t\tJOINT Chest
\t\t{
\t\t\tOFFSET\t 0.000000\t 0.000000\t 0.200000
\t\t\tCHANNELS 3 Zrotation Xrotation Yrotation
\t\t\tEnd Site
\t\t\t{
\t\t\t\tOFFSET\t 0.000000\t 0.000000\t 0.050000
\t\t\t}
\t\t}
\t}
}
MOTION
Frames:    {frames}
Frame Time: 0.0111111
{rows}
"""

# Pelvis 6채널(Xpos Ypos Zpos Zrot Xrot Yrot) + Spine 3채널 + Chest 3채널 = 12
REST_ROWS = " 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0\n 1.0 2.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0"


def _write(tmp_path: Path, rows: str, frames: int = 2) -> Path:
    path = tmp_path / "clip.bvh"
    text = HIERARCHY_BVH.replace("{frames}", str(frames)).replace("{rows}", rows)
    path.write_text(text, encoding="utf-8")
    return path


def test_parse_bvh_reads_hierarchy_channels_and_frames(tmp_path: Path) -> None:
    clip = bvh_skeleton.parse_bvh(_write(tmp_path, REST_ROWS))

    assert [joint.name for joint in clip.joints] == ["Pelvis", "Spine", "Chest"]
    assert [joint.parent for joint in clip.joints] == [-1, 0, 1]
    assert clip.joints[0].channels == (
        "Xposition",
        "Yposition",
        "Zposition",
        "Zrotation",
        "Xrotation",
        "Yrotation",
    )
    assert clip.joints[1].channels == ("Zrotation", "Xrotation", "Yrotation")
    assert clip.joints[1].offset == (0.0, 0.0, 0.1)
    assert clip.frame_count == 2
    assert len(clip.frames) == 2
    assert len(clip.frames[0]) == 12


def _close(actual: tuple[float, float, float], expected: tuple[float, float, float]) -> bool:
    return all(abs(a - b) < 1e-9 for a, b in zip(actual, expected))


def test_forward_kinematics_chains_offsets_from_root_translation(tmp_path: Path) -> None:
    clip = bvh_skeleton.parse_bvh(_write(tmp_path, REST_ROWS))

    world = bvh_skeleton.forward_kinematics(clip, 1)

    assert _close(world["Pelvis"], (1.0, 2.0, 3.0))
    assert _close(world["Spine"], (1.0, 2.0, 3.1))
    assert _close(world["Chest"], (1.0, 2.0, 3.3))


def test_forward_kinematics_composes_rotations_in_zxy_order(tmp_path: Path) -> None:
    """Spine에 Zrot=90, Xrot=90. ZXY(Rz@Rx)면 Chest offset (0,0,0.2)가 +x로 간다.

    XZY 같은 다른 순서면 -y로 가므로 이 단언이 순서를 판별한다.
    """
    rows = " 0.0 0.0 1.0 0.0 0.0 0.0 90.0 90.0 0.0 0.0 0.0 0.0"
    clip = bvh_skeleton.parse_bvh(_write(tmp_path, rows, frames=1))

    world = bvh_skeleton.forward_kinematics(clip, 0)

    assert _close(world["Spine"], (0.0, 0.0, 1.1))
    assert _close(world["Chest"], (0.2, 0.0, 1.1))


def test_forward_kinematics_subset_matches_full_result(tmp_path: Path) -> None:
    """손가락 40관절을 건너뛰어도 요청한 관절 값은 전체 계산과 같아야 한다."""
    clip = bvh_skeleton.parse_bvh(_write(tmp_path, REST_ROWS))

    full = bvh_skeleton.forward_kinematics(clip, 1)
    subset = bvh_skeleton.forward_kinematics(clip, 1, joints={"Chest"})

    assert set(subset) == {"Chest"}
    assert _close(subset["Chest"], full["Chest"])


def test_normalize_joint_name_strips_exporter_artifacts() -> None:
    assert bvh_skeleton.normalize_joint_name("mixamorig:LeftHandIndex2") == "LeftHandIndex2"
    assert bvh_skeleton.normalize_joint_name("RightArm") == "RightUpperArm"
    assert bvh_skeleton.normalize_joint_name("LeftUpperArm") == "LeftUpperArm"
    assert bvh_skeleton.normalize_joint_name("Pelvis") == "Pelvis"


def test_parse_bvh_normalizes_joint_names(tmp_path: Path) -> None:
    """벤더 BVH는 좌우 명명이 비대칭이다(LeftUpperArm vs RightArm). 파서가 흡수한다."""
    text = HIERARCHY_BVH.replace("JOINT Chest", "JOINT RightArm").replace(
        "{frames}", "1"
    ).replace("{rows}", " 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0")
    path = tmp_path / "named.bvh"
    path.write_text(text, encoding="utf-8")

    clip = bvh_skeleton.parse_bvh(path)

    assert [joint.name for joint in clip.joints] == ["Pelvis", "Spine", "RightUpperArm"]
