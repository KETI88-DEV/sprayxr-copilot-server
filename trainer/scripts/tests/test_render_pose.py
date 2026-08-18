#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

render_pose = importlib.import_module("render_pose")
simulator_csv = importlib.import_module("simulator_csv")

Vec3 = tuple[float, float, float]


def _pose() -> dict[str, Vec3]:
    return {name: (0.0, 0.0, 0.9) for name in simulator_csv.JOINT_KEYS.values()}


def test_bones_only_reference_joints_the_export_actually_has() -> None:
    """뼈대가 없는 관절을 그리려 하면 조용히 빈 그림이 나온다."""
    known = set(simulator_csv.JOINT_KEYS.values())

    unknown = {name for bone in render_pose.BONES for name in bone} - known

    assert unknown == set()


def test_bones_cover_both_arms_and_both_legs() -> None:
    joined = {name for bone in render_pose.BONES for name in bone}

    for required in ("LeftHand", "RightHand", "LeftFoot", "RightFoot", "Head"):
        assert required in joined


def test_drawable_bones_skips_pairs_missing_from_the_pose() -> None:
    partial = {"Pelvis": (0.0, 0.0, 0.9), "Spine": (0.0, 0.0, 1.05)}

    segments = render_pose.drawable_bones(partial)

    assert segments == [((0.0, 0.0, 0.9), (0.0, 0.0, 1.05))]


def test_drawable_bones_returns_every_bone_for_a_complete_pose() -> None:
    segments = render_pose.drawable_bones(_pose())

    assert len(segments) == len(render_pose.BONES)


def test_render_comparison_writes_an_image(tmp_path: Path) -> None:
    target = tmp_path / "compare.png"

    render_pose.render_comparison(_pose(), _pose(), target, ("trainee", "expert"))

    assert target.is_file()
    assert target.stat().st_size > 0


def test_ascii_label_replaces_glyphs_the_font_cannot_draw() -> None:
    """컨테이너에 한글 폰트가 없어 한글 라벨은 경고 없이 두부로 나온다."""
    assert render_pose.ascii_label("trainee") == "trainee"
    assert render_pose.ascii_label("훈련생").isascii()
