#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

build_pose_reference = importlib.import_module("build_pose_reference")

HEADER = "인덱스,시간(초),분사,Ray point X,Ray Point Y,Ray Point Z"


def _csv(tmp_path: Path, rows: list[str]) -> Path:
    path = tmp_path / "3_1000001_1.csv"
    path.write_text("\n".join([HEADER, *rows]) + "\n", encoding="utf-8-sig")
    return path


def test_summarize_joint_positions_returns_centroid_and_radial_spread() -> None:
    samples = [
        {"RightHand": (0.0, 0.0, 0.0)},
        {"RightHand": (0.1, 0.0, 0.0)},
        {"RightHand": (-0.1, 0.0, 0.0)},
    ]

    summary = build_pose_reference.summarize_joint_positions(samples)

    assert summary["RightHand"]["mean"] == [0.0, 0.0, 0.0]
    assert abs(summary["RightHand"]["spread"] - math.sqrt(0.02 / 3)) < 1e-5
    assert summary["RightHand"]["n"] == 3


def test_summarize_joint_positions_skips_joints_missing_from_a_sample() -> None:
    samples = [
        {"RightHand": (0.0, 0.0, 0.0), "Chest": (0.0, 0.0, 0.3)},
        {"RightHand": (0.0, 0.0, 0.0)},
    ]

    summary = build_pose_reference.summarize_joint_positions(samples)

    assert summary["RightHand"]["n"] == 2
    assert summary["Chest"]["n"] == 1


def test_iter_bursts_with_rows_carries_the_original_csv_row_index(tmp_path: Path) -> None:
    """패스 구간을 BVH 프레임으로 되돌리려면 원본 행 번호가 필요하다."""
    rows = [
        "0,0.00,Off,,,",
        "1,0.01,On,0.0,0.0,1.0",
        "2,0.02,On,0.1,0.0,1.0",
        "3,0.03,Off,,,",
        "4,0.04,On,0.5,0.0,1.0",
    ]

    bursts = list(build_pose_reference.iter_bursts_with_rows(_csv(tmp_path, rows)))

    assert len(bursts) == 2
    assert bursts[0].row_indices == [1, 2]
    assert bursts[0].points == [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0)]
    assert bursts[0].times == [0.01, 0.02]
    assert bursts[1].row_indices == [4]


def test_bucket_frames_maps_a_pass_to_the_bvh_frames_covering_it() -> None:
    """패스는 CSV 행 구간으로 나오고, 자세 표본은 그 행에 정합된 BVH 프레임에서 온다."""
    burst = build_pose_reference.Burst(
        row_indices=list(range(5, 17)),
        points=[(0.02 * i, 0.0, 0.8) for i in range(12)],
        times=[0.01 * i for i in range(12)],
    )
    frame_for_row = [row // 2 for row in range(20)]

    mapped = build_pose_reference.bucket_frames(burst, frame_for_row)

    assert len(mapped) == 1
    bucket, frames = mapped[0]
    assert bucket == "horizontal-mid-+x"
    assert frames == [2, 3, 4, 5, 6, 7, 8]


def test_session_quality_rejects_an_alignment_no_independent_signal_confirms() -> None:
    """작은 잔차는 정합의 증거가 아니다. 실측에서 잔차 14.4mm에 교차검증 -0.04가 나왔다."""
    quality = build_pose_reference.session_quality(
        residual=0.014, coverage=0.74, verification=-0.04
    )

    assert quality.usable is False
    assert quality.reason == "alignment_unverified"


def test_session_quality_rejects_a_session_with_a_large_alignment_residual() -> None:
    quality = build_pose_reference.session_quality(
        residual=0.120, coverage=0.75, verification=0.95
    )

    assert quality.usable is False
    assert quality.reason == "alignment_residual"


def test_session_quality_rejects_a_session_the_bvh_barely_covers() -> None:
    quality = build_pose_reference.session_quality(
        residual=0.010, coverage=0.21, verification=0.95
    )

    assert quality.usable is False
    assert quality.reason == "path_coverage"


def test_session_quality_accepts_a_clean_session() -> None:
    quality = build_pose_reference.session_quality(
        residual=0.014, coverage=0.74, verification=0.93
    )

    assert quality.usable is True
    assert quality.reason is None


def test_iter_bursts_with_rows_drops_spray_frames_without_a_ray_hit(tmp_path: Path) -> None:
    rows = [
        "0,0.00,On,,,",
        "1,0.01,On,0.0,0.0,1.0",
    ]

    bursts = list(build_pose_reference.iter_bursts_with_rows(_csv(tmp_path, rows)))

    assert [burst.row_indices for burst in bursts] == [[1]]
