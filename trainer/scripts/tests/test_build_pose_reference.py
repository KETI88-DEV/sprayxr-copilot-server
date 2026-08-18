#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

build_pose_reference = importlib.import_module("build_pose_reference")
pose_similarity = importlib.import_module("pose_similarity")

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


def _facing_pose(x: float = 0.0, y: float = 0.0) -> dict[str, tuple[float, float, float]]:
    return {
        "Pelvis": (x, y, 0.9),
        "LeftShoulder": (x - 0.18, y, 1.32),
        "RightShoulder": (x + 0.18, y, 1.32),
        "RightHand": (x + 0.3, y + 0.35, 1.0),
    }


def _forward_burst() -> object:
    return build_pose_reference.Burst(
        row_indices=list(range(12)),
        points=[(0.0, 0.02 * i, 0.8) for i in range(12)],
        times=[0.01 * i for i in range(12)],
    )


def test_body_bucket_frames_labels_the_pass_in_the_worker_frame() -> None:
    """자세가 패스와 같은 인덱스로 들어오므로 진행 방향을 작업자 기준으로 적을 수 있다."""
    poses = [_facing_pose() for _ in range(12)]

    mapped = build_pose_reference.body_bucket_frames(
        _forward_burst(), list(range(12)), poses
    )

    assert len(mapped) == 1
    bucket, frames = mapped[0]
    assert bucket == "horizontal-mid-forward"
    assert frames == list(range(12))


def test_body_bucket_frames_give_the_same_label_when_the_worker_is_turned() -> None:
    angle = math.radians(137.0)
    cos, sin = math.cos(angle), math.sin(angle)
    turned_points = [
        (0.0 * cos - (0.02 * i) * sin, 0.0 * sin + (0.02 * i) * cos, 0.8) for i in range(12)
    ]
    turned_pose = {
        name: (px * cos - py * sin, px * sin + py * cos, pz)
        for name, (px, py, pz) in _facing_pose().items()
    }
    burst = build_pose_reference.Burst(
        row_indices=list(range(12)), points=turned_points, times=[0.01 * i for i in range(12)]
    )

    mapped = build_pose_reference.body_bucket_frames(burst, list(range(12)), [turned_pose] * 12)

    assert mapped[0][0] == "horizontal-mid-forward"


def test_build_payload_keeps_the_degenerate_root_out_of_the_artifact() -> None:
    """정규화된 루트는 항상 원점이라 mean [0,0,0] spread 0의 무의미한 항목이 된다."""
    sample = {"Pelvis": (0.0, 0.0, 0.0), "Chest": (0.0, 0.0, 0.3), "RightHand": (0.3, 0.3, 1.0)}
    payload = build_pose_reference.build_payload(
        {"horizontal-mid-+y": [sample] * 40}, "10", "전형기", accepted=1, rejected=[]
    )

    joints = payload["buckets"]["horizontal-mid-+y"]["joints"]  # type: ignore[index]
    assert "Pelvis" not in joints
    assert set(joints) == {"Chest", "RightHand"}


def test_build_payload_drops_joints_outside_the_scoring_set() -> None:
    sample = {"Chest": (0.0, 0.0, 0.3), "LeftToe": (0.1, 0.0, -0.8)}
    payload = build_pose_reference.build_payload(
        {"horizontal-mid-+y": [sample] * 40}, "10", "전형기", accepted=1, rejected=[]
    )

    joints = payload["buckets"]["horizontal-mid-+y"]["joints"]  # type: ignore[index]
    assert set(joints) == {"Chest"}


def _world_csv(tmp_path: Path, drop: float = 0.0) -> Path:
    make_dummy_session = importlib.import_module("make_dummy_session")
    poses: list[dict[str, tuple[float, float, float]]] = []
    for index in range(240):
        sweep = 0.01 * (index % 40)
        lean = 0.2 * sweep
        poses.append({
            "Pelvis": (0.0, 0.0, 0.9),
            "Spine": (lean, 0.0, 1.05),
            "Chest": (lean, 0.0, 1.2),
            "LeftShoulder": (-0.18 + lean, 0.0, 1.32),
            "RightShoulder": (0.18 + lean, 0.0, 1.32),
            "LeftUpperArm": (-0.22 + lean, 0.02, 1.28),
            "RightUpperArm": (0.22 + lean, 0.02, 1.28),
            "LeftForeArm": (-0.28 + lean, 0.18, 1.10),
            "RightForeArm": (0.28 + sweep, 0.18, 1.10 - drop),
            "LeftHand": (-0.30 + lean, 0.34, 1.02),
            "RightHand": (0.30 + sweep, 0.34, 1.02),
            "Head": (lean, 0.0, 1.45),
        })
    return make_dummy_session.write_session(
        tmp_path / "10_9000001_1.csv", poses, 0.011, sequence_frames=40
    )


def test_iter_csv_passes_reads_poses_and_buckets_from_the_csv_alone(tmp_path: Path) -> None:
    """실납품은 월드 좌표 CSV다. BVH 없이 이 파일만으로 레퍼런스를 만들 수 있어야 한다."""
    entries = list(build_pose_reference.iter_csv_passes(_world_csv(tmp_path)))

    assert entries
    sequence, bucket, pose = entries[0]
    assert sequence == 1
    assert bucket.count("-") == 2
    assert "RightHand" in pose


def test_iter_csv_passes_numbers_sequences_from_one(tmp_path: Path) -> None:
    entries = list(build_pose_reference.iter_csv_passes(_world_csv(tmp_path)))

    assert min(entry[0] for entry in entries) == 1
    assert max(entry[0] for entry in entries) > 1


def test_collect_csv_session_groups_passes_by_bucket(tmp_path: Path) -> None:
    grouped = build_pose_reference.collect_csv_session(_world_csv(tmp_path))

    assert grouped
    assert all(isinstance(poses, list) and poses for poses in grouped.values())


def test_csv_collection_refuses_bind_pose_columns(tmp_path: Path) -> None:
    """관절이 상수면 모든 표본이 같은 자세가 되어 유사도가 무의미해진다."""
    make_dummy_session = importlib.import_module("make_dummy_session")
    frozen = [
        {name: (0.0, 0.0, 1.0) for name in make_dummy_session.EXPORT_JOINTS}
        for _ in range(60)
    ]
    path = make_dummy_session.write_session(tmp_path / "10_9000002_1.csv", frozen, 0.011)

    with pytest.raises(ValueError, match="바인드 포즈"):
        _ = build_pose_reference.collect_csv_session(path)


def test_shrunk_spread_falls_back_to_the_pooled_estimate_without_data() -> None:
    """산포는 2차 모멘트라 표본이 얇으면 불안정하다. 평균과 달리 보강이 필요하다."""
    assert build_pose_reference.shrunk_spread(0.30, 0, 0.10) == pytest.approx(0.10)


def test_shrunk_spread_keeps_the_bucket_estimate_when_data_is_plentiful() -> None:
    value = build_pose_reference.shrunk_spread(0.30, 100000, 0.10)

    assert value == pytest.approx(0.30, abs=1e-3)


def test_shrunk_spread_blends_variances_at_the_half_weight_point() -> None:
    value = build_pose_reference.shrunk_spread(
        0.30, build_pose_reference.SPREAD_SHRINK_WEIGHT, 0.10
    )

    assert value == pytest.approx(math.sqrt((0.09 + 0.01) / 2))


def test_build_payload_shrinks_a_thin_bucket_spread_toward_the_pooled_value() -> None:
    tight = [{"RightHand": (0.0, 0.0, 1.0 + 0.001 * i)} for i in range(10)]
    wide = [{"RightHand": (0.0, 0.0, 1.0 + 0.05 * i)} for i in range(60)]

    payload = build_pose_reference.build_payload(
        {"horizontal-mid-forward": tight, "horizontal-high-forward": wide},
        "10", "전형기", accepted=1, rejected=[],
    )

    thin = payload["buckets"]["horizontal-mid-forward"]["joints"]["RightHand"]  # type: ignore[index]
    raw = build_pose_reference.summarize_joint_positions(tight)["RightHand"]
    assert float(thin["spread"]) > float(raw["spread"])  # type: ignore[arg-type]


def test_pass_mean_pose_averages_the_normalized_frames_of_one_pass() -> None:
    """레퍼런스는 프레임이 아니라 패스 요약에서 추정해야 긴 패스가 과대 가중되지 않는다."""
    poses = [
        {"Pelvis": (0.0, 0.0, 0.9), "RightHand": (0.0, 0.0, 1.0 + 0.1 * index)}
        for index in range(3)
    ]

    mean = build_pose_reference.pass_mean_pose(poses, [0, 1, 2])

    assert mean["RightHand"] == pytest.approx((0.0, 0.0, 0.2), abs=1e-9)


def test_pass_mean_pose_skips_frames_missing_a_joint() -> None:
    poses = [
        {"Pelvis": (0.0, 0.0, 0.0), "RightHand": (0.0, 0.0, 1.0)},
        {"Pelvis": (0.0, 0.0, 0.0)},
        {"Pelvis": (0.0, 0.0, 0.0), "RightHand": (0.0, 0.0, 3.0)},
    ]

    mean = build_pose_reference.pass_mean_pose(poses, [0, 1, 2])

    assert mean["RightHand"] == pytest.approx((0.0, 0.0, 2.0), abs=1e-9)


def test_pass_mean_pose_is_empty_without_usable_frames() -> None:
    assert build_pose_reference.pass_mean_pose([], []) == {}


def test_bvh_burst_is_co_indexed_with_the_pose_stream_by_construction() -> None:
    """BVH는 자세와 손 궤적이 같은 프레임에서 나온다. 정합 문제가 아예 생기지 않는다."""
    poses = [
        {"RightHand": (0.0, 0.02 * i, 0.8), "Pelvis": (0.0, 0.0, 0.9)} for i in range(12)
    ]

    burst = build_pose_reference.bvh_burst(poses, frame_time=0.01)

    assert burst.row_indices == list(range(12))
    assert burst.points[3] == (0.0, 0.06, 0.8)
    assert burst.times[3] == pytest.approx(0.03)


def test_bvh_burst_is_empty_without_the_gun_hand() -> None:
    burst = build_pose_reference.bvh_burst([{"Pelvis": (0.0, 0.0, 0.9)}], frame_time=0.01)

    assert burst.row_indices == []


def test_co_indexed_quality_rejects_data_that_is_not_actually_co_indexed() -> None:
    """--co-indexed는 입력에 대한 단언이다. 79세션 BVH는 7509프레임 대 15366행으로 맞지 않는다."""
    quality = build_pose_reference.co_indexed_quality(pose_count=7509, row_count=15366)

    assert quality.usable is False
    assert quality.reason == "not_co_indexed"


def test_co_indexed_quality_accepts_matching_frame_counts() -> None:
    quality = build_pose_reference.co_indexed_quality(pose_count=500, row_count=500)

    assert quality.usable is True
    assert quality.reason is None
    assert quality.verification == 1.0
    assert quality.residual == 0.0


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


def test_iter_csv_pass_frames_yields_real_frames_not_the_mean(tmp_path: Path) -> None:
    """평균 자세는 팔다리가 중심으로 수축해 사람이 취할 수 없다. 그림에는 실제 프레임을 쓴다."""
    path = _world_csv(tmp_path)

    framed = list(build_pose_reference.iter_csv_pass_frames(path))
    averaged = list(build_pose_reference.iter_csv_passes(path))

    assert len(framed) == len(averaged)
    sequence, bucket, frames = framed[0]
    assert sequence == averaged[0][0]
    assert bucket == averaged[0][1]
    assert len(frames) > 1
    assert frames[0] != frames[-1]


def test_normalization_keeps_bone_lengths_so_the_drawing_is_honest(tmp_path: Path) -> None:
    """정규화는 평행이동과 수직축 회전뿐이라 등거리다. 뼈 길이가 바뀌면 그림이 거짓이 된다."""
    import csv as _csv
    import math

    from simulator_csv import BONES, joint_world_positions

    path = _world_csv(tmp_path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        raw = [joint_world_positions(row) for row in _csv.DictReader(handle)]

    for index in (0, 37, 120):
        source = raw[index]
        moved = pose_similarity.normalize_pose(source)
        for start, end in BONES:
            assert math.dist(moved[start], moved[end]) == pytest.approx(
                math.dist(source[start], source[end]), abs=1e-9
            ), f"{start}->{end}"
