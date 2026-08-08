#!/usr/bin/env python3
"""BVH 자세를 숙련자 버킷 분포로 요약해 레퍼런스 아티팩트를 만든다.

버킷 키는 궤적 레퍼런스와 같은 stroke 종류·위치 분류(pass_segmentation.bucket_key)를
그대로 쓴다. 자세는 CSV가 아니라 BVH FK에서 오고, 둘은 타임베이스가 달라
pose_alignment으로 정합한 뒤에야 같은 패스에 묶인다.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterator, Mapping, Sequence

import bvh_skeleton
import pose_alignment
import pose_similarity
from pass_segmentation import body_bucket_key, segment_burst
from simulator_csv import RAY_POINT_COLUMNS

logger = logging.getLogger(__name__)

Vec3 = tuple[float, float, float]

SPRAY_COLUMN: Final = "분사"
TIME_COLUMN: Final = "시간(초)"
SPRAY_ON: Final = "On"
GUN_COLUMN: Final = "한길숏건"
CAMERA_COLUMN: Final = "카메라"
ALIGNMENT_JOINTS: Final = ("RightHand", "Head")
GUN_HAND_JOINT: Final = "RightHand"

MIN_BUCKET_POSES: Final = 30
MAX_ALIGNMENT_RESIDUAL_M: Final = 0.030
MIN_PATH_COVERAGE: Final = 0.45

# 정합에 쓰지 않은 신호가 이만큼 따라와야 대응을 믿는다. 현재 데이터의 최고치는
# 0.80이라 전 세션이 걸러진다. 그래야 무의미한 레퍼런스가 조용히 만들어지지 않는다.
MIN_ALIGNMENT_VERIFICATION: Final = 0.85


@dataclass(frozen=True, slots=True)
class Burst:
    row_indices: list[int]
    points: list[Vec3]
    times: list[float]


@dataclass(frozen=True, slots=True)
class SessionSignals:
    """align은 정합에 쓰고, verify는 정합에 쓰지 않고 결과를 검증하는 데만 쓴다."""

    align: list[float]
    verify: list[float]
    path_length: float


@dataclass(frozen=True, slots=True)
class SessionQuality:
    residual: float
    coverage: float
    verification: float
    usable: bool
    reason: str | None


def session_quality(residual: float, coverage: float, verification: float) -> SessionQuality:
    """정합을 믿을 수 없거나 BVH가 세션을 거의 담지 못한 회차를 레퍼런스에서 제외한다.

    교차검증을 먼저 본다. 잔차는 상수 신호에서도 작게 나와 정합의 증거가 못 되므로
    단독으로는 통과 근거가 될 수 없다.
    """
    if verification < MIN_ALIGNMENT_VERIFICATION:
        return SessionQuality(residual, coverage, verification, False, "alignment_unverified")
    if residual > MAX_ALIGNMENT_RESIDUAL_M:
        return SessionQuality(residual, coverage, verification, False, "alignment_residual")
    if coverage < MIN_PATH_COVERAGE:
        return SessionQuality(residual, coverage, verification, False, "path_coverage")
    return SessionQuality(residual, coverage, verification, True, None)


def _pass_frames(
    burst: Burst, frame_for_row: Sequence[int], start: int, end: int
) -> list[int]:
    rows = burst.row_indices[start : end + 1]
    return sorted({frame_for_row[row] for row in rows if row < len(frame_for_row)})


def body_bucket_frames(
    burst: Burst, frame_for_row: Sequence[int], poses: Sequence[Mapping[str, Vec3]]
) -> list[tuple[str, list[int]]]:
    """패스마다 (작업자 기준 버킷, 대응 프레임들)을 낸다.

    진행 방향을 작업자 기준으로 적으려면 그 순간의 자세가 필요하다. 런타임처럼 자세와
    CSV가 같은 인덱스로 들어올 때만 성립한다.
    """
    mapped: list[tuple[str, list[int]]] = []
    for metrics in segment_burst(burst.points, burst.times):
        frames = _pass_frames(burst, frame_for_row, metrics.frame_start, metrics.frame_end)
        if not frames:
            continue
        middle = frames[len(frames) // 2]
        if middle >= len(poses):
            continue
        start = burst.points[metrics.frame_start]
        end = burst.points[metrics.frame_end]
        heading = pose_similarity.body_heading(
            pose_similarity.facing_yaw(poses[middle]),
            (end[0] - start[0], end[1] - start[1]),
        )
        mapped.append(
            (body_bucket_key(metrics.elevation_deg, metrics.mid_height_m, heading), frames)
        )
    return mapped


def bvh_burst(poses: Sequence[Mapping[str, Vec3]], frame_time: float) -> Burst:
    """BVH 클립 전체를 하나의 버스트로 본다.

    자세와 건을 쥔 손의 궤적이 같은 프레임에서 나오므로 정합 문제가 생기지 않는다.
    다만 BVH에는 분사 신호가 없어 비분사 이동도 함께 들어온다.
    """
    hands = [pose[GUN_HAND_JOINT] for pose in poses if GUN_HAND_JOINT in pose]
    return Burst(
        row_indices=list(range(len(hands))),
        points=hands,
        times=[index * frame_time for index in range(len(hands))],
    )


def co_indexed_quality(pose_count: int, row_count: int) -> SessionQuality:
    """동일 인덱스 대응은 단언이지 관측이 아니다. 프레임 수가 다르면 그 단언이 틀린 것이다."""
    if pose_count != row_count or row_count == 0:
        return SessionQuality(0.0, 0.0, 0.0, False, "not_co_indexed")
    return SessionQuality(0.0, 1.0, 1.0, True, None)


def bucket_frames(burst: Burst, frame_for_row: Sequence[int]) -> list[tuple[str, list[int]]]:
    """버스트를 패스로 나눠 (버킷, 그 패스에 정합된 BVH 프레임들)을 낸다."""
    mapped: list[tuple[str, list[int]]] = []
    for metrics in segment_burst(burst.points, burst.times):
        rows = burst.row_indices[metrics.frame_start : metrics.frame_end + 1]
        frames = sorted({frame_for_row[row] for row in rows if row < len(frame_for_row)})
        if frames:
            mapped.append((metrics.bucket, frames))
    return mapped


def summarize_joint_positions(
    samples: Sequence[Mapping[str, Vec3]],
) -> dict[str, dict[str, object]]:
    """관절별 중심과 반경 RMS 산포. 산포가 유사도의 σ 분모가 된다."""
    grouped: dict[str, list[Vec3]] = defaultdict(list)
    for sample in samples:
        for name, position in sample.items():
            grouped[name].append(position)

    summary: dict[str, dict[str, object]] = {}
    for name, positions in grouped.items():
        count = len(positions)
        mean = tuple(sum(p[axis] for p in positions) / count for axis in range(3))
        variance = sum(math.dist(p, mean) ** 2 for p in positions) / count
        summary[name] = {
            "n": count,
            "mean": [round(value, 5) for value in mean],
            "spread": round(math.sqrt(variance), 5),
        }
    return summary


def iter_bursts_with_rows(csv_path: Path) -> Iterator[Burst]:
    """분사 중이고 착탄점이 있는 연속 구간. 원본 CSV 행 번호를 함께 낸다."""
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = [column.strip() for column in next(reader)]
        try:
            spray = header.index(SPRAY_COLUMN)
            time = header.index(TIME_COLUMN)
            ray = [header.index(column) for column in RAY_POINT_COLUMNS]
        except ValueError:
            logger.warning("dialect mismatch, skipped: %s", csv_path.name)
            return

        current = Burst([], [], [])
        for index, row in enumerate(reader):
            if not row:
                continue
            if row[spray].strip() == SPRAY_ON and row[ray[0]].strip():
                current.row_indices.append(index)
                current.points.append(
                    (float(row[ray[0]]), float(row[ray[1]]), float(row[ray[2]]))
                )
                current.times.append(float(row[time]))
                continue
            if current.row_indices:
                yield current
                current = Burst([], [], [])
        if current.row_indices:
            yield current


def _vec(row: Mapping[str, str], prefix: str) -> Vec3:
    return (
        float(row[f"{prefix} Position X"]),
        float(row[f"{prefix} Position Y"]),
        float(row[f"{prefix} Position Z"]),
    )


def _path_length(points: Sequence[Vec3]) -> float:
    return sum(math.dist(points[i], points[i + 1]) for i in range(len(points) - 1))


def csv_alignment_signal(csv_path: Path) -> SessionSignals:
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = [column.strip() for column in next(reader)]
        rows = [dict(zip(header, row)) for row in reader if row]
    cameras = [_vec(row, CAMERA_COLUMN) for row in rows]
    guns = [_vec(row, GUN_COLUMN) for row in rows]
    return SessionSignals(
        align=[math.dist(gun, camera) for gun, camera in zip(guns, cameras)],
        verify=[camera[2] for camera in cameras],
        path_length=_path_length(cameras),
    )


def bvh_poses(clip: bvh_skeleton.BvhClip) -> tuple[list[dict[str, Vec3]], SessionSignals]:
    wanted = (*pose_similarity.SCORING_JOINTS, pose_similarity.ROOT_JOINT)
    poses = [
        bvh_skeleton.forward_kinematics(clip, index, joints=wanted)
        for index in range(len(clip.frames))
    ]
    hands = [pose[ALIGNMENT_JOINTS[0]] for pose in poses]
    heads = [pose[ALIGNMENT_JOINTS[1]] for pose in poses]
    return poses, SessionSignals(
        align=[math.dist(hand, head) for hand, head in zip(hands, heads)],
        verify=[head[2] for head in heads],
        path_length=_path_length(heads),
    )


def pass_mean_pose(
    poses: Sequence[Mapping[str, Vec3]], frames: Sequence[int]
) -> dict[str, Vec3]:
    """패스 하나를 자세 표본 하나로 요약한다.

    프레임을 독립 표본으로 넣으면 긴 패스가 과대 가중되고 산포가 인위적으로 줄어든다.
    패스 내 산포가 4.2cm(전역 0.356m 대비)로 작아 평균이 안정적인 대표값이다.
    """
    totals: dict[str, list[float]] = {}
    counts: dict[str, int] = {}
    for frame in frames:
        if frame >= len(poses):
            continue
        for name, position in pose_similarity.normalize_pose(poses[frame]).items():
            slot = totals.setdefault(name, [0.0, 0.0, 0.0])
            for axis in range(3):
                slot[axis] += position[axis]
            counts[name] = counts.get(name, 0) + 1
    return {
        name: (slot[0] / counts[name], slot[1] / counts[name], slot[2] / counts[name])
        for name, slot in totals.items()
    }


def collect_session(
    bvh_path: Path, csv_path: Path, co_indexed: bool = False
) -> tuple[dict[str, list[dict[str, Vec3]]], SessionQuality]:
    """co_indexed면 자세와 CSV가 같은 프레임이라고 보고 작업자 기준 버킷으로 모은다."""
    clip = bvh_skeleton.parse_bvh(bvh_path)
    poses, bvh = bvh_poses(clip)
    csv_signals = csv_alignment_signal(csv_path)

    if co_indexed:
        alignment = pose_alignment.co_indexed_alignment(len(csv_signals.align))
        quality = co_indexed_quality(len(poses), len(csv_signals.align))
    else:
        alignment = pose_alignment.align_session(bvh.align, csv_signals.align)
        verification = pose_alignment.cross_validate(
            alignment.frame_for_row, bvh.verify, csv_signals.verify
        )
        coverage = (
            bvh.path_length / csv_signals.path_length if csv_signals.path_length > 0.0 else 0.0
        )
        quality = session_quality(alignment.residual, coverage, verification)

    by_bucket: dict[str, list[dict[str, Vec3]]] = defaultdict(list)
    if not quality.usable:
        return by_bucket, quality

    for burst in iter_bursts_with_rows(csv_path):
        pairs = (
            body_bucket_frames(burst, alignment.frame_for_row, poses)
            if co_indexed
            else bucket_frames(burst, alignment.frame_for_row)
        )
        for bucket, frames in pairs:
            summary = pass_mean_pose(poses, frames)
            if summary:
                by_bucket[bucket].append(summary)
    return by_bucket, quality


def build_payload(
    by_bucket: Mapping[str, Sequence[Mapping[str, Vec3]]],
    level: str,
    worker: str,
    accepted: int,
    rejected: Sequence[dict[str, object]],
) -> dict[str, object]:
    scored = frozenset(pose_similarity.SCORING_JOINTS)
    buckets: dict[str, object] = {}
    total = 0
    for bucket, samples in sorted(by_bucket.items()):
        if len(samples) < MIN_BUCKET_POSES:
            continue
        total += len(samples)
        selected = [
            {name: position for name, position in sample.items() if name in scored}
            for sample in samples
        ]
        buckets[bucket] = {"n": len(samples), "joints": summarize_joint_positions(selected)}

    return {
        "level": level,
        "status": "ok" if buckets else "insufficient_data",
        "reference_quality": "single_worker",
        "worker_count": 1,
        "worker_ids": [worker],
        "scoring_joints": list(pose_similarity.SCORING_JOINTS),
        "correction_joints": list(pose_similarity.CORRECTION_JOINTS),
        "session_count": accepted,
        "rejected_sessions": list(rejected),
        "pose_count": total,
        "min_bucket_poses": MIN_BUCKET_POSES,
        "buckets": buckets,
    }


def pair_sessions(bvh_root: Path, csv_root: Path) -> list[tuple[str, Path, Path]]:
    csv_index = {path.stem: path for path in csv_root.glob("**/*.csv")}
    pairs: list[tuple[str, Path, Path]] = []
    for bvh_path in sorted(bvh_root.glob("**/*.bvh")):
        stem = bvh_path.name.removesuffix("_Recorded.bvh")
        csv_path = csv_index.get(stem)
        if csv_path is None:
            logger.warning("no CSV for %s", bvh_path.name)
            continue
        pairs.append((stem, bvh_path, csv_path))
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build expert pose reference per level from BVH.")
    parser.add_argument("--bvh-root", type=Path, required=True)
    parser.add_argument("--csv-root", type=Path, required=True)
    parser.add_argument("--level", default="all", help="Level key such as 10, or all.")
    parser.add_argument("--exclude-date", default=None, help="Leave-one-date-out fold to drop.")
    parser.add_argument(
        "--co-indexed",
        action="store_true",
        help="자세와 CSV가 같은 프레임으로 들어온다고 단언한다. 프레임 수가 다르면 거부된다.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    if not args.bvh_root.is_dir() or not args.csv_root.is_dir():
        logger.error("roots not found: %s %s", args.bvh_root, args.csv_root)
        return 1

    worker = args.bvh_root.name.removeprefix("bvh_")
    levels: dict[str, list[tuple[str, Path, Path]]] = defaultdict(list)
    for stem, bvh_path, csv_path in pair_sessions(args.bvh_root, args.csv_root):
        if args.exclude_date is not None and args.exclude_date in bvh_path.parent.name:
            continue
        levels[stem.split("_")[0]].append((stem, bvh_path, csv_path))

    targets = sorted(levels) if args.level == "all" else [args.level]
    payloads: dict[str, object] = {}
    for level in targets:
        merged: dict[str, list[dict[str, Vec3]]] = defaultdict(list)
        rejected: list[dict[str, object]] = []
        accepted = 0
        for stem, bvh_path, csv_path in levels.get(level, []):
            by_bucket, quality = collect_session(bvh_path, csv_path, args.co_indexed)
            if not quality.usable:
                rejected.append({
                    "session": stem,
                    "reason": quality.reason,
                    "residual_mm": round(quality.residual * 1000.0, 1),
                    "coverage": round(quality.coverage, 3),
                    "verification": round(quality.verification, 3),
                })
                continue
            accepted += 1
            for bucket, samples in by_bucket.items():
                merged[bucket].extend(samples)
        payload = build_payload(merged, level, worker, accepted, rejected)
        payload["bucket_frame"] = "body" if args.co_indexed else "world"
        payloads[level] = payload
        logger.info(
            "level %s: %d accepted, %d rejected, %d buckets",
            level, accepted, len(rejected), len(merged),
        )

    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / "expert_pose_reference.json"
    output.write_text(json.dumps(payloads, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("wrote %s (%d level(s))", output, len(payloads))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
