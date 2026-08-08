#!/usr/bin/env python3
"""레벨별 자세 레퍼런스를 leave-one-date-out으로 검증한다.

레퍼런스를 만든 회차로 그 회차를 채점하면 낙관 편향이 생긴다. 날짜 하나를 빼고 만든
레퍼런스로 빠진 날짜를 채점해야 "처음 보는 작업을 얼마나 설명하는가"가 나온다.

BVH는 자세와 건을 쥔 손의 궤적이 같은 프레임에서 나오므로 정합 문제가 없다. 다만
분사 신호가 없어 비분사 이동도 패스로 잡힌다. 지표는 절대 정확도가 아니라 자기정합성이다.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median, quantiles
from typing import Final, Mapping, Sequence

import build_pose_reference
import bvh_skeleton
import pose_similarity

logger = logging.getLogger(__name__)

Vec3 = tuple[float, float, float]
Samples = dict[str, list[dict[str, Vec3]]]

CALIBRATION_POINTS: Final = (10, 25, 50, 75, 90)
MIN_CALIBRATION_SAMPLES: Final = 100


@dataclass(frozen=True, slots=True)
class ScoreResult:
    scored: int
    matched: int
    similarities: list[float]


@dataclass(frozen=True, slots=True)
class FoldResult:
    level: str
    held_out: str
    scored: int
    matched: int
    similarities: list[float]


def score_samples(samples: Mapping[str, Sequence[Mapping[str, Vec3]]],
                  reference: Mapping[str, object]) -> ScoreResult:
    """레퍼런스에 없는 버킷은 점수를 만들지 않고 미매칭으로만 센다."""
    buckets = reference.get("buckets")
    scored = 0
    matched = 0
    similarities: list[float] = []
    for bucket, poses in samples.items():
        entry = buckets.get(bucket) if isinstance(buckets, Mapping) else None
        for pose in poses:
            scored += 1
            if not isinstance(entry, Mapping):
                continue
            sigmas = pose_similarity.joint_sigmas(pose, entry)
            if not sigmas:
                continue
            matched += 1
            similarities.append(pose_similarity.similarity_score(sigmas))
    return ScoreResult(scored, matched, similarities)


def summarize(folds: Sequence[FoldResult]) -> dict[str, object]:
    scored = sum(fold.scored for fold in folds)
    matched = sum(fold.matched for fold in folds)
    values = sorted(value for fold in folds for value in fold.similarities)
    summary: dict[str, object] = {
        "folds": len(folds),
        "scored": scored,
        "matched": matched,
        "match_rate": matched / scored if scored else 0.0,
        "p25": None,
        "p50": None,
        "p75": None,
    }
    if values:
        summary["p50"] = median(values)
        if len(values) >= 4:
            lower, _, upper = quantiles(values, n=4)
            summary["p25"] = lower
            summary["p75"] = upper
    return summary


def calibration_block(folds: Sequence[FoldResult]) -> dict[str, float] | None:
    """MCP가 질의 점수의 숙련자 대비 위치를 매길 때 읽는 백분위 사다리.

    교차적합된 값만 담는다. 표본이 얇으면 만들지 않는다. 없는 백분위가 틀린 백분위보다 낫다.
    """
    values = sorted(value for fold in folds for value in fold.similarities)
    if len(values) < MIN_CALIBRATION_SAMPLES:
        return None
    last = len(values) - 1
    block: dict[str, float] = {"n": float(len(values))}
    for point in CALIBRATION_POINTS:
        block[f"p{point}"] = values[min(last, int(len(values) * point / 100.0))]
    return block


def collect_bvh_samples(bvh_path: Path) -> Samples:
    clip = bvh_skeleton.parse_bvh(bvh_path)
    poses, _ = build_pose_reference.bvh_poses(clip)
    burst = build_pose_reference.bvh_burst(poses, clip.frame_time)
    samples: Samples = defaultdict(list)
    for bucket, frames in build_pose_reference.body_bucket_frames(
        burst, burst.row_indices, poses
    ):
        summary = build_pose_reference.pass_mean_pose(poses, frames)
        if summary:
            samples[bucket].append(summary)
    return samples


def _merged(entries: Sequence[Samples]) -> Samples:
    merged: Samples = defaultdict(list)
    for entry in entries:
        for bucket, poses in entry.items():
            merged[bucket].extend(poses)
    return merged


def run_folds(by_date: Mapping[str, Sequence[Samples]], level: str, worker: str) -> list[FoldResult]:
    dates = sorted(by_date)
    folds: list[FoldResult] = []
    for held_out in dates:
        others = [item for date in dates if date != held_out for item in by_date[date]]
        if not others:
            continue
        reference = build_pose_reference.build_payload(
            _merged(others), level, worker, accepted=len(others), rejected=[]
        )
        if reference.get("status") != "ok":
            logger.warning("level %s fold %s: reference insufficient", level, held_out)
            continue
        for sample in by_date[held_out]:
            result = score_samples(sample, reference)
            folds.append(
                FoldResult(level, held_out, result.scored, result.matched, result.similarities)
            )
    return folds


def _write_calibration(reference: Path, calibration: Mapping[str, dict[str, float]]) -> int:
    """아티팩트에 백분위를 심는다. 보정이 없는 레벨은 기존 값을 지우고 비워 둔다."""
    if not reference.is_file():
        logger.error("reference not found: %s", reference)
        return 0
    payload = json.loads(reference.read_text(encoding="utf-8"))
    written = 0
    for level, entry in payload.items():
        if not isinstance(entry, dict):
            continue
        block = calibration.get(level)
        if block is None:
            entry.pop("expert_similarity", None)
            continue
        entry["expert_similarity"] = block
        written += 1
    reference.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leave-one-date-out check for the pose reference.")
    parser.add_argument("--bvh-root", type=Path, required=True)
    parser.add_argument("--level", default="all", help="Level key such as 10, or all.")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON report path.")
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="레퍼런스 아티팩트에 교차적합 백분위(expert_similarity)를 써넣는다.",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    if not args.bvh_root.is_dir():
        logger.error("bvh root not found: %s", args.bvh_root)
        return 1

    worker = args.bvh_root.name.removeprefix("bvh_")
    cached: dict[str, dict[str, list[Samples]]] = defaultdict(lambda: defaultdict(list))
    paths = sorted(args.bvh_root.glob("**/*.bvh"))
    for index, path in enumerate(paths, 1):
        level = path.name.split("_")[0]
        if args.level != "all" and level != args.level:
            continue
        cached[level][path.parent.name].append(collect_bvh_samples(path))
        logger.info("scanned %d/%d %s", index, len(paths), path.name)

    report: dict[str, object] = {}
    calibration: dict[str, dict[str, float]] = {}
    for level in sorted(cached, key=lambda value: (len(value), value)):
        folds = run_folds(cached[level], level, worker)
        summary = summarize(folds)
        block = calibration_block(folds)
        if block is not None:
            calibration[level] = block
        summary["dates"] = sorted(cached[level])
        report[level] = summary
        logger.info(
            "level %s: folds %d, matched %d/%d (%.1f%%), similarity p50 %s",
            level, summary["folds"], summary["matched"], summary["scored"],
            100.0 * float(summary["match_rate"]),  # type: ignore[arg-type]
            "n/a" if summary["p50"] is None else f"{float(summary['p50']):.3f}",  # type: ignore[arg-type]
        )

    if args.reference is not None:
        written = _write_calibration(args.reference, calibration)
        logger.info("wrote calibration for %d level(s) into %s", written, args.reference)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
