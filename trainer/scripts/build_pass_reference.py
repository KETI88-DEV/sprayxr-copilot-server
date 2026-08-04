#!/usr/bin/env python3
"""전문가 CSV 코퍼스에서 패스 버킷별 궤적 지표 분포를 만든다.

산출물은 판정의 **처방** 쪽이다. 위반 판정(규범)은 여기서 나온 분포에서 임계를
역산해 쓰되, 분포 자체를 '정답'으로 규정하지 않는다. 6인 실측에서 스트로크 방향은
수평 36%/사선 31%/수직 33%로 갈렸고 '가로 일직선'은 규범이 아니었다.

버킷 키는 pass_segmentation.bucket_key와 동일하며, 관절 월드 좌표가 확보되면
자세 레퍼런스가 같은 키를 재사용한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Final, Iterator, Sequence

from pass_segmentation import PassMetrics, Vec3, segment_burst
from simulator_csv import RAY_POINT_COLUMNS

logger = logging.getLogger(__name__)

SPRAY_COLUMN: Final = "분사"
TIME_COLUMN: Final = "시간(초)"
SPRAY_ON: Final = "On"
MIN_BUCKET_PASSES: Final = 30
METRIC_NAMES: Final = ("straightness_rms_mm", "elevation_deg", "speed_mm_per_s", "length_mm")
PERCENTILES: Final = (10, 25, 50, 75, 90, 99)


def iter_bursts(csv_path: Path) -> Iterator[tuple[list[Vec3], list[float]]]:
    """분사 중이고 착탄점이 있는 연속 구간을 버스트로 낸다."""
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

        points: list[Vec3] = []
        times: list[float] = []
        for row in reader:
            if not row:
                continue
            if row[spray].strip() == SPRAY_ON and row[ray[0]].strip():
                points.append((float(row[ray[0]]), float(row[ray[1]]), float(row[ray[2]])))
                times.append(float(row[time]))
                continue
            if points:
                yield points, times
                points, times = [], []
        if points:
            yield points, times


def worker_of(path: Path, root: Path) -> str:
    top = path.relative_to(root).parts[0]
    return top if "_" not in top else top.split("_")[0]


def level_of(path: Path) -> str:
    return path.name.split("_")[0]


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    position = (len(values) - 1) * q / 100.0
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (position - low)


def summarize(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    count = len(ordered)
    mean = sum(ordered) / count
    variance = sum((v - mean) ** 2 for v in ordered) / count
    summary: dict[str, float | int] = {
        "n": count,
        "mean": round(mean, 3),
        "std": round(math.sqrt(variance), 3),
    }
    for q in PERCENTILES:
        summary[f"p{q}"] = round(percentile(ordered, q), 3)
    return summary


def collect_passes(paths: Sequence[Path], root: Path) -> dict[str, list[tuple[str, PassMetrics]]]:
    by_bucket: dict[str, list[tuple[str, PassMetrics]]] = defaultdict(list)
    for index, path in enumerate(paths, 1):
        worker = worker_of(path, root)
        for points, times in iter_bursts(path):
            for metrics in segment_burst(points, times):
                by_bucket[metrics.bucket].append((worker, metrics))
        if index % 50 == 0:
            logger.info("scanned %d/%d files", index, len(paths))
    return by_bucket


def build_payload(
    by_bucket: dict[str, list[tuple[str, PassMetrics]]], level: str, workers: set[str]
) -> dict[str, object]:
    buckets: dict[str, object] = {}
    total = 0
    for bucket, entries in sorted(by_bucket.items()):
        if len(entries) < MIN_BUCKET_PASSES:
            continue
        total += len(entries)
        bucket_workers = sorted({worker for worker, _ in entries})
        metrics: dict[str, object] = {
            name: summarize([getattr(item, name) for _, item in entries])
            for name in METRIC_NAMES
        }
        metrics["worker_ids"] = bucket_workers
        metrics["worker_count"] = len(bucket_workers)
        buckets[bucket] = metrics

    quality = "multi_worker" if len(workers) >= 2 else "single_worker"
    return {
        "level": level,
        "status": "ok" if buckets else "insufficient_data",
        "reference_quality": quality,
        "worker_count": len(workers),
        "worker_ids": sorted(workers),
        "pass_count": total,
        "min_bucket_passes": MIN_BUCKET_PASSES,
        "buckets": buckets,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build expert pass-trajectory reference per level.")
    parser.add_argument("--sources", type=Path, required=True, help="Professionals corpus root.")
    parser.add_argument("--level", default="all", help="Level key such as 10, 1-A, or all.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    root: Path = args.sources
    if not root.is_dir():
        logger.error("sources not found: %s", root)
        return 1

    paths = sorted(p for p in root.glob("**/*.csv"))
    levels: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        levels[level_of(path)].append(path)
    targets = sorted(levels) if args.level == "all" else [args.level]

    args.out.mkdir(parents=True, exist_ok=True)
    payloads: dict[str, object] = {}
    for level in targets:
        selected = levels.get(level, [])
        if not selected:
            logger.warning("no CSV for level %s", level)
            continue
        workers = {worker_of(path, root) for path in selected}
        by_bucket = collect_passes(selected, root)
        payloads[level] = build_payload(by_bucket, level, workers)
        logger.info("level %s: %d files, %d buckets", level, len(selected), len(by_bucket))

    output = args.out / "expert_pass_reference.json"
    output.write_text(json.dumps(payloads, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("wrote %s (%d level(s))", output, len(payloads))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
