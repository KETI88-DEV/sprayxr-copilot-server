#!/usr/bin/env python3
"""교정을 따랐을 때 유사도가 실제로 얼마나 오르는지 잰다.

지표의 전부는 레퍼런스 분리다. 교정에 쓴 분포로 다시 채점하면 교정된 관절이 정의상
기준 위치에 놓이므로 향상이 자동 보장된다. 그래서 교정 레퍼런스(A)와 채점 레퍼런스(B)를
서로 겹치지 않는 날짜에서 만들고, 질의 패스는 둘 다에 쓰이지 않은 세 번째 날짜에서 뽑는다.

두 번째 방어선은 집합 포함이다. 채점 관절 12개가 교정 관절 6개를 진부분집합으로 포함하므로
지시를 완벽히 따라도 나머지 6개는 그대로 남아 유사도가 1.0이 되지 않는다.

지표는 gap closure G = (S1-S0)/(1-S0)다. 이미 기준에 가까운 패스는 분모가 작아 지표가
요동치므로 개선 여지가 있는 경우만 센다.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Final, Mapping, Sequence

import build_pose_reference
import pose_similarity

logger = logging.getLogger(__name__)

Vec3 = tuple[float, float, float]

# D0 >= 1.0, 즉 평균 sigma가 1을 넘는 패스만 센다. exp(-0.5)와 같은 경계다.
MAX_ELIGIBLE_BEFORE: Final = 0.607


@dataclass(frozen=True, slots=True)
class Outcome:
    sequence: int
    bucket: str
    before: float
    after: float
    corrections: int

    @property
    def closure(self) -> float | None:
        return pose_similarity.gap_closure(self.before, self.after)


def measure_correction(
    pose: Mapping[str, Vec3],
    bucket: str,
    correction: Mapping[str, object],
    scoring: Mapping[str, object],
    sequence: int = 0,
) -> Outcome | None:
    """교정은 correction 분포로 만들고 점수는 scoring 분포로 잰다. 둘은 달라야 한다."""
    correction_buckets = correction.get("buckets")
    scoring_buckets = scoring.get("buckets")
    if not isinstance(correction_buckets, Mapping) or not isinstance(scoring_buckets, Mapping):
        return None
    source = correction_buckets.get(bucket)
    target = scoring_buckets.get(bucket)
    if not isinstance(source, Mapping) or not isinstance(target, Mapping):
        return None

    corrections = pose_similarity.build_corrections(pose, source)
    if not corrections:
        return None

    fixed = pose_similarity.apply_corrections(pose, source, corrections)
    before = pose_similarity.similarity_score(pose_similarity.joint_sigmas(pose, target))
    after = pose_similarity.similarity_score(pose_similarity.joint_sigmas(fixed, target))
    return Outcome(sequence, bucket, before, after, len(corrections))


def eligible(outcomes: Sequence[Outcome]) -> list[Outcome]:
    """개선 여지가 있는 패스만. 이미 기준에 가까우면 분모가 작아 지표가 요동친다."""
    return [item for item in outcomes if item.before <= MAX_ELIGIBLE_BEFORE]


def summarize(outcomes: Sequence[Outcome]) -> dict[str, object]:
    closures = [item.closure for item in outcomes if item.closure is not None]
    return {
        "n": len(outcomes),
        "median_closure": median(closures) if closures else None,
        "median_before": median([item.before for item in outcomes]) if outcomes else None,
        "median_after": median([item.after for item in outcomes]) if outcomes else None,
    }


def _reference_from(samples: Sequence[dict[str, list[dict[str, Vec3]]]], level: str) -> dict[str, object]:
    merged: dict[str, list[dict[str, Vec3]]] = defaultdict(list)
    for entry in samples:
        for bucket, poses in entry.items():
            merged[bucket].extend(poses)
    return build_pose_reference.build_payload(merged, level, "expert", len(samples), [])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leave-one-date-out correction gain with split references.")
    parser.add_argument("--world-csv-root", type=Path, required=True)
    parser.add_argument("--level", default="10")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    if not args.world_csv_root.is_dir():
        logger.error("root not found: %s", args.world_csv_root)
        return 1

    by_date: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(args.world_csv_root.glob("**/*.csv")):
        if path.stem.split("_")[0] == args.level:
            by_date[path.parent.name].append(path)

    dates = sorted(by_date)
    if len(dates) < 3:
        logger.error("need at least 3 dates to split correction and scoring references, got %d", len(dates))
        return 1

    cached = {
        date: [build_pose_reference.collect_csv_session(path) for path in paths]
        for date, paths in by_date.items()
    }

    honest: list[Outcome] = []
    gamed: list[Outcome] = []
    for held_out in dates:
        others = [date for date in dates if date != held_out]
        half = max(1, len(others) // 2)
        first = [entry for date in others[:half] for entry in cached[date]]
        second = [entry for date in others[half:] for entry in cached[date]]
        if not first or not second:
            continue
        reference_a = _reference_from(first, args.level)
        reference_b = _reference_from(second, args.level)
        if reference_a.get("status") != "ok" or reference_b.get("status") != "ok":
            continue

        for path in by_date[held_out]:
            for sequence, bucket, pose in build_pose_reference.iter_csv_passes(path):
                split = measure_correction(pose, bucket, reference_a, reference_b, sequence)
                if split is not None:
                    honest.append(split)
                same = measure_correction(pose, bucket, reference_a, reference_a, sequence)
                if same is not None:
                    gamed.append(same)

    report = {
        "level": args.level,
        "dates": dates,
        "split_reference": summarize(eligible(honest)),
        "same_reference": summarize(eligible(gamed)),
    }
    print(f"레벨 {args.level} | 날짜 {len(dates)}개")
    for name, key in (("교정A / 채점B (정직)", "split_reference"), ("교정A / 채점A (게이밍)", "same_reference")):
        entry = report[key]
        closure = entry["median_closure"]
        rendered = "n/a" if closure is None else f"{float(closure):.3f}"
        print(
            f"  {name:<24} n={entry['n']:<5} 개선율 중앙 {rendered}"
            f"  유사도 {float(entry['median_before'] or 0):.3f} -> {float(entry['median_after'] or 0):.3f}"
        )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
