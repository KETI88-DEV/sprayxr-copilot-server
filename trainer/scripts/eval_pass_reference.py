#!/usr/bin/env python3
"""Leave-one-out으로 패스 레퍼런스의 자기정합성을 검증한다.

숙련자 한 명을 빼고 나머지로 레퍼런스를 만든 뒤, 뺀 사람을 채점한다. 숙련자가 다른
숙련자 기준에서 대량으로 이탈로 잡히면 그 기준은 쓸 수 없다. 임계값은 이 분포에서
역산하며, 먼저 임계를 정하고 적용하는 순서가 아니다.
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Final

from build_pass_reference import build_payload, collect_passes, level_of, percentile, worker_of
from pass_coaching import assess_pass

logger = logging.getLogger(__name__)

SIGMA_THRESHOLDS: Final = (1.0, 1.5, 2.0, 3.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leave-one-out check for the expert pass reference.")
    parser.add_argument("--sources", type=Path, required=True, help="Professionals corpus root.")
    parser.add_argument("--level", default="10", help="Level key such as 10 or 1-A.")
    parser.add_argument("--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    root: Path = args.sources
    paths = [p for p in sorted(root.glob("**/*.csv")) if level_of(p) == args.level]
    if not paths:
        logger.error("no CSV for level %s", args.level)
        return 1

    by_worker: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        by_worker[worker_of(path, root)].append(path)
    workers = sorted(by_worker)

    print(f"레벨 {args.level} · 작업자 {len(workers)}명 · CSV {len(paths)}개")
    print(f"{'제외한 사람':<10}{'패스':>7}{'버킷매칭':>9}{'유사도 p10/p50/p90':>22}   " + "  ".join(f">{t}σ" for t in SIGMA_THRESHOLDS))

    overall: list[float] = []
    for held_out in workers:
        train_paths = [p for w in workers if w != held_out for p in by_worker[w]]
        reference = build_payload(
            collect_passes(train_paths, root), args.level, {w for w in workers if w != held_out}
        )
        if reference["status"] != "ok":
            print(f"{held_out:<10} insufficient_data")
            continue

        similarities: list[float] = []
        flagged = {t: 0 for t in SIGMA_THRESHOLDS}
        matched = total = 0
        for bucket_entries in collect_passes(by_worker[held_out], root).values():
            for _, metrics in bucket_entries:
                total += 1
                result = assess_pass(metrics, reference)
                if not result.matched:
                    continue
                matched += 1
                similarities.append(result.similarity_pct)
                worst = max((abs(s.sigma) for s in result.scores), default=0.0)
                for threshold in SIGMA_THRESHOLDS:
                    if worst > threshold:
                        flagged[threshold] += 1

        if not similarities:
            print(f"{held_out:<10} no matched pass")
            continue
        similarities.sort()
        overall += similarities
        rates = "  ".join(
            f"{100 * flagged[t] / len(similarities):4.1f}%" for t in SIGMA_THRESHOLDS
        )
        print(
            f"{held_out:<10}{total:>7}{100 * matched / total:>8.1f}%"
            f"{percentile(similarities, 10):>7.1f} /{percentile(similarities, 50):>6.1f} /{percentile(similarities, 90):>6.1f}"
            f"   {rates}"
        )

    if overall:
        overall.sort()
        print(
            f"\n전체 {len(overall)}패스 유사도 "
            f"p10={percentile(overall, 10):.1f} p50={percentile(overall, 50):.1f} p90={percentile(overall, 90):.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
