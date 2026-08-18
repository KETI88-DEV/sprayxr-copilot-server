#!/usr/bin/env python3
"""같은 스트로크를 다른 시퀀스에서 했을 때 자세가 얼마나 달랐는지 관절별로 보여준다.

측정 논리는 세 단계다.

1. 분할 - 분사 버스트를 방향 전환에서 잘라 패스로 나눈다. 버스트는 스트로크가 아니라
   평균 4개 패스를 담으므로 패스가 분석 단위다.
2. 분류 - 패스마다 (스트로크 종류, 높이대, 작업자 기준 진행 방향)으로 버킷 키를 만든다.
   방향을 월드 축이 아니라 몸 기준으로 적어 작업자가 어디를 보고 서 있든 같은 스트로크가
   같은 버킷에 들어간다.
3. 대조 - 같은 버킷이면서 다른 시퀀스에 있는 패스끼리 견준다. 패스를 평균 자세 하나로
   요약하고 루트를 원점으로 옮긴 뒤 수직축 둘레로만 회전시켜 정규화한다. 관절별 거리를
   숙련자 산포로 나눠 sigma를 내고, 유사도는 exp(-mean(sigma^2)/2)로 0~1에 둔다.

정렬에 자세를 쓰지 않는 것이 핵심이다. 자세로 짝을 고르면 보고할 차이를 스스로 줄이게 된다.
짝은 버킷으로만 정하고 버킷은 건 궤적에서 나온다.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping, Sequence

import build_pose_reference
import pose_similarity
from simulator_csv import has_world_joint_motion, joint_world_positions

logger = logging.getLogger(__name__)

Vec3 = tuple[float, float, float]

DEFAULT_PAIR_LIMIT: Final = 5
AXIS_LABELS: Final = ("오른쪽", "앞", "위")


@dataclass(frozen=True, slots=True)
class PassSample:
    sequence: int
    bucket: str
    pose: dict[str, Vec3]


def same_bucket_pairs(
    samples: Sequence[PassSample], limit: int = DEFAULT_PAIR_LIMIT
) -> list[tuple[PassSample, PassSample]]:
    """같은 버킷이면서 다른 시퀀스인 패스 짝. 같은 시퀀스끼리는 상황이 같아 비교 가치가 낮다."""
    grouped: dict[str, list[PassSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.bucket].append(sample)

    pairs: list[tuple[PassSample, PassSample]] = []
    for bucket in sorted(grouped):
        group = grouped[bucket]
        for index, left in enumerate(group):
            partner = next(
                (item for item in group[index + 1 :] if item.sequence != left.sequence), None
            )
            if partner is not None:
                pairs.append((left, partner))
            if len(pairs) >= limit:
                return pairs[:limit]
    return pairs[:limit]


def session_passes(csv_path: Path) -> list[PassSample]:
    """납품 형식 CSV 한 개를 패스 표본으로 바꾼다. 관절과 궤적이 같은 행이라 정합이 항등이다."""
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not has_world_joint_motion(rows, pose_similarity.CORRECTION_JOINTS):
        raise ValueError(
            f"{csv_path.name}: 관절 Position이 움직이지 않는다. 바인드 포즈 오프셋으로 보인다."
        )

    poses = [joint_world_positions(row) for row in rows]
    identity = list(range(len(poses)))
    samples: list[PassSample] = []
    for sequence, burst in enumerate(build_pose_reference.iter_bursts_with_rows(csv_path), 1):
        for bucket, frames in build_pose_reference.body_bucket_frames(burst, identity, poses):
            summary = build_pose_reference.pass_mean_pose(poses, frames)
            if summary:
                samples.append(PassSample(sequence, bucket, summary))
    return samples


def bucket_scales(samples: Sequence[PassSample], bucket: str) -> dict[str, float]:
    """해당 버킷 패스들의 관절별 산포. 레퍼런스 아티팩트가 없을 때 쓰는 자체 척도다."""
    poses = [sample.pose for sample in samples if sample.bucket == bucket]
    summary = build_pose_reference.summarize_joint_positions(poses)
    return {name: float(entry["spread"]) for name, entry in summary.items()}  # type: ignore[arg-type]


def reference_scales(reference: Path, level: str, bucket: str) -> dict[str, float] | None:
    payload = json.loads(reference.read_text(encoding="utf-8"))
    entry = payload.get(level)
    if not isinstance(entry, dict):
        return None
    found = entry.get("buckets", {}).get(bucket)
    if not isinstance(found, dict):
        return None
    return {name: float(item["spread"]) for name, item in found["joints"].items()}


def render(pair: tuple[PassSample, PassSample], comparison: pose_similarity.PoseComparison) -> str:
    left, right = pair
    lines = [
        f"버킷 {left.bucket} | 시퀀스 {left.sequence} vs {right.sequence}",
    ]
    if comparison.similarity is None:
        lines.append("  견줄 수 있는 관절이 없어 유사도를 내지 않는다.")
        return "\n".join(lines)

    lines.append(f"  자세 유사도 {comparison.similarity:.3f}")
    if not comparison.deltas:
        lines.append("  관절 차이 없음")
        return "\n".join(lines)

    lines.append(f"  {'관절':<12} {'거리':>8} {'sigma':>7}   신체 기준 차이")
    for delta in comparison.deltas:
        axes = ", ".join(
            f"{name} {value * 100:+.1f}cm"
            for name, value in zip(AXIS_LABELS, delta.body_delta)
        )
        lines.append(
            f"  {delta.label:<12} {delta.distance * 100:7.1f}cm {delta.sigma:7.2f}   {axes}"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare same-bucket passes across sequences.")
    parser.add_argument("--session", type=Path, required=True, help="납품 형식 CSV.")
    parser.add_argument("--reference", type=Path, default=None, help="숙련자 레퍼런스 아티팩트.")
    parser.add_argument("--level", default="10", help="--reference를 쓸 때의 레벨 키.")
    parser.add_argument("--pairs", type=int, default=DEFAULT_PAIR_LIMIT)
    parser.add_argument("--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    if not args.session.is_file():
        logger.error("session not found: %s", args.session)
        return 1

    samples = session_passes(args.session)
    pairs = same_bucket_pairs(samples, args.pairs)
    source = "레퍼런스 아티팩트" if args.reference else "이 세션 자체 분포"
    print(f"패스 {len(samples)}개, 비교 짝 {len(pairs)}개 | sigma 척도: {source}")

    for pair in pairs:
        left, right = pair
        scales: Mapping[str, float] | None = None
        if args.reference is not None:
            scales = reference_scales(args.reference, args.level, left.bucket)
        if scales is None:
            scales = bucket_scales(samples, left.bucket)
        comparison = pose_similarity.compare_poses(left.pose, right.pose, scales)
        print()
        print(render(pair, comparison))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
