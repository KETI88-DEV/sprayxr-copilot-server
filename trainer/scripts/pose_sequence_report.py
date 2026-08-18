#!/usr/bin/env python3
"""시퀀스별 자세 유사도 통계.

채점 단위는 패스지만 보고 단위는 시퀀스다. 한 시퀀스(분사 버스트)는 패스 여러 개를
담으므로, 그 패스들의 유사도를 평균하고 반복해서 벗어난 관절을 센다.

한 번 벗어난 관절과 매번 벗어나는 관절은 뜻이 다르다. occurrence_rate가 그 구분이다.
레퍼런스에 없는 버킷의 패스는 평균에서 빼고 unavailable로만 센다. 못 잰 것을 0점으로
넣으면 시퀀스 평균이 근거 없이 내려간다.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import build_pose_reference
import pose_similarity

logger = logging.getLogger(__name__)

Vec3 = tuple[float, float, float]
PassEntry = tuple[int, str, Mapping[str, Vec3]]


@dataclass(frozen=True, slots=True)
class ProblemJoint:
    joint: str
    label: str
    occurrence_count: int
    occurrence_rate: float
    average_sigma: float


@dataclass(frozen=True, slots=True)
class SequenceStats:
    sequence: int
    evaluated: int
    unavailable: int
    average_similarity: float | None
    problem_joints: list[ProblemJoint]


def _problem_joints(
    hits: Mapping[str, list[float]], evaluated: int
) -> list[ProblemJoint]:
    joints = [
        ProblemJoint(
            joint=name,
            label=pose_similarity.JOINT_LABELS.get(name, name),
            occurrence_count=len(values),
            occurrence_rate=len(values) / evaluated,
            average_sigma=sum(values) / len(values),
        )
        for name, values in hits.items()
    ]
    joints.sort(key=lambda item: (item.occurrence_count, item.average_sigma), reverse=True)
    return joints


def sequence_stats(
    entries: Iterable[PassEntry], reference: Mapping[str, object]
) -> list[SequenceStats]:
    """entries는 iter_csv_passes가 내는 (시퀀스, 버킷, 정규화된 패스 자세)다."""
    buckets = reference.get("buckets")
    grouped: dict[int, list[tuple[str, Mapping[str, Vec3]]]] = defaultdict(list)
    for sequence, bucket, pose in entries:
        grouped[sequence].append((bucket, pose))

    correctable = frozenset(pose_similarity.CORRECTION_JOINTS)
    report: list[SequenceStats] = []
    for sequence in sorted(grouped):
        similarities: list[float] = []
        hits: dict[str, list[float]] = defaultdict(list)
        unavailable = 0
        for bucket, pose in grouped[sequence]:
            entry = buckets.get(bucket) if isinstance(buckets, Mapping) else None
            sigmas = (
                pose_similarity.joint_sigmas(pose, entry)
                if isinstance(entry, Mapping)
                else {}
            )
            if not sigmas:
                unavailable += 1
                continue
            similarities.append(pose_similarity.similarity_score(sigmas))
            for name, value in sigmas.items():
                if name in correctable and value > pose_similarity.CORRECTION_SIGMA_THRESHOLD:
                    hits[name].append(value)

        evaluated = len(similarities)
        report.append(
            SequenceStats(
                sequence=sequence,
                evaluated=evaluated,
                unavailable=unavailable,
                average_similarity=sum(similarities) / evaluated if evaluated else None,
                problem_joints=_problem_joints(hits, evaluated) if evaluated else [],
            )
        )
    return report


def as_payload(stats: Sequence[SequenceStats]) -> list[dict[str, object]]:
    """계약 5.4.3의 자세 집계 필드명에 맞춘 직렬화."""
    return [
        {
            "sequence_id": item.sequence,
            "evaluated_pass_count": item.evaluated,
            "unavailable_pass_count": item.unavailable,
            "average_similarity_score": (
                None if item.average_similarity is None else round(item.average_similarity, 3)
            ),
            "problem_joints": [
                {
                    "joint": joint.joint,
                    "label": joint.label,
                    "occurrence_count": joint.occurrence_count,
                    "occurrence_rate": round(joint.occurrence_rate, 3),
                    "average_sigma": round(joint.average_sigma, 2),
                }
                for joint in item.problem_joints
            ],
        }
        for item in stats
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-sequence pose similarity statistics.")
    parser.add_argument("--session", type=Path, required=True, help="납품 형식 월드 좌표 CSV.")
    parser.add_argument("--reference", type=Path, required=True, help="숙련자 레퍼런스 아티팩트.")
    parser.add_argument("--level", default="10")
    parser.add_argument("--out", type=Path, default=None, help="JSON 출력 경로.")
    parser.add_argument("--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    if not args.session.is_file() or not args.reference.is_file():
        logger.error("session or reference not found")
        return 1

    payload = json.loads(args.reference.read_text(encoding="utf-8"))
    entry = payload.get(args.level)
    if not isinstance(entry, dict) or entry.get("status") != "ok":
        logger.error("level %s has no usable reference", args.level)
        return 1

    stats = sequence_stats(build_pose_reference.iter_csv_passes(args.session), entry)
    rendered = as_payload(stats)

    scored = sum(item.evaluated for item in stats)
    missing = sum(item.unavailable for item in stats)
    print(f"시퀀스 {len(stats)}개 | 채점 패스 {scored}개 | 미채점 {missing}개")
    print(f"{'시퀀스':>6} {'채점':>5} {'미채점':>6} {'평균유사도':>10}  반복 이탈 관절")
    for item, row in zip(stats, rendered):
        average = "n/a" if item.average_similarity is None else f"{item.average_similarity:.3f}"
        joints = ", ".join(
            f"{joint.label} {joint.occurrence_rate:.0%}" for joint in item.problem_joints[:3]
        )
        print(f"{item.sequence:>6} {item.evaluated:>5} {item.unavailable:>6} {average:>10}  {joints}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rendered, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
