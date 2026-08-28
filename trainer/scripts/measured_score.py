#!/usr/bin/env python3
"""실측 3점 축의 유사도와 교정.

근거가 측정값이라는 것이 이 축의 전부다. 관절 축은 IK 산출물을 채점하지만 여기서는
머리·양손의 실제 위치를 쓴다. 그래서 교정문이 "오른손을 5cm 띄우세요"가 되고 작업자가
그대로 따를 수 있다.

가중은 오른손 > 머리 > 왼손이다. 오른손이 건이고 도막 품질을 직접 결정한다. 왼손은
플래시라 작업 결과와의 인과가 약하다.

각도를 내지 않는다. REQ-19 correction_target_gate 통과 전까지 각도 지시는 근거보다
높은 정밀도를 표시한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Mapping, Sequence

Vec3 = tuple[float, float, float]

DEFAULT_WEIGHTS: Final[dict[str, float]] = {"right": 1.0, "head": 0.4, "left": 0.15}
POINT_LABELS: Final[dict[str, str]] = {"right": "오른손", "head": "머리", "left": "왼손"}
POINT_ORDER: Final[tuple[str, ...]] = ("right", "head", "left")

# 산포 하한. 측정 잡음이 작을 때 σ가 무한히 부풀지 않게 한다.
SIGMA_FLOOR_M: Final = 0.02
CORRECTION_SIGMA_THRESHOLD: Final = 2.0
MAX_SIGMA: Final = 6.0
MIN_CORRECTION_CM: Final = 1.0


@dataclass(frozen=True, slots=True)
class PointDelta:
    """한 점의 이탈. local_delta는 질의에서 레퍼런스로 가는 스트로크 프레임 벡터다."""

    point: str
    sigma: float
    local_delta: Vec3


@dataclass(frozen=True, slots=True)
class MeasuredResult:
    score: float
    deltas: list[PointDelta]
    samples: int


def _mean(points: Sequence[Vec3]) -> Vec3:
    count = len(points)
    return (
        sum(point[0] for point in points) / count,
        sum(point[1] for point in points) / count,
        sum(point[2] for point in points) / count,
    )


def score_track(
    query: Mapping[str, list[Vec3]],
    reference: Mapping[str, list[Vec3]],
    spreads: Mapping[str, float],
    weights: Mapping[str, float],
) -> MeasuredResult:
    """정렬 격자 위 세 점의 가중 유사도. 못 잰 점은 0점이 아니라 제외다."""
    deltas: list[PointDelta] = []
    samples = 0
    for name in POINT_ORDER:
        left = query.get(name) or []
        right = reference.get(name) or []
        if not left or not right:
            continue
        samples = max(samples, min(len(left), len(right)))
        mean_delta = _mean(
            [(r[0] - q[0], r[1] - q[1], r[2] - q[2]) for q, r in zip(left, right)]
        )
        spread = max(spreads.get(name, SIGMA_FLOOR_M), SIGMA_FLOOR_M)
        magnitude = math.sqrt(sum(axis * axis for axis in mean_delta))
        deltas.append(
            PointDelta(
                point=name,
                sigma=min(MAX_SIGMA, magnitude / spread),
                local_delta=mean_delta,
            )
        )

    if not deltas:
        return MeasuredResult(score=0.0, deltas=[], samples=0)

    total_weight = sum(weights.get(delta.point, 0.0) for delta in deltas)
    if total_weight <= 0.0:
        return MeasuredResult(score=0.0, deltas=deltas, samples=samples)
    weighted = (
        sum(weights.get(delta.point, 0.0) * delta.sigma for delta in deltas) / total_weight
    )
    return MeasuredResult(
        score=math.exp(-0.5 * weighted * weighted), deltas=deltas, samples=samples
    )


def corrections_text(deltas: Sequence[PointDelta]) -> list[str]:
    """통상 범위를 벗어난 점만. 각도가 아니라 센티미터로 낸다."""
    lines: list[str] = []
    for delta in sorted(deltas, key=lambda item: item.sigma, reverse=True):
        if delta.sigma <= CORRECTION_SIGMA_THRESHOLD:
            continue
        parts: list[str] = []
        normal_cm = delta.local_delta[0] * 100.0
        along_cm = delta.local_delta[1] * 100.0
        up_cm = delta.local_delta[2] * 100.0
        # 어간만 모으고 마지막에 "고 "로 이어 붙인다. 각 조각에 "고"를 넣고 잘라내면
        # "15cm 올리 유지하세요" 같은 비문이 된다.
        if abs(normal_cm) >= MIN_CORRECTION_CM:
            verb = "띄우" if normal_cm > 0 else "붙이"
            parts.append(f"벽에서 {abs(normal_cm):.0f}cm 더 {verb}")
        if abs(up_cm) >= MIN_CORRECTION_CM:
            verb = "올리" if up_cm > 0 else "내리"
            parts.append(f"{abs(up_cm):.0f}cm {verb}")
        if abs(along_cm) >= MIN_CORRECTION_CM:
            direction = "앞으로" if along_cm > 0 else "뒤로"
            parts.append(f"진행 방향 {direction} {abs(along_cm):.0f}cm 옮기")
        if not parts:
            continue
        label = POINT_LABELS.get(delta.point, delta.point)
        lines.append(f"{label}을 {'고 '.join(parts)}세요.")
    return lines


def apply_measured(
    query: Mapping[str, list[Vec3]],
    deltas: Sequence[PointDelta],
    fraction: float = 1.0,
) -> dict[str, list[Vec3]]:
    """교정을 따랐을 때의 궤적. 지목된 점만 fraction만큼 레퍼런스 쪽으로 옮긴다."""
    moves = {
        delta.point: delta.local_delta
        for delta in deltas
        if delta.sigma > CORRECTION_SIGMA_THRESHOLD
    }
    output: dict[str, list[Vec3]] = {}
    for name, points in query.items():
        shift = moves.get(name)
        if shift is None:
            output[name] = list(points)
            continue
        output[name] = [
            (
                point[0] + shift[0] * fraction,
                point[1] + shift[1] * fraction,
                point[2] + shift[2] * fraction,
            )
            for point in points
        ]
    return output
