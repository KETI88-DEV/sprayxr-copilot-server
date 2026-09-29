#!/usr/bin/env python3
"""자세 유사도 파이프라인을 **세 단계로 나눈다** — 각자 부를 수 있고 중간 출력을 넘긴다.

```
① resolve : 시퀀스 → 어느 숙련자 버킷인가            → ResolveResult
② align   : 착탄 경로 + 버킷 프로토타입 → 위상 대응   → AlignResult
③ score   : 위상 자세 + 위상 레퍼런스 → 점수·교정     → ScoreResult
```

## 왜 나누나

한 덩어리면 "점수가 이상하다"가 왔을 때 **어디가 이상한지 못 가린다.** 버킷을 잘못
찾았는지, 정렬이 휘었는지, σ가 큰 것인지가 한 숫자에 섞인다. 나누면 각 단계의 출력이
그 자체로 판정 가능하고, 중간 산물을 저장해 두고 뒤 단계만 다시 돌릴 수 있다.

각 결과는 `to_dict()`로 JSON이 된다. 메인 에이전트가 ①만 불러 버킷을 확인하고,
②를 따로 불러 정렬을 보고, ③만 다시 부르는 것이 가능하다.

## 단계 사이에 넘어가는 것

- ① → ②: `band`(버킷 키)와 `prototype`(그 버킷의 착탄 경로 무게중심)
- ② → ③: `phase_of`(query 프레임 → 위상) 와 `warp`
- ③: 위상별 자세를 위상별 레퍼런스에 대고 잰 σ·점수·교정

**어느 단계든 못 하면 그 자리에서 이유를 남기고 멈춘다.** 다음 단계가 없는 값을 지어내지
않는다 — 착탄이 안 움직이는 시퀀스(건 고정)는 ②에서 `path_missing`으로 멈추고,
호출자가 점 레퍼런스 경로로 내려갈지 정한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Final, Mapping, Sequence

if __package__:
    from . import arm_reference, sequence_dtw, world_axes
else:
    import arm_reference
    import sequence_dtw
    import world_axes

Vec3 = tuple[float, float, float]

# 단계마다 낼 수 있는 사유의 닫힌 집합. 지어낸 문자열이 나가면 소비자가 모르는 값을 받는다.
STAGE_REASONS: Final[frozenset[str]] = frozenset(
    {
        "reference_data_missing",
        "reference_single_worker",
        "reach_unavailable",
        "path_missing",
        "prototype_missing",
        "alignment_failed",
        "quality_insufficient",
        "phase_reference_missing",
    }
)


@dataclass(frozen=True)
class ResolveResult:
    """① 어느 숙련자 버킷인가."""

    ok: bool
    reason: str | None
    band: str | None = None
    reach_m: float | None = None
    support: int | None = None
    workers: int | None = None
    quality: str = "unavailable"
    prototype: tuple[Vec3, ...] = ()
    prototype_support: Mapping[str, float] | None = None
    has_phase_reference: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": "resolve",
            "ok": self.ok,
            "reason": self.reason,
            "band": self.band,
            "reach_m": None if self.reach_m is None else round(self.reach_m, 4),
            "support": self.support,
            "workers": self.workers,
            "reference_quality": self.quality,
            "prototype_points": len(self.prototype),
            "prototype_support": (
                None
                if self.prototype_support is None
                else {k: round(float(v), 4) for k, v in self.prototype_support.items()}
            ),
            "has_phase_reference": self.has_phase_reference,
        }


@dataclass(frozen=True)
class AlignResult:
    """② 착탄 경로를 버킷 프로토타입에 맞춘 결과."""

    ok: bool
    reason: str | None
    # query 프레임 인덱스 → 위상. 솎기 전 **원본 프레임** 기준이다.
    phase_of: tuple[tuple[int, int], ...] = ()
    phases: int = 0
    warp: float = 0.0
    cost: float = math.inf
    path_span_m: float = 0.0
    covered_phases: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": "align",
            "ok": self.ok,
            "reason": self.reason,
            "phases": self.phases,
            "covered_phases": self.covered_phases,
            # 휜 양은 **지워지지 않고 보고된다.** 크면 속도 배분이 숙련자와 다르다는 뜻이다.
            "warp": round(self.warp, 4),
            "cost_m": None if self.cost == math.inf else round(self.cost, 4),
            "path_span_m": round(self.path_span_m, 4),
            "frame_to_phase": [list(pair) for pair in self.phase_of],
        }


@dataclass(frozen=True)
class ScoreResult:
    """③ 위상 자세를 위상 레퍼런스에 대고 잰 결과."""

    ok: bool
    reason: str | None
    score: float | None = None
    percentile: int | None = None
    sigmas: tuple[arm_reference.JointSigma, ...] = ()
    corrections: tuple[arm_reference.Correction, ...] = ()
    scored_phases: int = 0
    mode: str = "phase"

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": "score",
            "ok": self.ok,
            "reason": self.reason,
            "mode": self.mode,
            "similarity_score": None if self.score is None else round(self.score, 4),
            "expert_percentile": self.percentile,
            "scored_phases": self.scored_phases,
            "scored_joints": [item.joint for item in self.sigmas],
            "joint_sigmas": {item.joint: round(item.sigma, 3) for item in self.sigmas},
            "corrections": [_correction_dict(item) for item in self.corrections],
        }


def _correction_dict(item: arm_reference.Correction) -> dict[str, Any]:
    return {
        "joint": item.joint,
        "label": item.label,
        "severity": item.severity,
        # 각도도 방향도 싣지 않는다 — `correction_mode="joint_only"`.
        "angle_delta_deg": None,
        "direction": None,
        "evidence": {
            "sigma": round(item.sigma, 3),
            "distance_m": round(item.distance_m, 4),
            "radial_sigma": None if item.radial_sigma is None else round(item.radial_sigma, 3),
            "angular_sigma": None if item.angular_sigma is None else round(item.angular_sigma, 3),
            "radius_delta_m": (
                None if item.radius_delta_m is None else round(item.radius_delta_m, 4)
            ),
            "expected_gain": (
                None if item.expected_gain is None else round(item.expected_gain, 4)
            ),
        },
        "correction_text": f"{item.label}을(를) 보세요.",
    }


# --- ① reference 찾기 --------------------------------------------------------


def resolve(
    reach_m: float | None, buckets: Mapping[str, Any] | None
) -> ResolveResult:
    """reach로 버킷을 고른다. 없으면 **추정하지 않는다.**"""
    if reach_m is None:
        return ResolveResult(ok=False, reason="reach_unavailable")
    if not isinstance(buckets, Mapping):
        return ResolveResult(ok=False, reason="reference_data_missing")
    band = arm_reference.reach_band(reach_m)
    entry = buckets.get(band)
    if not isinstance(entry, Mapping):
        return ResolveResult(ok=False, reason="reference_data_missing", band=band, reach_m=reach_m)
    quality = arm_reference.reference_quality(entry)
    if quality == "unavailable":
        return ResolveResult(
            ok=False, reason="reference_data_missing", band=band, reach_m=reach_m,
            support=int(entry.get("n", 0) or 0), workers=int(entry.get("workers", 0) or 0),
        )
    raw = entry.get("prototype")
    prototype: tuple[Vec3, ...] = ()
    if isinstance(raw, Sequence):
        prototype = tuple(
            (float(p[0]), float(p[1]), float(p[2]))
            for p in raw
            if isinstance(p, Sequence) and len(p) == 3
        )
    phases = entry.get("phases")
    return ResolveResult(
        ok=True,
        reason=None if quality == "multi_worker" else "reference_single_worker",
        band=band,
        reach_m=reach_m,
        support=int(entry.get("n", 0) or 0),
        workers=int(entry.get("workers", 0) or 0),
        quality=quality,
        prototype=prototype,
        prototype_support=(
            entry.get("prototype_support")
            if isinstance(entry.get("prototype_support"), Mapping)
            else None
        ),
        has_phase_reference=isinstance(phases, Sequence) and len(phases) > 0,
    )


# --- ② DTW 정렬 --------------------------------------------------------------


def align(
    impact_points: Sequence[Vec3], resolved: ResolveResult
) -> AlignResult:
    """착탄 경로를 버킷 프로토타입에 맞춘다.

    **착탄이 안 움직이면 정렬할 모양이 없다**(`path_missing`). 실물 샘플 SEQ-002가
    그 경우다 — 건을 고정해 `raypoint`가 전 프레임 같다. 억지로 정렬하면 잡음에 맞춘다.
    """
    if not resolved.ok:
        return AlignResult(ok=False, reason=resolved.reason)
    if not resolved.prototype:
        return AlignResult(ok=False, reason="prototype_missing")
    shape = sequence_dtw.path_shape(impact_points)
    if shape is None:
        return AlignResult(ok=False, reason="path_missing")

    indices = sequence_dtw.decimate(shape)
    thinned = [shape[i] for i in indices]
    path, cost = sequence_dtw.align(thinned, resolved.prototype)
    if not path:
        return AlignResult(ok=False, reason="alignment_failed")

    phase_of = tuple((indices[q], p) for q, p in path)
    span = max(math.dist(point, shape[0]) for point in shape)
    return AlignResult(
        ok=True,
        reason=resolved.reason,
        phase_of=phase_of,
        phases=len(resolved.prototype),
        warp=sequence_dtw.warp_amount(path, len(thinned), len(resolved.prototype)),
        cost=cost,
        path_span_m=span,
        covered_phases=len({p for _, p in path}),
    )


# --- ③ 유사도 측정 -----------------------------------------------------------


def score(
    frame_poses: Sequence[Mapping[str, Vec3]],
    aligned: AlignResult,
    bucket: Mapping[str, Any],
    *,
    excluded: Sequence[str] = (),
    labels: Mapping[str, str] | None = None,
) -> ScoreResult:
    """위상마다 자세를 모아 위상 레퍼런스에 대고 잰다."""
    if not aligned.ok:
        return ScoreResult(ok=False, reason=aligned.reason)
    phase_reference = bucket.get("phases")
    if not isinstance(phase_reference, Sequence) or not phase_reference:
        return ScoreResult(ok=False, reason="phase_reference_missing")

    per_phase: list[dict[str, list[list[float]]]] = [{} for _ in range(aligned.phases)]
    for frame_index, phase in aligned.phase_of:
        if not 0 <= phase < aligned.phases or frame_index >= len(frame_poses):
            continue
        for name, point in frame_poses[frame_index].items():
            slot = per_phase[phase].setdefault(name, [[], [], []])
            for axis in range(3):
                slot[axis].append(point[axis])

    phase_poses: list[Mapping[str, Vec3] | None] = []
    for slot in per_phase:
        if not slot:
            phase_poses.append(None)
            continue
        phase_poses.append(
            {
                name: (
                    sum(axes[0]) / len(axes[0]),
                    sum(axes[1]) / len(axes[1]),
                    sum(axes[2]) / len(axes[2]),
                )
                for name, axes in slot.items()
                if all(axes)
            }
        )

    reference = [
        item if isinstance(item, Mapping) else None
        for item in list(phase_reference)[: aligned.phases]
    ]
    sigmas = arm_reference.phase_sigmas(phase_poses, reference, excluded=excluded)
    if len(sigmas) < arm_reference.MIN_SCORED_JOINTS:
        return ScoreResult(ok=False, reason="quality_insufficient")

    value = arm_reference.similarity_score(sigmas)
    if value is None:
        return ScoreResult(ok=False, reason="quality_insufficient")
    corrections = arm_reference.build_corrections(
        sigmas, labels or {}, support=aligned.covered_phases
    )
    scored = sum(
        1
        for pose, entry in zip(phase_poses, reference)
        if pose is not None
        and isinstance(entry, Mapping)
        and int(entry.get("n", 0) or 0) >= arm_reference.MIN_PHASE_SAMPLES
    )
    return ScoreResult(
        ok=True,
        reason=aligned.reason,
        score=value,
        percentile=arm_reference.expert_percentile(bucket, value, mode="phase"),
        sigmas=tuple(sigmas),
        corrections=tuple(corrections),
        scored_phases=scored,
    )


def reach_of(impact_points: Sequence[Vec3], tool_points: Sequence[Vec3]) -> float | None:
    """reach = 착탄 수직 − 건 수직. 양쪽에서 같은 정의여야 하므로 여기 한 곳에 둔다."""
    if not impact_points or not tool_points:
        return None
    impacts = sorted(world_axes.vertical_of(p) for p in impact_points)
    tools = sorted(world_axes.vertical_of(p) for p in tool_points)
    return impacts[len(impacts) // 2] - tools[len(tools) // 2]
