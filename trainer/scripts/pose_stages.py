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
from dataclasses import dataclass, field, replace
from typing import Any, Final, Mapping, Sequence

if __package__:
    from . import arm_reference, joint_names, sequence_dtw, skeleton_gate, world_axes
else:
    import arm_reference
    import joint_names
    import sequence_dtw
    import skeleton_gate
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
    # 교정마다 프레임별 월드 계획. 시뮬레이터가 그대로 적용한다.
    plans: tuple[tuple[Mapping[str, Any], ...], ...] = ()
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
            "corrections": [
                _correction_dict(item, self.plans[index] if index < len(self.plans) else ())
                for index, item in enumerate(self.corrections)
            ],
        }




# 뼈 길이가 이보다 틀어지면 그대로 옮길 수 없다 — 게이트 R1의 허용폭과 같은 값을 쓴다.
BONE_BREAK_TOLERANCE_M: Final = skeleton_gate.BONE_TOLERANCE_M


def _parent_of(joint: str) -> str | None:
    for child, parent in skeleton_gate.BONES:
        if child == joint:
            return parent
    return None


def _feasibility(
    item: arm_reference.Correction, plan: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """**그대로 옮기면 뼈가 깨지는가.**

    우리 레퍼런스는 관절마다 따로 분포를 갖는다 — 사슬을 모델링하지 않는다. 그래서
    한 관절만 목표로 옮기면 부모와의 뼈 길이가 달라진다. 좌표만 주고 끝내면 받는 쪽이
    불가능한 골격을 만든다. **깨지는 양을 같이 실어 IK가 필요하다는 것을 알린다.**

    `requires_ik`가 참이면 `per_frame.target`은 **도달 목표**이지 대입할 좌표가 아니다.
    """
    parent = _parent_of(item.joint)
    if parent is None or not plan:
        return {"requires_ik": False, "parent_joint": parent, "bone_break_m": None}
    worst = 0.0
    rig = None
    for step in plan:
        current = step.get("current")
        target = step.get("target")
        parent_point = step.get("parent")
        if not (isinstance(current, Sequence) and isinstance(target, Sequence)):
            continue
        if not isinstance(parent_point, Sequence):
            continue
        before = math.dist(tuple(current), tuple(parent_point))
        after = math.dist(tuple(target), tuple(parent_point))
        rig = before if rig is None else rig
        worst = max(worst, abs(after - before))
    return {
        "requires_ik": worst > BONE_BREAK_TOLERANCE_M,
        "parent_joint": parent,
        "bone_length_m": None if rig is None else round(rig, 4),
        "bone_break_m": round(worst, 4),
    }

def _round3(point: Vec3) -> list[float]:
    return [round(value, 4) for value in point]


def _correction_dict(
    item: arm_reference.Correction,
    plan: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """교정 하나를 JSON으로. **`target`이 시뮬레이터가 그대로 적용할 값이다.**

    `correction_text`는 사람이 읽는 서술이고, 기계가 쓰는 것은 `target.per_frame`이다.
    둘을 나누는 이유: 문장은 반증할 수 없고, 좌표는 적용해 보고 확인할 수 있다.
    """
    return {
        "joint": item.joint,
        # 시뮬레이터가 보낸 이름 그대로 돌려준다 — 받는 쪽이 어휘를 다시 옮기지 않게.
        "joint_sim": joint_names.to_sim(item.joint),
        "label": item.label,
        "severity": item.severity,
        # 각도 지시는 여전히 없다 — 오일러 회전순서가 미확정이다.
        # `angle_from_expert_deg`는 **두 위치벡터 사이의 각**이라 규약을 타지 않는다.
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
            "angle_from_expert_deg": (
                None
                if item.angle_from_expert_deg is None
                else round(item.angle_from_expert_deg, 1)
            ),
            "phase_span": (
                None
                if item.phase_span is None
                else [round(item.phase_span[0], 3), round(item.phase_span[1], 3)]
            ),
            # 모의값이다 — 아래 `target`을 그대로 적용했을 때의 점수 상승분.
            # `verify_correction`이 실제로 적용해 이 값과 맞는지 확인한다.
            "expected_gain": (
                None if item.expected_gain is None else round(item.expected_gain, 4)
            ),
        },
        "target": None if item.target is None else {
            **_feasibility(item, plan),
            # 시뮬레이터가 무엇을 바꿔야 하는지. **월드 좌표다** — 우리 몸 프레임을
            # 받는 쪽이 다시 구현하지 않아도 된다.
            "frame": "world",
            "mode": item.dominant,
            "radius_m": (
                None if item.target.radius_m is None else round(item.target.radius_m, 4)
            ),
            "heading": None if item.target.heading is None else _round3(item.target.heading),
            "per_frame": list(plan),
        },
        "correction_text": item.text(),
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
    frame_poses: Sequence[Mapping[str, Vec3]],  # **월드** 관절
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

    phase_poses, reference = _phase_view(frame_poses, aligned, phase_reference)
    sigmas = arm_reference.phase_sigmas(phase_poses, reference, excluded=excluded)
    if len(sigmas) < arm_reference.MIN_SCORED_JOINTS:
        return ScoreResult(ok=False, reason="quality_insufficient")

    value = arm_reference.similarity_score(sigmas)
    if value is None:
        return ScoreResult(ok=False, reason="quality_insufficient")
    worst_reference = _worst_phase_reference(sigmas, phase_poses, reference)
    corrections = arm_reference.build_corrections(
        sigmas, labels or {}, support=aligned.covered_phases, reference=worst_reference
    )
    plans = [tuple(correction_plan(item, frame_poses, aligned)) for item in corrections]

    # **`expected_gain`은 `target`을 그대로 적용했을 때의 상승분이다.**
    # 예전에는 "σ를 0으로 만들면"이라는 상한이었는데, `target`은 주범 성분 하나만
    # 고치므로 예고와 지시가 달랐다 — 실측에서 예고 0.0579에 실현 0.0284였다.
    # 예고가 지시를 따라가야 검증이 뜻을 갖는다.
    adjusted: list[arm_reference.Correction] = []
    for item, plan in zip(corrections, plans):
        gain = None
        if plan:
            moved = apply_plan(frame_poses, item.joint, plan)
            after_poses, after_reference = _phase_view(moved, aligned, phase_reference)
            after = arm_reference.similarity_score(
                arm_reference.phase_sigmas(after_poses, after_reference, excluded=excluded)
            )
            gain = None if after is None else after - value
        adjusted.append(replace(item, expected_gain=gain))

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
        corrections=tuple(adjusted),
        plans=tuple(plans),
        scored_phases=scored,
    )


def _phase_view(
    frame_poses: Sequence[Mapping[str, Vec3]],
    aligned: AlignResult,
    phase_reference: Sequence[Any],
) -> tuple[list[Mapping[str, Vec3] | None], list[Mapping[str, object] | None]]:
    """월드 관절 → 위상별 자세, 그리고 그에 맞춘 위상 레퍼런스.

    위상 평균을 내기 전에 **프레임마다 정규화한다** — 날것 좌표를 먼저 평균하면
    프레임마다 다른 몸 방향이 한 점에 섞인다.
    """
    normalized_frames = [arm_reference.normalize(joints) or {} for joints in frame_poses]

    per_phase: list[dict[str, list[list[float]]]] = [{} for _ in range(aligned.phases)]
    for frame_index, phase in aligned.phase_of:
        if not 0 <= phase < aligned.phases or frame_index >= len(normalized_frames):
            continue
        for name, point in normalized_frames[frame_index].items():
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
    return phase_poses, reference




def _worst_phase_reference(
    sigmas: Sequence[arm_reference.JointSigma],
    phase_poses: Sequence[Mapping[str, Vec3] | None],
    reference: Sequence[Mapping[str, object] | None],
) -> dict[str, object]:
    """교정 목표를 낼 때 쓸 관절별 레퍼런스. 위상마다 값이 달라 **하나를 골라야 한다.**

    `phase_span`의 시작 위상을 쓴다 — 벗어나기 **시작한** 곳이 교정의 기준이다.
    """
    out: dict[str, object] = {}
    for item in sigmas:
        index = 0
        if item.phase_span is not None and len(reference) > 0:
            index = min(len(reference) - 1, int(item.phase_span[0] * len(reference)))
        entry = reference[index] if index < len(reference) else None
        if isinstance(entry, Mapping):
            joints = entry.get("joints")
            if isinstance(joints, Mapping) and item.joint in joints:
                out[item.joint] = joints[item.joint]
    return out


def correction_plan(
    correction: arm_reference.Correction,
    frame_joints: Sequence[Mapping[str, Vec3]],
    aligned: AlignResult,
) -> list[dict[str, Any]]:
    """교정을 **프레임마다 월드 좌표로** 편다. 시뮬레이터가 그대로 쓰는 값이다.

    `phase_span` 안의 위상에 대응하는 프레임만 낸다 — 스트로크 전체를 고치라는 것이
    아니라 **벗어난 구간**을 고치라는 것이다.
    """
    target = correction.target
    if target is None or not aligned.ok:
        return []
    low, high = correction.phase_span or (0.0, 1.0)
    phases = max(aligned.phases, 1)
    wanted = {
        frame_index
        for frame_index, phase in aligned.phase_of
        if low <= phase / phases < high or (high >= 1.0 and phase == phases - 1)
    }
    out: list[dict[str, Any]] = []
    for frame_index in sorted(wanted):
        if frame_index >= len(frame_joints):
            continue
        joints = frame_joints[frame_index]
        body = arm_reference.body_frame(joints)
        current = joints.get(correction.joint)
        if body is None or current is None:
            continue
        local = arm_reference.normalize(joints)
        if local is None or correction.joint not in local:
            continue
        moved = arm_reference.corrected_point(local[correction.joint], target)
        if moved is None:
            continue
        world = arm_reference.to_world(body, moved)
        entry: dict[str, Any] = {
            "index": frame_index,
            "current": _round3(current),
            "target": _round3(world),
            "delta": _round3(
                (world[0] - current[0], world[1] - current[1], world[2] - current[2])
            ),
        }
        parent = _parent_of(correction.joint)
        parent_point = joints.get(parent) if parent else None
        if parent_point is not None:
            # 받는 쪽이 IK를 풀 때 필요하고, 뼈가 얼마나 깨지는지도 여기서 나온다.
            entry["parent"] = _round3(parent_point)
        out.append(entry)
    return out

def reach_of(impact_points: Sequence[Vec3], tool_points: Sequence[Vec3]) -> float | None:
    """reach = 착탄 수직 − 건 수직. 양쪽에서 같은 정의여야 하므로 여기 한 곳에 둔다."""
    if not impact_points or not tool_points:
        return None
    impacts = sorted(world_axes.vertical_of(p) for p in impact_points)
    tools = sorted(world_axes.vertical_of(p) for p in tool_points)
    return impacts[len(impacts) // 2] - tools[len(tools) // 2]


# --- 교정 정확도 -------------------------------------------------------------


@dataclass(frozen=True)
class VerifyResult:
    """교정을 **실제로 적용해** 다시 잰 결과.

    `"{label}을(를) 보세요"` 같은 문장은 적용할 수 없어 반증도 못 한다. `target.per_frame`이
    월드 좌표라 replay가 그대로 적용할 수 있고, 그래서 **예고와 실현을 견줄 수 있다.**

    이것이 재는 것은 **내적 일관성**이다 — 교정이 말한 대로 움직이면 예고한 만큼
    점수가 오르는가. **효능이 아니다** — 실제 작업자가 이 교정을 받고 나아지는지는
    런타임 데이터가 있어야 한다. 그 주장은 여기서 하지 않는다.
    """

    ok: bool
    reason: str | None
    before: float | None = None
    after: float | None = None
    expected_gain: float | None = None
    realized_gain: float | None = None

    @property
    def error(self) -> float | None:
        if self.expected_gain is None or self.realized_gain is None:
            return None
        return self.realized_gain - self.expected_gain

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": "verify",
            "ok": self.ok,
            "reason": self.reason,
            "before": None if self.before is None else round(self.before, 4),
            "after": None if self.after is None else round(self.after, 4),
            "expected_gain": (
                None if self.expected_gain is None else round(self.expected_gain, 4)
            ),
            "realized_gain": (
                None if self.realized_gain is None else round(self.realized_gain, 4)
            ),
            "gain_error": None if self.error is None else round(self.error, 4),
        }


def apply_plan(
    frame_poses: Sequence[Mapping[str, Vec3]],
    joint: str,
    plan: Sequence[Mapping[str, Any]],
) -> list[dict[str, Vec3]]:
    """`target.per_frame`을 월드 관절에 그대로 적용한다 — 시뮬레이터가 할 일과 같다."""
    moved = [dict(joints) for joints in frame_poses]
    for step in plan:
        index = step.get("index")
        target = step.get("target")
        if not isinstance(index, int) or not isinstance(target, Sequence) or len(target) != 3:
            continue
        if 0 <= index < len(moved):
            moved[index][joint] = (float(target[0]), float(target[1]), float(target[2]))
    return moved


def verify_correction(
    frame_poses: Sequence[Mapping[str, Vec3]],
    aligned: AlignResult,
    bucket: Mapping[str, Any],
    *,
    excluded: Sequence[str] = (),
) -> VerifyResult:
    """첫 교정을 적용해 보고 예고한 상승분과 견준다.

    교정이 없으면 잴 것이 없다(`no_correction`). 그것도 정직한 결과다 —
    "고칠 곳이 없다"는 판정은 검증 대상이 아니다.
    """
    before = score(frame_poses, aligned, bucket, excluded=excluded)
    if not before.ok or before.score is None:
        return VerifyResult(ok=False, reason=before.reason)
    if not before.corrections:
        return VerifyResult(ok=False, reason="no_correction", before=before.score)
    correction = before.corrections[0]
    plan = before.plans[0] if before.plans else ()
    if not plan:
        return VerifyResult(ok=False, reason="no_plan", before=before.score)

    moved = apply_plan(frame_poses, correction.joint, plan)
    after = score(moved, aligned, bucket, excluded=excluded)
    if not after.ok or after.score is None:
        return VerifyResult(ok=False, reason=after.reason, before=before.score)
    return VerifyResult(
        ok=True,
        reason=None,
        before=before.score,
        after=after.score,
        expected_gain=correction.expected_gain,
        realized_gain=after.score - before.score,
    )
