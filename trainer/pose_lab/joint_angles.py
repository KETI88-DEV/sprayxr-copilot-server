#!/usr/bin/env python3
"""관절 각도 **추정표**(표시 전용). 점수·교정에 들어가지 않는다(D4).

관절에는 추적 장치가 없다. 실측 강체는 머리(HMD)·건·플래쉬 셋뿐이고 나머지 20관절은
시뮬레이터 IK가 세 강체에서 푼 값이다. 그래서 여기 값은 **추정**이라고 부른다 — 측정이 아니다.
채점(`measured_score`)·교정(`measured_pose`)·동작 키(`motion_key`)는 이 모듈을 부르지 않는다.

각도는 CSV 자기 리그의 FK(`pose_fk_csv.forward_kinematics`) 관절 위치에서 세 점 사이 각으로
잰다. 세 점 규약 하나만 쓴다(`AngleSpec.points` = (a, b, c) → b에서의 내각):

- 팔꿈치 = ∠(상완, 전완, 손) — 180°가 편 팔이다.
- 무릎 = ∠(허벅지, 종아리, 발) — 180°가 편 다리다.
- 어깨 벌림 = ∠(골반, 상완, 전완) — 0°면 팔이 골반 쪽(몸통을 따라 아래)을, 180°면 그 반대를 향한다.

**품질을 숨기지 않는다.** 값 옆에 그 관절을 움직이는 로컬 회전 컬럼의 성질을 같이 낸다:

- `zero_pct` — 그 회전이 `ZERO_DEG`(1°) 미만인 분사 프레임 비율. 오른 전완 회전은 표본 40세션
  287,712프레임의 87.6%가 여기 해당하고(모듈 `orientation` 독스트링), 그 프레임에서 오른
  팔꿈치는 정확히 180°(편 팔)로 나온다. 비율이 100%면 그 각은 그 패스에서 **상수**다.
- `jump_pct` — 이웃한 분사 프레임 사이 그 회전의 측지각이 `JUMP_DEG`(30°)를 넘은 비율.
  상완 회전이 프레임 사이 급변한다는 사실이 여기서 보인다.
- `spread_deg` — 각 자체의 표준편차. 0°면 화면의 한 값이 패스 전체를 대표한다는 뜻이다.

프레임은 분사 중이고 착탄점이 있는 프레임만 쓴다(`orientation`·`composite`와 같은 구간).
`jump_pct`는 이웃 프레임 사이 값이라 **연속 구간**이어야 한다 — 그래서 균등 추출로 솎지 않고
가운데를 중심으로 한 연속 창 `MAX_FRAMES`개를 쓴다. 몇 개를 봤는지는 `samples`로 낸다.
"""

from __future__ import annotations

import importlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping, Sequence

import numpy as np

_SCRIPTS_DIR = str(Path(__file__).resolve().parents[1] / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

measured_pose = importlib.import_module("measured_pose")
pose_fk_csv = importlib.import_module("pose_fk_csv")
rotation_math = importlib.import_module("rotation_math")

from trainer.pose_lab import skeleton  # noqa: E402

# 이웃 프레임 사이 값을 재므로 연속 구간이어야 한다. 실제 패스는 수십~수백 프레임이라 거의 걸리지 않는다.
MAX_FRAMES: Final = 600
# 회전이 "없다"고 볼 각. `orientation` 독스트링이 인용한 사전 측정과 같은 문턱이다(1° 미만).
ZERO_DEG: Final = 1.0
# 이웃 프레임 사이 "급변"으로 볼 각.
JUMP_DEG: Final = 30.0

TITLE: Final = "관절 각도 — 시뮬레이터 IK 관절로 계산한 추정값"
NOTE: Final = (
    "관절에는 추적 장치가 없습니다 — 아래 값은 세 실측 강체에서 시뮬레이터 IK가 푼 관절로 계산한 "
    "추정이고, 점수·교정에 들어가지 않습니다."
)
QUALITY_NOTE: Final = (
    "괄호 안은 그 관절을 움직이는 회전 컬럼의 성질입니다 — 회전이 0°인 프레임 비율, 이웃 프레임 사이 "
    f"{JUMP_DEG:.0f}°를 넘게 튄 비율, 그리고 각 자체의 표준편차. 0° 비율이 100%면 그 각은 이 패스에서 상수입니다."
)


@dataclass(frozen=True, slots=True)
class AngleSpec:
    key: str
    joint: str
    label: str
    points: tuple[str, str, str]  # (a, b, c) → b에서의 내각
    driver_column: str  # 이 각을 움직이는 CSV 로컬 회전 컬럼 그룹
    driver_label: str


ANGLE_SPECS: Final[tuple[AngleSpec, ...]] = (
    AngleSpec(
        key="right_elbow",
        joint="오른 팔꿈치",
        label="상완–전완–손 사이 각(180°가 편 팔)",
        points=("RightUpperArm", "RightForeArm", "RightHand"),
        driver_column="Right Fore Arm",
        driver_label="오른 전완 회전",
    ),
    AngleSpec(
        key="left_elbow",
        joint="왼 팔꿈치",
        label="상완–전완–손 사이 각(180°가 편 팔)",
        points=("LeftUpperArm", "LeftForeArm", "LeftHand"),
        driver_column="Left Fore Arm",
        driver_label="왼 전완 회전",
    ),
    AngleSpec(
        key="right_shoulder",
        joint="오른 어깨",
        label="골반–상완–전완 사이 각(0°가 몸통을 따라 아래)",
        points=("Pelvis", "RightUpperArm", "RightForeArm"),
        driver_column="Right Upper Arm",
        driver_label="오른 상완 회전",
    ),
    AngleSpec(
        key="left_shoulder",
        joint="왼 어깨",
        label="골반–상완–전완 사이 각(0°가 몸통을 따라 아래)",
        points=("Pelvis", "LeftUpperArm", "LeftForeArm"),
        driver_column="Left Upper Arm",
        driver_label="왼 상완 회전",
    ),
    AngleSpec(
        key="right_knee",
        joint="오른 무릎",
        label="허벅지–종아리–발 사이 각(180°가 편 다리)",
        points=("RightThigh", "RightCalf", "RightFoot"),
        driver_column="Right Calf",
        driver_label="오른 종아리 회전",
    ),
    AngleSpec(
        key="left_knee",
        joint="왼 무릎",
        label="허벅지–종아리–발 사이 각(180°가 편 다리)",
        points=("LeftThigh", "LeftCalf", "LeftFoot"),
        driver_column="Left Calf",
        driver_label="왼 종아리 회전",
    ),
)

REASON_TEXT: Final[dict[str, str]] = {
    "no_spraying_frame": "분사 중 착탄 프레임이 없어 계산하지 않습니다.",
    "no_joint_columns": "이 패스에는 관절 컬럼이 없어 계산하지 않습니다.",
}


@dataclass(frozen=True, slots=True)
class AngleStats:
    """한 관절·한 패스의 추정 각과 그 각이 얼마나 상수인지."""

    mean_deg: float
    spread_deg: float
    zero_pct: float
    jump_pct: float
    samples: int


def spraying_window(rows: Sequence[Mapping[str, str]]) -> list[Mapping[str, str]]:
    """분사 중 착탄 프레임의 **연속** 창(가운데 중심, 최대 `MAX_FRAMES`개)의 원본 행."""
    gun = measured_pose.active_gun(rows)
    if gun is None:
        return []
    picked = [
        row
        for frame, row in measured_pose.measured_frames_with_rows(rows)
        if frame.spraying and frame.hit is not None
    ]
    if len(picked) <= MAX_FRAMES:
        return picked
    start = (len(picked) - MAX_FRAMES) // 2
    return picked[start : start + MAX_FRAMES]


def _angle_at(points: Mapping[str, Sequence[float]], spec: AngleSpec) -> float | None:
    a, b, c = spec.points
    if a not in points or b not in points or c not in points:
        return None
    pb = np.asarray(points[b], dtype=float)
    u = np.asarray(points[a], dtype=float) - pb
    v = np.asarray(points[c], dtype=float) - pb
    angle = skeleton._angle_between_deg(u, v)  # noqa: SLF001 — 같은 패키지의 각도 계산 하나만 쓴다.
    return None if not math.isfinite(angle) else float(angle)


def _driver_matrices(rows: Sequence[Mapping[str, str]], column: str) -> list[np.ndarray | None]:
    out: list[np.ndarray | None] = []
    for row in rows:
        try:
            euler = tuple(float(row[f"{column} Rotation {axis}"]) for axis in "XYZ")
        except (KeyError, ValueError):
            out.append(None)
            continue
        out.append(np.asarray(pose_fk_csv.euler_matrix(euler), dtype=float))
    return out


def _identity_angle_deg(matrix: np.ndarray) -> float:
    return skeleton.rotation_angle_deg(np.eye(3), matrix)


def pass_angles(rows: Sequence[Mapping[str, str]]) -> dict[str, AngleStats | None]:
    """키 → 추정 각 통계. 프레임이 없거나 관절이 없으면 그 키는 None."""
    window = spraying_window(rows)
    if not window:
        return {spec.key: None for spec in ANGLE_SPECS}
    offsets = pose_fk_csv.bind_offsets(dict(window[0]))
    if not offsets:
        return {spec.key: None for spec in ANGLE_SPECS}
    frames = [pose_fk_csv.forward_kinematics(dict(row), offsets) for row in window]

    out: dict[str, AngleStats | None] = {}
    for spec in ANGLE_SPECS:
        values = [angle for angle in (_angle_at(points, spec) for points in frames) if angle is not None]
        matrices = _driver_matrices(window, spec.driver_column)
        known = [m for m in matrices if m is not None]
        if not values or not known:
            out[spec.key] = None
            continue
        zero = sum(1 for m in known if _identity_angle_deg(m) < ZERO_DEG)
        pairs = [
            (a, b)
            for a, b in zip(matrices, matrices[1:])
            if a is not None and b is not None
        ]
        jumps = sum(1 for a, b in pairs if skeleton.rotation_angle_deg(a, b) > JUMP_DEG)
        out[spec.key] = AngleStats(
            mean_deg=float(np.mean(values)),
            spread_deg=float(np.std(values)),
            zero_pct=100.0 * zero / len(known),
            jump_pct=100.0 * jumps / len(pairs) if pairs else 0.0,
            samples=len(values),
        )
    return out


def _deg_display(stats: AngleStats | None) -> str:
    return "—" if stats is None else f"{stats.mean_deg:.0f}°"


def quality_display(spec: AngleSpec, stats: AngleStats | None) -> str:
    """값 옆 괄호. 상수인 관절을 상수인 줄 모르게 보여주지 않는 것이 이 문자열의 일이다."""
    if stats is None:
        return "—"
    return (
        f"{spec.driver_label} 0° {stats.zero_pct:.0f}% · "
        f"{JUMP_DEG:.0f}° 초과 변화 {stats.jump_pct:.0f}% · 표준편차 {stats.spread_deg:.0f}°"
    )


def joint_angles_block(
    query_rows: Sequence[Mapping[str, str]], reference_rows: Sequence[Mapping[str, str]]
) -> dict[str, object]:
    query = pass_angles(query_rows)
    reference = pass_angles(reference_rows)
    rows: list[dict[str, object]] = []
    for spec in ANGLE_SPECS:
        q = query[spec.key]
        r = reference[spec.key]
        reason: str | None = None
        if q is None or r is None:
            reason = "no_spraying_frame" if not spraying_window(query_rows) else "no_joint_columns"
        rows.append(
            {
                "key": spec.key,
                "joint": spec.joint,
                "label": spec.label,
                "query": None if q is None else q.mean_deg,
                "reference": None if r is None else r.mean_deg,
                "display": {"query": _deg_display(q), "reference": _deg_display(r)},
                "quality": {
                    "query": None if q is None else _quality_dict(q),
                    "reference": None if r is None else _quality_dict(r),
                },
                "quality_display": {
                    "query": quality_display(spec, q),
                    "reference": quality_display(spec, r),
                },
                "samples": {
                    "query": 0 if q is None else q.samples,
                    "reference": 0 if r is None else r.samples,
                },
                "reason": reason,
                "reason_text": None if reason is None else REASON_TEXT[reason],
            }
        )
    return {"title": TITLE, "note": NOTE, "quality_note": QUALITY_NOTE, "rows": rows}


def _quality_dict(stats: AngleStats) -> dict[str, float]:
    return {
        "zero_pct": stats.zero_pct,
        "jump_pct": stats.jump_pct,
        "spread_deg": stats.spread_deg,
        "zero_deg": ZERO_DEG,
        "jump_deg": JUMP_DEG,
    }
