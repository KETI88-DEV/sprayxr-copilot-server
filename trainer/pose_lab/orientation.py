#!/usr/bin/env python3
"""건 각도 참고표(표시 전용). 점수·교정에 들어가지 않는다.

각도 축은 사전등록 측정(커버리지 Q1, 전이 Q3, 채점 혼합 Q4)을 하나도 통과하지 못했다. 그래서
교정으로 발부하지 않고 채점에 섞지 않는다. 여기서는 실측으로 검증된 두 값만 참고로 보인다.

- 건 입사각: 건 분사축(`measured_pose.GUN_FORWARD`, 규약 YXZ+ — (착탄점−건)과 64,098패스 중앙값
  3.89°)과 착탄면 법선 사이 각 `arccos|s·n|`. 법선은 **검증된 정렬로 배치한 메시**에서 3D
  최근접 삼각형의 것이다. 메시가 없으면(레벨 1·2 등) 계산하지 않는다 — 스트로크 기저 법선으로
  대체하지 않는다(실제 법선과 중앙값 44° 어긋난다). `|·|`이라 반사 정렬(det<0)에서도 같다.
- 건 앙각: `asin(vertical_of(s))`.

HMD 전방축은 수평 방위로만 검증돼(착탄 방위 순환 상관 0.785) 시선 앙각·시선-분사 각은 싣지 않는다.
관절 각도는 싣지 않는다 — CSV 전완 회전은 늘 0°이고 상완 회전은 프레임 사이 급변한다.
"""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path
from typing import Final, Mapping, Sequence

import numpy as np

_SCRIPTS_DIR = str(Path(__file__).resolve().parents[1] / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

measured_pose = importlib.import_module("measured_pose")
world_axes = importlib.import_module("world_axes")

from trainer.pose_lab import composite  # noqa: E402

MAX_FRAMES: Final = 200

TITLE: Final = "건 각도 — 건 실측 회전으로 계산한 참고값"
HEADER_NOTE: Final = (
    "점수·교정에 들어가지 않는 참고값입니다. 이 값을 기준에 맞췄을 때의 개선 효과는 검증되지 않았습니다."
)
WEIGHT_NOTE: Final = "유사도는 머리·오른손·왼손 위치를 핵심 지표로 가중합니다."
JOINT_ANGLES_NOTE: Final = (
    "관절 각도는 표시하지 않습니다 — CSV의 전완 회전은 늘 0°이고 상완 회전은 프레임 사이 급변해 "
    "팔꿈치·어깨 각이 측정돼 있지 않습니다. 스켈레톤은 이 회전으로 그린 아바타 추정입니다."
)
EXCLUDED_NOTE: Final = "HMD 전방축은 수평 방위로만 검증돼 시선 앙각·시선-분사 각은 계산하지 않습니다."

LABELS: Final[dict[str, str]] = {
    "gun_incidence_deg": "건 입사각(착탄면 법선 기준)",
    "gun_pitch_deg": "건 앙각(위가 +)",
}
REASON_TEXT: Final[dict[str, str]] = {
    "no_verified_surface": "이 레벨은 검증된 벽면 정렬이 없어 입사각을 계산하지 않습니다.",
    "no_rotation": "건 회전 컬럼을 읽지 못해 계산하지 않습니다.",
    "no_spraying_frame": "분사 중 착탄 프레임이 없어 계산하지 않습니다.",
}


def _spraying(rows: Sequence[Mapping[str, str]]) -> tuple[list[tuple[object, Mapping[str, str]]], str | None]:
    gun = measured_pose.active_gun(rows)
    if gun is None:
        return [], None
    paired = [
        (frame, row)
        for frame, row in measured_pose.measured_frames_with_rows(rows)
        if frame.spraying and frame.hit is not None
    ]
    if len(paired) > MAX_FRAMES:
        picks = np.linspace(0, len(paired) - 1, MAX_FRAMES).astype(int)
        paired = [paired[i] for i in picks]
    return paired, gun


def pass_angles(
    rows: Sequence[Mapping[str, str]], geometry: composite.MeshGeometry | None
) -> dict[str, tuple[float | None, int, str | None]]:
    """키 → (패스 평균 각, 표본 수, 사유)."""
    paired, gun = _spraying(rows)
    if not paired or gun is None:
        return {key: (None, 0, "no_spraying_frame") for key in LABELS}
    directions: list[np.ndarray] = []
    hits: list[Sequence[float]] = []
    guns: list[Sequence[float]] = []
    for frame, row in paired:
        s = composite.spray_direction(row, gun)
        if s is None:
            continue
        directions.append(s / np.linalg.norm(s))
        hits.append(frame.hit)  # type: ignore[attr-defined]
        guns.append(frame.right)  # type: ignore[attr-defined]
    if not directions:
        return {key: (None, 0, "no_rotation") for key in LABELS}

    pitches = [math.degrees(math.asin(max(-1.0, min(1.0, world_axes.vertical_of(tuple(s)))))) for s in directions]
    out: dict[str, tuple[float | None, int, str | None]] = {
        "gun_pitch_deg": (float(np.mean(pitches)), len(pitches), None)
    }
    if geometry is None:
        out["gun_incidence_deg"] = (None, 0, "no_verified_surface")
    else:
        _d, rows_idx, _c = geometry.nearest(np.asarray(hits, dtype=float))
        normals = geometry.normals[rows_idx]
        cosines = np.abs(np.sum(np.stack(directions) * normals, axis=1))
        angles = np.degrees(np.arccos(np.clip(cosines, 0.0, 1.0)))
        out["gun_incidence_deg"] = (float(np.mean(angles)), int(len(angles)), None)
    return out


def _deg_display(value: float | None) -> str:
    return "—" if value is None else f"{value:.0f}°"


def orientation_block(
    query_rows: Sequence[Mapping[str, str]],
    reference_rows: Sequence[Mapping[str, str]],
    geometry: composite.MeshGeometry | None,
) -> dict[str, object]:
    query = pass_angles(query_rows, geometry)
    reference = pass_angles(reference_rows, geometry)
    items = []
    for key in ("gun_incidence_deg", "gun_pitch_deg"):
        q_value, q_n, q_reason = query[key]
        r_value, r_n, r_reason = reference[key]
        reason = q_reason or r_reason
        items.append(
            {
                "key": key,
                "label": LABELS[key],
                "query": q_value,
                "reference": r_value,
                "samples": {"query": q_n, "reference": r_n},
                "display": {"query": _deg_display(q_value), "reference": _deg_display(r_value)},
                "reason": reason,
                "reason_text": None if reason is None else REASON_TEXT[reason],
            }
        )
    return {
        "items": items,
        "title": TITLE,
        "header_note": HEADER_NOTE,
        "weight_note": WEIGHT_NOTE,
        "joint_angles_note": JOINT_ANGLES_NOTE,
        "excluded_note": EXCLUDED_NOTE,
    }
