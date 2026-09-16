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
대신 **수평 방위만** 쓰는 `hmd_heading_to_hit_deg`(HMD 전방축의 수평 방위와 머리→착탄 수평 방위 사이
각)를 관절 축 평가표(`joint_axes_block`)의 머리 행에 싣는다.

관절 각도는 **추정**으로 싣는다(`joint_angles.py`, D4) — 점수·교정에는 넣지 않는다. 관절에는 추적
장치가 없고(시뮬레이터 IK 산출물), 오른 전완 회전은 표본 40세션 287,712프레임의 87.6%에서 1° 미만
(편 팔)이며, 상완 회전은 프레임 사이 30° 넘게 급변한다(1.7%).
왼 전완 회전은 0°가 아니다(중앙값 114°) — 옛 문구 "전완 회전은 늘 0°"는 오른쪽에만 맞았다
(스크래치패드 r3_forearm.py, seed 20260915).
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
    "관절 각도(팔꿈치·어깨·무릎)는 아래 관절 각도 표에 추정으로 싣습니다 — 관절에는 추적 장치가 없는 시뮬레이터 "
    "IK 값이라 점수·교정에 들어가지 않고, 각마다 그 회전이 0°인 프레임 비율과 프레임 사이 급변 비율을 같이 적습니다."
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


MINUS: Final = "\u2212"


def _deg_display(value: float | None) -> str:
    """정수 도. 음수 부호는 교정 수치(`instructions.figure`)와 같은 U+2212다 — 한 화면에 두 마이너스를 섞지 않는다."""
    if value is None:
        return "—"
    text = f"{value:.0f}"
    if text.startswith("-"):
        text = text[1:] if text == "-0" else MINUS + text[1:]
    return f"{text}°"


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


# --- 관절 축 평가표(사용자 요청 5번) --------------------------------------------------------
#
# "관절 축 평가"를 관절마다 **무엇을 잴 수 있고 무엇을 못 재는지**로 답한다. 잴 수 있는 것은 실측 강체
# 두 개(HMD·건)의 축뿐이다. 나머지 관절은 측정값이 아니라서 값 대신 사유를 싣는다 — 값을 지어내지 않는다.
JOINT_AXES_TITLE: Final = "관절 축 평가 — 관절마다 잴 수 있는 것"
JOINT_AXES_NOTE: Final = (
    "실측 강체(HMD·건)가 붙은 축만 값을 냅니다. 점수·교정에 들어가지 않는 참고값입니다."
)
STATUS_TEXT: Final[dict[str, str]] = {
    "measured": "실측",
    "measured_above": "실측 · 위 표",
    "not_evaluated": "평가 안 함",
    "not_measured": "측정 없음",
}
HMD_HEADING_LABEL: Final = "HMD 시선 수평 방위와 착탄 방위 사이 각"
JOINT_AXIS_ROWS: Final[tuple[dict[str, str], ...]] = (
    {"key": "head_hmd_heading", "joint": "머리(HMD)", "status": "measured"},
    {
        "key": "right_gun_axis",
        "joint": "오른손(건)",
        "status": "measured_above",
        "reason_text": "건 분사축의 입사각·앙각을 위 표에 싣습니다.",
    },
    {
        "key": "left_flash_axis",
        "joint": "왼손(플래쉬)",
        "status": "not_evaluated",
        "reason_text": "플래쉬 회전은 기록되지만 전방축을 착탄 같은 독립 근거로 검증하지 못했습니다.",
    },
    {
        "key": "elbow_shoulder",
        "joint": "팔꿈치·어깨",
        "status": "not_measured",
        "reason_text": "추적 장치가 없는 IK 관절입니다 — 축을 실측으로 검증할 수 없습니다. 각도 추정값은 관절 각도 표에 있습니다(오른 전완 회전은 프레임의 88%가 0°, 곧 편 팔입니다).",
    },
    {
        "key": "torso_pelvis",
        "joint": "몸통·골반",
        "status": "not_measured",
        "reason_text": "추적 장치가 없고, 골반 Position 컬럼은 월드 좌표가 아닙니다.",
    },
    {
        "key": "knee_ankle",
        "joint": "무릎·발목",
        "status": "not_measured",
        "reason_text": "추적 장치가 없고, IK 다리가 해부학 범위 안인 분사 프레임은 37%뿐입니다. 무릎 각 추정값은 관절 각도 표에 있습니다.",
    },
)


def hmd_heading_to_hit_deg(rows: Sequence[Mapping[str, str]]) -> tuple[float | None, int, str | None]:
    """분사 중 착탄 프레임마다 HMD 전방축의 **수평** 방위와 머리→착탄 **수평** 방위 사이 각(부호 없음)의 평균."""
    paired, gun = _spraying(rows)
    if not paired or gun is None:
        return None, 0, "no_spraying_frame"
    angles: list[float] = []
    for frame, row in paired:
        try:
            rotation = tuple(float(row[f"{measured_pose.HEAD_GROUP} Rotation {axis}"]) for axis in "XYZ")
        except (KeyError, ValueError):
            continue
        forward_yaw = measured_pose.facing_yaw(rotation)
        head = frame.head  # type: ignore[attr-defined]
        hit = frame.hit  # type: ignore[attr-defined]
        target_yaw = world_axes.yaw_of((hit[0] - head[0], hit[1] - head[1], hit[2] - head[2]))
        if forward_yaw is None or target_yaw is None:
            continue
        delta = abs((forward_yaw - target_yaw + math.pi) % (2.0 * math.pi) - math.pi)
        angles.append(math.degrees(delta))
    if not angles:
        return None, 0, "no_rotation"
    return float(np.mean(angles)), len(angles), None


# 첫 화면 요약(사용자 요청 "관절 축 평가도 필요해 보임"). 전체 표는 아래에 그대로 두고, 여기서는
# **무엇을 잴 수 있고 무엇을 못 재는지**만 한눈에 준다. 값·사유 문자열은 전부 아래 표와 같은 것을 쓴다 —
# 화면은 다시 계산하지 않는다.
JOINT_AXES_SUMMARY_TITLE: Final = "관절 축 평가"
MEASURED_STATUSES: Final[frozenset[str]] = frozenset({"measured", "measured_above"})
# 첫 화면 요약은 세 줄을 넘지 않는다 — 제목 · 상태 수 · 실측 축이 있는 관절. 못 재는 관절의 이름과
# 사유는 바로 아래 전체 표가 그대로 싣는다.
SUMMARY_COUNT_ORDER: Final[tuple[str, ...]] = ("measured", "not_evaluated", "not_measured")
SUMMARY_COUNT_LABEL: Final[dict[str, str]] = {
    "measured": "실측",
    "not_evaluated": "평가 안 함",
    "not_measured": "측정 없음",
}


def _joint_axes_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    counts = {key: 0 for key in SUMMARY_COUNT_ORDER}
    for row in rows:
        status = str(row["status"])
        counts["measured" if status in MEASURED_STATUSES else status] += 1
    counts_text = " · ".join(f"{SUMMARY_COUNT_LABEL[key]} {counts[key]}" for key in SUMMARY_COUNT_ORDER)
    chips: list[dict[str, object]] = []
    for row in rows:
        if row["status"] not in MEASURED_STATUSES:
            continue
        display = row.get("display")
        value = None
        if isinstance(display, Mapping) and not row.get("reason"):
            value = f"대상 {display['query']} · 기준 {display['reference']}"
        chips.append(
            {
                "key": row["key"],
                "joint": row["joint"],
                "status": row["status"],
                "status_text": row["status_text"],
                "value": value,
                "reason_text": row.get("reason_text") if row.get("reason") else None,
            }
        )
    return {
        "title": JOINT_AXES_SUMMARY_TITLE,
        "counts": counts,
        "counts_text": f"관절 {len(rows)}곳 — {counts_text}",
        "chips": chips,
    }


def joint_axes_block(
    query_rows: Sequence[Mapping[str, str]], reference_rows: Sequence[Mapping[str, str]]
) -> dict[str, object]:
    q_value, q_n, q_reason = hmd_heading_to_hit_deg(query_rows)
    r_value, r_n, r_reason = hmd_heading_to_hit_deg(reference_rows)
    rows: list[dict[str, object]] = []
    for spec in JOINT_AXIS_ROWS:
        row: dict[str, object] = {
            "key": spec["key"],
            "joint": spec["joint"],
            "status": spec["status"],
            "status_text": STATUS_TEXT[spec["status"]],
            "label": None,
            "query": None,
            "reference": None,
            "samples": None,
            "display": None,
            "reason": None,
            "reason_text": spec.get("reason_text"),
        }
        if spec["key"] == "head_hmd_heading":
            reason = q_reason or r_reason
            row.update(
                {
                    "label": HMD_HEADING_LABEL,
                    "query": q_value,
                    "reference": r_value,
                    "samples": {"query": q_n, "reference": r_n},
                    "display": {"query": _deg_display(q_value), "reference": _deg_display(r_value)},
                    "reason": reason,
                    "reason_text": None if reason is None else REASON_TEXT[reason],
                }
            )
        rows.append(row)
    return {
        "title": JOINT_AXES_TITLE,
        "note": JOINT_AXES_NOTE,
        "rows": rows,
        "summary": _joint_axes_summary(rows),
    }
