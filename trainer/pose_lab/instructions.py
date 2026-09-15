#!/usr/bin/env python3
"""발부된 축 교정(`measured_score.AxisCorrection`)을 사람이 읽는 문장·표시 문자열로 옮긴다.

**표시 전용이다.** 발부 로직·`magnitude_cm`·축 선택·VLM 계약 문장(`corrections_text`,
`stance_text`)은 여기서 바꾸지 않는다. 같은 구조체 하나에서 문장(`text`)과 수치(`figure`)를
함께 지으므로 둘이 다른 숫자를 말할 수 없다. 화면은 이 문자열을 다시 포맷하지 않는다 —
Python과 JS는 반올림 동률 처리가 달라 두 규칙이 섞이면 숫자가 갈린다.

점 라벨은 계측 대상과 맞춘다: 교정 화살표가 나가는 곳은 손 관절이 아니라 **건 마커**이고
머리는 **HMD**다. 조사는 괄호 앞 기본 라벨(`measured_score.POINT_LABELS`)에서 뽑는다.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Final

_SCRIPTS_DIR = str(Path(__file__).resolve().parents[1] / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

measured_score = importlib.import_module("measured_score")

DISPLAY_POINT_LABELS: Final[dict[str, str]] = {"right": "오른손(건)", "head": "머리(HMD)"}

# '벽'·'진행' 축은 스트로크 기저(착탄점 궤적의 접선과 월드 수직)에서 추정한 축이다. 실제
# 착탄면 법선과 레벨 3~10에서 중앙값 44° 어긋난다(각도 축 진단). 그 사실을 이름에 드러낸다.
AXIS_TRACE: Final[dict[str, str]] = {
    "wall": "벽(스트로크 기준) 축",
    "vertical": "수직 축",
    "lateral": "진행(스트로크 기준) 축",
}

# wall 축이 실제 착탄면 법선과 이보다 크게 어긋나면 "벽 쪽으로 붙이세요"라고 쓰지 않는다 —
# 화면에 그린 벽과 다른 방향을 가리키기 때문이다. 진단 분포(스트로크 법선 vs 메시 법선 중앙값
# 44°)를 **본 뒤** 정한 표시 규칙이지 게이트가 아니다. 채점·magnitude·축은 바뀌지 않는다.
#
# 그때도 방향은 버리지 않는다. `stroke_frame.Basis.normal`은 정의상 **수평**이고 착탄점에서 작업자
# (건) 쪽이 양수다. 그래서 wall 성분의 부호는 벽면이 어떻게 기울었든 "착탄 궤적 쪽으로 수평으로
# 다가가느냐/멀어지느냐"를 정확히 말한다(`text_mode = "stroke_relative"`). 옛 강등 문장("방향 대신
# 수치로 안내")은 행동 방향이 없어 교정 어휘로 기능하지 않았다.
SURFACE_ANGLE_AXIS_ONLY_DEG: Final = 30.0

STROKE_RELATIVE_RIGHT: Final[dict[str, str]] = {
    "붙이": "{obj} 착탄 궤적 쪽으로 수평으로 {m}cm 붙이세요.",
    "띄우": "{obj} 착탄 궤적에서 수평으로 {m}cm 띄우세요.",
}
# 머리: 부호 규칙은 `HEAD_PHRASES`와 같다(magnitude > 0 = 작업자 쪽 = 물러섬).
STROKE_RELATIVE_HEAD: Final[dict[bool, str]] = {
    True: "착탄 궤적에서 수평으로 {m}cm 더 물러서세요.",
    False: "착탄 궤적 쪽으로 수평으로 {m}cm 더 다가서세요.",
}

MINUS: Final = "−"


class NoPhrase(ValueError):
    """표에 없는 (점, 축, 방향) 조합. 문장을 지어내지 않는다."""


def figure(magnitude_cm: float) -> str:
    """부호(U+2212 또는 +) + 소수 1자리. 예: −15.4"""
    return (MINUS if magnitude_cm < 0 else "+") + f"{abs(magnitude_cm):.1f}"


def sigma_display(sigma: float) -> str:
    return f"σ{sigma:.2f}"


def trace(point: str, axis: str) -> str:
    return f"{DISPLAY_POINT_LABELS[point]} · {AXIS_TRACE[axis]}"


def _object(point: str) -> str:
    base = measured_score.POINT_LABELS[point]
    return f"{DISPLAY_POINT_LABELS[point]}{measured_score._object_particle(base)}"  # noqa: SLF001


def _right_text(axis: str, direction: str, amount: str) -> str:
    obj = _object("right")
    if axis == "wall" and direction == "붙이":
        return f"{obj} 벽 쪽으로 {amount}cm 붙이세요."
    if axis == "wall" and direction == "띄우":
        return f"{obj} 벽에서 {amount}cm 띄우세요."
    if axis == "vertical" and direction == "올리":
        return f"{obj} {amount}cm 올리세요."
    if axis == "vertical" and direction == "내리":
        return f"{obj} {amount}cm 내리세요."
    if axis == "lateral" and direction in ("앞으로", "뒤로"):
        return f"{obj} 진행 방향 {direction} {amount}cm 옮기세요."
    if axis == "lateral" and direction == "":
        # 수직 스트로크: 좌우를 판정할 기준이 없다. 방향을 지어내지 않는다.
        return f"{obj} 벽을 따라 {amount}cm 옮기세요."
    raise NoPhrase(f"no phrase for right/{axis}/{direction!r}")


# 머리 문장. 부호 규칙은 `measured_score.STANCE_PHRASES`의 (axis, magnitude > 0) 키와 같다.
HEAD_PHRASES: Final[dict[tuple[str, bool], str]] = {
    ("wall", True): "벽에서 {m}cm 더 떨어져 서세요.",
    ("wall", False): "벽 쪽으로 {m}cm 더 붙어 서세요.",
    ("vertical", True): "자세를 {m}cm 높여 작업하세요.",
    ("vertical", False): "자세를 {m}cm 낮춰 작업하세요.",
}


def axis_text(correction: object, surface_angle_deg: float | None = None) -> tuple[str, str]:
    """(text, text_mode). 표에 없는 조합이면 `NoPhrase`(ValueError)."""
    point = getattr(correction, "point")
    axis = getattr(correction, "axis")
    magnitude = float(getattr(correction, "magnitude_cm"))
    direction = str(getattr(correction, "direction"))
    if point not in DISPLAY_POINT_LABELS or axis not in AXIS_TRACE:
        raise NoPhrase(f"no phrase for {point}/{axis}")
    amount = f"{abs(magnitude):.1f}"

    if axis == "wall" and surface_angle_deg is not None and surface_angle_deg > SURFACE_ANGLE_AXIS_ONLY_DEG:
        # 표에 없는 조합이면 이 문장도 내지 않는다(방향 문장과 같은 경계).
        _directive(point, axis, magnitude, direction, amount)
        if point == "right":
            text = STROKE_RELATIVE_RIGHT[direction].format(obj=_object("right"), m=amount)
        else:
            text = STROKE_RELATIVE_HEAD[magnitude > 0].format(m=amount)
        return text, "stroke_relative"
    return _directive(point, axis, magnitude, direction, amount), "directive"


def _directive(point: str, axis: str, magnitude: float, direction: str, amount: str) -> str:
    if point == "right":
        return _right_text(axis, direction, amount)
    key = (axis, magnitude > 0)
    if key not in HEAD_PHRASES:
        raise NoPhrase(f"no phrase for head/{axis}")
    return HEAD_PHRASES[key].format(m=amount)


def surface_angle_display(surface_angle_deg: float | None) -> str | None:
    return None if surface_angle_deg is None else f"착탄면과 {surface_angle_deg:.0f}°"


def axis_display(correction: object, surface_angle_deg: float | None = None) -> dict[str, object]:
    """문장을 뺀 표시 필드. 문장이 없어도 수치와 추적 라벨은 싣는다."""
    point = getattr(correction, "point")
    axis = getattr(correction, "axis")
    # 착탄면 각은 '벽' 축의 이름이 기하와 맞는지를 말한다. 다른 축에는 싣지 않는다.
    angle = surface_angle_deg if axis == "wall" else None
    reason = None if axis != "wall" or angle is not None else "no_verified_surface"
    return {
        "figure": figure(float(getattr(correction, "magnitude_cm"))),
        "sigma_display": sigma_display(float(getattr(correction, "sigma"))),
        "trace": trace(point, axis),
        "surface_angle_deg": angle,
        "surface_angle_display": surface_angle_display(angle),
        "surface_reason": reason,
    }


def axis_instruction(correction: object, surface_angle_deg: float | None = None) -> dict[str, object]:
    """{text, text_mode, figure, sigma_display, trace}. 표에 없는 조합이면 ValueError."""
    text, mode = axis_text(correction, surface_angle_deg)
    shown = axis_display(correction, surface_angle_deg)
    return {
        "text": text,
        "text_mode": mode,
        "figure": shown["figure"],
        "sigma_display": shown["sigma_display"],
        "trace": shown["trace"],
    }


IMPROVEMENT_LABELS: Final[dict[str, str]] = {
    "same": "③ 교정을 그대로 따랐을 때(같은 기준 패스로 잰 값)",
    "split": "A 반쪽 산포로 다시 고른 교정으로 B 반쪽에서 잰 값 — ③ 목록과 발부 축이 다를 수 있음",
}

DASH: Final = "—"


def score_display(value: float | None) -> str:
    return DASH if value is None else f"{value:.3f}"


def deviation_display(cm: float) -> str:
    return f"{cm:.1f}"
