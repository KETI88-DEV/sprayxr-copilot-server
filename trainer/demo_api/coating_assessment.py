"""v1 흐름이 PR #8 도막 기준(`trainer/zone_engine`)으로 판정하게 하는 얇은 어댑터.

수치 규칙(복호 식, 10 µm 1회 차감, 1600 µm 상한, 1 µm 미만 미도장, 등급 경계, 6구간,
칠한 곳 평균)은 PR #8 함수를 그대로 부르므로 같다. 다른 것은 입력을 다루는 방식이다.

PR #8 엔진과 다른 점:

1. 크롭 밖 화소는 "미도장(관측됨)"이고 실시간에도 등급을 낸다. PR #8은 크롭 밖을 "모름"으로
   두고 일부만 관측된 Zone의 등급을 보류한다(`zone_engine/numerical.py` place_crop·assess_zone,
   `docs/corrected-zone/ENGINE_CONTRACT_KR.md`).
2. 목표가 없으면 160 µm로 판정하고 `default_target`으로 표시한다. PR #8은 목표를 채우지 않고
   등급을 보류한다.
3. 높이맵 한 장을 "세션 끝 누적"으로 읽는다. 회차 합산이 없다. PR #8은 현재 회차분이면 앞 회차와
   합산하고 앞 회차가 없으면 보류한다(`zone_engine/engine.py`).
4. 크롭 안에서 B=0인 화소는 앞선 값을 지우지 않는다(`coating_progress.py`). PR #8의 누적 크롭은
   크롭 사각형 전체를 교체한다(`numerical.overwrite_cumulative_crops`).
5. 전체 면 PNG의 encoding을 파일에서 추정한다(B가 전부 0이면 옛 형식). PR #8은 보내는 쪽이
   선언하고 섞어 추론하지 않는다.
6. 실시간 크롭은 요청의 `scale_um`으로 복호해 아틀라스에 최근접으로 놓고, complete는 표면 크기가
   프로필과 다르면 최근접으로 맞춘다. PR #8은 눈금 20000 고정이고 크기가 다르면 거부한다.
   실시간 크롭 복호(`coating_progress._decode_crop`)는 PIL이 여는 그림을 모두 RGB로 바꿔 받는다.
   PR #8 복호는 8비트 RGB·RGBA PNG만 받는다(complete 높이맵은 PR #8 복호를 그대로 쓴다).
7. `gate_status`, `coverage_percent`, `overpaint_percent`, `completion_percent`는 PR #8 출력이
   아니다. PR #8의 6구간 개수에서 이 어댑터가 만든 값이다. PR #8에는 면적 게이트가 없다.
   `gate_status="complete"`는 면적 95% 충족에 더해 PR #8 등급이 `below`가 아니어야 한다.
8. 조절 후보의 표본은 시퀀스 평균 속도·거리다. PR #8은 그 Zone에 착탄한 분사 중 프레임의
   속도·거리를 쓴다.
9. 세션 집계의 미도장률·6구간을 항상 낸다. PR #8은 최신 Zone 평가가 하나라도 `available`이
   아니면 보류한다.
10. 판정 표면이 좌하단 원점이라 Zone 화소를 더하는 순서가 PR #8(좌상단 원점)과 다르다. 칠한 곳
    평균이 마지막 자리(1e-14 µm 수준, 실시간 표면은 float32라 1e-7 수준)에서 다를 수 있고, 평균이
    목표와 정확히 같은 동률에서는 등급이 갈릴 수 있다.
11. 측정값 입구(`assess_readings`)는 게이지 측정값처럼 이미 µm인 물리량을 받으므로 10 µm를 빼지
    않고 판정한다. 같은 이유로 `v+10-10`이 부동소수로 `v`와 1 ulp 다를 수 있어, 평균이 목표와
    정확히 같은 동률에서만 등급이 갈릴 수 있다.
"""

from __future__ import annotations

import base64
import hashlib
import io
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final

import numpy as np
from PIL import Image

from trainer.zone_engine import behavior, numerical
from trainer.zone_engine.contracts import HeightPng
from trainer.zone_engine.engine import CRITERION_VERSION
from trainer.zone_engine.errors import EngineError

# PR #8 엔진은 목표를 기본값으로 채우지 않는다. v1 흐름만 160 µm로 판정하고 그 사실을 표시한다(D3).
DEFAULT_TARGET_UM: Final = 160.0
GATE_COVERAGE_PERCENT: Final = 95.0
# 호출 지점이 zone_engine을 직접 읽지 않게 다시 내보낸다.
SIX_BAND_KEYS: Final = numerical.SIX_BAND_KEYS
STATUS_BY_GRADE: Final = {"below": "under_app", "normal": "normal", "above": "over_app"}
# 이 흐름은 PR #8 엔진이 보류할 입력에도 등급을 낸다. 그래서 PR #8 문자열을 그대로 쓰지 않고 접두를 붙인다.
ASSESSMENT_VERSION: Final = f"coating-v1-adapter/{CRITERION_VERSION}"

# 표시용 사본. 판정·계산에는 쓰지 않는다(판정은 numerical이 한다).
# numerical은 이 수를 상수로 내보내지 않는다(grade_for_mean, six_band_counts 안에 있다).
# PR #8 경계가 바뀌면 test_display_edges_match_the_engine이 깨진다.
DISPLAY_BAND_FACTORS: Final = (0.9, 1.0, 1.5, 2.0, 2.5)  # 6구간 경계 ÷ 목표. [1]~[2]가 정상 구간

BAND_LABELS: Final[tuple[str, ...]] = (
    f"{DISPLAY_BAND_FACTORS[0] * 100:g}% 미만",
    *(f"{low * 100:g}~{high * 100:g}%" for low, high in zip(DISPLAY_BAND_FACTORS, DISPLAY_BAND_FACTORS[1:])),
    f"{DISPLAY_BAND_FACTORS[-1] * 100:g}% 이상",
)
OVER_PHRASE: Final[str] = f"목표의 {DISPLAY_BAND_FACTORS[2]:g}배 이상"
RULE_NOTE: Final[str] = f"등급은 칠한 곳의 평균 두께로 정합니다(목표 미만은 저도막, {OVER_PHRASE}은 과도막)."
GRADE_SHORT_LABELS: Final[dict[str, str]] = {
    "below": "목표 미만", "normal": "정상", "above": f"{DISPLAY_BAND_FACTORS[2]:g}배 이상", "none": "미도장"}

GRADE_WORDS: Final = {"below": "저도막", "normal": "정상", "above": "과도막"}
ADJUSTMENT_WORDS: Final = {
    ("speed_mm_s", "decrease"): "이동 속도를 낮추는 쪽",
    ("speed_mm_s", "increase"): "이동 속도를 높이는 쪽",
    ("distance_mm", "decrease"): "이격거리를 줄이는 쪽",
    ("distance_mm", "increase"): "이격거리를 늘리는 쪽",
}
_BEHAVIOR_FIELDS: Final = (("speed_mm_s", "avg_speed_mms"), ("distance_mm", "avg_distance_mm"))


def resolve_target(target_um: float | None) -> tuple[float, str]:
    """None → (160.0, "default_target"), 값 → (float(값), "session_target")."""
    if target_um is None:
        return DEFAULT_TARGET_UM, "default_target"
    return float(target_um), "session_target"


def session_target(session: Any) -> float | None:
    """세션 행 → 판정에 넘길 목표. 세션이 목표를 명시하지 않았으면 None."""
    # target_basis가 없는 행(예전 세션, 시험 스텁)은 저장된 목표를 세션 목표로 본다.
    if session is None or getattr(session, "target_basis", None) == "default_target":
        return None
    return getattr(session, "target_thickness_um", None)


def band_edges_um(target_um: float) -> list[float]:
    """6구간 경계(µm) — 표시용."""
    return [factor * target_um for factor in DISPLAY_BAND_FACTORS]


def normal_band_um(target_um: float) -> list[float]:
    """[목표, 과도막 경계] — 표시용."""
    return band_edges_um(target_um)[1:3]


def painted_surface_from_png(raw: bytes) -> np.ndarray:
    """전체 면 높이맵 PNG → float64 [H, W], 좌상단 원점. 칠한 자리의 원시 두께(µm), 그 밖은 0."""
    try:
        with Image.open(io.BytesIO(raw)) as image:
            width, height = image.size
            # 옛 형식 파일(B가 전부 0)을 전부 미도장으로 만들지 않으려고 파일에서 추정한다. PR #8은 보내는 쪽이 선언한다.
            encoding = "rg16_b8" if image.getchannel("B").getbbox() else "legacy_rg_positive"
        spec = HeightPng.model_validate({
            "data_base64": base64.b64encode(raw).decode("ascii"),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "encoding": encoding,
            # 세션 끝 누적 한 장으로 읽는다. PR #8의 회차 합산은 하지 않는다(S3).
            "basis": "current_round",
            "width": width, "height": height, "origin": "top_left",
            "crop": {"x0": 0, "y0": 0, "width": width, "height": height,
                     "atlas_width": width, "atlas_height": height},
        })
        decoded = numerical.decode_height_png(spec)
    except (EngineError, OSError) as exc:  # pydantic.ValidationError는 이미 ValueError다
        raise ValueError(f"height PNG cannot be decoded: {exc}") from exc
    return np.where(decoded.source_painted, decoded.raw_um, 0.0)


def assess_zone(
    surface: np.ndarray,
    zone_mask: np.ndarray,
    target_um: float | None = None,
    *,
    speed_mm_s: Sequence[float] = (),
    distance_mm: Sequence[float] = (),
) -> dict[str, Any] | None:
    """Zone 하나의 평가(반올림하지 않은 값). 마스크 모양이 표면과 다르거나 마스크가 비면 None."""
    if zone_mask.shape != surface.shape or not zone_mask.any():
        return None
    rows = np.flatnonzero(zone_mask.any(axis=1))
    cols = np.flatnonzero(zone_mask.any(axis=0))
    box = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
    mask = zone_mask[box].astype(bool)
    raw = np.asarray(surface[box], dtype=np.float64)
    # 어느 크롭에도 들지 않은 화소를 "아직 칠하지 않음"으로 본다. PR #8 엔진은 "모름"으로 두고 등급을 보류한다.
    surface_obj = numerical.Surface(raw, np.ones(raw.shape, dtype=bool), raw > 0)
    target, target_basis = resolve_target(target_um)
    result = numerical.assess_zone(surface_obj, mask, target)
    quality = result["quality"]
    observed = result["metrics"]["observed_statistics"]

    zone_pixel_count = quality["zone_pixel_count"]
    grade = quality["grade"]
    bands = quality["six_band_counts"]
    counts = [bands[key] for key in numerical.SIX_BAND_KEYS]
    coverage_percent = 100.0 * sum(counts[2:]) / zone_pixel_count
    corrected, painted = numerical.corrected_values(surface_obj)
    filled = np.where(painted, np.clip(corrected / target, 0.0, 1.0), 0.0)
    if grade == "above":
        gate_status = "overpaint_risk"
    # 2026-11-12 책임자 결정: 면적을 채워도 칠한 곳 평균이 목표 미만(저도막)이면 다음 Zone으로 보내지 않는다.
    elif coverage_percent >= GATE_COVERAGE_PERCENT and grade != "below":
        gate_status = "complete"
    else:
        gate_status = "needs_more"

    # 식별자 인자는 PR #8 근거 기록용이다. 여기서는 후보의 변수·방향만 읽는다.
    coaching, _evidence, _stats = behavior.build_coaching(
        namespace="v1", quality=quality,
        behavior_values={"speed_mm_s": list(speed_mm_s), "distance_mm": list(distance_mm)},
        behavior_source_ids={}, observation_event_id="v1", contract_version="v1", revision=1)
    return {
        "analysis_status": quality["analysis_status"],
        "reason_code": quality["reason_codes"][0] if quality["reason_codes"] else None,
        "grade": grade,
        "status": STATUS_BY_GRADE.get(grade),
        "target_thickness_um": target,
        "target_basis": target_basis,
        "painted_mean_um": quality["painted_mean_um"],
        "painted_sum_um": observed["painted_sum_um"],
        "painted_pixel_count": quality["painted_pixel_count"],
        "zone_pixel_count": zone_pixel_count,
        "unpainted_fraction": quality["unpainted_fraction"],
        "six_band_counts": bands,
        "coverage_percent": coverage_percent,
        "overpaint_percent": 100.0 * sum(counts[3:]) / zone_pixel_count,
        "completion_percent": 100.0 * float(filled[mask].sum()) / zone_pixel_count,
        "gate_status": gate_status,
        "normal_band_um": normal_band_um(target),
        # PR #8의 statement 문장은 내부 용어가 있어 옮기지 않는다.
        "adjustments": [
            {key: action[key] for key in ("variable", "direction", "priority", "evidence_kind")}
            for action in coaching["actions"]],
        "adjustment_reason": coaching["reason_code"],
    }


def session_summary(assessments: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """`assess_zone` 결과들의 세션 집계. 세션 등급은 만들지 않는다(PR #8에도 없다)."""
    items = list(assessments)
    painted_pixel_count = sum(item["painted_pixel_count"] for item in items)
    zone_pixel_count = sum(item["zone_pixel_count"] for item in items)
    grade_counts = {"below": 0, "normal": 0, "above": 0, "none": 0}
    for item in items:
        grade_counts[item["grade"] or "none"] += 1
    target, target_basis = (
        (items[0]["target_thickness_um"], items[0]["target_basis"]) if items else resolve_target(None))
    return {
        "zone_count": len(items),
        "grade_counts": grade_counts,
        # 화소 수 가중이다(PR #8 세션 집계와 같은 식). Zone 평균의 평균이 아니다.
        "painted_mean_um": (
            sum(item["painted_sum_um"] for item in items) / painted_pixel_count if painted_pixel_count else None),
        "painted_pixel_count": painted_pixel_count,
        "zone_pixel_count": zone_pixel_count,
        # PR #8은 available이 아닌 Zone이 있으면 보류한다. v1은 판정한 Zone을 전부 관측된 것으로 보고 항상 낸다.
        "unpainted_fraction": (
            (zone_pixel_count - painted_pixel_count) / zone_pixel_count if zone_pixel_count else None),
        "six_band_counts": {
            key: sum(item["six_band_counts"][key] for item in items) for key in numerical.SIX_BAND_KEYS},
        "target_thickness_um": target,
        "target_basis": target_basis,
    }


def _below_tenths(mean_um: float, target_um: float) -> float:
    """저도막 평균을 0.1 단위로 내린다. 반올림하면 159.96이 목표 160.0으로 올라가 "목표 미만"과 모순된다.

    내린 값은 정수 반올림 결과도 원래 평균과 같다(153.45 → 153.4 → "153"). 0.1 반올림은 153.5 → "154"가 된다.
    """
    # round(…, 3): 89.99999 같은 부동소수 잡음을 89.9로 내리지 않는다. min: 그래도 목표에는 닿지 않는다.
    return min(math.floor(round(mean_um * 10, 3)) / 10, round(target_um - 0.1, 1))


def zone_row(map_num: int | str, zone_id: str, assessment: Mapping[str, Any]) -> dict[str, Any]:
    """저장·응답에 싣는 행. 반올림은 여기서만 한다."""
    mean_um = assessment["painted_mean_um"]
    return {
        "zone_id": str(zone_id),
        "map_num": int(map_num),
        "status": assessment["status"],
        "grade": assessment["grade"],
        "thickness_um": None if mean_um is None else (
            _below_tenths(mean_um, assessment["target_thickness_um"]) if assessment["grade"] == "below"
            else round(mean_um, 1)),
        "target_thickness_um": assessment["target_thickness_um"],
        "target_basis": assessment["target_basis"],
        "normal_band_um": assessment["normal_band_um"],
        "unpainted_fraction": round(assessment["unpainted_fraction"], 4),
        "six_band_counts": assessment["six_band_counts"],
        "coverage_percent": round(assessment["coverage_percent"], 1),
        "overpaint_percent": round(assessment["overpaint_percent"], 1),
        "completion_percent": round(assessment["completion_percent"], 1),
        "gate_status": assessment["gate_status"],
        "adjustments": assessment["adjustments"],
        "adjustment_reason": assessment["adjustment_reason"],
    }


def behavior_by_zone(bodies: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, list[float]]]:
    """저장된 시퀀스 본문 → {zone_id: {"speed_mm_s": [...], "distance_mm": [...]}}."""
    # PR #8은 Zone에 착탄한 분사 중 프레임을 표본으로 쓴다. v1에는 시퀀스 평균만 있다.
    samples: dict[str, dict[str, list[float]]] = {}
    for body in bodies:
        zone_id = str(body.get("zone_id") or "").strip()
        if zone_id in {"", "-1", "0"}:
            continue
        result = body.get("sequence_result") or {}
        zone = samples.setdefault(zone_id, {"speed_mm_s": [], "distance_mm": []})
        for name, field in _BEHAVIOR_FIELDS:
            value = result.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
                zone[name].append(float(value))
    return samples


def basis_label(target_um: float, target_basis: str) -> str:
    return f"{'기본' if target_basis == 'default_target' else '세션'} 목표 {target_um:g}µm 기준"


def mean_text(mean_um: float, target_um: float) -> str:
    """칠한 곳 평균의 표기(µm 앞 숫자). 저도막 행은 `zone_row`가 0.1 단위로 내린 값이 온다."""
    whole = math.floor(mean_um + 0.5)  # 사사오입. f"{:.0f}"는 x.5를 짝수 쪽으로 붙인다.
    if mean_um < target_um <= whole:
        # 평균 159.6을 "160µm"로 적으면 목표 160에 "저도막"이 모순으로 읽힌다. 목표 아래 소수 한 자리로 적는다.
        return f"{min(mean_um, target_um - 0.1):.1f}"
    return f"{mean_um:.1f}" if whole == target_um and mean_um != target_um else str(whole)


def grade_phrase(row: Mapping[str, Any]) -> str:
    """`zone_row` 한 줄 → 훈련생에게 보이는 등급 표기."""
    unpainted = f"미도장 {row['unpainted_fraction'] * 100:.0f}%"
    if row["grade"] is None:
        return f"칠한 곳 없음({unpainted})"
    return f"{GRADE_WORDS[row['grade']]}(칠한 곳 평균 {mean_text(row['thickness_um'], row['target_thickness_um'])}µm, {unpainted})"


def adjustment_sentence(adjustments: Sequence[Mapping[str, Any]]) -> str | None:
    if not adjustments:
        return None
    words = ", ".join(
        ADJUSTMENT_WORDS[(item["variable"], item["direction"])]
        for item in sorted(adjustments, key=lambda item: item["priority"]))
    return f"조절 후보: {words}. 원인으로 확정된 것은 아닙니다."


def distribution_sentence(six_band_counts: Mapping[str, int]) -> str | None:
    counts = [six_band_counts.get(key, 0) for key in numerical.SIX_BAND_KEYS]
    total = sum(counts)
    if not total:
        return None
    shares = ", ".join(f"{label} {100.0 * count / total:.0f}%" for label, count in zip(BAND_LABELS, counts))
    return f"칠한 면적의 두께 분포: 목표의 {shares}."


def assess_readings(
    values_um: Sequence[float],
    target_um: float | None = None,
    *,
    speed_mm_s: Sequence[float] = (),
    distance_mm: Sequence[float] = (),
) -> dict[str, Any]:
    """측정값 목록(이미 µm인 물리량) → PR #8 등급·6구간·조절 후보. 높이맵이 아니므로 10 µm를 빼지 않는다."""
    if not len(values_um):
        raise ValueError("values_um must not be empty")
    values = np.asarray(values_um, dtype=np.float64)
    # PR #8 판정은 높이맵에서 DeadVolume 10 µm를 한 번 빼므로, 측정값에는 미리 더해 상쇄한다.
    raw = (values + numerical.DEAD_VOLUME_UM).reshape(1, -1)
    result = assess_zone(raw, np.ones(raw.shape, dtype=bool), target_um, speed_mm_s=speed_mm_s, distance_mm=distance_mm)
    assert result is not None  # 마스크가 비지 않고 모양이 같다
    target = result["target_thickness_um"]
    result["point_grades"] = [
        numerical.grade_for_mean(float(v), target) if v >= numerical.MIN_PAINTED_UM else None for v in values]
    return result
