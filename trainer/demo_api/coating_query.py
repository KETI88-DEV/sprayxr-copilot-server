"""Factual coating answers from authenticated viewer statistics or saved reports."""
from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from typing import Any, Final

from trainer.demo_api.coating_assessment import (
    ADJUSTMENT_WORDS, RULE_NOTE, adjustment_sentence, basis_label, distribution_sentence, grade_phrase, mean_text)
from trainer.demo_api.session_coating import EXPERT_WORDS

_ZONE_NUMBER = re.compile(r"(?:zone|존|구역)\s*(\d+)", re.I)
_EXPERT = re.compile(r"전문가|expert", re.I)
# 뷰어 기록에 목표 대비 판정이 없을 때 전문가 대비 답 앞에 밝히는 사유. 사유와 다른 말을 하지 않는다.
_TARGET_MISSING = {
    "height_map_unavailable": "이 기록은 높이맵 원본이 없어 목표 대비 판정을 낼 수 없습니다.",
    "height_map_size_mismatch": "이 기록의 높이맵 크기가 Zone 기준 자료와 달라 목표 대비 판정을 내지 않았습니다.",
    "zone_reference_unavailable": "이 콘텐츠의 Zone 기준 자료가 없어 목표 대비 판정을 낼 수 없습니다.",
}
_TARGET_NOT_SENT = "이 질문에는 목표 대비 판정 자료가 함께 오지 않았습니다."
_VIEWER_NOTE = "최종 높이맵 기준이며 재생 시점의 도막이 아닙니다."
# 전문가 대비 행이 있는 기록에서만 붙인다(전문가 본인 기록 등은 화면의 '전문가 대비'가 비어 있다).
_VIEWER_EXPERT_NOTE = " 전문가 기록과 견준 두께는 화면의 '전문가 대비'에서 볼 수 있습니다."


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and value >= 0 else None


def viewer_coating_context(payload: Mapping[str, Any], *, target: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """전문가 대비 행(`zones`)과 목표 대비 평가(`target`, PR #8 기준)를 함께 싣는다."""
    # target 없이 불리는 곳(자세 탭 대체 경로)은 높이맵이 없는 것이 아니라 도막 문맥이 오지 않은 것이다.
    target = dict(target) if target is not None else {
        "analysis_status": "not_available", "reason_code": "target_assessment_not_loaded"}
    evaluation = payload.get("pose_evaluation") or {}
    zones = evaluation.get("zones") or {}
    rows = []
    for zone in zones.values() if isinstance(zones, Mapping) else []:
        if not isinstance(zone, Mapping):
            continue
        metric = next((m for m in zone.get("metrics", [])
                       if isinstance(m, Mapping) and m.get("key") == "coating_thickness"), {})
        actual, reference = _number(metric.get("actual")), _number(metric.get("reference"))
        if metric.get("available") is not True or actual is None or reference is None or reference <= 0:
            continue
        rows.append({"map_num": str(zone.get("map_num")), "zone_id": str(zone.get("zone_id")),
                     "actual_um": actual, "reference_um": reference})
    available = bool(rows) or target.get("analysis_status") == "available"
    return {"analysis_status": "available" if available else "not_available", "scope": "final_record",
            "comparison_basis": "expert_zone_median", "zones": rows, "target": target}


def parse_viewer_coating_context(value: str | None, task_type: str) -> dict[str, Any] | None:
    if not value or not value.startswith(("motion_viewer_analysis:", "motion_viewer_playback:")):
        return None
    try:
        payload = json.loads(value.split(":", 1)[1])
        if not isinstance(payload, dict):
            return None
        from trainer.demo_api.viewer_context import verify_context
        claims = payload.get("coating_context")
        if isinstance(claims, dict) and verify_context("coating_status", claims, payload.get("coating_signature")):
            playback = payload.get("playback_context")
            if isinstance(playback, dict):
                map_num, zone_id = str(playback.get("map_num")), str(playback.get("zone_id"))
                target = claims.get("target")
                target_rows = target.get("zones") if isinstance(target, Mapping) else None
                signed_rows = [*(claims.get("zones") or []), *(target_rows if isinstance(target_rows, list) else [])]
                if any(isinstance(row, Mapping) and str(row.get("map_num")) == map_num and str(row.get("zone_id")) == zone_id
                       for row in signed_rows):
                    # A client selector can choose authenticated rows, never create values.
                    return {**claims, "playback_context": {"map_num": map_num, "zone_id": zone_id}}
            return claims
        # Previously loaded pose tabs already contain authenticated thickness metrics.
        claims = payload.get("analysis_context")
        if task_type == "pose_recomm" and isinstance(claims, dict) and verify_context(task_type, claims, payload.get("signature")):
            return viewer_coating_context({"pose_evaluation": claims})
    except (TypeError, ValueError, OverflowError):
        pass
    return None


def report_coating_context(found: tuple[str, Mapping[str, Any]] | None) -> dict[str, Any]:
    rows = found[1].get("coating_zones") if found else None
    # 칠한 곳이 없는 새 행은 status가 None이다. grade 열쇠가 있으면 새 행이다.
    rows = [dict(row) for row in rows if isinstance(row, Mapping)
            and (row.get("status") in {"normal", "under_app", "over_app"} or "grade" in row)] if isinstance(rows, list) else []
    return {"analysis_status": "available" if rows else "not_available", "scope": "completed_session",
            "comparison_basis": "saved_coating_report", "zones": rows}


def _zone_label(row: Mapping[str, Any]) -> str:
    return f"Map {row.get('map_num', '?')} · Zone {row.get('zone_id')}"


def _graded(rows: list[dict[str, Any]], grade: str | None) -> list[dict[str, Any]]:
    """그 등급의 행. 저도막·과도막은 목표에서 먼 순서다."""
    group = [row for row in rows if row["grade"] == grade]
    if grade in ("below", "above"):
        group.sort(key=lambda row: -abs(row["thickness_um"] - row["target_thickness_um"]) / row["target_thickness_um"])
    return group


def _listed(labels: list[str], *, detailed: bool) -> str:
    """'n개 — a, b, c 외 k곳'. 빠른 안내는 3곳까지만 적는다."""
    shown = labels if detailed else labels[:3]
    return f"{len(labels)}개 — {', '.join(shown)}" + (f" 외 {len(labels) - len(shown)}곳" if len(labels) > len(shown) else "")


def _sum_bands(rows: list[dict[str, Any]]) -> dict[str, int]:
    total: dict[str, int] = {}
    for row in rows:
        for key, count in row["six_band_counts"].items():
            total[key] = total.get(key, 0) + count
    return total


def _expert_line(rows: list[dict[str, Any]], *, detailed: bool) -> str:
    """목표 대비 줄 뒤에 붙는 참고 한 줄. 전문가 대비 낱말은 목표 대비 등급 낱말과 섞지 않는다."""
    parts = []
    for key, word in EXPERT_WORDS.items():
        found = [str(row["zone_id"]) for row in rows if (row.get("expert") or {}).get("status") == key]
        shown = found if detailed else found[:3]
        parts.append(f"{word} {len(found)}개" + (f"(Zone {'·'.join(shown)}{' 외' if len(found) > len(shown) else ''})" if key != "normal" and found else ""))
    return "참고(전문가 대비, 보정 전 두께): " + ", ".join(parts) + "."


def _grade_answer(rows: list[dict[str, Any]], prompt: str, *, detailed: bool, viewer: bool = False,
                  expert_rows: bool = False, report_expert: bool = False) -> str:
    """저장된 행의 등급을 읽어 옮긴다. 등급을 다시 계산하지 않는다."""
    basis = basis_label(rows[0]["target_thickness_um"], rows[0]["target_basis"])
    text = [f"{basis}으로 Zone별 도막을 판정했습니다. {RULE_NOTE}"]
    if _ZONE_NUMBER.search(prompt):
        text += [f"{_zone_label(row)}: {grade_phrase(row)}"
                 + (f" · 전문가 대비 {EXPERT_WORDS[row['expert']['status']]}" if report_expert and row.get("expert") else "")
                 for row in rows]
    else:
        for word, grade in (("저도막", "below"), ("과도막", "above")):
            # grade_phrase의 "저도막(칠한 곳 평균 …)"에서 괄호 부분만 Zone 이름 뒤에 붙인다.
            group = [_zone_label(row) + grade_phrase(row).removeprefix(word) for row in _graded(rows, grade)]
            text.append(f"{word}: " + (_listed(group, detailed=detailed) if group else "없음"))
        text.append(f"정상: {len(_graded(rows, 'normal'))}개")
        unpainted = [_zone_label(row) for row in _graded(rows, None)]
        text.append("칠한 곳 없음: " + (_listed(unpainted, detailed=detailed) if unpainted else "0개"))
    if "분포" in prompt:
        text.append(distribution_sentence(_sum_bands(rows)) or "")
    if "미도장" in prompt or "안 칠" in prompt:
        shares = [(row["zone_id"], f"{row['unpainted_fraction'] * 100:.0f}%")
                  for row in sorted(rows, key=lambda row: -row["unpainted_fraction"])]
        # 0%로 적히는 Zone은 "큰 곳"이 아니다.
        largest = [f"Zone {zone_id}({share})" for zone_id, share in shares if share != "0%"][:3]
        # 미도장 비율에는 합격 기준이 없다. 있다/없다로 판정하지 않고 숫자만 옮긴다.
        text.append(("미도장 비율이 큰 곳: " + ", ".join(largest) + "." if largest else "판정한 Zone의 미도장 비율은 모두 1% 미만입니다.")
                    + " 미도장 비율에는 합격 기준이 없어 숫자만 알려 드립니다.")
    if report_expert and any(row.get("expert") for row in rows):
        text.append(_expert_line(rows, detailed=detailed))
    if viewer:
        text.append(_VIEWER_NOTE + (_VIEWER_EXPERT_NOTE if expert_rows else ""))
    return "\n".join(line for line in text if line)


# --- 조절 후보와 실측의 비교(이슈 #19) ---------------------------------------------------------------------
#
# PR #8 규칙의 조절 후보는 등급만 보고 정한 방향이다(저도막 → 속도↓·거리↓). 2026-11-14 운영 기록의 저도막 8곳 중
# 6곳은 이미 전문가보다 느리고 가까웠다. 규칙은 다른 개발자 소관이라 고치지 않는다(이슈 #19). 여기서는 그 Zone의
# 실측 속도·이격거리와 전문가 중앙값을 함께 보고, 실측과 엇갈리는 후보는 조언으로 말하지 않는다.
_MEASURE_FIELDS: Final = {"speed_mm_s": ("speed_mms", "reference_speed_mms", "이동 속도", "mm/s", "빠름", "느림"),
                          "distance_mm": ("distance_mm", "reference_distance_mm", "이격거리", "mm", "멂", "가까움")}
_SIMILAR: Final = 0.05  # 전문가 중앙값과 ±5% 안이면 "비슷"으로 보고 후보와 엇갈린다고 하지 않는다
UNDER_REVIEW: Final = "규칙의 조절 후보({words})는 실측 비교와 맞지 않아 검토 중이라 조언으로 드리지 않습니다."


def zone_measures(claims: Mapping[str, Any] | None) -> dict[str, dict[str, float]]:
    """서명된 뷰어 대화 자료 → Zone 번호별 실측·전문가 중앙값(속도·이격거리) 그리고 목표 대비 등급."""
    out: dict[str, dict[str, Any]] = {}
    for zone in ((claims or {}).get("zones") or {}).values():
        if not isinstance(zone, Mapping) or zone.get("zone_id") is None:
            continue
        values = {key: float(zone[key]) for key in ("speed_mms", "reference_speed_mms", "distance_mm", "reference_distance_mm")
                  if isinstance(zone.get(key), (int, float)) and not isinstance(zone.get(key), bool) and zone[key] > 0}
        grade = (zone.get("coating_target") or {}).get("grade") if isinstance(zone.get("coating_target"), Mapping) else None
        out[str(zone["zone_id"])] = {**values, **({"grade": grade} if grade else {})}
    return out


def measured_relation(measure: Mapping[str, Any] | None, variable: str) -> str | None:
    """이 기록이 전문가 중앙값보다 "lower"·"higher"·"similar"인가. 값이 없으면 None."""
    own, reference = _MEASURE_FIELDS[variable][:2]
    mine, expert = (measure or {}).get(own), (measure or {}).get(reference)
    if not isinstance(mine, (int, float)) or not isinstance(expert, (int, float)) or expert <= 0:
        return None
    ratio = mine / expert - 1
    return "lower" if ratio <= -_SIMILAR else "higher" if ratio >= _SIMILAR else "similar"


def contradicted(adjustment: Mapping[str, Any], measure: Mapping[str, Any] | None) -> bool:
    """이미 전문가보다 낮은데 낮추라거나, 이미 높은데 높이라는 후보."""
    relation = measured_relation(measure, str(adjustment.get("variable")))
    return (adjustment.get("direction"), relation) in {("decrease", "lower"), ("increase", "higher")}


def measured_sentence(zone_id: str, measure: Mapping[str, Any] | None) -> str | None:
    """'실측 이동 속도 2,230 mm/s(전문가 3,514 mm/s보다 37% 느림), 이격거리 …'."""
    parts = []
    for variable, (own, reference, title, unit, up, down) in _MEASURE_FIELDS.items():
        relation = measured_relation(measure, variable)
        if relation is None:
            continue
        mine, expert = measure[own], measure[reference]  # type: ignore[index]
        word = {"lower": down, "higher": up, "similar": "비슷"}[relation]
        gap = f"{abs(mine / expert - 1) * 100:.0f}% {word}" if relation != "similar" else "비슷"
        parts.append(f"{title} {mine:,.0f} {unit}(전문가 {expert:,.0f} {unit}보다 {gap})" if relation != "similar"
                     else f"{title} {mine:,.0f} {unit}(전문가 {expert:,.0f} {unit}와 비슷)")
    if not parts:
        return None
    text = f"Zone {zone_id} 실측: " + ", ".join(parts) + "."
    grade = (measure or {}).get("grade")
    slow_close = {measured_relation(measure, name) for name in _MEASURE_FIELDS} == {"lower"}
    fast_far = {measured_relation(measure, name) for name in _MEASURE_FIELDS} == {"higher"}
    if grade == "below" and slow_close:
        text += " 가까이·천천히 칠하면 보통 도막이 두꺼워지는 쪽이라, 이 두 값으로는 얇게 나온 까닭을 설명하기 어렵습니다."
    elif grade == "above" and fast_far:
        text += " 멀리·빠르게 칠하면 보통 도막이 얇아지는 쪽이라, 이 두 값으로는 두껍게 나온 까닭을 설명하기 어렵습니다."
    return text


def _candidate_line(adjustments: list[Mapping[str, Any]], measure: Mapping[str, Any] | None, zone_id: str) -> str:
    """조절 후보 문장. 실측이 있으면 엇갈리는 후보는 빼고 실측 비교와 검토 중임을 말한다."""
    if measure is None:
        candidates = adjustment_sentence(adjustments)
        return candidates.partition(" 원인으로")[0] if candidates else "속도·이격거리 기록이 없어 조절 후보를 내지 못했습니다."
    kept = [item for item in adjustments if not contradicted(item, measure)]
    dropped = [item for item in adjustments if contradicted(item, measure)]
    parts = [measured_sentence(zone_id, measure) or ""]
    if kept:
        parts.append((adjustment_sentence(kept) or "").partition(" 원인으로")[0])
    if dropped:
        words = ", ".join(ADJUSTMENT_WORDS[(item["variable"], item["direction"])] for item in sorted(dropped, key=lambda item: item["priority"]))
        parts.append(UNDER_REVIEW.format(words=words))
    return " ".join(part for part in parts if part) or "속도·이격거리 기록이 없어 조절 후보를 내지 못했습니다."


_THIN: Final = r"얇|저도막|두께가\s*부족|분사량이\s*줄|도막이\s*줄"
_CLOSE_SLOW: Final = r"가깝|가까웠|가까워|짧았|짧아|짧고|짧은\s*(?:거리|이격)|느렸|느려|느리고|느린\s*속도|낮았|낮아|낮고|낮은\s*속도"
_THICK: Final = r"두꺼|두껍|과도막"
_FAR_FAST: Final = r"멀었|멀어|멀고|먼\s*거리|길었|길어|빨랐|빨라|빠르고|빠른\s*속도|높았|높아|과도한\s*(?:분사\s*)?(?:거리|이격|속도)"
_CONCESSIVE: Final = r"는데도|은데도|인데도|음에도|에도\s*불구|데도"
_ADVICE: Final = {("distance_mm", "decrease"): r"(?:거리|이격)\S*\s*(?:더\s*)?(?:줄|좁히|짧게)|더\s*가까이|가까이\s*(?:가져|대|붙|이동|다가)",
                  ("distance_mm", "increase"): r"(?:거리|이격)\S*\s*(?:더\s*)?(?:늘|넓히|길게)|더\s*멀리|멀리\s*(?:떨어|이동)",
                  ("speed_mm_s", "decrease"): r"속도\S*\s*(?:더\s*)?(?:낮추|줄이|늦추)|더\s*천천히|천천히\s*(?:움직|이동|칠|뿌)",
                  ("speed_mm_s", "increase"): r"속도\S*\s*(?:더\s*)?(?:높이|올리)|더\s*빠르게|빠르게\s*(?:움직|이동|칠|뿌)"}
_GENERAL_PHYSICS: Final = "일반적으로 가까이·천천히 칠할수록 도막이 두꺼워지고, 멀리·빠르게 칠할수록 얇아집니다."


def inverted_cause(sentence: str) -> bool:
    """"가깝고 느려서 얇아졌다"처럼 물리 방향이 거꾸로 된 원인 문장. "~는데도 얇았다"는 거꾸로가 아니다."""
    if re.search(_CONCESSIVE, sentence):
        return False
    return bool((re.search(_THIN, sentence) and re.search(_CLOSE_SLOW, sentence))
                or (re.search(_THICK, sentence) and re.search(_FAR_FAST, sentence)))


def _contradicts_measure(sentence: str, measures: Mapping[str, Mapping[str, Any]], zones: list[str]) -> bool:
    for zone in zones:
        for (variable, direction), pattern in _ADVICE.items():
            if re.search(pattern, sentence) and contradicted({"variable": variable, "direction": direction}, measures.get(zone)):
                return True
    return False


def drop_contradicted_causes(answer: str, measures: Mapping[str, Mapping[str, Any]] | None, *, default_zone: str | None = None) -> str:
    """모델 답에서 물리가 거꾸로 된 원인 문장과 실측 방향과 엇갈리는 조언 문장을 뺀다.

    다 빠지면 그 Zone의 실측 비교(없으면 일반적인 방향 한 문장)로 답한다 — 빈 답이나 틀린 인과를 내보내지 않는다.
    """
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", answer) if part.strip()]
    kept: list[str] = []
    dropped_physics = False
    for sentence in sentences:
        named = [a or b for a, b in re.findall(r"(?:zone|존|구역)\s*(\d+)|(\d+)\s*번\s*(?:zone|존|구역)", sentence, re.I)]
        zones = named or ([default_zone] if default_zone else [])
        if inverted_cause(sentence):
            dropped_physics = True
        elif not (measures and _contradicts_measure(sentence, measures, zones)):
            kept.append(sentence)
    if len(kept) == len(sentences):
        return answer
    if kept:
        return " ".join(kept)
    fallback = measured_sentence(default_zone, (measures or {}).get(default_zone)) if default_zone else None
    return fallback or (_GENERAL_PHYSICS if dropped_physics else "제공된 자료로는 답할 수 없습니다.")


def answer_distribution_cause(context: Mapping[str, Any], *, detailed: bool,
                              measures: Mapping[str, Mapping[str, Any]] | None = None) -> str | None:
    """error_analysis 답: 분포와 저도막·과도막 Zone의 조절 후보. 도막 자료가 없으면 None.

    `measures`(뷰어 Zone 실측)가 있으면 후보를 실측과 맞춰 본다(_candidate_line).
    """
    zones = context.get("zones") or []
    if context.get("comparison_basis") == "expert_zone_median":
        # 뷰어의 전문가 대비 행에는 등급이 없다. 서명된 목표 대비 행만 읽는다.
        zones = _target_rows(context) or []
    rows = [row for row in zones if "grade" in row]
    if context.get("analysis_status") != "available" or not rows:
        # 2026-11 이전에 저장된 리포트 행에는 등급·분포가 없다. 완료한 세션에 "완료 분석이 없다"고 답하지 않는다.
        return ("이 세션의 완료 분석은 이전 기준으로 저장되어 있어 도막 분포와 조절 후보를 낼 수 없습니다."
                if zones and context.get("comparison_basis") == "saved_coating_report" else None)
    basis = basis_label(rows[0]["target_thickness_um"], rows[0]["target_basis"])
    text = [" ".join(filter(None, (f"{basis}으로 본 도막 분포입니다.", distribution_sentence(_sum_bands(rows)))))]
    for word, grade in (("저도막", "below"), ("과도막", "above")):
        group = _graded(rows, grade)
        for row in group if detailed else group[:3]:
            # 어댑터 문장의 "원인으로 확정된 것은 아닙니다."는 끝줄에서 한 번만 말한다.
            measure = measures.get(str(row["zone_id"])) if measures else None
            text.append(f"{word} Zone {row['zone_id']}(칠한 곳 평균 {mean_text(row['thickness_um'], row['target_thickness_um'])}µm): "
                        + _candidate_line(row["adjustments"], measure, str(row["zone_id"])))
    if len(text) == 1:
        return text[0] + "\n저도막·과도막으로 판정된 Zone이 없습니다."
    return "\n".join([*text, "조절 후보는 등급에 따른 일반적인 방향이며 원인으로 확정된 것은 아닙니다."])


def _target_rows(context: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    """뷰어 문맥의 목표 대비 행. 판정이 없으면 None."""
    target = context.get("target")
    return target.get("zones") or [] if isinstance(target, Mapping) and target.get("analysis_status") == "available" else None


def _selected(rows: list[dict[str, Any]], prompt: str) -> list[dict[str, Any]]:
    """질문이 Map·Zone 번호를 말하면 그 행만 남긴다."""
    selected_map = re.search(r"map\s*(\d+)", prompt, re.I)
    if selected_map:
        rows = [row for row in rows if str(row.get("map_num")) == selected_map[1]]
    zone = _ZONE_NUMBER.search(prompt)
    if zone:
        rows = [row for row in rows if str(row.get("zone_id")) == zone[1]]
    return rows


def target_advice_facts(context: Mapping[str, Any], *, measures: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any] | None:
    """조언 질문의 프롬프트에 넣는 줄인 목표 대비 판정: 재생 위치의 Zone과 저도막·과도막 Zone, 최대 6개.

    `measures`가 있으면 Zone마다 실측·전문가 중앙값을 싣고, 실측과 엇갈리는 후보는 `adjustments`에서 빼 따로 둔다.
    """
    rows = _target_rows(context)
    if not rows:
        return None
    playback = context.get("playback_context") or {}
    here = [row for row in rows if (str(row["map_num"]), str(row["zone_id"])) == (str(playback.get("map_num")), str(playback.get("zone_id")))]
    flagged = [row for row in _graded(rows, "below") + _graded(rows, "above") if row not in here]
    def fact(row: Mapping[str, Any]) -> dict[str, Any]:
        item = {"map_num": row["map_num"], "zone_id": row["zone_id"], "grade": row["grade"],
                "label": grade_phrase(row), "adjustments": row["adjustments"]}
        measure = measures.get(str(row["zone_id"])) if measures else None
        if measure is None:
            return item
        item["adjustments"] = [a for a in row["adjustments"] if not contradicted(a, measure)]
        item["adjustments_under_review"] = [a for a in row["adjustments"] if contradicted(a, measure)]
        item["measured"] = {variable: {"record": measure[own], "expert_median": measure[reference],
                                       "relation": measured_relation(measure, variable)}
                            for variable, (own, reference, *_rest) in _MEASURE_FIELDS.items()
                            if measured_relation(measure, variable) is not None}
        return item

    return {"basis": context["target"]["basis_label"], "zones": [fact(row) for row in (here + flagged)[:6]]}


def target_findings_line(context: Mapping[str, Any] | None) -> str | None:
    """뷰어 '고칠 점' 답 끝의 도막 한 줄. 서명된 목표 대비 판정이 없으면 None."""
    rows = _target_rows(context) if context else None
    if rows is None:
        return None
    groups = [f"{word} Zone " + "·".join(str(row["zone_id"]) for row in _graded(rows, grade))
              for word, grade in (("저도막", "below"), ("과도막", "above")) if _graded(rows, grade)]
    return f"도막({context['target']['basis_label']}): " + (", ".join(groups) or "저도막·과도막 Zone 없음") + "."


def answer_coating_question(context: dict[str, Any], prompt: str, *, detailed: bool) -> str:
    rows = _selected(context.get("zones") or [], prompt)
    note = None
    if context.get("comparison_basis") == "expert_zone_median" and not _EXPERT.search(prompt):
        target_rows = _target_rows(context)
        if target_rows is None:
            target = context.get("target") if isinstance(context.get("target"), Mapping) else {}
            note = _TARGET_MISSING.get(target.get("reason_code"), _TARGET_NOT_SENT) + " 아래는 전문가 기록과 견준 값입니다."
        elif _selected(target_rows, prompt):
            return _grade_answer(_selected(target_rows, prompt), prompt, detailed=detailed, viewer=True,
                                 expert_rows=bool(context.get("zones")))
    if context.get("analysis_status") != "available" or not rows:
        return ("도막의 과다·부족 영역을 판단할 신뢰 가능한 Zone별 도막 분석 자료가 없습니다. "
                "뷰어의 기록을 다시 불러오거나 세션 완료 후 도막 분석 결과를 확인해 주세요.")
    if context.get("comparison_basis") == "saved_coating_report":
        if all("grade" in row for row in rows):
            return _grade_answer(rows, prompt, detailed=detailed, report_expert=True)
        # 2026-11 이전에 저장된 리포트 행에는 grade가 없다.
        groups = [(label, [row for row in rows if row.get("status") == status])
                  for label, status in (("도막 부족", "under_app"), ("도막 과다", "over_app"))]
        text = ["분석 가능한 Zone의 저장된 최종 도막 결과를 기준으로 확인했습니다."]
        for label, group in groups:
            text.append(f"{label}: " + (", ".join(f"Map {r.get('map_num', '?')} · Zone {r.get('zone_id')}" for r in group) or "판정된 영역 없음"))
        return "\n".join(text)

    valid = []
    for row in rows:
        actual, reference = _number(row.get("actual_um")), _number(row.get("reference_um"))
        if actual is not None and reference is not None and reference > 0:
            valid.append({**row, "delta": actual - reference, "delta_percent": (actual / reference - 1) * 100})
    if not valid:
        return "Zone별 도막 두께 비교값이 없어 과다·부족 영역을 판단할 수 없습니다."
    text = [*filter(None, [note]), "최종 높이맵의 Zone별 대표 두께를 전문가 기준과 비교했습니다."]
    for label, group in (("전문가 기준보다 두꺼운 영역", sorted((r for r in valid if r["delta"] > 1e-6), key=lambda r: -r["delta_percent"])),
                         ("전문가 기준보다 얇은 영역", sorted((r for r in valid if r["delta"] < -1e-6), key=lambda r: r["delta_percent"]))):
        text.append(f"{label}: {len(group)}개" + (" (차이가 큰 3곳)" if not detailed and len(group) > 3 else ""))
        for row in group if detailed else group[:3]:
            text.append(f"- Map {row['map_num']} · Zone {row['zone_id']}: {row['actual_um']:g}µm / 전문가 {row['reference_um']:g}µm ({row['delta_percent']:+.1f}%)")
    same = [row for row in valid if abs(row["delta"]) <= 1e-6]
    if same:
        text.append("전문가 기준과 같은 대표 두께:")
        for row in same if detailed else same[:3]:
            text.append(f"- Map {row['map_num']} · Zone {row['zone_id']}: {row['actual_um']:g}µm / 전문가 {row['reference_um']:g}µm")
    text.append("최종 높이맵 표본의 Zone별 중앙값 비교이며, 재생 시점의 누적 도막이나 목표 두께의 합격·불합격 판정은 아닙니다.")
    return "\n".join(text)
