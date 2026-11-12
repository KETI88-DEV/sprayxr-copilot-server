"""Factual coating answers from authenticated viewer statistics or saved reports."""
from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from typing import Any

from trainer.demo_api.coating_assessment import (
    RULE_NOTE, adjustment_sentence, basis_label, distribution_sentence, grade_phrase, mean_text)

_ZONE_NUMBER = re.compile(r"(?:zone|존|구역)\s*(\d+)", re.I)


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and value >= 0 else None


def viewer_coating_context(payload: Mapping[str, Any]) -> dict[str, Any]:
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
    return {"analysis_status": "available" if rows else "not_available", "scope": "final_record",
            "comparison_basis": "expert_zone_median", "zones": rows}


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
                if any(isinstance(row, Mapping) and str(row.get("map_num")) == map_num and str(row.get("zone_id")) == zone_id
                       for row in claims.get("zones", [])):
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


def _grade_answer(rows: list[dict[str, Any]], prompt: str, *, detailed: bool) -> str:
    """저장된 행의 등급을 읽어 옮긴다. 등급을 다시 계산하지 않는다."""
    basis = basis_label(rows[0]["target_thickness_um"], rows[0]["target_basis"])
    text = [f"{basis}으로 Zone별 도막을 판정했습니다. {RULE_NOTE}"]
    if _ZONE_NUMBER.search(prompt):
        text += [f"{_zone_label(row)}: {grade_phrase(row)}" for row in rows]
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
    return "\n".join(line for line in text if line)


def answer_distribution_cause(context: Mapping[str, Any], *, detailed: bool) -> str | None:
    """error_analysis 답: 분포와 저도막·과도막 Zone의 조절 후보. 도막 자료가 없으면 None."""
    zones = context.get("zones") or []
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
            candidates = adjustment_sentence(row["adjustments"])
            # 어댑터 문장의 "원인으로 확정된 것은 아닙니다."는 끝줄에서 한 번만 말한다.
            text.append(f"{word} Zone {row['zone_id']}(칠한 곳 평균 {mean_text(row['thickness_um'], row['target_thickness_um'])}µm): "
                        + (candidates.partition(" 원인으로")[0] if candidates else "속도·이격거리 기록이 없어 조절 후보를 내지 못했습니다."))
    if len(text) == 1:
        return text[0] + "\n저도막·과도막으로 판정된 Zone이 없습니다."
    return "\n".join([*text, "조절 후보는 등급에 따른 일반적인 방향이며 원인으로 확정된 것은 아닙니다."])


def answer_coating_question(context: dict[str, Any], prompt: str, *, detailed: bool) -> str:
    rows = context.get("zones") or []
    selected_map = re.search(r"map\s*(\d+)", prompt, re.I)
    if selected_map:
        rows = [row for row in rows if str(row.get("map_num")) == selected_map[1]]
    target = _ZONE_NUMBER.search(prompt)
    if target:
        rows = [row for row in rows if str(row.get("zone_id")) == target[1]]
    if context.get("analysis_status") != "available" or not rows:
        return ("도막의 과다·부족 영역을 판단할 신뢰 가능한 Zone별 도막 분석 자료가 없습니다. "
                "뷰어의 기록을 다시 불러오거나 세션 완료 후 도막 분석 결과를 확인해 주세요.")
    if context.get("comparison_basis") == "saved_coating_report":
        if all("grade" in row for row in rows):
            return _grade_answer(rows, prompt, detailed=detailed)
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
    text = ["최종 높이맵의 Zone별 대표 두께를 전문가 기준과 비교했습니다."]
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
