"""Factual coating answers from authenticated viewer statistics or saved reports."""
from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from typing import Any


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
    rows = [dict(row) for row in rows if isinstance(row, Mapping) and row.get("status") in {"normal", "under_app", "over_app"}] if isinstance(rows, list) else []
    return {"analysis_status": "available" if rows else "not_available", "scope": "completed_session",
            "comparison_basis": "saved_coating_report", "zones": rows}


def answer_coating_question(context: dict[str, Any], prompt: str, *, detailed: bool) -> str:
    rows = context.get("zones") or []
    selected_map = re.search(r"map\s*(\d+)", prompt, re.I)
    if selected_map:
        rows = [row for row in rows if str(row.get("map_num")) == selected_map[1]]
    target = re.search(r"(?:zone|존|구역)\s*(\d+)", prompt, re.I)
    if target:
        rows = [row for row in rows if str(row.get("zone_id")) == target[1]]
    if context.get("analysis_status") != "available" or not rows:
        return ("도막의 과다·부족 영역을 판단할 신뢰 가능한 Zone별 도막 분석 자료가 없습니다. "
                "뷰어의 기록을 다시 불러오거나 세션 완료 후 도막 분석 결과를 확인해 주세요.")
    if context.get("comparison_basis") == "saved_coating_report":
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
