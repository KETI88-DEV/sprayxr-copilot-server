"""Expert-reference sequence order and UV stroke-direction evaluation."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.telemetry_contracts import (
    AnalyzeSequenceRequest,
    EngineFeedback,
    learning_mode_from_scenario,
    process_from_tool_type,
)

ALGORITHM_VERSION = "sequence-reference-1.0"
_NO_ZONE = {"", "-1", "0", "none", "null", "nan"}


@dataclass(frozen=True)
class SequenceDecision:
    feedback: EngineFeedback
    text_guide: str | None
    anomaly_type: str | None = None


def _zone_id(body: dict[str, Any]) -> str | None:
    zone = str(body.get("zone_id", "")).strip()
    return None if zone.lower() in _NO_ZONE else zone


def _map_num(body: dict[str, Any]) -> int:
    for frame in body.get("sequence_info") or []:
        metrics = frame.get("gun_metrics") if isinstance(frame, dict) else None
        if not isinstance(metrics, dict):
            continue
        raw = metrics.get("raytexmapnum", metrics.get("raytex_map_num", 1))
        try:
            return max(1, int(float(raw)))
        except (TypeError, ValueError):
            continue
    return 1


def observed_direction(body: dict[str, Any]) -> str | None:
    """Classify a sequence from TSB's pixel-space ``raytexuv`` trace."""
    points: list[tuple[float, float]] = []
    for frame in body.get("sequence_info") or []:
        metrics = frame.get("gun_metrics") if isinstance(frame, dict) else None
        uv = metrics.get("raytexuv") if isinstance(metrics, dict) else None
        if not isinstance(uv, (list, tuple)) or len(uv) < 2:
            continue
        try:
            point = (float(uv[0]), float(uv[1]))
        except (TypeError, ValueError):
            continue
        if all(math.isfinite(value) for value in point):
            points.append(point)
    if len(points) < 2:
        return None
    x_travel = sum(abs(b[0] - a[0]) for a, b in zip(points, points[1:]))
    y_travel = sum(abs(b[1] - a[1]) for a, b in zip(points, points[1:]))
    if x_travel + y_travel < 5.0:
        return "stationary"
    displacement = math.dist(points[0], points[-1])
    path_length = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
    if x_travel >= y_travel * 1.5:
        return "horizontal"
    if y_travel >= x_travel * 1.5:
        return "vertical"
    return "diagonal" if path_length and displacement / path_length >= 0.5 else "curved_irregular"


class SequenceAnalysisEngine:
    def __init__(self, reference_root: Path) -> None:
        self._root = Path(reference_root)

    def _profile(self, process: str, mode_id: int) -> dict[str, Any] | None:
        path = self._root / "generated" / "profiles" / process / "learning" / f"mode_{mode_id:02d}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload.get("aggregate", {}).get("zones"), dict):
                return None
            return payload
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

    @staticmethod
    def _flow(profile: dict[str, Any]) -> list[str]:
        orders = profile.get("aggregate", {}).get("macro_orders") or []
        if not orders or not isinstance(orders[0], dict):
            return []
        return [str(item["zone_id"]) for item in orders[0].get("order", []) if isinstance(item, dict) and item.get("zone_id") is not None]

    @staticmethod
    def _expected(profile: dict[str, Any], zone_id: str, map_num: int = 1) -> str | None:
        zones = profile["aggregate"]["zones"]
        direct = zones.get(f"{map_num}:{zone_id}") or zones.get(zone_id)
        if isinstance(direct, dict) and isinstance(direct.get("expected_direction"), str):
            return str(direct["expected_direction"])
        matches = [value for value in zones.values() if isinstance(value, dict) and str(value.get("zone_id")) == zone_id]
        if len(matches) == 1 and isinstance(matches[0].get("expected_direction"), str):
            return str(matches[0]["expected_direction"])
        return None

    @staticmethod
    def _context(body: dict[str, Any], scenario_id: str | None = None) -> tuple[str, int] | None:
        scenario = scenario_id or str(body.get("scenario_id", ""))
        mode = learning_mode_from_scenario(scenario)
        setting = body.get("sequence_setting")
        process = process_from_tool_type(setting.get("tool_type")) if isinstance(setting, dict) else None
        return (process, mode) if process is not None and mode is not None else None

    @staticmethod
    def _visit_decisions(zones: list[str], flow: list[str]) -> list[str | None]:
        decisions: list[str | None] = []
        progress = -1
        seen: set[str] = set()
        for zone in zones:
            if zone not in flow:
                decisions.append("unexpected_zone")
                continue
            index = flow.index(zone)
            if zone in seen:
                decisions.append("zone_revisit")
            elif index == progress + 1:
                decisions.append(None)
                progress = index
            elif index > progress + 1:
                decisions.append("zone_skipped")
                progress = index
            else:
                decisions.append("zone_out_of_order")
            seen.add(zone)
        return decisions

    def evaluate(self, request: AnalyzeSequenceRequest, store: SequenceStore) -> SequenceDecision:
        bodies = store.session_bodies(request.session_id)
        context = self._context(request.model_dump(mode="json"))
        if context is None:
            return SequenceDecision(EngineFeedback(analysis_status="not_available", reason_code="sequence_reference_context_unavailable"), None)
        profile = self._profile(*context)
        if profile is None or not self._flow(profile):
            return SequenceDecision(EngineFeedback(analysis_status="not_available", reason_code="sequence_reference_unavailable"), None)
        visits: list[str] = []
        for body in bodies:
            zone = _zone_id(body)
            if zone is not None and (not visits or visits[-1] != zone):
                visits.append(zone)
        flow = self._flow(profile)
        order_reason = self._visit_decisions(visits, flow)[-1] if visits else None
        expected = self._expected(profile, request.zone_id, _map_num(request.model_dump(mode="json")))
        actual = observed_direction(request.model_dump(mode="json"))
        direction_error = expected is not None and actual is not None and expected != actual
        if order_reason is not None:
            missed = flow[max(0, flow.index(request.zone_id) - 1)] if request.zone_id in flow else None
            text = (
                f"권장 순서에서 이탈했습니다. {missed}번 구역을 먼저 확인하세요."
                if missed else "전문가 권장 Zone 순서를 확인하세요."
            )
            return SequenceDecision(
                EngineFeedback(analysis_status="available", reason_code=order_reason),
                text,
                "sequence_order_deviation",
            )
        if direction_error:
            return SequenceDecision(
                EngineFeedback(analysis_status="available", reason_code="direction_deviation"),
                f"{request.zone_id}번 구역은 {expected} 방향으로 도장하세요.",
                "direction_deviation",
            )
        next_zone = flow[flow.index(request.zone_id) + 1] if request.zone_id in flow and flow.index(request.zone_id) + 1 < len(flow) else None
        text = f"현재 순서가 적정합니다. 다음 권장 구역은 {next_zone}번입니다." if next_zone else "현재 순서가 적정합니다."
        return SequenceDecision(EngineFeedback(analysis_status="available", reason_code=None), text)

    def report_bodies(self, bodies: Iterable[dict[str, Any]], *, scenario_id: str) -> dict[str, Any]:
        items = list(bodies)
        context = self._context(items[0], scenario_id) if items else None
        profile = self._profile(*context) if context is not None else None
        flow = self._flow(profile) if profile is not None else []
        if profile is None or not flow:
            return {"analysis_status": "unavailable", "reason_code": "sequence_reference_unavailable", "algorithm_version": ALGORITHM_VERSION, "guideline": "해당 콘텐츠의 전문가 Zone 순서 기준이 없습니다."}

        visits: list[str] = []
        first_visits: list[str] = []
        visit_counts: Counter[str] = Counter()
        for body in items:
            zone = _zone_id(body)
            if zone is None:
                continue
            if not visits or visits[-1] != zone:
                visits.append(zone)
                visit_counts[zone] += 1
                if zone not in first_visits:
                    first_visits.append(zone)
        decisions = self._visit_decisions(visits, flow)
        deviations: Counter[str] = Counter(zone for zone, reason in zip(visits, decisions) if reason is not None)
        evaluated = len(visits)
        on_track = sum(reason is None for reason in decisions)

        direction_rows: dict[str, Counter[str]] = defaultdict(Counter)
        for body in items:
            zone = _zone_id(body)
            if zone is None:
                continue
            expected = self._expected(profile, zone, _map_num(body))
            actual = observed_direction(body)
            if expected is not None and actual is not None:
                direction_rows[zone]["evaluated"] += 1
                direction_rows[zone]["matched" if expected == actual else "error"] += 1
        direction_zone_summary: list[dict[str, Any]] = []
        for zone in flow:
            counts = direction_rows.get(zone)
            expected = self._expected(profile, zone)
            if not counts or expected is None:
                continue
            total = counts["evaluated"]
            direction_zone_summary.append({
                "zone_id": zone, "expected_direction": expected,
                "evaluated_sequence_count": total, "matched_sequence_count": counts["matched"],
                "direction_error_count": counts["error"],
                "direction_adherence_rate": round(counts["matched"] / total, 3),
            })
        direction_total = sum(row["evaluated_sequence_count"] for row in direction_zone_summary)
        direction_matched = sum(row["matched_sequence_count"] for row in direction_zone_summary)
        direction_rate = round(direction_matched / direction_total, 3) if direction_total else None
        direction_summary = None if not direction_total else {
            "evaluated_sequence_count": direction_total,
            "matched_sequence_count": direction_matched,
            "direction_error_count": direction_total - direction_matched,
            "direction_adherence_rate": direction_rate,
            "affected_zone_ids": [row["zone_id"] for row in direction_zone_summary if row["direction_error_count"]],
            "zone_summary": direction_zone_summary,
        }
        adherence = round(on_track / evaluated, 3) if evaluated else 0.0
        guideline = f"전문가 권장 순서 {'→'.join(flow)}를 기준으로 방문 순서 준수율은 {adherence * 100:.1f}%입니다."
        if direction_rate is not None:
            guideline += f" 방향 준수율은 {direction_rate * 100:.1f}%입니다."
        return {
            "analysis_status": "available", "reason_code": None,
            "algorithm_version": ALGORITHM_VERSION, "guideline": guideline,
            "recommended_flow": flow, "first_visit_flow": first_visits,
            "evaluated_visit_count": evaluated, "on_track_visit_count": on_track,
            "skipped_zone_count": len([zone for zone in flow if zone not in first_visits]),
            "adherence_rate": adherence, "deviation_count": evaluated - on_track,
            "direction_adherence_rate": direction_rate,
            "zone_summary": [{
                "zone_id": zone, "visit_count": visit_counts[zone],
                "revisit_count": max(0, visit_counts[zone] - 1),
                "deviation_count": deviations[zone],
            } for zone in dict.fromkeys(visits)],
            "direction_summary": direction_summary,
        }
