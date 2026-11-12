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
    CoatingProgress,
    learning_mode_from_scenario,
    process_from_tool_type,
)

ALGORITHM_VERSION = "sequence-reference-2.0"
_NO_ZONE = {"", "-1", "0", "none", "null", "nan"}


def expert_reference_flows(profile: dict[str, Any]) -> list[list[str]]:
    """Expand the foreman's measured macro orders into one flow per recorded run.

    Older generated profiles prepend an ``expert_zone_orders`` entry aggregated
    from every worker.  It is deliberately excluded: evaluation must use only
    the measured expert runs retained in the remaining macro-order rows.
    """
    flows: list[list[str]] = []
    for row in profile.get("aggregate", {}).get("macro_orders") or []:
        if not isinstance(row, dict) or row.get("source") == "expert_zone_orders":
            continue
        flow = [
            str(item["zone_id"])
            for item in row.get("order") or []
            if isinstance(item, dict) and item.get("zone_id") is not None
        ]
        if not flow:
            continue
        try:
            count = max(1, int(row.get("session_count", 1)))
        except (TypeError, ValueError):
            count = 1
        flows.extend([flow] * count)
    return flows


def _lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_item in left:
        current = [0]
        for index, right_item in enumerate(right, start=1):
            current.append(previous[index - 1] + 1 if left_item == right_item else max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def _position_distribution(references: list[list[str]]) -> list[list[dict[str, Any]]]:
    rows: list[list[dict[str, Any]]] = []
    total = len(references)
    for index in range(max((len(flow) for flow in references), default=0)):
        counts = Counter(flow[index] for flow in references if index < len(flow))
        rows.append([
            {"zone_id": zone, "count": count, "probability": round(count / total, 3)}
            for zone, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        ])
    return rows


def score_visit_flow(actual: list[str], references: list[list[str]]) -> dict[str, Any]:
    """Score duration-independent first visits against all valid expert runs.

    Seventy percent rewards similarity to the closest real expert performance;
    thirty percent rewards statistically common Zone choices at each rank.
    """
    distributions = _position_distribution(references)
    if not actual or not references:
        return {
            "best_reference_similarity": 0.0,
            "position_similarity": 0.0,
            "adherence_rate": 0.0,
            "position_distribution": distributions,
        }
    best = max(
        _lcs_length(actual, reference) / max(len(actual), len(reference), 1)
        for reference in references
    )
    position_scores: list[float] = []
    for index, zone in enumerate(actual):
        row = distributions[index] if index < len(distributions) else []
        maximum = max((int(item["count"]) for item in row), default=0)
        count = next((int(item["count"]) for item in row if item["zone_id"] == zone), 0)
        position_scores.append(count / maximum if maximum else 0.0)
    position = sum(position_scores) / len(position_scores)
    return {
        "best_reference_similarity": round(best, 3),
        "position_similarity": round(position, 3),
        "adherence_rate": round(best * 0.7 + position * 0.3, 3),
        "position_distribution": distributions,
    }


def recommend_next(
    current: str | None, visited: set[str], transitions: dict[tuple[str, str], int], flow: list[str]
) -> str | None:
    """다음에 칠할 구역. 지금 구역에서 전문가가 가장 많이 간, 아직 안 칠한 구역(같으면 권장 순서에서 앞선 쪽).

    그런 전이가 없으면 권장 순서의 첫 미방문 구역. 순서 **판정**은 권장 순서로 하되, 훈련생이 다른 순서로 갔을 때의
    **추천**은 이 확률로 한다(2026-10-16 결정).
    """
    candidates = [
        (count, -flow.index(target) if target in flow else -len(flow), target)
        for (source, target), count in transitions.items()
        if source == current and target not in visited
    ]
    if candidates:
        return max(candidates)[2]
    return next((zone for zone in flow if zone not in visited), None)


@dataclass(frozen=True)
class SequenceDecision:
    feedback: EngineFeedback
    text_guide: str | None
    anomaly_type: str | None = None
    direction_guide: str | None = None
    directions: tuple[str, str] | None = None   # 방향 이탈의 (기대, 관측) 열쇠. 게이트 전 판정(decide)에만 있고 뷰어 알림 값으로 쓴다


def gate_sequence_decision(
    request: AnalyzeSequenceRequest, decision: SequenceDecision,
    progress: CoatingProgress | None,
) -> SequenceDecision:
    """Apply the same physical completion gate to HTTP and anomaly SSE text."""
    if request.process_type != "spray" or _zone_id(request.model_dump(mode="json")) is None:
        return decision
    direction = (getattr(decision, "direction_guide", None) or decision.text_guide
                 if decision.feedback.reason_code == "direction_deviation" else None)
    if (progress is not None and progress.available and progress.zone_id == request.zone_id
            and progress.status == "complete" and progress.recommendation == "move_next_zone"):
        text = progress.message
        if direction:
            text = f"{direction} {text}"
        return SequenceDecision(decision.feedback, text, decision.anomaly_type, direction)
    if progress is not None and progress.available:
        text = progress.message
    else:
        text = f"현재 Zone {request.zone_id}의 목표 도막 충족 여부를 확인할 수 없습니다. 구역 이동 전 도막과 공식 구역 데이터를 확인하세요."
    if direction:
        text = f"{direction} {text}"
    return SequenceDecision(decision.feedback, text, decision.anomaly_type, direction)


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
            if not isinstance(payload, dict):
                return None
            if not isinstance(payload.get("aggregate", {}).get("zones"), dict):
                return None
            return payload
        except (OSError, AttributeError, ValueError, TypeError, json.JSONDecodeError):
            return None

    def rescore_report(
        self, report: dict[str, Any], *, scenario_id: str, process: str = "spray"
    ) -> dict[str, Any]:
        """Refresh a stored report with today's multi-run expert reference."""
        mode = learning_mode_from_scenario(scenario_id)
        profile = self._profile(process, mode) if mode is not None else None
        references = expert_reference_flows(profile) if profile is not None else []
        actual = [str(zone) for zone in report.get("first_visit_flow") or []]
        if not references or not actual:
            return report
        score = score_visit_flow(actual, references)
        updated = dict(report)
        updated.update({
            "algorithm_version": ALGORITHM_VERSION,
            "recommended_flow": self._flow(profile),
            "reference_session_count": len(references),
            "reference_flows": references,
            "position_distribution": score["position_distribution"],
            "best_reference_similarity": score["best_reference_similarity"],
            "position_similarity": score["position_similarity"],
            "adherence_rate": score["adherence_rate"],
        })
        return updated

    @staticmethod
    def _flow(profile: dict[str, Any]) -> list[str]:
        flows = expert_reference_flows(profile)
        if not flows:
            return []
        counts = Counter(tuple(flow) for flow in flows)
        return list(max(counts, key=lambda flow: (counts[flow], -flows.index(list(flow)))))

    @staticmethod
    def _transitions(profile: dict[str, Any]) -> dict[tuple[str, str], int]:
        """Foreman-only transition counts derived from the measured runs."""
        output: Counter[tuple[str, str]] = Counter()
        for flow in expert_reference_flows(profile):
            output.update(zip(flow, flow[1:]))
        return dict(output)

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
    def _visit_decisions(zones: list[str], flow: list[str], references: list[list[str]] | None = None) -> list[str | None]:
        decisions: list[str | None] = []
        if references:
            seen: set[str] = set()
            distributions = _position_distribution(references)
            first_index = 0
            for zone in zones:
                if zone in seen:
                    decisions.append("zone_revisit")
                    continue
                valid = {str(item["zone_id"]) for item in distributions[first_index]} if first_index < len(distributions) else set()
                appears_later = any(zone in reference[first_index + 1:] for reference in references)
                decisions.append(None if zone in valid else ("zone_skipped" if appears_later else "zone_out_of_order"))
                seen.add(zone)
                first_index += 1
            return decisions
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

    def planned_flow(self, *, scenario_id: str, process: str = "spray") -> list[str]:
        """Reference work plan, independent of a trainee's realtime completion."""
        plan = self.reference_plan(scenario_id=scenario_id, process=process)
        return list(plan["flow"]) if plan is not None else []

    def reference_plan(self, *, scenario_id: str, process: str = "spray") -> dict[str, Any] | None:
        """Measured expert plan; this does not authorize realtime Zone movement."""
        mode = learning_mode_from_scenario(scenario_id)
        profile = self._profile(process, mode) if mode is not None else None
        flow = self._flow(profile) if profile is not None else []
        if not flow:
            return None
        references = expert_reference_flows(profile)
        return {"analysis_status": "available", "reference_basis": "measured_expert_runs",
                "scenario_id": scenario_id, "process": process, "flow": flow,
                "reference_session_count": len(references)}

    def evaluate(self, request: AnalyzeSequenceRequest, store: SequenceStore) -> SequenceDecision:
        return gate_sequence_decision(request, self.decide(request, store), None)

    def decide(self, request: AnalyzeSequenceRequest, store: SequenceStore) -> SequenceDecision:
        """게이트(도막 진행 문장) 전의 순서·방향 판정. 공개 경로는 `evaluate`로 이 판정에 게이트를 씌운다."""
        if _zone_id(request.model_dump(mode="json")) is None:
            return SequenceDecision(EngineFeedback(analysis_status="not_available", reason_code="zone_unresolved"), None)
        bodies = store.session_bodies_through(request.session_id, request.sequence_id)
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
        references = expert_reference_flows(profile)
        order_reason = self._visit_decisions(visits, flow, references)[-1] if visits else None
        expected = self._expected(profile, request.zone_id, _map_num(request.model_dump(mode="json")))
        actual = observed_direction(request.model_dump(mode="json"))
        direction_error = expected is not None and actual is not None and expected != actual
        if order_reason is not None:
            # 판정은 권장 순서, 추천은 지금 위치에서의 전문가 전이(recommend_next).
            next_zone = recommend_next(request.zone_id, set(visits), self._transitions(profile), flow)
            text = (
                f"권장 순서에서 이탈했습니다. 지금 위치에서는 다음으로 {next_zone}번 구역을 권합니다."
                if next_zone else "전문가 권장 Zone 순서를 확인하세요."
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
                directions=(str(expected), str(actual)),
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
        references = expert_reference_flows(profile)
        decisions = self._visit_decisions(visits, flow, references)
        if not visits:
            return {
                "analysis_status": "unavailable", "reason_code": "sequence_data_unavailable",
                "algorithm_version": ALGORITHM_VERSION,
                "guideline": "Zone이 판정된 시퀀스가 없어 순서와 방향을 평가할 수 없습니다.",
            }
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
        flow_score = score_visit_flow(first_visits, references)
        adherence = flow_score["adherence_rate"]
        guideline = (
            f"전형기 {len(references)}회 실측 순서를 기준으로 순서 유사도는 {adherence * 100:.1f}%입니다. "
            f"대표 흐름은 {'→'.join(flow)}입니다."
        )
        if direction_rate is not None:
            guideline += f" 방향 준수율은 {direction_rate * 100:.1f}%입니다."
        return {
            "analysis_status": "available", "reason_code": None,
            "algorithm_version": ALGORITHM_VERSION, "guideline": guideline,
            "recommended_flow": flow, "first_visit_flow": first_visits,
            "reference_session_count": len(references),
            "reference_flows": references,
            "position_distribution": flow_score["position_distribution"],
            "best_reference_similarity": flow_score["best_reference_similarity"],
            "position_similarity": flow_score["position_similarity"],
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
