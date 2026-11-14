"""누락·보완·다음 Zone 질문의 답 — 서명된 방문 순서와 최종 도막 게이트만 쓴다.

모델에 맡기면 칠한 Zone을 "빠졌다"고 지어냈다(2026-11-14 운영 q11). 뷰어에서 "다음 구역"은 재생 위치를
몰라 "확인할 수 없습니다"였다(q16). 여기서는 계산만 한다:

* 방문: 대화 자료의 `actual_order`(첫 방문 순서)와 `zone_first_visit_seconds`(재생 시각 기준 첫 방문).
* 도막: 서명된 목표 대비 판정 행의 `gate_status`(면적 95% + 저도막 아님, `coating_assessment`가 정한 값).
  최종 높이맵 하나뿐이라 재생 중간 시점의 완료 여부는 모른다.
* 다음 Zone: 전문가 전이(`recommend_next`) — 재생 위치까지 칠한 Zone은 뺀다.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from trainer.demo_api.copilot_intent import is_coating_advice_question, is_general_painting_knowledge
from trainer.demo_api.reference_order_query import is_current_next_question, is_knowledge_ask
from trainer.demo_api.sequence_analysis import recommend_next

_STATUS = re.compile(r"빼먹|빠뜨|빠진|누락|미수행|안\s*칠|못\s*칠|덜\s*칠|다시\s*칠|덧칠|재도장|보완|남은\s*(?:구역|zone|존|곳|데)", re.I)
_NO_ZONE = {"", "-1", "0", "none", "null", "nan"}


def first_visit_seconds(trace: Iterable[Mapping[str, Any]], *, offset: float = 0.0) -> dict[str, float]:
    """Zone마다 처음 분사가 닿은 재생 시각(초). `offset`은 CSV 첫 행과 재생 0초의 차이."""
    out: dict[str, float] = {}
    for item in trace:
        zone, seconds = item.get("zone_id"), item.get("t")
        if zone and str(zone) not in out and isinstance(seconds, (int, float)) and math.isfinite(seconds):
            out[str(zone)] = round(max(0.0, float(seconds) - offset), 3)
    return out


def transition_counts(flows: Iterable[list[str]]) -> list[dict[str, Any]]:
    """전문가 수행에서 Zone A 다음에 Zone B로 간 횟수. 서명되는 자료라 정렬한 목록으로 싣는다."""
    counts: Counter[tuple[str, str]] = Counter()
    for flow in flows:
        counts.update(zip(flow, flow[1:]))
    return [{"from": a, "to": b, "count": n} for (a, b), n in sorted(counts.items())]


def order_progress_facts(order: Mapping[str, Any]) -> dict[str, Any]:
    """순서 평가에서 대화 자료(서명됨)로 옮기는 값. 없으면 싣지 않는다(예전 응답과 같은 모양)."""
    return {key: order[key] for key in ("zone_first_visit_seconds", "expert_transitions") if order.get(key) is not None}


def progress_question(prompt: str) -> str | None:
    """"next"(다음 Zone) · "status"(누락·보완) · None."""
    if is_current_next_question(prompt):
        return "next"
    if (_STATUS.search(prompt) and not is_coating_advice_question(prompt) and not is_knowledge_ask(prompt)
            and not is_general_painting_knowledge(prompt)):
        return "status"
    return None


def _zones(values: object) -> list[str]:
    return [str(item) for item in values] if isinstance(values, list) else []


def _gate_rows(coating: Mapping[str, Any] | None) -> list[Mapping[str, Any]] | None:
    target = coating.get("target") if isinstance(coating, Mapping) else None
    if not isinstance(target, Mapping) or target.get("analysis_status") != "available":
        return None
    return [row for row in target.get("zones") or [] if isinstance(row, Mapping) and row.get("gate_status")]


_LISTED = 3  # 글로 적는 Zone 수. 나머지는 도막 탭에서 본다(2026-11-14 운영 답이 13곳을 다 늘어놓아 읽을 수 없었다).
_VIEWER_FOOTER = "전체 목록은 도막 탭에서 볼 수 있습니다(세션 끝 도막 기준)."


def _area_text(row: Mapping[str, Any]) -> str:
    """'Zone 3(목표 두께 이상 0%)'. 등급 낱말·평균 두께는 적지 않는다(면적이 모자라 보완인데 "정상"이 붙으면 엇갈려 읽힌다)."""
    area = row.get("coverage_percent")
    return f"Zone {row['zone_id']}(목표 두께 이상 {area:g}%)" if isinstance(area, (int, float)) else f"Zone {row['zone_id']}"


def _zone_list(zones: list[str]) -> str:
    shown = zones[:_LISTED]
    return "Zone " + "·".join(shown) + (f" 외 {len(zones) - len(shown)}곳" if len(zones) > len(shown) else "")


def gate_lines(rows: list[Mapping[str, Any]] | None, *, order: list[str], visited: set[str] | None = None,
               footer: str = _VIEWER_FOOTER, lead: str = "") -> tuple[list[str], dict[str, list[str]]]:
    """최종 도막 게이트 요약 줄과 Zone 목록. 보완 Zone 수, 목표 두께 이상 면적이 가장 낮은 3곳, 과도막·완료 Zone.

    `visited`가 있으면 방문한 Zone만 본다. `lead`는 개수 줄 앞에 붙일 말("9개 Zone을 모두 칠했지만 ")이다.
    """
    if rows is None:
        return ["도막 완료 판정 자료가 없어 보완이 필요한 Zone은 알 수 없습니다."], {}
    rank = {zone: index for index, zone in enumerate(order)}
    rows = sorted((row for row in rows if visited is None or str(row["zone_id"]) in visited),
                  key=lambda row: (rank.get(str(row["zone_id"]), len(rank)), str(row["zone_id"])))
    groups = {status: [row for row in rows if row["gate_status"] == status] for status in ("needs_more", "overpaint_risk", "complete")}
    touch, total = groups["needs_more"], len(rows)
    lines = []
    if touch and len(touch) == total and lead:
        lines.append(f"{lead}모두 보완이 필요합니다.")
    elif touch and len(touch) == total:
        lines.append(f"{total}개 Zone 모두 보완이 필요합니다.")
    elif touch:
        lines.append(f"{lead}칠한 Zone {total}개 중 {len(touch)}개는 보완이 필요합니다." if lead or visited is not None
                     else f"Zone {total}개 중 {len(touch)}개는 보완이 필요합니다.")
    elif lead:
        lines.append(f"{lead}보완이 필요한 Zone은 없습니다.")
    if touch:
        lowest = sorted(touch, key=lambda row: float(row.get("coverage_percent") or 0.0))[:_LISTED]
        lines.append("가장 부족한 곳: " + ", ".join(_area_text(row) for row in lowest) + ".")
    if groups["overpaint_risk"]:
        lines.append(f"과도막이라 더 칠하지 않아야 할 Zone: {_zone_list([str(row['zone_id']) for row in groups['overpaint_risk']])}.")
    done = [str(row["zone_id"]) for row in groups["complete"]]
    if done:
        lines.append(f"완료한 Zone: {_zone_list(done)}.")
    lines.append(footer)
    return lines, {"touch_up_zone_ids": [str(row["zone_id"]) for row in touch],
                   "overpaint_zone_ids": [str(row["zone_id"]) for row in groups["overpaint_risk"]],
                   "complete_zone_ids": done}


def status_answer(claims: Mapping[str, Any], coating: Mapping[str, Any] | None) -> tuple[str, dict[str, Any]]:
    """미방문 / 방문했으나 보완 필요 / 완료. 미방문 줄은 미방문 Zone이 있을 때만 쓴다."""
    reference, actual = _zones(claims.get("reference_order")), _zones(claims.get("actual_order"))
    rows = _gate_rows(coating)
    known = list(dict.fromkeys([*reference, *(str(row["zone_id"]) for row in rows or [])]))
    unvisited = [zone for zone in known if zone not in actual]
    lines = [f"아직 칠하지 않은 Zone: {_zone_list(unvisited)}."] if unvisited else []
    lead = f"{len(known)}개 Zone을 모두 칠했지만 " if known and not unvisited else ""
    gate, groups = gate_lines(rows, order=reference or actual, visited=set(actual), lead=lead)
    if not known and not rows:
        lines.append("권장 기준 Zone 자료가 없어 칠하지 않은 Zone은 판단할 수 없습니다.")
    return "\n".join([*lines, *gate]), {"scope": "viewer_progress", "analysis_status": "available",
                                         "unvisited_zone_ids": unvisited, "visited_zone_ids": actual, **groups}


def _visited_until(claims: Mapping[str, Any], current: str | None, seconds: float | None) -> list[str]:
    """재생 위치까지 칠한 Zone(첫 방문 순서). 첫 방문 시각이 없으면 첫 방문 순서에서 지금 Zone까지."""
    actual = _zones(claims.get("actual_order"))
    times = claims.get("zone_first_visit_seconds")
    if isinstance(times, Mapping) and times and seconds is not None:
        return [zone for zone in actual if isinstance(times.get(zone), (int, float)) and times[zone] <= seconds + 1e-6]
    return actual[:actual.index(current) + 1] if current in actual else []


def next_answer(claims: Mapping[str, Any], coating: Mapping[str, Any] | None, playback: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    reference, actual = _zones(claims.get("reference_order")), _zones(claims.get("actual_order"))
    current = str(playback["zone_id"]) if playback.get("zone_id") is not None else None
    current = None if current is None or current.strip().lower() in _NO_ZONE else current  # "-1"은 Zone 사이 이동 중
    seconds = playback.get("playhead_seconds")
    seconds = float(seconds) if isinstance(seconds, (int, float)) and not isinstance(seconds, bool) and math.isfinite(seconds) else None
    visited = _visited_until(claims, current, seconds)
    if current and current not in visited:
        visited.append(current)  # 첫 착탄 시각 직전이라도 재생 위치가 있는 Zone은 칠하는 중이다
    place = f"Zone {current}" if current else "이동 중"
    where = f"재생 위치({place}, {seconds:.1f}초)" if seconds is not None else f"재생 위치({place})"
    scope: dict[str, Any] = {"scope": "viewer_progress", "analysis_status": "available", "current_zone_id": current,
                             "visited_zone_ids": visited, "next_zone_id": None, "basis": None}
    unknown_gate = "재생 시점의 도막 완료 여부는 알 수 없습니다(세션 끝 도막만 있습니다)."
    remaining = [zone for zone in reference if zone not in visited]
    if not reference:
        done = f"{where}까지 칠한 Zone은 {'·'.join(visited)}입니다. " if visited else ""
        return done + "권장 순서 기준이 없어 다음 Zone을 정할 수 없습니다. " + unknown_gate, scope
    if not remaining:
        gate, groups = gate_lines(_gate_rows(coating), order=reference or actual)
        scope.update(groups)
        head = f"{where}까지 권장 Zone을 모두 한 번 이상 칠했습니다. {unknown_gate}"
        return "\n".join([head, *gate]), scope
    transitions = {(str(item["from"]), str(item["to"])): int(item["count"])
                   for item in claims.get("expert_transitions") or [] if isinstance(item, Mapping)}
    choice = recommend_next(current, set(visited), transitions, reference)
    count = transitions.get((str(current), str(choice)), 0)
    scope.update(next_zone_id=choice, basis="expert_transition" if count else "recommended_flow")
    total = sum(n for (source, _), n in transitions.items() if source == current)
    reason = (f"전문가 기록에서 Zone {current} 다음에 간 Zone 가운데 아직 칠하지 않은 곳으로 가장 많았습니다({total}회 중 {count}회)."
              if count else "권장 순서에서 아직 칠하지 않은 첫 Zone입니다.")
    done = f"{where}까지 칠한 Zone은 {'·'.join(visited)}입니다. " if visited else f"{where}까지 칠한 Zone이 없습니다. "
    lines = [done + f"다음은 Zone {choice}를 권장합니다. {reason}"]
    if current in actual and actual.index(current) + 1 < len(actual) and actual[actual.index(current) + 1] not in visited:
        lines.append(f"이 기록에서 실제로 다음에 처음 칠한 곳은 Zone {actual[actual.index(current) + 1]}입니다.")
    lines.append(unknown_gate)
    return "\n".join(lines), scope


def viewer_progress_answer(value: str | None, prompt: str, kind: str) -> tuple[str, dict[str, Any]] | None:
    """서명된 뷰어 대화 자료가 있을 때만 답한다. 없으면 None(기존 경로)."""
    from trainer.demo_api.coating_query import parse_viewer_coating_context
    from trainer.demo_api.dialogue import viewer_claims
    found = viewer_claims(value)
    if found is None or not _zones(found[1].get("actual_order")):
        return None
    payload, claims = found
    coating = parse_viewer_coating_context(value, "knowledge_search")
    if kind == "status":
        return status_answer(claims, coating)
    playback = payload.get("playback_context") if isinstance(payload.get("playback_context"), Mapping) else {}
    return next_answer(claims, coating, playback)
