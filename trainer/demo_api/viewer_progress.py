"""누락·보완·다음 Zone 질문의 답 — 서명된 방문 순서와 최종 도막 게이트만 쓴다.

모델에 맡기면 칠한 Zone을 "빠졌다"고 지어냈다(2026-11-14 운영 q11). 뷰어에서 "다음 구역"은 재생 위치를
몰라 "확인할 수 없습니다"였다(q16). 여기서는 계산만 한다:

* 방문: 대화 자료의 `actual_order`(첫 방문 순서)와 `zone_first_visit_seconds`(재생 시각 기준 첫 방문).
* 도막: 서명된 판정 행의 `gate_status`(정상 범위 안 면적 95% + 평균 정상 범위, `coating_assessment`가 정한 값).
  최종 높이맵 하나뿐이라 재생 중간 시점의 완료 여부는 모른다.
* 다음 Zone: 전문가 전이(`recommend_next`) — 재생 위치까지 칠한 Zone은 뺀다.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from trainer.demo_api.coating_assessment import SIX_BAND_KEYS, resolve_band
from trainer.demo_api.pose_query import josa
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
    return {key: order[key] for key in (
        "zone_first_visit_seconds", "expert_transitions", "position_distribution", "reference_session_count",
        "best_reference_similarity", "position_similarity") if order.get(key) is not None}


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
_VIEWER_FOOTER = "전체 목록은 도막 탭에서 확인하세요. 세션이 끝났을 때의 도막 기준입니다."


def _num(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def normal_band(item: Mapping[str, Any]) -> tuple[float, float]:
    """판정 행(또는 target 블록)의 정상 범위. 기록에 범위가 없을 때만 목표에서 `resolve_band`로 구한다."""
    value = item.get("normal_band_um")
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(_num(v) for v in value):
        return float(value[0]), float(value[1])
    target = item.get("target_thickness_um")
    return resolve_band(float(target) if _num(target) else None)


def band_text(item: Mapping[str, Any]) -> str:
    low, high = normal_band(item)
    return f"{low:g}~{high:g}µm"


def below_percent(row: Mapping[str, Any]) -> float | None:
    """Zone 면적 가운데 하한 미만(미도장 포함) 비율(%). 6구간 앞 두 칸이 칠한 곳 중 하한 미만이다. 자료가 없으면 None."""
    counts, unpainted = row.get("six_band_counts"), row.get("unpainted_fraction")
    if not isinstance(counts, Mapping) or not _num(unpainted):
        return None
    values = [counts.get(key, 0) for key in SIX_BAND_KEYS]
    if not all(_num(v) for v in values):
        return None
    painted = sum(values)
    share = unpainted + ((1 - unpainted) * (values[0] + values[1]) / painted if painted else 0.0)
    return round(100.0 * share, 1)


def shortfall_side(row: Mapping[str, Any]) -> str | None:
    """정상 범위 밖 면적이 어느 쪽에 더 많은가: "under"(하한 미만) · "over"(상한 초과) · None(자료 없음·같음)."""
    below, over = below_percent(row), row.get("overpaint_percent")
    if below is None or not _num(over) or below == over:
        return None
    return "under" if below > over else "over"


def shortfall_text(row: Mapping[str, Any]) -> str:
    """완료 전 Zone의 범위 밖 면적. "보완(더 칠하기)"은 하한 미만이 상한 초과보다 많을 때만 말한다."""
    low, high = normal_band(row)
    below, over = below_percent(row), row.get("overpaint_percent")
    if not _num(over):
        return ""
    if below is None:
        return f"상한({high:g}µm)을 넘는 면적은 {over:g}%입니다."
    text = f"범위 밖 면적은 하한({low:g}µm) 미만(미도장 포함) {below:g}%, 상한({high:g}µm) 초과 {over:g}%"
    return text + {"under": "라 보완(더 칠하기)이 필요합니다.", "over": "로 주로 상한 초과입니다.", None: "입니다."}[shortfall_side(row)]


def _area_text(row: Mapping[str, Any], *, first: bool) -> str:
    """'Zone 3(정상 범위 안 0%)', 둘째부터 'Zone 8(1.9%)'. 등급 낱말·평균 두께는 적지 않는다
    (면적이 모자라 보완인데 "정상"이 붙으면 엇갈려 읽힌다)."""
    area = row.get("coverage_percent")
    if not isinstance(area, (int, float)):
        return f"Zone {row['zone_id']}"
    return f"Zone {row['zone_id']}({'정상 범위 안 ' if first else ''}{area:g}%)"


def _zone_list(zones: list[str]) -> str:
    shown = zones[:_LISTED]
    return "Zone " + "·".join(shown) + (f" 외 {len(zones) - len(shown)}곳" if len(zones) > len(shown) else "")


def gate_lines(rows: list[Mapping[str, Any]] | None, *, order: list[str], visited: set[str] | None = None,
               footer: str = _VIEWER_FOOTER, lead: str = "") -> tuple[list[str], dict[str, list[str]]]:
    """최종 도막 게이트 요약 줄과 Zone 목록. 보완 Zone 수, 정상 범위 안 면적이 가장 낮은 3곳, 과도막·완료 Zone.

    완료 전(needs_more)이라도 범위 밖 면적이 주로 상한 초과인 Zone은 보완(더 칠하기)으로 묶지 않는다(`shortfall_side`).

    `visited`가 있으면 방문한 Zone만 본다. `lead`는 개수 줄 앞에 붙일 말("9개 Zone을 모두 칠했지만 ")이다.
    """
    if rows is None:
        return ["도막 완료를 판정할 자료가 없어 보완이 필요한 Zone은 알 수 없습니다."], {}
    rank = {zone: index for index, zone in enumerate(order)}
    rows = sorted((row for row in rows if visited is None or str(row["zone_id"]) in visited),
                  key=lambda row: (rank.get(str(row["zone_id"]), len(rank)), str(row["zone_id"])))
    groups = {status: [row for row in rows if row["gate_status"] == status] for status in ("needs_more", "overpaint_risk", "complete")}
    sides = {id(row): shortfall_side(row) for row in groups["needs_more"]}
    touch, total = [row for row in groups["needs_more"] if sides[id(row)] == "under"], len(rows)
    over_band = [row for row in groups["needs_more"] if sides[id(row)] == "over"]
    unknown = [row for row in groups["needs_more"] if sides[id(row)] is None]
    lines = []
    if lead and touch and len(touch) == total:
        lines.append(f"{lead}칠했지만 {total}개 모두 보완이 필요합니다.")
    elif lead and touch:
        lines.append(f"{lead}칠했고 그중 {len(touch)}개는 보완이 필요합니다.")
    elif lead:
        lines.append(f"{lead}칠했고 보완이 필요한 Zone은 없습니다.")
    elif touch and visited is not None:
        lines.append(f"지금까지 칠한 Zone {total}개 가운데 {len(touch)}개는 보완이 필요합니다.")
    elif touch and len(touch) == total:
        lines.append(f"{total}개 Zone 모두 보완이 필요합니다.")
    elif touch:
        lines.append(f"{total}개 Zone 가운데 {len(touch)}개는 보완이 필요합니다.")
    if touch:
        lowest = sorted(touch, key=lambda row: float(row.get("coverage_percent") or 0.0))[:_LISTED]
        lines.append("가장 부족한 곳은 " + ", ".join(_area_text(row, first=index == 0) for index, row in enumerate(lowest)) + "입니다.")
    if over_band:
        lines.append(f"완료 전이지만 정상 범위를 벗어난 면적이 주로 상한 초과인 Zone은 {_zone_list([str(row['zone_id']) for row in over_band])}입니다.")
    if unknown:
        lines.append(f"정상 범위 안 면적이 95%에 못 미친 다른 Zone은 {_zone_list([str(row['zone_id']) for row in unknown])}입니다.")
    if groups["overpaint_risk"]:
        lines.append(f"과도막이라 더 칠하면 안 되는 Zone은 {_zone_list([str(row['zone_id']) for row in groups['overpaint_risk']])}입니다.")
    done = [str(row["zone_id"]) for row in groups["complete"]]
    if done:
        lines.append(f"완료한 Zone은 {_zone_list(done)}입니다.")
    lines.append(footer)
    return lines, {"touch_up_zone_ids": [str(row["zone_id"]) for row in touch],
                   "over_band_zone_ids": [str(row["zone_id"]) for row in over_band],
                   "overpaint_zone_ids": [str(row["zone_id"]) for row in groups["overpaint_risk"]],
                   "complete_zone_ids": done}


def status_answer(claims: Mapping[str, Any], coating: Mapping[str, Any] | None) -> tuple[str, dict[str, Any]]:
    """미방문 / 방문했으나 보완 필요 / 완료. 미방문 줄은 미방문 Zone이 있을 때만 쓴다."""
    reference, actual = _zones(claims.get("reference_order")), _zones(claims.get("actual_order"))
    rows = _gate_rows(coating)
    known = list(dict.fromkeys([*reference, *(str(row["zone_id"]) for row in rows or [])]))
    unvisited = [zone for zone in known if zone not in actual]
    lines = [f"아직 칠하지 않은 Zone은 {_zone_list(unvisited)}입니다."] if unvisited else []
    lead = f"{len(known)}개 Zone을 모두 " if known and not unvisited else ""
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


def _painted(zones: list[str]) -> str:
    """'Zone 9를', 'Zone 9와 8을', 'Zone 9·8·5를'(목적격까지)."""
    if len(zones) == 2:
        return josa(josa(f"Zone {zones[0]}", "과", "와") + f" {zones[1]}", "을", "를")
    return josa("Zone " + "·".join(zones), "을", "를")


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
    unknown_gate = "도막은 세션이 끝났을 때 값만 있어서 이 시점에 다 칠해졌는지는 알 수 없습니다."
    remaining = [zone for zone in reference if zone not in visited]
    if not reference:
        done = f"{where}까지 {_painted(visited)} 칠했습니다. " if visited else ""
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
    total = sum(n for (source, _), n in transitions.items() if source == current)
    # 전문가 전이 횟수가 같은 Zone이 여럿이면 함께 든다(권장 순서가 앞선 쪽부터, 최대 3곳). 하나만 고르면 근거 없이 갈린다.
    tied = sorted((target for (source, target), n in transitions.items() if source == current and n == count and target not in visited),
                  key=lambda zone: reference.index(zone) if zone in reference else len(reference))[:3] if count else []
    scope.update(next_zone_id=choice, basis="expert_transition" if count else "recommended_flow",
                 next_zone_candidates=tied if len(tied) > 1 else [choice])
    done = f"{where}까지 {_painted(visited)} 칠했습니다. " if visited else f"{where}까지 칠한 Zone이 없습니다. "
    if len(tied) > 1:
        names = [f"Zone {zone}" for zone in tied]
        options = ", ".join(names[:-2] + [josa(names[-2], "이나", "나") + " " + names[-1]])
        places = {2: "두 곳이", 3: "세 곳이"}[len(tied)]
        lines = [done + f"다음은 {josa(options, '을', '를')} 권합니다. "
                 f"전문가 기록에서 Zone {current} 다음으로 {places} 똑같이 많았습니다({total}회 중 각 {count}회)."]
    else:
        reason = (f"전문가 기록에서 Zone {current} 다음으로 많이 간 곳 가운데 아직 칠하지 않은 Zone입니다({total}회 중 {count}회)."
                  if count else "권장 순서에서 아직 칠하지 않은 첫 Zone입니다.")
        lines = [done + f"다음은 {josa(f'Zone {choice}', '을', '를')} 권합니다. {reason}"]
    if current in actual and actual.index(current) + 1 < len(actual) and actual[actual.index(current) + 1] not in visited:
        lines.append(f"이 기록에서는 실제로 {josa(f'Zone {actual[actual.index(current) + 1]}', '을', '를')} 다음에 처음 칠했습니다.")
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
