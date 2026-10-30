"""Compact, bounded context for session-aware Co-Pilot answers.

Raw simulator sequences can contain hundreds of 90/120 Hz frames and base64
heatmaps.  They stay in SQLite for the analysis engines; the language model is
given only stable session facts so latency and prompt size do not grow with the
training duration.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from typing import Any

from trainer.demo_api.telemetry_contracts import process_from_tool_type

_NO_ZONE = {"", "-1", "0", "none", "null", "nan"}
_METRICS = ("avg_speed_mms", "avg_distance_mm", "avg_thickness_um")


def _zone(body: dict[str, Any]) -> str | None:
    value = str(body.get("zone_id", "")).strip()
    return None if value.lower() in _NO_ZONE else value


def _number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return round(parsed, 3) if math.isfinite(parsed) else None


def _sequence_fact(body: dict[str, Any]) -> dict[str, Any]:
    result = body.get("sequence_result")
    result = result if isinstance(result, dict) else {}
    fact: dict[str, Any] = {
        "sequence_id": str(body.get("sequence_id", "")),
        "zone_id": _zone(body),
    }
    for key in _METRICS:
        value = _number(result.get(key))
        if value is not None:
            fact[key] = value
    return fact


def summarize_session(bodies: list[dict[str, Any]], *, recent_limit: int = 3) -> str:
    """Return deterministic compact JSON for every received sequence in a session."""
    if not bodies:
        return ""

    facts = [_sequence_fact(body) for body in bodies]
    visits: list[str] = []
    counts: Counter[str] = Counter()
    for fact in facts:
        zone = fact.get("zone_id")
        if not isinstance(zone, str):
            continue
        counts[zone] += 1
        if not visits or visits[-1] != zone:
            visits.append(zone)

    current_zone = next(
        (str(fact["zone_id"]) for fact in reversed(facts) if fact.get("zone_id") is not None),
        None,
    )
    averages: dict[str, float] = {}
    for key in _METRICS:
        values = [float(fact[key]) for fact in facts if key in fact]
        if values:
            averages[key] = round(sum(values) / len(values), 3)

    setting = bodies[-1].get("sequence_setting")
    setting = setting if isinstance(setting, dict) else {}
    process = process_from_tool_type(setting.get("tool_type"))
    summary = {
        "scenario_id": str(bodies[-1].get("scenario_id", "")) or None,
        "process": process,
        "sequence_count": len(bodies),
        "latest_sequence_id": str(bodies[-1].get("sequence_id", "")) or None,
        "current_zone": current_zone,
        "zone_visit_order": visits,
        "zone_sequence_counts": dict(counts),
        "session_averages": averages,
        "recent_sequences": facts[-max(1, min(int(recent_limit), 10)):],
    }
    return json.dumps(summary, ensure_ascii=False, separators=(",", ":"))


def build_query_instruction(
    prompt: str,
    *,
    client_context: str | None,
    session_summary: str,
    recent_turns: list[dict[str, str]],
    response_mode: str = "quick",
) -> str:
    """Add server-owned context without changing the external query contract."""
    context = (client_context or "").strip()
    parts = [
        "[사용자 질문]",
        prompt,
        "[응답 수준]",
        (
            "상세 분석: 현재 상황을 먼저 요약하고, 확인된 근거와 측정값, 원인, 권장 조치를 "
            "구분해 설명하세요. 현재 데이터에 없는 값은 추측하지 마세요."
            if response_mode == "detailed"
            else (
                "빠른 안내: 반드시 완결된 한국어 문장 2개 이내로 답하세요. "
                "표·목록·제목·배경 설명은 쓰지 말고, 첫 문장은 핵심 판단, "
                "두 번째 문장은 즉시 조치로 구성하세요."
            )
        ),
    ]
    if context or session_summary or recent_turns:
        parts.extend([
        "[답변 지침]",
        (
            "비정형 표현과 대명사는 아래 세션 및 최근 대화 문맥으로 해석하세요. "
            "질문에 먼저 직접 답하고, 제공되지 않은 측정값은 추측하지 마세요. "
            "avg_speed_mms의 단위는 mm/s, avg_distance_mm의 단위는 mm, "
            "avg_thickness_um의 단위는 µm입니다. 한국어로 자연스럽고 간결하게 답하세요."
        ),
        ])
    if context:
        parts.extend(("[클라이언트 현재 문맥 — 관찰 참고이며 검증된 점수·분석이 아님]", context[:2000]))
    if session_summary:
        parts.extend(("[현재 세션 누적 요약]", session_summary))
    if recent_turns:
        parts.append("[최근 대화]")
        for turn in recent_turns:
            parts.append(f"사용자: {turn['prompt'][:500]}")
            parts.append(f"Co-Pilot: {turn['answer'][:1000]}")
    return "\n".join(parts)
