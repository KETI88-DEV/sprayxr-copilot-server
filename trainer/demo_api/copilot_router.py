"""POST /api/v1/copilot/query (v2.5).

``task_type`` picks the path (plan D3); the semantic router is not consulted.

* ``knowledge_search``: the live pipeline singleton shared with ``/api/chat``
  runs ``run(force_route="fast")`` in a worker thread inside the shared
  generation slot. ``use_rag='N'`` skips retrieval and returns no citations.
  ``answer`` is the parsed prose alone and ``citations`` is what it cites (a
  subset of this request's retrieved chunks), matching the §9.2 example: the
  ``[답변]``/``[근거]`` scaffold the model is trained to emit is stripped at this
  boundary, so no document id reaches ``answer`` at all. ``degraded`` is true when
  no full answer was produced or generation hit its deadline.
* ``pose_recomm`` (명세서 p.9 "전문가 작업 자세 유사도 비교"): 이미 올라간 시퀀스를
  채점한다. 대상은 ``context``(예: ``SEQ-012``)가 가리키는 시퀀스이고, 없으면 그
  ``session_id``의 가장 최근 시퀀스다. 채점은 MCP 자세 엔진에 위임하고 결과는
  ``pose_context``에 싣는다. **자세는 여기서 채점한다** — ``telemetry/sequence``는
  시퀀스를 데이터베이스에 올리는 일만 한다(p.4).
* ``trajectory_recomm`` / ``error_analysis`` (plan D2): HTTP 200 SUCCESS with a short
  "준비 중" answer and ``context_status="not_available"``. 엔진이 없다.

Every response uses the v2.5 wrapper. Request validation failures under
``/api/v1/`` are wrapped as ERROR 422; other routes keep FastAPI's format.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Final

from anyio import to_thread
from fastapi import APIRouter, Body, FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from trainer.demo_api.copilot_contracts import CopilotQueryData, CopilotQueryRequest, CopilotResponse
from trainer.demo_api.copilot_intent import classify_question_intent, fallback_intent, preset_intent
from trainer.demo_api.copilot_live_analysis import live_analysis
from trainer.demo_api.copilot_session_context import build_query_instruction, summarize_session, viewer_definition_context
from trainer.demo_api.pose_query import (
    answer_order_question,
    answer_pose_question,
    asks_about_zones,
    score_pose,
    requested_zone,
    zone_unavailable,
)
from trainer.demo_api.query_language import TranslationUnavailableError, translate_text
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.sequence_analysis import SequenceAnalysisEngine
from trainer.demo_api.v1_store import V1Store
from trainer.demo_api.query_runtime import generation_slot_for, select_query_pipeline, translation_pipeline_for
from trainer.scripts.generation_slot import GENERATION_SLOT  # Legacy slot identity for unconfigured clients.
from trainer.scripts.live_response_contract import (
    FAILED_REPAIR_PATHS,
    parse_knowledge_answer,
    retrieved_chunk_ids,
)
from trainer.scripts.openai_backend import OpenAIBackendUnavailableError

if TYPE_CHECKING:
    from trainer.demo_api.chat_service import _Pipeline

logger = logging.getLogger(__name__)

COPILOT_QUERY_PATH: Final[str] = "/api/v1/copilot/query"
_API_V1_PREFIX: Final[str] = "/api/v1/"
_SUCCESS_MESSAGE: Final[str] = "Copilot query completed"
_MOTION_VIEWER_CONTEXT_PREFIX: Final[str] = "motion_viewer_analysis:"

# Answers for task types whose MCP tools are not finished (plan D2).
PENDING_TASK_ANSWERS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "trajectory_recomm": "이동 궤적·도장 순서 추천 기능은 준비 중입니다. 도장 지식 질문에는 지금 답변할 수 있습니다.",
        "error_analysis": "도막 분포 이상 원인 분석 기능은 준비 중입니다. 도장 지식 질문에는 지금 답변할 수 있습니다.",
    }
)

# 자세 채점이 대상을 찾지 못했을 때의 답변. 왜 못 했는지를 말한다 — 점수를 지어내지 않는다.
_POSE_REASON_ANSWERS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "session_id_missing": "어느 세션의 자세를 비교할지 알 수 없습니다. session_id를 함께 보내주세요.",
        "no_sequence_for_session": "이 세션에 올라온 분사 시퀀스가 없습니다. 먼저 시퀀스를 전송해 주세요.",
        "sequence_not_found": "가리키신 시퀀스를 찾지 못했습니다. context의 시퀀스 식별자를 확인해 주세요.",
        "engine_unavailable": "자세 분석 엔진에 연결할 수 없습니다.",
    }
)
_POSE_FALLBACK_ANSWER: Final[str] = "이 시퀀스로는 자세를 판정할 수 없습니다."

PipelineProvider = Callable[[], "_Pipeline"]


class _InvalidViewerPlayback(ValueError):
    """A signed evaluation was accompanied by an invalid playback selector."""


def _validated_viewer_playback(value: object, analysis: Mapping[str, object], *, task_type: str) -> dict[str, object]:
    # Playback is client state, not a measurement. Only allow it to select a
    # map/Zone already authenticated by the server's evaluation signature.
    allowed = {"playhead_seconds", "source_index", "map_num", "zone_id"}
    if not isinstance(value, dict) or set(value) - allowed:
        raise _InvalidViewerPlayback
    seconds = value.get("playhead_seconds")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise _InvalidViewerPlayback
    try:
        finite_seconds = math.isfinite(seconds)
    except OverflowError:
        raise _InvalidViewerPlayback from None
    if not finite_seconds or seconds < 0:
        raise _InvalidViewerPlayback
    source_index = value.get("source_index")
    if source_index is not None and (isinstance(source_index, bool) or not isinstance(source_index, int) or source_index < 0):
        raise _InvalidViewerPlayback
    map_num, zone_id = value.get("map_num"), value.get("zone_id")
    result: dict[str, object] = {"playhead_seconds": seconds, "map_num": None, "zone_id": None}
    if source_index is not None:
        result["source_index"] = source_index
    if map_num is None and zone_id is None:
        return result
    if any(isinstance(item, bool) or not isinstance(item, (str, int)) for item in (map_num, zone_id)):
        raise _InvalidViewerPlayback
    map_text, zone_text = str(map_num), str(zone_id)
    if not map_text.isdigit() or not zone_text.isdigit() or int(zone_text) <= 0:
        raise _InvalidViewerPlayback
    key = f"{map_text}:{zone_text}"
    if task_type == "pose_recomm":
        zones = analysis.get("zones")
        row = zones.get(key) if isinstance(zones, Mapping) else None
        if not isinstance(row, Mapping) or str(row.get("zone_id")) != zone_text or str(row.get("map_num", map_text)) != map_text:
            raise _InvalidViewerPlayback
    elif task_type == "trajectory_recomm":
        keys = analysis.get("zone_keys")
        if not isinstance(keys, list) or key not in keys:
            raise _InvalidViewerPlayback
    else:
        raise _InvalidViewerPlayback
    result.update(map_num=map_text, zone_id=zone_text)
    return result


def _parse_motion_viewer_context(value: str | None, *, task_type: str) -> dict[str, object] | None:
    """Accept viewer measurements only when signed by the analysis server."""
    if not value or not value.startswith(_MOTION_VIEWER_CONTEXT_PREFIX):
        return None
    try:
        payload = json.loads(value[len(_MOTION_VIEWER_CONTEXT_PREFIX):])
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    analysis_context = payload.get("analysis_context")
    if not isinstance(analysis_context, dict):
        return None
    signature = payload.get("signature")
    if not isinstance(signature, str):
        return None
    # Transport-only record objects are not part of the signed analysis.
    analysis_context = {key: item for key, item in analysis_context.items() if key != "record"}
    from trainer.demo_api.viewer_context import verify_context
    if not verify_context(task_type, analysis_context, signature):
        return None
    if "playback_context" in payload:
        analysis_context["playback_context"] = _validated_viewer_playback(
            payload["playback_context"], analysis_context, task_type=task_type
        )
    return analysis_context


def _asks_current_viewer_pose(prompt: str) -> bool:
    # A current playback selection must not change an explicitly global query.
    if re.search(r"전체|전반|종합|평균|모든|최악|가장.*(?:zone|구역)|(?:zone|구역).*가장", prompt, re.I):
        return False
    return bool(re.search(r"지금|현재\s*(?:자세|동작|zone|구역)|이\s*(?:zone|구역)|current\s+(?:pose|zone)|this\s+zone", prompt, re.I))


def _current_viewer_pose_context(prompt: str, context: dict[str, object]) -> dict[str, object]:
    if "playback_context" not in context or not _asks_current_viewer_pose(prompt):
        return context
    playback = context["playback_context"]
    assert isinstance(playback, dict)  # normalized only after verifying the evaluation
    explicit_zone = requested_zone(prompt, None)
    if explicit_zone is not None and explicit_zone != playback.get("zone_id"):
        return context
    zones = context.get("zones")
    key = f"{playback.get('map_num')}:{playback.get('zone_id')}"
    row = zones.get(key) if isinstance(zones, Mapping) else None
    scored = isinstance(row, Mapping) and isinstance(row.get("similarity_percent"), (int, float))
    return {
        "status": "available" if context.get("status") == "available" and scored else "not_available",
        "scope": "current_zone", "playback_context": playback,
        "map_num": playback.get("map_num"), "zone_id": playback.get("zone_id"),
        "similarity_percent": row.get("similarity_percent") if scored else None,
        "zones": {key: row} if scored else {},
        "corrections": (row.get("corrections") or []) if scored else [],
    }


def _viewer_answer(task_type: str, prompt: str, context: Mapping[str, object], intent: str | None) -> str:
    """Build the factual answer from signed measures, never client prose.

    뷰어의 `similarity_percent`는 도구 각도·몸통 기울기·칠하는 방식·도막 두께·이격거리를 묶은 **작업 유사도**다.
    "자세 유사도"는 전문가 자세(팔 관절 엔진)와의 유사도에만 쓴다 — 같은 말로 다른 숫자를 내지 않는다.
    """
    selected_intent = intent if intent and intent != "knowledge" else fallback_intent(task_type, prompt)
    if task_type == "trajectory_recomm":
        flow = context.get("recommended_flow")
        flow = flow if isinstance(flow, list) else []
        if not flow:
            return "현재 재생 기록에서는 전문가 Zone 순서 기준을 계산할 수 없습니다."
        if selected_intent == "order_direction":
            direction = context.get("direction_summary")
            classification = direction.get("classification") if isinstance(direction, Mapping) else None
            return f"현재 기록의 주요 도포 방향은 {classification}입니다." if classification else "현재 기록에서는 주요 도포 방향을 계산할 수 없습니다."
        if selected_intent == "order_adherence":
            adherence = _percent_text(context.get("adherence_rate"))
            if adherence is None:
                return "현재 기록의 전문가 순서 유사도를 계산할 수 없습니다."
            answer = f"현재 기록의 전문가 순서 유사도는 {adherence}입니다."
            closest = _percent_text(context.get("best_reference_similarity"))
            positional = _percent_text(context.get("position_similarity"))
            if closest is not None and positional is not None:
                answer += f" 가장 가까운 전문가 수행과 {closest}, 순번별 통계와 {positional} 유사합니다."
            return answer
        # No visit-count or on-track-count is inferred from a blended score.
        report = dict(context)
        report.update({"analysis_status": "available", "first_visit_flow": context.get("selected_flow") or []})
        return answer_order_question((None, {"sequence_report": report}), prompt=prompt, intent=selected_intent).answer

    zones = context.get("zones")
    rows = list(zones.values()) if isinstance(zones, Mapping) else zones if isinstance(zones, list) else []
    rows = [row for row in rows if isinstance(row, Mapping) and isinstance(row.get("similarity_percent"), (int, float))]
    if context.get("scope") == "current_zone":
        if context.get("status") != "available" or not rows:
            return "질문 시점의 Zone 작업 비교값이 없습니다. 재생 위치를 확인한 뒤 다시 질문해 주세요."
        zone = rows[0]
        corrections = zone.get("corrections")
        guidance = " ".join(item for item in corrections if isinstance(item, str)) if isinstance(corrections, list) else ""
        if re.search(r"오른|right|왼|left", prompt, re.I):
            side = "오른" if re.search(r"오른|right", prompt, re.I) else "왼"
            matching = [item for item in corrections if isinstance(item, str) and side in item] if isinstance(corrections, list) else []
            guidance = " ".join(matching) if matching else f"이 Zone에는 {side}손에 특정된 교정 항목이 없습니다."
        return (f"현재 재생 구간(Map {context.get('map_num')} · Zone {zone.get('zone_id')})의 작업 유사도는 "
                f"{_percent_text(zone.get('similarity_percent'), already_percent=True)}입니다. "
                + (guidance or "별도의 작업 개선 항목은 없습니다."))
    if context.get("status") != "available" or not rows:
        return "현재 재생 기록에서는 작업 비교값을 계산할 수 없습니다."
    target = requested_zone(prompt, None)
    if target is not None:
        zone = next((row for row in rows if str(row.get("zone_id")) == target), None)
        if zone is None:
            return f"현재 재생 기록에는 Zone {target}의 작업 비교값이 없습니다."
        corrections = zone.get("corrections")
        guidance = " ".join(str(item) for item in corrections if isinstance(item, str)) if isinstance(corrections, list) else ""
        return f"Zone {target} 작업 유사도는 {_percent_text(zone.get('similarity_percent'), already_percent=True)}입니다." + (f" {guidance}" if guidance else " 별도의 작업 개선 항목은 없습니다.")
    if selected_intent in {"pose_right", "pose_left"}:
        side = "오른손" if selected_intent == "pose_right" else "왼손"
        corrections = context.get("corrections")
        messages = [str(item.get("message") or item.get("label") or "") if isinstance(item, Mapping) else str(item)
                    for item in corrections] if isinstance(corrections, list) else []
        matching = [message for message in messages if side in message or ("오른" if side == "오른손" else "왼") in message]
        return f"{side} 개선 안내: {' '.join(matching)}" if matching else f"현재 기록에서 {side}에 특정된 개선 항목은 없습니다. 전체 작업 유사도는 {_percent_text(context.get('similarity_percent'), already_percent=True)}입니다."
    worst = min(rows, key=lambda row: float(row["similarity_percent"]))
    if selected_intent == "pose_worst_zone" or any(word in prompt for word in ("문제", "최악", "낮", "어느", "어디")):
        corrections = worst.get("corrections")
        guidance = " ".join(str(item) for item in corrections if isinstance(item, str)) if isinstance(corrections, list) else ""
        return f"현재 기록에서 작업 유사도가 가장 낮은 구역은 Zone {worst.get('zone_id')}({_percent_text(worst.get('similarity_percent'), already_percent=True)})입니다." + (f" {guidance}" if guidance else "")
    return (f"현재 재생 기록의 전체 작업 유사도는 {_percent_text(context.get('similarity_percent'), already_percent=True)}입니다. "
            f"가장 낮은 구역은 Zone {worst.get('zone_id')}({_percent_text(worst.get('similarity_percent'), already_percent=True)})입니다.")


def _percent_text(value: object, *, already_percent: bool = False) -> str | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if 0 <= number <= 1 and not already_percent:
        number *= 100
    return f"{round(number)}%"


def _flow_text(value: object) -> str | None:
    if not isinstance(value, list) or not value:
        return None
    return " → ".join(str(item) for item in value)


def _stable_order_prefix(
    recommended_value: object, distribution_value: object, *, threshold: float = 0.6
) -> tuple[list[str], float | None]:
    if not isinstance(recommended_value, list) or not isinstance(distribution_value, list):
        return [], None
    stable: list[str] = []
    probabilities: list[float] = []
    for expected, row in zip(recommended_value, distribution_value):
        if not isinstance(row, list) or not row or not isinstance(row[0], dict):
            break
        top = row[0]
        try:
            probability = float(top.get("probability", 0))
        except (TypeError, ValueError):
            break
        if str(top.get("zone_id")) != str(expected) or probability < threshold:
            break
        stable.append(str(expected))
        probabilities.append(probability)
    return stable, min(probabilities) if probabilities else None


def _structured_detail_fallback(
    task_type: str,
    base_answer: str,
    analysis_context: dict[str, object],
    *,
    interpretation: str | None = None,
) -> str:
    """Build a standalone analysis, not a quick answer with text appended."""
    if task_type == "trajectory_recomm":
        recommended_value = analysis_context.get("recommended_flow")
        recommended = _flow_text(recommended_value)
        selected = _flow_text(
            analysis_context.get("selected_flow") or analysis_context.get("first_visit_flow")
        )
        stable_prefix, stable_confidence = _stable_order_prefix(
            recommended_value, analysis_context.get("position_distribution")
        )
        adherence_value = analysis_context.get("adherence_rate")
        adherence = _percent_text(adherence_value)
        try:
            adherence_number = float(adherence_value)  # type: ignore[arg-type]
            if adherence_number <= 1:
                adherence_number *= 100
        except (TypeError, ValueError):
            adherence_number = -1
        tendencies: list[str] = []
        if len(stable_prefix) >= 2 and stable_confidence is not None:
            tendencies.append(
                f"전문가 데이터에서는 Zone {' → '.join(stable_prefix)} 구간이 가장 뚜렷했습니다. "
                f"전문가 작업의 약 {stable_confidence * 100:.0f}%가 이 시작 흐름을 따랐습니다."
            )
        elif stable_prefix and stable_confidence is not None:
            tendencies.append(
                f"Zone {stable_prefix[0]}에서 시작하는 경향이 약 {stable_confidence * 100:.0f}%로 가장 높았습니다."
            )
        if isinstance(recommended_value, list) and len(stable_prefix) < len(recommended_value):
            start_at = len(stable_prefix) + 1
            tendencies.append(
                f"{start_at}번째 Zone 이후에는 작업 회차에 따라 순서 차이가 나타납니다. "
                "이 구간은 하나의 순서만 정답으로 판단하지 않고 대표 경로와의 연결성을 함께 평가합니다."
            )
        # The deterministic sections below already contain every verified
        # order statistic.  An LLM paraphrase here often repeats the quick
        # answer verbatim, making detailed mode feel like quick mode with an
        # extra paragraph.  Keep trajectory detail concise and non-duplicated.

        comparison: list[str] = []
        if selected:
            comparison.append(f"실제 수행: {selected}")
        if adherence:
            comparison.append(f"순서 준수율은 {adherence}입니다.")
        metrics = (
            ("가장 가까운 전문가 수행과의 유사도", "는", "best_reference_similarity"),
            ("순번별 통계 유사도", "는", "position_similarity"),
        )
        for label, particle, key in metrics:
            value = _percent_text(analysis_context.get(key))
            if value is not None:
                comparison.append(f"{label}{particle} {value}입니다.")
        direction = analysis_context.get("direction_summary")
        if isinstance(direction, dict) and direction.get("classification"):
            direction_label = {
                "horizontal": "수평",
                "vertical": "수직",
                "diagonal": "대각선",
                "irregular": "비정형",
            }.get(str(direction["classification"]), str(direction["classification"]))
            comparison.append(f"주요 도포 방향은 {direction_label}입니다.")
        segment_lines: list[str] = []
        if stable_prefix:
            confidence_text = f" ({stable_confidence * 100:.0f}%)" if stable_confidence is not None else ""
            segment_lines.append(f"- 우선 적용 구간: Zone {' → '.join(stable_prefix)}{confidence_text}")
        if isinstance(recommended_value, list) and len(stable_prefix) < len(recommended_value):
            flexible = " → ".join(str(zone) for zone in recommended_value[len(stable_prefix):])
            segment_lines.append(f"- 유동 구간: Zone {flexible}\n  작업 회차별 차이를 허용하는 대표 경로입니다.")
        if comparison:
            segment_lines.append("- 현재 기록 비교\n  " + "\n  ".join(comparison))
        action = (
            "현재 순서를 유지하되 각 Zone 내부의 속도·이격거리와 도포 패턴을 함께 점검하세요."
            if recommended and selected and recommended == selected
            else (
                f"Zone {stable_prefix[0]}에서 시작한 뒤 Zone {stable_prefix[1]}로 이동하세요. "
                "이후에는 대표 경로를 참고하되 현재 위치와 완료한 Zone을 고려해 다음 Zone을 선택하세요."
                if len(stable_prefix) >= 2 else
                f"{recommended or '표시된 권장 순서'}를 참고하고 Zone 전환 직전에 다음 Zone을 확인하세요."
            )
        )
        return (
            "[권장 Zone 순서]\n" + (f"Zone {recommended}" if recommended else "권장 순서를 계산할 수 없습니다.")
            + "\n\n[전문가 작업 경향]\n" + ("\n\n".join(tendencies) or "전문가 작업 경향을 계산할 수 없습니다.")
            + "\n\n[구간별 판단]\n" + ("\n".join(segment_lines) or "구간별 판단 자료가 없습니다.")
            + ("\n\n[상황 해석]\n" + interpretation if interpretation else "")
            + "\n\n[권장 조치]\n" + action
        )
    elif task_type == "pose_recomm":
        similarity = _percent_text(analysis_context.get("similarity_score", analysis_context.get("similarity_percent")), already_percent="similarity_score" not in analysis_context)
        current_zone = analysis_context.get("scope") == "current_zone"
        scope_label = f"현재 재생 구간(Map {analysis_context.get('map_num')} · Zone {analysis_context.get('zone_id')})"
        # 엔진(팔 관절) 점수 `similarity_score`만 자세 유사도다. 뷰어의 `similarity_percent`는 작업 유사도.
        engine = "similarity_score" in analysis_context
        measure = "자세" if engine else "작업"
        comparison = [f"{scope_label if current_zone else '전체'} {measure} 유사도는 {similarity}입니다."] if similarity else []
        zones = analysis_context.get("zones")
        zone_rows = list(zones.values()) if isinstance(zones, dict) else zones
        worst_text = None
        if isinstance(zone_rows, list) and not current_zone:
            candidates = [row for row in zone_rows if isinstance(row, dict)]
            scored = [
                (float(row.get("similarity_percent", 100)), row)
                for row in candidates
                if isinstance(row.get("similarity_percent"), (int, float))
            ]
            if scored:
                _, worst = min(scored, key=lambda item: item[0])
                score = _percent_text(worst.get("similarity_percent"), already_percent=True)
                worst_text = f"가장 낮은 구역은 Zone {worst.get('zone_id')}({score})입니다."
                comparison.append(worst_text)
        corrections = analysis_context.get("corrections")
        messages: list[str] = []
        if isinstance(corrections, list):
            messages = [
                str(item.get("message") or item.get("label") or item.get("joint") or "") if isinstance(item, dict) else str(item)
                for item in corrections[:3]
            ]
            messages = [message for message in messages if message and message != "None"]
        verdict = (f"{scope_label}의 전문가 {measure} 평가를 기준으로 확인했습니다."
                   if current_zone else f"전문가 {'자세와의' if engine else '작업과의'} 차이를 Zone별로 확인했습니다.")
        if interpretation is None:
            interpretation = worst_text or base_answer
        action = " ".join(messages) if messages else (
            "다음 분사에서도 같은 조건으로 자세를 비교하고, 관절별 교정이 제시되면 해당 위치를 우선 확인하세요."
            if engine else "다음 분사에서도 같은 조건으로 작업을 비교하고, 항목별 교정이 제시되면 해당 항목을 우선 확인하세요."
        )
        return (
            "[상세 분석]\n[종합 판단]\n" + verdict
            + "\n\n[정량 비교]\n" + ("\n".join(comparison) or f"비교 가능한 {measure} 지표가 없습니다.")
            + "\n\n[해석]\n" + interpretation
            + "\n\n[권장 조치]\n" + action
        )
    return "[상세 분석]\n" + (interpretation or base_answer)


def _trim_dangling_detail_label(candidate: str) -> str:
    return re.sub(r"(?:\n|\s)*(?:근거|분석|상세)\s*:\s*$", "", candidate).strip()


def _detail_interpretation(base_answer: str, candidate: str) -> str | None:
    """Keep useful LLM interpretation, but reject a copied quick answer."""
    if candidate.startswith(base_answer.strip()):
        candidate = candidate[len(base_answer.strip()):].strip()
        # A response built by appending statistics to the quick answer is exactly
        # what detailed mode must avoid; the structured formatter handles it better.
        return None
    return candidate if len(candidate) >= 30 else None


def _grounded_detail(candidate: str, base_answer: str, context: Mapping[str, object]) -> bool:
    """Reject novel numbers and reordered Zone paths before prose reaches clients."""
    allowed = set(re.findall(r"\d+(?:\.\d+)?", base_answer))
    flows: list[list[str]] = []
    recommended: list[list[str]] = []

    def collect(value: object, key: str = "") -> None:
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                collect(child, str(child_key))
        elif isinstance(value, list):
            if key.endswith("flow") or key == "recommended_flow":
                flows.append([str(item) for item in value])
                if key == "recommended_flow":
                    recommended.append([str(item) for item in value])
            for child in value:
                collect(child, key)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            allowed.add(str(value))
            allowed.add(f"{value:.0f}")
            if 0 <= value <= 1:
                allowed.add(f"{value * 100:.0f}")
        elif isinstance(value, str):
            allowed.update(re.findall(r"\d+(?:\.\d+)?", value))

    collect(context)
    if any(number not in allowed for number in re.findall(r"\d+(?:\.\d+)?", candidate)):
        return False
    for match in re.finditer(r"\d+(?:\s*(?:→|->)\s*\d+)+", candidate):
        sequence = re.findall(r"\d+", match.group())
        prefix = candidate[max(0, match.start() - 40):match.start()]
        recommendation_at = max(prefix.rfind(word) for word in ("권장", "추천", "대표"))
        actual_at = max(prefix.rfind(word) for word in ("실제", "수행", "현재"))
        allowed_flows = recommended if recommended and recommendation_at > actual_at else flows
        if not any(any(flow[start:start + len(sequence)] == sequence
                       for start in range(len(flow))) for flow in allowed_flows):
            return False
    return True


def error_response(code: int, message: str) -> JSONResponse:
    envelope = CopilotResponse(status="ERROR", code=code, message=message, data=None)
    return JSONResponse(status_code=code, content=envelope.model_dump(mode="json"))


def _success_response(
    query: CopilotQueryRequest,
    started: float,
    *,
    answer: str,
    retrieval_used: bool,
    citations: list[str],
    degraded: bool = False,
    pose_context: dict[str, object] | None = None,
    sequence_context: dict[str, object] | None = None,
    context_status: str = "not_available",
) -> JSONResponse:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    # Wall clock from handler entry to just before serialization, slot wait included.
    latency_ms = int((time.monotonic() - started) * 1000)
    envelope = CopilotResponse(
        status="SUCCESS",
        code=200,
        message=_SUCCESS_MESSAGE,
        data=CopilotQueryData(
            request_id=query.request_id,
            session_id=query.session_id,
            user_id=query.user_id,
            scenario_id=query.scenario_id,
            generated_at=generated_at,
            latency_ms=latency_ms,
            degraded=degraded,
            answer=answer,
            resolved_intents=[query.task_type],
            context_status=context_status,  # type: ignore[arg-type]
            sequence_context=sequence_context,
            coating_context=None,
            pose_context=pose_context,
            retrieval_used=retrieval_used,
            citations=citations,
        ),
    )
    logger.info(
        "copilot query done request_id=%s task_type=%s latency_ms=%d retrieval=%s citations=%d",
        query.request_id,
        query.task_type,
        latency_ms,
        retrieval_used,
        len(citations),
    )
    return JSONResponse(status_code=200, content=envelope.model_dump(mode="json"))


async def _execute_copilot_query(
    query: CopilotQueryRequest,
    *,
    pipeline_provider: PipelineProvider | None,
    pending_answers: Mapping[str, str] = PENDING_TASK_ANSWERS,
    sequences: SequenceStore,
    mcp_base_url: str,
    transport: object | None = None,
    started: float | None = None,
    reports: V1Store | None = None,
    sequence_engine: SequenceAnalysisEngine | None = None,
) -> JSONResponse:
    """Execute one normalized query for both text and voice transports."""
    request_started = time.monotonic() if started is None else started

    intent: str | None = None
    intent_pipeline: object | None = None
    if query.task_type in ("pose_recomm", "trajectory_recomm"):
        classifier = None
        intent = preset_intent(query.task_type, query.prompt) if query.response_mode == "quick" else None
        if pipeline_provider is not None and intent is None:
            try:
                intent_pipeline = await to_thread.run_sync(pipeline_provider)
                method = getattr(intent_pipeline, "classify_copilot_intent", None)
                if callable(method):
                    def classifier(task_type: str, prompt: str) -> str:
                        with generation_slot_for(intent_pipeline):
                            return str(method(task_type, prompt))
            except Exception:  # noqa: BROAD_EXCEPT_OK - keyword fallback keeps structured answers available
                logger.exception("copilot intent pipeline unavailable request_id=%s", query.request_id)
        if intent is None:
            intent = await to_thread.run_sync(
                lambda: classify_question_intent(query.task_type, query.prompt, classifier)
            )
        logger.info("copilot intent request_id=%s task=%s intent=%s", query.request_id, query.task_type, intent)
    # If the model is unavailable, keep the selected structured tab operational;
    # only a successful semantic-model classification may cross-route to RAG.
    knowledge_routed = intent == "knowledge" and classifier is not None
    viewer_context_requested = bool(query.context and query.context.startswith(_MOTION_VIEWER_CONTEXT_PREFIX))
    try:
        viewer_analysis = _parse_motion_viewer_context(query.context, task_type=query.task_type)
    except _InvalidViewerPlayback:
        return error_response(422, "Invalid motion viewer playback context")
    if viewer_analysis is not None and query.task_type == "pose_recomm" and "playback_context" in viewer_analysis and _asks_current_viewer_pose(query.prompt):
        knowledge_routed = False

    async def _synthesize_structured_detail(
        base_answer: str,
        analysis_context: dict[str, object],
        *,
        available: bool,
    ) -> tuple[str, bool]:
        """Turn verified engine output into prose only for the requested detailed mode."""
        if query.response_mode != "detailed" or not available or pipeline_provider is None:
            return base_answer, False
        try:
            pipeline = intent_pipeline or await to_thread.run_sync(pipeline_provider)
            session_summary = ""
            recent_turns: list[dict[str, str]] = []
            if query.session_id:
                session_summary, recent_turns = await to_thread.run_sync(
                    lambda: (
                        summarize_session(sequences.session_bodies(query.session_id)),
                        sequences.recent_query_turns(query.session_id, limit=4),
                    )
                )
            instruction = build_query_instruction(
                query.prompt,
                # The verified viewer summary is appended below in a strict JSON block.
                # Repeating its transport envelope as free-form context wastes tokens.
                client_context=None if viewer_analysis is not None else query.context,
                session_summary=session_summary,
                recent_turns=recent_turns,
                response_mode="detailed",
            )
            verified = {
                "task_type": query.task_type,
                "verified_answer": base_answer,
                "analysis_context": analysis_context,
            }
            instruction += "\n[서버 구조화 분석 결과]\n" + json.dumps(
                verified, ensure_ascii=False, separators=(",", ":")
            )
            instruction += (
                "\n위 분석 결과에 있는 사실만 사용해 질문에 자연스럽게 답하세요. "
                "수치와 Zone 순서는 바꾸거나 새로 만들지 마세요."
            )

            def _guarded_detail_run() -> Mapping[str, object]:
                with generation_slot_for(pipeline):
                    return pipeline.run(
                        instruction=instruction,
                        force_route="fast",
                        session_id=query.session_id,
                        enable_thinking=False,
                        use_rag=False,
                        max_tokens=256,
                        request_id=query.request_id,
                    )

            result = await to_thread.run_sync(_guarded_detail_run)
            parsed = parse_knowledge_answer(str(result.get("response", "")), [])
            candidate = _trim_dangling_detail_label(parsed.body)
            grounded = _grounded_detail(candidate, base_answer, analysis_context)
            interpretation = _detail_interpretation(base_answer, candidate) if candidate and grounded else None
            return _structured_detail_fallback(
                query.task_type,
                base_answer,
                analysis_context,
                interpretation=interpretation,
            ), bool(result.get("deadline_hit", False)) or not grounded
        except Exception:  # noqa: BROAD_EXCEPT_OK - verified deterministic answer remains available
            logger.exception(
                "copilot structured detail synthesis failed request_id=%s task_type=%s",
                query.request_id,
                query.task_type,
            )
        return _structured_detail_fallback(query.task_type, base_answer, analysis_context), True

    if viewer_analysis is not None and query.task_type in ("pose_recomm", "trajectory_recomm") and not knowledge_routed:
        analysis_context = (_current_viewer_pose_context(query.prompt, viewer_analysis)
                            if query.task_type == "pose_recomm" else viewer_analysis)
        available = analysis_context.get("scope") != "current_zone" or analysis_context.get("status") == "available"
        verified_answer = _viewer_answer(query.task_type, query.prompt, analysis_context, intent)
        answer, detail_degraded = await _synthesize_structured_detail(
            verified_answer, analysis_context, available=available
        )
        return _success_response(
            query, request_started, answer=answer, retrieval_used=False, citations=[],
            degraded=detail_degraded,
            pose_context=analysis_context if query.task_type == "pose_recomm" else None,
            sequence_context=analysis_context if query.task_type == "trajectory_recomm" else None,
            context_status="available" if available else "not_available",
        )
    # Unsigned/forged viewer claims can still resolve against saved server data.
    if viewer_context_requested:
        query = query.model_copy(update={"context": None})

    context = (query.context or "").strip()
    # 시퀀스(`SEQ-012`)를 가리키지 않은 자세·순서 질문은 complete 가 남긴 세션 결과로 답한다(pose_query 참고).
    # complete 뒤에는 질문에 "구역"이 없어도 그렇다 — 뷰어 채팅은 칩으로 유형만 고른다.
    session_id = query.session_id
    if not query.scenario_id and session_id:
        def _saved_scenario() -> str | None:
            latest = sequences.latest_sequence_id(session_id)
            body = sequences.load_body(session_key=session_id, sequence_id=latest) if latest else None
            if body and body.get("scenario_id"):
                return str(body["scenario_id"])
            get_session = getattr(reports, "get_session", None)
            session = get_session(session_id) if callable(get_session) else None
            return session.scenario_id if session is not None else None
        scenario = await to_thread.run_sync(_saved_scenario)
        if scenario:
            query = query.model_copy(update={"scenario_id": scenario})
    found = (
        await to_thread.run_sync(lambda: reports.latest_complete_analysis(session_id))
        if reports is not None and session_id and query.task_type in ("pose_recomm", "trajectory_recomm")
        and (not context or context.isdigit())
        else None
    )
    if found is not None and session_id:
        captured = found[1].get("sequences")
        captured_ids = {str(item["sequence_id"]) for item in captured
                        if isinstance(item, Mapping) and item.get("sequence_id")} if isinstance(captured, list) else set()
        if captured_ids:
            latest = await to_thread.run_sync(lambda: sequences.latest_sequence_id(session_id))
            if latest is not None and latest not in captured_ids:
                found = None
    if found is None and session_id and not context and query.task_type in ("pose_recomm", "trajectory_recomm") and not knowledge_routed and (reports is not None or query.task_type == "trajectory_recomm" or asks_about_zones(query.prompt, query.context)):
        # Keep explicit-sequence pose queries on their original scoring path.
        found = await to_thread.run_sync(lambda: live_analysis(
            sequences, session_id=session_id, scenario_id=query.scenario_id,
            task_type=query.task_type, sequence_engine=sequence_engine,
            mcp_base_url=mcp_base_url, transport=transport,
            zone_id=requested_zone(query.prompt, query.context) if query.task_type == "pose_recomm" else None,
        ))
    elif found is None and session_id and context.isdigit() and query.task_type == "pose_recomm" and not knowledge_routed:
        found = await to_thread.run_sync(lambda: live_analysis(
            sequences, session_id=session_id, scenario_id=query.scenario_id,
            task_type=query.task_type, sequence_engine=sequence_engine,
            mcp_base_url=mcp_base_url, transport=transport, zone_id=context,
        ))
    if query.task_type == "trajectory_recomm" and not knowledge_routed and session_id:
        if found is not None and sequence_engine is not None and query.scenario_id:
            report_id, analysis = found
            refreshed = dict(analysis)
            stored_report = refreshed.get("sequence_report")
            if isinstance(stored_report, dict):
                refreshed["sequence_report"] = sequence_engine.rescore_report(
                    stored_report, scenario_id=query.scenario_id
                )
                found = (report_id, refreshed)
        order = answer_order_question(found, prompt=query.prompt, intent=intent)
        logger.info(
            "copilot trajectory_recomm request_id=%s status=%s reason=%s",
            query.request_id, order.analysis_status, order.reason_code,
        )
        answer, detail_degraded = await _synthesize_structured_detail(
            order.answer, order.context, available=order.analysis_status == "available"
        )
        return _success_response(
            query, request_started, answer=answer, retrieval_used=False, citations=[],
            degraded=detail_degraded,
            sequence_context=order.context,
            context_status="available" if order.analysis_status == "available" else "not_available",

        )

    if query.task_type == "pose_recomm" and not knowledge_routed and (
        found is not None or asks_about_zones(query.prompt, query.context)
    ):
        zone = (
            answer_pose_question(found, prompt=query.prompt, context=query.context, intent=intent)
            if session_id
            else zone_unavailable("session_id_missing")
        )
        logger.info(
            "copilot pose_recomm zone request_id=%s zone=%s status=%s reason=%s",
            query.request_id, zone.context.get("zone_id"), zone.analysis_status, zone.reason_code,
        )
        live_context = found[1].get("live_pose_context") if found is not None else None
        report_pose = found[1].get("pose_report") if found is not None else None
        limited_confidence = isinstance(report_pose, Mapping) and report_pose.get("confidence") == "limited"
        if isinstance(report_pose, Mapping):
            for key in ("confidence", "limited_sequence_count", "reason_codes"):
                if key in report_pose:
                    zone.context[key] = report_pose[key]
        base_answer = zone.answer
        if isinstance(live_context, dict):
            zone.context.update(live_context)
            base_answer = "최근 저장된 분사 중 최대 3회 표본을 분석한 중간 결과입니다. " + base_answer
        answer, detail_degraded = await _synthesize_structured_detail(
            base_answer, zone.context, available=zone.analysis_status == "available"
        )
        if isinstance(live_context, dict) and query.response_mode == "detailed":
            answer = "최근 저장된 분사 중 최대 3회 표본을 분석한 중간 결과입니다.\n" + answer
        return _success_response(
            query, request_started, answer=answer, retrieval_used=False, citations=[],
            degraded=detail_degraded or limited_confidence or bool(isinstance(live_context, dict) and live_context.get("reason_codes")),
            pose_context=zone.context,
            context_status="available" if zone.analysis_status == "available" else "not_available",

        )

    if query.task_type == "pose_recomm" and not knowledge_routed:
        pose = await to_thread.run_sync(
            lambda: score_pose(
                sequences,
                session_id=query.session_id,
                context=query.context,
                mcp_base_url=mcp_base_url,
                transport=transport,  # type: ignore[arg-type]
            )
        )
        if pose.analysis_status == "available":
            answer = pose.summary or "숙련자 자세와 비교했습니다."
        else:
            answer = _POSE_REASON_ANSWERS.get(pose.reason_code or "", _POSE_FALLBACK_ANSWER)
        logger.info(
            "copilot pose_recomm request_id=%s sequence=%s status=%s reason=%s",
            query.request_id, pose.sequence_id, pose.analysis_status, pose.reason_code,
        )
        answer, detail_degraded = await _synthesize_structured_detail(
            answer, pose.as_context(), available=pose.analysis_status == "available"
        )
        return _success_response(
            query, request_started, answer=answer, retrieval_used=False, citations=[],
            degraded=detail_degraded or bool(pose.reason_code and pose.analysis_status == "available"),
            pose_context=pose.as_context(),
            context_status="available" if pose.analysis_status == "available" else "not_available",

        )

    if query.task_type != "knowledge_search" and not knowledge_routed:
        answer = pending_answers.get(query.task_type) or PENDING_TASK_ANSWERS[query.task_type]
        return _success_response(
            query, request_started, answer=answer, retrieval_used=False, citations=[]
        )

    if pipeline_provider is None:
        return error_response(503, "Copilot pipeline is not configured")
    try:
        pipeline = intent_pipeline or await to_thread.run_sync(pipeline_provider)
    except Exception:  # noqa: BROAD_EXCEPT_OK - construction failures surface as not ready
        logger.exception("copilot query: pipeline construction failed request_id=%s", query.request_id)
        return error_response(503, "Copilot pipeline is not ready")

    # Explicit Y/N remains a backwards-compatible override.  When omitted, any
    # question that reaches this knowledge-generation branch needs retrieval;
    # structured pose/order/report questions have already returned above.
    use_rag = query.use_rag != "N"
    session_summary = ""
    recent_turns: list[dict[str, str]] = []
    if query.session_id:
        session_summary, recent_turns = await to_thread.run_sync(
            lambda: (
                summarize_session(sequences.session_bodies(query.session_id)),
                sequences.recent_query_turns(query.session_id, limit=4),
            )
        )
    definition_context = (viewer_definition_context(query.prompt, query.context)
                          if query.response_mode == "quick" and query.task_type == "knowledge_search" else None)
    instruction = build_query_instruction(
        query.prompt,
        client_context=None if viewer_analysis is not None else (definition_context or query.context),
        session_summary=session_summary,
        recent_turns=[] if definition_context is not None else recent_turns,
        response_mode=query.response_mode,
    )
    generation_instruction = (build_query_instruction(
        query.prompt,
        client_context=None if viewer_analysis is not None else (definition_context or query.context),
        session_summary=session_summary,
        recent_turns=[] if definition_context is not None else recent_turns,
        response_mode=query.response_mode,
        concise=True,
    ) if query.response_mode == "quick" else None)

    def _guarded_run() -> Mapping[str, object]:
        with generation_slot_for(pipeline):
            return pipeline.run(
                instruction=instruction,
                force_route="fast",
                session_id=query.session_id,
                enable_thinking=False,
                use_rag=use_rag,
                max_tokens=128 if query.response_mode == "quick" else 512,
                request_id=query.request_id,
                **({
                    "quick_cache_scope": json.dumps([query.user_id, query.scenario_id, query.task_type]),
                    "concise_answer": True,
                    "generation_instruction": generation_instruction,
                    **({"retrieval_instruction": query.prompt}
                       if getattr(pipeline, "quick_query_runtime", None) is not None else {}),
                    # Persistent dialogue is already in instruction. Do not add
                    # the pipeline's second, recursively summarized copy.
                    "use_pipeline_history": False,
                } if query.response_mode == "quick" else {}),
            )

    try:
        result = await to_thread.run_sync(_guarded_run)
    except OpenAIBackendUnavailableError:
        logger.exception("copilot query: model server unavailable request_id=%s", query.request_id)
        return error_response(503, "Model server unavailable")
    except Exception:  # noqa: BROAD_EXCEPT_OK - never leak internals past the boundary
        logger.exception("copilot query: pipeline failed request_id=%s", query.request_id)
        return error_response(500, "Copilot query failed")

    retrieved = retrieved_chunk_ids(result.get("rag_chunks") or []) if use_rag else []
    parsed = parse_knowledge_answer(str(result.get("response", "")), retrieved)
    answer = parsed.body
    if query.response_mode == "quick":
        answer = re.sub(r"\n+\s*\[?근거\]?\s*:?\s*$", "", answer).rstrip()
    logger.info(
        "copilot answer normalized request_id=%s repair_path=%s citations=%d dropped=%d",
        query.request_id,
        parsed.repair_path,
        len(parsed.citations),
        len(parsed.dropped),
    )
    return _success_response(
        query,
        request_started,
        answer=answer,
        retrieval_used=use_rag,
        citations=list(parsed.citations),
        degraded=parsed.repair_path in FAILED_REPAIR_PATHS or bool(result.get("deadline_hit", False)),
        context_status="available" if session_summary else "not_available",

    )


async def execute_copilot_query(
    query: CopilotQueryRequest,
    *,
    pipeline_provider: PipelineProvider | None,
    pending_answers: Mapping[str, str] = PENDING_TASK_ANSWERS,
    sequences: SequenceStore,
    mcp_base_url: str,
    transport: object | None = None,
    started: float | None = None,
    reports: V1Store | None = None,
    sequence_engine: SequenceAnalysisEngine | None = None,
) -> JSONResponse:
    """Normalize foreign questions, keep Korean analysis, then localize only prose."""
    request_started = time.monotonic() if started is None else started
    response_language = query.response_language or query.language
    original_provider = pipeline_provider
    if original_provider is not None:
        # A per-request binding switches only the HTTP generator, never the shared model.
        pipeline_provider = lambda: select_query_pipeline(original_provider(), query.response_mode)
    pipeline: object | None = None

    async def _translate(text: str, source: str, target: str) -> str:
        nonlocal pipeline
        if source == target:
            return text
        if pipeline_provider is None:
            raise TranslationUnavailableError("Translation engine unavailable")
        if pipeline is None:
            pipeline = await to_thread.run_sync(pipeline_provider)
        translator = translation_pipeline_for(pipeline, source, target)
        def guarded() -> str:
            with generation_slot_for(translator):
                return translate_text(translator, text, source_language=source, target_language=target)
        return await to_thread.run_sync(guarded)

    try:
        normalized = await _translate(query.prompt, query.language, "ko")
        if len(normalized) > 8000:
            raise TranslationUnavailableError("Translated question exceeds analysis limit")
        analysis_query = query.model_copy(update={"prompt": normalized})
        response = await _execute_copilot_query(
            analysis_query, pipeline_provider=pipeline_provider, pending_answers=pending_answers,
            sequences=sequences, mcp_base_url=mcp_base_url, transport=transport,
            started=request_started, reports=reports, sequence_engine=sequence_engine,
        )
        if response.status_code != 200:
            return response
        body = json.loads(response.body)
        data = body["data"]
        data["answer"] = await _translate(data["answer"], "ko", response_language)
    except Exception:  # Translation failures never masquerade as a requested-language answer.
        logger.exception("copilot translation failed request_id=%s", query.request_id)
        return error_response(503, "Query translation unavailable; retry or use Korean")

    if {"language", "response_language"} & query.model_fields_set:
        data["language"] = query.language
        data["response_language"] = response_language
    else:
        # Legacy clients (including the separate rc1 gateway) validate the old
        # response strictly. Only opt-in multilingual calls add language metadata.
        data.pop("language", None)
        data.pop("response_language", None)
    data["latency_ms"] = max(0, int((time.monotonic() - request_started) * 1000))
    data["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if query.session_id:
        try:
            sequences.record_query_turn(
                session_key=query.session_id, user_id=query.user_id,
                prompt=query.prompt, answer=data["answer"], task_type=query.task_type,
                created_at=data["generated_at"],
            )
        except Exception:
            logger.exception("copilot query: failed to store dialogue request_id=%s", query.request_id)
    return JSONResponse(status_code=200, content=body)


def create_copilot_router(
    pipeline_provider: PipelineProvider | None,
    *,
    pending_answers: Mapping[str, str] = PENDING_TASK_ANSWERS,
    store: SequenceStore | None = None,
    mcp_base_url: str | None = None,
    transport: object | None = None,
    reports: V1Store | None = None,
    sequence_engine: SequenceAnalysisEngine | None = None,
) -> APIRouter:
    """``pipeline_provider`` returns the live pipeline singleton; ``None`` when live mode is off.

    ``reports`` is the complete-report store; without it zone questions fall back to scoring one sequence.
    """
    router = APIRouter()
    sequences = store if store is not None else SequenceStore()
    mcp_base = mcp_base_url or os.environ.get("MCP_BASE_URL", "http://localhost:8100")

    @router.post(
        COPILOT_QUERY_PATH,
        response_model=CopilotResponse,
        tags=["3. AI Co-Pilot 질의"],
        summary="3-1. 텍스트 기반 Co-Pilot 질의",
        description=(
            "Replay 및 일반 UI용 텍스트 질의입니다. 답변 수준에 따라 Qwen3-8B 또는 Qwen3.6-27B 모델과 RAG를 사용합니다.\n\n"
            "- **빠른 안내 (`quick`, 기본값) · Qwen3-8B 모델**: 질문의 핵심부터 짧게 답합니다. 필수 안전 조건은 함께 안내합니다.\n"
            "  - 사용자·세션 맥락·검색 근거가 모두 같은 반복 질문은 최대 2분간 생성 결과를 재사용합니다.\n"
            "  - 새 시퀀스·대화·문맥·근거가 반영되면 다시 생성합니다. 새 질문의 처리 시간은 모델 부하에 따라 달라집니다.\n"
            "- **상세 분석 (`detailed`) · Qwen3.6-27B 모델**: 근거·원인·권장 조치를 확장해 설명합니다.\n"
            "- **질문 언어**: `language`로 지정합니다. 생략 시 `ko`입니다.\n"
            "- **답변 언어**: `response_language`로 지정하며, 생략/null이면 질문 언어를 따릅니다.\n"
            "- 외국어 질문을 한국어 분석·검색으로 연결하고 최종 `answer`를 요청 언어로 번역합니다.\n"
            "- 점수·단위·Zone 순서·구조화 context·citations는 유지합니다. 번역 처리 시간이 추가됩니다.\n"
            "  - 일부 언어는 번역 품질을 위해 기존 Qwen3.6-27B 모델을 사용합니다.\n"
            "- 번역에 실패하거나 수치가 달라지면 한국어를 외국어 응답으로 표시하지 않고 503을 반환합니다."
        ),
    )
    async def copilot_query(query: Annotated[CopilotQueryRequest, Body(openapi_examples={
        "korean": {"summary": "한국어 질문·답변 (기존 호출 호환)", "value": {
            "user_id": "1234", "scenario_id": "3", "session_id": "3_11000346_1",
            "prompt": "스프레이 도장 시 적정 이격거리를 알려주세요.",
            "task_type": "knowledge_search", "response_mode": "quick", "use_rag": "Y",
            "language": "ko", "response_language": "ko",
        }},
        "vietnamese": {"summary": "베트남어 질문 → 베트남어 답변", "value": {
            "user_id": "1234", "scenario_id": "3", "session_id": "3_11000346_1",
            "prompt": "Khoảng cách phun phù hợp là bao nhiêu?",
            "task_type": "knowledge_search", "response_mode": "quick", "use_rag": "Y",
            "language": "vi", "response_language": "vi",
        }},
        "english_to_korean": {"summary": "영어 질문 → 한국어 답변", "value": {
            "user_id": "1234", "prompt": "How can I keep a consistent spray distance?",
            "task_type": "knowledge_search", "response_mode": "detailed", "use_rag": "Y",
            "language": "en", "response_language": "ko",
        }},
    })]) -> JSONResponse:
        return await execute_copilot_query(
            query,
            pipeline_provider=pipeline_provider,
            pending_answers=pending_answers,
            sequences=sequences,
            mcp_base_url=mcp_base,
            transport=transport,
            reports=reports,
            sequence_engine=sequence_engine,
        )

    return router


def install_copilot_error_handlers(app: FastAPI) -> None:
    """Wraps request validation errors under ``/api/v1/`` in the v2.5 ERROR envelope."""

    async def _on_validation_error(request: Request, exc: Exception) -> JSONResponse:
        assert isinstance(exc, RequestValidationError)
        if request.url.path.startswith(_API_V1_PREFIX):
            logger.info(
                "copilot request rejected path=%s errors=%s",
                request.url.path,
                [(error.get("loc"), error.get("type")) for error in exc.errors()],
            )
            # Only stable metadata crosses the boundary; errors.input/ctx can
            # contain transcripts, credentials, or custom exception contents.
            first = exc.errors()[0] if exc.errors() else {}
            path = ".".join(re.sub(r"[^A-Za-z0-9_\-]", "_", str(item))[:40]
                            for item in first.get("loc", ())[:5])[:120]
            reasons = {
                "missing": "Field required", "extra_forbidden": "Extra field is not permitted",
                "literal_error": "Unsupported value", "enum": "Unsupported value",
                "string_too_short": "Value is too short", "string_too_long": "Value is too long",
                "int_parsing": "Integer required", "float_parsing": "Number required",
                "json_invalid": "Malformed JSON", "model_attributes_type": "Object required",
                "greater_than": "Value is below the allowed range",
                "greater_than_equal": "Value is below the allowed range",
                "less_than": "Value is above the allowed range",
                "less_than_equal": "Value is above the allowed range",
            }
            reason = reasons.get(str(first.get("type")), "Invalid value")
            return error_response(422, f"Invalid request: {path or 'request'} ({reason})")
        return await request_validation_exception_handler(request, exc)

    app.add_exception_handler(RequestValidationError, _on_validation_error)
