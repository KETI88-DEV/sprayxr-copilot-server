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
from trainer.demo_api.coating_query import answer_coating_question, parse_viewer_coating_context, report_coating_context
from trainer.demo_api.copilot_intent import classify_question_intent, fallback_intent, preset_intent, is_coating_status_question, is_coating_advice_question, is_general_painting_knowledge
from trainer.demo_api.dialogue import (authenticated_facts, conversation_key, drop_ungrounded, parse_plan, plain_answer,
                                       selected_direction_answer, ungrounded_measurements, viewer_findings_answer,
                                       viewer_measurement_answer, viewer_metadata_context, without_hanzi)
from trainer.demo_api.playback_direction import playback_direction_from_context, playback_direction_answer, simple_direction_question, requested_direction_scope, aggregate_direction, LABELS as DIRECTION_LABELS
from trainer.demo_api.copilot_live_analysis import current_zone, live_analysis
from trainer.demo_api.copilot_session_context import build_query_instruction, summarize_session, viewer_definition_context
from trainer.demo_api.pose_query import (
    answer_order_question,
    answer_pose_question,
    answer_sequence_question,
    answer_zone_question,
    asked_arm,
    asks_about_zones,
    asks_whole_session,
    score_pose,
    unsupported_pose_part,
    requested_zone,
    zone_unavailable,
)
from trainer.demo_api.query_language import TranslationUnavailableError, translate_text
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.viewer_arm_pose import arm_pose_answer, parse_viewer_arm_pose
from trainer.demo_api.sequence_analysis import SequenceAnalysisEngine
from trainer.demo_api.reference_order_query import actual_order_intent, current_next_answer, is_current_next_question, is_reference_order_question, resolve_reference_target, reference_order_answer
from trainer.demo_api.v1_store import V1Store
from trainer.demo_api.query_runtime import generation_slot_for, retrieval_question, select_query_pipeline, translation_pipeline_for
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


# analysis(채점·분석 결과를 옮긴 답, 모델 미사용) · documents(문서 검색 + 모델) · conversation(모델 대화)
ANSWER_SOURCE_HEADER: Final = "X-Copilot-Answer-Source"
_EVALUATION_QUESTION: Final = re.compile(r"잘한|못한|고쳐야|고칠|개선할|개선점|문제점|부족한 점")
# 대화 계획 모델이 속도·Zone 질문을 도포 방향으로 분류한다(2026-11-11 Spark: "지금 속도는 적당해?"에 방향을 답함).
_DIRECTION_WORD: Final = re.compile(r"방향|수평|수직|대각|가로|세로|패턴|direction", re.I)


def _session_findings(found: tuple[str, Mapping[str, Any]] | None) -> str:
    """평가를 묻는 자유 질문의 답: 완료 분석의 판정(자세 최저 구역, 도막 과다·부족 구역)을 그대로 전한다.

    모델에 판정을 넘겨 풀어 쓰게 하면 8B가 없는 판정을 만든다(2026-11-11 Spark: 도막이 정상인 9번 구역을 "도막 부족").
    """
    pose, coating = answer_zone_question(found, context=None), report_coating_context(found)
    parts = [pose.answer] if pose.analysis_status == "available" else []
    if coating["analysis_status"] == "available":
        parts.append(answer_coating_question(coating, "", detailed=False))
    return "\n".join(parts) or "이 세션은 완료 분석 결과가 없어 잘한 점이나 고칠 점을 판정할 수 없습니다."


def _selected_sequence(context: str | None) -> str | None:
    """`context`가 시퀀스 ID(`SEQ-012`)면 그것. 구역 번호나 뷰어 문맥은 시퀀스가 아니다."""
    value = (context or "").strip()
    return value if value and not value.isdigit() and ":" not in value and len(value) <= 64 else None


def _pose_engine_question(query: CopilotQueryRequest) -> bool:
    """자세 탭의 자세 질문은 대화 계획·모델 분류를 거치지 않고 채점 엔진으로 간다.

    분류가 빗나가면 자유 대화가 채점 없이 수치와 판정을 지어낸다(2026-11-11 재현).
    """
    return (query.task_type == "pose_recomm" and not is_coating_status_question(query.prompt)
            and not is_coating_advice_question(query.prompt)
            and bool(re.search(r"자세(?!히|하게|한)|동작|교정|고쳐|고치|pose|posture|팔꿈치|어깨|무릎|허리|elbow|shoulder|knee|(?:zone|존|구역)\s*\d+",
                               query.prompt, re.I)))


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
            return f"전체 기록의 주요 도포 방향은 {DIRECTION_LABELS.get(classification, '비정형')}입니다." if classification and classification != "unavailable" else "현재 기록에서는 주요 도포 방향을 계산할 수 없습니다."
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
        if context.get("zone_id") is None:
            return "현재 재생 위치는 Zone ID가 없는 구간이어서 Zone 기준 작업 비교값을 표시할 수 없습니다."
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
    intent: str | None = None,
) -> str:
    """검증된 수치만 덧붙인 문장으로 답한다 — 제목 틀과 고정 문장은 쓰지 않는다(2026-11-11 지시)."""
    if task_type == "trajectory_recomm" and intent not in (None, "order_reference"):
        evidence = ""
        if intent == "order_missing":
            expected = {str(z) for z in analysis_context.get("recommended_flow", [])}
            visited = {str(z) for z in analysis_context.get("selected_flow", analysis_context.get("first_visit_flow", []))}
            if expected:
                evidence = f"\n전문가 기준 {len(expected)}개 Zone 중 방문 {len(expected & visited)}개, 누락 {len(expected - visited)}개입니다. Zone 방문 여부와 Zone 내부 도막 충족 여부는 별도로 평가합니다."
        elif intent == "order_adherence":
            evidence = "\n전문가 작업 흐름과 순번별 분포를 비교한 순서 유사도입니다. 도막 두께와 자세 평가는 포함하지 않습니다."
        return base_answer + evidence
    if task_type == "trajectory_recomm":
        recommended_value = analysis_context.get("recommended_flow")
        recommended = _flow_text(recommended_value)
        if not recommended:
            return base_answer
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
        # 자료에서 나온 안내만 쓴다 — 어느 기록에나 붙는 일반 권고 문장은 넣지 않는다.
        action = (
            ["실제 수행 순서가 권장 순서와 같습니다."] if selected and recommended == selected
            else [f"Zone {stable_prefix[0]}에서 시작한 뒤 Zone {stable_prefix[1]}로 이동하세요."] if len(stable_prefix) >= 2
            else []
        )
        return "\n".join([f"전문가 기준 권장 순서는 Zone {recommended}입니다.", *tendencies, *segment_lines, *action])
    elif task_type == "pose_recomm" and analysis_context.get("scope") != "current_zone":
        zones = analysis_context.get("zones")
        zone_rows = list(zones.values()) if isinstance(zones, dict) else zones
        scored = [
            (float(row["similarity_percent"]), row)
            for row in (zone_rows if isinstance(zone_rows, list) else [])
            if isinstance(row, dict) and isinstance(row.get("similarity_percent"), (int, float))
        ]
        if scored:
            _, worst = min(scored, key=lambda item: item[0])
            score = _percent_text(worst.get("similarity_percent"), already_percent=True)
            worst_text = f"가장 낮은 구역은 Zone {worst.get('zone_id')}({score})입니다."
            if worst_text not in base_answer:
                return base_answer + "\n" + worst_text
    return base_answer


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
    coating_context: dict[str, object] | None = None,
    context_status: str = "not_available",
    generated: bool = False,
) -> JSONResponse:
    """`generated`는 모델이 문장을 만든 답에만 준다. 나머지는 채점·분석 결과를 그대로 옮긴 답이다.

    답을 만든 곳은 응답 헤더로만 알린다 — 본문 키를 엄격히 검사하는 기존 클라이언트가 있어 본문은 바꾸지 않는다.
    화면이 요청한 답변 수준만 보고 모델 이름을 붙이면 채점 결과에도 모델 이름이 붙는다(2026-11-11 뷰어 점검).
    """
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
            coating_context=coating_context,
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
    source = "documents" if retrieval_used else "conversation" if generated else "analysis"
    return JSONResponse(status_code=200, content=envelope.model_dump(mode="json"), headers={ANSWER_SOURCE_HEADER: source})


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
    coating_engine: object | None = None,
    conversation_id: str | None = None,
) -> JSONResponse:
    """Execute one normalized query for both text and voice transports."""
    request_started = time.monotonic() if started is None else started

    dialogue_turns = await to_thread.run_sync(lambda: sequences.recent_query_turns(
        query.session_id or "CHAT:" + query.user_id, limit=6,
        user_id=query.user_id, conversation_key=conversation_id,
    ))
    facts = authenticated_facts(query.context)
    try:
        # Pose/order playback validation applies only when consuming that
        # signed evaluation. Coating has its own independently signed claims.
        viewer_analysis = (_parse_motion_viewer_context(query.context, task_type=query.task_type)
                           if query.task_type in ("pose_recomm", "trajectory_recomm") else None)
    except _InvalidViewerPlayback:
        return error_response(422, "Invalid motion viewer playback context")
    relevant_viewer = viewer_analysis is not None and query.task_type in ("pose_recomm", "trajectory_recomm")
    if is_current_next_question(query.prompt):
        answer, scope, progress = await to_thread.run_sync(lambda: current_next_answer(
            query, sequences, reports, sequence_engine=sequence_engine, coating_engine=coating_engine,
        ))
        return _success_response(query, request_started, answer=answer, retrieval_used=False,
            citations=[], sequence_context=scope, coating_context=progress,
            context_status="available" if scope["analysis_status"] == "available" else "not_available")
    has_reference_plan = callable(getattr(sequence_engine, "reference_plan", None))
    explicit_actual_order = actual_order_intent(query.prompt)
    async def reference_answer():
        scenario, process = await to_thread.run_sync(
            lambda: resolve_reference_target(query, sequences, reports, facts=facts)
        )
        reference_query = query.model_copy(update={"scenario_id": scenario})
        answer, scope = await to_thread.run_sync(lambda: reference_order_answer(
            sequence_engine, scenario_id=scenario, process=process,
            detailed=query.response_mode == "detailed",
        ))
        return _success_response(
            reference_query, request_started, answer=answer, retrieval_used=False,
            citations=[], sequence_context=scope,
            context_status="available" if scope["analysis_status"] == "available" else "not_available",
        )
    # A full reference plan is available before any trainee sequence or complete
    # report. It never authorizes realtime movement or supplies actual visits.
    if (has_reference_plan and not relevant_viewer and query.task_type != "error_analysis"
            and is_reference_order_question(query.prompt)):
        return await reference_answer()
    frame_direction = playback_direction_from_context(query.context)
    viewer_requested = bool(query.context and query.context.startswith(("motion_viewer_analysis:", "motion_viewer_playback:")))
    def frame_answer():
        if requested_direction_scope(query.prompt) != "frame":
            answer, scope = aggregate_direction(query.context, query.prompt)
            return _success_response(query, request_started, answer=answer, retrieval_used=False,
                                     citations=[], sequence_context=scope,
                                     context_status="available" if scope.get("analysis_status") == "available" else "not_available")
        available = frame_direction is not None and frame_direction.get("analysis_status") == "available"
        return _success_response(query, request_started, answer=playback_direction_answer(frame_direction),
                                 retrieval_used=False, citations=[], sequence_context=frame_direction or {"scope": "current_frame", "analysis_status": "unavailable"},
                                 context_status="available" if available else "not_available")
    if viewer_requested and (simple_direction_question(query.prompt) or preset_intent("trajectory_recomm", query.prompt) == "order_direction"):
        return frame_answer()
    if viewer_requested and preset_intent(query.task_type, query.prompt) is None:
        signed_answer = viewer_measurement_answer(query.context, query.prompt) or (
            viewer_findings_answer(query.context) if _EVALUATION_QUESTION.search(query.prompt) else None)
        if signed_answer is not None:
            return _success_response(query, request_started, answer=signed_answer, retrieval_used=False, citations=[],
                                     context_status="available")
    decision = None
    planned_pipeline = None
    if query.task_type != "error_analysis" and pipeline_provider is not None and preset_intent(query.task_type, query.prompt) is None and not is_coating_status_question(query.prompt) and not _pose_engine_question(query):
        try:
            planned_pipeline = await to_thread.run_sync(pipeline_provider)
        except Exception:
            # Preserve engine-only responses when the optional generator is offline.
            planned_pipeline = None
        planner = select_query_pipeline(planned_pipeline, "quick") if planned_pipeline else None
        if getattr(planner, "supports_copilot_dialogue", callable(getattr(planner, "plan_copilot_dialogue", None))):
            try:
                def plan():
                    with generation_slot_for(planner):
                        return parse_plan(planner.plan_copilot_dialogue(query.prompt, task_type=query.task_type,
                                          recent_turns=dialogue_turns, facts=facts))
                decision = await to_thread.run_sync(plan)
            except Exception:
                logger.exception("copilot dialogue planning failed request_id=%s", query.request_id)
                decision = {"kind": "clarify", "intent": "knowledge", "scope": "unspecified"}

    if decision is not None and query.task_type == "knowledge_search" and is_general_painting_knowledge(query.prompt):
        # 계획 모델은 직전 자세·순서 대화에 끌려 일반 지식 질문을 기록 후속 대화로 분류한다(2026-11-12 재현).
        decision = {"kind": "coating_knowledge", "intent": "knowledge", "scope": "unspecified"}
    if decision is not None:
        intent = decision["intent"]
        if intent == "order_reference" and explicit_actual_order:
            intent = explicit_actual_order
            decision = {**decision, "intent": intent}
        if has_reference_plan and not relevant_viewer and intent == "order_reference" and decision["kind"] == "lookup":
            return await reference_answer()
        cross_tab = ((intent.startswith("pose_") and query.task_type != "pose_recomm")
                     or (intent.startswith("order_") and query.task_type != "trajectory_recomm"))
        if viewer_requested and intent == "order_direction" and decision["kind"] in {"lookup", "correction"} and _DIRECTION_WORD.search(query.prompt):
            return frame_answer()
        zone_lookup = (intent == "order_direction" or intent.startswith("pose_")) and decision["scope"] in {"zone", "playback"}
        if decision["kind"] in {"chat", "clarify", "correction"} or cross_tab or zone_lookup:
            try:
                if query.session_id:
                    facts["training_summary"] = await to_thread.run_sync(
                        lambda: summarize_session(sequences.session_bodies(query.session_id),
                                                  selected_sequence_id=_selected_sequence(query.context)))
                    if _EVALUATION_QUESTION.search(query.prompt):
                        found_report = (await to_thread.run_sync(lambda: reports.latest_complete_analysis(query.session_id))
                                        if reports is not None else None)
                        return _success_response(query, request_started, answer=_session_findings(found_report),
                                                 retrieval_used=False, citations=[],
                                                 context_status="available" if found_report else "not_available")
                def converse():
                    with generation_slot_for(planned_pipeline):
                        return planned_pipeline.converse_copilot(query.prompt, recent_turns=dialogue_turns,
                                                                 facts=facts, response_mode=query.response_mode)
                answer = plain_answer(await to_thread.run_sync(converse))
                if ungrounded_measurements(answer, facts=facts, prompt=query.prompt):
                    # 자료에 없는 측정값은 내보내지 않는다. 한 번 다시 생성하고, 그래도 남으면 그 부분을 뺀다.
                    answer = drop_ungrounded(await to_thread.run_sync(converse), facts=facts, prompt=query.prompt)
                return _success_response(query, request_started, answer=answer, retrieval_used=False, generated=True,
                                         citations=[], context_status="available" if facts else "not_available")
            except Exception:
                logger.exception("copilot conversation unavailable request_id=%s", query.request_id)
                return error_response(503, "Conversation engine temporarily unavailable; retry")

    trusted_coating = parse_viewer_coating_context(query.context, query.task_type)
    coating_advice_context = (trusted_coating if is_coating_advice_question(query.prompt) and trusted_coating and trusted_coating.get("analysis_status") == "available" else None)
    if is_coating_status_question(query.prompt) or (decision and decision["intent"] == "coating_status" and decision["kind"] == "lookup"):
        coating = trusted_coating
        if coating is None:
            found_coating = (await to_thread.run_sync(lambda: reports.latest_complete_analysis(query.session_id))
                             if reports is not None and query.session_id else None)
            coating = report_coating_context(found_coating)
        available = coating.get("analysis_status") == "available"
        return _success_response(query, request_started,
            answer=answer_coating_question(coating, query.prompt, detailed=query.response_mode == "detailed"),
            retrieval_used=False, citations=[], coating_context=coating,
            context_status="available" if available else "not_available")

    intent: str | None = decision["intent"] if decision else None
    intent_pipeline: object | None = planned_pipeline
    classifier = None
    if query.task_type in ("pose_recomm", "trajectory_recomm"):
        classifier = None
        intent = intent or preset_intent(query.task_type, query.prompt)
        if intent is None and _pose_engine_question(query):
            intent = fallback_intent(query.task_type, query.prompt)
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
    if intent == "order_reference" and explicit_actual_order:
        intent = explicit_actual_order
    if has_reference_plan and not relevant_viewer and intent == "order_reference":
        return await reference_answer()
    # If the model is unavailable, keep the selected structured tab operational;
    # only a successful semantic-model classification may cross-route to RAG.
    knowledge_routed = intent == "knowledge" and (classifier is not None or decision is not None)
    if viewer_requested and intent == "order_direction" and not knowledge_routed and (
            query.task_type == "trajectory_recomm" or _DIRECTION_WORD.search(query.prompt)):
        return frame_answer()
    viewer_context_requested = bool(query.context and query.context.startswith(_MOTION_VIEWER_CONTEXT_PREFIX))
    if viewer_analysis is not None and query.task_type == "pose_recomm" and "playback_context" in viewer_analysis and _asks_current_viewer_pose(query.prompt) and not is_coating_advice_question(query.prompt):
        knowledge_routed = False

    async def _synthesize_structured_detail(
        base_answer: str,
        analysis_context: dict[str, object],
        *,
        available: bool,
    ) -> tuple[str, bool]:
        """Explain computed results directly without a second generation pass."""
        if query.response_mode != "detailed" or not available:
            return base_answer, False
        return _structured_detail_fallback(query.task_type, base_answer, analysis_context, intent=intent), False

    if viewer_analysis is not None and query.task_type in ("pose_recomm", "trajectory_recomm") and not knowledge_routed:
        declined = unsupported_pose_part(query.prompt) if query.task_type == "pose_recomm" else None
        if declined is not None:
            return _success_response(query, request_started, answer=declined, retrieval_used=False, citations=[],
                                     context_status="not_available")
        analysis_context = (_current_viewer_pose_context(query.prompt, viewer_analysis)
                            if query.task_type == "pose_recomm" else viewer_analysis)
        available = analysis_context.get("scope") != "current_zone" or analysis_context.get("status") == "available"
        verified_answer = _viewer_answer(query.task_type, query.prompt, analysis_context, intent)
        arm_pose = parse_viewer_arm_pose(query.context) if query.task_type == "pose_recomm" else None
        side = asked_arm(query.prompt, intent) if query.task_type == "pose_recomm" else None
        if arm_pose is not None:
            arm_answer = arm_pose_answer(
                arm_pose, current=analysis_context.get("scope") == "current_zone",
                playback=viewer_analysis.get("playback_context"),  # type: ignore[arg-type]
                zone_id=requested_zone(query.prompt, None), side=side,
            )
            # 자세 유사도(시퀀스별 팔 관절 채점)를 먼저, 전문가 기록 대비 작업 유사도를 이어서 답한다.
            # 팔 방향을 물으면 팔 채점 결과만 답한다 — 작업 유사도의 개선 안내에는 팔 방향 항목이 없다.
            verified_answer = arm_answer if side is not None else arm_answer + "\n" + verified_answer
            analysis_context = {**analysis_context, "arm_pose": {key: item for key, item in arm_pose.items() if key != "sequences"}}
            available = available or arm_pose.get("status") == "available"
        elif side is not None:
            # 팔 채점이 없으면 작업 개선 안내만 남는다. 거기에 팔별 항목이 없다는 말이 "팔에 문제가 없다"로 읽히지 않게 밝힌다.
            verified_answer += "\n팔 자세 채점 결과를 받지 못해 팔별 교정 여부는 이 답에 없습니다. 자세 채점이 끝난 뒤 다시 질문해 주세요."
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
    if query.task_type == "pose_recomm" and not knowledge_routed and viewer_analysis is None:
        declined = unsupported_pose_part(query.prompt)
        if declined is not None:
            return _success_response(query, request_started, answer=declined, retrieval_used=False, citations=[],
                                     context_status="not_available")
        if context and not context.isdigit() and asks_whole_session(query.prompt, intent):
            context = ""  # 세션 전체를 묻는 질문은 붙인 시퀀스 대신 세션 결과로 답한다
    found = (
        await to_thread.run_sync(lambda: reports.latest_complete_analysis(session_id))
        if reports is not None and session_id and query.task_type in ("pose_recomm", "trajectory_recomm")
        # 순서 답은 세션 단위라 붙인 시퀀스와 무관하다. 자세는 시퀀스를 붙이면 그 분사를 채점한다.
        and (query.task_type == "trajectory_recomm" or not context or context.isdigit())
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
    working_zone = live_zone = None
    if found is None and session_id and not context and query.task_type == "pose_recomm" and not knowledge_routed:
        # 훈련 중에는 지금 작업하는 구역으로 답한다. 세션 전체를 묻는 질문만 올라온 분사 전체를 본다.
        working_zone = await to_thread.run_sync(lambda: current_zone(sequences, session_id))
        live_zone = requested_zone(query.prompt, query.context) or (
            None if asks_whole_session(query.prompt, intent) else working_zone)
    if found is None and session_id and (not context or query.task_type == "trajectory_recomm") and query.task_type in ("pose_recomm", "trajectory_recomm") and not knowledge_routed and (reports is not None or query.task_type == "trajectory_recomm" or asks_about_zones(query.prompt, query.context)):
        # Keep explicit-sequence pose queries on their original scoring path.
        found = await to_thread.run_sync(lambda: live_analysis(
            sequences, session_id=session_id, scenario_id=query.scenario_id,
            task_type=query.task_type, sequence_engine=sequence_engine,
            mcp_base_url=mcp_base_url, transport=transport,
            zone_id=live_zone,
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
            answer_pose_question(found, prompt=query.prompt, context=live_zone or query.context, intent=intent)
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
            base_answer = ("지금 작업 중인 구역 기준입니다. " if live_zone and live_zone == working_zone
                           else "" if live_zone else "지금까지 올라온 분사 기준입니다. ") + base_answer
        answer, detail_degraded = await _synthesize_structured_detail(
            base_answer, zone.context, available=zone.analysis_status == "available"
        )
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
            answer = answer_sequence_question(pose, prompt=query.prompt, intent=intent)
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
    recent_turns: list[dict[str, str]] = dialogue_turns
    if query.session_id:
        session_summary, recent_turns = await to_thread.run_sync(
            lambda: (
                summarize_session(sequences.session_bodies(query.session_id),
                                  selected_sequence_id=_selected_sequence(query.context)),
                dialogue_turns,
            )
        )
    definition_context = (viewer_definition_context(query.prompt, query.context)
                          if query.response_mode == "quick" and query.task_type == "knowledge_search" else None)
    knowledge_client_context = (definition_context or (json.dumps(facts, ensure_ascii=False, separators=(",", ":")) if facts else viewer_metadata_context(query.context))) if viewer_requested else (definition_context or query.context)
    if coating_advice_context is not None:
        knowledge_client_context = "[검증된 최종 도막 비교 자료]\n" + json.dumps(coating_advice_context, ensure_ascii=False, separators=(",", ":"))
        knowledge_client_context += "\n최종 기록의 전문가 대비 대표 두께 비교이며 재생 시점의 현재 도막 판정이 아닙니다. 질문의 원인·방법·관계를 직접 설명하고 확인된 측정 사실과 일반적인 가능한 원인을 구분하세요. 권장 Zone 순서를 대신 답하지 마세요."
    instruction = build_query_instruction(
        query.prompt,
        client_context=knowledge_client_context,
        session_summary=session_summary,
        recent_turns=[] if definition_context is not None else recent_turns,
        response_mode=query.response_mode,
        concise=query.response_mode == "detailed",
    )
    generation_instruction = (build_query_instruction(
        query.prompt,
        client_context=knowledge_client_context,
        session_summary=session_summary,
        recent_turns=[] if definition_context is not None else recent_turns,
        response_mode=query.response_mode,
        concise=True,
    ) if query.response_mode == "quick" else instruction)

    def _guarded_run() -> Mapping[str, object]:
        with generation_slot_for(pipeline):
            return pipeline.run(
                instruction=instruction,
                force_route="fast",
                use_pipeline_history=False,
                session_id=query.session_id,
                enable_thinking=False,
                use_rag=use_rag,
                max_tokens=128,
                **({"compact_analysis": True} if query.response_mode == "detailed" else {}),
                request_id=query.request_id,
                **({
                    "quick_cache_scope": json.dumps([query.user_id, query.scenario_id, query.task_type]),
                    "concise_answer": True,
                    "generation_instruction": generation_instruction,
                    **({"retrieval_instruction": query.prompt}
                       if getattr(pipeline, "quick_query_runtime", None) is not None else {}),
                } if query.response_mode == "quick" else {
                    # 응답 수준 지시문이 검색어에 섞이면 무관한 문단이 잡힌다(Issue #9).
                    "retrieval_instruction": retrieval_question(query.prompt, context=instruction),
                }),
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
        coating_context=coating_advice_context,
        context_status="available" if session_summary or coating_advice_context or facts else "not_available",
        generated=True,
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
    coating_engine: object | None = None,
    conversation_id: str | None = None,
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
        # These language pairs are more reliable from English than from Korean
        # on the existing 27B translator. Validate values and language on BOTH
        # legs; the Korean analysis and public response language stay unchanged.
        if source == "ko" and target in {"ne", "my", "si", "km"}:
            english = await _translate(text, "ko", "en")
            return await _translate(english, "en", target)
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
            coating_engine=coating_engine, conversation_id=conversation_id,
        )
        if response.status_code != 200:
            return response
        body = json.loads(response.body)
        data = body["data"]
        # 분석 답은 항상 한국어다. 모델이 섞어 낸 한자는 번역 전에 뺀다.
        data["answer"] = await _translate(without_hanzi(data["answer"], prompt=normalized), "ko", response_language)
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
    try:
        sequences.record_query_turn(
            session_key=query.session_id or "CHAT:" + query.user_id, user_id=query.user_id,
            prompt=query.prompt, answer=data["answer"], task_type=query.task_type,
            created_at=data["generated_at"], conversation_key=conversation_id,
        )
    except Exception:
        logger.exception("copilot query: failed to store dialogue request_id=%s", query.request_id)
    return JSONResponse(status_code=200, content=body,
                        headers={ANSWER_SOURCE_HEADER: response.headers.get(ANSWER_SOURCE_HEADER, "analysis")})


def create_copilot_router(
    pipeline_provider: PipelineProvider | None,
    *,
    pending_answers: Mapping[str, str] = PENDING_TASK_ANSWERS,
    store: SequenceStore | None = None,
    mcp_base_url: str | None = None,
    transport: object | None = None,
    reports: V1Store | None = None,
    sequence_engine: SequenceAnalysisEngine | None = None,
    coating_engine: object | None = None,
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
            "- **상세 분석 (`detailed`) · Qwen3.6-27B 모델**: 기존 27B 빠른 답변 수준으로 결론·근거·조치를 간결하게 분석합니다(2문장 또는 최대 2개 짧은 항목).\n"
            "- **대화 맥락**: 일반 지식·일상 대화·계산·후속 질문도 처리합니다. 탭 선택이 질문 의도를 강제하지 않습니다. 도장 지식이 필요한 경우에 RAG를 사용합니다.\n"
            "  - 선택 헤더 `X-Copilot-Conversation`: 채팅 창 식별자(영문·숫자·`._:-`, 최대 128자). 사용자·훈련 세션과 함께 대화 이력을 분리합니다. 생략하면 기본 대화를 사용합니다. 인증 수단은 아닙니다.\n"
            "  - 선택 헤더 `X-Copilot-Source`: `motion-viewer`(3D VIEWER), `assistant-test`(TEST). DB 표시용 출처 힌트이며 훈련 대상 작업자 ID와 별개입니다.\n"
            "- **훈련 전 기준 순서**: `scenario_id`가 확인되고 등록된 기준이 있으면 시퀀스·리포트 없이도 Zone 작업 순서를 안내합니다. 공정이 불명확하면 등록된 스프레이·블라스팅 기준을 구분합니다.\n"
            "  - 기준 순서 조회는 실제 방문·준수율·도막 완료 판정이 아닙니다. 현재 다음 Zone 이동 질문은 저장된 시퀀스와 도막 완료 상태로 판단합니다.\n"
            "- **질문 언어**: `language`로 지정합니다. 생략 시 `ko`입니다.\n"
            "- **답변 언어**: `response_language`로 지정하며, 생략/null이면 질문 언어를 따릅니다.\n"
            "- 외국어 질문을 한국어 분석·검색으로 연결하고 최종 `answer`를 요청 언어로 번역합니다.\n"
            "- 점수·단위·Zone 순서·구조화 context·citations는 유지합니다. 번역 처리 시간이 추가됩니다.\n"
            "  - 일부 언어는 번역 품질을 위해 기존 Qwen3.6-27B 모델을 사용합니다.\n"
            "- 번역에 실패하거나 수치가 달라지면 한국어를 외국어 응답으로 표시하지 않고 503을 반환합니다."
        ),
    )
    async def copilot_query(request: Request, query: Annotated[CopilotQueryRequest, Body(openapi_examples={
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
            conversation_id=conversation_key(request.headers.get("x-copilot-conversation")),
            pipeline_provider=pipeline_provider,
            pending_answers=pending_answers,
            sequences=sequences,
            mcp_base_url=mcp_base,
            transport=transport,
            reports=reports,
            sequence_engine=sequence_engine,
            coating_engine=coating_engine,
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
