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
import os
import re
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from anyio import to_thread
from fastapi import APIRouter, FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from trainer.demo_api.copilot_contracts import CopilotQueryData, CopilotQueryRequest, CopilotResponse
from trainer.demo_api.copilot_intent import classify_question_intent
from trainer.demo_api.copilot_session_context import build_query_instruction, summarize_session
from trainer.demo_api.pose_query import (
    answer_order_question,
    answer_pose_question,
    asks_about_zones,
    score_pose,
    zone_unavailable,
)
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.sequence_analysis import SequenceAnalysisEngine
from trainer.demo_api.v1_store import V1Store
from trainer.scripts.generation_slot import GENERATION_SLOT
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


def _parse_motion_viewer_context(value: str | None) -> tuple[str, dict[str, object]] | None:
    """Read the compact, already-verified analysis attached by the motion viewer."""
    if not value or not value.startswith(_MOTION_VIEWER_CONTEXT_PREFIX):
        return None
    try:
        payload = json.loads(value[len(_MOTION_VIEWER_CONTEXT_PREFIX):])
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    verified_answer = payload.get("verified_answer")
    analysis_context = payload.get("analysis_context")
    if not isinstance(verified_answer, str) or not verified_answer.strip():
        return None
    if not isinstance(analysis_context, dict):
        return None
    return verified_answer.strip(), analysis_context


def _percent_text(value: object) -> str | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if 0 <= number <= 1:
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
            + "\n\n[권장 조치]\n" + action
        )
    elif task_type == "pose_recomm":
        similarity = _percent_text(analysis_context.get("similarity_percent"))
        comparison = [f"전체 자세 유사도는 {similarity}입니다."] if similarity else []
        zones = analysis_context.get("zones")
        zone_rows = list(zones.values()) if isinstance(zones, dict) else zones
        worst_text = None
        if isinstance(zone_rows, list):
            candidates = [row for row in zone_rows if isinstance(row, dict)]
            scored = [
                (float(row.get("similarity_percent", 100)), row)
                for row in candidates
                if isinstance(row.get("similarity_percent"), (int, float))
            ]
            if scored:
                _, worst = min(scored, key=lambda item: item[0])
                score = _percent_text(worst.get("similarity_percent"))
                worst_text = f"가장 낮은 구역은 Zone {worst.get('zone_id')}({score})입니다."
                comparison.append(worst_text)
        corrections = analysis_context.get("corrections")
        messages: list[str] = []
        if isinstance(corrections, list):
            messages = [
                str(item.get("message")) if isinstance(item, dict) and item.get("message") else str(item)
                for item in corrections[:3]
            ]
            messages = [message for message in messages if message and message != "None"]
        verdict = "전문가 자세와의 차이를 Zone별로 확인했습니다."
        if interpretation is None:
            interpretation = worst_text or base_answer
        action = " ".join(messages) if messages else "현재 자세를 유지하되 몸통 기울기와 양손 위치를 다음 Zone에서도 일정하게 유지하세요."
        return (
            "[상세 분석]\n[종합 판단]\n" + verdict
            + "\n\n[정량 비교]\n" + ("\n".join(comparison) or "비교 가능한 자세 지표가 없습니다.")
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
    store: SequenceStore | None = None,
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
    if store is not None and query.session_id:
        try:
            store.record_query_turn(
                session_key=query.session_id,
                user_id=query.user_id,
                prompt=query.prompt,
                answer=answer,
                task_type=query.task_type,
                created_at=generated_at,
            )
        except Exception:  # noqa: BROAD_EXCEPT_OK - chat memory must not break the API response
            logger.exception("copilot query: failed to store dialogue request_id=%s", query.request_id)
    return JSONResponse(status_code=200, content=envelope.model_dump(mode="json"))


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
    """Execute one normalized query for both text and voice transports."""
    request_started = time.monotonic() if started is None else started

    intent: str | None = None
    intent_pipeline: object | None = None
    if query.task_type in ("pose_recomm", "trajectory_recomm"):
        classifier = None
        if pipeline_provider is not None:
            try:
                intent_pipeline = await to_thread.run_sync(pipeline_provider)
                method = getattr(intent_pipeline, "classify_copilot_intent", None)
                if callable(method):
                    def classifier(task_type: str, prompt: str) -> str:
                        with GENERATION_SLOT:
                            return str(method(task_type, prompt))
            except Exception:  # noqa: BROAD_EXCEPT_OK - keyword fallback keeps structured answers available
                logger.exception("copilot intent pipeline unavailable request_id=%s", query.request_id)
        intent = await to_thread.run_sync(
            lambda: classify_question_intent(query.task_type, query.prompt, classifier)
        )
        logger.info("copilot intent request_id=%s task=%s intent=%s", query.request_id, query.task_type, intent)
    # If the model is unavailable, keep the selected structured tab operational;
    # only a successful semantic-model classification may cross-route to RAG.
    knowledge_routed = intent == "knowledge" and classifier is not None
    viewer_analysis = _parse_motion_viewer_context(query.context)

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
                with GENERATION_SLOT:
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
            interpretation = _detail_interpretation(base_answer, candidate) if candidate else None
            return _structured_detail_fallback(
                query.task_type,
                base_answer,
                analysis_context,
                interpretation=interpretation,
            ), bool(result.get("deadline_hit", False))
        except Exception:  # noqa: BROAD_EXCEPT_OK - verified deterministic answer remains available
            logger.exception(
                "copilot structured detail synthesis failed request_id=%s task_type=%s",
                query.request_id,
                query.task_type,
            )
        return _structured_detail_fallback(query.task_type, base_answer, analysis_context), True

    # Expert CSV records shown by the motion viewer do not necessarily have a
    # sessions/complete report.  The viewer therefore attaches the compact
    # structured analysis it is already rendering.  Keep the server response as
    # the single source of truth: quick returns it verbatim, detailed lets the
    # LLM explain the same verified facts without inventing new measurements.
    if (
        viewer_analysis is not None
        and query.task_type in ("pose_recomm", "trajectory_recomm")
        and not knowledge_routed
    ):
        verified_answer, analysis_context = viewer_analysis
        answer, detail_degraded = await _synthesize_structured_detail(
            verified_answer, analysis_context, available=True
        )
        return _success_response(
            query,
            request_started,
            answer=answer,
            retrieval_used=False,
            citations=[],
            degraded=detail_degraded,
            pose_context=analysis_context if query.task_type == "pose_recomm" else None,
            sequence_context=analysis_context if query.task_type == "trajectory_recomm" else None,
            context_status="available",
            store=sequences,
        )

    context = (query.context or "").strip()
    # 시퀀스(`SEQ-012`)를 가리키지 않은 자세·순서 질문은 complete 가 남긴 세션 결과로 답한다(pose_query 참고).
    # complete 뒤에는 질문에 "구역"이 없어도 그렇다 — 뷰어 채팅은 칩으로 유형만 고른다.
    session_id = query.session_id
    found = (
        await to_thread.run_sync(lambda: reports.latest_complete_analysis(session_id))
        if reports is not None and session_id and query.task_type in ("pose_recomm", "trajectory_recomm")
        and (not context or context.isdigit())
        else None
    )
    if query.task_type == "trajectory_recomm" and not knowledge_routed and reports is not None and session_id:
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
            store=sequences,
        )

    if query.task_type == "pose_recomm" and not knowledge_routed and reports is not None and (
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
        answer, detail_degraded = await _synthesize_structured_detail(
            zone.answer, zone.context, available=zone.analysis_status == "available"
        )
        return _success_response(
            query, request_started, answer=answer, retrieval_used=False, citations=[],
            degraded=detail_degraded,
            pose_context=zone.context,
            context_status="available" if zone.analysis_status == "available" else "not_available",
            store=sequences,
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
            degraded=detail_degraded,
            pose_context=pose.as_context(),
            context_status="available" if pose.analysis_status == "available" else "not_available",
            store=sequences,
        )

    if query.task_type != "knowledge_search" and not knowledge_routed:
        answer = pending_answers.get(query.task_type) or PENDING_TASK_ANSWERS[query.task_type]
        return _success_response(
            query, request_started, answer=answer, retrieval_used=False, citations=[], store=sequences
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
    instruction = build_query_instruction(
        query.prompt,
        client_context=query.context,
        session_summary=session_summary,
        recent_turns=recent_turns,
        response_mode=query.response_mode,
    )

    def _guarded_run() -> Mapping[str, object]:
        with GENERATION_SLOT:
            return pipeline.run(
                instruction=instruction,
                force_route="fast",
                session_id=query.session_id,
                enable_thinking=False,
                use_rag=use_rag,
                max_tokens=96 if query.response_mode == "quick" else 256,
                request_id=query.request_id,
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
        store=sequences,
    )


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
        description="Replay 및 일반 UI용 텍스트 질의입니다. 빠른 응답을 위해 27B 모델의 thinking을 끄고 RAG 근거를 함께 반환합니다.",
    )
    async def copilot_query(query: CopilotQueryRequest) -> JSONResponse:
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
            return error_response(422, "Invalid request")
        return await request_validation_exception_handler(request, exc)

    app.add_exception_handler(RequestValidationError, _on_validation_error)
