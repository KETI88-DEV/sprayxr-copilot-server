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

import logging
import os
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
        return _success_response(
            query, request_started, answer=order.answer, retrieval_used=False, citations=[],
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
        return _success_response(
            query, request_started, answer=zone.answer, retrieval_used=False, citations=[],
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
        return _success_response(
            query, request_started, answer=answer, retrieval_used=False, citations=[],
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

    use_rag = query.use_rag == "Y"
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
    )

    def _guarded_run() -> Mapping[str, object]:
        with GENERATION_SLOT:
            return pipeline.run(
                instruction=instruction,
                force_route="fast",
                session_id=query.session_id,
                enable_thinking=False,
                use_rag=use_rag,
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
        answer=parsed.body,
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
