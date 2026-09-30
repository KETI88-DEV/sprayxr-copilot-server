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
* ``trajectory_recomm`` / ``pose_recomm`` / ``error_analysis`` (plan D2): HTTP 200
  SUCCESS with a short "준비 중" answer and ``context_status="not_available"``.
  Neither the pipeline nor MCP is touched.

Every response uses the v2.5 wrapper. Request validation failures under
``/api/v1/`` are wrapped as ERROR 422; other routes keep FastAPI's format.
"""

from __future__ import annotations

import logging
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
        "pose_recomm": "숙련자 작업 자세 비교 기능은 준비 중입니다. 도장 지식 질문에는 지금 답변할 수 있습니다.",
        "error_analysis": "도막 분포 이상 원인 분석 기능은 준비 중입니다. 도장 지식 질문에는 지금 답변할 수 있습니다.",
    }
)

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
            context_status="not_available",
            sequence_context=None,
            coating_context=None,
            pose_context=None,
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


def create_copilot_router(
    pipeline_provider: PipelineProvider | None,
    *,
    pending_answers: Mapping[str, str] = PENDING_TASK_ANSWERS,
) -> APIRouter:
    """``pipeline_provider`` returns the live pipeline singleton; ``None`` when live mode is off."""
    router = APIRouter()

    # response_model: Swagger 가 계약 응답을 그대로 보여주게 한다. JSONResponse 만 두면
    # 요청 스키마는 나오는데 응답 스키마가 비어 SIM 이 무엇을 받는지 알 수 없다.
    @router.post(COPILOT_QUERY_PATH, response_model=CopilotResponse)
    async def copilot_query(query: CopilotQueryRequest) -> JSONResponse:
        started = time.monotonic()
        if query.task_type != "knowledge_search":
            answer = pending_answers.get(query.task_type) or PENDING_TASK_ANSWERS[query.task_type]
            return _success_response(query, started, answer=answer, retrieval_used=False, citations=[])

        if pipeline_provider is None:
            return error_response(503, "Copilot pipeline is not configured")
        provider = pipeline_provider
        try:
            pipeline = await to_thread.run_sync(provider)
        except Exception:  # noqa: BROAD_EXCEPT_OK - construction failures surface as not ready
            logger.exception("copilot query: pipeline construction failed request_id=%s", query.request_id)
            return error_response(503, "Copilot pipeline is not ready")

        use_rag = query.use_rag == "Y"

        def _guarded_run() -> Mapping[str, object]:
            with GENERATION_SLOT:
                return pipeline.run(
                    instruction=query.prompt,
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

        # ``citations`` is what the answer cites, not what search returned: the parser
        # intersects the model's ids with this request's retrieved chunks and renders the
        # ``[근거]`` line from the survivors, so text and array cannot disagree.
        # ``use_rag='N'`` retrieves nothing, so every document id leaves the prose.
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
            started,
            # ``body``, not ``answer``: the contract's §9.2 example is marker-free prose and
            # the ids travel in ``citations``. The ``[답변]``/``[근거]`` scaffold is the
            # model's training form, not part of the public contract.
            answer=parsed.body,
            retrieval_used=use_rag,
            citations=list(parsed.citations),
            # No full answer was produced, or generation hit its deadline.
            degraded=parsed.repair_path in FAILED_REPAIR_PATHS or bool(result.get("deadline_hit", False)),
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
