"""Opt-in public corrected Zone v2 draft routes.

These routes are registered only when the corrected engine URL is configured.
They do not reuse legacy coating status names or silently project corrected
results into v1 responses.
"""

from __future__ import annotations

import asyncio
from typing import Annotated

from anyio import to_thread
from fastapi import APIRouter, HTTPException, Query

from trainer.demo_api.engine_contracts import (
    CONTRACT_VERSION,
    EVALUATION_MODE,
    AcceptedEventResponse,
    AssessmentResponse,
    CopilotQueryRequest,
    CopilotQueryResponse,
    CorrectedRequest,
    ProcessingStatus,
    SequenceRequest,
    SessionCompleteRequest,
    SessionEndRequest,
    SessionReportResponse,
    SessionStartRequest,
    SessionStatusResponse,
    ZoneObservationRequest,
    coating_progress_from_engine_state,
    engine_event_from_request,
)
from trainer.demo_api.engine_store import (
    CorrectedStore,
    CorrectedStoreConflict,
    CorrectedStoreNotFound,
    CorrectedStoreOrderError,
    EventReceipt,
)
from trainer.demo_api.engine_worker import CorrectedOutboxWorker


def _processing(receipt: EventReceipt) -> ProcessingStatus:
    return ProcessingStatus(
        source_cursor=receipt.source_cursor,
        outbox_status=receipt.outbox_status,  # type: ignore[arg-type]
        attempts=receipt.attempts,
        last_error=receipt.last_error,
        last_confirmed_engine_cursor=receipt.last_confirmed_engine_cursor,
        session_revision=receipt.session_revision,
    )


def _accepted(receipt: EventReceipt) -> AcceptedEventResponse:
    return AcceptedEventResponse(
        status="duplicate" if receipt.duplicate else "accepted",
        evaluation_mode=EVALUATION_MODE,
        contract_version=CONTRACT_VERSION,
        event_id=receipt.event_id,
        event_type=receipt.event_type,  # type: ignore[arg-type]
        session_id=receipt.session_id,
        processing=_processing(receipt),
        assessment_id=receipt.assessment_id,
        assessment_revision=receipt.assessment_revision,
        coating_progress=coating_progress_from_engine_state(
            receipt.engine_state,
            source_cursor=receipt.source_cursor,
            confirmed_through_source_cursor=receipt.last_confirmed_engine_cursor,
        ),
        engine_state=receipt.engine_state,
    )


def create_corrected_router(
    *,
    store: CorrectedStore,
    worker: CorrectedOutboxWorker,
    source_instance_id: str,
) -> APIRouter:
    router = APIRouter(prefix="/api/v2", tags=["corrected Zone v2 통합 초안"])

    async def _accept(body: CorrectedRequest) -> EventReceipt:
        event = engine_event_from_request(body, source_instance_id=source_instance_id)
        try:
            receipt = await to_thread.run_sync(lambda: store.accept(body, event))
        except CorrectedStoreConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except CorrectedStoreOrderError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except CorrectedStoreNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        worker.notify()
        return receipt

    async def _wait_for_engine(event_id: str, wait_ms: int) -> EventReceipt:
        deadline = asyncio.get_running_loop().time() + wait_ms / 1000
        receipt = await to_thread.run_sync(lambda: store.event_receipt(event_id))
        while receipt.outbox_status in {"pending", "processing", "failed_retryable"}:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            await asyncio.sleep(min(0.05, remaining))
            receipt = await to_thread.run_sync(lambda: store.event_receipt(event_id))
        return receipt

    @router.post("/sessions/start", response_model=AcceptedEventResponse, status_code=202)
    async def sessions_start(body: SessionStartRequest) -> AcceptedEventResponse:
        return _accepted(await _accept(body))

    @router.post("/telemetry/sequence", response_model=AcceptedEventResponse, status_code=202)
    async def telemetry_sequence(body: SequenceRequest) -> AcceptedEventResponse:
        return _accepted(await _accept(body))

    @router.post("/copilot/analyze-sequence", response_model=AcceptedEventResponse)
    async def analyze_sequence(
        body: SequenceRequest,
        wait_ms: Annotated[int, Query(ge=0, le=5000)] = 1000,
    ) -> AcceptedEventResponse:
        receipt = await _accept(body)
        # Analyze is an idempotent read/wait over the already accepted event.
        # It never inserts a second physical sequence or engine event.
        if wait_ms:
            receipt = await _wait_for_engine(receipt.event_id, wait_ms)
        return _accepted(receipt)

    @router.post("/zones/observations", response_model=AcceptedEventResponse, status_code=202)
    async def zone_observation(body: ZoneObservationRequest) -> AcceptedEventResponse:
        return _accepted(await _accept(body))

    @router.post("/sessions/end", response_model=AcceptedEventResponse, status_code=202)
    async def sessions_end(body: SessionEndRequest) -> AcceptedEventResponse:
        return _accepted(await _accept(body))

    @router.post("/sessions/complete", response_model=AcceptedEventResponse, status_code=202)
    async def sessions_complete(body: SessionCompleteRequest) -> AcceptedEventResponse:
        return _accepted(await _accept(body))

    @router.get("/sessions/{session_id}/status", response_model=SessionStatusResponse)
    async def session_status(session_id: str) -> SessionStatusResponse:
        value = await to_thread.run_sync(lambda: store.session_status(session_id))
        if value is None:
            raise HTTPException(status_code=404, detail="corrected session not found")
        value["coating_progress"] = coating_progress_from_engine_state(
            value["engine_state"],
            source_cursor=value["source_cursor"],
            confirmed_through_source_cursor=value["last_confirmed_engine_cursor"],
        )
        return SessionStatusResponse.model_validate(value)

    @router.get("/assessments/{assessment_id}", response_model=AssessmentResponse)
    async def assessment(assessment_id: str) -> AssessmentResponse:
        value = await to_thread.run_sync(lambda: store.get_assessment(assessment_id))
        if value is None:
            raise HTTPException(status_code=404, detail="corrected assessment not found")
        explanation = await to_thread.run_sync(lambda: store.get_explanation(assessment_id))
        explanation_status = await to_thread.run_sync(
            lambda: store.explanation_status(assessment_id)
        )
        return AssessmentResponse(
            assessment=value,
            explanation=explanation,
            explanation_status=explanation_status,  # type: ignore[arg-type]
        )

    @router.get("/sessions/{session_id}/report", response_model=SessionReportResponse)
    async def session_report(session_id: str) -> SessionReportResponse:
        value = await to_thread.run_sync(lambda: store.session_report(session_id))
        if value is None:
            raise HTTPException(status_code=404, detail="corrected session not found")
        if value["official_summary"] is None:
            raise HTTPException(status_code=409, detail="corrected session is not complete")
        return SessionReportResponse.model_validate(value)

    @router.post("/copilot/query", response_model=CopilotQueryResponse)
    async def copilot_query(body: CopilotQueryRequest) -> CopilotQueryResponse:
        value = await to_thread.run_sync(lambda: store.get_assessment(body.assessment_id))
        if value is None:
            raise HTTPException(status_code=404, detail="corrected assessment not found")
        if value.session_id != body.session_id:
            raise HTTPException(status_code=409, detail="assessment belongs to another session")
        if value.revision != body.revision:
            raise HTTPException(status_code=409, detail="assessment revision mismatch")
        explanation = await to_thread.run_sync(
            lambda: store.get_explanation(body.assessment_id)
        )
        if explanation is None:
            answer_status = "pending"
            answer = "저장된 평가의 설명 생성이 진행 중입니다. 같은 assessment_id와 revision으로 다시 조회해 주세요."
            validation = None
        else:
            answer_status = explanation.status
            answer = explanation.text
            validation = explanation.validation
        await to_thread.run_sync(
            lambda: store.record_query(
                session_id=body.session_id,
                assessment_id=body.assessment_id,
                revision=body.revision,
                question=body.question,
                answer=answer,
                answer_status=answer_status,
            )
        )
        return CopilotQueryResponse(
            session_id=body.session_id,
            assessment_id=body.assessment_id,
            revision=body.revision,
            assessment_type=value.assessment_type,
            answer_status=answer_status,  # type: ignore[arg-type]
            answer=answer,
            evidence_ids=value.evidence_ids,
            explanation_validation=validation,
        )

    return router
