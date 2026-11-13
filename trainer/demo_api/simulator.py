# pyright: reportUnusedFunction=false
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from trainer.demo_api import simulator_stub as stub
from trainer.demo_api.monitor import MonitorRecorder
from trainer.demo_api.sim_contracts import (
    Phase,
    SessionEventInput,
    SessionEventOutput,
    SessionStartInput,
    SessionReviewOutput,
    SessionStartOutput,
    SimSessionResponse,
    SimSessionView,
)
from trainer.demo_api.sim_session import SimSessionState, SimSessionStore
from trainer.demo_api.zone_resolver import ZoneResolver
from trainer.scripts.session_coaching import SessionSummary

TInput = TypeVar("TInput", bound=BaseModel)
TOutput = TypeVar("TOutput", bound=BaseModel)


def _to_view(state: SimSessionState) -> SimSessionView:
    return SimSessionView(
        session_id=state.session_id,
        level_id=state.level_id,
        phase=state.phase,
        recommended_flow=list(state.recommended_flow),
        current_zone_id=state.current_zone_id,
        completed_zone_ids=list(state.completed_zone_ids),
        last_event_id=state.last_event_id,
        anomaly_count=state.anomaly_count,
        unresolved_zone_id=state.unresolved_zone_id,
        updated_at=state.updated_at,
    )


def create_simulator_router(
    recorder: MonitorRecorder, store: SimSessionStore, resolver: ZoneResolver
) -> APIRouter:
    """Deterministic three-phase simulator interface that records every exchange.

    Every call writes exactly two monitor events - the raw request body as
    ``input`` and the produced response as ``output`` - so rejected payloads stay
    observable instead of disappearing behind a 422. Accepted calls additionally
    advance ``store``, which is what the sequence snapshot and the chat
    ``environment`` read from.
    """
    router = APIRouter(prefix="/api/sim", tags=["simulator"])

    async def _run_phase(
        request: Request,
        phase: Phase,
        path_session_id: str | None,
        model_cls: type[TInput],
        build_output: Callable[[TInput], TOutput],
        on_success: Callable[[TInput, TOutput], Awaitable[None]] | None = None,
    ) -> JSONResponse:
        error: dict[str, object]

        try:
            raw: object = await request.json()
        except ValueError:
            session_id = path_session_id or "unknown"
            error = {"error": "invalid_json"}
            _ = recorder.record(session_id, phase, "input", dict(error))
            _ = recorder.record(session_id, phase, "output", error)
            return JSONResponse(status_code=422, content=error)

        body: dict[str, object] = raw if isinstance(raw, dict) else {}
        session_id = path_session_id or str(body.get("session_id", "")) or "unknown"
        _ = recorder.record(session_id, phase, "input", body)

        if path_session_id is not None and "session_id" in body and body["session_id"] != path_session_id:
            error = {
                "error": "session_id_mismatch",
                "path": path_session_id,
                "body": body["session_id"],
            }
            _ = recorder.record(session_id, phase, "output", error)
            return JSONResponse(status_code=422, content=error)

        try:
            payload = model_cls.model_validate(body)
        except ValidationError as exc:
            # url/context off: pydantic puts raw exception objects in ctx, which
            # JSONResponse could not serialize.
            error = {
                "error": "validation_error",
                "detail": exc.errors(include_url=False, include_context=False),
            }
            _ = recorder.record(session_id, phase, "output", error)
            return JSONResponse(status_code=422, content=error)

        output = build_output(payload)
        output_payload: dict[str, object] = output.model_dump(mode="json")
        # 상태 전이를 output 이벤트보다 먼저 확정한다. 구독자는 이 이벤트를 재조회
        # 신호로 쓰므로, 순서가 반대면 갱신 전 상태를 읽고 다시 받을 신호가 없다.
        if on_success is not None:
            await on_success(payload, output)
        _ = recorder.record(session_id, phase, "output", output_payload)
        return JSONResponse(status_code=200, content=output_payload)

    @router.post("/sessions/start")
    async def start(request: Request) -> JSONResponse:
        async def _apply(payload: SessionStartInput, output: SessionStartOutput) -> None:
            _ = store.start(
                session_id=payload.session_id,
                level_id=payload.level_id,
                recommended_flow=output.recommended_flow,
            )

        return await _run_phase(
            request, "start", None, SessionStartInput, stub.build_start_output, _apply
        )

    @router.post("/sessions/{session_id}/events")
    async def events(session_id: str, request: Request) -> JSONResponse:
        async def _apply(payload: SessionEventInput, output: SessionEventOutput) -> None:
            target = payload.zone_id
            if target is None:
                coordinates = (
                    payload.telemetry.coordinates if payload.telemetry is not None else None
                )
                if coordinates is not None:
                    target = await resolver.resolve(list(coordinates))
            _ = store.runtime(
                session_id=session_id,
                event_id=payload.event_id,
                target_zone_id=target,
                anomaly=output.kind == "anomaly",
            )

        return await _run_phase(
            request,
            "runtime",
            session_id,
            SessionEventInput,
            lambda payload: stub.build_event_output(session_id, payload),
            _apply,
        )

    @router.post("/sessions/{session_id}/complete")
    async def complete(session_id: str, request: Request) -> JSONResponse:
        async def _apply(payload: SessionSummary, output: SessionReviewOutput) -> None:
            _ = (payload, output)
            _ = store.end(session_id=session_id)

        return await _run_phase(
            request, "session_end", session_id, SessionSummary, stub.build_review_output, _apply
        )

    @router.get("/session")
    def session_state() -> JSONResponse:
        state = store.current()
        response = SimSessionResponse(session=None if state is None else _to_view(state))
        return JSONResponse(content=response.model_dump(mode="json"))

    return router
