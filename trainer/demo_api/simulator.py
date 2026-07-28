# pyright: reportUnusedFunction=false
from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from trainer.demo_api import simulator_stub as stub
from trainer.demo_api.monitor import MonitorRecorder
from trainer.demo_api.sim_contracts import Phase, SessionEventInput, SessionStartInput
from trainer.scripts.session_coaching import SessionSummary

TInput = TypeVar("TInput", bound=BaseModel)


def create_simulator_router(recorder: MonitorRecorder) -> APIRouter:
    """Deterministic three-phase simulator interface that records every exchange.

    Every call writes exactly two monitor events - the raw request body as
    ``input`` and the produced response as ``output`` - so rejected payloads stay
    observable instead of disappearing behind a 422.
    """
    router = APIRouter(prefix="/api/sim", tags=["simulator"])

    async def _run_phase(
        request: Request,
        phase: Phase,
        path_session_id: str | None,
        model_cls: type[TInput],
        build_output: Callable[[TInput], BaseModel],
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

        output_payload: dict[str, object] = build_output(payload).model_dump(mode="json")
        _ = recorder.record(session_id, phase, "output", output_payload)
        return JSONResponse(status_code=200, content=output_payload)

    @router.post("/sessions/start")
    async def start(request: Request) -> JSONResponse:
        return await _run_phase(request, "start", None, SessionStartInput, stub.build_start_output)

    @router.post("/sessions/{session_id}/events")
    async def events(session_id: str, request: Request) -> JSONResponse:
        return await _run_phase(
            request,
            "runtime",
            session_id,
            SessionEventInput,
            lambda payload: stub.build_event_output(session_id, payload),
        )

    @router.post("/sessions/{session_id}/complete")
    async def complete(session_id: str, request: Request) -> JSONResponse:
        return await _run_phase(
            request, "session_end", session_id, SessionSummary, stub.build_review_output
        )

    return router
