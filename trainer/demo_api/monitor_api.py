# pyright: reportUnusedFunction=false
from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Final

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from trainer.demo_api.monitor import MonitorRecorder

_REPLAY_LIMIT: Final[int] = 10_000


def _frame(event: str, data: dict[str, object]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


async def monitor_sse_body(recorder: MonitorRecorder, cursor: int) -> AsyncGenerator[bytes, None]:
    """Replay buffered events after ``cursor``, then forward live ones.

    Subscribing before taking the snapshot is what makes the handoff lossless:
    an event recorded between the two lands in the subscriber queue and is
    de-duplicated by seq, instead of falling into the gap between them.
    """
    send, receive = recorder.subscribe()
    try:
        snapshot = recorder.snapshot(cursor=cursor, limit=_REPLAY_LIMIT)
        if snapshot.dropped_before is not None:
            yield _frame("gap", {"dropped_before": snapshot.dropped_before})

        last = cursor
        for event in snapshot.events:
            yield _frame("monitor", event.model_dump(mode="json"))
            last = event.seq

        async with receive:
            async for event in receive:
                if event.seq <= last:
                    continue
                yield _frame("monitor", event.model_dump(mode="json"))
                last = event.seq
    finally:
        recorder.unsubscribe(send)
        send.close()


def create_monitor_router(recorder: MonitorRecorder) -> APIRouter:
    router = APIRouter(prefix="/api/monitor", tags=["monitor"])

    @router.get("/events")
    def events(cursor: int = 0, limit: int = 200) -> JSONResponse:
        snapshot = recorder.snapshot(cursor=cursor, limit=limit)
        return JSONResponse(content=snapshot.model_dump(mode="json"))

    @router.get("/stream")
    async def stream(cursor: int = 0) -> StreamingResponse:
        return StreamingResponse(
            monitor_sse_body(recorder, cursor),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return router
