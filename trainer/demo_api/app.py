# pyright: reportUnusedFunction=false, reportAny=false, reportUnusedCallResult=false
from __future__ import annotations

import argparse
import logging
import os
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import cast

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from trainer.demo_api import chat_service as chat_service_mod
from trainer.demo_api.chat_service import ChatService, build_default_live_runner
from trainer.demo_api.contracts import ChatRequest
from trainer.demo_api.copilot_router import create_copilot_router, install_copilot_error_handlers
from trainer.demo_api.telemetry_router import create_telemetry_router
from trainer.demo_api.v1_router import create_v1_router
from trainer.demo_api.monitor import MonitorRecorder
from trainer.demo_api.monitor_api import create_monitor_router
from trainer.demo_api.readiness import evaluate_live_readiness
from trainer.demo_api.sequence import default_snapshot, session_snapshot
from trainer.demo_api.sim_session import SimSessionStore
from trainer.demo_api.simulator import create_simulator_router
from trainer.demo_api.zone_resolver import McpZoneResolver

logger = logging.getLogger(__name__)
_LEVEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\-_.]{0,63}$")


def create_app(
    chat_service: ChatService | None = None,
    *,
    probe_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    recorder = MonitorRecorder()
    store = SimSessionStore()
    mcp_base_url = os.environ.get("MCP_BASE_URL", "http://localhost:8100")
    resolver = McpZoneResolver(mcp_base_url)

    if chat_service is not None:
        service = chat_service
    else:
        live_mode = os.environ.get("DEMO_LIVE_MODE", "off")
        if live_mode not in {"off", "on"}:
            raise RuntimeError("DEMO_LIVE_MODE must be 'off' or 'on'")

        logger.info("DEMO_LIVE_MODE=%s", live_mode)

        if live_mode == "on":
            service = ChatService(live_runner=build_default_live_runner(store))
        else:
            service = ChatService()
    live = service.live_runner is not None

    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
        if live and os.environ.get("LIVE_EAGER_INIT") == "1":
            app.state.eager_init_thread = chat_service_mod.start_eager_pipeline_init()
        yield

    app = FastAPI(title="Ship Painting Demo API", lifespan=_lifespan)
    app.state.chat_service = service
    app.state.eager_init_thread = None

    raw_allow_origins = os.environ.get("DEMO_UI_ORIGINS")
    if raw_allow_origins is None:
        allow_origins = ["*"]
    else:
        allow_origins = [origin.strip() for origin in raw_allow_origins.split(",") if origin.strip()]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # 계약 밖. Swagger 는 계약 15개만 보여야 한다 — demo-ui 와 compose 헬스체크가 쓰므로 라우트는 남긴다.
    @app.get("/api/health", include_in_schema=False)
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "servers": 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @app.get("/api/ready", include_in_schema=False)
    async def ready() -> JSONResponse:
        body: dict[str, object] = {"ready": True, "mode": "live" if live else "demo", "checks": {}}
        if live:
            is_ready, checks = await evaluate_live_readiness(
                pipeline=chat_service_mod.loaded_pipeline(),
                mcp_base_url=mcp_base_url,
                transport=probe_transport,
            )
            body.update(ready=is_ready, checks=checks)
        body["timestamp"] = datetime.now(timezone.utc).isoformat()
        return JSONResponse(status_code=200 if body["ready"] else 503, content=body)

    @app.get("/api/sequence/{level_id}", include_in_schema=False)
    def sequence(level_id: str) -> dict[str, object]:
        if _LEVEL_ID_PATTERN.fullmatch(level_id) is None:
            raise HTTPException(status_code=400, detail="invalid level_id")
        state = store.current()
        if state is not None and state.level_id == level_id:
            return session_snapshot(state).model_dump(mode="json")
        return default_snapshot(level_id).model_dump(mode="json")

    @app.post("/api/chat", include_in_schema=False)
    async def chat(request: ChatRequest) -> StreamingResponse:
        service = cast(ChatService, app.state.chat_service)

        async def _sse_body() -> AsyncIterator[bytes]:
            async for event in service.stream(request):
                frame = f"event: {event.event}\ndata: {event.model_dump_json()}\n\n"
                yield frame.encode("utf-8")

        return StreamingResponse(
            _sse_body(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    app.state.monitor_recorder = recorder
    app.state.sim_session_store = store
    # v2.5 이전 데모 인터페이스. 계약의 /sessions/*·/telemetry/sequence 가 대응한다.
    app.include_router(create_simulator_router(recorder, store, resolver), include_in_schema=False)
    app.include_router(create_monitor_router(recorder), include_in_schema=False)
    # knowledge_search shares the live pipeline singleton; resolved per call so it is built once, lazily.
    app.include_router(
        create_copilot_router(
            (lambda: chat_service_mod._get_or_build_pipeline()) if live else None,
            mcp_base_url=mcp_base_url,
        )
    )
    # 명세서 15개 중 12개: 세션·사용자·리포트·이상 알림 SSE·테스트.
    app.include_router(create_v1_router())
    # POST /api/v1/telemetry/sequence: 시퀀스를 데이터베이스에 올린다. 자세 채점은
    # copilot/query 의 pose_recomm 이 한다(명세서 p.4 vs p.9).
    # recorder 를 함께 준다 — v2.5 요청·응답이 /api/monitor/{events,stream} 에 보인다.
    app.include_router(create_telemetry_router(recorder=recorder))
    install_copilot_error_handlers(app)

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--host", default="0.0.0.0")
    _ = parser.add_argument("--port", default=8120, type=int)
    _ = parser.add_argument("--reload", action="store_true")
    _ = parser.add_argument("--live", action="store_true")
    raw_args = cast(dict[str, object], vars(parser.parse_args()))
    host = cast(str, raw_args.get("host", "0.0.0.0"))
    port = cast(int, raw_args.get("port", 8120))
    reload_mode = cast(bool, raw_args.get("reload", False))
    live_mode = cast(bool, raw_args.get("live", False))

    if live_mode:
        os.environ["DEMO_LIVE_MODE"] = "on"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger.info("Starting demo API on %s:%s", host, port)
    uvicorn.run("trainer.demo_api.app:app", host=host, port=port, reload=reload_mode)


app = create_app()


if __name__ == "__main__":
    main()
