"""FastAPI entry point for the separate corrected CPU engine on port 8122."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .assets import ZoneAssetCatalog
from .contracts import EventRequest
from .engine import CorrectedZoneEngine
from .errors import EngineError


DEFAULT_DB_PATH = "/state/engine.db"
DEFAULT_MAX_BODY_BYTES = 64 * 1024 * 1024


def _positive_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be positive")
    return value


def build_engine_from_env() -> CorrectedZoneEngine:
    db_path = Path(os.environ.get("ZONE_ENGINE_DB", DEFAULT_DB_PATH))
    asset_root_raw = os.environ.get("ZONE_ASSET_ROOT")
    manifest_path = (
        Path(__file__).resolve().parent / "assets" / "zone_asset_manifest.json"
    )
    assets = ZoneAssetCatalog(
        manifest_path,
        asset_root=Path(asset_root_raw) if asset_root_raw else None,
    )
    return CorrectedZoneEngine(db_path, assets=assets)


def create_app(engine: CorrectedZoneEngine | None = None) -> FastAPI:
    app = FastAPI(
        title="SprayXR corrected Zone engine",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.engine = engine or build_engine_from_env()
    app.state.max_body_bytes = _positive_env_int(
        "ZONE_ENGINE_MAX_BODY_BYTES", DEFAULT_MAX_BODY_BYTES
    )

    @app.exception_handler(EngineError)
    async def engine_error_handler(_request: Request, exc: EngineError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "reason_code": exc.reason_code,
                    "message": exc.message,
                }
            },
        )

    @app.exception_handler(ValidationError)
    async def payload_validation_error_handler(
        _request: Request, exc: ValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "reason_code": "event_payload_contract_invalid",
                    "detail": exc.errors(),
                }
            },
        )

    @app.post("/internal/v1/events")
    async def events(request: Request) -> JSONResponse:
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={"error": {"reason_code": "content_length_invalid"}},
                )
            if declared_length > app.state.max_body_bytes:
                return JSONResponse(
                    status_code=413,
                    content={"error": {"reason_code": "request_body_too_large"}},
                )
        chunks: list[bytes] = []
        total = 0
        async for chunk in request.stream():
            total += len(chunk)
            if total > app.state.max_body_bytes:
                return JSONResponse(
                    status_code=413,
                    content={"error": {"reason_code": "request_body_too_large"}},
                )
            chunks.append(chunk)
        raw = b"".join(chunks)
        try:
            document: Any = json.loads(raw)
            event = EventRequest.model_validate(document)
        except (json.JSONDecodeError, UnicodeDecodeError, ValidationError) as exc:
            detail = exc.errors() if isinstance(exc, ValidationError) else str(exc)
            return JSONResponse(
                status_code=422,
                content={
                    "error": {
                        "reason_code": "event_contract_invalid",
                        "detail": detail,
                    }
                },
            )
        response = await run_in_threadpool(app.state.engine.process_event, event)
        return JSONResponse(status_code=200, content=response)

    @app.get("/internal/v1/sessions/{session_id}")
    async def session(session_id: str) -> JSONResponse:
        result = await run_in_threadpool(app.state.engine.get_session, session_id)
        return JSONResponse(status_code=200, content=result)

    @app.get("/internal/v1/assessments/{assessment_id}")
    async def assessment(assessment_id: str) -> JSONResponse:
        result = await run_in_threadpool(app.state.engine.get_assessment, assessment_id)
        return JSONResponse(status_code=200, content=result)

    @app.get("/internal/v1/health")
    async def health() -> JSONResponse:
        result = await run_in_threadpool(app.state.engine.health)
        return JSONResponse(status_code=200, content=result)

    return app


def main() -> None:
    import uvicorn

    host = os.environ.get("ZONE_ENGINE_HOST", "0.0.0.0")
    port = _positive_env_int("ZONE_ENGINE_PORT", 8122)
    uvicorn.run(create_app(), host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
