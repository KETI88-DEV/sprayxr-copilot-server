from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class MCPResponse(BaseModel):
    status: str
    result: Mapping[str, object] | None = None
    error: Mapping[str, object] | None = None
    timestamp: str


class MCPError(BaseModel):
    code: str
    message: str
    details: Mapping[str, object] | None = None


class MCPException(Exception):
    code: str
    message: str
    details: Mapping[str, object] | None
    status_code: int

    def __init__(
        self,
        code: str,
        message: str,
        details: Mapping[str, object] | None = None,
        status_code: int = 400,
    ) -> None:
        self.code = code
        self.message = message
        self.details = details
        self.status_code = status_code
        super().__init__(message)


def iso_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def success_response(result: Mapping[str, object]) -> MCPResponse:
    return MCPResponse(status="success", result=result, error=None, timestamp=iso_timestamp())


def error_response(code: str, message: str, details: Mapping[str, object] | None = None) -> MCPResponse:
    payload = MCPError(code=code, message=message, details=details)
    return MCPResponse(status="error", result=None, error=payload.model_dump(), timestamp=iso_timestamp())


async def _handle_mcp_exception(_: Request, exc: Exception) -> JSONResponse:
    mcp_exc = exc if isinstance(exc, MCPException) else MCPException("SERVER_ERROR", "internal error", status_code=500)
    response = error_response(code=mcp_exc.code, message=mcp_exc.message, details=mcp_exc.details)
    return JSONResponse(status_code=mcp_exc.status_code, content=response.model_dump())


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(MCPException, _handle_mcp_exception)
