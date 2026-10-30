"""Operator-only database access and opt-in TSB API-key enforcement."""
from __future__ import annotations

import base64
import binascii
import hmac
import json
import os
import secrets
from pathlib import Path
from typing import Any

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


class OperatorCredentials:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._value: tuple[str, str] | None = None

    def ensure(self) -> tuple[str, str]:
        if self._value is not None:
            return self._value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"username": "admin", "password": secrets.token_urlsafe(24)}
        try:
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(data, stream)
        username, password = data.get("username"), data.get("password")
        if not isinstance(username, str) or not isinstance(password, str) or not password:
            raise ValueError("Operator credentials are invalid")
        self._value = username, password
        return self._value

    def valid(self, authorization: str) -> bool:
        if not authorization.startswith("Basic ") or len(authorization) > 2048:
            return False
        try:
            supplied = base64.b64decode(authorization[6:], validate=True).decode("utf-8")
        except (ValueError, binascii.Error, UnicodeDecodeError):
            return False
        username, separator, password = supplied.partition(":")
        expected_user, expected_password = self.ensure()
        return bool(separator) and hmac.compare_digest(username.encode(), expected_user.encode()) and hmac.compare_digest(password.encode(), expected_password.encode())


class AccessControlMiddleware:
    def __init__(
        self, app: ASGIApp, *, credentials: OperatorCredentials,
        protect_admin: bool, api_key: str | None = None,
    ) -> None:
        self.app = app
        self.credentials = credentials
        self.protect_admin = protect_admin
        self.api_key = api_key

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") == "OPTIONS":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        headers = dict(scope.get("headers", []))
        authorization = headers.get(b"authorization", b"").decode("latin-1")
        operator_path = path == "/admin" or path.startswith("/admin/") or path.startswith("/api/monitor/")
        if self.protect_admin and operator_path and not self.credentials.valid(authorization):
            await JSONResponse(
                {"status": "ERROR", "code": 401, "message": "Operator authentication required", "data": None},
                status_code=401, headers={"WWW-Authenticate": 'Basic realm="Co-Pilot Operator", charset="UTF-8"', "Cache-Control": "no-store"},
            )(scope, receive, send)
            return
        if self.api_key and path.startswith("/api/v1/"):
            supplied = headers.get(b"x-api-key", b"").decode("latin-1")
            if authorization.startswith("Bearer "):
                supplied = authorization[7:]
            if not hmac.compare_digest(supplied.encode(), self.api_key.encode()):
                await JSONResponse(
                    {"status": "ERROR", "code": 401, "message": "API key required", "data": None}, status_code=401,
                )(scope, receive, send)
                return
        await self.app(scope, receive, send)
