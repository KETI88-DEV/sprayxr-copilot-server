"""Separate DB administrator credentials, expiring cookies and CSRF checks."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import HTTPException, Request

COOKIE = "copilot_db_admin"
SESSION_SECONDS = 2 * 60 * 60
ITERATIONS = 600_000


def fail(code: int, message: str) -> None:
    raise HTTPException(code, message, headers={"Cache-Control": "no-store"})


def configure_credentials(path: Path, username: str, password: str) -> None:
    """Explicit operator provisioning; the password is never persisted in plaintext."""
    if not username or not password:
        raise ValueError("administrator credentials are required")
    salt = secrets.token_bytes(16)
    data = {"username": username, "algorithm": "pbkdf2_sha256", "iterations": ITERATIONS,
            "salt": salt.hex(), "password_hash": hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS).hex(),
            "signing_key": secrets.token_hex(32)}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".db-admin-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(data, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@dataclass(frozen=True)
class AdminSession:
    username: str
    csrf: str
    expires: float


class DBAdminAuth:
    def __init__(self, credentials_path: Path) -> None:
        self.path = credentials_path
        self._sessions: dict[str, AdminSession] = {}
        self._failures: dict[str, list[float]] = {}
        self._guard = threading.Lock()

    def _credentials(self) -> dict:
        try:
            with self.path.open("r", encoding="utf-8") as source:
                data = json.loads(source.read(8193))
            assert data["algorithm"] == "pbkdf2_sha256"
            assert isinstance(data["username"], str) and data["username"]
            assert 100_000 <= data["iterations"] <= 2_000_000
            assert len(bytes.fromhex(data["salt"])) == 16
            assert len(bytes.fromhex(data["password_hash"])) == 32
            assert len(bytes.fromhex(data["signing_key"])) == 32
            return data
        except (OSError, ValueError, KeyError, TypeError, AssertionError):
            fail(503, "관리자 계정이 아직 설정되지 않았습니다.")

    def same_origin(self, request: Request, *, required: bool = False) -> None:
        origin = request.headers.get("origin")
        if origin:
            parsed = urlsplit(origin)
            expected = urlsplit(str(request.base_url))
            if (parsed.scheme, parsed.netloc) != (expected.scheme, expected.netloc) or parsed.path not in ("", "/"):
                fail(403, "같은 사이트에서만 관리자 기능을 사용할 수 있습니다.")
        elif required or request.headers.get("sec-fetch-site") == "cross-site":
            fail(403, "관리자 요청의 출처를 확인할 수 없습니다.")

    def login(self, request: Request, username: str, password: str) -> tuple[str, AdminSession]:
        self.same_origin(request, required=True)
        now = time.time()
        client = request.client.host if request.client else "unknown"
        with self._guard:
            recent = [at for at in self._failures.get(client, []) if now - at < 60]
            if len(recent) >= 5:
                fail(429, "로그인 시도가 많습니다. 1분 뒤 다시 시도하세요.")
            # Reserve a slot before the expensive hash to bound concurrent attempts.
            if len(self._failures) > 1024:
                self._failures = {client: recent}
            self._failures[client] = [*recent, now]
        credentials = self._credentials()
        supplied = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(credentials["salt"]), credentials["iterations"])
        valid = hmac.compare_digest(supplied, bytes.fromhex(credentials["password_hash"]))
        valid = hmac.compare_digest(username.encode(), credentials["username"].encode()) and valid
        if not valid:
            fail(401, "관리자 ID 또는 비밀번호가 올바르지 않습니다.")
        session = AdminSession(username, secrets.token_urlsafe(32), now + SESSION_SECONDS)
        token = secrets.token_urlsafe(48)
        with self._guard:
            self._failures.pop(client, None)
            self._sessions = {key: value for key, value in self._sessions.items() if value.expires > now}
            if len(self._sessions) >= 256:
                self._sessions.pop(next(iter(self._sessions)))
            self._sessions[token] = session
        return token, session

    def session(self, request: Request) -> AdminSession | None:
        self.same_origin(request)
        with self._guard:
            token = request.cookies.get(COOKIE, "")
            session = self._sessions.get(token)
            if session and session.expires <= time.time():
                self._sessions.pop(token, None)
                return None
            return session

    def require(self, request: Request, *, write: bool = False) -> AdminSession:
        session = self.session(request)
        if session is None:
            fail(401, "관리자 로그인이 필요합니다.")
        if write:
            self.same_origin(request, required=True)
            if not hmac.compare_digest(request.headers.get("x-db-csrf", "").encode(), session.csrf.encode()):
                fail(403, "관리자 확인 정보가 만료되었습니다. 화면을 새로고침하세요.")
        return session

    def logout(self, request: Request) -> None:
        self.require(request, write=True)
        with self._guard:
            self._sessions.pop(request.cookies.get(COOKIE, ""), None)

    def sign(self, data: dict) -> str:
        raw = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        return hmac.new(bytes.fromhex(self._credentials()["signing_key"]), raw, "sha256").hexdigest()
