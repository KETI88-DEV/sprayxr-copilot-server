"""Private management API alongside the still-public read-only DB viewer."""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from trainer.demo_api.db_admin_auth import COOKIE, SESSION_SECONDS, DBAdminAuth, fail
from trainer.demo_api.db_admin_store import DBAdminStore
from trainer.demo_api.sequence_store import DEFAULT_DB_PATH
from trainer.demo_api.page_access import PageAccessStore


class Target(BaseModel):
    model_config = ConfigDict(extra="forbid")
    table: str = Field(max_length=100)
    rowid: int = Field(ge=-(2**63), le=2**63-1)


class Update(Target):
    version: str = Field(max_length=128)
    fields: dict[str, Any]


class Delete(Target):
    preview_token: str = Field(max_length=160)
    confirmation: str = Field(max_length=256)


def create_db_admin_router() -> APIRouter:
    router = APIRouter(include_in_schema=False)
    path = Path(os.environ.get("SEQUENCE_DB_PATH", DEFAULT_DB_PATH))
    auth = DBAdminAuth(Path(os.environ.get("DB_ADMIN_CREDENTIALS_PATH", str(path.parent / "db_admin_credentials.json"))))
    store = DBAdminStore(path, auth)
    page_access_store = PageAccessStore.for_database(path)
    from trainer.demo_api.feedback import create_feedback_admin_router
    router.include_router(create_feedback_admin_router(auth, str(path)))
    from trainer.demo_api.training_verification import create_training_verification_router
    router.include_router(create_training_verification_router(auth))

    def response(data: dict):
        return JSONResponse(data, headers={"Cache-Control": "no-store"})

    @router.get("/admin/manage")
    def management_page():
        # Deferred import avoids a dependency cycle with the shared viewer shell.
        from trainer.demo_api.db_viewer import _page, _sidebar, _read_database, _table_counts
        from trainer.demo_api.db_admin_ui import ADMIN_HTML
        with _read_database() as conn:
            tables = _table_counts(conn) if conn is not None else []
        return _page(ADMIN_HTML, _sidebar(tables, None, "manage"))

    @router.get("/admin/assets/db_admin.js")
    def script():
        return FileResponse(Path(__file__).parent / "static/db_admin.js", media_type="text/javascript", headers={"Cache-Control": "no-cache"})

    @router.get("/admin/api/manage/status")
    def status(request: Request):
        session = auth.session(request)
        return response({"authenticated": bool(session), **({"username": session.username,
            "csrf_token": session.csrf, "expires_at": session.expires} if session else {})})

    @router.post("/admin/api/manage/login")
    async def login(request: Request):
        auth.same_origin(request, required=True)
        if request.headers.get("content-type", "").split(";", 1)[0].strip() != "application/json":
            fail(415, "JSON 형식으로 로그인 정보를 입력하세요.")
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > 4096:
                fail(413, "로그인 입력이 너무 깁니다.")
        try:
            payload = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            fail(422, "로그인 입력을 확인하세요.")
        if not isinstance(payload, dict) or set(payload) != {"username", "password"} or any(
            not isinstance(value, str) or not 1 <= len(value) <= 1024 for value in payload.values()
        ):
            fail(422, "관리자 ID와 비밀번호를 입력하세요.")
        # Hashing runs in a worker; it must not block live simulator requests.
        from anyio import to_thread
        token, session = await to_thread.run_sync(lambda: auth.login(request, payload["username"], payload["password"]))
        result = response({"authenticated": True, "username": session.username,
                           "csrf_token": session.csrf, "expires_at": session.expires})
        result.set_cookie(COOKIE, token, max_age=SESSION_SECONDS, httponly=True,
                          secure=request.url.scheme == "https", samesite="strict", path="/admin")
        return result

    @router.post("/admin/api/manage/logout")
    def logout(request: Request):
        auth.logout(request)
        result = response({"authenticated": False})
        result.delete_cookie(COOKIE, path="/admin", httponly=True, samesite="strict")
        return result

    @router.get("/admin/api/manage/records")
    def records(request: Request, table: str = Query(default="sessions", max_length=100),
                q: str = Query(default="", max_length=100), page: int = Query(default=1, ge=1)):
        auth.require(request)
        return response(store.records(table, q, page))

    @router.get("/admin/api/manage/record")
    def record(request: Request, table: str = Query(max_length=100), rowid: int = Query(ge=-(2**63), le=2**63-1)):
        auth.require(request)
        return response(store.record(table, rowid))

    @router.post("/admin/api/manage/update")
    async def update(request: Request):
        session = auth.require(request, write=True)
        body = await body_model(request, Update)
        from anyio import to_thread
        return response(await to_thread.run_sync(lambda: store.update(body.table, body.rowid, body.version, body.fields, session.username)))

    @router.post("/admin/api/manage/preview-delete")
    async def preview(request: Request):
        auth.require(request, write=True)
        body = await body_model(request, Target)
        from anyio import to_thread
        return response(await to_thread.run_sync(lambda: store.preview_delete(body.table, body.rowid)))

    @router.post("/admin/api/manage/delete")
    async def delete(request: Request):
        session = auth.require(request, write=True)
        body = await body_model(request, Delete)
        from anyio import to_thread
        return response(await to_thread.run_sync(lambda: store.delete(body.table, body.rowid, body.preview_token, body.confirmation, session.username)))

    @router.get("/admin/api/manage/history")
    def history(request: Request):
        auth.require(request)
        return response(store.history())

    @router.get('/admin/api/manage/page-access')
    def page_access(request: Request, page: int = Query(default=1, ge=1),
                    limit: int = Query(default=50, ge=1, le=100),
                    page_filter: str = Query(default='all', pattern='^(all|home|docs|admin|motion-viewer|query-chat|feedback)$'),
                    client_ip: str = Query(default='', max_length=45),
                    source: str = Query(default='all', pattern='^(all|direct|internal|external)$'),
                    since: datetime | None = None, until: datetime | None = None):
        auth.require(request)
        from trainer.demo_api.request_provenance import valid_ip
        if client_ip and valid_ip(client_ip) is None:
            fail(422, 'IP 주소 형식을 확인하세요.')
        from trainer.demo_api.page_access import utc_stamp
        if since and until and utc_stamp(since) > utc_stamp(until):
            fail(422, '조회 시작 시각은 종료 시각보다 늦을 수 없습니다.')
        selected = getattr(request.app.state, 'page_access_store', page_access_store)
        try:
            return response(selected.records(page=page, limit=limit, page_filter=page_filter, client_ip=client_ip,
                source=source, since=since, until=until))
        except (OSError, sqlite3.Error):
            fail(503, '페이지 접근 기록을 읽을 수 없습니다. 잠시 후 다시 시도하세요.')

    return router


async def body_model(request, model):
    # Authenticate before parsing and avoid returning the supplied content in errors.
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > 16_384:
            fail(413, "관리 입력이 너무 깁니다.")
    try:
        return model.model_validate_json(bytes(raw))
    except ValueError:
        fail(422, "관리 요청의 입력 항목을 확인하세요.")
