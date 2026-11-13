"""Bounded local feedback receipts, separate from trainee and copilot writes."""
from __future__ import annotations

import html
import json
import math
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from urllib.parse import urlencode, urlsplit
from uuid import UUID

from anyio import to_thread
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from trainer.demo_api import __version__
from trainer.demo_api.request_provenance import capture_provenance
from trainer.demo_api.sequence_store import DEFAULT_DB_PATH

KINDS = {'improvement': '개선', 'bug': '오류', 'question': '질문'}
PAGES = {'docs': 'API 문서', 'database': 'DB 뷰어', 'motion': '3D 모션 뷰어'}
STATUSES = {'new': '접수', 'reviewing': '검토 중', 'done': '반영 완료'}
_SCHEMA = """CREATE TABLE IF NOT EXISTS user_feedback (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 submission_id TEXT NOT NULL UNIQUE,
 message TEXT NOT NULL,
 name TEXT NOT NULL DEFAULT '',
 kind TEXT NOT NULL CHECK(kind IN ('improvement','bug','question')),
 page TEXT NOT NULL CHECK(page IN ('docs','database','motion')),
 server_version TEXT NOT NULL,
 created_at TEXT NOT NULL,
 client_ip TEXT,
 caller_source TEXT NOT NULL,
 caller_origin TEXT,
 caller_referer TEXT,
 source_evidence TEXT NOT NULL,
 status TEXT NOT NULL DEFAULT 'new' CHECK(status IN ('new','reviewing','done')),
 status_updated_at TEXT,
 status_updated_by TEXT,
 answer_context TEXT
)"""


def _clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if '\x00' in value:
        raise ValueError('invalid feedback text')
    return value


class AnswerFeedbackContext(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    conversation_id: str = Field(min_length=1, max_length=80)
    question: str = Field(min_length=1, max_length=2000)
    answer: str = Field(min_length=1, max_length=8000)
    task: Literal['pose_recomm', 'trajectory_recomm', 'error_analysis', 'knowledge_search']
    response_mode: Literal['quick', 'detailed']
    answer_source: str | None = Field(default=None, max_length=40)
    zone_id: str | None = Field(default=None, max_length=8)
    map_num: int | None = Field(default=None, ge=0, le=99)
    playhead_seconds: float | None = Field(default=None, ge=0, le=86400, allow_inf_nan=False)
    record: str | None = Field(default=None, max_length=200)
    session_id: str | None = Field(default=None, max_length=80)
    worker_id: str | None = Field(default=None, max_length=80)
    correct_answer: str = Field(default='', max_length=4000)

    @field_validator('conversation_id', 'question', 'answer', 'answer_source', 'zone_id', 'record',
                     'session_id', 'worker_id', 'correct_answer')
    @classmethod
    def trimmed(cls, value: str | None) -> str | None:
        return _clean_text(value)

    @field_validator('conversation_id', 'question', 'answer')
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value:
            raise ValueError('invalid feedback text')
        return value


class FeedbackSubmission(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    submission_id: str = Field(min_length=36, max_length=36)
    message: str = Field(min_length=1, max_length=4000)
    name: str = Field(default='', max_length=80)
    kind: Literal['improvement', 'bug', 'question'] = 'improvement'
    page: Literal['docs', 'database', 'motion']
    answer: AnswerFeedbackContext | None = None

    @model_validator(mode='after')
    def answer_only_on_motion(self):
        if self.answer is not None and self.page != 'motion':
            raise ValueError('answer feedback belongs to the motion viewer')
        return self

    @field_validator('submission_id')
    @classmethod
    def valid_submission_id(cls, value: str) -> str:
        return str(UUID(value))

    @field_validator('message', 'name')
    @classmethod
    def trimmed_text(cls, value: str, info) -> str:
        value = value.strip()
        if (info.field_name == 'message' and not value) or '\x00' in value:
            raise ValueError('invalid feedback text')
        return value


class FeedbackStatus(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    feedback_id: str = Field(min_length=36, max_length=36)
    status: Literal['new', 'reviewing', 'done']


class FeedbackStore:
    """Lazily add one table to the existing SQLite file; never create users."""
    def __init__(self, db_path: str | None = None):
        self.path = db_path if db_path is not None else os.environ.get('SEQUENCE_DB_PATH', DEFAULT_DB_PATH)
        self._conn: sqlite3.Connection | None = None
        self._guard = threading.Lock()

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            if self.path != ':memory:':
                Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.path, check_same_thread=False, timeout=5)
            conn.row_factory = sqlite3.Row
            try:
                with conn:
                    conn.execute(_SCHEMA)
                    conn.execute('CREATE INDEX IF NOT EXISTS idx_feedback_status ON user_feedback(status,id)')
                    # Production DBs predate answer_context; CREATE TABLE IF NOT EXISTS leaves them untouched.
                    if 'answer_context' not in {r[1] for r in conn.execute('PRAGMA table_info(user_feedback)')}:
                        conn.execute('ALTER TABLE user_feedback ADD COLUMN answer_context TEXT')
            except BaseException:
                conn.close()
                raise
            self._conn = conn
        return self._conn

    def submit(self, body: FeedbackSubmission, provenance: dict) -> tuple[dict, bool]:
        with self._guard:
            conn = self._connection()
            context = json.dumps(body.answer.model_dump(), ensure_ascii=False, sort_keys=True) if body.answer else None
            with conn:
                # The transaction also protects idempotency across API workers.
                conn.execute('BEGIN IMMEDIATE')
                row = conn.execute('SELECT * FROM user_feedback WHERE submission_id=?', (body.submission_id,)).fetchone()
                duplicate = row is not None
                if duplicate:
                    if (any(row[key] != getattr(body, key) for key in ('message', 'name', 'kind', 'page'))
                            or row['answer_context'] != context):
                        raise HTTPException(409, '전송 번호가 이미 사용되었습니다. 의견 창을 다시 열어 보내세요.')
                else:
                    created_at = datetime.now(timezone.utc).isoformat(timespec='seconds')
                    conn.execute('''INSERT INTO user_feedback
                        (submission_id,message,name,kind,page,server_version,created_at,client_ip,
                         caller_source,caller_origin,caller_referer,source_evidence,answer_context)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                        (body.submission_id, body.message, body.name, body.kind, body.page, __version__, created_at,
                         provenance.get('client_ip'), provenance['caller_source'], provenance.get('caller_origin'),
                         provenance.get('caller_referer'), provenance['source_evidence'], context))
                    row = conn.execute('SELECT * FROM user_feedback WHERE submission_id=?', (body.submission_id,)).fetchone()
                return {'status': 'received', 'feedback_id': row['submission_id'],
                        'created_at': row['created_at'], 'duplicate': duplicate}, duplicate

    def set_status(self, feedback_id: str, status: str, username: str) -> dict:
        with self._guard:
            conn = self._connection()
            stamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
            with conn:
                cursor = conn.execute('''UPDATE user_feedback SET status=?,status_updated_at=?,status_updated_by=?
                                         WHERE submission_id=?''', (status, stamp, username, feedback_id))
                if cursor.rowcount != 1:
                    raise HTTPException(404, '의견을 찾을 수 없습니다.')
            return {'feedback_id': feedback_id, 'status': status, 'status_updated_at': stamp,
                    'status_updated_by': username}

    def close(self) -> None:
        with self._guard:
            if self._conn is not None:
                self._conn.close()
                self._conn = None


def _same_site(request: Request) -> None:
    origin = request.headers.get('origin')
    if origin:
        try:
            parsed, expected = urlsplit(origin), urlsplit(str(request.base_url))
        except ValueError:
            raise HTTPException(403, '현재 사이트의 의견 창에서 보내주세요.') from None
        if (parsed.scheme, parsed.netloc) != (expected.scheme, expected.netloc) or parsed.path not in ('', '/'):
            raise HTTPException(403, '현재 사이트의 의견 창에서 보내주세요.')
    elif request.headers.get('sec-fetch-site') == 'cross-site':
        raise HTTPException(403, '현재 사이트의 의견 창에서 보내주세요.')


async def _submission_body(request: Request) -> FeedbackSubmission:
    if request.headers.get('content-type', '').split(';', 1)[0].strip().lower() != 'application/json':
        raise HTTPException(415, '의견은 JSON 형식으로 보내주세요.')
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > 98_304:
            raise HTTPException(413, '입력이 너무 깁니다. 의견은 4,000자, 답변 정보는 각 항목의 글자 수 이내로 입력하세요.')
    try:
        return FeedbackSubmission.model_validate_json(bytes(raw))
    except ValueError:
        # Never echo untrusted messages or client-supplied fields into error text.
        raise HTTPException(422, '의견 내용(1~4,000자), 이름(80자 이내), 유형, 페이지와 답변 정보를 확인하세요.') from None


def create_feedback_router(store: FeedbackStore | None = None) -> APIRouter:
    router = APIRouter(include_in_schema=False)
    store = store if store is not None else FeedbackStore()

    @router.post('/api/feedback')
    async def submit_feedback(request: Request):
        _same_site(request)
        body = await _submission_body(request)
        provenance = capture_provenance(request.scope)
        try:
            data, duplicate = await to_thread.run_sync(lambda: store.submit(body, provenance))
        except (OSError, sqlite3.Error):
            raise HTTPException(503, '의견을 저장하지 못했습니다. 입력을 유지한 채 잠시 후 다시 보내주세요.') from None
        return JSONResponse(data, status_code=200 if duplicate else 201, headers={'Cache-Control': 'no-store'})

    def asset_handler(filename: str, content_type: str):
        def asset():
            return FileResponse(Path(__file__).parent/'static'/filename, media_type=content_type,
                                headers={'Cache-Control': 'no-cache'})
        return asset
    for name, media_type in (('copilot_feedback.js', 'text/javascript'), ('copilot_feedback.css', 'text/css'),
                             ('copilot_feedback_admin.js', 'text/javascript')):
        router.add_api_route('/assets/'+name, asset_handler(name, media_type), methods=['GET'], include_in_schema=False)
    return router


def create_feedback_admin_router(auth, db_path: str) -> APIRouter:
    """Use the existing DB administrator's session, Origin and CSRF checks."""
    router = APIRouter(include_in_schema=False)
    store = FeedbackStore(db_path)

    @router.post('/admin/api/manage/feedback-status')
    async def update_feedback_status(request: Request):
        from trainer.demo_api.db_admin_router import body_model
        session = auth.require(request, write=True)
        body = await body_model(request, FeedbackStatus)
        try:
            result = await to_thread.run_sync(lambda: store.set_status(body.feedback_id, body.status, session.username))
        except (OSError, sqlite3.Error):
            raise HTTPException(503, '의견 상태를 저장하지 못했습니다. 잠시 후 다시 시도하세요.') from None
        return JSONResponse(result, headers={'Cache-Control': 'no-store'})
    return router


def render_feedback_list(conn: sqlite3.Connection | None, *, status: str = 'all', page: int = 1) -> str:
    status = status if status in STATUSES else 'all'
    exists = conn is not None and conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='user_feedback'").fetchone()
    params = [] if status == 'all' else [status]
    where = '' if status == 'all' else ' WHERE status=?'
    total = conn.execute('SELECT COUNT(*) FROM user_feedback'+where, params).fetchone()[0] if exists else 0
    pages = max(1, math.ceil(total/25)); page = min(page, pages)
    rows = conn.execute('SELECT * FROM user_feedback'+where+' ORDER BY id DESC LIMIT 25 OFFSET ?', [*params, (page-1)*25]).fetchall() if exists else []
    escape = lambda value: html.escape(str(value if value is not None else '미기록'))
    options = ''.join(f'<option value="{key}"'+(' selected' if status == key else '')+f'>{label}</option>' for key,label in {'all':'전체', **STATUSES}.items())
    rendered = []
    for row in rows:
        row = dict(row)
        choices = ''.join(f'<option value="{key}"'+(' selected' if row['status'] == key else '')+f'>{label}</option>' for key,label in STATUSES.items())
        kind_label, answer_line = KINDS.get(row['kind'], row['kind']), ''
        if row.get('answer_context'):
            kind_label = '답변 오류' if row['kind'] == 'bug' else '답변 메모'
            try:
                question = json.loads(row['answer_context']).get('question')
            except (ValueError, AttributeError):
                question = None
            answer_line = f'<div class="feedback-answer">질문: {escape(question)}</div>'
        detail_url = escape('/admin/api/record?'+urlencode({'table':'user_feedback','rowid':row['id']}))
        rendered.append(f'''<tr><td>{escape(row['created_at'])}<br><small>{escape(row['client_ip'])}</small></td>
<td>{escape(kind_label)}<br>{escape(row['name'] or '익명')}</td>
<td><div class="feedback-message">{escape(row['message'])}</div>{answer_line}<button type="button" class="record-button" data-detail-url="{detail_url}">상세 보기</button></td>
<td>{escape(PAGES.get(row['page'],row['page']))}<br><small>v{escape(row['server_version'])}</small></td>
<td><label class="feedback-status-label">처리 상태<select data-feedback-status="{escape(row['submission_id'])}" disabled>{choices}</select></label>
<button type="button" data-feedback-save="{escape(row['submission_id'])}" disabled>저장</button>
<small data-feedback-result="{escape(row['submission_id'])}" role="status" aria-live="polite"></small></td></tr>''')
    previous = f'<a href="/admin/feedback?{urlencode({"status":status,"page":page-1})}">← 이전</a>' if page > 1 else '<span></span>'
    following = f'<a href="/admin/feedback?{urlencode({"status":status,"page":page+1})}">다음 →</a>' if page < pages else '<span></span>'
    return f'''<div class="topbar"><div><div class="eyebrow">USER FEEDBACK</div><h1>사용자 의견</h1><div class="sub">페이지에서 접수한 개선·오류·질문을 확인합니다. 전체 {total:,}건</div></div></div>
<p id="feedback-review-state" class="sub" role="status" aria-live="polite">상태 변경은 <a href="/admin/manage">관리자 로그인</a> 후 사용할 수 있습니다.</p>
<div class="toolbar"><form method="get" action="/admin/feedback"><label for="feedback-filter">처리 상태</label><select id="feedback-filter" name="status">{options}</select><button>조회</button></form></div>
<div class="table-wrap"><table><thead><tr><th>접수 시각(UTC) · IP</th><th>유형 · 이름</th><th>의견</th><th>페이지 · 버전</th><th>처리 상태</th></tr></thead><tbody>{''.join(rendered) or '<tr><td colspan="5" class="empty">접수된 의견이 없습니다.</td></tr>'}</tbody></table></div>
<div class="pagination">{previous}<span>{page} / {pages} 페이지</span>{following}</div><script defer src="/assets/copilot_feedback_admin.js"></script>'''
