"""Bounded local feedback receipts, separate from trainee and copilot writes."""
from __future__ import annotations

import csv
import html
import io
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
from trainer.demo_api.answer_feedback import AnswerFeedbackStore, QueryRating, QueryRatingReview

from anyio import to_thread
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from trainer.demo_api import __version__
from trainer.demo_api.request_provenance import capture_provenance
from trainer.demo_api.sequence_store import DEFAULT_DB_PATH

KINDS = {'improvement': '개선', 'bug': '오류', 'question': '질문'}
PAGES = {'docs': 'API 문서', 'database': 'DB 뷰어', 'motion': '3D 모션 뷰어'}
STATUSES = {'new': '접수', 'reviewing': '처리 중', 'done': '처리 완료'}


def source_instance() -> str:
    return os.environ.get('COPILOT_INSTANCE_ID', 'main').strip()[:80] or 'main'
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
    resolution: str = Field(default='', max_length=4000)


def _add_answer_column(conn: sqlite3.Connection) -> None:
    """Production DBs predate answer_context; another API worker may add it at the same moment."""
    if 'answer_context' in {r[1] for r in conn.execute('PRAGMA table_info(user_feedback)')}:
        return
    try:
        conn.execute('ALTER TABLE user_feedback ADD COLUMN answer_context TEXT')
    except sqlite3.OperationalError as error:
        if 'duplicate column name' not in str(error):
            raise

# (JSON key, CSV header); [5:18] come from answer_context, the rest from table columns.
_EXPORT_FIELDS = [('feedback_id', '피드백 ID'), ('created_at', '접수 시각(UTC)'), ('status', '처리 상태'),
                  ('label', '종류(답변 오류/답변 메모)'), ('message', '메모'), ('correct_answer', '맞는 답'),
                  ('question', '질문'), ('answer', '답'), ('task', '질문 종류'), ('response_mode', '답변 수준'),
                  ('answer_source', '답 출처'), ('zone_id', 'Zone'), ('map_num', 'Map'),
                  ('playhead_seconds', '재생 위치(초)'), ('record', '기록'), ('session_id', '세션'),
                  ('worker_id', '작업자'), ('conversation_id', '대화 ID'), ('server_version', '서버 버전')]


def answer_feedback_rows(db_path: str, status: str | None) -> list[dict]:
    """Fresh read-only connection: an export never takes write locks. shortcut: whole result in memory, stream when it outgrows a few thousand rows."""
    try:
        conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    except sqlite3.Error:
        return []
    try:
        conn.row_factory = sqlite3.Row
        sql = 'SELECT * FROM user_feedback WHERE answer_context IS NOT NULL' + (' AND status=?' if status else '') + ' ORDER BY id'
        raw = conn.execute(sql, [status] if status else []).fetchall()
    except sqlite3.Error:  # no table or no column yet
        return []
    finally:
        conn.close()
    rows = []
    for row in raw:
        try:
            context = json.loads(row['answer_context'])
        except ValueError:
            context = {}
        if not isinstance(context, dict):
            context = {}
        rows.append({'feedback_id': row['submission_id'], 'created_at': row['created_at'], 'status': row['status'],
                     'label': 'wrong_answer' if row['kind'] == 'bug' else 'memo', 'message': row['message'],
                     **{key: context.get(key) for key, _ in _EXPORT_FIELDS[5:18]}, 'server_version': row['server_version']})
    return rows


def _csv_cell(value) -> str:
    text = '' if value is None else str(value)
    return "'" + text if text[:1] in ('=', '+', '-', '@') else text


def _download(body: bytes, media: str, ext: str) -> Response:
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%MZ')
    return Response(body, media_type=media, headers={'Content-Disposition': f'attachment; filename="answer_feedback_{stamp}.{ext}"',
                                                     'Cache-Control': 'no-store'})


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
                    _add_answer_column(conn)
                    existing = {r[1] for r in conn.execute('PRAGMA table_info(user_feedback)')}
                    for name, ddl in (('origin_server', "TEXT NOT NULL DEFAULT ''"), ('updated_at', "TEXT NOT NULL DEFAULT ''"), ('resolution', "TEXT NOT NULL DEFAULT ''")):
                        if name not in existing:
                            try:
                                conn.execute(f'ALTER TABLE user_feedback ADD COLUMN {name} {ddl}')
                            except sqlite3.OperationalError as error:
                                if 'duplicate column name' not in str(error):
                                    raise
                    conn.execute("UPDATE user_feedback SET origin_server=? WHERE origin_server=''", (source_instance(),))
                    conn.execute("UPDATE user_feedback SET updated_at=COALESCE(status_updated_at,created_at) WHERE updated_at='' ")
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
                    conn.execute('UPDATE user_feedback SET origin_server=?,updated_at=? WHERE submission_id=?',
                                 (source_instance(), created_at, body.submission_id))
                return {'status': 'received', 'feedback_id': row['submission_id'],
                        'created_at': row['created_at'], 'duplicate': duplicate}, duplicate

    def set_status(self, feedback_id: str, status: str, username: str, resolution: str = '') -> dict:
        with self._guard:
            conn = self._connection()
            stamp = datetime.now(timezone.utc).isoformat(timespec='microseconds')
            with conn:
                cursor = conn.execute('''UPDATE user_feedback SET status=?,status_updated_at=?,status_updated_by=?,resolution=?,updated_at=?
                                         WHERE submission_id=?''', (status, stamp, username, resolution.strip(), stamp, feedback_id))
                if cursor.rowcount != 1:
                    raise HTTPException(404, '의견을 찾을 수 없습니다.')
            return {'feedback_id': feedback_id, 'status': status, 'status_updated_at': stamp,
                    'status_updated_by': username, 'resolution': resolution.strip()}

    def public_rows(self, status: str, page: int) -> dict:
        with self._guard:
            conn = self._connection()
            where, args = (' WHERE status=?', [status]) if status in STATUSES else ('', [])
            total = conn.execute('SELECT COUNT(*) FROM user_feedback'+where, args).fetchone()[0]
            rows = conn.execute('SELECT submission_id,message,name,kind,page,created_at,status,status_updated_at,resolution,origin_server FROM user_feedback'+where+' ORDER BY created_at DESC LIMIT 20 OFFSET ?', [*args,(page-1)*20]).fetchall()
            return {'items':[dict(r) for r in rows], 'total':total, 'page':page, 'page_size':20, 'statuses':STATUSES}

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


async def _bounded_model(request: Request, model, *, limit: int = 98_304):
    if request.headers.get('content-type','').split(';',1)[0].strip().lower() != 'application/json':
        raise HTTPException(415, 'JSON 형식으로 보내주세요.')
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw)>limit:
            raise HTTPException(413, '입력이 너무 깁니다.')
    try:
        return model.model_validate_json(bytes(raw))
    except ValueError:
        raise HTTPException(422, '입력 항목을 확인하세요.') from None


def create_feedback_router(store: FeedbackStore | None = None) -> APIRouter:
    router = APIRouter(include_in_schema=False)
    store = store if store is not None else FeedbackStore()
    ratings = AnswerFeedbackStore(store.path, source_instance())

    @router.get('/api/feedback/list')
    async def public_feedback(status: Literal['all','new','reviewing','done'] = 'all', page: int = Query(default=1, ge=1, le=100000)):
        return JSONResponse(await to_thread.run_sync(lambda: store.public_rows(status, page)), headers={'Cache-Control':'no-store'})

    @router.get('/feedback', response_class=HTMLResponse)
    def feedback_page():
        from trainer.demo_api.portal_navigation import portal_document
        document = (Path(__file__).parent/'static/feedback_page.html').read_text(encoding='utf-8')
        return HTMLResponse(portal_document(document, 'feedback'), headers={'Cache-Control':'no-store'})

    @router.post('/api/query-feedback')
    async def rate_query(request: Request):
        _same_site(request)
        body = await _bounded_model(request, QueryRating)
        try:
            receipt = await to_thread.run_sync(lambda: ratings.submit(body, capture_provenance(request.scope)))
        except (OSError, sqlite3.Error):
            raise HTTPException(503, '답변 평가를 저장하지 못했습니다. 잠시 후 다시 보내주세요.') from None
        return JSONResponse(receipt, status_code=200 if receipt['duplicate'] else 201, headers={'Cache-Control':'no-store'})

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
                             ('copilot_feedback_admin.js', 'text/javascript'), ('feedback_page.js','text/javascript'),
                             ('answer_feedback_admin.js','text/javascript')):
        router.add_api_route('/assets/'+name, asset_handler(name, media_type), methods=['GET'], include_in_schema=False)
    return router


def create_feedback_admin_router(auth, db_path: str) -> APIRouter:
    """Use the existing DB administrator's session, Origin and CSRF checks."""
    router = APIRouter(include_in_schema=False)
    store = FeedbackStore(db_path)
    ratings = AnswerFeedbackStore(db_path, source_instance())

    @router.get('/admin/query-feedback', response_class=HTMLResponse)
    def rating_review_page():
        from trainer.demo_api.db_viewer import _page, _sidebar, _read_database, _table_counts
        from trainer.demo_api.feedback_review_ui import REVIEW_HTML
        with _read_database() as conn:
            tables = _table_counts(conn) if conn is not None else []
        return _page(REVIEW_HTML, _sidebar(tables, None, 'feedback'))

    @router.get('/admin/api/feedback/query-ratings')
    async def rating_list(request: Request, status: Literal['pending','approved','rejected'] | None = None):
        auth.require(request)
        return JSONResponse({'items':await to_thread.run_sync(ratings.rows, status)}, headers={'Cache-Control':'no-store'})

    @router.post('/admin/api/manage/query-feedback-review')
    async def review_rating(request: Request):
        session = auth.require(request, write=True)
        body = await _bounded_model(request, QueryRatingReview)
        return JSONResponse(await to_thread.run_sync(lambda: ratings.review(body, session.username)), headers={'Cache-Control':'no-store'})

    @router.get('/admin/api/feedback/training-candidates.jsonl')
    async def training_candidates(request: Request):
        auth.require(request)
        rows = await to_thread.run_sync(ratings.training_rows)
        body = ''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows)
        return _download(body.encode(), 'application/x-ndjson; charset=utf-8', 'jsonl')

    @router.get('/admin/api/feedback/sync-bundle')
    async def sync_export(request: Request, offset: int = Query(default=0, ge=0)):
        auth.require(request)
        from trainer.demo_api.feedback_sync import export_bundle
        return JSONResponse(await to_thread.run_sync(lambda: export_bundle(store, ratings, offset)), headers={'Cache-Control':'no-store'})

    @router.post('/admin/api/manage/feedback-sync')
    async def sync_import(request: Request):
        session = auth.require(request, write=True)
        from trainer.demo_api.feedback_sync import FeedbackBundle, import_bundle
        body = await _bounded_model(request, FeedbackBundle, limit=4_194_304)
        return JSONResponse(await to_thread.run_sync(lambda: import_bundle(store, ratings, body, session.username)), headers={'Cache-Control':'no-store'})

    @router.post('/admin/api/manage/feedback-status')
    async def update_feedback_status(request: Request):
        from trainer.demo_api.db_admin_router import body_model
        session = auth.require(request, write=True)
        body = await body_model(request, FeedbackStatus)
        try:
            result = await to_thread.run_sync(lambda: store.set_status(body.feedback_id, body.status, session.username, body.resolution))
        except (OSError, sqlite3.Error):
            raise HTTPException(503, '의견 상태를 저장하지 못했습니다. 잠시 후 다시 시도하세요.') from None
        return JSONResponse(result, headers={'Cache-Control': 'no-store'})
    @router.get('/admin/api/feedback/answers.jsonl')
    async def answers_jsonl(request: Request, status: Literal['new', 'reviewing', 'done'] | None = Query(default=None)):
        auth.require(request)
        rows = await to_thread.run_sync(answer_feedback_rows, db_path, status)
        body = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
        return _download(body.encode('utf-8'), 'application/x-ndjson; charset=utf-8', 'jsonl')

    @router.get('/admin/api/feedback/answers.csv')
    async def answers_csv(request: Request, status: Literal['new', 'reviewing', 'done'] | None = Query(default=None)):
        auth.require(request)
        rows = await to_thread.run_sync(answer_feedback_rows, db_path, status)
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow([header for _, header in _EXPORT_FIELDS])
        for r in rows:
            record = {**r, 'label': '답변 오류' if r['label'] == 'wrong_answer' else '답변 메모'}
            writer.writerow([_csv_cell(record[key]) for key, _ in _EXPORT_FIELDS])
        return _download(('\ufeff' + out.getvalue()).encode('utf-8'), 'text/csv; charset=utf-8', 'csv')
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
<label class="feedback-status-label">처리 내용<textarea data-feedback-resolution maxlength="4000" rows="3" disabled>{escape(row.get('resolution') or '')}</textarea></label>
<button type="button" data-feedback-save="{escape(row['submission_id'])}" disabled>저장</button>
<small data-feedback-result="{escape(row['submission_id'])}" role="status" aria-live="polite"></small></td></tr>''')
    previous = f'<a href="/admin/feedback?{urlencode({"status":status,"page":page-1})}">← 이전</a>' if page > 1 else '<span></span>'
    following = f'<a href="/admin/feedback?{urlencode({"status":status,"page":page+1})}">다음 →</a>' if page < pages else '<span></span>'
    return f'''<div class="topbar"><div><div class="eyebrow">USER FEEDBACK</div><h1>사용자 의견</h1><div class="sub">페이지에서 접수한 개선·오류·질문을 확인합니다. 전체 {total:,}건</div></div></div>
<p id="feedback-review-state" class="sub" role="status" aria-live="polite">상태 변경은 <a href="/admin/manage">관리자 로그인</a> 후 사용할 수 있습니다.</p>
<p class="sub"><a href="/feedback">공개 의견 목록</a> · <a href="/admin/query-feedback">Query 답변 평가·학습 후보 검토</a></p>
<div class="toolbar"><form method="get" action="/admin/feedback"><label for="feedback-filter">처리 상태</label><select id="feedback-filter" name="status">{options}</select><button>조회</button></form>
<a href="/admin/api/feedback/answers.jsonl">답변 피드백 내려받기(JSONL)</a> <a href="/admin/api/feedback/answers.csv">(CSV)</a> <small>내려받기는 관리자 로그인 후 사용할 수 있습니다.</small></div>
<div class="table-wrap"><table><thead><tr><th>접수 시각(UTC) · IP</th><th>유형 · 이름</th><th>의견</th><th>페이지 · 버전</th><th>처리 상태</th></tr></thead><tbody>{''.join(rendered) or '<tr><td colspan="5" class="empty">접수된 의견이 없습니다.</td></tr>'}</tbody></table></div>
<div class="pagination">{previous}<span>{page} / {pages} 페이지</span>{following}</div><script defer src="/assets/copilot_feedback_admin.js"></script>'''
