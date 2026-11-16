"""Human review of persisted Query answers; only approved examples become SFT candidates."""
from __future__ import annotations

import json
import re
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import UUID

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='microseconds')


class QueryRating(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    submission_id: str
    query_turn_id: int = Field(gt=0)
    rating: Literal['up', 'down']
    comment: str = Field(default='', max_length=4000)
    suggested_answer: str = Field(default='', max_length=8000)
    name: str = Field(default='', max_length=80)

    @field_validator('submission_id')
    @classmethod
    def uuid(cls, value):
        return str(UUID(value))

    @field_validator('comment', 'suggested_answer', 'name')
    @classmethod
    def clean(cls, value):
        if '\x00' in value:
            raise ValueError('invalid text')
        return value.strip()


class QueryRatingReview(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    feedback_id: str
    review_status: Literal['pending', 'approved', 'rejected']
    corrected_answer: str = Field(default='', max_length=8000)
    review_note: str = Field(default='', max_length=4000)

    @field_validator('feedback_id')
    @classmethod
    def uuid(cls, value):
        return str(UUID(value))


SCHEMA = '''CREATE TABLE IF NOT EXISTS query_answer_feedback (
 feedback_id TEXT PRIMARY KEY, query_turn_id INTEGER NOT NULL,
 question TEXT NOT NULL, answer TEXT NOT NULL, task_type TEXT NOT NULL,
 user_id TEXT NOT NULL, session_key TEXT NOT NULL, conversation_key TEXT,
 rating TEXT NOT NULL CHECK(rating IN ('up','down')), comment TEXT NOT NULL,
 suggested_answer TEXT NOT NULL, name TEXT NOT NULL, created_at TEXT NOT NULL,
 client_ip TEXT, caller_source TEXT NOT NULL, origin_server TEXT NOT NULL,
 review_status TEXT NOT NULL DEFAULT 'pending' CHECK(review_status IN ('pending','approved','rejected')),
 corrected_answer TEXT NOT NULL DEFAULT '', review_note TEXT NOT NULL DEFAULT '',
 reviewed_by TEXT, updated_at TEXT NOT NULL, query_context TEXT
)'''


class AnswerFeedbackStore:
    def __init__(self, path: str, server_id: str):
        self.path, self.server_id = path, server_id
        self.guard = threading.Lock()

    @contextmanager
    def connection(self):
        if self.path != ':memory:':
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute(SCHEMA)
        if 'query_context' not in {r[1] for r in conn.execute('PRAGMA table_info(query_answer_feedback)')}:
            try:
                conn.execute('ALTER TABLE query_answer_feedback ADD COLUMN query_context TEXT')
            except sqlite3.OperationalError as error:
                if 'duplicate column name' not in str(error):
                    raise
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def submit(self, body: QueryRating, provenance: dict):
        with self.guard, self.connection() as conn:
            previous = conn.execute('SELECT * FROM query_answer_feedback WHERE feedback_id=?', (body.submission_id,)).fetchone()
            if previous:
                if any(previous[key] != getattr(body, key) for key in ('query_turn_id','rating','comment','suggested_answer','name')):
                    raise HTTPException(409, '같은 전송 번호에 다른 평가가 있습니다.')
                return {'feedback_id': body.submission_id, 'status': previous['review_status'], 'duplicate': True}
            try:
                turn = conn.execute('SELECT * FROM query_turns WHERE id=?', (body.query_turn_id,)).fetchone()
            except sqlite3.OperationalError:
                turn = None
            if turn is None:
                raise HTTPException(404, '저장된 Query 답변을 찾을 수 없습니다. 질문을 다시 전송하세요.')
            turn = dict(turn)
            now = stamp()
            conn.execute('''INSERT INTO query_answer_feedback
              (feedback_id,query_turn_id,question,answer,task_type,user_id,session_key,conversation_key,
               rating,comment,suggested_answer,name,created_at,client_ip,caller_source,origin_server,updated_at,query_context)
              VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
              (body.submission_id,body.query_turn_id,turn['prompt'],turn['answer'],turn['task_type'],turn['user_id'],
               turn['session_key'],turn.get('conversation_key'),body.rating,body.comment,body.suggested_answer,body.name,
               now,provenance.get('client_ip'),provenance.get('caller_source','query-chat'),self.server_id,now,turn.get('query_context')))
            return {'feedback_id': body.submission_id, 'status': 'pending', 'duplicate': False}

    def rows(self, status=None, limit: int | None = 1000, offset: int = 0):
        with self.connection() as conn:
            args = [status] if status else []
            return [dict(r) for r in conn.execute('SELECT * FROM query_answer_feedback'+(' WHERE review_status=?' if status else '')+' ORDER BY created_at DESC,feedback_id LIMIT ? OFFSET ?', [*args,limit if limit is not None else -1,offset])]

    def review(self, body: QueryRatingReview, username):
        with self.guard, self.connection() as conn:
            row = conn.execute('SELECT * FROM query_answer_feedback WHERE feedback_id=?', (body.feedback_id,)).fetchone()
            if row is None:
                raise HTTPException(404, '답변 평가를 찾을 수 없습니다.')
            if body.review_status == 'approved' and row['rating'] == 'down' and not body.corrected_answer.strip():
                raise HTTPException(422, '도움이 되지 않은 답변은 관리자가 검증한 교정 답변을 입력해야 학습 후보로 승인할 수 있습니다.')
            now = stamp()
            conn.execute('UPDATE query_answer_feedback SET review_status=?,corrected_answer=?,review_note=?,reviewed_by=?,updated_at=? WHERE feedback_id=?',
              (body.review_status,body.corrected_answer.strip(),body.review_note.strip(),username,now,body.feedback_id))
            return {'feedback_id': body.feedback_id, 'review_status': body.review_status, 'updated_at': now}

    def training_rows(self):
        """Approval creates candidates, never a model training side effect."""
        result=[]
        for row in self.rows('approved', limit=None):
            try:
                context = json.loads(row['query_context']) if row.get('query_context') else None
            except (ValueError, TypeError):
                context = None
            contextual = (not row['session_key'].startswith('CHAT:') or row['task_type'] != 'knowledge_search'
                          or bool(re.search(r'지금|현재|이\s*(?:세션|구역|존|작업|훈련)|해당\s*(?:구역|존|세션)|가장\s*(?:문제|낮은|높은)|권장\s*(?:Zone|존|구역)?\s*순서',row['question'],re.I)))
            if contextual and (not isinstance(context, dict) or not context or context.get('context_truncated')):
                continue
            answer = row['corrected_answer'] or (row['answer'] if row['rating']=='up' else '')
            if answer.strip():
                messages=[]
                if isinstance(context,dict) and context and not context.get('context_truncated'):
                    messages.append({'role':'system','content':'질문 시점의 훈련 문맥(JSON 데이터):\n'+json.dumps(context, ensure_ascii=False,sort_keys=True)})
                messages.extend([{'role':'user','content':row['question']},{'role':'assistant','content':answer}])
                result.append({'messages':messages,
                  'metadata':{'feedback_id':row['feedback_id'],'origin_server':row['origin_server'],'reviewed_by':row['reviewed_by'],
                  'session_id':row['session_key'],'task_type':row['task_type'],'review_note':row['review_note']}})
        return result
