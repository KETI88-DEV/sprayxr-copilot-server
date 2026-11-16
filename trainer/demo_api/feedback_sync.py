"""Authenticated, repeatable feedback transfer for disconnected Spark instances."""
from __future__ import annotations

import json
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PeerOpinion(BaseModel):
    model_config = ConfigDict(extra='ignore', strict=True)
    submission_id: str
    message: str = Field(min_length=1,max_length=4000)
    name: str = Field(default='',max_length=80)
    kind: Literal['improvement','bug','question']
    page: Literal['docs','database','motion']
    server_version: str = Field(max_length=80)
    created_at: str = Field(max_length=80)
    client_ip: str | None = Field(default=None,max_length=80)
    caller_source: str = Field(max_length=80)
    caller_origin: str | None = Field(default=None,max_length=1000)
    caller_referer: str | None = Field(default=None,max_length=2000)
    source_evidence: str = Field(max_length=2000)
    status: Literal['new','reviewing','done']
    status_updated_at: str | None = Field(default=None,max_length=80)
    status_updated_by: str | None = Field(default=None,max_length=80)
    answer_context: str | None = Field(default=None,max_length=98000)
    origin_server: str = Field(min_length=1,max_length=80)
    updated_at: str = Field(min_length=1,max_length=80)
    resolution: str = Field(default='',max_length=4000)

    @field_validator('submission_id')
    @classmethod
    def uuid(cls, value):
        return str(UUID(value))


class PeerRating(BaseModel):
    model_config = ConfigDict(extra='ignore', strict=True)
    feedback_id: str
    query_turn_id: int = Field(gt=0)
    question: str = Field(min_length=1,max_length=16000)
    answer: str = Field(min_length=1,max_length=32000)
    task_type: str = Field(max_length=80)
    user_id: str = Field(max_length=160)
    session_key: str = Field(max_length=160)
    conversation_key: str | None = Field(default=None,max_length=160)
    rating: Literal['up','down']
    comment: str = Field(max_length=4000)
    suggested_answer: str = Field(max_length=8000)
    name: str = Field(max_length=80)
    created_at: str = Field(max_length=80)
    client_ip: str | None = Field(default=None,max_length=80)
    caller_source: str = Field(max_length=80)
    origin_server: str = Field(min_length=1,max_length=80)
    review_status: Literal['pending','approved','rejected']
    corrected_answer: str = Field(max_length=8000)
    review_note: str = Field(max_length=4000)
    reviewed_by: str | None = Field(default=None,max_length=80)
    updated_at: str = Field(min_length=1,max_length=80)
    query_context: str | None = Field(default=None,max_length=131072)

    @field_validator('feedback_id')
    @classmethod
    def uuid(cls, value):
        return str(UUID(value))


class FeedbackBundle(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    schema_version: Literal[1] = 1
    source_instance: str = Field(min_length=1,max_length=80)
    opinions: list[PeerOpinion] = Field(default_factory=list,max_length=1000)
    answer_ratings: list[PeerRating] = Field(default_factory=list,max_length=1000)
    next_offset: int | None = Field(default=None,ge=0)


def export_bundle(store, ratings, offset: int = 0, page_size: int = 50) -> dict:
    with store._guard:
        conn = store._connection()
        opinions = [dict(r) for r in conn.execute('SELECT * FROM user_feedback ORDER BY created_at DESC,submission_id LIMIT ? OFFSET ?', (page_size+1,offset))]
    answers=ratings.rows(limit=page_size+1,offset=offset)
    # Keep an exported page below the bounded import request size even for rich contexts.
    while page_size>1 and len(json.dumps({'opinions':opinions[:page_size],'answer_ratings':answers[:page_size]},ensure_ascii=False).encode())>3_000_000:
        page_size=max(1,page_size//2)
    more=len(opinions)>page_size or len(answers)>page_size
    return {'schema_version':1,'source_instance':ratings.server_id,'opinions':opinions[:page_size],
            'answer_ratings':answers[:page_size],'next_offset':offset+page_size if more else None}


def _merge(conn, table: str, key: str, row: dict, immutable: tuple[str,...]):
    old = conn.execute(f'SELECT * FROM {table} WHERE {key}=?',(row[key],)).fetchone()
    if old is None:
        fields = list(row)
        conn.execute(f'INSERT INTO {table} ({",".join(fields)}) VALUES ({",".join("?" for _ in fields)})',[row[f] for f in fields])
        return 'inserted'
    old = dict(old)
    if any(old.get(f)!=row.get(f) for f in immutable):
        return 'conflict'
    # A repeated export never rolls a newer local administrator decision back.
    if row['updated_at'] <= old['updated_at']:
        return 'unchanged'
    mutable = [f for f in row if f not in immutable and f != key]
    conn.execute(f'UPDATE {table} SET '+','.join(f'{f}=?' for f in mutable)+f' WHERE {key}=?',[*[row[f] for f in mutable],row[key]])
    return 'updated'


def import_bundle(store, ratings, bundle: FeedbackBundle, username: str) -> dict:
    totals = dict(inserted=0,updated=0,unchanged=0,conflict=0)
    conflicts=[]
    with store._guard:
        conn=store._connection()
        with conn:
            for item in bundle.opinions:
                row=item.model_dump()
                result=_merge(conn,'user_feedback','submission_id',row,('submission_id','message','name','kind','page','created_at','origin_server','answer_context','client_ip','caller_source','caller_origin','caller_referer','source_evidence','server_version'))
                totals[result]+=1
                if result=='conflict': conflicts.append(row['submission_id'])
    with ratings.guard, ratings.connection() as conn:
        for item in bundle.answer_ratings:
            row=item.model_dump()
            # Imported turn IDs are scoped to origin_server and are snapshots, never local lookups.
            if row['review_status']=='approved' and row['rating']=='down' and not row['corrected_answer'].strip():
                totals['conflict']+=1; conflicts.append(row['feedback_id']); continue
            result=_merge(conn,'query_answer_feedback','feedback_id',row,('feedback_id','query_turn_id','question','answer','task_type','user_id','session_key','conversation_key','rating','comment','suggested_answer','name','created_at','origin_server','client_ip','caller_source','query_context'))
            totals[result]+=1
            if result=='conflict': conflicts.append(row['feedback_id'])
    return {**totals,'conflicts':conflicts,'source_instance':bundle.source_instance,'reviewed_by':username}
