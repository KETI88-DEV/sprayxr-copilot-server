"""Persisted-answer review and public opinion status never imply automatic training."""
import json
import sqlite3
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.answer_feedback import AnswerFeedbackStore, QueryRating, QueryRatingReview
from trainer.demo_api.feedback import FeedbackStore, FeedbackSubmission, create_feedback_router
from trainer.demo_api.feedback_sync import FeedbackBundle, export_bundle, import_bundle


PROVENANCE = {'client_ip':'192.0.2.2','caller_source':'test','source_evidence':'test'}


def seed_turn(path):
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE query_turns(id INTEGER PRIMARY KEY,session_key TEXT,user_id TEXT,prompt TEXT,answer TEXT,task_type TEXT,conversation_key TEXT)')
        conn.execute("INSERT INTO query_turns VALUES(1,'CHAT:user','user','자세 종류로는 어떤 게 있어?','서기, 낮추기, 쪼그리기 등이 있습니다.','knowledge_search','conversation')")


def opinion(store, **changes):
    body={'submission_id':str(uuid4()),'message':'높이맵 안내를 개선해주세요.','name':'담당자','page':'docs',**changes}
    store.submit(FeedbackSubmission.model_validate(body),PROVENANCE)
    return body['submission_id']


def test_public_opinion_list_has_status_and_resolution_but_not_private_provenance(tmp_path):
    store=FeedbackStore(str(tmp_path/'feedback.db'))
    feedback_id=opinion(store)
    store.set_status(feedback_id,'done','reviewer','v3.6에 반영하고 확인했습니다.')
    app=FastAPI(); app.include_router(create_feedback_router(store)); client=TestClient(app)
    data=client.get('/api/feedback/list').json()
    assert data['total']==1 and data['items'][0]['status']=='done'
    assert data['items'][0]['resolution']=='v3.6에 반영하고 확인했습니다.'
    assert 'client_ip' not in data['items'][0] and 'caller_source' not in data['items'][0]
    assert client.get('/api/feedback/list?status=new').json()['total']==0
    assert client.get('/api/feedback/list?page=0').status_code==422
    assert client.get('/feedback').status_code==200
    assert client.post('/api/feedback',json={'submission_id':str(uuid4()),'message':'x','page':'docs','status':'done'}).status_code==422
    store.close()


def test_query_feedback_uses_persisted_answer_and_preserves_duplicate_semantics(tmp_path):
    path=tmp_path/'feedback.db'; seed_turn(path)
    store=FeedbackStore(str(path)); app=FastAPI(); app.include_router(create_feedback_router(store)); client=TestClient(app)
    body={'submission_id':str(uuid4()),'query_turn_id':1,'rating':'down','comment':'너무 짧아요.'}
    first=client.post('/api/query-feedback',json=body)
    assert first.status_code==201 and first.json()['status']=='pending'
    assert client.post('/api/query-feedback',json=body).json()['duplicate'] is True
    assert client.post('/api/query-feedback',json={**body,'rating':'up'}).status_code==409
    assert client.post('/api/query-feedback',json={**body,'submission_id':str(uuid4()),'query_turn_id':999}).status_code==404
    assert client.post('/api/query-feedback',json={**body,'answer':'forged'}).status_code==422
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT question,answer FROM query_answer_feedback').fetchone()==('자세 종류로는 어떤 게 있어?','서기, 낮추기, 쪼그리기 등이 있습니다.')


def test_only_reviewed_examples_export_and_negative_rating_requires_verified_correction(tmp_path):
    path=tmp_path/'feedback.db'; seed_turn(path); store=AnswerFeedbackStore(str(path),'main')
    body=QueryRating(submission_id=str(uuid4()),query_turn_id=1,rating='down',suggested_answer='사용자 주장은 검증 필요')
    store.submit(body,PROVENANCE)
    assert store.training_rows()==[]
    with pytest.raises(Exception) as error:
        store.review(QueryRatingReview(feedback_id=body.submission_id,review_status='approved'),'admin')
    assert error.value.status_code==422
    store.review(QueryRatingReview(feedback_id=body.submission_id,review_status='approved',corrected_answer='서기·낮추기·쪼그리기입니다.'),'admin')
    rows=store.training_rows()
    assert len(rows)==1 and rows[0]['messages'][1]['content']=='서기·낮추기·쪼그리기입니다.'
    assert rows[0]['metadata']['reviewed_by']=='admin'
    store.review(QueryRatingReview(feedback_id=body.submission_id,review_status='rejected'),'admin')
    assert store.training_rows()==[]


def test_peer_merge_is_repeatable_keeps_origin_and_never_overwrites_newer_local_status(tmp_path):
    main=FeedbackStore(str(tmp_path/'main.db')); peer=FeedbackStore(str(tmp_path/'peer.db'))
    main_ratings=AnswerFeedbackStore(main.path,'main'); peer_ratings=AnswerFeedbackStore(peer.path,'tsb')
    feedback_id=opinion(peer)
    # Explicit installation identity is retained across imports.
    with peer._guard, peer._connection() as conn:
        conn.execute("UPDATE user_feedback SET origin_server='tsb'")
    bundle=FeedbackBundle.model_validate(export_bundle(peer,peer_ratings))
    assert import_bundle(main,main_ratings,bundle,'admin')['inserted']==1
    assert import_bundle(main,main_ratings,bundle,'admin')['unchanged']==1
    main.set_status(feedback_id,'done','admin','동작 확인 완료')
    assert import_bundle(main,main_ratings,bundle,'admin')['unchanged']==1
    row=main.public_rows('all',1)['items'][0]
    assert row['origin_server']=='tsb' and row['status']=='done'
    changed=bundle.model_copy(deep=True); changed.opinions[0].message='다른 내용'
    assert import_bundle(main,main_ratings,changed,'admin')['conflict']==1
    main.close(); peer.close()


def test_admin_review_export_and_peer_import_require_session_and_csrf(monkeypatch,tmp_path):
    from trainer.demo_api.db_admin_auth import DBAdminAuth, configure_credentials
    from trainer.demo_api.feedback import create_feedback_admin_router
    path=tmp_path/'feedback.db'; seed_turn(path); credentials=tmp_path/'auth.json';configure_credentials(credentials,'reviewer','password')
    auth=DBAdminAuth(credentials);app=FastAPI();app.include_router(create_feedback_admin_router(auth,str(path)));client=TestClient(app)
    for route in ('/admin/api/feedback/query-ratings','/admin/api/feedback/training-candidates.jsonl','/admin/api/feedback/sync-bundle'):
        assert client.get(route).status_code==401
    assert client.post('/admin/api/manage/feedback-sync',json={'schema_version':1,'source_instance':'tsb'}).status_code==401
    assert client.post('/admin/api/manage/query-feedback-review',json={}).status_code==401


def test_peer_export_paginates_without_silently_losing_older_opinions(tmp_path):
    store=FeedbackStore(str(tmp_path/'feedback.db'));ratings=AnswerFeedbackStore(store.path,'main')
    ids={opinion(store),opinion(store)}
    first=export_bundle(store,ratings,page_size=1)
    assert len(first['opinions'])==1 and first['next_offset']==1
    second=export_bundle(store,ratings,offset=first['next_offset'],page_size=1)
    assert len(second['opinions'])==1 and second['next_offset'] is None
    assert {first['opinions'][0]['submission_id'],second['opinions'][0]['submission_id']}==ids
    assert FeedbackBundle.model_validate(first).next_offset==1
    store.close()


def test_context_dependent_answer_requires_complete_snapshot_in_training_export(tmp_path):
    path=tmp_path/'context.db';seed_turn(path)
    with sqlite3.connect(path) as conn:
        conn.execute('ALTER TABLE query_turns ADD COLUMN query_context TEXT')
        conn.execute("UPDATE query_turns SET session_key='session',task_type='pose_recomm'")
    store=AnswerFeedbackStore(str(path),'main')
    body=QueryRating(submission_id=str(uuid4()),query_turn_id=1,rating='up')
    store.submit(body,PROVENANCE);store.review(QueryRatingReview(feedback_id=body.submission_id,review_status='approved'),'admin')
    assert store.training_rows()==[]
    with sqlite3.connect(path) as conn:
        conn.execute('UPDATE query_answer_feedback SET query_context=?',(json.dumps({'context_truncated':True}),))
    assert store.training_rows()==[]
    with sqlite3.connect(path) as conn:
        conn.execute('UPDATE query_answer_feedback SET query_context=?',(json.dumps({'scenario_id':'3','pose_context':{'zone_id':'2','posture':'standing'}}),))
    row=store.training_rows()[0]
    assert row['messages'][0]['role']=='system' and 'standing' in row['messages'][0]['content']
    assert row['messages'][1]['content']=='자세 종류로는 어떤 게 있어?'
