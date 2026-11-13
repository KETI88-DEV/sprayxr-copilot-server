"""Public feedback is bounded data; only the existing DB admin can review it."""
import importlib
import importlib.util
import sqlite3
from html.parser import HTMLParser
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def feedback_module():
    assert importlib.util.find_spec('trainer.demo_api.feedback') is not None, 'The feedback submission boundary is missing'
    return importlib.import_module('trainer.demo_api.feedback')


def payload(**changes):
    return {'submission_id': str(uuid4()), 'message': '구역 선택 후 안내를 더 쉽게 보고 싶습니다.',
            'name': '테스트 담당자', 'kind': 'improvement', 'page': 'motion', **changes}


@pytest.fixture
def feedback_client(monkeypatch, tmp_path):
    from trainer.demo_api.access_control import AccessControlMiddleware, OperatorCredentials
    from trainer.demo_api.db_viewer import create_db_viewer_router
    from trainer.demo_api.db_admin_auth import configure_credentials
    module = feedback_module()
    path = tmp_path / 'feedback.db'
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE users(user_id TEXT, name TEXT)')
        conn.execute('INSERT INTO users VALUES(?,?)', ('COPILOT-DEMO-USER', '모의훈련자'))
    monkeypatch.setenv('SEQUENCE_DB_PATH', str(path))
    credentials = tmp_path / 'admin.json'
    configure_credentials(credentials, 'reviewer', 'test-feedback-password')
    monkeypatch.setenv('DB_ADMIN_CREDENTIALS_PATH', str(credentials))
    app = FastAPI()
    app.add_middleware(AccessControlMiddleware, credentials=OperatorCredentials(tmp_path/'operator.json'), protect_admin=True, api_key='private-api-key')
    app.include_router(module.create_feedback_router())
    app.include_router(create_db_viewer_router())
    return TestClient(app, client=('192.0.2.42', 12345)), path


def test_feedback_persists_in_the_existing_database_without_creating_a_trainee(feedback_client):
    client, path = feedback_client
    body = payload(message="  안내 문구를 개선해주세요.  ", name='  담당자  ')
    response = client.post('/api/feedback', json=body)
    assert response.status_code == 201, response.text
    receipt = response.json()
    assert receipt['feedback_id'] == body['submission_id'] and receipt['duplicate'] is False
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        row = dict(conn.execute('SELECT * FROM user_feedback').fetchone())
        assert row['message'] == '안내 문구를 개선해주세요.' and row['name'] == '담당자'
        assert row['page'] == 'motion' and row['kind'] == 'improvement' and row['status'] == 'new'
        assert row['client_ip'] == '192.0.2.42' and row['server_version']
        assert row['created_at'].endswith('+00:00')
        assert conn.execute('SELECT COUNT(*) FROM users').fetchone()[0] == 1


@pytest.mark.parametrize('changes', [
    {'message': ''}, {'message': '   \n'}, {'message': '가' * 4001}, {'message': 42},
    {'name': '가' * 81}, {'kind': 'delete'}, {'page': 'https://evil.test'},
    {'submission_id': 'not-a-uuid'}, {'status': 'done'}, {'client_ip': '203.0.113.1'}, {'user_id': 'NEW-USER'},
])
def test_invalid_feedback_never_enters_storage(feedback_client, changes):
    client, path = feedback_client
    response = client.post('/api/feedback', json=payload(**changes))
    assert response.status_code == 422, response.text
    assert 'input' not in response.text
    with sqlite3.connect(path) as conn:
        exists = conn.execute("SELECT 1 FROM sqlite_master WHERE name='user_feedback'").fetchone()
        assert not exists or conn.execute('SELECT COUNT(*) FROM user_feedback').fetchone()[0] == 0


def test_limit_accepts_four_thousand_korean_characters_and_optional_name(feedback_client):
    client, _ = feedback_client
    body = payload(message='가' * 4000)
    body.pop('name')
    assert client.post('/api/feedback', json=body).status_code == 201


def test_feedback_bounds_raw_body_and_requires_json_and_same_site(feedback_client):
    client, _ = feedback_client
    assert client.post('/api/feedback', content=' ' * 98305, headers={'Content-Type': 'application/json'}).status_code == 413
    assert client.post('/api/feedback', content='message=hello').status_code == 415
    assert client.post('/api/feedback', content='{', headers={'Content-Type': 'application/json'}).status_code == 422
    assert client.post('/api/feedback', json=payload(), headers={'Origin': 'https://unrelated.test'}).status_code == 403


def answer_context(**changes):
    return {'conversation_id': 'conv-1', 'question': '어느 구역이 문제인가요?', 'answer': '3번 구역입니다.',
            'task': 'pose_recomm', 'response_mode': 'quick', 'answer_source': 'engine', 'zone_id': '3',
            'map_num': 2, 'playhead_seconds': 12.5, 'record': 'SEQ-012', 'session_id': 'S-1',
            'worker_id': 'W-1', 'correct_answer': '4번 구역입니다.', **changes}


def test_answer_feedback_persists_context_json(feedback_client):
    import json
    client, path = feedback_client
    ctx = answer_context()
    response = client.post('/api/feedback', json=payload(kind='bug', answer=ctx))
    assert response.status_code == 201, response.text
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute('SELECT * FROM user_feedback').fetchone()
    assert json.loads(row['answer_context']) == ctx and row['kind'] == 'bug' and row['page'] == 'motion'


def test_legacy_database_gets_the_answer_column_without_losing_rows(feedback_client):
    import json, re
    module = feedback_module()
    _, path = feedback_client
    old = re.sub(r"\n answer_context TEXT,?", '', module._SCHEMA)
    old = old.replace(' status_updated_by TEXT,', ' status_updated_by TEXT')
    assert 'answer_context' not in old
    with sqlite3.connect(path) as conn:
        conn.execute(old)
        conn.execute("INSERT INTO user_feedback(submission_id,message,kind,page,server_version,created_at,caller_source,source_evidence) VALUES('old','옛 의견','bug','docs','1','t','x','y')")
    store = module.FeedbackStore(str(path))
    body = module.FeedbackSubmission.model_validate(payload(kind='improvement', answer=answer_context()))
    store.submit(body, {'caller_source': 'x', 'source_evidence': 'y'})
    store.close()
    with sqlite3.connect(path) as conn:
        rows = conn.execute('SELECT message,answer_context FROM user_feedback ORDER BY id').fetchall()
    assert rows[0] == ('옛 의견', None) and json.loads(rows[1][1])['question'] == '어느 구역이 문제인가요?'


def test_answer_on_other_page_or_with_unknown_key_is_rejected(feedback_client):
    client, path = feedback_client
    assert client.post('/api/feedback', json=payload(page='docs', answer=answer_context())).status_code == 422
    assert client.post('/api/feedback', json=payload(answer=answer_context(extra='x'))).status_code == 422
    with sqlite3.connect(path) as conn:
        exists = conn.execute("SELECT 1 FROM sqlite_master WHERE name='user_feedback'").fetchone()
        assert not exists or conn.execute('SELECT COUNT(*) FROM user_feedback').fetchone()[0] == 0


def test_same_submission_with_different_answer_conflicts(feedback_client):
    client, _ = feedback_client
    body = payload(kind='bug', answer=answer_context())
    assert client.post('/api/feedback', json=body).status_code == 201
    assert client.post('/api/feedback', json=body).status_code == 200
    assert client.post('/api/feedback', json={**body, 'answer': answer_context(answer='다른 답')}).status_code == 409


def test_admin_list_labels_answer_feedback_and_escapes_question(feedback_client):
    client, _ = feedback_client
    ctx = answer_context(question='<script>x</script>')
    assert client.post('/api/feedback', json=payload(kind='bug', answer=ctx)).status_code == 201
    text = client.get('/admin/feedback').text
    assert '답변 오류' in text and '&lt;script&gt;x&lt;/script&gt;' in text and '<script>x</script>' not in text
    assert 'class="feedback-answer"' in text


def test_large_answer_feedback_body_limit(feedback_client):
    client, _ = feedback_client
    big = answer_context(question='가' * 2000, answer='나' * 8000, correct_answer='다' * 4000)
    body = payload(kind='bug', message='라' * 4000, answer=big)
    assert 50_000 < len(__import__('json').dumps(body, ensure_ascii=False).encode()) < 98_304
    assert client.post('/api/feedback', json=body).status_code == 201
    assert client.post('/api/feedback', content=' ' * 98_305, headers={'Content-Type': 'application/json'}).status_code == 413


def test_retries_are_idempotent_but_other_people_on_same_ip_can_submit(feedback_client):
    client, path = feedback_client
    body = payload()
    first = client.post('/api/feedback', json=body)
    retry = client.post('/api/feedback', json=body)
    assert first.status_code == 201 and retry.status_code == 200
    assert retry.json()['duplicate'] is True and retry.json()['feedback_id'] == first.json()['feedback_id']
    assert client.post('/api/feedback', json={**body, 'message': '다른 내용'}).status_code == 409
    assert client.post('/api/feedback', json=payload()).status_code == 201
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT COUNT(*) FROM user_feedback').fetchone()[0] == 2


def test_forwarded_ip_and_client_version_never_override_server_provenance(feedback_client):
    client, path = feedback_client
    assert client.post('/api/feedback', json=payload(), headers={'X-Forwarded-For': '203.0.113.9', 'X-Real-IP': '203.0.113.8'}).status_code == 201
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT client_ip FROM user_feedback').fetchone()[0] == '192.0.2.42'
    assert client.post('/api/feedback', json=payload(server_version='fake')).status_code == 422


def test_invalid_asgi_peer_stays_unknown(monkeypatch, tmp_path):
    module = feedback_module()
    path = tmp_path / 'unknown-ip.db'
    monkeypatch.setenv('SEQUENCE_DB_PATH', str(path))
    app = FastAPI(); app.include_router(module.create_feedback_router())
    client = TestClient(app)
    assert client.post('/api/feedback', json=payload(), headers={'X-Forwarded-For': '203.0.113.1'}).status_code == 201
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT client_ip FROM user_feedback').fetchone()[0] is None


def test_stored_feedback_is_escaped_in_public_list_and_read_as_literal_data(feedback_client):
    client, path = feedback_client
    message = '<img src=x onerror=alert(1)>\n\'; DROP TABLE users; --'
    result = client.post('/api/feedback', json=payload(message=message, name='<script>bad</script>'))
    assert result.status_code == 201
    listing = client.get('/admin/feedback')
    assert listing.status_code == 200 and '&lt;img src=x onerror=alert(1)&gt;' in listing.text
    assert '<img src=x onerror=alert(1)>' not in listing.text and '<script>bad</script>' not in listing.text
    assert client.get('/admin?table=user_feedback').status_code == 200
    detail = client.get('/admin/api/record', params={'table': 'user_feedback', 'rowid': 1})
    assert detail.json()['sections'][0]['data']['message'] == message
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT COUNT(*) FROM users').fetchone()[0] == 1


def test_only_existing_admin_session_with_csrf_can_change_feedback_status(feedback_client):
    client, path = feedback_client
    receipt = client.post('/api/feedback', json=payload()).json()
    target = {'feedback_id': receipt['feedback_id'], 'status': 'reviewing'}
    route = '/admin/api/manage/feedback-status'
    assert client.post(route, json=target, headers={'Origin': 'http://testserver'}).status_code == 401
    assert client.get('/admin/api/manage/status').json()['authenticated'] is False
    login = client.post('/admin/api/manage/login', json={'username': 'reviewer', 'password': 'test-feedback-password'}, headers={'Origin': 'http://testserver'})
    assert login.status_code == 200
    headers = {'Origin': 'http://testserver', 'X-DB-CSRF': login.json()['csrf_token']}
    assert client.post(route, json=target).status_code == 403
    assert client.post(route, json=target, headers={**headers, 'Origin': 'https://elsewhere.test'}).status_code == 403
    assert client.post(route, json={**target, 'status': 'arbitrary'}, headers=headers).status_code == 422
    assert client.post(route, json={**target, 'message': 'overwritten'}, headers=headers).status_code == 422
    assert client.post(route, json=target, headers=headers).status_code == 200
    assert client.post(route, json={**target, 'feedback_id': str(uuid4())}, headers=headers).status_code == 404
    with sqlite3.connect(path) as conn:
        row = conn.execute('SELECT status,status_updated_by,status_updated_at,message FROM user_feedback').fetchone()
        assert row[:2] == ('reviewing', 'reviewer') and row[2]
        assert row[3] == '구역 선택 후 안내를 더 쉽게 보고 싶습니다.'
        assert conn.execute('SELECT COUNT(*) FROM users').fetchone()[0] == 1


def test_feedback_api_and_assets_do_not_expand_public_copilot_openapi(feedback_client):
    client, _ = feedback_client
    assert not any('feedback' in path for path in client.get('/openapi.json').json()['paths'])
    for path in ('/assets/copilot_feedback.js', '/assets/copilot_feedback.css', '/assets/copilot_feedback_admin.js'):
        assert client.get(path).status_code == 200


def test_asset_routes_cannot_be_used_to_read_other_server_files(feedback_client):
    client, _ = feedback_client
    original = client.get('/assets/copilot_feedback.js')
    substituted = client.get('/assets/copilot_feedback.js', params={'filename':'../db_admin_auth.py', 'content_type':'text/plain'})
    assert substituted.status_code == 200
    assert substituted.content == original.content
    assert substituted.headers['content-type'].startswith('text/javascript')


def test_json_looking_feedback_remains_literal_text_in_record_details(feedback_client):
    client, _ = feedback_client
    literal = '{"message":"literal user text"}'
    assert client.post('/api/feedback', json=payload(message=literal)).status_code == 201
    detail = client.get('/admin/api/record', params={'table':'user_feedback', 'rowid':1}).json()
    assert detail['sections'][0]['data']['message'] == literal


def test_unwritable_database_returns_a_failure_without_a_false_receipt(monkeypatch, tmp_path):
    module = feedback_module()
    blocker = tmp_path/'file'; blocker.write_text('not a directory')
    app = FastAPI(); app.include_router(module.create_feedback_router(module.FeedbackStore(str(blocker/'feedback.db'))))
    result = TestClient(app).post('/api/feedback', json=payload())
    assert result.status_code == 503
    assert 'feedback_id' not in result.json()


class Elements(HTMLParser):
    def __init__(self, markup):
        super().__init__(); self.by_id = {}; self.feed(markup)
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            assert attrs['id'] not in self.by_id, 'Duplicate element ID'
            self.by_id[attrs['id']] = (tag, attrs)


@pytest.mark.parametrize('page', ['docs', 'database', 'motion'])
def test_shared_navigation_renders_an_accessible_bounded_feedback_form(page):
    from trainer.demo_api.portal_navigation import portal_document
    document = portal_document('<html><head></head><body><main>page</main></body></html>', page)
    nodes = Elements(document).by_id
    assert 'copilot-feedback-open' in nodes, 'Shared navigation is missing feedback access'
    assert nodes['copilot-feedback-open'][1]['aria-controls'] == 'copilot-feedback-dialog'
    assert nodes['copilot-feedback-dialog'][0] == 'dialog'
    assert nodes['copilot-feedback-dialog'][1]['aria-labelledby'] == 'copilot-feedback-title'
    assert nodes['copilot-feedback-message'][1]['maxlength'] == '4000'
    assert 'required' in nodes['copilot-feedback-message'][1]
    assert nodes['copilot-feedback-name'][1]['maxlength'] == '80'
    assert nodes['copilot-feedback-state'][1]['aria-live'] == 'polite'
    assert portal_document(document, page) == document


def test_answer_column_migration_tolerates_another_worker_adding_it_first(tmp_path):
    """두 작업자가 동시에 열을 더하려 해도 늦은 쪽이 실패하지 않는다."""
    import sqlite3
    from trainer.demo_api.feedback import _SCHEMA, _add_answer_column
    conn = sqlite3.connect(tmp_path / "race.db")
    conn.execute(_SCHEMA)
    _add_answer_column(conn)
    _add_answer_column(conn)
    columns = [row[1] for row in conn.execute("PRAGMA table_info(user_feedback)")]
    assert columns.count("answer_context") == 1


def admin_login(client):
    login = client.post('/admin/api/manage/login', json={'username': 'reviewer', 'password': 'test-feedback-password'}, headers={'Origin': 'http://testserver'})
    assert login.status_code == 200


def seed_answers(client):
    client.post('/api/feedback', json=payload(message='일반 의견'))
    wrong = client.post('/api/feedback', json=payload(kind='bug', message='틀렸습니다', answer=answer_context())).json()
    client.post('/api/feedback', json=payload(message='=SUM(A1)', answer=answer_context(correct_answer='+1 맞는 답')))
    return wrong['feedback_id']


def test_admin_downloads_answer_feedback_jsonl_only_answer_rows(feedback_client):
    import json
    client, _ = feedback_client
    wrong_id = seed_answers(client)
    admin_login(client)
    response = client.get('/admin/api/feedback/answers.jsonl')
    assert response.status_code == 200
    assert response.headers['content-type'] == 'application/x-ndjson; charset=utf-8'
    assert response.headers['cache-control'] == 'no-store'
    assert response.headers['content-disposition'].startswith('attachment; filename="answer_feedback_') and response.headers['content-disposition'].endswith('.jsonl"')
    rows = [json.loads(line) for line in response.content.decode('utf-8').splitlines()]
    assert [r['label'] for r in rows] == ['wrong_answer', 'memo']
    first = rows[0]
    assert first['feedback_id'] == wrong_id and first['question'] == '어느 구역이 문제인가요?' and first['correct_answer'] == '4번 구역입니다.'
    assert first['message'] == '틀렸습니다' and first['conversation_id'] == 'conv-1' and first['map_num'] == 2 and first['server_version']
    assert '어느 구역이 문제인가요?' in response.content.decode('utf-8')  # not \u-escaped
    client.post('/admin/api/manage/feedback-status', json={'feedback_id': wrong_id, 'status': 'done'},
                headers={'Origin': 'http://testserver', 'X-DB-CSRF': client.get('/admin/api/manage/status').json()['csrf_token']})
    done = client.get('/admin/api/feedback/answers.jsonl', params={'status': 'done'}).content.decode().splitlines()
    assert len(done) == 1 and json.loads(done[0])['feedback_id'] == wrong_id
    assert client.get('/admin/api/feedback/answers.jsonl', params={'status': 'bogus'}).status_code == 422


def test_admin_downloads_answer_feedback_csv_with_bom_and_formula_guard(feedback_client):
    import csv, io
    client, _ = feedback_client
    seed_answers(client)
    admin_login(client)
    response = client.get('/admin/api/feedback/answers.csv')
    assert response.status_code == 200 and response.headers['content-type'] == 'text/csv; charset=utf-8'
    assert response.content.startswith('﻿'.encode('utf-8')) and response.headers['content-disposition'].endswith('.csv"')
    table = list(csv.reader(io.StringIO(response.content.decode('utf-8-sig'))))
    assert table[0][:5] == ['피드백 ID', '접수 시각(UTC)', '처리 상태', '종류(답변 오류/답변 메모)', '메모'] and len(table[0]) == 19
    assert [r[3] for r in table[1:]] == ['답변 오류', '답변 메모']
    assert table[2][4] == "'=SUM(A1)" and table[2][5] == "'+1 맞는 답"


@pytest.mark.parametrize('route', ['answers.jsonl', 'answers.csv'])
def test_answer_feedback_download_requires_admin_session(feedback_client, route):
    client, _ = feedback_client
    seed_answers(client)
    response = client.get('/admin/api/feedback/' + route)
    assert response.status_code == 401
    assert '어느 구역이' not in response.text and '틀렸습니다' not in response.text


def test_answer_feedback_download_is_empty_without_table_or_column(feedback_client):
    import re
    client, path = feedback_client
    admin_login(client)
    jsonl, csv_ = client.get('/admin/api/feedback/answers.jsonl'), client.get('/admin/api/feedback/answers.csv')
    assert jsonl.status_code == 200 and jsonl.content == b''
    assert csv_.status_code == 200 and len(csv_.content.decode('utf-8-sig').strip().splitlines()) == 1
    old = re.sub(r"\n answer_context TEXT,?", '', feedback_module()._SCHEMA).replace(' status_updated_by TEXT,', ' status_updated_by TEXT')
    with sqlite3.connect(path) as conn:
        conn.execute(old)
    assert client.get('/admin/api/feedback/answers.jsonl').content == b''
    assert len(client.get('/admin/api/feedback/answers.csv').content.decode('utf-8-sig').strip().splitlines()) == 1


def test_feedback_list_links_to_answer_downloads(feedback_client):
    client, _ = feedback_client
    text = client.get('/admin/feedback').text
    assert 'href="/admin/api/feedback/answers.jsonl"' in text and 'href="/admin/api/feedback/answers.csv"' in text
    assert '답변 피드백 내려받기(JSONL)' in text and '관리자 로그인' in text
