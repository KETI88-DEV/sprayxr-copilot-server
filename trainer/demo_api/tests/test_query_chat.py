from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.query_chat import create_query_chat_router
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.request_provenance import capture_provenance


def test_chat_uses_same_api_and_server_receipt_for_feedback():
    app = FastAPI()
    app.include_router(create_query_chat_router())
    client = TestClient(app)
    response = client.get('/query-chat')
    assert response.status_code == 200
    assert response.text.count('id="copilot-navigation"') == 1
    assert 'aria-current="page"' in response.text
    js = client.get('/assets/query_chat.js').text
    assert '/api/v1/copilot/query' in js and '/api/query-feedback' in js
    assert 'X-Copilot-Query-Turn' in js
    assert "user_id:'query-chat'" in js
    assert 'innerHTML' not in js
    assert 'window.location' not in js
    assert client.get('/assets/query_chat.js?name=/etc/passwd&media=text/plain').text == js


def test_chat_loads_evidence_labels_before_the_query_client():
    app = FastAPI()
    app.include_router(create_query_chat_router())
    client = TestClient(app)
    html = client.get('/query-chat').text
    assert html.index('/assets/query_answer_source.js') < html.index('/assets/query_chat.js')
    asset = client.get('/assets/query_answer_source.js')
    assert asset.status_code == 200
    assert asset.headers['cache-control'] == 'no-cache'
    assert 'CopilotAnswerSource' in asset.text


def test_chat_has_question_sidebar_library_hooks_and_allowlisted_preset_asset():
    app = FastAPI()
    app.include_router(create_query_chat_router())
    client = TestClient(app)
    html = client.get('/query-chat').text
    assert 'id="question-sidebar"' in html
    assert 'aria-controls="question-sidebar"' in html
    assert 'data-knowledge-open' in html
    assert html.index('/assets/query_chat_presets.js') < html.index('/assets/query_chat.js')
    presets = client.get('/assets/query_chat_presets.js')
    assert presets.status_code == 200 and presets.headers['cache-control'] == 'no-cache'
    assert 'CopilotQuestionPresets' in presets.text
    assert client.get('/assets/query_chat_presets.js?filename=/etc/passwd').text == presets.text
    assert client.get('/assets/not-allowlisted.js').status_code == 404


def test_recorded_answer_receipt_identifies_the_saved_answer():
    store = SequenceStore(':memory:')
    first = store.record_query_turn(session_key='CHAT:query-chat',user_id='query-chat',
        prompt='질문',answer='답변',task_type='knowledge_search',created_at='2026-11-16T00:00:00Z')
    second = store.record_query_turn(session_key='CHAT:query-chat',user_id='query-chat',
        prompt='다음 질문',answer='다음 답변',task_type='knowledge_search',created_at='2026-11-16T00:00:01Z')
    assert isinstance(first, int) and second > first
    assert store._conn.execute('SELECT answer FROM query_turns WHERE id=?',(first,)).fetchone()[0] == '답변'


def test_chat_provenance_is_distinct_from_simulator():
    facts = capture_provenance({'headers':[(b'x-copilot-source',b'query-chat')],
                               'client':('192.168.0.100',12345),'path':'/api/v1/copilot/query'})
    assert facts['caller_source'] == 'query-chat'
