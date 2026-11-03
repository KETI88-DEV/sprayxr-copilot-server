import sqlite3

from fastapi.testclient import TestClient
from trainer.demo_api.app import create_app


def test_real_http_calls_link_to_session_dashboard_and_parameter_modal(monkeypatch, tmp_path):
    path=tmp_path/'integration.db'
    monkeypatch.setenv('DEMO_LIVE_MODE','off')
    monkeypatch.setenv('SEQUENCE_DB_PATH',str(path))
    app=create_app()
    with TestClient(app) as client:
        body={'session_id':'3-INTEGRATION','user_id':'W','scenario_id':'3',
              'started_at':'2026-11-03T00:00:00Z','worker':{'skill_level':'novice'}}
        assert client.post('/api/v1/sessions/start',json=body).status_code==200
        assert client.post('/api/v1/sessions/start',json=body).status_code==200
        assert client.post('/api/v1/copilot/analyze-sequence',json={'session_id':'3-INTEGRATION','sequence_id':'SEQ-1'}).status_code==422
        ended={'session_id':'3-INTEGRATION','user_id':'W','scenario_id':'3','ended_at':'2026-11-03T00:05:00Z'}
        assert client.post('/api/v1/sessions/end',json=ended).status_code==200
        assert client.get('/admin/sessions').status_code==200
        assert client.get('/admin/sessions?limit=200').status_code==200
        page=client.get('/admin/sessions/view',params={'session_id':'3-INTEGRATION'})
        assert page.status_code==200 and 'data-detail-url=' in page.text
        assert page.headers['cache-control']=='no-store'
        errors=client.get('/admin/sessions/view',params={'session_id':'3-INTEGRATION','q':'SEQ-1','outcome':'errors','order':'oldest'})
        assert errors.status_code==200 and 'HTTP 422' in errors.text
        assert '/api/v1/sessions/end' not in errors.text
        with sqlite3.connect(path) as conn:
            rows=conn.execute('SELECT id,status_code,category FROM api_calls WHERE session_id=? ORDER BY id',('3-INTEGRATION',)).fetchall()
        assert len(rows)==4
        assert [r[1] for r in rows]==[200,200,422,200]
        detail=client.get('/admin/api/session-event',params={'session_id':'3-INTEGRATION','event_key':f'call:{rows[0][0]}'})
        assert detail.status_code==200
        assert '3-INTEGRATION' in detail.text and 'novice' in detail.text
        assert client.get('/admin/api/session-event',params={'session_id':'OTHER','event_key':f'call:{rows[0][0]}'}).status_code==404
        for method in ('POST','PUT','PATCH','DELETE'):
            assert client.request(method,'/admin/sessions').status_code==405
        assert client.get('/admin/assets/db_viewer.js').status_code==200
