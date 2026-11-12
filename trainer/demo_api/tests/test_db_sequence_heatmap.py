import base64
import io
import json
import sqlite3
from urllib.parse import urlencode

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from trainer.demo_api.db_viewer import create_db_viewer_router


def _client(tmp_path, monkeypatch):
    path = tmp_path / 'data.db'
    stream = io.BytesIO()
    Image.new('RGB', (4, 4), (1, 0, 255)).save(stream, format='PNG')
    heatmap = dict(map_num=1, x0=0, y0=0, width=4, height=4, img_width=4, img_height=4,
                   format='v2;thickness=RG16;alpha=B8;scale_um=20000', encoding='png;base64',
                   data=base64.b64encode(stream.getvalue()).decode())
    body = {'session_id': '999_1234_1', 'sequence_id': 'SEQ-001', 'scenario_id': '999',
            'sequence_setting': {'tool_type': 'spray_gun'}, 'sequence_result': {'cur_sequence_heatmap': [heatmap]}}
    with sqlite3.connect(path) as c:
        c.execute('CREATE TABLE sessions(session_id TEXT,user_id TEXT,scenario_id TEXT,session_status TEXT,started_at TEXT,ended_at TEXT,created_at TEXT,target_thickness_um REAL)')
        c.execute('INSERT INTO sessions VALUES(?,?,?,?,?,?,?,?)', (body['session_id'], 'worker', '999', 'ACTIVE', '2026-11-04T01:00:00Z', None, '2026-11-04T01:00:00Z', 150))
        c.execute('CREATE TABLE sequences(session_key TEXT,sequence_id TEXT,received_at TEXT,body TEXT)')
        c.execute('INSERT INTO sequences VALUES(?,?,?,?)', (body['session_id'], 'SEQ-001', '2026-11-04T01:01:00Z', json.dumps(body)))
        c.execute('CREATE TABLE api_calls(id INTEGER PRIMARY KEY,session_id TEXT,sequence_id TEXT,status_code INTEGER,path TEXT,method TEXT,request_json TEXT,response_json TEXT,received_at TEXT,request_id TEXT)')
        c.execute('INSERT INTO api_calls VALUES(?,?,?,?,?,?,?,?,?,?)', (1, body['session_id'], 'SEQ-001', 200, '/api/v1/copilot/analyze-sequence', 'POST', json.dumps({'capture': 'truncated'}), '{}', '2026-11-04T01:01:00Z', 'request-1'))
    monkeypatch.setenv('SEQUENCE_DB_PATH', str(path))
    app = FastAPI()
    app.include_router(create_db_viewer_router())
    return TestClient(app), path, body


def test_sequence_record_links_decoded_analysis_and_original_parameters(tmp_path, monkeypatch):
    client, path, body = _client(tmp_path, monkeypatch)
    detail = client.get('/admin/api/record?table=sequences&rowid=1').json()
    assert detail['sections'][0]['data']['body'] == body
    assert detail['heatmaps_url'].startswith('/admin/api/sequence-heatmaps?')
    result = client.get(detail['heatmaps_url'])
    assert result.status_code == 200
    data = result.json()
    assert data['available'] and data['sequence_id'] == 'SEQ-001'
    assert data['analysis']['summary']['painted_pixels'] == 16
    assert data['analysis']['summary']['target_thickness_um'] == 150
    assert data['analysis']['summary']['coverage_percent'] is None
    assert client.get(data['image_url']).content.startswith(b'\x89PNG')
    assert client.get(data['thickness_image_url']).content.startswith(b'\x89PNG')
    assert client.get('/admin/assets/db_sequence_heatmap.js').status_code == 200
    with sqlite3.connect(path) as c:
        assert c.execute('SELECT COUNT(*) FROM sequences').fetchone()[0] == 1
        assert json.loads(c.execute('SELECT body FROM sequences').fetchone()[0]) == body


def test_session_event_and_truncated_success_call_use_registered_sequence(tmp_path, monkeypatch):
    client, path, body = _client(tmp_path, monkeypatch)
    for url in ['/admin/api/session-event?' + urlencode({'session_id': body['session_id'], 'event_key': 'sequence:SEQ-001'}),
                '/admin/api/record?table=api_calls&rowid=1',
                '/admin/api/session-event?' + urlencode({'session_id': body['session_id'], 'event_key': 'call:1'})]:
        response = client.get(url)
        assert response.status_code == 200
        result = client.get(response.json()['heatmaps_url'])
        assert result.status_code == 200 and result.json()['available']
    with sqlite3.connect(path) as c:
        c.execute('UPDATE api_calls SET status_code=409')
    rejected = client.get('/admin/api/record?table=api_calls&rowid=1').json()
    assert not rejected.get('heatmaps_url')


def test_corrupt_base64_and_no_heatmaps_leave_detail_available(tmp_path, monkeypatch):
    client, path, body = _client(tmp_path, monkeypatch)
    body['sequence_result']['cur_sequence_heatmap'][0]['data'] = 'not base64'
    with sqlite3.connect(path) as c:
        c.execute('UPDATE sequences SET body=?', (json.dumps(body),))
    result = client.get('/admin/api/sequence-heatmaps?table=sequences&rowid=1').json()
    assert result['available']
    assert result['analysis']['status'] == 'unavailable'
    assert result['image_url'] is None
    body['sequence_result'] = {}
    with sqlite3.connect(path) as c:
        c.execute('UPDATE sequences SET body=?', (json.dumps(body),))
    detail = client.get('/admin/api/record?table=sequences&rowid=1')
    assert detail.status_code == 200 and not detail.json().get('heatmaps_url')
    assert client.get('/admin/api/sequence-heatmaps?table=sequences&rowid=1').json()['available'] is False


def test_heatmap_endpoints_are_scoped_read_only_and_out_of_range_safe(tmp_path, monkeypatch):
    client, path, body = _client(tmp_path, monkeypatch)
    assert client.get('/admin/api/sequence-heatmaps?table=admin_change_log&rowid=1').status_code == 404
    assert client.get('/admin/api/sequence-heatmaps?table=sequences&rowid=999').status_code == 404
    assert client.get('/admin/api/sequence-heatmaps?table=sequences&rowid=1&map_index=99').status_code in {404, 422}
    assert client.get('/admin/api/sequence-heatmaps?table=sequences&rowid=1&session_id=x&event_key=y').status_code == 422
    assert client.get('/admin/api/sequence-heatmaps/image?table=sequences&rowid=1&map_index=4').status_code == 404
    assert client.post('/admin/api/sequence-heatmaps?table=sequences&rowid=1').status_code == 405


def test_multiple_crops_are_selected_without_using_the_first_repeatedly(tmp_path, monkeypatch):
    client, path, body = _client(tmp_path, monkeypatch)
    import copy
    second = copy.deepcopy(body['sequence_result']['cur_sequence_heatmap'][0])
    second['x0'] = 10
    body['sequence_result']['cur_sequence_heatmap'].append(second)
    with sqlite3.connect(path) as c:
        c.execute('UPDATE sequences SET body=?', (json.dumps(body),))
    data = client.get('/admin/api/sequence-heatmaps?table=sequences&rowid=1&map_index=1').json()
    assert len(data['maps']) == 2 and data['selected_index'] == 1
    assert data['maps'][1]['x0'] == 10


def test_nonfinite_legacy_geometry_does_not_break_event_or_fallback_json(tmp_path, monkeypatch):
    client, path, body = _client(tmp_path, monkeypatch)
    body['sequence_result']['cur_sequence_heatmap'][0]['x0'] = float('nan')
    with sqlite3.connect(path) as c:
        c.execute('UPDATE sequences SET body=?', (json.dumps(body),))
    for source in [{'session_id': body['session_id'], 'event_key': 'sequence:SEQ-001'},
                   {'table': 'api_calls', 'rowid': 1}]:
        response = client.get('/admin/api/sequence-heatmaps', params=source)
        assert response.status_code == 200
        assert response.json()['analysis']['summary']['coverage_percent'] is None


def test_all_stored_crop_indices_can_be_viewed(tmp_path, monkeypatch):
    client, path, body = _client(tmp_path, monkeypatch)
    sample = body['sequence_result']['cur_sequence_heatmap'][0]
    body['sequence_result']['cur_sequence_heatmap'] = [dict(sample, x0=i) for i in range(65)]
    with sqlite3.connect(path) as c:
        c.execute('UPDATE sequences SET body=?', (json.dumps(body),))
    response = client.get('/admin/api/sequence-heatmaps?table=sequences&rowid=1&map_index=64')
    assert response.status_code == 200
    assert response.json()['selected_index'] == 64


def test_default_target_session_is_shown_as_default(tmp_path, monkeypatch):
    client, path, body = _client(tmp_path, monkeypatch)
    url = client.get('/admin/api/record?table=sequences&rowid=1').json()['heatmaps_url']
    assert client.get(url).json()['analysis']['summary']['target_basis'] == 'session_target'
    with sqlite3.connect(path) as c:
        c.execute('ALTER TABLE sessions ADD COLUMN target_basis TEXT')
        c.execute("UPDATE sessions SET target_thickness_um=160, target_basis='default_target'")
    analysis = client.get(url).json()['analysis']
    assert analysis['summary']['target_basis'] == 'default_target'
    assert analysis['summary']['target_thickness_um'] == 160
    assert analysis['basis_label'] == '기본 목표 160µm 기준'
