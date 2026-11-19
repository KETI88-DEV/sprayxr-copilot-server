"""Page visits are bounded, private operational data, never trainee identity."""
import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.testclient import TestClient


def client_for(store):
    from trainer.demo_api.page_access import PageAccessMiddleware
    app = FastAPI(docs_url=None)

    @app.get('/docs')
    def docs():
        return HTMLResponse('docs')

    @app.get('/')
    def home():
        return HTMLResponse('home')

    @app.get('/admin')
    def admin():
        return HTMLResponse('admin')

    @app.get('/feedback')
    def feedback():
        return HTMLResponse('feedback')

    @app.get('/admin/assets/script.js')
    def static():
        return HTMLResponse('not a page')

    @app.get('/api/health')
    def health():
        return HTMLResponse('not a page')

    app.add_middleware(PageAccessMiddleware, store=store)
    return TestClient(app, client=('192.0.2.17', 12345))


def test_only_successful_html_page_gets_are_recorded_without_sensitive_query_or_headers(tmp_path):
    from trainer.demo_api.page_access import PageAccessStore
    store = PageAccessStore(tmp_path / 'visits.sqlite3')
    client = client_for(store)
    assert client.get('/?token=private&worker=Someone', headers={
        'X-Forwarded-For': '203.0.113.9', 'X-Copilot-Source': 'tsb',
        'Referer': 'http://external.example/secret?password=private',
        'Cookie': 'sensitive=private',
    }).status_code == 200
    client.get('/docs')
    client.get('/admin', headers={'Referer': 'http://testserver/'})
    client.get('/admin/assets/script.js')
    client.get('/api/health')
    client.get('/missing')
    client.post('/docs')
    client.head('/docs')
    client.get('/docs', headers={'Sec-Purpose': 'prefetch'})
    result = store.records()
    assert result['total'] == 3
    assert [row['path'] for row in result['rows']] == ['/admin', '/docs', '/']
    first = result['rows'][-1]
    assert first['client_ip'] == '192.0.2.17'
    assert first['page'] == 'home'
    assert first['source'] == 'external'
    assert first['identity'] == 'unconfirmed'
    assert first['source_basis'] == 'request_header_estimate'
    assert 'private' not in str(result) and 'Someone' not in str(result)
    assert '203.0.113.9' not in str(result) and 'tsb' not in str(result)


def test_public_feedback_html_is_included_as_a_page(tmp_path):
    from trainer.demo_api.page_access import PageAccessStore
    store = PageAccessStore(tmp_path / 'visits.db')
    client = client_for(store)
    assert client.get('/feedback').status_code == 200
    assert store.records()['rows'][0]['page'] == 'feedback'


def test_visit_filters_pagination_and_retention_are_bounded(tmp_path):
    from trainer.demo_api.page_access import PageAccessStore
    store = PageAccessStore(tmp_path / 'visits.sqlite3', max_rows=3, retention_days=30)
    now = datetime.now(timezone.utc)
    for days_ago, page, ip in ((40, 'home', '192.0.2.1'), (3, 'docs', '192.0.2.2'),
                              (2, 'home', '192.0.2.1'), (1, 'home', '192.0.2.2'), (0, 'admin', '192.0.2.1')):
        store.append(created_at=(now - timedelta(days=days_ago)).isoformat(), client_ip=ip,
                     path={'home': '/', 'docs': '/docs', 'admin': '/admin'}[page],
                     source='direct', source_basis='unavailable')
    result = store.records(limit=2)
    assert result['total'] == 3 and len(result['rows']) == 2
    assert store.records(page=2, limit=2)['rows'][0]['page'] == 'home'
    assert store.records(page_filter='home')['total'] == 2
    assert store.records(client_ip='192.0.2.2')['total'] == 1
    assert store.records(since=now - timedelta(hours=12))['total'] == 1
    assert store.records(until=now - timedelta(hours=12))['total'] == 2
    assert store.records(source='external')['total'] == 0
    with sqlite3.connect(store.path) as conn:
        assert conn.execute('SELECT COUNT(*) FROM page_access').fetchone()[0] == 3
    assert result['retention_days'] == 30


def test_retention_drops_expired_rows_and_rejects_public_database_path(tmp_path, monkeypatch):
    from trainer.demo_api.page_access import PageAccessStore
    database = tmp_path / 'records.db'
    monkeypatch.setenv('PAGE_ACCESS_DB_PATH', str(database))
    with pytest.raises(ValueError, match='separate'):
        PageAccessStore.for_database(database)
    monkeypatch.delenv('PAGE_ACCESS_DB_PATH')
    store = PageAccessStore.for_database(database)
    assert store.path != database
    store.append(created_at=(datetime.now(timezone.utc) - timedelta(days=31)).isoformat(),
                 client_ip=None, path='/', source='direct', source_basis='unavailable')
    assert store.records()['total'] == 0
    with sqlite3.connect(store.path) as conn:
        assert conn.execute('SELECT COUNT(*) FROM page_access').fetchone()[0] == 0


def test_visit_storage_failure_does_not_break_served_page(tmp_path):
    from trainer.demo_api.page_access import PageAccessStore
    blocking = tmp_path / 'not-a-directory'
    blocking.write_text('blocked')
    client = client_for(PageAccessStore(blocking / 'visits.db'))
    assert client.get('/docs').status_code == 200


def network_store(tmp_path):
    from trainer.demo_api.page_access import PageAccessStore
    store = PageAccessStore(tmp_path / 'visits.db')
    for ip in ('192.0.2.17', '192.0.2.77', '192.0.2.200', '2001:db8::7', '203.0.113.8'):
        store.append(created_at=datetime.now(timezone.utc).isoformat(), client_ip=ip,
                     path='/docs', source='direct', source_basis='unavailable')
    return store


def test_current_ip_network_rules_use_longest_prefix_without_confirming_identity(tmp_path, monkeypatch):
    store = network_store(tmp_path)
    rules = tmp_path / 'network-rules.json'
    rules.write_text(json.dumps({'version': 1, 'rules': [
        {'network': '192.0.2.0/24', 'label': '파트너 공유망', 'basis': '운영자가 제공한 공유망 범위'},
        {'network': '192.0.2.0/25', 'label': '검증 구역망', 'basis': '운영자가 확인한 구역망 범위'},
        {'network': '192.0.2.77', 'label': '현재 운영 PC', 'basis': '운영자가 확인한 현재 연결 IP'},
        {'network': '2001:db8::/32', 'label': 'IPv6 공유망', 'basis': '문서로 확인한 시험망'},
    ]}), encoding='utf-8')
    monkeypatch.setenv('PAGE_ACCESS_IP_LABELS_PATH', str(rules))
    result = store.records()
    rows = {row['client_ip']: row for row in result['rows']}
    assert rows['192.0.2.17']['network_label'] == '검증 구역망 (추정)'
    assert rows['192.0.2.77']['network_label'] == '현재 운영 PC (추정)'
    assert rows['192.0.2.200']['network_label'] == '파트너 공유망 (추정)'
    assert rows['2001:db8::7']['network_label'] == 'IPv6 공유망 (추정)'
    assert rows['203.0.113.8']['network_label'] == '미확인'
    assert rows['192.0.2.77']['network_basis'] == '현재 규칙·조회 시 분류: 운영자가 확인한 현재 연결 IP'
    assert all(row['identity'] == 'unconfirmed' for row in result['rows'])
    assert result['network_classification_basis'] == 'current_rules_at_read'
    assert str(rules) not in str(result) and 'rules' not in str(rows)


def test_network_rule_updates_reclassify_old_visits_without_storing_duplicate_ip_labels(tmp_path, monkeypatch):
    store = network_store(tmp_path)
    rules = tmp_path / 'network-rules.json'
    monkeypatch.setenv('PAGE_ACCESS_IP_LABELS_PATH', str(rules))
    assert store.records(client_ip='192.0.2.77')['rows'][0]['network_label'] == '미확인'
    for label in ('이전 공유망', '현재 공유망'):
        rules.write_text(json.dumps({'version': 1, 'rules': [
            {'network': '192.0.2.77/32', 'label': label, 'basis': '공유망 운영자 제공 정보'}]}), encoding='utf-8')
        row = store.records(client_ip='192.0.2.77')['rows'][0]
        assert row['network_label'] == label + ' (추정)'
        assert row['network_basis'].startswith('현재 규칙·조회 시 분류:')
    with sqlite3.connect(store.path) as conn:
        assert 'network_label' not in {row[1] for row in conn.execute('PRAGMA table_info(page_access)')}


@pytest.mark.parametrize('payload', [
    'not-json',
    '{"version":1,"rules":[]}' + ' ' * 65536,
    json.dumps({'version': 1, 'rules': [{'network': '192.0.2.0/99', 'label': 'private', 'basis': 'private'}]}),
    json.dumps({'version': 1, 'rules': [{'network': '192.0.2.17', 'label': 'private', 'basis': 'private', 'secret': 'private'}]}),
    json.dumps({'version': 1, 'rules': [{'network': '192.0.2.17', 'label': 'private', 'basis': 'private'}] * 129}),
    json.dumps({'version': 1, 'rules': [
        {'network': '192.0.2.17', 'label': 'private', 'basis': 'private'},
        {'network': '192.0.2.17/32', 'label': 'conflicting', 'basis': 'private'}]}),
    json.dumps({'version': 1, 'rules': [{'network': '192.0.2.17', 'label': 'x' * 81, 'basis': 'private'}]}),
    json.dumps({'version': 1, 'rules': [{'network': '192.0.2.17', 'label': 'private', 'basis': 'x' * 241}]}),
    json.dumps({'version': 1, 'rules': [{'network': '192.0.2.17', 'label': 'private\nlabel', 'basis': 'private'}]}),
])
def test_invalid_or_excessive_ip_rules_fail_closed_without_disclosing_configuration(tmp_path, monkeypatch, payload):
    store = network_store(tmp_path)
    rules = tmp_path / 'sensitive-path.json'
    rules.write_text(payload, encoding='utf-8')
    monkeypatch.setenv('PAGE_ACCESS_IP_LABELS_PATH', str(rules))
    result = store.records()
    assert all(row['network_label'] == '미확인' for row in result['rows'])
    assert str(rules) not in str(result) and 'private' not in str(result)


def test_forwarded_ip_is_never_used_for_network_classification(tmp_path, monkeypatch):
    from trainer.demo_api.page_access import PageAccessStore
    rules = tmp_path / 'network-rules.json'
    rules.write_text(json.dumps({'version': 1, 'rules': [
        {'network': '203.0.113.9', 'label': '파트너망', 'basis': '운영자 제공 IP'}]}), encoding='utf-8')
    monkeypatch.setenv('PAGE_ACCESS_IP_LABELS_PATH', str(rules))
    store = PageAccessStore(tmp_path / 'visits.db')
    client_for(store).get('/docs', headers={'X-Forwarded-For': '203.0.113.9'})
    row = store.records()['rows'][0]
    assert row['client_ip'] == '192.0.2.17' and row['network_label'] == '미확인'
