import json
import re

from fastapi.testclient import TestClient

from trainer.demo_api.app import create_app


def _index(document):
    match = re.search(r'<script id="api-docs-index" type="application/json">(.*?)</script>', document, re.S)
    assert match
    return json.loads(match.group(1))


def test_docs_search_indexes_only_public_operations_and_keeps_detailed_rules(monkeypatch, tmp_path):
    monkeypatch.setenv('DEMO_LIVE_MODE', 'off')
    monkeypatch.setenv('SEQUENCE_DB_PATH', str(tmp_path / 'docs.db'))
    client = TestClient(create_app())
    document = client.get('/docs').text
    entries = _index(document)
    schema = client.get('/openapi.json').json()
    public = {(method.upper(), path) for path, operations in schema['paths'].items()
              for method in operations if method in {'get', 'post', 'put', 'patch', 'delete', 'head', 'options'}}
    assert {(entry['method'], entry['path']) for entry in entries} == public
    assert not any(entry['path'].startswith('/admin') for entry in entries)
    assert all(entry['href'].startswith('#/') for entry in entries)
    assert 'id="api-docs-search"' in document
    assert 'aria-label="API 분류 바로가기"' in document
    assert client.get('/docs/assets/api_docs.js').status_code == 200
    description = schema['info']['description']
    assert '<details>' in description and '</details>' in description
    assert '누적 히트맵' in description and '접근 제어' in description


def test_docs_search_metadata_cannot_close_its_script_or_inject_links():
    from trainer.demo_api.api_docs import api_docs_document

    schema = {'tags': [{'name': 'tag </script><script>bad()</script>'}], 'paths': {
        '/example': {'post': {'tags': ['tag </script><script>bad()</script>'],
                              'summary': '</script><script>alert(1)</script>', 'operationId': 'test'}}}}
    document = api_docs_document('<html><head></head><body><div id="swagger-ui"></div></body></html>', schema)
    assert _index(document)[0]['summary'] == '</script><script>alert(1)</script>'
    assert '<script>alert(1)</script>' not in document
    assert '<script>bad()</script>' not in document
