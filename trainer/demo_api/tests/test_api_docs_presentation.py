import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from trainer.demo_api.app import create_app


_SWAGGER_ASSETS = (
    ('swagger-ui-bundle.js', {'application/javascript', 'text/javascript'}, b'SwaggerUIBundle'),
    ('swagger-ui.css', {'text/css'}, b'.swagger-ui'),
    ('favicon-32x32.png', {'image/png'}, b'\x89PNG\r\n\x1a\n'),
)


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


@pytest.mark.parametrize('prefix', ['', '/copilot'])
def test_docs_serves_swagger_ui_without_an_external_connection(monkeypatch, tmp_path, prefix):
    monkeypatch.setenv('DEMO_LIVE_MODE', 'off')
    monkeypatch.setenv('SEQUENCE_DB_PATH', str(tmp_path / 'offline-docs.db'))
    client = TestClient(create_app(), root_path=prefix)
    page = client.get(f'{prefix}/docs')
    assert page.status_code == 200

    resources = set(re.findall(r'<(?:script|link)\b[^>]*\b(?:src|href)="([^"]+)"', page.text))
    assert resources and all(url.startswith('/') and not url.startswith('//') for url in resources)
    for name, content_types, marker in _SWAGGER_ASSETS:
        url = f'{prefix}/docs/assets/{name}'
        assert url in resources
        asset = client.get(url)
        assert asset.status_code == 200
        assert asset.headers['content-type'].split(';', 1)[0] in content_types
        assert marker in asset.content

    assert f"url: '{prefix}/openapi.json'" in page.text
    assert '"validatorUrl": null' in page.text
    schema = client.get(f'{prefix}/openapi.json')
    assert schema.status_code == 200
    paths = schema.json()['paths']
    assert 'post' in paths['/api/v1/copilot/query']
    assert not any(path.startswith('/docs') for path in paths)


def test_docs_swagger_assets_are_packaged_in_the_release():
    release_list = Path(__file__).resolve().parents[3] / 'deploy/release/release_files.txt'
    packaged = set(release_list.read_text(encoding='utf-8').splitlines())
    assert {f'trainer/demo_api/static/swagger-ui/{name}' for name, _, _ in _SWAGGER_ASSETS} <= packaged
