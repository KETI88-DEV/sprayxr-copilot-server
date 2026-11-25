"""Presentation helpers for finding public API operations without resetting Try it out."""
from __future__ import annotations

import html
import json
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

DOCS_ASSET_PREFIX = '/docs/assets'
_STATIC_DIR = Path(__file__).with_name('static')
_SWAGGER_ASSETS = {
    'swagger-ui-bundle.js': 'application/javascript',
    'swagger-ui.css': 'text/css',
    'favicon-32x32.png': 'image/png',
}
_METHODS = {'get', 'post', 'put', 'patch', 'delete', 'head', 'options', 'trace'}
_CSS = """
#api-docs-tools{position:sticky;top:calc(var(--copilot-nav-height,52px) + var(--copilot-banner-height,0px));z-index:100;background:#f5f8fd;border-bottom:1px solid #dce4f1;padding:10px 20px;font-family:system-ui,sans-serif;color:#23334b;box-sizing:border-box}
.api-docs-tools-inner{max-width:1460px;margin:auto;position:relative}.api-docs-find{display:flex;align-items:center;gap:10px}.api-docs-find label{font-size:12px;font-weight:700;white-space:nowrap}.api-docs-find input{box-sizing:border-box;min-width:0;width:min(480px,100%);padding:8px 11px;font-size:13px;border:1px solid #bdcbe1;border-radius:7px;background:#fff;color:#172b49}.api-docs-find input:focus{outline:2px solid #8cb1ff;outline-offset:1px}.api-docs-find button{border:0;background:#e3ebf8;border-radius:6px;padding:7px 10px;color:#304b7b;cursor:pointer}.api-docs-find small{font-size:11px;color:#61738f;white-space:nowrap}
.api-docs-sections{display:flex;gap:6px;overflow-x:auto;margin-top:8px;padding-bottom:2px}.api-docs-sections a{flex:none;text-decoration:none;font-size:12px;font-weight:650;padding:5px 10px;color:#40577d;background:#e9eff9;border-radius:6px;white-space:nowrap}.api-docs-sections a:hover,.api-docs-sections a:focus-visible{background:#d6e2fc;color:#244aa7}
#api-docs-results{position:absolute;top:42px;left:0;right:0;max-height:340px;overflow:auto;background:#fff;border:1px solid #c6d3e7;border-radius:9px;box-shadow:0 12px 25px #12234426;z-index:2}#api-docs-results[hidden]{display:none}.api-docs-result{display:grid;grid-template-columns:55px minmax(170px,1fr) minmax(180px,1fr);align-items:center;gap:10px;padding:11px 13px;border-bottom:1px solid #edf1f7;text-decoration:none;color:#213c66;font-size:12px}.api-docs-result:hover,.api-docs-result:focus-visible{background:#edf3ff}.api-docs-method{font-size:10px;font-weight:800;color:#3157d5}.api-docs-path{overflow-wrap:anywhere;font-family:ui-monospace,Consolas,monospace}.api-docs-result small{color:#667994;line-height:1.5}.api-docs-empty{margin:0;padding:18px;font-size:13px;color:#677990}
body[data-portal-page=docs] #swagger-ui .opblock,body[data-portal-page=docs] #swagger-ui .opblock-tag{scroll-margin-top:calc(var(--copilot-nav-height,52px) + var(--api-docs-tools-height,100px) + 12px)}
body[data-portal-page=docs] .swagger-ui .info .description details{margin:18px 0;padding:13px 16px;border:1px solid #dce4f1;border-radius:9px;background:#f7f9fd}body[data-portal-page=docs] .swagger-ui .info .description summary{font-weight:700;color:#375482;cursor:pointer}body[data-portal-page=docs] .swagger-ui .info .description details[open] summary{margin-bottom:14px}
@media(max-width:650px){#api-docs-tools{padding:9px 12px}.api-docs-find{gap:7px}.api-docs-find small{display:none}.api-docs-result{grid-template-columns:45px minmax(0,1fr)}.api-docs-result small{grid-column:2}.api-docs-sections a{font-size:11px;padding:5px 8px}}
"""


def api_docs_document(document: str, schema: dict, *, root_path: str = '') -> str:
    entries = []
    for path, operations in schema.get('paths', {}).items():
        for method, operation in operations.items():
            if method not in _METHODS or not isinstance(operation, dict):
                continue
            tag = next(iter(operation.get('tags') or ['default']))
            operation_id = operation.get('operationId')
            href = '#/' + quote(str(tag), safe='')
            if operation_id:
                href += '/' + quote(str(operation_id), safe='')
            entries.append({'method': method.upper(), 'path': path, 'summary': operation.get('summary', ''),
                            'tag': tag, 'href': href})
    links = ''.join(
        '<a href="#/' + quote(str(tag['name']), safe='') + '">' + html.escape(str(tag['name'])) + '</a>'
        for tag in schema.get('tags', []) if isinstance(tag, dict) and tag.get('name')
    )
    index = json.dumps(entries, ensure_ascii=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    tools = ('<section id="api-docs-tools" aria-label="API 탐색"><div class="api-docs-tools-inner">'
             '<div class="api-docs-find"><label for="api-docs-search">API 찾기</label>'
             '<input id="api-docs-search" type="search" placeholder="경로·기능·분류 검색 (예: query, 도막, 리포트)" '
             'autocomplete="off" aria-controls="api-docs-results" aria-expanded="false">'
             '<button type="button" id="api-docs-clear" hidden>지우기</button>'
             f'<small id="api-docs-count" role="status">전체 {len(entries)}개 API</small></div>'
             '<nav class="api-docs-sections" aria-label="API 분류 바로가기">' + links + '</nav>'
             '<div id="api-docs-results" hidden></div></div></section>')
    document = document.replace('</head>', f'<style>{_CSS}</style></head>', 1)
    document = document.replace('<div id="swagger-ui">', tools + '<div id="swagger-ui">', 1)
    asset = html.escape(root_path.rstrip('/') + DOCS_ASSET_PREFIX + '/api_docs.js', quote=True)
    return document.replace('</body>', '<script id="api-docs-index" type="application/json">' + index +
                            f'</script><script defer src="{asset}"></script></body>', 1)


def create_api_docs_router() -> APIRouter:
    router = APIRouter(prefix=DOCS_ASSET_PREFIX, include_in_schema=False)

    @router.get('/api_docs.js')
    def script() -> FileResponse:
        return FileResponse(_STATIC_DIR / 'api_docs.js',
                            media_type='application/javascript', headers={'Cache-Control': 'no-cache'})

    @router.get('/{filename}')
    def swagger_asset(filename: str) -> FileResponse:
        media_type = _SWAGGER_ASSETS.get(filename)
        if media_type is None:
            raise HTTPException(status_code=404, detail='Not Found')
        return FileResponse(_STATIC_DIR / 'swagger-ui' / filename, media_type=media_type,
                            headers={'Cache-Control': 'no-cache'})

    return router
