"""Dedicated chat client for the same simulator Query API and answer feedback."""
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, HTMLResponse

from trainer.demo_api.portal_navigation import portal_document


def create_query_chat_router() -> APIRouter:
    router = APIRouter(include_in_schema=False)
    static = Path(__file__).with_name('static')

    @router.get('/query-chat', response_class=HTMLResponse)
    def query_chat():
        return HTMLResponse(portal_document((static / 'query_chat.html').read_text(encoding='utf-8'), 'chat'),
                            headers={'Cache-Control': 'no-store'})

    def asset_handler(filename: str, content_type: str):
        def asset():
            return FileResponse(static / filename, media_type=content_type, headers={'Cache-Control': 'no-cache'})
        return asset

    for filename, content_type in [('query_chat.js', 'text/javascript'), ('query_chat.css', 'text/css')]:
        router.add_api_route('/assets/' + filename, asset_handler(filename, content_type), methods=['GET'])
    return router
