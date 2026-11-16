"""Dedicated chat client for the same simulator Query API and answer feedback."""
from pathlib import Path

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse, HTMLResponse

from trainer.demo_api.portal_navigation import portal_document
from trainer.demo_api.query_chat_data import QueryChatDatasets
from trainer.demo_api.training_data_catalog import TrainingDataCatalog


def create_query_chat_router(catalog: TrainingDataCatalog | None = None) -> APIRouter:
    router = APIRouter(include_in_schema=False)
    static = Path(__file__).with_name('static')
    datasets = QueryChatDatasets(catalog)

    @router.get('/query-chat/api/options')
    def dataset_options(worker_id: str = Query('all', max_length=120),
                        process: str = Query('all', pattern='^(all|spray|blasting|grinding)$'),
                        scenario_id: str = Query('all', pattern='^(all|[1-9]|10)$')):
        return datasets.options(worker_id, process, scenario_id)

    @router.get('/query-chat/api/context')
    def dataset_context(dataset_id: str = Query(pattern='^[0-9a-f]{32}$'),
                        zone_key: str = Query('all', min_length=1, max_length=160)):
        return datasets.context(dataset_id, zone_key)

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
