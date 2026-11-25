"""Home, packaged release posts and a small read-only training overview."""
from __future__ import annotations
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from trainer.demo_api import __version__
from trainer.demo_api.portal_navigation import portal_document
from trainer.demo_api.training_data_catalog import TrainingDataCatalog
from trainer.demo_api.training_worker_identity import is_unknown_worker

_ASSETS = Path(__file__).with_name('assets')
_QUERY_QA_FILES = {
    1: 'query_api_qa_2026-11-25.json',
    2: 'query_api_qa_round2_2026-11-25.json',
    3: 'query_api_qa_round3_2026-11-25.json',
}

@lru_cache(maxsize=1)
def release_posts():
    path=_ASSETS/'portal_releases.json'
    data=json.loads(path.read_text(encoding='utf-8'))
    return data['updates']


@lru_cache(maxsize=len(_QUERY_QA_FILES))
def query_qa_report(round_number: int = 1):
    path = _ASSETS / _QUERY_QA_FILES[round_number]
    return json.loads(path.read_text(encoding='utf-8'))


def create_home_router(catalog: TrainingDataCatalog | None=None) -> APIRouter:
    router=APIRouter(include_in_schema=False)
    catalog=catalog or TrainingDataCatalog()
    static=Path(__file__).with_name('static')

    @router.get('/')
    def landing():
        return RedirectResponse('/home',status_code=307)

    @router.get('/home',response_class=HTMLResponse)
    def home():
        return HTMLResponse(portal_document((static/'portal_home.html').read_text(encoding='utf-8'),'home'),
                            headers={'Cache-Control':'no-store'})

    @router.get('/home/api/updates')
    def updates():
        return {'updates':release_posts(),'count':len(release_posts())}

    @router.get('/home/api/updates/{version}')
    def update_detail(version: str):
        match=next((row for row in release_posts() if row['version']==version),None)
        if match is None:
            raise HTTPException(404,'업데이트 기록을 찾을 수 없습니다.')
        if match.get('has_query_qa'):
            return {**match, 'query_qa': query_qa_report(match.get('query_qa_round', 1))}
        return match

    @router.get('/home/api/summary')
    def summary():
        index=catalog._index()
        rows=[row for row in index.get('records',[]) if isinstance(row,dict)]
        complete=sum(row.get('quality_status')=='complete' for row in rows)
        workers={str(row.get('worker_id','')) for row in rows
                 if not is_unknown_worker(row.get('worker_id'),row.get('worker_name'))}
        return {'version':__version__,'training':{'total':len(rows),'complete':complete,
                'needs_review':len(rows)-complete,'known_workers':len(workers),
                'unnamed_records':sum(is_unknown_worker(row.get('worker_id'),row.get('worker_name')) for row in rows),
                'processes':dict(Counter(str(row.get('process','unknown')) for row in rows))},
                'count_basis':'등록 카탈로그 기준','release_count':len(release_posts())}

    def static_asset(filename: str,content_type: str):
        def asset():
            return FileResponse(static/filename,media_type=content_type,headers={'Cache-Control':'no-cache'})
        return asset

    for name,mime in [('portal_home.js','text/javascript'),('portal_home.css','text/css')]:
        router.add_api_route('/assets/'+name,static_asset(name,mime),methods=['GET'])
    return router
