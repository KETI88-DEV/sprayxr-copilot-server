"""Read-only previews of heatmaps already retained in sequence JSON records."""
from __future__ import annotations

import json
import math
import sqlite3
from functools import lru_cache
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response

from trainer.demo_api.db_record_detail import json_safe, record_detail
from trainer.demo_api.db_sessions import session_event_detail
from trainer.demo_api.sequence_heatmap_analysis import prepare_heatmap
from trainer.demo_api.telemetry_contracts import process_from_tool_type

BASE = '/admin/api/sequence-heatmaps'
SEMANTICS = ('해당 시퀀스가 전송한 영역 크롭의 누적 도막 상태입니다. 이전 시퀀스·회차의 도막을 포함할 수 있으며, '
             '이번 시퀀스의 증분 도포량이나 전체 Zone의 최종 상태를 의미하지 않습니다.')


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')}


def _find_body(value, depth=0):
    if depth > 12:
        return None
    if isinstance(value, dict):
        result = value.get('sequence_result')
        if isinstance(result, dict) and result.get('cur_sequence_heatmap'):
            return value
        for key, child in value.items():
            if key in {'sequence_info', 'data'} and not isinstance(child, (dict, list)):
                continue
            if key == 'sequence_info':
                continue
            if isinstance(child, str) and key in {'body', 'request_json', 'payload', 'request_body'}:
                try:
                    child = json.loads(child)
                except (ValueError, TypeError):
                    continue
            found = _find_body(child, depth + 1)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_body(child, depth + 1)
            if found is not None:
                return found
    return None


def _source_detail(conn, source):
    if source.get('table') is not None:
        return record_detail(conn, source['table'], source['rowid'])
    return session_event_detail(conn, source['session_id'], source['event_key'])


def _stored_call_body(conn, source):
    required = {'status_code', 'path', 'session_id', 'sequence_id'}
    if not required.issubset(_columns(conn, 'api_calls')):
        return None
    call = None
    if source.get('table') == 'api_calls':
        call = conn.execute('SELECT * FROM api_calls WHERE rowid=?', (source['rowid'],)).fetchone()
    elif source.get('event_key', '').startswith('call:'):
        call = conn.execute('SELECT * FROM api_calls WHERE id=? AND session_id=?',
                            (source['event_key'][5:], source['session_id'])).fetchone()
    if call is None or not isinstance(call['status_code'], int) or not 200 <= call['status_code'] < 300:
        return None
    if str(call['path']).rstrip('/') not in {'/api/v1/copilot/analyze-sequence', '/api/v1/telemetry/sequence'}:
        return None
    if not {'session_key', 'sequence_id', 'body'}.issubset(_columns(conn, 'sequences')):
        return None
    row = conn.execute('SELECT body FROM sequences WHERE session_key=? AND sequence_id=?',
                       (call['session_id'], call['sequence_id'])).fetchone()
    if row is not None:
        try:
            return _find_body(json.loads(row['body']))
        except (TypeError, ValueError):
            pass
    return None


def _context(conn, detail, source):
    body = _find_body(detail)
    note = None
    if body is None:
        body = _stored_call_body(conn, source)
        if body is not None:
            note = '호출 로그에 생략된 히트맵은 연결된 저장 시퀀스에서 읽었습니다.'
    if body is None:
        return None
    maps = body.get('sequence_result', {}).get('cur_sequence_heatmap')
    if isinstance(maps, dict):
        maps = [maps]
    if not isinstance(maps, list):
        return None
    # A malformed individual crop is retained as an error card, not a shifted index.
    maps = [item if isinstance(item, dict) else {} for item in maps]
    if not maps:
        return None
    session_id = body.get('session_id') or source.get('session_id')
    scenario = body.get('scenario_id')
    target = None
    columns = _columns(conn, 'sessions')
    if session_id and 'session_id' in columns:
        session = conn.execute('SELECT * FROM sessions WHERE session_id=?', (str(session_id),)).fetchone()
        if session is not None:
            if scenario is None and 'scenario_id' in columns:
                scenario = session['scenario_id']
            # A default stored by the session is not a target the session chose.
            if 'target_thickness_um' in columns and not (
                    'target_basis' in columns and session['target_basis'] == 'default_target'):
                value = session['target_thickness_um']
                if isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
                    target = float(value)
    setting = body.get('sequence_setting')
    process = process_from_tool_type(setting.get('tool_type')) if isinstance(setting, dict) else None
    return {'maps': maps, 'session_id': session_id, 'sequence_id': body.get('sequence_id'),
            'scenario_id': str(scenario) if scenario is not None else '', 'target': target,
            'process': process or 'unknown', 'source_note': note}


def attach_heatmap_link(conn, detail: dict, **source) -> dict:
    if _context(conn, detail, source) is not None:
        detail['heatmaps_url'] = BASE + '?' + urlencode({k: v for k, v in source.items() if v is not None})
    return detail


@lru_cache(maxsize=12)
def _prepared(item_json: str, scenario_id: str, process: str, target: float | None):
    return prepare_heatmap(json.loads(item_json), scenario_id=scenario_id, process=process,
                           target_thickness_um=target)


def _source(table: str | None = Query(None, max_length=200), rowid: int | None = Query(None, ge=-(2**63), le=2**63-1),
            session_id: str | None = Query(None, max_length=256), event_key: str | None = Query(None, max_length=400)):
    row_source = table is not None and rowid is not None and session_id is None and event_key is None
    event_source = session_id is not None and event_key is not None and table is None and rowid is None
    if not (row_source or event_source):
        raise HTTPException(422, '레코드 또는 세션 이벤트 중 하나의 조회 대상을 지정하세요.')
    return {k: v for k, v in dict(table=table, rowid=rowid, session_id=session_id, event_key=event_key).items() if v is not None}


def create_sequence_heatmap_router(read_database) -> APIRouter:
    router = APIRouter(include_in_schema=False)

    def load(conn, source):
        detail = _source_detail(conn, source) if conn is not None else None
        if detail is None:
            raise HTTPException(404, '해당 기록을 찾을 수 없습니다.')
        return _context(conn, detail, source)

    def prepared(context, index):
        if context is None or index >= len(context['maps']):
            raise HTTPException(404, '해당 히트맵을 찾을 수 없습니다.')
        return _prepared(json.dumps(context['maps'][index], sort_keys=True, ensure_ascii=False),
                         context['scenario_id'], context['process'], context['target'])

    @router.get(BASE)
    def analysis(source: dict = Depends(_source), map_index: int = Query(0, ge=0)):
        with read_database() as conn:
            context = load(conn, source)
        if context is None:
            return {'available': False, 'maps': [], 'semantics': SEMANTICS, 'reason': '시퀀스 히트맵이 없습니다.'}
        result = prepared(context, map_index)
        maps = []
        for i, item in enumerate(context['maps']):
            maps.append({'index': i, 'label': f'크롭 {i+1}',
                         **{k: item.get(k) for k in ('map_num', 'x0', 'y0', 'width', 'height', 'img_width', 'img_height')}})
        image_base = BASE + '/image?' + urlencode({**source, 'map_index': map_index})
        return json_safe({'available': True, **{k: v for k, v in context.items() if k not in {'maps', 'target'}},
                'semantics': SEMANTICS, 'maps': maps, 'selected_index': map_index,
                'analysis': result.payload, 'image_url': image_base + '&view=raw' if result.original_png is not None else None,
                'thickness_image_url': image_base + '&view=thickness' if result.thickness_png is not None else None})

    @router.get(BASE + '/image')
    def image(source: dict = Depends(_source), map_index: int = Query(0, ge=0),
              view: str = Query('raw', pattern='^(raw|thickness)$')):
        with read_database() as conn:
            context = load(conn, source)
        item = prepared(context, map_index)
        raw = item.original_png if view == 'raw' else item.thickness_png
        if raw is None:
            raise HTTPException(404, '이미지로 표시할 수 없는 히트맵입니다.')
        return Response(raw, media_type='image/png', headers={'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff'})

    return router
