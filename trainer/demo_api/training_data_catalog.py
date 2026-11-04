"""Read-only catalog and exports for generated recording feature datasets."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import zipfile
from collections import Counter
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response
from trainer.demo_api.training_data_dates import acquisition_date


class TrainingDataCatalog:
    def __init__(self, output_root: Path | None = None, source_root: Path | None = None):
        db = Path(os.getenv('SEQUENCE_DB_PATH', '/output/eval_results/sequences.db'))
        self.root = Path(output_root or os.getenv('TRAINING_DATA_ROOT', str(db.parent / 'training_datasets'))).resolve()
        self.sources = Path(source_root or os.getenv('EXPERT_REFERENCE_ROOT', '/expert_reference_data')).resolve()

    def _index(self) -> dict:
        try:
            value = json.loads((self.root / 'catalog.json').read_text(encoding='utf-8'))
            if not isinstance(value, dict) or not isinstance(value.get('records'), list):
                return {}
            return value
        except (OSError, ValueError):
            return {}

    @staticmethod
    def _safe(root: Path, relative: str) -> Path:
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise FileNotFoundError('등록된 파일을 찾을 수 없습니다.')
        return path

    def _rows(self, index: dict) -> list[dict]:
        result = []
        for row in index.get('records', []):
            dataset_id = row.get('dataset_id', '')
            if len(dataset_id) != 32 or any(c not in '0123456789abcdef' for c in dataset_id):
                continue
            try:
                p = self._safe(self.root, f'records/{dataset_id}.json')
                if p.stat().st_size == row.get('json_bytes'):
                    result.append(row)
            except OSError:
                continue
        return result

    def _selected(self, *, worker_id='all', process='all', scenario_id='all', acquired_at='all', q='', **_) -> tuple[dict, list[dict], list[dict]]:
        index = self._index()
        rows = self._rows(index)
        needle = q.strip().casefold()
        selected = [r for r in rows if (worker_id == 'all' or r['worker_id'] == worker_id)
                    and (process == 'all' or r['process'] == process)
                    and (scenario_id == 'all' or r['scenario_id'] == scenario_id)
                    and (acquired_at == 'all' or (acquisition_date(r.get('acquired_at')) is None if acquired_at == 'unknown'
                                                  else acquisition_date(r.get('acquired_at')) == acquired_at))
                    and (not needle or needle in ' '.join(str(r.get(k, '')) for k in ('dataset_name', 'worker_name', 'mode_label', 'scenario_id')).casefold())]
        selected.sort(key=lambda r: (r['worker_name'], r['process'], r['scenario_id'].zfill(8), r['dataset_name']))
        return index, rows, selected

    def list(self, *, page=1, limit=25, **filters) -> dict:
        index, rows, selected = self._selected(**filters)
        def grouped(field, label_field=None, labels=None):
            counts = Counter(r[field] for r in selected)
            names = {r[field]: r.get(label_field, r[field]) for r in selected}
            ordered = sorted(counts.items(), key=lambda item: (not item[0].isdigit(), int(item[0]) if item[0].isdigit() else 0, item[0]))
            return [{'id': k, 'label': (labels or {}).get(k, names[k]), 'count': v} for k, v in ordered]
        def options(field, label_field=None, labels=None):
            values = {r[field]: (labels or {}).get(r[field], r.get(label_field, r[field])) for r in rows}
            return [{'id': k, 'label': v} for k, v in sorted(values.items(), key=lambda x: x[0].zfill(8))]
        labels = {'spray': '스프레이', 'blasting': '블라스팅'}
        pages = max(1, math.ceil(len(selected) / limit))
        page = min(max(1, page), pages)
        dates = [day for row in selected if (day := acquisition_date(row.get('acquired_at'))) is not None]
        all_dates = {acquisition_date(row.get('acquired_at')) for row in rows}
        date_options = [{'id': day, 'label': day} for day in sorted(all_dates - {None}, reverse=True)]
        if None in all_dates:
            date_options.append({'id': 'unknown', 'label': '미기록'})
        return {'available': bool(index), 'total': len(rows), 'filtered_count': len(selected), 'page': page, 'limit': limit, 'pages': pages,
                'build': {k: v for k, v in index.items() if k != 'records'},
                'summary': {'total': len(selected), 'worker_count': len({r['worker_id'] for r in selected}),
                            'complete': sum(r['quality_status'] == 'complete' for r in selected),
                            'needs_review': sum(r['quality_status'] == 'needs_review' for r in selected),
                            'failed': index.get('failed_count', 0) if len(selected) == len(rows) else 0,
                            'heightmap_count': sum(r['heightmap_count'] for r in selected),
                            'dated_count': len(dates), 'acquisition_date_min': min(dates) if dates else None,
                            'acquisition_date_max': max(dates) if dates else None,
                            'by_worker': grouped('worker_id', 'worker_name'), 'by_mode': grouped('scenario_id', 'mode_label'),
                            'by_process': grouped('process', labels=labels)},
                'options': {'workers': options('worker_id', 'worker_name'), 'processes': options('process', labels=labels),
                            'modes': options('scenario_id', 'mode_label'), 'dates': date_options},
                'items': [{k: v for k, v in r.items() if k != 'artifact_sha256'} for r in selected[(page - 1) * limit:page * limit]]}

    def artifact(self, dataset_id: str) -> Path:
        row = next((r for r in self._rows(self._index()) if r['dataset_id'] == dataset_id), None)
        if row is None:
            raise FileNotFoundError('학습 데이터를 찾을 수 없습니다.')
        path = self._safe(self.root, f'records/{dataset_id}.json')
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['artifact_sha256']:
            raise ValueError('학습 데이터 파일 검증에 실패했습니다.')
        return path

    def detail(self, dataset_id: str) -> dict:
        return json.loads(self.artifact(dataset_id).read_text(encoding='utf-8'))

    def image(self, dataset_id: str, image_index: int, view='raw') -> Path:
        data = self.detail(dataset_id)
        if image_index < 0 or image_index >= len(data['provenance']['heightmaps']):
            raise FileNotFoundError('텍스처를 찾을 수 없습니다.')
        if view == 'thickness':
            if not data['coating']['maps'][image_index].get('thickness_image_url'):
                raise FileNotFoundError('두께 미리보기가 없습니다.')
            return self._safe(self.root, f'previews/{dataset_id}-{image_index}.png')
        item = data['provenance']['heightmaps'][image_index]
        path = self._safe(self.sources, item['relative_path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('원본 텍스처 검증에 실패했습니다.')
        return path

    def csv_export(self, **filters) -> bytes:
        _, _, rows = self._selected(**filters)
        out = io.StringIO(newline='')
        writer = csv.writer(out)
        fields = ['dataset_id', 'dataset_name', 'worker_name', 'role', 'process', 'scenario_id', 'mode_label',
                  'acquired_at', 'quality_status', 'frame_count', 'zone_count', 'heightmap_count', 'json_bytes']
        writer.writerow(fields)
        for row in rows:
            values = [(acquisition_date(row.get(k)) or '') if k == 'acquired_at' else '' if row.get(k) is None else str(row[k]) for k in fields]
            writer.writerow(["'" + v if v.startswith(('=', '+', '-', '@')) else v for v in values])
        return ('\ufeff' + out.getvalue()).encode('utf-8')

    def zip_export(self, **filters) -> bytes:
        index, _, rows = self._selected(**filters)
        if sum(r['json_bytes'] for r in rows) > 250 * 1024 * 1024:
            raise ValueError('내보낼 데이터가 큽니다. 작업자나 모드로 범위를 좁혀 주세요.')
        out = io.BytesIO()
        with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for row in rows:
                archive.write(self.artifact(row['dataset_id']), f"records/{row['dataset_id']}.json")
            archive.writestr('manifest.json', json.dumps({'schema_version': index.get('schema_version'), 'count': len(rows),
                                                        'counting_rule': index.get('counting_rule'), 'records': rows}, ensure_ascii=False, indent=2))
        return out.getvalue()


def create_training_data_router(catalog: TrainingDataCatalog | None = None) -> APIRouter:
    store = catalog or TrainingDataCatalog()
    router = APIRouter(prefix='/admin/api/training-data', include_in_schema=False)

    def filters(worker_id: str, process: str, scenario_id: str, acquired_at: str, q: str) -> dict:
        if acquired_at not in {'all', 'unknown'} and acquisition_date(acquired_at) != acquired_at:
            raise HTTPException(422, '훈련날짜는 YYYY-MM-DD 형식의 유효한 날짜여야 합니다.')
        return dict(worker_id=worker_id, process=process, scenario_id=scenario_id, acquired_at=acquired_at, q=q)

    @router.get('')
    def listing(worker_id: str = Query('all', max_length=120), process: str = Query('all', pattern='^(all|spray|blasting)$'),
                scenario_id: str = Query('all', max_length=120), acquired_at: str = Query('all', max_length=10), q: str = Query('', max_length=100),
                page: int = Query(1, ge=1), limit: int = Query(25, ge=1, le=200)):
        return store.list(page=page, limit=limit, **filters(worker_id, process, scenario_id, acquired_at, q))

    @router.get('/export.csv')
    def export_csv(worker_id: str = Query('all', max_length=120), process: str = Query('all', pattern='^(all|spray|blasting)$'),
                   scenario_id: str = Query('all', max_length=120), acquired_at: str = Query('all', max_length=10), q: str = Query('', max_length=100)):
        return Response(store.csv_export(**filters(worker_id, process, scenario_id, acquired_at, q)), media_type='text/csv; charset=utf-8',
                        headers={'Content-Disposition': 'attachment; filename="training-datasets.csv"'})

    @router.get('/export.zip')
    def export_zip(worker_id: str = Query('all', max_length=120), process: str = Query('all', pattern='^(all|spray|blasting)$'),
                   scenario_id: str = Query('all', max_length=120), acquired_at: str = Query('all', max_length=10), q: str = Query('', max_length=100)):
        try:
            content = store.zip_export(**filters(worker_id, process, scenario_id, acquired_at, q))
        except (OSError, ValueError) as error:
            raise HTTPException(409, str(error)) from error
        return Response(content, media_type='application/zip', headers={'Content-Disposition': 'attachment; filename="training-datasets.zip"'})

    @router.get('/{dataset_id}')
    def detail(dataset_id: str):
        try:
            return store.detail(dataset_id)
        except (OSError, ValueError, KeyError) as error:
            raise HTTPException(404, str(error)) from error

    @router.get('/{dataset_id}/download')
    def download(dataset_id: str):
        try:
            path = store.artifact(dataset_id)
            name = store.detail(dataset_id)['metadata']['dataset_name']
        except (OSError, ValueError, KeyError) as error:
            raise HTTPException(404, str(error)) from error
        return FileResponse(path, media_type='application/json', filename=f'{name}.json', headers={'Cache-Control': 'no-store'})

    @router.get('/{dataset_id}/images/{image_index}')
    def image(dataset_id: str, image_index: int, view: str = Query('raw', pattern='^(raw|thickness)$')):
        try:
            path = store.image(dataset_id, image_index, view)
        except (OSError, ValueError, KeyError) as error:
            raise HTTPException(404, str(error)) from error
        return FileResponse(path, media_type='image/png', headers={'Cache-Control': 'private, max-age=60'})

    return router
