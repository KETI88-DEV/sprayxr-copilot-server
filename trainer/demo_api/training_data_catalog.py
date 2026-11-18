"""Read-only catalog and exports for generated recording feature datasets."""
from __future__ import annotations

import csv
import gzip
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
from trainer.demo_api.training_worker_identity import worker_group_id, worker_group_label, worker_sort_key


_PROCESS_ORDER = {'spray': 0, 'blasting': 1, 'grinding': 2}


def _process_sort_key(value: str) -> int:
    # Python's stable sort preserves catalog order for unfamiliar processes.
    return _PROCESS_ORDER.get(value, len(_PROCESS_ORDER))


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_motion_sha256(path: Path) -> str | None:
    """Hash every original CSV cell, excluding only two Zone enrichments.

    Preserve row order, timestamps and cell spelling. No numeric tolerances,
    basename heuristics, or feature-summary comparisons establish equivalence.
    A changed/extra original column therefore stays a distinct recording.
    """
    csv.field_size_limit(max(csv.field_size_limit(), 64 * 1024 * 1024))
    opener = gzip.open if path.name.endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream, strict=True)
        columns = [name.strip().casefold() for name in next(reader, [])]
        if (len(set(columns)) != len(columns) or '' in columns
                or not {'ray point x', 'ray point y', 'ray point z', '분사'} <= set(columns)
                or not {'시간(초)', 'time'} & set(columns)):
            return None
        original = sorted((name, index) for index, name in enumerate(columns)
                          if name not in {'zone id', 'cur_sequence_heatmap'})
        digest = hashlib.sha256()
        digest.update(json.dumps([name for name, _ in original], ensure_ascii=False).encode('utf-8') + b'\n')
        frames = 0
        for row in reader:
            if not row:
                continue
            if len(row) != len(columns):
                return None
            digest.update(json.dumps([row[index] for _, index in original], ensure_ascii=False,
                                     separators=(',', ':')).encode('utf-8') + b'\n')
            frames += 1
        return digest.hexdigest() if frames else None


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
                    available = row.get('zone_id_available')
                    if not isinstance(available, bool):
                        available = row.get('zone_count', 0) > 0
                    result.append({**row, 'zone_id_available': available,
                                   'zone_id_status': 'present' if available else 'absent'})
            except OSError:
                continue
        return result

    @staticmethod
    def _filtered(rows: list[dict], *, worker_id='all', process='all', scenario_id='all', acquired_at='all', zone_mapping='all', quality_status='all', q='', **_) -> list[dict]:
        needle = q.strip().casefold()
        return [r for r in rows if (worker_id == 'all' or r['worker_id'] == worker_id
                                   or worker_group_id(r['worker_id'], r.get('worker_name')) == worker_id)
                    and (process == 'all' or r['process'] == process)
                    and (scenario_id == 'all' or r['scenario_id'] == scenario_id)
                    and (zone_mapping == 'all' or r['zone_id_status'] == zone_mapping)
                    and (quality_status == 'all' or (r.get('quality_status') == 'complete' if quality_status == 'complete'
                                                    else r.get('quality_status') != 'complete'))
                    and (acquired_at == 'all' or (acquisition_date(r.get('acquired_at')) is None if acquired_at == 'unknown'
                                                  else acquisition_date(r.get('acquired_at')) == acquired_at))
                    and (not needle or needle in ' '.join(str(r.get(k, '')) for k in ('dataset_name', 'worker_name', 'mode_label', 'scenario_id')).casefold())]

    def _selected(self, **filters) -> tuple[dict, list[dict], list[dict]]:
        index = self._index()
        rows = self._rows(index)
        selected = self._filtered(rows, **filters)
        selected.sort(key=lambda r: (worker_sort_key(r['worker_id'], r.get('worker_name')),
                                     _process_sort_key(r['process']), r['scenario_id'].zfill(8), r['dataset_name']))
        return index, rows, selected

    def list(self, *, page=1, limit=25, **filters) -> dict:
        index, rows, selected = self._selected(**filters)
        def grouped(field, label_field=None, labels=None):
            counts = Counter(r[field] for r in selected)
            names = {r[field]: r.get(label_field, r[field]) for r in selected}
            ordered = (sorted(counts.items(), key=lambda item: _process_sort_key(item[0])) if field == 'process'
                       else sorted(counts.items(), key=lambda item: (not item[0].isdigit(), int(item[0]) if item[0].isdigit() else 0, item[0])))
            return [{'id': k, 'label': (labels or {}).get(k, names[k]), 'count': v} for k, v in ordered]
        def options(field, label_field=None, labels=None):
            eligible = self._filtered(rows, **{**filters, field: 'all'})
            values = {r[field]: (labels or {}).get(r[field], r.get(label_field, r[field])) for r in eligible}
            ordered = (sorted(values.items(), key=lambda item: _process_sort_key(item[0])) if field == 'process'
                       else sorted(values.items(), key=lambda x: x[0].zfill(8)))
            return [{'id': k, 'label': v} for k, v in ordered]
        def worker_groups(eligible, *, counts=False):
            groups = {}
            for row in eligible:
                key = worker_group_id(row['worker_id'], row.get('worker_name'))
                group = groups.setdefault(key, {'id': key, 'label': worker_group_label(row['worker_id'], row.get('worker_name')),
                                                'count': 0})
                group['count'] += 1
            ordered = sorted(groups.values(), key=lambda item: worker_sort_key(item['id'], item['label']))
            if counts:
                ordered.sort(key=lambda item: -item['count'])
                return ordered
            return [{k: item[k] for k in ('id', 'label')} for item in ordered]
        labels = {'spray': '스프레이', 'blasting': '블라스팅', 'grinding': '그라인딩'}
        quality_rows = self._filtered(rows, **{**filters, 'quality_status': 'all'})
        qualities = [{'id': value, 'label': label} for value, label in
                     [('needs_review', '확인 필요'), ('complete', '완료')]
                     if any((r.get('quality_status') == 'complete') == (value == 'complete') for r in quality_rows)]
        pages = max(1, math.ceil(len(selected) / limit))
        page = min(max(1, page), pages)
        dates = [day for row in selected if (day := acquisition_date(row.get('acquired_at'))) is not None]
        all_dates = {acquisition_date(row.get('acquired_at')) for row in rows}
        date_options = [{'id': day, 'label': day} for day in sorted(all_dates - {None}, reverse=True)]
        if None in all_dates:
            date_options.append({'id': 'unknown', 'label': '미기록'})
        return {'available': bool(index), 'total': len(rows), 'filtered_count': len(selected), 'page': page, 'limit': limit, 'pages': pages,
                'build': {k: v for k, v in index.items() if k != 'records'},
                'summary': {'total': len(selected), 'worker_count': len({worker_group_id(r['worker_id'], r.get('worker_name')) for r in selected}),
                            'complete': sum(r['quality_status'] == 'complete' for r in selected),
                            'needs_review': sum(r['quality_status'] == 'needs_review' for r in selected),
                            'failed': index.get('failed_count', 0) if len(selected) == len(rows) else 0,
                            'heightmap_count': sum(r['heightmap_count'] for r in selected),
                            'with_zone_id': sum(r['zone_id_available'] for r in selected),
                            'without_zone_id': sum(not r['zone_id_available'] for r in selected),
                            'dated_count': len(dates), 'acquisition_date_min': min(dates) if dates else None,
                            'acquisition_date_max': max(dates) if dates else None,
                            'by_worker': worker_groups(selected, counts=True), 'by_mode': grouped('scenario_id', 'mode_label'),
                            'by_process': grouped('process', labels=labels)},
                'options': {'workers': worker_groups(self._filtered(rows, **{**filters, 'worker_id': 'all'})),
                            'processes': options('process', labels=labels),
                            'modes': options('scenario_id', 'mode_label'), 'qualities': qualities, 'dates': date_options,
                            'zone_mappings': [{'id': 'present', 'label': '있음'}, {'id': 'absent', 'label': '없음'}]},
                'items': [{k: v for k, v in r.items() if k != 'artifact_sha256'} for r in selected[(page - 1) * limit:page * limit]]}

    def audit_duplicates(self, *, verify_sources: bool = False, metadata_only: bool = False) -> dict:
        """Return review evidence only; never mutate catalog, DB, or sources.

        The default verifies artifact bytes, then compares declared
        provenance hashes. ``verify_sources=True`` additionally hashes the real
        source files and compares their complete original CSV motion streams.
        ``metadata_only=True`` reads just the index's declared source hashes;
        it explicitly leaves both artifact and original bytes unverified.
        Even a verified match is a review group, never a deletion instruction.
        """
        if verify_sources and metadata_only:
            raise ValueError('metadata_only and verify_sources are mutually exclusive')
        index = self._index()
        rows = index.get('records', []) if metadata_only else self._rows(index)
        inspected, unavailable = [], []
        for row in rows:
            try:
                if metadata_only:
                    if (not isinstance(row, dict) or not isinstance(row.get('dataset_id'), str)
                            or len(row['dataset_id']) != 32
                            or any(c not in '0123456789abcdef' for c in row['dataset_id'])
                            or not isinstance(row.get('process'), str) or not row['process']):
                        raise ValueError('invalid_index_record')
                    artifact = {}
                    source = {'sha256': row.get('source_sha256')}
                else:
                    artifact = self._detail_row(row)
                    source = artifact['provenance']['source_csv']
                declared = source.get('sha256')
                if (not isinstance(declared, str) or len(declared) != 64
                        or any(c not in '0123456789abcdef' for c in declared)):
                    raise ValueError('missing_source_hash')
                if not metadata_only and any(artifact.get('metadata', {}).get(field) != row.get(field)
                       for field in ('worker_id', 'process', 'scenario_id')):
                    raise ValueError('artifact_identity_mismatch')
                motion = None
                if verify_sources:
                    path = self._safe(self.sources, source['relative_path'])
                    before = path.stat()
                    if source.get('bytes') not in (None, before.st_size) or _source_sha256(path) != declared:
                        raise ValueError('source_hash_mismatch')
                    try:
                        motion = _raw_motion_sha256(path)
                    except (OSError, ValueError, csv.Error, UnicodeError, EOFError):
                        # Identical source bytes still provide evidence when a
                        # legacy CSV is unsuitable for the stricter motion pass.
                        motion = None
                    after = path.stat()
                    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        raise ValueError('source_changed_during_audit')
                inspected.append({'row': row, 'source_sha256': declared, 'raw_motion_sha256': motion,
                                  'heightmap_hashes': sorted(item.get('sha256', '')
                                      for item in artifact.get('provenance', {}).get('heightmaps', []))})
            except (OSError, ValueError, KeyError, TypeError) as error:
                reason = str(error) if isinstance(error, ValueError) and str(error) in {
                    'missing_source_hash', 'artifact_identity_mismatch', 'source_hash_mismatch',
                    'source_changed_during_audit', 'invalid_index_record'} else 'artifact_or_source_unavailable'
                unavailable.append({'dataset_id': row.get('dataset_id') if isinstance(row, dict) else None,
                                    'reason_code': reason})

        parents = list(range(len(inspected)))
        def parent(index):
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index
        matches = {}
        for index, item in enumerate(inspected):
            for kind in ('source_sha256', 'raw_motion_sha256'):
                if item[kind]:
                    key = (item['row']['process'], kind, item[kind])
                    if key in matches:
                        parents[parent(index)] = parent(matches[key])
                    else:
                        matches[key] = index
        buckets = {}
        for index, item in enumerate(inspected):
            buckets.setdefault(parent(index), []).append(item)
        groups = []
        for members in buckets.values():
            if len(members) < 2:
                continue
            records = [item['row'] for item in members]
            reasons = []
            if metadata_only:
                reasons.append('artifact_bytes_not_verified')
            if not verify_sources:
                reasons.append('source_bytes_not_verified')
            if any(row.get('frame_count') == 0 for row in records):
                reasons.append('empty_recording')
            if len({row.get('worker_id') for row in records}) != 1:
                reasons.append('worker_identity_conflict')
            if any(not row.get('acquired_at') for row in records):
                reasons.append('acquisition_time_unknown')
            if len({row.get('acquired_at') for row in records}) != 1:
                reasons.append('acquisition_time_conflict')
            if len({row.get('scenario_id') for row in records}) != 1:
                reasons.append('scenario_conflict')
            if len({tuple(item['heightmap_hashes']) for item in members}) != 1:
                reasons.append('heightmap_provenance_conflict')
            ranked = sorted(records, key=lambda row: (
                -int(row.get('zone_id_available') is True),
                -int(row.get('source_kind') == 'converted'),
                -int(row.get('quality_status') == 'complete'), row['dataset_id']))
            groups.append({'process': records[0]['process'],
                           'match_basis': 'source_sha256' if len({item['source_sha256'] for item in members}) == 1 else 'raw_motion_sha256',
                           'source_bytes_verified': verify_sources,
                           'artifact_bytes_verified': not metadata_only,
                           'dataset_ids': sorted(row['dataset_id'] for row in records),
                           'record_count': len(records), 'review_reasons': reasons,
                           'preferred_for_review': ranked[0]['dataset_id'],
                           'preference_basis': 'zone_mapped_then_converted_then_complete',
                           'requires_operator_review': True})
        groups.sort(key=lambda group: (group['process'], group['dataset_ids']))
        return {'audit_version': 1, 'read_only': True, 'catalog_record_count': len(rows),
                'inspected_count': len(inspected), 'source_bytes_verified': verify_sources,
                'artifact_bytes_verified': not metadata_only,
                'motion_verified_count': sum(item['raw_motion_sha256'] is not None for item in inspected),
                'match_group_count': len(groups), 'matching_record_count': sum(group['record_count'] for group in groups),
                'additional_matching_records': sum(group['record_count'] - 1 for group in groups),
                'groups': groups, 'unavailable': unavailable,
                'policy': 'Review only. Filename/session labels never prove duplication; retain uncertain identity/time evidence.'}

    def artifact(self, dataset_id: str) -> Path:
        row = next((r for r in self._rows(self._index()) if r['dataset_id'] == dataset_id), None)
        if row is None:
            raise FileNotFoundError('학습 데이터를 찾을 수 없습니다.')
        path = self._safe(self.root, f'records/{dataset_id}.json')
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['artifact_sha256']:
            raise ValueError('학습 데이터 파일 검증에 실패했습니다.')
        return path

    def detail(self, dataset_id: str) -> dict:
        row = next((r for r in self._rows(self._index()) if r['dataset_id'] == dataset_id), None)
        if row is None:
            raise FileNotFoundError('학습 데이터를 찾을 수 없습니다.')
        return self._detail_row(row)

    def _detail_row(self, row: dict) -> dict:
        """Load an already selected catalog row, verifying and parsing the same bytes.

        The viewer selects all records for a mode from one validated index. It
        must not reload and stat the entire catalog for each selected artifact.
        """
        path = self._safe(self.root, f"records/{row['dataset_id']}.json")
        content = path.read_bytes()
        if len(content) != row['json_bytes'] or hashlib.sha256(content).hexdigest() != row['artifact_sha256']:
            raise ValueError('학습 데이터 파일 검증에 실패했습니다.')
        return json.loads(content)

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
                  'acquired_at', 'quality_status', 'frame_count', 'zone_count', 'zone_id_available', 'zone_id_status',
                  'source_kind', 'sequence_heatmap_count', 'heightmap_count', 'json_bytes']
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

    def filters(worker_id: str, process: str, scenario_id: str, acquired_at: str, q: str, zone_mapping: str, quality_status: str) -> dict:
        if acquired_at not in {'all', 'unknown'} and acquisition_date(acquired_at) != acquired_at:
            raise HTTPException(422, '훈련날짜는 YYYY-MM-DD 형식의 유효한 날짜여야 합니다.')
        return dict(worker_id=worker_id, process=process, scenario_id=scenario_id, acquired_at=acquired_at, q=q,
                    zone_mapping=zone_mapping, quality_status=quality_status)

    @router.get('')
    def listing(worker_id: str = Query('all', max_length=120), process: str = Query('all', pattern='^(all|spray|blasting|grinding)$'),
                scenario_id: str = Query('all', max_length=120), acquired_at: str = Query('all', max_length=10), q: str = Query('', max_length=100),
                zone_mapping: str = Query('all', pattern='^(all|present|absent)$'),
                quality_status: str = Query('all', pattern='^(all|complete|needs_review)$'),
                page: int = Query(1, ge=1), limit: int = Query(25, ge=1, le=200)):
        return store.list(page=page, limit=limit, **filters(worker_id, process, scenario_id, acquired_at, q, zone_mapping, quality_status))

    @router.get('/export.csv')
    def export_csv(worker_id: str = Query('all', max_length=120), process: str = Query('all', pattern='^(all|spray|blasting|grinding)$'),
                   scenario_id: str = Query('all', max_length=120), acquired_at: str = Query('all', max_length=10), q: str = Query('', max_length=100),
                   zone_mapping: str = Query('all', pattern='^(all|present|absent)$'),
                   quality_status: str = Query('all', pattern='^(all|complete|needs_review)$')):
        return Response(store.csv_export(**filters(worker_id, process, scenario_id, acquired_at, q, zone_mapping, quality_status)), media_type='text/csv; charset=utf-8',
                        headers={'Content-Disposition': 'attachment; filename="training-datasets.csv"'})

    @router.get('/export.zip')
    def export_zip(worker_id: str = Query('all', max_length=120), process: str = Query('all', pattern='^(all|spray|blasting|grinding)$'),
                   scenario_id: str = Query('all', max_length=120), acquired_at: str = Query('all', max_length=10), q: str = Query('', max_length=100),
                   zone_mapping: str = Query('all', pattern='^(all|present|absent)$'),
                   quality_status: str = Query('all', pattern='^(all|complete|needs_review)$')):
        try:
            content = store.zip_export(**filters(worker_id, process, scenario_id, acquired_at, q, zone_mapping, quality_status))
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


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Read-only training catalog duplicate evidence audit')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--verify-sources', action='store_true',
                      help='Read/hash every selected source and compare original CSV motion cells')
    mode.add_argument('--metadata-only', action='store_true',
                      help='Read only index source_sha256 declarations; do not verify artifact/source bytes')
    arguments = parser.parse_args()
    print(json.dumps(TrainingDataCatalog(arguments.output_root, arguments.source_root)
                     .audit_duplicates(verify_sources=arguments.verify_sources,
                                       metadata_only=arguments.metadata_only), ensure_ascii=False, indent=2))
