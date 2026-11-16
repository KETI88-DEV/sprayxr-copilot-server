"""Zone availability is based on mapped values, including imported raw data."""

import csv
import hashlib
import io
import json
import zipfile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.training_data_catalog import TrainingDataCatalog, create_training_data_router


@pytest.fixture
def zone_catalog(tmp_path):
    """Use persisted artifacts directly so ingestion is not under test here."""
    output = tmp_path / 'built'
    records = output / 'records'
    records.mkdir(parents=True)
    definitions = [
        {'dataset_name': 'legacy-mapped', 'zone_count': 2},
        {'dataset_name': 'legacy-unassigned', 'zone_count': 0},
        {'dataset_name': 'raw-grinding', 'process': 'grinding', 'zone_count': 3,
         'zone_id_available': False, 'zone_id_status': 'absent', 'source_kind': 'raw',
         'sequence_heatmap_count': 7, 'worker_id': 'raw-worker', 'worker_name': '원본 작업자'},
        {'dataset_name': 'converted-mapped', 'zone_count': 1, 'zone_id_available': True,
         'zone_id_status': 'present', 'source_kind': 'converted', 'sequence_heatmap_count': 0},
    ]
    rows = []
    for number, definition in enumerate(definitions, 1):
        dataset_id = f'{number:032x}'
        row = {'dataset_id': dataset_id, 'worker_id': 'legacy-worker', 'worker_name': '기존 작업자',
               'role': 'trainee', 'process': 'spray', 'scenario_id': '3', 'mode_label': '시나리오 3',
               'acquired_at': '2026-10-20', 'quality_status': 'needs_review', 'frame_count': 3,
               'heightmap_count': 0, **definition}
        artifact = {'schema_version': '1.0', 'dataset_id': dataset_id, 'metadata': dict(row),
                    'features': {'frame_count': 3, 'zones': []}, 'quality': {'status': 'needs_review', 'issues': []},
                    'coating': {'maps': []},
                    'provenance': {'source_csv': {'filename': definition['dataset_name'] + '.csv',
                                                  'headers': ['Zone ID']}, 'heightmaps': []}}
        content = json.dumps(artifact, ensure_ascii=False).encode('utf-8')
        (records / f'{dataset_id}.json').write_bytes(content)
        rows.append({**row, 'json_bytes': len(content), 'artifact_sha256': hashlib.sha256(content).hexdigest()})
    (output / 'catalog.json').write_text(json.dumps({'schema_version': '1.0', 'records': rows}), encoding='utf-8')
    return TrainingDataCatalog(output, tmp_path / 'source')


def test_zone_availability_uses_explicit_boolean_then_legacy_mapped_zone_count(zone_catalog):
    # A Zone ID header does not establish availability; explicit false wins over old counts.
    listing = zone_catalog.list()
    items = {item['dataset_name']: item for item in listing['items']}
    assert {name: (item['zone_id_available'], item['zone_id_status']) for name, item in items.items()} == {
        'legacy-mapped': (True, 'present'), 'legacy-unassigned': (False, 'absent'),
        'raw-grinding': (False, 'absent'), 'converted-mapped': (True, 'present'),
    }
    assert listing['summary']['with_zone_id'] == 2
    assert listing['summary']['without_zone_id'] == 2
    assert listing['options']['zone_mappings'] == [
        {'id': 'present', 'label': '있음'}, {'id': 'absent', 'label': '없음'},
    ]


def test_zone_filter_keeps_raw_unmapped_records_and_counts_current_selection(zone_catalog):
    absent = zone_catalog.list(zone_mapping='absent')
    assert absent['total'] == 4 and absent['filtered_count'] == 2
    assert {item['dataset_name'] for item in absent['items']} == {'legacy-unassigned', 'raw-grinding'}
    assert absent['summary']['with_zone_id'] == 0
    assert absent['summary']['without_zone_id'] == 2
    present = zone_catalog.list(zone_mapping='present')
    assert {item['dataset_name'] for item in present['items']} == {'legacy-mapped', 'converted-mapped'}
    grinding = zone_catalog.list(zone_mapping='absent', process='grinding', acquired_at='2026-10-20')
    assert grinding['filtered_count'] == 1
    assert grinding['items'][0]['source_kind'] == 'raw'
    assert grinding['items'][0]['sequence_heatmap_count'] == 7
    assert grinding['summary']['without_zone_id'] == 1
    assert {'id': 'grinding', 'label': '그라인딩'} in grinding['options']['processes']
    assert zone_catalog.list(zone_mapping='present', process='grinding')['filtered_count'] == 0


def test_zone_csv_export_has_availability_and_retains_import_metadata(zone_catalog):
    exported = list(csv.DictReader(io.StringIO(zone_catalog.csv_export(zone_mapping='absent').decode('utf-8-sig'))))
    assert {row['dataset_name'] for row in exported} == {'legacy-unassigned', 'raw-grinding'}
    assert {(row['zone_id_available'], row['zone_id_status']) for row in exported} == {('False', 'absent')}
    raw = next(row for row in exported if row['dataset_name'] == 'raw-grinding')
    assert raw['source_kind'] == 'raw' and raw['sequence_heatmap_count'] == '7'


@pytest.fixture
def zone_client(zone_catalog):
    app = FastAPI()
    app.include_router(create_training_data_router(zone_catalog))
    with TestClient(app) as client:
        yield client


def test_learning_routes_accept_grinding_and_apply_zone_filter_to_all_exports(zone_client):
    params = {'process': 'grinding', 'zone_mapping': 'absent'}
    listing = zone_client.get('/admin/api/training-data', params=params)
    assert listing.status_code == 200
    assert listing.json()['filtered_count'] == 1
    item = listing.json()['items'][0]
    detail = zone_client.get('/admin/api/training-data/' + item['dataset_id']).json()
    assert detail['metadata']['zone_id_available'] is False
    assert detail['metadata']['sequence_heatmap_count'] == 7
    csv_response = zone_client.get('/admin/api/training-data/export.csv', params=params)
    assert csv_response.status_code == 200
    rows = list(csv.DictReader(io.StringIO(csv_response.content.decode('utf-8-sig'))))
    assert len(rows) == 1 and rows[0]['dataset_name'] == 'raw-grinding'
    zip_response = zone_client.get('/admin/api/training-data/export.zip', params=params)
    assert zip_response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(zip_response.content)) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        assert manifest['count'] == 1 and manifest['records'][0]['zone_id_status'] == 'absent'
        archived = archive.read('records/' + item['dataset_id'] + '.json')
    downloaded = zone_client.get('/admin/api/training-data/' + item['dataset_id'] + '/download')
    assert downloaded.status_code == 200 and downloaded.content == archived


@pytest.mark.parametrize('endpoint', ['', '/export.csv', '/export.zip'])
def test_learning_routes_reject_unknown_zone_mapping_and_process(zone_client, endpoint):
    url = '/admin/api/training-data' + endpoint
    assert zone_client.get(url, params={'zone_mapping': 'unknown'}).status_code == 422
    assert zone_client.get(url, params={'process': 'grinding-other'}).status_code == 422


def test_quality_filter_matches_list_counts_and_both_exports(zone_catalog):
    path = zone_catalog.root / 'catalog.json'
    index = json.loads(path.read_text(encoding='utf-8'))
    for row, quality in zip(index['records'], ['complete', 'needs_review', 'failed', 'unknown']):
        row['quality_status'] = quality
    path.write_text(json.dumps(index), encoding='utf-8')
    app = FastAPI()
    app.include_router(create_training_data_router(zone_catalog))
    with TestClient(app) as client:
        for quality, expected in [('complete', {'legacy-mapped'}),
                                  ('needs_review', {'legacy-unassigned', 'raw-grinding', 'converted-mapped'})]:
            params = {'quality_status': quality}
            listing = client.get('/admin/api/training-data', params=params).json()
            assert listing['filtered_count'] == len(expected)
            assert {row['dataset_name'] for row in listing['items']} == expected
            rows = list(csv.DictReader(io.StringIO(client.get('/admin/api/training-data/export.csv', params=params).content.decode('utf-8-sig'))))
            assert {row['dataset_name'] for row in rows} == expected
            with zipfile.ZipFile(io.BytesIO(client.get('/admin/api/training-data/export.zip', params=params).content)) as archive:
                assert json.loads(archive.read('manifest.json'))['count'] == len(expected)
        for endpoint in ['', '/export.csv', '/export.zip']:
            assert client.get('/admin/api/training-data'+endpoint, params={'quality_status':'wrong'}).status_code == 422
