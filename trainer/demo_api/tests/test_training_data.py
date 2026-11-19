import csv
import hashlib
import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from trainer.demo_api.training_data_build import build_training_datasets
from trainer.demo_api.training_data_catalog import TrainingDataCatalog, create_training_data_router


def record(root: Path, name='3_1234_1', *, worker=None, zone='1'):
    base = root / 'workers' / worker if worker else root
    if worker:
        base.mkdir(parents=True, exist_ok=True)
        (base / 'worker.json').write_text(json.dumps({'display_name': '새 작업자', 'role': 'trainee'}), encoding='utf-8')
    folder = base / 'raw/spray/learning/mode_03' / name
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f'{name}.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['시간(초)', '분사', 'Zone ID', 'Ray Texture Map Number', 'Distance (Input)', 'Speed (Input)', 'Ray point X', 'Ray Point Y', 'Ray Point Z'])
        writer.writerows([[0, 'Off', '-1', 1, 200, 400, 0, 0, 0], [1, 'On', zone, 1, 250, 500, 1, 2, 3], [2, 'On', zone, 1, 350, 700, 2, 3, 4]])
    Image.new('RGB', (4, 4), (1, 0, 255)).save(folder / 'Map01_1234_1_1_Height.png')
    return folder


def test_real_json_artifacts_and_rebuild_count_do_not_duplicate(tmp_path):
    root, output = tmp_path / 'source', tmp_path / 'built'
    record(root)
    first = build_training_datasets(root, output)
    assert first['record_count'] == 1
    catalog = TrainingDataCatalog(output, root)
    listing = catalog.list()
    item = listing['items'][0]
    data = catalog.detail(item['dataset_id'])
    assert data['features']['frame_count'] == 3
    assert data['features']['sequence_count'] == 1
    assert data['features']['zones'][0]['distance_mm']['median'] == 300
    assert data['metadata']['acquired_at'] is None
    assert data['metadata']['worker_name'] == '전형기'
    assert data['coating']['maps'][0]['image_url'].startswith('/admin/api/training-data/')
    assert json.loads(catalog.artifact(item['dataset_id']).read_text(encoding='utf-8')) == data
    build_training_datasets(root, output)
    assert TrainingDataCatalog(output, root).list()['total'] == 1
    assert len(list((output / 'records').glob('*.json'))) == 1


def test_workers_filters_and_quality_are_separate_from_expert_profiles(tmp_path):
    root, output = tmp_path / 'source', tmp_path / 'built'
    record(root)
    record(root, '3_5678_1', worker='trainee-2', zone='-1')
    build_training_datasets(root, output)
    catalog = TrainingDataCatalog(output, root)
    selected = catalog.list(worker_id='trainee-2')
    assert selected['total'] == 2 and selected['filtered_count'] == 1
    assert selected['summary']['worker_count'] == 1
    assert selected['items'][0]['quality_status'] == 'needs_review'
    assert {option['id'] for option in selected['options']['workers']} == {'jeon_hyeonggi', 'trainee-2'}
    assert catalog.list(q='새 작업자')['filtered_count'] == 1
    assert catalog.list(scenario_id='10')['filtered_count'] == 0
    import io
    exported = list(csv.DictReader(io.StringIO(catalog.csv_export(worker_id='trainee-2').decode('utf-8-sig'))))
    assert exported[0]['zone_count'] == '0'


def test_training_date_filters_summary_and_all_exports_together(tmp_path):
    import io
    import zipfile
    root, output = tmp_path / 'source', tmp_path / 'built'
    for name, worker, day in [('3_A_1', None, '2025-08-31'), ('3_B_1', 'other', '2025-09-01'),
                              ('3_C_1', 'other', '2025-08-31'), ('3_D_1', None, None)]:
        folder = record(root, name, worker=worker)
        if day:
            (folder / 'metadata.json').write_text(json.dumps({'acquired_at': day}), encoding='utf-8')
    build_training_datasets(root, output)
    catalog = TrainingDataCatalog(output, root)
    selected = catalog.list(acquired_at='2025-08-31')
    assert selected['total'] == 4 and selected['filtered_count'] == 2
    assert selected['summary']['dated_count'] == 2
    assert selected['options']['dates'] == [
        {'id': '2025-09-01', 'label': '2025-09-01'}, {'id': '2025-08-31', 'label': '2025-08-31'},
        {'id': 'unknown', 'label': '미기록'},
    ]
    assert catalog.list(acquired_at='unknown')['filtered_count'] == 1
    app = FastAPI()
    app.include_router(create_training_data_router(catalog))
    params = {'acquired_at': '2025-08-31', 'worker_id': 'other', 'process': 'spray', 'scenario_id': '3'}
    with TestClient(app) as client:
        listing = client.get('/admin/api/training-data', params=params).json()
        assert listing['filtered_count'] == 1 and listing['items'][0]['dataset_name'] == '3_C_1'
        csv_rows = list(csv.DictReader(io.StringIO(client.get('/admin/api/training-data/export.csv', params=params).content.decode('utf-8-sig'))))
        assert len(csv_rows) == 1 and csv_rows[0]['acquired_at'] == '2025-08-31'
        zipped = client.get('/admin/api/training-data/export.zip', params=params)
        with zipfile.ZipFile(io.BytesIO(zipped.content)) as archive:
            manifest = json.loads(archive.read('manifest.json'))
            assert manifest['count'] == 1 and manifest['records'][0]['dataset_name'] == '3_C_1'
        for endpoint in ['', '/export.csv', '/export.zip']:
            assert client.get('/admin/api/training-data' + endpoint, params={'acquired_at': '2025-02-30'}).status_code == 422


def test_download_export_and_images_are_read_only_and_allowlisted(tmp_path):
    root, output = tmp_path / 'source', tmp_path / 'built'
    record(root)
    build_training_datasets(root, output)
    app = FastAPI()
    app.include_router(create_training_data_router(TrainingDataCatalog(output, root)))
    with TestClient(app) as client:
        listing = client.get('/admin/api/training-data').json()
        dataset_id = listing['items'][0]['dataset_id']
        detail = client.get('/admin/api/training-data/' + dataset_id)
        assert detail.status_code == 200
        assert client.get(detail.json()['coating']['maps'][0]['image_url']).headers['content-type'] == 'image/png'
        assert client.get('/admin/api/training-data/' + dataset_id + '/download').status_code == 200
        assert client.get('/admin/api/training-data/export.csv').content.startswith(b'\xef\xbb\xbf')
        archive = client.get('/admin/api/training-data/export.zip')
        assert archive.status_code == 200 and archive.content.startswith(b'PK')
        assert client.get('/admin/api/training-data/not-a-dataset').status_code == 404
        assert client.get('/admin/api/training-data/' + dataset_id + '/images/999').status_code == 404
        assert client.post('/admin/api/training-data').status_code == 405


def test_missing_catalog_and_bad_record_do_not_claim_completed_data(tmp_path):
    assert TrainingDataCatalog(tmp_path / 'built', tmp_path / 'source').list()['available'] is False
    root, output = tmp_path / 'source', tmp_path / 'built'
    folder = record(root)
    (folder / '3_1234_1.csv').write_text('broken\nvalue\n', encoding='utf-8')
    report = build_training_datasets(root, output)
    assert report['failed_count'] == 1
    assert TrainingDataCatalog(output, root).list()['total'] == 0


def test_same_record_copy_is_deduplicated(tmp_path):
    import shutil
    root, output = tmp_path / 'source', tmp_path / 'built'
    folder = record(root)
    copy = folder.parent / 'duplicate'
    shutil.copytree(folder, copy)
    report = build_training_datasets(root, output)
    assert report['record_count'] == 1 and report['duplicate_count'] == 1


def test_independent_recordings_and_expert_bands_are_retained(tmp_path):
    root, output = tmp_path / 'source', tmp_path / 'built'
    record(root, '3_1234_1')
    record(root, '3_5678_1')
    build_training_datasets(root, output)
    catalog = TrainingDataCatalog(output, root)
    assert catalog.list()['total'] == 2
    detail = catalog.detail(catalog.list()['items'][0]['dataset_id'])
    assert detail['coating']['maps'][0]['reference']['record_count'] == 2
    assert detail['coating']['maps'][0]['reference']['median_um'] > 0


def test_legacy_zone_correction_does_not_remap_new_workers(tmp_path):
    root, output = tmp_path / 'source', tmp_path / 'built'
    record(root, '2_1234_1', zone='1')
    record(root, '2_5678_1', worker='new-worker', zone='1')
    build_training_datasets(root, output)
    catalog = TrainingDataCatalog(output, root)
    legacy = catalog.list(worker_id='jeon_hyeonggi')['items'][0]
    current = catalog.list(worker_id='new-worker')['items'][0]
    assert catalog.detail(legacy['dataset_id'])['features']['zones'][0]['zone_id'] == '3'
    assert catalog.detail(current['dataset_id'])['features']['zones'][0]['zone_id'] == '1'


def test_old_records_without_criterion_are_left_as_is(tmp_path):
    root, output = tmp_path / 'source', tmp_path / 'built'
    record(root)
    build_training_datasets(root, output)
    catalog = TrainingDataCatalog(output, root)
    item = catalog.list()['items'][0]
    face = catalog.detail(item['dataset_id'])['coating']['maps'][0]
    assert face['criterion_version'].startswith('coating-v1-adapter/')
    path = catalog.artifact(item['dataset_id'])
    data = json.loads(path.read_text(encoding='utf-8'))
    old = data['coating']['maps'][0]
    for key in ('criterion_version', 'thickness_basis', 'basis_label', 'band_labels', 'grade_labels', 'rule_note'):
        del old[key]
    old['summary'].update(within_target_percent=50.0, target_tolerance_percent=10.0)
    path.write_text(json.dumps(data), encoding='utf-8')
    # 옛 기록처럼 보이게 목록의 크기·해시도 맞춘다(저장된 기록을 다시 쓰는 일은 하지 않는다).
    index_path = output / 'catalog.json'
    index = json.loads(index_path.read_text(encoding='utf-8'))
    for row in index['records']:
        if row['dataset_id'] == item['dataset_id']:
            row['json_bytes'], row['artifact_sha256'] = path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest()
    index_path.write_text(json.dumps(index), encoding='utf-8')
    before = path.read_bytes()
    shown = TrainingDataCatalog(output, root).detail(item['dataset_id'])['coating']['maps'][0]
    assert 'criterion_version' not in shown and shown['summary']['within_target_percent'] == 50.0
    assert path.read_bytes() == before


def test_catalog_rejects_escaped_paths(tmp_path):
    root, output = tmp_path / 'source', tmp_path / 'built'
    record(root)
    build_training_datasets(root, output)
    catalog = TrainingDataCatalog(output, root)
    item = catalog.list()['items'][0]
    p = catalog.artifact(item['dataset_id'])
    data = json.loads(p.read_text(encoding='utf-8'))
    data['provenance']['heightmaps'][0]['relative_path'] = '../secret.png'
    p.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises((ValueError, FileNotFoundError)):
        TrainingDataCatalog(output, root).image(item['dataset_id'], 0)


def test_training_data_overview_is_public_without_operational_database(tmp_path, monkeypatch):
    from trainer.demo_api.db_viewer import create_db_viewer_router
    monkeypatch.setenv('SEQUENCE_DB_PATH', str(tmp_path / 'missing.db'))
    monkeypatch.setenv('TRAINING_DATA_ROOT', str(tmp_path / 'built'))
    app = FastAPI()
    app.include_router(create_db_viewer_router())
    with TestClient(app) as client:
        home = client.get('/admin')
        assert '학습 데이터' in home.text
        response = client.get('/admin/training-data')
        assert response.status_code == 200
        assert '학습 데이터' in response.text and '/admin/assets/training_data.js' in response.text
        assert 'id="td-verification-tab"' in response.text and 'id="tv-root"' in response.text
        assert '품질 완료 등록' in response.text and '전체 훈련 시간 연속 재생' in response.text
        assert 'id="tv-start" type="button" hidden' in response.text
        assert client.get('/admin/assets/training_data.js').status_code == 200
        for suffix in ['js', 'css']:
            asset = client.get('/admin/assets/training_verification.' + suffix)
            assert asset.status_code == 200 and asset.headers['cache-control'] == 'no-cache'
        assert client.get('/admin/api/training-data').json()['available'] is False


def test_acquisition_dates_survive_rebuild_and_exports_without_clock_time(tmp_path):
    from trainer.demo_api.training_data_dates import apply_acquisition_dates
    root, output = tmp_path / 'source', tmp_path / 'built'
    folder = record(root)
    (folder / 'metadata.json').write_text(json.dumps({'target_thickness_um': 150}), encoding='utf-8')
    build_training_datasets(root, output)
    original = TrainingDataCatalog(output, root).list()['items'][0]
    index = json.loads((output / 'catalog.json').read_text(encoding='utf-8'))
    artifact = TrainingDataCatalog(output, root).detail(original['dataset_id'])
    manifest = {'records': [{**{k: original[k] for k in ('dataset_id', 'dataset_name', 'process', 'worker_id')},
                            'expected_artifact_sha256': index['records'][0]['artifact_sha256'], 'acquired_at': '2025-09-01',
                            'date_source': {'kind': 'original_date_folder', 'precision': 'day', 'folder': '전형기_20250901', 'source_csv': '3_1234_1.csv'}}]}
    report = apply_acquisition_dates(root, output, manifest)
    assert report['updated_count'] == 1 and Path(report['backup_path']).is_dir()
    catalog = TrainingDataCatalog(output, root)
    updated = catalog.detail(original['dataset_id'])
    assert updated['metadata']['acquired_at'] == '2025-09-01'
    assert 'converted_at' not in updated['metadata']
    assert updated['features'] == artifact['features'] and updated['coating'] == artifact['coating']
    metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
    assert metadata['target_thickness_um'] == 150 and metadata['acquired_at'] == '2025-09-01'
    assert catalog.list()['summary']['acquisition_date_min'] == '2025-09-01'
    assert catalog.list()['summary']['dated_count'] == 1
    import io
    csv_rows = list(csv.DictReader(io.StringIO(catalog.csv_export().decode('utf-8-sig'))))
    assert csv_rows[0]['acquired_at'] == '2025-09-01' and 'converted_at' not in csv_rows[0]
    assert apply_acquisition_dates(root, output, manifest)['updated_count'] == 0
    build_training_datasets(root, output)
    rebuilt = TrainingDataCatalog(output, root).detail(original['dataset_id'])
    assert rebuilt['metadata']['acquired_at'] == '2025-09-01'
    assert 'converted_at' not in rebuilt['metadata']
    assert rebuilt['provenance']['acquisition_date_source']['precision'] == 'day'


def test_bad_date_mapping_leaves_existing_files_unchanged(tmp_path):
    from trainer.demo_api.training_data_dates import apply_acquisition_dates
    root, output = tmp_path / 'source', tmp_path / 'built'
    folder = record(root)
    build_training_datasets(root, output)
    before = (output / 'catalog.json').read_bytes()
    row = TrainingDataCatalog(output, root).list()['items'][0]
    for date_value, name in [('2025-02-30', row['dataset_name']), ('2025-09-01', 'wrong-record')]:
        manifest = {'records': [{**row, 'dataset_name': name, 'acquired_at': date_value,
                                'date_source': {'kind': 'original_date_folder', 'precision': 'day', 'folder': 'original', 'source_csv': 'record.csv'}}]}
        with pytest.raises(ValueError):
            apply_acquisition_dates(root, output, manifest)
        assert (output / 'catalog.json').read_bytes() == before
        assert not (folder / 'metadata.json').exists()


def test_date_normalization_keeps_calendar_date_without_guessing_missing_values():
    from trainer.demo_api.training_data_dates import acquisition_date
    assert acquisition_date('2026-08-21') == '2026-08-21'
    assert acquisition_date('2026-08-21T16:00:00Z') == '2026-08-22'
    assert acquisition_date('2026-08-21T10:20:30') == '2026-08-21'
    assert acquisition_date('2025-02-30') is None and acquisition_date(None) is None
