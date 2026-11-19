"""Grouping choices preserves original URLs and absent input invalidation."""
import hashlib
import json
import csv
import os
import shutil
from pathlib import Path

import pytest

from trainer.demo_api.tests.test_motion_viewer import _client, _write_csv, _write_height_map
from trainer.demo_api.tests.test_training_motion_catalog import publish


@pytest.mark.parametrize('worker, name, expected_worker', [
    ('new_worker', '새 작업자', 'new_worker'),
    ('legacy_foreman', '전형기', 'jeon_hyeonggi'),
    ('unknown_source', '', '__unknown_workers__'),
])
def test_dataset_selection_resolves_the_same_record_that_actually_plays(monkeypatch, tmp_path, worker, name, expected_worker):
    output = tmp_path / 'built'; monkeypatch.setenv('TRAINING_DATA_ROOT', str(output))
    row = publish(tmp_path, output, worker=worker, worker_name=name)
    client = _client(tmp_path)
    response = client.get('/motion-viewer/api/selection', params={'dataset_id': row['dataset_id']})
    assert response.status_code == 200, response.text
    selection = response.json()
    assert selection == {'dataset_id': row['dataset_id'], 'worker_id': expected_worker, 'process': 'spray',
                         'mode_id': 4, 'session_id': 'dataset_' + row['dataset_id'] if not name else row['session_id']}
    playback = client.get('/motion-viewer/api/motion', params={key: value for key, value in selection.items() if key != 'dataset_id'})
    assert playback.status_code == 200, playback.text
    assert playback.json()['dataset_id'] == row['dataset_id']
    assert playback.json()['selected_session_id'] == selection['session_id']


def test_dataset_selection_does_not_substitute_unplayable_or_missing_records(monkeypatch, tmp_path):
    output = tmp_path / 'built'; monkeypatch.setenv('TRAINING_DATA_ROOT', str(output))
    row = publish(tmp_path, output, quality='needs_review')
    client = _client(tmp_path)
    assert client.get('/motion-viewer/api/selection', params={'dataset_id': row['dataset_id']}).status_code == 404
    assert client.get('/motion-viewer/api/selection', params={'dataset_id': '0' * 32}).status_code == 404


def test_dataset_selection_rejects_an_artifact_shadowed_by_the_same_logical_session(monkeypatch, tmp_path):
    output = tmp_path / 'built'; monkeypatch.setenv('TRAINING_DATA_ROOT', str(output))
    row = publish(tmp_path, output)
    index_path = output / 'catalog.json'
    (output / 'records' / ('1' * 32 + '.json')).write_bytes(
        (output / 'records' / (row['dataset_id'] + '.json')).read_bytes())
    index = json.loads(index_path.read_text(encoding='utf-8'))
    index['records'].append({**row, 'dataset_id': '1' * 32, 'source_kind': 'legacy'})
    index_path.write_text(json.dumps(index), encoding='utf-8')
    client = _client(tmp_path)
    response = client.get('/motion-viewer/api/selection', params={'dataset_id': '1' * 32})
    assert response.status_code == 409
    assert '다른 등록 자료' in response.json()['detail']


@pytest.mark.parametrize('endpoint', ['motion', 'arm-pose', 'alerts'])
@pytest.mark.parametrize('source_kind', ['catalog', 'legacy'])
def test_original_unknown_worker_urls_select_only_that_workers_original_session(monkeypatch, tmp_path, endpoint, source_kind):
    output = tmp_path / 'built'; monkeypatch.setenv('TRAINING_DATA_ROOT', str(output))
    client = _client(tmp_path)
    for worker_id, offset in [('unknown_a', 3), ('unknown_b', 8)]:
        if source_kind == 'legacy':
            worker_root = tmp_path / 'workers' / worker_id
            profile = worker_root / 'generated/profiles/spray/learning/mode_04.json'
            profile.parent.mkdir(parents=True)
            profile.write_bytes((tmp_path / 'generated/profiles/spray/learning/mode_04.json').read_bytes())
            _write_csv(worker_root / 'raw/spray/learning/mode_04/4_1001_1.csv', offset=offset)
        else:
            row = publish(tmp_path, output, worker=worker_id, worker_name='', session='4_1001_1')
            source = tmp_path / 'imports' / worker_id / '4_1001_1.csv'
            _write_csv(source, offset=offset)
            artifact_path = output / 'records' / (row['dataset_id'] + '.json')
            artifact = json.loads(artifact_path.read_text(encoding='utf-8'))
            artifact['provenance']['source_csv']['relative_path'] = source.relative_to(tmp_path).as_posix()
            content = json.dumps(artifact).encode(); artifact_path.write_bytes(content)
            index = json.loads((output / 'catalog.json').read_text(encoding='utf-8'))
            for item in index['records']:
                if item['dataset_id'] == row['dataset_id']:
                    item.update(json_bytes=len(content), artifact_sha256=hashlib.sha256(content).hexdigest())
            (output / 'catalog.json').write_text(json.dumps(index), encoding='utf-8')
    monkeypatch.setattr('trainer.demo_api.viewer_arm_pose.build_arm_pose_evaluation',
                        lambda path, **kwargs: {'selected_source': str(path)})
    monkeypatch.setattr('trainer.demo_api.viewer_alerts.sequence_alerts_for',
                        lambda path, **kwargs: {'selected_source': str(path)})
    for worker_id, offset in [('unknown_a', 3), ('unknown_b', 8)]:
        response = client.get('/motion-viewer/api/' + endpoint, params={
            'worker_id': worker_id, 'process': 'spray', 'mode_id': 4, 'session_id': '4_1001_1'})
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload['selected_session_id'] == '4_1001_1'
        if endpoint == 'motion':
            assert payload['worker']['worker_id'] == worker_id
            assert payload['frames'][0]['joints']['Pelvis'][0] == offset
            assert payload['sampled_session_count'] == 1
        else:
            result = payload['arm_pose_evaluation' if endpoint == 'arm-pose' else 'sequence_alerts']
            assert worker_id in result['selected_source']
    unknown = client.get('/motion-viewer/api/catalog').json()['workers'][-1]
    assert unknown['worker_id'] == '__unknown_workers__'
    sessions = unknown['processes'][0]['modes'][0]['sessions']
    assert len(sessions) == 2
    if source_kind == 'legacy' and endpoint == 'motion':
        bucket_query = {'worker_id': unknown['worker_id'], 'process': 'spray', 'mode_id': 4, 'session_id': sessions[0]}
        bucket = client.get('/motion-viewer/api/motion', params=bucket_query).json()
        _write_csv(tmp_path / 'workers/unknown_a/raw/spray/learning/mode_04/4_1002_1.csv', offset=4)
        assert client.get('/motion-viewer/api/cache-status', params={'token': bucket['viewer_cache']['token']}).json() == {'valid': False}
        assert client.get('/motion-viewer/api/motion', params=bucket_query).json()['sampled_session_count'] == 3


def test_missing_legacy_reference_csv_creation_invalidates_payload(tmp_path):
    client = _client(tmp_path)
    missing = tmp_path / 'raw/spray/learning/mode_04/4_1002_1.csv'
    missing.unlink()
    query = {'process': 'spray', 'mode_id': 4, 'session_id': '4_1001_1'}
    first = client.get('/motion-viewer/api/motion', params=query).json()
    assert first['sampled_session_count'] == 1
    token = first['viewer_cache']['token']
    _write_csv(missing, offset=0.1)
    assert client.get('/motion-viewer/api/cache-status', params={'token': token}).json() == {'valid': False}
    assert client.get('/motion-viewer/api/motion', params=query).json()['sampled_session_count'] == 2


def test_declared_missing_heightmap_creation_invalidates_payload(tmp_path):
    name = 'Map01_1002_1_1_Height.png'
    client = _client(tmp_path, heightmaps=[name])
    query = {'process': 'spray', 'mode_id': 4, 'session_id': '4_1002_1'}
    first = client.get('/motion-viewer/api/motion', params=query).json()
    assert first['coating_evaluation']['target']['analysis_status'] == 'not_available'
    token = first['viewer_cache']['token']
    _write_height_map(tmp_path, name, 181)
    assert client.get('/motion-viewer/api/cache-status', params={'token': token}).json() == {'valid': False}
    assert client.get('/motion-viewer/api/motion', params=query).json()['coating_evaluation']['target']['analysis_status'] == 'available'


def test_missing_imported_record_source_creation_invalidates_and_refreshes_catalog_summary(monkeypatch, tmp_path):
    output = tmp_path / 'built'; monkeypatch.setenv('TRAINING_DATA_ROOT', str(output))
    client = _client(tmp_path)
    publish(tmp_path, output, session='4_2201_1')
    publish(tmp_path, output, session='4_2202_1')
    missing = tmp_path / 'imports/converted/4_2202_1_Height.csv'; missing.unlink()
    query = {'worker_id': 'new_worker', 'process': 'spray', 'mode_id': 4, 'session_id': '4_2201_1'}
    first = client.get('/motion-viewer/api/motion', params=query).json()
    assert first['sampled_session_count'] == 1
    token = first['viewer_cache']['token']
    _write_csv(missing, offset=5)
    assert client.get('/motion-viewer/api/cache-status', params={'token': token}).json() == {'valid': False}
    assert client.get('/motion-viewer/api/motion', params=query).json()['sampled_session_count'] == 2


def test_equal_size_mtime_csv_replacement_refreshes_computation_not_only_token(tmp_path):
    client = _client(tmp_path)
    query = {'process': 'spray', 'mode_id': 4, 'session_id': '4_1001_1'}
    first = client.get('/motion-viewer/api/motion', params=query).json()
    source = tmp_path / 'raw/spray/learning/mode_04/4_1001_1.csv'
    before = source.stat()
    with source.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle); fields = reader.fieldnames; rows = list(reader)
    rows[0]['Pelvis Position X'] = '9.0'; rows[0]['Ray point X'] = '9.0'
    replacement = source.with_suffix('.replacement')
    with replacement.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    assert replacement.stat().st_size == before.st_size
    os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns)); os.replace(replacement, source)
    assert client.get('/motion-viewer/api/cache-status', params={'token': first['viewer_cache']['token']}).json() == {'valid': False}
    second = client.get('/motion-viewer/api/motion', params=query).json()
    assert second['frames'][0]['joints']['Pelvis'][0] == 9
    assert second['paint_trace'][0]['point'][0] == 9
    assert second['frame_direction_context']['snapshot_id'] != first['frame_direction_context']['snapshot_id']


def test_official_labelmap_change_refreshes_coating_and_catalog_mask_caches(monkeypatch, tmp_path):
    from PIL import Image
    from trainer.demo_api import motion_viewer as viewer
    from trainer.zone_reference import ZoneReferenceCatalog
    original_assets = Path(viewer.__file__).with_name('assets') / 'zone_reference'
    local_api = tmp_path / 'module'; assets = local_api / 'assets/zone_reference'
    assets.mkdir(parents=True); (assets / 'color').mkdir()
    for source in original_assets.glob('*.csv'):
        shutil.copyfile(source, assets / source.name)
    labelmap = assets / 'color/Learn04_01.png'
    shutil.copyfile(original_assets / 'color/Learn04_01.png', labelmap)
    old_catalog = ZoneReferenceCatalog(assets)
    monkeypatch.setattr(ZoneReferenceCatalog, 'default', classmethod(lambda cls: old_catalog))
    monkeypatch.setattr(viewer, '__file__', str(local_api / 'motion_viewer.py'))
    client = _client(tmp_path, heightmaps=['Map01_1002_1_1_Height.png'])
    _write_height_map(tmp_path, 'Map01_1002_1_1_Height.png', 181)
    query = {'process': 'spray', 'mode_id': 4, 'session_id': '4_1002_1'}
    first = client.get('/motion-viewer/api/motion', params=query).json()
    assert first['coating_evaluation']['target']['analysis_status'] == 'available'
    Image.new('RGB', (2048, 2048), (0, 0, 0)).save(labelmap)
    assert client.get('/motion-viewer/api/cache-status', params={'token': first['viewer_cache']['token']}).json() == {'valid': False}
    second = client.get('/motion-viewer/api/motion', params=query).json()
    assert second['coating_evaluation']['target']['analysis_status'] == 'not_available'


def test_equal_size_mtime_artifact_replacement_is_revalidated(monkeypatch, tmp_path):
    from trainer.demo_api.training_motion_catalog import TrainingMotionCatalog
    output = tmp_path / 'built'; monkeypatch.setenv('TRAINING_DATA_ROOT', str(output))
    row = publish(tmp_path, output)
    catalog = TrainingMotionCatalog(tmp_path)
    assert catalog.profile('new_worker', 'spray', 4) is not None
    artifact = output / 'records' / (row['dataset_id'] + '.json')
    before = artifact.stat(); replacement = artifact.with_suffix('.replacement')
    content = artifact.read_bytes().replace(b'300', b'900'); assert len(content) == before.st_size
    replacement.write_bytes(content); os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
    os.replace(replacement, artifact)
    assert catalog.profile('new_worker', 'spray', 4) is None


def test_equal_size_mtime_csv_replacement_refreshes_spray_windows_and_arm_input(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from trainer.demo_api import viewer_arm_pose
    from trainer.demo_api.viewer_alerts import spray_windows
    from trainer.demo_api.tests.test_viewer_alerts import _write_strokes
    source = tmp_path / 'record.csv'; _write_strokes(source, [('1', 'h')])
    def analyze(*, store, session_id, **kwargs):
        zone = store.session_bodies(session_id)[0]['zone_id']
        return SimpleNamespace(analysis_status='unavailable', guideline='source Zone ' + zone, counts={})
    monkeypatch.setattr(viewer_arm_pose, 'analyze_session', analyze)
    get_arm = lambda: viewer_arm_pose.build_arm_pose_evaluation(source, record_id='R', process='spray', mode_id=4, mcp_base_url='unused')
    assert spray_windows(source, mode_id=4, legacy=False)[0][-1] == '1'
    assert get_arm()['guideline'] == 'source Zone 1'
    before = source.stat()
    _write_strokes(source.with_suffix('.replacement'), [('2', 'h')])
    replacement = source.with_suffix('.replacement')
    assert replacement.stat().st_size == before.st_size
    os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns)); os.replace(replacement, source)
    assert spray_windows(source, mode_id=4, legacy=False)[0][-1] == '2'
    assert get_arm()['guideline'] == 'source Zone 2'


def test_sequence_alert_cache_refreshes_when_reference_profile_changes(tmp_path):
    from trainer.demo_api.tests.test_viewer_alerts import _client as alerts_client, _get
    client = alerts_client(tmp_path)
    first = _get(client, '/motion-viewer/api/alerts')['sequence_alerts']
    path = tmp_path / 'generated/profiles/spray/learning/mode_04.json'
    profile = json.loads(path.read_text(encoding='utf-8'))
    for zone in profile['aggregate']['zones'].values():
        zone['expected_direction'] = 'vertical'
    path.write_text(json.dumps(profile), encoding='utf-8')
    second = _get(client, '/motion-viewer/api/alerts')['sequence_alerts']
    assert second != first
