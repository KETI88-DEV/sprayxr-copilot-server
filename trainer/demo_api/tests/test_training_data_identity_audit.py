"""Worker presentation groups and conservative read-only duplicate evidence."""
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.training_data_catalog import TrainingDataCatalog, create_training_data_router
from trainer.demo_api.training_data_ui import render_training_data
from trainer.demo_api.training_worker_identity import (
    UNKNOWN_WORKER_ID, is_unknown_worker, selection_worker_id,
    worker_group_id, worker_group_label, worker_sort_key,
)


@pytest.mark.parametrize('name', [None, '', '  ', '미기록', '미확인', '이름 미확인',
                                 'unknown', ' UNKNOWN ', '이름 미확인 · IDabc123',
                                 '이름 미확인 · a', '이름 미확인 · b',
                                 '이름 미확인·a', '이름 미확인 · 123', '이름 미확인 ·'])
def test_unknown_presentation_is_one_bucket_without_rewriting_source_ids(name):
    assert is_unknown_worker('original-worker', name)
    assert worker_group_id('original-worker', name) == UNKNOWN_WORKER_ID
    assert worker_group_label('original-worker', name) == '이름 미확인'
    assert selection_worker_id('original-worker', name) == 'original-worker'


def test_known_homonyms_and_existing_legacy_aliases_stay_distinct():
    assert worker_group_id('a', '동일이름') == 'a'
    assert worker_group_id('b', '동일이름') == 'b'
    assert not is_unknown_worker('a', '미확인연구원')
    assert not is_unknown_worker('a', '이름 미확인자')
    assert selection_worker_id('legacy', '전형기') == 'jeon_hyeonggi'
    assert selection_worker_id('legacy', '김경호') == 'kim_kyungho'
    assert [name for _, name in sorted([('z', '이름 미확인'), ('c', '홍길동'),
                                        ('a', '가나다'), ('b', '김연수')],
                                       key=lambda pair: worker_sort_key(*pair))] == [
        '가나다', '김연수', '홍길동', '이름 미확인']


@pytest.mark.parametrize('name', ['이름 미확인자', '미확인연구원', '이름 미확인자 · a',
                                 '이름 미확인 연구원 · a', '이름 미확인 - a', '이름 미확인: a'])
def test_similar_known_names_without_the_explicit_placeholder_delimiter_are_preserved(name):
    assert not is_unknown_worker('original-worker', name)
    assert worker_group_id('original-worker', name) == 'original-worker'
    assert worker_group_label('original-worker', name) == name


class FixtureCatalog:
    def __init__(self, root):
        self.root, self.source = root / 'built', root / 'sources'
        (self.root / 'records').mkdir(parents=True)
        self.source.mkdir()
        self.rows = []

    def add(self, *, worker='w', name='김연수', process='spray', mode='1',
            quality='complete', zone=True, day='2026-09-01', source_kind='raw',
            source=None, source_hash=None, source_relative=None, **extra):
        ident = f'{len(self.rows) + 1:032x}'
        metadata = dict(dataset_id=ident, dataset_name='same-name', worker_id=worker,
                        worker_name=name, role='trainee', process=process, scenario_id=mode,
                        mode_label='학습모드 ' + mode, acquired_at=day,
                        source_kind=source_kind, quality_status=quality, **extra)
        source_data = {}
        if source is not None:
            relative = f'{ident}/same-name.csv'
            path = self.source / relative
            path.parent.mkdir(parents=True)
            path.write_bytes(source)
            source_data = dict(relative_path=source_relative or relative, bytes=len(source),
                               sha256=source_hash or hashlib.sha256(source).hexdigest(), filename='same-name.csv')
        artifact = dict(metadata=metadata, provenance={'source_csv': source_data, 'heightmaps': []},
                        quality={'status': quality}, features={})
        payload = json.dumps(artifact, ensure_ascii=False).encode()
        (self.root / 'records' / (ident + '.json')).write_bytes(payload)
        self.rows.append({**metadata, 'json_bytes': len(payload),
                          'artifact_sha256': hashlib.sha256(payload).hexdigest(),
                          'source_sha256': source_data.get('sha256'),
                          'heightmap_count': 0, 'zone_id_available': zone, 'zone_count': int(zone),
                          'frame_count': 2, 'sequence_heatmap_count': 0})
        return ident

    def catalog(self):
        (self.root / 'catalog.json').write_text(json.dumps({'records': self.rows}), encoding='utf-8')
        return TrainingDataCatalog(self.root, self.source)


def test_worker_groups_sort_counts_names_and_preserve_raw_rows_and_exports(tmp_path):
    fixture = FixtureCatalog(tmp_path)
    fixture.add(worker='u1', name='이름 미확인 · IDabc')
    fixture.add(worker='u2', name='이름 미확인 · IDdef', process='blasting')
    fixture.add(worker='u2', name='이름 미확인 · IDdef', process='grinding')
    fixture.add(worker='z', name='홍길동')
    fixture.add(worker='a', name='가나다')
    fixture.add(worker='a', name='가나다')
    fixture.add(worker='b', name='가나다')
    catalog = fixture.catalog()
    result = catalog.list()
    assert result['summary']['by_worker'] == [
        {'id': UNKNOWN_WORKER_ID, 'label': '이름 미확인', 'count': 3},
        {'id': 'a', 'label': '가나다', 'count': 2},
        {'id': 'b', 'label': '가나다', 'count': 1},
        {'id': 'z', 'label': '홍길동', 'count': 1},
    ]
    assert result['options']['workers'] == [
        {'id': 'a', 'label': '가나다'}, {'id': 'b', 'label': '가나다'},
        {'id': 'z', 'label': '홍길동'}, {'id': UNKNOWN_WORKER_ID, 'label': '이름 미확인'},
    ]
    assert result['summary']['worker_count'] == 4
    selected = catalog.list(worker_id=UNKNOWN_WORKER_ID)
    assert selected['filtered_count'] == 3
    assert {row['worker_id'] for row in selected['items']} == {'u1', 'u2'}
    assert all(row['worker_name'].startswith('이름 미확인 · ID') for row in selected['items'])
    assert catalog.list(worker_id='u1')['filtered_count'] == 1, 'old raw-ID URLs still select only their ID'
    exported = list(csv.DictReader(io.StringIO(catalog.csv_export(worker_id=UNKNOWN_WORKER_ID).decode('utf-8-sig'))))
    assert len(exported) == 3
    with zipfile.ZipFile(io.BytesIO(catalog.zip_export(worker_id=UNKNOWN_WORKER_ID))) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        assert {row['worker_id'] for row in manifest['records']} == {'u1', 'u2'}


def test_group_selection_keeps_all_other_facets_search_and_direct_http_urls(tmp_path):
    fixture = FixtureCatalog(tmp_path)
    fixture.add(worker='u1', name='이름 미확인 · IDabc')
    fixture.add(worker='u2', name=None, quality='needs_review', mode='2', process='blasting')
    fixture.add(worker='known', name='가나다', process='blasting')
    catalog = fixture.catalog()
    result = catalog.list(worker_id=UNKNOWN_WORKER_ID, quality_status='complete', scenario_id='1', q='same-name')
    assert result['filtered_count'] == 1
    assert [item['id'] for item in result['options']['workers']] == ['known', UNKNOWN_WORKER_ID]
    assert [item['id'] for item in result['options']['processes']] == ['spray']
    assert [item['id'] for item in result['options']['modes']] == ['1']
    assert [item['id'] for item in result['options']['qualities']] == ['complete']
    assert catalog.list(worker_id=UNKNOWN_WORKER_ID, q='missing')['filtered_count'] == 0
    app = FastAPI(); app.include_router(create_training_data_router(catalog))
    with TestClient(app) as client:
        params = {'worker_id': UNKNOWN_WORKER_ID, 'quality_status': 'needs_review',
                  'process': 'blasting', 'scenario_id': '2'}
        assert client.get('/admin/api/training-data', params=params).json()['filtered_count'] == 1
        assert len(client.get('/admin/api/training-data/export.csv', params=params).text.splitlines()) == 2


def test_process_summary_and_choices_use_domain_order_and_stable_unknown_tail(tmp_path):
    fixture = FixtureCatalog(tmp_path)
    for process in ('unknown-z', 'grinding', 'blasting', 'unknown-a', 'spray'):
        fixture.add(process=process)
    result = fixture.catalog().list()
    expected = ['spray', 'blasting', 'grinding', 'unknown-z', 'unknown-a']
    assert [item['id'] for item in result['summary']['by_process']] == expected
    assert [item['id'] for item in result['options']['processes']] == expected
    assert '이름 미확인은 1개 그룹' in render_training_data()


def motion_csv(*, zone=None, x='1', time='1', other='original'):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    header = ['시간(초)', '분사', 'Ray point X', 'Ray Point Y', 'Ray Point Z', 'Other Original']
    cells = [time, 'On', x, '2', '3', other]
    if zone is not None:
        header += ['Zone ID', 'Cur_Sequence_heatmap']
        cells += [zone, '{"derived":true}']
    writer.writerow(header)
    writer.writerow(cells)
    return stream.getvalue().encode('utf-8-sig')


def snapshot(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob('*') if path.is_file()}


def test_verified_original_motion_ignores_only_zone_enrichment_prefers_zone_and_never_writes(tmp_path):
    fixture = FixtureCatalog(tmp_path)
    first = fixture.add(source=motion_csv(), zone=False)
    second = fixture.add(source=motion_csv(zone='7'), zone=True, source_kind='converted')
    catalog = fixture.catalog()
    before = snapshot(tmp_path)
    metadata_only = catalog.audit_duplicates()
    assert metadata_only['match_group_count'] == 0, 'different declared source hashes are not guessed equivalent'
    result = catalog.audit_duplicates(verify_sources=True)
    assert result['motion_verified_count'] == 2 and result['match_group_count'] == 1
    assert result['additional_matching_records'] == 1
    group = result['groups'][0]
    assert group['dataset_ids'] == [first, second]
    assert group['match_basis'] == 'raw_motion_sha256'
    assert group['source_bytes_verified'] is True
    assert group['preferred_for_review'] == second
    assert group['requires_operator_review'] is True
    assert snapshot(tmp_path) == before


def test_same_filename_different_motion_time_original_columns_or_process_is_preserved(tmp_path):
    fixture = FixtureCatalog(tmp_path)
    fixture.add(source=motion_csv())
    fixture.add(source=motion_csv(x='9'))
    fixture.add(source=motion_csv(time='2'))
    fixture.add(source=motion_csv(other='different'))
    fixture.add(source=motion_csv(), process='blasting')
    result = fixture.catalog().audit_duplicates(verify_sources=True)
    assert result['motion_verified_count'] == 5
    assert result['groups'] == []


def test_equal_original_hash_is_evidence_but_identity_and_date_conflicts_require_review(tmp_path):
    fixture = FixtureCatalog(tmp_path)
    fixture.add(source=motion_csv(), worker='a', day=None)
    fixture.add(source=motion_csv(), worker='b', day='2026-10-01')
    catalog = fixture.catalog()
    metadata = catalog.audit_duplicates()['groups'][0]
    assert metadata['source_bytes_verified'] is False
    assert 'source_bytes_not_verified' in metadata['review_reasons']
    result = catalog.audit_duplicates(verify_sources=True)
    assert result['groups'][0]['match_basis'] == 'source_sha256'
    assert result['groups'][0]['review_reasons'] == [
        'worker_identity_conflict', 'acquisition_time_unknown', 'acquisition_time_conflict']
    assert result['groups'][0]['requires_operator_review'] is True


def test_source_hash_mismatch_and_escaping_path_are_excluded_from_verified_audit(tmp_path):
    fixture = FixtureCatalog(tmp_path)
    fixture.add(source=motion_csv())
    mismatch = fixture.add(source=motion_csv(x='5'), source_hash=hashlib.sha256(motion_csv()).hexdigest())
    escape = fixture.add(source=motion_csv(), source_relative='../outside.csv')
    result = fixture.catalog().audit_duplicates(verify_sources=True)
    assert result['inspected_count'] == 1 and result['groups'] == []
    assert result['unavailable'] == [
        {'dataset_id': mismatch, 'reason_code': 'source_hash_mismatch'},
        {'dataset_id': escape, 'reason_code': 'artifact_or_source_unavailable'},
    ]


def test_changed_artifact_cannot_supply_unverified_duplicate_hash(tmp_path):
    fixture = FixtureCatalog(tmp_path)
    fixture.add(source=motion_csv())
    ident = fixture.add(source=motion_csv())
    catalog = fixture.catalog()
    path = catalog.root / 'records' / (ident + '.json')
    path.write_bytes(path.read_bytes().replace(b'"spray"', b'"splay"'))
    result = catalog.audit_duplicates()
    assert result['inspected_count'] == 1 and result['groups'] == []
    assert result['unavailable'] == [{'dataset_id': ident, 'reason_code': 'artifact_or_source_unavailable'}]


def test_index_only_audit_never_reads_artifacts_or_sources_and_labels_unverified_evidence(tmp_path, monkeypatch):
    fixture = FixtureCatalog(tmp_path)
    fixture.add(source=motion_csv())
    fixture.add(source=motion_csv())
    catalog = fixture.catalog()
    def forbidden(*args, **kwargs):
        raise AssertionError('metadata-only must not read artifact/source files')
    monkeypatch.setattr(catalog, '_detail_row', forbidden)
    monkeypatch.setattr(catalog, '_rows', forbidden)
    monkeypatch.setattr(catalog, '_safe', forbidden)
    result = catalog.audit_duplicates(metadata_only=True)
    assert result['inspected_count'] == 2 and result['match_group_count'] == 1
    assert result['source_bytes_verified'] is result['artifact_bytes_verified'] is False
    assert result['motion_verified_count'] == 0
    assert result['groups'][0]['review_reasons'] == ['artifact_bytes_not_verified', 'source_bytes_not_verified']
    with pytest.raises(ValueError, match='mutually exclusive'):
        catalog.audit_duplicates(metadata_only=True, verify_sources=True)


def test_matching_empty_recordings_are_explicitly_kept_as_review_evidence(tmp_path):
    fixture = FixtureCatalog(tmp_path)
    fixture.add(source=b'header-only\n', worker='a')
    fixture.add(source=b'header-only\n', worker='b')
    for row in fixture.rows:
        row['frame_count'] = 0
    result = fixture.catalog().audit_duplicates(metadata_only=True)
    assert 'empty_recording' in result['groups'][0]['review_reasons']
    assert 'worker_identity_conflict' in result['groups'][0]['review_reasons']
    assert result['groups'][0]['requires_operator_review'] is True
