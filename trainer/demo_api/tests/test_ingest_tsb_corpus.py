import hashlib
import importlib
import json
from pathlib import Path

import pytest


def driver():
    return importlib.import_module('trainer.scripts.ingest_tsb_corpus')


def row(ident, name='3_100_1', process='spray', **extra):
    return dict(dataset_id=ident * 32, dataset_name=name, process=process,
                worker_name='전형기', worker_id='expert', role='expert',
                zone_count=1, frame_count=100, json_bytes=2,
                artifact_sha256=hashlib.sha256(b'{}').hexdigest(), **extra)


def test_legacy_and_converted_height_names_share_one_logical_key():
    m = driver()
    assert m.record_key(row('a')) == m.record_key(row('b', name='3_100_1_Height'))
    assert m.record_key(row('a')) != m.record_key(row('b', process='blasting'))


def test_zone_enriched_record_wins_over_larger_unmapped_candidate():
    m = driver()
    old = row('a')
    raw = row('b', zone_id_available=False, source_kind='raw', quality_rank=[0, 0, 2, 10000])
    rich = row('c', name='3_100_1_Height', zone_id_available=True, source_kind='converted', quality_rank=[1, 1, 3, 20])
    winners, stats = m.select_records([old], [raw, rich])
    assert [r['dataset_id'] for r in winners] == [rich['dataset_id']]
    assert stats['superseded_count'] == 2


def test_explicit_absent_overrides_legacy_nonzero_zone_count():
    m = driver()
    old = row('a', zone_id_available=False)
    valid = row('b', zone_id_available=True, source_kind='raw')
    assert m.select_records([old], [valid])[0][0]['dataset_id'] == valid['dataset_id']


def test_equal_hash_and_identity_preserves_existing_record():
    m = driver()
    old = row('a', source_sha256='same', enrichment_sha256='equivalent')
    new = row('b', source_kind='converted', source_sha256='same', enrichment_sha256='equivalent')
    winners, stats = m.select_records([old], [new])
    assert winners[0]['dataset_id'] == old['dataset_id']
    assert stats['exact_duplicate_count'] == 1


def test_same_csv_with_new_heightmaps_is_enrichment_not_exact_duplicate():
    m = driver()
    old = row('a', source_sha256='same', source_kind='raw', heightmap_count=0)
    new = row('b', source_sha256='same', source_kind='raw', heightmap_count=9)
    winners, stats = m.select_records([old], [new])
    assert winners[0]['dataset_id'] == new['dataset_id']
    assert stats['exact_duplicate_count'] == 0


def test_receipt_and_all_file_sizes_required_before_reading_job(tmp_path):
    m = driver()
    source = tmp_path / 'imports/run/archive/raw/data.csv'
    image = source.parent / 'map.png'
    source.parent.mkdir(parents=True)
    source.write_bytes(b'csv')
    image.write_bytes(b'png')
    job = {'source': {'relative_path': 'imports/run/archive/raw/data.csv', 'bytes': 3},
           'heightmaps': [{'relative_path': 'imports/run/archive/raw/map.png', 'bytes': 3}],
           'required_receipts': ['imports/run/transfer/batch.receipt.json']}
    assert not m.job_ready(job, tmp_path)
    receipt = tmp_path / 'imports/run/transfer/batch.receipt.json'
    receipt.parent.mkdir()
    manifest = [['data.csv', 3], ['map.png', 3]]
    manifest_hash = hashlib.sha256(json.dumps(manifest, ensure_ascii=True, separators=(',', ':')).encode()).hexdigest()
    payload = {'batch_id': 'batch', 'kind': 'raw', 'sha256': 'a'*64,
               'manifest': manifest, 'manifest_sha256': manifest_hash}
    receipt.write_text(json.dumps(payload))
    assert m.job_ready(job, tmp_path)
    payload['batch_id'] = 'unrelated-batch'
    receipt.write_text(json.dumps(payload))
    assert not m.job_ready(job, tmp_path)
    payload['batch_id'] = 'batch'
    payload['manifest'] = [['other.csv', 3]]
    receipt.write_text(json.dumps(payload))
    assert not m.job_ready(job, tmp_path)
    payload['manifest'] = manifest
    receipt.write_text(json.dumps(payload))
    image.write_bytes(b'incomplete')
    assert not m.job_ready(job, tmp_path)


@pytest.mark.parametrize('relative', ['../outside', '/absolute', 'C:\\absolute', 'archive/../outside'])
def test_manifest_path_traversal_rejected(tmp_path, relative):
    with pytest.raises(ValueError):
        driver().safe_file(tmp_path, relative)


def test_publish_checks_artifacts_and_detects_concurrent_catalog_changes(tmp_path):
    m = driver()
    root = tmp_path / 'output'
    root.mkdir()
    (root / 'records').mkdir()
    item = row('a')
    (root / 'records' / (item['dataset_id'] + '.json')).write_bytes(b'{}')
    report = {'records': [item], 'record_count': 1}
    current_hash = m.publish_catalog(root, report, expected_hash=None)
    assert len(current_hash) == 64
    assert json.loads((root / 'catalog.json').read_text())['record_count'] == 1
    (root / 'catalog.json').write_text('{"other":true}')
    with pytest.raises(RuntimeError, match='concurrent'):
        m.publish_catalog(root, report, expected_hash=current_hash)
    assert json.loads((root / 'catalog.json').read_text()) == {'other': True}


def test_publish_missing_artifact_keeps_previous_catalog(tmp_path):
    m = driver()
    root = tmp_path / 'out'
    root.mkdir()
    (root / 'catalog.json').write_text('{"records":[]}')
    previous = hashlib.sha256((root / 'catalog.json').read_bytes()).hexdigest()
    with pytest.raises((ValueError, FileNotFoundError)):
        m.publish_catalog(root, {'records': [row('a')]}, expected_hash=previous)
    assert hashlib.sha256((root / 'catalog.json').read_bytes()).hexdigest() == previous


def test_publication_rechecks_catalog_after_validation(tmp_path, monkeypatch):
    m = driver()
    dest = tmp_path / 'catalog.json'
    dest.write_text('{"records":[]}')
    before = m.file_hash(dest)
    def competing_writer(*args, **kwargs):
        dest.write_text('{"records":[],"external_change":true}')
    monkeypatch.setattr(m, 'validate_row', competing_writer)
    with pytest.raises(RuntimeError, match='concurrent'):
        m.publish_catalog(tmp_path, {'records': [row('a')]}, expected_hash=before)
    assert json.loads(dest.read_text())['external_change'] is True


def test_resume_recovers_only_own_committed_catalog_after_receipt_crash(tmp_path):
    m = driver()
    item = row('a')
    report = {'import_run_id': 'owned-run', 'records': [item]}
    (tmp_path / 'catalog.json').write_text(json.dumps(report))
    actual = m.file_hash(tmp_path / 'catalog.json')
    assert m.recover_publication(tmp_path, {'catalog_sha256': 'stale'}, [item], 'owned-run') == actual
    with pytest.raises(RuntimeError, match='concurrent'):
        m.recover_publication(tmp_path, {'catalog_sha256': 'stale'}, [item], 'another-run')
    with pytest.raises(RuntimeError, match='concurrent'):
        m.recover_publication(tmp_path, {'catalog_sha256': 'stale'}, [], 'owned-run')
