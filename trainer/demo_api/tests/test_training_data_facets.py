"""Faceted choices are computed from the same real, readable catalog as results."""
import hashlib
import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.training_data_catalog import TrainingDataCatalog, create_training_data_router


@pytest.fixture
def catalog(tmp_path: Path) -> TrainingDataCatalog:
    output = tmp_path / 'built'
    (output / 'records').mkdir(parents=True)
    rows = []
    matrix = [
        ('a', '전형기', 'spray', '1', 'complete', '1_101_1'),
        ('a', '전형기', 'spray', '2', 'needs_review', '2_102_1'),
        ('a', '전형기', 'blasting', '1', 'complete', '1_103_1'),
        ('a', '전형기', 'spray', '1', 'needs_review', '1_104_1'),
        ('b', '판득황', 'blasting', '1', 'needs_review', '1_201_1'),
        ('b', '판득황', 'blasting', '2', 'needs_review', '2_202_1'),
        ('c', '동일이름', 'spray', '2', 'complete', '2_301_1'),
        ('c', '동일이름', 'spray', '3', 'needs_review', '3_302_1'),
        ('d', '동일이름', 'grinding', '3', 'complete', '3_401_1'),
        ('d', '동일이름', 'blasting', '2', 'complete', '2_402_1'),
        ('b', '판득황', 'blasting', '4', 'needs_review', '4_203_1'),
    ]
    for index, (worker, name, process, mode, quality, session) in enumerate(matrix, 1):
        ident = f'{index:032x}'
        metadata = dict(dataset_id=ident, dataset_name=session, worker_id=worker, worker_name=name,
                        role='worker', process=process, scenario_id=mode, mode_label='학습모드 ' + mode,
                        acquired_at='2026-09-04' if worker != 'b' else None)
        artifact = {'schema_version': 'record-features-1.2', 'metadata': metadata,
                    'quality': {'status': quality, 'issues': []}, 'features': {},
                    'coating': {'maps': []}, 'provenance': {}}
        payload = json.dumps(artifact, ensure_ascii=False).encode('utf-8')
        (output / 'records' / (ident + '.json')).write_bytes(payload)
        rows.append({**metadata, 'quality_status': quality, 'json_bytes': len(payload),
                     'artifact_sha256': hashlib.sha256(payload).hexdigest(), 'heightmap_count': 0,
                     'zone_id_available': quality == 'complete', 'zone_count': int(quality == 'complete'),
                     'frame_count': 10, 'source_kind': 'converted', 'sequence_heatmap_count': 0})
    (output / 'catalog.json').write_text(json.dumps({'schema_version': 'record-features-1.2',
        'record_count': len(rows), 'import_status': 'complete', 'records': rows}, ensure_ascii=False), encoding='utf-8')
    return TrainingDataCatalog(output, tmp_path / 'source')


def ids(result: dict, facet: str) -> set[str]:
    return {option['id'] for option in result['options'][facet]}


def test_quality_filters_workers_but_does_not_lock_its_own_choices(catalog):
    result = catalog.list(quality_status='complete')
    assert result['filtered_count'] == 5
    assert ids(result, 'workers') == {'a', 'c', 'd'}
    assert ids(result, 'qualities') == {'complete', 'needs_review'}
    assert '판득황' not in {option['label'] for option in result['options']['workers']}


def test_each_facet_ignores_only_itself_and_keeps_other_conditions(catalog):
    result = catalog.list(worker_id='a', process='spray', scenario_id='1', quality_status='complete')
    assert [row['dataset_name'] for row in result['items']] == ['1_101_1']
    assert ids(result, 'workers') == {'a'}
    assert ids(result, 'processes') == {'spray', 'blasting'}
    assert ids(result, 'modes') == {'1'}
    assert ids(result, 'qualities') == {'complete', 'needs_review'}


def test_selected_worker_is_not_the_only_worker_option(catalog):
    result = catalog.list(worker_id='a', quality_status='complete')
    assert ids(result, 'workers') == {'a', 'c', 'd'}
    assert ids(result, 'processes') == {'spray', 'blasting'}
    assert ids(result, 'modes') == {'1'}


def test_same_name_different_worker_ids_are_not_collapsed(catalog):
    result = catalog.list(q='동일이름', quality_status='complete')
    assert result['filtered_count'] == 3
    assert result['options']['workers'] == [
        {'id': 'c', 'label': '동일이름'}, {'id': 'd', 'label': '동일이름'},
    ]
    assert ids(result, 'processes') == {'spray', 'blasting', 'grinding'}
    assert ids(result, 'modes') == {'2', '3'}


def test_search_is_retained_for_all_four_facet_calculations(catalog):
    result = catalog.list(q='2_302', quality_status='complete')
    assert result['filtered_count'] == 0
    assert all(not result['options'][facet] for facet in ('workers', 'processes', 'modes', 'qualities'))
    result = catalog.list(q='3_302', quality_status='complete')
    assert result['filtered_count'] == 0
    assert ids(result, 'qualities') == {'needs_review'}
    assert all(not result['options'][facet] for facet in ('workers', 'processes', 'modes'))


def test_empty_filter_intersection_stays_empty_without_silent_reset(catalog):
    result = catalog.list(worker_id='b', process='blasting', quality_status='complete')
    assert result['filtered_count'] == 0 and result['items'] == []
    assert ids(result, 'workers') == {'a', 'd'}
    assert ids(result, 'qualities') == {'needs_review'}
    assert ids(result, 'processes') == ids(result, 'modes') == set()


def test_no_records_means_no_facet_options(catalog):
    result = catalog.list(q='does-not-exist')
    assert all(not result['options'][facet] for facet in ('workers', 'processes', 'modes', 'qualities'))


def test_missing_artifacts_cannot_contribute_facet_values(catalog):
    (catalog.root / 'records' / (f'{9:032x}' + '.json')).unlink()
    result = catalog.list(quality_status='complete')
    assert 'grinding' not in ids(result, 'processes')
    assert '3' not in ids(result, 'modes')


def test_legacy_date_and_zone_filters_also_constrain_facets(catalog):
    dated = catalog.list(acquired_at='unknown', quality_status='complete')
    assert dated['filtered_count'] == 0
    assert ids(dated, 'workers') == set()
    assert ids(dated, 'qualities') == {'needs_review'}
    zones = catalog.list(zone_mapping='absent', worker_id='a')
    assert ids(zones, 'workers') == {'a', 'b', 'c'}
    assert ids(zones, 'modes') == {'1', '2'}
    assert ids(zones, 'qualities') == {'needs_review'}


def test_direct_query_url_has_same_options_and_exports_keep_full_selection(catalog):
    app = FastAPI(); app.include_router(create_training_data_router(catalog))
    with TestClient(app) as client:
        result = client.get('/admin/api/training-data?worker_id=a&process=spray&scenario_id=1&quality_status=complete').json()
        assert result['filtered_count'] == 1
        assert ids(result, 'processes') == {'spray', 'blasting'}
        assert ids(result, 'qualities') == {'complete', 'needs_review'}
        csv = client.get('/admin/api/training-data/export.csv?worker_id=a&process=spray&scenario_id=1&quality_status=complete')
        assert csv.status_code == 200 and len(csv.content.decode('utf-8-sig').splitlines()) == 2
        assert '1_101_1' in csv.text and '1_104_1' not in csv.text
        assert client.get('/admin/api/training-data?quality_status=invalid').status_code == 422


def test_four_facets_reuse_one_filesystem_catalog_scan(catalog, monkeypatch):
    original = catalog._rows
    scans = []

    def scanned(index):
        scans.append(index)
        return original(index)

    monkeypatch.setattr(catalog, '_rows', scanned)
    result = catalog.list(worker_id='a', quality_status='complete')
    assert len(scans) == 1
    assert ids(result, 'workers') == {'a', 'c', 'd'}
    assert ids(result, 'qualities') == {'complete', 'needs_review'}


def test_review_facet_never_advertises_an_unsupported_legacy_quality_value(catalog):
    path = catalog.root / 'catalog.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    data['records'][1]['quality_status'] = 'failed'
    path.write_text(json.dumps(data), encoding='utf-8')
    result = catalog.list(worker_id='a', scenario_id='2', quality_status='needs_review')
    assert result['filtered_count'] == 1
    assert ids(result, 'qualities') == {'needs_review'}


@pytest.mark.parametrize(('filters', 'expected'), [
    ({'worker_id': 'd'}, {'workers': {'a', 'b', 'c', 'd'}, 'processes': {'blasting', 'grinding'},
                         'modes': {'2', '3'}, 'qualities': {'complete'}}),
    ({'process': 'grinding'}, {'workers': {'d'}, 'processes': {'spray', 'blasting', 'grinding'},
                              'modes': {'3'}, 'qualities': {'complete'}}),
    ({'scenario_id': '4'}, {'workers': {'b'}, 'processes': {'blasting'}, 'modes': {'1', '2', '3', '4'},
                           'qualities': {'needs_review'}}),
    ({'quality_status': 'complete'}, {'workers': {'a', 'c', 'd'}, 'processes': {'spray', 'blasting', 'grinding'},
                                     'modes': {'1', '2', '3'}, 'qualities': {'complete', 'needs_review'}}),
    ({'q': '4_203'}, {'workers': {'b'}, 'processes': {'blasting'}, 'modes': {'4'}, 'qualities': {'needs_review'}}),
])
def test_every_filter_and_search_constrains_all_other_facets(catalog, filters, expected):
    result = catalog.list(**filters)
    assert {facet: ids(result, facet) for facet in expected} == expected
