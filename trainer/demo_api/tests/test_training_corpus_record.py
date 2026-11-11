"""One imported recording remains verifiable even without Zone annotations."""

import base64
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import importlib
import json
import struct
import zlib

import pytest


def build_record(*args, **kwargs):
    module = importlib.import_module('trainer.demo_api.training_corpus_record')
    return module.build_import_record(*args, **kwargs)


def png_bytes():
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    return (b'\x89PNG\r\n\x1a\n'
            + chunk(b'IHDR', struct.pack('>IIBBBBB', 2, 2, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(b'\0\x01\x00\xff\x02\x00\xff' * 2))
            + chunk(b'IEND', b''))


def write_job(tmp_path, columns=None, rows=None, *, process='spray', source_kind='raw', candidate_id='candidate-a'):
    source_root = tmp_path / 'source'
    path = source_root / 'imports/run/archive' / source_kind / process / 'Sequences/3_1000_1.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(columns or ['time', '분사', 'Distance (Input)', 'Speed (Input)'])
        writer.writerows(rows if rows is not None else [[0, 'Off', 999, 999], [1, 'On', 200, 400], [2, 'On', 400, 800]])
    job = {'candidate_id': candidate_id, 'logical_record_key': 'worker-a:spray:session-1',
           'session_id': 'session-1', 'process': process, 'scenario_id': '3', 'content_type': 'learning',
           'worker_id': 'worker-a', 'worker_name': '작업자 A', 'role': 'trainee', 'acquired_at': '2026-09-29',
           'source_kind': source_kind, 'source_inventory_id': 'inventory-1',
           'source': {'relative_path': path.relative_to(source_root).as_posix(),
                      'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns},
           'source_sql': {'db': 'database.db', 'matchstatus': 'matched', 'workid': 1000, 'orderid': 1,
                          'performanceids': [15], 'qualification': 'exact'},
           'issues': [], 'heightmaps': [], 'metadata': {'target_thickness_um': 100, 'operator_note': '원본 메모'},
           'provenance': {'archive_id': 'archive-1'}}
    return source_root, tmp_path / 'output', path, job


def read_artifact(output_root, row):
    path = output_root / 'records' / f"{row['dataset_id']}.json"
    raw = path.read_bytes()
    assert row['json_bytes'] == len(raw)
    assert row['artifact_sha256'] == hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def attach_heightmap(source_root, job, *, name='15_S9_Map01_Height.png'):
    path = source_root / 'imports/run/archive/raw/spray/Final' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png_bytes())
    job['heightmaps'] = [{'relative_path': path.relative_to(source_root).as_posix(), 'filename': name,
                         'map_num': 1, 'performance_id': 15, 'snapshot_sequence_index': 9,
                         'semantics': 'last_saved_snapshot_per_map'}]
    return path


def test_raw_without_zone_retains_metrics_original_hash_and_provenance(tmp_path):
    root, output, source, job = write_job(tmp_path)
    job.update({'qualification': 'unknown', 'identity_evidence': {'kind': 'sql_join', 'expertise': 'unknown'},
                'sequence_sidecar_count': 7, 'sequence_raw_sidecar_count': 2, 'role': None})
    original = source.read_bytes()
    row = build_record(job, root, output, catalog=None)
    data = read_artifact(output, row)
    assert data['schema_version'] == 'record-features-1.2'
    assert data['dataset_type'] == 'record_features'
    assert data['features']['frame_count'] == 3
    assert data['features']['global_metrics']['distance_mm']['median'] == 300
    assert data['features']['global_metrics']['speed_mms']['median'] == 600
    assert row['zone_id_available'] is False and row['zone_id_status'] == 'absent'
    assert row['source_kind'] == 'raw' and row['sequence_heatmap_count'] == 0
    assert data['metadata']['operator_note'] == '원본 메모'
    assert data['provenance']['archive_id'] == 'archive-1'
    assert data['provenance']['source_sql'] == job['source_sql']
    assert data['provenance']['source_inventory_id'] == 'inventory-1'
    assert data['provenance']['qualification'] == 'unknown'
    assert data['provenance']['identity_evidence'] == {'kind': 'sql_join', 'expertise': 'unknown'}
    assert data['metadata']['qualification'] == 'unknown' and data['metadata']['role'] == 'unknown'
    assert data['metadata']['sequence_sidecar_count'] == 7 and data['metadata']['sequence_raw_sidecar_count'] == 2
    assert data['provenance']['sequence_sidecar_count'] == 7 and data['provenance']['sequence_raw_sidecar_count'] == 2
    assert data['provenance']['source_csv']['relative_path'] == job['source']['relative_path']
    assert data['provenance']['source_csv']['sha256'] == hashlib.sha256(original).hexdigest()
    assert data['provenance']['source_csv']['bytes'] == len(original)
    assert row['source_sha256'] == hashlib.sha256(original).hexdigest()
    assert row['logical_record_key'] == job['logical_record_key'] and row['candidate_id'] == 'candidate-a'
    assert source.read_bytes() == original
    assert row['quality_status'] == 'needs_review'
    assert not (output / 'catalog.json').exists()


def test_converted_sparse_payload_and_official_zone_survive_without_legacy_remapping(tmp_path):
    image = {'map_num': '01', 'x0': 125, 'y0': 87, 'width': 1472, 'height': 924,
             'img_width': 2, 'img_height': 2, 'format': 'v2;thickness=RG16;alpha=B8;scale_um=20000',
             'encoding': 'png;base64', 'data': base64.b64encode(png_bytes()).decode('ascii')}
    root, output, source, job = write_job(tmp_path,
        ['time', '분사', 'Zone ID', 'Ray Texture Map Number', 'cur_sequence_heatmap'],
        [[0, 'On', '1', '01', json.dumps([image])]], source_kind='converted')
    job['scenario_id'] = '2'
    original = source.read_bytes()
    row = build_record(job, root, output, catalog=None)
    data = read_artifact(output, row)
    assert data['features']['zones'][0]['zone_id'] == '1'
    assert data['features']['zones'][0]['map_num'] == 1
    assert data['features']['sequence_heatmaps'][0]['maps'] == [image]
    assert row['source_kind'] == 'converted' and row['sequence_heatmap_count'] == 1
    assert source.read_bytes() == original


def test_positive_zone_has_priority_over_more_raw_frames(tmp_path):
    root, output, _, raw_job = write_job(tmp_path)
    raw_row = build_record(raw_job, root, output)
    _, _, _, zone_job = write_job(tmp_path, ['time', '분사', 'Zone ID'], [[0, 'On', 2]],
                                 source_kind='converted', candidate_id='candidate-b')
    zone_row = build_record(zone_job, root, output)
    assert zone_row['zone_id_available'] is True
    assert zone_row['quality_rank'] > raw_row['quality_rank']


@pytest.mark.parametrize('content', [b'', b'time,spray\n'])
def test_empty_csv_creates_review_artifact_with_known_zero_frames(tmp_path, content):
    root, output, source, job = write_job(tmp_path)
    source.write_bytes(content)
    job['source']['bytes'] = len(content)
    row = build_record(job, root, output)
    data = read_artifact(output, row)
    assert row['quality_status'] == 'needs_review'
    assert data['features']['analysis_status'] == 'unavailable'
    assert data['features']['frame_count'] == 0
    assert data['quality']['issues']
    assert data['provenance']['source_csv']['sha256'] == hashlib.sha256(content).hexdigest()
    assert data['features'].get('global_metrics') in (None, {})


def test_unreadable_csv_keeps_provenance_without_claiming_zero_frames(tmp_path):
    root, output, source, job = write_job(tmp_path)
    source.write_bytes(b'\xff\xfetime,spray\n\xff\xfe')
    job['source']['bytes'] = source.stat().st_size
    row = build_record(job, root, output)
    data = read_artifact(output, row)
    assert data['features']['analysis_status'] == 'unavailable'
    assert data['features']['frame_count'] is None
    assert data['features'].get('global_metrics') in (None, {})
    assert data['quality']['issues'] and row['quality_status'] == 'needs_review'


@pytest.mark.parametrize('relative', ['../outside.csv', r'..\outside.csv', '/outside.csv', r'C:\outside.csv'])
def test_source_path_traversal_is_rejected_without_artifact(tmp_path, relative):
    root, output, _, job = write_job(tmp_path)
    job['source']['relative_path'] = relative
    with pytest.raises(ValueError):
        build_record(job, root, output)
    assert not list(output.glob('records/*.json'))


def test_source_directory_and_declared_size_mismatch_are_rejected(tmp_path):
    root, output, source, job = write_job(tmp_path)
    job['source']['bytes'] += 1
    with pytest.raises(ValueError):
        build_record(job, root, output)
    job['source']['relative_path'] = source.parent.relative_to(root).as_posix()
    job['source'].pop('bytes')
    with pytest.raises((ValueError, FileNotFoundError)):
        build_record(job, root, output)


def test_spray_preview_preserves_original_image_map_and_snapshot_labels(tmp_path):
    root, output, _, job = write_job(tmp_path)
    original_image = attach_heightmap(root, job)
    row = build_record(job, root, output)
    data = read_artifact(output, row)
    image = data['provenance']['heightmaps'][0]
    coating = data['coating']['maps'][0]
    assert image['map_num'] == 1 and image['performance_id'] == 15 and image['snapshot_sequence_index'] == 9
    assert image['sha256'] == hashlib.sha256(png_bytes()).hexdigest() and image['bytes'] == len(png_bytes())
    assert coating['map_num'] == 1 and coating['performance_id'] == 15 and coating['snapshot_sequence_index'] == 9
    assert coating['image_url'] == f"/admin/api/training-data/{row['dataset_id']}/images/0"
    assert coating['thickness_image_url'] == coating['image_url'] + '?view=thickness'
    assert (output / 'previews' / f"{row['dataset_id']}-0.png").read_bytes().startswith(b'\x89PNG')
    assert original_image.read_bytes() == png_bytes()


@pytest.mark.parametrize('process', ['grinding', 'blasting'])
def test_non_spray_texture_is_preserved_without_thickness_analysis(tmp_path, process):
    root, output, _, job = write_job(tmp_path, process=process)
    attach_heightmap(root, job)
    row = build_record(job, root, output)
    data = read_artifact(output, row)
    assert data['coating']['status'] == 'not_applicable'
    assert data['coating']['maps'][0]['status'] == 'not_applicable'
    assert data['coating']['maps'][0]['thickness_image_url'] is None
    assert not list(output.glob('previews/*.png'))


def test_missing_optional_image_creates_review_artifact_and_preserves_link(tmp_path):
    root, output, _, job = write_job(tmp_path)
    job['heightmaps'] = [{'relative_path': 'imports/missing.png', 'filename': 'missing.png', 'map_num': 4,
                         'performance_id': 88, 'snapshot_sequence_index': 3, 'semantics': 'last_snapshot'}]
    row = build_record(job, root, output)
    data = read_artifact(output, row)
    assert row['frame_count'] == 3 and row['quality_status'] == 'needs_review'
    assert data['quality']['issues']
    assert data['provenance']['heightmap_candidates'][0]['relative_path'] == 'imports/missing.png'
    assert data['provenance']['heightmaps'] == [] and data['coating']['maps'] == []


def test_dataset_ids_isolate_duplicate_candidates_and_atomic_retries(tmp_path):
    root, output, _, job = write_job(tmp_path)
    jobs = [dict(job, candidate_id=f'candidate-{i}') for i in range(2)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(lambda item: build_record(item, root, output), jobs + jobs))
    assert rows[0]['dataset_id'] != rows[1]['dataset_id']
    assert rows[0]['dataset_id'] == rows[2]['dataset_id']
    assert len(rows[0]['dataset_id']) == 32
    assert len(list(output.glob('records/*.json'))) == 2
    assert all(read_artifact(output, row)['dataset_id'] == row['dataset_id'] for row in rows)
    assert not list(output.rglob('*.tmp'))
