"""Build one verifiable feature dataset per source recording, outside the API DB."""
from __future__ import annotations

import argparse
import base64
import binascii
import csv
import gzip
import hashlib
import json
import math
import re
import statistics
import struct
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trainer.demo_api.training_data_coating import analyze_heightmap, render_thickness_preview
from trainer.demo_api.training_data_dates import acquisition_date
from trainer.expert_zone_mapping import canonical_expert_zone_id
from trainer.zone_reference import ZoneReferenceCatalog

VERSION = 'record-features-1.1'

# Sparse PNG cells exceed csv's 128 KiB default. Keep both parsing and binary
# decoding bounded, independently of the (usually much smaller) row width.
_CSV_FIELD_LIMIT = 64 * 1024 * 1024
_HEATMAP_PNG_LIMIT = 16 * 1024 * 1024
_HEATMAP_BASE64_LIMIT = ((_HEATMAP_PNG_LIMIT + 2) // 3) * 4
_HEATMAP_FORMAT = 'v2;thickness=RG16;alpha=B8;scale_um=20000'


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {'count': 0, 'mean': None, 'median': None, 'p25': None, 'p75': None}
    ordered = sorted(values)
    def percentile(q: float) -> float:
        i = (len(ordered) - 1) * q
        low = int(i)
        return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (i - low)
    return {'count': len(values), 'mean': round(statistics.fmean(values), 6),
            'median': round(statistics.median(values), 6), 'p25': round(percentile(.25), 6),
            'p75': round(percentile(.75), 6)}


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def worker_sources(root: Path) -> list[dict]:
    meta = _read_json(root / 'worker.json')
    sources = [{'root': root, 'worker_id': 'jeon_hyeonggi', 'worker_name': meta.get('display_name') or '전형기',
                'role': meta.get('role') or 'expert'}]
    for folder in sorted((root / 'workers').glob('*')):
        if folder.is_dir() and (folder / 'raw').is_dir() and folder.resolve().is_relative_to(root.resolve()):
            meta = _read_json(folder / 'worker.json')
            sources.append({'root': folder, 'worker_id': folder.name,
                            'worker_name': meta.get('display_name') or folder.name, 'role': meta.get('role') or 'trainee'})
    return sources


def _reject_json_constant(value: str) -> None:
    raise ValueError(f'JSON의 유한하지 않은 숫자: {value}')


def _sequence_heatmap_maps(cell: str) -> list[dict] | None:
    """Validate sparse payloads without resizing PNGs or loading any sidecar."""
    if len(cell) > _CSV_FIELD_LIMIT:
        return None
    try:
        maps = json.loads(cell, parse_constant=_reject_json_constant)
        if not isinstance(maps, list) or len(maps) > 256:
            return None
        for image in maps:
            if not isinstance(image, dict) or image.get('encoding') != 'png;base64' or image.get('format') != _HEATMAP_FORMAT:
                return None
            if image.get('map_num') is None:
                return None
            # Crop coordinates refer to the full source texture; img_* refer
            # to the smaller encoded PNG and must never replace width/height.
            sizes = [_number(image.get(name)) for name in ('x0', 'y0', 'width', 'height', 'img_width', 'img_height')]
            if any(value is None or not value.is_integer() for value in sizes):
                return None
            if any(value < 0 for value in sizes[:2]) or any(value <= 0 for value in sizes[2:]):
                return None
            encoded = image.get('data')
            if not isinstance(encoded, str) or len(encoded) > _HEATMAP_BASE64_LIMIT:
                return None
            png = base64.b64decode(encoded, validate=True)
            if len(png) > _HEATMAP_PNG_LIMIT or len(png) < 33 or png[:16] != b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR':
                return None
            if struct.unpack('>II', png[16:24]) != (int(sizes[4]), int(sizes[5])):
                return None
        return maps
    except (ValueError, TypeError, OverflowError, binascii.Error, RecursionError):
        return None


def _map_identity(value: str) -> tuple[int | None, str | None]:
    label = value.strip()
    number = _number(label)
    if number is not None:
        return (int(number), None) if number >= 1 and number.is_integer() else (None, None)
    return None, label if label.casefold() not in {'', 'none', 'nan', 'null'} else None


def _features(path: Path, process: str, scenario: str, *, legacy_expert=False, content_type='learning') -> dict:
    opener = gzip.open if path.name.endswith('.gz') else open
    zones: dict[tuple[int | None, str | None, str], dict] = {}
    order: list[dict] = []
    sequences, frames, spraying, unmapped = 0, 0, 0, 0
    active = zone_available = False
    first_time = last_time = None
    metric_names = {'distance_mm': 'distance (input)', 'speed_mms': 'speed (input)',
                    'distance_real_mm': 'distance (real)', 'speed_real_mms': 'speed (real)'}
    global_values = {name: [] for name in metric_names}
    global_ray_min, global_ray_max, global_ray_count = [math.inf] * 3, [-math.inf] * 3, 0
    heatmaps, invalid_heatmaps = [], 0
    zone_cache: dict[str, str | None] = {}
    map_cache: dict[str, tuple[int | None, str | None]] = {}
    map_order: list[dict] = []
    seen_maps: set[tuple[int | None, str | None]] = set()
    remap_type = content_type if legacy_expert else 'source'
    mode_id = int(scenario) if scenario.isdigit() else 0
    if csv.field_size_limit() < _CSV_FIELD_LIMIT:
        csv.field_size_limit(_CSV_FIELD_LIMIT)
    with opener(path, 'rt', encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream)
        columns = [column.strip() for column in next(reader, [])]
        indices = {column.casefold(): index for index, column in enumerate(columns)}
        if '분사' not in indices:
            raise ValueError('필수 열(분사)이 없습니다.')
        spray_index = indices['분사']
        time_index = indices.get('시간(초)', indices.get('time'))
        zone_index = indices.get('zone id')
        map_index = indices.get('ray texture map number')
        heatmap_index = indices.get('cur_sequence_heatmap')
        metric_indices = [(name, indices[column]) for name, column in metric_names.items() if column in indices]
        ray_indices = [indices.get(f'ray point {axis}') for axis in 'xyz']
        # csv.reader performs the expensive wide-row split in C. Only selected
        # columns are touched here; no 173-key dictionary is built per frame.
        for row in reader:
            if not row:  # DictReader previously skipped empty physical lines.
                continue
            frame_index = frames
            frames += 1
            width = len(row)
            t = _number(row[time_index]) if time_index is not None and time_index < width else None
            if t is not None:
                if first_time is None:
                    first_time = t
                last_time = t
            on = spray_index < width and row[spray_index].strip().casefold() in {'on', '1', 'true'}
            if on and not active:
                sequences += 1
            active = on
            if heatmap_index is not None and heatmap_index < width:
                cell = row[heatmap_index].strip()
                if cell and cell != '[]':
                    maps = _sequence_heatmap_maps(cell)
                    if maps is None:
                        invalid_heatmaps += 1
                    elif maps:
                        heatmaps.append({'frame_index': frame_index, 'time_seconds': t,
                                         'spray_run_index': sequences or None, 'maps': maps})
            zone = None
            if zone_index is not None and zone_index < width:
                raw_zone = row[zone_index].strip()
                if raw_zone not in zone_cache:
                    number = _number(raw_zone)
                    value = str(int(number)) if number is not None and number > 0 and number.is_integer() else None
                    if value and process in {'spray', 'blasting'}:
                        value = canonical_expert_zone_id(process, remap_type, mode_id, value)
                    zone_cache[raw_zone] = value
                zone = zone_cache[raw_zone]
                if zone:
                    zone_available = True
            if not on:
                continue
            spraying += 1
            raw_map = row[map_index] if map_index is not None and map_index < width else ''
            if raw_map not in map_cache:
                map_cache[raw_map] = _map_identity(raw_map)
            map_num, map_label = map_cache[raw_map]
            map_key = (map_num, map_label)
            if (map_num is not None or map_label is not None) and map_key not in seen_maps:
                seen_maps.add(map_key)
                identity = {'map_num': map_num}
                if map_label is not None:
                    identity['map_label'] = map_label
                map_order.append(identity)
            metrics = []
            for name, index in metric_indices:
                value = _number(row[index]) if index < width else None
                if value is not None and value >= 0:
                    global_values[name].append(value)
                    metrics.append((name, value))
            ray = [_number(row[index]) if index is not None and index < width else None for index in ray_indices]
            has_ray = all(value is not None for value in ray)
            if has_ray:
                global_ray_count += 1
                for axis, value in enumerate(ray):
                    global_ray_min[axis] = min(global_ray_min[axis], value)
                    global_ray_max[axis] = max(global_ray_max[axis], value)
            if not zone:
                unmapped += 1
                continue
            key = (map_num, map_label, zone)
            if key not in zones:
                identity = {'map_num': map_num, 'zone_id': zone}
                if map_label is not None:
                    identity['map_label'] = map_label
                zones[key] = {**identity, 'frame_count': 0,
                              'metrics': {name: [] for name in metric_names}, 'ray_count': 0,
                              'ray_min': [math.inf] * 3, 'ray_max': [-math.inf] * 3}
                order.append(identity)
            z = zones[key]
            z['frame_count'] += 1
            for name, value in metrics:
                z['metrics'][name].append(value)
            if has_ray:
                z['ray_count'] += 1
                for axis, value in enumerate(ray):
                    z['ray_min'][axis] = min(z['ray_min'][axis], value)
                    z['ray_max'][axis] = max(z['ray_max'][axis], value)
    if not frames:
        raise ValueError('CSV에 프레임이 없습니다.')
    output_zones = []
    for z in zones.values():
        output = {'map_num': z['map_num'], 'zone_id': z['zone_id'], 'frame_count': z['frame_count'],
                  **{name: _distribution(values) for name, values in z['metrics'].items()},
                  'ray_point_bounds_m': {'min': z['ray_min'], 'max': z['ray_max']} if z['ray_count'] else None}
        if 'map_label' in z:
            output['map_label'] = z['map_label']
        output_zones.append(output)
    return {'frame_count': frames, 'spraying_frame_count': spraying, 'unmapped_spraying_frames': unmapped,
            'mapped_spraying_frame_count': spraying - unmapped,
            'zone_id_column_present': zone_index is not None, 'zone_id_available': zone_available,
            'zone_id_status': 'present' if zone_available else 'absent',
            'sequence_count': sequences, 'duration_seconds': round(max(0, last_time - first_time), 6) if first_time is not None else None,
            'source_columns': columns, 'zone_order': order, 'zones': output_zones,
            'map_order': map_order, 'map_labels': [identity['map_label'] for identity in map_order if 'map_label' in identity],
            'global_metrics': {**{name: _distribution(values) for name, values in global_values.items()},
                               'ray_point_bounds_m': {'min': global_ray_min, 'max': global_ray_max} if global_ray_count else None},
            'sequence_heatmap_count': len(heatmaps), 'sequence_heatmaps': heatmaps,
            'sequence_heatmap_invalid_count': invalid_heatmaps,
            'definitions': {'sequence_count': '분사 Off에서 On으로 전환된 연속 분사 구간 수',
                            'zone_order': '면 번호와 Zone별 최초 방문 순서. 여러 회차의 순서 통계는 별도 전문가 프로필에서 계산합니다.',
                            'map_order': '분사 프레임에서 면의 최초 등장 순서. 정수 번호 또는 원본 면 이름을 보존하며 미기록 면을 만들지 않습니다.',
                            'distance_mm': 'Distance (Input)의 유효한 분사 프레임 분포',
                            'speed_mms': 'Speed (Input)의 유효한 분사 프레임 분포',
                            'global_metrics': 'Zone 유무와 무관한 전체 유효 분사 프레임의 분포. 거리 mm, 속도 mm/s, 착탄 좌표 m.',
                            'sequence_heatmaps': '원본 sparse 행별 PNG 페이로드. frame_index는 0부터, spray_run_index는 On 전환으로 추정한 최근 분사 구간(1부터)이며 공식 sequence_id가 아닙니다.'}}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def attach_expert_references(output_root: Path, rows: list[dict]) -> None:
    """Use a distribution of per-record medians, keeping pixel/sample units distinct."""
    records = {}
    buckets: dict[tuple, list[float]] = {}
    for row in rows:
        path = output_root / 'records' / f"{row['dataset_id']}.json"
        data = _read_json(path)
        records[row['dataset_id']] = data
        if row['role'] != 'expert':
            continue
        for face in data['coating']['maps']:
            prefix = (row['process'], row['scenario_id'], face['map_num'], face.get('distribution_domain'))
            for zone_id, summary in [('', face.get('summary', {}))] + [(z['zone_id'], z.get('summary', {})) for z in face.get('zones', [])]:
                value = _number(summary.get('median_um'))
                if value is not None:
                    buckets.setdefault((*prefix, zone_id), []).append(value)
    for row in rows:
        data = records[row['dataset_id']]
        for face in data['coating']['maps']:
            prefix = (row['process'], row['scenario_id'], face['map_num'], face.get('distribution_domain'))
            for zone_id, destination in [('', face)] + [(z['zone_id'], z) for z in face.get('zones', [])]:
                values = buckets.get((*prefix, zone_id), [])
                if values:
                    stats = _distribution(values)
                    destination['reference'] = {'record_count': stats['count'], 'median_um': stats['median'],
                                                'p25_um': stats['p25'], 'p75_um': stats['p75'],
                                                'basis': '동일 훈련·면·Zone의 전문가 회차별 도막 중앙값 분포'}
        path = output_root / 'records' / f"{row['dataset_id']}.json"
        _write_json(path, data)
        row['json_bytes'], row['artifact_sha256'] = path.stat().st_size, _sha(path)


def build_training_datasets(source_root: Path, output_root: Path, *, catalog=None) -> dict:
    source_root, output_root = Path(source_root).resolve(), Path(output_root).resolve()
    if not (source_root / 'raw').is_dir() and not (source_root / 'workers').is_dir():
        raise ValueError('원본 raw 폴더가 없습니다.')
    if catalog is None:
        try:
            catalog = ZoneReferenceCatalog.default()
        except (OSError, ValueError, KeyError):
            catalog = None
    converted_at = datetime.now(timezone.utc).isoformat()
    rows, failures, duplicates = [], [], 0
    seen: set[str] = set()
    for worker in worker_sources(source_root):
        raw = worker['root'] / 'raw'
        paths = sorted(set(raw.rglob('*.csv')) | set(raw.rglob('*.csv.gz')))
        for csv_path in paths:
            try:
                if not csv_path.resolve().is_relative_to(source_root):
                    raise ValueError('원본 범위를 벗어난 경로입니다.')
                parts = csv_path.relative_to(raw).parts
                if len(parts) < 3 or parts[0] not in {'spray', 'blasting'}:
                    raise ValueError('훈련종류 폴더를 확인해 주세요.')
                process, content_type = parts[:2]
                name = csv_path.name.removesuffix('.gz').removesuffix('.csv')
                scenario = name.split('_', 1)[0]
                csv_hash = _sha(csv_path)
                dataset_id = hashlib.sha256(f"{worker['worker_id']}:{process}:{scenario}:{name}:{csv_hash}".encode()).hexdigest()[:32]
                if dataset_id in seen:
                    duplicates += 1
                    continue
                features = _features(csv_path, process, scenario, legacy_expert=worker['root'] == source_root,
                                     content_type=content_type)
                meta = _read_json(csv_path.parent / 'metadata.json')
                issues = []
                if not features['zones']:
                    issues.append('분사 중 유효한 Zone ID가 없어 Zone 기반 학습에 확인이 필요합니다.')
                image_paths = sorted(set(csv_path.parent.glob('*_Height.png')))
                if not image_paths:
                    issues.append('연결된 최종 높이맵이 없습니다.')
                maps, image_sources = [], []
                target = _number(meta.get('target_thickness_um'))
                target = target if target is not None and target > 0 else None
                for i, path in enumerate(image_paths):
                    if not path.resolve().is_relative_to(source_root):
                        raise ValueError('텍스처가 원본 범위를 벗어났습니다.')
                    match = re.search(r'Map(\d+)', path.name, re.I)
                    map_num = int(match.group(1)) if match else i + 1
                    image_sources.append({'filename': path.name, 'relative_path': path.relative_to(source_root).as_posix(),
                                          'sha256': _sha(path), 'bytes': path.stat().st_size, 'map_num': map_num})
                    preview = None
                    if process == 'spray':
                        try:
                            analysis = analyze_heightmap(path, scenario_id=scenario, map_num=map_num,
                                                         catalog=catalog, target_thickness_um=target)
                            preview = output_root / 'previews' / f'{dataset_id}-{i}.png'
                            preview.parent.mkdir(parents=True, exist_ok=True)
                            preview.write_bytes(render_thickness_preview(path, vmax_um=analysis.get('preview_max_um')))
                        except (OSError, ValueError) as error:
                            analysis = {'status': 'unavailable', 'reason': f'높이맵 해석 실패: {type(error).__name__}', 'zones': [], 'histogram': []}
                            issues.append(f'{path.name}: 높이맵을 확인해 주세요.')
                    else:
                        analysis = {'status': 'not_applicable', 'reason': '블라스팅 텍스처는 도막 두께로 해석하지 않습니다.', 'zones': [], 'histogram': []}
                    maps.append({**analysis, 'map_num': map_num, 'filename': path.name,
                                 'image_url': f'/admin/api/training-data/{dataset_id}/images/{i}',
                                 'thickness_image_url': f'/admin/api/training-data/{dataset_id}/images/{i}?view=thickness' if preview else None})
                status = 'needs_review' if issues else 'complete'
                metadata = {'dataset_id': dataset_id, 'dataset_name': name, 'worker_id': worker['worker_id'],
                            'worker_name': worker['worker_name'], 'role': worker['role'], 'process': process,
                            'scenario_id': scenario, 'content_type': content_type,
                            'mode_label': f'학습모드 {scenario}' if content_type == 'learning' else f'실전 {scenario}',
                            'acquired_at': acquisition_date(meta.get('acquired_at')),
                            'quality_status': status, 'frame_count': features['frame_count'],
                            'zone_count': len(features['zones']), 'heightmap_count': len(maps)}
                artifact = {'schema_version': VERSION, 'dataset_id': dataset_id,
                            'dataset_type': 'record_features', 'metadata': metadata,
                            'provenance': {'source_csv': {'filename': csv_path.name, 'relative_path': csv_path.relative_to(source_root).as_posix(),
                                                          'sha256': csv_hash, 'bytes': csv_path.stat().st_size},
                                           'heightmaps': image_sources, 'converter_version': VERSION,
                                           'acquisition_date_source': meta.get('acquisition_date_source'),
                                           'date_note': '취득일은 원본 근거의 연월일만 기록하며 파일 수정일로 추정하지 않습니다.'},
                            'quality': {'status': status, 'issues': issues}, 'features': features,
                            'coating': {'status': 'available' if maps and process == 'spray' else 'not_applicable',
                                        'semantics': '기록의 최종 도막 상태. 시퀀스별 증분이나 해당 회차만의 기여량은 아닙니다.', 'maps': maps},
                            'units': {'distance': 'mm', 'speed': 'mm/s', 'ray_point': 'm', 'thickness': 'µm', 'time': 's'},
                            'intended_use': 'Zone별 수치 특성과 최종 도막 분포를 보존하는 학습용 특징 데이터입니다. LLM 가중치 학습 완료를 의미하지 않습니다.'}
                artifact_path = output_root / 'records' / f'{dataset_id}.json'
                _write_json(artifact_path, artifact)
                rows.append({**metadata, 'json_bytes': artifact_path.stat().st_size, 'artifact_sha256': _sha(artifact_path)})
                seen.add(dataset_id)
            except (OSError, ValueError, csv.Error, UnicodeError) as error:
                failures.append({'source': csv_path.relative_to(source_root).as_posix(), 'reason': str(error)})
    attach_expert_references(output_root, rows)
    report = {'schema_version': VERSION, 'converted_at': converted_at, 'record_count': len(rows),
              'failed_count': len(failures), 'duplicate_count': duplicates,
              'counting_rule': '훈련 기록 1회의 CSV와 연결된 높이맵을 변환한 검증 가능한 JSON 1개를 1건으로 집계합니다.',
              'records': rows, 'failures': failures}
    _write_json(output_root / 'catalog.json', report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description='원본 기록별 학습용 특징 JSON 구축')
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    report = build_training_datasets(args.source_root, args.output_root)
    print(json.dumps({k: v for k, v in report.items() if k not in {'records', 'failures'}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
