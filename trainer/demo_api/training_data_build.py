"""Build one verifiable feature dataset per source recording, outside the API DB."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trainer.demo_api.training_data_coating import analyze_heightmap, render_thickness_preview
from trainer.demo_api.training_data_dates import acquisition_date
from trainer.expert_zone_mapping import canonical_expert_zone_id
from trainer.zone_reference import ZoneReferenceCatalog

VERSION = 'record-features-1.1'


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
    sources = [{'root': root, 'worker_id': 'jeon_hyeonggi', 'worker_name': meta.get('display_name') or '전형기 반장',
                'role': meta.get('role') or 'expert'}]
    for folder in sorted((root / 'workers').glob('*')):
        if folder.is_dir() and (folder / 'raw').is_dir() and folder.resolve().is_relative_to(root.resolve()):
            meta = _read_json(folder / 'worker.json')
            sources.append({'root': folder, 'worker_id': folder.name,
                            'worker_name': meta.get('display_name') or folder.name, 'role': meta.get('role') or 'trainee'})
    return sources


def _features(path: Path, process: str, scenario: str, *, legacy_expert=False, content_type='learning') -> dict:
    opener = gzip.open if path.name.endswith('.gz') else open
    zones: dict[tuple[int, str], dict] = {}
    order: list[dict] = []
    sequences, frames, spraying, unmapped = 0, 0, 0, 0
    active = False
    first_time = last_time = None
    metric_names = {'distance_mm': 'distance (input)', 'speed_mms': 'speed (input)',
                    'distance_real_mm': 'distance (real)', 'speed_real_mms': 'speed (real)'}
    with opener(path, 'rt', encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        columns = [str(c).strip() for c in reader.fieldnames or []]
        lower = {c.casefold() for c in columns}
        if not {'분사', 'zone id'}.issubset(lower):
            raise ValueError('필수 열(분사, Zone ID)이 없습니다.')
        for original in reader:
            row = {str(k).strip().casefold(): v for k, v in original.items() if k is not None}
            frames += 1
            t = _number(row.get('시간(초)', row.get('time')))
            if t is not None:
                if first_time is None:
                    first_time = t
                last_time = t
            on = str(row.get('분사', '')).strip().casefold() in {'on', '1', 'true'}
            if on and not active:
                sequences += 1
            active = on
            if not on:
                continue
            spraying += 1
            zone = str(row.get('zone id', '')).strip()
            if process in {'spray', 'blasting'}:
                zone = canonical_expert_zone_id(process, content_type if legacy_expert else 'source',
                                                int(scenario) if scenario.isdigit() else 0, zone) or ''
            if zone.casefold() in {'', '-1', '0', 'none', 'nan', 'null'}:
                unmapped += 1
                continue
            map_value = _number(row.get('ray texture map number'))
            map_num = int(map_value) if map_value is not None and map_value >= 1 else 1
            key = (map_num, zone)
            if key not in zones:
                zones[key] = {'map_num': map_num, 'zone_id': zone, 'frame_count': 0,
                              'metrics': {k: [] for k in metric_names}, 'ray_count': 0,
                              'ray_min': [math.inf] * 3, 'ray_max': [-math.inf] * 3}
                order.append({'map_num': map_num, 'zone_id': zone})
            z = zones[key]
            z['frame_count'] += 1
            for name, column in metric_names.items():
                value = _number(row.get(column))
                if value is not None and value >= 0:
                    z['metrics'][name].append(value)
            ray = [_number(row.get(f'ray point {a}')) for a in 'xyz']
            if all(v is not None for v in ray):
                z['ray_count'] += 1
                z['ray_min'] = [min(a, b) for a, b in zip(z['ray_min'], ray)]
                z['ray_max'] = [max(a, b) for a, b in zip(z['ray_max'], ray)]
    if not frames:
        raise ValueError('CSV에 프레임이 없습니다.')
    output_zones = []
    for z in zones.values():
        output_zones.append({'map_num': z['map_num'], 'zone_id': z['zone_id'], 'frame_count': z['frame_count'],
                             **{k: _distribution(v) for k, v in z['metrics'].items()},
                             'ray_point_bounds_m': {'min': z['ray_min'], 'max': z['ray_max']} if z['ray_count'] else None})
    return {'frame_count': frames, 'spraying_frame_count': spraying, 'unmapped_spraying_frames': unmapped,
            'sequence_count': sequences, 'duration_seconds': round(max(0, last_time - first_time), 6) if first_time is not None else None,
            'source_columns': columns, 'zone_order': order, 'zones': output_zones,
            'definitions': {'sequence_count': '분사 Off에서 On으로 전환된 연속 분사 구간 수',
                            'zone_order': '면 번호와 Zone별 최초 방문 순서. 여러 회차의 순서 통계는 별도 전문가 프로필에서 계산합니다.',
                            'distance_mm': 'Distance (Input)의 유효한 분사 프레임 분포',
                            'speed_mms': 'Speed (Input)의 유효한 분사 프레임 분포'}}


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
