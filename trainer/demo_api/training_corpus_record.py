"""Build one immutable-source import artifact without changing its catalog."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import stat
import tempfile
from typing import Any

from trainer.demo_api.training_data_build import _features, _number, _sha
from trainer.demo_api.training_data_coating import analyze_heightmap, render_thickness_preview
from trainer.demo_api.training_data_dates import acquisition_date


VERSION = 'record-features-1.2'
_FINAL_MAP_SEMANTICS = ('기록에서 면별 마지막으로 저장된 도막 상태입니다. '
                        '시퀀스별 증분이나 해당 회차만의 기여량은 아닙니다.')


def _relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError('원본 상대 경로가 없습니다.')
    normalized = value.replace('\\', '/')
    path = Path(normalized)
    windows = PureWindowsPath(value)
    if path.is_absolute() or windows.drive or windows.root or '..' in path.parts:
        raise ValueError('원본 범위를 벗어난 경로입니다.')
    return path.as_posix()


def _source_file(root: Path, relative: Any) -> tuple[Path, str]:
    relative = _relative_path(relative)
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('원본 범위를 벗어난 경로입니다.')
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError('원본이 일반 파일이 아닙니다.')
    return path, relative


def _fingerprint(path: Path, declared: dict) -> dict:
    before = path.stat()
    size = declared.get('bytes')
    if size is not None and (not isinstance(size, int) or isinstance(size, bool) or size != before.st_size):
        raise ValueError('선언된 원본 파일 크기가 실제 크기와 다릅니다.')
    digest = _sha(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('해시 계산 중 원본 파일이 변경되었습니다.')
    expected_sha = declared.get('sha256')
    if expected_sha is not None and expected_sha != digest:
        raise ValueError('선언된 원본 체크섬이 실제 체크섬과 다릅니다.')
    return {'sha256': digest, 'bytes': after.st_size, 'mtime_ns': after.st_mtime_ns}


def _atomic_write(root: Path, relative: str, payload: bytes) -> None:
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination = destination.resolve()
    if not destination.is_relative_to(root):
        raise ValueError('출력 범위를 벗어난 경로입니다.')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f'.{destination.name}.',
                                         suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _known_empty_csv(path: Path) -> bool:
    """Confirm no data rows; unreadable or malformed files have unknown counts."""
    opener = gzip.open if path.name.endswith('.gz') else open
    try:
        with opener(path, 'rt', encoding='utf-8-sig', newline='') as stream:
            reader = csv.reader(stream, strict=True)
            next(reader, None)
            return next((row for row in reader if row), None) is None
    except (OSError, ValueError, csv.Error, UnicodeError, EOFError):
        return False


def _extract(path: Path, process: str, scenario: str, content_type: str, issues: list) -> dict:
    try:
        features = _features(path, process, scenario, legacy_expert=False, content_type=content_type)
    except (OSError, ValueError, csv.Error, UnicodeError, EOFError) as error:
        issues.append(f'CSV 특징 분석을 사용할 수 없습니다: {type(error).__name__}: {error}')
        return {'analysis_status': 'unavailable', 'analysis_reason': str(error),
                'frame_count': 0 if _known_empty_csv(path) else None,
                'zone_id_available': False, 'zone_id_status': 'absent',
                'zones': [], 'zone_order': [], 'sequence_heatmaps': [],
                'sequence_heatmap_count': 0}
    features['analysis_status'] = 'available'
    if not features.get('zone_id_available'):
        issues.append('유효한 Zone ID가 없어 Zone 기반 학습에 확인이 필요합니다.')
    if features.get('sequence_heatmap_invalid_count', 0):
        issues.append(f"잘못된 시퀀스 히트맵 {features['sequence_heatmap_invalid_count']}개를 확인해 주세요.")
    return features


def build_import_record(job: dict, source_root: Path, output_root: Path, *, catalog=None) -> dict:
    """Write one features JSON and optional previews; return an unindexed row.

    Source CSVs and textures remain byte-for-byte unchanged. Missing optional
    textures and unavailable CSV analysis remain reviewable records, whereas
    invalid source identity, containment or size raises before publication.
    ``quality_rank`` and import identity fields let the caller select winners.
    """
    source_root, output_root = Path(source_root).resolve(), Path(output_root).resolve()
    process = str(job.get('process', ''))
    if process not in {'spray', 'blasting', 'grinding'}:
        raise ValueError('지원하지 않는 훈련 공정입니다.')
    candidate_id = job.get('candidate_id')
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError('가져오기 후보 ID가 없습니다.')
    source = job.get('source')
    if not isinstance(source, dict):
        raise ValueError('CSV 원본 정보가 없습니다.')
    csv_path, csv_relative = _source_file(source_root, source.get('relative_path'))
    measured = _fingerprint(csv_path, source)
    scenario = str(job.get('scenario_id', ''))
    content_type = str(job.get('content_type', 'learning'))
    source_kind = str(job.get('source_kind', 'raw'))
    worker_id = str(job.get('worker_id', ''))
    session_id = str(job.get('session_id', ''))
    identity = json.dumps([worker_id, process, session_id, measured['sha256'], candidate_id], ensure_ascii=False)
    dataset_id = hashlib.sha256(identity.encode('utf-8')).hexdigest()[:32]
    issues = list(job.get('issues') or [])
    features = _extract(csv_path, process, scenario, content_type, issues)
    optional_metadata = dict(job.get('metadata') or {})
    target = _number(optional_metadata.get('target_thickness_um'))
    target = target if target is not None and target > 0 else None
    heightmap_candidates = list(job.get('heightmaps') or [])
    image_sources, maps = [], []
    if not heightmap_candidates:
        issues.append('연결된 최종 높이맵이 없습니다.')
    for candidate in heightmap_candidates:
        # Unsafe paths are rejected before opening. Optional missing or invalid
        # files remain in heightmap_candidates and do not discard the CSV.
        relative = _relative_path(candidate.get('relative_path'))
        try:
            image_path, relative = _source_file(source_root, relative)
            image_measured = _fingerprint(image_path, candidate)
        except (OSError, ValueError) as error:
            issues.append(f'{relative}: 연결된 높이맵 확인이 필요합니다: {error}')
            continue
        map_num = candidate.get('map_num')
        numeric_map = _number(map_num)
        map_num = int(numeric_map) if numeric_map is not None and numeric_map >= 1 and numeric_map.is_integer() else None
        image_index = len(image_sources)
        image_sources.append({**candidate, 'filename': image_path.name, 'relative_path': relative,
                              **image_measured, 'map_num': map_num})
        preview_available = False
        if process == 'spray':
            try:
                analysis = analyze_heightmap(image_path, scenario_id=scenario, map_num=map_num,
                                             catalog=catalog, target_thickness_um=target)
                if analysis.get('status') != 'unavailable':
                    preview_bytes = render_thickness_preview(image_path, vmax_um=analysis.get('preview_max_um'))
                    _atomic_write(output_root, f'previews/{dataset_id}-{image_index}.png', preview_bytes)
                    preview_available = True
                else:
                    issues.append(f'{image_path.name}: 높이맵 해석을 사용할 수 없습니다.')
            except (OSError, ValueError) as error:
                analysis = {'status': 'unavailable', 'reason': f'높이맵 해석 실패: {type(error).__name__}',
                            'zones': [], 'histogram': []}
                issues.append(f'{image_path.name}: 높이맵을 확인해 주세요.')
        else:
            analysis = {'status': 'not_applicable', 'reason': f'{process} 텍스처는 도막 두께로 해석하지 않습니다.',
                        'zones': [], 'histogram': []}
        labels = {key: value for key, value in candidate.items()
                  if key not in {'filename', 'relative_path', 'sha256', 'bytes', 'mtime_ns'}}
        maps.append({**analysis, **labels, 'map_num': map_num, 'filename': image_path.name,
                     'image_url': f'/admin/api/training-data/{dataset_id}/images/{image_index}',
                     'thickness_image_url': f'/admin/api/training-data/{dataset_id}/images/{image_index}?view=thickness'
                     if preview_available else None})
    current = csv_path.stat()
    if (current.st_size, current.st_mtime_ns) != (measured['bytes'], measured['mtime_ns']):
        raise ValueError('특징 분석 중 CSV 원본이 변경되었습니다.')
    status = 'needs_review' if issues else 'complete'
    zone_available = bool(features.get('zone_id_available'))
    metadata = {**optional_metadata, 'dataset_id': dataset_id,
                'dataset_name': optional_metadata.get('dataset_name') or csv_path.name.removesuffix('.gz').removesuffix('.csv'),
                'worker_id': worker_id, 'worker_name': job.get('worker_name') or worker_id,
                'role': job.get('role') or 'unknown', 'process': process, 'scenario_id': scenario,
                'content_type': content_type, 'session_id': session_id,
                'mode_label': job.get('mode_label') or (f'학습모드 {scenario}' if content_type == 'learning' else f'실전 {scenario}'),
                'acquired_at': acquisition_date(job.get('acquired_at', optional_metadata.get('acquired_at'))),
                'quality_status': status, 'frame_count': features.get('frame_count'),
                'analysis_status': features['analysis_status'], 'zone_count': len(features.get('zones', [])),
                'zone_id_available': zone_available, 'zone_id_status': 'present' if zone_available else 'absent',
                'source_kind': source_kind, 'sequence_heatmap_count': features.get('sequence_heatmap_count', 0),
                'heightmap_count': len(maps)}
    for key in ('qualification', 'sequence_sidecar_count', 'sequence_raw_sidecar_count'):
        if key in job:
            metadata[key] = job[key]
    provenance = dict(job.get('provenance') or {})
    provenance.update({'source_csv': {**provenance.get('source_csv', {}), **source,
                                      'filename': csv_path.name, 'relative_path': csv_relative, **measured},
                       'heightmaps': image_sources, 'heightmap_candidates': heightmap_candidates,
                       'source_kind': source_kind, 'source_inventory_id': job.get('source_inventory_id'),
                       'source_sql': job.get('source_sql'), 'candidate_id': candidate_id,
                       'logical_record_key': job.get('logical_record_key'), 'session_id': session_id,
                       'converter_version': VERSION})
    for key in ('qualification', 'identity_evidence', 'sequence_sidecar_count', 'sequence_raw_sidecar_count'):
        if key in job:
            provenance[key] = job[key]
    if 'acquisition_date_source' in job:
        provenance['acquisition_date_source'] = job['acquisition_date_source']
    provenance.setdefault('date_note', '취득일은 원본 근거의 연월일만 기록하며 파일 수정일로 추정하지 않습니다.')
    if process != 'spray':
        coating_status = 'not_applicable'
    elif not maps or all(item.get('status') == 'unavailable' for item in maps):
        coating_status = 'unavailable'
    elif all(item.get('status') == 'available' for item in maps):
        coating_status = 'available'
    else:
        coating_status = 'partial'
    artifact = {'schema_version': VERSION, 'dataset_id': dataset_id, 'dataset_type': 'record_features',
                'metadata': metadata, 'provenance': provenance, 'features': features,
                'coating': {'status': coating_status, 'semantics': _FINAL_MAP_SEMANTICS, 'maps': maps},
                'units': {'distance': 'mm', 'speed': 'mm/s', 'ray_point': 'm', 'thickness': 'µm', 'time': 's'},
                'quality': {'status': status, 'issues': issues},
                'intended_use': '기록별 수치 특징, 원본 시퀀스 히트맵과 최종 도막 분포를 보존하는 학습용 데이터입니다. LLM 가중치 학습 완료를 의미하지 않습니다.'}
    payload = json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')
    _atomic_write(output_root, f'records/{dataset_id}.json', payload)
    return {**metadata, 'json_bytes': len(payload), 'artifact_sha256': hashlib.sha256(payload).hexdigest(),
            'logical_record_key': job.get('logical_record_key'), 'candidate_id': candidate_id,
            'source_sha256': measured['sha256'],
            'quality_rank': [int(zone_available), int(features.get('sequence_heatmap_count', 0) > 0),
                             {'converted': 3, 'raw': 2, 'legacy': 1}.get(source_kind, 0),
                             features.get('frame_count') if features.get('frame_count') is not None else -1]}
