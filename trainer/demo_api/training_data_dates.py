"""Import acquisition calendar dates from verified original-recording folders."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shutil
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4


def acquisition_date(value) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    try:
        if len(value) == 10:
            return date.fromisoformat(value).isoformat()
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone(timedelta(hours=9)))
        return parsed.date().isoformat()
    except ValueError:
        return None


def _write(path: Path, data: dict):
    temp = path.with_name(path.name + '.date-import.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def _sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def apply_acquisition_dates(source_root: Path, dataset_root: Path, manifest: dict) -> dict:
    from trainer.demo_api.training_data_catalog import TrainingDataCatalog
    source_root, dataset_root = Path(source_root).resolve(), Path(dataset_root).resolve()
    index_path = dataset_root / 'catalog.json'
    before_index = index_path.read_bytes()
    index = json.loads(before_index)
    rows = {row['dataset_id']: row for row in index['records']}
    entries = manifest.get('records')
    if not isinstance(entries, list) or not entries:
        raise ValueError('취득일 매핑이 비어 있습니다.')
    store = TrainingDataCatalog(dataset_root, source_root)
    planned, seen = [], set()
    for entry in entries:
        dataset_id = entry.get('dataset_id')
        if dataset_id in seen or dataset_id not in rows:
            raise ValueError('중복되었거나 등록되지 않은 데이터 ID입니다.')
        seen.add(dataset_id)
        row = rows[dataset_id]
        if any(entry.get(key) != row.get(key) for key in ('dataset_name', 'worker_id', 'process')):
            raise ValueError('원본 기록 식별 정보가 일치하지 않습니다.')
        acquired = acquisition_date(entry.get('acquired_at'))
        if acquired is None or acquired != entry['acquired_at']:
            raise ValueError('취득일은 유효한 YYYY-MM-DD 날짜여야 합니다.')
        origin = entry.get('date_source')
        if not isinstance(origin, dict) or origin.get('kind') != 'original_date_folder' or origin.get('precision') != 'day':
            raise ValueError('원본 날짜 폴더의 취득일 근거가 필요합니다.')
        if acquired.replace('-', '') not in str(origin.get('folder', '')):
            raise ValueError('원본 폴더명과 취득일이 일치하지 않습니다.')
        if Path(str(origin.get('source_csv', ''))).stem.removesuffix('_copy') != row['dataset_name']:
            raise ValueError('날짜 근거의 CSV와 해당 훈련 기록이 다릅니다.')
        artifact_path = store.artifact(dataset_id)
        artifact = json.loads(artifact_path.read_text(encoding='utf-8'))
        already_dated = (artifact['metadata'].get('acquired_at') == acquired
                         and artifact['provenance'].get('acquisition_date_source') == origin
                         and 'converted_at' not in artifact['metadata'] and 'converted_at' not in row)
        expected = entry.get('expected_artifact_sha256')
        if expected and expected != row['artifact_sha256'] and not already_dated:
            raise ValueError('매핑 생성 이후 학습 데이터가 변경되었습니다.')
        csv_info = artifact['provenance']['source_csv']
        csv_path = (source_root / csv_info['relative_path']).resolve()
        if not csv_path.is_relative_to(source_root) or not csv_path.is_file() or _sha(csv_path) != csv_info['sha256']:
            raise ValueError('원본 CSV의 경로 또는 체크섬을 확인해 주세요.')
        metadata_path = (csv_path.parent / 'metadata.json').resolve()
        if not metadata_path.is_relative_to(source_root):
            raise ValueError('메타데이터 경로가 원본 자료 범위를 벗어납니다.')
        previous = metadata_path.read_bytes() if metadata_path.exists() else None
        metadata = json.loads(previous.decode('utf-8-sig')) if previous is not None else {}
        if not isinstance(metadata, dict):
            raise ValueError('기존 metadata.json 형식이 올바르지 않습니다.')
        previous_date = acquisition_date(metadata.get('acquired_at'))
        if previous_date and previous_date != acquired:
            raise ValueError('기존 원본 메타데이터의 취득일과 충돌합니다.')
        new_metadata = {**metadata, 'acquired_at': acquired, 'acquisition_date_source': origin}
        if already_dated and new_metadata == metadata:
            continue
        artifact['metadata']['acquired_at'] = acquired
        artifact['metadata'].pop('converted_at', None)
        artifact['provenance']['acquisition_date_source'] = origin
        artifact['provenance']['date_note'] = '원본 날짜 폴더와 훈련 기록을 대조한 취득일(연월일)입니다.'
        planned.append((dataset_id, artifact, metadata_path, previous, new_metadata))
    if not planned:
        return {'updated_count': 0, 'backup_path': None}
    suffix = uuid4().hex[:12]
    stage = dataset_root.with_name(dataset_root.name + '.dates-stage-' + suffix)
    backup = dataset_root.with_name(dataset_root.name + '.before-dates-' + suffix)
    if stage.parent.resolve() != dataset_root.parent or backup.parent.resolve() != dataset_root.parent:
        raise ValueError('데이터 백업 경로를 확인해 주세요.')
    shutil.copytree(dataset_root, stage, symlinks=True)
    for dataset_id, artifact, _, _, _ in planned:
        path = stage / 'records' / f'{dataset_id}.json'
        if not path.resolve().is_relative_to(stage.resolve()):
            raise ValueError('잘못된 학습 파일 경로입니다.')
        _write(path, artifact)
        row = rows[dataset_id]
        row.pop('converted_at', None)
        row['acquired_at'], row['json_bytes'], row['artifact_sha256'] = artifact['metadata']['acquired_at'], path.stat().st_size, _sha(path)
    _write(stage / 'catalog.json', index)
    journal = {'records': [{'path': p.relative_to(source_root).as_posix(), 'before_base64': base64.b64encode(old).decode() if old is not None else None}
                           for _, _, p, old, _ in planned], 'manifest': manifest}
    _write(backup.with_suffix(backup.suffix + '.metadata.json'), journal)
    if index_path.read_bytes() != before_index:
        raise ValueError('다른 작업이 학습 목록을 변경했습니다. 최신 목록으로 다시 대조해 주세요.')
    moved_old = False
    try:
        for _, _, path, _, metadata in planned:
            _write(path, metadata)
        dataset_root.rename(backup)
        moved_old = True
        stage.rename(dataset_root)
    except Exception:
        if moved_old and not dataset_root.exists():
            backup.rename(dataset_root)
        for _, _, path, previous, _ in planned:
            if previous is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(previous)
        raise
    return {'updated_count': len(planned), 'backup_path': str(backup)}


def main():
    parser = argparse.ArgumentParser(description='원본 폴더 근거의 학습 데이터 취득일 반영')
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(apply_acquisition_dates(args.source_root, args.dataset_root,
                                           json.loads(args.manifest.read_text(encoding='utf-8'))), ensure_ascii=False))


if __name__ == '__main__':
    main()
