"""Resume verified corpus jobs and atomically publish one winner per recording."""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re
import sqlite3
import tempfile
import time
import uuid


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def emit(event, **fields):
    print(json.dumps({'time': utc_now(), 'event': event, **fields}, ensure_ascii=False), flush=True)


def relative_parts(relative: str) -> list[str]:
    if (not isinstance(relative, str) or not relative or '\x00' in relative
            or PureWindowsPath(relative).drive or relative.startswith(('/', '\\'))):
        raise ValueError('unsafe relative path')
    parts = relative.replace('\\', '/').split('/')
    if any(part in {'', '.', '..'} for part in parts):
        raise ValueError('unsafe relative path')
    return parts


def safe_file(root: Path, relative: str) -> Path:
    parts = relative_parts(relative)
    root = root.resolve()
    path = root.joinpath(*parts).resolve()
    if not path.is_relative_to(root):
        raise ValueError('unsafe relative path')
    return path


def record_key(row: dict) -> str:
    if row.get('logical_record_key'):
        return row['logical_record_key']
    name = str(row.get('session_id') or row.get('dataset_name', ''))
    name = re.sub(r'\.csv(?:\.gz)?$', '', name, flags=re.I)
    name = re.sub(r'(?:_(?:Height|copy))+$', '', name, flags=re.I)
    return f"{row['process']}:{name}"


def rank(row: dict) -> tuple:
    available = row.get('zone_id_available')
    available = available if isinstance(available, bool) else (row.get('zone_count') or 0) > 0
    return (int(available), int((row.get('sequence_heatmap_count') or 0) > 0),
            {'converted': 3, 'raw': 2, 'legacy': 1}.get(row.get('source_kind', 'legacy'), 1),
            row.get('zone_count') or 0, row.get('sequence_heatmap_count') or 0,
            row.get('heightmap_count') or 0,
            row.get('frame_count') if isinstance(row.get('frame_count'), int) else -1)


def select_records(existing: list[dict], imported: list[dict]) -> tuple[list[dict], dict]:
    winners = {}
    exact = 0
    for row in [*existing, *sorted(imported, key=lambda item: item['dataset_id'])]:
        key = record_key(row)
        previous = winners.get(key)
        if previous is None:
            winners[key] = row
        elif (row.get('source_sha256') and row.get('source_sha256') == previous.get('source_sha256')
              and row.get('enrichment_sha256')
              and row['enrichment_sha256'] == previous.get('enrichment_sha256')):
            exact += 1
        elif rank(row) > rank(previous):
            winners[key] = row
    result = sorted(winners.values(), key=lambda item: (item['worker_name'], item['process'], item['dataset_name']))
    return result, {'superseded_count': len(existing) + len(imported) - len(result),
                    'exact_duplicate_count': exact}


def read_receipt(source_root: Path, relative: str) -> dict:
    receipt = safe_file(source_root, relative)
    value = json.loads(receipt.read_text(encoding='utf-8'))
    batch_id = receipt.name.removesuffix('.receipt.json')
    if (value.get('batch_id') != batch_id or value.get('kind') not in {'raw', 'converted_height'}
            or not re.fullmatch('[0-9a-f]{64}', value.get('sha256', ''))):
        raise ValueError('invalid transfer receipt')
    manifest = value.get('manifest')
    encoded = json.dumps(manifest, ensure_ascii=True, separators=(',', ':')).encode()
    if hashlib.sha256(encoded).hexdigest() != value.get('manifest_sha256'):
        raise ValueError('invalid transfer manifest checksum')
    base = receipt.parent.parent / 'archive' / ('raw' if value['kind'] == 'raw' else 'converted')
    prefix = base.relative_to(source_root.resolve()).as_posix() + '/'
    entries = {}
    for path, size in manifest:
        relative = prefix + '/'.join(relative_parts(path))
        if (not isinstance(size, int) or isinstance(size, bool) or size < 0
                or relative in entries):
            raise ValueError('invalid transfer manifest entry')
        entries[relative] = size
    return entries


class ReceiptCache:
    def __init__(self, source_root):
        self.root, self.items, self.checked = source_root, {}, {}

    def get(self, relative):
        # Completed receipts are immutable: batch IDs bind the source snapshot,
        # and transport publishes them only after checksum and extraction checks.
        if relative in self.items:
            return self.items[relative]
        now = time.monotonic()
        if now - self.checked.get(relative, -100) >= 5:
            self.checked[relative] = now
            try:
                self.items[relative] = read_receipt(self.root, relative)
            except (OSError, ValueError, TypeError):
                self.items.pop(relative, None)
        return self.items.get(relative)


def job_ready(job: dict, source_root: Path, cache=None) -> bool:
    manifests = []
    for relative in job.get('required_receipts', []):
        try:
            manifest = cache.get(relative) if cache is not None else read_receipt(source_root, relative)
            if manifest is None:
                return False
            manifests.append(manifest)
        except (OSError, ValueError, TypeError):
            return False
    for item in [job['source'], *job.get('heightmaps', [])]:
        path = safe_file(source_root, item['relative_path'])
        if job.get('required_receipts') and not any(manifest.get(item['relative_path']) == item.get('bytes') for manifest in manifests):
            return False
        if not path.is_file() or (item.get('bytes') is not None and path.stat().st_size != item['bytes']):
            return False
    return True


def file_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', dir=path.parent, prefix='.' + path.name,
                                         suffix='.tmp', encoding='utf-8', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def validate_row(root: Path, row: dict, *, checksum=False):
    ident = row.get('dataset_id', '')
    if not re.fullmatch('[0-9a-f]{32}', ident):
        raise ValueError('invalid dataset ID')
    path = safe_file(root, f'records/{ident}.json')
    if path.stat().st_size != row['json_bytes']:
        raise ValueError('artifact size mismatch')
    if checksum and file_hash(path) != row['artifact_sha256']:
        raise ValueError('artifact checksum mismatch')


def publish_catalog(root: Path, report: dict, *, expected_hash: str | None) -> str:
    import fcntl
    with (root / '.catalog.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        destination = root / 'catalog.json'
        if file_hash(destination) != expected_hash:
            raise RuntimeError('concurrent catalog change detected; preserve it and reconcile before retry')
        for row in report['records']:
            validate_row(root, row)
        if file_hash(destination) != expected_hash:
            raise RuntimeError('concurrent catalog change during validation')
        atomic_json(destination, report)
        return file_hash(destination)


def recover_publication(root, published, expected_records, owner_id):
    current_hash = file_hash(root / 'catalog.json')
    if not published or published['catalog_sha256'] == current_hash:
        return current_hash
    current = json.loads((root / 'catalog.json').read_text(encoding='utf-8'))
    if current.get('import_run_id') == owner_id and current.get('records') == expected_records:
        return current_hash
    raise RuntimeError('concurrent catalog change after last publication')


def enrich_row(row, artifact):
    row = dict(row)
    features = artifact.get('features', {})
    row['source_sha256'] = artifact.get('provenance', {}).get('source_csv', {}).get('sha256')
    for field in ('zone_id_available', 'sequence_heatmap_count'):
        if field in features:
            row[field] = features[field]
    metadata = artifact.get('metadata', {})
    evidence = {'metadata': {k: metadata.get(k) for k in ('worker_name', 'role', 'acquired_at', 'qualification')},
                'zone_id_available': row.get('zone_id_available'),
                'zone_count': row.get('zone_count'), 'sequence_heatmap_count': row.get('sequence_heatmap_count'),
                'heightmaps': artifact.get('provenance', {}).get('heightmaps', [])}
    row['enrichment_sha256'] = hashlib.sha256(json.dumps(evidence, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return row


def _build(job, source_root, output_root):
    from trainer.demo_api.training_corpus_record import build_import_record
    row = build_import_record(job, Path(source_root), Path(output_root))
    artifact = json.loads((Path(output_root) / 'records' / (row['dataset_id'] + '.json')).read_text(encoding='utf-8'))
    return enrich_row(row, artifact)


def run(plan_path: Path, source_root: Path, output_root: Path, state_root: Path,
        *, workers=4, wait_for_files=True, publish_every=50):
    # Linux production driver: separate from operational session SQLite.
    import fcntl
    state_root.mkdir(parents=True, exist_ok=True)
    lock = (state_root / 'ingest.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    owner_path = state_root / 'owner.json'
    if not owner_path.exists():
        atomic_json(owner_path, {'import_run_id': uuid.uuid4().hex})
    owner_id = json.loads(owner_path.read_text())['import_run_id']
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    jobs = {job['candidate_id']: job for job in plan['candidates']}
    db = sqlite3.connect(state_root / 'ingest.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, status TEXT NOT NULL, row_json TEXT, error TEXT, updated_at TEXT NOT NULL)')
    for ident, job in jobs.items():
        fingerprint = hashlib.sha256(json.dumps(job, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        previous = db.execute('SELECT fingerprint FROM jobs WHERE id=?', (ident,)).fetchone()
        if previous and previous[0] != fingerprint:
            raise RuntimeError('job metadata changed; use a new isolated state root')
        db.execute('INSERT OR IGNORE INTO jobs VALUES (?,?,?,NULL,NULL,?)', (ident, fingerprint, 'pending', utc_now()))
    db.commit()
    baseline_path = state_root / 'baseline-catalog.json'
    output_root.mkdir(parents=True, exist_ok=True)
    if not baseline_path.exists():
        current = output_root / 'catalog.json'
        snapshot = current.read_bytes() if current.exists() else None
        baseline = json.loads(snapshot) if snapshot is not None else {'records': []}
        baseline['_baseline_source_sha256'] = hashlib.sha256(snapshot).hexdigest() if snapshot is not None else None
        atomic_json(baseline_path, baseline)
    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    existing = baseline.get('records', [])
    for i, row in enumerate(existing):
        path = output_root / 'records' / f"{row['dataset_id']}.json"
        validate_row(output_root, row, checksum=True)
        existing[i] = enrich_row(row, json.loads(path.read_text(encoding='utf-8')))
    completed = {}
    for ident, encoded in db.execute("SELECT id,row_json FROM jobs WHERE status='complete'"):
        if ident in jobs:
            row = json.loads(encoded)
            validate_row(output_root, row, checksum=True)
            completed[ident] = enrich_row(row, json.loads((output_root / 'records' / (row['dataset_id'] + '.json')).read_text(encoding='utf-8')))
    failed = {ident: error for ident, error in db.execute("SELECT id,error FROM jobs WHERE status='failed'") if ident in jobs}
    pending = {ident: job for ident, job in jobs.items() if ident not in completed and ident not in failed}
    # A previous run may have published a subset. Its hash is recorded separately.
    publication_state = state_root / 'publication.json'
    published = json.loads(publication_state.read_text()) if publication_state.exists() else None
    if not published and '_baseline_source_sha256' in baseline and file_hash(output_root / 'catalog.json') != baseline['_baseline_source_sha256']:
        raise RuntimeError('concurrent catalog change after initial baseline snapshot')
    expected_hash = recover_publication(output_root, published, select_records(existing, list(completed.values()))[0], owner_id)
    def publish(status):
        nonlocal expected_hash
        rows, dedup = select_records(existing, list(completed.values()))
        report = {'schema_version': 'record-features-1.2', 'converted_at': utc_now(), 'import_run_id': owner_id,
                  'record_count': len(rows), 'failed_count': len(failed),
                  'duplicate_count': dedup['superseded_count'],
                  'counting_rule': '공정과 훈련 기록 ID가 같은 데이터는 유효한 Zone ID·시퀀스 히트맵을 우선해 1건으로 집계합니다.',
                  'import_status': status, 'import_summary': {**plan['summary'], **dedup,
                      'built_candidates': len(completed), 'pending_candidates': len(jobs) - len(completed) - len(failed),
                      'failed_candidates': len(failed)},
                  'records': rows,
                  'failures': [{'candidate_id': ident, 'reason': error} for ident, error in sorted(failed.items())]}
        expected_hash = publish_catalog(output_root, report, expected_hash=expected_hash)
        atomic_json(publication_state, {'catalog_sha256': expected_hash, 'updated_at': utc_now(),
                                       'record_count': len(rows), 'status': status})
        emit('published', status=status, records=len(rows), built=len(completed), failed=len(failed), **dedup)
    running = {}
    receipt_cache = ReceiptCache(source_root)
    ready_queue, queued = deque(), set()
    next_ready_check = 0
    since_publish = 0
    last_status = last_publish = time.monotonic()
    emit('started', jobs=len(jobs), already_complete=len(completed), failed=len(failed), workers=workers)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        while pending or running:
            if len(running) < workers:
                now = time.monotonic()
                if now >= next_ready_check:
                    ready = [ident for ident, job in pending.items() if ident not in queued and job_ready(job, source_root, receipt_cache)]
                    ready.sort(key=lambda ident: (jobs[ident]['source_kind'] != 'converted', jobs[ident]['source']['bytes']))
                    ready_queue.extend(ready)
                    queued.update(ready)
                    next_ready_check = now + 5
                for _ in range(min(len(ready_queue), workers - len(running))):
                    ident = ready_queue.popleft()
                    queued.remove(ident)
                    job = pending.pop(ident)
                    running[pool.submit(_build, job, str(source_root), str(output_root))] = ident
            if not running:
                if not wait_for_files:
                    break
                time.sleep(5)
            else:
                done, _ = wait(running, timeout=5, return_when=FIRST_COMPLETED)
                for future in done:
                    ident = running.pop(future)
                    try:
                        row = future.result()
                        validate_row(output_root, row, checksum=True)
                        completed[ident] = row
                        db.execute('UPDATE jobs SET status=?,row_json=?,error=NULL,updated_at=? WHERE id=?',
                                   ('complete', json.dumps(row, ensure_ascii=False), utc_now(), ident))
                    except Exception as error:
                        failed[ident] = f'{type(error).__name__}: {error}'
                        db.execute('UPDATE jobs SET status=?,error=?,updated_at=? WHERE id=?',
                                   ('failed', failed[ident], utc_now(), ident))
                        emit('job_failed', candidate_id=ident, error=failed[ident])
                    db.commit()
                    since_publish += 1
            now = time.monotonic()
            if since_publish and (since_publish >= publish_every or now - last_publish >= 60):
                publish('in_progress')
                since_publish = 0
                last_publish = now
            if now - last_status >= 30:
                emit('progress', complete=len(completed), failed=len(failed), pending=len(pending), running=len(running))
                last_status = now
    status = 'complete' if len(completed) + len(failed) == len(jobs) and not failed else 'needs_review' if failed else 'waiting_for_files'
    publish(status)
    db.close()
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()
    return {'status': status, 'complete': len(completed), 'failed': len(failed), 'pending': len(pending)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--state-root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4, choices=range(1, 9))
    parser.add_argument('--available-only', action='store_true')
    args = parser.parse_args()
    result = run(args.plan, args.source_root, args.output_root, args.state_root,
                 workers=args.workers, wait_for_files=not args.available_only)
    emit('finished', **result)
    return 1 if result['failed'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
