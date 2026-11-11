"""Build a deterministic, metadata-only plan for the archived TSB corpus.

The inventory, sanitized SQL identity index and old labeled catalog are the only
inputs. Corpus files, image pixels, remote hosts and application APIs are never
opened. All payload references are relative to /expert_reference_data.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sqlite3
from typing import Any

try:
    from trainer.scripts.tsb_sql_identity import resolve_session
except ModuleNotFoundError as exc:
    if exc.name != 'trainer.scripts.tsb_sql_identity':
        raise
    # The local import workspace retains the same sanitized parser under its
    # original name. Production installs it as trainer.scripts.tsb_sql_identity.
    from sql_identity_import import resolve_session


ARCHIVE_PREFIX = 'imports/tsb-20261111/archive'
SESSION = re.compile(r'([^_]+)_(\d+)_(\d+)(?:_(?:Height|copy))*\Z', re.I)
RAW_HEIGHT = re.compile(r'Map(\d+)_(\d+)_(\d+)_(\d+)_Height\.png\Z', re.I)
SEQUENCE_HEIGHT = re.compile(r'(?:(.+)_)?S(\d+)_Map(\d+)_Height\.(png|raw)\Z', re.I)
PROCESS = {'blast': 'blasting', 'spray': 'spray', 'grind': 'grinding'}


def _hash(namespace: str, value: str) -> str:
    return hashlib.sha256((namespace + '\0' + value).encode('utf-8')).hexdigest()


def _stem(filename: str) -> str:
    return re.sub(r'\.csv(?:\.gz)?\Z', '', filename, flags=re.I)


def _session(filename: str) -> dict[str, Any] | None:
    match = SESSION.fullmatch(_stem(filename))
    if not match:
        return None
    scenario, work, order = match.groups()
    return {'session_id': f'{scenario}_{int(work)}_{int(order)}',
            'scenario_id': scenario, 'work_id': int(work), 'order_id': int(order)}


def _safe_relative(row: dict[str, Any]) -> str:
    """Reject traversal, rooted/drive paths and inventory path disagreement."""
    value = str(row['relative_path']).replace('\\', '/')
    parts = value.split('/')
    if (not value or value.startswith('/') or '\x00' in value or ':' in value
            or any(part in {'', '.', '..'} for part in parts)):
        raise ValueError(f"unsafe relative path in inventory row {row['id']}")
    if parts[-1] != row['filename']:
        raise ValueError(f"relative path and filename disagree in inventory row {row['id']}")
    root_text = str(row['root'])
    path_type = PureWindowsPath if ('\\' in root_text or re.match(r'^[A-Za-z]:', root_text)) else PurePosixPath
    root = path_type(root_text)
    path = path_type(str(row['path']))
    if not root.is_absolute() or not path.is_absolute():
        raise ValueError(f"inventory row {row['id']} requires an absolute root and path")
    try:
        actual_relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path not contained by root in inventory row {row['id']}") from exc
    if any(part in {'.', '..'} for part in actual_relative.parts) or actual_relative != path_type(*parts):
        raise ValueError(f"path does not agree with root and relative path in inventory row {row['id']}")
    return '/'.join(parts)


def _process(row: dict[str, Any]) -> str | None:
    if row['source_kind'] == 'converted_height':
        return 'spray'
    return PROCESS.get(row['relative_path'].split('/', 1)[0].casefold())


def _reference(row: dict[str, Any]) -> dict[str, Any]:
    kind = 'raw' if row['source_kind'] == 'raw' else 'converted'
    return {'relative_path': f"{ARCHIVE_PREFIX}/{kind}/{row['relative_path']}",
            'bytes': row['bytes'], 'mtime_ns': row['mtime_ns']}


def _common(rows: list[dict[str, Any]], field: str) -> Any:
    """Retain a value only when every SQL performance supplies the same value."""
    if not rows:
        return None
    value = rows[0].get(field)
    return value if all(row.get(field) == value for row in rows) else None


def _distinct(rows: list[dict[str, Any]], field: str) -> list[Any]:
    values = {json.dumps(row.get(field), sort_keys=True): row.get(field) for row in rows}
    return [values[key] for key in sorted(values, key=lambda key: (values[key] is None, key))]


def _legacy_records(catalog: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in catalog.get('records', []):
        session = _session(str(row.get('dataset_name', '')))
        if session and row.get('process') in PROCESS.values():
            records[f"{row['process']}:{session['session_id']}"].append(row)
    return records


def _identity(index: dict[str, Any], session: dict[str, Any], process: str,
              legacy: list[dict[str, Any]]) -> dict[str, Any]:
    resolved = resolve_session(index, session['session_id'] + '.csv', database='spray', source_table='result')
    rows = resolved['candidates']
    issues = []
    if resolved['status'] != 'matched':
        issues.append(resolved['status'])
    for field in ('user_id', 'display_name', 'acquisition_date', 'qualification', 'sql_scenario_id'):
        if len(_distinct(rows, field)) > 1:
            issues.append(f'conflicting_sql_{field}')
    sql_user = _common(rows, 'user_id')
    database = _common(rows, 'database') or 'spray'
    logical_key = f"{process}:{session['session_id']}"
    worker_hash = _hash('tsb-sql-user', f'{database}|{sql_user}') if sql_user else _hash('tsb-unresolved-session', logical_key)
    worker_id = ('tsb_user_' if sql_user else 'tsb_unknown_') + worker_hash[:24]
    name = _common(rows, 'display_name')
    name = str(name).strip() if name else None
    evidence = ['sql_result_user_join'] if name else []
    sql_names_absent = not rows or all(not row.get('display_name') for row in rows)
    legacy_name = _common(legacy, 'worker_name')
    if not name and sql_names_absent and legacy_name and str(legacy_name).strip():
        name = str(legacy_name).strip()
        evidence = ['existing_labeled_record']
    if not name:
        name = f'이름 미확인 · ID{worker_hash[:10]}'
        issues.append('missing_worker_name')
    explicit_expert = re.sub(r'\s+', '', name) == '전형기'
    exact_legacy_expert = bool(legacy and all(row.get('role') == 'expert' for row in legacy))
    if exact_legacy_expert and 'existing_labeled_record' not in evidence:
        evidence.append('existing_labeled_record')
    role = 'expert' if explicit_expert or exact_legacy_expert else 'worker'
    sql_qualification = _common(rows, 'qualification') or 'unknown'
    qualification = 'expert' if role == 'expert' else ('nonexpert' if sql_qualification == 'nonexpert' else 'unknown')
    if sql_qualification == 'expert' and role != 'expert':
        issues.append('sql_expert_qualification_not_person_verified')
    acquired = _common(rows, 'acquisition_date')
    if not acquired:
        issues.append('missing_acquisition_date')
    if any(row.get('process') not in {None, 'unknown', process} for row in rows):
        issues.append('sql_process_differs_from_source_directory')
    source_sql = {
        'match_status': resolved['status'], 'database': database,
        'source_table': 'result', 'work_id': session['work_id'], 'order_id': session['order_id'],
        'performance_id_candidates': _distinct(rows, 'performance_id'),
        'sql_scenario_id': _common(rows, 'sql_scenario_id'),
        'sql_scenario_id_candidates': _distinct(rows, 'sql_scenario_id'),
        'qualification': sql_qualification,
    }
    if 'available_order_ids' in resolved:
        source_sql['available_order_ids'] = resolved['available_order_ids']
    return {'worker_id': worker_id, 'worker_name': name, 'role': role,
            'qualification': qualification, 'acquired_at': acquired or None,
            'identity_evidence': evidence, 'source_sql': source_sql, 'issues': issues}


def _raw_heightmaps(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        match = RAW_HEIGHT.fullmatch(row['filename'])
        output.append({**_reference(row), 'source_inventory_id': row['id'], 'filename': row['filename'],
                       'map_num': int(match.group(1)), 'performance_id': int(match.group(4)),
                       'semantics': 'raw_final_heightmap'})
    return sorted(output, key=lambda item: (item['map_num'], item['performance_id'], item['relative_path']))


def _converted_heightmaps(rows: list[dict[str, Any]], session_id: str) -> tuple[list[dict[str, Any]], int, int]:
    last: dict[int, tuple[int, list[dict[str, Any]]]] = {}
    png_count, raw_count = 0, 0
    for row in rows:
        match = SEQUENCE_HEIGHT.fullmatch(row['filename'])
        if not match:
            continue
        if match.group(1):
            parsed = _session(match.group(1))
            if not parsed or parsed['session_id'] != session_id:
                continue
        if match.group(4).casefold() != 'png':
            raw_count += 1
            continue
        png_count += 1
        sequence, map_num = int(match.group(2)), int(match.group(3))
        if map_num not in last or sequence > last[map_num][0]:
            last[map_num] = sequence, [row]
        elif sequence == last[map_num][0]:
            last[map_num][1].append(row)
    maps = []
    for map_num, (sequence, selected) in sorted(last.items()):
        for row in sorted(selected, key=lambda value: value['relative_path']):
            maps.append({**_reference(row), 'source_inventory_id': row['id'],
                         'filename': row['filename'], 'map_num': map_num,
                         'performance_id': None, 'snapshot_sequence_index': sequence,
                         'semantics': 'last_available_sequence_snapshot'})
    return maps, png_count, raw_count


def create_import_plan(inventory: Path, index: Path, existing_catalog: Path) -> dict[str, Any]:
    """Return every session source candidate without reading or changing payloads."""
    with Path(index).open(encoding='utf-8') as stream:
        identity_index = json.load(stream)
    with Path(existing_catalog).open(encoding='utf-8') as stream:
        legacy_index = _legacy_records(json.load(stream))
    csvs = []
    raw_maps: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    converted_sidecars: dict[str, list[dict[str, Any]]] = defaultdict(list)
    archive_counts: Counter[str] = Counter()
    archive_bytes = 0
    connection = sqlite3.connect(Path(inventory).resolve().as_uri() + '?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    try:
        query = 'SELECT id,path,root,source_kind,relative_path,bytes,mtime_ns,extension,filename FROM files ORDER BY id'
        for raw in connection.execute(query):
            row = dict(raw)
            if row['source_kind'] not in {'raw', 'converted_height'}:
                raise ValueError(f"unsupported source kind in inventory row {row['id']}")
            row['relative_path'] = _safe_relative(row)
            archive_counts[row['source_kind']] += 1
            archive_bytes += row['bytes']
            lower = row['filename'].casefold()
            if lower.endswith(('.csv', '.csv.gz')):
                csvs.append(row)
            if row['source_kind'] == 'raw' and (match := RAW_HEIGHT.fullmatch(row['filename'])):
                process = _process(row)
                if process:
                    raw_maps[(process, int(match.group(2)), int(match.group(3)))].append(row)
            elif row['source_kind'] == 'converted_height' and SEQUENCE_HEIGHT.fullmatch(row['filename']):
                converted_sidecars[str(PurePosixPath(row['relative_path']).parent)].append(row)
    finally:
        connection.close()

    candidates = []
    excluded = []
    for row in csvs:
        session = _session(row['filename'])
        process = _process(row)
        if not session or not process:
            reason = ('summary_without_session_identifier' if re.fullmatch(r'[^_]+_Export', _stem(row['filename']), re.I)
                      else 'invalid_identifier' if not session else 'unknown_source_process')
            excluded.append({'source_inventory_id': row['id'], 'source': _reference(row), 'reason': reason})
            continue
        logical_key = f"{process}:{session['session_id']}"
        identity = _identity(identity_index, session, process, legacy_index.get(logical_key, []))
        source = _reference(row)
        candidate = {
            'candidate_id': _hash('tsb-corpus-source', source['relative_path'])[:32],
            'logical_record_key': logical_key, 'session_id': session['session_id'],
            'process': process, 'scenario_id': session['scenario_id'],
            'content_type': ('learning' if session['scenario_id'].isdigit() and 1 <= int(session['scenario_id']) <= 10 else 'practice/unit'),
            **identity,
            'source_kind': 'raw' if row['source_kind'] == 'raw' else 'converted',
            'source_inventory_id': row['id'], 'source': source,
        }
        if candidate['source_kind'] == 'raw':
            candidate['heightmaps'] = _raw_heightmaps(raw_maps.get((process, session['work_id'], session['order_id']), []))
            candidate['sequence_sidecar_count'] = 0
            candidate['sequence_raw_sidecar_count'] = 0
            performances = {heightmap['performance_id'] for heightmap in candidate['heightmaps']}
            if len(performances) > 1:
                candidate['issues'].append('ambiguous_heightmap_performance')
            if performances and candidate['source_sql']['performance_id_candidates']:
                if performances - set(candidate['source_sql']['performance_id_candidates']):
                    candidate['issues'].append('heightmap_performance_not_in_sql_candidates')
        else:
            folder = str(PurePosixPath(row['relative_path']).parent)
            maps, png_count, raw_count = _converted_heightmaps(converted_sidecars.get(folder, []), session['session_id'])
            candidate['heightmaps'] = maps
            candidate['sequence_sidecar_count'] = png_count
            candidate['sequence_raw_sidecar_count'] = raw_count
            if any(Counter(m['map_num'] for m in maps)[n] > 1 for n in {m['map_num'] for m in maps}):
                candidate['issues'].append('ambiguous_last_sequence_snapshot')
        if not candidate['heightmaps']:
            candidate['issues'].append('missing_heightmap')
        candidates.append(candidate)

    candidates.sort(key=lambda candidate: (candidate['logical_record_key'], candidate['source']['relative_path']))
    groups: dict[str, list[str]] = defaultdict(list)
    for candidate in candidates:
        groups[candidate['logical_record_key']].append(candidate['candidate_id'])
    duplicate_groups = [{'logical_record_key': key, 'candidate_ids': ids, 'candidate_count': len(ids)}
                        for key, ids in sorted(groups.items()) if len(ids) > 1]
    summary = {
        'archive_file_count': sum(archive_counts.values()), 'archive_bytes': archive_bytes,
        'archive_files_by_source_kind': dict(sorted(archive_counts.items())),
        'csv_file_count': len(csvs), 'candidate_count': len(candidates),
        'logical_record_count': len(groups),
        'ancillary_file_count': sum(archive_counts.values()) - len(candidates),
        'excluded_csv_count': len(excluded),
        'excluded_csv_reasons': dict(Counter(item['reason'] for item in excluded)),
        'by_process': dict(Counter(c['process'] for c in candidates)),
        'by_source_kind': dict(Counter(c['source_kind'] for c in candidates)),
        'by_content_type': dict(Counter(c['content_type'] for c in candidates)),
        'sql_match_status': dict(Counter(c['source_sql']['match_status'] for c in candidates)),
        'missing_worker_name': sum('missing_worker_name' in c['issues'] for c in candidates),
        'missing_acquired_at': sum(c['acquired_at'] is None for c in candidates),
        'existing_labeled_identity_count': sum('existing_labeled_record' in c['identity_evidence'] for c in candidates),
        'by_role': dict(Counter(c['role'] for c in candidates)),
        'by_qualification': dict(Counter(c['qualification'] for c in candidates)),
        'missing_heightmap': sum(not c['heightmaps'] for c in candidates),
        'heightmap_reference_count': sum(len(c['heightmaps']) for c in candidates),
        'ambiguous_heightmap_performance': sum('ambiguous_heightmap_performance' in c['issues'] for c in candidates),
        'duplicate_group_count': len(duplicate_groups),
        'duplicate_candidate_count': sum(group['candidate_count'] for group in duplicate_groups),
        'duplicate_extra_candidate_count': sum(group['candidate_count'] - 1 for group in duplicate_groups),
    }
    return {'format_version': 1, 'archive_mount': '/expert_reference_data',
            'archive_prefix': ARCHIVE_PREFIX, 'summary': summary,
            'rules': {
                'scenario': 'Filename prefix is authoritative; SQL ScenarioID is separate metadata.',
                'identity': 'Exact spray.result WorkID/OrderID join; all SQL performances retained; no alternate-order date inference.',
                'worker_id': 'SHA-256 namespace of database scope and SQL UserID; unresolved identity is session-scoped.',
                'legacy': 'Names may fill absent SQL names only for the exact process/session record; an exact labeled expert record can retain its verified role.',
                'deduplication': 'Candidates are retained; logical key is process plus canonical session_id; no winner selected.',
                'raw_heightmap': 'All same-process WorkID/OrderID MapNN/PerformanceID PNGs retained, without averaging.',
                'converted_heightmap': 'Last available numeric S sequence per map in the CSV folder; not claimed as a final map.',
                'raw_payload': '.raw files remain ancillary archives; dtype is not inferred.',
                'privacy': 'No raw SQL UserID, emails, credentials, RFID or contacts are emitted.',
            },
            'candidates': candidates, 'excluded_csvs': excluded, 'duplicate_groups': duplicate_groups}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', required=True, type=Path)
    parser.add_argument('--sql-index', required=True, type=Path)
    parser.add_argument('--existing-catalog', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = create_import_plan(args.inventory, args.sql_index, args.existing_catalog)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result['summary'], ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
