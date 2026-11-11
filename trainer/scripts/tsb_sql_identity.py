"""Read selected MySQL dump literals without executing SQL or evaluating code.

The output deliberately excludes RFID, contact fields, credentials and unrelated
dump data. Unknown enum meanings remain unknown rather than becoming labels.
"""
import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path


SELECTED_TABLES = {'user', 'job', 'gun', 'result', 'result_exam', 'scenario', 'plane', 'unit'}
_NUMBER = re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z')
_IDENTIFIER = r'`([^`]+)`'
_INSERT = re.compile(r'INSERT(?:\s+IGNORE)?\s+INTO\s+(?:`([^`]+)`\.)?`([^`]+)`\s*(?:\((.*?)\))?\s+VALUES\s*', re.I | re.S)
_CREATE = re.compile(r'CREATE\s+TABLE(?:\s+IF\s+NOT\s+EXISTS)?\s+(?:`([^`]+)`\.)?`([^`]+)`\s*\(', re.I)
_ESCAPES = {'0': '\0', 'b': '\b', 'n': '\n', 'r': '\r', 't': '\t', 'Z': '\x1a'}


@dataclass
class DumpData:
    rows: dict = field(default_factory=dict)
    schemas: dict = field(default_factory=dict)
    table_counts: dict = field(default_factory=dict)


def parse_mysql_values(text):
    """Yield literal tuples from an INSERT VALUES tail. No SQL expressions."""
    i, n = 0, len(text)

    def whitespace(pos):
        while pos < n and text[pos].isspace():
            pos += 1
        return pos

    while True:
        i = whitespace(i)
        if i >= n or text[i] == ';':
            if text[i:].strip() not in {'', ';'}:
                raise ValueError('Unexpected text after literal tuples')
            return
        if text[i] != '(':
            raise ValueError('Expected a literal tuple')
        i += 1
        row = []
        while True:
            i = whitespace(i)
            if i >= n:
                raise ValueError('Unterminated literal tuple')
            if text[i] in {'\'', '"'}:
                quote = text[i]
                i += 1
                chars = []
                while i < n:
                    char = text[i]
                    i += 1
                    if char == '\\':
                        if i >= n:
                            raise ValueError('Unterminated string escape')
                        escaped = text[i]
                        i += 1
                        chars.append(_ESCAPES.get(escaped, escaped))
                    elif char == quote:
                        if i < n and text[i] == quote:
                            chars.append(quote)
                            i += 1
                        else:
                            break
                    else:
                        chars.append(char)
                else:
                    raise ValueError('Unterminated quoted string')
                value = ''.join(chars)
            else:
                start = i
                while i < n and text[i] not in {',', ')'}:
                    i += 1
                token = text[start:i].strip()
                if token.upper() == 'NULL':
                    value = None
                elif _NUMBER.fullmatch(token):
                    value = float(token) if any(c in token for c in '.eE') else int(token)
                else:
                    # Never echo an untrusted value in errors/logs.
                    raise ValueError('Only numbers, quoted strings and NULL are supported')
            row.append(value)
            i = whitespace(i)
            if i >= n:
                raise ValueError('Unterminated literal tuple')
            if text[i] == ')':
                i += 1
                yield tuple(row)
                break
            if text[i] != ',':
                raise ValueError('Expected a tuple separator')
            i += 1
        i = whitespace(i)
        if i < n and text[i] == ',':
            i += 1
            if whitespace(i) >= n or text[whitespace(i)] != '(':
                raise ValueError('Expected a tuple after row separator')
        elif i < n and text[i] != ';':
            raise ValueError('Expected a row separator')


def _statement_complete(text):
    quote = None
    escaped = False
    for char in text:
        if quote:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == quote:
                quote = None
        elif char in {'\'', '"', '`'}:
            quote = char
        elif char == ';':
            return True
    return False


def read_dump(path, tables=None):
    """Read mysqldump CREATE/INSERT text, keeping database scopes separate."""
    wanted = set(SELECTED_TABLES if tables is None else tables)
    data = DumpData()
    database = 'unspecified'
    create_key = None
    pending = ''
    with Path(path).open('r', encoding='utf-8-sig', errors='strict') as stream:
        for line in stream:
            if pending:
                pending += line
                if not _statement_complete(pending):
                    continue
                statement, pending = pending, ''
            else:
                db_match = re.search(r'^--\s+Host:.*?\bDatabase:\s*(\S+)', line)
                use_match = re.match(r'USE\s+`([^`]+)`\s*;', line, re.I)
                if db_match or use_match:
                    database = (db_match or use_match).group(1)
                    continue
                match = _CREATE.match(line)
                if match:
                    create_key = (match.group(1) or database, match.group(2))
                    data.schemas[create_key] = []
                    continue
                if create_key:
                    column = re.match(r'\s*`([^`]+)`\s+', line)
                    if column:
                        data.schemas[create_key].append(column.group(1))
                    if line.lstrip().startswith(')'):
                        create_key = None
                    continue
                match = _INSERT.match(line)
                if not match or match.group(2) not in wanted:
                    continue
                if not _statement_complete(line):
                    pending = line
                    continue
                statement = line
            match = _INSERT.match(statement)
            if not match:
                raise ValueError('Malformed INSERT header')
            key = (match.group(1) or database, match.group(2))
            columns = re.findall(_IDENTIFIER, match.group(3)) if match.group(3) else data.schemas.get(key)
            if not columns:
                raise ValueError('INSERT table has no known column schema')
            rows = data.rows.setdefault(key, [])
            for values in parse_mysql_values(statement[match.end():]):
                if len(values) != len(columns):
                    raise ValueError(f'Column count mismatch in {key[0]}.{key[1]}')
                rows.append(dict(zip(columns, values)))
            data.table_counts[key] = len(rows)
    if pending or create_key:
        raise ValueError('Incomplete dump statement')
    return data


def build_identity_index(data):
    """Build sanitized result→user/job/gun joins and exact path-stem aliases."""
    records = []
    users = []
    for database in sorted({key[0] for key in data.rows}):
        jobs = {str(r['JobID']): r.get('JobName') for r in data.rows.get((database, 'job'), [])}
        guns = {str(r['GunID']): r for r in data.rows.get((database, 'gun'), [])}
        user_lookup = {str(r['UserID']): r for r in data.rows.get((database, 'user'), [])}
        for user_id, user in user_lookup.items():
            users.append({'database': database, 'user_id': user_id, 'display_name': user.get('UserName'),
                          'job_id': user.get('JobID'), 'job_name': jobs.get(str(user.get('JobID'))),
                          'level_raw': user.get('Level')})
        for table in ('result', 'result_exam'):
            for row in data.rows.get((database, table), []):
                user_id = str(row.get('UserID', ''))
                user = user_lookup.get(user_id, {})
                job_name = jobs.get(str(user.get('JobID')))
                gun = guns.get(str(row.get('GunID')), {})
                qualification, source = _qualification(user, job_name)
                aliases = sorted({_path_identifier(row.get(field)) for field in ('DataPath', 'TexturePath')} - {None})
                work_date = row.get('WorkDate')
                record = {
                    'database': database, 'source_table': table,
                    'work_id': row.get('WorkID'), 'order_id': row.get('OrderID'),
                    'performance_id': row.get('PerformanceID'), 'user_id': user_id,
                    'sql_scenario_id': row.get('ScenarioID'),
                    'display_name': user.get('UserName'), 'work_date': work_date,
                    'acquisition_date': _acquisition_date(work_date),
                    'worker_type_raw': row.get('WorkerType'), 'user_level_raw': user.get('Level'),
                    'job_id': user.get('JobID'), 'job_name': job_name,
                    'gun_id': row.get('GunID'), 'gun_type_raw': gun.get('GunType'),
                    'gun_name': gun.get('GunName'), 'process': _process(gun.get('GunName')),
                    'process_source': 'gun.GunName' if _process(gun.get('GunName')) != 'unknown' else None,
                    'qualification': qualification, 'qualification_source': source,
                    'aliases': aliases,
                    'data_path_alias': _path_identifier(row.get('DataPath')),
                    'texture_path_alias': _path_identifier(row.get('TexturePath')),
                }
                records.append(record)
    by_identifier = {}
    by_work_id = {}
    by_session = {}
    for position, record in enumerate(records):
        for alias in record['aliases']:
            by_identifier.setdefault(alias, []).append(position)
        by_work_id.setdefault(str(record['work_id']), []).append(position)
        session_key = f"{record['database']}|{record['work_id']}|{record['order_id']}"
        by_session.setdefault(session_key, []).append(position)
    return {'format_version': 1, 'users': users, 'records': records,
            'by_identifier': by_identifier, 'by_work_id': by_work_id, 'by_session': by_session}


def resolve_identifier(index, identifier):
    """Return exact path aliases; no identity inference from numeric tokens."""
    key = _path_identifier(identifier)
    return [index['records'][p] for p in index['by_identifier'].get(key, [])]


def resolve_session(index, identifier, database='spray', source_table='result'):
    """Resolve external scenario-label_WorkID_OrderID names, preserving ambiguity.

    The external scenario label is distinct from SQL result.ScenarioID and is
    never a user ID or process code. The join is database/WorkID/OrderID.
    No alternate OrderID or database is used when the requested key is absent.
    """
    name = _path_identifier(identifier)
    if re.fullmatch(r'[^_]+_Export', name or '', re.I):
        return {'status': 'summary_without_session_identifier', 'candidates': [], 'identifier': name}
    match = re.fullmatch(r'([^_]+)_(\d+)_(\d+)(?:(?:_Height|_copy))*', name or '', re.I)
    if not match:
        return {'status': 'invalid_identifier', 'candidates': [], 'identifier': name}
    scenario_label = match.group(1)
    scenario_id = int(scenario_label) if scenario_label.isdigit() else None
    work_id, order_id = int(match.group(2)), int(match.group(3))
    rows = [index['records'][position]
            for position in index['by_session'].get(f'{database}|{work_id}|{order_id}', [])
            if source_table is None or index['records'][position]['source_table'] == source_table]
    result = {'identifier': name, 'database': database, 'scenario_id': scenario_id,
              'scenario_label': scenario_label, 'work_id': work_id, 'order_id': order_id, 'candidates': rows}
    if not rows:
        work_rows = [index['records'][position]
                     for position in index['by_work_id'].get(str(work_id), [])
                     if index['records'][position]['database'] == database and
                     (source_table is None or index['records'][position]['source_table'] == source_table)]
        result['status'] = 'order_not_found' if work_rows else 'work_id_not_found'
        result['available_order_ids'] = sorted({row['order_id'] for row in work_rows})
    else:
        result['status'] = 'matched' if len(rows) == 1 else 'ambiguous_performance'
        result['unknown_fields'] = sorted({field for row in rows
                                          for field in ('display_name', 'acquisition_date') if not row[field]})
    return result


def _path_identifier(value):
    if not value:
        return None
    name = str(value).replace('\\', '/').rsplit('/', 1)[-1]
    return re.sub(r'\.(?:csv|png|jpg|jpeg|txt|dat|replay)$', '', name, flags=re.I) or None


def _acquisition_date(value):
    if not value:
        return None
    match = re.search(r'(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})', str(value))
    if not match:
        return None
    try:
        return date(*(int(x) for x in match.groups())).isoformat()
    except ValueError:
        return None


def _qualification(user, job_name):
    name = re.sub(r'\s+', '', str(user.get('UserName', ''))).casefold()
    explicit_metadata = ' '.join(str(x or '') for x in (job_name, user.get('Department'))).casefold()
    if any(word in explicit_metadata for word in ('외국인', '외국 국적', 'foreign national', 'foreigner')):
        return 'nonexpert', 'user_rule:foreign_national'
    if name in {'전형기', 'jeonhyeonggi', 'jeonhyunggi'}:
        return 'expert', 'user_rule:jeon_hyeonggi'
    label = str(job_name or '').strip().casefold()
    if label in {'비숙련자', '비전문가', '초보자', '비숙련', 'nonexpert', 'non-expert', 'beginner'}:
        return 'nonexpert', 'job.JobName'
    if label in {'숙련자', '전문가', '숙련', 'expert', 'professional'}:
        return 'expert', 'job.JobName'
    return 'unknown', None


def _process(gun_name):
    label = str(gun_name or '').casefold()
    if 'blast' in label or '블라스' in label:
        return 'blasting'
    if 'spray' in label or '스프레' in label or '도장' in label:
        return 'spray'
    if 'grind' in label or '그라인' in label or '연마' in label:
        return 'grinding'
    return 'unknown'


def mapping_report(data, index):
    return {
        'format_version': 1,
        'table_rows': {f'{db}.{table}': count for (db, table), count in sorted(data.table_counts.items())},
        'table_links': ['result/result_exam.UserID = user.UserID within the same database',
                        'user.JobID = job.JobID within the same database',
                        'result/result_exam.GunID = gun.GunID within the same database'],
        'field_meanings': {'work_id': 'result.WorkID; simulator training index',
                           'order_id': 'result.OrderID', 'performance_id': 'result.PerformanceID',
                           'user_id': 'result.UserID joined to user.UserID',
                           'acquisition_date': 'ISO date parsed from result.WorkDate',
                           'aliases': 'exact basename without known extension from result.DataPath/TexturePath'},
        'records': len(index['records']), 'users': len(index['users']),
        'qualification_counts': dict(Counter(r['qualification'] for r in index['records'])),
        'process_counts': dict(Counter(r['process'] for r in index['records'])),
        'records_missing_user_name': sum(not r['display_name'] for r in index['records']),
        'records_missing_acquisition_date': sum(not r['acquisition_date'] for r in index['records']),
        'unknowns': ['WorkerType, Level and numeric GunType enum meanings require external documentation.',
                     'No nationality inferred from a person name.',
                     'The SQL snapshot cannot supply metadata for later or absent WorkID values.'],
        'identity_rule': 'External scenario_WorkID_OrderID → (database=spray, source_table=result, WorkID, OrderID) → result.UserID → user.UserID. Preserve all PerformanceID candidates.',
        'external_filename_fields': ['scenario_label', 'work_id', 'order_id'],
        'scenario_rule': 'External scenario_label is not SQL result.ScenarioID; the latter is retained as sql_scenario_id.',
        'source_path_alias_rule': 'SQL DataPath is WorkID_OrderID_PerformanceID.replay; it is retained separately from the external scenario prefix.',
        'process_rule': 'Use explicit source directory Spray/Blast/Grind for this corpus; numeric filename prefixes are scenario IDs, not process or user IDs.',
        'qualification_rules': ['Explicit foreign-national metadata → nonexpert',
                                '전형기 / Jeon Hyeonggi → expert',
                                'Unambiguous job.JobName qualification labels; otherwise unknown'],
        'sanitization': 'Only display names, source IDs, explicit job/level/worker-type metadata, process and acquisition date retained.'
    }


def source_coverage(index, corpus_root=None, height_root=None):
    """Report read-only filename-to-session coverage without reading CSV data."""
    groups = {}
    sources = []
    if corpus_root:
        root = Path(corpus_root)
        sources.extend((path, root, path.relative_to(root).parts[0]) for path in root.rglob('*.csv'))
    if height_root:
        root = Path(height_root)
        sources.extend((path, root, 'converted_1_Height') for path in root.iterdir()
                       if path.is_dir() and path.name.endswith('1_Height'))
    for path, root, group in sources:
        report = groups.setdefault(group, {'total': 0, 'status_counts': {}, 'missing_name': 0,
                                         'missing_date': 0, 'qualification_counts': {}, 'unmatched_examples': []})
        result = resolve_session(index, path.name)
        report['total'] += 1
        status = result['status']
        report['status_counts'][status] = report['status_counts'].get(status, 0) + 1
        if status != 'matched':
            if len(report['unmatched_examples']) < 12:
                report['unmatched_examples'].append({key: result[key] for key in
                                                    ('identifier', 'status', 'available_order_ids') if key in result})
            continue
        row = result['candidates'][0]
        report['missing_name'] += not bool(row['display_name'])
        report['missing_date'] += not bool(row['acquisition_date'])
        qualification = row['qualification']
        report['qualification_counts'][qualification] = report['qualification_counts'].get(qualification, 0) + 1
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sql_path', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--corpus-root', type=Path)
    parser.add_argument('--height-root', type=Path)
    args = parser.parse_args()
    data = read_dump(args.sql_path)
    index = build_identity_index(data)
    report = mapping_report(data, index)
    if args.corpus_root or args.height_root:
        report['source_coverage'] = source_coverage(index, args.corpus_root, args.height_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, value in [('sql_identity_index.json', index), ('sql_mapping_report.json', report)]:
        (args.output_dir / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'records': report['records'], 'users': report['users'],
                      'outputs': ['sql_identity_index.json', 'sql_mapping_report.json']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
