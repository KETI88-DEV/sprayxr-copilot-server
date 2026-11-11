"""Local, synthetic-fixture tests: no corpus payload or network access."""
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from trainer.scripts.tsb_corpus_plan import create_import_plan


class ImportPlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inventory = self.root / 'inventory.sqlite'
        self.index_path = self.root / 'identity.json'
        self.catalog_path = self.root / 'catalog.json'
        self.rows = []
        self.records = []
        self.existing = []

    def source(self, relative, kind='raw', **overrides):
        root = 'E:\\corpus' if kind == 'raw' else 'C:\\Sequences'
        row = dict(id=len(self.rows) + 1, path=root + '\\' + relative,
                   root=root, source_kind=kind, relative_path=relative,
                   bytes=100, mtime_ns=123,
                   extension=Path(relative).suffix.lower(),
                   filename=relative.replace('\\', '/').rsplit('/', 1)[-1])
        row.update(overrides)
        self.rows.append(row)
        return row

    def sql(self, work, order=1, **overrides):
        row = dict(database='spray', source_table='result', work_id=work,
                   order_id=order, performance_id=1, user_id='shared-account',
                   sql_scenario_id=99, display_name='작업자',
                   acquisition_date='2026-09-02', qualification='unknown',
                   process='unknown', aliases=[])
        row.update(overrides)
        self.records.append(row)
        return row

    def plan(self):
        with sqlite3.connect(self.inventory) as con:
            con.execute('CREATE TABLE files (id INTEGER, path TEXT, root TEXT, '
                        'source_kind TEXT, relative_path TEXT, bytes INTEGER, '
                        'mtime_ns INTEGER, extension TEXT, filename TEXT)')
            for row in self.rows:
                con.execute('INSERT INTO files VALUES (?,?,?,?,?,?,?,?,?)', tuple(row.values()))
        con.close()
        index = dict(records=self.records, by_session={}, by_work_id={}, by_identifier={})
        for n, row in enumerate(self.records):
            key = f"{row['database']}|{row['work_id']}|{row['order_id']}"
            index['by_session'].setdefault(key, []).append(n)
            index['by_work_id'].setdefault(str(row['work_id']), []).append(n)
        self.index_path.write_text(json.dumps(index), encoding='utf-8')
        self.catalog_path.write_text(json.dumps({'records': self.existing}), encoding='utf-8')
        return create_import_plan(self.inventory, self.index_path, self.catalog_path)

    def test_filename_scenario_and_process_are_authoritative(self):
        for scenario in ('1-A', '1-ABC', '1E-A', 'Unit7', '3'):
            self.source(f'Blast\\Sequences\\{scenario}_10_1.csv')
        self.sql(10, sql_scenario_id=8, process='spray')
        result = self.plan()
        by_scenario = {c['scenario_id']: c for c in result['candidates']}
        self.assertEqual(set(by_scenario), {'1-A', '1-ABC', '1E-A', 'Unit7', '3'})
        self.assertEqual(by_scenario['1-A']['process'], 'blasting')
        self.assertEqual(by_scenario['3']['content_type'], 'learning')
        self.assertEqual(by_scenario['Unit7']['content_type'], 'practice/unit')
        self.assertEqual(by_scenario['3']['source_sql']['sql_scenario_id'], 8)

    def test_process_collisions_and_source_variants_are_not_deduplicated(self):
        self.source('Blast\\Sequences\\1_10_1.csv')
        self.source('Spray\\Sequences\\1_10_1.csv')
        self.source('1_10_1_Height\\1_10_1_Height.csv', 'converted_height')
        self.source('Spray\\Sequences\\1_10_1_copy.csv.gz')
        result = self.plan()
        self.assertEqual(len(result['candidates']), 4)
        keys = [c['logical_record_key'] for c in result['candidates']]
        self.assertEqual(keys.count('spray:1_10_1'), 3)
        self.assertEqual(keys.count('blasting:1_10_1'), 1)
        self.assertEqual(result['summary']['duplicate_group_count'], 1)
        self.assertEqual(len({c['candidate_id'] for c in result['candidates']}), 4)

    def test_export_and_raw_payloads_are_ancillary(self):
        self.source('Blast\\TextureFiles\\10\\1_Export.csv')
        self.source('Spray\\Sequences\\1_10_1.raw')
        self.source('Spray\\Sequences\\bad.csv')
        result = self.plan()
        self.assertEqual(result['candidates'], [])
        self.assertEqual(result['summary']['excluded_csv_reasons'],
                         {'summary_without_session_identifier': 1, 'invalid_identifier': 1})
        self.assertEqual(result['summary']['ancillary_file_count'], 3)
        self.assertEqual(result['summary']['archive_file_count'], 3)

    def test_missing_name_does_not_expose_private_user_id(self):
        self.source('Spray\\Sequences\\1_10_1.csv')
        self.source('Spray\\Sequences\\1_11_1.csv')
        self.sql(10, display_name=None, user_id='private-email@example.com')
        self.sql(11, display_name=None, user_id='private-email@example.com')
        result = self.plan()
        a, b = result['candidates']
        self.assertEqual(a['worker_id'], b['worker_id'])
        self.assertTrue(a['worker_name'].startswith('이름 미확인 · ID'))
        self.assertNotIn('private-email@example.com', json.dumps(result))
        self.assertEqual(a['role'], 'worker')

    def test_legacy_name_is_exact_record_evidence_not_shared_account_mapping(self):
        self.source('Spray\\Sequences\\1_10_1.csv')
        self.source('Spray\\Sequences\\1_11_1.csv')
        self.source('Blast\\Sequences\\1_10_1.csv')
        self.sql(10, display_name=None)
        self.sql(11, display_name=None)
        self.existing = [dict(dataset_name='1_10_1', process='spray',
                              worker_name='전형기', worker_id='old_expert', role='expert')]
        candidates = {c['logical_record_key']: c for c in self.plan()['candidates']}
        self.assertEqual(candidates['spray:1_10_1']['role'], 'expert')
        self.assertEqual(candidates['spray:1_10_1']['identity_evidence'], ['existing_labeled_record'])
        self.assertEqual(candidates['spray:1_11_1']['role'], 'worker')
        self.assertEqual(candidates['blasting:1_10_1']['role'], 'worker')

    def test_sql_expert_job_does_not_promote_an_unverified_person(self):
        self.source('Spray\\Sequences\\1_10_1.csv')
        self.sql(10, qualification='expert')
        candidate = self.plan()['candidates'][0]
        self.assertEqual(candidate['role'], 'worker')
        self.assertEqual(candidate['qualification'], 'unknown')
        self.assertEqual(candidate['source_sql']['qualification'], 'expert')

    def test_exact_legacy_expert_role_is_record_evidence_with_present_sql_name(self):
        self.source('Spray\\Sequences\\1_10_1.csv')
        self.sql(10, display_name='확인된 전문가')
        self.existing = [dict(dataset_name='1_10_1', process='spray',
                              worker_name='확인된 전문가', role='expert')]
        candidate = self.plan()['candidates'][0]
        self.assertEqual(candidate['role'], 'expert')
        self.assertIn('existing_labeled_record', candidate['identity_evidence'])

    def test_deterministic_ids_do_not_depend_on_inventory_row_id_or_mtime(self):
        self.source('Spray\\Sequences\\1_10_1.csv')
        first = self.plan()
        with sqlite3.connect(self.inventory) as con:
            con.execute('UPDATE files SET id=99,mtime_ns=999')
        con.close()
        second = create_import_plan(self.inventory, self.index_path, self.catalog_path)
        self.assertEqual(first['candidates'][0]['candidate_id'], second['candidates'][0]['candidate_id'])
        self.assertEqual(first['candidates'][0]['worker_id'], second['candidates'][0]['worker_id'])

    def test_ambiguous_performance_retains_common_fields_and_all_ids(self):
        self.source('Spray\\Sequences\\1_10_1.csv')
        self.sql(10, performance_id=2)
        self.sql(10, performance_id=3)
        candidate = self.plan()['candidates'][0]
        self.assertEqual(candidate['worker_name'], '작업자')
        self.assertEqual(candidate['acquired_at'], '2026-09-02')
        self.assertEqual(candidate['source_sql']['performance_id_candidates'], [2, 3])
        self.assertIn('ambiguous_performance', candidate['issues'])

    def test_conflicting_fields_and_missing_order_dates_remain_unknown(self):
        self.source('Spray\\Sequences\\1_10_1.csv')
        self.source('Spray\\Sequences\\1_10_9.csv')
        self.sql(10, performance_id=2)
        self.sql(10, performance_id=3, user_id='different', display_name='다른사람',
                 acquisition_date='2026-10-01')
        candidates = {c['session_id']: c for c in self.plan()['candidates']}
        self.assertIsNone(candidates['1_10_1']['acquired_at'])
        self.assertTrue(candidates['1_10_1']['worker_name'].startswith('이름 미확인'))
        self.assertIsNone(candidates['1_10_9']['acquired_at'])
        self.assertEqual(candidates['1_10_9']['source_sql']['match_status'], 'order_not_found')

    def test_unsafe_relative_or_mismatched_absolute_path_fails_closed(self):
        self.source('Spray\\Sequences\\..\\..\\outside.csv')
        with self.assertRaisesRegex(ValueError, 'unsafe|contain|relative'):
            self.plan()

    def test_absolute_path_must_agree_with_root_and_relative_path(self):
        self.source('Spray\\Sequences\\1_10_1.csv', path='E:\\outside\\1_10_1.csv')
        with self.assertRaisesRegex(ValueError, 'contain|agree|root'):
            self.plan()

    def test_raw_heightmaps_are_scoped_by_process_work_order_and_performance(self):
        self.source('Blast\\Sequences\\1_10_1.csv')
        self.source('Spray\\Sequences\\1_10_1.csv')
        for performance in (2, 3):
            self.source(f'Blast\\TextureFiles\\10\\Map01_10_1_{performance}_Height.png')
        self.source('Blast\\TextureFiles\\10\\Map01_10_2_2_Height.png')
        self.source('Spray\\TextureFiles\\10\\Map01_10_1_7_Height.png')
        self.source('Blast\\TextureFiles\\10\\Map01_10_1_2_Height.raw')
        result = self.plan()
        by_process = {c['process']: c for c in result['candidates']}
        self.assertEqual([m['performance_id'] for m in by_process['blasting']['heightmaps']], [2, 3])
        self.assertEqual([m['performance_id'] for m in by_process['spray']['heightmaps']], [7])
        self.assertIn('ambiguous_heightmap_performance', by_process['blasting']['issues'])
        self.assertEqual(by_process['blasting']['heightmaps'][0]['source_inventory_id'], 3)
        self.assertEqual(by_process['blasting']['heightmaps'][0]['bytes'], 100)
        self.assertTrue(by_process['blasting']['heightmaps'][0]['relative_path'].startswith(
            'imports/tsb-20261111/archive/raw/Blast/'))

    def test_converted_uses_last_numeric_sequence_per_map_in_same_folder(self):
        self.source('1_10_1_Height\\1_10_1_Height.csv', 'converted_height')
        for name in ('1_10_1_S2_Map01_Height.png', '1_10_1_S10_Map01_Height.png',
                     '1_10_1_S03_Map02_Height.png', '1_10_1_S03_Map02_Height.raw'):
            self.source('1_10_1_Height\\' + name, 'converted_height')
        self.source('1_11_1_Height\\1_10_1_S99_Map01_Height.png', 'converted_height')
        candidate = self.plan()['candidates'][0]
        self.assertEqual(candidate['sequence_sidecar_count'], 3)
        self.assertEqual(candidate['sequence_raw_sidecar_count'], 1)
        self.assertEqual([(m['map_num'], m['snapshot_sequence_index'])
                          for m in candidate['heightmaps']], [(1, 10), (2, 3)])
        self.assertEqual(candidate['heightmaps'][0]['semantics'], 'last_available_sequence_snapshot')
        self.assertEqual(candidate['heightmaps'][0]['source_inventory_id'], 3)
        self.assertEqual(candidate['heightmaps'][0]['bytes'], 100)

    def test_converted_accepts_bare_sequence_names_scoped_to_session_folder(self):
        self.source('1_10_1_Height\\1_10_1_Height.csv', 'converted_height')
        self.source('1_10_1_Height\\S01_Map01_Height.png', 'converted_height')
        self.source('1_10_1_Height\\S02_Map01_Height.png', 'converted_height')
        candidate = self.plan()['candidates'][0]
        self.assertEqual(candidate['sequence_sidecar_count'], 2)
        self.assertEqual(candidate['heightmaps'][0]['snapshot_sequence_index'], 2)


if __name__ == '__main__':
    unittest.main()
