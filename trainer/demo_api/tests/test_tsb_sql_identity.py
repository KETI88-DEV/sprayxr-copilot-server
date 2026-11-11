import tempfile
import unittest
import json
from pathlib import Path

from trainer.scripts import tsb_sql_identity as sqlmap


class MysqlValuesTests(unittest.TestCase):
    def test_escaped_strings_do_not_split_tuples_or_execute_text(self):
        # Breaking quote/escape handling would split the text or change its value.
        values = r"(1,'O\'Brien, (test);','C:\\data\\x','line\nnext','it''s'),(2,NULL,'NULL','','x')"
        self.assertEqual(list(sqlmap.parse_mysql_values(values)), [
            (1, "O'Brien, (test);", r"C:\data\x", "line\nnext", "it's"),
            (2, None, 'NULL', '', 'x'),
        ])

    def test_multiple_insert_tuples_keep_numbers_and_null_distinct(self):
        self.assertEqual(list(sqlmap.parse_mysql_values("(1,-2,1.25,NULL),(3,0,2e2,'NULL');")), [
            (1, -2, 1.25, None), (3, 0, 200.0, 'NULL')])

    def test_untrusted_expressions_are_rejected(self):
        with self.assertRaises(ValueError):
            list(sqlmap.parse_mysql_values("(1,LOAD_FILE('/private'))"))

    def test_trailing_separator_and_truncated_strings_are_rejected(self):
        for value in ["(1),", "(1,'broken)", "(1);DROP TABLE user;"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                list(sqlmap.parse_mysql_values(value))


class DumpLinkTests(unittest.TestCase):
    def parse(self, text):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'input.sql'
            path.write_text(text, encoding='utf-8')
            return sqlmap.read_dump(path)

    def test_database_scopes_and_explicit_insert_columns_are_retained(self):
        data = self.parse("""-- Host: localhost    Database: spray_exam
CREATE TABLE `user` (
  `UserID` varchar(20),
  `UserName` varchar(20),
  PRIMARY KEY (`UserID`)
);
INSERT INTO `user` VALUES ('u','First');
-- Host: localhost    Database: spray
CREATE TABLE `user` (
  `UserID` varchar(20),
  `UserName` varchar(20)
);
INSERT INTO `user` (`UserName`,`UserID`) VALUES ('Second','u');
""")
        self.assertEqual(data.rows[('spray_exam', 'user')], [{'UserID': 'u', 'UserName': 'First'}])
        self.assertEqual(data.rows[('spray', 'user')], [{'UserName': 'Second', 'UserID': 'u'}])

    def test_identifier_joins_result_to_user_without_treating_workid_as_userid(self):
        data = sqlmap.DumpData(rows={
            ('spray', 'user'): [{'UserID': 'operator-A', 'UserName': '전형기', 'JobID': 4}],
            ('spray', 'job'): [{'JobID': 4, 'JobName': '교육생'}],
            ('spray', 'gun'): [{'GunID': 2, 'GunType': 3, 'GunName': '블라스팅'}],
            ('spray', 'result'): [{'WorkID': 11000346, 'OrderID': 1, 'PerformanceID': 1,
                'UserID': 'operator-A', 'WorkDate': '2026-11-09 10:02:03',
                'DataPath': 'Data/3_11000346_1.csv', 'GunID': 2, 'WorkerType': 0}],
        })
        index = sqlmap.build_identity_index(data)
        match = sqlmap.resolve_identifier(index, '3_11000346_1')
        self.assertEqual(len(match), 1)
        self.assertEqual(match[0]['user_id'], 'operator-A')
        self.assertEqual(match[0]['display_name'], '전형기')
        self.assertEqual(match[0]['work_id'], 11000346)
        self.assertEqual(match[0]['acquisition_date'], '2026-11-09')
        self.assertEqual(match[0]['qualification'], 'expert')

    def test_unknown_qualification_stays_unknown_and_foreign_rule_overrides(self):
        data = sqlmap.DumpData(rows={
            ('spray', 'user'): [
                {'UserID': 'a', 'UserName': '국내작업자', 'JobID': 1},
                {'UserID': 'b', 'UserName': 'Foreign worker', 'JobID': 2},
            ],
            ('spray', 'job'): [{'JobID': 1, 'JobName': '교육생'}, {'JobID': 2, 'JobName': '외국인'}],
            ('spray', 'result'): [
                {'WorkID': 12, 'OrderID': 1, 'PerformanceID': 1, 'UserID': 'a', 'WorkerType': 0},
                {'WorkID': 13, 'OrderID': 1, 'PerformanceID': 1, 'UserID': 'b', 'WorkerType': 1},
            ],
        })
        index = sqlmap.build_identity_index(data)
        self.assertEqual(index['records'][0]['qualification'], 'unknown')
        self.assertEqual(index['records'][1]['qualification'], 'nonexpert')
        self.assertEqual(index['records'][1]['qualification_source'], 'user_rule:foreign_national')

    def test_external_identifier_uses_session_key_and_retains_performance_ambiguity(self):
        data = sqlmap.DumpData(rows={('spray', 'result'): [
            {'WorkID': 42, 'OrderID': 1, 'PerformanceID': 1, 'UserID': 'operator'},
            {'WorkID': 42, 'OrderID': 1, 'PerformanceID': 2, 'UserID': 'operator'},
        ], ('spray_exam', 'result'): [
            {'WorkID': 42, 'OrderID': 1, 'PerformanceID': 1, 'UserID': 'different'},
        ]})
        index = sqlmap.build_identity_index(data)
        match = sqlmap.resolve_session(index, '9_42_1_Height_copy')
        self.assertEqual(match['status'], 'ambiguous_performance')
        self.assertEqual(match['scenario_id'], 9)
        self.assertEqual([r['performance_id'] for r in match['candidates']], [1, 2])
        self.assertEqual({r['user_id'] for r in match['candidates']}, {'operator'})

    def test_missing_order_never_falls_back_to_another_order(self):
        data = sqlmap.DumpData(rows={('spray', 'result'): [
            {'WorkID': 1000364, 'OrderID': 0, 'PerformanceID': 1, 'UserID': ''},
        ]})
        result = sqlmap.resolve_session(sqlmap.build_identity_index(data), '1_1000364_1_Height')
        self.assertEqual(result['status'], 'order_not_found')
        self.assertEqual(result['candidates'], [])
        self.assertEqual(result['available_order_ids'], [0])

    def test_nonnumeric_scenario_labels_and_copy_suffixes_keep_the_same_session_join(self):
        index = sqlmap.build_identity_index(sqlmap.DumpData(rows={('spray', 'result'): [
            {'WorkID': 42, 'OrderID': 1, 'PerformanceID': 1, 'ScenarioID': 0, 'UserID': 'u'},
        ]}))
        for name, label in [('Unit7_42_1.csv', 'Unit7'), ('1-ABC_42_1_copy.csv', '1-ABC')]:
            with self.subTest(name=name):
                result = sqlmap.resolve_session(index, name)
                self.assertEqual(result['status'], 'matched')
                self.assertEqual(result['scenario_label'], label)
                self.assertEqual(result['candidates'][0]['sql_scenario_id'], 0)

    def test_export_summary_is_distinguished_from_missing_session(self):
        result = sqlmap.resolve_session(sqlmap.build_identity_index(sqlmap.DumpData()), '1_Export.csv')
        self.assertEqual(result['status'], 'summary_without_session_identifier')

    def test_index_does_not_export_unnecessary_private_fields(self):
        index = sqlmap.build_identity_index(sqlmap.DumpData(rows={('spray', 'user'): [
            {'UserID': 'u', 'UserName': '작업자', 'RFID': 'private-rfid',
             'Email': 'private-email', 'Password': 'private-password', 'Department': 'private-department'},
        ]}))
        encoded = json.dumps(index, ensure_ascii=False)
        for private_value in ['private-rfid', 'private-email', 'private-password', 'private-department']:
            self.assertNotIn(private_value, encoded)
        self.assertEqual(index['users'][0]['display_name'], '작업자')

    def test_path_aliases_ignore_empty_basenames_and_normalize_replay_extensions(self):
        data = sqlmap.DumpData(rows={('spray', 'result'): [
            {'WorkID': 42, 'OrderID': 1, 'PerformanceID': 1, 'UserID': 'u',
             'DataPath': 'Data/42_1_1.replay', 'TexturePath': 'Texture/'},
        ]})
        index = sqlmap.build_identity_index(data)
        self.assertEqual(index['records'][0]['aliases'], ['42_1_1'])
        self.assertNotIn('', index['by_identifier'])


if __name__ == '__main__':
    unittest.main()

