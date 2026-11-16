"""Vendor localization rows are archived, screened, and never certified wholesale."""
import csv
import hashlib

from trainer.multilingual.tsb_import import build_corpus


HEADER = ['한국어','English','中文','Tiếng Việt','Myanmar','नेपाली',"O'zbek tili",'ภาษาไทย','Indonesia','සිංහල','বাংলা','']


def write_source(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(HEADER)
        writer.writerows(rows)
    return path


def test_import_keeps_original_cells_hash_rows_and_multiline_locations(tmp_path):
    source = write_source(tmp_path/'sample.csv', [['노즐','Nozzle','喷嘴','Vòi phun','','','','','','','',''],
                                                ['긴\n안내','Long\nguide','','','','','','','','','','extra']])
    raw = source.read_bytes()
    result = build_corpus([source], selected_rows={('sample.csv',2):('en','vi')})
    assert source.read_bytes() == raw
    assert result['source_files'][0]['sha256'] == hashlib.sha256(raw).hexdigest()
    first, second = result['rows']
    assert first['row'] == 2 and first['line_start'] == 2 and first['values']['vi'] == 'Vòi phun'
    assert second['values']['ko'] == '긴\n안내' and second['extra_cells'] == ['extra']
    assert second['line_end'] > second['line_start']
    assert 'unexpected_extra_cells' in second['issues']
    assert result['reference_terms'][0]['source']['row'] == 2


def test_screening_separates_missing_conflicting_script_numeric_and_placeholder_cells(tmp_path):
    rows = [
        ['노즐 519','Nozzle 519','喷嘴 519','Vòi phun 519','नेपाली','','','','','තුණ්ඩ 523','',''],
        ['노즐 519','Different nozzle 519','','','','','','','','','',''],
        ['목표 {value}','Target {Value}','','','','','','','','','',''],
        ['도막','Coating Film','','Coating Film','','','','','','','',''],
    ]
    result = build_corpus([write_source(tmp_path/'sample.csv',rows)],
                          selected_rows={('sample.csv',2):('en','my','si'), ('sample.csv',4):('en',), ('sample.csv',5):('vi',)})
    first, _, template, copied = result['rows']
    assert 'conflicting_translation' in first['cell_issues']['en']
    assert 'script_mismatch' in first['cell_issues']['my']
    assert 'numeric_mismatch' in first['cell_issues']['si']
    assert 'empty' in first['cell_issues']['ne']
    assert 'placeholder_mismatch' in template['cell_issues']['en']
    assert 'english_fallback' in copied['cell_issues']['vi']
    assert result['reference_terms'] == []


def test_unknown_source_revisions_are_preserved_but_not_automatically_activated(tmp_path):
    result = build_corpus([write_source(tmp_path/'VR_Language.csv', [['노즐','Nozzle','','','','','','','','','','']])])
    assert len(result['rows']) == 1 and result['reference_terms'] == []
    assert 'source_revision_not_reviewed' in result['source_files'][0]['issues']


def test_parenthesized_identifiers_and_arithmetic_are_case_sensitive_templates(tmp_path):
    rows = [['재작업 ((WorkID)번 작업-(OrderID)회차)',
             'Rework ((WorkID)-(OrderID))','','','','','','((WorkId)-(orderId))',
             '((workid)-(orderid))','((වැඩ කිරීම)-(SPRIDED))','',''],
            ['다음 회차 ((OrderID+1))','Next ((OrderID+1))','','','','','','((orderId+1))','','','','']]
    result = build_corpus([write_source(tmp_path/'sample.csv',rows)],
                          selected_rows={('sample.csv',2):('en','th','id'),('sample.csv',3):('en','th')})
    first, second = result['rows']
    assert first['kind']=='template' and second['kind']=='template'
    assert 'template_text' in first['cell_issues']['en']
    assert 'placeholder_mismatch' not in first['cell_issues']['en']
    assert 'placeholder_mismatch' in first['cell_issues']['th']
    assert 'placeholder_mismatch' in first['cell_issues']['id']
    assert 'placeholder_mismatch' in first['cell_issues']['si']
    assert 'placeholder_mismatch' in second['cell_issues']['th']
    assert result['reference_terms']==[]
