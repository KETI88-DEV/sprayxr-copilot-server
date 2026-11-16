"""Archive TSB localization CSVs and screen a small translation-reference subset.

The CSVs are user-provided simulator UI text, not independently attested coating
standards or training labels. Every original cell remains in the archive; only
explicitly selected source revisions/rows can supply optional translator hints.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping

LANGUAGES = ('ko','en','zh','vi','my','ne','uz','th','id','si','bn')
HEADERS = ('한국어','English','中文','Tiếng Việt','Myanmar','नेपाली',"O'zbek tili",'ภาษาไทย','Indonesia','සිංහල','বাংলা')
SOURCE_HASHES = {
    'VR_Language.csv':'634c52afcce29a13c0f6892b0b95bde9436dc5a82cd380821d26fb042c50865e',
    'ManageIOS_Language.csv':'7b46862b1ed85473efea430ec1325f2070124f46e2a01cc34c2b768b690d48b6',
    'StartIOS_Language.csv':'b1b0abfbca05304f30af393fb1d6f2d3ea2bd206eb6a553150df440f82daf874',
}
REFERENCE_LANGUAGES = ('en','zh','vi','my','ne','uz','th','id')
# Sinhala is not aligned reliably with the Korean rows; Bengali is not an API
# language. Other uncertain cells are archived, not repaired or made mandatory.
LANGUAGE_POLICY = {'si':'quarantined_alignment', 'bn':'unsupported_api_language',
                   'ru':'no_source_column', 'km':'no_source_column'}
_CORE = ('en','zh','vi','id')
SELECTED_ROWS = {
    ('VR_Language.csv',4):_CORE, ('VR_Language.csv',8):_CORE,
    ('VR_Language.csv',10):_CORE, ('VR_Language.csv',24):_CORE,
    ('VR_Language.csv',25):_CORE, ('VR_Language.csv',47):('en','zh','id'),
    ('VR_Language.csv',147):_CORE, ('VR_Language.csv',148):_CORE,
    ('VR_Language.csv',161):_CORE, ('VR_Language.csv',385):REFERENCE_LANGUAGES,
    ('ManageIOS_Language.csv',26):('en','zh','vi','ne','th','id'),
    ('ManageIOS_Language.csv',27):REFERENCE_LANGUAGES,
    ('StartIOS_Language.csv',224):REFERENCE_LANGUAGES,
    ('StartIOS_Language.csv',271):REFERENCE_LANGUAGES,
    ('StartIOS_Language.csv',279):REFERENCE_LANGUAGES,
}
_SCRIPTS = {'ko':r'[가-힣]', 'en':r'[A-Za-z]', 'zh':r'[\u3400-\u9fff]',
            'vi':r'[A-Za-zÀ-ỹ]', 'my':r'[\u1000-\u109f]', 'ne':r'[\u0900-\u097f]',
            'uz':r'[A-Za-z]', 'th':r'[\u0e01-\u0e5b]', 'id':r'[A-Za-z]',
            'si':r'[\u0d80-\u0dff]', 'bn':r'[\u0980-\u09ff]'}
_PLACEHOLDER = re.compile(r'\{[^{}]+\}|%[A-Za-z_][A-Za-z0-9_]*%|%[sdif]|__\w+__|'
                          r'\((?i:[A-Za-z_][A-Za-z0-9_]*ID)(?:\s*[+-]\s*\d+)?\)')
_NUMBER = re.compile(r'[-+−]?\d+(?:\.\d+)?')
_UNIT = re.compile(r'(?<![A-Za-z])(?:mm/s|m/s|mm|cm|µm|μm|um|bar|MPa|kPa|psi)(?![A-Za-z])|[%°]')
_IDENTIFIER = re.compile(r'\b(?:SEQ-[A-Za-z0-9_-]+|[A-Za-z0-9-]+_\d+_\d+)\b')
_DOMAIN = re.compile(r'도막|노즐|스프레이|블라스|그라인|이격|분사|표면|압력|유량|도포|겹침')


def normalized(text: str) -> str:
    return re.sub(r'\s+','',unicodedata.normalize('NFKC',text)).casefold()


def _cell_issues(source: str, target: str, english: str, language: str) -> list[str]:
    issues = []
    if not target.strip():
        return ['empty']
    if language in LANGUAGE_POLICY:
        issues.append(LANGUAGE_POLICY[language])
    if language != 'en' and normalized(target)==normalized(english) and re.search(r'[A-Za-z]{3}',english):
        issues.append('english_fallback')
    if any(c.isalpha() for c in target) and not re.search(_SCRIPTS[language],target):
        issues.append('script_mismatch')
    if _NUMBER.findall(unicodedata.normalize('NFKC',source)) != _NUMBER.findall(unicodedata.normalize('NFKC',target)):
        issues.append('numeric_mismatch')
    if Counter(_UNIT.findall(source.replace('μm','µm'))) != Counter(_UNIT.findall(target.replace('μm','µm'))):
        issues.append('unit_mismatch')
    if _PLACEHOLDER.findall(source) != _PLACEHOLDER.findall(target):
        issues.append('placeholder_mismatch')
    if _IDENTIFIER.findall(source) != _IDENTIFIER.findall(target):
        issues.append('identifier_mismatch')
    if _PLACEHOLDER.search(source) or _PLACEHOLDER.search(target):
        issues.append('template_text')
    if re.search(r'/n/|[\r\n]|<[^>]+>',source+target):
        issues.append('layout_markup')
    return list(dict.fromkeys(issues))


def build_corpus(paths: Iterable[Path], *, selected_rows: Mapping[tuple[str,int],Iterable[str]] | None = None) -> dict:
    explicit_selection = selected_rows is not None
    selection = SELECTED_ROWS if selected_rows is None else selected_rows
    files, rows = [], []
    for path in paths:
        path = Path(path)
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        reader = csv.reader(io.StringIO(raw.decode('utf-8-sig'),newline=''))
        header = next(reader)
        if any(name not in header for name in HEADERS):
            raise ValueError(f'{path.name}: missing required language columns')
        indexes = {language:header.index(name) for language,name in zip(LANGUAGES,HEADERS)}
        extra_indexes = [i for i in range(len(header)) if i not in indexes.values()]
        reviewed = SOURCE_HASHES.get(path.name)==sha
        file_info = {'file':path.name,'sha256':sha,'bytes':len(raw),'encoding':'utf-8-sig',
                     'header':header,'issues':[] if reviewed else ['source_revision_not_reviewed'], 'row_count':0}
        files.append(file_info)
        previous = reader.line_num
        for row_number,cells in enumerate(reader,2):
            line_start,previous = previous+1,reader.line_num
            values = {language:cells[i] if i<len(cells) else '' for language,i in indexes.items()}
            extra = [cells[i] if i<len(cells) else '' for i in extra_indexes] + cells[len(header):]
            issues = []
            if len(cells)!=len(header):
                issues.append('column_count_mismatch')
            if any(value.strip() for value in extra):
                issues.append('unexpected_extra_cells')
            ko = values['ko']
            kind = ('template' if _PLACEHOLDER.search(ko) else 'guidance' if len(ko)>40 or re.search(r'/n/|[\r\n]|[.!?。]$',ko)
                    else 'domain_label' if _DOMAIN.search(ko) else 'ui_label')
            row = {'file':path.name,'row':row_number,'line_start':line_start,'line_end':reader.line_num,
                   'cells':cells,'extra_cells':extra,'values':values,'kind':kind,'issues':issues,
                   'cell_issues':{language:_cell_issues(ko,values[language],values['en'],language) for language in LANGUAGES[1:]},
                   'source_sha256':sha,'source_reviewed':reviewed}
            if not ko.strip():
                row['issues'].append('empty_korean')
            rows.append(row)
            file_info['row_count'] += 1
    groups = defaultdict(list)
    for row in rows:
        if normalized(row['values']['ko']):
            groups[normalized(row['values']['ko'])].append(row)
    for group in groups.values():
        if len(group)>1:
            for row in group:
                row['issues'].append('duplicate_korean_key')
            for language in LANGUAGES[1:]:
                variants = {normalized(row['values'][language]) for row in group if row['values'][language].strip()}
                if len(variants)>1:
                    for row in group:
                        row['cell_issues'][language].append('conflicting_translation')
    terms = []
    for row in rows:
        ko = row['values']['ko'].strip()
        languages = selection.get((row['file'],row['row']),()) if explicit_selection or row['source_reviewed'] else ()
        if not languages or row['kind']!='domain_label' or any(issue!='duplicate_korean_key' for issue in row['issues']):
            continue
        translations = {language:row['values'][language].strip() for language in languages
                        if language in REFERENCE_LANGUAGES and not row['cell_issues'][language]
                        and 0<len(row['values'][language].strip())<=120}
        if translations:
            terms.append({'ko':ko,'translations':translations,'status':'user_provided_reference',
                          'source':{'file':row['file'],'sha256':row['source_sha256'],'row':row['row'],
                                    'line_start':row['line_start'],'line_end':row['line_end']}})
    digest = hashlib.sha256(json.dumps({'sources':[(f['file'],f['sha256']) for f in files], 'terms':terms},
                                       ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    return {'schema_version':1,'version':'tsb-localization-20261116-'+digest[:16],
            'purpose':'optional_translation_reference_not_model_training_or_technical_evidence',
            'source_directory':'D:/KETI/2026/선박 도장 스프레이/다국어 데이터',
            'language_policy':LANGUAGE_POLICY, 'source_files':files,'rows':rows,'reference_terms':terms,
            'summary':{'row_count':len(rows),'kind_counts':dict(Counter(r['kind'] for r in rows)),
                       'row_issue_counts':dict(Counter(issue for r in rows for issue in r['issues'])),
                       'cell_issue_counts':dict(Counter(issue for r in rows for issues in r['cell_issues'].values() for issue in issues)),
                       'reference_term_count':len(terms),
                       'reference_pairs_by_language':dict(Counter(lang for term in terms for lang in term['translations']))}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path(__file__).with_name('tsb_reference.json'))
    options = parser.parse_args()
    data = build_corpus([options.input_dir / name for name in SOURCE_HASHES])
    options.output.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'version':data['version'],**data['summary']},ensure_ascii=False))


if __name__=='__main__':
    main()
