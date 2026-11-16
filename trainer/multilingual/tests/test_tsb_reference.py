"""Only the reviewed subset is offered as optional translation context."""
from trainer.multilingual import tsb_reference as reference


def corpus():
    return {'version':'test-version', 'reference_terms':[
        {'ko':'노즐', 'translations':{'en':'Nozzle','vi':'Vòi phun','si':'unsafe','bn':'বাঙালি'},
         'source':{'file':'sample.csv','sha256':'abc','row':2}},
        {'ko':'도막', 'translations':{'en':'Coating Film','vi':'Màng sơn'},
         'source':{'file':'sample.csv','sha256':'abc','row':3}},
    ]}


def test_reference_matches_in_both_directions_with_provenance_and_a_limit(monkeypatch):
    monkeypatch.setattr(reference, 'load_corpus', corpus)
    pairs = reference.reference_hints('노즐을 확인하세요.', 'ko','vi')
    assert len(pairs)==1 and pairs[0]['source']=='노즐' and pairs[0]['target']=='Vòi phun'
    assert pairs[0]['provenance']['row']==2 and pairs[0]['status']=='user_provided_reference'
    assert reference.reference_hints('Vòi phun có vấn đề.', 'vi','ko')[0]['target']=='노즐'
    assert len(reference.reference_hints('도막과 노즐', 'ko','en', limit=1))==1
    assert reference.reference_hints('일반 인사말', 'ko','en')==[]


def test_existing_attested_terms_take_priority_and_latin_substrings_do_not_match(monkeypatch):
    monkeypatch.setattr(reference, 'load_corpus', corpus)
    assert reference.reference_hints('건도막 두께', 'ko','en',
        protected_pairs=[{'source':'건도막 두께','target':'dry film thickness'}])==[]
    assert reference.reference_hints('nozzletter', 'en','ko')==[]


def test_deferred_languages_are_never_promoted_even_if_a_corpus_contains_them(monkeypatch):
    monkeypatch.setattr(reference, 'load_corpus', corpus)
    for language in ('si','bn','ru','km'):
        assert reference.reference_hints('노즐', 'ko',language)==[]
        assert reference.reference_hints('unsafe', language,'ko')==[]


def test_packaged_subset_has_real_provenance_without_ui_guidance_or_quarantined_cells():
    data = reference.load_corpus()
    assert data['summary']['row_count']==1004 and len(data['source_files'])==3
    assert data['reference_terms']
    source_hashes={item['file']:item['sha256'] for item in data['source_files']}
    for term in data['reference_terms']:
        assert term['source']['sha256']==source_hashes[term['source']['file']]
        assert term['source']['row']>=2 and len(term['ko'])<=40
        assert not {'si','bn'} & term['translations'].keys()
        assert not any(token in term['ko'] for token in ('/n/','\n','{','<'))


def test_packaged_dynamic_work_and_order_templates_remain_quarantined():
    rows = {row['row']:row for row in reference.load_corpus()['rows'] if row['file']=='StartIOS_Language.csv'}
    for number in (2,3,9,10):
        assert rows[number]['kind']=='template'
        assert 'placeholder_mismatch' in rows[number]['cell_issues']['th']
        assert 'placeholder_mismatch' in rows[number]['cell_issues']['id']
