import base64,json
import pytest
from fastapi.testclient import TestClient
from trainer.demo_api.app import create_app


@pytest.fixture(autouse=True)
def evidence_policies(monkeypatch,tmp_path):
    """Synthetic review policy, independent of corrected production chunk IDs."""
    from trainer.demo_api import knowledge_library
    exclusions=tmp_path/'exclusions.json'
    exclusions.write_text(json.dumps({'조각':{'DOC-doc_0008-C0006':'사실 오류: PRIVATE_AUDIT_MARKER'}}),encoding='utf-8')
    corrections=tmp_path/'corrections.json'
    corrections.write_text(json.dumps({'조각':{}}),encoding='utf-8')
    monkeypatch.setattr(knowledge_library,'_POLICY_PATH',exclusions)
    monkeypatch.setattr(knowledge_library,'_CORRECTIONS_PATH',corrections,raising=False)

def client_for(monkeypatch,tmp_path):
    monkeypatch.setenv('DEMO_LIVE_MODE','off')
    monkeypatch.setenv('SEQUENCE_DB_PATH',str(tmp_path/'portal.db'))
    monkeypatch.setenv('TRAINING_DATA_ROOT',str(tmp_path/'datasets'))
    monkeypatch.setenv('EXPERT_REFERENCE_ROOT',str(tmp_path/'expert'))
    return TestClient(create_app())

def test_home_is_the_default_landing_page_and_lists_historical_releases(monkeypatch,tmp_path):
    client=client_for(monkeypatch,tmp_path)
    response=client.get('/',follow_redirects=False)
    assert response.status_code in (302,303,307,308)
    assert response.headers['location']=='/home'
    home=client.get('/home')
    assert home.status_code==200 and '업데이트' in home.text
    assert home.text.index('data-portal-link="home"')<home.text.index('data-portal-link="docs"')
    updates=client.get('/home/api/updates').json()['updates']
    assert len(updates)>=38
    assert any(row['version']=='v3.11.0' and row['date'] for row in updates)
    detail=client.get('/home/api/updates/v3.11.0')
    assert detail.status_code==200 and detail.json()['body']

def test_home_counts_registered_data_without_exposing_internal_addresses(monkeypatch,tmp_path):
    root=tmp_path/'datasets';root.mkdir()
    (root/'catalog.json').write_text(json.dumps({'records':[
        {'worker_id':'known','worker_name':'전형기','quality_status':'complete'},
        {'worker_id':'a','worker_name':'이름 미확인 · a','quality_status':'needs_review'},
        {'worker_id':'b','worker_name':'이름 미확인 · b','quality_status':'needs_review'}]}),encoding='utf-8')
    data=client_for(monkeypatch,tmp_path).get('/home/api/summary').json()
    assert data['training']['total']==3 and data['training']['complete']==1
    assert data['training']['needs_review']==2 and data['training']['known_workers']==1
    raw=json.dumps(data)
    assert '192.168.' not in raw and '172.18.' not in raw and 'password' not in raw

def library_client(monkeypatch,tmp_path):
    metadata=tmp_path/'metadata.json'
    metadata.write_text(json.dumps({'id_map':[
        {'chunk_id':'DOC-doc_0001-C0001','doc_id':'doc_0001','text':'도장 전 표면 준비를 확인합니다.','source_type':'pdf','domain_category':'coating'},
        {'chunk_id':'DOC-doc_0001-C0002','doc_id':'doc_0001','text':'작업 전 안전을 점검합니다.','source_type':'pdf','domain_category':'coating'}]},ensure_ascii=False),encoding='utf-8')
    config=tmp_path/'library.yaml';config.write_text('paths:\n  metadata_json: '+str(metadata).replace('\\','/')+'\n',encoding='utf-8')
    monkeypatch.setenv('CONFIG_PATH',str(config))
    return client_for(monkeypatch,tmp_path)

def test_knowledge_library_uses_registered_text_and_citation_ids(monkeypatch,tmp_path):
    client=library_client(monkeypatch,tmp_path)
    response=client.get('/query-chat/api/knowledge/documents?q=안전')
    assert response.status_code==200
    docs=response.json()['documents'];assert len(docs)==1 and docs[0]['chunk_count']==2
    assert docs[0]['original_available'] is False and docs[0]['image_count']==0
    detail=client.get('/query-chat/api/knowledge/documents/doc_0001').json()
    assert detail['chunks'][0]['text']=='도장 전 표면 준비를 확인합니다.'
    assert detail['chunks'][0]['chunk_id']=='DOC-doc_0001-C0001'
    assert client.get('/query-chat/api/knowledge/chunks/DOC-doc_0001-C0002').json()['text']=='작업 전 안전을 점검합니다.'
    assert client.get('/query-chat/api/knowledge/documents/not-registered').status_code==404

def test_registered_images_are_supported_and_path_traversal_is_not_served(monkeypatch,tmp_path):
    media=tmp_path/'expert'/'knowledge_library';media.mkdir(parents=True)
    image=base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a7t8AAAAASUVORK5CYII=')
    (media/'manual.png').write_bytes(image)
    (tmp_path/'expert'/'private.png').write_bytes(image)
    (media/'manifest.json').write_text(json.dumps({'documents':{'doc_0001':{'title':'표면 준비 학습자료','assets':[{'file':'manual.png','label':'표면 예시'},{'file':'../private.png','label':'허용하지 않는 경로'}]}}}),encoding='utf-8')
    client=library_client(monkeypatch,tmp_path)
    detail=client.get('/query-chat/api/knowledge/documents/doc_0001').json()
    assert detail['title']=='표면 준비 학습자료'
    assert len(detail['assets'])==1 and detail['assets'][0]['kind']=='image'
    response=client.get(detail['assets'][0]['url'])
    assert response.status_code==200 and response.content==image
    assert response.headers['content-type'].startswith('image/png')
    assert client.get('/query-chat/api/knowledge/files/not-registered').status_code==404

def test_media_root_symlink_cannot_expose_an_unregistered_directory(monkeypatch,tmp_path):
    media=tmp_path/'expert'/'knowledge_library'
    media.parent.mkdir()
    outside=tmp_path/'outside';outside.mkdir()
    image=base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a7t8AAAAASUVORK5CYII=')
    (outside/'manual.png').write_bytes(image)
    (outside/'manifest.json').write_text(json.dumps({'documents':{'doc_0001':{'assets':[{'file':'manual.png'}]}}}),encoding='utf-8')
    media.symlink_to(outside,target_is_directory=True)
    client=library_client(monkeypatch,tmp_path)
    detail=client.get('/query-chat/api/knowledge/documents/doc_0001').json()
    assert detail['assets']==[] and detail['original_available'] is False

def test_replacing_the_index_preserving_size_and_mtime_refreshes_text(monkeypatch,tmp_path):
    import os
    client=library_client(monkeypatch,tmp_path)
    route='/query-chat/api/knowledge/chunks/DOC-doc_0001-C0002'
    assert '안전' in client.get(route).json()['text']
    metadata=tmp_path/'metadata.json';old=metadata.stat();raw=metadata.read_text(encoding='utf-8')
    replacement=metadata.with_suffix('.new');replacement.write_text(raw.replace('안전','도장'),encoding='utf-8')
    assert replacement.stat().st_size==old.st_size
    os.utime(replacement,ns=(old.st_atime_ns,old.st_mtime_ns));replacement.replace(metadata)
    assert '도장' in client.get(route).json()['text']

def test_static_assets_ignore_filename_and_content_type_query_parameters(monkeypatch,tmp_path):
    import os
    from pathlib import Path
    from trainer.demo_api import knowledge_library
    client=client_for(monkeypatch,tmp_path)
    secret=tmp_path/'protected.txt';secret.write_text('PRIVATE_TEST_FILE',encoding='utf-8')
    traversal=os.path.relpath(secret,Path(knowledge_library.__file__).parent/'static')
    for path,mime in [('/assets/portal_home.js','text/javascript'),('/assets/knowledge_library.js','text/javascript'),
                      ('/assets/portal_home.css','text/css'),('/assets/knowledge_library.css','text/css')]:
        original=client.get(path)
        assert original.status_code==200
        for filename in [str(secret),traversal]:
            response=client.get(path,params={'filename':filename,'content_type':'text/html'})
            assert response.status_code==200 and response.content==original.content
            assert 'PRIVATE_TEST_FILE' not in response.text
            assert response.headers['content-type'].startswith(mime)

def test_excluded_chunks_are_redacted_in_preview_direct_lookup_and_download(monkeypatch,tmp_path):
    client=library_client(monkeypatch,tmp_path)
    path=tmp_path/'metadata.json';data=json.loads(path.read_text(encoding='utf-8'))
    data['id_map'].append({'chunk_id':'DOC-doc_0008-C0006','doc_id':'doc_0001','text':'BAD_REVIEWED_CONTENT','source_type':'pdf'})
    path.write_text(json.dumps(data),encoding='utf-8')
    item=client.get('/query-chat/api/knowledge/chunks/DOC-doc_0008-C0006').json()
    assert item['excluded'] is True and 'BAD_REVIEWED_CONTENT' not in item['text']
    assert item['exclusion_reason']=='사실 오류'
    preview=client.get('/query-chat/api/knowledge/documents/doc_0001').json()
    assert preview['excluded_count']==1
    assert 'BAD_REVIEWED_CONTENT' not in json.dumps(preview)
    response=client.get('/query-chat/api/knowledge/documents/doc_0001/text')
    assert 'BAD_REVIEWED_CONTENT' not in response.text and '제외' in response.text


def test_excluded_text_does_not_supply_the_document_title_or_search_results(monkeypatch,tmp_path):
    client=library_client(monkeypatch,tmp_path)
    path=tmp_path/'metadata.json';data=json.loads(path.read_text(encoding='utf-8'))
    data['id_map'].insert(0,{'chunk_id':'DOC-doc_0008-C0006','doc_id':'doc_0001','text':'EXCLUDED_TITLE_MARKER','source_type':'pdf'})
    path.write_text(json.dumps(data),encoding='utf-8')
    listed=client.get('/query-chat/api/knowledge/documents').json()['documents']
    assert listed[0]['title']=='도장 전 표면 준비를 확인합니다.'
    assert listed[0]['excluded_count']==1
    assert client.get('/query-chat/api/knowledge/documents?q=EXCLUDED_TITLE_MARKER').json()['total']==0


def test_policy_replacement_refreshes_redaction_without_changing_metadata(monkeypatch,tmp_path):
    import os
    from trainer.demo_api import knowledge_library
    policy=tmp_path/'policy.json'
    policy.write_text(json.dumps({'조각':{'DOC-doc_0001-C0009':'표 깨짐: PRIVATE_REASON_MARKER'}}),encoding='utf-8')
    monkeypatch.setattr(knowledge_library,'_POLICY_PATH',policy,raising=False)
    client=library_client(monkeypatch,tmp_path)
    route='/query-chat/api/knowledge/chunks/DOC-doc_0001-C0002'
    assert '안전' in client.get(route).json()['text']
    old=policy.stat();replacement=policy.with_suffix('.new')
    replacement.write_text(policy.read_text(encoding='utf-8').replace('C0009','C0002'),encoding='utf-8')
    assert replacement.stat().st_size==old.st_size
    os.utime(replacement,ns=(old.st_atime_ns,old.st_mtime_ns));replacement.replace(policy)
    response=client.get(route)
    assert response.json()['excluded'] is True
    assert response.json()['exclusion_reason']=='표 깨짐'
    assert '안전' not in response.text and 'PRIVATE_REASON_MARKER' not in response.text


def test_missing_or_invalid_exclusion_policy_closes_all_learning_reads(monkeypatch,tmp_path):
    from trainer.demo_api import knowledge_library
    policy=tmp_path/'policy.json'
    monkeypatch.setattr(knowledge_library,'_POLICY_PATH',policy,raising=False)
    client=library_client(monkeypatch,tmp_path)
    routes=['/query-chat/api/knowledge/documents',
            '/query-chat/api/knowledge/documents/doc_0001',
            '/query-chat/api/knowledge/chunks/DOC-doc_0001-C0001',
            '/query-chat/api/knowledge/documents/doc_0001/text',
            '/query-chat/api/knowledge/files/anything']
    for invalid in [None,'{invalid',json.dumps({'wrong_key':{}}),json.dumps({'조각':[]})]:
        if invalid is not None:
            policy.write_text(invalid,encoding='utf-8')
        for route in routes:
            response=client.get(route)
            assert response.status_code==503
            assert '표면 준비' not in response.text


def test_corrected_text_replaces_source_in_title_search_preview_direct_lookup_and_download(monkeypatch,tmp_path):
    from trainer.demo_api import knowledge_library
    client=library_client(monkeypatch,tmp_path)
    metadata=tmp_path/'metadata.json';data=json.loads(metadata.read_text(encoding='utf-8'))
    data['id_map'][0]['text']='SUPERSEDED_SOURCE_MARKER'
    metadata.write_text(json.dumps(data),encoding='utf-8')
    knowledge_library._CORRECTIONS_PATH.write_text(json.dumps({'조각':{'DOC-doc_0001-C0001':{
        '본문':'CORRECTED_TITLE corrected_search_term','원인':'PRIVATE_AUDIT_MARKER',
        '원본':'PRIVATE_AUDIT_MARKER','바꾼 곳':'PRIVATE_AUDIT_MARKER'}}}),encoding='utf-8')
    docs=client.get('/query-chat/api/knowledge/documents').json()['documents']
    assert docs[0]['title']=='CORRECTED_TITLE corrected_search_term'
    assert '교정본' in docs[0]['title_source'] and docs[0]['corrected_count']==1
    assert client.get('/query-chat/api/knowledge/documents?q=corrected_search_term').json()['total']==1
    assert client.get('/query-chat/api/knowledge/documents?q=SUPERSEDED_SOURCE_MARKER').json()['total']==0
    preview=client.get('/query-chat/api/knowledge/documents/doc_0001').json()
    assert preview['corrected_count']==1 and '교정본' in preview['text_basis']
    item=client.get('/query-chat/api/knowledge/chunks/DOC-doc_0001-C0001').json()
    assert item['corrected'] is True and item['excluded'] is False
    assert item['text']=='CORRECTED_TITLE corrected_search_term'
    assert item['correction_basis']=='원본 대조 교정본'
    download=client.get('/query-chat/api/knowledge/documents/doc_0001/text')
    assert 'CORRECTED_TITLE corrected_search_term' in download.text and '원본 대조 교정본' in download.text
    for response in [json.dumps(docs),json.dumps(preview),json.dumps(item),download.text]:
        assert 'SUPERSEDED_SOURCE_MARKER' not in response and 'PRIVATE_AUDIT_MARKER' not in response


def test_exclusion_precedes_a_registered_correction(monkeypatch,tmp_path):
    from trainer.demo_api import knowledge_library
    client=library_client(monkeypatch,tmp_path)
    knowledge_library._POLICY_PATH.write_text(json.dumps({'조각':{'DOC-doc_0001-C0001':'표 깨짐: PRIVATE_AUDIT_MARKER'}}),encoding='utf-8')
    knowledge_library._CORRECTIONS_PATH.write_text(json.dumps({'조각':{'DOC-doc_0001-C0001':{'본문':'CORRECTION_WITHHELD_MARKER'}}}),encoding='utf-8')
    item=client.get('/query-chat/api/knowledge/chunks/DOC-doc_0001-C0001').json()
    assert item['excluded'] is True and item['corrected'] is False
    assert 'CORRECTION_WITHHELD_MARKER' not in json.dumps(item)
    assert client.get('/query-chat/api/knowledge/documents?q=CORRECTION_WITHHELD_MARKER').json()['total']==0
    assert client.get('/query-chat/api/knowledge/documents/doc_0001').json()['corrected_count']==0


def test_same_size_and_mtime_correction_replacement_refreshes_unchanged_metadata(monkeypatch,tmp_path):
    import os
    from trainer.demo_api import knowledge_library
    client=library_client(monkeypatch,tmp_path)
    policy=knowledge_library._CORRECTIONS_PATH
    policy.write_text(json.dumps({'조각':{'DOC-doc_0001-C0001':{'본문':'CORRECTED_VERSION_A'}}}),encoding='utf-8')
    route='/query-chat/api/knowledge/chunks/DOC-doc_0001-C0001'
    assert client.get(route).json()['text']=='CORRECTED_VERSION_A'
    old=policy.stat();replacement=policy.with_suffix('.new')
    replacement.write_text(policy.read_text(encoding='utf-8').replace('VERSION_A','VERSION_B'),encoding='utf-8')
    assert replacement.stat().st_size==old.st_size
    os.utime(replacement,ns=(old.st_atime_ns,old.st_mtime_ns));replacement.replace(policy)
    assert client.get(route).json()['text']=='CORRECTED_VERSION_B'
    assert client.get('/query-chat/api/knowledge/documents').json()['documents'][0]['title']=='CORRECTED_VERSION_B'
    assert client.get('/query-chat/api/knowledge/documents?q=CORRECTED_VERSION_A').json()['total']==0


def test_missing_or_malformed_correction_policy_closes_all_learning_reads(monkeypatch,tmp_path):
    from trainer.demo_api import knowledge_library
    client=library_client(monkeypatch,tmp_path)
    policy=tmp_path/'required-corrections.json'
    monkeypatch.setattr(knowledge_library,'_CORRECTIONS_PATH',policy,raising=False)
    routes=['/query-chat/api/knowledge/documents','/query-chat/api/knowledge/documents/doc_0001',
            '/query-chat/api/knowledge/chunks/DOC-doc_0001-C0001',
            '/query-chat/api/knowledge/documents/doc_0001/text','/query-chat/api/knowledge/files/anything']
    malformed=[None,'{invalid',json.dumps({'wrong_key':{}}),json.dumps({'조각':[]}),
               json.dumps({'조각':{'DOC-doc_0001-C0001':'invalid'}}),
               json.dumps({'조각':{'DOC-doc_0001-C0001':{'본문':''}}}),
               json.dumps({'조각':{'DOC-doc_0001-C0001':{'본문':'x'*(64*1024+1)}}}),
               json.dumps({'조각':{'invalid\nID':{'본문':'SAFE_MARKER'}}}),
               ' '*(1024*1024+1)]
    for invalid in malformed:
        if invalid is not None:
            policy.write_text(invalid,encoding='utf-8')
        for route in routes:
            response=client.get(route)
            assert response.status_code==503
            assert '표면 준비' not in response.text
