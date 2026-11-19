"""Completed Zone-mapped artifacts automatically become playable viewer records."""
import csv
import hashlib
import json

from trainer.demo_api.training_motion_catalog import TrainingMotionCatalog
from trainer.demo_api.tests.test_motion_viewer import _client, _write_csv


def publish(root, output, *, worker='new_worker', worker_name='새 작업자', session='4_2001_1', quality='complete', mapped=True):
    source = root / 'imports' / 'converted' / (session + '_Height.csv')
    _write_csv(source, offset=0.3)
    dataset_id = hashlib.md5((worker + session).encode()).hexdigest()
    row = {'dataset_id':dataset_id, 'dataset_name':session+'_Height', 'session_id':session,
           'worker_id':worker,'worker_name':worker_name,'role':'trainee','process':'spray','scenario_id':'4',
           'mode_label':'학습모드 4','quality_status':quality,'zone_id_available':mapped,
           'zone_count':1 if mapped else 0,'source_kind':'converted','frame_count':5,'heightmap_count':0}
    artifact = {'metadata':row,'features':{'zone_order':[{'map_num':1,'zone_id':'1'}],
        'zones':[{'map_num':1,'zone_id':'1','speed_mms':{'median':1600},'distance_mm':{'median':260},
                  'speed_real_mms':{'median':300},'distance_real_mm':{'median':250}}]},
        'provenance':{'source_csv':{'relative_path':source.relative_to(root).as_posix()},'heightmaps':[]}}
    content=json.dumps(artifact,ensure_ascii=False).encode()
    records=output/'records';records.mkdir(parents=True,exist_ok=True)
    (records/(dataset_id+'.json')).write_bytes(content)
    row.update(json_bytes=len(content),artifact_sha256=hashlib.sha256(content).hexdigest())
    path=output/'catalog.json'
    index=json.loads(path.read_text()) if path.exists() else {'records':[]}
    index['records']=[r for r in index['records'] if r['dataset_id']!=dataset_id]+[row]
    path.write_text(json.dumps(index),encoding='utf-8')
    return row


def test_catalog_auto_refreshes_and_only_includes_complete_mapped_records(monkeypatch,tmp_path):
    output=tmp_path/'built';monkeypatch.setenv('TRAINING_DATA_ROOT',str(output))
    publish(tmp_path,output)
    publish(tmp_path,output,worker='review_worker',session='4_2002_1',quality='needs_review')
    publish(tmp_path,output,worker='unmapped_worker',session='4_2003_1',mapped=False)
    client=_client(tmp_path)
    catalog=client.get('/motion-viewer/api/catalog').json()
    workers={worker['worker_id']:worker for worker in catalog['workers']}
    assert 'new_worker' in workers
    assert 'review_worker' not in workers and 'unmapped_worker' not in workers
    assert workers['new_worker']['processes'][0]['modes'][0]['sessions']==['4_2001_1']
    publish(tmp_path,output,worker='later_worker',session='4_2004_1')
    assert 'later_worker' in {w['worker_id'] for w in client.get('/motion-viewer/api/catalog').json()['workers']}


def test_auto_registered_record_actually_plays_with_source_zone_ids(monkeypatch,tmp_path):
    output=tmp_path/'built';monkeypatch.setenv('TRAINING_DATA_ROOT',str(output))
    publish(tmp_path,output)
    client=_client(tmp_path)
    response=client.get('/motion-viewer/api/motion',params={'worker_id':'new_worker','process':'spray',
        'mode_id':4,'session_id':'4_2001_1'})
    assert response.status_code==200,response.text
    body=response.json()
    assert body['worker']['display_name']=='새 작업자'
    assert body['selected_session_id']=='4_2001_1' and len(body['frames'])==5
    assert body['frames'][0]['zone_id']=='1'
    assert body['paint_trace'] and body['query_context_signatures']


def test_session_zone_metrics_use_measured_real_columns_not_simulator_input(monkeypatch,tmp_path):
    output=tmp_path/'built';monkeypatch.setenv('TRAINING_DATA_ROOT',str(output))
    publish(tmp_path,output)
    profile=TrainingMotionCatalog(tmp_path).profile('new_worker','spray',4)
    zone=profile['sessions'][0]['analysis']['zones']['1:1']
    assert (zone['speed_mms'],zone['distance_mm'])==(300,250)
    assert profile['aggregate']['zones']['1:1']['speed_mms']['median']==300


def test_auto_catalog_does_not_duplicate_existing_worker_or_session(monkeypatch,tmp_path):
    output=tmp_path/'built';monkeypatch.setenv('TRAINING_DATA_ROOT',str(output))
    publish(tmp_path,output,worker='jeon_hyeonggi',session='4_1001_1')
    client=_client(tmp_path)
    catalog=client.get('/motion-viewer/api/catalog').json()
    foremen=[w for w in catalog['workers'] if w['worker_id']=='jeon_hyeonggi']
    assert len(foremen)==1
    sessions=foremen[0]['processes'][0]['modes'][0]['sessions']
    assert sessions.count('4_1001_1')==1


def test_profile_rejects_source_paths_outside_the_registered_root(monkeypatch,tmp_path):
    output=tmp_path/'built';monkeypatch.setenv('TRAINING_DATA_ROOT',str(output))
    row=publish(tmp_path,output)
    path=output/'records'/(row['dataset_id']+'.json')
    artifact=json.loads(path.read_text(encoding='utf-8'));artifact['provenance']['source_csv']['relative_path']='../outside.csv'
    content=json.dumps(artifact).encode();path.write_bytes(content)
    index=json.loads((output/'catalog.json').read_text());index['records'][0].update(json_bytes=len(content),artifact_sha256=hashlib.sha256(content).hexdigest())
    (output/'catalog.json').write_text(json.dumps(index))
    assert TrainingMotionCatalog(tmp_path).profile('new_worker','spray',4) is None


def test_known_sql_worker_alias_merges_into_legacy_selector_and_preserves_source_id(monkeypatch,tmp_path):
    output=tmp_path/'built';monkeypatch.setenv('TRAINING_DATA_ROOT',str(output))
    publish(tmp_path,output,worker='tsb_user_sql86',worker_name='전형기',session='4_2100_1')
    publish(tmp_path,output,worker='tsb_other',worker_name='다른 작업자',session='4_2101_1')
    client=_client(tmp_path)
    workers=client.get('/motion-viewer/api/catalog').json()['workers']
    assert 'tsb_user_sql86' not in {w['worker_id'] for w in workers}
    foremen=[w for w in workers if w['worker_id']=='jeon_hyeonggi']
    assert len(foremen)==1
    assert '4_2100_1' in foremen[0]['processes'][0]['modes'][0]['sessions']
    profile=TrainingMotionCatalog(tmp_path).profile('jeon_hyeonggi','spray',4)
    assert profile['sessions'][0]['source_worker_id']=='tsb_user_sql86'
    response=client.get('/motion-viewer/api/motion',params={'worker_id':'jeon_hyeonggi','process':'spray',
        'mode_id':4,'session_id':'4_2100_1'})
    assert response.status_code==200,response.text
    assert len(response.json()['frames'])==5


def test_converted_csv_large_sparse_cell_and_explicit_blasting_tool(tmp_path):
    from trainer.demo_api.motion_viewer import _frames
    source=tmp_path/'imports'/'converted'/'4_9000_1_Height.csv'
    _write_csv(source,offset=0)
    with source.open(encoding='utf-8-sig',newline='') as handle:
        reader=csv.DictReader(handle);fields=list(reader.fieldnames);rows=list(reader)
    fields+=['cur_sequence_heatmap',*[f'블라스트 건 Position {axis}' for axis in 'XYZ']]
    with source.open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for row in rows:
            row.update({'cur_sequence_heatmap':'x'*200000,'블라스트 건 Position X':2,
                        '블라스트 건 Position Y':1,'블라스트 건 Position Z':3})
            writer.writerow(row)
    frames=_frames(source,5,'blasting')
    assert len(frames)==5 and frames[0]['tracked']['right_hand']==[2.0,1.0,3.0]


def test_worker_choices_are_name_sorted_with_one_unknown_bucket_and_original_ids(monkeypatch, tmp_path):
    output = tmp_path / 'built'; monkeypatch.setenv('TRAINING_DATA_ROOT', str(output))
    publish(tmp_path, output, worker='z', worker_name='하 작업자', session='4_2201_1')
    publish(tmp_path, output, worker='a', worker_name='가 작업자', session='4_2202_1')
    publish(tmp_path, output, worker='raw-unknown-a', worker_name='미기록', session='4_2203_1')
    # Equal source session IDs under unknown workers must remain separate records.
    publish(tmp_path, output, worker='raw-unknown-b', worker_name='', session='4_2203_1')
    client = _client(tmp_path)
    workers = client.get('/motion-viewer/api/catalog').json()['workers']
    assert [item['display_name'] for item in workers] == ['전형기', '가 작업자', '하 작업자', '이름 미확인']
    assert workers[0]['role'] == 'expert'
    unknown = workers[-1]
    assert unknown['worker_id'] == '__unknown_workers__'
    sessions = unknown['processes'][0]['modes'][0]['sessions']
    assert len(sessions) == 2 and len(set(sessions)) == 2
    profile = TrainingMotionCatalog(tmp_path).profile('__unknown_workers__', 'spray', 4)
    assert {row['source_worker_id'] for row in profile['sessions']} == {'raw-unknown-a', 'raw-unknown-b'}
    assert {row['source_session_id'] for row in profile['sessions']} == {'4_2203_1'}
    payload = client.get('/motion-viewer/api/motion', params={
        'worker_id': '__unknown_workers__', 'process': 'spray', 'mode_id': 4, 'session_id': sessions[0]}).json()
    assert payload['worker']['worker_id'] in {'raw-unknown-a', 'raw-unknown-b'}
    assert payload['worker']['selection_worker_id'] == '__unknown_workers__'
