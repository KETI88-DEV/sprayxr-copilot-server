import hashlib
import json
import os
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.db_admin_auth import DBAdminAuth
from trainer.demo_api.training_data_catalog import TrainingDataCatalog
from trainer.demo_api.training_verification import ConstructionVerification, create_training_verification_router


def fixtures(tmp_path, count=2):
    root, sources = tmp_path / 'datasets', tmp_path / 'sources'
    (root / 'records').mkdir(parents=True); sources.mkdir()
    rows=[]
    for i in range(count):
        did=f'{i:032x}'; source=sources/f'{i}.csv';source.write_text(f'time,x\n0,{i+1}\n1,{i+2}\n')
        meta={'dataset_id':did,'dataset_name':f'3_TEST_{i}','worker_id':'qa','worker_name':'정성오',
              'process':'spray','scenario_id':'3','session_id':f'3_TEST_{i}',
              'quality_status':'complete','frame_count':3,'zone_id_available':True}
        artifact={'schema_version':1,'dataset_id':did,'dataset_type':'record_features','metadata':meta,
                  'quality':{'status':'complete','issues':[]},'features':{'frame_count':3,'zone_id_available':True},
                  'provenance':{'source_csv':{'relative_path':source.name,'bytes':source.stat().st_size,
                                 'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}}}
        raw=json.dumps(artifact,ensure_ascii=False).encode();(root/'records'/f'{did}.json').write_bytes(raw)
        rows.append({**meta,'json_bytes':len(raw),'artifact_sha256':hashlib.sha256(raw).hexdigest()})
    (root/'catalog.json').write_text(json.dumps({'records':rows}))
    catalog=TrainingDataCatalog(root,sources)
    return catalog,ConstructionVerification(catalog,tmp_path/'verification'),rows


def proof(row,run_id):
    return {'run_id':run_id,'artifact_sha256':row['artifact_sha256'],'status':'passed','dataset_id':row['dataset_id'],
            'session_id':row['session_id'],'renderer':'canvas-2d','frames_loaded':10,'duration_seconds':10,
            'samples':[{'position':name,'time_before':at+.01,'time_after':at+.2,
                        'frame_before':int(at),'frame_after':int(at)+1,'draw_before':10*i,'draw_after':10*i+5,
                        'renderer_ready':True,'context_lost':False}
                       for i,(name,at) in enumerate([('start',0),('middle',5),('end',8.5)])]}


def test_frozen_complete_denominator_keeps_missing_and_corrupt_artifacts(tmp_path):
    catalog,store,rows=fixtures(tmp_path,3)
    (catalog.root/'records'/f"{rows[0]['dataset_id']}.json").unlink()
    (catalog.root/'records'/f"{rows[1]['dataset_id']}.json").write_text('{}')
    result=store.audit()
    assert result['summary']['registered']==3
    assert result['summary']['json_passed']==1
    assert result['summary']['valid']==0
    assert len(result['items'])==3
    assert result['summary']['browser_passed']==0
    assert all(item['browser_status']!='passed' for item in result['items'])


def test_real_browser_proof_required_and_snapshot_changes_rejected(tmp_path):
    catalog,store,rows=fixtures(tmp_path)
    store.audit()
    bad=proof(rows[0],store.status()['run_id']);bad['samples'][0]['frame_after']=bad['samples'][0]['frame_before']
    with pytest.raises(ValueError):store.record_playback(rows[0]['dataset_id'],bad)
    assert store.status()['summary']['valid']==0
    store.record_playback(rows[0]['dataset_id'],proof(rows[0],store.status()['run_id']))
    assert store.status()['summary']['valid']==1
    store.record_playback(rows[0]['dataset_id'],proof(rows[0],store.status()['run_id']))
    assert store.status()['summary']['valid']==1
    index=catalog._index();index['extra']='changed';(catalog.root/'catalog.json').write_text(json.dumps(index))
    assert store.status()['stale'] is True
    with pytest.raises(ValueError):store.record_playback(rows[1]['dataset_id'],proof(rows[1],store.status()['run_id']))


def test_duplicate_sessions_and_source_escape_not_counted(tmp_path):
    catalog,store,rows=fixtures(tmp_path)
    index=catalog._index();index['records'][1]['session_id']=rows[0]['session_id']
    (catalog.root/'catalog.json').write_text(json.dumps(index))
    result=store.audit()
    assert result['summary']['duplicates']==0
    assert result['summary']['registered']==2
    assert result['summary']['valid']==0
    public=json.dumps(result)
    assert str(catalog.sources) not in public and str(catalog.root) not in public


def test_identical_source_under_different_names_is_excluded_once(tmp_path):
    catalog,store,rows=fixtures(tmp_path)
    source=catalog.sources/'1.csv';source.write_bytes((catalog.sources/'0.csv').read_bytes())
    path=catalog.root/'records'/f"{rows[1]['dataset_id']}.json";data=json.loads(path.read_bytes())
    data['provenance']['source_csv']['sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    raw=json.dumps(data,ensure_ascii=False).encode();path.write_bytes(raw)
    index=catalog._index();index['records'][1].update(json_bytes=len(raw),artifact_sha256=hashlib.sha256(raw).hexdigest())
    (catalog.root/'catalog.json').write_text(json.dumps(index))
    result=store.audit()
    assert result['summary']['duplicates']==1 and result['summary']['pending']==1


def test_temporary_output_error_does_not_permanently_block_restart(tmp_path,monkeypatch):
    catalog,store,rows=fixtures(tmp_path,1)
    original=store._save
    monkeypatch.setattr(store,'_save',lambda:(_ for _ in ()).throw(OSError('disk')))
    with pytest.raises(OSError):store.audit()
    monkeypatch.setattr(store,'_save',original)
    assert store.audit()['summary']['json_passed']==1


def test_legacy_schema_uses_actual_mapped_zones_not_a_new_optional_flag(tmp_path):
    catalog,store,rows=fixtures(tmp_path,1)
    path=catalog.root/'records'/f"{rows[0]['dataset_id']}.json";data=json.loads(path.read_bytes())
    data['schema_version']='record-features-1.1'
    data['features'].pop('zone_id_available');data['features']['zones']=[{'map_num':1,'zone_id':'2'}]
    raw=json.dumps(data,ensure_ascii=False).encode();path.write_bytes(raw)
    index=catalog._index();index['records'][0].update(json_bytes=len(raw),artifact_sha256=hashlib.sha256(raw).hexdigest())
    (catalog.root/'catalog.json').write_text(json.dumps(index))
    assert store.audit()['summary']['json_passed']==1


def test_browser_proof_rechecks_actual_json_not_only_declared_hash(tmp_path):
    catalog,store,rows=fixtures(tmp_path,1);store.audit()
    path=catalog.root/'records'/f"{rows[0]['dataset_id']}.json"
    path.write_bytes(path.read_bytes().replace(b'qa',b'qb'))
    with pytest.raises(ValueError):store.record_playback(rows[0]['dataset_id'],proof(rows[0],store.status()['run_id']))
    assert store.status()['summary']['valid']==0


def test_source_content_changes_of_same_size_are_not_treated_as_verified(tmp_path):
    catalog,store,rows=fixtures(tmp_path,1)
    source=catalog.sources/'0.csv';source.write_text('time,x\n0,9\n1,2\n')
    result=store.audit()
    assert result['summary']['json_passed']==1
    assert result['summary']['source_passed']==0
    assert result['items'][0]['reasons']==['source_hash_mismatch']


def test_start_middle_end_names_cannot_mask_three_start_samples(tmp_path):
    catalog,store,rows=fixtures(tmp_path,1);store.audit()
    value=proof(rows[0],store.status()['run_id']);value['samples'][1]['time_before']=.01;value['samples'][1]['time_after']=.1
    with pytest.raises(ValueError):store.record_playback(rows[0]['dataset_id'],value)


def test_public_read_but_writes_require_existing_admin_and_csrf(tmp_path):
    catalog,store,rows=fixtures(tmp_path)
    auth=DBAdminAuth(tmp_path/'credentials.json')
    app=FastAPI();app.include_router(create_training_verification_router(auth,store))
    with TestClient(app) as client:
        assert client.get('/admin/api/training-verification').status_code==200
        assert client.post('/admin/api/training-verification/start').status_code==401
        assert client.post('/admin/api/training-verification/playback/'+rows[0]['dataset_id'],json=proof(rows[0],'a'*32)).status_code==401


@pytest.mark.parametrize('changed_file', ['json', 'source'])
@pytest.mark.parametrize('change', ['delete', 'same_size_write'])
def test_verified_file_drift_invalidates_current_count_but_preserves_snapshot(tmp_path, changed_file, change):
    catalog,store,rows=fixtures(tmp_path,1)
    store.audit()
    store.record_playback(rows[0]['dataset_id'],proof(rows[0],store.status()['run_id']))
    before=store.status()
    assert before['summary']['valid']==1
    path=(catalog.root/'records'/f"{rows[0]['dataset_id']}.json" if changed_file=='json'
          else catalog.sources/'0.csv')
    previous=path.stat()
    if change=='delete':
        path.unlink()
    else:
        raw=path.read_bytes()
        replacement=raw.replace(b'qa',b'qb') if changed_file=='json' else raw.replace(b'0,1',b'0,9')
        assert replacement!=raw and len(replacement)==len(raw)
        path.write_bytes(replacement)
        os.utime(path,ns=(previous.st_atime_ns,previous.st_mtime_ns+1_000_000))
    after=store.status()
    assert after['stale'] is True
    assert after['summary']['valid']==0 and after['summary']['goal_met'] is False
    assert after['summary']['snapshot_valid']==1 and after['summary']['browser_passed']==1
    assert ('artifact_changed' if changed_file=='json' else 'source_changed') in after['stale_reasons']
    assert after['stale_reason']
    assert after['items'][0]['browser_status']=='passed'


@pytest.mark.parametrize('corrupt_before_first_read', [False, True])
def test_legacy_frozen_result_checks_json_hash_before_caching_current_fingerprint(tmp_path, monkeypatch, corrupt_before_first_read):
    catalog,store,rows=fixtures(tmp_path,1)
    store.audit()
    store.record_playback(rows[0]['dataset_id'],proof(rows[0],store.status()['run_id']))
    # Old frozen runs have no JSON file stat, only the original artifact SHA.
    run_path=store.root/'runs'/f"{store.status()['run_id']}.json"
    frozen=json.loads(run_path.read_text())
    frozen['items'][0].pop('_artifact',None)
    frozen.pop('viewer_canonical_sha256',None)
    run_path.write_text(json.dumps(frozen))
    path=catalog.root/'records'/f"{rows[0]['dataset_id']}.json"
    previous=path.stat()
    if corrupt_before_first_read:
        path.write_bytes(path.read_bytes().replace(b'qa',b'qb'))
        # Matching legacy metadata must not bless changed bytes on first load.
        os.utime(path,ns=(previous.st_atime_ns,previous.st_mtime_ns))
    reopened=ConstructionVerification(catalog,store.root)
    import trainer.demo_api.training_verification as module
    original=module._sha
    artifact_reads=[]
    def observed_sha(candidate):
        if candidate==path:artifact_reads.append(candidate)
        return original(candidate)
    monkeypatch.setattr(module,'_sha',observed_sha)
    first=reopened.status()
    assert first['stale'] is corrupt_before_first_read
    assert first['summary']['valid']==(0 if corrupt_before_first_read else 1)
    assert artifact_reads==[path]
    assert '_artifact' not in first['items'][0] and '_source' not in first['items'][0]
    if not corrupt_before_first_read:
        assert reopened.status()['summary']['valid']==1
        assert artifact_reads==[path]  # Subsequent reads only need stat checks.
        path.unlink()
        assert reopened.status()['stale'] is True
    assert json.loads(run_path.read_text())==frozen  # Historical evidence is immutable on read.


def test_successful_playback_retry_replaces_current_failure_reason(tmp_path):
    catalog,store,rows=fixtures(tmp_path,1)
    store.audit()
    failed=proof(rows[0],store.status()['run_id']);failed['status']='failed';failed['samples']=[]
    store.record_playback(rows[0]['dataset_id'],failed)
    assert store.status()['items'][0]['reasons']==['browser_playback_failed']
    store.record_playback(rows[0]['dataset_id'],proof(rows[0],store.status()['run_id']))
    current=store.status()
    assert current['summary']['browser_failed']==0 and current['summary']['valid']==1
    assert 'browser_playback_failed' not in current['items'][0]['reasons']


def test_new_renderer_signature_ignores_line_endings_but_invalidates_changed_code(tmp_path,monkeypatch):
    import trainer.demo_api.training_verification as module
    verifier=tmp_path/'training_verification.py';verifier.write_bytes(b'verifier\n')
    viewer=tmp_path/'motion_viewer.py';viewer.write_bytes(b'renderer\r\n')
    monkeypatch.setattr(module,'__file__',str(verifier))
    catalog,store,rows=fixtures(tmp_path/'data',1)
    report=store.audit()
    assert report['viewer_canonical_sha256']==hashlib.sha256(b'renderer\n').hexdigest()
    store.record_playback(rows[0]['dataset_id'],proof(rows[0],report['run_id']))
    previous=viewer.stat()
    viewer.write_bytes(b'renderer\n')
    os.utime(viewer,ns=(previous.st_atime_ns,previous.st_mtime_ns+1_000_000))
    assert store.status()['summary']['valid']==1
    viewer.write_bytes(b'changed renderer\n')
    current=store.status()
    assert current['stale'] is True and 'viewer_changed' in current['stale_reasons']
    assert current['summary']['valid']==0 and current['summary']['snapshot_valid']==1
