"""Frozen construction evidence; browser playback is never inferred from JSON validity."""
from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import math
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field
from trainer.demo_api import __version__

from trainer.demo_api.db_admin_auth import DBAdminAuth
from trainer.demo_api.training_data_catalog import TrainingDataCatalog
from trainer.demo_api.training_motion_catalog import TrainingMotionCatalog

DEFINITION = '동일 공정의 훈련 기록 1회에서 가공한 JSON과 연결 원천 모션 자료를 1건으로 산정하며 복사본은 중복 집계하지 않습니다.'
SCOPE = 'JSON·원천 파일 SHA-256은 전수 확인하고, 실제 브라우저의 시작·중간·끝 구간 재생을 별도로 확인합니다. 공정별 동일 원천 바이트는 중복 제외합니다. Zone 보강본 간 원래 모션의 동일성, 전체 시간 연속 재생, 라벨 정확도·AI 성능 시험은 포함하지 않습니다.'
_ID = re.compile(r'^[0-9a-f]{32}$')


def _now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _sha(path: Path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def _atomic(path: Path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex)
    try:
        temp.write_text(json.dumps(data,ensure_ascii=False,allow_nan=False),encoding='utf-8')
        temp.replace(path)
    finally:temp.unlink(missing_ok=True)


class ConstructionVerification:
    def __init__(self,catalog: TrainingDataCatalog,root: Path | None=None):
        self.catalog=catalog
        self.root=Path(root or os.getenv('TRAINING_VERIFICATION_ROOT',str(catalog.root.parent/'training_verification')))
        self._guard=threading.RLock();self._running=False;self._report=None
        self._catalog_cache=None
        self._viewer_cache=None
        self._legacy_artifact_fingerprints={}

    def _catalog_digest(self):
        path=self.catalog.root/'catalog.json'
        try:
            stat=path.stat();key=(stat.st_size,stat.st_mtime_ns)
            if not self._catalog_cache or self._catalog_cache[0]!=key:self._catalog_cache=(key,_sha(path))
            return self._catalog_cache[1]
        except OSError:return None

    def _canonical_viewer_digest(self):
        path=Path(__file__).with_name('motion_viewer.py')
        try:
            before=self._fingerprint(path.stat());key=(before['bytes'],before['mtime_ns'])
            if not self._viewer_cache or self._viewer_cache[0]!=key:
                digest=hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                if self._fingerprint(path.stat())!=before:return None
                self._viewer_cache=(key,digest)
            return self._viewer_cache[1]
        except OSError:return None

    def _load(self):
        if self._report is None:
            try:
                pointer=json.loads((self.root/'latest.json').read_text())
                rid=pointer['run_id']
                if not _ID.fullmatch(rid):raise ValueError('invalid_run')
                self._report=json.loads((self.root/'runs'/f'{rid}.json').read_text())
            except (OSError,ValueError,KeyError,TypeError):pass
        return self._report

    def _save(self):
        _atomic(self.root/'runs'/f"{self._report['run_id']}.json",self._report)
        _atomic(self.root/'latest.json',{'run_id':self._report['run_id']})

    @staticmethod
    def _fingerprint(stat):
        return {'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}

    def _current_file_changes(self,report):
        """Detect drift without rewriting historical results or rehashing CSVs.

        Older frozen runs did not retain a JSON stat. Their first read must
        compare actual bytes with the frozen SHA before caching a cheap stat;
        merely taking the current stat would bless a pre-existing change.
        """
        changed=set()
        for item in report['items']:
            if item['json_status']=='passed':
                try:
                    path=self.catalog._safe(self.catalog.root,f"records/{item['dataset_id']}.json")
                    current=self._fingerprint(path.stat())
                    expected=item.get('_artifact')
                    if expected is None:
                        key=(report['run_id'],item['dataset_id'],item['artifact_sha256'])
                        expected=self._legacy_artifact_fingerprints.get(key)
                        if expected is None:
                            digest=_sha(path)
                            if digest!=item['artifact_sha256'] or self._fingerprint(path.stat())!=current:
                                changed.add('artifact_changed');continue
                            expected=current
                            self._legacy_artifact_fingerprints[key]=expected
                    if current!=expected:changed.add('artifact_changed')
                except (OSError,ValueError,TypeError):changed.add('artifact_changed')
            if item['source_status']=='passed':
                try:
                    source=item.get('_source') or {}
                    path=self.catalog._safe(self.catalog.sources,source.get('relative_path',''))
                    expected={key:source.get(key) for key in ['bytes','mtime_ns']}
                    if self._fingerprint(path.stat())!=expected:changed.add('source_changed')
                except (OSError,ValueError,TypeError,AttributeError):changed.add('source_changed')
        return sorted(changed)

    def _public(self):
        report=copy.deepcopy(self._load())
        if report is None:
            registered=sum(isinstance(r,dict) and r.get('quality_status')=='complete'
                           for r in self.catalog._index().get('records',[]))
            return {'phase':'not_started','definition':DEFINITION,'scope':SCOPE,'target':1000,
                    'stale':False,'summary':{'registered':registered,'json_passed':0,'source_passed':0,
                    'browser_passed':0,'browser_failed':0,'duplicates':0,'valid':0,'pending':registered,
                    'goal_met':False},'items':[]}
        items=report['items'];valid=sum(r['json_status']=='passed' and r['source_status']=='passed'
                  and not r['duplicate'] and r['browser_status']=='passed' for r in items)
        report['stale_reasons']=(['catalog_changed'] if report['catalog_sha256']!=self._catalog_digest() else [])+self._current_file_changes(report)
        # Legacy raw hashes remain historical evidence: CRLF checkout bytes and
        # the same deployed LF code must not be mistaken for different code.
        if report.get('viewer_canonical_sha256') and report['viewer_canonical_sha256']!=self._canonical_viewer_digest():
            report['stale_reasons'].append('viewer_changed')
        report['stale']=bool(report['stale_reasons'])
        report['stale_reason']=' '.join(label for reason,label in [
            ('catalog_changed','등록 목록이 검증 당시와 달라졌습니다.'),
            ('artifact_changed','JSON 파일이 검증 당시와 달라졌거나 없어졌습니다.'),
            ('source_changed','연결 원천 파일이 검증 당시와 달라졌거나 없어졌습니다.'),
            ('viewer_changed','3D 재생 코드가 검증 당시와 달라졌거나 없어졌습니다.')]
            if reason in report['stale_reasons'])
        if report['phase']=='checking_json' and not self._running:report['phase']='interrupted'
        report['summary']={'registered':len(items),'json_passed':sum(r['json_status']=='passed' for r in items),
            'source_passed':sum(r['source_status']=='passed' for r in items),
            'browser_passed':sum(r['browser_status']=='passed' for r in items),
            'browser_failed':sum(r['browser_status']=='failed' for r in items),
            'duplicates':sum(r['duplicate'] for r in items),
            'pending':sum(r['browser_status']=='pending' for r in items),
            'snapshot_valid':valid,'valid':0 if report['stale'] else valid,
            'goal_met':not report['stale'] and valid>=1000}
        for item in items:
            item.pop('_source',None);item.pop('_artifact',None)
        return report

    def status(self):
        with self._guard:return self._public()

    def _prepare(self):
        if self._running:raise ValueError('verification_running')
        digest=self._catalog_digest();index=self.catalog._index()
        if not digest:raise ValueError('catalog_unavailable')
        viewer_digest=self._canonical_viewer_digest()
        if not viewer_digest:raise ValueError('viewer_unavailable')
        rows=[copy.deepcopy(r) for r in index.get('records',[]) if isinstance(r,dict) and r.get('quality_status')=='complete']
        if self._catalog_digest()!=digest:raise ValueError('catalog_changed')
        items=[]
        for row in rows:
            items.append({**{k:str(row.get(k) or '')[:200] for k in ['dataset_id','dataset_name','worker_id','worker_name','process','scenario_id']},
                'session_id':TrainingMotionCatalog.session_id(row),'artifact_sha256':str(row.get('artifact_sha256') or ''),
                'duplicate':False,'json_status':'pending','source_status':'pending',
                'browser_status':'pending','reasons':[]})
        self._report={'run_id':uuid.uuid4().hex,'method_version':1,'definition':DEFINITION,'scope':SCOPE,
            'target':1000,'phase':'checking_json','started_at':_now(),'checked_at':None,
            'catalog_sha256':digest,'application_version':__version__,
            'verifier_sha256':_sha(Path(__file__)),
            'viewer_sha256':_sha(Path(__file__).with_name('motion_viewer.py')),
            'viewer_canonical_sha256':viewer_digest,
            'processed':0,'items':items}
        self._save();self._legacy_artifact_fingerprints.clear();self._running=True;return rows

    def _check(self,row,item):
        try:
            did=item['dataset_id']
            if not _ID.fullmatch(did):raise ValueError('invalid_dataset_id')
            path=self.catalog._safe(self.catalog.root,f'records/{did}.json')
            before=self._fingerprint(path.stat())
            payload=path.read_bytes()
            if hashlib.sha256(payload).hexdigest()!=row.get('artifact_sha256') or len(payload)!=row.get('json_bytes'):
                raise ValueError('json_hash_or_size_mismatch')
            if self._fingerprint(path.stat())!=before:raise ValueError('artifact_changed')
            item['_artifact']=before
            data=json.loads(payload,parse_constant=lambda value:(_ for _ in ()).throw(ValueError('non_finite_json')))
            if not isinstance(data,dict):raise ValueError('json_not_object')
            meta=data.get('metadata') or {};features=data.get('features') or {};quality=data.get('quality') or {}
            if data.get('dataset_id')!=did or any(meta.get(k)!=row.get(k) for k in ['worker_id','process','scenario_id']):
                raise ValueError('json_identity_mismatch')
            if quality.get('status')!='complete' or quality.get('issues')!=[] or meta.get('quality_status')!='complete':
                raise ValueError('quality_inconsistent')
            frames=features.get('frame_count')
            zone_available=features.get('zone_id_available')
            if zone_available is None:
                # record-features-1.1 predates the availability flag; its actual
                # mapped Zone feature rows are the evidence, never the UI count.
                zone_available=any(isinstance(z,dict) and re.fullmatch(r'[1-9][0-9]*',str(z.get('zone_id','')))
                                   for z in features.get('zones',[]))
            if isinstance(frames,bool) or not isinstance(frames,int) or frames<2 or not zone_available:
                raise ValueError('motion_metadata_incomplete')
            item['json_status']='passed';item['frame_count']=frames
            source=(data.get('provenance') or {}).get('source_csv') or {}
            relative=source.get('relative_path')
            if not isinstance(relative,str):raise ValueError('source_link_missing')
            source_path=self.catalog._safe(self.catalog.sources,relative)
            before=source_path.stat()
            if source.get('bytes')!=before.st_size:raise ValueError('source_size_mismatch')
            if not re.fullmatch(r'[0-9a-f]{64}',str(source.get('sha256',''))):raise ValueError('source_hash_missing')
            if _sha(source_path)!=source['sha256']:raise ValueError('source_hash_mismatch')
            after=source_path.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('source_changed')
            item['_source']={'relative_path':relative,'bytes':after.st_size,'mtime_ns':after.st_mtime_ns}
            item['source_sha256']=source['sha256']
            item['source_status']='passed'
            if item['process'] not in {'spray','blasting','grinding'} or not item['scenario_id'].isdigit() or not 1<=int(item['scenario_id'])<=10:
                item['browser_status']='failed';item['reasons'].append('viewer_scenario_unsupported')
            if item['duplicate']:item['reasons'].append('duplicate_training_record')
        except (OSError,ValueError,KeyError,TypeError,AttributeError):
            import sys
            error=sys.exc_info()[1]
            allowed={'invalid_dataset_id','json_hash_or_size_mismatch','non_finite_json','json_not_object','artifact_changed',
                     'json_identity_mismatch','quality_inconsistent','motion_metadata_incomplete',
                     'source_link_missing','source_size_mismatch','source_hash_missing','source_hash_mismatch','source_changed'}
            reason=str(error) if str(error) in allowed else 'artifact_or_source_unavailable'
            item['reasons'].append(reason)
            if item['json_status']!='passed':item['json_status']='failed'
            if item['source_status']!='passed':item['source_status']='failed'
            item['browser_status']='skipped'

    def _audit_rows(self,rows):
        seen=set()
        try:
            for i,row in enumerate(rows):
                with self._guard:
                    item=copy.deepcopy(self._report['items'][i])
                # Large raw CSV hashes must not hold the status/read lock.
                self._check(row,item)
                with self._guard:
                    self._report['items'][i]=item
                    if item['source_status']=='passed':
                        key=(item['process'],item['source_sha256'])
                        if key in seen:
                            item['duplicate']=True;item['browser_status']='skipped'
                            item['reasons'].append('duplicate_source_content')
                        seen.add(key)
                    self._report['processed']=i+1
                    if i%20==0:self._save()
            with self._guard:
                self._report['phase']='awaiting_playback' if any(x['browser_status']=='pending' for x in self._report['items']) else 'finished'
                self._report['checked_at']=_now();self._save()
        finally:
            with self._guard:self._running=False
        return self.status()

    def audit(self):
        with self._guard:rows=self._prepare()
        return self._audit_rows(rows)

    def start(self):
        with self._guard:
            rows=self._prepare()
            threading.Thread(target=self._audit_rows,args=(rows,),daemon=True,name='training-construction-verification').start()
            return self._public()

    def record_playback(self,dataset_id,proof):
        with self._guard:
            report=self._load()
            if not report or self._running or report['catalog_sha256']!=self._catalog_digest():raise ValueError('verification_not_current')
            if proof.get('run_id')!=report['run_id']:raise ValueError('verification_run_changed')
            matches=[r for r in report['items'] if r['dataset_id']==dataset_id and not r['duplicate']]
            if len(matches)!=1:raise ValueError('dataset_not_unique')
            row=matches[0]
            if row['duplicate'] or row['json_status']!='passed' or row['source_status']!='passed':raise ValueError('dataset_not_eligible')
            path=self.catalog._safe(self.catalog.root,f'records/{dataset_id}.json')
            if proof.get('dataset_id')!=dataset_id or proof.get('artifact_sha256')!=row['artifact_sha256'] or _sha(path)!=row['artifact_sha256']:
                raise ValueError('artifact_changed')
            source=row.get('_source') or {}
            stat=self.catalog._safe(self.catalog.sources,source.get('relative_path','')).stat()
            if (stat.st_size,stat.st_mtime_ns)!=(source.get('bytes'),source.get('mtime_ns')):raise ValueError('source_changed')
            if proof.get('status')=='passed':
                if proof.get('session_id')!=row['session_id'] or proof.get('renderer')!='canvas-2d' or proof.get('frames_loaded',0)<2:
                    raise ValueError('invalid_playback_identity')
                samples=proof.get('samples',[])
                if len(samples)!=3 or {s.get('position') for s in samples}!={'start','middle','end'}:raise ValueError('incomplete_playback_scope')
                duration=proof.get('duration_seconds',0)
                if isinstance(duration,bool) or not isinstance(duration,(float,int)) or not math.isfinite(duration) or duration<=0:
                    raise ValueError('invalid_playback_duration')
                for s in samples:
                    nums=[s.get(k) for k in ['time_before','time_after','frame_before','frame_after','draw_before','draw_after']]
                    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in nums):raise ValueError('invalid_playback_sample')
                    if not s.get('renderer_ready') or s.get('context_lost') or nums[1]<=nums[0] or nums[3]<=nums[2] or nums[5]<=nums[4]:
                        raise ValueError('playback_did_not_advance')
                    low,high={'start':(0,.15),'middle':(.35,.65),'end':(.8,1)}[s['position']]
                    if not low<=nums[0]/duration<=high or nums[1]>duration+.05 or nums[3]>=proof['frames_loaded']:
                        raise ValueError('playback_scope_mismatch')
            elif proof.get('status')!='failed':raise ValueError('invalid_playback_status')
            row['browser_status']=proof['status'];row['browser_checked_at']=_now()
            row['playback']={k:proof.get(k) for k in ['renderer','frames_loaded','duration_seconds','selected_session_id','samples']}
            if proof['status']=='failed':row['reasons'].append('browser_playback_failed')
            else:row['reasons']=[reason for reason in row['reasons'] if reason!='browser_playback_failed']
            report['phase']='awaiting_playback' if any(x['browser_status']=='pending' for x in report['items']) else 'finished'
            report['checked_at']=_now();self._save();return self._public()


class PlaybackSample(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    position: str=Field(pattern='^(start|middle|end)$')
    time_before: float=Field(ge=0);time_after: float=Field(ge=0)
    frame_before: int=Field(ge=0);frame_after: int=Field(ge=0)
    draw_before: int=Field(ge=0);draw_after: int=Field(ge=0)
    renderer_ready: bool;context_lost: bool


class PlaybackProof(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    run_id: str=Field(pattern='^[0-9a-f]{32}$')
    artifact_sha256: str=Field(pattern='^[0-9a-f]{64}$')
    status: str=Field(pattern='^(passed|failed)$')
    dataset_id: str=Field(pattern='^[0-9a-f]{32}$')
    session_id: str=Field(max_length=200)
    renderer: str=Field(default='canvas-2d',max_length=30)
    frames_loaded: int=Field(default=0,ge=0)
    duration_seconds: float=Field(default=0,ge=0)
    selected_session_id: str | None=Field(default=None,max_length=200)
    samples: list[PlaybackSample]=Field(default_factory=list,max_length=3)


def create_training_verification_router(auth: DBAdminAuth,store: ConstructionVerification | None=None):
    store=store or ConstructionVerification(TrainingDataCatalog())
    router=APIRouter(prefix='/admin/api/training-verification',include_in_schema=False)

    @router.get('')
    def status():return store.status()

    @router.post('/start')
    def start(request: Request):
        auth.require(request,write=True)
        try:return store.start()
        except ValueError as error:raise HTTPException(409,str(error)) from error

    @router.post('/playback/{dataset_id}')
    def playback(dataset_id: str,proof: PlaybackProof,request: Request):
        auth.require(request,write=True)
        try:return store.record_playback(dataset_id,proof.model_dump())
        except (ValueError,OSError) as error:raise HTTPException(409,'검증 대상이나 재생 증빙을 확인해 주세요.') from error

    @router.get('/export.json')
    def export():
        return Response(json.dumps(store.status(),ensure_ascii=False,indent=2),media_type='application/json',
                        headers={'Content-Disposition':'attachment; filename="training-verification.json"','Cache-Control':'no-store'})

    @router.get('/export.csv')
    def export_csv():
        text=io.StringIO();fields=['dataset_id','dataset_name','worker_name','process','scenario_id','session_id','json_status','source_status','browser_status','duplicate','browser_checked_at','reasons']
        writer=csv.DictWriter(text,fieldnames=fields);writer.writeheader()
        for item in store.status()['items']:
            row={key:str(item.get(key,'')) for key in fields}
            writer.writerow({k:"'"+v if v.startswith(('=','+','-','@')) else v for k,v in row.items()})
        return Response('\ufeff'+text.getvalue(),media_type='text/csv; charset=utf-8',
                        headers={'Content-Disposition':'attachment; filename="training-verification.csv"','Cache-Control':'no-store'})
    return router
