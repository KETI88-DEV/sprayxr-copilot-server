'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
class Element {
  constructor(tag='div',doc){this.tagName=tag.toUpperCase();this.ownerDocument=doc;this.children=[];this.listeners={};this.attrs={};this.dataset={};this.style={};this.value='';this._text='';this.hidden=false;this.disabled=false;this.open=false;this.isConnected=true;this.classList={add(){},remove(){}};}
  set textContent(value){this._text=String(value);this.children=[];}get textContent(){return this._text+this.children.map(n=>n.textContent||'').join(' ');}
  append(...nodes){for(const node of nodes){node.parentNode=this;this.children.push(node);}}replaceChildren(...nodes){this.children=[];this._text='';this.append(...nodes);}
  setAttribute(k,v){this.attrs[k]=String(v);}getAttribute(k){return this.attrs[k]??null;}removeAttribute(k){delete this.attrs[k];if(k==='href')delete this.href;}
  addEventListener(k,fn){(this.listeners[k]||=[]).push(fn);}async emit(k,extra={}){for(const fn of this.listeners[k]||[])await fn({target:this,key:'',preventDefault(){},...extra});}
  focus(){this.ownerDocument.activeElement=this;}showModal(){this.open=true;}close(){this.open=false;void this.emit('close');}
  querySelectorAll(selector){return walk(this).filter(n=>selector==='[tabindex]'?n.attrs.tabindex!==undefined:['BUTTON','A','SELECT','INPUT'].includes(n.tagName));}
  closest(selector){if(selector==='[hidden]')return this.hidden?this:this.parentNode?.closest?.(selector)||null;return null;}
  getBoundingClientRect(){return{left:0,top:0,right:1000,bottom:900};}
}
const walk=node=>node.children.flatMap(child=>[child,...walk(child)]);
const settle=async()=>{for(let i=0;i<5;i++)await new Promise(resolve=>setImmediate(resolve));};
const metric=(key,label,value,unit='mm',status='available',group='distance_mm')=>({key,label,value,unit,status,group});
const fixture=id=>({dataset_id:id,metadata:{dataset_name:'시험 기록',worker_name:'훈련자 <img src=x>',worker_id:'trainee',role:'trainee',process:'spray',scenario_id:'3',session_id:'session-1'},
  pipeline:[{key:'csv',label:'CSV 원천',status:'stored',description:'저장된 분석에 연결된 원천',values:{frame_count:1000}}],
  source_preview:{status:'available',columns:['Timestamp','Distance <x>'],rows:[['0','150']],total_columns:20,sampled_rows:1,truncated:true,omitted_columns:['BodyPose']},
  summary:{frame_count:1000,duration_seconds:30,spray_interval_count:3,zone_count:2,metrics:[metric('distance_mm_mean','거리 (입력) 평균',150),metric('speed_mms_mean','속도 (입력) 평균',null,'mm/s','unavailable','speed_mms')],pose:{status:'unavailable',reason:'저장된 자세 분류 없음'},direction:{status:'unavailable',reason:'저장된 방향 분류 없음'}},
  zones:[{key:'1:2',map_num:1,zone_id:'2',sequence:{first_visit_index:1},metrics:[metric('distance_mm_mean','거리 (입력) 평균',0)],coating:{status:'unavailable'}},{key:'1:3',map_num:1,zone_id:'3',sequence:{first_visit_index:2},metrics:[],coating:{status:'unavailable'}}],
  coating:{basis:'stored',maps:[],reason:'맵 없음'},links:{json:'/admin/api/training-data/'+id,json_download:'/admin/api/training-data/'+id+'/download',motion:'/motion-viewer?dataset_id='+id},
  scope:{result_kind:'stored_record_features',notes:['저장된 특징 데이터 · 전체 원시 프레임 아님']},reconvert:{available:true,max_source_bytes:268435456}});
function setup({authenticated=false,fetcher}={}){
  const elements=new Map(),calls=[],events={},views=[],timers=new Map();let auth=authenticated,csrf='csrf-proof',timer=0;
  const shell=new Set(['tv-inspection','tv-inspection-title','tv-inspection-description','tv-inspection-close','tv-inspection-status','tv-inspection-content','td-detail-content']);
  const document={activeElement:null,hidden:false,getElementById(id){if(elements.has(id))return elements.get(id);for(const root of elements.values()){const found=walk(root).find(node=>node.id===id);if(found)return found;}if(!shell.has(id))return null;const node=new Element('div',document);node.id=id;elements.set(id,node);return node;},createElement:tag=>new Element(tag,document),createElementNS:(_,tag)=>new Element(tag,document),querySelector(){return null;}};
  document.body=new Element('body',document);for(const id of ['tv-inspection','tv-inspection-title','tv-inspection-description','tv-inspection-close','tv-inspection-status','tv-inspection-content'])document.getElementById(id);
  document.getElementById('tv-inspection').append(...['tv-inspection-title','tv-inspection-description','tv-inspection-close','tv-inspection-status','tv-inspection-content'].map(id=>document.getElementById(id)));
  const window={location:new URL('http://localhost/admin/training-data?view=verification'),addEventListener:(name,fn)=>events[name]=fn,
    TrainingDataPresentation:{coating:(data,options)=>{views.push({kind:'coating',data,options});const node=new Element('section',document);node.textContent='저장된 맵 단위 도막';return node;},json:data=>{views.push({kind:'json',data});const node=new Element('section',document);node.textContent='실제 저장 JSON';return node;}}};
  const fetch=async(url,options={})=>{calls.push({url,options});let value;
    if(url==='/admin/api/manage/status')value={authenticated:auth,csrf_token:auth?csrf:'',expires_at:Date.now()/1000+600};
    else value=await fetcher?.(url,options);
    return{ok:value?.http_status?false:true,status:value?.http_status||200,headers:{get:()=>null},json:async()=>value};};
  const script=path.join(__dirname,'../static/training_inspection.js');
  vm.runInNewContext(fs.existsSync(script)?fs.readFileSync(script,'utf8'):'',{document,window,fetch,URL,URLSearchParams,AbortController,Date,Intl,console,setTimeout:(fn)=>{timers.set(++timer,fn);return timer;},clearTimeout:id=>timers.delete(id)});
  assert.equal(typeof window.TrainingInspection?.open,'function','inspection dialog must provide the record-opening behavior');
  return{get:id=>document.getElementById(id),window,document,calls,views,events,timers,setAuth:value=>auth=value,setCsrf:value=>csrf=value,trigger:new Element('button',document)};
}
const A='a'.repeat(32),B='b'.repeat(32),J='c'.repeat(32);

test('public inspection shows stored source, honest missing metrics and actual worker role without a POST',async()=>{
  const app=setup({fetcher:async()=>fixture(A)});await app.window.TrainingInspection.open(A,app.trigger);
  const text=app.get('tv-inspection-content').textContent;
  assert.match(text,/저장된/);assert.match(text,/훈련자 <img src=x>/);assert.match(text,/미기록|없음/);assert.doesNotMatch(text,/전문가 기준 데이터 1,201/);
  assert.match(text,/Timestamp/);assert.match(text,/150.*mm/);assert.equal(app.get('ti-reconvert').hidden,true);assert.ok(!app.calls.some(x=>x.options.method==='POST'));
  assert.equal(walk(app.get('tv-inspection-content')).some(node=>node.tagName==='IMG'&&node.src==='x'),false);
});
test('Zone detail keeps zero as data and labels first visit separately from an expert recommendation',async()=>{
  const app=setup({fetcher:async()=>fixture(A)});await app.window.TrainingInspection.open(A,app.trigger);
  app.get('ti-zone').value='1:2';await app.get('ti-zone').emit('change');
  const text=app.get('ti-analysis').textContent;assert.match(text,/Zone 2/);assert.match(text,/0.*mm/);assert.match(text,/첫 방문.*1/);assert.doesNotMatch(text,/권장 순서|전문가 순서/);
});
test('rapid record changes abort older reads and ignore a late stale result',async()=>{
  const pending=[];const app=setup({fetcher:(url,options)=>new Promise(resolve=>pending.push({url,options,resolve}))});
  const first=app.window.TrainingInspection.open(A,app.trigger);await settle();const second=app.window.TrainingInspection.open(B,app.trigger);await settle();
  assert.equal(pending[0].options.signal.aborted,true);pending[1].resolve({...fixture(B),metadata:{dataset_name:'두 번째'}});await second;
  pending[0].resolve({...fixture(A),metadata:{dataset_name:'오래된 첫 번째'}});await first;
  assert.equal(app.get('tv-inspection-title').textContent,'두 번째');
});
test('closing the dedicated modal aborts its read, restores focus and leaves the library detail intact',async()=>{
  const app=setup({fetcher:async()=>fixture(A)});app.get('td-detail-content').textContent='기존 상세 유지';
  await app.window.TrainingInspection.open(A,app.trigger);await app.get('tv-inspection-close').emit('click');await settle();
  assert.equal(app.get('tv-inspection').open,false);assert.equal(app.document.activeElement,app.trigger);assert.equal(app.get('td-detail-content').textContent,'기존 상세 유지');
});
test('raw JSON is lazy and the existing coating/JSON presentation is reused with separate IDs',async()=>{
  const app=setup({fetcher:async url=>url.endsWith('/inspection')?fixture(A):{dataset_id:A,features:{distance:123}}});
  await app.window.TrainingInspection.open(A,app.trigger);assert.equal(app.calls.filter(x=>x.url==='/admin/api/training-data/'+A).length,0);
  assert.equal(app.views[0].kind,'coating');assert.equal(app.views[0].options.idPrefix,'tv-inspection');
  await app.get('ti-json-load').emit('click');assert.equal(app.views.at(-1).kind,'json');assert.equal(app.views.at(-1).data.features.distance,123);
  assert.equal(app.get('ti-motion').href,'/motion-viewer?dataset_id='+A);
});
test('admin reconversion uses CSRF, shows only real returned steps, and logout retains public analysis',async()=>{
  const job={dataset_id:A,job_id:J,status:'completed',steps:[{key:'read',label:'CSV 읽기',status:'completed',detail:'실제 1000행'}],result_available:true,download_url:`/admin/api/training-data/${A}/reconvert/${J}/download`};
  const app=setup({authenticated:true,fetcher:async url=>url.endsWith('/inspection')?fixture(A):job});
  await app.window.TrainingInspection.open(A,app.trigger);await app.get('ti-reconvert').emit('click');
  const post=app.calls.find(x=>x.options.method==='POST');assert.equal(post.options.headers['X-DB-CSRF'],'csrf-proof');assert.equal(post.options.credentials,'same-origin');
  assert.match(app.get('ti-job-details').textContent,/실제 1000행/);assert.ok(app.get('ti-job-download').href.endsWith('/download'));
  app.setAuth(false);await app.window.TrainingInspection.refreshAuth();
  assert.equal(app.get('ti-reconvert').hidden,true);assert.equal(app.get('ti-job-details').textContent,'');assert.match(app.get('ti-analysis').textContent,/150/);
  assert.ok(!app.get('ti-job-download').href);
});
test('an unavailable source remains reviewable and is never portrayed as a running extraction',async()=>{
  const data=fixture(A);data.source_preview={status:'unavailable',reason:'연결 CSV 없음'};data.reconvert={available:false,reason:'원천 없음'};
  const app=setup({authenticated:true,fetcher:async()=>data});await app.window.TrainingInspection.open(A,app.trigger);
  assert.match(app.get('tv-inspection-content').textContent,/연결 CSV 없음/);assert.match(app.get('ti-analysis').textContent,/150/);assert.equal(app.get('ti-reconvert').disabled,true);
  assert.ok(!app.calls.some(x=>x.options.method==='POST'));
});
test('rapid clicks start only one real reconversion while administrator status is checked',async()=>{
  const job={dataset_id:A,job_id:J,status:'completed',steps:[],result_available:false};
  const app=setup({authenticated:true,fetcher:async url=>url.endsWith('/inspection')?fixture(A):job});
  await app.window.TrainingInspection.open(A,app.trigger);
  await Promise.all([app.get('ti-reconvert').emit('click'),app.get('ti-reconvert').emit('click')]);
  assert.equal(app.calls.filter(row=>row.options.method==='POST').length,1);
});
test('a whole-session classification is not relabeled as a selected Zone classification',async()=>{
  const data=fixture(A);data.summary.pose={status:'available',classification:'전체 회차 분류 A'};
  const app=setup({fetcher:async()=>data});await app.window.TrainingInspection.open(A,app.trigger);
  assert.match(app.get('ti-analysis').textContent,/전체 회차 분류 A/);
  app.get('ti-zone').value='1:2';await app.get('ti-zone').emit('change');
  assert.doesNotMatch(app.get('ti-analysis').textContent,/전체 회차 분류 A/);assert.match(app.get('ti-analysis').textContent,/분류 결과가 저장되지/);
});

test('statistics are compact and chart details do not repeat every value in cards',async()=>{
  const data=fixture(A);data.metadata.role='worker';data.metadata.qualification='등록 근거 미확인';data.summary.metrics=[...['mean','median','p25','p75'].map((stat,index)=>metric('distance_mm_'+stat,'거리 (입력) '+stat,index+1))];
  data.summary.pose={status:'unavailable',reason:'stored_features_do_not_include_pose_scores'};
  const app=setup({fetcher:async()=>data});await app.window.TrainingInspection.open(A,app.trigger);
  const content=app.get('ti-analysis'),details=walk(content).find(n=>n.tagName==='DETAILS');
  assert.ok(details,'all-stat charts belong to expandable details');assert.equal(details.open,false);
  const cards=walk(content).filter(n=>n.tagName==='DL').flatMap(n=>n.children);
  assert.equal(cards.length,6,'four session counts and only mean/median statistic cards');
  assert.match(app.get('tv-inspection-content').textContent,/작업자.*등록 근거 미확인/);assert.doesNotMatch(content.textContent,/stored_features_do_not_include/);
});

test('non-spray Zone detail explicitly has no coating thickness assessment',async()=>{
  const data=fixture(A);data.metadata.process='grinding';data.coating={status:'not_applicable',maps:[],reason:'coating_not_applicable_to_process'};data.zones[0].coating={status:'not_applicable',metrics:[]};
  const app=setup({fetcher:async()=>data});await app.window.TrainingInspection.open(A,app.trigger);app.get('ti-zone').value='1:2';await app.get('ti-zone').emit('change');
  assert.match(app.get('ti-analysis').textContent,/도막 두께 판정 대상이 아닙니다/);assert.equal(app.views.filter(v=>v.kind==='coating').length,0);
});

test('Tab stays within the modal and Escape restores the opening control',async()=>{
  const app=setup({fetcher:async()=>fixture(A)});await app.window.TrainingInspection.open(A,app.trigger);
  const dialog=app.get('tv-inspection'),nodes=dialog.querySelectorAll('button,a[href],input,select,textarea,[tabindex]').filter(n=>!n.hidden&&!n.disabled&&!n.closest('[hidden]'));
  let prevented=false;nodes.at(-1).focus();await dialog.emit('keydown',{key:'Tab',preventDefault(){prevented=true;}});assert.equal(prevented,true);assert.equal(app.document.activeElement,nodes[0]);
  await dialog.emit('keydown',{key:'Tab',shiftKey:true});assert.equal(app.document.activeElement,nodes.at(-1));
  await dialog.emit('cancel');assert.equal(dialog.open,false);assert.equal(app.document.activeElement,app.trigger);
});

test('a submitted job survives closing and is recovered only after its record is rendered again',async()=>{
  let resolvePost,reads=0;const job={dataset_id:A,job_id:J,status:'completed',steps:[],result_available:false};
  const app=setup({authenticated:true,fetcher:async(url,options)=>{if(url.endsWith('/inspection'))return fixture(A);if(options.method==='POST')return new Promise(resolve=>resolvePost=resolve);reads++;return job;}});
  await app.window.TrainingInspection.open(A,app.trigger);const submitted=app.get('ti-reconvert').emit('click');await settle();app.window.TrainingInspection.close();resolvePost(job);await submitted;
  assert.equal(reads,0);await app.window.TrainingInspection.open(A,app.trigger);assert.equal(reads,1);assert.match(app.get('ti-job-details').textContent,/실제 재변환 완료/);
});

test('a job never exposes an external result URL and worker failure is not labeled as corrupt data',async()=>{
  let job={dataset_id:A,job_id:J,status:'completed',steps:[],result_available:true,download_url:'https://evil.invalid/result'};
  const app=setup({authenticated:true,fetcher:async url=>url.endsWith('/inspection')?fixture(A):job});await app.window.TrainingInspection.open(A,app.trigger);await app.get('ti-reconvert').emit('click');
  assert.equal(app.get('ti-job-download').hidden,true);assert.ok(!app.get('ti-job-download').href);
  job={...job,status:'failed',reason:'extraction_worker_failed'};await app.window.TrainingInspection.refreshAuth();
  assert.match(app.get('ti-job-details').textContent,/재변환 작업을 완료하지 못했습니다/);assert.doesNotMatch(app.get('ti-job-details').textContent,/데이터 손상|메모리 부족|extraction_worker_failed/);
});

test('an expired job releases the start control without losing public stored analysis',async()=>{
  let expired=false;const job={dataset_id:A,job_id:J,status:'running',steps:[],result_available:false};
  const app=setup({authenticated:true,fetcher:async(url,options)=>url.endsWith('/inspection')?fixture(A):expired&&options.method!=='POST'?{http_status:404,detail:'job_not_found'}:job});
  await app.window.TrainingInspection.open(A,app.trigger);await app.get('ti-reconvert').emit('click');assert.equal(app.get('ti-reconvert').disabled,true);
  expired=true;await app.window.TrainingInspection.refreshAuth();assert.equal(app.get('ti-reconvert').disabled,false);assert.match(app.get('ti-analysis').textContent,/150/);
});

test('a late aborted job error cannot erase the new record or its administrator state',async()=>{
  let finishPoll;const job={dataset_id:A,job_id:J,status:'running',steps:[],result_available:false};
  const app=setup({authenticated:true,fetcher:async(url,options)=>{if(url.endsWith('/inspection'))return fixture(url.includes(B)?B:A);if(options.method==='POST')return job;return new Promise(resolve=>finishPoll=resolve);}});
  await app.window.TrainingInspection.open(A,app.trigger);const started=app.get('ti-reconvert').emit('click');await settle();await app.window.TrainingInspection.open(B,app.trigger);
  finishPoll({http_status:401,detail:'expired'});await started;
  assert.equal(app.get('ti-reconvert').hidden,false);assert.equal(app.get('tv-inspection-title').textContent,'시험 기록');assert.match(app.get('ti-analysis').textContent,/150/);
});

test('focus returns to the equivalent action when periodic table refresh replaced its original button',async()=>{
  const app=setup({fetcher:async()=>fixture(A)}),replacement=new Element('button',app.document);app.trigger.dataset.verificationAction='inspection-action';
  let selector;app.document.querySelector=value=>{selector=value;return replacement;};await app.window.TrainingInspection.open(A,app.trigger);app.trigger.isConnected=false;app.window.TrainingInspection.close();
  assert.match(selector,/inspection-action/);assert.equal(app.document.activeElement,replacement);
});

test('a late POST authorization failure cannot clear a newer authenticated session',async()=>{
  let finishPost;const app=setup({authenticated:true,fetcher:async(url,options)=>options.method==='POST'?new Promise(resolve=>finishPost=resolve):fixture(url.includes(B)?B:A)});
  await app.window.TrainingInspection.open(A,app.trigger);const submitted=app.get('ti-reconvert').emit('click');await settle();app.window.TrainingInspection.close();
  app.setCsrf('new-session-csrf');await app.window.TrainingInspection.open(B,app.trigger);finishPost({http_status:401,detail:'expired'});await submitted;
  assert.equal(app.get('ti-reconvert').hidden,false);assert.equal(app.get('ti-job-details').textContent,'');assert.match(app.get('ti-analysis').textContent,/150/);
});

test('an authorization failure from the current POST still clears its administrator controls',async()=>{
  const app=setup({authenticated:true,fetcher:async(url,options)=>options.method==='POST'?{http_status:403,detail:'expired'}:fixture(A)});
  await app.window.TrainingInspection.open(A,app.trigger);await app.get('ti-reconvert').emit('click');assert.equal(app.get('ti-reconvert').hidden,true);assert.match(app.get('ti-analysis').textContent,/150/);
});
