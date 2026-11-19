'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
class Element {
  constructor(tag='div'){this.tagName=tag;this.children=[];this.events={};this.attributes={};this.dataset={};this.style={};this.value='';this.textContent='';this.hidden=false;this.disabled=false;}
  append(...nodes){this.children.push(...nodes);} replaceChildren(...nodes){this.children=nodes;this.textContent='';}
  addEventListener(event,fn){this.events[event]=fn;}setAttribute(key,value){this.attributes[key]=value;}focus(){Element.active=this;}
}
const text=node=>[node.textContent,...node.children.map(text)].filter(Boolean).join(' ');
const item=(id,status='pending')=>({dataset_id:id.repeat(32),dataset_name:'기록 '+id,worker_name:'작업자 '+id,process:'spray',scenario_id:'4',session_id:'session-'+id,json_status:'passed',source_status:'passed',browser_status:status,duplicate:false,reasons:[]});
async function page({authenticated=false,report,view='verification'}={}){
  report ||= {phase:'not_started',target:1000,stale:false,definition:'한 훈련 기록 = 1건',scope:'시작·중간·끝 재생, 전체 연속 재생 아님',summary:{registered:8,json_passed:0,source_passed:0,browser_passed:0,browser_failed:0,duplicates:0,valid:0,pending:8,goal_met:false},items:[]};
  const nodes=new Map(),requests=[],events={};const get=id=>{if(!nodes.has(id))nodes.set(id,new Element());return nodes.get(id);};
  for(const id of ['tv-status-filter','tv-process-filter'])get(id).value='all';
  Element.active=null;
  const document={get activeElement(){return Element.active;},getElementById:get,createElement:tag=>new Element(tag),hidden:false,addEventListener:(name,fn)=>events[name]=fn};
  const window={location:{href:'http://localhost/admin/training-data?view='+view},history:{replaceState(){}},addEventListener:(name,fn)=>events[name]=fn};
  const fetch=async(url,options={})=>{requests.push({url,options});return {ok:true,status:200,json:async()=>url.includes('/manage/status')?{authenticated,csrf_token:'csrf-test',expires_at:Date.now()/1000+3600}:report};};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../static/training_verification.js'),'utf8'),{document,window,fetch,URL,URLSearchParams,AbortController,Date,setTimeout:()=>1,clearTimeout(){}});
  await new Promise(resolve=>setImmediate(resolve));
  return {get,requests,events,document};
}
test('unverified registration is not presented as browser success or valid construction',async()=>{
  const {get,requests}=await page();
  assert.equal(get('tv-kpi-registered').textContent,'8');assert.equal(get('tv-kpi-valid').textContent,'0');
  assert.match(get('tv-phase').textContent,/미검증|시작 전/);assert.match(text(get('tv-items')),/검증.*시작/);
  assert.equal(get('tv-start').hidden,true);assert.ok(!requests.some(row=>row.options.method==='POST'));
});
test('only authenticated administrators see and invoke the bounded JSON/source start action',async()=>{
  const {get,requests}=await page({authenticated:true});
  assert.equal(get('tv-start').hidden,false);await get('tv-start').events.click();
  const request=requests.find(row=>row.url.endsWith('/start'));
  assert.equal(request.options.method,'POST');assert.equal(request.options.headers['X-DB-CSRF'],'csrf-test');
  assert.equal(request.options.credentials,'same-origin');
});
test('failed/pending filters keep honest states and link the exact dataset JSON and playback',async()=>{
  const report={phase:'awaiting_playback',run_id:'run',target:1000,stale:false,summary:{registered:3,valid:1,pending:1},items:[item('a','passed'),{...item('b','failed'),reasons:['browser_playback_failed']},item('c')]};
  const {get}=await page({report});get('tv-status-filter').value='failed';get('tv-status-filter').events.change();
  assert.match(text(get('tv-items')),/기록 b/);assert.doesNotMatch(text(get('tv-items')),/기록 a|기록 c/);
  const links=get('tv-items').children[0].children.at(-1).children;
  assert.ok(links.some(row=>row.href==='/motion-viewer?dataset_id='+ 'b'.repeat(32)));
  assert.ok(links.some(row=>row.href==='/admin/api/training-data/'+ 'b'.repeat(32)));
  get('tv-status-filter').value='pending';get('tv-status-filter').events.change();assert.match(text(get('tv-items')),/기록 c.*미검증/);
});
test('stale evidence is visible and never reports the target as achieved',async()=>{
  const {get}=await page({report:{phase:'finished',target:1000,stale:true,summary:{registered:1000,valid:0,snapshot_valid:1000,goal_met:false},items:[]}});
  assert.equal(get('tv-kpi-valid').textContent,'0');assert.equal(get('tv-stale').hidden,false);assert.match(get('tv-goal').textContent,/미달|미충족/);
});
test('stale evidence explains the server-confirmed artifact or source drift reason',async()=>{
  const reason='검증 후 연결 원천 파일이 변경되었습니다. 다시 점검해 주세요.';
  const {get}=await page({report:{phase:'finished',target:1000,stale:true,stale_reason:reason,summary:{registered:1,valid:0,goal_met:false},items:[]}});
  assert.equal(get('tv-stale').textContent,reason);
});
test('refreshing evidence keeps keyboard focus on the same dataset action',async()=>{
  const report={phase:'awaiting_playback',target:1000,summary:{registered:1,pending:1},items:[item('a')]};
  const {get,document}=await page({report});
  const original=get('tv-items').children[0].children.at(-1).children[1];original.focus();
  await get('tv-refresh').events.click();
  const replacement=get('tv-items').children[0].children.at(-1).children[1];
  assert.notEqual(replacement,original);assert.equal(document.activeElement,replacement);assert.equal(replacement.textContent,'3D 재생');
});
test('library and verification tabs use real tab semantics and keyboard navigation',async()=>{
  const {get}=await page({view:'library'});assert.equal(get('tv-root').hidden,true);
  await get('td-library-tab').events.keydown({key:'ArrowRight',preventDefault(){}});
  assert.equal(get('tv-root').hidden,false);assert.equal(get('td-library-panel').hidden,true);assert.equal(get('td-verification-tab').attributes['aria-selected'],'true');
});
