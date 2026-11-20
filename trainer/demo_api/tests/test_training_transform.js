'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
class Element {
  constructor(tag='div',doc){this.tagName=tag.toUpperCase();this.ownerDocument=doc;this.children=[];this.listeners={};this.attrs={};this.dataset={};this.style={setProperty(){}};this._text='';this.value='';this.hidden=false;this.disabled=false;}
  set textContent(v){this._text=String(v);this.children=[];}get textContent(){return this._text+this.children.map(c=>c.textContent||'').join(' ');}
  append(...children){this.children.push(...children);for(const c of children)c.parentNode=this;}replaceChildren(...children){this._text='';this.children=[];this.append(...children);}
  setAttribute(k,v){this.attrs[k]=String(v);}getAttribute(k){return this.attrs[k];}addEventListener(k,fn){(this.listeners[k]||=[]).push(fn);}
  async emit(k){for(const fn of this.listeners[k]||[])await fn({target:this});}scrollIntoView(){}remove(){this.parentNode.children=this.parentNode.children.filter(n=>n!==this);}
}
const walk=node=>[node,...node.children.flatMap(walk)],A='a'.repeat(32),B='b'.repeat(32);
const inspection=id=>({dataset_id:id,metadata:{dataset_name:'원본 <script>',worker_name:'전형기',process:'spray',scenario_id:'4'},
 source_preview:{status:'available',columns:['시간(초)','분사','Zone ID','Ray Texture Map Number','Distance (Input)','Ray Point X','Ray Point Y','Ray Point Z'],rows:[['0','Off','1','1','0','0','1','2'],['1','On','2','1','123','1','1','2']],sampled_rows:2},
 summary:{frame_count:1200,zone_count:2,spray_interval_count:4},
 zones:[{key:'1:1',map_num:1,zone_id:'1',sequence:{first_visit_index:1},metrics:[{key:'distance_mm_mean',value:0,status:'available',unit:'mm',label:'거리 평균'}]},
 {key:'2:1',map_num:2,zone_id:'1',sequence:{first_visit_index:2},metrics:[{key:'distance_mm_mean',value:320,status:'available',unit:'mm',label:'거리 평균'}]}],
 coating:{maps:[{map_num:1,image_url:`/admin/api/training-data/${id}/images/0`,thickness_image_url:`/admin/api/training-data/${id}/images/0?view=thickness`,summary:{mean_um:171},zones:[{zone_id:'1',hex_color:'#FF0000',summary:{mean_um:0}}]},
 {map_num:2,image_url:`/admin/api/training-data/${id}/images/1`,summary:{mean_um:190},zones:[{zone_id:'1',summary:{mean_um:189}}]}]}});
const artifact=id=>({dataset_id:id,metadata:{process:'spray'},features:{frame_count:1200,sequence_count:4,zone_order:[{map_num:1,zone_id:'1'},{map_num:2,zone_id:'1'}],zones:[{map_num:1,zone_id:'1',distance_mm:{count:10,mean:0}},{map_num:2,zone_id:'1',distance_mm:{count:20,mean:320}}],sequence_heatmaps:[{data:'BIG_BASE64'}]},coating:inspection(id).coating});
function setup(fetcher){
 const elements=new Map(),calls=[],events={},scenes=[];
 const doc={getElementById(id){return elements.get(id)||[...elements.values()].flatMap(walk).find(n=>n.id===id)||null;},createElement:tag=>new Element(tag,doc),createElementNS:(_,tag)=>new Element(tag,doc)};
 for(const id of ['tt-root','tt-record','tt-status','tt-retry','tt-zones','tt-source','tt-visual','tt-json','tt-meta','tt-actions','tt-explanation','tt-view-zone','tt-view-metrics','tt-view-coating']){const n=new Element(id==='tt-record'?'select':'div',doc);n.id=id;elements.set(id,n);}
 const window={location:new URL('http://localhost/admin/training-data?view=verification'),addEventListener:(name,fn)=>events[name]=fn,
 TrainingTransformScene:{create(host,options){const scene={options,selected:[],destroyed:false,selectZone(v){this.selected.push(v);},destroy(){this.destroyed=true;}};scenes.push(scene);return scene;}}};
 const fetch=async(url,options={})=>{calls.push({url,options});const result=await fetcher(url,options);return{ok:true,status:200,headers:{get:()=>null},json:async()=>result};};
 const script=path.join(__dirname,'../static/training_transform.js');vm.runInNewContext(fs.existsSync(script)?fs.readFileSync(script,'utf8'):'',{document:doc,window,fetch,AbortController,URL,URLSearchParams,Intl,console,TextDecoder,TextEncoder});
 assert.equal(typeof window.TrainingTransform?.show,'function','transformation example controller must exist');
 return{get:id=>doc.getElementById(id),window,calls,scenes,events};
}
const normal=async url=>url.endsWith('/inspection')?inspection(url.includes(B)?B:A):url.includes('?')?{items:[{dataset_id:A,dataset_name:'예시',zone_count:2,heightmap_count:1}]}:artifact(url.includes(B)?B:A);
test('activation reads a real record, never posts or eagerly loads motion, and preserves escaped source text',async()=>{
 const app=setup(normal);assert.equal(app.calls.length,0);await app.window.TrainingTransform.activate(true);
 assert.match(app.get('tt-status').textContent,/원본 <script>/);assert.match(app.get('tt-source').textContent,/Ray Point/);
 assert.ok(app.calls.every(c=>!c.options.method||c.options.method==='GET'));assert.ok(!app.calls.some(c=>c.url.includes('/motion-viewer/')));
 assert.equal(walk(app.get('tt-source')).some(n=>n.tagName==='SCRIPT'),false);
});
test('selected Zone uses map AND Zone identity in saved JSON and actual statistics, retaining zero',async()=>{
 const app=setup(normal);await app.window.TrainingTransform.show(A);await app.get('tt-view-metrics').emit('click');
 const buttons=walk(app.get('tt-zones')).filter(n=>n.tagName==='BUTTON');await buttons.find(n=>n.dataset.zoneKey==='1:1').emit('click');
 assert.match(app.get('tt-visual').textContent,/0\s*mm/);assert.match(app.get('tt-json').textContent,/"mean": 0/);assert.doesNotMatch(app.get('tt-json').textContent,/"mean": 320/);
 await buttons.find(n=>n.dataset.zoneKey==='2:1').emit('click');assert.match(app.get('tt-json').textContent,/"mean": 320/);assert.doesNotMatch(app.get('tt-json').textContent,/"mean": 0/);
});
test('the coating image is a same-record map with separate Zone statistics, not expert aggregation',async()=>{
 const app=setup(normal);await app.window.TrainingTransform.show(A);await app.get('tt-view-coating').emit('click');
 await walk(app.get('tt-zones')).find(n=>n.dataset.zoneKey==='1:1').emit('click');
 const images=walk(app.get('tt-visual')).filter(n=>n.tagName==='IMG');assert.equal(images.length,2);assert.ok(images.every(n=>n.src.includes(A+'/images/0')));
 assert.match(app.get('tt-visual').textContent,/맵 1 전체/);assert.match(app.get('tt-visual').textContent,/0\s*µm/);assert.doesNotMatch(app.get('tt-json').textContent,/BIG_BASE64/);
});
test('late results of an aborted previous record never overwrite a new record',async()=>{
 let finish;const app=setup(async url=>url.includes(A)?new Promise(resolve=>finish=resolve):normal(url));
 const first=app.window.TrainingTransform.show(A);const second=app.window.TrainingTransform.show(B);await second;
 finish(inspection(A));await first;assert.match(app.get('tt-actions').textContent,/JSON/);assert.equal(app.get('tt-record').value,B);
 assert.equal(app.calls[0].options.signal.aborted,true);
});
test('missing raw JSON or missing data stays explicit, without simulated transformation or classification',async()=>{
 const app=setup(async url=>{if(!url.endsWith('/inspection'))throw new Error('too large');return{...inspection(A),source_preview:{status:'unavailable'},zones:[],coating:{maps:[]}};});
 await app.window.TrainingTransform.show(A);await app.get('tt-view-coating').emit('click');
 assert.match(app.get('tt-json').textContent,/다운로드|불러오지/);assert.match(app.get('tt-source').textContent,/없|못|불가/);assert.match(app.get('tt-visual').textContent,/없|못|미기록/);
 assert.ok(!app.calls.some(c=>c.options.method==='POST'));
});
test('3D is explicit and its requests are destroyed when the example changes or leaves the tab',async()=>{
 const app=setup(normal);await app.window.TrainingTransform.show(A);const button=walk(app.get('tt-visual')).find(n=>n.id==='tt-load-scene');assert.ok(button);await button.emit('click');assert.equal(app.scenes.length,1);
 await app.window.TrainingTransform.show(B);assert.equal(app.scenes[0].destroyed,true);await app.window.TrainingTransform.activate(false);
});
test('returning to the exam tab restores its 3D loading control after preview cleanup',async()=>{
 const app=setup(normal);await app.window.TrainingTransform.activate(true);
 await walk(app.get('tt-visual')).find(n=>n.id==='tt-load-scene').emit('click');
 await app.window.TrainingTransform.activate(false);assert.equal(app.scenes[0].destroyed,true);
 await app.window.TrainingTransform.activate(true);
 assert.ok(walk(app.get('tt-visual')).some(n=>n.id==='tt-load-scene'),'the destroyed scene must be replaceable without changing the record');
});
