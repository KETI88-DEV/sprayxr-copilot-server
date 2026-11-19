'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../motion_viewer.py'), 'utf8');
const functions = source.split('\n').filter(line => /^(function (option|currentWorker|currentProcess|currentMode|fillWorkers|fillProcesses|fillModes|fillSessions|sessionOptions|applySelection|playbackProbe)\(|window\.__motionPlaybackProbe=)/.test(line)).join('\n');
function selector() {
  return {value: '', style: {}, dataset: {}, children: [], replaceChildren(...rows) { this.children=rows; this.value=String(rows[0]?.value || ''); }};
}
function viewer() {
  const elements=Object.fromEntries(['worker','process','mode','session','status'].map(id=>[id,selector()]));
  const mode={mode_id:4,session_count:2,sessions:['record-a','record-b']};
  const processes=[{process:'spray',label:'스프레이',modes:[mode]}];
  let loads=0;
  const context={catalog:{workers:[{worker_id:'trainee',display_name:'김경호',role:'trainee',processes},{worker_id:'jeon_hyeonggi',display_name:'전형기',role:'expert',processes}]},
    $:id=>elements[id],document:{createElement:()=>({style:{},dataset:{}})},refreshSessionLabels(){},loadMotion(){loads++;},window:{},
    data:null,index:0,playheadSeconds:0,playing:false,renderedFrameIndex:null,rendererFrameCount:0,ctx:{isContextLost:()=>false},canvas:{width:800,height:500}};
  vm.createContext(context);vm.runInContext(functions,context);
  return {context,elements,run:code=>vm.runInContext(code,context),loads:()=>loads};
}
test('expert worker stays first, labelled and distinguished even without native option colors',()=>{
  const view=viewer();view.run('fillWorkers()');
  const [first]=view.elements.worker.children;
  assert.equal(first.value,'jeon_hyeonggi');assert.equal(first.textContent,'전형기 · 전문가 기준');
  assert.equal(view.elements.worker.dataset.referenceRole,'expert');assert.ok(view.elements.worker.style.color);
  assert.notEqual(first.style.color,view.elements.worker.children[1].style.color,'native menus that honor option styles distinguish the expert');
});
test('trainees select actual records and never a misleading combined expert baseline',()=>{
  const view=viewer();view.run("fillWorkers('trainee');fillProcesses()");
  assert.deepEqual(view.elements.session.children.map(row=>row.value),['record-a','record-b']);
  assert.ok(!view.elements.session.children.some(row=>/종합 기준/.test(row.textContent)));
  assert.equal(view.loads(),1);
});
test('resolved dataset selection cascades through process, mode and exact session before loading',()=>{
  const view=viewer();view.run("applySelection({dataset_id:'exact-id',worker_id:'trainee',process:'spray',mode_id:4,session_id:'record-b'})");
  assert.equal(view.elements.worker.value,'trainee');assert.equal(view.elements.process.value,'spray');assert.equal(view.elements.mode.value,'4');assert.equal(view.elements.session.value,'record-b');assert.equal(view.loads(),1);
});
test('a stale or invalid dataset link never silently loads another recording',()=>{
  const view=viewer();
  assert.throws(()=>view.run("applySelection({dataset_id:'exact-id',worker_id:'trainee',process:'blasting',mode_id:4,session_id:'record-b'})"),/재생 목록과 일치하지 않습니다/);
  assert.equal(view.loads(),0);
});
test('playback probe reports actual renderer state without inferring that loaded frames were drawn',()=>{
  const view=viewer();view.context.data={dataset_id:'dataset-id',selected_session_id:'record-b',frames:[{},{}],duration_seconds:2};
  const before=view.run('window.__motionPlaybackProbe()');
  assert.equal(before.framesLoaded,2);assert.equal(before.rendererFrameCount,0);assert.equal(before.renderedFrameIndex,null);assert.equal(before.rendererReady,false);
  view.context.renderedFrameIndex=1;view.context.rendererFrameCount=3;view.context.playheadSeconds=1.2;
  const after=view.run('window.__motionPlaybackProbe()');
  assert.equal(after.datasetId,'dataset-id');assert.equal(after.currentTime,1.2);assert.equal(after.renderedFrameIndex,1);assert.equal(after.rendererReady,true);assert.equal(after.renderer,'canvas-2d');assert.doesNotMatch(JSON.stringify(after),/source_path|source_csv|root/);
});

test('only completed scene draws advance the playback probe renderer counter',()=>{
  const view=viewer(), context=view.context;
  Object.assign(context, {data:{frames:[{t:0}],statistics:[{}],paint_trace:[]}, referencePreview:false,viewCenter:null,
    interpolatedFrame:()=>({frame:{t:0,joints:{}},index:0}),drawFloorGrid(){},referencePointCloud(){},drawZoneFlowGuides(){},drawSkeletonView(){},updateZonePanel(){},
    dispatchEvent(){},CustomEvent:class{constructor(type,options){this.type=type;this.detail=options.detail;}}});
  context.ctx.clearRect=()=>{};context.canvas.getBoundingClientRect=()=>({width:800,height:500});
  for(const id of ['objectMesh','zoneSurface','overlayFrame'])view.elements[id]=selector();
  const draw=source.slice(source.indexOf('function draw(){'),source.indexOf('\nlet atEnd='));
  view.run(draw);view.run('draw()');
  const rendered=view.run('window.__motionPlaybackProbe()');
  assert.equal(rendered.rendererFrameCount,1);assert.equal(rendered.renderedFrameIndex,0);assert.equal(rendered.rendererReady,true);
  context.drawSkeletonView=()=>{throw new Error('canvas draw failed');};
  assert.throws(()=>view.run('draw()'),/canvas draw failed/);
  assert.equal(view.run('window.__motionPlaybackProbe()').rendererFrameCount,1,'failed draw is not counted');
});
