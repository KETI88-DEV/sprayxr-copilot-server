/* Read-only, single-record explanation. No extraction jobs or verification writes. */
(() => {
  'use strict';
  const $=id=>document.getElementById(id);if(!$('tt-root'))return;
  const API='/admin/api/training-data',ID=/^[0-9a-f]{32}$/,LIMIT=4*1024*1024;
  const number=new Intl.NumberFormat('ko-KR',{maximumFractionDigits:2});
  const state={active:false,started:false,id:null,data:null,raw:null,rawError:'',controller:null,listController:null,view:'zone',zone:'whole',scene:null,sceneHost:null};
  const el=(tag,text,cls)=>{const node=document.createElement(tag);if(text!=null)node.textContent=String(text);if(cls)node.className=cls;return node;};
  const array=v=>Array.isArray(v)?v:[],object=v=>v&&typeof v==='object'&&!Array.isArray(v)?v:{};
  const finite=v=>typeof v==='number'&&Number.isFinite(v);
  const amount=(v,unit='')=>finite(v)?number.format(v)+(unit?' '+unit:''):'미기록';
  const pick=(v,keys)=>Object.fromEntries(keys.filter(k=>Object.hasOwn(object(v),k)).map(k=>[k,v[k]]));
  const key=v=>`${v.map_num??v.map_label??'unknown'}:${v.zone_id??'unknown'}`;
  const current=()=>array(state.data?.zones).find(z=>z.key===state.zone);
  const selectedMap=()=>current()?.map_num;
  const link=(label,path)=>{const node=el('a',label);node.href=path;node.target='_blank';node.rel='noopener noreferrer';return node;};
  const metric=(scope,id)=>array(scope?.metrics).find(m=>m.key===id&&m.status==='available')?.value;
  const color=zone=>{
    const map=array(state.data?.coating?.maps).find(m=>String(m.map_num)===String(zone.map_num));
    const raw=array(map?.zones).find(z=>String(z.zone_id)===String(zone.zone_id))?.hex_color;
    return /^#[0-9a-f]{6}$/i.test(raw||'')?raw:'#6c86bd';
  };
  function destroyScene(){state.scene?.destroy();state.scene=null;state.sceneHost=null;}
  async function request(url,signal){
    const response=await fetch(url,{signal,credentials:'same-origin',cache:'no-store',headers:{Accept:'application/json'}});
    if(!response.ok)throw new Error(`자료 조회 실패 (HTTP ${response.status})`);
    if(Number(response.headers?.get?.('Content-Length'))>LIMIT)throw new Error('JSON 표시 한도를 넘었습니다. 전체 자료는 다운로드로 확인합니다.');
    if(response.body?.getReader){
      const reader=response.body.getReader(),parts=[];let bytes=0;
      try{while(true){const chunk=await reader.read();if(chunk.done)break;bytes+=chunk.value.byteLength;if(bytes>LIMIT)throw new Error('JSON 표시 한도를 넘었습니다. 전체 자료는 다운로드로 확인합니다.');parts.push(chunk.value);}}
      catch(error){await reader.cancel().catch(()=>{});throw error;}finally{reader.releaseLock();}
      const bytesArray=new Uint8Array(bytes);let offset=0;for(const part of parts){bytesArray.set(part,offset);offset+=part.byteLength;}
      return JSON.parse(new TextDecoder().decode(bytesArray));
    }
    const data=await response.json();if(JSON.stringify(data).length>LIMIT)throw new Error('JSON 표시 한도를 넘었습니다.');return data;
  }
  function note(text){return el('p',text,'tt-footnote');}
  function card(label,v,unit){const node=el('div',null,'tt-metric');node.append(el('span',label),el('strong',amount(v,unit)));return node;}
  function ensureOption(id,label){if(!array([...$('tt-record').children]).some(n=>n.value===id)){const option=el('option',label||id);option.value=id;$('tt-record').append(option);}$('tt-record').value=id;$('tt-record').disabled=false;}
  function renderZones(){
    const host=$('tt-zones');host.replaceChildren();const zones=array(state.data?.zones);
    for(const zone of [null,...zones]){
      const button=el('button',zone?`맵 ${zone.map_num??zone.map_label??'?'} · Zone ${zone.zone_id}`:'전체 훈련','tt-zone-chip');
      button.type='button';const identity=zone?.key||'whole';button.dataset.zoneKey=identity;button.setAttribute('aria-pressed',String(identity===state.zone));
      button.style.setProperty('--zone-color',zone?color(zone):'#5272ab');
      button.addEventListener('click',()=>{state.zone=identity;renderZones();renderSource();renderVisual();renderJSON();});host.append(button);
    }
  }
  function renderSource(){
    const host=$('tt-source');host.replaceChildren();const preview=state.data?.source_preview||{};
    if(state.view==='coating'){
      host.append(el('strong','CSV + 연결된 최종 높이맵'),note('도막 두께는 높이맵의 픽셀값에서 환산합니다. 위치 좌표만으로 두께를 만들지 않습니다.'));
    }else host.append(el('strong',state.view==='metrics'?'분사 중 입력 거리·속도':'시간 · 분사 상태 · 착탄 위치 · Zone ID'));
    if(preview.status!=='available'){host.append(el('p','연결된 CSV 일부를 불러올 수 없습니다.','tt-empty'));return;}
    const columns=array(preview.columns),wanted=state.view==='metrics'?/시간|^time$|분사|zone id|distance \(input\)|speed \(input\)/i:/시간|^time$|분사|zone id|ray texture map|ray point/i;
    const indices=columns.map((name,i)=>wanted.test(name)?i:-1).filter(i=>i>=0).slice(0,8);
    if(!indices.length){host.append(el('p','이 원본의 해당 열은 미리보기에 없습니다. 전체 CSV 정보는 자료 상세에서 확인합니다.','tt-empty'));return;}
    const wrap=el('div',null,'tt-csv-wrap'),table=el('table'),thead=el('thead'),head=el('tr');
    for(const i of indices){const th=el('th',columns[i]);th.scope='col';head.append(th);}thead.append(head);table.append(thead);const body=el('tbody');
    for(const row of array(preview.rows).slice(0,6)){const tr=el('tr');for(const i of indices)tr.append(el('td',array(row)[i]??''));body.append(tr);}table.append(body);wrap.append(table);host.append(wrap);
    host.append(note('CSV 시작 부분의 일부 행입니다. 선택한 Zone으로 원시 행을 필터링한 표는 아닙니다.'));
    host.append(el('p',state.view==='metrics'?'가공: 분사 On인 유효 수치를 모아 Zone별 평균·중앙값을 구합니다.':'가공: 분사 On 구간을 묶고, 면 번호와 Zone ID별로 착탄 위치와 수치를 집계합니다.','tt-flow-note'));
  }
  function renderOrder(host){
    const ordered=array(state.data?.zones).filter(z=>finite(z.sequence?.first_visit_index)).sort((a,b)=>a.sequence.first_visit_index-b.sequence.first_visit_index);
    const list=el('div',null,'tt-order');for(const zone of ordered){const chip=el('span',`${zone.sequence.first_visit_index}. M${zone.map_num??'?'} Z${zone.zone_id}`,'tt-zone-chip');chip.style.setProperty('--zone-color',color(zone));list.append(chip);}
    host.append(el('h4','이 기록의 첫 방문 순서'),list.children.length?list:note('방문 순서 미기록'));
  }
  function renderZone(host){
    renderOrder(host);
    if(state.scene&&state.sceneHost){host.append(state.sceneHost);state.scene.selectZone(state.zone);return;}
    const intro=el('div',null,'tt-empty');intro.append(el('strong','착탄 좌표를 3차원 도장 경로로 확인'),note('3D 미리보기는 연결 CSV를 읽으므로 최초 조회에 시간이 걸릴 수 있습니다.'));
    const button=el('button','3D 구역·경로 불러오기','tt-load-scene');button.id='tt-load-scene';button.type='button';intro.append(button);host.append(intro);
    button.addEventListener('click',()=>{
      if(!window.TrainingTransformScene){button.disabled=true;intro.append(note('3D 미리보기 기능을 불러오지 못했습니다. 아래 전체 모션뷰어 링크를 이용해 주세요.'));return;}
      const id=state.id,sceneHost=el('div',null,'tt-scenehost');intro.className='tt-scene-wrapper';intro.replaceChildren(sceneHost);state.sceneHost=sceneHost;
      state.scene=window.TrainingTransformScene.create(sceneHost,{datasetId:id,zoneKey:state.zone,
        onZoneChange:identity=>{if(state.id!==id||state.sceneHost!==sceneHost)return;state.zone=identity==='whole'||array(state.data?.zones).some(z=>z.key===identity)?identity:'whole';renderZones();renderSource();renderVisual();renderJSON();},
        onStatus:({message})=>{if(state.id===id&&state.sceneHost===sceneHost)$('tt-status').textContent=message;}});
    });
  }
  function renderMetrics(host){
    const scope=current()||state.data?.summary||{},grid=el('div',null,'tt-metrics');
    for(const [id,label,unit]of [['distance_mm_mean','입력 이격거리 평균','mm'],['distance_mm_median','입력 이격거리 중앙값','mm'],['speed_mms_mean','입력 이동속도 평균','mm/s'],['speed_mms_median','입력 이동속도 중앙값','mm/s']])grid.append(card(label,metric(scope,id),unit));
    host.append(grid,note('분사 중 유효한 입력값의 저장 통계입니다. 미기록 값은 채우지 않습니다.'));
  }
  function imageURL(url){try{const parsed=new URL(url,window.location.href);return parsed.origin===window.location.origin&&new RegExp(`^/admin/api/training-data/${state.id}/images/\\d+$`).test(parsed.pathname)?parsed.pathname+parsed.search:null;}catch(_){return null;}}
  function appendImage(host,label,url){
    const figure=el('figure'),stage=el('div',null,'tt-image-stage'),source=imageURL(url);figure.append(el('figcaption',label));
    if(source){const image=el('img');image.src=source;image.alt=label;image.loading='lazy';image.addEventListener('error',()=>stage.replaceChildren(note('이미지를 불러올 수 없습니다.')));stage.append(image);}else stage.append(note('이미지 미기록'));
    figure.append(stage);host.append(figure);
  }
  function renderCoating(host){
    if(['blasting','grinding'].includes(state.data?.metadata?.process)){host.append(el('p','이 공정은 스프레이 도막 두께 분석 대상이 아닙니다.','tt-empty'));return;}
    const allMaps=array(state.data?.coating?.maps),map=allMaps.find(m=>selectedMap()!=null&&String(m.map_num)===String(selectedMap()))||(state.zone==='whole'?allMaps[0]:null);
    if(!map){host.append(el('p','선택 범위에 연결된 높이맵이 없습니다.','tt-empty'));return;}
    const pair=el('div',null,'tt-image-pair');appendImage(pair,'원본 높이맵',map.image_url);appendImage(pair,'두께로 환산한 분포',map.thickness_image_url);host.append(pair);
    const zone=current(),zoneSummary=zone?array(map.zones).find(z=>String(z.zone_id)===String(zone.zone_id))?.summary:null;
    const summary=zone?object(zoneSummary):object(map.summary),grid=el('div',null,'tt-metrics');grid.append(card(zone?'선택 Zone 평균':'표시 맵 평균',summary.mean_um,'µm'),card(zone?'선택 Zone 중앙값':'표시 맵 중앙값',summary.median_um,'µm'));host.append(grid);
    host.append(note(`이미지: 맵 ${map.map_num??'?'} 전체 · ${zone?'수치만 선택 Zone 기준':'첫 번째 연결 맵 기준'}입니다. 다른 Zone을 잘라낸 이미지가 아닙니다.`));
    host.append(note('해당 기록의 최종 높이맵입니다. 전문가 여러 회차 평균이나 시간별 도막 누적 재현이 아닙니다.'));
  }
  function renderVisual(){
    const host=$('tt-visual');host.replaceChildren();
    for(const view of ['zone','metrics','coating'])$('tt-view-'+view).setAttribute('aria-pressed',String(state.view===view));
    const zone=current();host.append(el('strong',zone?`맵 ${zone.map_num??zone.map_label??'?'} · Zone ${zone.zone_id}`:'전체 훈련'));
    if(state.view==='zone')renderZone(host);else if(state.view==='metrics')renderMetrics(host);else renderCoating(host);
    $('tt-explanation').textContent=({zone:'착탄 좌표와 Zone ID → 면·Zone별 작업 범위와 첫 방문 순서 → JSON의 zones·zone_order',metrics:'분사 중 입력 거리·속도 → 평균·중앙값 등 분포 통계 → JSON의 distance_mm·speed_mms',coating:'연결 높이맵의 픽셀값 → 도막 두께와 Zone별 분포 → JSON의 coating.maps'})[state.view];
  }
  function renderJSON(){
    const host=$('tt-json');host.replaceChildren();
    if(!state.raw){host.append(el('p',state.rawError?'저장 JSON 원문을 불러오지 못했습니다. 전체 JSON 다운로드로 확인해 주세요.':'저장 JSON 항목을 불러오는 중입니다.','tt-empty'));return;}
    const raw=state.raw,features=object(raw.features),selection=current();
    const chosen=array(features.zones).filter(z=>!selection||key(z)===selection.key).slice(0,selection?1:2);
    const excerpt={...pick(raw,['schema_version','dataset_id']),metadata:pick(raw.metadata,['worker_name','process','scenario_id'])};
    if(state.view==='coating'){
      excerpt.coating={...pick(raw.coating,['status']),maps:array(raw.coating?.maps).filter(m=>selectedMap()==null||String(m.map_num)===String(selectedMap())).slice(0,2).map(m=>({...pick(m,['map_num','summary']),zones:array(m.zones).filter(z=>!selection||String(z.zone_id)===String(selection.zone_id)).slice(0,selection?1:2).map(z=>pick(z,['zone_id','summary']))}))};
    }else{
      excerpt.features={...pick(features,['frame_count','sequence_count']),...(state.view==='zone'?pick(features,['zone_order']):{}),...(selection?{}:pick(features,['global_metrics'])),zones:chosen.map(z=>pick(z,state.view==='zone'?['map_num','map_label','zone_id','frame_count','ray_point_bounds_m']:['map_num','map_label','zone_id','distance_mm','speed_mms']))};
    }
    const pre=el('pre',JSON.stringify(excerpt,null,2),'tt-code');pre.tabIndex=0;pre.setAttribute('aria-label','저장된 학습 JSON의 실제 항목 발췌');host.append(pre,note('저장 JSON의 실제 키·값을 일부 발췌했습니다. 전체 통계는 최대 2개 Zone을 표시하며, 원문 다운로드에 모든 항목이 포함됩니다.'));
  }
  function render(){
    const data=state.data,meta=data.metadata||{};ensureOption(state.id,`${meta.worker_name||'이름 미확인'} · ${meta.mode_label||'모드 '+(meta.scenario_id||'?')} · ${meta.dataset_name||state.id}`);
    $('tt-meta').replaceChildren(...[['프레임',data.summary?.frame_count],['분사 구간',data.summary?.spray_interval_count],['Zone',data.summary?.zone_count]].map(([label,v])=>el('span',`${label} ${amount(v)}개`,'tt-meta-chip')));
    $('tt-actions').replaceChildren(link('저장 JSON 다운로드 ↓',`${API}/${state.id}/download`),link('전체 3D 모션뷰어 ↗',`/motion-viewer?dataset_id=${state.id}`));
    const inspect=el('button','자료 상세 보기');inspect.type='button';inspect.addEventListener('click',()=>window.TrainingInspection?.open(state.id,inspect));$('tt-actions').append(inspect);
    renderZones();renderSource();renderVisual();renderJSON();$('tt-status').textContent=`${meta.worker_name||'이름 미확인'} · ${meta.dataset_name||state.id} · 저장된 가공 결과를 설명하는 예시입니다.`;
  }
  async function show(id){
    if(!ID.test(id||''))return;state.controller?.abort();destroyScene();const controller=new AbortController();state.controller=controller;state.id=id;state.data=null;state.raw=null;state.rawError='';state.zone='whole';
    $('tt-root').setAttribute('aria-busy','true');$('tt-status').textContent='실제 기록의 CSV 일부와 저장 정보를 불러오는 중입니다.';$('tt-retry').hidden=true;
    for(const name of ['source','visual','json','zones','meta','actions'])$('tt-'+name).replaceChildren();
    try{
      const data=await request(`${API}/${id}/inspection`,controller.signal);if(state.controller!==controller||controller.signal.aborted)return;
      if(data.dataset_id!==id)throw new Error('기록 식별 정보가 일치하지 않습니다.');state.data=data;render();
      try{const raw=await request(`${API}/${id}`,controller.signal);if(raw.dataset_id!==id)throw new Error('저장 JSON의 기록이 일치하지 않습니다.');if(state.controller===controller&&!controller.signal.aborted){state.raw=raw;renderJSON();}}
      catch(error){if(error.name!=='AbortError'&&state.controller===controller){state.rawError=error.message;renderJSON();}}
    }catch(error){if(error.name!=='AbortError'&&state.controller===controller){$('tt-status').textContent=error.message;$('tt-retry').hidden=false;}}
    finally{if(state.controller===controller)$('tt-root').setAttribute('aria-busy','false');}
  }
  async function activate(active){
    state.active=active;if(!active){state.controller?.abort();state.listController?.abort();destroyScene();if(!state.data)state.started=false;return;}
    if(state.started){if(state.data&&state.raw){renderVisual();return;}return state.id?show(state.id):undefined;}state.started=true;
    const controller=new AbortController();state.listController=controller;$('tt-status').textContent='예시로 사용할 품질 완료 스프레이 기록을 찾는 중입니다.';
    try{
      let result=await request(`${API}?quality_status=complete&process=spray&worker_id=jeon_hyeonggi&limit=200`,controller.signal);
      if(!array(result.items).length)result=await request(`${API}?quality_status=complete&process=spray&limit=200`,controller.signal);
      if(controller!==state.listController||controller.signal.aborted)return;
      const rows=array(result.items).filter(r=>ID.test(r.dataset_id||''));rows.sort((a,b)=>Number(b.heightmap_count>0)-Number(a.heightmap_count>0)||(Number(b.zone_count)||0)-(Number(a.zone_count)||0));
      $('tt-record').replaceChildren();for(const row of rows)ensureOption(row.dataset_id,`${row.worker_name||'이름 미확인'} · ${row.mode_label||'모드 '+(row.scenario_id||'?')} · ${row.dataset_name||row.dataset_id}`);
      if(rows.length)await show(rows[0].dataset_id);else{$('tt-status').textContent='예시에 사용할 품질 완료 스프레이 기록이 없습니다.';state.started=false;}
    }catch(error){if(error.name!=='AbortError'){state.started=false;$('tt-status').textContent=error.message;$('tt-retry').hidden=false;}}
  }
  for(const view of ['zone','metrics','coating'])$('tt-view-'+view).addEventListener('click',()=>{if(!state.data)return;if(state.view!==view&&view!=='zone')destroyScene();state.view=view;renderSource();renderVisual();renderJSON();});
  $('tt-record').addEventListener('change',()=>show($('tt-record').value));$('tt-retry').addEventListener('click',()=>state.id?show(state.id):activate(true));
  window.addEventListener('pagehide',()=>{state.controller?.abort();state.listController?.abort();destroyScene();});
  window.TrainingTransform=Object.freeze({activate,show});
})();
