/* Exam inspection reads saved features. Only an explicit administrator action starts extraction. */
(() => {
  'use strict';
  const $=id=>document.getElementById(id),dialog=$('tv-inspection');if(!dialog)return;
  const API='/admin/api/training-data',ID=/^[0-9a-f]{32}$/,NS='http://www.w3.org/2000/svg';
  const number=new Intl.NumberFormat('ko-KR',{maximumFractionDigits:2});
  const state={id:null,data:null,trigger:null,read:null,authRead:null,jobRead:null,csrf:'',authRevision:0,expires:0,job:null,jobs:new Map(),poll:null,expiry:null,starting:false};
  const el=(tag,text,cls)=>{const node=document.createElement(tag);if(text!==undefined&&text!==null)node.textContent=String(text);if(cls)node.className=cls;return node;};
  const numeric=value=>typeof value==='number'&&Number.isFinite(value);
  const value=(v,unit='')=>numeric(v)?number.format(v)+(unit?' '+unit:''):'미기록';
  const processLabel=v=>({spray:'스프레이',blasting:'블라스팅',grinding:'그라인딩'}[v]||v||'미기록');
  const roleLabel=v=>({expert:'전문가 기준',worker:'작업자',trainee:'훈련자',unknown:'신원·자격 미확인',unverified:'자격 미확인'}[v]||v||'미기록');
  const qualificationLabel=v=>v==null?'미기록':typeof v==='object'?JSON.stringify(v):({unknown:'미확인',expert:'전문가'}[v]||v);
  const reasons={source_preview_unavailable:'원천 CSV 일부를 표시할 수 없습니다.',source_columns_unavailable:'CSV 열 정보를 확인할 수 없습니다.',source_unavailable:'연결 원천 CSV를 사용할 수 없습니다.',source_changed:'연결 원천 CSV가 등록 당시와 달라졌습니다.',source_too_large:'단건 재변환의 원천 크기 한도를 넘습니다.',source_expanded_too_large:'CSV 처리 크기 한도를 넘었습니다.',stored_features_do_not_include_pose_scores:'이 특징 JSON에는 자세 채점 결과가 저장되지 않았습니다.',stored_features_do_not_include_direction_angles:'이 특징 JSON에는 방향 각도 분류가 저장되지 않았습니다.',coating_not_applicable_to_process:'이 공정은 도막 두께 판정 대상이 아닙니다.',process_not_recorded:'공정 정보가 없어 도막 판정 적용 여부를 확인할 수 없습니다.',extraction_worker_failed:'재변환 작업을 완료하지 못했습니다.',feature_extraction_unavailable:'특징 추출을 완료할 수 없습니다.',extraction_timeout:'재변환 작업 시간 한도를 넘었습니다.',extraction_memory_limit:'재변환 작업의 메모리 한도를 넘었습니다.',extraction_cancelled:'재변환 작업이 중단됐습니다.',result_too_large:'재변환 결과 크기 한도를 넘었습니다.',inspection_record_too_large:'이 기록은 상세 표시 크기 한도를 넘습니다.',inspection_response_too_large:'이 기록은 상세 응답 크기 한도를 넘습니다.',dataset_not_found:'해당 기록을 찾을 수 없습니다.',dataset_unavailable:'해당 등록 기록을 사용할 수 없습니다.',job_not_found:'재변환 작업이 만료됐거나 존재하지 않습니다.',reconversion_busy:'다른 단건 재변환이 진행 중입니다.'};
  const reason=text=>reasons[text]||text;
  Object.assign(reasons,{reconversion_job_unavailable:'재변환 작업이 만료됐거나 존재하지 않습니다.',reconversion_result_not_ready:'재변환 결과가 아직 준비되지 않았습니다.',reconversion_already_running:'다른 단건 재변환이 진행 중입니다.',source_process_unsupported:'등록된 공정 정보로는 CSV 재변환을 실행할 수 없습니다.',inspection_service_closed:'단건 재변환 서비스를 사용할 수 없습니다.'});
  const fieldLabel=key=>({filename:'원천 파일',bytes:'원천 크기 (바이트)',schema_version:'JSON 형식',frame_count:'프레임 수',map_count:'맵 수',sequence_count:'분사 구간 수',zone_count:'Zone 수',duration_seconds:'훈련 시간 (초)',global_metrics:'전체 통계',zones:'Zone별 통계',zone_order:'첫 방문 순서',map_order:'맵 방문 순서'}[key]||key);
  const primary=metrics=>(Array.isArray(metrics)?metrics:[]).filter(m=>/_(mean|median)(?:_um)?$/.test(m.key||''));
  const section=(title,note)=>{const node=el('section',null,'td-panel ti-section');node.append(el('h3',title));if(note)node.append(el('p',note,'td-help'));return node;};
  function metricCards(metrics){
    const grid=el('dl',null,'ti-metrics');
    for(const metric of Array.isArray(metrics)?metrics:[]){const card=el('div');card.append(el('dt',metric.label||metric.key),el('dd',metric.status==='available'?value(metric.value,metric.unit):'미기록'));if(numeric(metric.sample_count))card.append(el('small',`유효 분사 프레임 ${number.format(metric.sample_count)}개`));grid.append(card);}
    if(!grid.children.length)grid.append(el('p','저장된 통계가 없습니다.','ti-missing'));return grid;
  }
  function metricCharts(metrics){
    const groups=new Map();for(const metric of Array.isArray(metrics)?metrics:[]){const key=String(metric.group||metric.unit||'stats');if(!groups.has(key))groups.set(key,[]);groups.get(key).push(metric);}
    const host=el('div',null,'ti-chart-grid');
    for(const group of groups.values()){
      const entries=group.filter(m=>m.status==='available'&&numeric(m.value));const card=el('div',null,'ti-stat-chart');
      card.append(el('h4',(group[0].label||'저장 통계').replace(/\s*(평균|중앙값|25백분위|75백분위)$/,'')+' · '+(group[0].unit||'단위 미기록')));
      if(!entries.length){card.append(el('p','이 통계는 저장되지 않았습니다.','ti-missing'));host.append(card);continue;}
      const lo=Math.min(0,...entries.map(m=>m.value)),hi=Math.max(0,...entries.map(m=>m.value)),span=hi-lo||1;
      const svg=document.createElementNS(NS,'svg');svg.setAttribute('viewBox',`0 0 560 ${entries.length*38+28}`);svg.setAttribute('role','img');svg.setAttribute('aria-label','저장된 통계값 · '+entries.map(m=>`${m.label} ${value(m.value,m.unit)}`).join(', '));
      const scale=v=>190+(v-lo)/span*245;
      for(const [index,metric] of entries.entries()){
        const y=index*38+20,label=document.createElementNS(NS,'text');label.setAttribute('x','0');label.setAttribute('y',String(y+4));label.textContent=metric.label;svg.append(label);
        const bar=document.createElementNS(NS,metric.value===0?'circle':'rect');
        if(metric.value===0){bar.setAttribute('cx',String(scale(0)));bar.setAttribute('cy',String(y));bar.setAttribute('r','3');}
        else{bar.setAttribute('x',String(Math.min(scale(0),scale(metric.value))));bar.setAttribute('y',String(y-8));bar.setAttribute('height','16');bar.setAttribute('width',String(Math.abs(scale(metric.value)-scale(0))));bar.setAttribute('rx','3');}
        svg.append(bar);const amount=document.createElementNS(NS,'text');amount.setAttribute('x','450');amount.setAttribute('y',String(y+4));amount.textContent=value(metric.value,metric.unit);svg.append(amount);
      }
      card.append(svg,el('p','저장된 요약 통계 비교 · 원시 프레임 분포나 새로운 판정 결과가 아닙니다.','td-help'));host.append(card);
    }return host;
  }
  function sourcePreview(data){
    const node=section('CSV 입력 미리보기','등록 JSON에 연결된 CSV의 일부 행입니다. 전체 원시 모션은 3D 재생에서 확인합니다.');const preview=data.source_preview||{};
    if(preview.status!=='available'){node.append(el('p',reason(preview.reason)||'원천 CSV 미리보기를 사용할 수 없습니다.','ti-missing'));return node;}
    node.append(el('p',`미리보기 ${value(preview.sampled_rows)}행 · 전체 ${value(preview.total_columns)}열${preview.truncated?' · 일부 열·값 생략':''}`,'td-help'));
    const scroll=el('div',null,'ti-csv-scroll'),table=el('table'),thead=el('thead'),head=el('tr');
    for(const name of (preview.columns||[]).slice(0,40)){const th=el('th',name);th.scope='col';head.append(th);}thead.append(head);table.append(thead);
    const tbody=el('tbody');for(const values of (preview.rows||[]).slice(0,30)){const row=el('tr');for(const cell of (Array.isArray(values)?values:[]).slice(0,40))row.append(el('td',String(cell??'').slice(0,600)));tbody.append(row);}table.append(tbody);scroll.append(table);node.append(scroll);
    if(preview.omitted_columns?.length)node.append(el('p','생략된 열: '+preview.omitted_columns.slice(0,40).join(', '),'td-help'));return node;
  }
  function savedPipeline(data){
    const node=section('CSV → 정보 → JSON','아래 단계와 수치는 등록 JSON에 저장된 분석 결과입니다. 조회만으로 새 분석을 실행하지 않습니다.');const list=el('ol',null,'ti-pipeline');
    for(const step of Array.isArray(data.pipeline)?data.pipeline:[]){const item=el('li');item.append(el('strong',step.label),el('span',step.status==='not_applicable'?'공정 비해당':['unavailable','missing','failed'].includes(step.status)?'기록 확인 필요':'저장 결과','ti-step-state'));if(step.description)item.append(el('p',step.description));for(const [key,v]of Object.entries(step.values||{}).slice(0,12))item.append(el('small',`${fieldLabel(key)}: ${numeric(v)?number.format(v):String(v??'미기록')}`));list.append(item);}node.append(list);return node;
  }
  function classification(title,data){
    const node=el('div',null,'ti-classification');node.append(el('strong',title));
    const actual=data?.classification??data?.label??data?.value;
    node.append(el('p',data?.status==='available'&&actual!=null?(typeof actual==='object'?JSON.stringify(actual):actual):(reason(data?.reason)||'분류 결과가 저장되지 않았습니다.')));return node;
  }
  function renderAnalysis(key='whole'){
    if(!state.data)return;const data=state.data,zone=(data.zones||[]).find(z=>z.key===key),selected=zone||data.summary||{};const host=$('ti-analysis');host.replaceChildren();
    const node=section(zone?`맵 ${zone.map_num??'미기록'} · Zone ${zone.zone_id??'미기록'}`:'전체 훈련 요약',zone?'선택 Zone에 저장된 집계와 해당 회차의 방문 순서입니다.':'한 훈련 회차에서 저장된 정보입니다.');
    if(zone)node.append(el('p',numeric(zone.sequence?.first_visit_index)?`해당 기록의 첫 방문 순서 ${number.format(zone.sequence.first_visit_index)}`:'첫 방문 순서 미기록','ti-order'));
    else node.append(metricCards([{label:'프레임',value:selected.frame_count,status:'available',unit:'개'},{label:'훈련 시간',value:selected.duration_seconds,status:'available',unit:'초'},{label:'분사 구간',value:selected.spray_interval_count,status:'available',unit:'개'},{label:'Zone',value:selected.zone_count,status:'available',unit:'개'}]));
    node.append(metricCards(primary(selected.metrics)));const charts=el('details',null,'ti-stat-details');charts.append(el('summary','통계 그래프 · 평균·중앙값·사분위 보기'),metricCharts(selected.metrics));node.append(charts);
    const classifications=el('div',null,'ti-classifications');classifications.append(classification('자세 분류',selected.pose),classification('방향 분류',selected.direction));node.append(classifications);
    const notApplicable=data.coating?.status==='not_applicable'||zone?.coating?.status==='not_applicable';
    if(zone?.coating){node.append(el('h4','선택 Zone의 저장 도막 정보'));if(notApplicable)node.append(el('p',reasons.coating_not_applicable_to_process,'ti-missing'));else node.append(metricCards(zone.coating.metrics));}
    host.append(node);
    const views=window.TrainingDataPresentation;
    if(notApplicable){if(!zone)host.append(el('p',reasons.coating_not_applicable_to_process,'ti-missing'));}
    else if(views?.coating){
      const note=el('p','아래 PNG와 히스토그램은 연결된 맵 단위입니다. 선택 Zone 수치와 집계 범위가 다릅니다.','ti-map-domain');host.append(note);
      const maps=Array.isArray(data.coating?.maps)?data.coating.maps:[];
      const coating=zone?{...data.coating,maps:maps.filter(map=>String(map.map_num)===String(zone.map_num))}:data.coating;
      host.append(views.coating({dataset_id:data.dataset_id,metadata:data.metadata,coating},{idPrefix:'tv-inspection',isolated:true}));
    }else host.append(el('p','맵 표시 기능을 불러오지 못했습니다. 저장 JSON과 다운로드로 자료를 확인할 수 있습니다.','ti-missing'));
  }
  function link(label,href,id){const node=el('a',label,'td-export');node.href=href;if(id)node.id=id;node.target='_blank';node.rel='noopener noreferrer';return node;}
  async function request(url,options={},bounded=false){
    const response=await fetch(url,{...options,credentials:'same-origin',cache:'no-store',headers:{Accept:'application/json',...options.headers}});
    if(bounded&&Number(response.headers?.get?.('Content-Length'))>4194304)throw new Error('브라우저 표시 한도를 넘는 JSON입니다. 다운로드로 확인해 주세요.');
    const data=await response.json();if(!response.ok){const error=new Error(typeof data?.detail==='string'?reason(data.detail):`요청 실패 (HTTP ${response.status})`);error.status=response.status;throw error;}
    return data;
  }
  function render(data){
    const body=$('tv-inspection-content');body.replaceChildren();const meta=data.metadata||{};
    $('tv-inspection-title').textContent=meta.dataset_name||data.dataset_id;
    const intro=section('기록과 자료 구분','품질 완료 여부와 작업자 자격은 별개의 정보입니다.');const grid=el('dl',null,'ti-meta-grid');
    const appendMeta=(target,entries)=>{for(const [label,text]of entries){const card=el('div');card.append(el('dt',label),el('dd',text));target.append(card);}};
    appendMeta(grid,[['작업자',meta.worker_name||meta.worker_id||'미확인'],['등록 역할',roleLabel(meta.role)],['등록 자격 정보',qualificationLabel(meta.qualification)],['공정',processLabel(meta.process)]]);intro.append(grid);
    const provenance=el('details',null,'ti-record-provenance'),extra=el('dl',null,'ti-meta-grid');provenance.append(el('summary','세션·자료 범위 보기'));appendMeta(extra,[['모드·시나리오',meta.mode_label||meta.scenario_id||'미기록'],['세션',meta.session_id||'미기록'],['자료 종류','저장된 회차별 특징 JSON']]);provenance.append(extra);
    for(const note of data.scope?.notes||[])provenance.append(el('p',note,'td-help'));intro.append(provenance);body.append(intro);
    const controls=el('div',null,'ti-scope-controls'),whole=el('button','전체 훈련','td-button-secondary');whole.id='ti-whole';whole.type='button';whole.setAttribute('aria-pressed','true');
    const field=el('label','Zone 선택 '),select=el('select');select.id='ti-zone';select.setAttribute('aria-label','Zone별 정보 선택');const initial=el('option','전체 훈련');initial.value='whole';select.append(initial);
    for(const zone of data.zones||[]){const option=el('option',`맵 ${zone.map_num??'미기록'} · Zone ${zone.zone_id??'미기록'}`);option.value=zone.key;select.append(option);}select.value='whole';select.disabled=!(data.zones?.length);field.append(select);controls.append(whole,field);body.append(controls);
    const analysis=el('div');analysis.id='ti-analysis';body.append(analysis);
    select.addEventListener('change',()=>{whole.setAttribute('aria-pressed',String(select.value==='whole'));renderAnalysis(select.value);});whole.addEventListener('click',()=>{select.value='whole';whole.setAttribute('aria-pressed','true');renderAnalysis();});renderAnalysis();
    body.append(savedPipeline(data),sourcePreview(data));
    const json=section('저장 JSON과 실제 모션','JSON은 가공된 정보이며 전체 모션 프레임은 연결 원천에서 재생합니다.');const actions=el('div',null,'ti-links');
    actions.append(link('저장 JSON 다운로드 ↓',`${API}/${state.id}/download`),link('해당 기록 3D 재생 ↗',`/motion-viewer?dataset_id=${state.id}`,'ti-motion'));
    const load=el('button','저장 JSON 펼치기','td-button-secondary');load.id='ti-json-load';load.type='button';actions.append(load);json.append(actions);const jsonBody=el('div');json.append(jsonBody);body.append(json);
    load.addEventListener('click',async()=>{const current=state.read,id=state.id;load.disabled=true;try{const raw=await request(`${API}/${id}`,{signal:current.signal},true);if(current!==state.read||current.signal.aborted)return;if(JSON.stringify(raw).length>4194304)throw new Error('브라우저 표시 한도를 넘는 JSON입니다. 다운로드로 확인해 주세요.');jsonBody.replaceChildren(window.TrainingDataPresentation?.json?window.TrainingDataPresentation.json(raw,id):el('pre',JSON.stringify(raw,null,2)));load.textContent='저장 JSON을 불러왔습니다.';}catch(error){if(error.name!=='AbortError'&&current===state.read){jsonBody.replaceChildren(el('p',error.message,'ti-error'));load.disabled=false;}}});
    const job=section('관리자 단건 CSV 재변환','실제 CSV를 다시 읽는 선택 작업입니다. 결과는 별도 시험 샘플이며 등록 JSON·기존 검증 회차를 변경하지 않습니다.');
    const start=el('button','CSV 단건 재변환');start.id='ti-reconvert';start.type='button';start.hidden=true;start.addEventListener('click',startJob);
    const login=link('관리자 로그인 ↗','/admin/manage','ti-job-login');const note=el('p',reason(data.reconvert?.reason)||'원천 크기와 서버 작업 제한 범위 안에서만 실행합니다.','td-help');
    const details=el('div');details.id='ti-job-details';const download=link('재변환 시험 샘플 JSON ↓','#','ti-job-download');download.hidden=true;download.removeAttribute('href');
    job.append(start,login,note,details,download);body.append(job);adminControls();
  }
  function available(){return Boolean(state.csrf)&&Date.now()<state.expires;}
  function adminControls(){const start=$('ti-reconvert');if(!start||!state.data)return;start.hidden=!available();start.disabled=state.starting||state.data.reconvert?.available!==true||['queued','running'].includes(state.job?.status);const login=$('ti-job-login');if(login)login.hidden=available();}
  function clearProtected(){clearTimeout(state.poll);state.jobRead?.abort();state.job=null;$('ti-job-details')?.replaceChildren();const download=$('ti-job-download');if(download){download.hidden=true;download.removeAttribute('href');}adminControls();}
  function loseAuth(){state.csrf='';state.authRevision++;state.expires=0;clearTimeout(state.expiry);clearProtected();}
  async function refreshAuth(){
    state.authRead?.abort();const controller=new AbortController();state.authRead=controller;
    try{const session=await request('/admin/api/manage/status',{signal:controller.signal});if(controller!==state.authRead||controller.signal.aborted)return;if(!session.authenticated){loseAuth();return;}const csrf=session.csrf_token||'';if(csrf!==state.csrf)state.authRevision++;state.csrf=csrf;state.expires=Number(session.expires_at||0)*1000;clearTimeout(state.expiry);state.expiry=setTimeout(loseAuth,Math.max(0,state.expires-Date.now()));adminControls();if(dialog.open&&state.id&&state.jobs.has(state.id))await pollJob();}
    catch(error){if(error.name!=='AbortError'&&controller===state.authRead)loseAuth();}
  }
  function renderJob(job){
    if(!available()||job.dataset_id!==state.id)return;const host=$('ti-job-details');if(!host)return;state.job=job;host.replaceChildren();
    const labels={queued:'작업 대기',running:'실제 재변환 진행 중',completed:'실제 재변환 완료',failed:'재변환 실패'};host.append(el('strong',labels[job.status]||'작업 상태 확인 필요'));
    const steps=el('ol',null,'ti-job-steps'),statuses={pending:'대기',running:'처리 중',completed:'완료',failed:'실패'};
    for(const step of job.steps||[]){const item=el('li');item.append(el('strong',step.label),el('span',statuses[step.status]||'상태 미기록'));if(step.detail)item.append(el('p',step.detail));steps.append(item);}host.append(steps);
    if(job.reason)host.append(el('p',reason(job.reason),'ti-error'));if(job.comparison)host.append(el('p',`저장 특징과 비교: 변경 필드 ${(job.comparison.changed_fields||[]).map(fieldLabel).join(', ')||'없음'} · 도막·전문가 분포의 재분석과 구분합니다.`,'td-help'));
    const download=$('ti-job-download');download.hidden=true;download.removeAttribute('href');
    if(job.status==='completed'&&job.result_available===true&&ID.test(job.job_id)){
      try{const url=new URL(job.download_url,window.location.href);if(url.origin===window.location.origin&&url.pathname===`${API}/${state.id}/reconvert/${job.job_id}/download`){download.href=url.pathname;download.hidden=false;}}catch(_){}
    }adminControls();
  }
  async function pollJob(){
    if(!dialog.open||!available()||!state.id||!state.data)return;const id=state.id,jobId=state.jobs.get(id);if(!ID.test(jobId||''))return;
    clearTimeout(state.poll);state.jobRead?.abort();const controller=new AbortController();state.jobRead=controller;
    try{const job=await request(`${API}/${id}/reconvert/${jobId}`,{signal:controller.signal});if(controller!==state.jobRead||controller.signal.aborted||state.id!==id||!available())return;if(job.dataset_id!==id||job.job_id!==jobId)throw new Error('재변환 작업 식별 정보가 맞지 않습니다.');renderJob(job);if(['queued','running'].includes(job.status))state.poll=setTimeout(pollJob,1000);}
    catch(error){if(error.name==='AbortError'||controller!==state.jobRead||controller.signal.aborted||state.id!==id)return;if([401,403].includes(error.status)){loseAuth();return;}if(error.status===404){state.jobs.delete(id);clearProtected();}$('ti-job-details')?.replaceChildren(el('p',error.message,'ti-error'));}
  }
  async function startJob(){
    if(state.starting)return;const id=state.id;let submittedAuth=null;state.starting=true;adminControls();
    try{
      await refreshAuth();if(!dialog.open||state.id!==id||!available()||state.data?.reconvert?.available!==true)return;
      // Keep the POST response when a dialog closes: the server job continues, and its ID is recoverable after login.
      submittedAuth=state.authRevision;const job=await request(`${API}/${id}/reconvert`,{method:'POST',headers:{'X-DB-CSRF':state.csrf}});
      if(job.dataset_id!==id||!ID.test(job.job_id||''))throw new Error('재변환 작업 식별 정보가 맞지 않습니다.');state.jobs.set(id,job.job_id);
      if(dialog.open&&state.id===id&&available()){renderJob(job);if(['queued','running'].includes(job.status))await pollJob();}
    }catch(error){if([401,403].includes(error.status)){if(submittedAuth===state.authRevision)loseAuth();}else if(dialog.open&&state.id===id)$('ti-job-details')?.replaceChildren(el('p',error.message,'ti-error'));}
    finally{state.starting=false;adminControls();}
  }
  async function open(id,trigger){
    if(!ID.test(id||''))return;state.read?.abort();state.jobRead?.abort();clearTimeout(state.poll);state.id=id;state.data=null;state.job=null;state.trigger=trigger||document.activeElement;
    const controller=new AbortController();state.read=controller;$('tv-inspection-title').textContent='시험자료 확인 중';$('tv-inspection-status').textContent='저장된 분석과 CSV 일부를 불러오는 중입니다.';$('tv-inspection-content').replaceChildren();dialog.setAttribute('aria-busy','true');
    if(!dialog.open){document.body.classList.add('tv-inspection-open');if(dialog.showModal)dialog.showModal();else dialog.setAttribute('open','');}$('tv-inspection-close').focus();
    const auth=refreshAuth();
    try{const data=await request(`${API}/${id}/inspection`,{signal:controller.signal});if(controller!==state.read||controller.signal.aborted||!dialog.open)return;if(data.dataset_id!==id)throw new Error('조회한 기록의 식별 정보가 맞지 않습니다.');state.data=data;render(data);$('tv-inspection-status').textContent='저장된 분석을 불러왔습니다. 새 CSV 재변환은 관리자 버튼으로 별도 실행합니다.';await auth;if(available()&&state.jobs.has(id))await pollJob();}
    catch(error){if(error.name!=='AbortError'&&controller===state.read){$('tv-inspection-status').textContent='자료를 불러오지 못했습니다.';const retry=el('button','다시 불러오기','td-button-secondary');retry.type='button';retry.addEventListener('click',()=>open(id,trigger));$('tv-inspection-content').replaceChildren(el('p',error.message,'ti-error'),retry);}}
    finally{if(controller===state.read)dialog.setAttribute('aria-busy','false');}
  }
  function afterClose(){state.read?.abort();state.authRead?.abort();clearProtected();document.body.classList.remove('tv-inspection-open');if(state.trigger?.isConnected)state.trigger.focus();else{const action=state.trigger?.dataset?.verificationAction==='inspection-action'?'inspection-action':'inspection';document.querySelector?.(`[data-verification-dataset="${state.id}"][data-verification-action="${action}"]`)?.focus();}}
  function close(){if(dialog.close)dialog.close();else{dialog.removeAttribute('open');afterClose();}}
  $('tv-inspection-close').addEventListener('click',close);dialog.addEventListener('close',afterClose);dialog.addEventListener('cancel',event=>{event.preventDefault();close();});
  dialog.addEventListener('click',event=>{if(event.target!==dialog)return;const r=dialog.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)close();});
  dialog.addEventListener('keydown',event=>{if(event.key!=='Tab')return;const nodes=[...dialog.querySelectorAll('button,a[href],input,select,textarea,[tabindex]')].filter(n=>!n.disabled&&!n.hidden&&!n.closest('[hidden]')&&n.tabIndex!==-1);if(!nodes.length)return;const first=nodes[0],last=nodes.at(-1);if(event.shiftKey&&document.activeElement===first){event.preventDefault();last.focus();}else if(!event.shiftKey&&document.activeElement===last){event.preventDefault();first.focus();}});
  window.addEventListener('focus',()=>{if(dialog.open)refreshAuth();});window.addEventListener('pagehide',()=>{state.read?.abort();state.authRead?.abort();clearProtected();clearTimeout(state.expiry);});
  window.TrainingInspection=Object.freeze({open,close,refreshAuth});
})();
