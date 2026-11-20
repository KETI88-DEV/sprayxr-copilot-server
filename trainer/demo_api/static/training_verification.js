/* Construction evidence stays separate from conversion quality and model training. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  if (!$('tv-root')) return;
  const API = '/admin/api/training-verification';
  const state = {report: null, page: 1, active: false, timer: null, controller: null, csrf: '', expires: 0, starting: false};
  const count = value => new Intl.NumberFormat('ko-KR').format(Number.isFinite(Number(value)) ? Math.max(0, Number(value)) : 0);
  const el = (tag, value, cls = '') => {const node = document.createElement(tag); if (value != null) node.textContent = String(value); if (cls) node.className = cls; return node;};
  const labels = {passed: '통과', failed: '실패', pending: '미검증', skipped: '제외'};
  const reasons = {browser_playback_failed:'실제 브라우저 재생 실패', duplicate_training_record:'동일 훈련 기록 중복', duplicate_source_content:'동일 원천 모션 내용 중복',
    json_hash_or_size_mismatch:'JSON 해시·크기 불일치', json_identity_mismatch:'JSON 식별 정보 불일치', quality_inconsistent:'변환 품질 정보 불일치',
    motion_metadata_incomplete:'모션 정보 불충분', source_link_missing:'연결 원천 없음', source_size_mismatch:'원천 크기 불일치',
    source_hash_missing:'원천 해시 없음', source_hash_mismatch:'원천 해시 불일치', viewer_scenario_unsupported:'뷰어 미지원 공정·모드',
    invalid_dataset_id:'데이터 식별자 형식 오류', non_finite_json:'JSON에 유효하지 않은 수치 포함', json_not_object:'JSON 구조 오류',
    source_changed:'점검 중 원천 내용 변경', verification_not_current:'등록 자료가 검증 당시와 달라 재점검 필요', verification_run_changed:'검증 회차가 바뀌어 이전 재생 증빙 제외',
    verification_running:'이미 JSON·원천 점검이 진행 중입니다.', catalog_unavailable:'등록 목록을 읽을 수 없습니다.', catalog_changed:'등록 목록이 변경되어 다시 점검해야 합니다.',
    artifact_or_source_unavailable:'JSON 또는 연결 원천을 읽을 수 없음'};
  const processLabel = value => ({spray:'스프레이',blasting:'블라스팅',grinding:'그라인딩'}[value] || value || '미기록');
  function badge(value) {return el('span', labels[value] || '미검증', 'tv-badge tv-' + (Object.hasOwn(labels, value) ? value : 'pending'));}
  function matches(item) {
    const filter = $('tv-status-filter').value, process = $('tv-process-filter').value, search = $('tv-search').value.trim().toLocaleLowerCase();
    const statuses = [item.json_status, item.source_status, item.browser_status];
    if (filter === 'passed' && !(statuses.every(value => value === 'passed') && !item.duplicate)) return false;
    if (filter === 'failed' && !statuses.includes('failed')) return false;
    if (filter === 'pending' && !statuses.includes('pending')) return false;
    if (filter === 'duplicates' && !item.duplicate) return false;
    if (process !== 'all' && process !== item.process) return false;
    return !search || [item.dataset_id,item.dataset_name,item.worker_id,item.worker_name,item.session_id].join(' ').toLocaleLowerCase().includes(search);
  }
  function renderRows() {
    const focused=document.activeElement;let focusTarget=null;
    const track=(node,id,action)=>{node.dataset.verificationDataset=id;node.dataset.verificationAction=action;if(focused?.dataset?.verificationDataset===id&&focused.dataset.verificationAction===action)focusTarget=node;};
    const inspect=(id,trigger)=>{if(window.TrainingInspection?.open)return window.TrainingInspection.open(id,trigger);$('tv-message').textContent='자료 상세 기능을 불러오지 못했습니다. 새로고침 후 다시 확인해 주세요.';};
    const items = (Array.isArray(state.report?.items) ? state.report.items : []).filter(matches), pages = Math.max(1, Math.ceil(items.length / 25));
    state.page = Math.min(state.page, pages);
    const rows = items.slice((state.page - 1) * 25, state.page * 25).map(item => {
      const row = el('tr'), name = el('td'),validId=/^[0-9a-f]{32}$/.test(item.dataset_id);
      const open=el(validId?'button':'strong',item.dataset_name || item.dataset_id,'tv-record-open');
      if(validId){open.type='button';track(open,item.dataset_id,'inspection');open.addEventListener('click',()=>inspect(item.dataset_id,open));row.className='tv-record-row';row.addEventListener('click',event=>{if(!event.target?.closest?.('a,button,input,select'))inspect(item.dataset_id,open);});}
      name.append(open,el('small',item.session_id));
      row.append(name, el('td',item.worker_name || item.worker_id || '미확인'), el('td',processLabel(item.process) + ' · 모드 ' + (item.scenario_id || '미기록')));
      for (const status of [item.json_status,item.source_status,item.browser_status]) {const cell = el('td');cell.append(badge(status));row.append(cell);}
      const note = el('td');if(item.duplicate)note.append(el('span','중복 제외','tv-duplicate'));
      const why = [...new Set(Array.isArray(item.reasons) ? item.reasons : [])];
      if (why.length) {const list = el('ul'); for (const reason of why) {const item = el('li',reasons[reason] || '검증 사유 확인 필요');item.title=String(reason);list.append(item);} note.append(list);}
      row.append(note);
      const actions = el('td', null, 'tv-row-actions');
      if (validId) for (const [label,href,action] of [['JSON 보기',`/admin/api/training-data/${item.dataset_id}`,'json'],['3D 재생',`/motion-viewer?dataset_id=${item.dataset_id}`,'playback']]) {
        const link=el('a',label);link.href=href;link.target='_blank';link.rel='noopener';link.dataset.verificationDataset=item.dataset_id;link.dataset.verificationAction=action;
        if(focused?.dataset?.verificationDataset===item.dataset_id&&focused.dataset.verificationAction===action)focusTarget=link;
        actions.append(link);
      }
      if(validId){const detail=el('button','자료 상세','tv-inspect-action');detail.type='button';track(detail,item.dataset_id,'inspection-action');detail.addEventListener('click',()=>inspect(item.dataset_id,detail));actions.append(detail);}
      row.append(actions);return row;
    });
    if (!rows.length) {const row=el('tr'),cell=el('td',state.report?.phase==='not_started'?'구축 검증을 아직 시작하지 않았습니다. 등록 건수는 재생 검증 통과 건수가 아닙니다.':'조건에 맞는 검증 기록이 없습니다.');cell.colSpan=8;row.append(cell);rows.push(row);}
    $('tv-items').replaceChildren(...rows);focusTarget?.focus({preventScroll:true});$('tv-pagination').textContent=`조회 ${count(items.length)}건 · ${state.page} / ${pages}페이지`;
    $('tv-prev').disabled=state.page<=1;$('tv-next').disabled=state.page>=pages;
  }
  function adminControl() {
    const available=Boolean(state.csrf)&&Date.now()<state.expires;
    $('tv-start').hidden=!available;$('tv-login').hidden=available;
    $('tv-start').disabled=state.starting||state.report?.phase==='checking_json';
  }
  function render(report) {
    if (!report || typeof report !== 'object' || !report.summary) throw new Error('검증 결과 형식을 확인해 주세요.');
    state.report=report;
    const summary=report.summary;
    for (const key of ['registered','json_passed','source_passed','browser_passed','valid','pending']) $('tv-kpi-'+key.replaceAll('_','-')).textContent=count(summary[key]);
    $('tv-definition').textContent=report.definition || '동일 공정의 훈련 기록 1회에서 가공한 JSON과 연결 모션 자료를 1건으로 집계합니다.';
    $('tv-scope').textContent=report.scope || '시작·중간·끝의 실제 브라우저 재생을 확인합니다. 전체 연속 재생과 AI 성능 검증은 포함하지 않습니다.';
    const phases={not_started:'미검증 · 시작 전',checking_json:'JSON·원천 연결 점검 중',awaiting_playback:'브라우저 재생 미검증 기록 있음',finished:'검증 회차 종료',interrupted:'점검 중단 · 다시 시작 필요'};
    $('tv-phase').textContent=phases[report.phase] || '검증 상태 확인 필요';
    $('tv-stale').textContent=typeof report.stale_reason==='string'&&report.stale_reason.trim()?report.stale_reason:'등록 목록이 검증 당시와 달라졌습니다. 이전 검증 결과는 현재 유효 구축 건수에 포함하지 않습니다.';
    $('tv-stale').hidden=!report.stale;
    const examTarget=Number(report.exam?.target)>0?Number(report.exam.target):1200;
    const met=!report.stale&&Number(summary.valid)>=examTarget;
    $('tv-goal').textContent=met?`시험 수량 기준 ${count(examTarget)}건 충족`:`시험 수량 기준 ${count(examTarget)}건 미충족 · ${count(summary.valid)}건 기술 검증 통과`;
    $('tv-exam-quantity').textContent=`검증 회차의 품질 완료 등록 ${count(summary.registered)}건 · 시험자료 수량 기준 ${count(examTarget)}건`;
    $('tv-historical-target').textContent=`보존된 검증 회차의 당시 기준 ${count(report.exam?.historical_run_target??report.target??1000)}건 · 회차와 기존 증빙을 유지합니다.`;
    const auditing=report.phase==='checking_json',eligible=(Array.isArray(report.items)?report.items:[]).filter(item=>item.json_status==='passed'&&item.source_status==='passed'&&!item.duplicate).length;
    $('tv-progress').max=Math.max(1,auditing?Number(summary.registered)||0:eligible);
    $('tv-progress').value=auditing?Number(report.processed)||0:Number(summary.browser_passed||0)+Number(summary.browser_failed||0);
    $('tv-progress').setAttribute('aria-label',auditing?'JSON·연결 원천 점검 진행률':'실제 브라우저 재생 대상 확인 진행률');
    $('tv-progress-text').textContent=`재생 통과 ${count(summary.browser_passed)} · 재생 실패 ${count(summary.browser_failed)} · 미검증 ${count(summary.pending)} · 중복 제외 ${count(summary.duplicates)}`;
    const checked=report.checked_at?new Date(report.checked_at):null;
    const checkedText=checked&&!Number.isNaN(checked.getTime())?new Intl.DateTimeFormat('ko-KR',{dateStyle:'medium',timeStyle:'medium',timeZone:'Asia/Seoul'}).format(checked)+' KST':'점검 중';
    $('tv-run').textContent=report.run_id?`검증 결과 시각 ${checkedText} · 회차 ${report.run_id}`:'검증 결과 없음';
    adminControl();renderRows();
  }
  async function request(url, options={}) {
    const response=await fetch(url,{...options,credentials:'same-origin',cache:'no-store',headers:{Accept:'application/json',...options.headers}});
    const data=await response.json();if(!response.ok)throw new Error(typeof data.detail==='string'?(reasons[data.detail]||data.detail):`조회 실패 (HTTP ${response.status})`);return data;
  }
  async function auth() {
    try {const session=await request('/admin/api/manage/status');state.csrf=session.authenticated?session.csrf_token||'':'';state.expires=Number(session.expires_at||0)*1000;}
    catch (_) {state.csrf='';state.expires=0;}adminControl();
  }
  function schedule() {
    clearTimeout(state.timer);
    if (state.active) state.timer=setTimeout(()=>{adminControl();if(document.hidden)schedule();else load();},state.report?.phase==='checking_json'||state.report?.phase==='awaiting_playback'?3000:15000);
  }
  async function load() {
    state.controller?.abort();const controller=new AbortController();state.controller=controller;$('tv-table').setAttribute('aria-busy','true');
    try {const report=await request(API,{signal:controller.signal});if(state.controller===controller&&!controller.signal.aborted){render(report);$('tv-message').textContent='';}}
    catch(error) {if(error.name!=='AbortError')$('tv-message').textContent=error.message || '검증 결과를 읽을 수 없습니다.';}
    finally {if(state.controller===controller){$('tv-table').setAttribute('aria-busy','false');schedule();}}
  }
  async function selectTab(view,focus=false) {
    state.active=view==='verification';
    for (const [id,active] of [['td-library-tab',!state.active],['td-verification-tab',state.active]]) {$(id).setAttribute('aria-selected',String(active));$(id).tabIndex=active?0:-1;if(active&&focus)$(id).focus();}
    $('td-library-panel').hidden=state.active;$('tv-root').hidden=!state.active;$('td-library-meta').hidden=state.active;
    window.TrainingTransform?.activate(state.active);
    const url=new URL(window.location.href);if(state.active)url.searchParams.set('view','verification');else url.searchParams.delete('view');window.history.replaceState(null,'',url);
    if(state.active){await Promise.all([auth(),load()]);}else{clearTimeout(state.timer);state.controller?.abort();}
  }
  for (const [id,view] of [['td-library-tab','library'],['td-verification-tab','verification']]) {
    $(id).addEventListener('click',()=>selectTab(view));
    $(id).addEventListener('keydown',event=>{if(['ArrowLeft','ArrowRight','Home','End'].includes(event.key)){event.preventDefault();return selectTab(event.key==='Home'?'library':event.key==='End'?'verification':view==='library'?'verification':'library',true);}});
  }
  for (const id of ['tv-status-filter','tv-process-filter','tv-search']) $(id).addEventListener(id==='tv-search'?'input':'change',()=>{state.page=1;renderRows();});
  $('tv-prev').addEventListener('click',()=>{if(state.page>1){state.page--;renderRows();}});
  $('tv-next').addEventListener('click',()=>{state.page++;renderRows();});
  $('tv-refresh').addEventListener('click',()=>Promise.all([auth(),load()]));
  $('tv-start').addEventListener('click',async()=>{
    await auth();if(!state.csrf||Date.now()>=state.expires){$('tv-message').textContent='관리자 로그인 후 시작할 수 있습니다.';return;}
    state.starting=true;adminControl();
    try {render(await request(API+'/start',{method:'POST',headers:{'X-DB-CSRF':state.csrf}}));$('tv-message').textContent='JSON·원천 점검을 시작했습니다. 실제 브라우저 재생은 운영 검증이 진행되면 별도로 반영됩니다.';schedule();}
    catch(error){$('tv-message').textContent=error.message;}
    finally{state.starting=false;adminControl();}
  });
  window.addEventListener('focus',()=>{if(state.active)auth();});
  window.addEventListener('pagehide',()=>{clearTimeout(state.timer);state.controller?.abort();});
  selectTab(new URL(window.location.href).searchParams.get('view')==='verification'?'verification':'library');
})();
