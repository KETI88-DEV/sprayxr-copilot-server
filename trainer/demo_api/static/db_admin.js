(() => {
  'use strict';
  const root = document.getElementById('manage-login');
  if (!root) return;
  const $ = id => document.getElementById(id);
  const labels = {users:'훈련자',sessions:'훈련 세션',sequences:'시퀀스 기록',sequence_analysis_results:'시퀀스 분석 결과',query_turns:'질문·답변',completions:'최종 리포트',api_calls:'API 호출 이력',
    session_id:'세션 ID',user_id:'훈련자 ID',name:'이름',scenario_id:'콘텐츠',creation_origin:'출처',session_status:'상태',sequence_id:'시퀀스 ID',report_id:'리포트 ID',report_status:'리포트 상태',skill_level:'숙련도',dominant_hand:'주 사용 손',age:'나이',gender:'성별',prompt:'질문',path:'호출 경로',created_at:'생성 시각',received_at:'수신 시각',requested_at:'요청 시각',imported_at:'가져온 시각',submitted_at:'접수 시각',started_at:'시작 시각',__created_at:'생성 시각 미기록'};
  const text = (tag, value, cls = '') => { const node = document.createElement(tag); node.textContent = String(value ?? '미기록'); if (cls) node.className = cls; return node; };
  const stamp = value => { const date = new Date(value); return Number.isNaN(date.getTime()) ? value : new Intl.DateTimeFormat('ko-KR',{dateStyle:'short',timeStyle:'medium',timeZone:'Asia/Seoul'}).format(date); };
  const display = (key, value) => {
    if (value === null || value === '') return '미기록';
    if (key === 'creation_origin') return ({real:'일반 API 연동',test:'모의 테스트'})[value] || '출처 미기록';
    if (key.endsWith('_at')) return stamp(value);
    const names = {ACTIVE:'진행 중',COMPLETED:'종료',pending:'분석 대기',complete:'완료',failed:'실패',novice:'일반 훈련자',expert:'전문가',right:'오른손',left:'왼손',unknown:'미기록'};
    return names[value] || String(value);
  };
  let csrf = '', page = 1, total = 0, editing, deleting, expiration;
  let listRequest = 0;
  let editRequest = 0, deleteRequest = 0;
  let accessRequest = 0, accessPage = 1, accessTotal = 0;
  const message = (id, value, error = false) => { $(id).textContent = value; $(id).classList.toggle('error', error); };
  function locked() {
    csrf = ''; ++listRequest; ++editRequest; ++deleteRequest; ++accessRequest; clearTimeout(expiration);
    $('manage-workspace').hidden = true; root.hidden = false;
    $('manage-badge').textContent = '로그인 필요';
    $('manage-password').value = '';
    // Remove privileged data from the DOM on logout or expiry, including closed panels.
    editing = null; deleting = null;
    for (const id of ['manage-records','manage-history','manage-access-records','manage-account','manage-expiry','manage-access-pagination','manage-edit-summary','manage-edit-fields','manage-delete-summary','manage-delete-counts','manage-delete-notice','manage-confirmation-value']) $(id).replaceChildren();
    $('manage-confirmation').value = '';
    $('manage-access-panel').open = false;
    for (const id of ['manage-edit','manage-delete']) if ($(id).open) $(id).close();
  }
  async function api(path, options = {}) {
    const headers = {Accept:'application/json', ...(options.body ? {'Content-Type':'application/json'} : {}), ...(options.method === 'POST' ? {'X-DB-CSRF':csrf} : {})};
    const result = await fetch('/admin/api/manage/' + path, {...options, headers, credentials:'same-origin', cache:'no-store'});
    const data = await result.json();
    if (!result.ok) {
      if (result.status === 401 && path !== 'login') { locked(); message('manage-login-message','로그인이 만료되었습니다. 다시 로그인하세요.',true); }
      throw new Error(typeof data.detail === 'string' ? data.detail : '입력값을 확인해 주세요.');
    }
    return data;
  }
  async function authenticated(data) {
    csrf = data.csrf_token; root.hidden = true; $('manage-workspace').hidden = false;
    $('manage-badge').textContent = '관리자 접속 중';
    $('manage-account').textContent = `${data.username} · 수정·삭제 권한 활성화`;
    $('manage-expiry').textContent = `로그인 유지: ${stamp(new Date(data.expires_at * 1000).toISOString())}까지`;
    clearTimeout(expiration); expiration = setTimeout(() => { locked(); message('manage-login-message','관리자 접속 시간이 만료되었습니다.',true); }, Math.max(0,data.expires_at * 1000 - Date.now()));
    await load(); await history();
  }
  $('manage-login-form').addEventListener('submit', async event => {
    event.preventDefault(); const button = event.submitter || event.currentTarget.querySelector('button'); button.disabled = true;
    message('manage-login-message','로그인 확인 중…');
    const username = $('manage-username').value, password = $('manage-password').value;
    try { const data = await api('login',{method:'POST',body:JSON.stringify({username,password})}); $('manage-password').value = ''; message('manage-login-message',''); await authenticated(data); }
    catch(error) { message('manage-login-message',error.message,true); }
    finally { button.disabled = false; }
  });
  $('manage-logout').addEventListener('click', async () => {
    try { await api('logout',{method:'POST'}); locked(); message('manage-login-message','로그아웃했습니다.'); }
    catch(error) { message('manage-message',error.message,true); }
  });
  async function load() {
    const request = ++listRequest, selected = $('manage-table').value;
    $('manage-records').setAttribute('aria-busy','true');
    try {
      const params = new URLSearchParams({table:selected,q:$('manage-query').value,page});
      const data = await api('records?' + params);
      if (request !== listRequest || !csrf) return;
      total = data.total; page = data.page;
      const available = Object.entries(data.tables);
      $('manage-table').replaceChildren(...available.map(([value,label]) => new Option(label,value)));
      $('manage-table').value = selected;
      const table = document.createElement('table'), thead = document.createElement('thead'), head = document.createElement('tr');
      const columns = data.columns.filter(key => !['age','gender','dominant_hand'].includes(key) || selected === 'users');
      for (const key of columns) head.append(text('th',labels[key] || key)); head.append(text('th','관리'));
      thead.append(head); table.append(thead);
      const tbody = document.createElement('tbody');
      for (const item of data.rows) {
        const row = document.createElement('tr'), controls = document.createElement('td'); controls.className = 'manage-row-actions';
        const detail = text('button','상세','manage-secondary'); detail.type='button';
        detail.dataset.detailUrl = '/admin/api/record?' + new URLSearchParams({table:selected,rowid:item.rowid}); controls.append(detail);
        if (selected === 'users' || selected === 'sessions') { const button=text('button','수정'); button.type='button'; button.addEventListener('click',() => edit(selected,item.rowid)); controls.append(button); }
        const remove=text('button','삭제','manage-danger-outline'); remove.type='button'; remove.addEventListener('click',() => preview(selected,item.rowid)); controls.append(remove);
        for (const key of columns) { const value=display(key,item.values[key]); const cell=text('td',value.length>100?value.slice(0,100)+'…':value); cell.title=value; row.append(cell); }
        row.append(controls);
        tbody.append(row);
      }
      if (!data.rows.length) { const row=document.createElement('tr'), cell=text('td','조건에 맞는 기록이 없습니다.','empty'); cell.colSpan=columns.length+1; row.append(cell); tbody.append(row); }
      table.append(tbody); $('manage-records').replaceChildren(table);
      $('manage-page').textContent=`전체 ${total.toLocaleString()}건 · ${page} / ${Math.max(1,Math.ceil(total/25))}페이지`;
      $('manage-previous').disabled=page<=1; $('manage-next').disabled=page*25>=total;
    } catch(error) { if(request===listRequest)message('manage-message',error.message,true); }
    finally { if(request===listRequest)$('manage-records').setAttribute('aria-busy','false'); }
  }
  $('manage-search').addEventListener('submit',event => {event.preventDefault();page=1;load();});
  $('manage-table').addEventListener('change',() => {page=1;message('manage-message','');load();});
  $('manage-reload').addEventListener('click',() => {load();history();if($('manage-access-panel').open)access();});
  $('manage-previous').addEventListener('click',() => {if(page>1){page--;load();}});
  $('manage-next').addEventListener('click',() => {if(page*25<total){page++;load();}});
  function summary(target, values) { return `${labels[target]} · ` + Object.entries(values).filter(([key])=>['user_id','session_id','name','sequence_id','report_id'].includes(key)).map(([key,value])=>`${labels[key]}: ${value ?? '미기록'}`).join(' / '); }
  async function edit(table,rowid) {
    const request = ++editRequest;
    try {
      const data = await api('record?'+new URLSearchParams({table,rowid}));
      if(request !== editRequest || !csrf)return;
      editing=data;
      $('manage-edit-summary').textContent=summary(table,editing.summary) + (table==='sessions'?' · 출처 변경은 관리자 수동 분류로 이력에 기록됩니다.':'');
      $('manage-edit-fields').replaceChildren();
      for (const [key,spec] of Object.entries(editing.fields)) {
        const label=text('label',spec.label,'manage-label'); let field;
        if(spec.choices) { field=document.createElement('select'); for(const [value,name] of Object.entries(spec.choices))field.append(new Option(name,value)); }
        else {field=document.createElement('input');field.type=spec.type==='number'?'number':'text';if(spec.max_length)field.maxLength=spec.max_length;if(spec.min!==undefined)field.min=spec.min;if(spec.max!==undefined)field.max=spec.max;}
        field.name=key; field.value=editing.values[key]??''; label.append(field); $('manage-edit-fields').append(label);
      }
      message('manage-edit-message',''); $('manage-edit').showModal();
    } catch(error) {message('manage-message',error.message,true);}
  }
  $('manage-edit-form').addEventListener('submit',async event => {
    event.preventDefault(); if(!editing)return;
    const fields={};
    for(const field of $('manage-edit-fields').querySelectorAll('input,select')) {
      const spec=editing.fields[field.name]; let value=field.value;
      if(spec.nullable && value==='')value=null; else if(spec.type==='number')value=Number(value);
      if(value!==editing.values[field.name])fields[field.name]=value;
    }
    if(!Object.keys(fields).length){message('manage-edit-message','변경된 항목이 없습니다.');return;}
    const button=event.submitter;button.disabled=true;
    busy($('manage-edit'),true);
    try {await api('update',{method:'POST',body:JSON.stringify({table:editing.table,rowid:editing.rowid,version:editing.version,fields})});$('manage-edit').close();message('manage-message','수정했습니다. 변경 전 DB 백업과 변경 이력을 저장했습니다.');await load();await history();}
    catch(error){message('manage-edit-message',error.message,true);}
    finally{busy($('manage-edit'),false);button.disabled=false;}
  });
  async function preview(table,rowid) {
    const request = ++deleteRequest;
    try {
      const data=await api('preview-delete',{method:'POST',body:JSON.stringify({table,rowid})});
      if(request !== deleteRequest || !csrf)return;
      deleting={table,rowid,...data};
      $('manage-delete-summary').textContent=summary(table,data.summary);
      $('manage-delete-counts').replaceChildren();
      for(const [key,count] of Object.entries(data.counts)){const cell=text('div',labels[key]||key);cell.append(text('strong',count.toLocaleString()+'건'));$('manage-delete-counts').append(cell);}
      $('manage-delete-notice').textContent=data.notice;
      $('manage-confirmation-value').textContent=data.confirmation;
      $('manage-confirmation').value='';$('manage-delete-submit').disabled=true;message('manage-delete-message','');
      $('manage-delete').showModal();
    }catch(error){message('manage-message',error.message,true);}
  }
  $('manage-confirmation').addEventListener('input',()=>{$('manage-delete-submit').disabled=!deleting||$('manage-confirmation').value!==deleting.confirmation;});
  $('manage-delete-form').addEventListener('submit',async event=>{
    event.preventDefault();if(!deleting)return;const button=$('manage-delete-submit');button.disabled=true;
    busy($('manage-delete'),true);
    try {
      await api('delete',{method:'POST',body:JSON.stringify({table:deleting.table,rowid:deleting.rowid,preview_token:deleting.preview_token,confirmation:$('manage-confirmation').value})});
      $('manage-delete').close();message('manage-message','확인한 기록을 삭제했습니다. 삭제 전 DB 백업과 변경 이력을 저장했습니다.');await load();await history();
    }catch(error){message('manage-delete-message',error.message,true);}
    finally{busy($('manage-delete'),false);button.disabled=!deleting||$('manage-confirmation').value!==deleting.confirmation;}
  });
  function busy(dialog,value){dialog.dataset.busy=value?'1':'0';for(const field of dialog.querySelectorAll('input,select,button'))field.disabled=value;}
  for(const button of document.querySelectorAll('[data-manage-close]'))button.addEventListener('click',()=>button.closest('dialog').close());
  for(const id of ['manage-edit','manage-delete'])$(id).addEventListener('cancel',event=>{if($(id).dataset.busy==='1')event.preventDefault();});
  $('manage-edit').addEventListener('close',()=>{++editRequest;editing=null;});$('manage-delete').addEventListener('close',()=>{++deleteRequest;deleting=null;});
  async function history() {
    const activeCsrf=csrf;
    try {
      const data=await api('history');if(!csrf||csrf!==activeCsrf)return;$('manage-history').replaceChildren();
      if(!data.rows.length)$('manage-history').append(text('p','아직 관리자 변경 이력이 없습니다.','manage-notes'));
      for(const item of data.rows){const entry=text('div',`${item.action==='delete'?'삭제':'수정'} · ${labels[item.table_name]||item.table_name} #${item.record_rowid}`,'manage-audit-item');const meta=text('span',`${item.actor} · ${stamp(item.created_at)}`);meta.append(text('small',`변경 전 백업: ${item.backup_name}`));entry.append(meta);$('manage-history').append(entry);}
    }catch(error){message('manage-message',error.message,true);}
  }
  const accessSources={direct:'직접 방문·출처 없음 (미확인)',internal:'사이트 내부 이동 (추정)',external:'외부 링크 (추정)'};
  const accessPages={home:'Home',docs:'API 문서',admin:'DB 뷰어·관리자','motion-viewer':'모션 뷰어','query-chat':'AI 질문',feedback:'사용자 의견'};
  async function access() {
    const request=++accessRequest;
    $('manage-access-records').setAttribute('aria-busy','true');message('manage-access-message','');
    try {
      const params=new URLSearchParams({page:accessPage,limit:50,page_filter:$('manage-access-page').value,client_ip:$('manage-access-ip').value.trim(),source:$('manage-access-source').value});
      for(const key of ['since','until']){const value=$('manage-access-'+key).value;if(value)params.set(key,new Date(value+':00+09:00').toISOString());}
      const data=await api('page-access?'+params);if(request!==accessRequest||!csrf)return;
      accessPage=data.page;accessTotal=data.total;
      const table=document.createElement('table'),head=document.createElement('thead'),tr=document.createElement('tr');
      for(const value of ['접근 시각 (KST)','연결 IP','IP망 라벨 (추정)','IP망 분류 근거','페이지','경로','이동 출처 (추정)','이동 출처 근거','신원'])tr.append(text('th',value));head.append(tr);table.append(head);
      const body=document.createElement('tbody');
      for(const item of data.rows){const row=document.createElement('tr');for(const value of [stamp(item.created_at),item.client_ip||'미기록',item.network_label||'미확인',item.network_basis||'현재 IP망 규칙의 일치 근거 없음',accessPages[item.page]||item.page,item.path,accessSources[item.source]||'미확인',item.source_basis==='request_header_estimate'?'요청 헤더·미검증':'근거 없음','미확인']){const cell=text('td',value);cell.title=String(value);row.append(cell);}body.append(row);}
      if(!data.rows.length){const row=document.createElement('tr'),cell=text('td','조건에 맞는 접근 기록이 없습니다.','empty');cell.colSpan=9;row.append(cell);body.append(row);}
      table.append(body);$('manage-access-records').replaceChildren(table);
      $('manage-access-pagination').textContent=`전체 ${accessTotal.toLocaleString()}건 · ${accessPage} / ${Math.max(1,Math.ceil(accessTotal/50))}페이지`;
      $('manage-access-previous').disabled=accessPage<=1;$('manage-access-next').disabled=accessPage*50>=accessTotal;
    }catch(error){if(request===accessRequest&&csrf)message('manage-access-message',error.message,true);}
    finally{if(request===accessRequest)$('manage-access-records').setAttribute('aria-busy','false');}
  }
  $('manage-access-panel').addEventListener('toggle',()=>{if($('manage-access-panel').open&&csrf)access();});
  $('manage-access-search').addEventListener('submit',event=>{event.preventDefault();accessPage=1;access();});
  $('manage-access-previous').addEventListener('click',()=>{if(accessPage>1){accessPage--;access();}});
  $('manage-access-next').addEventListener('click',()=>{if(accessPage*50<accessTotal){accessPage++;access();}});
  api('status').then(data=>data.authenticated?authenticated(data):locked()).catch(error=>{locked();message('manage-login-message',error.message,true);});
})();
