(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const make = (tag, cls, text) => { const e=document.createElement(tag); if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e; };
  const uuid = () => globalThis.crypto?.randomUUID?.() || 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g,c=>{const r=Math.random()*16|0;return(c==='x'?r:(r&3|8)).toString(16);});
  let conversation='query-chat:'+uuid(), busy=false;
  let selectedContext=null,contextPending=false,contextError=false,selectionGeneration=0;
  const smallScreen=window.matchMedia('(max-width:760px)');
  function controls(){ $('send').disabled=busy||contextPending||contextError;$('new-chat').disabled=busy; }
  function drawer(open){
    const mobile=smallScreen.matches;
    $('chat-sidebar').classList.toggle('is-open',open&&mobile);
    $('chat-sidebar').inert=mobile&&!open;
    $('chat-backdrop').hidden=!open||!mobile;
    $('open-settings').setAttribute('aria-expanded',String(open&&mobile));
    if(open&&mobile){$('close-settings').focus();}
  }
  function closeSettings(){drawer(false);if(smallScreen.matches)$('open-settings').focus();}
  $('open-settings').addEventListener('click',()=>drawer(true));
  $('close-settings').addEventListener('click',closeSettings);
  $('chat-backdrop').addEventListener('click',closeSettings);
  $('context-chip').addEventListener('click',()=>{drawer(true);if(!smallScreen.matches)$('training-worker').focus();});
  $('apply-training').addEventListener('click',()=>{closeSettings();if(!smallScreen.matches)$('prompt').focus();});
  document.addEventListener('keydown',event=>{
    if(!smallScreen.matches||!$('chat-sidebar').classList.contains('is-open'))return;
    if(event.key==='Escape'){event.preventDefault();closeSettings();}
    if(event.key==='Tab'){
      const nodes=[...$('chat-sidebar').querySelectorAll('button,select,input,a')].filter(e=>!e.disabled&&e.getClientRects().length);
      if(event.shiftKey&&document.activeElement===nodes[0]){event.preventDefault();nodes.at(-1)?.focus();}
      else if(!event.shiftKey&&document.activeElement===nodes.at(-1)){event.preventDefault();nodes[0]?.focus();}
    }
  });
  function viewport(){
    const v=window.visualViewport;
    document.documentElement.style.setProperty('--chat-viewport-height',`${smallScreen.matches&&v?v.height:window.innerHeight}px`);
    document.documentElement.style.setProperty('--chat-viewport-top',`${smallScreen.matches&&v?v.offsetTop:0}px`);
  }
  window.addEventListener('resize',viewport);
  window.visualViewport?.addEventListener('resize',viewport);
  window.visualViewport?.addEventListener('scroll',viewport);
  smallScreen.addEventListener('change',()=>{drawer(false);viewport();});
  for(const name of ['gesturestart','gesturechange'])document.addEventListener(name,event=>{if(smallScreen.matches&&event.cancelable)event.preventDefault();},{passive:false});
  document.addEventListener('touchmove',event=>{if(smallScreen.matches&&event.touches.length>1&&event.cancelable)event.preventDefault();},{passive:false});
  drawer(false);viewport();
  function sizePrompt(){const field=$('prompt');field.style.height='auto';field.style.height=Math.min(smallScreen.matches?120:150,Math.max(28,field.scrollHeight))+'px';}
  $('prompt').addEventListener('input',sizePrompt);
  async function requestJson(url,options,timeoutMs){
    const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),timeoutMs);
    try{const response=await fetch(url,{...options,signal:controller.signal});const body=await response.json();return {response,body};}
    catch(error){if(error.name==='AbortError')throw new Error('응답 확인 시간이 지났습니다. 연결을 확인한 뒤 다시 시도하세요.');throw error;}
    finally{clearTimeout(timer);}
  }
  function bottom(){ $('messages').scrollTop=$('messages').scrollHeight; }
  function options(id,rows,placeholder,value=''){
    const select=$(id);select.replaceChildren(new Option(placeholder,''));
    for(const row of rows||[])select.add(new Option(row.label,String(row.id)));
    select.value=value;select.disabled=!(rows||[]).length;
  }
  function invalidateContext(){
    selectedContext=null;contextPending=false;contextError=false;
    $('context-label').textContent='일반 질문 · 훈련 데이터 선택';controls();
    $('context-chip').title='훈련 데이터 선택';
  }
  function clearAfter(level){
    const ids=['training-worker','training-process','training-mode','training-session'];
    const labels=['작업자 선택','종류 선택','훈련모드 선택','훈련세션 선택'];
    for(let index=level+1;index<ids.length;index++)options(ids[index],[],labels[index]);
    $('training-zone').replaceChildren(new Option('전체 Zone','all'));$('training-zone').disabled=true;
    invalidateContext();
  }
  async function loadOptions(level=-1){
    const generation=++selectionGeneration;
    const values={worker_id:$('training-worker').value||'all',process:$('training-process').value||'all',scenario_id:$('training-mode').value||'all'};
    $('training-status').textContent='조건에 맞는 데이터를 불러오는 중…';
    try{
      const {response,body}=await requestJson('/query-chat/api/options?'+new URLSearchParams(values),{},20000);
      if(generation!==selectionGeneration)return;
      if(!response.ok)throw new Error('훈련 데이터 목록을 불러오지 못했습니다.');
      if(level<0)options('training-worker',body.workers,'작업자 선택',values.worker_id==='all'?'':values.worker_id);
      if(level<=0)options('training-process',body.processes,'종류 선택');
      if(level<=1)options('training-mode',body.modes,'훈련모드 선택');
      options('training-session',body.sessions,'훈련세션 선택');
      $('training-status').textContent=values.worker_id==='all'?'작업자부터 차례대로 선택하세요.':values.process==='all'?'종류를 선택하세요.':values.scenario_id==='all'?'훈련모드를 선택하세요.':`조건에 맞는 훈련세션 ${(body.sessions||[]).length.toLocaleString()}건`;
    }catch(error){if(generation===selectionGeneration)$('training-status').textContent=error.message||'목록을 불러오지 못했습니다.';}
  }
  function describeContext(data){
    const s=data.selection||{},process={spray:'스프레이',blasting:'블라스팅',grinding:'그라인딩'}[s.process]||s.process;
    const zone=s.zone_key==='all'?'전체 Zone':(data.zones||[]).find(row=>row.id===s.zone_key)?.label||s.zone_key;
    return [s.worker_name,process,s.mode_label,s.record,zone].filter(Boolean).join(' · ');
  }
  async function loadContext(){
    const id=$('training-session').value;
    if(!id){selectionGeneration++;$('training-zone').disabled=true;invalidateContext();return;}
    const generation=++selectionGeneration;
    contextPending=true;contextError=false;selectedContext=null;controls();
    $('training-status').textContent='선택한 기록의 문맥을 불러오는 중…';
    const zone=$('training-zone').value||'all';
    try{
      const {response,body}=await requestJson('/query-chat/api/context?'+new URLSearchParams({dataset_id:id,zone_key:zone}),{},30000);
      if(generation!==selectionGeneration)return;
      if(!response.ok||!body.context)throw new Error(typeof body.detail==='string'?body.detail:'선택한 기록을 불러오지 못했습니다.');
      selectedContext=body;
      $('training-zone').replaceChildren(new Option('전체 Zone','all'),...(body.zones||[]).map(row=>new Option(row.label,String(row.id))));
      $('training-zone').value=zone;$('training-zone').disabled=!(body.zones||[]).length;
      $('context-label').textContent=describeContext(body);
      $('context-chip').title=describeContext(body);
      const notes=[];
      if(body.warning)notes.push(body.warning);
      if(body.selection?.quality_status!=='complete')notes.push('확인 필요 데이터');
      if(!(body.zones||[]).length)notes.push('Zone 정보 없음 · 기록 전체로 질문');
      $('training-status').textContent=notes.length?notes.join(' / '):'선택한 훈련 기록과 Zone을 기준으로 답합니다.';
      conversation='query-chat:'+uuid();
    }catch(error){if(generation===selectionGeneration){contextError=true;$('training-status').textContent=(error.message||'문맥을 불러오지 못했습니다.')+' 다시 선택하거나 해제해 주세요.';$('context-label').textContent='훈련 문맥을 불러오지 못했습니다';}}
    finally{if(generation===selectionGeneration){contextPending=false;controls();}}
  }
  ['training-worker','training-process','training-mode'].forEach((id,level)=>$(id).addEventListener('change',()=>{clearAfter(level);loadOptions(level);}));
  $('training-session').addEventListener('change',()=>{$('training-zone').replaceChildren(new Option('전체 Zone','all'));loadContext();});
  $('training-zone').addEventListener('change',loadContext);
  $('clear-training').addEventListener('click',()=>{selectionGeneration++;$('training-worker').value='';clearAfter(0);$('training-status').textContent='일반 질문으로 전환했습니다.';conversation='query-chat:'+uuid();});
  loadOptions();
  function add(role,text){ $('welcome')?.remove();const card=make('article','message '+role);card.append(make('div','message-role',role==='user'?'사용자':'AI Co-Pilot'),make('div','message-text',text));$('messages').append(card);bottom();return card; }
  function rating(card,turnId){
    if(!Number.isInteger(turnId)||turnId<1){card.append(make('p','answer-meta','답변 저장 번호를 확인하지 못해 평가를 남길 수 없습니다.'));return;}
    const tools=make('div','answer-tools'), good=make('button','','👍 도움 됨'),bad=make('button','','👎 개선 필요'),box=make('div','feedback-box');
    good.type=bad.type='button';good.setAttribute('aria-pressed','false');bad.setAttribute('aria-pressed','false');box.hidden=true;
    const label=make('label','','평가 코멘트 (선택)'),comment=make('textarea');comment.rows=2;comment.maxLength=2000;comment.placeholder='어떤 점이 좋았거나 아쉬웠나요?';comment.id='comment-'+turnId;label.htmlFor=comment.id;
    const correctLabel=make('label','','더 나은 답변 (선택)'),correct=make('textarea');correct.rows=2;correct.maxLength=4000;correct.id='correct-'+turnId;correctLabel.htmlFor=correct.id;correct.placeholder='원하셨던 답변이나 수정할 내용을 적어주세요.';
    const save=make('button','','평가 보내기'),status=make('p');status.setAttribute('role','status');save.type='button';box.append(label,comment,correctLabel,correct,save,status);tools.append(good,bad);card.append(tools,box);
    let chosen=null,submission=uuid(),sent=false,pending=false,lastPayload=null;
    function choose(value){if(sent||pending)return;chosen=value;good.setAttribute('aria-pressed',String(value==='up'));bad.setAttribute('aria-pressed',String(value==='down'));box.hidden=false;comment.focus();bottom();}
    good.addEventListener('click',()=>choose('up'));bad.addEventListener('click',()=>choose('down'));
    save.addEventListener('click',async()=>{
      if(!chosen||sent||pending)return;pending=true;save.disabled=good.disabled=bad.disabled=true;comment.disabled=correct.disabled=true;status.textContent='저장 중…';
      const fields={query_turn_id:turnId,rating:chosen,comment:comment.value.trim(),suggested_answer:correct.value.trim(),name:$('reviewer-name').value.trim()},serialized=JSON.stringify(fields);
      if(lastPayload!==null&&lastPayload!==serialized)submission=uuid();lastPayload=serialized;
      try{const {response,body:data}=await requestJson('/api/query-feedback',{method:'POST',headers:{'Content-Type':'application/json','X-Copilot-Source':'query-chat'},body:JSON.stringify({submission_id:submission,...fields})},12000);if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:'평가를 저장하지 못했습니다.');sent=true;status.textContent='평가가 저장되었습니다. 검토 후 학습 개선에 활용합니다.';save.textContent='접수 완료';}
      catch(error){status.textContent=error.message||'연결을 확인하고 다시 시도하세요.';}
      finally{pending=false;save.disabled=good.disabled=bad.disabled=sent;comment.disabled=correct.disabled=sent;}
    });
  }
  $('response-mode').addEventListener('change',()=>{$('mode-label').textContent=$('response-mode').selectedOptions[0].textContent;});
  $('new-chat').addEventListener('click',()=>{if(busy)return;conversation='query-chat:'+uuid();$('messages').replaceChildren();$('chat-status').textContent='새 대화를 시작합니다.';if(!smallScreen.matches)$('prompt').focus();});
  document.querySelectorAll('[data-prompt]').forEach(button=>button.addEventListener('click',()=>{$('prompt').value=button.dataset.prompt;$('prompt').focus();}));
  $('prompt').addEventListener('keydown',event=>{if(event.key==='Enter'&&!event.shiftKey&&!event.isComposing){event.preventDefault();$('chat-form').requestSubmit();}});
  $('chat-form').addEventListener('submit',async event=>{
    event.preventDefault();const prompt=$('prompt').value.trim();if(!prompt||busy||contextPending||contextError)return;
    const mode=$('response-mode').value,lang=$('response-language').value;
    const payload={user_id:'query-chat',prompt,task_type:'knowledge_search',response_mode:mode,summarize:false,language:'ko',response_language:lang,use_rag:'Y',request_id:'CHAT-'+uuid()};
    const questionContext=selectedContext;
    if(questionContext)Object.assign(payload,{scenario_id:questionContext.scenario_id,session_id:questionContext.session_id,context:questionContext.context});
    busy=true;controls();add('user',prompt);$('prompt').value='';sizePrompt();const card=add('assistant','답변을 작성하고 있습니다…');card.querySelector('.message-text').classList.add('pending');const started=performance.now();$('chat-status').textContent='질문을 처리하고 있습니다…';
    try{const {response,body}=await requestJson('/api/v1/copilot/query',{method:'POST',headers:{'Content-Type':'application/json','X-Copilot-Conversation':conversation,'X-Copilot-Source':'query-chat'},body:JSON.stringify(payload)},90000);if(!response.ok||body.status!=='SUCCESS'||!body.data?.answer)throw new Error(typeof body.message==='string'?body.message:'질문을 처리하지 못했습니다. 잠시 후 다시 시도하세요.');card.querySelector('.message-text').textContent=body.data.answer;window.CopilotAnswerSource.render(card,body.data,response.headers.get('X-Copilot-Answer-Source')||'');const elapsed=((performance.now()-started)/1000).toFixed(1);card.append(make('div','answer-meta',`${mode==='quick'?'빠른 안내':'상세 분석'} · ${elapsed}초`));if(questionContext)card.append(make('div','answer-meta',describeContext(questionContext)));rating(card,Number(response.headers.get('X-Copilot-Query-Turn')));$('chat-status').textContent='답변 아래에서 평가와 코멘트를 남길 수 있습니다.';}
    catch(error){card.classList.add('error');card.querySelector('.message-text').textContent=error.message||'서버 연결을 확인하세요.';$('chat-status').textContent='전송하지 못했습니다. 질문을 다시 입력해 주세요.';if(!$('prompt').value)$('prompt').value=prompt;}
    finally{card.querySelector('.message-text').classList.remove('pending');busy=false;controls();bottom();if(!smallScreen.matches)$('prompt').focus();}
  });
})();
