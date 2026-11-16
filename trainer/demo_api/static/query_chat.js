(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const make = (tag, cls, text) => { const e=document.createElement(tag); if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e; };
  const uuid = () => globalThis.crypto?.randomUUID?.() || 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g,c=>{const r=Math.random()*16|0;return(c==='x'?r:(r&3|8)).toString(16);});
  let conversation='query-chat:'+uuid(), busy=false;
  async function requestJson(url,options,timeoutMs){
    const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),timeoutMs);
    try{const response=await fetch(url,{...options,signal:controller.signal});const body=await response.json();return {response,body};}
    catch(error){if(error.name==='AbortError')throw new Error('응답 확인 시간이 지났습니다. 연결을 확인한 뒤 다시 시도하세요.');throw error;}
    finally{clearTimeout(timer);}
  }
  function bottom(){ $('messages').scrollTop=$('messages').scrollHeight; }
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
  $('new-chat').addEventListener('click',()=>{if(busy)return;conversation='query-chat:'+uuid();$('messages').replaceChildren();$('chat-status').textContent='새 대화를 시작합니다.';$('prompt').focus();});
  document.querySelectorAll('[data-prompt]').forEach(button=>button.addEventListener('click',()=>{$('prompt').value=button.dataset.prompt;$('prompt').focus();}));
  $('prompt').addEventListener('keydown',event=>{if(event.key==='Enter'&&!event.shiftKey&&!event.isComposing){event.preventDefault();$('chat-form').requestSubmit();}});
  $('chat-form').addEventListener('submit',async event=>{
    event.preventDefault();const prompt=$('prompt').value.trim();if(!prompt||busy)return;
    const mode=$('response-mode').value,lang=$('response-language').value;
    const payload={user_id:'query-chat',prompt,task_type:'knowledge_search',response_mode:mode,language:'ko',response_language:lang,use_rag:'Y',request_id:'CHAT-'+uuid()};
    for(const [id,key] of [['scenario-id','scenario_id'],['session-id','session_id'],['context','context']]){const value=$(id).value.trim();if(value)payload[key]=value;}
    busy=true;$('send').disabled=$('new-chat').disabled=true;add('user',prompt);$('prompt').value='';const card=add('assistant','답변을 작성하고 있습니다…');card.querySelector('.message-text').classList.add('pending');const started=performance.now();$('chat-status').textContent='질문을 처리하고 있습니다…';
    try{const {response,body}=await requestJson('/api/v1/copilot/query',{method:'POST',headers:{'Content-Type':'application/json','X-Copilot-Conversation':conversation,'X-Copilot-Source':'query-chat'},body:JSON.stringify(payload)},90000);if(!response.ok||body.status!=='SUCCESS'||!body.data?.answer)throw new Error(typeof body.message==='string'?body.message:'질문을 처리하지 못했습니다. 잠시 후 다시 시도하세요.');card.querySelector('.message-text').textContent=body.data.answer;const elapsed=((performance.now()-started)/1000).toFixed(1);card.append(make('div','answer-meta',`${mode==='quick'?'빠른 안내 · Qwen3-8B':'상세 분석 · Qwen3.6-27B'} · ${elapsed}초`));rating(card,Number(response.headers.get('X-Copilot-Query-Turn')));$('chat-status').textContent='답변 아래에서 평가와 코멘트를 남길 수 있습니다.';}
    catch(error){card.classList.add('error');card.querySelector('.message-text').textContent=error.message||'서버 연결을 확인하세요.';$('chat-status').textContent='전송하지 못했습니다. 질문을 다시 입력해 주세요.';if(!$('prompt').value)$('prompt').value=prompt;}
    finally{card.querySelector('.message-text').classList.remove('pending');busy=false;$('send').disabled=$('new-chat').disabled=false;bottom();$('prompt').focus();}
  });
})();
