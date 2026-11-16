(() => {
  'use strict';
  const byId=id=>document.getElementById(id),node=(tag,text)=>{const e=document.createElement(tag);e.textContent=text;return e;};let csrf;
  async function load(){
    const state=byId('answer-review-state'),items=byId('answer-review-items');
    try{
      const session=await fetch('/admin/api/manage/status',{cache:'no-store'}).then(r=>r.json());
      if(!session.authenticated){state.textContent='관리자 로그인 후 답변 평가를 검토할 수 있습니다.';return;}
      csrf=session.csrf_token;const status=byId('answer-review-filter').value;
      const r=await fetch('/admin/api/feedback/query-ratings'+(status?'?status='+status:''),{cache:'no-store'});if(!r.ok)throw new Error('목록을 불러오지 못했습니다.');
      const data=await r.json();items.replaceChildren();state.textContent=`${data.items.length}건 · ${session.username} 관리자`;
      for(const row of data.items){
        const card=node('article','');card.className='answer-review-card';
        const meta=node('p',`${row.rating==='up'?'👍 도움이 됨':'👎 도움이 안 됨'} · ${row.name||'익명'} · ${row.origin_server} · ${new Date(row.created_at).toLocaleString('ko-KR')}`);meta.className='answer-review-meta';
        card.append(meta,node('strong','질문'),node('pre',row.question),node('strong','실제 응답'),node('pre',row.answer),node('p','사용자 코멘트 · '+(row.comment||'없음')));
        if(row.suggested_answer)card.append(node('p','사용자 제안 · '+row.suggested_answer));
        const answerLabel=node('label','관리자가 검증한 교정 답변'),answer=node('textarea','');answer.rows=4;answer.maxLength=8000;answer.value=row.corrected_answer;answerLabel.append(answer);
        const noteLabel=node('label','검토 메모'),note=node('textarea','');note.rows=2;note.maxLength=4000;note.value=row.review_note;noteLabel.append(note);
        const select=node('select','');for(const [value,text]of Object.entries({pending:'미검토',approved:'학습 후보 승인',rejected:'학습 제외'})){const o=node('option',text);o.value=value;o.selected=row.review_status===value;select.append(o);}
        const save=node('button','검토 저장'),result=node('span','');result.setAttribute('role','status');
        save.addEventListener('click',async()=>{save.disabled=true;result.textContent='저장 중…';try{const r=await fetch('/admin/api/manage/query-feedback-review',{method:'POST',headers:{'Content-Type':'application/json','X-DB-CSRF':csrf},body:JSON.stringify({feedback_id:row.feedback_id,review_status:select.value,corrected_answer:answer.value,review_note:note.value})});const data=await r.json();if(!r.ok)throw new Error(data.detail||'저장하지 못했습니다.');result.textContent='저장했습니다.';}catch(e){result.textContent=e.message;}finally{save.disabled=false;}});
        card.append(answerLabel,noteLabel,select,save,result);items.append(card);
      }
    }catch(e){state.textContent=e.message||'연결을 확인하세요.';}
  }
  byId('answer-review-filter').addEventListener('change',load);load();
})();
