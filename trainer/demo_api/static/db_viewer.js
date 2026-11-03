(() => {
  'use strict';
  const dialog = document.getElementById('db-detail-modal');
  if (!dialog) return;
  const title = document.getElementById('db-detail-title');
  const content = document.getElementById('db-detail-content');
  const copy = document.getElementById('db-detail-copy');
  let controller, payload;
  const el = (tag, cls, text) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = String(text);
    return node;
  };
  const scalar = (value) => value === null ? 'null' : typeof value === 'string' ? value : JSON.stringify(value);
  function tree(key, value, depth = 0) {
    if (value === null || typeof value !== 'object') {
      const node = el('div', 'json-leaf');
      node.append(el('span', 'json-key', `${key}: `));
      const text = scalar(value);
      const leaf = el('span', value === null ? 'json-null' : typeof value === 'number' ? 'json-number' : 'json-value', text.length > 1200 ? text.slice(0,1200) + ' …' : text);
      node.append(leaf);
      if (text.length > 1200) {
        const more = el('button','json-more','문자열 더 보기'); more.type='button';
        more.addEventListener('click', () => { leaf.textContent = text.slice(0,100000); more.remove(); if (text.length > 100000) node.append(el('span','json-null',' (전체 값은 JSON 복사로 확인)')); });
        node.append(more);
      }
      return node;
    }
    const array = Array.isArray(value);
    const keys = array ? null : Object.keys(value);
    const length = array ? value.length : keys.length;
    const branch = el('details','json-branch');
    branch.append(el('summary','',`${key} ${array ? '[' : '{'} ${length.toLocaleString()}개 ${array ? ']' : '}'}`));
    let loaded = false, offset = 0;
    const children = el('div'); branch.append(children);
    function appendBatch() {
      const end = Math.min(offset+50,length);
      for (;offset<end;offset++) { const k=array?offset:keys[offset]; children.append(tree(k,value[k],depth+1)); }
      if (offset<length) {
        const more=el('button','json-more',`다음 50개 보기 (${offset.toLocaleString()} / ${length.toLocaleString()})`); more.type='button';
        more.addEventListener('click',()=>{more.remove();appendBatch();}); children.append(more);
      }
    }
    branch.addEventListener('toggle',()=>{if(branch.open&&!loaded){loaded=true;appendBatch();}});
    if (depth===0) branch.open=true;
    return branch;
  }
  function show(data) {
    title.textContent=data.title||'상세 파라미터'; content.replaceChildren();
    if(data.notice) content.append(el('p','detail-notice',data.notice));
    if(data.meta?.length){
      const meta=el('dl','detail-meta');
      for(const item of data.meta){const cell=el('div');cell.append(el('dt','',item.label),el('dd','',item.value??'미기록'));meta.append(cell);}
      content.append(meta);
    }
    for (const section of data.sections||[]) {
      const box=el('details','detail-section');box.open=true;box.append(el('summary','',section.title||'파라미터'));
      const body=el('div','detail-parameters');body.append(tree('data',section.data));box.append(body);content.append(box);
    }
    if(!data.sections?.length) content.append(el('p','detail-loading','표시할 상세 파라미터가 없습니다.'));
  }
  async function open(url){
    controller?.abort();controller=new AbortController();const current=controller;payload=null;copy.disabled=true;
    title.textContent='상세 파라미터';content.replaceChildren(el('div','detail-loading','파라미터를 불러오고 있습니다…'));
    if(!dialog.open) dialog.showModal();
    try{
      const result=await fetch(url,{signal:current.signal,headers:{Accept:'application/json'}});
      if(!result.ok) throw new Error(result.status===404?'해당 기록을 찾을 수 없습니다.':'상세 내용을 불러오지 못했습니다.');
      const data=await result.json();if(current!==controller||!dialog.open)return;payload=data;copy.disabled=false;show(data);
    }catch(error){if(error.name!=='AbortError')content.replaceChildren(el('p','detail-error',error.message));}
  }
  document.addEventListener('click',(event)=>{
    const target=event.target.closest('[data-detail-url]');if(!target)return;
    if(target.tagName==='TR'&&event.target.closest('a,button,summary,input,select'))return;
    event.preventDefault();open(target.dataset.detailUrl);
  });
  document.addEventListener('keydown',(event)=>{if((event.key==='Enter'||event.key===' ')&&event.target.matches('tr[data-detail-url]')){event.preventDefault();open(event.target.dataset.detailUrl);}});
  dialog.querySelector('.detail-close').addEventListener('click',()=>dialog.close());
  dialog.addEventListener('close',()=>{controller?.abort();payload=null;});
  dialog.addEventListener('click',(event)=>{if(event.target===dialog){const r=dialog.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)dialog.close();}});
  copy.addEventListener('click',async()=>{
    if(!payload)return;
    try{
      const text=JSON.stringify(payload,null,2);
      if(navigator.clipboard&&window.isSecureContext) await navigator.clipboard.writeText(text);
      else{
        const field=el('textarea');field.value=text;field.style.cssText='position:fixed;left:0;top:0;width:1px;height:1px;opacity:0';dialog.append(field);
        let copied=false;try{field.focus();field.select();copied=document.execCommand('copy');}finally{field.remove();}
        if(!copied)throw new Error('copy unavailable');
      }
      copy.textContent='복사 완료';
    }catch{copy.textContent='브라우저에서 복사를 지원하지 않습니다';}
    setTimeout(()=>{copy.textContent='JSON 복사';},2000);
  });
})();
