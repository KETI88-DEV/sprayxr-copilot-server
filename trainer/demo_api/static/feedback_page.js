(() => {
  'use strict';
  const byId=id=>document.getElementById(id); let page=1;
  const node=(tag,text,cls)=>{const n=document.createElement(tag);n.textContent=text;if(cls)n.className=cls;return n;};
  async function load(){
    byId('opinion-state').textContent='불러오는 중…';
    try{
      const r=await fetch('/api/feedback/list?'+new URLSearchParams({status:byId('opinion-filter').value,page}),{cache:'no-store'});
      if(!r.ok)throw new Error('의견을 불러오지 못했습니다.'); const data=await r.json();
      byId('opinion-items').replaceChildren();
      for(const row of data.items){
        const card=node('article','','opinion-card'),meta=node('div','','opinion-meta'),status=node('span',data.statuses[row.status]||row.status,'opinion-status'); status.dataset.status=row.status;
        meta.append(status,node('span',row.name||'익명'),node('time',new Date(row.created_at).toLocaleString('ko-KR')),node('span',row.origin_server));
        card.append(meta,node('div',row.message,'opinion-message'));
        if(row.resolution)card.append(node('div','처리 내용 · '+row.resolution,'opinion-resolution'));
        byId('opinion-items').append(card);
      }
      byId('opinion-state').textContent=data.total?'':'접수된 의견이 없습니다.';
      byId('opinion-total').textContent=`총 ${data.total.toLocaleString()}건`;
      byId('opinion-page').textContent=`${page} / ${Math.max(1,Math.ceil(data.total/data.page_size))}`;
      byId('opinion-prev').disabled=page<=1;byId('opinion-next').disabled=page*data.page_size>=data.total;
    }catch(e){byId('opinion-state').textContent=e.message||'연결을 확인하세요.';}
  }
  byId('opinion-filter').addEventListener('change',()=>{page=1;load();});
  byId('opinion-prev').addEventListener('click',()=>{page--;load();});byId('opinion-next').addEventListener('click',()=>{page++;load();});load();
})();
