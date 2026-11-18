(() => {
'use strict';
const make=(tag,cls,text)=>{const el=document.createElement(tag);if(cls)el.className=cls;if(text!==undefined)el.textContent=text;return el;};
const dialog=make('dialog','knowledge-library');dialog.setAttribute('aria-labelledby','knowledge-library-title');
const header=make('header'),heading=make('div'),title=make('h2','','NCS·매뉴얼 학습 자료');title.id='knowledge-library-title';
heading.append(title,make('p','','검색에 연결된 텍스트와 등록된 원본 자료를 확인하세요.'));
const close=make('button','knowledge-close','×');close.type='button';close.setAttribute('aria-label','학습 자료 닫기');header.append(heading,close);
const note=make('p','knowledge-note','텍스트는 검색용 문단으로 재구성된 내용이며 원본 대조 교정본은 별도로 표시합니다. 제외 문단의 본문은 표시하지 않습니다. 등록된 원본 PDF·이미지는 검색용 교정 텍스트와 구분됩니다.');
const search=make('input','knowledge-search');search.type='search';search.placeholder='자료 내용 검색';search.setAttribute('aria-label','학습 자료 검색');
const layout=make('div','knowledge-layout'),list=make('section','knowledge-list'),detail=make('section','knowledge-detail');list.setAttribute('aria-label','학습 자료 목록');detail.setAttribute('aria-label','선택한 자료');detail.setAttribute('aria-live','polite');layout.append(list,detail);dialog.append(header,note,search,layout);document.body.append(dialog);
let listGeneration=0,detailGeneration=0,selectedId='',lastFocus=null,timer=null,total=0,offset=0;
async function read(url){const response=await fetch(url,{cache:'no-store',headers:{'X-Copilot-Source':'query-chat'}});const data=await response.json();if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:'학습 자료를 불러오지 못했습니다.');return data;}
function failure(target,error){target.replaceChildren(make('p','knowledge-empty',error.message||'연결을 확인해 주세요.'));}
function appendDocumentRows(rows){
 for(const row of rows){
  const button=make('button','knowledge-doc');button.type='button';button.dataset.docId=row.doc_id;
  button.append(make('strong','',row.title),make('small','',row.chunk_count.toLocaleString()+'개 문단'+(row.original_available?' · 원본 연결됨':' · 텍스트')+(row.corrected_count?' · '+row.corrected_count.toLocaleString()+'개 교정':'')+(row.excluded_count?' · '+row.excluded_count.toLocaleString()+'개 제외':'')));
  button.setAttribute('aria-pressed',String(row.doc_id===selectedId));button.addEventListener('click',()=>showDocument(row.doc_id));list.append(button);
 }
}
async function loadDocuments(more=false){
 const generation=++listGeneration,q=search.value.trim();if(!more){offset=0;list.replaceChildren(make('p','knowledge-empty','자료를 불러오는 중…'));}else list.querySelector('.knowledge-more')?.remove();
 try{
  const data=await read('/query-chat/api/knowledge/documents?'+new URLSearchParams({q,offset:String(offset),limit:'50'}));
  if(generation!==listGeneration)return;
  if(!more)list.replaceChildren();total=data.total;appendDocumentRows(data.documents||[]);offset+=(data.documents||[]).length;
  if(!offset){list.append(make('p','knowledge-empty','조건에 맞는 학습 자료가 없습니다.'));return;}
  if(offset<total){const button=make('button','knowledge-more','자료 더 보기 ('+offset.toLocaleString()+'/'+total.toLocaleString()+')');button.type='button';button.addEventListener('click',()=>loadDocuments(true));list.append(button);}
 }catch(error){if(generation===listGeneration)failure(list,error);}
}
function safeAssetUrl(value){try{const url=new URL(value,location.origin);return url.origin===location.origin&&url.pathname.startsWith('/query-chat/api/knowledge/files/')?url.href:null;}catch{return null;}}
function chunkArticle(chunk,label){
 const article=make('article','knowledge-chunk'+(chunk.excluded?' knowledge-excluded':''));article.append(make('small','',label));
 if(chunk.excluded)article.append(make('p','knowledge-warning','검색·학습 안내에서 제외됨 · '+(chunk.exclusion_reason||'검토 제외')));
 else if(chunk.corrected)article.append(make('p','knowledge-correction','원본 대조 교정본'));
 article.append(make('p','',chunk.text));return article;
}
async function showDocument(id,more=false,next=0){
 const generation=more?detailGeneration:++detailGeneration;
 if(!more){selectedId=id;detail.replaceChildren(make('p','knowledge-empty','본문을 불러오는 중…'));for(const b of list.querySelectorAll('[data-doc-id]'))b.setAttribute('aria-pressed',String(b.dataset.docId===id));}
 try{
  const data=await read('/query-chat/api/knowledge/documents/'+encodeURIComponent(id)+'?offset='+next+'&limit=20');
  if(generation!==detailGeneration)return;
  if(!more){
   detail.replaceChildren(make('h3','',data.title),make('p','knowledge-basis',data.text_basis+' · '+data.chunk_count.toLocaleString()+'개 문단'));
   if(data.corrected_count)detail.append(make('p','knowledge-correction',data.corrected_count.toLocaleString()+'개 문단은 원본 대조 교정본입니다. 검색과 다운로드에도 이 교정 텍스트를 사용합니다.'));
   if(data.excluded_count)detail.append(make('p','knowledge-warning','확인된 오류·손상이 있는 '+data.excluded_count.toLocaleString()+'개 문단은 검색·학습 안내에서 제외되어 본문을 표시하지 않습니다.'));
   const links=make('div','knowledge-document-tools'),download=make('a','','검색용 텍스트 다운로드');download.href='/query-chat/api/knowledge/documents/'+encodeURIComponent(id)+'/text';links.append(download);detail.append(links);
   if(data.assets?.length){
    const media=make('div','knowledge-originals');media.append(make('h4','','연결된 원본 PDF·이미지'),make('p','knowledge-basis',data.original_basis||'별도 등록된 원본 자료입니다. 검색용 교정 텍스트와 구분됩니다.'));
    for(const asset of data.assets){const url=safeAssetUrl(asset.url);if(!url)continue;const link=make('a','',asset.label+' ↗');link.href=url;link.target='_blank';link.rel='noopener noreferrer';media.append(link);if(asset.kind==='image'){const image=make('img');image.src=url;image.alt=asset.label;image.loading='lazy';media.append(image);}}
    detail.append(media);
   }else detail.append(make('p','knowledge-no-images','이 문서에는 원본 PDF·이미지가 연결되어 있지 않습니다. 현재 보관된 검색 텍스트를 표시합니다.'));
  }
  detail.querySelector('.knowledge-next')?.remove();
  for(const [index,chunk] of (data.chunks||[]).entries()){
   detail.append(chunkArticle(chunk,'문단 '+(next+index+1)+(chunk.page?' · '+chunk.page+'쪽':'')));
  }
  if(data.next_offset!==null&&data.next_offset!==undefined){const button=make('button','knowledge-next','다음 문단 보기');button.type='button';button.addEventListener('click',()=>{button.disabled=true;showDocument(id,true,data.next_offset).finally(()=>{if(button.isConnected)button.disabled=false;});});detail.append(button);}
 }catch(error){if(generation===detailGeneration)failure(detail,error);}
}
async function showCitations(ids){
 const generation=++detailGeneration;detail.replaceChildren(make('h3','','이 답변에 사용한 근거 문단'));
 const status=make('p','knowledge-basis','사용한 문단을 확인하고 있습니다.');detail.append(status);
 for(const id of ids){
  try{const row=await read('/query-chat/api/knowledge/chunks/'+encodeURIComponent(id));if(generation!==detailGeneration)return;const article=chunkArticle(row,row.chunk_id);const button=make('button','knowledge-doc-link','문서의 다른 문단 보기');button.type='button';button.addEventListener('click',()=>showDocument(row.doc_id));article.append(button);detail.append(article);}
  catch{if(generation!==detailGeneration)return;detail.append(make('p','knowledge-empty','근거 문단을 찾을 수 없습니다: '+id));}
 }
 if(generation===detailGeneration)status.textContent='모델의 답변과 실제 검색 문단을 비교할 수 있습니다.';
}
function open(button,ids=[]){
 document.dispatchEvent(new Event('copilot:close-panels'));lastFocus=button;if(!dialog.open)dialog.showModal();close.focus();loadDocuments();if(ids.length)showCitations(ids);else{detailGeneration++;detail.replaceChildren(make('p','knowledge-empty','왼쪽에서 자료를 선택하세요.'));}
}
function hide(){listGeneration++;detailGeneration++;dialog.close();lastFocus?.focus();}
close.addEventListener('click',hide);dialog.addEventListener('close',()=>lastFocus?.focus());dialog.addEventListener('click',event=>{if(event.target===dialog)hide();});
search.addEventListener('input',()=>{clearTimeout(timer);timer=setTimeout(()=>loadDocuments(),250);});
document.addEventListener('click',event=>{
 const button=event.target.closest?.('[data-knowledge-open],[data-knowledge-citations]');if(!button)return;
 let ids=[];if(button.dataset.knowledgeCitations){try{ids=JSON.parse(button.dataset.knowledgeCitations).filter(x=>typeof x==='string'&&/^DOC-[A-Za-z0-9_-]+-C\d+$/.test(x)).slice(0,8);}catch{}}
 open(button,ids);
});
})();
