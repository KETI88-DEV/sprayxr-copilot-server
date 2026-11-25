(() => {
'use strict';
const $=id=>document.getElementById(id),make=(tag,cls,text)=>{const e=document.createElement(tag);if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e;};
let posts=[],installed='',lastFocus=null;
const count=value=>Number.isFinite(value)?value.toLocaleString('ko-KR'):'—';
async function read(path){const r=await fetch(path,{cache:'no-store'});const data=await r.json();if(!r.ok)throw new Error('조회 실패');return data;}
function teaser(body){return (body||'').split('\n').map(s=>s.trim()).find(s=>s&&!s.startsWith('#'))?.replace(/^[-*]\s*/,'').replace(/\*\*|\x60/g,'')||'상세 변경 내용을 확인하세요.';}
function render(){
 const needle=$('release-search').value.trim().toLowerCase(),rows=posts.filter(p=>[p.version,p.title,p.body].join(' ').toLowerCase().includes(needle));
 $('update-count').textContent=count(rows.length)+'개';$('release-list').replaceChildren();
 if(!rows.length){$('release-list').append(make('p','empty-state','검색 조건에 맞는 업데이트가 없습니다.'));return;}
 for(const post of rows){
  const button=make('button','release-row');button.type='button';const top=make('div','release-row-top');
  top.append(make('strong','release-version',post.version));
  if(post.version.replace(/^v/,'')===installed)top.append(make('span','installed-label','현재 버전'));
  top.append(make('time','release-row-date',post.date||'날짜 미기록'));
  button.append(top,make('span','release-row-title',post.title===post.version?'기능 및 사용성 개선':post.title),make('span','release-row-summary',teaser(post.body)),make('span','release-row-more','상세 보기 →'));
  button.addEventListener('click',()=>open(post,button));$('release-list').append(button);
 }
}
function notes(body){
 const target=$('release-body');target.replaceChildren();let code=false,pre=null;
 for(const line of (body||'').split('\n')){
  if(line.trim().startsWith('\x60\x60\x60')){code=!code;if(code){pre=make('pre');target.append(pre);}continue;}
  if(code){pre.textContent+=line+'\n';continue;}
  const value=line.trim();if(!value)continue;
  if(/^#{1,6}\s/.test(value))target.append(make('h3','',value.replace(/^#{1,6}\s+/,'')));
  else if(/^[-*]\s/.test(value))target.append(make('p','release-bullet','• '+value.replace(/^[-*]\s+/,'')));
  else target.append(make('p','',value));
 }
}
const verdictNames={pass:'통과',partial:'부분 통과',fail:'실패'};
function answerCard(title,result,extraClass=''){
 const card=make('div','qa-answer '+extraClass);
 card.append(make('h4','',title),make('p','qa-answer-text',result.answer));
 if(result.note)card.append(make('p','qa-note',result.note));
 return card;
}
function queryAnswers(report,target){
 const questions=report.questions;
 target.replaceChildren(make('h3','','시험 질문과 답변'));
 target.append(make('p','qa-meta',`질문 ${questions.length}개 · 최초 시험 ${report.baseline_version} · 재시험 ${report.retested_version}`));
 const before=report.baseline_counts,after=report.retest_counts;
 target.append(make('p','qa-meta',`최초 통과 ${before.pass}개, 부분 통과 ${before.partial}개, 실패 ${before.fail}개`));
 target.append(make('p','qa-meta',`재시험 통과 ${after.pass}개, 부분 통과 ${after.partial}개, 실패 ${after.fail}개`));
 const filter=make('label','qa-filter'),checkbox=make('input');checkbox.type='checkbox';
 filter.append(checkbox,document.createTextNode('최초 시험에서 실패한 질문만 보기'));
 const list=make('div','qa-list');target.append(filter,list);
 function renderQuestions(){
  list.replaceChildren();let group='';
  for(const item of questions){
   if(checkbox.checked&&item.baseline.verdict!=='fail')continue;
   if(group!==item.group){group=item.group;list.append(make('h4','qa-group',group));}
   const question=make('details','qa-question');question.dataset.questionId=item.id;
   const heading=make('summary','qa-question-summary');
   heading.append(make('span','qa-id',item.id),make('span','qa-prompt',item.question),
                  make('span','qa-verdict '+item.baseline.verdict,verdictNames[item.baseline.verdict]));
   const answers=make('div','qa-answer-grid');
   answers.append(answerCard('최초 실제 답변',item.baseline));
   if(item.baseline.verdict!=='pass'){
    if(item.retest?.verdict==='pass')answers.append(answerCard('교정 후 실제 답변 · 통과',item.retest));
    else{
     const status=item.retest?'재시험 '+verdictNames[item.retest.verdict]:'재시험 미실시';
     answers.append(answerCard('기대 답변 · '+status,{answer:item.expected_answer}));
     if(item.retest)answers.append(answerCard('재시험 실제 답변 · '+verdictNames[item.retest.verdict],item.retest,'qa-observed'));
    }
   }else if(item.retest){
    answers.append(answerCard('재시험 실제 답변 · '+verdictNames[item.retest.verdict],item.retest));
   }
   question.append(heading,answers);list.append(question);
  }
 }
 checkbox.addEventListener('change',renderQuestions);renderQuestions();
}
async function open(post,button){
 lastFocus=button;$('release-version').textContent=post.version;$('release-title').textContent=post.title;
 $('release-date').textContent=(post.date||'날짜 미기록')+' · '+(post.date_source||post.source||'변경 이력');
 const ui=$('release-ui-changes');ui.replaceChildren();if(post.ui_changes?.length){ui.append(make('h3','','화면이 이렇게 달라졌습니다'));for(const text of post.ui_changes)ui.append(make('p','',text));}
 notes(post.body);
 const qa=$('release-query-qa');qa.replaceChildren();qa.hidden=!post.has_query_qa;
 const source=$('release-source');source.hidden=!post.url;if(post.url)source.href=post.url;
 $('release-dialog').showModal();$('release-close').focus();
 if(!post.has_query_qa)return;
 const loading=make('p','qa-meta','시험 결과를 불러오는 중입니다.');loading.setAttribute('role','status');qa.append(loading);
 try{
  const detail=await read('/home/api/updates/'+encodeURIComponent(post.version));
  if($('release-dialog').open&&$('release-version').textContent===post.version)queryAnswers(detail.query_qa,qa);
 }catch{
  if($('release-dialog').open&&$('release-version').textContent===post.version)loading.textContent='시험 결과를 불러오지 못했습니다.';
 }
}
function close(){$('release-dialog').close();lastFocus?.focus();}
$('release-close').addEventListener('click',close);$('release-done').addEventListener('click',close);
$('release-dialog').addEventListener('click',e=>{if(e.target===$('release-dialog'))close();});
$('release-dialog').addEventListener('close',()=>lastFocus?.focus());
$('release-search').addEventListener('input',render);
read('/home/api/updates').then(data=>{posts=data.updates||[];render();}).catch(()=>{$('release-list').replaceChildren(make('p','empty-state','업데이트 기록을 불러오지 못했습니다. 새로고침 후 다시 확인하세요.'));});
read('/home/api/summary').then(data=>{installed=data.version;$('build-version').textContent='v'+installed;const t=data.training;$('stat-total').textContent=count(t.total);$('stat-complete').textContent=count(t.complete);$('stat-review').textContent=count(t.needs_review);$('stat-workers').textContent=count(t.known_workers);render();}).catch(()=>{});
async function status(){
 try{
  const response=await fetch('/api/ready',{cache:'no-store'}),data=await response.json(),checks=data.checks||{};
  const ready=data.ready===true;$('server-state').textContent=ready?'정상 운영 중':'서비스 준비 중';$('server-dot').classList.toggle('ready',ready);
  $('stat-rag').textContent=count(checks.pipeline?.rag_vectors);
  for(const [id,key] of [['status-detailed','llm'],['status-quick','quick_llm'],['status-stt','speech_to_text'],['status-mcp','mcp']])$(id).textContent=checks[key]?.ok?'준비됨':'확인 필요';
  $('status-updated').textContent='마지막 확인 '+new Date().toLocaleTimeString('ko-KR',{hour:'2-digit',minute:'2-digit'});
 }catch{$('server-state').textContent='서버 상태 확인 필요';$('server-dot').classList.remove('ready');$('status-updated').textContent='연결을 확인하고 새로고침해 주세요.';}
}
status();setInterval(()=>{if(!document.hidden)status();},60000);
})();
