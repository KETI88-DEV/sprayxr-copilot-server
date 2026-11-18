'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
class Element{
  constructor(tag='div',text='',value=''){this.tagName=tag;this.textContent=text;this.value=value;this.children=[];this.events={};this.attributes={};this.className='';this.style={setProperty(){}};this.classList={add(){},remove(){},toggle(){},contains(){return false;}};this.selectedOptions=[{textContent:'빠른 안내 · Qwen3-8B'}];this.scrollHeight=30;}
  append(...items){this.children.push(...items);for(const item of items)if(item&&typeof item==='object')item.parent=this;}
  replaceChildren(...items){this.children=[];this.append(...items);}
  add(item){this.append(item);}
  addEventListener(name,callback){this.events[name]=callback;}
  setAttribute(key,value){this.attributes[key]=value;}
  getAttribute(key){return this.attributes[key];}
  querySelector(selector){return this.find(item=>selector.startsWith('.')&&item.className.split(' ').includes(selector.slice(1)));}
  find(predicate){for(const child of this.children){if(predicate(child))return child;const nested=child.find?.(predicate);if(nested)return nested;}return null;}
  getClientRects(){return [{}];}
  focus(){}
  remove(){if(this.parent)this.parent.children=this.parent.children.filter(child=>child!==this);}
}
async function page(source,data,mode='quick'){
  const elements=new Map(),requests=[];
  const get=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
  const document={getElementById:get,createElement:tag=>new Element(tag),querySelectorAll:()=>[],addEventListener(){},documentElement:new Element()};
  const window={matchMedia:()=>({matches:false,addEventListener(){}}),addEventListener(){},innerHeight:800};
  const fetch=async(url,options={})=>{
    requests.push({url,options});
    if(url.startsWith('/query-chat/api/options'))return {ok:true,json:async()=>({workers:[],processes:[],modes:[],sessions:[]})};
    if(url==='/api/v1/copilot/query')return {ok:true,json:async()=>({status:'SUCCESS',data:{answer:'답변',...data}}),headers:{get:name=>name==='X-Copilot-Answer-Source'?source:name==='X-Copilot-Query-Turn'?'12':null}};
    throw new Error('Unexpected request '+url);
  };
  const context={window,document,Option:class extends Element{constructor(text,value){super('option',text,value);}},fetch,AbortController,setTimeout,clearTimeout,URLSearchParams,performance};
  vm.createContext(context);
  for(const name of ['query_answer_source.js','query_chat.js'])vm.runInContext(fs.readFileSync(path.join(__dirname,'../static',name),'utf8'),context);
  await new Promise(resolve=>setImmediate(resolve));
  get('prompt').value='저도막의 원인은?';get('response-mode').value=mode;get('response-language').value='ko';
  await get('chat-form').events.submit({preventDefault(){}});
  return {elements,requests,assistant:get('messages').children.find(item=>item.className==='message assistant')};
}
test('a completed Query answer renders its actual RAG source and keeps feedback controls',async()=>{
  const {assistant,requests}=await page('documents',{retrieval_used:true,citations:['DOC-doc_0001-C0001']});
  assert.equal(assistant.querySelector('.evidence-badge')?.textContent,'RAG');
  assert.ok(assistant.querySelector('.answer-tools'),'feedback controls remain attached to the stored turn');
  assert.ok(requests.some(request=>request.url==='/api/v1/copilot/query'));
});
test('a no-document response labels the same Query conversation LLM',async()=>{
  const {assistant}=await page('conversation',{retrieval_used:false,citations:[]});
  assert.equal(assistant.querySelector('.evidence-badge')?.textContent,'LLM');
});
test('a term translation is labelled glossary, not document RAG',async()=>{
  const {assistant}=await page('glossary',{retrieval_used:false,citations:['TSB:VR_Language.csv:27']});
  assert.equal(assistant.querySelector('.evidence-badge')?.textContent,'용어집');
  assert.doesNotMatch(assistant.querySelector('.answer-meta')?.textContent||'',/Qwen/,'dictionary lookup must not claim a model was invoked');
});

for(const mode of ['quick','detailed'])test(`the question page requests and renders the full ${mode} answer`,async()=>{
  const answer='표면을 먼저 준비하세요.\n근거와 작업 조건을 함께 확인하세요.';
  const {assistant,requests}=await page('documents',{answer,retrieval_used:true,citations:['DOC-doc_0001-C0001']},mode);
  const sent=JSON.parse(requests.find(request=>request.url==='/api/v1/copilot/query').options.body);
  assert.equal(sent.response_mode,mode);
  assert.equal(sent.summarize,false,'browser questions must opt out of the VR summary default');
  assert.equal(assistant.querySelector('.message-text').textContent,answer);
  assert.ok(assistant.querySelector('.answer-tools'),'full answers remain rateable');
});
