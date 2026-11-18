'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');

function page(width=390){
  let document;
  class Element{
    constructor(tag='div'){
      this.tagName=tag;this.children=[];this.events={};this.attributes={};this.dataset={};this.value='';
      this.className='';this.hidden=false;this.disabled=false;this.inert=false;this.style={setProperty(){}};
      this.scrollHeight=30;this.selectedOptions=[{textContent:'빠른 안내 · Qwen3-8B'}];
      const classes=new Set();this.classList={add:cls=>classes.add(cls),remove:cls=>classes.delete(cls),
        contains:cls=>classes.has(cls),toggle:(cls,on)=>on?classes.add(cls):classes.delete(cls)};
    }
    append(...nodes){for(const node of nodes){node.parent=this;this.children.push(node);}}
    add(node){this.append(node);}
    replaceChildren(...nodes){this.children=[];this.append(...nodes);}
    addEventListener(name,callback){(this.events[name]??=[]).push(callback);}
    emit(name,event={}){for(const callback of this.events[name]||[])callback({target:this,preventDefault(){this.defaultPrevented=true;},...event});}
    setAttribute(key,value){this.attributes[key]=value;}
    getAttribute(key){return this.attributes[key];}
    removeAttribute(key){delete this.attributes[key];}
    focus(){document.activeElement=this;}
    getClientRects(){return this.hidden?[]:[{}];}
    contains(node){return node===this||this.children.some(child=>child.contains(node));}
    querySelectorAll(){return this.descendants().filter(node=>['button','input','select','a','summary'].includes(node.tagName));}
    descendants(){return this.children.flatMap(child=>[child,...child.descendants()]);}
    remove(){if(this.parent)this.parent.children=this.parent.children.filter(node=>node!==this);}
  }
  const nodes=new Map(),events={},requests=[],media=[];
  const tags={'open-settings':'button','close-settings':'button','open-questions':'button','close-questions':'button',
    'preset-search':'input','training-worker':'select','training-process':'select','training-mode':'select',
    'training-session':'select','training-zone':'select','response-mode':'select','response-language':'select',
    'reviewer-name':'input','prompt':'textarea','send':'button','new-chat':'button','apply-training':'button',
    'context-chip':'button','clear-training':'button'};
  const get=id=>{if(!nodes.has(id))nodes.set(id,new Element(tags[id]));return nodes.get(id);};
  document={activeElement:null,getElementById:get,createElement:tag=>new Element(tag),querySelectorAll:()=>[],
    addEventListener:(name,callback)=>(events[name]??=[]).push(callback),
    dispatchEvent:event=>{for(const callback of events[event.type]||[])callback(event);},documentElement:new Element()};
  const anchor=new Element('a');get('chat-sidebar').append(get('close-settings'),get('response-mode'),get('training-worker'),anchor);
  get('question-sidebar').append(get('close-questions'),get('preset-search'),get('question-list'));
  get('conversation').append(get('open-settings'),get('open-questions'),get('new-chat'),get('prompt'));
  const window={innerHeight:800,visualViewport:{height:520,offsetTop:5,addEventListener(){}},addEventListener(){},
    matchMedia(query){const entry={matches:query.includes('max-width')?width<=760:width>=1180,query,listeners:[],addEventListener(_name,callback){this.listeners.push(callback);}};media.push(entry);return entry;}};
  const fetch=async(url,options)=>{requests.push({url,options});return {ok:true,json:async()=>({workers:[],processes:[],modes:[],sessions:[]})};};
  const context={window,document,fetch,Option:class extends Element{constructor(label,value){super('option');this.textContent=label;this.value=value;}},
    AbortController,setTimeout,clearTimeout,URLSearchParams,performance};
  vm.createContext(context);
  for(const file of ['query_chat_presets.js','query_chat.js'])vm.runInContext(fs.readFileSync(path.join(__dirname,'../static',file),'utf8'),context);
  return {get,document,window,requests,anchor,
    key(key,shiftKey=false){const event={key,shiftKey,preventDefault(){this.defaultPrevented=true;}};for(const callback of events.keydown||[])callback(event);return event;},
    focusOutside(node){node.focus();for(const callback of events.focusin||[])callback({target:node});},
    resize(next){for(const entry of media){const matches=entry.query.includes('max-width')?next<=760:next>=1180;if(entry.matches!==matches){entry.matches=matches;entry.listeners.forEach(callback=>callback());}}},
    presets:window.CopilotQuestionPresets};
}

test('48 categorized presets distinguish training questions and selecting a question only fills the composer',()=>{
  const app=page(390),groups=app.presets.groups;
  assert.equal(groups.length,8);assert.ok(app.presets.total>=40);
  const questions=groups.flatMap(group=>group.questions);
  assert.equal(new Set(questions.map(item=>item.id)).size,questions.length);
  assert.equal(questions.filter(item=>item.requiresContext).length,6);
  assert.ok(groups.some(group=>group.id==='blasting')&&groups.some(group=>group.id==='grinding'));
  app.get('open-questions').emit('click');
  const button=app.get('question-list').descendants().find(node=>node.dataset.presetId==='training-summary');
  const before=app.requests.length;button.emit('click');
  assert.equal(app.get('prompt').value,questions.find(item=>item.id==='training-summary').prompt);
  assert.equal(app.requests.length,before,'choosing a preset must not fetch or submit');
  assert.match(app.get('chat-status').textContent,/훈련 기록을 선택해야/);
  assert.equal(app.document.activeElement,app.get('prompt'));
  assert.equal(app.get('chat-backdrop').hidden,true);
});

test('preset search filters categories and matches prompt text without changing the input',()=>{
  const app=page(1280);
  app.get('preset-search').value='이슬점';app.get('preset-search').emit('input');
  assert.equal(app.get('question-list').descendants().filter(node=>node.dataset.presetId).length,1);
  assert.match(app.get('preset-count').textContent,/1개 질문/);
  assert.equal(app.get('prompt').value,'');
  app.get('preset-search').value='없는 질문 abc';app.get('preset-search').emit('input');
  assert.equal(app.get('question-list').descendants().filter(node=>node.dataset.presetId).length,0);
  assert.match(app.get('preset-count').textContent,/0개 질문/);
});

test('mobile drawers are exclusive, modal, trap focus, close on Escape/backdrop and restore the opener',()=>{
  const app=page(390);
  assert.equal(app.get('chat-sidebar').inert,true);assert.equal(app.get('question-sidebar').inert,true);
  app.get('open-settings').emit('click');
  assert.equal(app.get('chat-sidebar').getAttribute('aria-modal'),'true');
  assert.equal(app.get('conversation').inert,true);
  assert.equal(app.get('open-settings').getAttribute('aria-expanded'),'true');
  assert.equal(app.document.activeElement,app.get('close-settings'));
  assert.equal(app.key('Tab',true).defaultPrevented,true);assert.equal(app.document.activeElement,app.anchor);
  assert.equal(app.key('Tab').defaultPrevented,true);assert.equal(app.document.activeElement,app.get('close-settings'));
  app.focusOutside(app.get('prompt'));assert.equal(app.document.activeElement,app.get('close-settings'));
  app.get('open-questions').emit('click');
  assert.equal(app.get('chat-sidebar').inert,true);assert.equal(app.get('chat-sidebar').classList.contains('is-open'),false);
  assert.equal(app.get('question-sidebar').inert,false);
  assert.equal(app.get('open-settings').getAttribute('aria-expanded'),'false');
  app.key('Escape');
  assert.equal(app.document.activeElement,app.get('open-questions'));
  assert.equal(app.get('conversation').inert,false);assert.equal(app.get('chat-backdrop').hidden,true);
  app.get('open-settings').emit('click');app.get('chat-backdrop').emit('click');
  assert.equal(app.document.activeElement,app.get('open-settings'));
});

test('tablet questions use a drawer while desktop keeps both sidebars available and resets open modal state on resize',()=>{
  const app=page(1024);
  assert.equal(app.get('chat-sidebar').inert,false);assert.equal(app.get('question-sidebar').inert,true);
  app.get('open-questions').emit('click');assert.equal(app.get('chat-sidebar').inert,true);
  app.resize(1280);
  assert.equal(app.get('chat-sidebar').inert,false);assert.equal(app.get('question-sidebar').inert,false);
  assert.equal(app.get('conversation').inert,false);assert.equal(app.get('chat-backdrop').hidden,true);
  assert.equal(app.get('question-sidebar').getAttribute('aria-modal'),undefined);
  app.resize(390);assert.equal(app.get('chat-sidebar').inert,true);assert.equal(app.get('question-sidebar').inert,true);
});

test('library close-panels event removes drawer modality without restoring focus or trapping the dialog',()=>{
  for(const [width,opener] of [[390,'open-settings'],[390,'open-questions'],[1024,'open-questions']]){
    const app=page(width);
    app.get(opener).emit('click');
    const before=app.document.activeElement;
    assert.equal(app.get('conversation').inert,true);
    app.document.dispatchEvent({type:'copilot:close-panels'});
    assert.equal(app.document.activeElement,before,'handoff must let the library dialog choose focus');
    assert.equal(app.get('conversation').inert,false);
    assert.equal(app.get('chat-backdrop').hidden,true);
    for(const id of ['chat-sidebar','question-sidebar']){
      assert.equal(app.get(id).classList.contains('is-open'),false);
      assert.equal(app.get(id).getAttribute('aria-modal'),undefined);
    }
    assert.equal(app.get('open-settings').getAttribute('aria-expanded'),'false');
    assert.equal(app.get('open-questions').getAttribute('aria-expanded'),'false');
    assert.equal(app.get('chat-sidebar').inert,width<=760);
    assert.equal(app.get('question-sidebar').inert,true);
    const libraryControl=app.get('library-dialog-control');
    app.focusOutside(libraryControl);
    assert.equal(app.document.activeElement,libraryControl,'drawer focusin handler must not reclaim library focus');
    assert.notEqual(app.key('Tab').defaultPrevented,true,'native dialog owns keyboard focus after handoff');
    app.document.dispatchEvent({type:'copilot:close-panels'});
    assert.equal(app.document.activeElement,libraryControl,'repeated close event is focus-neutral');
  }
});

test('keyboard viewport, phone input font and existing zoom policy remain present',()=>{
  const css=fs.readFileSync(path.join(__dirname,'../static/query_chat.css'),'utf8');
  const html=fs.readFileSync(path.join(__dirname,'../static/query_chat.html'),'utf8');
  const js=fs.readFileSync(path.join(__dirname,'../static/query_chat.js'),'utf8');
  assert.match(css,/\.composer textarea,\.feedback-box textarea,\.settings input,\.settings select\{font-size:16px!important/);
  assert.match(css,/\.preset-search\{font-size:16px;min-height:44px/);
  assert.match(html,/interactive-widget=resizes-content/);assert.match(html,/user-scalable=no/);
  assert.match(js,/visualViewport/);assert.match(css,/min-width:1180px/);
});
