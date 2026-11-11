'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../static/copilot_feedback.js'), 'utf8');

class Element {
  constructor(id) { this.id=id; this.listeners={}; this.dataset={}; this.attrs={}; this.value=''; this.disabled=false; this.open=false; this.readOnly=false; }
  addEventListener(name, listener) { (this.listeners[name] ||= []).push(listener); }
  emit(name) { return Promise.all((this.listeners[name] || []).map(fn => fn({preventDefault(){}}))); }
  focus() { this.owner.activeElement=this; }
  showModal() { this.open=true; }
  close() { this.open=false; return this.emit('close'); }
  setAttribute(name, value) { this.attrs[name]=value; }
  removeAttribute(name) { delete this.attrs[name]; }
  set innerHTML(value) { throw new Error('Feedback must render text, not HTML'); }
}
function setup(fetcher) {
  const ids=['open','dialog','form','message','name','kind','submit','reset','state','close'];
  const nodes=Object.fromEntries(ids.map(id => [id,new Element('copilot-feedback-'+id)]));
  const document={body:{dataset:{portalPage:'motion'}},getElementById:id=>Object.values(nodes).find(node=>node.id===id)};
  Object.values(nodes).forEach(node=>{node.owner=document;});
  nodes.form.reportValidity=()=>nodes.message.value.length<=4000 && nodes.name.value.length<=80;
  nodes.form.reset=()=>{nodes.message.value='';nodes.name.value='';nodes.kind.value='improvement';};
  nodes.kind.value='improvement';
  let count=0;
  vm.runInNewContext(source,{document,fetch:fetcher,AbortController,setTimeout,clearTimeout,Uint8Array,crypto:{randomUUID:()=>`00000000-0000-4000-8000-${String(++count).padStart(12,'0')}`}});
  return {nodes,document};
}
const response=(status,body)=>({ok:status>=200&&status<300,status,json:async()=>body});

test('navigation opens the dialog, focuses the required field, and restores focus on close',async()=>{
  const {nodes,document}=setup(()=>{throw new Error('No submission expected');});
  await nodes.open.emit('click');
  assert.equal(nodes.dialog.open,true);assert.equal(document.activeElement,nodes.message);
  nodes.message.value='temporary';await nodes.reset.emit('click');
  assert.equal(nodes.message.value,'');assert.equal(document.activeElement,nodes.message);
  await nodes.close.emit('click');
  assert.equal(nodes.dialog.open,false);assert.equal(document.activeElement,nodes.open);
});

test('blank feedback is rejected with a focused field before any request',async()=>{
  let calls=0;const {nodes,document}=setup(()=>{calls++;});
  await nodes.open.emit('click');nodes.message.value=' \n ';
  await nodes.form.emit('submit');
  assert.equal(calls,0);assert.equal(nodes.state.dataset.state,'error');assert.equal(document.activeElement,nodes.message);
});

test('a pending request locks input and double submission; success clears the draft',async()=>{
  let resolve,requests=[];
  const {nodes,document}=setup((url,options)=>{requests.push({url,options});return new Promise(done=>{resolve=done;});});
  await nodes.open.emit('click');nodes.message.value='<script>literal</script>';nodes.name.value=' 담당자 ';
  const sending=nodes.form.emit('submit');
  assert.equal(nodes.submit.disabled,true);assert.equal(nodes.reset.disabled,true);assert.equal(nodes.kind.disabled,true);
  assert.equal(nodes.message.readOnly,true);assert.equal(nodes.form.attrs['aria-busy'],'true');
  await nodes.form.emit('submit');assert.equal(requests.length,1);
  const body=JSON.parse(requests[0].options.body);
  assert.equal(requests[0].url,'/api/feedback');assert.equal(body.page,'motion');assert.equal(body.message,'<script>literal</script>');
  assert.equal(body.name,'담당자');assert.deepEqual(Object.keys(body).sort(),['kind','message','name','page','submission_id']);
  resolve(response(201,{status:'received',feedback_id:body.submission_id}));await sending;
  assert.equal(nodes.state.dataset.state,'success');assert.equal(nodes.message.value,'');assert.equal(nodes.name.value,'');
  assert.equal(nodes.submit.disabled,false);assert.equal(nodes.message.readOnly,false);assert.equal(nodes.form.attrs['aria-busy'],undefined);
  assert.equal(document.activeElement,nodes.reset);
});

test('storage failure preserves input and retry keeps the idempotency key',async()=>{
  let requests=[];let failing=true;
  const {nodes}=setup(async(url,options)=>{
    const body=JSON.parse(options.body);requests.push(body);
    return failing?response(503,{}):response(200,{status:'received',feedback_id:body.submission_id});
  });
  await nodes.open.emit('click');nodes.message.value='Keep this draft';await nodes.form.emit('submit');
  assert.equal(nodes.state.dataset.state,'error');assert.equal(nodes.message.value,'Keep this draft');assert.equal(nodes.submit.disabled,false);
  failing=false;await nodes.form.emit('submit');
  assert.equal(requests.length,2);assert.equal(requests[0].submission_id,requests[1].submission_id);
  assert.equal(nodes.state.dataset.state,'success');
});

test('network errors and malformed receipts keep the draft and clear the loading state',async()=>{
  for (const fetcher of [async()=>{throw new TypeError('network down');},async()=>response(200,{})]) {
    const {nodes,document}=setup(fetcher);await nodes.open.emit('click');nodes.message.value='Draft';await nodes.form.emit('submit');
    assert.equal(nodes.state.dataset.state,'error');assert.equal(nodes.message.value,'Draft');assert.equal(nodes.submit.disabled,false);
    assert.equal(nodes.kind.disabled,false);assert.equal(document.activeElement,nodes.message);
    assert.match(nodes.state.textContent,/입력/);assert.doesNotMatch(nodes.state.textContent,/network down/);
  }
});
