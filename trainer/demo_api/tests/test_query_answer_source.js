'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const file=path.join(__dirname,'../static/query_answer_source.js');
const context={window:{}};
vm.runInNewContext(fs.existsSync(file)?fs.readFileSync(file,'utf8'):'',context);
const describe=(data,source)=>context.window.CopilotAnswerSource?.describe(data,source)||{label:'',kind:''};
test('actual cited document answer is labelled RAG',()=>{
  const result=describe({retrieval_used:true,citations:['DOC-doc_0001-C0001']},'documents');
  assert.equal(result.label,'RAG');assert.equal(result.kind,'rag');
});
test('requesting retrieval without usable citations never claims RAG',()=>{
  const result=describe({retrieval_used:true,citations:[],degraded:true},'documents');
  assert.notEqual(result.kind,'rag');assert.match(result.label,/확인 필요/);
});
test('conversation fallback is labelled LLM',()=>{
  const result=describe({retrieval_used:false,citations:[]},'conversation');
  assert.equal(result.label,'LLM');assert.equal(result.kind,'llm');
});
test('glossary and training records stay distinct from document RAG',()=>{
  assert.equal(describe({retrieval_used:false,citations:['TSB:VR_Language.csv:1']},'glossary').label,'용어집');
  assert.equal(describe({retrieval_used:false,citations:[]},'analysis').label,'분석 안내');
  assert.doesNotMatch(describe({retrieval_used:false,citations:[]},'analysis').description,/훈련 기록/);
});
test('legacy genuine citations work and arbitrary server strings are not rendered as labels',()=>{
  assert.equal(describe({retrieval_used:true,citations:['DOC-doc_0002-C0010']},'').label,'RAG');
  assert.equal(describe({retrieval_used:false,citations:['<img src=x onerror=alert(1)>']},'<script>').label,'답변 출처 미확인');
});
test('degraded supported document answer shows its limitation',()=>{
  assert.equal(describe({retrieval_used:true,citations:['DOC-doc_0001-C0001'],degraded:true},'documents').label,'RAG · 확인 필요');
});
