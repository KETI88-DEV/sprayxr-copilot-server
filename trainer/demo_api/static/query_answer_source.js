/* Labels describe the evidence actually used by the server, not the requested mode. */
(() => {
  'use strict';
  const documentId=/^DOC-[A-Za-z0-9_.:-]+-C\d+$/;
  function describe(data={},source=''){
    const citations=Array.isArray(data.citations)?[...new Set(data.citations.filter(value=>typeof value==='string'&&documentId.test(value)))]:[];
    const rag=data.retrieval_used===true&&citations.length>0&&source==='documents';
    let result;
    if(rag)result={kind:'rag',label:'RAG',description:'선박도장 문서에서 확인한 근거를 바탕으로 작성한 답변입니다.'};
    else if(source==='glossary')result={kind:'glossary',label:'용어집',description:'검증된 용어집 또는 검토된 TSB 용어자료의 번역입니다.'};
    else if(source==='conversation')result={kind:'llm',label:'LLM',description:'검색 문서 근거 없이 대화 맥락과 모델 지식으로 작성한 답변입니다.'};
    else if(source==='analysis')result={kind:'analysis',label:'분석 안내',description:'서버의 분석 결과 또는 고정 안내입니다.'};
    else result={kind:'unknown',label:'답변 출처 미확인',description:'이 답변의 근거 출처를 확인하지 못했습니다.'};
    if(data.degraded===true)result.label+=' · 확인 필요';
    result.citations=rag?citations:[];
    return result;
  }
  function render(metadata,data,source){
    const info=describe(data,source),label=document.createElement(info.kind==='rag'?'button':'span');
    label.className='answer-source source-'+info.kind;
    label.textContent=info.label;label.title=info.description;
    if(info.kind==='rag'){
      label.type='button';label.dataset.knowledgeCitations=JSON.stringify(info.citations);
      label.setAttribute('aria-label',info.label+' · 참고 학습자료 열기');
    }
    metadata.append(label);return label;
  }
  window.CopilotAnswerSource=Object.freeze({describe,render});
})();
