/* Labels describe the evidence actually used by the server, not the requested mode. */
(() => {
  'use strict';
  const documentId=/^DOC-[A-Za-z0-9_.:-]+$/;
  function describe(data={},source=''){
    const citations=Array.isArray(data.citations)?data.citations.filter(value=>typeof value==='string'&&documentId.test(value)):[];
    const rag=data.retrieval_used===true&&citations.length>0&&(source==='documents'||!source);
    let result;
    if(rag)result={kind:'rag',label:'RAG',description:'선박도장 문서에서 확인한 근거를 바탕으로 작성한 답변입니다.'};
    else if(source==='glossary')result={kind:'glossary',label:'용어집',description:'검증된 용어집 또는 검토된 TSB 용어자료의 번역입니다.'};
    else if(source==='conversation')result={kind:'llm',label:'LLM',description:'검색 문서 근거 없이 대화 맥락과 모델 지식으로 작성한 답변입니다.'};
    else if(source==='analysis')result={kind:'analysis',label:'분석 안내',description:'서버의 분석 결과 또는 고정 안내입니다.'};
    else result={kind:'unknown',label:'답변 출처 미확인',description:'이 답변의 근거 출처를 확인하지 못했습니다.'};
    if(data.degraded===true)result.label+=' · 확인 필요';
    return result;
  }
  function render(card,data,source){
    const info=describe(data,source),row=document.createElement('div'),badge=document.createElement('span');
    row.className='answer-evidence';badge.className='evidence-badge evidence-'+info.kind;
    badge.textContent=info.label;badge.title=info.description;row.append(badge);
    card.append(row);return row;
  }
  window.CopilotAnswerSource=Object.freeze({describe,render});
})();
