/* Question suggestions fill the composer; they never submit a request. */
(() => {
  'use strict';
  const groups = [
    {id:'coating',label:'도막과 도장 원리',questions:[
      ['wet-dry','습도막과 건조도막','습도막 두께와 건조도막 두께는 어떻게 다른가요?'],
      ['thin','저도막의 원인','저도막이 생기는 원인은 무엇인가요?'],
      ['thick','과도막의 문제','과도막이 생기면 어떤 문제가 발생하나요?'],
      ['uniform','균일한 도장','균일하게 칠하려면 어떻게 해야 하나요?'],
      ['target','목표 두께와 판정','목표 도막 두께와 도막 판정은 어떤 관계인가요?'],
      ['overlap','중첩과 재도장','스프레이 중첩과 재도장은 어떻게 다른가요?'],
    ]},
    {id:'spray',label:'스프레이 작업',questions:[
      ['distance','분사 거리','스프레이 건과 표면 사이 거리가 도막에 어떤 영향을 주나요?'],
      ['speed','이동 속도','스프레이 건 이동 속도를 바꾸면 도막 두께는 어떻게 달라지나요?'],
      ['angle','분사 각도','스프레이 건을 표면에 수직으로 유지해야 하는 이유는 무엇인가요?'],
      ['pattern','분사 패턴','스프레이 분사 패턴이 불균일할 때 무엇부터 확인해야 하나요?'],
      ['edges','모서리와 곡면','모서리와 곡면을 도장할 때 주의할 점은 무엇인가요?'],
      ['start-stop','분사 시작과 종료','스프레이 분사를 시작하고 멈추는 위치는 어떻게 잡아야 하나요?'],
    ]},
    {id:'blasting',label:'블라스팅과 표면 처리',questions:[
      ['purpose','표면 처리 목적','도장 전에 블라스팅으로 표면을 처리하는 이유는 무엇인가요?'],
      ['profile','표면 조도','블라스팅 표면 조도와 도료 부착력은 어떤 관계인가요?'],
      ['abrasive','연마재 선택','블라스팅 연마재를 고를 때 어떤 조건을 확인해야 하나요?'],
      ['contamination','먼지와 염분','블라스팅 후 먼지와 염분을 확인해야 하는 이유는 무엇인가요?'],
      ['nozzle','노즐 거리와 각도','블라스팅 노즐의 거리와 각도가 작업 결과에 어떤 영향을 주나요?'],
      ['rust','재발청','블라스팅 후 재발청을 줄이려면 무엇을 관리해야 하나요?'],
    ]},
    {id:'grinding',label:'그라인딩과 보수',questions:[
      ['purpose','그라인딩의 목적','도장 준비 작업에서 그라인딩은 언제 필요한가요?'],
      ['edge','모서리 다듬기','용접부와 날카로운 모서리를 다듬어야 하는 이유는 무엇인가요?'],
      ['disc','연마 공구 점검','그라인딩 전 연마 공구와 디스크에서 무엇을 점검해야 하나요?'],
      ['pressure','과도한 압력','그라인딩할 때 압력을 지나치게 주면 어떤 문제가 생기나요?'],
      ['repair','보수도장 경계','보수도장 전에 기존 도막의 경계는 어떻게 정리해야 하나요?'],
      ['cleanup','작업 후 정리','그라인딩 후 도장 전 표면에서 무엇을 제거해야 하나요?'],
    ]},
    {id:'safety',label:'안전과 작업 환경',questions:[
      ['ppe','보호구','스프레이 도장 전에 어떤 개인 보호구를 확인해야 하나요?'],
      ['ventilation','환기','밀폐된 공간에서 도장할 때 환기가 중요한 이유는 무엇인가요?'],
      ['sds','안전보건자료','도료의 안전보건자료에서 작업 전에 확인할 항목은 무엇인가요?'],
      ['ignition','화재와 점화원','도장 작업 주변의 점화원은 어떻게 관리해야 하나요?'],
      ['dew','결로와 이슬점','이슬점과 결로가 도장 품질에 어떤 영향을 주나요?'],
      ['dust','분진과 비산물','블라스팅과 그라인딩 작업의 분진과 비산물에는 어떻게 대비해야 하나요?'],
    ]},
    {id:'posture',label:'자세와 작업 순서',questions:[
      ['types','도장 자세의 종류','도장할 때 어떤 자세 유형이 있나요?'],
      ['stable','안정적인 자세','스프레이 작업에서 안정적인 자세를 유지하려면 어떻게 해야 하나요?'],
      ['arm','팔과 몸의 움직임','스프레이 건을 이동할 때 팔과 몸은 어떻게 움직여야 하나요?'],
      ['fatigue','피로와 휴식','반복 도장 작업에서 피로로 인한 자세 변화를 줄이려면 어떻게 해야 하나요?'],
      ['order','작업 순서 계획','도장 작업 순서를 계획할 때 어떤 점을 고려해야 하나요?'],
      ['corners','좁은 곳의 자세','좁은 곳이나 손이 닿기 어려운 곳을 도장할 때 자세에서 주의할 점은 무엇인가요?'],
    ]},
    {id:'training',label:'선택한 훈련 기록',questions:[
      ['summary','기록의 주요 문제','선택한 훈련 기록에서 우선 확인해야 할 문제를 알려주세요.',true],
      ['zone','문제가 큰 Zone','선택한 훈련에서 가장 문제가 있는 Zone은 어디인가요?',true],
      ['distance','기록의 분사 거리','선택한 훈련 기록의 분사 거리를 확인하고 개선할 점을 알려주세요.',true],
      ['speed','기록의 이동 속도','선택한 훈련 기록의 이동 속도에서 확인할 점을 알려주세요.',true],
      ['posture','기록의 자세','선택한 훈련 기록의 자세를 확인하고 개선 방향을 알려주세요.',true],
      ['sequence','기록의 작업 순서','선택한 훈련 기록의 Zone 방문 순서에서 확인할 점을 알려주세요.',true],
    ]},
    {id:'terms',label:'다국어와 도장 용어',questions:[
      ['dft','건조도막두께 용어','건조도막두께는 영어로 무엇인가요?'],
      ['wft','습도막두께 용어','습도막두께는 영어로 무엇인가요?'],
      ['vietnamese','베트남어 도장 용어','건조도막두께의 베트남어 용어를 알려주세요.'],
      ['indonesian','인도네시아어 도장 용어','건조도막두께의 인도네시아어 용어를 알려주세요.'],
      ['russian','러시아어 도장 용어','건조도막두께의 러시아어 용어를 알려주세요.'],
      ['explain','쉽게 설명하기','표면 처리, 도막 두께, 재도장의 뜻을 처음 배우는 사람에게 쉽게 설명해 주세요.'],
    ]},
  ].map(group => Object.freeze({...group, questions:Object.freeze(group.questions.map(([id,title,prompt,requiresContext=false]) =>
    Object.freeze({id:group.id+'-'+id,title,prompt,requiresContext})))}));
  function render(host,onChoose,query='') {
    const needle=query.trim().toLocaleLowerCase('ko-KR');
    host.replaceChildren();
    let count=0;
    for(const group of groups){
      const questions=group.questions.filter(item=>!needle||(group.label+' '+item.title+' '+item.prompt).toLocaleLowerCase('ko-KR').includes(needle));
      if(!questions.length)continue;
      const section=document.createElement('details'),heading=document.createElement('summary'),list=document.createElement('div');
      section.className='question-category';section.open=Boolean(needle)||count===0;
      heading.textContent=group.label+' · '+questions.length;list.className='question-items';section.append(heading,list);
      for(const item of questions){
        const button=document.createElement('button'),title=document.createElement('strong'),text=document.createElement('span');
        button.type='button';button.className='question-preset';button.dataset.presetId=item.id;
        title.textContent=item.title;text.textContent=item.prompt;button.append(title,text);
        if(item.requiresContext){const note=document.createElement('small');note.textContent='훈련 기록 선택 필요';button.append(note);}
        button.addEventListener('click',()=>onChoose(item));list.append(button);
      }
      host.append(section);count+=questions.length;
    }
    if(!count){const empty=document.createElement('p');empty.className='sub';empty.textContent='검색에 맞는 질문이 없습니다.';host.append(empty);}
    return count;
  }
  window.CopilotQuestionPresets=Object.freeze({groups:Object.freeze(groups),total:groups.reduce((sum,group)=>sum+group.questions.length,0),render});
})();
