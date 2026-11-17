"""Explicit translation uses the same Query envelope, receipt and scoped history."""
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.copilot_router import create_copilot_router
from trainer.demo_api.sequence_store import SequenceStore

URL='/api/v1/copilot/query'


def setup(pipeline=None):
    store=SequenceStore(':memory:')
    app=FastAPI();app.include_router(create_copilot_router((lambda:pipeline) if pipeline else None,store=store))
    return TestClient(app),store


def body(prompt,**values):
    return {'user_id':'COPILOT-AI-TEST','task_type':'knowledge_search','prompt':prompt,'language':'ko','response_language':'ko',**values}


@pytest.mark.parametrize('mode',['quick','detailed'])
@pytest.mark.parametrize(('prompt','answer','language'),[
    ('노즐을 영어로 번역해줘','Nozzle','en'),
    ('노즐을 중국어로 번역해줘','喷嘴','zh'),
    ('Nozzle를 한국어로 번역해줘','노즐','ko'),
    ('건조도막두께를 영어로 번역해줘','dry film thickness','en'),
])
def test_exact_dictionary_translation_needs_no_generator_and_preserves_receipt(mode,prompt,answer,language):
    client,store=setup()
    response=client.post(URL,json=body(prompt,response_mode=mode))
    assert response.status_code==200
    data=response.json()['data']
    assert data['answer']==answer and data['response_language']==language
    assert data['retrieval_used'] is False and data['citations']==[]
    assert response.headers['X-Copilot-Answer-Source']=='glossary'
    assert int(response.headers['X-Copilot-Query-Turn'])>0
    row=store._conn.execute('SELECT answer_source,query_context FROM query_turns').fetchone()
    assert row[0]=='glossary'
    context=json.loads(row[1])
    assert context['glossary_context']['target_language']==language
    assert context['glossary_context']['provenance']


def test_unknown_term_uses_translation_of_the_term_not_a_definition():
    class Translator:
        def __init__(self):self.calls=[]
        def translate_copilot_text(self,text,**languages):
            self.calls.append((text,languages));return 'Màng sơn mỏng'
    translator=Translator();client,_=setup(translator)
    response=client.post(URL,json=body('저도막을 베트남어로하면?'))
    assert response.status_code==200
    assert response.json()['data']['answer']=='Màng sơn mỏng'
    assert response.json()['data']['response_language']=='vi'
    # The complete term is not a dictionary entry, but its attested 도막 term
    # is supplied to the existing translator and appears in this translation.
    assert response.headers['X-Copilot-Answer-Source']=='glossary'
    assert translator.calls==[('저도막을',{'source_language':'auto','target_language':'vi'})]


def test_sentence_translation_keeps_full_text_numbers_and_glossary_hint():
    class Translator:
        def translate_copilot_text(self,text,**languages):
            assert text=='노즐을 300 mm 떨어뜨리세요.'
            return 'Giữ Vòi phun cách bề mặt 300 mm.'
    client,_=setup(Translator())
    response=client.post(URL,json=body('"노즐을 300 mm 떨어뜨리세요."를 베트남어로 번역해줘'))
    assert response.status_code==200
    assert '300 mm' in response.json()['data']['answer']
    assert response.headers['X-Copilot-Answer-Source']=='glossary'


def test_translation_followup_reuses_only_same_user_and_conversation_source_text():
    client,_=setup()
    headers={'X-Copilot-Conversation':'glossary-one'}
    first=client.post(URL,json=body('노즐을 영어로 번역해줘'),headers=headers)
    assert first.status_code==200
    second=client.post(URL,json=body('그럼 중국어로?'),headers=headers)
    assert second.status_code==200 and second.json()['data']['answer']=='喷嘴'
    isolated=client.post(URL,json=body('그럼 중국어로?',user_id='another-user'),headers=headers)
    assert isolated.status_code==503
    isolated=client.post(URL,json=body('그럼 중국어로?'),headers={'X-Copilot-Conversation':'glossary-two'})
    assert isolated.status_code==503


def test_unknown_chinese_sentence_to_korean_is_not_returned_unchanged():
    class Translator:
        def translate_copilot_text(self,text,**languages):
            assert languages=={'source_language':'auto','target_language':'ko'}
            return '표면을 확인하세요.'
    client,_=setup(Translator())
    response=client.post(URL,json=body('"检查表面。"를 한국어로 번역해줘'))
    assert response.status_code==200
    assert response.json()['data']['answer']=='표면을 확인하세요.'


@pytest.mark.parametrize(('prompt','target','result'),[
    ('Translate "Hãy sơn tường mới." into English.','en','Paint the new wall.'),
    ('Translate "Check 김씨의 nozzle." into Korean.','ko','김씨의 노즐을 확인하세요.'),
])
def test_question_language_and_a_proper_name_do_not_skip_source_translation(prompt,target,result):
    class Translator:
        calls=0
        def translate_copilot_text(self,text,**languages):
            self.calls+=1
            assert languages=={'source_language':'auto','target_language':target}
            return result
    translator=Translator();client,_=setup(translator)
    response=client.post(URL,json=body(prompt,language='en',response_language='en'))
    assert response.status_code==200 and response.json()['data']['answer']==result
    assert response.json()['data']['response_language']==target and translator.calls==1


def test_auto_source_keeps_glossary_hints_and_does_not_add_a_public_language_value():
    from trainer.demo_api.query_language import translation_messages
    messages=translation_messages('Check the dry film thickness.',source_language='auto',target_language='ko')
    request=json.loads(messages[-1]['content'])
    assert 'detect' in request['source_language'].lower()
    assert any(pair['target']=='건도막 두께' for pair in request['glossary'])
    client,_=setup()
    response=client.post(URL,json=body('질문',language='auto'))
    assert response.status_code==422


def test_language_instruction_does_not_turn_a_rag_procedure_question_into_a_heading_translation():
    class Pipeline:
        def plan_copilot_dialogue(self,*args,**kwargs):
            return {'kind':'coating_knowledge','intent':'knowledge','scope':'unspecified'}
        def run(self,**kwargs):
            assert '압력 조절 방법' in kwargs['instruction']
            return {'response':'[답변] 건의 압력 조절 장치를 확인하세요.\n[근거] DOC-method-C0001',
                    'rag_chunks':[{'chunk_id':'DOC-method-C0001','text':'건의 압력 조절 장치를 확인합니다.'}]}
        def translate_copilot_text(self,text,**languages):
            assert '압력 조절 장치' in text
            assert languages=={'source_language':'ko','target_language':'en'}
            return 'Check the pressure control on the gun.'
    client,_=setup(Pipeline())
    response=client.post(URL,json=body('영어로 분사 압력 조절 방법을 알려줘'))
    assert response.status_code==200
    assert response.json()['data']['answer']=='Check the pressure control on the gun.'
    assert response.json()['data']['retrieval_used'] is True
    assert response.headers['X-Copilot-Answer-Source']=='documents'


def test_bad_sentence_translation_cannot_change_values_or_claim_dictionary_success():
    class Translator:
        def translate_copilot_text(self,text,**languages):return 'Giữ Vòi phun cách bề mặt 500 mm.'
    client,_=setup(Translator())
    response=client.post(URL,json=body('"노즐을 300 mm 떨어뜨리세요."를 베트남어로 번역해줘'))
    assert response.status_code==503
    assert 'X-Copilot-Query-Turn' not in response.headers
