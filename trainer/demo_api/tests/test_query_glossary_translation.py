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
    ('도장 현장 용어표로 건조도막두께, 영어 한 단어/구만 알려줘.','dry film thickness','en'),
    ("스프레이건의 '노즐'을 중국어로, 용어만.",'喷嘴','zh'),
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


def test_unregistered_standard_term_explains_which_attestation_is_missing():
    client, _ = setup()
    response = client.post(URL, json=body("한국 현장 별명인 '갯물 찍힘'을 우즈벡 도장 규격 용어로 정확히 번역해줘."))
    assert response.status_code == 200
    data = response.json()["data"]
    assert all(term in data["answer"] for term in ("갯물 찍힘", "우즈베크어", "규격 문서"))
    assert data["response_language"] == "ko" and data["citations"] == []
    assert data["retrieval_used"] is False
    assert response.headers["X-Copilot-Answer-Source"] != "glossary"


def test_menu_label_translation_does_not_retrieve_documents_or_query_zone_order():
    class Translator:
        def translate_copilot_text(self, text, **languages):
            assert text == "다음 순서"
            assert languages["target_language"] == "en"
            return "Next step"

    client, _ = setup(Translator())
    response = client.post(URL, json=body("‘다음 순서’라고 써 있는 메뉴가 무슨 뜻인지 영어로 짧게 바꿔 쓰고 싶어."))
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["answer"] == "Next step" and data["response_language"] == "en"
    assert data["retrieval_used"] is False and data["citations"] == []
    assert response.headers["X-Copilot-Answer-Source"] == "conversation"


@pytest.mark.parametrize(("prompt", "language", "expected"), [
    ("During the test, what does 'not yet sprayed' mean on the coating chart?", "en", "도료가 도포되지 않은"),
    ("'Zone 7을 완료했습니다'는 내가 적어본 UI 문구인데, 철자 자연스럽게 바꿔 줘.", "ko", "‘7번 구역 작업 완료’"),
])
def test_literal_language_requests_do_not_make_session_claims(prompt, language, expected):
    client, _ = setup()
    response = client.post(URL, json=body(prompt, language=language))
    assert response.status_code == 200
    data = response.json()["data"]
    assert expected in data["answer"]
    assert data["retrieval_used"] is False and data["citations"] == []


@pytest.mark.parametrize(("prompt", "language", "source_text", "expected"), [
    ("저도막은 어떤 상태인가요? 대답만 영어로 부탁해요.", "ko", "요구 두께에 못 미친",
     "The dry film is too thin when its measured thickness is below the specified requirement."),
    ("Nếu tôi chưa phun, hệ thống có biết độ dày lớp sơn của tôi không?", "vi", "실제 도막 두께를 측정할 수 없습니다",
     "If you have not sprayed yet, the actual coating thickness cannot be measured. The target thickness is a separate setting."),
])
def test_language_override_or_foreign_input_preserves_the_intended_answer(prompt, language, source_text, expected):
    class Translator:
        def translate_copilot_text(self, text, **languages):
            assert source_text in text
            assert languages == {"source_language": "ko", "target_language": "en"}
            return expected

    client, _ = setup(Translator())
    response = client.post(URL, json=body(prompt, language=language, response_language="en" if language == "vi" else "ko"))
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["answer"] == expected and data["response_language"] == "en"
    assert data["retrieval_used"] is False and data["citations"] == []


def test_unrelated_definition_stays_on_topic_even_with_spray_as_background():
    client, _ = setup()
    prompt = "스프레이 작업 전에 도시락을 냉장고에 둘까 합니다. 상할까 걱정되는데 냉장 보관이 무슨 뜻인지 한 문장만요."
    response = client.post(URL, json=body(prompt))
    assert response.status_code == 200
    data = response.json()["data"]
    assert "음식" in data["answer"] and "온도" in data["answer"]
    assert data["answer"].count(".") == 1
    assert data["retrieval_used"] is False and data["citations"] == []


def test_vietnamese_standoff_instruction_has_two_clear_actions_and_keeps_requested_units():
    class Translator:
        def translate_copilot_text(self, _text, **_languages):
            pytest.fail("the controlled instruction does not require a model")

    client, _ = setup(Translator())
    prompt = "안내 문구 ‘건을 300 mm 떨어뜨리고, 도막은 160 µm를 목표로 하세요.’를 베트남어로 번역해 줘."
    response = client.post(URL, json=body(prompt, summarize=True))
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["answer"] == "Giữ súng phun cách bề mặt cần sơn 300 mm và đặt mục tiêu độ dày màng sơn là 160 µm."
    assert data["response_language"] == "vi" and data["citations"] == []
    assert data["retrieval_used"] is False
