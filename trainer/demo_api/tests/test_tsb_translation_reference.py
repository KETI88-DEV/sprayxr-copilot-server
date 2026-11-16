"""Text and voice translation share optional TSB hints, not a new forced glossary."""
import json

from trainer.demo_api import query_language as language


def test_translation_messages_use_vendor_terms_as_optional_request_data():
    messages = language.translation_messages('노즐 상태를 확인하세요.',source_language='ko',target_language='vi')
    request = json.loads(messages[-1]['content'])
    assert any(item['source']=='노즐' and item['target']=='Vòi phun' for item in request['translation_reference'])
    assert 'optional' in messages[0]['content'] and 'glossary takes priority' in messages[0]['content']


def test_source_attested_glossary_remains_authoritative_and_tsb_hints_are_not_required():
    request = json.loads(language.translation_messages('건도막 두께를 확인하세요.',source_language='ko',target_language='en')[-1]['content'])
    assert request['glossary']==[{'source':'건도막 두께','target':'dry film thickness'}]
    assert not request.get('translation_reference')
    class Translator:
        def translate_copilot_text(self,*args,**kwargs):
            return 'Check the spray tip.'
    assert language.translate_text(Translator(),'노즐을 확인하세요.',source_language='ko',target_language='en')=='Check the spray tip.'


def test_unsupported_or_quarantined_languages_do_not_receive_vendor_hints():
    for target in ('si','ru','km'):
        request = json.loads(language.translation_messages('노즐 상태를 확인하세요.',source_language='ko',target_language=target)[-1]['content'])
        assert not request.get('translation_reference')


def test_translation_cache_is_scoped_to_the_reference_corpus_version(monkeypatch):
    from trainer.scripts.quick_answer import TextCache
    version = ['one']
    monkeypatch.setattr(language,'reference_version',lambda:version[0])
    class Translator:
        _translation_cache = TextCache(capacity=4,ttl=120)
        calls = 0
        def translate_copilot_text(self,*args,**kwargs):
            self.calls += 1
            return 'Check the spray tip.'
    translator = Translator()
    for _ in range(2):
        language.translate_text(translator,'노즐을 확인하세요.',source_language='ko',target_language='en')
    assert translator.calls==1
    version[0]='two'
    language.translate_text(translator,'노즐을 확인하세요.',source_language='ko',target_language='en')
    assert translator.calls==2
