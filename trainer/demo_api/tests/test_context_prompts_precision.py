"""Curated passages cannot narrow a composite question or inherit a new topic."""
import pytest

from trainer.demo_api.context_prompts import matched_prompt
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.tests.test_query_contextual_dialogue import DialoguePipeline, ask, client


@pytest.mark.parametrize('prompt', [
    '저도막 말고 과도막 원인을 알려 주세요.',
    '저도막과 과도막의 원인을 비교해 주세요.',
    '과도막과 도막 두께 부족은 왜 생기나요?',
])
def test_multiple_known_subjects_do_not_force_one_curated_topic(prompt):
    assert matched_prompt(prompt) is None


def test_history_is_not_inherited_without_an_explicit_followup_decision():
    store=SequenceStore(':memory:')
    store.record_query_turn(session_key='CHAT:U',user_id='U',prompt='저도막의 원인은?',answer='이전 답',
        task_type='knowledge_search',created_at='2026-11-18T05:05:01Z',answer_source='documents')
    p=DialoguePipeline()
    ask(client(p,store),'도장 처짐은 왜 생겨?',response_mode='detailed')
    # 앞 질문의 저도막을 이어받지 않는다. 처짐(흘림)은 2026-11-20부터 제 묶음이 있다.
    assert '질문한 결함: 저도막' not in p.calls[-1]['instruction'] and '질문한 결함: 흘림' in p.calls[-1]['instruction']
    ask(client(p,store),'도장 일정은 왜 밀려?',response_mode='detailed')
    assert 'pinned_chunk_ids' not in p.calls[-1] and '[결함 원인 질문]' not in p.calls[-1]['instruction']


def test_an_explicit_followup_to_a_composite_subject_stays_generic():
    store=SequenceStore(':memory:')
    store.record_query_turn(session_key='CHAT:U',user_id='U',prompt='저도막과 과도막의 차이는?',answer='이전 답',
        task_type='knowledge_search',created_at='2026-11-18T05:05:01Z',answer_source='documents')
    p=DialoguePipeline({'kind':'coating_knowledge','intent':'knowledge','scope':'unspecified','context_relation':'followup'})
    ask(client(p,store),'그럼 원인은?',response_mode='detailed')
    assert 'pinned_chunk_ids' not in p.calls[-1]
    assert '[결함 원인 질문]' not in p.calls[-1]['instruction']


@pytest.mark.parametrize('prompt', ['저도막과 과도막이 생기면 어떤 문제가 있어?', '과도막 말고 저도막은 어떤 문제가 생겨?'])
def test_multiple_subjects_do_not_force_one_effect_type(prompt):
    assert matched_prompt(prompt) is None
