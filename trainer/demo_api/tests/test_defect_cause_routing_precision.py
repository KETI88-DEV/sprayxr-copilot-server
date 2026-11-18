"""Curated passages cannot narrow a composite question or inherit a new topic."""
import pytest

from trainer.demo_api.defect_cause import defect_cause_topic
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.tests.test_query_contextual_dialogue import DialoguePipeline, ask, client


@pytest.mark.parametrize('prompt', [
    '저도막 말고 과도막 원인을 알려 주세요.',
    '저도막과 과도막의 원인을 비교해 주세요.',
    '과도막과 도막 두께 부족은 왜 생기나요?',
])
def test_multiple_known_subjects_do_not_force_one_curated_topic(prompt):
    assert defect_cause_topic(prompt) is None


def test_history_is_not_inherited_without_an_explicit_followup_decision():
    store=SequenceStore(':memory:')
    store.record_query_turn(session_key='CHAT:U',user_id='U',prompt='저도막의 원인은?',answer='이전 답',
        task_type='knowledge_search',created_at='2026-11-18T05:05:01Z',answer_source='documents')
    p=DialoguePipeline()
    ask(client(p,store),'도장 처짐은 왜 생겨?',response_mode='detailed')
    assert 'pinned_chunk_ids' not in p.calls[-1]
    assert '[결함 원인 질문]' not in p.calls[-1]['instruction']


def test_an_explicit_followup_to_a_composite_subject_stays_generic():
    store=SequenceStore(':memory:')
    store.record_query_turn(session_key='CHAT:U',user_id='U',prompt='저도막과 과도막의 차이는?',answer='이전 답',
        task_type='knowledge_search',created_at='2026-11-18T05:05:01Z',answer_source='documents')
    p=DialoguePipeline({'kind':'coating_knowledge','intent':'knowledge','scope':'unspecified','context_relation':'followup'})
    ask(client(p,store),'그럼 원인은?',response_mode='detailed')
    assert 'pinned_chunk_ids' not in p.calls[-1]
    assert '[결함 원인 질문]' not in p.calls[-1]['instruction']
