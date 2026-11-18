"""An unavailable RAG answer can start a separately verified general explanation."""
import json
import pytest
import re

from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.tests.test_dialogue_pipeline import pipeline as real_pipeline
from trainer.demo_api.tests.test_query_evidence_fallback import ask,client
from trainer.scripts.answer_semantics import UNCONFIRMED_ANSWER


def unavailable_pipeline(verdict, *, judge_unavailable=False):
    p=real_pipeline()
    p.config['confidence']={'abstain_below_rerank_score':4.0}
    p.plan_copilot_dialogue=lambda *_args,**_kwargs:{'kind':'coating_knowledge','intent':'knowledge','scope':'unspecified'}
    def run(*,defer_without_evidence=False,evidence_score_threshold=None,**_kwargs):
        return {'response':UNCONFIRMED_ANSWER,'rag_chunks':[{'chunk_id':'DOC-one-C0001','text':'원인을 뒷받침하지 않는 자료','rerank_score':6.0}],
                'metadata':{'answer_review':{'verdict':verdict},'answer_trace':{'review':{'verdict':verdict}}}}
    p.run=run
    calls=[]
    def generate(_messages,**options):
        review=bool(options.get('answer_regex') and re.fullmatch(options['answer_regex'],'이유: 검토\n근거: 없음\n판정: OK'))
        calls.append('review' if review else 'generate')
        if review:
            if judge_unavailable:raise TimeoutError('judge temporarily unavailable')
            return '이유: 가능한 조건과 한계를 구분함\n근거: 없음\n판정: OK'
        return '[답변] 일반적으로 작업 조건에 따라 결과가 달라질 수 있습니다. 현재 원인을 확정하려면 기록을 확인해야 합니다.'
    p._generate_direct=generate
    return p,calls


@pytest.mark.parametrize('verdict',['budget_exhausted','unavailable','rejected','invalid_review','invalid_support'])
def test_failed_rag_is_discarded_before_a_fresh_answer_and_its_independent_review(verdict):
    p,calls=unavailable_pipeline(verdict)
    store=SequenceStore(':memory:')
    response=ask(client(p,store))
    data=response.json()['data']
    assert response.status_code==200 and not data['degraded']
    assert calls==['generate','review']
    assert '작업 조건에 따라' in data['answer'] and '충분히 확인하지 못했습니다' not in data['answer']
    assert data['citations']==[] and not data['retrieval_used']
    assert response.headers['X-Copilot-Answer-Source']=='conversation'
    saved=json.loads(store._conn.execute('SELECT query_context FROM query_turns ORDER BY id DESC LIMIT 1').fetchone()[0])
    assert saved['answer_trace']['fallback_reason']=='rag_answer_unavailable'
    assert saved['answer_trace']['review']['verdict']==verdict


def test_a_fresh_fallback_with_an_unavailable_judge_is_an_error_not_an_approved_answer():
    p,calls=unavailable_pipeline('budget_exhausted',judge_unavailable=True)
    response=ask(client(p))
    assert response.status_code==503 and response.json()['data'] is None
    assert calls==['generate','review']


def test_reviewed_document_only_limitation_is_not_prefixed_as_a_general_explanation():
    p,_=unavailable_pipeline('budget_exhausted')
    candidate='제공된 문서에 해당 정보가 없습니다.'
    def generate(_messages,**options):
        if re.fullmatch(options.get('answer_regex',''), '이유: 검토\n근거: 없음\n판정: OK'):
            return '이유: 문서만 요구한 한계를 밝힘\n근거: 없음\n판정: OK'
        return '[답변] '+candidate
    p._generate_direct=generate
    response=ask(client(p),prompt='문서에 명시된 내용만 답해 주세요. 없으면 보충하지 마세요.')
    assert response.status_code==200
    data=response.json()['data']
    assert data['answer']==candidate
    assert data['citations']==[] and not data['retrieval_used']


@pytest.mark.parametrize('extra',[{'use_rag':'N'},{'context':'현재 사용자 지정 자료를 바탕으로만 답해 주세요.'}])
def test_explicit_no_rag_or_client_context_does_not_widen_failed_rag_into_general_knowledge(extra):
    p,calls=unavailable_pipeline('budget_exhausted')
    response=ask(client(p),**extra)
    assert response.status_code==200 and response.json()['data']['degraded']
    assert calls==[]
