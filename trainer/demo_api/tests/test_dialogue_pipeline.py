import json
import pytest
from trainer.demo_api.dialogue import parse_plan, plan_regex, viewer_dialogue_context
from trainer.scripts.agent_pipeline import AgentPipeline


class Counter:
    def count_messages(self, messages):
        return sum(len(m['content']) // 3 + 4 for m in messages)


def pipeline():
    p = AgentPipeline.__new__(AgentPipeline)
    p.config = {'generation': {'max_new_tokens': 768}}
    p.model = object()
    p._prompt_token_counter = lambda: Counter()
    p.calls = []
    def generate(messages, **kwargs):
        p.calls.append((messages, kwargs))
        return '{"kind":"chat","intent":"knowledge","scope":"unspecified"}' if kwargs.get('answer_regex') else '1과 3은 4입니다.'
    p._generate_direct = generate
    return p


def test_general_conversation_has_history_and_no_painting_qa_scaffold():
    p = pipeline()
    answer = p.converse_copilot('그 둘을 더하면?', recent_turns=[{'prompt':'1과 3을 기억해','answer':'기억했습니다.'}], facts={}, response_mode='quick', inference_context_limit=4096)
    messages, options = p.calls[-1]
    assert answer == '1과 3은 4입니다.'
    assert [m['role'] for m in messages] == ['system','user','assistant','user']
    assert '1과 3' in messages[1]['content'] and messages[-1]['content'] == '그 둘을 더하면?'
    assert options['answer_regex'] is None and options['max_tokens'] == 1024
    assert options['enable_thinking'] is False


def test_planner_sees_current_utterance_and_previous_turns():
    import re
    p=pipeline()
    decision=p.plan_copilot_dialogue('야',task_type='trajectory_recomm',recent_turns=[{'prompt':'도포 방향은?','answer':'대각선'}],facts={},inference_context_limit=4096)
    messages, options=p.calls[-1]
    data=json.loads(messages[-1]['content'])
    assert data['current_utterance']=='야' and data['recent_dialogue'][0]['prompt']=='도포 방향은?'
    assert re.fullmatch(plan_regex(),json.dumps(decision))
    assert options['max_tokens'] <= 192
    expanded = {"kind": "coating_knowledge", "intent": "knowledge", "scope": "unspecified",
                "context_relation": "followup", "resolved_question": "거리가 적절하다는 조건에서 저도막이 남는 원인은?"}
    assert re.fullmatch(options['answer_regex'], json.dumps(expanded, ensure_ascii=False))
    assert parse_plan(expanded) == expanded


def test_context_trimming_keeps_current_question_and_whole_exchanges():
    p=pipeline()
    p.converse_copilot('현재 질문',recent_turns=[{'prompt':'오래된 질문'*300,'answer':'오래된 답변'*300} for _ in range(6)],facts={},response_mode='quick',inference_context_limit=2048)
    messages,_=p.calls[-1]
    assert messages[0]['role']=='system' and messages[-1]['content']=='현재 질문'
    assert len(messages)%2==0 and len(messages)<14


@pytest.mark.parametrize('value',[{}, {'kind':'chat','intent':'knowledge','scope':'unknown'}, {'kind':'chat','intent':'knowledge','scope':'record','extra':'instruction'}])
def test_plans_must_follow_closed_contract(value):
    with pytest.raises(ValueError):parse_plan(value)


@pytest.mark.parametrize("extra", [{"context_relation": "other"}, {"resolved_question": "가" * 601}])
def test_dialogue_interpretation_is_bounded_after_decoding(extra):
    with pytest.raises(ValueError):
        parse_plan({"kind": "coating_knowledge", "intent": "knowledge", "scope": "unspecified", **extra})


def test_viewer_facts_keep_zone_pattern_separate_from_global_direction():
    result=viewer_dialogue_context({'direction_summary':{'classification':'diagonal'},'pose_evaluation':{'zones':{'1:10':{'map_num':'1','zone_id':'10','metrics':[{'key':'paint_pattern','available':True,'actual':'horizontal','reference':'horizontal'}]}}}})
    assert result['record_direction']=='diagonal' and result['zones']['1:10']['direction']=='horizontal'


@pytest.mark.parametrize("mode",["quick","detailed"])
def test_real_query_pipeline_never_reads_or_writes_unscoped_shared_memory(mode, monkeypatch):
    import sys
    from types import ModuleType
    from trainer.scripts.tests.test_agent_pipeline_use_rag import _pipeline
    from trainer.demo_api.tests.test_dialogue_conversation import client, body
    from trainer.demo_api.sequence_store import SequenceStore
    monkeypatch.setitem(sys.modules,"search_index",ModuleType("search_index"))
    p,prompts=_pipeline()
    p.plan_copilot_dialogue=lambda *args,**kwargs: {"kind":"coating_knowledge","intent":"knowledge","scope":"unspecified"}
    p.session_context.update("SHARED",{"instruction":"OTHER-USER-SECRET","response":"PRIVATE-MEMORY","tool_calls":[]})
    before=p.session_context.get_context("SHARED")
    result=client(p,SequenceStore(":memory:")).post("/api/v1/copilot/query",json=body("저도막이 뭐야?",task_type="knowledge_search",session_id="SHARED",response_mode=mode,use_rag="N"))
    assert result.status_code==200,result.text
    assert prompts and "OTHER-USER-SECRET" not in json.dumps(prompts)
    assert "PRIVATE-MEMORY" not in json.dumps(prompts)
    assert p.session_context.get_context("SHARED")==before


def test_planner_trims_history_by_tokens_not_only_character_count():
    p=pipeline()
    p.plan_copilot_dialogue("야",task_type="trajectory_recomm",recent_turns=[{"prompt":"과거질문"*100,"answer":"과거답변"*100} for _ in range(6)],facts={},inference_context_limit=1500)
    messages,_=p.calls[-1]
    assert len(json.loads(messages[-1]["content"])["recent_dialogue"])<6
    assert Counter().count_messages(messages)+128<=1500
