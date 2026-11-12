"""Conversation remains contextual across arbitrary subjects, clients and viewer corrections."""
import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.copilot_router import create_copilot_router
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.viewer_context import sign_context


class DialoguePipeline:
    def __init__(self):
        self.plans = []
        self.conversations = []
        self.rag_calls = []
    def plan_copilot_dialogue(self, prompt, *, task_type, recent_turns, facts, **kwargs):
        self.plans.append({"prompt": prompt, "history": recent_turns, "facts": facts})
        return {"kind": "correction" if "수평" in prompt else "chat", "intent": "order_direction" if "수평" in prompt else "knowledge", "scope": "zone" if "수평" in prompt else "unspecified"}
    def converse_copilot(self, prompt, *, recent_turns, facts, response_mode, **kwargs):
        self.conversations.append({"prompt": prompt, "history": recent_turns, "facts": facts})
        if "수평" in prompt:
            return "선택한 Zone 10은 수평형입니다. 앞서 대각선은 전체 기록 기준이었습니다."
        if prompt == "야":
            return "네, 말씀하세요."
        if prompt == "그 둘을 더하면?":
            return "1과 3을 더하면 4입니다." if recent_turns else "어떤 두 수를 말하시는지 알려주세요."
        return "1과 3을 기억하겠습니다." if "기억" in prompt else "1950년 6월 25일입니다."
    def classify_copilot_intent(self, task, prompt):
        return "order_direction"
    def run(self, **kwargs):
        self.rag_calls.append(kwargs)
        return {"response": "[답변] 현재 기록의 방향은 diagonal입니다.", "rag_chunks": []}


def client(pipeline, store):
    app = FastAPI()
    app.include_router(create_copilot_router(lambda: pipeline, store=store))
    return TestClient(app)


def body(prompt, **kwargs):
    return {"user_id": "COPILOT-DEMO-USER", "task_type": "trajectory_recomm", "prompt": prompt, **kwargs}


def test_unrelated_questions_and_greetings_do_not_search_painting_manuals():
    pipeline = DialoguePipeline()
    c = client(pipeline, SequenceStore(":memory:"))
    answer = c.post("/api/v1/copilot/query", json=body("625전쟁 날짜는?")).json()["data"]
    assert answer["answer"] == "1950년 6월 25일입니다." and answer["retrieval_used"] is False
    answer = c.post("/api/v1/copilot/query", json=body("야")).json()["data"]
    assert answer["answer"] == "네, 말씀하세요." and answer["retrieval_used"] is False
    assert pipeline.rag_calls == []


def test_conversation_header_preserves_no_session_dialogue_without_creating_training_sessions():
    pipeline = DialoguePipeline(); store = SequenceStore(":memory:"); c = client(pipeline, store)
    headers = {"X-Copilot-Conversation": "viewer-thread-A"}
    c.post("/api/v1/copilot/query", json=body("1과 3을 기억해"), headers=headers)
    response = c.post("/api/v1/copilot/query", json=body("그 둘을 더하면?"), headers=headers)
    assert response.json()["data"]["answer"] == "1과 3을 더하면 4입니다."
    assert pipeline.plans[-1]["history"][-1]["prompt"] == "1과 3을 기억해"
    other = c.post("/api/v1/copilot/query", json=body("그 둘을 더하면?"), headers={"X-Copilot-Conversation": "viewer-thread-B"})
    assert "어떤 두 수" in other.json()["data"]["answer"]
    assert pipeline.plans[-1]["history"] == []


def test_same_training_session_does_not_mix_other_users_or_conversation_windows():
    pipeline = DialoguePipeline(); store = SequenceStore(":memory:"); c = client(pipeline, store)
    c.post("/api/v1/copilot/query", json=body("1과 3을 기억해", session_id="SHARED"), headers={"X-Copilot-Conversation":"A"})
    different = c.post("/api/v1/copilot/query", json=body("그 둘을 더하면?", session_id="SHARED"), headers={"X-Copilot-Conversation":"B"})
    assert "어떤 두 수" in different.json()["data"]["answer"]
    c.post("/api/v1/copilot/query", json=body("그 둘을 더하면?", user_id="ANOTHER", session_id="SHARED"), headers={"X-Copilot-Conversation":"A"})
    assert not pipeline.plans[-1]["history"]


def test_direction_correction_receives_current_zone_and_record_facts():
    pipeline=DialoguePipeline(); c=client(pipeline, SequenceStore(":memory:"))
    facts={"record_direction":"diagonal", "zones":{"1:10":{"map_num":"1","zone_id":"10","direction":"horizontal"}}}
    from trainer.demo_api.playback_direction import build_playback_snapshot
    from trainer.demo_api.tests.test_playback_direction import trace
    claims,signature=build_playback_snapshot(trace([[0,1,0],[0,1.01,0],[0,1.02,0]]),record_id="REC",duration_seconds=1)
    context="motion_viewer_analysis:"+json.dumps({"dialogue_context":facts,"dialogue_signature":sign_context("dialogue", facts),
        "frame_direction_context":claims,"frame_direction_signature":signature,
        "playback_context":{"map_num":"1","zone_id":"10","playhead_seconds":.1}})
    result=c.post("/api/v1/copilot/query", json=body("지금은 수평방향일텐데?", context=context)).json()["data"]
    assert "수직" in result["answer"] and "대표" not in result["answer"]
    assert "수평" not in result["answer"] and result["sequence_context"]["scope"] == "current_frame"
    assert pipeline.plans[-1]["facts"]["record_direction"] == "diagonal"
    assert pipeline.plans[-1]["facts"]["selected_zone"]["direction"] == "horizontal"
    assert pipeline.rag_calls == []


def test_unsigned_viewer_facts_are_not_used_as_current_measurements():
    pipeline=DialoguePipeline(); c=client(pipeline, SequenceStore(":memory:"))
    context="motion_viewer_analysis:"+json.dumps({"dialogue_context":{"record_direction":"forged"}, "dialogue_signature":"0"*64})
    response=c.post("/api/v1/copilot/query", json=body("지금은 수평방향일텐데?", context=context)).json()
    assert "forged" not in json.dumps(response)
    assert "현재 프레임" in response["data"]["answer"] and pipeline.conversations == []


def test_pose_followup_uses_selected_zone_corrections_instead_of_global_corrections():
    pipeline=DialoguePipeline()
    pipeline.plan_copilot_dialogue=lambda *a,**kw: {"kind":"lookup","intent":"pose_right","scope":"zone"}
    def answer(prompt, *, recent_turns, facts, **kwargs):
        return facts["selected_zone"]["corrections"][0]["message"]
    pipeline.converse_copilot=answer
    c=client(pipeline,SequenceStore(":memory:"))
    facts={"zones":{"1:1":{"map_num":"1","zone_id":"1","corrections":[{"message":"오른손을 5 cm 올리세요."}]}}}
    ctx="motion_viewer_analysis:"+json.dumps({"dialogue_context":facts,"dialogue_signature":sign_context("dialogue",facts),"playback_context":{"map_num":"1","zone_id":"1"}})
    result=c.post("/api/v1/copilot/query",json=body("그럼 오른손은?",task_type="pose_recomm",context=ctx)).json()["data"]
    assert result["answer"]=="오른손을 5 cm 올리세요." and result["retrieval_used"] is False
    assert not pipeline.rag_calls


class HistoryBiasedPipeline(DialoguePipeline):
    """계획 모델이 직전 자세 대화에 끌려 지식 질문을 기록 후속 대화로 분류하던 상황(2026-11-12 카나리 재현)."""
    def plan_copilot_dialogue(self, prompt, *, task_type, recent_turns, facts, **kwargs):
        self.plans.append({"prompt": prompt, "history": recent_turns, "facts": facts})
        if recent_turns:
            return {"kind": "lookup", "intent": "pose_overall", "scope": "record"}
        return {"kind": "coating_knowledge", "intent": "knowledge", "scope": "unspecified"}
    def converse_copilot(self, prompt, **kwargs):
        self.conversations.append(prompt)
        return "적정 이격 거리 관련 데이터는 제공되지 않았습니다."
    def run(self, **kwargs):
        self.rag_calls.append(kwargs)
        return {"response": "[답변] 적정 이격거리는 30~40cm입니다.", "rag_chunks": []}


def test_knowledge_question_after_pose_question_still_searches_documents():
    pipeline = HistoryBiasedPipeline(); c = client(pipeline, SequenceStore(":memory:"))
    headers = {"X-Copilot-Conversation": "viewer-thread-K"}
    c.post("/api/v1/copilot/query", json=body("지금 자세는 어때?", task_type="pose_recomm"), headers=headers)
    response = c.post("/api/v1/copilot/query", json=body("적정 이격거리는?", task_type="knowledge_search"), headers=headers)
    assert "적정 이격거리는?" not in [plan["prompt"] for plan in pipeline.plans], "일반 지식 질문은 기록에 끌리는 계획 모델을 부르지 않는다"
    assert response.headers["X-Copilot-Answer-Source"] == "documents"
    assert "30~40cm" in response.json()["data"]["answer"] and pipeline.conversations == []


def test_record_followup_in_knowledge_tab_stays_in_conversation():
    pipeline = HistoryBiasedPipeline(); c = client(pipeline, SequenceStore(":memory:"))
    headers = {"X-Copilot-Conversation": "viewer-thread-F"}
    c.post("/api/v1/copilot/query", json=body("지금 자세는 어때?", task_type="pose_recomm"), headers=headers)
    for prompt in ("그럼 이 Zone 은?", "아까 그거 다시 말해줘"):
        response = c.post("/api/v1/copilot/query", json=body(prompt, task_type="knowledge_search"), headers=headers)
        assert response.headers["X-Copilot-Answer-Source"] == "conversation"
    assert pipeline.rag_calls == []


def test_knowledge_question_without_history_searches_documents():
    pipeline = HistoryBiasedPipeline(); c = client(pipeline, SequenceStore(":memory:"))
    response = c.post("/api/v1/copilot/query", json=body("적정 이격거리는?", task_type="knowledge_search"))
    assert response.headers["X-Copilot-Answer-Source"] == "documents"
    assert pipeline.plans == [] and pipeline.conversations == []


def test_knowledge_tab_presets_after_pose_question_search_documents_without_planning():
    pipeline = HistoryBiasedPipeline(); c = client(pipeline, SequenceStore(":memory:"))
    headers = {"X-Copilot-Conversation": "viewer-thread-P"}
    c.post("/api/v1/copilot/query", json=body("지금 자세는 어때?", task_type="pose_recomm"), headers=headers)
    planned = len(pipeline.plans)
    for prompt in ("저도막이 뭐야?", "적정 이격거리는?", "적정 분사 속도는?", "과도막의 원인은?"):
        response = c.post("/api/v1/copilot/query", json=body(prompt, task_type="knowledge_search"), headers=headers)
        assert response.headers["X-Copilot-Answer-Source"] == "documents", prompt
    assert len(pipeline.plans) == planned and pipeline.conversations == []


def test_record_followups_in_coating_tab_keep_the_planner_chat_route():
    pipeline = DialoguePipeline(); c = client(pipeline, SequenceStore(":memory:"))
    headers = {"X-Copilot-Conversation": "viewer-thread-C"}
    for prompt in ("속도는 괜찮았어?", "제 이격거리는 적당했나요?", "이격거리는 어땠어?", "나의 속도는?", "그럼 속도는?"):
        response = c.post("/api/v1/copilot/query", json=body(prompt, task_type="knowledge_search"), headers=headers)
        assert response.headers["X-Copilot-Answer-Source"] == "conversation", prompt
    assert pipeline.rag_calls == []
