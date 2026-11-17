"""Meaningful dialogue survives routing without promoting generated text to facts."""
from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.copilot_router import create_copilot_router
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.tests.document_evidence import supported_document_result
from trainer.demo_api.tests.test_pose_engine_first import ARM, _viewer_context
from trainer.demo_api.viewer_context import sign_context


QUERY = "/api/v1/copilot/query"
ORIGINAL_ANSWER = "저도막은 표면 장력의 영향으로 모서리에서 도막이 얇아지는 현상입니다."


class DialoguePipeline:
    def __init__(self, decision=None):
        self.decision = decision or {"kind": "coating_knowledge", "intent": "knowledge", "scope": "unspecified"}
        self.calls = []
        self.plans = []
        self.conversations = []

    def plan_copilot_dialogue(self, prompt, **kwargs):
        self.plans.append((prompt, kwargs))
        return self.decision

    def converse_copilot(self, prompt, **kwargs):
        self.conversations.append((prompt, kwargs))
        return "프랑스의 수도는 파리입니다."

    def run(self, **kwargs):
        self.calls.append(kwargs)
        return supported_document_result("[답변] 이동 속도와 패스 겹침을 확인하세요. 거리가 적절했다는 조건만으로 원인을 확정할 수는 없습니다.")


def client(pipeline, store=None):
    app = FastAPI()
    app.include_router(create_copilot_router(lambda: pipeline, store=store or SequenceStore(":memory:")))
    return TestClient(app)


def previous(store, *, prompt="저도막이 생기는 원인은 무엇인가요?", answer=ORIGINAL_ANSWER):
    store.record_query_turn(session_key="CHAT:U", user_id="U", prompt=prompt, answer=answer,
                            task_type="knowledge_search", created_at="2026-11-16T05:05:01Z", answer_source="documents")


def ask(c, prompt, **kwargs):
    response = c.post(QUERY, json={"user_id": "U", "task_type": "knowledge_search", "prompt": prompt, **kwargs})
    assert response.status_code == 200, response.text
    return response.json()["data"]


def test_generated_previous_answer_survives_as_unverified_dialogue_for_correction():
    store = SequenceStore(":memory:")
    previous(store)
    pipeline = DialoguePipeline()
    result = ask(client(pipeline, store), "이격거리는 적절했는데?")
    assert result["retrieval_used"] is True and len(pipeline.calls) == 1
    instruction = pipeline.calls[0]["instruction"]
    assert ORIGINAL_ANSWER in instruction
    assert "미검증" in instruction
    assert "사용자: 저도막이 생기는 원인은 무엇인가요?" in instruction
    generation = pipeline.calls[0]["generation_instruction"]
    assert generation.rsplit("[지금 답할 질문", 1)[1].endswith("이격거리는 적절했는데?")
    planned_turn = pipeline.plans[0][1]["recent_turns"][-1]
    assert ORIGINAL_ANSWER in planned_turn["answer"] and "미검증" in planned_turn["answer"]


@pytest.mark.parametrize("task", ["knowledge_search", "pose_recomm", "trajectory_recomm"])
def test_original_cause_question_reaches_generator_separately_from_wrapped_instruction(task):
    pipeline = DialoguePipeline()
    prompt = "저도막이 생기는 원인은 무엇인가요?"
    data = ask(client(pipeline), prompt, task_type=task)
    assert data["retrieval_used"] is True
    assert pipeline.calls[0]["answer_question"] == prompt
    assert "원인" in pipeline.calls[0]["instruction"]


@pytest.mark.parametrize("task", ["knowledge_search", "pose_recomm", "trajectory_recomm"])
def test_generic_body_technique_question_does_not_become_recorded_arm_score(task):
    pipeline = DialoguePipeline()
    context = _viewer_context(sign_context("arm_pose", ARM))
    data = ask(client(pipeline), "팔꿈치 자세를 편하게 유지하는 방법은?", task_type=task, context=context)
    assert data["retrieval_used"] is True
    assert len(pipeline.calls) == 1
    assert data["pose_context"] is None
    assert "60%" not in data["answer"] and "SEQ-001" not in data["answer"]


def test_explicit_current_arm_measurement_keeps_verified_engine_path():
    pipeline = DialoguePipeline()
    context = _viewer_context(sign_context("arm_pose", ARM))
    data = ask(client(pipeline), "지금 오른손 자세 어때?", task_type="pose_recomm", context=context)
    assert data["retrieval_used"] is False
    assert pipeline.calls == [] and pipeline.plans == []
    assert "65°" in data["answer"] and "교정" in data["answer"]


@pytest.mark.parametrize("task", ["knowledge_search", "pose_recomm", "trajectory_recomm"])
@pytest.mark.parametrize("question", ["이 Zone에서 팔꿈치 부담을 줄이려면 어떻게 해야 해?",
                                    "이 Zone에서 오른손을 어떻게 움직이면 좋아?"])
def test_current_zone_body_method_question_reaches_qwen_with_verified_arm_facts(task, question):
    pipeline = DialoguePipeline()
    context = _viewer_context(sign_context("arm_pose", ARM))
    data = ask(client(pipeline), question, task_type=task, context=context)
    assert data["retrieval_used"] is True and len(pipeline.calls) == 1
    instruction = pipeline.calls[0]["generation_instruction"]
    assert "selected_zone_arm_pose" in instruction and "80" in instruction
    assert question in instruction
    assert data["pose_context"] is None


def test_forged_arm_values_cannot_become_explanation_facts():
    pipeline = DialoguePipeline()
    raw = json.loads(_viewer_context(sign_context("arm_pose", ARM)).split(":", 1)[1])
    raw["arm_pose_context"]["zones"]["8"]["similarity_percent"] = 9999
    data = ask(client(pipeline), "이 Zone에서 팔꿈치 부담을 줄이려면 어떻게 해야 해?", task_type="pose_recomm",
               context="motion_viewer_analysis:" + json.dumps(raw))
    assert data["retrieval_used"] is True
    instruction = pipeline.calls[0]["generation_instruction"]
    assert "9999" not in instruction and "selected_zone_arm_pose" not in instruction


def test_pose_score_calculation_principle_is_explanation_not_a_saved_score_lookup():
    from trainer.demo_api.copilot_contracts import CopilotQueryRequest
    from trainer.demo_api.copilot_router import _pose_engine_question
    question = "자세 유사도 계산 원리는?"
    query = CopilotQueryRequest(user_id="U", task_type="pose_recomm", prompt=question)
    assert not _pose_engine_question(query)
    pipeline = DialoguePipeline()
    data = ask(client(pipeline), question, task_type="pose_recomm", context=_viewer_context(sign_context("arm_pose", ARM)))
    assert data["retrieval_used"] is True and pipeline.calls
    assert data["pose_context"] is None


@pytest.mark.parametrize("task", ["knowledge_search", "pose_recomm", "trajectory_recomm"])
def test_general_importance_of_correction_is_not_a_recorded_correction_lookup(task):
    pipeline = DialoguePipeline()
    data = ask(client(pipeline), "일반적으로 팔꿈치 교정이 중요한 이유는?", task_type=task,
               context=_viewer_context(sign_context("arm_pose", ARM)))
    assert data["retrieval_used"] is True and len(pipeline.calls) == 1
    assert data["pose_context"] is None


def test_current_direction_correction_uses_current_playback_not_previous_generated_direction(monkeypatch):
    from trainer.demo_api import copilot_router
    store = SequenceStore(":memory:")
    previous(store, prompt="도포 방향은?", answer="현재 기록의 주요 도포 방향은 대각선입니다.")
    pipeline = DialoguePipeline({"kind": "correction", "intent": "order_direction", "scope": "playback"})
    direction = {"analysis_status": "available", "label": "수평", "classification": "horizontal", "playhead_seconds": 3.0}
    monkeypatch.setattr(copilot_router, "playback_direction_from_context", lambda _: direction)
    order = {"recommended_flow": ["1", "2"], "selected_flow": ["1"], "zone_keys": ["1:1", "1:2"]}
    context = "motion_viewer_analysis:" + json.dumps({"analysis_context": order,
        "signature": sign_context("trajectory_recomm", order),
        "playback_context": {"map_num": "1", "zone_id": "1", "playhead_seconds": 3.0}})
    data = ask(client(pipeline, store), "지금은 수평일텐데?", task_type="trajectory_recomm", context=context)
    assert "수평" in data["answer"] and "대각선" not in data["answer"]
    assert data["retrieval_used"] is False and pipeline.calls == []


def test_new_unrelated_topic_uses_latest_question_without_repeating_painting_answer():
    store = SequenceStore(":memory:")
    previous(store)
    pipeline = DialoguePipeline({"kind": "chat", "intent": "knowledge", "scope": "unspecified"})
    data = ask(client(pipeline, store), "프랑스의 수도는?", task_type="pose_recomm")
    assert data["answer"] == "프랑스의 수도는 파리입니다."
    assert pipeline.calls == [] and pipeline.conversations[-1][0] == "프랑스의 수도는?"


def test_prior_generated_numbers_are_not_added_to_authenticated_facts():
    store = SequenceStore(":memory:")
    previous(store, answer="이격거리는 999 mm입니다.")
    pipeline = DialoguePipeline({"kind": "chat", "intent": "knowledge", "scope": "unspecified"})
    ask(client(pipeline, store), "고마워")
    facts = pipeline.conversations[-1][1]["facts"]
    assert "999" not in json.dumps(facts)


def test_accumulated_conditions_and_scope_correction_remain_available_to_qwen():
    store = SequenceStore(":memory:")
    previous(store)
    previous(store, prompt="이격거리는 적절했는데?", answer="그렇다면 속도와 패스 겹침을 확인해야 합니다.")
    pipeline = DialoguePipeline()
    ask(client(pipeline, store), "속도도 일정했어. 모서리 말고 평판 전체가 얇은 이유는?")
    instruction = pipeline.calls[-1]["generation_instruction"]
    assert "이격거리는 적절했는데?" in instruction
    assert "그렇다면 속도와 패스 겹침을 확인해야 합니다." in instruction
    assert instruction.endswith("속도도 일정했어. 모서리 말고 평판 전체가 얇은 이유는?")
    assert all("999" not in json.dumps(plan[1]["facts"]) for plan in pipeline.plans)


def test_semantic_followup_relation_keeps_scope_correction_in_document_path():
    store = SequenceStore(":memory:")
    previous(store)
    pipeline = DialoguePipeline({"kind": "correction", "intent": "knowledge", "scope": "unspecified", "context_relation": "followup"})
    data = ask(client(pipeline, store), "평평한 면 전체가 얇아지는 것도 같은 원리야?")
    assert data["retrieval_used"] is True and pipeline.conversations == []
    instruction = pipeline.calls[-1]["generation_instruction"]
    assert "[문서 후속 질문]" in instruction and ORIGINAL_ANSWER in instruction
    assert pipeline.calls[-1]["answer_question"] == "평평한 면 전체가 얇아지는 것도 같은 원리야?"


def test_semantic_new_topic_does_not_inherit_the_previous_document_subject():
    store = SequenceStore(":memory:")
    previous(store)
    pipeline = DialoguePipeline({"kind": "chat", "intent": "knowledge", "scope": "unspecified", "context_relation": "new_topic"})
    data = ask(client(pipeline, store), "그럼 프랑스의 수도는?")
    assert data["answer"] == "프랑스의 수도는 파리입니다."
    assert data["retrieval_used"] is False and pipeline.calls == []


@pytest.mark.parametrize("relation,marker", [("new_topic", "[새 주제 질문]"), ("followup", "[문서 후속 질문]")])
def test_dialogue_relation_survives_assembling_signed_coating_advice(relation, marker):
    from trainer.demo_api.query_runtime import retrieval_question
    from trainer.demo_api.tests.test_query_question_routing import context
    store = SequenceStore(":memory:")
    previous(store)
    pipeline = DialoguePipeline({"kind": "coating_knowledge", "intent": "knowledge", "scope": "zone",
                                 "context_relation": relation, "resolved_question": ""})
    question = "그런데 이 Zone의 도막 기포를 어떻게 예방하죠?"
    data = ask(client(pipeline, store), question, context=context())
    assert data["retrieval_used"] is True
    call = pipeline.calls[-1]
    instruction = call["generation_instruction"]
    assert "[검증된 최종 도막 비교 자료]" in instruction
    assert marker in instruction
    if relation == "new_topic":
        retrieval = retrieval_question(call["retrieval_instruction"], context=instruction)
        assert "기포" in retrieval and "저도막" not in retrieval


@pytest.mark.parametrize("mode", ["quick", "detailed"])
def test_resolved_followup_question_retains_original_subject_and_accumulated_user_conditions(mode):
    store = SequenceStore(":memory:")
    previous(store)
    previous(store, prompt="이격거리는 적절했는데?", answer="그렇다면 이동 속도를 확인할 수 있습니다.")
    previous(store, prompt="도료 유량은 정상 범위였어요.", answer="유량이 정상이라는 조건도 함께 검토해야 합니다.")
    original = "그런데 평평한 면 전체가 얇은 이유는?"
    resolved = "이격거리가 적절하고 도료 유량이 정상이라는 사용자 조건에서 평평한 면 전체에 저도막이 생기는 가능한 원인은?"
    pipeline = DialoguePipeline({"kind": "coating_knowledge", "intent": "knowledge", "scope": "unspecified",
                                 "context_relation": "followup", "resolved_question": resolved})
    data = ask(client(pipeline, store), original, response_mode=mode)
    assert data["retrieval_used"] is True and len(pipeline.calls) == 1
    call = pipeline.calls[0]
    assert original in call["retrieval_instruction"]
    assert call["answer_question"] == original
    generation = call.get("generation_instruction") or call["instruction"]
    assert resolved in generation
    assert generation.endswith(original)
    assert "미검증" in generation
    assert len(pipeline.plans[0][1]["recent_turns"]) == 3
    from trainer.demo_api.query_runtime import retrieval_question
    search = retrieval_question(call["retrieval_instruction"], context=call["instruction"])
    assert "저도막이 생기는 원인은" in search and "도료 유량은 정상 범위였어요" in search
    assert original in search


@pytest.mark.parametrize("mode", ["quick", "detailed"])
def test_planner_declarative_answer_cannot_replace_the_original_user_subject(mode):
    from trainer.demo_api.query_runtime import retrieval_question
    store = SequenceStore(":memory:")
    previous(store)
    bad = "분사 거리는 적절하지만, 도막의 손상 여부는 다른 요인에 따라 달라질 수 있습니다."
    pipeline = DialoguePipeline({"kind": "coating_knowledge", "intent": "knowledge", "scope": "unspecified",
                                 "context_relation": "followup", "resolved_question": bad})
    original = "이격거리는 적절했는데?"
    ask(client(pipeline, store), original, response_mode=mode)
    call = pipeline.calls[-1]
    assert "적절했는데" in call["retrieval_instruction"] and call["answer_question"] == original
    search = retrieval_question(call["retrieval_instruction"], context=call["instruction"])
    assert "저도막이 생기는 원인은 무엇인가요?" in search and "적절했는데" in search
    generation = call.get("generation_instruction") or call["instruction"]
    assert "손상" not in search and bad not in generation


@pytest.mark.parametrize("advice", ["도막 손상이 생기는 원인은?", "거리 조건을 제외한 얇은 도막 원인"])
@pytest.mark.parametrize("mode", ["quick", "detailed"])
def test_unverified_resolution_is_only_advice_and_cannot_supersede_user_history(advice, mode):
    from trainer.demo_api.query_runtime import retrieval_question
    store = SequenceStore(":memory:")
    previous(store)
    pipeline = DialoguePipeline({"kind": "coating_knowledge", "intent": "knowledge", "scope": "unspecified",
                                 "context_relation": "followup", "resolved_question": advice})
    original = "이격거리는 적절했는데?"
    ask(client(pipeline, store), original, response_mode=mode)
    call = pipeline.calls[-1]
    assert "적절했는데" in call["retrieval_instruction"]
    instruction = call.get("generation_instruction") or call["instruction"]
    assert advice in instruction and instruction.index(advice) < instruction.index("[최근 대화]")
    assert instruction.endswith(original)
    search = retrieval_question(call["retrieval_instruction"], context=call["instruction"])
    assert "저도막이 생기는 원인은 무엇인가요?" in search and "적절했는데" in search
    assert "손상" not in search


def test_resolution_advice_marker_does_not_remove_the_raw_followup_search_context():
    from trainer.demo_api.query_runtime import retrieval_question
    context = ("[문서 후속 질문]\n[후속 질문 해석 — 미검증 대화 참고]\n도막 손상 원인은?\n"
               "[최근 대화]\n사용자: 저도막이 생기는 원인은?\nCo-Pilot: 이전 답변\n"
               "사용자: 도료 유량은 정상 범위였어요.\n")
    result = retrieval_question("이격거리는 적절했는데?", context=context)
    assert "저도막이 생기는 원인은?" in result and "도료 유량은 정상 범위였어요" in result
    assert "손상" not in result
    switched = retrieval_question("기포 예방 방법은?", context="[새 주제 질문]\n" + context)
    assert "저도막" not in switched and "기포" in switched


@pytest.mark.parametrize("origin,advice,prefix", [
    ("저도막이 생기는 원인은?", "블라스팅 작업에서 도막 손상이 생기는 원인은?", "선박 도장 스프레이 건 작업"),
    ("블라스팅 노즐에서 분진이 많이 생기는 원인은?", "스프레이 도장에서 손상이 생기는 원인은?", "선박 블라스팅 노즐 작업"),
])
def test_resolution_advice_cannot_choose_the_retrieval_process(origin, advice, prefix):
    from trainer.demo_api.query_runtime import retrieval_question
    store = SequenceStore(":memory:")
    previous(store, prompt=origin)
    pipeline = DialoguePipeline({"kind": "coating_knowledge", "intent": "knowledge", "scope": "unspecified",
                                 "context_relation": "followup", "resolved_question": advice})
    ask(client(pipeline, store), "이격거리는 적절했는데?")
    call = pipeline.calls[-1]
    query = retrieval_question(call["retrieval_instruction"], context=call["instruction"])
    assert query.startswith(prefix), query


def test_new_topic_or_an_explicit_current_process_does_not_inherit_an_old_process():
    from trainer.demo_api.query_runtime import retrieval_question
    context = "[문서 후속 질문]\n[최근 대화]\n사용자: 블라스팅 분진 원인은?"
    assert retrieval_question("스프레이 이격거리는?", context=context).startswith("선박 도장 스프레이")
    assert retrieval_question("기포 원인은?", context="[새 주제 질문]\n" + context).startswith("선박 도장 스프레이")


def test_foreign_previous_answer_is_preserved_for_a_korean_followup():
    store = SequenceStore(":memory:")
    previous(store, prompt="Why is the coating thin?", answer="Lớp sơn mỏng có thể liên quan đến khoảng cách phun.")
    pipeline = DialoguePipeline()
    ask(client(pipeline, store), "이격거리는 적절했는데?")
    instruction = pipeline.calls[-1]["generation_instruction"]
    assert "Why is the coating thin?" in instruction
    assert "Lớp sơn mỏng" in instruction and "미검증" in instruction
    assert instruction.endswith("이격거리는 적절했는데?")


def test_generation_trace_is_stored_privately_and_omits_internal_prompts():
    class TracedPipeline(DialoguePipeline):
        def run(self, **kwargs):
            result = super().run(**kwargs)
            result["metadata"] = {"answer_trace": {
                "policy_version": "context-v1", "review": {"verdict": "accepted"},
                "selected_evidence": [{"chunk_id": "DOC-1", "excerpt": "관련 문단"}],
                "system_prompt": "PRIVATE-SYSTEM-PROMPT",
            }}
            return result
    store = SequenceStore(":memory:")
    result = ask(client(TracedPipeline(), store), "저도막이 생기는 원인은 무엇인가요?")
    assert "answer_trace" not in result
    context = json.loads(store._conn.execute("SELECT query_context FROM query_turns ORDER BY id DESC LIMIT 1").fetchone()[0])
    trace = context["answer_trace"]
    assert trace["review"]["verdict"] == "accepted"
    assert trace["selected_evidence"][0]["chunk_id"] == "DOC-1"
    assert "PRIVATE-SYSTEM-PROMPT" not in json.dumps(trace)


@pytest.mark.parametrize("character", ["가", "😀"])
def test_generation_trace_has_a_byte_bound_and_ignores_nonfinite_metadata(character):
    from fastapi.responses import JSONResponse
    from trainer.demo_api.copilot_router import _with_answer_trace
    trace = {key: character * 10000 for key in ("policy_version", "question", "retrieval_query", "initial_answer", "final_answer")}
    trace["selected_evidence"] = [{"chunk_id": character * 1000, "excerpt": character * 10000,
                                    "sha256": character * 1000, "rerank_score": float("inf")} for _ in range(20)]
    trace["review"] = {"verdict": character * 10000, "reason": character * 10000}
    response = _with_answer_trace(JSONResponse({"ok": True}), {"metadata": {"answer_trace": trace}})
    result = response._copilot_answer_trace
    assert len(json.dumps(result, ensure_ascii=False).encode("utf-8")) <= 16384
    assert len(result["selected_evidence"]) == 4
    assert all("rerank_score" not in item for item in result["selected_evidence"])


def test_semantic_safety_followups_keep_the_original_hazard_for_two_and_three_turns():
    from trainer.demo_api.safety_guard import HIGH_PRESSURE_ANSWER
    store = SequenceStore(":memory:")
    pipeline = DialoguePipeline({"kind": "chat", "intent": "knowledge", "scope": "unspecified",
                                 "context_relation": "followup", "resolved_question": ""})
    c = client(pipeline, store)
    first = ask(c, "분사되는 건 끝을 손가락으로 막아서 확인해도 돼?")
    assert first["answer"] == HIGH_PRESSURE_ANSWER and pipeline.plans == []
    second = ask(c, "압력을 낮췄는데?")
    assert second["answer"] == HIGH_PRESSURE_ANSWER
    turns = store.recent_query_turns("CHAT:U", user_id="U")
    assert turns[-1]["safety_class"] == "hand_tip"
    third = ask(c, "살짝만 해도?")
    assert third["answer"] == HIGH_PRESSURE_ANSWER
    turns = store.recent_query_turns("CHAT:U", user_id="U")
    assert turns[-1]["safety_class"] == "hand_tip"
    assert pipeline.calls == [] and pipeline.conversations == []


def test_semantic_new_topic_does_not_inherit_a_previous_hazard():
    store = SequenceStore(":memory:")
    pipeline = DialoguePipeline({"kind": "chat", "intent": "knowledge", "scope": "unspecified",
                                 "context_relation": "new_topic", "resolved_question": ""})
    c = client(pipeline, store)
    ask(c, "분사되는 건 끝을 손가락으로 막아서 확인해도 돼?")
    data = ask(c, "프랑스의 수도는?")
    assert data["answer"] == "프랑스의 수도는 파리입니다."
    assert store.recent_query_turns("CHAT:U", user_id="U")[-1].get("safety_class") is None


def test_semantic_followup_reuses_the_existing_guard_for_other_hazard_classes():
    from trainer.demo_api.safety_guard import CLASSES
    class UnsafePipeline(DialoguePipeline):
        def run(self, **kwargs):
            self.calls.append(kwargs)
            return {"response": "[답변] 네, 잠깐은 괜찮습니다.", "rag_chunks": []}
    store = SequenceStore(":memory:")
    pipeline = UnsafePipeline({"kind": "chat", "intent": "knowledge", "scope": "unspecified",
                               "context_relation": "followup", "resolved_question": ""})
    c = client(pipeline, store)
    ask(c, "작업 중에 담배 피워도 돼?")
    data = ask(c, "시간은 짧은데?")
    assert data["answer"] == CLASSES["ignition"].answer
    assert store.recent_query_turns("CHAT:U", user_id="U")[-1]["safety_class"] == "ignition"
    assert pipeline.conversations == []


@pytest.mark.parametrize("verdict,degraded", [
    ("ok", False), ("rewritten", False), ("cached", False), ("not_required", False), ("attribution_corrected", False),
    ("budget_exhausted", True), ("rejected", True), ("invalid_review", True),
    ("unavailable", True), ("insufficient", True),
])
def test_semantic_review_failure_is_visible_in_existing_degraded_flag(verdict, degraded):
    class ReviewedPipeline(DialoguePipeline):
        def run(self, **kwargs):
            result = super().run(**kwargs)
            result["metadata"] = {"answer_review": {"verdict": verdict}}
            return result
    data = ask(client(ReviewedPipeline()), "저도막이 생기는 원인은 무엇인가요?")
    assert data["degraded"] is degraded


def test_detailed_search_keeps_training_terms_found_verbatim_in_the_manuals():
    from trainer.demo_api.query_runtime import retrieval_question
    for term in ("저도막", "과도막", "이격거리", "오버랩"):
        assert term in retrieval_question(f"{term}의 원인은?", expand_terms=False)
    assert "두께 미달" in retrieval_question("저도막이 뭐야?")  # 빠른 안내 기본값은 그대로
