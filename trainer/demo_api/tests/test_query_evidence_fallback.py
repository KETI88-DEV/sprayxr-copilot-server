"""Query reports actual document use and keeps contextual fallback separate from errors."""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.copilot_router import create_copilot_router
from trainer.demo_api.sequence_store import SequenceStore


DOC = "DOC-doc_0007-C0002"
PROMPT = "도막이 얇아지는 원인은 무엇인가요?"
PUBLIC_KEYS = {
    "request_id", "session_id", "user_id", "scenario_id", "generated_at", "latency_ms",
    "degraded", "answer", "resolved_intents", "context_status",
    "sequence_context", "coating_context", "pose_context", "retrieval_used", "citations",
}


class Pipeline:
    config = {"confidence": {"abstain_below_rerank_score": 4.0, "quick_abstain_below_rerank_score": 2.0}}

    def __init__(self, *, missing=False, response=None, error=None, verdict="not_required"):
        self.missing, self.error, self.verdict = missing, error, verdict
        self.response = response if response is not None else "[답변] 작업 속도와 도료 공급 상태를 확인하세요.\n[근거] " + DOC
        self.runs, self.conversations = [], []

    def plan_copilot_dialogue(self, prompt, **kwargs):
        return {"kind": "coating_knowledge", "intent": "knowledge", "scope": "unspecified"}

    def run(self, *, defer_without_evidence=False, evidence_score_threshold=None, **kwargs):
        self.runs.append({**kwargs, "defer_without_evidence": defer_without_evidence,
                          "evidence_score_threshold": evidence_score_threshold})
        if self.error:
            raise self.error
        if self.missing and defer_without_evidence and kwargs.get("use_rag"):
            return {"response": "", "rag_chunks": [], "citations": [], "metadata": {
                "retrieval_decision": {"status": "no_relevant_evidence", "reason": "empty", "attempted": True},
            }}
        return {"response": self.response, "rag_chunks": [
            {"chunk_id": DOC, "text": "작업 속도와 도료 공급 상태는 도막 형성에 영향을 줍니다.", "rerank_score": 6.0}],
            "metadata": {"answer_review": {"verdict": self.verdict}}}

    def converse_copilot(self, prompt, **kwargs):
        self.conversations.append({"prompt": prompt, **kwargs})
        return "일반적으로 작업 속도와 도료 공급 상태를 함께 확인하세요."


def client(pipeline, store=None):
    app = FastAPI()
    app.include_router(create_copilot_router(lambda: pipeline, store=store or SequenceStore(":memory:")))
    return TestClient(app)


def ask(c, *, prompt=PROMPT, mode="detailed", user="COPILOT-AI-TEST", conversation="fallback-A", **kwargs):
    return c.post("/api/v1/copilot/query", headers={"X-Copilot-Conversation": conversation}, json={
        "user_id": user, "prompt": prompt, "task_type": "knowledge_search", "response_mode": mode, **kwargs,
    })


@pytest.mark.parametrize("mode,threshold", [("quick", 2.0), ("detailed", 4.0)])
def test_missing_document_evidence_uses_contextual_llm_without_new_body_fields(mode, threshold):
    pipeline = Pipeline(missing=True)
    response = ask(client(pipeline), mode=mode)
    data = response.json()["data"]
    assert response.status_code == 200
    assert "작업 속도" in data["answer"] and "근거를 찾지 못해" not in data["answer"]
    assert data["retrieval_used"] is False and data["citations"] == [] and data["degraded"] is False
    assert response.headers["X-Copilot-Answer-Source"] == "conversation"
    assert set(data) == PUBLIC_KEYS
    assert pipeline.runs[0]["defer_without_evidence"] is True
    assert pipeline.runs[0]["evidence_score_threshold"] == threshold
    assert pipeline.conversations[0]["prompt"] == PROMPT
    assert pipeline.conversations[0]["response_mode"] == mode


def test_fallback_keeps_user_and_window_scoped_dialogue():
    pipeline = Pipeline(missing=True)
    c = client(pipeline)
    ask(c)
    ask(c, prompt="이격거리는 적절했는데?")
    history = pipeline.conversations[-1]["recent_turns"]
    assert history[-1]["prompt"] == PROMPT
    ask(c, prompt="이격거리는 적절했는데?", conversation="fallback-B")
    assert pipeline.conversations[-1]["recent_turns"] == []
    ask(c, prompt="이격거리는 적절했는데?", user="OTHER-TEST")
    assert pipeline.conversations[-1]["recent_turns"] == []


def test_supported_document_citation_is_the_only_document_source_signal():
    pipeline = Pipeline()
    response = ask(client(pipeline))
    data = response.json()["data"]
    assert data["retrieval_used"] is True and data["citations"] == [DOC]
    assert response.headers["X-Copilot-Answer-Source"] == "documents"
    assert pipeline.conversations == []


@pytest.mark.parametrize("response", [
    "[답변] 일반 설명입니다.",
    "[답변] 일반 설명입니다.\n[근거] DOC-invented-C0001",
])
def test_search_attempt_alone_does_not_mark_the_answer_as_document_supported(response):
    pipeline = Pipeline(response=response)
    result = ask(client(pipeline))
    data = result.json()["data"]
    assert data["retrieval_used"] is False and data["citations"] == []
    assert result.headers["X-Copilot-Answer-Source"] == "conversation"
    assert pipeline.conversations == []


def test_search_or_generation_failure_does_not_masquerade_as_general_knowledge():
    pipeline = Pipeline(error=RuntimeError("retrieval backend down"))
    response = ask(client(pipeline))
    assert response.status_code == 500
    assert response.json()["data"] is None and pipeline.conversations == []


@pytest.mark.parametrize("verdict", ["unavailable", "invalid_review", "invalid_support", "rejected", "budget_exhausted"])
def test_semantic_review_failure_does_not_trigger_unreviewed_conversation(verdict):
    pipeline = Pipeline(response="[답변] 답변 근거를 충분히 확인하지 못했습니다.", verdict=verdict)
    response = ask(client(pipeline))
    assert response.json()["data"]["degraded"] is True
    assert response.json()["data"]["retrieval_used"] is False
    assert pipeline.conversations == []


@pytest.mark.parametrize("mode", ["quick", "detailed"])
def test_valid_review_insufficient_falls_back_even_with_high_rerank_score(mode):
    pipeline = Pipeline(response="[답변] 질문에 맞는 설명을 충분히 확인하지 못했습니다.", verdict="insufficient")
    response = ask(client(pipeline), mode=mode)
    data = response.json()["data"]
    assert response.status_code == 200 and data["degraded"] is False
    assert "작업 속도" in data["answer"]
    assert data["retrieval_used"] is False and data["citations"] == []
    assert response.headers["X-Copilot-Answer-Source"] == "conversation"
    assert len(pipeline.runs) == 1 and len(pipeline.conversations) == 1
    assert pipeline.conversations[0]["prompt"] == PROMPT


def test_review_insufficient_never_bypasses_the_safety_document_path():
    pipeline = Pipeline(response="[답변] 질문에 맞는 설명을 충분히 확인하지 못했습니다.", verdict="insufficient")
    response = ask(client(pipeline), prompt="에어리스 건 팁을 손으로 막아도 되나요?")
    assert pipeline.conversations == []
    assert "막" in response.json()["data"]["answer"]
    assert response.json()["data"]["retrieval_used"] is False


def test_review_insufficient_without_rag_is_not_a_general_fallback_signal():
    pipeline = Pipeline(response="[답변] 질문에 맞는 설명을 충분히 확인하지 못했습니다.", verdict="insufficient")
    response = ask(client(pipeline), use_rag="N")
    assert response.json()["data"]["degraded"] is True
    assert pipeline.conversations == []


def test_review_insufficient_preserves_verified_mixed_record_facts():
    from trainer.demo_api.tests.viewer_record import AT_35S, context, payload

    pipeline = Pipeline(response="[답변] 질문에 맞는 설명을 충분히 확인하지 못했습니다.", verdict="insufficient")
    response = ask(client(pipeline), prompt="순서 준수율 45%는 어떻게 나온 거고, 숙련자들끼리는 순서가 얼마나 일치해?",
                   task_type="trajectory_recomm", context=context("trajectory_recomm", payload(), AT_35S))
    data = response.json()["data"]
    assert "45.1%" in data["answer"] and data["degraded"] is True
    assert data["retrieval_used"] is False and data["citations"] == []
    assert pipeline.conversations == []


@pytest.mark.parametrize("missing,verdict,reason", [
    (True, "not_required", "retrieval_empty"),
    (False, "insufficient", "review_insufficient"),
])
def test_private_diagnostics_identify_the_exact_general_fallback_reason(missing, verdict, reason):
    import json

    store = SequenceStore(":memory:")
    pipeline = Pipeline(missing=missing, verdict=verdict)
    response = ask(client(pipeline, store))
    assert response.status_code == 200
    assert set(response.json()["data"]) == PUBLIC_KEYS
    row = store._conn.execute("SELECT query_context FROM query_turns ORDER BY id DESC LIMIT 1").fetchone()
    assert json.loads(row[0])["answer_trace"]["fallback_reason"] == reason


def test_weak_definition_evidence_does_not_hide_a_review_outage():
    class Weak(Pipeline):
        def run(self, *, defer_without_evidence=False, evidence_score_threshold=None, **kwargs):
            result = super().run(defer_without_evidence=defer_without_evidence,
                                 evidence_score_threshold=evidence_score_threshold, **kwargs)
            result["rag_chunks"][0]["rerank_score"] = 1.0
            return result

    pipeline = Weak(response="[답변] 질문에 맞는 설명을 충분히 확인하지 못했습니다.", verdict="unavailable")
    response = ask(client(pipeline), prompt="새깅이 뭐지?")
    assert response.json()["data"]["degraded"] is True
    assert pipeline.conversations == []


def test_explicit_no_rag_keeps_override_and_removes_even_model_supplied_citations():
    pipeline = Pipeline(missing=True)
    data = ask(client(pipeline), use_rag="N").json()["data"]
    assert pipeline.runs[0]["use_rag"] is False
    assert pipeline.runs[0]["defer_without_evidence"] is False
    assert data["retrieval_used"] is False and data["citations"] == []
    assert pipeline.conversations == []


def test_safety_question_cannot_use_the_ordinary_no_evidence_fallback():
    pipeline = Pipeline(missing=True)
    response = ask(client(pipeline), prompt="에어리스 건 팁을 손으로 막아도 되나요?")
    assert pipeline.conversations == []
    assert "막" in response.json()["data"]["answer"]
    assert not any(call["defer_without_evidence"] for call in pipeline.runs)


def test_legacy_pipeline_signature_is_not_given_new_opt_in_arguments():
    class Legacy(Pipeline):
        def run(self, *, instruction, answer_question, force_route, use_pipeline_history, session_id,
                enable_thinking, use_rag, max_tokens, compact_analysis, request_id, retrieval_instruction):
            return super().run(instruction=instruction, use_rag=use_rag)

    pipeline = Legacy()
    response = ask(client(pipeline))
    assert response.status_code == 200 and response.json()["data"]["retrieval_used"] is True


@pytest.mark.parametrize("mode", ["quick", "detailed"])
def test_fallback_keeps_the_selected_generator_backend(mode):
    from trainer.demo_api.query_runtime import QuickQueryRuntime

    pipeline = Pipeline(missing=True)
    backend = object()
    pipeline.quick_query_runtime = QuickQueryRuntime(backend, None, 8192)
    response = ask(client(pipeline), mode=mode)
    assert response.status_code == 200
    assert (pipeline.conversations[0].get("inference_backend") is backend) is (mode == "quick")


def test_failed_review_citations_are_not_presented_as_supported_documents():
    pipeline = Pipeline(verdict="rejected")
    response = ask(client(pipeline))
    assert response.json()["data"]["retrieval_used"] is False
    assert response.json()["data"]["citations"] == []
    assert pipeline.conversations == []


def test_verified_record_lookup_keeps_its_deterministic_analysis_source():
    from trainer.demo_api.tests.viewer_record import AT_3S, context, payload

    pipeline = Pipeline(missing=True)
    response = ask(client(pipeline), prompt="이 Zone 속도랑 이격거리 숙련자랑 비교해줘.",
                   context=context("knowledge_search", payload(), AT_3S))
    data = response.json()["data"]
    assert response.headers["X-Copilot-Answer-Source"] == "analysis"
    assert "251 mm" in data["answer"]
    assert pipeline.runs == [] and pipeline.conversations == []


def test_missing_evidence_never_discards_the_verified_part_of_a_mixed_question():
    from trainer.demo_api.copilot_router import MIXED_NO_EVIDENCE
    from trainer.demo_api.tests.viewer_record import AT_35S, context, payload

    class Mixed(Pipeline):
        def run(self, *, defer_without_evidence=False, evidence_score_threshold=None, **kwargs):
            self.runs.append({"defer_without_evidence": defer_without_evidence, **kwargs})
            return {"response": "", "rag_chunks": [], "metadata": {}}

    pipeline = Mixed()
    response = ask(client(pipeline), prompt="순서 준수율 45%는 어떻게 나온 거고, 숙련자들끼리는 순서가 얼마나 일치해?",
                   task_type="trajectory_recomm", context=context("trajectory_recomm", payload(), AT_35S))
    data = response.json()["data"]
    assert "45.1%" in data["answer"] and MIXED_NO_EVIDENCE in data["answer"]
    assert data["retrieval_used"] is False and data["citations"] == []
    assert pipeline.conversations == []
    assert pipeline.runs and pipeline.runs[0]["defer_without_evidence"] is False


def signed_thickness_context():
    import json
    from trainer.demo_api.viewer_context import sign_context

    claims = {"record_id": "fallback-signed-record", "thickness": "12 µm"}
    return "motion_viewer_playback:" + json.dumps({
        "record": "스프레이 학습모드 3", "playback_context": {"map_num": "1", "zone_id": "8"},
        "dialogue_context": claims, "dialogue_signature": sign_context("dialogue", claims),
    })


class ThicknessEchoPipeline(Pipeline):
    def converse_copilot(self, prompt, **kwargs):
        self.conversations.append({"prompt": prompt, **kwargs})
        return "현재 도막은 12 µm입니다. 저도막은 도막이 필요한 수준보다 얇은 상태입니다."


@pytest.mark.parametrize("missing,verdict", [(True, "not_required"), (False, "insufficient")])
def test_quick_general_fallback_excludes_signed_measurements_from_generation_and_validation(missing, verdict):
    import json

    pipeline = ThicknessEchoPipeline(missing=missing, verdict=verdict)
    response = ask(client(pipeline), prompt="저도막이란 무엇인가요?", mode="quick", context=signed_thickness_context())
    data = response.json()["data"]
    assert response.status_code == 200 and pipeline.conversations
    assert "12" not in json.dumps(pipeline.conversations[0]["facts"], ensure_ascii=False)
    assert "12" not in data["answer"] and "저도막" in data["answer"]
    assert data["retrieval_used"] is False and data["citations"] == []


def test_detailed_general_fallback_keeps_the_existing_signed_measurement_scope():
    import json

    pipeline = ThicknessEchoPipeline(missing=True)
    response = ask(client(pipeline), prompt="저도막이란 무엇인가요?", mode="detailed", context=signed_thickness_context())
    assert response.status_code == 200
    assert "12 µm" in json.dumps(pipeline.conversations[0]["facts"], ensure_ascii=False)
    assert "12 µm" in response.json()["data"]["answer"]


def test_metadata_only_definition_fallback_drops_irrelevant_playback_history():
    import json

    pipeline = Pipeline(missing=True)
    store = SequenceStore(":memory:")
    store.record_query_turn(session_key="CHAT:COPILOT-AI-TEST", user_id="COPILOT-AI-TEST",
                            prompt="현재 도막은?", answer="현재 도막은 12 µm입니다.", task_type="knowledge_search",
                            created_at="2026-11-17T01:00:00Z", answer_source="analysis", conversation_key="fallback-A")
    metadata = "motion_viewer_playback:" + json.dumps({"record": "스프레이 학습모드 3",
                                                       "playback_context": {"map_num": "1", "zone_id": "8"}})
    response = ask(client(pipeline, store), prompt="저도막이란 무엇인가요?", mode="quick", context=metadata)
    assert response.status_code == 200
    assert pipeline.conversations[0]["recent_turns"] == []


@pytest.mark.parametrize("prompt", [
    "면접에서 긴장하지 않으려면 어떻게 하면 좋을까?",
    "새로운 언어를 꾸준히 공부하려면 어떻게 하면 좋을까?",
])
def test_semantic_general_chat_is_not_redirected_to_documents_by_question_shape(prompt):
    class General(Pipeline):
        def plan_copilot_dialogue(self, prompt, **kwargs):
            return {"kind": "chat", "intent": "knowledge", "scope": "unspecified"}

    pipeline = General()
    response = ask(client(pipeline), prompt=prompt, mode="quick")
    assert response.status_code == 200 and response.json()["data"]["degraded"] is False
    assert pipeline.runs == [] and len(pipeline.conversations) == 1
    assert pipeline.conversations[0]["prompt"] == prompt
    assert response.headers["X-Copilot-Answer-Source"] == "conversation"


def test_positive_painting_knowledge_still_uses_documents_when_planner_would_say_chat():
    class MistakenPlanner(Pipeline):
        def plan_copilot_dialogue(self, prompt, **kwargs):
            return {"kind": "chat", "intent": "knowledge", "scope": "unspecified"}

    pipeline = MistakenPlanner()
    response = ask(client(pipeline), prompt="도막 기포를 예방하려면 어떻게 해야 하나요?", mode="quick")
    assert response.status_code == 200 and len(pipeline.runs) == 1
    assert pipeline.conversations == []
    assert response.headers["X-Copilot-Answer-Source"] == "documents"


@pytest.mark.parametrize("prompt", ["훈련모드는 몇 가지가 있어?", "학습 모드는 총 몇 개야?"])
def test_confirmed_simulator_mode_configuration_does_not_require_ncs_document_support(prompt):
    from trainer.demo_api.simulator_context import SIMULATOR_CONTEXT

    class Modes(Pipeline):
        def __init__(self):
            super().__init__()
            self.plans = []

        def plan_copilot_dialogue(self, prompt, **kwargs):
            self.plans.append(prompt)
            return super().plan_copilot_dialogue(prompt, **kwargs)

        def converse_copilot(self, prompt, **kwargs):
            self.conversations.append({"prompt": prompt, **kwargs})
            return "훈련모드는 1~10번, 총 10개입니다."

    pipeline = Modes()
    response = ask(client(pipeline), prompt=prompt, mode="quick")
    data = response.json()["data"]
    assert response.status_code == 200 and data["degraded"] is False
    assert pipeline.runs == [] and pipeline.plans == []
    assert pipeline.conversations[0]["simulator_context"] == SIMULATOR_CONTEXT
    assert pipeline.conversations[0]["facts"]["simulator_configuration"]["training_mode_ids"] == list(range(1, 11))
    assert "1~10" in data["answer"] and data["retrieval_used"] is False and data["citations"] == []
    assert response.headers["X-Copilot-Answer-Source"] == "conversation"


def test_mode_number_in_a_coating_technique_question_is_not_a_mode_catalog_question():
    pipeline = Pipeline()
    response = ask(client(pipeline), prompt="훈련모드 3에서 도막 기포를 방지하려면 어떻게 해야 하나요?", mode="quick")
    assert response.status_code == 200 and len(pipeline.runs) == 1
    assert pipeline.conversations == []


@pytest.mark.parametrize("verdict", ["unavailable", "invalid_review", "invalid_support", "rejected", "budget_exhausted"])
def test_a_withheld_review_failure_is_not_labelled_as_a_successful_conversation(verdict):
    pipeline = Pipeline(response="[답변] 제공된 자료만으로는 질문에 맞는 설명을 충분히 확인하지 못했습니다.", verdict=verdict)
    response = ask(client(pipeline), mode="quick")
    assert response.json()["data"]["degraded"] is True and pipeline.conversations == []
    assert response.headers["X-Copilot-Answer-Source"] == "analysis"


@pytest.mark.parametrize("prompt", [
    "헤그만 게이지 눈금 6은 입자 크기가 몇 µm를 뜻하나요?",
    "크실렌의 LD50 값은 얼마인가요?",
    "SVR이 뭐야?",
])
def test_positive_technical_requests_do_not_skip_evidence_when_planner_says_chat(prompt):
    class Technical(Pipeline):
        def plan_copilot_dialogue(self, prompt, **kwargs):
            return {"kind": "chat", "intent": "knowledge", "scope": "unspecified"}

    pipeline = Technical()
    response = ask(client(pipeline), prompt=prompt, mode="quick")
    assert response.status_code == 200
    assert len(pipeline.runs) == 1 and pipeline.conversations == []


@pytest.mark.parametrize("prompt", ["이 기록은 훈련모드 몇 번이야?", "현재 학습모드 번호는 뭐야?"])
def test_selected_mode_questions_are_not_the_global_simulator_catalog(prompt):
    from trainer.demo_api.simulator_context import simulator_configuration_facts

    assert simulator_configuration_facts(prompt) is None


def test_selected_record_mode_retains_its_signed_facts_instead_of_catalog_facts():
    import json
    from trainer.demo_api.viewer_context import sign_context

    claims = {"record_id": "selected-training-record", "scenario_id": "3"}
    selected = "motion_viewer_playback:" + json.dumps({"record": "스프레이 학습모드 3", "playback_context": {},
        "dialogue_context": claims, "dialogue_signature": sign_context("dialogue", claims)})
    pipeline = Pipeline(missing=True)
    response = ask(client(pipeline), prompt="이 기록은 훈련모드 몇 번이야?", mode="quick", context=selected)
    assert response.status_code == 200 and pipeline.conversations
    assert pipeline.conversations[0]["facts"]["scenario_id"] == "3"
    assert "simulator_configuration" not in pipeline.conversations[0]["facts"]
