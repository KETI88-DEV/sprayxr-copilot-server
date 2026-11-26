"""Summary presentation is independent of model routing and preserves guarded facts."""
from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.answer_feedback import AnswerFeedbackStore, QueryRating
from trainer.demo_api.copilot_contracts import CopilotQueryRequest
from trainer.demo_api.copilot_router import create_copilot_router, install_copilot_error_handlers
from trainer.demo_api.query_runtime import QuickQueryRuntime
from trainer.demo_api.safety_guard import HIGH_PRESSURE_ANSWER
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.tests.test_voice_router import _Stt, _client as voice_client, _post as voice_post
from trainer.demo_api.voice_router import TEST_TEXT_QUERY_PATH, TEST_VOICE_QUERY_PATH

CORE = "표면의 먼지를 먼저 제거하세요."
EXPLANATION = "이 과정은 도료가 표면에 고르게 부착되도록 준비하는 단계입니다."
EXTRA = "작업 결과를 살펴보고 도막의 외관도 점검하세요."
BODY = "\n".join((CORE, EXPLANATION, EXTRA))
QUESTION = "도장 표면 준비 방법을 설명해 주세요."


class Documents:
    def __init__(self, body=BODY):
        self.calls = []
        self.model = SimpleNamespace(name="27B")
        self.quick = SimpleNamespace(name="8B")
        self.quick_query_runtime = QuickQueryRuntime(self.quick, object(), 8192)
        # A shared cached result must remain full across presentation choices.
        self.result = {"response": "[답변] " + body + "\n[근거] DOC-a-C0001",
                       "rag_chunks": [{"chunk_id": "DOC-a-C0001", "text": body}]}

    def run(self, **kwargs):
        self.calls.append(kwargs)
        # Simulate the existing first generation following its conditional
        # style, rather than expecting a postprocessor to delete the answer.
        instruction = kwargs.get("generation_instruction", kwargs["instruction"])
        if self.result["response"] == "[답변] " + BODY + "\n[근거] DOC-a-C0001" and "핵심 요약" in instruction:
            return {**self.result, "response": "[답변] " + CORE + "\n[근거] DOC-a-C0001"}
        return self.result


def client(pipeline, store=None):
    app = FastAPI()
    app.include_router(create_copilot_router(lambda: pipeline, store=store or SequenceStore(":memory:")))
    install_copilot_error_handlers(app)
    return TestClient(app)


def request(**changes):
    return {"user_id": "COPILOT-AI-TEST", "prompt": QUESTION,
            "task_type": "knowledge_search", "use_rag": "Y", **changes}


def test_query_contract_defaults_to_summary_without_changing_model_default():
    query = CopilotQueryRequest(**request())
    assert query.summarize is True
    assert query.response_mode == "quick"
    schema = CopilotQueryRequest.model_json_schema()
    assert schema["properties"]["summarize"]["type"] == "boolean"
    assert schema["properties"]["summarize"]["default"] is True


@pytest.mark.parametrize("mode", ["quick", "detailed"])
@pytest.mark.parametrize("summary", [True, False])
def test_generation_focus_is_independent_of_model_and_summary_mode(mode, summary):
    from trainer.demo_api.copilot_session_context import build_query_instruction
    from trainer.demo_api.dialogue import conversation_messages
    from trainer.scripts.answer_semantics import answer_system_prompt

    prompts = [
        build_query_instruction(QUESTION, client_context=None, session_summary="", recent_turns=[],
                                response_mode=mode, summarize=summary),
        answer_system_prompt(detailed=mode == "detailed", has_evidence=True, summarize=summary),
        *(conversation_messages(QUESTION, recent_turns=[], facts={}, response_mode=mode, summarize=summary,
                                knowledge_fallback=fallback)[0]["content"] for fallback in (True, False)),
    ]
    for prompt in prompts:
        assert "묻지 않은 관련 지표" in prompt and "결론을 반복" in prompt
        assert "필요한 안전 안내" in prompt and "답할 수 없는 구체적인 이유" in prompt


@pytest.mark.parametrize("mode", ["quick", "detailed"])
@pytest.mark.parametrize("summary", [None, True, False])
def test_public_query_summary_and_model_selection_are_independent(mode, summary):
    pipeline = Documents()
    payload = request(response_mode=mode)
    if summary is not None:
        payload["summarize"] = summary
    response = client(pipeline).post("/api/v1/copilot/query", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["answer"] == (BODY if summary is False else CORE)
    assert data["citations"] == ["DOC-a-C0001"] and data["retrieval_used"] is True
    assert response.headers["x-copilot-answer-source"] == "documents"
    if mode == "quick":
        assert pipeline.calls[0]["inference_backend"] is pipeline.quick
    else:
        assert "inference_backend" not in pipeline.calls[0]
    assert pipeline.model.name == "27B"


def test_summary_retains_late_warning_numeric_units_whole_zone_order_and_limit():
    numeric = "이격거리는 300 mm를 유지하세요."
    order = "Zone 순서는 7→4→8→5→6→3→2→1→9입니다."
    warning = "어지럽거나 숨이 차면 작업을 중단하고 안전한 곳으로 이동하세요."
    limit = "이 값은 대표 통계이며 현재 순간의 실측값은 아닙니다."
    body = "\n".join((CORE, EXPLANATION, numeric, order, warning, limit))
    response = client(Documents(body)).post("/api/v1/copilot/query", json=request())
    assert response.status_code == 200
    answer = response.json()["data"]["answer"]
    for essential in (CORE, numeric, order, warning, limit):
        assert essential in answer
    assert EXPLANATION in answer
    assert "…" not in answer and "(9개)" not in answer


def test_summary_never_cuts_a_long_sentence_or_a_decimal_measurement():
    core = "표면에 남아 있는 먼지와 이물질을 제거한 뒤 제조사가 지정한 전처리 상태를 충분히 확인하고 작업을 진행하세요."
    numeric = "확인된 평균 두께는 187.5 µm입니다."
    assert len(core) > 60
    response = client(Documents("\n".join((core, EXPLANATION, numeric)))).post(
        "/api/v1/copilot/query", json=request())
    assert response.status_code == 200
    answer = response.json()["data"]["answer"]
    assert core in answer and numeric in answer
    assert EXPLANATION in answer


@pytest.mark.parametrize("summary", [True, False])
def test_fixed_high_pressure_safety_answer_is_not_reduced(summary):
    response = client(Documents()).post("/api/v1/copilot/query", json=request(
        prompt="에어리스 스프레이건 팁을 손으로 막아도 되나요?", summarize=summary))
    assert response.status_code == 200
    assert response.json()["data"]["answer"] == HIGH_PRESSURE_ANSWER


@pytest.mark.parametrize("mode", ["quick", "detailed"])
@pytest.mark.parametrize("summary", [None, "true", "false"])
def test_voice_form_propagates_summary_default_and_false_for_both_models(mode, summary):
    pipeline = Documents()
    fields = {"response_mode": mode}
    if summary is not None:
        fields["summarize"] = summary
    response = voice_post(voice_client(_Stt(QUESTION), pipeline), **fields)
    assert response.status_code == 200, response.text
    assert response.json()["data"]["answer"] == (BODY if summary == "false" else CORE)


@pytest.mark.parametrize("summary", [None, "true", "false"])
def test_public_text_test_defaults_and_reports_the_summary_choice(summary):
    params = {"prompt": QUESTION}
    if summary is not None:
        params["summarize"] = summary
    response = voice_client(_Stt(), Documents()).post(TEST_TEXT_QUERY_PATH, params=params)
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["answer"] == (BODY if summary == "false" else CORE)
    assert data["test_context"]["input_used"]["query_parameters"]["summarize"] is (summary != "false")


@pytest.mark.parametrize("choices", [(True, False), (False, True)])
def test_shared_full_result_is_presented_per_request_and_history_feedback_keep_choice(tmp_path, choices):
    path = tmp_path / "query.db"
    sequences = SequenceStore(str(path))
    pipeline = Documents()
    api = client(pipeline, sequences)
    answers, turn_ids = [], []
    for summary in choices:
        response = api.post("/api/v1/copilot/query", json=request(summarize=summary),
                            headers={"X-Copilot-Conversation": "same-conversation"})
        assert response.status_code == 200, response.text
        answers.append(response.json()["data"]["answer"])
        turn_ids.append(int(response.headers["x-copilot-query-turn"]))
    assert answers == [CORE if summary else BODY for summary in choices]
    assert pipeline.result["response"] == "[답변] " + BODY + "\n[근거] DOC-a-C0001"
    assert [row["answer"] for row in sequences.recent_query_turns(
        "CHAT:COPILOT-AI-TEST", user_id="COPILOT-AI-TEST", conversation_key="same-conversation")] == answers
    feedback = AnswerFeedbackStore(str(path), "main")
    for turn in turn_ids:
        feedback.submit(QueryRating(submission_id=str(uuid4()), query_turn_id=turn, rating="up"),
                        {"caller_source": "assistant-test", "client_ip": "127.0.0.1"})
    with sqlite3.connect(path) as connection:
        rows = connection.execute("SELECT answer,query_context FROM query_answer_feedback ORDER BY query_turn_id").fetchall()
    assert [row[0] for row in rows] == answers
    assert [json.loads(row[1])["summarize"] for row in rows] == list(choices)


def test_summary_is_applied_before_answer_translation(monkeypatch):
    import trainer.demo_api.copilot_router as router
    seen = []
    def translate(_pipeline, text, **kwargs):
        seen.append(text)
        return "Remove surface dust first."
    monkeypatch.setattr(router, "translate_text", translate)
    response = client(Documents()).post("/api/v1/copilot/query", json=request(response_language="en"))
    assert response.status_code == 200
    assert seen == [CORE]
    assert response.json()["data"]["answer"] == "Remove surface dust first."


@pytest.mark.parametrize("summary", [True, False])
def test_general_conversation_uses_the_same_final_summary_gate(summary):
    from trainer.demo_api.tests.viewer_record import RecordingPipeline
    core = "발표 전에 핵심 내용을 먼저 정리하세요."
    body = core + "\n전달할 내용을 실제로 말하면서 연습해 보세요."
    class Conversation(RecordingPipeline):
        def converse_copilot(self, prompt, **kwargs):
            return core if kwargs.get("summarize") else body
    pipeline = Conversation(plan={"kind": "chat", "intent": "knowledge", "scope": "unspecified"}, converse=body)
    response = client(pipeline).post("/api/v1/copilot/query", json=request(
        prompt="회의 발표를 준비하는 방법을 설명해 주세요.", summarize=summary))
    assert response.status_code == 200
    assert response.json()["data"]["answer"] == (core if summary else body)
    assert response.headers["x-copilot-answer-source"] == "conversation"


def test_explicit_sentence_translation_keeps_the_whole_text_even_when_summary_is_true(monkeypatch):
    import trainer.demo_api.copilot_router as router
    source = "표면을 살펴보세요. 작업 준비를 시작하세요."
    translated = "Inspect the surface. Begin preparing the work."
    seen = []
    def translate(_pipeline, text, **kwargs):
        seen.append(text)
        return translated
    monkeypatch.setattr(router, "translate_text", translate)
    response = client(Documents()).post("/api/v1/copilot/query", json=request(
        prompt='"' + source + '"를 영어로 번역해줘', summarize=True))
    assert response.status_code == 200
    assert seen == [source]
    assert response.json()["data"]["answer"] == translated
    assert response.json()["data"]["citations"] == []


def test_signed_record_summary_keeps_zone_findings_and_absence_claims():
    from trainer.demo_api.tests.viewer_record import ask
    answer = ask("도막이 과하거나 모자란 영역이 있어?", tab="coating", summarize=True)["answer"]
    for phrase in ("저도막: 8개 —", "과도막: 없음", "Zone 9"):
        assert phrase in answer


def test_voice_and_public_test_openapi_expose_summary_boolean_default_true():
    schema = voice_client(_Stt(), Documents()).get("/openapi.json").json()
    voice = schema["paths"]["/api/v1/copilot/query-voice"]["post"]
    form_ref = voice["requestBody"]["content"]["multipart/form-data"]["schema"]["$ref"].split("/")[-1]
    field = schema["components"]["schemas"][form_ref]["properties"]["summarize"]
    assert field["type"] == "boolean" and field["default"] is True
    for path in (TEST_TEXT_QUERY_PATH, TEST_VOICE_QUERY_PATH):
        field = next(item for item in schema["paths"][path]["post"]["parameters"] if item["name"] == "summarize")
        assert field["schema"]["type"] == "boolean" and field["schema"]["default"] is True


def test_invalid_summary_is_rejected_before_generation_or_stt():
    pipeline = Documents()
    response = client(pipeline).post("/api/v1/copilot/query", json=request(summarize="later"))
    assert response.status_code == 422 and not pipeline.calls
    stt = _Stt()
    response = voice_post(voice_client(stt, pipeline), summarize="later")
    assert response.status_code == 422 and not stt.calls and not pipeline.calls


@pytest.mark.parametrize("question,introduction,items", [
    ("저도막이 생기는 원인을 설명해 주세요.", "저도막은 도막이 얇게 형성된 상태입니다.",
     ("건의 이동이 빠르면 도료가 충분히 쌓이기 어렵습니다.", "토출량이 적으면 형성되는 막도 얇아집니다.")),
    (QUESTION, "표면 준비는 도장 전에 진행하는 작업입니다.",
     ("먼지를 제거하세요.", "기름기를 닦으세요.", "표면을 말리세요.")),
])
def test_summary_keeps_requested_causes_or_steps_after_an_introductory_sentence(question, introduction, items):
    body = introduction + "\n" + "\n".join("- " + item for item in items) + "\n자세한 배경 설명은 작업 이해에 도움이 됩니다."
    response = client(Documents(body)).post("/api/v1/copilot/query", json=request(prompt=question))
    assert response.status_code == 200
    answer = response.json()["data"]["answer"]
    for item in items:
        assert item in answer
    assert answer == body.replace("\n", " ")


@pytest.mark.parametrize("choices", [(True, False), (False, True)])
def test_conditional_first_generation_and_cache_scope_do_not_cross_summary_modes(choices):
    class Cache(Documents):
        def __init__(self):
            super().__init__()
            self.cache = {}
        def run(self, **kwargs):
            self.calls.append(kwargs)
            summary_instruction = "핵심 요약" in kwargs["generation_instruction"]
            text = self.cache.setdefault(kwargs["quick_cache_scope"], CORE if summary_instruction else BODY)
            return {"response": "[답변] " + text + "\n[근거] DOC-a-C0001", "rag_chunks": self.result["rag_chunks"]}
    pipeline = Cache()
    api = client(pipeline)
    for summary in choices:
        response = api.post("/api/v1/copilot/query", json=request(summarize=summary, response_mode="quick"))
        assert response.status_code == 200
        assert response.json()["data"]["answer"] == (CORE if summary else BODY)
    assert len(pipeline.cache) == 2
    for summary, call in zip(choices, pipeline.calls):
        assert ("핵심 요약" in call["generation_instruction"]) is summary
        assert call.get("answer_summary", False) is summary
        scope = json.loads(call["quick_cache_scope"])
        assert scope == ["COPILOT-AI-TEST", None, "knowledge_search", True] if summary else scope == ["COPILOT-AI-TEST", None, "knowledge_search"]


def test_rag_and_conversation_first_generation_use_summary_style_without_reducing_the_ceiling():
    from trainer.scripts.answer_semantics import answer_system_prompt
    from trainer.scripts.agent_pipeline import AgentPipeline
    assert "핵심 요약" in answer_system_prompt(detailed=True, has_evidence=True, summarize=True)
    assert "요약하지" in answer_system_prompt(detailed=True, has_evidence=True)
    pipeline = object.__new__(AgentPipeline)
    calls = []
    def complete(messages, **kwargs):
        calls.append((messages, kwargs))
        return CORE
    pipeline._dialogue_completion = complete
    for summary in (True, False):
        assert pipeline.converse_copilot(QUESTION, recent_turns=[], facts={}, response_mode="detailed", summarize=summary) == CORE
    assert "핵심 요약" in calls[0][0][0]["content"] and "요약하지" not in calls[0][0][0]["content"]
    assert "요약하지" in calls[1][0][0]["content"]
    assert [call[1]["max_tokens"] for call in calls] == [1024, 1024]


@pytest.mark.parametrize("summary", [True, False])
def test_precomposed_analysis_keeps_the_required_recovery_action(summary):
    response = client(Documents()).post("/api/v1/copilot/query", json=request(
        prompt="도막 분석 결과를 알려주세요.", task_type="error_analysis", session_id="missing-report", summarize=summary))
    assert response.status_code == 200
    answer = response.json()["data"]["answer"]
    assert "신뢰 가능한 Zone별 도막 분석 자료가 없습니다." in answer
    assert "뷰어의 기록을 다시 불러오거나 세션 완료 후 도막 분석 결과를 확인해 주세요." in answer


@pytest.mark.parametrize("body", [
    "원인은 다음과 같습니다.\n건의 이동이 빠르면 도료가 충분히 쌓이기 어렵습니다.\n토출량이 적으면 형성되는 막도 얇아집니다.",
    "해당 세정제를 사용할 수 있습니다.\n방폭형 설비를 갖춘 작업장에서만 사용하세요.",
], ids=["causes_after_intro", "approval_condition"])
def test_summary_formatting_never_deletes_approved_nonbullet_causes_or_conditions(body):
    response = client(Documents(body)).post("/api/v1/copilot/query", json=request())
    assert response.status_code == 200
    assert response.json()["data"]["answer"] == body.replace("\n", " ")


@pytest.mark.parametrize("expression,result", [("2**3", "8"), ("2**3**2", "512")])
@pytest.mark.parametrize("summary", [True, False])
def test_summary_preserves_literal_exponentiation_and_backticks(expression, result, summary):
    from trainer.demo_api.tests.viewer_record import RecordingPipeline
    body = f"파이썬에서 `{expression}`은 {result}입니다."
    pipeline = RecordingPipeline(plan={"kind": "chat", "intent": "knowledge", "scope": "unspecified"}, converse=body)
    response = client(pipeline).post("/api/v1/copilot/query", json=request(
        prompt=f"파이썬에서 {expression}은 무슨 뜻인가요?", summarize=summary))
    assert response.status_code == 200
    assert response.json()["data"]["answer"] == body
