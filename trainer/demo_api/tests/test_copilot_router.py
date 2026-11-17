"""v2.5 POST /api/v1/copilot/query: knowledge_search through the live pipeline, other task types pending."""

from __future__ import annotations

import re
import threading
import time
from typing import Any

import anyio
import httpx
import pytest
from fastapi.testclient import TestClient

from trainer.demo_api import chat_service as chat_service_mod
from trainer.demo_api import copilot_router
from trainer.demo_api.app import create_app
from trainer.demo_api.quick_answer import one_line
from trainer.scripts import agent_pipeline
from trainer.scripts.openai_backend import OpenAIBackendUnavailableError

URL = "/api/v1/copilot/query"
_DATA_KEYS = {
    "request_id",
    "session_id",
    "user_id",
    "scenario_id",
    "generated_at",
    "latency_ms",
    "degraded",
    "answer",
    "resolved_intents",
    "context_status",
    "sequence_context",
    "coating_context",
    "pose_context",
    "retrieval_used",
    "citations",
}
# pose_recomm 은 이제 실제로 채점한다(명세서 p.9). 준비 중 응답은 둘만 남는다.
# trajectory_recomm 은 세션이 있으면 complete 리포트로 답한다(아래 별도 시험). 엔진이 없는 것은 error_analysis 뿐이다.
_PENDING = ["error_analysis"]
_GENERATED_UUID = "3f0c1b2a-9d8e-4c7b-a6f5-0e1d2c3b4a59"


class _FakePipeline:
    def __init__(
        self,
        *,
        response: str = "[답변] 저도막은 규정 최소 두께보다 얇은 도막이다.\n[근거] DOC-doc_0001-C0001",
        citations: tuple[str, ...] = ("DOC-doc_0001-C0001", "DOC-doc_0002-C0003"),
        sleep_sec: float = 0.0,
        exc: Exception | None = None,
    ) -> None:
        self.response = response
        self.citations = citations
        self.sleep_sec = sleep_sec
        self.exc = exc
        self.calls: list[dict[str, Any]] = []
        self._guard = threading.Lock()
        self.active = 0
        self.peak = 0

    def run(self, **kwargs: Any) -> dict[str, Any]:
        with self._guard:
            self.calls.append(kwargs)
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            if self.sleep_sec:
                time.sleep(self.sleep_sec)
            if self.exc is not None:
                raise self.exc
            return {
                "request_id": kwargs.get("request_id") or _GENERATED_UUID,
                # run() invents a session id when none is given; it must not reach the client.
                "session_id": kwargs.get("session_id") or _GENERATED_UUID,
                "route": "fast",
                "response": self.response,
                # run() returns both: ``citations`` is the evaluation-facing union it has
                # always been, ``rag_chunks`` is what search actually returned. The v2.5
                # boundary reconciles against ``rag_chunks``.
                "citations": list(self.citations),
                "tool_calls": [],
                "rag_chunks": [{"chunk_id": chunk_id, "text": ""} for chunk_id in self.citations],
            }
        finally:
            with self._guard:
                self.active -= 1


class _IntentPipeline(_FakePipeline):
    def __init__(self, intent: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.intent = intent
        self.intent_calls: list[tuple[str, str]] = []

    def classify_copilot_intent(self, task_type: str, prompt: str) -> str:
        self.intent_calls.append((task_type, prompt))
        return self.intent


def _body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "user_id": "W-017",
        "scenario_id": None,
        "session_id": None,
        "request_id": "req-query-0044",
        "prompt": "저도막이 무엇인지 설명해줘.",
        "task_type": "knowledge_search",
        "use_rag": "Y",
        "context": None,
    }
    body.update(overrides)
    return body


def _app(monkeypatch: pytest.MonkeyPatch, pipeline: object | None, *, live: bool = True) -> Any:
    monkeypatch.setenv("DEMO_LIVE_MODE", "on" if live else "off")
    # Nothing listens on the discard port: MCP is unreachable for every test here.
    monkeypatch.setenv("MCP_BASE_URL", "http://127.0.0.1:9")
    monkeypatch.delenv("LIVE_EAGER_INIT", raising=False)
    monkeypatch.setattr(chat_service_mod, "_pipeline_singleton", pipeline)
    return create_app()


def _client(monkeypatch: pytest.MonkeyPatch, pipeline: object | None, *, live: bool = True) -> TestClient:
    return TestClient(_app(monkeypatch, pipeline, live=live))


def test_knowledge_search_runs_the_pipeline_on_the_fast_route(monkeypatch: pytest.MonkeyPatch) -> None:
    """(a) Given knowledge_search, When queried, Then run() gets force_route='fast' and thinking off."""
    pipeline = _FakePipeline()

    response = _client(monkeypatch, pipeline).post(URL, json=_body())

    assert response.status_code == 200
    assert "정의 질문은 뜻만 한 문장" in pipeline.calls[0]["generation_instruction"]
    assert [{key: value for key, value in call.items() if key != "generation_instruction"} for call in pipeline.calls] == [
        {
            "instruction": (
                "[사용자 질문]\n저도막이 무엇인지 설명해줘.\n"
                "[응답 수준]\n빠른 안내: 반드시 완결된 한국어 문장 2개 이내로 답하세요. "
                "표·목록·제목·배경 설명은 쓰지 말고, 첫 문장은 핵심 판단, 두 번째 문장은 즉시 조치로 구성하세요."
            ),
            "force_route": "fast",
            "session_id": None,
            "enable_thinking": False,
            "use_rag": True,
            "max_tokens": 128,
            "request_id": "req-query-0044",
            "quick_cache_scope": '["W-017", null, "knowledge_search"]',
            "concise_answer": True,
            "use_pipeline_history": False,
        }
    ]


def test_detailed_mode_requests_evidence_and_action_from_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """상세 분석은 같은 빠른 생성 경로를 쓰되 더 깊고 구조적인 답변을 지시한다."""
    pipeline = _FakePipeline()

    response = _client(monkeypatch, pipeline).post(
        URL, json=_body(response_mode="detailed", context="현재 Zone은 3입니다.")
    )

    assert response.status_code == 200
    instruction = pipeline.calls[0]["instruction"]
    assert "상세 분석" in instruction
    assert "2문장" in instruction
    assert "근거" in instruction
    assert "권장 조치" in instruction
    assert pipeline.calls[0]["force_route"] == "fast"
    assert pipeline.calls[0]["enable_thinking"] is False
    assert pipeline.calls[0]["max_tokens"] == 128
    assert pipeline.calls[0]["compact_analysis"] is True


def test_detailed_mode_searches_documents_with_the_question_not_the_answer_directive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """응답 수준 지시문이 검색어에 섞이면 질문과 무관한 문단이 잡힌다(2026-11-11 재현, Issue #9)."""
    pipeline = _FakePipeline()

    response = _client(monkeypatch, pipeline).post(
        URL, json=_body(prompt="적정 이격거리는?", response_mode="detailed")
    )

    assert response.status_code == 200
    retrieval = pipeline.calls[0]["retrieval_instruction"]
    assert "응답 수준" not in retrieval
    assert "상세 분석" not in retrieval
    assert "이격거리" in retrieval  # 사용자 낱말을 그대로 검색한다(매뉴얼에 같은 낱말이 있음)
    assert "상세 분석" in pipeline.calls[0]["instruction"]  # 생성 지시는 그대로


def test_detailed_order_mode_explains_verified_report_without_extra_generation() -> None:
    from fastapi import FastAPI

    from trainer.demo_api.sequence_store import SequenceStore

    class _Reports:
        def latest_complete_analysis(self, _session_id: str) -> tuple[str, dict[str, object]]:
            return "RPT-ORDER", {"sequence_report": {
                "analysis_status": "available",
                "recommended_flow": ["1", "3", "2"],
                "first_visit_flow": ["1", "2", "3"],
                "evaluated_visit_count": 3,
                "on_track_visit_count": 2,
                "adherence_rate": 0.67,
                "reference_session_count": 6,
                "position_distribution": [],
            }}

    pipeline = _IntentPipeline(
        "order_reference",
        response=(
            "[답변] 전문가 6회 기준으로 1번 구역에서 시작한 뒤 3번과 2번 구역 순서가 권장됩니다. "
            "현재 수행은 두 번째 단계가 달라 다음에는 3번 구역을 먼저 진행하세요.\n근거:"
        ),
    )
    app = FastAPI()
    app.include_router(copilot_router.create_copilot_router(
        lambda: pipeline,
        store=SequenceStore(":memory:"),
        reports=_Reports(),  # type: ignore[arg-type]
    ))
    client = TestClient(app)

    quick = client.post(URL, json=_body(
        task_type="trajectory_recomm", session_id="SESSION-ORDER",
        prompt="권장 Zone 순서는?", response_mode="quick",
    )).json()["data"]
    detailed = client.post(URL, json=_body(
        task_type="trajectory_recomm", session_id="SESSION-ORDER",
        prompt="권장 Zone 순서는?", response_mode="detailed",
    )).json()["data"]

    assert quick["answer"].startswith("권장 작업 순서는 Zone 1→3→2입니다.")  # 빠른 안내는 화살표를 붙여 쓴다
    assert detailed["answer"].replace(" → ", "→") == quick["answer"]  # 같은 조회 요청은 응답 모드만으로 불필요한 통계를 더하지 않는다.
    assert "[" not in detailed["answer"]  # 제목 틀 없이 문장으로 낸다(2026-11-11 지시)
    assert "Zone 1 → 3 → 2" in detailed["answer"]
    assert "유동 구간" not in detailed["answer"] and "67%" not in detailed["answer"]
    assert detailed["sequence_context"]["adherence_rate"] == .67  # 구조화된 근거는 온전히 유지한다.
    assert not detailed["answer"].rstrip().endswith("근거:")
    assert pipeline.calls == []
    assert detailed["degraded"] is False


@pytest.mark.parametrize(
    ("task_type", "intent", "context"),
    [
        ("pose_recomm", None, {"similarity_score": 0.8, "corrections": []}),
        ("trajectory_recomm", "order_missing", {"recommended_flow": ["1", "2"], "selected_flow": ["1"]}),
        ("trajectory_recomm", "order_adherence", {"adherence_rate": 0.67}),
        ("trajectory_recomm", "order_reference", {"recommended_flow": ["1", "3", "2"], "selected_flow": ["1", "2", "3"],
                                                  "adherence_rate": 0.67, "position_distribution": []}),
        ("trajectory_recomm", "order_reference", {}),
        ("knowledge_search", None, {}),
    ],
)
def test_detailed_engine_answers_have_no_section_labels_or_canned_sentences(
    task_type: str, intent: str | None, context: dict[str, object],
) -> None:
    """상세 분석도 채점·조회 결과를 제목 틀과 고정 문장 없이 낸다(2026-11-11 지시)."""
    base = "숙련자 팔 자세 대비 유사도 80% — 숙련자 75백분위."

    answer = copilot_router._structured_detail_fallback(task_type, base, context, intent=intent)

    assert "[" not in answer
    assert "다음 분사에서도 같은 조건으로" not in answer
    assert "Zone별로 확인했습니다" not in answer
    assert "계산할 수 없습니다" not in answer
    if task_type != "trajectory_recomm" or not context:
        assert answer == base


def test_motion_viewer_client_claims_are_not_used_as_verified_analysis() -> None:
    import json
    from fastapi import FastAPI

    from trainer.demo_api.sequence_store import SequenceStore

    class _ReportsWithoutSession:
        def latest_complete_analysis(self, _session_id: str) -> None:
            return None

    pipeline = _IntentPipeline(
        "order_reference",
        response=(
            "[답변] 권장 작업 순서는 Zone 1 → 3 → 2입니다. "
            "우선 Zone 1에서 시작한 뒤 Zone 3으로 이동하세요.\n근거:"
        ),
    )
    app = FastAPI()
    app.include_router(copilot_router.create_copilot_router(
        lambda: pipeline,
        store=SequenceStore(":memory:"),
        reports=_ReportsWithoutSession(),  # type: ignore[arg-type]
    ))
    client = TestClient(app)
    viewer_context = "motion_viewer_analysis:" + json.dumps({
        "verified_answer": "전문가 대표 흐름은 1 → 3 → 2입니다.",
        "analysis_context": {
            "recommended_flow": ["1", "3", "2"],
            "selected_flow": ["1", "2", "3"],
            "adherence_rate": 0.67,
                "best_reference_similarity": 0.81,
                "position_similarity": 0.74,
                "position_distribution": [
                    [{"zone_id": "1", "probability": 0.83}],
                    [{"zone_id": "3", "probability": 0.67}],
                    [{"zone_id": "2", "probability": 0.33}],
                ],
        },
    }, ensure_ascii=False)

    quick = client.post(URL, json=_body(
        task_type="trajectory_recomm", session_id="3_1000793_1",
        prompt="권장 Zone 순서는?", response_mode="quick", context=viewer_context,
    )).json()["data"]
    detailed = client.post(URL, json=_body(
        task_type="trajectory_recomm", session_id="3_1000793_1",
        prompt="권장 Zone 순서는?", response_mode="detailed", context=viewer_context,
    )).json()["data"]

    assert quick["context_status"] == "not_available"
    assert detailed["context_status"] == "not_available"
    assert "1 → 3 → 2" not in quick["answer"]
    assert "recommended_flow" not in detailed["sequence_context"]
    assert pipeline.calls == []


def test_response_mode_defaults_to_quick_and_rejects_unknown_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quick = _client(monkeypatch, _FakePipeline()).post(URL, json=_body())
    invalid = _client(monkeypatch, _FakePipeline()).post(
        URL, json=_body(response_mode="slow")
    )

    assert quick.status_code == 200
    assert invalid.status_code == 422
    assert invalid.json() == _error(422, "Invalid request: body.response_mode (Unsupported value)")


def test_quick_mode_removes_an_unfinished_evidence_heading(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _FakePipeline(response="[답변] 표면 상태를 확인하고 해당 부위를 재도장하세요.\n근거:")

    data = _client(monkeypatch, pipeline).post(
        URL, json=_body(response_mode="quick", use_rag="N")
    ).json()["data"]

    assert data["answer"] == "표면 상태를 확인하고 해당 부위를 재도장하세요."


def test_session_query_injects_accumulated_sequences_and_recent_dialogue() -> None:
    """A follow-up question must be grounded in every prior sequence and turn of that session."""
    import json
    from fastapi import FastAPI
    from trainer.demo_api.sequence_store import SequenceStore

    store = SequenceStore(":memory:")
    for sequence_id, zone_id, speed in (("SEQ-001", "1", 420.0), ("SEQ-002", "3", 480.0)):
        body = {
            "session_id": "SESSION-CONTEXT",
            "scenario_id": "3",
            "sequence_id": sequence_id,
            "zone_id": zone_id,
            "sequence_setting": {"tool_type": "spray_gun"},
            "sequence_info": [],
            "sequence_result": {"avg_speed_mms": speed, "avg_distance_mm": 250.0},
        }
        store.record(
            session_key="SESSION-CONTEXT",
            sequence_id=sequence_id,
            body_sha256=sequence_id,
            frames=0,
            received_at="2026-10-23T00:00:00Z",
            body=json.dumps(body),
        )
    store.record_query_turn(
        session_key="SESSION-CONTEXT",
        user_id="W-017",
        prompt="방금 속도가 어땠어?",
        answer="두 번째 시퀀스에서 빨라졌습니다.",
        task_type="knowledge_search",
        created_at="2026-10-23T00:00:01Z",
    )
    pipeline = _FakePipeline(response="[답변] 앞선 맥락을 반영한 답변")
    app = FastAPI()
    app.include_router(copilot_router.create_copilot_router(lambda: pipeline, store=store))

    data = TestClient(app).post(
        URL,
        json=_body(session_id="SESSION-CONTEXT", prompt="그럼 지금은?", use_rag="N"),
    ).json()["data"]

    instruction = pipeline.calls[0]["instruction"]
    assert "그럼 지금은?" in instruction
    assert '"sequence_count":2' in instruction
    assert '"zone_visit_order":["1","3"]' in instruction
    assert "avg_speed_mms의 단위는 mm/s" in instruction
    assert "방금 속도가 어땠어?" in instruction
    assert "두 번째 시퀀스에서 빨라졌습니다." in instruction
    assert data["answer"] == "앞선 맥락을 반영한 답변"


def test_successful_session_query_is_available_to_the_next_follow_up() -> None:
    """The server persists completed turns so a later request can resolve pronouns naturally."""
    from fastapi import FastAPI
    from trainer.demo_api.sequence_store import SequenceStore

    store = SequenceStore(":memory:")
    pipeline = _FakePipeline(response="[답변] 첫 답변")
    app = FastAPI()
    app.include_router(copilot_router.create_copilot_router(lambda: pipeline, store=store))
    client = TestClient(app)

    first = client.post(URL, json=_body(session_id="S-TURNS", prompt="권장 자세는?", use_rag="N"))
    assert first.status_code == 200
    pipeline.response = "[답변] 문맥을 이은 두 번째 답변"
    second = client.post(URL, json=_body(session_id="S-TURNS", prompt="왜 그래?", use_rag="N"))

    assert second.status_code == 200
    assert "권장 자세는?" in pipeline.calls[1]["instruction"]
    assert "첫 답변" in pipeline.calls[1]["instruction"]


def test_pose_tab_knowledge_question_is_semantically_routed_to_rag(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _IntentPipeline("knowledge")

    data = _client(monkeypatch, pipeline).post(
        URL,
        json=_body(task_type="pose_recomm", prompt="저도막이 무엇인지 알려줘"),
    ).json()["data"]

    assert pipeline.intent_calls == [("pose_recomm", "저도막이 무엇인지 알려줘")]
    assert len(pipeline.calls) == 1
    assert pipeline.calls[0]["instruction"].startswith(
        "[사용자 질문]\n저도막이 무엇인지 알려줘\n[응답 수준]\n빠른 안내:"
    )
    assert "표·목록·제목" in pipeline.calls[0]["instruction"]
    assert data["answer"] == "저도막은 규정 최소 두께보다 얇은 도막이다."
    assert data["retrieval_used"] is True


def test_success_envelope_has_every_v25_field(monkeypatch: pytest.MonkeyPatch) -> None:
    """(b) Given a knowledge answer, When returned, Then the v2.5 wrapper and data fields are exact."""
    response = _client(monkeypatch, _FakePipeline()).post(URL, json=_body(scenario_id="1-A"))

    body = response.json()
    assert {k: body[k] for k in ("status", "code", "message")} == {
        "status": "SUCCESS",
        "code": 200,
        "message": "Copilot query completed",
    }
    data = body["data"]
    assert set(data) == _DATA_KEYS
    assert data["request_id"] == "req-query-0044"
    assert data["user_id"] == "W-017"
    assert data["scenario_id"] == "1-A"
    assert data["resolved_intents"] == ["knowledge_search"]
    assert data["context_status"] == "not_available"
    assert (data["sequence_context"], data["coating_context"], data["pose_context"]) == (None, None, None)
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", data["generated_at"])
    assert isinstance(data["latency_ms"], int) and data["latency_ms"] >= 0
    # §9.2 의 예시는 마커 없는 산문이다. 근거 id 는 citations 로만 간다.
    assert data["answer"] == "저도막은 규정 최소 두께보다 얇은 도막이다."
    assert "[답변]" not in data["answer"]
    assert "DOC-" not in data["answer"]
    assert data["degraded"] is False


def test_answer_gets_the_fast_knowledge_normalization(monkeypatch: pytest.MonkeyPatch) -> None:
    scaffold = "[분석] SENTINEL_DEFINITION\n[판정] SENTINEL_VERDICT\n[조치]\n1. SENTINEL_ACTION\n[근거] DOC-doc_0001-C0001"

    data = _client(monkeypatch, _FakePipeline(response=scaffold)).post(URL, json=_body()).json()["data"]

    assert data["answer"] == "SENTINEL_DEFINITION"
    assert "[답변]" not in data["answer"]
    assert "SENTINEL_VERDICT" not in data["answer"]
    assert "SENTINEL_ACTION" not in data["answer"]


@pytest.mark.parametrize(
    ("use_rag", "run_flag", "retrieval_used", "citations"),
    [
        # Two chunks were retrieved; the answer cites one. ``citations`` reports the cited
        # one, not the retrieval set.
        ("Y", True, True, ["DOC-doc_0001-C0001"]),
        ("N", False, False, []),
        ("y", True, True, ["DOC-doc_0001-C0001"]),
        ("n", False, False, []),
    ],
)
def test_use_rag_controls_retrieval_and_citations(
    monkeypatch: pytest.MonkeyPatch, use_rag: str, run_flag: bool, retrieval_used: bool, citations: list[str]
) -> None:
    """(c) Given use_rag Y/N, When queried, Then run(use_rag), retrieval_used and citations follow it."""
    pipeline = _FakePipeline()

    data = _client(monkeypatch, pipeline).post(URL, json=_body(use_rag=use_rag)).json()["data"]

    assert pipeline.calls[0]["use_rag"] is run_flag
    assert data["retrieval_used"] is retrieval_used
    assert data["citations"] == citations


def test_use_rag_n_leaves_no_document_id_in_the_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Given use_rag='N', When the answer still cites a document, Then the id leaves both fields."""
    data = _client(monkeypatch, _FakePipeline()).post(URL, json=_body(use_rag="N")).json()["data"]

    assert "DOC-" not in data["answer"]
    assert data["citations"] == []


def test_citation_outside_the_retrieved_chunks_is_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Given an id the search never returned, When answered, Then it reaches neither field."""
    pipeline = _FakePipeline(
        response="[답변] SENTINEL_BODY\n[근거] DOC-doc_9999-C0001",
        citations=("DOC-doc_0001-C0001",),
    )

    data = _client(monkeypatch, pipeline).post(URL, json=_body()).json()["data"]

    assert data["answer"] == "SENTINEL_BODY"
    assert data["citations"] == []


def test_request_id_is_optional_per_the_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    """Given no request_id, When queried, Then it is accepted and echoed as null (명세서 p.9)."""
    body = _body()
    del body["request_id"]

    data = _client(monkeypatch, _FakePipeline()).post(URL, json=body).json()["data"]

    assert data["request_id"] is None


def test_omitted_use_rag_defaults_to_y_and_uses_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omission enables RAG, matching an explicit Y request."""
    from trainer.demo_api.copilot_contracts import CopilotQueryRequest

    pipeline = _FakePipeline()
    body = _body()
    del body["use_rag"]

    parsed = CopilotQueryRequest(**body)
    assert parsed.use_rag == "Y"

    data = _client(monkeypatch, pipeline).post(URL, json=body).json()["data"]

    assert pipeline.calls[0]["use_rag"] is True
    assert data["retrieval_used"] is True

    automatic = _client(monkeypatch, pipeline).post(
        URL, json=_body(use_rag="auto")
    )
    assert automatic.status_code == 422
    assert automatic.json()["status"] == "ERROR"
    assert len(pipeline.calls) == 1


def test_degraded_is_true_when_no_answer_could_be_produced(monkeypatch: pytest.MonkeyPatch) -> None:
    """Given a generation with no answer section, When answered, Then degraded reports it."""
    pipeline = _FakePipeline(response="[판정] SENTINEL_VERDICT\n[근거] DOC-doc_0001-C0001")

    data = _client(monkeypatch, pipeline).post(URL, json=_body()).json()["data"]

    assert data["degraded"] is True
    assert data["answer"] == "답변을 생성하지 못했습니다."
    assert data["citations"] == []


def test_pending_task_types_are_not_degraded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Given a pending task type, When answered, Then degraded is false: nothing was truncated."""
    data = (
        _client(monkeypatch, _FakePipeline())
        .post(URL, json=_body(task_type="pose_recomm"))
        .json()["data"]
    )

    assert data["degraded"] is False
    assert data["context_status"] == "not_available"


def test_null_session_id_stays_null_even_when_run_invents_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """(d) Given session_id null, When run() returns a generated uuid, Then the response keeps null."""
    response = _client(monkeypatch, _FakePipeline()).post(URL, json=_body(session_id=None))

    assert response.json()["data"]["session_id"] is None
    assert _GENERATED_UUID not in response.text


def test_given_session_id_is_passed_through_and_echoed(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _FakePipeline()

    data = _client(monkeypatch, pipeline).post(URL, json=_body(session_id="SESSION-001")).json()["data"]

    assert pipeline.calls[0]["session_id"] == "SESSION-001"
    assert data["session_id"] == "SESSION-001"


def test_omitted_optional_fields_are_null(monkeypatch: pytest.MonkeyPatch) -> None:
    body = {k: v for k, v in _body().items() if k not in {"scenario_id", "session_id", "context"}}

    data = _client(monkeypatch, _FakePipeline()).post(URL, json=body).json()["data"]

    assert (data["scenario_id"], data["session_id"]) == (None, None)


def test_latency_covers_the_pipeline_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """(e) Given a 300 ms run, When queried, Then latency_ms is at least 300."""
    data = _client(monkeypatch, _FakePipeline(sleep_sec=0.3)).post(URL, json=_body()).json()["data"]

    assert data["latency_ms"] >= 300


def test_latency_includes_waiting_for_the_generation_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    """Given the slot is held for 400 ms after the handler entered, When answered, Then latency counts that wait."""
    pipeline = _FakePipeline()
    client = _client(monkeypatch, pipeline)
    entered = threading.Event()

    def _provider() -> _FakePipeline:
        entered.set()  # the handler clock is already running when the provider is called
        return pipeline

    monkeypatch.setattr(chat_service_mod, "_get_or_build_pipeline", _provider)
    results: list[dict[str, Any]] = []
    assert agent_pipeline._GENERATION_SLOT.acquire(timeout=5)
    worker = threading.Thread(target=lambda: results.append(client.post(URL, json=_body()).json()["data"]))
    try:
        worker.start()
        assert entered.wait(timeout=10)
        time.sleep(0.4)
    finally:
        agent_pipeline._GENERATION_SLOT.release()
    worker.join(timeout=10)

    assert len(results) == 1
    assert results[0]["latency_ms"] >= 400


def _error(code: int, message: str) -> dict[str, Any]:
    return {"status": "ERROR", "code": code, "message": message, "data": None}


def test_pipeline_exception_becomes_error_envelope_without_leaking(monkeypatch: pytest.MonkeyPatch) -> None:
    """(f) Given run() raises, When queried, Then ERROR 500 with data null and no internal detail."""
    pipeline = _FakePipeline(exc=RuntimeError("SECRET /output/indexes/faiss.index"))

    response = _client(monkeypatch, pipeline).post(URL, json=_body())

    assert response.status_code == 500
    assert response.json() == _error(500, "Copilot query failed")
    assert "SECRET" not in response.text


def test_model_server_outage_is_503(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _FakePipeline(exc=OpenAIBackendUnavailableError("HTTP 503 from http://vllm:8000"))

    response = _client(monkeypatch, pipeline).post(URL, json=_body())

    assert response.status_code == 503
    assert response.json() == _error(503, "Model server unavailable")
    assert "vllm" not in response.text


def test_pipeline_construction_failure_is_503(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom() -> object:
        raise RuntimeError("SECRET model load")

    app = _app(monkeypatch, None)
    monkeypatch.setattr(chat_service_mod, "_construct_default_pipeline", _boom)

    response = TestClient(app).post(URL, json=_body())

    assert response.status_code == 503
    assert response.json() == _error(503, "Copilot pipeline is not ready")
    assert "SECRET" not in response.text


def test_demo_mode_does_not_answer_knowledge_search(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _FakePipeline()

    response = _client(monkeypatch, pipeline, live=False).post(URL, json=_body())

    assert response.status_code == 503
    assert response.json() == _error(503, "Copilot pipeline is not configured")
    assert pipeline.calls == []


@pytest.fixture
def network_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def _sync(self: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        raise httpx.ConnectError("network disabled in test")

    async def _async(self: httpx.AsyncHTTPTransport, request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        raise httpx.ConnectError("network disabled in test")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _sync)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", _async)
    return calls


@pytest.mark.parametrize("task_type", _PENDING)
@pytest.mark.parametrize("use_rag", ["Y", "N"])
def test_pending_task_types_answer_not_available_without_pipeline_or_mcp(
    monkeypatch: pytest.MonkeyPatch, network_calls: list[str], task_type: str, use_rag: str
) -> None:
    """(g) Given a task type whose MCP is unfinished, When queried, Then 200 not_available, no run, no MCP."""
    pipeline = _FakePipeline()
    constructed: list[object] = []
    app = _app(monkeypatch, None)
    monkeypatch.setattr(chat_service_mod, "_construct_default_pipeline", lambda: constructed.append(pipeline) or pipeline)

    response = TestClient(app).post(URL, json=_body(task_type=task_type, use_rag=use_rag, session_id="SESSION-001"))

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "SUCCESS"
    data = body["data"]
    assert set(data) == _DATA_KEYS
    assert data["answer"] == one_line(copilot_router.PENDING_TASK_ANSWERS[task_type])  # 빠른 안내 한 줄
    assert "완료 분석" in data["answer"]
    assert data["resolved_intents"] == [task_type]
    assert data["context_status"] == "not_available"
    assert (data["sequence_context"], data["coating_context"], data["pose_context"]) == (None, None, None)
    assert data["retrieval_used"] is False
    assert data["citations"] == []
    assert data["session_id"] == "SESSION-001"
    assert pipeline.calls == []
    assert constructed == []
    assert network_calls == []


def test_pending_answers_are_configurable() -> None:
    """pose_recomm 은 이제 실제로 채점하므로 아직 엔진이 없는 task_type 으로 시험한다."""
    from fastapi import FastAPI

    from trainer.demo_api.sequence_store import SequenceStore

    router = copilot_router.create_copilot_router(
        None,
        pending_answers={"trajectory_recomm": "사용자 정의 안내"},
        store=SequenceStore(":memory:"),
    )
    app = FastAPI()
    app.include_router(router)

    data = TestClient(app).post(URL, json=_body(task_type="trajectory_recomm")).json()["data"]

    assert data["answer"] == "사용자 정의 안내"


def test_knowledge_search_succeeds_while_mcp_is_unreachable(
    monkeypatch: pytest.MonkeyPatch, network_calls: list[str]
) -> None:
    """(h) Given MCP_BASE_URL pointing nowhere, When the app is created and queried, Then knowledge_search is 200."""
    pipeline = _FakePipeline()

    response = _client(monkeypatch, pipeline).post(URL, json=_body())

    assert response.status_code == 200
    assert response.json()["status"] == "SUCCESS"
    assert network_calls == []


@pytest.mark.parametrize(
    "body",
    [
        _body(extra_field="x"),
        _body(use_rag="true"),
        _body(use_rag=True),
        _body(task_type="free_chat"),
        {k: v for k, v in _body().items() if k != "user_id"},
        _body(prompt=""),
        _body(prompt="   "),
    ],
    ids=["extra", "use_rag-string", "use_rag-bool", "task_type", "no-user_id", "empty", "blank"],
)
def test_invalid_requests_get_the_error_envelope(monkeypatch: pytest.MonkeyPatch, body: dict[str, Any]) -> None:
    """(i) Given an invalid body, When posted, Then 422 in the v2.5 ERROR wrapper; run() is not called."""
    pipeline = _FakePipeline()

    response = _client(monkeypatch, pipeline).post(URL, json=body)

    assert response.status_code == 422
    assert response.json()["status"] == "ERROR"
    assert response.json()["data"] is None
    assert response.json()["message"].startswith("Invalid request: body.")
    assert pipeline.calls == []


def test_malformed_json_gets_the_error_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _client(monkeypatch, _FakePipeline()).post(
        URL, content=b"{not json", headers={"content-type": "application/json"}
    )

    assert response.status_code == 422
    assert response.json()["status"] == "ERROR"
    assert response.json()["data"] is None
    assert response.json()["message"].startswith("Invalid request: body.")


def test_legacy_routes_keep_fastapi_validation_format(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _client(monkeypatch, _FakePipeline()).post("/api/chat", json={"message": "hi", "session_id": "s", "bogus": 1})

    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)
    assert "status" not in response.json()


def test_copilot_shares_the_generation_slot_with_live_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    """(j) Given the live SSE path holds the slot, When a copilot query arrives, Then it waits for that slot."""
    pipeline = _FakePipeline()
    client = _client(monkeypatch, pipeline)
    results: list[int] = []

    assert copilot_router.GENERATION_SLOT is agent_pipeline._GENERATION_SLOT
    assert agent_pipeline._GENERATION_SLOT.acquire(timeout=5)
    worker = threading.Thread(target=lambda: results.append(client.post(URL, json=_body()).status_code))
    try:
        worker.start()
        time.sleep(0.3)
        assert pipeline.calls == []
    finally:
        agent_pipeline._GENERATION_SLOT.release()
    worker.join(timeout=10)

    assert results == [200]
    assert len(pipeline.calls) == 1


def test_concurrent_queries_run_one_at_a_time(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _FakePipeline(sleep_sec=0.2)
    app = _app(monkeypatch, pipeline)

    async def _both() -> list[httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            responses: list[httpx.Response] = []

            async def _one(request_id: str) -> None:
                responses.append(await client.post(URL, json=_body(request_id=request_id)))

            async with anyio.create_task_group() as group:
                group.start_soon(_one, "req-1")
                group.start_soon(_one, "req-2")
            return responses

    responses = anyio.run(_both)

    assert sorted(r.json()["data"]["request_id"] for r in responses) == ["req-1", "req-2"]
    assert len(pipeline.calls) == 2
    assert pipeline.peak == 1
    assert min(r.json()["data"]["latency_ms"] for r in responses) >= 200


def test_order_question_without_sequences_explains_missing_data(
    monkeypatch: pytest.MonkeyPatch, network_calls: list[str]
) -> None:
    """순서 질문은 complete 리포트로 답한다. 아직 없으면 '준비 중'이 아니라 언제 생기는지를 말한다. 파이프라인·MCP 없음."""
    app = _app(monkeypatch, None)

    data = TestClient(app).post(URL, json=_body(task_type="trajectory_recomm", session_id="SESSION-NONE")).json()["data"]

    assert "분사 시퀀스나 완료 리포트가 없습니다" in data["answer"]
    assert data["context_status"] == "not_available"
    assert data["sequence_context"]["reason_code"] == "session_report_unavailable"
    assert network_calls == []
