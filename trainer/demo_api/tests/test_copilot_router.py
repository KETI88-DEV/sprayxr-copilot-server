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
    "answer",
    "resolved_intents",
    "context_status",
    "sequence_context",
    "coating_context",
    "pose_context",
    "retrieval_used",
    "citations",
}
_PENDING = ["trajectory_recomm", "pose_recomm", "error_analysis"]
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
                "citations": list(self.citations),
                "tool_calls": [],
                "rag_chunks": [],
            }
        finally:
            with self._guard:
                self.active -= 1


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
    assert pipeline.calls == [
        {
            "instruction": "저도막이 무엇인지 설명해줘.",
            "force_route": "fast",
            "session_id": None,
            "enable_thinking": False,
            "use_rag": True,
            "request_id": "req-query-0044",
        }
    ]


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
    assert data["answer"].startswith("[답변] 저도막은 규정 최소 두께보다 얇은 도막이다.")


def test_answer_gets_the_fast_knowledge_normalization(monkeypatch: pytest.MonkeyPatch) -> None:
    scaffold = "[분석] SENTINEL_DEFINITION\n[판정] SENTINEL_VERDICT\n[조치]\n1. SENTINEL_ACTION\n[근거] DOC-doc_0001-C0001"

    data = _client(monkeypatch, _FakePipeline(response=scaffold)).post(URL, json=_body()).json()["data"]

    assert "SENTINEL_DEFINITION" in data["answer"]
    assert data["answer"].startswith("[답변]")
    assert "SENTINEL_VERDICT" not in data["answer"]
    assert "SENTINEL_ACTION" not in data["answer"]


@pytest.mark.parametrize(
    ("use_rag", "run_flag", "retrieval_used", "citations"),
    [
        ("Y", True, True, ["DOC-doc_0001-C0001", "DOC-doc_0002-C0003"]),
        ("N", False, False, []),
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
    assert data["answer"] == copilot_router.PENDING_TASK_ANSWERS[task_type]
    assert "준비 중" in data["answer"]
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
    router = copilot_router.create_copilot_router(None, pending_answers={"pose_recomm": "사용자 정의 안내"})
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)

    data = TestClient(app).post(URL, json=_body(task_type="pose_recomm")).json()["data"]

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
        {k: v for k, v in _body().items() if k != "request_id"},
        {k: v for k, v in _body().items() if k != "user_id"},
        _body(prompt=""),
        _body(prompt="   "),
    ],
    ids=["extra", "use_rag-string", "use_rag-bool", "task_type", "no-request_id", "no-user_id", "empty", "blank"],
)
def test_invalid_requests_get_the_error_envelope(monkeypatch: pytest.MonkeyPatch, body: dict[str, Any]) -> None:
    """(i) Given an invalid body, When posted, Then 422 in the v2.5 ERROR wrapper; run() is not called."""
    pipeline = _FakePipeline()

    response = _client(monkeypatch, pipeline).post(URL, json=body)

    assert response.status_code == 422
    assert response.json() == _error(422, "Invalid request")
    assert pipeline.calls == []


def test_malformed_json_gets_the_error_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _client(monkeypatch, _FakePipeline()).post(
        URL, content=b"{not json", headers={"content-type": "application/json"}
    )

    assert response.status_code == 422
    assert response.json() == _error(422, "Invalid request")


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
