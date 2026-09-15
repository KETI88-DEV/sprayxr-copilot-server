"""AgentPipeline on the OpenAI-compatible backend: routes, tool loop, errors, cancellation."""

from __future__ import annotations

import asyncio
import json
import socket
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

import anyio
import httpx
import pytest
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from trainer.scripts import agent_pipeline
from trainer.scripts.agent_pipeline import AgentPipeline, SemanticRouter, SessionContext
from trainer.scripts.openai_backend import (
    ContextLengthExceededError,
    OpenAIBackendSettings,
    OpenAIBackendUnavailableError,
    OpenAIChatBackend,
)
from trainer.scripts.tool_calling_adapter import SYNTHESIS_INSTRUCTION, ToolCallingAdapter

_CONFIGS = Path(__file__).resolve().parents[2] / "configs"
PipelineEvent = tuple[str, dict[str, Any]]


def _config(base_url: str = "http://vllm.test:8000/v1") -> dict[str, Any]:
    return {
        "llm": {
            "backend": "openai",
            "base_url": base_url,
            "served_model_name": "shipcoat-v6-1-fp8",
            "connect_timeout_sec": 2.0,
            "request_timeout_sec": 10.0,
            "max_model_len": 8192,
            "pass_native_tools": False,
            "thinking_off_for_tools": False,
            "append_citations": True,
            "tool_deadline_sec": 150,
        },
        "generation": {
            "temperature": 0.3,
            "top_p": 0.8,
            "top_k": 20,
            "repetition_penalty": 1.15,
            "do_sample": True,
            "max_new_tokens": 1024,
            "max_new_tokens_tool": 256,
        },
        "prompt": {
            "system": "SYSTEM-TOOLS",
            "fast_system": "SYSTEM-FAST",
            "context_template": "문서:\n{context}",
        },
    }


def _sse(*chunks: dict[str, Any], done: bool = True) -> bytes:
    body = "".join(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n" for chunk in chunks)
    if done:
        body += "data: [DONE]\n\n"
    return body.encode("utf-8")


def _delta(content: str | None = None, finish_reason: str | None = None) -> dict[str, Any]:
    delta = {} if content is None else {"content": content}
    return {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}


class _ScriptedServer:
    """Fake vLLM that answers the n-th chat request with the n-th scripted text."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.payloads: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.payloads.append(json.loads(request.content))
        text = self.replies[len(self.payloads) - 1]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse(_delta(text), _delta(finish_reason="stop")),
        )


@pytest.fixture
def fake_search(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    queries: list[str] = []
    module = ModuleType("search_index")

    def _search_and_rerank(query: str, *_args: Any, **_kwargs: Any) -> tuple[list[dict[str, Any]], dict[str, float]]:
        queries.append(query)
        return [{"chunk_id": "DOC-doc_0001-C0001", "text": "저도막은 규정 두께 미달이다.", "rerank_score": 0.9}], {"total_ms": 1.0}

    setattr(module, "search_and_rerank", _search_and_rerank)
    monkeypatch.setitem(sys.modules, "search_index", module)
    return queries


def _pipeline(config: dict[str, Any], transport: httpx.BaseTransport | None, mcp_base_url: str = "http://127.0.0.1:9") -> AgentPipeline:
    agent_pipeline._ensure_scripts_path()
    pipe = AgentPipeline.__new__(AgentPipeline)
    pipe.config = config
    pipe.backend = "openai"
    pipe.session_context = SessionContext(max_turns=3)
    pipe.router = SemanticRouter()
    pipe.model = OpenAIChatBackend(OpenAIBackendSettings.from_config(config, environ={}), transport=transport)
    pipe.tokenizer = None
    pipe.tool_adapter = ToolCallingAdapter(
        schema_dir=_CONFIGS / "tool_schemas",
        protocol_path=_CONFIGS / "tool_calling_protocol.json",
        mcp_base_url=mcp_base_url,
        pass_native_tools=False,
    )
    pipe.index = pipe.id_map = pipe.embedder = pipe.reranker = None
    return pipe


@contextmanager
def _serve(app: Any) -> Iterator[str]:
    """Runs a real uvicorn server on an ephemeral port in a background thread."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="off"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline or not thread.is_alive():
            raise RuntimeError("fake server did not start")
        time.sleep(0.01)
    try:
        yield f"http://127.0.0.1:{sock.getsockname()[1]}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()


def test_openai_backend_loads_without_local_model_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """Given backend openai and no llm.model_path, When loading, Then only the HTTP client is built."""

    def _forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("local model weights must not load for the openai backend")

    monkeypatch.setattr(agent_pipeline.AutoModelForCausalLM, "from_pretrained", _forbidden)
    monkeypatch.setattr(agent_pipeline.AutoTokenizer, "from_pretrained", _forbidden)
    pipe = AgentPipeline.__new__(AgentPipeline)
    pipe.config = _config()
    pipe.backend = "openai"

    model, tokenizer = pipe._load_model_and_tokenizer("/output/adapters/qwen36_27b_shipcoat_lora_v6_1")

    assert isinstance(model, OpenAIChatBackend)
    assert model.settings.served_model_name == "shipcoat-v6-1-fp8"
    assert tokenizer is None


def test_fast_route_is_one_request_capped_at_route_budget_with_thinking_off(fake_search: list[str]) -> None:
    server = _ScriptedServer(["[답변] 저도막은 규정 두께 미달입니다.\n[근거] DOC-doc_0001-C0001"])
    pipe = _pipeline(_config(), httpx.MockTransport(server))

    result = pipe.run(instruction="저도막이 뭐야", force_route="fast", enable_thinking=True)

    assert fake_search == ["저도막이 뭐야"]
    assert len(server.payloads) == 1
    payload = server.payloads[0]
    assert payload["max_tokens"] == 256
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert "tools" not in payload and "tool_choice" not in payload
    assert payload["messages"][0] == {"role": "system", "content": "SYSTEM-FAST"}
    assert result["route"] == "fast"
    assert result["response"].startswith("[답변] 저도막은 규정 두께 미달입니다.")


def _fake_mcp(received: list[dict[str, Any]]) -> Starlette:
    async def _dft(request: Request) -> JSONResponse:
        received.append(await request.json())
        return JSONResponse(
            {"status": "success", "result": {"verdict": "불합격", "mean": 103.3}, "error": None, "timestamp": "t"}
        )

    return Starlette(routes=[Route("/dft_judge/evaluate", _dft, methods=["POST"])])


def test_tool_loop_round_trip_through_fake_vllm_and_fake_mcp(
    fake_search: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a tool call, a silent second round and forced synthesis, When run, Then every request follows D6."""
    monkeypatch.setenv("TOOL_ADAPTER_FORCE_SYNTHESIS", "1")
    received: list[dict[str, Any]] = []
    tool_call = (
        '<tool_call>{"name": "dft_judge", "arguments": {"measurements": [82, 88, 140], '
        '"spec": {"nominal": 100, "tolerance_rule": "90/10"}}}</tool_call>'
    )
    synthesis = "[분석] 두 측정값이 90% 기준 미만입니다.\n[판정] 불합격\n[근거] DOC-doc_0001-C0001"
    server = _ScriptedServer([tool_call, "", synthesis])

    with _serve(_fake_mcp(received)) as mcp_url:
        pipe = _pipeline(_config(), httpx.MockTransport(server), mcp_base_url=mcp_url)
        result = pipe.run(instruction="DFT 82 88 140 판정해줘", force_route="training_review", enable_thinking=True)

    assert received == [{"measurements": [82, 88, 140], "spec": {"nominal": 100, "tolerance_rule": "90/10"}}]
    assert len(server.payloads) == 3
    assert all("tools" not in p and "tool_choice" not in p for p in server.payloads)
    assert [p["max_tokens"] for p in server.payloads] == [256, 256, 1024]
    assert [p["chat_template_kwargs"]["enable_thinking"] for p in server.payloads] == [True, True, False]
    tool_messages = [m for m in server.payloads[1]["messages"] if m["role"] == "tool"]
    assert len(tool_messages) == 1 and "불합격" in tool_messages[0]["content"]
    assert server.payloads[1]["messages"][-2] == {"role": "assistant", "content": tool_call}
    assert server.payloads[2]["messages"][-1] == {"role": "user", "content": SYNTHESIS_INSTRUCTION}
    assert [call["name"] for call in result["tool_calls"]] == ["dft_judge"]
    assert result["tool_calls"][0]["result"]["status"] == "success"
    assert result["response"].startswith("[분석] 두 측정값이 90% 기준 미만입니다.")


def test_realtime_synthesis_uses_realtime_route_cap(fake_search: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOOL_ADAPTER_FORCE_SYNTHESIS", "1")
    received: list[dict[str, Any]] = []
    tool_call = '<tool_call>{"name": "dft_judge", "arguments": {"measurements": [95], "spec": {"nominal": 100, "tolerance_rule": "90/10"}}}</tool_call>'
    server = _ScriptedServer([tool_call, tool_call, "[판정] 합격"])

    with _serve(_fake_mcp(received)) as mcp_url:
        pipe = _pipeline(_config(), httpx.MockTransport(server), mcp_base_url=mcp_url)
        _ = pipe.run(instruction="DFT 95 판정", force_route="training_realtime", enable_thinking=False)

    # training_realtime allows 2 rounds, then forces synthesis at the route cap.
    assert [p["max_tokens"] for p in server.payloads] == [256, 256, 256]
    assert server.payloads[2]["messages"][-1]["content"] == SYNTHESIS_INSTRUCTION


def _collect(pipe: AgentPipeline, instruction: str) -> list[PipelineEvent]:
    async def _drain() -> list[PipelineEvent]:
        return [event async for event in pipe.run_streaming(instruction=instruction)]

    return anyio.run(_drain)


@pytest.mark.parametrize(
    ("exc", "message", "retryable"),
    [
        (ContextLengthExceededError("maximum context length is 8192 SECRET-PROMPT"), "prompt exceeds the model context window", False),
        (OpenAIBackendUnavailableError("HTTP 503 from http://vllm-internal:8000 SECRET-URL"), "streaming pipeline failed", True),
    ],
    ids=["context-length", "unavailable"],
)
def test_run_streaming_maps_backend_failures_to_one_sanitized_error(exc: Exception, message: str, retryable: bool) -> None:
    pipe = AgentPipeline.__new__(AgentPipeline)
    pipe.router = SemanticRouter()

    def _raise(**_kwargs: Any) -> dict[str, Any]:
        raise exc

    pipe.run = _raise

    events = _collect(pipe, "저도막이 뭐야")

    assert [kind for kind, _ in events] == ["error"]
    payload = events[0][1]
    assert payload["message"] == message
    assert payload["retryable"] is retryable
    assert payload["ttft_ms"] is None
    assert isinstance(payload["total_ms"], int)
    assert "SECRET" not in json.dumps(payload)


def _slow_vllm(state: dict[str, Any]) -> Starlette:
    async def _chat(request: Request) -> StreamingResponse:
        state["payload"] = await request.json()

        async def _body() -> Any:
            try:
                for index in range(400):
                    state["sent"] += 1
                    state["streaming"].set()
                    yield _sse(_delta(f"토큰{index} "), done=False)
                    await asyncio.sleep(0.05)
                yield _sse(_delta(finish_reason="stop"))
                state["completed"] = True
            finally:
                state["closed"].set()

        return StreamingResponse(_body(), media_type="text/event-stream")

    return Starlette(routes=[Route("/v1/chat/completions", _chat, methods=["POST"])])


def test_consumer_leaving_run_streaming_aborts_the_upstream_generation(
    fake_search: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a slow vLLM stream, When the SSE consumer closes, Then the upstream request is disconnected."""
    monkeypatch.setattr(agent_pipeline, "_PROGRESS_INTERVAL_SEC", 0.05)
    state: dict[str, Any] = {"sent": 0, "completed": False, "streaming": threading.Event(), "closed": threading.Event()}

    with _serve(_slow_vllm(state)) as url:
        pipe = _pipeline(_config(base_url=f"{url}/v1"), transport=None)

        async def _consume_then_leave() -> None:
            stream = pipe.run_streaming(instruction="저도막이 뭐야")
            with anyio.fail_after(15):
                async for kind, payload in stream:
                    if kind == "status" and payload.get("stage") == "generating" and state["sent"] >= 3:
                        break
            await stream.aclose()

        anyio.run(_consume_then_leave)

        assert state["closed"].wait(timeout=5), "fake vLLM never saw the client disconnect"
        assert state["completed"] is False
        assert state["sent"] < 400
        assert state["payload"]["max_tokens"] == 256
        acquired = agent_pipeline._GENERATION_SLOT.acquire(timeout=5)
        assert acquired, "generation slot was not released after cancellation"
        agent_pipeline._GENERATION_SLOT.release()
