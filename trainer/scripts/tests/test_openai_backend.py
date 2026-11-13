"""OpenAI-compatible (vLLM) chat backend: request contract, stream assembly, error mapping."""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import anyio
import httpx
import pytest

from trainer.scripts.openai_backend import (
    ContextLengthExceededError,
    GenerationCancelledError,
    OpenAIBackendRequestError,
    OpenAIBackendSettings,
    OpenAIBackendTimeoutError,
    OpenAIBackendUnavailableError,
    OpenAIChatBackend,
    generation_cancel_scope,
    probe_openai_server,
)
from trainer.scripts.tool_calling_adapter import ToolCallingAdapter, ToolInfrastructureError

_CONFIGS = Path(__file__).resolve().parents[2] / "configs"

# Sampling values deliberately differ from every code default (0.3/0.7, 0.8, 20, 1.15)
# so a backend that silently falls back to a default fails these tests.
_GENERATION = {
    "temperature": 0.31,
    "top_p": 0.83,
    "top_k": 17,
    "repetition_penalty": 1.19,
    "do_sample": True,
    "max_new_tokens": 1024,
    "max_new_tokens_tool": 256,
}


def _config(**llm: Any) -> dict[str, Any]:
    base = {
        "backend": "openai",
        "base_url": "http://vllm.test:8000/v1",
        "served_model_name": "shipcoat-v6-1-fp8",
        "api_key_env": "TEST_OPENAI_BACKEND_KEY",
        "connect_timeout_sec": 1.5,
        "request_timeout_sec": 7.0,
    }
    base.update(llm)
    return {"llm": base, "generation": dict(_GENERATION)}


def _sse(*chunks: dict[str, Any], done: bool = True) -> bytes:
    body = "".join(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n" for chunk in chunks)
    if done:
        body += "data: [DONE]\n\n"
    return body.encode("utf-8")


def _delta(content: str | None = None, finish_reason: str | None = None, **delta: Any) -> dict[str, Any]:
    if content is not None:
        delta["content"] = content
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


class _Recorder:
    def __init__(self, body: bytes, status: int = 200) -> None:
        self.body = body
        self.status = status
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(self.status, headers={"content-type": "text/event-stream"}, content=self.body)

    def payload(self, index: int = 0) -> dict[str, Any]:
        return json.loads(self.requests[index].content)


def _backend(handler: Any, config: dict[str, Any] | None = None) -> OpenAIChatBackend:
    settings = OpenAIBackendSettings.from_config(config or _config(), environ={})
    return OpenAIChatBackend(settings, transport=httpx.MockTransport(handler))


def test_request_carries_every_d6_field_and_never_tools() -> None:
    """Given a chat call, When sent, Then the payload is exactly the D6 contract with no tools."""
    recorder = _Recorder(_sse(_delta("답변", finish_reason="stop")))
    backend = _backend(recorder)
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "저도막이 뭐야"}]

    text = backend.complete_chat(messages, max_tokens=256, enable_thinking=False)

    assert text == "답변"
    assert len(recorder.requests) == 1
    request = recorder.requests[0]
    assert request.method == "POST"
    assert str(request.url) == "http://vllm.test:8000/v1/chat/completions"
    assert recorder.payload() == {
        "model": "shipcoat-v6-1-fp8",
        "messages": messages,
        "max_tokens": 256,
        "temperature": 0.31,
        "top_p": 0.83,
        "top_k": 17,
        "repetition_penalty": 1.19,
        "chat_template_kwargs": {"enable_thinking": False},
        "skip_special_tokens": True,
        "stream": True,
    }


def test_enable_thinking_is_forwarded_per_call() -> None:
    recorder = _Recorder(_sse(_delta("x", finish_reason="stop")))
    backend = _backend(recorder)

    backend.complete_chat([{"role": "user", "content": "q"}], max_tokens=64, enable_thinking=True)

    assert recorder.payload()["chat_template_kwargs"] == {"enable_thinking": True}
    assert recorder.payload()["max_tokens"] == 64


@pytest.mark.parametrize("missing", ["temperature", "top_p", "top_k", "repetition_penalty"])
def test_settings_refuse_to_fall_back_to_code_default_sampling(missing: str) -> None:
    """Given a config without an explicit sampling key, When settings load, Then it fails naming the key."""
    config = _config()
    del config["generation"][missing]

    with pytest.raises(ValueError, match=missing):
        OpenAIBackendSettings.from_config(config, environ={})


@pytest.mark.parametrize("missing", ["base_url", "served_model_name"])
def test_settings_require_server_address_and_model_name(missing: str) -> None:
    config = _config()
    del config["llm"][missing]

    with pytest.raises(ValueError, match=missing):
        OpenAIBackendSettings.from_config(config, environ={})


def test_greedy_config_sends_zero_temperature() -> None:
    config = _config()
    config["generation"]["do_sample"] = False
    recorder = _Recorder(_sse(_delta("x", finish_reason="stop")))

    _backend(recorder, config).complete_chat([{"role": "user", "content": "q"}], max_tokens=8, enable_thinking=False)

    assert recorder.payload()["temperature"] == 0.0


def test_authorization_header_only_when_key_env_is_set() -> None:
    recorder = _Recorder(_sse(_delta("x", finish_reason="stop")))
    with_key = OpenAIBackendSettings.from_config(_config(), environ={"TEST_OPENAI_BACKEND_KEY": "s3cret"})
    without_key = OpenAIBackendSettings.from_config(_config(), environ={"TEST_OPENAI_BACKEND_KEY": ""})

    for settings in (with_key, without_key):
        OpenAIChatBackend(settings, transport=httpx.MockTransport(recorder)).complete_chat(
            [{"role": "user", "content": "q"}], max_tokens=8, enable_thinking=False
        )

    assert recorder.requests[0].headers["authorization"] == "Bearer s3cret"
    assert "authorization" not in recorder.requests[1].headers


def test_spark_serving_config_satisfies_the_openai_backend_contract() -> None:
    """Given the shipped Spark config, When settings load, Then D6 sampling and the shared window are explicit."""
    import yaml

    with (_CONFIGS / "rag_config_spark_vllm.yaml").open(encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    with (_CONFIGS / "rag_config_fewshot_27b.yaml").open(encoding="utf-8") as fh:
        baseline = yaml.safe_load(fh)

    settings = OpenAIBackendSettings.from_config(config, environ={})

    assert config["llm"]["backend"] == "openai"
    assert settings.base_url.endswith("/v1")
    assert (settings.sampling.temperature, settings.sampling.top_p, settings.sampling.top_k) == (0.3, 0.8, 20)
    assert config["generation"]["repetition_penalty"] == 1.0
    assert config["llm"]["max_model_len"] == baseline["llm"]["max_model_len"] == 8192
    assert config["llm"]["pass_native_tools"] is False
    assert config["llm"]["tokenizer_path"].startswith("/")
    assert config["llm"].get("api_key_env") == "VLLM_API_KEY"
    assert config["embedding"]["model_name"].startswith("/")
    assert config["reranker"]["model_name"].startswith("/")
    assert config["prompt"] == baseline["prompt"]
    assert config["reranker"]["top_k_rerank"] == baseline["reranker"]["top_k_rerank"]


def test_settings_carry_timeouts() -> None:
    settings = OpenAIBackendSettings.from_config(_config(), environ={})

    assert settings.connect_timeout_sec == 1.5
    assert settings.request_timeout_sec == 7.0


def test_tool_call_split_across_sse_chunks_and_bytes_is_reassembled_and_parsed() -> None:
    """Given a <tool_call> cut across SSE events and TCP reads, When accumulated, Then the adapter parses it."""
    pieces = [
        "측정값을 확인합니다.\n<tool",
        '_call>{"name": "dft_',
        'judge", "arguments": {"measurements": [82, 88], ',
        '"spec": {"nominal": 100, "tolerance_rule": "90/10"}}}</tool_',
        "call>",
    ]
    body = _sse(
        _delta(role="assistant"),
        _delta(reasoning_content="내부 추론은 content가 아니다"),
        *[_delta(piece) for piece in pieces],
        _delta(finish_reason="stop"),
        {"id": "chatcmpl-1", "object": "chat.completion.chunk", "choices": [], "usage": {"completion_tokens": 40}},
    )

    def _handler(_request: httpx.Request) -> httpx.Response:
        def _reads() -> Iterator[bytes]:
            for start in range(0, len(body), 7):
                yield body[start : start + 7]

        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_reads())

    text = _backend(_handler).complete_chat([{"role": "user", "content": "q"}], max_tokens=256, enable_thinking=False)

    assert text == "".join(pieces)
    adapter = ToolCallingAdapter(
        schema_dir=_CONFIGS / "tool_schemas",
        protocol_path=_CONFIGS / "tool_calling_protocol.json",
        pass_native_tools=False,
    )
    calls = adapter.parse_tool_calls(text)
    assert [(call["name"], call["arguments"]) for call in calls] == [
        ("dft_judge", {"measurements": [82, 88], "spec": {"nominal": 100, "tolerance_rule": "90/10"}})
    ]


def test_length_finish_reason_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    recorder = _Recorder(_sse(_delta("잘린 답"), _delta(finish_reason="length")))

    with caplog.at_level(logging.WARNING, logger="trainer.scripts.openai_backend"):
        text = _backend(recorder).complete_chat([{"role": "user", "content": "q"}], max_tokens=32, enable_thinking=False)

    assert text == "잘린 답"
    assert "finish_reason=length" in caplog.text
    assert "max_tokens=32" in caplog.text


def test_normal_stop_is_not_logged_as_truncation(caplog: pytest.LogCaptureFixture) -> None:
    recorder = _Recorder(_sse(_delta("답"), _delta(finish_reason="stop")))

    with caplog.at_level(logging.WARNING, logger="trainer.scripts.openai_backend"):
        _backend(recorder).complete_chat([{"role": "user", "content": "q"}], max_tokens=32, enable_thinking=False)

    assert "finish_reason=length" not in caplog.text


_VLLM_CONTEXT_ERROR = {
    "object": "error",
    "message": (
        "This model's maximum context length is 8192 tokens. However, you requested 9300 tokens "
        "(8276 in the messages, 1024 in the completion). Please reduce the length of the messages or completion."
    ),
    "type": "BadRequestError",
    "param": None,
    "code": 400,
}


def _status_handler(status: int, body: dict[str, Any]) -> Any:
    def _handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body)

    return _handler


def _raising_handler(exc: Exception) -> Any:
    def _handler(_request: httpx.Request) -> httpx.Response:
        raise exc

    return _handler


@pytest.mark.parametrize(
    ("handler", "expected", "infrastructure", "retryable"),
    [
        (_status_handler(400, _VLLM_CONTEXT_ERROR), ContextLengthExceededError, False, False),
        (
            _status_handler(400, {"object": "error", "message": "top_k must be -1 or at least 1", "code": 400}),
            OpenAIBackendRequestError,
            False,
            False,
        ),
        (_status_handler(404, {"object": "error", "message": "The model `x` does not exist."}), OpenAIBackendRequestError, False, False),
        (_status_handler(500, {"object": "error", "message": "EngineCore died"}), OpenAIBackendUnavailableError, True, True),
        (_status_handler(503, {"detail": "starting"}), OpenAIBackendUnavailableError, True, True),
        (_raising_handler(httpx.ReadTimeout("read timed out")), OpenAIBackendTimeoutError, True, True),
        (_raising_handler(httpx.ConnectTimeout("connect timed out")), OpenAIBackendTimeoutError, True, True),
        (_raising_handler(httpx.ConnectError("connection refused")), OpenAIBackendUnavailableError, True, True),
    ],
    ids=["400-context", "400-other", "404", "500", "503", "read-timeout", "connect-timeout", "connect-refused"],
)
def test_http_failures_map_to_backend_errors(
    handler: Any, expected: type[Exception], infrastructure: bool, retryable: bool
) -> None:
    with pytest.raises(expected) as caught:
        _backend(handler).complete_chat([{"role": "user", "content": "q"}], max_tokens=8, enable_thinking=False)

    assert isinstance(caught.value, ToolInfrastructureError) is infrastructure
    assert getattr(caught.value, "retryable") is retryable
    if expected is not ContextLengthExceededError:
        assert not isinstance(caught.value, ContextLengthExceededError)


def test_error_object_inside_stream_is_not_returned_as_text() -> None:
    body = _sse(_delta("부분"), {"object": "error", "message": "engine dead", "code": 500}, done=False)

    with pytest.raises(OpenAIBackendUnavailableError):
        _backend(_Recorder(body)).complete_chat([{"role": "user", "content": "q"}], max_tokens=8, enable_thinking=False)


def test_stream_that_ends_without_done_or_finish_reason_is_an_error() -> None:
    body = _sse(_delta("잘린 연결"), done=False)

    with pytest.raises(OpenAIBackendUnavailableError):
        _backend(_Recorder(body)).complete_chat([{"role": "user", "content": "q"}], max_tokens=8, enable_thinking=False)


def test_cancelled_scope_sends_no_request() -> None:
    recorder = _Recorder(_sse(_delta("x", finish_reason="stop")))
    event = threading.Event()
    event.set()

    with generation_cancel_scope(event), pytest.raises(GenerationCancelledError):
        _backend(recorder).complete_chat([{"role": "user", "content": "q"}], max_tokens=8, enable_thinking=False)

    assert recorder.requests == []


def test_cancel_mid_stream_stops_reading_the_upstream_body() -> None:
    event = threading.Event()
    produced: list[int] = []

    def _handler(_request: httpx.Request) -> httpx.Response:
        def _reads() -> Iterator[bytes]:
            for index in range(50):
                produced.append(index)
                if index == 2:
                    event.set()  # the consumer leaves while tokens are still arriving
                yield _sse(_delta(f"t{index} "), done=False)

        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_reads())

    with generation_cancel_scope(event), pytest.raises(GenerationCancelledError):
        _backend(_handler).complete_chat([{"role": "user", "content": "q"}], max_tokens=512, enable_thinking=False)

    assert len(produced) < 10


def _probe(handler: Any, config: dict[str, Any] | None = None) -> Any:
    settings = OpenAIBackendSettings.from_config(config or _config(), environ={})

    async def _run() -> Any:
        return await probe_openai_server(settings, transport=httpx.MockTransport(handler))

    return anyio.run(_run)


def _server(health: int = 200, models: list[str] | None = None) -> Any:
    seen: list[str] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == "/health":
            return httpx.Response(health)
        if request.url.path == "/v1/models":
            ids = models if models is not None else ["shipcoat-v6-1-fp8"]
            return httpx.Response(200, json={"object": "list", "data": [{"id": name, "object": "model"} for name in ids]})
        return httpx.Response(404)

    _handler.seen = seen  # type: ignore[attr-defined]
    return _handler


def test_probe_reports_healthy_server_serving_the_configured_model() -> None:
    handler = _server()

    probe = _probe(handler)

    assert (probe.health_ok, probe.model_served) == (True, True)
    assert handler.seen == ["/health", "/v1/models"]


def test_probe_reports_model_missing_from_the_server() -> None:
    probe = _probe(_server(models=["some-other-model"]))

    assert (probe.health_ok, probe.model_served) == (True, False)


def test_probe_reports_unhealthy_server() -> None:
    probe = _probe(_server(health=503))

    assert probe.health_ok is False


def test_probe_never_raises_when_server_is_unreachable() -> None:
    probe = _probe(_raising_handler(httpx.ConnectError("refused")))

    assert (probe.health_ok, probe.model_served) == (False, False)
