from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator
from typing import TypeVar

import pytest

from trainer.demo_api import chat_service as chat_service_mod
from trainer.demo_api.chat_service import ChatService, LiveRunner
from trainer.demo_api.contracts import ChatRequest

T = TypeVar("T")

PipelineEvent = tuple[str, dict[str, object]]


def _collect(agen: AsyncIterator[T]) -> list[T]:
    async def _run() -> list[T]:
        return [event async for event in agen]

    return asyncio.run(_run())


class _FakePipeline:
    async def run_streaming(
        self,
        *,
        instruction: str,
        session_id: str,
        enable_thinking: bool,
        environment: dict[str, object],
    ) -> AsyncIterator[PipelineEvent]:
        _ = (instruction, session_id, enable_thinking, environment)
        yield "complete", {
            "ttft_ms": 1,
            "synthesis_ms": 1,
            "total_ms": 1,
            "tool_call_count": 0,
            "rag_chunk_count": 0,
            "deadline_hit": False,
        }


def _raising_runner(exc: Exception) -> LiveRunner:
    async def _runner(_request: ChatRequest) -> AsyncIterator[PipelineEvent]:
        yield "status", {"stage": "rag", "chunks": 0}
        raise exc

    return _runner


def test_default_runner_constructs_pipeline_off_event_loop_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given the default live runner, When first driven, Then construction runs off the loop thread."""
    construction_threads: list[int] = []

    def _fake_construct() -> object:
        construction_threads.append(threading.get_ident())
        return _FakePipeline()

    monkeypatch.setattr(chat_service_mod, "_pipeline_singleton", None)
    monkeypatch.setattr(chat_service_mod, "_construct_default_pipeline", _fake_construct)

    runner = chat_service_mod.build_default_live_runner()
    request = ChatRequest(message="hi", session_id="s-1", mode="live")

    loop_thread = threading.get_ident()
    out = _collect(ChatService(live_runner=runner).stream(request))

    assert [event.event for event in out] == ["session", "complete"]
    assert len(construction_threads) == 1
    assert construction_threads[0] != loop_thread


def test_default_runner_initializes_pipeline_once_under_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given concurrent first live requests, When they race, Then construction runs once, off-loop."""
    calls: list[int] = []
    entered = threading.Event()
    release = threading.Event()

    def _fake_construct() -> object:
        calls.append(threading.get_ident())
        entered.set()
        _ = release.wait(timeout=5.0)
        return _FakePipeline()

    monkeypatch.setattr(chat_service_mod, "_pipeline_singleton", None)
    monkeypatch.setattr(chat_service_mod, "_construct_default_pipeline", _fake_construct)

    runner = chat_service_mod.build_default_live_runner()
    loop_thread = threading.get_ident()

    async def _one() -> list[object]:
        request = ChatRequest(message="hi", session_id="s", mode="live")
        return [event async for event in ChatService(live_runner=runner).stream(request)]

    async def _drive() -> list[list[object]]:
        tasks = [asyncio.create_task(_one()) for _ in range(5)]
        entered_ok = await asyncio.to_thread(entered.wait, 5.0)
        assert entered_ok, "constructor never entered a worker thread"
        assert not any(task.done() for task in tasks)
        release.set()
        return list(await asyncio.gather(*tasks))

    results = asyncio.run(_drive())

    assert len(results) == 5
    assert all(len(result) == 2 for result in results)
    assert len(calls) == 1
    assert calls[0] != loop_thread


def test_live_adapter_exception_is_sanitized_logged_timed_and_retryable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Given the runner raises, When streamed, Then a generic retryable timed error; secret only in logs."""
    secret = "SECRET-ADAPTER-STACK-42"
    request = ChatRequest(message="hi", session_id="s-1", mode="live")

    with caplog.at_level(logging.ERROR):
        out = _collect(ChatService(live_runner=_raising_runner(RuntimeError(secret))).stream(request))

    assert out[0].event == "session"
    err = out[-1]
    assert err.event == "error"
    assert err.data["message"] == "streaming pipeline failed"
    assert secret not in str(err.data)
    assert err.data["retryable"] is True
    assert err.data["ttft_ms"] is None
    assert isinstance(err.data["total_ms"], int)
    assert err.data["total_ms"] >= 0
    assert secret in caplog.text


def test_live_construction_failure_is_sanitized_logged_and_retryable(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Given pipeline construction raises, When streamed, Then a generic retryable timed error; secret only in logs."""
    secret = "SECRET-MODEL-LOAD-PATH"

    def _boom() -> object:
        raise RuntimeError(secret)

    monkeypatch.setattr(chat_service_mod, "_pipeline_singleton", None)
    monkeypatch.setattr(chat_service_mod, "_construct_default_pipeline", _boom)

    runner = chat_service_mod.build_default_live_runner()
    request = ChatRequest(message="hi", session_id="s-1", mode="live")
    with caplog.at_level(logging.ERROR):
        out = _collect(ChatService(live_runner=runner).stream(request))

    err = out[-1]
    assert err.event == "error"
    assert err.data["message"] == "streaming pipeline failed"
    assert secret not in str(err.data)
    assert err.data["retryable"] is True
    assert err.data["ttft_ms"] is None
    assert isinstance(err.data["total_ms"], int)
    assert secret in caplog.text


def test_live_unknown_event_kind_is_generic_nonretryable_and_timed() -> None:
    """Given an unknown pipeline kind, When streamed, Then a generic non-retryable timed error; kind not leaked."""
    leaky_kind = "SECRET_KIND_LEAK_9000"

    async def _runner(_request: ChatRequest) -> AsyncIterator[PipelineEvent]:
        yield leaky_kind, {"text": "x"}

    request = ChatRequest(message="hi", session_id="s-1", mode="live")
    out = _collect(ChatService(live_runner=_runner).stream(request))

    err = out[-1]
    assert err.event == "error"
    assert err.data["message"] == "invalid streaming response"
    assert leaky_kind not in str(err.data)
    assert err.data["retryable"] is False
    assert err.data["ttft_ms"] is None
    assert isinstance(err.data["total_ms"], int)
    assert err.data["total_ms"] >= 0


def test_live_pipeline_error_payload_is_preserved_verbatim() -> None:
    """Given a pipeline-originated error event, When streamed, Then it passes through untouched."""
    pipeline_error: dict[str, object] = {
        "message": "streaming pipeline failed",
        "retryable": True,
        "ttft_ms": None,
        "total_ms": 55,
    }

    async def _runner(_request: ChatRequest) -> AsyncIterator[PipelineEvent]:
        yield "error", pipeline_error

    request = ChatRequest(message="hi", session_id="s-1", mode="live")
    out = _collect(ChatService(live_runner=_runner).stream(request))

    assert out[-1].event == "error"
    assert out[-1].data == pipeline_error
