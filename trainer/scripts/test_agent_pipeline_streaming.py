#!/usr/bin/env python3
"""Bounded, deterministic regression tests for AgentPipeline streaming (no GPU).

Locks the three streaming-path quality defects fixed alongside these tests:

  * ``_stream_generate`` must not hang when ``model.generate`` raises. The
    worker has to unblock the ``TextIteratorStreamer`` (whose ``queue.get``
    uses ``timeout=None``), be joined, and re-raise the original error.
  * ``run_streaming`` buffered-success event contract: exact
    status/tool_result/delta/complete order plus timing and count fields.
  * ``run_streaming`` error boundary: blank input is non-retryable; exceptions
    escaping ``run`` are logged and surfaced as generic retryable errors. All
    error events carry ``ttft_ms=None`` and an integer ``total_ms``.

The pipeline is built via ``AgentPipeline.__new__`` so no model, RAG index, or
MCP backend is loaded; ``run``/``model``/``tokenizer`` are replaced with fakes.
``agent_pipeline`` is imported dynamically to match the sibling test convention
and avoid basedpyright's implicit-relative-import error.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import sys
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

# Add parent to path for imports (mirrors sibling test modules).
sys.path.insert(0, str(Path(__file__).parent))

agent_pipeline = importlib.import_module("agent_pipeline")
AgentPipeline = agent_pipeline.AgentPipeline


def _collect(agen: AsyncIterator[tuple[str, dict[str, Any]]]) -> list[tuple[str, dict[str, Any]]]:
    """Drain an async generator to a list on a private event loop."""

    async def _drain() -> list[tuple[str, dict[str, Any]]]:
        return [event async for event in agen]

    return asyncio.run(_drain())


class _DummyTokenizer:
    """Minimal tokenizer stand-in: callable and exposes ``eos_token_id``."""

    eos_token_id = 0

    def __call__(self, _texts: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"input_ids": [[1, 2, 3]]}


class _RaisingModel:
    """Fake model whose ``generate`` raises before producing any token."""

    def generate(self, **_kwargs: Any) -> None:
        raise RuntimeError("boom-in-generate")


def test_stream_generate_terminates_and_reraises_when_generate_raises() -> None:
    """Given generate() raises, When consumed, Then it unblocks and re-raises (no hang)."""
    pipe = AgentPipeline.__new__(AgentPipeline)
    pipe.tokenizer = _DummyTokenizer()
    pipe.model = _RaisingModel()

    captured: dict[str, BaseException] = {}
    done = threading.Event()

    def _consume() -> None:
        try:
            list(pipe._stream_generate("hi", {"do_sample": False}, 8))
        except Exception as exc:  # SUT is expected to re-raise the worker error
            captured["error"] = exc
        finally:
            done.set()

    runner = threading.Thread(target=_consume, daemon=True)
    runner.start()
    finished = done.wait(timeout=5.0)

    assert finished, "_stream_generate hung: streamer never received a stop signal"
    assert isinstance(captured.get("error"), RuntimeError)
    assert str(captured["error"]) == "boom-in-generate"


def _fake_run_success(**_kwargs: Any) -> dict[str, Any]:
    return {
        "rag_chunks": [{"chunk_id": "c1"}, {"chunk_id": "c2"}],
        "tool_calls": [
            {"name": "dft_judge", "arguments": {"nominal": 200}, "result": {"verdict": "pass"}},
        ],
        "response": "A" * 24 + "B" * 6,
        "synthesis_ms": 42,
        "deadline_hit": False,
    }


def test_run_streaming_success_emits_exact_event_contract() -> None:
    """Given a buffered run(), When streamed, Then exact status/tool_result/delta/complete order."""
    pipe = AgentPipeline.__new__(AgentPipeline)
    pipe.run = _fake_run_success

    events = _collect(pipe.run_streaming(instruction="hi"))

    assert [kind for kind, _ in events] == ["status", "tool_result", "delta", "delta", "complete"]
    assert events[0] == ("status", {"stage": "rag", "chunks": 2})
    assert events[1] == (
        "tool_result",
        {"name": "dft_judge", "arguments": {"nominal": 200}, "result": {"verdict": "pass"}},
    )
    assert events[2] == ("delta", {"text": "A" * 24})
    assert events[3] == ("delta", {"text": "B" * 6})

    kind, payload = events[4]
    assert kind == "complete"
    assert payload["synthesis_ms"] == 42
    assert payload["tool_call_count"] == 1
    assert payload["rag_chunk_count"] == 2
    assert payload["deadline_hit"] is False
    assert isinstance(payload["ttft_ms"], int)
    assert payload["ttft_ms"] >= 0
    assert isinstance(payload["total_ms"], int)
    assert payload["total_ms"] >= 0


def test_run_streaming_blank_instruction_rejected_without_invoking_run() -> None:
    """Given a blank instruction, When streamed, Then a non-retryable error and run() is never invoked."""
    pipe = AgentPipeline.__new__(AgentPipeline)
    run_calls: list[dict[str, Any]] = []

    def _tracking_run(**kwargs: Any) -> dict[str, Any]:
        run_calls.append(kwargs)
        return {"rag_chunks": [], "tool_calls": [], "response": "", "synthesis_ms": 0, "deadline_hit": False}

    pipe.run = _tracking_run

    events = _collect(pipe.run_streaming(instruction="   "))

    assert run_calls == []
    assert len(events) == 1
    kind, payload = events[0]
    assert kind == "error"
    assert payload["message"] == "instruction must not be empty"
    assert payload["retryable"] is False
    assert payload["ttft_ms"] is None
    assert isinstance(payload["total_ms"], int)
    assert payload["total_ms"] >= 0


def test_run_streaming_unexpected_error_is_generic_retryable_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Given run() raises RuntimeError, When streamed, Then generic retryable error; detail only in logs."""
    pipe = AgentPipeline.__new__(AgentPipeline)
    secret = "SECRET-INTERNAL-STACK-9f3c"

    def _raise_runtime(**_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError(secret)

    pipe.run = _raise_runtime

    with caplog.at_level(logging.ERROR):
        events = _collect(pipe.run_streaming(instruction="hi"))

    assert len(events) == 1
    kind, payload = events[0]
    assert kind == "error"
    assert payload["message"] == "streaming pipeline failed"
    assert secret not in payload["message"]
    assert payload["retryable"] is True
    assert payload["ttft_ms"] is None
    assert isinstance(payload["total_ms"], int)
    assert payload["total_ms"] >= 0
    assert secret in caplog.text


def test_run_streaming_valueerror_from_run_is_generic_retryable_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Given nonblank run() raises an internal ValueError, When streamed, Then generic retryable error; no leak."""
    pipe = AgentPipeline.__new__(AgentPipeline)
    secret = "secret config path"

    def _raise_value(**_kwargs: Any) -> dict[str, Any]:
        raise ValueError(secret)

    pipe.run = _raise_value

    with caplog.at_level(logging.ERROR):
        events = _collect(pipe.run_streaming(instruction="paint zone 3"))

    assert len(events) == 1
    kind, payload = events[0]
    assert kind == "error"
    assert payload["message"] == "streaming pipeline failed"
    assert secret not in str(payload)
    assert payload["retryable"] is True
    assert payload["ttft_ms"] is None
    assert isinstance(payload["total_ms"], int)
    assert payload["total_ms"] >= 0
    assert secret in caplog.text
    assert any(record.levelno == logging.ERROR for record in caplog.records)
    assert any("unexpected failure" in record.getMessage() for record in caplog.records)
