from __future__ import annotations

import asyncio  # noqa: ANYIO_OK
import threading
import time
from collections.abc import AsyncIterator
from typing import Any

import pytest

from trainer.scripts import agent_pipeline
from trainer.scripts.agent_pipeline import AgentPipeline

PipelineEvent = tuple[str, dict[str, Any]]


def _collect(agen: AsyncIterator[PipelineEvent]) -> list[PipelineEvent]:
    async def _drain() -> list[PipelineEvent]:
        return [event async for event in agen]

    return asyncio.run(_drain())


def _finished_run() -> dict[str, Any]:
    return {
        "rag_chunks": [],
        "tool_calls": [],
        "response": "ok",
        "synthesis_ms": 0,
        "deadline_hit": False,
    }


def test_run_streaming_reports_progress_while_run_is_slow(monkeypatch: pytest.MonkeyPatch) -> None:
    pipe = AgentPipeline.__new__(AgentPipeline)
    pipe.router = agent_pipeline.SemanticRouter()

    def _slow_run(**_kwargs: Any) -> dict[str, Any]:
        time.sleep(0.35)
        return _finished_run()

    pipe.run = _slow_run
    monkeypatch.setattr(agent_pipeline, "_PROGRESS_INTERVAL_SEC", 0.1)

    events = _collect(pipe.run_streaming(instruction="도막 기준"))

    progress = [
        payload for kind, payload in events if kind == "status" and payload["stage"] == "generating"
    ]
    assert len(progress) >= 2
    assert all(isinstance(payload["elapsed_sec"], int) for payload in progress)
    assert any(kind == "status" and payload["stage"] == "rag" for kind, payload in events)
    assert events[-1][0] == "complete"


def test_run_streaming_serializes_generation_across_concurrent_requests() -> None:
    pipe = AgentPipeline.__new__(AgentPipeline)
    pipe.router = agent_pipeline.SemanticRouter()
    guard = threading.Lock()
    active = 0
    peak = 0

    def _tracking_run(**_kwargs: Any) -> dict[str, Any]:
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
        time.sleep(0.15)
        with guard:
            active -= 1
        return _finished_run()

    pipe.run = _tracking_run

    async def _drain() -> None:
        async def _one() -> list[PipelineEvent]:
            return [event async for event in pipe.run_streaming(instruction="도막 기준")]

        _ = await asyncio.gather(_one(), _one(), _one())

    asyncio.run(_drain())

    assert peak == 1
