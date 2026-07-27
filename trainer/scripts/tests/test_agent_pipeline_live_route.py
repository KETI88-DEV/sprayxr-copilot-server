from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import anyio

from trainer.demo_api.sequence import default_tool_environment
from trainer.scripts.agent_pipeline import AgentPipeline, SemanticRouter

PipelineEvent = tuple[str, dict[str, Any]]


def _collect(stream: AsyncIterator[PipelineEvent]) -> list[PipelineEvent]:
    async def _drain() -> list[PipelineEvent]:
        return [event async for event in stream]

    return anyio.run(_drain)


def _pipeline_capturing_route(route: str, tool_calls: list[dict[str, Any]]) -> tuple[AgentPipeline, dict[str, Any]]:
    pipe = AgentPipeline.__new__(AgentPipeline)
    pipe.router = SemanticRouter()
    captured: dict[str, Any] = {}

    def _fake_run(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "route": route,
            "rag_chunks": [],
            "tool_calls": tool_calls,
            "response": "답변",
            "synthesis_ms": 0,
            "deadline_hit": False,
        }

    pipe.run = _fake_run
    return pipe, captured


def test_live_stream_routes_knowledge_question_to_fast_despite_ambient_environment() -> None:
    """Given ambient sequence context, When a definition is asked, Then the fast route is selected."""
    pipe, captured = _pipeline_capturing_route("fast", [])
    ambient = dict(default_tool_environment("SP0035-L01"))

    _ = _collect(pipe.run_streaming(instruction="블라스팅이 뭐야", environment=ambient))

    assert captured["force_route"] == "fast"


_DIAGNOSTIC_SCAFFOLD = "\n".join(
    [
        "[분석] SENTINEL_DEFINITION",
        "[판정] SENTINEL_VERDICT",
        "[원인] SENTINEL_CAUSE",
        "[조치]",
        "1. SENTINEL_ACTION",
        "[근거] DOC-doc_0086-C0001",
    ]
)


def _stream_answer(route: str, instruction: str) -> str:
    pipe = AgentPipeline.__new__(AgentPipeline)
    pipe.router = SemanticRouter()
    pipe.run = lambda **_kwargs: {
        "route": route,
        "rag_chunks": [],
        "tool_calls": [],
        "response": _DIAGNOSTIC_SCAFFOLD,
        "synthesis_ms": 0,
        "deadline_hit": False,
    }
    events = _collect(pipe.run_streaming(instruction=instruction, environment={}))
    return "".join(payload["text"] for kind, payload in events if kind == "delta")


def test_fast_route_answer_drops_unsupported_verdict_cause_and_action() -> None:
    """Given a toolless fast answer, When streamed, Then only answer and evidence reach the client."""
    answer = _stream_answer("fast", "블라스팅이 뭐야")

    assert "SENTINEL_DEFINITION" in answer
    assert "DOC-doc_0086-C0001" in answer
    assert "SENTINEL_VERDICT" not in answer
    assert "SENTINEL_CAUSE" not in answer
    assert "SENTINEL_ACTION" not in answer


def test_diagnostic_route_answer_is_streamed_unchanged() -> None:
    """Given the same scaffold on a diagnostic route, When streamed, Then every section survives."""
    answer = _stream_answer("training_realtime", "도막 상태 판정해줘")

    assert answer == _DIAGNOSTIC_SCAFFOLD


def test_live_stream_keeps_sequence_question_on_realtime_route_with_same_environment() -> None:
    """Given the same ambient context, When the sequence is asked, Then the realtime route is kept."""
    existing_call = {
        "name": "sequence_validation",
        "arguments": {"current_zone_id": "A-1"},
        "result": {"status": "success", "result": {"next_zone": "A-2"}},
    }
    pipe, captured = _pipeline_capturing_route("training_realtime", [existing_call])
    ambient = dict(default_tool_environment("SP0035-L01"))

    _ = _collect(pipe.run_streaming(instruction="다음 순서가 어떻게 돼?", environment=ambient))

    assert captured["force_route"] == "training_realtime"
