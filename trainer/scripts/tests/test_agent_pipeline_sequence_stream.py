from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import anyio

from trainer.scripts.agent_pipeline import AgentPipeline, SemanticRouter

PipelineEvent = tuple[str, dict[str, Any]]


def _collect(stream: AsyncIterator[PipelineEvent]) -> list[PipelineEvent]:
    async def _drain() -> list[PipelineEvent]:
        return [event async for event in stream]

    return anyio.run(_drain)


class _ToolAdapter:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.calls: list[tuple[str, dict[str, Any], str | None]] = []

    async def dispatch_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        session_id: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append((tool_name, arguments, session_id))
        return self.result


def _result(*, route: str, tool_calls: list[dict[str, Any]], response: str) -> dict[str, Any]:
    return {
        "route": route,
        "rag_chunks": [],
        "tool_calls": tool_calls,
        "response": response,
        "synthesis_ms": 0,
        "deadline_hit": False,
    }


def test_sequence_stream_dispatches_missing_validation_and_uses_tool_message() -> None:
    pipe = AgentPipeline.__new__(AgentPipeline)
    pipe.router = SemanticRouter()
    adapter = _ToolAdapter(
        {
            "status": "success",
            "result": {
                "next_zone": "A-2",
                "message": "현재 블록(BLK-A) 내 다음 구역: A-2.",
            },
        }
    )
    pipe.tool_adapter = adapter
    pipe.run = lambda **_kwargs: _result(
        route="training_realtime",
        tool_calls=[],
        response="검증 없이 A-3을 칠하세요.",
    )
    environment = {
        "zone_map": [{"zone_id": "A-2"}],
        "current_zone_id": "A-1",
        "structure_type": "block",
    }

    events = _collect(
        pipe.run_streaming(
            instruction="다음 순서가 어떻게 돼?",
            environment=environment,
            session_id="sequence-session",
        )
    )

    assert adapter.calls == [("sequence_validation", environment, "sequence-session")]
    tool_events = [payload for kind, payload in events if kind == "tool_result"]
    assert [event["name"] for event in tool_events] == ["sequence_validation"]
    answer = "".join(payload["text"] for kind, payload in events if kind == "delta")
    assert answer == "현재 블록(BLK-A) 내 다음 구역: A-2."
    assert "A-3" not in answer
    assert events[-1][0] == "complete"
    assert events[-1][1]["tool_call_count"] == 1


def test_sequence_stream_does_not_duplicate_existing_validation() -> None:
    pipe = AgentPipeline.__new__(AgentPipeline)
    pipe.router = SemanticRouter()
    adapter = _ToolAdapter({"status": "success", "result": {"next_zone": "A-3"}})
    pipe.tool_adapter = adapter
    existing_call = {
        "name": "sequence_validation",
        "arguments": {"current_zone_id": "A-1"},
        "result": {"status": "success", "result": {"next_zone": "A-2"}},
    }
    pipe.run = lambda **_kwargs: _result(
        route="training_realtime",
        tool_calls=[existing_call],
        response="A-2를 칠하세요.",
    )

    events = _collect(pipe.run_streaming(instruction="다음 구역", environment={"zone_map": []}))

    assert adapter.calls == []
    assert [payload["name"] for kind, payload in events if kind == "tool_result"] == [
        "sequence_validation"
    ]
    assert events[-1][1]["tool_call_count"] == 1


def test_non_sequence_realtime_stream_does_not_dispatch_sequence_validation() -> None:
    pipe = AgentPipeline.__new__(AgentPipeline)
    pipe.router = SemanticRouter()
    adapter = _ToolAdapter({"status": "success", "result": {"next_zone": "A-2"}})
    pipe.tool_adapter = adapter
    pipe.run = lambda **_kwargs: _result(
        route="training_realtime",
        tool_calls=[],
        response="도막 상태를 확인하세요.",
    )

    events = _collect(pipe.run_streaming(instruction="지금 도막 상태 어때?", environment={}))

    assert adapter.calls == []
    assert all(kind != "tool_result" for kind, _payload in events)
    assert events[-1][1]["tool_call_count"] == 0


def test_sequence_stream_stops_when_fallback_validation_times_out() -> None:
    pipe = AgentPipeline.__new__(AgentPipeline)
    pipe.router = SemanticRouter()
    adapter = _ToolAdapter(
        {
            "status": "error",
            "error": {"code": "TIMEOUT", "message": "internal MCP timeout detail"},
        }
    )
    pipe.tool_adapter = adapter
    pipe.run = lambda **_kwargs: _result(
        route="training_realtime",
        tool_calls=[],
        response="검증되지 않은 답변",
    )

    events = _collect(pipe.run_streaming(instruction="다음 순서", environment={"zone_map": []}))

    assert [kind for kind, _payload in events] == ["status", "tool_result", "error"]
    assert events[-1][1]["message"] == "sequence validation failed"
    assert events[-1][1]["retryable"] is True
    assert "internal MCP timeout detail" not in str(events[-1][1])
