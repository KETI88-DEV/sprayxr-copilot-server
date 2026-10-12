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
    ambient = dict(default_tool_environment("TR-L01"))

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
        # The scaffold cites this chunk, so search must have returned it; a citation with
        # nothing retrieved is a state run() cannot produce.
        "rag_chunks": [{"chunk_id": "DOC-doc_0086-C0001", "text": ""}],
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


def _pipeline_by_route(tool_calls_on_diagnostic: list[dict[str, Any]]) -> tuple[AgentPipeline, list[str]]:
    """진단 경로에서는 주어진 도구 호출을, fast 경로에서는 지식 답을 낸다. 부른 경로를 기록한다."""
    pipe = AgentPipeline.__new__(AgentPipeline)
    pipe.router = SemanticRouter()
    routes: list[str] = []

    def _fake_run(**kwargs: Any) -> dict[str, Any]:
        route = kwargs["force_route"]
        routes.append(route)
        return {
            "route": route,
            "rag_chunks": [{"chunk_id": "DOC-doc_0086-C0001", "text": ""}],
            "tool_calls": tool_calls_on_diagnostic if route != "fast" else [],
            "response": _DIAGNOSTIC_SCAFFOLD if route != "fast" else "[분석] KNOWLEDGE_ANSWER\n[근거] DOC-doc_0086-C0001",
            "synthesis_ms": 0,
            "deadline_hit": False,
        }

    pipe.run = _fake_run
    return pipe, routes


def test_toolless_diagnostic_answer_is_re_answered_on_the_fast_route() -> None:
    """Given a diagnostic route that called no tool, When streamed, Then it is re-answered as knowledge.

    도구 결과가 없으면 [판정]·[원인]은 데이터 없이 채운 틀이다. 2026-10-12 데모에서
    "측정값 편차 또는 작업 조건 이탈 가능성 있음"·"anomaly_detection 도구로 … 생성합니다"가 그대로 나왔다.
    """
    pipe, routes = _pipeline_by_route([])

    events = _collect(pipe.run_streaming(instruction="도막 상태 판정해줘", environment={}))
    answer = "".join(payload["text"] for kind, payload in events if kind == "delta")

    assert routes == ["training_realtime", "fast"]
    assert "KNOWLEDGE_ANSWER" in answer
    assert "SENTINEL_VERDICT" not in answer
    assert answer.startswith("[답변]")


def test_diagnostic_answer_with_tool_results_is_streamed_unchanged() -> None:
    """Given a diagnostic route whose tools ran, When streamed, Then every section survives — 대조군."""
    tool = {"name": "anomaly_detection", "arguments": {}, "result": {"status": "success", "result": {}}}
    pipe, routes = _pipeline_by_route([tool])

    events = _collect(pipe.run_streaming(instruction="도막 상태 판정해줘", environment={}))
    answer = "".join(payload["text"] for kind, payload in events if kind == "delta")

    assert routes == ["training_realtime"]
    assert answer == _DIAGNOSTIC_SCAFFOLD


def test_live_stream_keeps_sequence_question_on_realtime_route_with_same_environment() -> None:
    """Given the same ambient context, When the sequence is asked, Then the realtime route is kept."""
    existing_call = {
        "name": "sequence_validation",
        "arguments": {"current_zone_id": "A-1"},
        "result": {"status": "success", "result": {"next_zone": "A-2"}},
    }
    pipe, captured = _pipeline_capturing_route("training_realtime", [existing_call])
    ambient = dict(default_tool_environment("TR-L01"))

    _ = _collect(pipe.run_streaming(instruction="다음 순서가 어떻게 돼?", environment=ambient))

    assert captured["force_route"] == "training_realtime"
