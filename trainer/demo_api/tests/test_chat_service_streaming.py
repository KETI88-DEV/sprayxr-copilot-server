from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TypeVar

from trainer.demo_api.chat_service import ChatService, LiveRunner
from trainer.demo_api.contracts import ChatRequest

T = TypeVar("T")

PipelineEvent = tuple[str, dict[str, object]]


def _collect(agen: AsyncIterator[T]) -> list[T]:
    async def _run() -> list[T]:
        return [event async for event in agen]

    return asyncio.run(_run())


def _runner_yielding(events: list[PipelineEvent]) -> LiveRunner:
    """Build a fake live runner that re-emits a fixed pipeline (kind, payload) stream."""

    async def _runner(_request: ChatRequest) -> AsyncIterator[PipelineEvent]:
        for kind, payload in events:
            yield kind, payload

    return _runner


def test_live_forwards_pipeline_events_in_order_with_session_first() -> None:
    """Given a live async stream, When consumed, Then session leads and events forward in order."""
    events: list[PipelineEvent] = [
        ("status", {"stage": "rag", "chunks": 2}),
        ("tool_result", {"name": "dft_judge", "arguments": {}, "result": {"verdict": "pass"}}),
        ("delta", {"text": "안녕"}),
        ("delta", {"text": "하세요"}),
        (
            "complete",
            {
                "ttft_ms": 5,
                "synthesis_ms": 10,
                "total_ms": 20,
                "tool_call_count": 1,
                "rag_chunk_count": 2,
                "deadline_hit": False,
            },
        ),
    ]
    request = ChatRequest(message="상태 알려줘", session_id="s-live", mode="live")

    out = _collect(ChatService(live_runner=_runner_yielding(events)).stream(request))

    names = [event.event for event in out]
    assert names == [
        "session",
        "status",
        "tool_result",
        "message_delta",
        "message_delta",
        "complete",
    ]
    assert out[0].data == {"session_id": "s-live", "mode": "live"}


def test_live_delta_maps_to_message_delta_preserving_text() -> None:
    """Given a pipeline delta, When forwarded, Then it becomes message_delta with identical text."""
    events: list[PipelineEvent] = [("delta", {"text": "chunk-0"})]
    request = ChatRequest(message="hi", session_id="s-1", mode="live")

    out = _collect(ChatService(live_runner=_runner_yielding(events)).stream(request))

    deltas = [event for event in out if event.event == "message_delta"]
    assert len(deltas) == 1
    assert deltas[0].data == {"text": "chunk-0"}
    assert not any(event.event == "delta" for event in out)


def test_live_complete_retains_timing_counts_and_deadline() -> None:
    """Given a complete payload, When forwarded, Then timing/count/deadline fields are untouched."""
    complete_payload: dict[str, object] = {
        "ttft_ms": 123,
        "synthesis_ms": 456,
        "total_ms": 789,
        "tool_call_count": 3,
        "rag_chunk_count": 5,
        "deadline_hit": True,
    }
    events: list[PipelineEvent] = [("delta", {"text": "x"}), ("complete", complete_payload)]
    request = ChatRequest(message="hi", session_id="s-1", mode="live")

    out = _collect(ChatService(live_runner=_runner_yielding(events)).stream(request))

    complete = next(event for event in out if event.event == "complete")
    assert complete.data == complete_payload


def test_live_error_event_timing_fields_survive() -> None:
    """Given a pipeline error event, When forwarded, Then ttft_ms/total_ms survive verbatim."""
    error_payload: dict[str, object] = {
        "message": "streaming pipeline failed",
        "retryable": True,
        "ttft_ms": None,
        "total_ms": 42,
    }
    events: list[PipelineEvent] = [("error", error_payload)]
    request = ChatRequest(message="hi", session_id="s-1", mode="live")

    out = _collect(ChatService(live_runner=_runner_yielding(events)).stream(request))

    assert out[0].event == "session"
    assert out[-1].event == "error"
    assert out[-1].data == error_payload


def test_live_sequence_update_passes_through_unchanged() -> None:
    """Given a pipeline sequence_update, When forwarded, Then name and payload are unchanged."""
    seq_payload: dict[str, object] = {
        "type": "update_sequence",
        "level_id": "SP0035-L01",
        "current_zone_id": "A-1",
        "next_zone_ids": ["A-2"],
        "completed_zone_ids": ["A-1"],
    }
    events: list[PipelineEvent] = [
        ("sequence_update", seq_payload),
        (
            "complete",
            {
                "ttft_ms": 1,
                "synthesis_ms": 1,
                "total_ms": 1,
                "tool_call_count": 0,
                "rag_chunk_count": 0,
                "deadline_hit": False,
            },
        ),
    ]
    request = ChatRequest(message="hi", session_id="s-1", mode="live")

    out = _collect(ChatService(live_runner=_runner_yielding(events)).stream(request))

    seq = next(event for event in out if event.event == "sequence_update")
    assert seq.data == seq_payload


def test_demo_complete_carries_locally_measured_timing() -> None:
    """Given demo mode, When streamed, Then complete carries local ttft_ms/synthesis_ms/total_ms/deadline_hit."""
    request = ChatRequest(message="에폭시 도료란 무엇인가?", session_id="s-demo")

    out = _collect(ChatService().stream(request))

    complete = next(event for event in out if event.event == "complete")
    for key in ("ttft_ms", "synthesis_ms", "total_ms"):
        assert key in complete.data, f"demo complete missing {key}"
        value = complete.data[key]
        assert isinstance(value, int)
        assert value >= 0
    assert "tool_call_count" in complete.data
    assert "rag_chunk_count" in complete.data
    assert "deadline_hit" in complete.data, "demo complete missing deadline_hit"
    assert complete.data["deadline_hit"] is False
