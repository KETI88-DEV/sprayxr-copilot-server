from __future__ import annotations

import asyncio
from typing import TypeVar

from collections.abc import AsyncIterator

from trainer.demo_api.chat_service import ChatService
from trainer.demo_api.contracts import ChatRequest

T = TypeVar("T")


def _collect(agen: AsyncIterator[T]) -> list[T]:
    async def _run() -> list[T]:
        return [event async for event in agen]

    return asyncio.run(_run())


def test_sequence_demo_emits_update_before_complete() -> None:
    request = ChatRequest(message="다음에 어떤 구역을 칠해야 해?", session_id="s-1")
    events = _collect(ChatService().stream(request))
    names = [event.event for event in events]
    assert names[0] == "session"
    assert "sequence_update" in names
    assert names[-1] == "complete"
    update = next(event for event in events if event.event == "sequence_update")
    assert update.data["next_zone_ids"] == ["A-2", "A-3"]
    assert update.data["current_zone_id"] == "A-1"
    assert update.data["completed_zone_ids"] == ["A-1"]


def test_live_without_runner_emits_error() -> None:
    request = ChatRequest(message="hi", session_id="s-1", mode="live")
    events = _collect(ChatService().stream(request))
    assert events[-1].event == "error"
    assert events[-1].data["retryable"] is False
