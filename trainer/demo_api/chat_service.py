# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Final, Protocol

from anyio import to_thread
from pydantic import ValidationError

from trainer.demo_api.contracts import ChatRequest, SequenceUpdate, StreamEvent
from trainer.demo_api.scenario_adapter import get_demo_scenario

logger = logging.getLogger(__name__)

# kind ∈ status|tool_result|delta|complete|error (sequence_update also allowed).
LiveRunner = Callable[[ChatRequest], AsyncIterator[tuple[str, dict[str, object]]]]

_CONFIG_PATH_DEFAULT: Final[str] = "/workspace/trainer/configs/rag_config_fewshot.yaml"
_ADAPTER_PATH_DEFAULT: Final[str] = "/output/adapters/qwen3_4b_shipcoat_lora/"
_MCP_BASE_URL_DEFAULT: Final[str] = "http://localhost:8100"


class _Pipeline(Protocol):
    def run_streaming(
        self,
        *,
        instruction: str,
        session_id: str,
        enable_thinking: bool,
        environment: dict[str, object],
    ) -> AsyncIterator[tuple[str, dict[str, object]]]:
        ...


_pipeline_singleton: _Pipeline | None = None
_pipeline_lock = threading.Lock()


class _StreamProtocolError(Exception):
    pass


class ChatService:
    def __init__(self, live_runner: LiveRunner | None = None) -> None:
        self.live_runner: LiveRunner | None = live_runner

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        t0 = time.monotonic()
        live_runner = self.live_runner
        if request.mode == "live" and live_runner is None:
            yield StreamEvent(
                event="error",
                data={"message": "live pipeline is not configured", "retryable": False},
            )
            return

        yield StreamEvent(event="session", data={"session_id": request.session_id, "mode": request.mode})

        if request.mode == "live":
            runner = live_runner
            assert runner is not None
            try:
                async for kind, payload in runner(request):
                    yield _to_stream_event(kind, payload)
            except (_StreamProtocolError, ValidationError):
                logger.exception("live stream: pipeline protocol violation")
                yield _boundary_error(t0, "invalid streaming response", retryable=False)
            except Exception:  # noqa: BROAD_EXCEPT_OK
                logger.exception("live stream: unexpected pipeline failure")
                yield _boundary_error(t0, "streaming pipeline failed", retryable=True)
            return

        scenario = _demo_scenario(request.message)
        for event in _iter_stream_events(request, scenario):
            yield event


def _construct_default_pipeline() -> _Pipeline:
    from trainer.scripts.agent_pipeline import AgentPipeline

    return AgentPipeline(
        config_path=os.environ.get("CONFIG_PATH", _CONFIG_PATH_DEFAULT),
        adapter_path=os.environ.get("ADAPTER_PATH", _ADAPTER_PATH_DEFAULT),
        mcp_base_url=os.environ.get("MCP_BASE_URL", _MCP_BASE_URL_DEFAULT),
    )


def _get_or_build_pipeline() -> _Pipeline:
    global _pipeline_singleton
    with _pipeline_lock:
        if _pipeline_singleton is None:
            _pipeline_singleton = _construct_default_pipeline()
        return _pipeline_singleton


def build_default_live_runner() -> LiveRunner:
    async def _runner(request: ChatRequest) -> AsyncIterator[tuple[str, dict[str, object]]]:
        pipeline = await to_thread.run_sync(_get_or_build_pipeline)
        async for event in pipeline.run_streaming(
            instruction=request.message,
            session_id=request.session_id,
            enable_thinking=False,
            environment={},
        ):
            yield event

    return _runner


def _boundary_error(t0: float, message: str, *, retryable: bool) -> StreamEvent:
    return StreamEvent(
        event="error",
        data={
            "message": message,
            "retryable": retryable,
            "ttft_ms": None,
            "total_ms": int((time.monotonic() - t0) * 1000),
        },
    )


def _to_stream_event(kind: str, payload: dict[str, object]) -> StreamEvent:
    match kind:
        case "delta":
            return StreamEvent(event="message_delta", data=payload)
        case "status" | "tool_result" | "sequence_update" | "complete" | "error":
            return StreamEvent(event=kind, data=payload)
        case _:
            raise _StreamProtocolError(f"unexpected pipeline event kind: {kind!r}")


def _iter_stream_events(request: ChatRequest, scenario: dict[str, object]) -> Iterator[StreamEvent]:
    t0 = time.monotonic()

    rag_chunks = scenario.get("rag_chunks") or []
    if not isinstance(rag_chunks, list):
        rag_chunks = []
    yield StreamEvent(
        event="status",
        data={"stage": "rag", "chunks": len(rag_chunks)},
    )

    tool_calls = scenario.get("tool_calls") or []
    if not isinstance(tool_calls, list):
        tool_calls = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        arguments = tool_call.get("args") or tool_call.get("arguments") or {}
        result = tool_call.get("result") or {}
        if not isinstance(arguments, dict):
            arguments = {}
        if not isinstance(result, dict):
            result = {}
        yield StreamEvent(
            event="tool_result",
            data={
                "name": str(tool_call.get("name", "")),
                "arguments": arguments,
                "result": result,
            },
        )

    xr_commands = scenario.get("xr_commands")
    if isinstance(xr_commands, dict):
        next_path = xr_commands.get("next_path")
        steps = xr_commands.get("steps") or []
        if isinstance(next_path, dict) and isinstance(steps, list):
            update = _sequence_update_from_payload(request.level_id, next_path, steps)
            if update is not None:
                yield StreamEvent(
                    event="sequence_update",
                    data=update.model_dump(mode="json"),
                )

    ttft_ms: int | None = None
    synthesis_start = time.monotonic()
    response_obj = scenario.get("response", "")
    response = response_obj if isinstance(response_obj, str) else ""
    if response:
        for index in range(0, len(response), 24):
            if ttft_ms is None:
                ttft_ms = int((time.monotonic() - t0) * 1000)
            yield StreamEvent(event="message_delta", data={"text": response[index : index + 24]})
    else:
        ttft_ms = int((time.monotonic() - t0) * 1000)
        yield StreamEvent(event="message_delta", data={"text": ""})

    yield StreamEvent(
        event="complete",
        data={
            "ttft_ms": ttft_ms,
            "synthesis_ms": int((time.monotonic() - synthesis_start) * 1000),
            "total_ms": int((time.monotonic() - t0) * 1000),
            "tool_call_count": len(tool_calls),
            "rag_chunk_count": len(rag_chunks),
            "deadline_hit": False,
        },
    )


def _demo_scenario(message: str) -> dict[str, object]:
    scenario = get_demo_scenario(message)
    normalized: dict[str, object] = {}
    normalized.update(scenario)
    return normalized


def _sequence_update_from_payload(
    level_id: str,
    next_path: dict[str, object] | None,
    steps: list[dict[str, object]] | None,
) -> SequenceUpdate | None:
    from trainer.demo_api.sequence import sequence_update_from_payload as _sequence_update_from_payload_impl

    return _sequence_update_from_payload_impl(level_id, next_path, steps)
