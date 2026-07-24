# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Final, Protocol

from trainer.demo_api.contracts import ChatRequest, SequenceUpdate, StreamEvent
from trainer.demo_api.scenario_adapter import get_demo_scenario

LiveRunner = Callable[[ChatRequest], dict[str, object]]

_CONFIG_PATH_DEFAULT: Final[str] = "/workspace/trainer/configs/rag_config_fewshot.yaml"
_ADAPTER_PATH_DEFAULT: Final[str] = "/output/adapters/qwen3_4b_shipcoat_lora/"
_MCP_BASE_URL_DEFAULT: Final[str] = "http://localhost:8100"


class _Pipeline(Protocol):
    def run(
        self,
        *,
        instruction: str,
        session_id: str,
        enable_thinking: bool,
        environment: dict[str, object],
    ) -> dict[str, object]:
        ...


_pipeline_singleton: _Pipeline | None = None


class ChatService:
    def __init__(self, live_runner: LiveRunner | None = None) -> None:
        self.live_runner: LiveRunner | None = live_runner

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        live_runner = self.live_runner
        if request.mode == "live" and live_runner is None:
            yield StreamEvent(
                event="error",
                data={"message": "live pipeline is not configured", "retryable": False},
            )
            return

        yield StreamEvent(event="session", data={"session_id": request.session_id, "mode": request.mode})

        scenario: dict[str, object]
        if request.mode == "live":
            runner = live_runner
            assert runner is not None

            def _run_live() -> dict[str, object]:
                return runner(request)

            try:
                scenario = await asyncio.to_thread(_run_live)
            except Exception as exc:  # noqa: BROAD_EXCEPT_OK
                yield StreamEvent(
                    event="error",
                    data={"message": str(exc), "retryable": True},
                )
                return
        else:
            scenario = _demo_scenario(request.message)

        for event in _iter_stream_events(request, scenario):
            yield event


def build_default_live_runner() -> LiveRunner:
    def _runner(request: ChatRequest) -> dict[str, object]:
        global _pipeline_singleton
        if _pipeline_singleton is None:
            from trainer.scripts.agent_pipeline import AgentPipeline

            _pipeline_singleton = AgentPipeline(
                config_path=os.environ.get("CONFIG_PATH", _CONFIG_PATH_DEFAULT),
                adapter_path=os.environ.get("ADAPTER_PATH", _ADAPTER_PATH_DEFAULT),
                mcp_base_url=os.environ.get("MCP_BASE_URL", _MCP_BASE_URL_DEFAULT),
            )

        return _pipeline_singleton.run(
            instruction=request.message,
            session_id=request.session_id,
            enable_thinking=False,
            environment={},
        )

    return _runner


def _iter_stream_events(request: ChatRequest, scenario: dict[str, object]) -> Iterator[StreamEvent]:
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

    response_obj = scenario.get("response", "")
    response = response_obj if isinstance(response_obj, str) else ""
    if response:
        for index in range(0, len(response), 24):
            yield StreamEvent(event="message_delta", data={"text": response[index : index + 24]})
    else:
        yield StreamEvent(event="message_delta", data={"text": ""})

    yield StreamEvent(
        event="complete",
        data={"tool_call_count": len(tool_calls), "rag_chunk_count": len(rag_chunks)},
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
