#!/usr/bin/env python3
# pyright: reportAny=false, reportExplicitAny=false
"""Typed HTTP/SSE measurement boundary for the latency benchmark."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterable
from typing import Any, ClassVar, Literal, NotRequired, Self, TypedDict

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

PathName = Literal["rag", "tools"]
JsonObject = dict[str, Any]


class EvalQuestion(TypedDict):
    q_id: str
    question: str
    tool_calls_required: list[str]
    context_data: NotRequired[JsonObject]


class BenchRow(TypedDict):
    q_id: str
    expected_path: PathName
    path: PathName | None
    ttft_ms: float | None
    backend_ttft_ms: float | None
    synthesis_ms: float | None
    total_ms: float
    backend_total_ms: float | None
    tool_call_count: int
    rag_chunk_count: int
    deadline_hit: bool
    degraded: bool
    mcp_timeout_count: int
    error: str | None


class _SseEnvelope(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", strict=True, allow_inf_nan=False
    )
    event: str
    data: JsonObject


class _CompletePayload(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", strict=True, allow_inf_nan=False
    )
    tool_call_count: int = Field(ge=0)
    rag_chunk_count: int = Field(ge=0)
    ttft_ms: float | None = Field(default=None, ge=0)
    synthesis_ms: float = Field(ge=0)
    total_ms: float = Field(ge=0)
    deadline_hit: bool
    degraded: bool = False


class _ErrorPayload(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", strict=True, allow_inf_nan=False
    )
    message: str = Field(min_length=1)
    retryable: bool
    ttft_ms: float | None = Field(default=None, ge=0)
    total_ms: float | None = Field(default=None, ge=0)


class _ToolError(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="ignore", strict=True, allow_inf_nan=False
    )
    code: str


class _ToolResult(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="ignore", strict=True, allow_inf_nan=False
    )
    status: Literal["success", "error"]
    result: JsonObject | None = None
    error: _ToolError | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        if self.status == "success" and (self.result is None or self.error is not None):
            raise ValueError("successful tool result requires result only")
        if self.status == "error" and (self.error is None or self.result is not None):
            raise ValueError("failed tool result requires error only")
        return self


class _MessageDelta(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", strict=True, allow_inf_nan=False
    )
    text: str = Field(min_length=1)


class _ToolResultEvent(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", strict=True, allow_inf_nan=False
    )
    name: str
    arguments: JsonObject
    result: JsonObject


def classify_path(tool_call_count: int) -> PathName:
    return "rag" if tool_call_count == 0 else "tools"


def parse_sse_frames(chunks: Iterable[str]) -> list[tuple[str, JsonObject]]:
    text = "".join(chunks).replace("\r\n", "\n")
    parsed: list[tuple[str, JsonObject]] = []
    for raw_frame in text.split("\n\n"):
        event_name = ""
        data_lines: list[str] = []
        for line in raw_frame.splitlines():
            if line.startswith("event:"):
                event_name = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data_lines.append(line.removeprefix("data:").lstrip())
        if not event_name or not data_lines:
            continue
        envelope = _SseEnvelope.model_validate_json("\n".join(data_lines))
        if envelope.event != event_name:
            raise ValueError(f"SSE event mismatch: {event_name} != {envelope.event}")
        parsed.append((event_name, envelope.data))
    return parsed


def _failed_row(
    question: EvalQuestion,
    started: float,
    message: str,
    *,
    observed_ttft: float | None = None,
    backend_ttft: float | None = None,
    backend_total: float | None = None,
    mcp_timeout_count: int = 0,
) -> BenchRow:
    return {
        "q_id": question["q_id"],
        "expected_path": "tools" if question["tool_calls_required"] else "rag",
        "path": None,
        "ttft_ms": observed_ttft,
        "backend_ttft_ms": backend_ttft,
        "synthesis_ms": None,
        "total_ms": (time.monotonic() - started) * 1000,
        "backend_total_ms": backend_total,
        "tool_call_count": 0,
        "rag_chunk_count": 0,
        "deadline_hit": False,
        "degraded": False,
        "mcp_timeout_count": mcp_timeout_count,
        "error": message,
    }


def _is_mcp_timeout(tool_result: _ToolResult) -> bool:
    return (
        tool_result.status == "error"
        and tool_result.error is not None
        and tool_result.error.code == "TIMEOUT"
    )


def measure_question(
    client: httpx.Client,
    api_url: str,
    question: EvalQuestion,
    timeout_sec: float,
) -> BenchRow:
    started = time.monotonic()
    deadline = started + timeout_sec
    observed_ttft: float | None = None
    mcp_timeout_count = 0
    frame_lines: list[str] = []
    try:
        with client.stream(
            "POST",
            f"{api_url.rstrip('/')}/api/chat",
            json={
                "message": question["question"],
                "session_id": f"bench-{uuid.uuid4().hex}",
                "mode": "live",
            },
            timeout=timeout_sec,
            headers={"Accept": "text/event-stream"},
        ) as response:
            _ = response.raise_for_status()
            for line in response.iter_lines():
                if time.monotonic() >= deadline:
                    return _failed_row(
                        question,
                        started,
                        f"overall sample timeout after {timeout_sec} seconds",
                        observed_ttft=observed_ttft,
                        mcp_timeout_count=mcp_timeout_count,
                    )
                if line:
                    frame_lines.append(line)
                    continue
                if not frame_lines:
                    continue
                frames = parse_sse_frames(["\n".join(frame_lines), "\n\n"])
                frame_lines.clear()
                if not frames:
                    continue
                kind, payload = frames[0]
                if kind == "message_delta":
                    try:
                        _ = _MessageDelta.model_validate(payload)
                    except ValidationError as exc:
                        raise ValueError(f"invalid message_delta event: {exc}") from exc
                    if observed_ttft is None:
                        observed_ttft = (time.monotonic() - started) * 1000
                elif kind == "tool_result":
                    try:
                        tool_event = _ToolResultEvent.model_validate(payload)
                        tool_result = _ToolResult.model_validate(tool_event.result)
                    except ValidationError as exc:
                        raise ValueError(f"invalid tool_result event: {exc}") from exc
                    if _is_mcp_timeout(tool_result):
                        mcp_timeout_count += 1
                elif kind == "error":
                    try:
                        error = _ErrorPayload.model_validate(payload)
                    except ValidationError as exc:
                        raise ValueError(f"invalid error event: {exc}") from exc
                    return _failed_row(
                        question,
                        started,
                        error.message,
                        observed_ttft=observed_ttft,
                        backend_ttft=error.ttft_ms,
                        backend_total=error.total_ms,
                        mcp_timeout_count=mcp_timeout_count,
                    )
                elif kind == "complete":
                    try:
                        complete = _CompletePayload.model_validate(payload)
                    except ValidationError as exc:
                        raise ValueError(f"invalid complete event: {exc}") from exc
                    if observed_ttft is None:
                        return _failed_row(
                            question,
                            started,
                            "complete event arrived before a valid message_delta",
                            mcp_timeout_count=mcp_timeout_count,
                        )
                    return {
                        "q_id": question["q_id"],
                        "expected_path": "tools" if question["tool_calls_required"] else "rag",
                        "path": classify_path(complete.tool_call_count),
                        "ttft_ms": observed_ttft,
                        "backend_ttft_ms": complete.ttft_ms,
                        "synthesis_ms": complete.synthesis_ms,
                        "total_ms": (time.monotonic() - started) * 1000,
                        "backend_total_ms": complete.total_ms,
                        "tool_call_count": complete.tool_call_count,
                        "rag_chunk_count": complete.rag_chunk_count,
                        "deadline_hit": complete.deadline_hit,
                        "degraded": complete.degraded,
                        "mcp_timeout_count": mcp_timeout_count,
                        "error": None,
                    }
    except (httpx.HTTPError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        return _failed_row(
            question,
            started,
            f"invalid SSE response: {exc}",
            observed_ttft=observed_ttft,
            mcp_timeout_count=mcp_timeout_count,
        )
    return _failed_row(
        question,
        started,
        "missing complete or error event",
        observed_ttft=observed_ttft,
        mcp_timeout_count=mcp_timeout_count,
    )
