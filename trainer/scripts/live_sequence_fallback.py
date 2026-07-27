from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Protocol

_SEQUENCE_INTENT_MARKERS: Final[tuple[str, ...]] = (
    "순서",
    "순번",
    "다음 구역",
    "다음 블록",
    "어디부터",
    "sequence",
    "next zone",
)
_RETRYABLE_ERROR_CODES: Final[frozenset[str]] = frozenset({"TIMEOUT", "SERVER_ERROR"})


class SequenceToolAdapter(Protocol):
    async def dispatch_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        session_id: str | None = None,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class LiveSequenceRequest:
    instruction: str
    route: str
    tool_calls: list[dict[str, Any]]
    response: str
    environment: dict[str, Any] | None
    session_id: str | None


@dataclass(frozen=True, slots=True)
class LiveSequenceResult:
    tool_calls: list[dict[str, Any]]
    response: str
    error_retryable: bool | None = None


async def enforce_live_sequence_validation(
    request: LiveSequenceRequest,
    adapter: SequenceToolAdapter | None,
) -> LiveSequenceResult:
    tool_calls = list(request.tool_calls)
    normalized_instruction = request.instruction.casefold()
    sequence_intent = request.route == "training_realtime" and any(
        marker in normalized_instruction for marker in _SEQUENCE_INTENT_MARKERS
    )
    sequence_called = any(call.get("name") == "sequence_validation" for call in tool_calls)
    if not sequence_intent or sequence_called:
        return LiveSequenceResult(tool_calls=tool_calls, response=request.response)
    if adapter is None:
        return LiveSequenceResult(
            tool_calls=tool_calls,
            response=request.response,
            error_retryable=False,
        )

    arguments = dict(request.environment or {})
    fallback_result = await adapter.dispatch_tool_call(
        "sequence_validation",
        arguments,
        request.session_id,
    )
    tool_calls.append(
        {
            "name": "sequence_validation",
            "arguments": arguments,
            "result": fallback_result,
        }
    )
    if fallback_result.get("status") != "success":
        error = fallback_result.get("error")
        error_code = str(error.get("code", "")) if isinstance(error, dict) else ""
        return LiveSequenceResult(
            tool_calls=tool_calls,
            response=request.response,
            error_retryable=error_code in _RETRYABLE_ERROR_CODES,
        )

    business_result = fallback_result.get("result")
    if not isinstance(business_result, dict):
        return LiveSequenceResult(tool_calls=tool_calls, response=request.response)
    message = business_result.get("message")
    if not isinstance(message, str) or not message.strip():
        return LiveSequenceResult(tool_calls=tool_calls, response=request.response)
    return LiveSequenceResult(tool_calls=tool_calls, response=message.strip())
