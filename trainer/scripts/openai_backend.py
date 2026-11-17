"""OpenAI-compatible chat backend (``llm.backend: openai``) for an external vLLM server.

Production inference on Spark calls a vLLM OpenAI-compatible server over HTTP;
the ``vllm`` package is never imported here. The request contract (plan D6):

* ``POST {base_url}/chat/completions`` with ``stream: true``; the text is
  accumulated from ``choices[].delta.content`` so a departing consumer can stop
  reading and close the connection, which makes vLLM abort the request.
* ``tools`` / ``tool_choice`` are never sent. The model keeps the text
  ``<tool_call>{...}</tool_call>`` protocol it was fine-tuned on, and the
  caller parses the returned text with ``ToolCallingAdapter``.
* Every request states ``model``, ``messages``, ``max_tokens`` (per round),
  ``temperature``, ``top_p``, ``top_k``, ``repetition_penalty``,
  ``chat_template_kwargs.enable_thinking`` and ``skip_special_tokens``.
  Sampling values must be explicit in ``generation``: vLLM would otherwise
  apply the checkpoint's generation_config (temperature 1.0 / top_p 0.95).

Cancellation rides a context variable so ``AgentPipeline.run()`` keeps its
signature: ``run_streaming`` installs a ``threading.Event`` with
:func:`generation_cancel_scope` inside its worker thread, and ``asyncio.run`` /
``asyncio.to_thread`` copy that context down to this client. The event is
checked before sending and between streamed lines, so a cancel lands within one
token interval once decoding has started; a request still in prefill is
abandoned when its first bytes arrive.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import re
import socket
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Final

import httpx

from trainer.scripts.tool_calling_adapter import ToolInfrastructureError

logger = logging.getLogger(__name__)

_CONTEXT_LENGTH_RE: Final[re.Pattern[str]] = re.compile(
    r"maximum context length|maximum model length|max_model_len|context length",
    flags=re.IGNORECASE,
)
_REQUIRED_SAMPLING_KEYS: Final[tuple[str, ...]] = ("temperature", "top_p", "top_k", "repetition_penalty")
_PROBE_TIMEOUT_SEC: Final[float] = 2.0

_cancel_event: contextvars.ContextVar[threading.Event | None] = contextvars.ContextVar(
    "openai_backend_cancel_event", default=None
)
_generation_deadline: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "openai_backend_generation_deadline", default=None
)


class OpenAIBackendError(Exception):
    """Base for failures of the remote chat backend."""

    retryable: bool = True


class ContextLengthExceededError(OpenAIBackendError):
    """HTTP 400: prompt plus ``max_tokens`` does not fit the server's ``--max-model-len``."""

    retryable = False


class OpenAIBackendRequestError(OpenAIBackendError):
    """Any other 4xx: the request or the deployment configuration is wrong; retrying will not help."""

    retryable = False


class OpenAIBackendUnavailableError(OpenAIBackendError, ToolInfrastructureError):
    """5xx, 408/429, connection failure or a broken stream: the same infrastructure family as MCP faults."""

    retryable = True


class OpenAIBackendTimeoutError(OpenAIBackendUnavailableError):
    """Connect or read timeout against the model server."""


class StructuredOutputUnsupportedError(OpenAIBackendError):
    """HTTP 400 for a request that carried a decoding constraint.

    Raised only when a constraint was sent and the message is not about context length, so
    the caller can retry once without it instead of failing the request. The deployed vLLM
    build is chosen outside this repo (``VLLM_IMAGE``), so a server that does not accept
    ``structured_outputs`` must not take the service down.

    Note this cannot catch a server that *ignores* the field: vLLM 0.29.0 answers 200 to a
    request carrying the older ``guided_grammar`` and generates unconstrained. Only a
    falsifiable probe catches that, which is why the field name is pinned by measurement
    (``answer_grammar`` module docstring) rather than by trying several.
    """

    retryable = False


class GenerationCancelledError(OpenAIBackendError):
    """The consumer left; the upstream request was closed."""

    retryable = False


@contextmanager
def generation_cancel_scope(event: threading.Event) -> Iterator[None]:
    """Makes ``event`` the cancel signal for backend calls made in this context."""
    token = _cancel_event.set(event)
    try:
        yield
    finally:
        _cancel_event.reset(token)


@contextmanager
def generation_deadline_scope(timeout_sec: float) -> Iterator[None]:
    """Bound a whole completion, including a continuously active token stream.

    Nested operations inherit the earliest deadline. Consumer cancellation stays
    independent, so an optional review timeout never cancels the parent query.
    """
    previous = _generation_deadline.get()
    deadline = time.monotonic() + max(0.0, timeout_sec)
    token = _generation_deadline.set(min(previous, deadline) if previous is not None else deadline)
    try:
        yield
    finally:
        _generation_deadline.reset(token)


@dataclass(frozen=True)
class SamplingParams:
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float

    @classmethod
    def from_generation_config(cls, generation: Mapping[str, Any]) -> SamplingParams:
        missing = [key for key in _REQUIRED_SAMPLING_KEYS if generation.get(key) is None]
        if missing:
            raise ValueError(
                "generation must set sampling explicitly for the openai backend; missing: " + ", ".join(missing)
            )
        do_sample = bool(generation.get("do_sample", True))
        return cls(
            temperature=float(generation["temperature"]) if do_sample else 0.0,
            top_p=float(generation["top_p"]),
            top_k=int(generation["top_k"]),
            repetition_penalty=float(generation["repetition_penalty"]),
        )


@dataclass(frozen=True)
class OpenAIBackendSettings:
    base_url: str
    served_model_name: str
    api_key: str | None
    connect_timeout_sec: float
    request_timeout_sec: float
    sampling: SamplingParams

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        environ: Mapping[str, str] | None = None,
    ) -> OpenAIBackendSettings:
        env = os.environ if environ is None else environ
        llm = config.get("llm", {})
        for key in ("base_url", "served_model_name"):
            if not isinstance(llm.get(key), str) or not llm.get(key):
                raise ValueError(f"llm.{key} is required for the openai backend")
        api_key_env = llm.get("api_key_env")
        api_key = env.get(str(api_key_env), "") if api_key_env else ""
        return cls(
            base_url=str(llm["base_url"]).rstrip("/"),
            served_model_name=str(llm["served_model_name"]),
            api_key=api_key or None,
            connect_timeout_sec=float(llm.get("connect_timeout_sec", 5.0)),
            request_timeout_sec=float(llm.get("request_timeout_sec", 180.0)),
            sampling=SamplingParams.from_generation_config(config.get("generation", {})),
        )

    @property
    def server_root(self) -> str:
        """``base_url`` without the ``/v1`` API prefix; vLLM serves ``/health`` there."""
        return self.base_url[: -len("/v1")] if self.base_url.endswith("/v1") else self.base_url

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}


def _error_message(body: str) -> str:
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return body
    if isinstance(parsed, dict):
        error = parsed.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return error["message"]
        for key in ("message", "detail"):
            if isinstance(parsed.get(key), str):
                return parsed[key]
    return body


def _status_error(status: int, message: str) -> OpenAIBackendError:
    if status == 400 and _CONTEXT_LENGTH_RE.search(message):
        return ContextLengthExceededError(message)
    if status >= 500 or status in (408, 429):
        return OpenAIBackendUnavailableError(f"model server HTTP {status}: {message[:300]}")
    return OpenAIBackendRequestError(f"model server HTTP {status}: {message[:300]}")


class OpenAIChatBackend:
    """Blocking streaming client; the pipeline calls it from worker threads under ``_GENERATION_SLOT``."""

    def __init__(self, settings: OpenAIBackendSettings, *, transport: httpx.BaseTransport | None = None) -> None:
        self.settings = settings
        self._client = httpx.Client(
            base_url=settings.base_url,
            headers=settings.headers(),
            timeout=httpx.Timeout(
                settings.request_timeout_sec,
                connect=settings.connect_timeout_sec,
                pool=settings.connect_timeout_sec,
            ),
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def build_payload(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int,
        enable_thinking: bool,
        answer_regex: str | None = None,
    ) -> dict[str, Any]:
        sampling = self.settings.sampling
        payload: dict[str, Any] = {
            "model": self.settings.served_model_name,
            "messages": [dict(message) for message in messages],
            "max_tokens": int(max_tokens),
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "top_k": sampling.top_k,
            "repetition_penalty": sampling.repetition_penalty,
            "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
            "skip_special_tokens": True,
            "stream": True,
        }
        if answer_regex:
            # Measured on vLLM 0.29.0 (Spark, 2026-09-29): this is the field that actually
            # constrains generation. ``guided_grammar`` — top level or under ``extra_body`` —
            # returns 200 and is silently ignored, and ``guided_decoding_backend`` is not a
            # field of this request at all. The keys above are untouched, so an
            # unconstrained request is byte-identical to before.
            payload["structured_outputs"] = {"regex": answer_regex}
        return payload

    def complete_chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int,
        enable_thinking: bool,
        answer_regex: str | None = None,
    ) -> str:
        """Returns the generated text; raises an :class:`OpenAIBackendError` subclass on failure."""
        cancel = _cancel_event.get()
        deadline = _generation_deadline.get()
        expired = threading.Event()
        active_response: list[httpx.Response] = []
        response_lock = threading.Lock()

        def check_cancelled() -> None:
            if expired.is_set() or (deadline is not None and time.monotonic() >= deadline):
                raise OpenAIBackendTimeoutError("model generation exceeded its total operation deadline")
            if cancel is not None and cancel.is_set():
                raise GenerationCancelledError("consumer left during generation")

        def expire() -> None:
            expired.set()
            with response_lock:
                response = active_response[0] if active_response else None
            if response is None:
                return
            # Closing/shutting down the active stream wakes a blocked socket read
            # and makes vLLM abort the request, rather than leaving generation in
            # the background after the API has stopped waiting.
            network = response.extensions.get("network_stream")
            try:
                connection = network.get_extra_info("socket") if network is not None else None
                if connection is not None:
                    connection.shutdown(socket.SHUT_RDWR)
            except (AttributeError, OSError):
                pass
            try:
                response.close()
            except Exception as exc:
                logger.debug("deadline stream close failed: %s", type(exc).__name__)

        check_cancelled()

        payload = self.build_payload(
            messages, max_tokens=max_tokens, enable_thinking=enable_thinking, answer_regex=answer_regex
        )
        parts: list[str] = []
        finish_reason: str | None = None
        timer = None
        request_options: dict[str, Any] = {}
        if deadline is not None:
            remaining = max(.001, deadline - time.monotonic())
            request_options["timeout"] = httpx.Timeout(
                remaining, connect=min(remaining, self.settings.connect_timeout_sec),
                pool=min(remaining, self.settings.connect_timeout_sec),
            )
            timer = threading.Timer(remaining, expire)
            timer.daemon = True
            timer.start()
        try:
            with self._client.stream("POST", "/chat/completions", json=payload, **request_options) as response:
                with response_lock:
                    active_response.append(response)
                check_cancelled()
                if response.status_code != 200:
                    body = response.read().decode("utf-8", errors="replace")
                    message = _error_message(body)
                    if (
                        answer_regex is not None
                        and response.status_code == 400
                        and not _CONTEXT_LENGTH_RE.search(message)
                    ):
                        raise StructuredOutputUnsupportedError(
                            f"model server rejected the decoding constraint: {message[:300]}"
                        )
                    raise _status_error(response.status_code, message)
                for line in response.iter_lines():
                    check_cancelled()
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    chunk = self._parse_chunk(data)
                    for choice in chunk.get("choices") or []:
                        content = (choice.get("delta") or {}).get("content")
                        if isinstance(content, str) and content:
                            parts.append(content)
                        if choice.get("finish_reason"):
                            finish_reason = str(choice["finish_reason"])
        except httpx.TimeoutException as exc:
            check_cancelled()
            raise OpenAIBackendTimeoutError(f"model server timed out: {type(exc).__name__}") from exc
        except httpx.TransportError as exc:
            check_cancelled()
            raise OpenAIBackendUnavailableError(f"model server unreachable: {type(exc).__name__}") from exc
        except RuntimeError:
            check_cancelled()
            raise
        finally:
            if timer is not None:
                timer.cancel()
        check_cancelled()

        if finish_reason is None:
            raise OpenAIBackendUnavailableError("model server stream ended without a finish_reason")
        if finish_reason == "length":
            logger.warning(
                "openai backend: generation stopped at max_tokens=%d (finish_reason=length, chars=%d)",
                max_tokens,
                sum(len(part) for part in parts),
            )
        return "".join(parts)

    @staticmethod
    def _parse_chunk(data: str) -> dict[str, Any]:
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError as exc:
            raise OpenAIBackendUnavailableError("model server sent a malformed stream chunk") from exc
        if not isinstance(chunk, dict):
            raise OpenAIBackendUnavailableError("model server sent a non-object stream chunk")
        if chunk.get("object") == "error" or "error" in chunk:
            error = chunk.get("error")
            message = error.get("message", "") if isinstance(error, dict) else str(chunk.get("message", ""))
            status = int(chunk.get("code") or (error.get("code") if isinstance(error, dict) else 0) or 500)
            raise _status_error(status if status >= 400 else 500, str(message))
        return chunk


@dataclass(frozen=True)
class OpenAIServerProbe:
    health_ok: bool
    model_served: bool


async def probe_openai_server(
    settings: OpenAIBackendSettings,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    timeout_sec: float = _PROBE_TIMEOUT_SEC,
) -> OpenAIServerProbe:
    """Checks vLLM ``/health`` and that ``/v1/models`` lists ``served_model_name``. Never raises."""
    health_ok = False
    model_served = False
    async with httpx.AsyncClient(headers=settings.headers(), timeout=timeout_sec, transport=transport) as client:
        try:
            health_ok = (await client.get(f"{settings.server_root}/health")).status_code == 200
        except httpx.HTTPError:
            logger.info("readiness: model server /health unreachable")
        try:
            response = await client.get(f"{settings.base_url}/models")
            if response.status_code == 200:
                data = response.json().get("data") or []
                model_served = any(
                    isinstance(item, dict) and item.get("id") == settings.served_model_name for item in data
                )
        except (httpx.HTTPError, ValueError, AttributeError):
            logger.info("readiness: model server /models unavailable")
    return OpenAIServerProbe(health_ok=health_ok, model_served=model_served)
