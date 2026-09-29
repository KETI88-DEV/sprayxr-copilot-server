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
import threading
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
    """HTTP 400 for a request that carried a decoding grammar.

    Raised only when a grammar was sent and the message is not about context length, so
    the caller can retry once without the grammar instead of failing the request. The
    deployed vLLM build is chosen outside this repo (``VLLM_IMAGE``), so a server that
    does not accept the grammar field must not take the service down.
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
        grammar: str | None = None,
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
        if grammar:
            # vLLM reads guided decoding options from extra_body; the keys above are
            # untouched so a grammar-less request is byte-identical to before.
            payload["extra_body"] = {"guided_grammar": grammar, "guided_decoding_backend": "xgrammar"}
        return payload

    def complete_chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int,
        enable_thinking: bool,
        grammar: str | None = None,
    ) -> str:
        """Returns the generated text; raises an :class:`OpenAIBackendError` subclass on failure."""
        cancel = _cancel_event.get()
        if cancel is not None and cancel.is_set():
            raise GenerationCancelledError("cancelled before the request was sent")

        payload = self.build_payload(
            messages, max_tokens=max_tokens, enable_thinking=enable_thinking, grammar=grammar
        )
        parts: list[str] = []
        finish_reason: str | None = None
        try:
            with self._client.stream("POST", "/chat/completions", json=payload) as response:
                if response.status_code != 200:
                    body = response.read().decode("utf-8", errors="replace")
                    message = _error_message(body)
                    if (
                        grammar is not None
                        and response.status_code == 400
                        and not _CONTEXT_LENGTH_RE.search(message)
                    ):
                        raise StructuredOutputUnsupportedError(
                            f"model server rejected the decoding grammar: {message[:300]}"
                        )
                    raise _status_error(response.status_code, message)
                for line in response.iter_lines():
                    if cancel is not None and cancel.is_set():
                        # Leaving the ``with`` closes the connection mid-body; vLLM aborts the request.
                        raise GenerationCancelledError("consumer left during generation")
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
            raise OpenAIBackendTimeoutError(f"model server timed out: {type(exc).__name__}") from exc
        except httpx.TransportError as exc:
            raise OpenAIBackendUnavailableError(f"model server unreachable: {type(exc).__name__}") from exc

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
