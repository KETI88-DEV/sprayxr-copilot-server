"""Fast-route decoding grammar: mode resolution, payload wiring, and the unsupported-server fallback."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from trainer.scripts.answer_grammar import (
    KNOWLEDGE_GRAMMAR_NO_EVIDENCE,
    KNOWLEDGE_GRAMMAR_WITH_EVIDENCE,
    knowledge_grammar,
    structured_output_fallback,
    structured_output_mode,
)
from trainer.scripts.openai_backend import (
    ContextLengthExceededError,
    OpenAIBackendRequestError,
    OpenAIBackendSettings,
    OpenAIChatBackend,
    StructuredOutputUnsupportedError,
)

_GENERATION = {
    "temperature": 0.31,
    "top_p": 0.83,
    "top_k": 17,
    "repetition_penalty": 1.19,
    "do_sample": True,
    "max_new_tokens": 1024,
}
_MESSAGES = [{"role": "user", "content": "저도막이 뭐야"}]


def _config(**llm: Any) -> dict[str, Any]:
    base = {
        "backend": "openai",
        "base_url": "http://vllm.test:8000/v1",
        "served_model_name": "shipcoat-v6-1-fp8",
    }
    base.update(llm)
    return {"llm": base, "generation": dict(_GENERATION)}


def _backend(handler: Any) -> OpenAIChatBackend:
    settings = OpenAIBackendSettings.from_config(_config(), environ={})
    return OpenAIChatBackend(settings, transport=httpx.MockTransport(handler))


def _sse_ok(text: str = "[답변] SENTINEL") -> bytes:
    chunk = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": "stop"}],
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\ndata: [DONE]\n\n".encode()


# --- mode resolution -------------------------------------------------------


def test_structured_output_defaults_to_off() -> None:
    """Given no setting, When resolved, Then the grammar is not sent."""
    assert structured_output_mode(_config()) == "off"


@pytest.mark.parametrize("raw", ["grammar", "GRAMMAR", " grammar "])
def test_grammar_mode_is_case_and_space_tolerant(raw: str) -> None:
    """Given a grammar setting in any casing, When resolved, Then grammar mode is selected."""
    assert structured_output_mode(_config(structured_output=raw)) == "grammar"


def test_json_mode_is_rejected_rather_than_silently_degraded() -> None:
    """Given the reserved json mode, When resolved, Then it fails loudly."""
    with pytest.raises(ValueError, match="not implemented"):
        structured_output_mode(_config(structured_output="json"))


@pytest.mark.parametrize("raw", ["on", "true", "ebnf", ""])
def test_unknown_mode_is_a_configuration_error(raw: str) -> None:
    """Given an unknown mode, When resolved, Then startup fails instead of guessing."""
    with pytest.raises(ValueError, match="structured_output"):
        structured_output_mode(_config(structured_output=raw))


def test_fallback_defaults_on_and_can_be_switched_off() -> None:
    """Given the fallback flag, When read, Then it defaults on and honours an explicit false."""
    assert structured_output_fallback(_config()) is True
    assert structured_output_fallback(_config(structured_output_fallback=False)) is False


# --- grammar content ------------------------------------------------------


def test_no_evidence_grammar_cannot_emit_a_document_id() -> None:
    """Given nothing retrieved, When the grammar is chosen, Then no citation rule exists."""
    grammar = knowledge_grammar(has_evidence=False)

    assert grammar == KNOWLEDGE_GRAMMAR_NO_EVIDENCE
    assert "DOC-" not in grammar
    assert "근거" not in grammar


def test_evidence_grammar_allows_at_most_one_evidence_line_with_three_ids() -> None:
    """Given retrieved chunks, When the grammar is chosen, Then evidence appears at most once."""
    grammar = knowledge_grammar(has_evidence=True)

    assert grammar == KNOWLEDGE_GRAMMAR_WITH_EVIDENCE
    assert "evidence?" in grammar
    assert '(", " cite){0,2}' in grammar
    assert grammar.count("evidence ::=") == 1


@pytest.mark.parametrize("grammar", [KNOWLEDGE_GRAMMAR_WITH_EVIDENCE, KNOWLEDGE_GRAMMAR_NO_EVIDENCE])
def test_both_grammars_open_with_the_answer_heading_and_forbid_a_bracket_line_start(grammar: str) -> None:
    """Given either grammar, When inspected, Then the body cannot restart a section heading."""
    assert grammar.startswith('root ::= "[답변] " body')
    assert "bodyhead ::= [^\\n\\[]" in grammar
    for label in ("판정", "원인", "조치", "분석"):
        assert label not in grammar


# --- payload wiring -------------------------------------------------------


def test_payload_without_a_grammar_carries_no_extra_body() -> None:
    """Given no grammar, When the payload is built, Then it is unchanged from the plain contract."""
    payload = _backend(lambda _r: httpx.Response(200)).build_payload(
        _MESSAGES, max_tokens=256, enable_thinking=False
    )

    assert "extra_body" not in payload


def test_payload_with_a_grammar_adds_only_extra_body() -> None:
    """Given a grammar, When the payload is built, Then extra_body is the sole addition."""
    backend = _backend(lambda _r: httpx.Response(200))
    plain = backend.build_payload(_MESSAGES, max_tokens=256, enable_thinking=False)
    guided = backend.build_payload(
        _MESSAGES, max_tokens=256, enable_thinking=False, grammar=KNOWLEDGE_GRAMMAR_NO_EVIDENCE
    )

    assert set(guided) - set(plain) == {"extra_body"}
    assert {key: guided[key] for key in plain} == plain
    assert guided["extra_body"] == {
        "guided_grammar": KNOWLEDGE_GRAMMAR_NO_EVIDENCE,
        "guided_decoding_backend": "xgrammar",
    }


def test_grammar_reaches_the_wire() -> None:
    """Given a grammar, When the request is sent, Then the server receives it verbatim."""
    sent: list[httpx.Request] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_sse_ok())

    _backend(_handler).complete_chat(
        _MESSAGES, max_tokens=256, enable_thinking=False, grammar=KNOWLEDGE_GRAMMAR_WITH_EVIDENCE
    )

    body = json.loads(sent[0].content)
    assert body["extra_body"]["guided_grammar"] == KNOWLEDGE_GRAMMAR_WITH_EVIDENCE


# --- server rejection -----------------------------------------------------


def test_400_on_a_guided_request_is_reported_as_unsupported_structured_output() -> None:
    """Given a server that rejects the grammar, When called, Then the cause is distinguishable."""
    handler = lambda _r: httpx.Response(400, json={"error": {"message": "unknown field guided_grammar"}})

    with pytest.raises(StructuredOutputUnsupportedError, match="guided_grammar"):
        _backend(handler).complete_chat(
            _MESSAGES, max_tokens=256, enable_thinking=False, grammar=KNOWLEDGE_GRAMMAR_NO_EVIDENCE
        )


def test_context_length_400_stays_a_context_error_even_with_a_grammar() -> None:
    """Given an over-long guided request, When rejected, Then it is not blamed on the grammar."""
    handler = lambda _r: httpx.Response(400, json={"error": {"message": "maximum context length is 8192"}})

    with pytest.raises(ContextLengthExceededError):
        _backend(handler).complete_chat(
            _MESSAGES, max_tokens=256, enable_thinking=False, grammar=KNOWLEDGE_GRAMMAR_NO_EVIDENCE
        )


def test_400_without_a_grammar_keeps_its_existing_classification() -> None:
    """Given a plain request rejected with 400, When raised, Then the old error type is unchanged."""
    handler = lambda _r: httpx.Response(400, json={"error": {"message": "bad request"}})

    with pytest.raises(OpenAIBackendRequestError):
        _backend(handler).complete_chat(_MESSAGES, max_tokens=256, enable_thinking=False)


# --- pipeline fallback ----------------------------------------------------


def _pipeline(handler: Any, **llm: Any) -> Any:
    from trainer.scripts.agent_pipeline import AgentPipeline

    pipe = AgentPipeline.__new__(AgentPipeline)
    pipe.config = _config(**llm)
    pipe.model = _backend(handler)
    return pipe


def test_grammar_is_only_chosen_for_the_remote_backend_in_grammar_mode() -> None:
    """Given each deployment shape, When a grammar is requested, Then only openai+grammar gets one."""
    assert _pipeline(lambda _r: httpx.Response(200))._knowledge_grammar(has_evidence=True) is None

    guided = _pipeline(lambda _r: httpx.Response(200), structured_output="grammar")
    assert guided._knowledge_grammar(has_evidence=True) == KNOWLEDGE_GRAMMAR_WITH_EVIDENCE
    assert guided._knowledge_grammar(has_evidence=False) == KNOWLEDGE_GRAMMAR_NO_EVIDENCE

    guided.model = object()
    assert guided._knowledge_grammar(has_evidence=True) is None


def test_a_server_that_rejects_the_grammar_is_retried_once_without_it() -> None:
    """Given a server without grammar support, When answering, Then the request still succeeds."""
    seen: list[bool] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        guided = "extra_body" in json.loads(request.content)
        seen.append(guided)
        if guided:
            return httpx.Response(400, json={"error": {"message": "unknown field guided_grammar"}})
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_sse_ok())

    pipe = _pipeline(_handler, structured_output="grammar")

    assert pipe._generate_direct(_MESSAGES, grammar=KNOWLEDGE_GRAMMAR_NO_EVIDENCE) == "[답변] SENTINEL"
    assert seen == [True, False]
    # The process stops paying for a rejection it already learned about.
    assert pipe._generate_direct(_MESSAGES, grammar=KNOWLEDGE_GRAMMAR_NO_EVIDENCE) == "[답변] SENTINEL"
    assert seen == [True, False, False]


def test_fallback_switched_off_lets_the_rejection_surface() -> None:
    """Given the fallback disabled, When the grammar is rejected, Then the caller sees the failure."""
    handler = lambda _r: httpx.Response(400, json={"error": {"message": "unknown field guided_grammar"}})
    pipe = _pipeline(handler, structured_output="grammar", structured_output_fallback=False)

    with pytest.raises(StructuredOutputUnsupportedError):
        pipe._generate_direct(_MESSAGES, grammar=KNOWLEDGE_GRAMMAR_NO_EVIDENCE)


def test_grammar_is_dropped_when_thinking_is_on() -> None:
    """Given thinking enabled, When generating, Then the grammar is not sent.

    The chat template pre-closes an empty ``<think>`` block only when thinking is off; with
    it on the prompt ends inside an open block, which this grammar's root forbids.
    """
    sent: list[bool] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        sent.append("extra_body" in json.loads(request.content))
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_sse_ok())

    pipe = _pipeline(_handler, structured_output="grammar")
    pipe._generate_direct(_MESSAGES, enable_thinking=True, grammar=KNOWLEDGE_GRAMMAR_NO_EVIDENCE)
    pipe._generate_direct(_MESSAGES, enable_thinking=False, grammar=KNOWLEDGE_GRAMMAR_NO_EVIDENCE)

    assert sent == [False, True]
