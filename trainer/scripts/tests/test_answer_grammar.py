"""Fast-route decoding constraint: mode resolution, payload wiring, and the unsupported-server fallback."""

from __future__ import annotations

import json
import re
from typing import Any

import httpx
import pytest

from trainer.scripts.answer_grammar import (
    ANSWER_REGEX_NO_EVIDENCE,
    ANSWER_REGEX_WITH_EVIDENCE,
    answer_regex,
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

DOC_A = "DOC-doc_0086-C0001"
DOC_B = "DOC-doc_0042-C0007"


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


@pytest.mark.parametrize("raw", ["on", "ON", " on "])
def test_on_mode_is_case_and_space_tolerant(raw: str) -> None:
    """Given a grammar setting in any casing, When resolved, Then grammar mode is selected."""
    assert structured_output_mode(_config(structured_output=raw)) == "on"


def test_obsolete_transport_names_are_rejected_rather_than_silently_mapped() -> None:
    """Given the reserved json mode, When resolved, Then it fails loudly."""
    for obsolete in ("json", "grammar"):
        with pytest.raises(ValueError, match="obsolete"):
            structured_output_mode(_config(structured_output=obsolete))


@pytest.mark.parametrize("raw", ["true", "ebnf", "regex", ""])
def test_unknown_mode_is_a_configuration_error(raw: str) -> None:
    """Given an unknown mode, When resolved, Then startup fails instead of guessing."""
    with pytest.raises(ValueError, match="structured_output"):
        structured_output_mode(_config(structured_output=raw))


def test_fallback_defaults_on_and_can_be_switched_off() -> None:
    """Given the fallback flag, When read, Then it defaults on and honours an explicit false."""
    assert structured_output_fallback(_config()) is True
    assert structured_output_fallback(_config(structured_output_fallback=False)) is False


# --- constraint content ------------------------------------------------------
#
# Asserted as behavior (what the pattern accepts), not as pattern text: the shapes below
# are the ones measured on the deployed server, and a text assertion would pass while the
# pattern stopped matching them.


@pytest.mark.parametrize(
    ("answer", "accepted"),
    [
        ("[답변] 한 줄 설명입니다.", True),
        ("[답변] 첫 줄\n둘째 줄", True),
        (f"[답변] 본문\n[근거] {DOC_A}", True),
        (f"[답변] 본문\n[근거] {DOC_A}, {DOC_B}", True),
        # An unconstrained probe appended "[분석] 이 답변은…" on the same line; the body
        # bans "[" outright so that cannot recur.
        ("[답변] 본문 [분석] 자평", False),
        (f"[답변] 본문\n[근거] {DOC_A}\n[근거] {DOC_B}", False),
        ("[답변] 본문\n[판정] 합격", False),
        ("[분석] 본문", False),
        ("[답변] ", False),
        ("본문만 있음", False),
        (f"[답변] 본문\n[근거] {DOC_A}\n추가 문장", False),
    ],
)
def test_evidence_pattern_accepts_exactly_the_contract_shapes(answer: str, accepted: bool) -> None:
    """Given a candidate answer, When matched, Then only the contract shape is admitted."""
    assert (re.fullmatch(ANSWER_REGEX_WITH_EVIDENCE, answer) is not None) is accepted


def test_no_evidence_pattern_cannot_emit_a_document_id() -> None:
    """Given nothing retrieved, When the pattern is chosen, Then no citation clause exists."""
    pattern = answer_regex(has_evidence=False)

    assert pattern == ANSWER_REGEX_NO_EVIDENCE
    assert "DOC-" not in pattern
    assert "근거" not in pattern
    assert re.fullmatch(pattern, f"[답변] 본문\n[근거] {DOC_A}") is None
    assert re.fullmatch(pattern, "[답변] 본문") is not None


def test_evidence_pattern_is_chosen_when_chunks_were_retrieved() -> None:
    """Given retrieved chunks, When the pattern is chosen, Then the evidence clause is present."""
    assert answer_regex(has_evidence=True) == ANSWER_REGEX_WITH_EVIDENCE


@pytest.mark.parametrize("pattern", [ANSWER_REGEX_WITH_EVIDENCE, ANSWER_REGEX_NO_EVIDENCE])
def test_no_unsupported_section_label_can_appear(pattern: str) -> None:
    """Given either pattern, When a verdict section is offered, Then it is rejected."""
    for label in ("판정", "원인", "조치", "분석"):
        assert re.fullmatch(pattern, f"[답변] 본문\n[{label}] 내용") is None


# --- payload wiring -------------------------------------------------------


def test_payload_without_a_constraint_is_unchanged() -> None:
    """Given no constraint, When the payload is built, Then it matches the plain contract."""
    payload = _backend(lambda _r: httpx.Response(200)).build_payload(
        _MESSAGES, max_tokens=256, enable_thinking=False
    )

    assert "structured_outputs" not in payload
    assert "extra_body" not in payload


def test_payload_with_a_constraint_adds_only_structured_outputs() -> None:
    """Given a constraint, When the payload is built, Then structured_outputs is the sole addition.

    ``guided_grammar`` and ``guided_decoding_backend`` must never be sent: vLLM 0.29.0
    answers 200 to the former and generates unconstrained, and the latter is not a field of
    ``ChatCompletionRequest`` at all.
    """
    backend = _backend(lambda _r: httpx.Response(200))
    plain = backend.build_payload(_MESSAGES, max_tokens=256, enable_thinking=False)
    guided = backend.build_payload(
        _MESSAGES, max_tokens=256, enable_thinking=False, answer_regex=ANSWER_REGEX_NO_EVIDENCE
    )

    assert set(guided) - set(plain) == {"structured_outputs"}
    assert {key: guided[key] for key in plain} == plain
    assert guided["structured_outputs"] == {"regex": ANSWER_REGEX_NO_EVIDENCE}
    assert "guided_grammar" not in json.dumps(guided)
    assert "guided_decoding_backend" not in json.dumps(guided)


def test_constraint_reaches_the_wire() -> None:
    """Given a constraint, When the request is sent, Then the server receives it verbatim."""
    sent: list[httpx.Request] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_sse_ok())

    _backend(_handler).complete_chat(
        _MESSAGES, max_tokens=256, enable_thinking=False, answer_regex=ANSWER_REGEX_WITH_EVIDENCE
    )

    body = json.loads(sent[0].content)
    assert body["structured_outputs"] == {"regex": ANSWER_REGEX_WITH_EVIDENCE}


# --- server rejection -----------------------------------------------------


def test_400_on_a_guided_request_is_reported_as_unsupported_structured_output() -> None:
    """Given a server that rejects the grammar, When called, Then the cause is distinguishable."""
    handler = lambda _r: httpx.Response(400, json={"error": {"message": "unknown field structured_outputs"}})

    with pytest.raises(StructuredOutputUnsupportedError, match="structured_outputs"):
        _backend(handler).complete_chat(
            _MESSAGES, max_tokens=256, enable_thinking=False, answer_regex=ANSWER_REGEX_NO_EVIDENCE
        )


def test_context_length_400_stays_a_context_error_even_with_a_grammar() -> None:
    """Given an over-long guided request, When rejected, Then it is not blamed on the grammar."""
    handler = lambda _r: httpx.Response(400, json={"error": {"message": "maximum context length is 8192"}})

    with pytest.raises(ContextLengthExceededError):
        _backend(handler).complete_chat(
            _MESSAGES, max_tokens=256, enable_thinking=False, answer_regex=ANSWER_REGEX_NO_EVIDENCE
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


def test_constraint_is_only_chosen_for_the_remote_backend_when_on() -> None:
    """Given each deployment shape, When a grammar is requested, Then only openai+grammar gets one."""
    assert _pipeline(lambda _r: httpx.Response(200))._answer_regex(has_evidence=True) is None

    guided = _pipeline(lambda _r: httpx.Response(200), structured_output="on")
    assert guided._answer_regex(has_evidence=True) == ANSWER_REGEX_WITH_EVIDENCE
    assert guided._answer_regex(has_evidence=False) == ANSWER_REGEX_NO_EVIDENCE

    guided.model = object()
    assert guided._answer_regex(has_evidence=True) is None


def test_a_server_that_rejects_the_constraint_is_retried_once_without_it() -> None:
    """Given a server without grammar support, When answering, Then the request still succeeds."""
    seen: list[bool] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        guided = "structured_outputs" in json.loads(request.content)
        seen.append(guided)
        if guided:
            return httpx.Response(400, json={"error": {"message": "unknown field structured_outputs"}})
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_sse_ok())

    pipe = _pipeline(_handler, structured_output="on")

    assert pipe._generate_direct(_MESSAGES, answer_regex=ANSWER_REGEX_NO_EVIDENCE) == "[답변] SENTINEL"
    assert seen == [True, False]
    # The process stops paying for a rejection it already learned about.
    assert pipe._generate_direct(_MESSAGES, answer_regex=ANSWER_REGEX_NO_EVIDENCE) == "[답변] SENTINEL"
    assert seen == [True, False, False]


def test_fallback_switched_off_lets_the_rejection_surface() -> None:
    """Given the fallback disabled, When the grammar is rejected, Then the caller sees the failure."""
    handler = lambda _r: httpx.Response(400, json={"error": {"message": "unknown field structured_outputs"}})
    pipe = _pipeline(handler, structured_output="on", structured_output_fallback=False)

    with pytest.raises(StructuredOutputUnsupportedError):
        pipe._generate_direct(_MESSAGES, answer_regex=ANSWER_REGEX_NO_EVIDENCE)


def test_constraint_is_dropped_when_thinking_is_on() -> None:
    """Given thinking enabled, When generating, Then the grammar is not sent.

    The chat template pre-closes an empty ``<think>`` block only when thinking is off; with
    it on the prompt ends inside an open block, which this grammar's root forbids.
    """
    sent: list[bool] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        sent.append("structured_outputs" in json.loads(request.content))
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_sse_ok())

    pipe = _pipeline(_handler, structured_output="on")
    pipe._generate_direct(_MESSAGES, enable_thinking=True, answer_regex=ANSWER_REGEX_NO_EVIDENCE)
    pipe._generate_direct(_MESSAGES, enable_thinking=False, answer_regex=ANSWER_REGEX_NO_EVIDENCE)

    assert sent == [False, True]
