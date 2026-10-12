"""Decode-time constraint for the fast knowledge answer.

The constraint mirrors the ``[답변]``/``[근거]`` text scaffold the v6.1 adapter was
trained on, rather than asking for JSON: constraining the shape the model already
learned keeps the token distribution close to the SFT target. See
``docs/superpowers/specs/2026-09-29-knowledge-answer-structured-output-design.md``.

**Measured against the deployed server (vLLM 0.29.0, Spark, 2026-09-29).** The
constraint travels as ``structured_outputs.regex`` on the chat request. What was
confirmed there, with a falsifiable probe (a sentinel the model has no reason to emit):

* ``structured_outputs`` really constrains generation — ``{"regex": "SENTINEL-[0-9][0-9][0-9]-END"}``
  produced exactly ``SENTINEL-102-END``.
* ``guided_grammar``, at the top level and nested under ``extra_body``, is **silently
  ignored**: the request returns 200 and the answer is unconstrained. Never send it.
* ``guided_decoding_backend`` is not a field of ``ChatCompletionRequest`` (only of the
  batch variant), so the backend is chosen by the server.
* ``structured_outputs.grammar`` accepts the GBNF dialect too, but ``regex`` is used
  because it has no dialect ambiguity.

What the constraint buys, structurally rather than by prompt text:

* ``[근거]`` can appear at most once (measured: 23 of 240 stored responses repeat it).
* ``[판정]``/``[원인]``/``[조치]`` cannot appear, so the fast route cannot publish a
  verdict it has no measurements for. The body forbids ``[`` **anywhere**, not only at a
  line start: an unconstrained probe appended ``[분석] 이 답변은…`` on the same line, and
  a line-start-only rule would have let that through.
* With nothing retrieved, the evidence clause is absent from the pattern, so a request
  that was given no documents cannot answer with document ids.

The ids the pattern admits are only shape-checked; membership in the request's retrieved
set is decided by ``live_response_contract.parse_knowledge_answer``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final, Literal, get_args

StructuredOutputMode = Literal["on", "off"]

_MODES: Final[tuple[str, ...]] = get_args(StructuredOutputMode)
DEFAULT_MODE: Final[StructuredOutputMode] = "off"

# A body line holds at least one character and no ``[`` at all; see the module docstring
# for why the bracket is banned outright rather than only at a line start.
# 한자를 이 문자 클래스에서 빼면 안 된다. 2026-10-12 Spark(vLLM 0.29 structured_outputs) 실측: 같은 프롬프트에
# 정규식만 바꿔 보내자, 옛 클래스는 정상 한국어였고 한자 범위를 뺀 클래스는 답이 **중국어**로 바뀌고 띄어쓰기가
# 사라졌다 — 엔진이 비ASCII 범위를 넣은 부정 클래스를 거꾸로 해석한다. 파이썬 re 와 로컬 xgrammar 0.2.8 로는
# 재현되지 않으므로 여기 시험으로 잡을 수 없다. 바꾸려면 Spark vLLM 에 직접 A/B 로 확인해야 한다.
_BODY_LINE: Final[str] = r"[^\n\[]+"
_DOC_ID: Final[str] = r"DOC-[A-Za-z0-9_-]+-C[0-9][0-9][0-9][0-9]"

ANSWER_REGEX_NO_EVIDENCE: Final[str] = rf"\[답변\] {_BODY_LINE}(\n{_BODY_LINE})*"

# Body lines cannot start with ``[`` and the evidence line must, so the newline branch is
# unambiguous. At most three ids, matching the observed citation count (median 1, max 3).
ANSWER_REGEX_WITH_EVIDENCE: Final[str] = (
    ANSWER_REGEX_NO_EVIDENCE + rf"(\n\[근거\] {_DOC_ID}(, {_DOC_ID}){{0,2}})?"
)


def answer_regex(*, has_evidence: bool) -> str:
    """The constraint for one fast knowledge request; both variants are module constants."""
    return ANSWER_REGEX_WITH_EVIDENCE if has_evidence else ANSWER_REGEX_NO_EVIDENCE


def structured_output_mode(config: Mapping[str, Any]) -> StructuredOutputMode:
    """``llm.structured_output``; an unknown value fails loudly rather than degrading silently."""
    raw = config.get("llm", {}).get("structured_output", DEFAULT_MODE)
    mode = str(raw).strip().lower()
    if mode in ("grammar", "json"):
        # Both named a transport that this deployment does not use. Not silently mapped to
        # "on": a config written against the old draft should be corrected, not guessed at.
        raise ValueError(
            f"llm.structured_output={raw!r} is obsolete; the constraint travels as "
            "structured_outputs.regex. Use 'on' or 'off'."
        )
    if mode not in _MODES:
        raise ValueError(f"llm.structured_output must be one of {', '.join(_MODES)}; got {raw!r}")
    return mode  # type: ignore[return-value]


def structured_output_fallback(config: Mapping[str, Any]) -> bool:
    """Whether a server that rejects the constraint should be retried without it (default on)."""
    return bool(config.get("llm", {}).get("structured_output_fallback", True))
