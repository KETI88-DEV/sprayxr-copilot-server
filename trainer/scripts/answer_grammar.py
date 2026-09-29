"""Decode-time grammars for the fast knowledge answer.

The grammar mirrors the ``[답변]``/``[근거]`` text scaffold the v6.1 adapter was
trained on, rather than asking for JSON: constraining the shape the model already
learned keeps the token distribution close to the SFT target. See
``docs/superpowers/specs/2026-09-29-knowledge-answer-structured-output-design.md`` §3.1.

What the grammar buys, structurally rather than by prompt text:

* ``[근거]`` can appear at most once (measured: 23 of 240 stored responses repeat it).
* ``[판정]``/``[원인]``/``[조치]`` cannot appear as headings, so the fast route cannot
  publish a verdict it has no measurements for.
* With nothing retrieved, ``cite`` is absent from the grammar, so a request that was
  given no documents cannot answer with document ids.

The ids the grammar admits are only shape-checked; membership in the request's
retrieved set is decided by ``live_response_contract.parse_knowledge_answer``.

**Unverified against a server.** The exact request field and the handling of the
multi-byte ``[답변]`` literal depend on the deployed vLLM build, which is chosen
outside this repo (``VLLM_IMAGE``). ``§3.6`` lists what to confirm; until then
``structured_output_fallback`` keeps a server that rejects the grammar serving.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final, Literal, get_args

StructuredOutputMode = Literal["grammar", "off"]

_MODES: Final[tuple[str, ...]] = get_args(StructuredOutputMode)
DEFAULT_MODE: Final[StructuredOutputMode] = "off"

# A line may contain brackets but must not start with one, which is what keeps a
# section heading from being reborn inside the answer body.
_SHARED_RULES: Final[str] = """
body ::= bodyline ("\\n" bodyline)*
bodyline ::= bodyhead bodytail*
bodyhead ::= [^\\n\\[]
bodytail ::= [^\\n]
"""

_CITE_RULES: Final[str] = """
cite ::= "DOC-" idchar+ "-C" digit digit digit digit
idchar ::= [a-zA-Z0-9_-]
digit ::= [0-9]
"""

KNOWLEDGE_GRAMMAR_WITH_EVIDENCE: Final[str] = (
    '\nroot ::= "[답변] " body evidence?\nevidence ::= "\\n[근거] " cite (", " cite){0,2}\n'
    + _SHARED_RULES
    + _CITE_RULES
).strip() + "\n"

KNOWLEDGE_GRAMMAR_NO_EVIDENCE: Final[str] = (
    '\nroot ::= "[답변] " body\n' + _SHARED_RULES
).strip() + "\n"


def knowledge_grammar(*, has_evidence: bool) -> str:
    """The grammar for one fast knowledge request; both variants are module constants."""
    return KNOWLEDGE_GRAMMAR_WITH_EVIDENCE if has_evidence else KNOWLEDGE_GRAMMAR_NO_EVIDENCE


def structured_output_mode(config: Mapping[str, Any]) -> StructuredOutputMode:
    """``llm.structured_output``; an unknown value fails loudly rather than degrading silently."""
    raw = config.get("llm", {}).get("structured_output", DEFAULT_MODE)
    mode = str(raw).strip().lower()
    if mode == "json":
        # Reserved for the spec's fallback (§3.1) and deliberately not silently degraded to
        # "off": a deployment that asked for structured output must not quietly lose it.
        raise ValueError("llm.structured_output='json' is reserved but not implemented; use 'grammar' or 'off'")
    if mode not in _MODES:
        raise ValueError(f"llm.structured_output must be one of {', '.join(_MODES)}; got {raw!r}")
    return mode  # type: ignore[return-value]


def structured_output_fallback(config: Mapping[str, Any]) -> bool:
    """Whether a server that rejects the grammar should be retried without it (default on)."""
    return bool(config.get("llm", {}).get("structured_output_fallback", True))
