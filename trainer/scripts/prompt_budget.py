"""Prompt token accounting against the model context window.

One place answers "how many tokens will this chat prompt occupy?".

* With a tokenizer, the count is exact: the messages are rendered through the
  model's own chat template (``add_generation_prompt=True``,
  ``enable_thinking=False``) and tokenized, which is what the HF path feeds to
  ``generate`` and what an OpenAI-compatible server (vLLM) checks against
  ``--max-model-len``. ``enable_thinking=False`` appends the empty
  ``<think>\\n\\n</think>\\n\\n`` block, so it is never shorter than the
  thinking-on render of the same messages.
* Without a tokenizer (a remote backend that has not loaded one), the fallback
  is a character-class upper bound that must not undercount; see
  :class:`CharClassTokenEstimator` for the evidence behind its costs.

Remote backends keep exact accounting by loading only the tokenizer locally
through :func:`load_token_counter`.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping, Sequence
from typing import Any, Final, Protocol

logger = logging.getLogger(__name__)

Message = Mapping[str, Any]

# Chat-template cost of one message with the Qwen3.6 template, measured
# 2026-09-15: system/user wrap in 5 tokens, an assistant turn in 9 (its empty
# think block included), the first tool result in 9 and each following one in
# 5. The fallback books every message at the worst case plus headroom.
FALLBACK_TOKENS_PER_MESSAGE: Final[int] = 16
# "<|im_start|>assistant\n<think>\n\n</think>\n\n" is 7 tokens.
FALLBACK_GENERATION_PROMPT_TOKENS: Final[int] = 16
# Headroom for the template's tool-declaration preamble around the schema JSON.
FALLBACK_TOOLS_PREAMBLE_TOKENS: Final[int] = 256

_HANGUL_SYLLABLE_FIRST: Final[int] = 0xAC00
_HANGUL_SYLLABLE_LAST: Final[int] = 0xD7A3


class PromptTokenCounter(Protocol):
    """Counts prompt tokens; ``exact`` says whether a real tokenizer is behind it."""

    exact: bool

    def count_messages(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> int: ...

    def count_text(self, text: str) -> int: ...


def _content_text(message: Message) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


class CharClassTokenEstimator:
    """Tokenizer-free upper bound on the token count of a prompt.

    Costs per character: ASCII 1 and any other non-Hangul character its UTF-8
    byte length are hard bounds for a byte-level BPE vocabulary, because every
    token spans at least one byte. A Hangul syllable costs 1, which is
    empirical: against the Qwen3.6-27B tokenizer, 156 of the 1,039 distinct
    syllables in the RAG index take 2-3 tokens on their own, yet this estimator
    undercounted none of the 10,829 index chunks (2026-09-15; median
    overestimate 1.92x, 1st percentile 1.24x).

    A flat characters-per-token ratio cannot give that guarantee here. The
    median Korean chunk is 0.53 tokens/char, but the 99th percentile is 0.82,
    the 99.9th 0.97, and chunks full of private-use PDF glyphs reach 2.33
    (each glyph is 3 byte tokens). The fast system prompt plus the eight most
    token-dense chunks averages 1.01 tokens/char, so even ``len(text)`` would
    undercount that prompt.
    """

    exact = False

    def count_text(self, text: str) -> int:
        total = 0
        for char in text:
            code = ord(char)
            if code < 0x80 or _HANGUL_SYLLABLE_FIRST <= code <= _HANGUL_SYLLABLE_LAST:
                total += 1
            else:
                total += len(char.encode("utf-8"))
        return total

    def count_messages(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> int:
        total = FALLBACK_GENERATION_PROMPT_TOKENS
        for message in messages:
            total += FALLBACK_TOKENS_PER_MESSAGE + self.count_text(_content_text(message))
        if tools:
            total += FALLBACK_TOOLS_PREAMBLE_TOKENS
            total += self.count_text(json.dumps(list(tools), ensure_ascii=False))
        return total


def _encoded_length(encoded: Any) -> int:
    if isinstance(encoded, Mapping) or hasattr(encoded, "keys"):
        input_ids = encoded["input_ids"]
    else:
        input_ids = encoded
    if input_ids and isinstance(input_ids[0], (list, tuple)):
        raise ValueError("Expected a single tokenized sequence, got a batch")
    return len(input_ids)


class TokenizerTokenCounter:
    """Exact prompt token count through the model's own chat template."""

    exact = True

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    def count_text(self, text: str) -> int:
        return _encoded_length(self._tokenizer(text, add_special_tokens=False))

    def count_messages(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> int:
        kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        if tools:
            kwargs["tools"] = list(tools)
        encoded = self._tokenizer.apply_chat_template([dict(m) for m in messages], **kwargs)
        return _encoded_length(encoded)


def token_counter_for(tokenizer: Any | None) -> PromptTokenCounter:
    """Exact counter when a chat tokenizer is at hand, conservative estimate otherwise."""
    if tokenizer is not None and callable(getattr(tokenizer, "apply_chat_template", None)):
        return TokenizerTokenCounter(tokenizer)
    return CharClassTokenEstimator()


def load_token_counter(
    tokenizer_path: str | os.PathLike[str] | None,
    *,
    trust_remote_code: bool = True,
) -> PromptTokenCounter:
    """Loads only a tokenizer, for backends whose model weights live elsewhere.

    An OpenAI-compatible remote server rejects prompts longer than its
    ``--max-model-len``; loading the same model's tokenizer here keeps the
    client-side budget exact. A missing path or a failed load falls back to
    :class:`CharClassTokenEstimator` and says so in the log.
    """
    if not tokenizer_path:
        return CharClassTokenEstimator()
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=trust_remote_code)
    except Exception:
        logger.warning(
            "Tokenizer load failed for %s; prompt budget falls back to the conservative estimate",
            tokenizer_path,
            exc_info=True,
        )
        return CharClassTokenEstimator()
    return TokenizerTokenCounter(tokenizer)


def context_window(config: Mapping[str, Any]) -> int:
    """``llm.max_model_len``: the context window every backend must share."""
    return int(config.get("llm", {}).get("max_model_len", 4096))


def reserved_generation_tokens(config: Mapping[str, Any]) -> int:
    return int(config.get("generation", {}).get("max_new_tokens", 768))


def prompt_token_budget(config: Mapping[str, Any]) -> int:
    """Prompt tokens that still leave ``generation.max_new_tokens`` inside the window."""
    return context_window(config) - reserved_generation_tokens(config)
