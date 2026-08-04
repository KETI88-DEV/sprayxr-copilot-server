from __future__ import annotations

from trainer.scripts.agent_pipeline import AgentPipeline


class _InputIds:
    shape = (1, 2)


class _Tokenizer:
    eos_token_id = 0

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        _ = (messages, tokenize, add_generation_prompt, enable_thinking)
        return "prompt"

    def __call__(
        self,
        prompts: list[str],
        *,
        return_tensors: str,
        return_attention_mask: bool,
    ) -> dict[str, _InputIds]:
        _ = (prompts, return_tensors, return_attention_mask)
        return {"input_ids": _InputIds()}

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        _ = skip_special_tokens
        return "x" * len(token_ids)


class _Model:
    def generate(self, **kwargs: _InputIds | int | float | bool) -> list[list[int]]:
        max_new_tokens = kwargs["max_new_tokens"]
        assert isinstance(max_new_tokens, int)
        return [[1, 2, *([3] * max_new_tokens)]]


def test_generate_direct_caps_fast_response_to_route_budget() -> None:
    pipeline = AgentPipeline.__new__(AgentPipeline)
    pipeline.model = _Model()
    pipeline.tokenizer = _Tokenizer()
    pipeline.config = {
        "generation": {
            "max_new_tokens": 1024,
            "do_sample": False,
            "repetition_penalty": 1.15,
        },
    }

    response = pipeline._generate_direct(
        [{"role": "user", "content": "도막 성공 기준 알려줘"}],
        enable_thinking=False,
    )

    assert len(response) == 256
