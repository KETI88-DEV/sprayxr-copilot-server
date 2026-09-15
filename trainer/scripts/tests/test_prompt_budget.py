"""Prompt token budget: exact counting, the tokenizer-free bound, trimming edges, tool-loop report."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from trainer.scripts.agent_pipeline import AgentPipeline
from trainer.scripts.prompt_budget import (
    CharClassTokenEstimator,
    TokenizerTokenCounter,
    load_token_counter,
    prompt_token_budget,
    token_counter_for,
)
from trainer.scripts.tool_calling_adapter import ToolCallingAdapter

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = Path("/models/Qwen3.6-27B")

# An index-style Korean chunk: Korean prose with digits, units and punctuation.
KOREAN_CHUNK = (
    "3. 용어의 정의\n(1) 이 지침에서 사용하는 용어의 정의는 다음과 같다.\n"
    "(가) “분무지역(Spray area)”이라 함은 일반적으로 옥외, 넓은 옥내 또는 공간 내의 일부에서 "
    "분무되는 지역을 말하며, 국소배기/환기시스템이 구비되어 있는 자동 운전공정에서는 분무작업의 "
    "직경로(Direct path of spray operation), 수동조작에서는 도장표면에 분사되는 180°가 최대 분무지역으로 한다.\n"
    "건조도막두께(DFT)는 90/10 규칙에 따라 측정점의 90% 이상이 규정치 100㎛ 이상이어야 하며, "
    "나머지 10%는 규정치의 90%(90㎛) 미만이어서는 안 된다. 에폭시 도료의 가사시간은 20℃에서 약 6시간이다."
)
# PDF extraction leaves private-use glyphs; each one is several byte tokens.
PRIVATE_USE_CHUNK = "강재 교체\n\n≤"
DIGIT_JSON = '{"measurements": [95, 102, 88, 105, 99], "coordinates": [{"x": 1240.2, "y": -210.5, "z": 340.1}]}'
FIXTURE_TEXTS = [KOREAN_CHUNK, PRIVATE_USE_CHUNK, DIGIT_JSON]


@pytest.fixture(scope="module")
def exact() -> TokenizerTokenCounter:
    if not (MODEL_PATH / "tokenizer.json").is_file():
        pytest.skip(f"real tokenizer not available at {MODEL_PATH}")
    from transformers import AutoTokenizer

    return TokenizerTokenCounter(AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True))


def _messages(user: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "너는 선박 도장 분야 전문 AI 어시스턴트이다."},
        {"role": "user", "content": user},
    ]


def test_exact_count_is_the_rendered_chat_template(exact: TokenizerTokenCounter) -> None:
    tokenizer = exact._tokenizer
    messages = _messages(KOREAN_CHUNK)
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

    assert rendered.endswith("<think>\n\n</think>\n\n")
    assert exact.count_messages(messages) == len(tokenizer(rendered, add_special_tokens=False)["input_ids"])


def test_legacy_half_length_estimate_undercounts_korean(exact: TokenizerTokenCounter) -> None:
    """The fixture must be able to catch an undercounting estimator."""
    assert exact.count_text(KOREAN_CHUNK) > len(KOREAN_CHUNK) // 2
    assert exact.count_text(PRIVATE_USE_CHUNK) > len(PRIVATE_USE_CHUNK) / 1.2


@pytest.mark.parametrize("text", FIXTURE_TEXTS, ids=["korean", "private_use", "digit_json"])
def test_fallback_never_undercounts_real_tokens(exact: TokenizerTokenCounter, text: str) -> None:
    estimator = CharClassTokenEstimator()

    assert estimator.count_text(text) >= exact.count_text(text)
    assert estimator.count_messages(_messages(text)) >= exact.count_messages(_messages(text))


def test_fallback_counts_tool_turns_at_least_as_high(exact: TokenizerTokenCounter) -> None:
    conversation = [
        *_messages(KOREAN_CHUNK),
        {"role": "assistant", "content": '<tool_call>{"name": "dft_judge", "arguments": {}}</tool_call>'},
        {"role": "tool", "content": DIGIT_JSON},
        {"role": "tool", "content": PRIVATE_USE_CHUNK},
    ]

    assert CharClassTokenEstimator().count_messages(conversation) >= exact.count_messages(conversation)


def test_counter_selection_and_loader_fallbacks(tmp_path: Path) -> None:
    assert isinstance(token_counter_for(None), CharClassTokenEstimator)
    assert isinstance(load_token_counter(None), CharClassTokenEstimator)
    assert isinstance(load_token_counter(tmp_path / "missing-tokenizer"), CharClassTokenEstimator)
    assert prompt_token_budget({"llm": {"max_model_len": 8192}, "generation": {"max_new_tokens": 1024}}) == 7168


def _config(max_model_len: int) -> dict[str, Any]:
    return {
        "llm": {"max_model_len": max_model_len},
        "generation": {"max_new_tokens": 1024},
        "prompt": {"system": "진단 시스템 프롬프트", "context_template": "문서:\n{context}"},
    }


def _chunks() -> list[dict[str, Any]]:
    return [
        {"chunk_id": f"DOC-doc_0001-C000{i}", "text": KOREAN_CHUNK, "rerank_score": 0.9 - i / 10}
        for i in range(1, 5)
    ]


def _pipeline(config: dict[str, Any], counter: Any | None) -> AgentPipeline:
    pipe = AgentPipeline.__new__(AgentPipeline)
    pipe.config = config
    if counter is not None:
        pipe._token_counter = counter
    return pipe


def _kept(messages: list[dict[str, str]], chunks: list[dict[str, Any]]) -> int:
    return sum(1 for c in chunks if f"[{c['chunk_id']}]" in messages[1]["content"])


def _build(pipe: AgentPipeline, chunks: list[dict[str, Any]], **kwargs: Any) -> list[dict[str, str]]:
    return pipe._build_messages(
        instruction="DFT 90/10 기준을 설명해줘",
        environment={"zone_id": "B-3"},
        rag_chunks=chunks,
        session_context_summary="",
        **kwargs,
    )


def test_trimming_keeps_a_prompt_exactly_at_budget_and_drops_one_token_over(exact: TokenizerTokenCounter) -> None:
    chunks = _chunks()
    three_chunk_tokens = exact.count_messages(_build(_pipeline(_config(1 << 20), exact), chunks[:3]))

    at_budget = _build(_pipeline(_config(three_chunk_tokens + 1024), exact), chunks)
    one_over = _build(_pipeline(_config(three_chunk_tokens + 1024 - 1), exact), chunks)

    assert _kept(at_budget, chunks) == 3
    assert exact.count_messages(at_budget) == three_chunk_tokens
    assert _kept(one_over, chunks) == 2


def test_native_tool_declarations_count_against_the_budget(exact: TokenizerTokenCounter) -> None:
    chunks = _chunks()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "dft_judge",
                "description": "DFT 합부 판정 " * 40,
                "parameters": {"type": "object", "properties": {"measurements": {"type": "array"}}},
            },
        }
    ]
    all_chunk_tokens = exact.count_messages(_build(_pipeline(_config(1 << 20), exact), chunks))
    window = all_chunk_tokens + 1024

    assert _kept(_build(_pipeline(_config(window), exact), chunks), chunks) == 4
    assert _kept(_build(_pipeline(_config(window), exact), chunks, tools=tools), chunks) < 4


def test_pipeline_without_tokenizer_trims_on_the_conservative_estimate(exact: TokenizerTokenCounter) -> None:
    chunks = _chunks()
    all_chunk_tokens = exact.count_messages(_build(_pipeline(_config(1 << 20), exact), chunks))
    config = _config(all_chunk_tokens + 1024)

    fallback_pipe = _pipeline(config, None)
    fallback = _build(fallback_pipe, chunks)

    assert fallback_pipe._prompt_token_counter().exact is False
    assert _kept(_build(_pipeline(config, exact), chunks), chunks) == 4
    assert _kept(fallback, chunks) < 4
    assert CharClassTokenEstimator().count_messages(fallback) <= prompt_token_budget(config)


class _ScriptedModel:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def generate_response(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> str:
        _ = (messages, tools)
        return self._responses.pop(0)


class _LengthCounter:
    exact = True

    def count_text(self, text: str) -> int:
        return len(text)

    def count_messages(self, messages: list[dict[str, Any]], *, tools: Any = None) -> int:
        _ = tools
        return sum(len(str(m.get("content", ""))) for m in messages)


def _adapter() -> ToolCallingAdapter:
    return ToolCallingAdapter(
        schema_dir=REPO_ROOT / "trainer/configs/tool_schemas",
        protocol_path=REPO_ROOT / "trainer/configs/tool_calling_protocol.json",
        mcp_base_url="http://mcp.invalid",
        pass_native_tools=False,
    )


def _run_loop(adapter: ToolCallingAdapter, **kwargs: Any) -> dict[str, Any]:
    async def _dispatch(name: str, arguments: dict[str, Any], session_id: str | None = None) -> dict[str, Any]:
        _ = (arguments, session_id)
        return {"status": "success", "result": {"tool": name, "values": list(range(200))}}

    adapter.dispatch_tool_call = _dispatch  # type: ignore[method-assign]
    model = _ScriptedModel(
        ['<tool_call>{"name":"dft_judge","arguments":{"measurements":[95],"spec":{"nominal":100}}}</tool_call>', "최종 답변"]
    )
    return asyncio.run(
        adapter.run_tool_loop(
            messages=[{"role": "user", "content": "판정해줘"}],
            model=model,
            tokenizer=object(),
            max_rounds=3,
            max_tokens=256,
            tool_max_tokens=128,
            **kwargs,
        )
    )


def test_tool_loop_reports_accumulated_prompt_growth_without_changing_output() -> None:
    plain = _run_loop(_adapter())
    measured = _run_loop(_adapter(), token_counter=_LengthCounter(), max_model_len=500)

    assert "context" not in plain
    assert measured["final_response"] == plain["final_response"] == "최종 답변"
    assert measured["rounds"] == plain["rounds"] == 2
    calls = measured["context"]["calls"]
    assert [c["stage"] for c in calls] == ["round", "round"]
    assert calls[0]["prompt_tokens"] == len("판정해줘")
    assert calls[1]["prompt_tokens"] > 500 > calls[0]["prompt_tokens"] + calls[0]["max_new_tokens"]
    assert measured["context"]["overflow"] is True
