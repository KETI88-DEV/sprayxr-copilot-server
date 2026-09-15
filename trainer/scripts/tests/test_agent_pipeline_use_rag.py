"""run(use_rag=...): retrieval can be skipped without changing the default for existing callers."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import pytest

from trainer.scripts import agent_pipeline
from trainer.scripts.agent_pipeline import AgentPipeline, SemanticRouter, SessionContext


@pytest.fixture
def search_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    module = ModuleType("search_index")

    def _search_and_rerank(query: str, *_args: Any, **_kwargs: Any) -> tuple[list[dict[str, Any]], dict[str, float]]:
        calls.append(query)
        return [{"chunk_id": "DOC-doc_0007-C0002", "text": "RAG-SENTINEL 문서 본문", "rerank_score": 0.8}], {"total_ms": 3.0}

    setattr(module, "search_and_rerank", _search_and_rerank)
    monkeypatch.setitem(sys.modules, "search_index", module)
    return calls


def _pipeline() -> tuple[AgentPipeline, list[list[dict[str, str]]]]:
    agent_pipeline._ensure_scripts_path()
    pipe = AgentPipeline.__new__(AgentPipeline)
    pipe.config = {
        "llm": {"max_model_len": 8192, "append_citations": True},
        "generation": {"max_new_tokens": 1024},
        "prompt": {"fast_system": "SYSTEM-FAST", "context_template": "문서:\n{context}"},
    }
    pipe.session_context = SessionContext(max_turns=3)
    pipe.router = SemanticRouter()
    pipe.tokenizer = None
    pipe.index = pipe.id_map = pipe.embedder = pipe.reranker = None
    prompts: list[list[dict[str, str]]] = []

    def _generate_direct(messages: list[dict[str, str]], enable_thinking: bool = False) -> str:
        prompts.append(messages)
        return "[답변] 저도막은 규정보다 얇은 도막이다."

    pipe._generate_direct = _generate_direct
    return pipe, prompts


def test_run_without_rag_skips_search_and_cites_nothing(search_calls: list[str]) -> None:
    pipe, prompts = _pipeline()

    result = pipe.run(instruction="저도막이 뭐야", force_route="fast", enable_thinking=False, use_rag=False)

    assert search_calls == []
    assert result["rag_chunks"] == []
    assert result["citations"] == []
    assert "RAG-SENTINEL" not in prompts[0][1]["content"]
    assert "[RAG 문맥]" not in prompts[0][1]["content"]
    assert result["response"] == "[답변] 저도막은 규정보다 얇은 도막이다."


def test_run_defaults_to_searching(search_calls: list[str]) -> None:
    pipe, prompts = _pipeline()

    result = pipe.run(instruction="저도막이 뭐야", force_route="fast", enable_thinking=False)

    assert search_calls == ["저도막이 뭐야"]
    assert [chunk["chunk_id"] for chunk in result["rag_chunks"]] == ["DOC-doc_0007-C0002"]
    assert "RAG-SENTINEL" in prompts[0][1]["content"]
    assert result["citations"] == ["DOC-doc_0007-C0002"]
