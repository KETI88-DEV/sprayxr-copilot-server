"""Detailed live replies keep short generation and visible-evidence budgets."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import pytest

from trainer.demo_api.copilot_session_context import build_query_instruction
from trainer.scripts.agent_pipeline import AgentPipeline, SemanticRouter, SessionContext
from trainer.scripts.openai_backend import OpenAIChatBackend


class RecordingBackend(OpenAIChatBackend):
    """Replace only the external completion request; keep pipeline budgeting real."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[dict[str, str]], dict[str, Any]]] = []

    def complete_chat(self, messages, **options):
        self.calls.append((messages, options))
        return "[답변] 분사 거리를 일정하게 유지하세요."


@pytest.fixture
def pipeline(monkeypatch):
    search = ModuleType("search_index")
    search_calls = []

    def search_and_rerank(query, *args):
        search_calls.append((query, args[-1]))
        return [
            {"chunk_id": f"DOC-fixture-C{index:04}", "text": f"RANKED-PASSAGE-{index}", "rerank_score": 1 - index / 10}
            for index in range(1, 7)
        ], {"total_ms": 3.0}

    search.search_and_rerank = search_and_rerank
    monkeypatch.setitem(sys.modules, "search_index", search)
    pipe = AgentPipeline.__new__(AgentPipeline)
    pipe.config = {
        "llm": {"max_model_len": 8192, "structured_output": "off", "append_citations": False},
        "generation": {"max_new_tokens": 1024},
        "retrieval": {"top_k": 12},
        "prompt": {
            "system": "DIAGNOSTIC-SCAFFOLD",
            "fast_system": " ".join(["VERBOSE-QA-SCAFFOLD"] * 80),
            "context_template": "문서:\n{context}",
        },
    }
    pipe.router = SemanticRouter()
    pipe.session_context = SessionContext(max_turns=3)
    pipe.index = pipe.id_map = pipe.embedder = pipe.reranker = None
    pipe.tokenizer = None
    pipe.model = RecordingBackend()
    return pipe, search_calls


def test_compact_analysis_limits_visible_evidence_after_full_retrieval(pipeline):
    pipe, search_calls = pipeline
    result = pipe.run(
        "분사 거리와 개선 방법을 설명해줘", force_route="fast", compact_analysis=True,
        max_tokens=192, retrieval_instruction="거리 개선", use_pipeline_history=False,
    )
    messages, options = pipe.model.calls[-1]

    assert search_calls == [("거리 개선", pipe.config)]
    assert pipe.config["retrieval"]["top_k"] == 12
    assert [chunk["chunk_id"] for chunk in result["rag_chunks"]] == [
        "DOC-fixture-C0001", "DOC-fixture-C0002", "DOC-fixture-C0003", "DOC-fixture-C0004",
    ]
    assert "RANKED-PASSAGE-4" in messages[1]["content"]
    assert "RANKED-PASSAGE-5" not in messages[1]["content"]
    assert result["citations"] == [
        "DOC-fixture-C0001", "DOC-fixture-C0002", "DOC-fixture-C0003", "DOC-fixture-C0004",
    ]
    assert options["max_tokens"] == 192
    assert options["enable_thinking"] is False
    assert "quick_cache_hit" not in result["metadata"]


def test_compact_analysis_replaces_verbose_qa_without_quick_sentence_rule(pipeline):
    pipe, _ = pipeline
    pipe.run("분사 거리와 개선 방법을 설명해줘", force_route="fast", compact_analysis=True, max_tokens=192)
    system = pipe.model.calls[-1][0][0]["content"]

    assert "VERBOSE-QA-SCAFFOLD" not in system
    assert "2문장" in system
    assert "수치" in system and "단위" in system
    assert "한 문장을 우선" not in system
    assert "[답변]" in system and "[근거]" in system


def test_compact_analysis_without_rag_never_invents_document_citation(pipeline):
    pipe, search_calls = pipeline
    result = pipe.run("현재 확인된 정보만 알려줘", force_route="fast", compact_analysis=True, use_rag=False, max_tokens=160)

    assert search_calls == []
    assert result["rag_chunks"] == [] and result["citations"] == []
    messages, options = pipe.model.calls[-1]
    assert "문서 번호" in messages[0]["content"]
    assert "[RAG 문맥]" not in messages[1]["content"]
    assert options["max_tokens"] == 160


def test_evaluation_default_keeps_all_evidence_prompt_budget_and_memory(pipeline):
    pipe, _ = pipeline
    result = pipe.run("분사 거리를 설명해줘", force_route="fast", session_id="EVAL")
    messages, options = pipe.model.calls[-1]

    assert len(result["rag_chunks"]) == 6
    assert "RANKED-PASSAGE-6" in messages[1]["content"]
    assert messages[0]["content"] == pipe.config["prompt"]["fast_system"]
    assert options["max_tokens"] == 256
    assert pipe.session_context.get_context("EVAL")


def test_compact_analysis_preserves_scoped_history_opt_out(pipeline):
    pipe, _ = pipeline
    pipe.session_context.update("SHARED", {"instruction": "OTHER-USER-SECRET", "response": "PRIVATE-MEMORY", "tool_calls": []})
    before = pipe.session_context.get_context("SHARED")
    pipe.run("현재 질문", force_route="fast", compact_analysis=True, session_id="SHARED", use_pipeline_history=False, max_tokens=192)

    assert "OTHER-USER-SECRET" not in str(pipe.model.calls[-1][0])
    assert "PRIVATE-MEMORY" not in str(pipe.model.calls[-1][0])
    assert pipe.session_context.get_context("SHARED") == before


@pytest.mark.parametrize("mode", ["quick", "detailed"])
def test_general_conversation_has_a_bounded_complete_reply_budget_for_both_modes(pipeline, mode):
    pipe, _ = pipeline
    answer = pipe.converse_copilot("안녕", recent_turns=[], facts={}, response_mode=mode)

    assert answer.startswith("[답변]")
    assert pipe.model.calls[-1][1]["max_tokens"] == 1024


def test_detailed_instruction_requests_the_complete_answer_first():
    instruction = build_query_instruction(
        "도포 방향의 근거와 개선 방법을 설명해줘", client_context=None, session_summary="", recent_turns=[], response_mode="detailed",
    )

    assert "질문에 먼저 직접 답" in instruction
    assert "2문장" not in instruction and "최대 2개" not in instruction
    assert "필요한 문단과 목록을 유지" in instruction and "추측하지" in instruction
    assert "원인, 권장 조치를 구분" not in instruction
