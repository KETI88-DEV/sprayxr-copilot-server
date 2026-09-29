from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import pytest

from trainer.scripts.agent_pipeline import AgentPipeline, SemanticRouter, SessionContext

DIAGNOSTIC_PROMPT = "진단용 시스템 프롬프트 [분석] [판정] [원인] [조치] [근거]"
KNOWLEDGE_PROMPT = "지식 질문용 시스템 프롬프트 [답변] [근거]"


def _config() -> dict[str, Any]:
    return {
        "llm": {"max_model_len": 8192, "append_citations": False},
        "generation": {"max_new_tokens": 1024},
        "prompt": {
            "system": DIAGNOSTIC_PROMPT,
            "fast_system": KNOWLEDGE_PROMPT,
            "context_template": "문서:\n{context}",
            "user_template": "질문: {question}",
        },
    }


def _pipeline(monkeypatch: pytest.MonkeyPatch) -> tuple[AgentPipeline, dict[str, Any]]:
    fake_search = ModuleType("search_index")
    setattr(
        fake_search,
        "search_and_rerank",
        lambda *_args, **_kwargs: ([], {}),
    )
    monkeypatch.setitem(sys.modules, "search_index", fake_search)

    pipe = AgentPipeline.__new__(AgentPipeline)
    pipe.config = _config()
    pipe.router = SemanticRouter()
    pipe.session_context = SessionContext(max_turns=3)
    pipe.index = None
    pipe.id_map = []
    pipe.embedder = None
    pipe.reranker = None

    captured: dict[str, Any] = {}

    def _fake_generate(
        messages: list[dict[str, str]], enable_thinking: bool = False, **_kwargs: object
    ) -> str:
        _ = enable_thinking
        captured["messages"] = messages
        return "블라스팅은 연마재를 분사해 강재 표면의 녹과 이물질을 제거하는 표면처리 공정이다."

    pipe._generate_direct = _fake_generate
    return pipe, captured


def test_fast_route_answers_knowledge_question_without_diagnostic_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a definition question, When the fast route runs, Then the knowledge prompt is used."""
    pipe, captured = _pipeline(monkeypatch)

    result = pipe.run(instruction="블라스팅이 뭐야", force_route="fast")

    assert result["route"] == "fast"
    assert captured["messages"][0]["content"] == KNOWLEDGE_PROMPT
    assert "[판정]" not in captured["messages"][0]["content"]


def test_tool_route_prompt_construction_keeps_the_diagnostic_system_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given no override, When messages are built, Then the diagnostic prompt is preserved."""
    pipe, _captured = _pipeline(monkeypatch)

    messages = pipe._build_messages(
        instruction="지금 도막 상태 어때?",
        environment={"zone_id": "A-2"},
        rag_chunks=[],
        session_context_summary="",
    )

    assert messages[0]["content"] == DIAGNOSTIC_PROMPT
