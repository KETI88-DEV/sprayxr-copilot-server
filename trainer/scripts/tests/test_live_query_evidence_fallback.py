"""Live-only evidence deferral must happen before generation or the answer cache."""
from __future__ import annotations

import sys
from types import ModuleType

import pytest

from trainer.scripts.tests.test_agent_pipeline_use_rag import _pipeline


DOC = "DOC-doc_0007-C0002"


def search(monkeypatch, chunks, *, error=None):
    calls = []
    module = ModuleType("search_index")

    def retrieve(query, *_args, **_kwargs):
        calls.append(query)
        if error is not None:
            raise error
        return chunks, {"total_ms": 3.0}

    module.search_and_rerank = retrieve
    monkeypatch.setitem(sys.modules, "search_index", module)
    return calls


def run_live(pipe, **kwargs):
    return pipe.run(instruction="원문 질문", answer_question="원문 질문", force_route="fast",
                    enable_thinking=False, use_pipeline_history=False,
                    defer_without_evidence=True, evidence_score_threshold=4.0, **kwargs)


@pytest.mark.parametrize("chunks,reason", [
    ([], "empty"),
    ([{"chunk_id": DOC, "text": "  \n", "rerank_score": 6.0}], "empty"),
    ([{"chunk_id": DOC, "text": "다른 주제 문단", "rerank_score": 3.9}], "below_threshold"),
])
def test_live_irrelevant_evidence_defers_without_generation_or_cache(monkeypatch, chunks, reason):
    calls = search(monkeypatch, chunks)
    pipe, generated = _pipeline()
    result = run_live(pipe, quick_cache_scope="AI-TEST")
    assert calls == ["원문 질문"]
    assert generated == []
    assert not hasattr(pipe, "_quick_answer_cache")
    assert result["response"] == ""
    assert result["rag_chunks"] == [] and result["citations"] == []
    assert result["metadata"]["retrieval_decision"] == {
        "status": "no_relevant_evidence", "reason": reason, "attempted": True,
    }


def test_live_supported_evidence_is_generated_once_and_cited(monkeypatch):
    search(monkeypatch, [{"chunk_id": DOC, "text": "저도막은 규정보다 얇은 도막이다.", "rerank_score": 4.0}])
    pipe, generated = _pipeline()
    def generate(messages, **_kwargs):
        generated.append(messages)
        return "[답변] 저도막은 규정보다 얇은 도막이다.\n[근거] " + DOC
    pipe._generate_direct = generate
    # A definition requires no extra semantic review model call.
    result = pipe.run(instruction="저도막이 뭐야?", answer_question="저도막이 뭐야?",
                      force_route="fast", enable_thinking=False, use_pipeline_history=False,
                      defer_without_evidence=True, evidence_score_threshold=4.0)
    assert len(generated) == 1
    assert result["rag_chunks"][0]["chunk_id"] == DOC
    assert result["citations"] == [DOC]
    assert result["metadata"].get("retrieval_decision", {}).get("status") != "no_relevant_evidence"


def test_default_evaluation_still_generates_without_evidence(monkeypatch):
    search(monkeypatch, [])
    pipe, generated = _pipeline()
    result = pipe.run(instruction="평가 질문", force_route="fast", enable_thinking=False)
    assert len(generated) == 1 and result["response"]
    assert "retrieval_decision" not in result["metadata"]


def test_explicit_no_rag_does_not_defer_or_search(monkeypatch):
    calls = search(monkeypatch, [])
    pipe, generated = _pipeline()
    result = run_live(pipe, use_rag=False)
    assert calls == [] and len(generated) == 1
    assert result["metadata"].get("retrieval_decision", {}).get("status") != "no_relevant_evidence"


def test_search_failure_is_not_reported_as_missing_evidence(monkeypatch):
    search(monkeypatch, [], error=RuntimeError("retrieval offline"))
    pipe, generated = _pipeline()
    with pytest.raises(RuntimeError, match="retrieval offline"):
        run_live(pipe)
    assert generated == []


@pytest.mark.parametrize("score", [None, float("nan"), float("inf"), "bad"])
def test_unknown_score_never_certifies_an_irrelevant_document(monkeypatch, score):
    search(monkeypatch, [{"chunk_id": DOC, "text": "문서 본문", "rerank_score": score}])
    pipe, _ = _pipeline()
    # Corrupt retrieval metadata is an error, not permission to use general knowledge.
    with pytest.raises(ValueError, match="score"):
        run_live(pipe)
