"""결함 원인 질문의 원문 근거 묶음(2026-11-18 '저도막 원인' 오답)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from trainer.demo_api.defect_cause import cause_context, defect_cause_topic, pinned_chunks
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.tests.test_query_contextual_dialogue import DialoguePipeline, ask, client
from trainer.scripts.agent_pipeline import AgentPipeline

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


@pytest.mark.parametrize("prompt, previous, expected", [
    ("저도막의 원인은?", [], "저도막"),
    ("저도막은 왜 생겨?", [], "저도막"),
    ("도막 두께 부족이 생기는 이유", [], "저도막"),
    ("과도막은 왜 생겨?", [], "과도막"),
    ("그럼 원인은?", ["저도막이 뭐야?"], "저도막"),
    ("저도막이 뭐야?", [], None),            # 정의 질문은 원인 묶음을 쓰지 않는다
    ("그럼 원인은?", [], None),
    ("처짐은 왜 생겨?", [], None),           # 묶음에 없는 결함
    ("그럼 원인은?", ["이격거리는 얼마가 적당해?"], None),
])
def test_cause_question_finds_its_defect(prompt, previous, expected):
    assert defect_cause_topic(prompt, previous) == expected


def test_bundle_uses_no_excluded_chunk():
    excluded = set(json.loads((CONFIGS / "evidence_exclusions.json").read_text(encoding="utf-8"))["조각"])
    bundle = json.loads((CONFIGS / "defect_cause_evidence.json").read_text(encoding="utf-8"))["결함"]
    assert all(entry["조각"] and not excluded & set(entry["조각"]) for entry in bundle.values())


def test_context_rule_forbids_flipping_cause_and_effect():
    rule = cause_context("과도막")
    assert "질문한 결함: 과도막" in rule and "방향을 바꾸지" in rule and "B가 A의 원인이라는 뜻이 아닙니다" in rule


def _pipeline(rows):
    pipeline = AgentPipeline.__new__(AgentPipeline)
    pipeline.id_map = rows
    pipeline.reranker = None
    pipeline.config = {"reranker": {"top_k_rerank": 4}}
    return pipeline


def test_pinned_passages_lead_skip_excluded_and_keep_the_evidence_count():
    rows = [{"chunk_id": f"C{i}", "text": f"본문 {i}"} for i in range(5)]
    search = SimpleNamespace(EXCLUDED_CHUNKS=frozenset({"C1"}))
    retrieved = [{"chunk_id": c, "text": "검색", "rerank_score": 5.0} for c in ("C2", "R1", "R2", "R3")]
    found = _pipeline(rows)._pin_evidence("저도막 원인", ["C0", "C1", "C2", "C9"], retrieved, search)
    assert [c["chunk_id"] for c in found] == ["C0", "C2", "R1", "R2"]
    assert [c.get("pinned", False) for c in found] == [True, True, False, False]
    assert found[0]["text"] == "본문 0"  # 색인 원문 그대로


def test_pinned_passage_uses_the_corrected_body_not_the_superseded_original():
    rows = [{"chunk_id": "C0", "text": "틀린 원본", "superseded": True}, {"chunk_id": "C0", "text": "고친 본문", "corrected": True}]
    found = _pipeline(rows)._pin_evidence("q", ["C0"], [], SimpleNamespace(EXCLUDED_CHUNKS=frozenset()))
    assert [c["text"] for c in found] == ["고친 본문"]


def test_router_pins_the_bundle_and_adds_the_rule_for_a_cause_question():
    pipeline = DialoguePipeline()
    ask(client(pipeline), "저도막의 원인은?", response_mode="detailed")
    call = pipeline.calls[-1]
    assert call["pinned_chunk_ids"] == pinned_chunks("저도막")
    assert "[결함 원인 질문]" in call["instruction"]


def test_router_inherits_the_defect_from_the_previous_question():
    store = SequenceStore(":memory:")
    store.record_query_turn(session_key="CHAT:U", user_id="U", prompt="저도막이 뭐야?", answer="저도막은 도막 두께 부족입니다.",
                            task_type="knowledge_search", created_at="2026-11-18T05:05:01Z", answer_source="documents")
    pipeline = DialoguePipeline({"kind": "coating_knowledge", "intent": "knowledge", "scope": "unspecified", "context_relation": "followup"})
    ask(client(pipeline, store), "그럼 원인은?", response_mode="quick")
    assert pipeline.calls[-1]["pinned_chunk_ids"] == pinned_chunks("저도막")


class _PinnedLowScore(DialoguePipeline):
    """골라 둔 조각의 재정렬 점수가 문턱(3.28)보다 낮은 운영 사례(과도막 묶음 최고 2.59)."""

    config = {"confidence": {"abstain_below_rerank_score": 3.28, "quick_abstain_below_rerank_score": 3.28}}

    def __init__(self, pinned: bool):
        super().__init__()
        self.pinned = pinned

    def run(self, **kwargs):
        self.calls.append(kwargs)
        chunk = {"chunk_id": "DOC-doc_0193-C0184", "text": "중첩률이 너무 높으면 같은 위치에 도료가 과도하게 겹쳐져 도막 두꺼움이 생긴다.",
                 "rerank_score": 1.5, **({"pinned": True} if self.pinned else {})}
        return {"response": "[답변] 중첩률이 너무 높으면 도료가 과도하게 겹쳐져 도막 두꺼움이 생깁니다.\n[근거] DOC-doc_0193-C0184",
                "rag_chunks": [chunk]}


@pytest.mark.parametrize("pinned, kept", [(True, True), (False, False)])
def test_weak_score_gate_skips_only_curated_passages(pinned, kept):
    result = ask(client(_PinnedLowScore(pinned)), "과도막은 왜 생겨?", response_mode="detailed")
    assert ("중첩률" in result["answer"]) is kept, result["answer"]


def test_router_leaves_other_questions_unpinned():
    pipeline = DialoguePipeline()
    ask(client(pipeline), "이격거리는 얼마가 적당해?", response_mode="quick")
    assert "pinned_chunk_ids" not in pipeline.calls[-1]
