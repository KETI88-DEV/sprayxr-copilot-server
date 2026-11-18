"""A broad question can use a verified subset without certifying unsupported claims."""
import pytest

from trainer.scripts.answer_semantics import review_answer


EVIDENCE = [
    {"chunk_id": "DOC-scope-C0001", "text": "다른 공정의 작업 순서입니다."},
    {"chunk_id": "DOC-condition-C0001", "text": "조건 A에서는 현상 P가 발생할 수 있습니다."},
]


def test_partial_support_gets_one_scoped_repair_and_a_new_source_check():
    outputs = iter([
        "이유: 일부 원인만 자료에 있음\n근거: 2\n판정: INSUFFICIENT",
        "이유: 조건 A에 한정한 설명과 일치\n근거: 2\n판정: OK",
    ])
    repairs = []
    def regenerate(instruction, _timeout):
        repairs.append(instruction)
        return "[답변] 자료에서 확인되는 한 가지 조건은 A입니다. 다른 원인은 이 자료만으로 확인되지 않습니다.\n[근거] DOC-scope-C0001"

    result = review_answer("P의 원인을 설명해 주세요.", "", "[답변] P의 모든 원인은 A와 B입니다.", EVIDENCE,
        complete=lambda *_: next(outputs), regenerate=regenerate, require_evidence=True, budget_seconds=5)
    assert result.verdict == "rewritten" and result.retries == 1 and result.calls == 2
    assert "한 가지 조건은 A" in result.text and "A와 B" not in result.text
    assert result.text.endswith("[근거] DOC-condition-C0001") and result.cacheable
    assert "DOC-condition-C0001" in repairs[0] and "부분" in repairs[0]


@pytest.mark.parametrize("support", ["없음", "4"])
def test_missing_or_out_of_range_support_cannot_trigger_partial_evidence_repair(support):
    result = review_answer("P의 원인은?", "", "[답변] A와 B입니다.", EVIDENCE,
        complete=lambda *_: "이유: 지원하는 자료가 없음\n근거: " + support + "\n판정: INSUFFICIENT",
        regenerate=lambda *_: pytest.fail("unsupported evidence must never trigger repair"),
        require_evidence=True, budget_seconds=5)
    assert result.verdict in {"insufficient", "invalid_support"} and not result.cacheable
    assert "A와 B" not in result.text


def test_partial_repair_is_withheld_when_the_second_check_still_finds_unsupported_claims():
    result = review_answer("P의 원인은?", "", "[답변] A와 B입니다.", EVIDENCE,
        complete=lambda *_: "이유: 일부만 자료에서 확인됨\n근거: 2\n판정: INSUFFICIENT",
        regenerate=lambda *_: "[답변] 여전히 A와 B가 모든 원인입니다.",
        require_evidence=True, budget_seconds=5)
    assert result.retries == 1 and result.calls == 2 and result.verdict == "insufficient"
    assert not result.cacheable and "모든 원인" not in result.text


@pytest.mark.parametrize("detailed", [False, True])
def test_live_pipeline_publishes_only_the_scoped_repair_and_newly_checked_citation(monkeypatch, detailed):
    from trainer.demo_api.tests.test_query_quick_latency import _pipeline

    p, _, _, chunks = _pipeline(monkeypatch)
    chunks[:] = [{**item, "rerank_score": 8.0 - index / 10} for index, item in enumerate(EVIDENCE)]
    outputs = iter([
        "[답변] A와 B가 모든 원인입니다.\n[근거] DOC-scope-C0001",
        "이유: A만 자료에서 확인됨\n근거: 2\n판정: INSUFFICIENT",
        "[답변] 확인되는 조건은 A입니다. 다른 원인은 이 자료로 확인되지 않습니다.\n[근거] DOC-scope-C0001",
        "이유: 한정된 설명이 자료와 일치\n근거: 2\n판정: OK",
    ])
    p._generate_direct = lambda *_args, **_kwargs: next(outputs)
    result = p.run(instruction="P의 원인은?", answer_question="P의 원인은?", force_route="fast",
                   compact_analysis=detailed, concise_answer=not detailed, use_pipeline_history=False)
    assert "A와 B" not in result["response"] and "확인되는 조건은 A" in result["response"]
    assert result["citations"] == ["DOC-condition-C0001"]
    assert result["metadata"]["answer_review"]["verdict"] == "rewritten"
