from __future__ import annotations

import re

import pytest

from trainer.scripts.live_response_contract import (
    FAILED_ANSWER,
    document_ids,
    is_contract_shaped,
    parse_knowledge_answer,
    retrieved_chunk_ids,
)

DOC_A = "DOC-doc_0086-C0001"
DOC_B = "DOC-doc_0042-C0007"
RETRIEVED = [DOC_A, DOC_B]

_DOC_ID_RE = re.compile(r"DOC-[a-zA-Z0-9_-]+-C\d+")

CANONICAL = "\n".join(
    [
        "[분석] SENTINEL_DEFINITION_BODY",
        "두 번째 줄 SENTINEL_SECOND_LINE",
        "[판정] SENTINEL_VERDICT",
        "[원인] SENTINEL_CAUSE",
        "[조치]",
        "1. SENTINEL_ACTION",
        f"[근거] {DOC_A}",
    ]
)

# The nine label sequences measured over 240 stored v6.1 responses
# (/output/eval_results/qwen36_27b_v6_1_nf4_baseline_965549e/seed_{42..49}).
MEASURED_LABEL_SEQUENCES = [
    ["분석", "판정", "원인", "조치", "근거"],
    ["분석", "판정", "조치", "근거"],
    ["분석", "판정", "원인", "조치", "근거", "근거"],
    ["분석", "판정", "조치", "근거", "근거"],
    ["분석", "근거"],
    ["분석", "조치", "근거"],
    ["분석", "조치", "근거", "근거"],
    ["분석", "판정", "원인", "근거"],
    ["판정", "원인", "조치", "근거"],
]


def _render(labels: list[str]) -> str:
    lines = []
    for index, label in enumerate(labels):
        payload = DOC_A if label == "근거" else f"SENTINEL_{label}_{index}"
        lines.append(f"[{label}] {payload}")
    return "\n".join(lines)


def test_canonical_scaffold_keeps_answer_and_renders_validated_evidence() -> None:
    """Given the trained five-section scaffold, When parsed, Then only answer and evidence remain."""
    parsed = parse_knowledge_answer(CANONICAL, RETRIEVED)

    assert parsed.answer == "\n".join(
        [
            "[답변] SENTINEL_DEFINITION_BODY",
            "두 번째 줄 SENTINEL_SECOND_LINE",
            f"[근거] {DOC_A}",
        ]
    )
    assert parsed.citations == (DOC_A,)
    assert parsed.dropped == ()
    assert parsed.dropped_sections == ("판정", "원인", "조치")
    assert parsed.repair_path == "sections"


def test_already_compliant_answer_round_trips() -> None:
    """Given an answer/evidence response, When parsed, Then it comes back unchanged."""
    compliant = f"[답변] SENTINEL_BODY\n[근거] {DOC_A}"

    parsed = parse_knowledge_answer(compliant, RETRIEVED)

    assert parsed.answer == compliant
    assert parsed.citations == (DOC_A,)


def test_inline_label_mention_is_not_treated_as_a_heading() -> None:
    """Given a label used mid-sentence, When parsed, Then the sentence survives intact."""
    prose = f"[답변] 문서에서 [판정]이라는 용어는 합부 결정을 뜻한다.\n[근거] {DOC_A}"

    assert parse_knowledge_answer(prose, RETRIEVED).answer == prose


def test_noncanonical_ordering_is_repaired_not_passed_through() -> None:
    """Given evidence before the verdict, When parsed, Then the verdict is still removed."""
    reordered = f"[분석] SENTINEL_BODY\n[근거] {DOC_A}\n[판정] SENTINEL_VERDICT"

    parsed = parse_knowledge_answer(reordered, RETRIEVED)

    assert parsed.answer == f"[답변] SENTINEL_BODY\n[근거] {DOC_A}"
    assert "SENTINEL_VERDICT" not in parsed.answer
    assert parsed.dropped_sections == ("판정",)


def test_duplicated_answer_sections_are_merged() -> None:
    """Given a repeated answer section, When parsed, Then both bodies are kept once."""
    duplicated = "\n".join(
        [
            "[분석] SENTINEL_ONE",
            "[판정] SENTINEL_VERDICT",
            "[분석] SENTINEL_TWO",
            f"[근거] {DOC_A}",
        ]
    )

    parsed = parse_knowledge_answer(duplicated, RETRIEVED)

    assert parsed.answer == f"[답변] SENTINEL_ONE\nSENTINEL_TWO\n[근거] {DOC_A}"


def test_duplicated_evidence_sections_collapse_to_one_line() -> None:
    """Given two evidence sections, When parsed, Then one evidence line carries both ids."""
    duplicated = "\n".join(["[분석] SENTINEL_BODY", f"[근거] {DOC_A}", f"[근거] {DOC_B}"])

    parsed = parse_knowledge_answer(duplicated, RETRIEVED)

    assert parsed.answer == f"[답변] SENTINEL_BODY\n[근거] {DOC_A}, {DOC_B}"
    assert parsed.citations == (DOC_A, DOC_B)


def test_preamble_before_the_first_heading_joins_the_answer_body() -> None:
    """Given prose ahead of the headings, When parsed, Then it leads the answer without nesting."""
    parsed = parse_knowledge_answer("설명 문장입니다.\n" + CANONICAL, RETRIEVED)

    assert parsed.answer == "\n".join(
        [
            "[답변] 설명 문장입니다.",
            "SENTINEL_DEFINITION_BODY",
            "두 번째 줄 SENTINEL_SECOND_LINE",
            f"[근거] {DOC_A}",
        ]
    )
    assert "[분석]" not in parsed.answer


def test_response_without_any_heading_gets_the_answer_heading() -> None:
    """Given a bare paragraph, When parsed, Then it is wrapped, not passed through."""
    parsed = parse_knowledge_answer("저도막은 규정 두께에 미달한 상태를 말한다.", RETRIEVED)

    assert parsed.answer == "[답변] 저도막은 규정 두께에 미달한 상태를 말한다."
    assert parsed.repair_path == "fallback"
    assert parsed.citations == ()


def test_citation_outside_the_retrieved_set_leaves_the_prose_and_the_array() -> None:
    """Given an id that was never retrieved, When parsed, Then it is dropped from both."""
    outside = "DOC-doc_9999-C0001"
    raw = f"[분석] SENTINEL_BODY({outside} 참조)\n[근거] {outside}"

    parsed = parse_knowledge_answer(raw, RETRIEVED)

    assert outside not in parsed.answer
    assert parsed.citations == ()
    assert parsed.dropped == (outside,)
    # The sentence is not rewritten; the id leaves and the separators are tidied.
    assert parsed.answer == "[답변] SENTINEL_BODY(참조)"


def test_mixed_citations_keep_only_the_retrieved_one() -> None:
    """Given one retrieved and one invented id, When parsed, Then only the retrieved id remains."""
    outside = "DOC-doc_9999-C0002"
    raw = f"[분석] SENTINEL_BODY\n[근거] {outside}, {DOC_B}"

    parsed = parse_knowledge_answer(raw, RETRIEVED)

    assert parsed.answer == f"[답변] SENTINEL_BODY\n[근거] {DOC_B}"
    assert parsed.citations == (DOC_B,)
    assert parsed.dropped == (outside,)


def test_empty_retrieved_set_strips_every_document_id() -> None:
    """Given use_rag='N' (nothing retrieved), When parsed, Then no document id survives."""
    raw = f"[분석] 규정은 {DOC_A} 에 있다.\n[근거] {DOC_A}"

    parsed = parse_knowledge_answer(raw, [])

    assert _DOC_ID_RE.search(parsed.answer) is None
    assert parsed.citations == ()
    assert parsed.dropped == (DOC_A,)


def test_verdict_only_response_is_refused_rather_than_published() -> None:
    """Given no answer section, When parsed, Then the verdict is not published as an answer."""
    raw = f"[판정] SENTINEL_VERDICT\n[원인] SENTINEL_CAUSE\n[근거] {DOC_A}"

    parsed = parse_knowledge_answer(raw, RETRIEVED)

    assert parsed.answer == f"[답변] {FAILED_ANSWER}"
    assert parsed.citations == ()
    assert parsed.repair_path == "no_answer_section"
    assert "SENTINEL_VERDICT" not in parsed.answer


def test_fullwidth_bracket_headings_are_sections_too() -> None:
    """use_rag='N' 실측(2026-10-12): 문서가 없자 모델이 정규식이 막는 ``[`` 대신 ``【】`` 로 진단 틀을 냈다."""
    raw = "\n".join(
        [
            "저도막은 건도막두께가 50μm 미만인 경우를 말합니다.",
            "",
            "【분석】 SENTINEL_ANALYSIS",
            "",
            "【판정】 이상 탐지",
            "",
            "【원인】 측정값 편차 또는 작업 조건 이탈이 확인되었습니다.",
            "",
            "【조치】",
            "",
            "1. 현재 상태를 유지하고 정기 점검을 계속하십시오.",
            "",
            "【위험】 화재·안전 위험이 높습니다.",
            "",
            "【근거】 참조 문서 ID:",
        ]
    )

    parsed = parse_knowledge_answer(raw, [])

    assert parsed.body == "저도막은 건도막두께가 50μm 미만인 경우를 말합니다.\nSENTINEL_ANALYSIS"
    assert parsed.dropped_sections == ("판정", "원인", "조치", "위험")
    assert is_contract_shaped(parsed.answer)


@pytest.mark.parametrize("raw", ["", "   ", "\n\n"])
def test_blank_response_yields_the_failure_answer(raw: str) -> None:
    """Given an empty generation, When parsed, Then a stated failure replaces silence."""
    parsed = parse_knowledge_answer(raw, RETRIEVED)

    assert parsed.answer == f"[답변] {FAILED_ANSWER}"
    assert parsed.repair_path == "empty"


def test_body_of_only_a_dropped_id_yields_the_failure_answer() -> None:
    """Given a body that is nothing but an unretrieved id, When parsed, Then the failure shows."""
    parsed = parse_knowledge_answer("[분석] DOC-doc_9999-C0003", RETRIEVED)

    assert parsed.answer == f"[답변] {FAILED_ANSWER}"
    assert parsed.repair_path == "body_empty"


@pytest.mark.parametrize("labels", MEASURED_LABEL_SEQUENCES, ids=lambda labels: "-".join(labels))
def test_every_measured_label_sequence_yields_the_contract_shape(labels: list[str]) -> None:
    """Given each shape seen in 240 stored responses, When parsed, Then the contract shape holds."""
    parsed = parse_knowledge_answer(_render(labels), RETRIEVED)

    lines = parsed.answer.split("\n")
    assert lines[0].startswith("[답변] ")
    evidence_lines = [line for line in lines if line.startswith("[근거]")]
    assert len(evidence_lines) <= 1
    if evidence_lines:
        assert evidence_lines[-1] is lines[-1]
    for label in ("[판정]", "[원인]", "[조치]", "[분석]"):
        assert label not in parsed.answer


@pytest.mark.parametrize("labels", MEASURED_LABEL_SEQUENCES, ids=lambda labels: "-".join(labels))
def test_every_document_id_in_the_answer_is_in_citations(labels: list[str]) -> None:
    """Given each measured shape, When parsed, Then prose ids never exceed the citation array."""
    parsed = parse_knowledge_answer(_render(labels), RETRIEVED)

    assert set(_DOC_ID_RE.findall(parsed.answer)) <= set(parsed.citations)


def test_retrieved_chunk_ids_preserves_order_dedupes_and_ignores_junk() -> None:
    """Given a rag_chunks list, When reduced to ids, Then order holds and non-dicts are skipped."""
    chunks = [
        {"chunk_id": DOC_B},
        {"chunk_id": DOC_A},
        {"chunk_id": DOC_B},
        {"chunk_id": ""},
        {"text": "no id"},
        "not a dict",
    ]

    assert retrieved_chunk_ids(chunks) == [DOC_B, DOC_A]


# --- shape measurement (the gate instrument) ------------------------------


@pytest.mark.parametrize(
    ("answer", "shaped"),
    [
        ("[답변] 본문입니다.", True),
        (f"[답변] 본문\n[근거] {DOC_A}", True),
        (f"[답변] 본문\n[근거] {DOC_A}, {DOC_B}", True),
        ("[답변] 첫 줄\n두 번째 줄", True),
        (f"[답변] 본문\n[근거] {DOC_A}\n[근거] {DOC_B}", False),
        (f"[답변] 본문\n[근거] {DOC_A}\n추가 문장", False),
        ("[답변] 본문\n[판정] 합격", False),
        ("[분석] 본문", False),
        ("[답변] ", False),
        ("본문만 있음", False),
        ("[답변] 본문\n[근거] 참조 문서 없음", False),
        ("", False),
    ],
)
def test_contract_shape_checker(answer: str, shaped: bool) -> None:
    """Given a published answer, When measured, Then only the contract shape passes."""
    assert is_contract_shaped(answer) is shaped


@pytest.mark.parametrize(
    "raw",
    [
        CANONICAL,
        "",
        "본문만 있는 문단",
        f"[판정] SENTINEL\n[근거] {DOC_A}",
        "설명 문장입니다.\n" + CANONICAL,
        f"[분석] SENTINEL\n[근거] {DOC_A}\n[근거] {DOC_B}",
        f"[분석] SENTINEL\n[근거] DOC-doc_9999-C0001",
        *[_render(labels) for labels in MEASURED_LABEL_SEQUENCES],
    ],
)
def test_parser_output_always_passes_the_shape_checker(raw: str) -> None:
    """Given any input, When parsed, Then the result satisfies the gate instrument."""
    assert is_contract_shaped(parse_knowledge_answer(raw, RETRIEVED).answer)


def test_document_ids_are_ordered_and_deduplicated() -> None:
    """Given repeated ids in prose, When extracted, Then each appears once in order."""
    text = f"{DOC_B} 그리고 {DOC_A} 그리고 다시 {DOC_B}"

    assert document_ids(text) == [DOC_B, DOC_A]
