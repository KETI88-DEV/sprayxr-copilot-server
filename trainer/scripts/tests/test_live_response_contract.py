from __future__ import annotations

from trainer.scripts.live_response_contract import normalize_fast_knowledge_response

CANONICAL = "\n".join(
    [
        "[분석] SENTINEL_DEFINITION_BODY",
        "두 번째 줄 SENTINEL_SECOND_LINE",
        "[판정] SENTINEL_VERDICT",
        "[원인] SENTINEL_CAUSE",
        "[조치]",
        "1. SENTINEL_ACTION",
        "[근거] DOC-doc_0086-C0001",
    ]
)


def test_canonical_diagnostic_scaffold_becomes_answer_and_evidence_only() -> None:
    """Given the trained five-section scaffold, When normalized, Then only answer and evidence remain."""
    normalized = normalize_fast_knowledge_response(CANONICAL)

    assert normalized == "\n".join(
        [
            "[답변] SENTINEL_DEFINITION_BODY",
            "두 번째 줄 SENTINEL_SECOND_LINE",
            "[근거] DOC-doc_0086-C0001",
        ]
    )


def test_already_compliant_answer_is_unchanged() -> None:
    """Given an answer/evidence response, When normalized, Then it is returned byte-for-byte."""
    compliant = "[답변] SENTINEL_BODY\n[근거] DOC-doc_0001-C0001"

    assert normalize_fast_knowledge_response(compliant) == compliant


def test_inline_label_mention_is_not_treated_as_a_heading() -> None:
    """Given a label used mid-sentence, When normalized, Then the text is untouched."""
    prose = "[답변] 문서에서 [판정]이라는 용어는 합부 결정을 뜻한다.\n[근거] DOC-doc_0002-C0002"

    assert normalize_fast_knowledge_response(prose) == prose


def test_noncanonical_ordering_is_left_untouched() -> None:
    """Given evidence before the verdict, When normalized, Then nothing is removed."""
    reordered = "[분석] SENTINEL_BODY\n[근거] DOC-doc_0003-C0003\n[판정] SENTINEL_VERDICT"

    assert normalize_fast_knowledge_response(reordered) == reordered


def test_duplicated_sections_are_left_untouched() -> None:
    """Given a repeated section, When normalized, Then the ambiguous response is preserved."""
    duplicated = "\n".join(
        [
            "[분석] SENTINEL_ONE",
            "[판정] SENTINEL_VERDICT",
            "[분석] SENTINEL_TWO",
            "[근거] DOC-doc_0004-C0004",
        ]
    )

    assert normalize_fast_knowledge_response(duplicated) == duplicated


def test_prose_before_the_first_heading_is_left_untouched() -> None:
    """Given a preamble outside any section, When normalized, Then the response is preserved."""
    preamble = "설명 문장입니다.\n" + CANONICAL

    assert normalize_fast_knowledge_response(preamble) == preamble
