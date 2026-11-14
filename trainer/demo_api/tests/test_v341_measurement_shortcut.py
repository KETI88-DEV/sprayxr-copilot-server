"""측정값 짧은 답이 우선순위·이유·긴 질문을 가로채지 않는다(v3.4.0 운영 질문 시험 2026-11-14 q22·q40)."""
from __future__ import annotations

import pytest

from trainer.demo_api import dialogue
from trainer.demo_api.tests.test_v341_record_causes import Q22, Q40, _documents
from trainer.demo_api.tests.viewer_record import AT_3S, ask, context, payload


@pytest.mark.parametrize("prompt", [Q40, Q22, "거리랑 속도 중에 뭐부터 고쳐야 해?", "지금 거리가 왜 문제야?", "이격거리를 어떻게 맞춰야 해?"])
def test_priority_why_and_long_questions_are_not_answered_with_measurements_only(prompt: str) -> None:
    signed = context("knowledge_search", payload(), AT_3S)
    assert dialogue.viewer_measurement_answer(signed, prompt) is None


def test_q40_reaches_the_pipeline() -> None:
    # v3.5.0 설계 표 v340 #40: "하나만 콕 집어"는 측정값 짧은 답이 아니라 기록의 우선순위(도막 게이트 최저 Zone 3)로 답한다.
    result = ask(Q40, tab="knowledge", pipeline=_documents("거리부터 일정하게 유지하세요."))
    assert result["source"] == "analysis" and "Zone 3" in result["answer"] and "속도 대표값" not in result["answer"]


@pytest.mark.parametrize("prompt", ["지금 속도 얼마야?", "지금 거리 얼마야?", "이 Zone 속도와 이격거리는 어땠어?"])
def test_plain_measurement_questions_still_use_the_signed_values(prompt: str) -> None:
    signed = context("knowledge_search", payload(), AT_3S)
    assert "대표값" in (dialogue.viewer_measurement_answer(signed, prompt) or "")
