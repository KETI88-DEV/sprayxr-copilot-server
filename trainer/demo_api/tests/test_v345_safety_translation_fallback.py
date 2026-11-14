"""v3.4.5 — 고정 고압 안전 답의 답 번역이 실패하면 503 대신 미리 옮겨 둔 문장을 낸다.

운영 v3.4.4(2026-11-14): 우즈베크어 T15는 질문 번역은 됐지만 고정 답을 우즈베크어로 옮기다
"Incomplete translation"으로 세 번 다 실패해 503이 나갔다. 번역 경로는 바꾸지 않고, 실패했을 때만 쓴다.
"""
from __future__ import annotations

import pytest

from trainer.demo_api.query_language import LANGUAGE_NAMES
from trainer.demo_api.safety_guard import HIGH_PRESSURE_ANSWER, fixed_answer_translation


@pytest.mark.parametrize("language", sorted(set(LANGUAGE_NAMES) - {"ko"}))
def test_every_response_language_has_a_fallback_for_the_high_pressure_answer(language: str) -> None:
    text = fixed_answer_translation(HIGH_PRESSURE_ANSWER, language)
    assert text and "건" not in text and "손가락" not in text   # 한국어가 남지 않는다


def test_only_the_fixed_answer_has_a_fallback() -> None:
    assert fixed_answer_translation("Zone 3은 목표 두께 이상 0%입니다.", "uz") is None
    assert fixed_answer_translation(HIGH_PRESSURE_ANSWER, "ko") is None
