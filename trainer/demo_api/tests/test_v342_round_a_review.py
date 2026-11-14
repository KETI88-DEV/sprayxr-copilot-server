"""v3.4.2 A차 3차 검토(2026-11-14, probe3) — 손-팁 고정 답의 오판·누락·후속 질문."""
from __future__ import annotations

import pytest

from trainer.demo_api.safety_guard import blocks_tip_by_hand, hand_on_tip


@pytest.mark.parametrize("text", [
    "손 위치 확인하는 팁 있어?",                       # '팁'은 요령, '확인'은 접촉이 아니다
    "오른손으로 건 잡고 노즐 각도 확인하려면?",
    "What hand position should I hold for the tip angle?",
])
def test_tip_as_advice_or_check_is_not_hand_on_tip(text: str) -> None:
    assert not blocks_tip_by_hand(text)


@pytest.mark.parametrize("text", [
    "건 끝을 손가락으로 막아서 확인해도 되나요?",        # T15 원문은 그대로 잡는다
    "건의 끝을 손가락으로 막아도 돼?",
    "총구를 손으로 막아도 돼?",
    "분무구에 손가락 대봐도 돼?",
    "건 머리를 손바닥으로 막아서 확인해도 돼?",
    "노즐에 손가락 붙여서 막혔는지 봐도 돼?",
    "Can I block the spray gun with my finger?",
])
def test_more_tip_words_are_hand_on_tip(text: str) -> None:
    assert blocks_tip_by_hand(text)


PREVIOUS = "건 끝을 손가락으로 막아서 확인해도 되나요?"


@pytest.mark.parametrize("text", ["도막 두께는 어떻게 되나요?", "다음 구역은 어떻게 돼?", "자세 점수는 괜찮아?"])
def test_new_question_after_hand_on_tip_is_not_a_follow_up(text: str) -> None:
    assert not hand_on_tip((text,), previous=PREVIOUS)


@pytest.mark.parametrize("text", ["그럼 해도 돼?", "그래도 돼?", "왜 안 되는데?", "잠깐이면 괜찮지?"])
def test_short_follow_up_after_hand_on_tip_keeps_the_fixed_answer(text: str) -> None:
    assert hand_on_tip((text,), previous=PREVIOUS)
