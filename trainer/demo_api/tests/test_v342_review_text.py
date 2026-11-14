"""v3.4.2 독립 검토(2026-11-14): 접속어 정리와 음수 각도 근거(review-v342/probe.py의 문장)."""
from __future__ import annotations

import pytest

from trainer.demo_api import dialogue


# --- 4. 접속어 ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("거리 999 mm. 즉시 압력을 빼세요.", "즉시 압력을 빼세요."),
    ("거리 999 mm. 결국은 다시 칠해야 합니다.", "결국은 다시 칠해야 합니다."),
    ("거리 999 mm. 따라서, 다시 칠하세요.", "다시 칠하세요."),
    ("압력이 너무 높습니다. 따라서 압력을 낮추세요.", "압력이 너무 높습니다. 따라서 압력을 낮추세요."),
    ("Zone 3은 999 mm입니다. 그리고 Zone 4는 300 mm로 적정입니다.", "Zone 4는 300 mm로 적정입니다."),
    ("속도가 999 mm/s입니다. 그러나 도막은 목표를 넘었습니다.", "도막은 목표를 넘었습니다."),
    # v3.4.3(2차 검토 #6): 앞을 가리키는 문장이 이어지면 함께 뺀다. 다 빠지면 호출한 쪽의 대체 문장이 답한다.
    ("거리 999 mm. 이로 인해 생긴 문제는 없습니다. 그 결과 점수가 올랐습니다.", ""),
])
def test_connectives_only_at_sentence_start_and_no_chain_drop(text: str, expected: str) -> None:
    assert dialogue.drop_sentences(text, lambda part: "999" in part)[0] == expected


def test_orphan_only_answer_falls_back_instead_of_going_empty() -> None:
    assert dialogue.drop_ungrounded("거리 999 mm. 이로 인해 얇아졌습니다.", facts={}, prompt="") == "세션 기록으로는 확인할 수 없습니다."


# --- 6. 음수 각도 -------------------------------------------------------------------------------------

def test_negative_angles_are_grounded() -> None:
    facts = {"selected_zone": {"metrics": [{"key": "tool_angle", "actual": -5.2, "reference": 3.1, "unit": "deg"}]}}
    assert dialogue.ungrounded_measurements("도구 각도는 -5.2°입니다.", facts=facts, prompt="") == []
