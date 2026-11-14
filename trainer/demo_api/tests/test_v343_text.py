"""v3.4.3 2차 검토 #6(2026-11-14, review-v342/probe2.py): 앞을 가리키는 문장 연쇄와 각도 부호."""
from __future__ import annotations

import pytest

from trainer.demo_api import dialogue


@pytest.mark.parametrize("text,expected", [
    ("압력 999 bar입니다. 즉시 압력을 빼세요.", "즉시 압력을 빼세요."),
    ("거리 999 mm입니다. 결국은 겹침이 문제입니다.", "결국은 겹침이 문제입니다."),
    ("거리 999 mm. 이로 인해 생긴 문제는 없습니다. 그 결과 점수가 올랐습니다.", ""),
    ("거리 999 mm. 그리고, 속도는 적정입니다.", "속도는 적정입니다."),
    ("거리 999 mm. 이로인해 도막이 얇습니다.", ""),
    ("속도 999 mm/s.\n따라서 겹침을 늘리세요.", "겹침을 늘리세요."),
])
def test_probe2_connectives(text: str, expected: str) -> None:
    assert dialogue.drop_sentences(text, lambda part: "999" in part)[0] == expected


def test_chain_of_point_back_sentences_falls_back_instead_of_going_empty() -> None:
    assert dialogue.drop_ungrounded("거리 999 mm. 이로 인해 생긴 문제는 없습니다. 그 결과 점수가 올랐습니다.", facts={}, prompt="") == (
        "세션 기록으로는 확인할 수 없습니다.")


def test_angle_sign_is_checked() -> None:
    facts = {"m": {"actual": -5.2, "unit": "deg"}}
    assert dialogue.drop_ungrounded("Zone 8 도구 각도는 -5.2°입니다.", facts=facts, prompt="") == "Zone 8 도구 각도는 -5.2°입니다."
    assert dialogue.drop_ungrounded("Zone 8 도구 각도는 5.2°입니다.", facts=facts, prompt="") == "세션 기록으로는 확인할 수 없습니다."
