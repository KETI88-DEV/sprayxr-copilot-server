import pytest
from trainer.demo_api.copilot_intent import is_domain_question
from trainer.demo_api.dialogue import drop_ungrounded, ungrounded_measurements

@pytest.mark.parametrize("prompt", [
    "가사 시간이 지난 도료 아까운데 그냥 써도 돼요?", "VR로 연습하다가 어지러우면 몇 분 쉬어야 해요?",
    "금속 도장 기능사 실기는 몇 점 이상이면 합격이에요?", "에어리스 팁 517은 무슨 뜻이에요?", "가사시간이 뭐지?"])
def test_painting_and_training_words_are_domain(prompt: str) -> None:
    assert is_domain_question(prompt)

@pytest.mark.parametrize("prompt", ["BTS 멤버는 몇 명이야?", "안녕", "오늘 점심 뭐 먹을까요?", "1+3은?",
                                    "한국전쟁은 언제 시작됐어?", "이 조건이면 괜찮아?", "파이썬 리스트 설명해줘"])
def test_off_topic_words_are_not_domain(prompt: str) -> None:
    assert not is_domain_question(prompt)

def test_extended_units_catch_invented_minutes_and_points() -> None:
    prompt = "VR로 연습하다가 어지러우면 몇 분 쉬어야 해요?"
    assert ungrounded_measurements("5~10분 정도 쉬는 것이 일반적입니다.", facts={}, prompt=prompt) == []
    assert ungrounded_measurements("5~10분 정도 쉬는 것이 일반적입니다.", facts={}, prompt=prompt, extended=True) == ["5~10분"]
    assert ungrounded_measurements("60점 이상이어야 합니다.", facts={}, prompt="", extended=True) == ["60점"]
    assert ungrounded_measurements("1.5m 떨어지세요.", facts={}, prompt="", extended=True) == ["1.5m"]

def test_extended_units_accept_values_in_facts_and_chunk_text() -> None:
    facts = {"guideline": "분사 2회 중 2회", "evidence": ["피도체와의 거리는 30 cm 정도로 균일하게"]}
    assert ungrounded_measurements("분사 2회, 거리 30cm, 250.8 mm", facts={**facts, "d": {"actual": 250.8, "unit": "mm"}},
                                   prompt="", extended=True) == []
    assert drop_ungrounded("쉬는 시간은 10분입니다. 멈추고 알리세요.", facts={}, prompt="", extended=True) == "멈추고 알리세요."

def test_celsius_is_not_read_as_angle() -> None:
    assert ungrounded_measurements("25℃ 이하", facts={"a": {"actual": 25, "unit": "deg"}}, prompt="", extended=True) == ["25℃"]


@pytest.mark.parametrize("text", ["3점검", "100초과", "2배열", "5분류", "1분기", "3회전", "3회복"])
def test_extended_units_ignore_korean_compounds(text: str) -> None:
    assert ungrounded_measurements(f"{text} 합니다.", facts={}, prompt="", extended=True) == []


@pytest.mark.parametrize("text, found", [("5분이 지나면", "5분"), ("3점을 받아야", "3점"), ("2회는 필요", "2회"), ("30초 동안", "30초")])
def test_extended_units_keep_josa_cases(text: str, found: str) -> None:
    assert ungrounded_measurements(text, facts={}, prompt="", extended=True) == [found]


def test_extended_leaves_existing_units_unchanged() -> None:
    answer, facts = "12mm, 80%, -5.2°, 99mm, 90%", {"a": {"actual": 12, "unit": "mm"}, "r": {"similarity": 0.8}, "g": {"actual": -5.2, "unit": "deg"}}
    assert ungrounded_measurements(answer, facts=facts, prompt="", extended=True) == ungrounded_measurements(answer, facts=facts, prompt="") == ["99mm", "90%"]
