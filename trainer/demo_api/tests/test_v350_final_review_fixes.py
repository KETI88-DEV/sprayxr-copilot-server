"""v3.5.0 최종 전체 검토(2026-11-15) 지적 C1·I2~I5의 재현 문장."""
from __future__ import annotations

import pytest

from trainer.demo_api.record_query import RecordFacts, answer_record_question, ground_generated
from trainer.demo_api.safety_guard import CLASSES
from trainer.demo_api.tests.viewer_record import AT_3S, AT_24S, AT_35S, RecordingPipeline, ask, context, payload


def _answer(prompt, playback=AT_3S, task="knowledge_search"):
    return answer_record_question(context(task, payload(), playback), prompt, task_type=task, recent_turns=[]).text


# ---- C1: 위험 질문은 출구 검사로 비어도 기록 답·"답할 수 없습니다"가 아니라 그 종류의 고정 문장이다
@pytest.mark.parametrize("prompt,document,hazard", [
    ("Zone 8 다시 칠해도 돼? 시너 냄새 맡으면서", "[답변] 30분 환기하세요.", "solvent"),
    ("어지러운데 계속 칠해도 돼?", "[답변] 15분 이상 쉬세요.", "health"),
])
def test_emptied_safety_answer_falls_back_to_the_hazard_sentence(prompt: str, document: str, hazard: str) -> None:
    pipeline = RecordingPipeline(document=document, chunks=[{"chunk_id": "DOC-x", "text": "작업 중 환기를 유지한다.", "rerank_score": 9.0}])
    data = ask(prompt, tab="knowledge", playback=AT_3S, pipeline=pipeline)
    assert data["answer"] == CLASSES[hazard].answer, data["answer"]


# ---- I2: "지금·여기"만으로는 일반 문서 문장을 Zone 판정으로 바꾸지 않는다
@pytest.mark.parametrize("prompt", ["지금 건조 시간은 얼마나 기다려야 해?", "여기 도료 희석은 어떻게 해?"])
def test_here_alone_does_not_turn_a_general_sentence_into_a_zone_verdict(prompt: str) -> None:
    sentence = "도료는 건조가 완료된 상태에서 덧칠합니다."
    pipeline = RecordingPipeline(document="[답변] " + sentence,
                                 chunks=[{"chunk_id": "DOC-x", "text": "건조가 완료된 상태에서 덧칠한다.", "rerank_score": 9.0}])
    data = ask(prompt, tab="knowledge", playback=AT_3S, pipeline=pipeline)
    assert sentence in data["answer"] and "Zone 8" not in data["answer"], data["answer"]


def test_a_named_zone_verdict_is_still_replaced() -> None:
    out = ground_generated("Zone 8은 완료된 상태입니다.", RecordFacts.from_context(context("knowledge_search", payload(), AT_3S)), zone="8", zone_asked=False)
    assert "보완" in out and "완료된 상태" not in out, out


# ---- I3: 맞는 전제의 "예."가 요구(올려야 해?·잘한 거지?)에 대한 답으로 읽히지 않는다
def test_speed_question_confirms_the_premise_without_yes() -> None:   # T44
    text = _answer("나 숙련자보다 느리던데, 속도를 숙련자만큼 올려야 해?")
    assert not text.startswith("예") and text.startswith("숙련자 기록보다 속도가 느린 것은 맞습니다.") and "이전 장비" in text, text


def test_correct_pose_number_is_not_answered_with_yes() -> None:
    text = _answer("내 자세 유사도가 54%면 잘한 거 맞지?", task="pose_recomm")
    assert not text.startswith("예") and text.count("54%") == 1 and "70%" in text, text


def test_correct_order_number_is_not_repeated_and_uses_the_right_particle() -> None:
    text = _answer("순서 준수율 45%면 괜찮은 거지?", task="trajectory_recomm")
    assert not text.startswith("예") and "준수율는" not in text and text.count("45") == 1, text


# ---- I4: 완료 개수 질문은 결정적 답, 모델이 말한 틀린 완료 개수는 바뀐다
@pytest.mark.parametrize("prompt", ["몇 개 Zone 끝났어요?", "얼마나 끝났어요?"])
def test_completed_zone_count_questions_are_answered_from_the_record(prompt: str) -> None:
    data = ask(prompt, tab="knowledge", playback=AT_35S)
    assert data["source"] == "analysis" and "9곳 중 0곳" in data["answer"], data["answer"]


def test_wrong_completed_zone_count_in_a_model_answer_is_replaced() -> None:
    facts = RecordFacts.from_context(context("knowledge_search", payload(), AT_35S))
    out = ground_generated("3개 Zone이 완료되었습니다.", facts, zone=None)
    assert "0곳" in out and "3개 Zone이" not in out, out


# ---- I5: Zone 완료 답은 술어가 있는 문장이다
def test_low_film_completion_answer_is_a_full_sentence() -> None:
    text = _answer("Zone 8 끝났어?", AT_35S)
    assert "Zone 8은 저도막(칠한 곳 평균 77µm, 미도장 0%)입니다." in text and "0%). " not in text, text


def test_contrast_completion_answer_uses_the_possessive() -> None:
    text = _answer("Zone 1 끝났어?", AT_24S)
    assert "Zone 1의 칠한 곳 평균 두께는" in text and "Zone 1은 칠한 곳" not in text, text


# ---- 재검토 추가: 원인 전제의 방향이 물리적으로 거꾸로면(가까움→얇음, 멂→두꺼움) 동조하지 않는다
@pytest.mark.parametrize("prompt,zone", [("거리가 가까워서 얇게 나온 거지?", "Zone 8"),   # Zone 8: 전문가보다 가깝고 저도막
                                         ("Zone 4 거리가 멀어서 두껍게 나온 거지?", "Zone 4")])   # Zone 4: 전문가보다 멂
def test_physically_reversed_cause_premise_is_not_affirmed(prompt: str, zone: str) -> None:
    text = _answer(prompt)
    assert text.startswith("아니요. ") and zone in text and not text.startswith("예"), text


def test_valid_cause_premise_matching_the_record_is_confirmed_without_yes() -> None:   # Zone 4: 멀어서 얇다, 실측도 멂
    text = _answer("Zone 4 거리가 멀어서 얇게 나온 거지?")
    assert text.startswith("Zone 4는 407 mm") and not text.startswith(("예", "아니요")), text


def test_cause_premise_against_the_measured_distance_still_says_no() -> None:   # T40: Zone 8은 가까운데 "멀어서"
    assert _answer("거리가 멀어서 얇게 나온 거지?").startswith("아니요. Zone 8")


# ---- 재검토 추가 2: 사실을 말하고 이유를 묻는 질문("가까웠는데 왜 얇아?")은 전제가 아니라 예·아니요로 열지 않는다
@pytest.mark.parametrize("prompt,fact", [("Zone 8 거리가 가까웠는데 왜 얇아?", "Zone 8은 251 mm"),
                                         ("Zone 4 거리가 멀었는데 왜 두꺼워?", "Zone 4는 407 mm")])
def test_stated_fact_with_a_why_question_gets_no_yes_or_no(prompt: str, fact: str) -> None:
    text = _answer(prompt)
    assert text.startswith(fact) and not text.startswith(("예", "아니요")), text
