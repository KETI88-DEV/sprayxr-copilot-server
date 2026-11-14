from trainer.demo_api.record_query import ASK_ZONE, answer_record_question, previous_subject
from trainer.demo_api.tests.viewer_record import AT_3S, AT_24S, AT_35S, context, payload


def answer(prompt, playback=AT_3S, task="knowledge_search", turns=()):
    return answer_record_question(context(task, payload(), playback), prompt, task_type=task, recent_turns=list(turns))


def test_wrong_metric_name_is_corrected_first() -> None:   # T27
    result = answer("화면에 87% 떠 있던데, 내 자세 유사도가 87%면 숙련자(70%)보다 잘한 거 맞지?", task="pose_recomm")
    assert result.text.startswith("아니요") and "87%는 작업 유사도" in result.text and "54%" in result.text and "참고용" in result.text


def test_cause_premise_against_the_measured_distance() -> None:   # T40
    turns = [{"prompt": "이 Zone 속도랑 이격거리 숙련자랑 비교해줘.", "answer": "…", "answer_source": "analysis",
              "subject": {"domain": "distance", "zone_ids": ["8"], "operator": "compare"}}]
    result = answer("거리가 멀어서 얇게 나온 거지?", turns=turns)
    assert result.text.startswith("아니요") and "Zone 8" in result.text and "54%" in result.text and "가까" in result.text


def test_conclusion_premises_are_answered_with_the_gate() -> None:   # T43, T34
    assert answer("Zone 1은 정상이라고 나왔으니까 다 끝난 거지? 거긴 다시 안 가도 되지?", AT_24S).text.startswith("아니요")
    t34 = answer("다 칠했으니까 이제 끝난 거죠?", AT_35S, task="trajectory_recomm").text
    assert t34.startswith("아니요") and "95%" in t34


def test_true_comparison_premise_starts_with_yes() -> None:   # T44
    text = answer("나 숙련자보다 느리던데, 속도를 숙련자만큼 올려야 해?").text
    assert text.startswith("예") and "이전 장비" in text


def test_missing_zone_is_named_not_explained() -> None:   # T33
    text = answer("Zone 10이랑 11은 왜 안 칠했다고 나와요?", task="trajectory_recomm").text
    assert "Zone 10" in text and "1~9" in text and "안 칠" not in text


def test_mixed_question_keeps_only_the_record_part_and_hands_over_the_knowledge_part() -> None:   # T38
    turns = [{"prompt": "왜 그렇게 얇게 나온 거야?", "answer": "…", "answer_source": "analysis",
              "subject": {"domain": "coating", "zone_ids": ["8"], "operator": "cause"}}]
    result = answer("그럼 거기부터 다시 칠하면 돼? 어떻게 칠해?", AT_35S, turns=turns)
    assert result.kind == "mixed" and result.knowledge_question == "어떻게 칠해?"
    assert "Zone 8" in result.text and "1.9%" in result.text


def test_caveat_line_and_subject() -> None:   # T36
    result = answer("칠 다 했는데 어디가 제일 얇게 나왔어?", AT_35S)
    assert result.text.splitlines()[-1].count("세션이 끝났을 때") == 1
    assert result.subject == {"domain": "coating", "zone_ids": ["8"], "operator": "argmin"}


def test_previous_subject_ignores_model_answers() -> None:
    assert previous_subject([{"answer_source": "documents", "subject": None}]) is None
    assert previous_subject([{"answer_source": "analysis", "subject": {"domain": "coating", "zone_ids": ["8"], "operator": "argmin"}},
                             {"answer_source": "conversation", "subject": None}]) is None
    subject = {"domain": "coating", "zone_ids": ["8"], "operator": "argmin"}
    assert previous_subject([{"answer_source": "analysis", "subject": subject}]) == subject


def test_single_zone_question_without_a_zone_asks_for_the_zone() -> None:   # Review Focus 1
    moving = {"map_num": None, "zone_id": None, "source_index": 5, "playhead_seconds": 0.5}
    assert answer("왜 이렇게 얇게 나왔어?", moving).text.startswith(ASK_ZONE)


def test_forged_or_missing_signature_turns_record_answers_off() -> None:   # Review Focus 2
    forged = context("knowledge_search", payload(), AT_3S).replace('"dialogue_signature": "', '"dialogue_signature": "0', 1)
    assert answer_record_question(forged, "어디가 제일 얇게 나왔어?", task_type="knowledge_search", recent_turns=[]) is None
    assert answer_record_question(None, "어디가 제일 얇게 나왔어?", task_type="error_analysis", recent_turns=[]) is None


def test_unanswerable_question_shaped_part_is_named_but_narration_is_silent() -> None:   # P7
    assert "그 부분은 이 기록에 없습니다." in answer("칠 다 했는데 어디가 제일 얇게 나왔어? 그리고 내 이름이 뭐야?").text
    assert "그 부분은" not in answer("칠 다 했는데 어디가 제일 얇게 나왔어?").text
