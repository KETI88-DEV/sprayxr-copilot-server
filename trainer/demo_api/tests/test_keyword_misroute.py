"""낱말 하나에 걸려 질문 의도와 다른 답 틀이 답하던 사례(2026-11-17 조사, 묶음 A~G)와 그 회귀 짝.

사례는 운영 질문 기록·v3.8.7 재시험에서 나온 질문이다. 같은 서명 문맥 두 벌(팔 자세 채점 있음·없음)과
빠른 안내·상세 분석 두 수준으로 라우터에 보낸다. 모델이 필요한 경로는 가짜 파이프라인으로 경로(answer_source)만 본다.
고정 자료는 2026-11-14 학습 기록 사본(viewer_record)이다.
"""
from __future__ import annotations

from typing import Any

import pytest

from trainer.demo_api.tests.test_scenario_reference_query import Pipeline as ReferencePipeline
from trainer.demo_api.tests.test_scenario_reference_query import client as reference_client
from trainer.demo_api.tests.test_scenario_reference_query import profile
from trainer.demo_api.tests.viewer_record import AT_35S, RecordingPipeline, ask, payload

ARMS = ("arm", "no_arm")
MODES = ("quick", "detailed")
DOCUMENT = "문서 답입니다."
CONVERSATION = "대화 답입니다."
NO_ARM_QUICK = "팔 자세 채점 결과가 없어"
NO_ARM_DETAILED = "팔 자세 채점 결과를 받지 못해"

# 사례 1(v3.8.7 재시험 #42 원문).
LONG_BEGINNER = (
    "안녕하세요 저는 이번에 처음 도장 훈련을 받는 사람인데요, 오늘 시뮬레이터에서 스프레이로 칠해 봤는데 화면에 저도막이라고 빨갛게 많이 뜨더라고요. "
    "선배는 건이 너무 가까우면 오히려 흘러내린다고 하고 다른 선배는 멀리 떨어지면 건조 분무가 생긴다고 해서 헷갈립니다. "
    "손목을 꺾지 말라는 얘기도 들었고 패스를 50% 겹치라는 얘기도 들었는데 그게 무슨 뜻인지 잘 모르겠어요. "
    "그리고 속도는 빨리 해야 하는지 천천히 해야 하는지도 모르겠고요. "
    "결국 제가 지금 제일 먼저 신경 써야 하는 게 거리인지 속도인지 겹침인지, 그리고 그 이유가 뭔지 하나만 콕 집어서 알려주시면 좋겠습니다. "
    "너무 길게 말고 핵심만요.")


def _data(arm: str) -> dict[str, Any]:
    data = payload()
    if arm == "no_arm":  # 운영 v3.8.7 재시험 문맥: 팔 자세 채점이 오지 않았다
        data["arm_pose_evaluation"] = None
        data["query_context_signatures"].pop("arm_pose", None)
    return data


def _ask(prompt: str, tab: str, arm: str = "arm", mode: str = "quick", **kwargs: Any) -> dict[str, Any]:
    found = ask(prompt, tab=tab, mode=mode, data=_data(arm), **kwargs)
    if mode == "quick":
        assert "\n" not in found["answer"], found["answer"]
    return found


both = pytest.mark.parametrize("arm, mode", [(arm, mode) for arm in ARMS for mode in MODES])


# ---------------------------------------------------------------- A. 일반 요인 중 고르기·"어떻게 나온"은 지식이다
@both
def test_a_choice_among_general_factors_is_answered_from_the_documents(arm: str, mode: str) -> None:
    """사례 1: "거리인지 속도인지 겹침인지 … 이유가 뭔지"에 "Zone 3부터 보완"이 답했다."""
    found = _ask(LONG_BEGINNER, "knowledge", arm, mode)
    assert found["source"] == "documents" and DOCUMENT in found["answer"], found
    assert "Zone" not in found["answer"], found["answer"]


@both
@pytest.mark.parametrize("prompt", ["가장 먼저 고쳐야 할 점은?", "뭐부터 고쳐야 돼? 딱 하나만 콕 집어줘."])
def test_a_pair_record_priority_questions_keep_the_record_priority(prompt: str, arm: str, mode: str) -> None:
    found = _ask(prompt, "knowledge", arm, mode)
    assert found["source"] == "analysis" and "Zone 3부터 보완" in found["answer"].split("\n")[0], found["answer"]


@both
def test_a_how_a_rate_is_computed_goes_to_the_documents_with_the_record_numbers(arm: str, mode: str) -> None:
    """사례 3: "준수율 45%는 어떻게 나온 거고"에 계산 설명 없이 수치만 답했다."""
    found = _ask("순서 준수율 45%는 어떻게 나온 거고, 숙련자들끼리는 순서가 얼마나 일치해?", "order", arm, mode)
    assert found["source"] == "documents" and found["calls"] == ["run"], found
    if mode == "quick":  # 먼저 물은 쪽(계산 방식)이 앞이다
        assert found["answer"].startswith(DOCUMENT), found["answer"]
    else:
        assert "45.1%" in found["answer"] and DOCUMENT in found["answer"], found["answer"]


@both
def test_a_pair_the_bare_rate_question_stays_a_record_lookup(arm: str, mode: str) -> None:
    found = _ask("순서 준수율은?", "order", arm, mode)
    assert found["source"] == "analysis" and "45" in found["answer"], found["answer"]


# ---------------------------------------------------------------- B. 신체 부위가 먼저, "뭐부터 고치"는 활용형과 상관없이 한 답
@both
def test_b_hand_question_is_not_answered_with_the_lowest_work_zone(arm: str, mode: str) -> None:
    """사례 4: 팔 채점이 없을 때 "오른손은 어케 고쳐야돼"에 "먼저 볼 곳은 Zone 6"이 답했다."""
    found = _ask("오른손은 어케 고쳐야돼", "pose", arm, mode)
    assert found["source"] == "analysis" and "먼저 볼 곳" not in found["answer"], found["answer"]
    if arm == "arm":
        assert "오른" in found["answer"] and NO_ARM_QUICK not in found["answer"], found["answer"]
    else:
        assert (found["answer"].startswith(NO_ARM_QUICK) if mode == "quick" else NO_ARM_DETAILED in found["answer"]), found["answer"]


@both
def test_b_first_fix_question_has_one_answer_for_every_conjugation(arm: str, mode: str) -> None:
    """사례 5: "뭐부터 고치면 돼요?"와 "뭐부터 고쳐야 돼?"가 다른 틀로 갔다."""
    answers = {prompt: _ask(prompt, "pose", arm, mode) for prompt in ("그럼 뭐부터 고치면 돼요?", "뭐부터 고쳐야 돼?", "뭐부터 고칠까?")}
    assert len({found["answer"] for found in answers.values()}) == 1, answers
    answer = next(iter(answers.values()))["answer"]
    if arm == "arm":  # 자세 탭은 팔 자세 우선 교정(T22)
        assert "오른쪽 팔꿈치" in answer, answer
    else:
        assert "Zone 6" in answer and "작업 유사도" in answer, answer


@both
def test_b_pair_zone_correction_question_keeps_the_zone_work_answer(arm: str, mode: str) -> None:
    found = _ask("이 Zone에서 어떻게 고쳐?", "pose", arm, mode)
    assert found["source"] == "analysis" and "먼저 볼 곳" not in found["answer"], found["answer"]
    assert ("자세 유사도" if arm == "arm" else "Zone 8") in found["answer"], found["answer"]


# ---------------------------------------------------------------- C. 팔 채점이 없으면 신체 부위·자세 질문에 그 사실을 먼저 밝힌다
@both
@pytest.mark.parametrize("prompt", ["팔꿈치를 더 펴야 해? 몇 도 정도?", "왼손은?", "왼팔은?"])
def test_c_body_question_without_arm_scoring_says_so(prompt: str, arm: str, mode: str) -> None:
    """사례 7·10: 쪽이 없는 신체 부위는 작업 유사도 기본 답이, "왼손은?"은 대화 모델이 답했다."""
    found = _ask(prompt, "pose", arm, mode)
    assert found["source"] == "analysis" and found["calls"] == [], found
    if arm == "arm":
        assert NO_ARM_QUICK not in found["answer"] and NO_ARM_DETAILED not in found["answer"], found["answer"]
    elif mode == "quick":
        assert found["answer"].startswith(NO_ARM_QUICK), found["answer"]
    else:
        assert NO_ARM_DETAILED in found["answer"] and "작업 유사도" in found["answer"], found["answer"]


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("prompt, worst", [("지금 자세는 어때?", None), ("자세가 제일 문제였던 구역이 어디야?", ("Zone 9", "Zone 6"))])
def test_c_pose_question_without_arm_scoring_says_so_in_the_detailed_answer(prompt: str, worst: tuple[str, str] | None, arm: str) -> None:
    """사례 8·9: "자세"를 물었는데 팔 자세 채점이 없다는 말 없이 작업 유사도만 답했다.

    빠른 안내 한 줄은 작업 유사도 답을 그대로 둔다(이름이 "작업 유사도"라 수치 혼동은 없다). 짝: 빠른 안내.
    """
    detailed, quick = _ask(prompt, "pose", arm, "detailed"), _ask(prompt, "pose", arm, "quick")
    if arm == "arm":
        assert "유사도" in detailed["answer"] and NO_ARM_DETAILED not in detailed["answer"], detailed["answer"]
    else:
        assert "작업 유사도" in detailed["answer"] and NO_ARM_DETAILED in detailed["answer"], detailed["answer"]
        assert "작업 유사도" in quick["answer"] and NO_ARM_DETAILED not in quick["answer"], quick["answer"]
    if worst is not None:
        assert worst[0 if arm == "arm" else 1] in detailed["answer"], detailed["answer"]


# ---------------------------------------------------------------- D. 자세 탭의 기능·낱말 질문은 채점 엔진이 답하지 않는다
@both
@pytest.mark.parametrize("prompt", ["지금 자세탭인데 지식관련 질문만 가능하니?", "자세가 영어로 뭐야"])
def test_d_tab_or_word_question_is_not_a_scoring_lookup(prompt: str, arm: str, mode: str) -> None:
    """사례 11·12: "자세" 낱말 하나에 채점 답이 나왔다."""
    found = _ask(prompt, "pose", arm, mode)
    assert found["source"] == "conversation" and found["answer"] == CONVERSATION, found


@both
@pytest.mark.parametrize("prompt", ["자세는?", "지금 자세는 어때?"])
def test_d_pair_bare_pose_lookups_stay_with_the_engine(prompt: str, arm: str, mode: str) -> None:
    found = _ask(prompt, "pose", arm, mode)
    assert found["source"] == "analysis" and "plan" not in found["calls"], found
    if arm == "arm":
        assert "자세 유사도" in found["answer"], found["answer"]


@both
def test_d_pair_pose_concept_question_goes_to_the_documents(arm: str, mode: str) -> None:
    found = _ask("자세 종류로는 어떤게있어?", "pose", arm, mode)
    assert found["source"] == "documents", found


# ---------------------------------------------------------------- E. 일반 명사·문서 산출물 낱말은 기록·기준 순서 표지가 아니다
@both
@pytest.mark.parametrize("prompt", ["건도막 두께는 PSPC 구역이랑 그 외 구역에서 어떤 법칙으로 판정해?", "미도장 부위는 정확히 어떤 상태를 말하나요?"])
def test_e_rule_and_definition_questions_go_to_the_documents(prompt: str, arm: str, mode: str) -> None:
    """사례 16·17: "구역·부위" 낱말 하나에 도막 상태 답이 나왔다."""
    found = _ask(prompt, "knowledge", arm, mode)
    assert found["source"] == "documents" and DOCUMENT in found["answer"], found


@both
@pytest.mark.parametrize("prompt", ["도막이 과하거나 모자란 영역이 있어?", "정상 두께로 칠한 존이 있어?"])
def test_e_pair_coating_state_questions_stay_record_lookups(prompt: str, arm: str, mode: str) -> None:
    found = _ask(prompt, "knowledge", arm, mode)
    assert found["source"] == "analysis" and "µm" in found["answer"], found


@pytest.mark.parametrize("mode", MODES)
def test_e_order_in_a_document_figure_is_not_the_registered_plan(tmp_path, mode: str) -> None:
    """사례 14: "마일스톤 차트 예시 그림 … 순서"에 등록된 Zone 기준 순서가 답했다."""
    profile(tmp_path, "spray", ["1", "3", "2"])
    response = reference_client(tmp_path, pipeline=ReferencePipeline()).post("/api/v1/copilot/query", json={
        "user_id": "COPILOT-AI-TEST", "task_type": "knowledge_search", "scenario_id": "3", "response_mode": mode,
        "prompt": "마일스톤 차트 예시 그림에서 이정표로 표시된 주요 작업은 무엇이고 어떤 순서로 배열되어 있나요?"})
    data = response.json()["data"]
    assert data["sequence_context"] is None and "기준 순서" not in data["answer"], data


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("prompt", ["이 피도면의 작업 순서를 구역 단위로 알려주세요.", "권장 Zone 순서는?"])
def test_e_pair_plan_questions_keep_the_registered_plan(tmp_path, prompt: str, mode: str) -> None:
    profile(tmp_path, "spray", ["1", "3", "2"])
    response = reference_client(tmp_path, pipeline=ReferencePipeline()).post("/api/v1/copilot/query", json={
        "user_id": "COPILOT-AI-TEST", "task_type": "knowledge_search", "scenario_id": "3", "response_mode": mode, "prompt": prompt})
    assert "1 → 3 → 2" in response.json()["data"]["answer"].replace("→", " → ").replace("  ", " ")


# ---------------------------------------------------------------- F. 빠른 안내의 섞인 답은 먼저 물은 조각이 앞이다
@both
def test_f_quick_mixed_answer_starts_with_the_part_asked_first(arm: str, mode: str) -> None:
    """사례 18: "과도막은 왜 생겨? 내 기록에서도…"의 빠른 안내에 기록 문장만 남았다."""
    found = _ask("과도막은 왜 생겨? 내 기록에서도 과도막 난 데 있어?", "knowledge", arm, mode)
    assert found["source"] == "documents" and "0곳" in found["answer"], found["answer"]
    if mode == "quick":
        assert found["answer"].startswith(DOCUMENT), found["answer"]
    else:  # 상세 분석은 지금처럼 기록 문장이 앞이다
        assert found["answer"].startswith("칠한 곳 평균") and found["answer"].endswith(DOCUMENT), found["answer"]


def test_f_pair_quick_mixed_answer_without_document_text_keeps_the_record_sentence() -> None:
    found = _ask("과도막은 왜 생겨? 내 기록에서도 과도막 난 데 있어?", "knowledge", pipeline=RecordingPipeline(document="[답변] "))
    assert found["answer"].startswith("칠한 곳 평균"), found["answer"]


@both
def test_f_pair_record_only_coating_question_stays_a_record_lookup(arm: str, mode: str) -> None:
    found = _ask("두껍게 너무 많이 뿌린 데도 있어요?", "knowledge", arm, mode)
    assert found["source"] == "analysis" and "0곳" in found["answer"], found["answer"]


# ---------------------------------------------------------------- G. "채웠나요·넘어가도 되나요"는 이 Zone의 완료 판정이다
@both
def test_g_filled_and_move_on_question_is_answered_with_the_zone_gate(arm: str, mode: str) -> None:
    """사례 20: 완료 질문은 버려지고 다음 Zone 추천만 답했다."""
    found = _ask("이 Zone 도막은 목표 두께를 채웠나요? 다음 Zone으로 넘어가도 되나요?", "knowledge", arm, mode)
    answer = found["answer"]
    assert found["source"] == "analysis" and "Zone 8" in answer and "95%" in answer, answer
    assert "다음은 Zone" not in answer and "그 부분은 이 기록에 없습니다" not in answer, answer


@both
def test_g_pair_next_zone_question_still_recommends_the_next_zone(arm: str, mode: str) -> None:
    found = _ask("다음엔 어디 칠해야 돼?", "order", arm, mode)
    assert found["source"] == "analysis" and "Zone 5" in found["answer"], found["answer"]


@both
def test_g_pair_finished_question_keeps_the_completion_verdict(arm: str, mode: str) -> None:
    found = _ask("다 칠했으니까 이제 끝난 거죠?", "order", arm, mode, playback=AT_35S)
    assert found["source"] == "analysis" and found["answer"].startswith("아니요") and "95%" in found["answer"], found["answer"]
