"""자세 답의 결함(2026-11-14 운영 시험 q2·q4·q5·q6·q7, 27B Zone 번호 오기)."""
from __future__ import annotations

import pytest

from trainer.demo_api import dialogue, pose_query, session_analysis as sa
from trainer.demo_api.pose_query import PoseQueryResult
from trainer.demo_api.tests.viewer_record import AT_3S, RecordingPipeline, ask, payload
from trainer.demo_api.viewer_arm_pose import arm_pose_answer

NOTE = pose_query.POSE_SCALE_NOTE


def _model(reply: str = "왼손 자세는 정상 범위 내에 있는 것으로 보입니다.") -> RecordingPipeline:
    return RecordingPipeline(plan={"kind": "chat", "intent": "knowledge", "scope": "unspecified"}, converse=reply)


# --- (a) Zone별 문제 -------------------------------------------------------------------------------------

@pytest.mark.parametrize("prompt", ["Zone별로 어디가 제일 문제야?", "가장 문제가 큰 Zone은?", "어느 구역 자세가 제일 나빠?"])
def test_worst_zone_questions_get_the_per_zone_pose_summary(prompt: str) -> None:
    result = ask(prompt, tab="pose", pipeline=_model())
    answer = result["answer"]
    assert result["calls"] == [], answer
    # 자세 유사도(팔 관절 엔진)가 낮은 Zone부터: Zone 9(50%), Zone 1(59%).
    assert answer.index("Zone 9(50%)") < answer.index("Zone 1(59%)")
    assert "오른쪽 팔꿈치" in answer and NOTE in answer
    # 작업 유사도는 다른 척도다. 섞어 답하지 않고 이름을 달아 따로 적는다.
    assert "작업 유사도" in answer.split("\n")[-1] and "Zone 6(79%)" in answer.split("\n")[-1]


def test_model_sentences_that_name_the_wrong_zone_are_dropped() -> None:
    data = payload()
    zones = dialogue.zone_fact_numbers(data["dialogue_evaluation"], data["arm_pose_evaluation"])
    # Zone 2의 교정(5.8cm)을 1번 구역에 붙였다.
    wrong = "1번 구역은 도구를 피도면에서 약 5.8cm 멀리 이동해야 합니다. 이동 속도를 일정하게 유지하세요."
    assert dialogue.drop_misattributed_zones(wrong, zones) == "이동 속도를 일정하게 유지하세요."
    right = "Zone 2는 도구를 피도면에서 약 5.8cm 멀리 이동해야 합니다."
    assert dialogue.drop_misattributed_zones(right, zones) == right
    assert dialogue.drop_misattributed_zones("Zone 50은 자세가 좋았습니다. 다른 곳을 보세요.", zones) == "다른 곳을 보세요."


def test_conversation_answers_pass_the_zone_check() -> None:
    reply = "1번 구역은 도구를 피도면에서 약 5.8cm 멀리 이동해야 합니다. 추가로 궁금한 점을 말씀해 주세요."
    result = ask("음.. 그거", tab="knowledge", pipeline=_model(reply))
    assert result["calls"][-1] == "converse"
    assert result["answer"] == "추가로 궁금한 점을 말씀해 주세요."


# --- (b) 손·손목·팔꿈치 ----------------------------------------------------------------------------------

@pytest.mark.parametrize("tab", ["pose", "knowledge"])
def test_right_hand_question_answers_from_the_arm_engine(tab: str) -> None:
    result = ask("오른손은 어케 고쳐야돼", tab=tab, pipeline=_model())
    answer = result["answer"]
    assert result["calls"] == [], answer
    # v3.4.1: 반경 교정은 몸을 기준으로 적고 할 일을 붙인다. 오른손을 물었는데 손목 교정이 없으면 그렇다고 밝힌다.
    assert "오른쪽 팔꿈치가 숙련자 평균보다 몸 쪽으로 20cm 더 붙어" in answer and "손목 교정은 없고" in answer and "Zone 6" not in answer


@pytest.mark.parametrize("prompt", ["왼손은?", "왼손은 어떻게 고쳐?", "손목은 어때?"])
def test_no_correction_is_not_called_normal(prompt: str) -> None:
    result = ask(prompt, tab="pose", pipeline=_model())
    answer = result["answer"]
    assert result["calls"] == [], answer
    assert "나오지 않았습니다" in answer and "정상 자세를 확정하지 않습니다" in answer
    assert "정상 범위" not in answer and "통상 범위 안" not in answer


@pytest.mark.parametrize("prompt,mode", [("팔꿈치를 더 펴야 해? 몇 도 정도?", "quick"), ("팔꿈치 각도가 숙련자랑 몇 도 차이 나?", "detailed")])
def test_elbow_questions_use_the_forearm_result_without_inventing_angles(prompt: str, mode: str) -> None:
    result = ask(prompt, tab="pose", mode=mode, pipeline=_model())
    answer = result["answer"]
    assert result["calls"] == [], answer
    assert "°" not in answer and "확인할 수 없습니다" not in answer
    assert "각도" in answer and "몸 쪽으로 20cm 더 붙어" in answer


def test_tsb_elbow_question_reads_the_forearm_joint() -> None:
    zone = {"zone_id": "3", "evaluated_sequence_count": 2, "average_similarity_score": 0.6,
            "problem_joints": [{"joint": "RightForeArm", "occurrence_count": 1, "occurrence_rate": 0.5, "average_distance": 0.2}]}
    found = ("r", {"pose_report": {"average_similarity_score": 0.6, "zone_summary": [zone]}})
    assert pose_query.unsupported_pose_part("팔꿈치 각도는 몇 도야?") is None
    assert "어깨" in (pose_query.unsupported_pose_part("어깨는 어때?") or "")
    answer = pose_query.answer_pose_question(found, prompt="팔꿈치는 어때?", context=None).answer
    assert "오른쪽 팔꿈치" in answer
    hand = pose_query.answer_pose_question(found, prompt="오른쪽 손목은?", context=None).answer
    assert "오른쪽 손목" in hand and "정상 자세를 확정하지 않습니다" in hand


# --- (c) 교정 없음 ≠ 정상 ----------------------------------------------------------------------------------

def test_sequence_without_arm_correction_is_not_called_normal() -> None:
    pose = PoseQueryResult("available", None, "SEQ-1", 0.7, "joint_only", [], "요약")
    answer = pose_query.answer_sequence_question(pose, prompt="왼팔은?")
    assert "통상 범위 안" not in answer and "정상 자세를 확정하지 않습니다" in answer


# --- (d) 우선 교정의 수치와 문장 ------------------------------------------------------------------------

def _measured(score: float, joint: str, *, distance: float, radius: float | None = None, angle: float | None = None) -> PoseQueryResult:
    mode = "radius" if radius is not None else "heading"
    return PoseQueryResult("available", None, None, score, "joint_only", [{"joint": joint}], None,
                           joint_distances={joint: distance},
                           joint_measures={joint: (mode, radius if radius is not None else angle)})


def test_priority_correction_states_the_corrections_own_measure() -> None:
    scored = [("9", _measured(0.5, "RightForeArm", distance=0.35, radius=-0.20))]
    priority = sa.priority_corrections(sa.zone_summary(scored), measures=sa.joint_measures(scored))
    text = priority[0]["correction_guidance"]
    assert "몸 쪽으로 평균 20cm 더 붙어" in text and "35" not in text and "위치가 숙련자 범위" not in text
    scored = [("4", _measured(0.6, "LeftHand", distance=0.12, angle=31.0))]
    text = sa.priority_corrections(sa.zone_summary(scored), measures=sa.joint_measures(scored))[0]["correction_guidance"]
    assert "방향" in text and "31°" in text and "cm" not in text


# --- (e) 70% 안팎 ----------------------------------------------------------------------------------------

def test_pose_percent_answers_explain_the_expert_scale() -> None:
    arm = payload()["arm_pose_evaluation"]
    assert NOTE in arm_pose_answer(arm, current=False, playback=None, zone_id=None)
    assert NOTE in arm_pose_answer(arm, current=True, playback=AT_3S, zone_id=None)
    assert NOTE in ask("지금 자세는 어때?", tab="pose")["answer"]
    assert "70% 안팎" in NOTE and "숙련자 평균 동작" in NOTE


def test_viewer_pose_panels_carry_the_same_hint() -> None:
    from pathlib import Path
    static = Path(pose_query.__file__).with_name("static")
    assert "숙련자도 보통 70% 안팎" in (static / "motion_viewer_v2.html").read_text(encoding="utf-8")
    assert "숙련자도 보통 70% 안팎" in (static / "motion_viewer_pose.js").read_text(encoding="utf-8")


# --- 독립 검토 지적(2026-11-14): 안전·통증·도구 질문은 팔 관절 엔진 답이 아니다 ----------------------------

@pytest.mark.parametrize("prompt", ["손목 보호대 착용해야 해?", "팔꿈치 통증 예방법", "오른쪽 손잡이 압력", "오른손으로 잡은 건 각도는?"])
@pytest.mark.parametrize("tab", ["pose", "knowledge"])
def test_safety_pain_and_tool_questions_skip_the_arm_engine(prompt: str, tab: str) -> None:
    result = ask(prompt, tab=tab, pipeline=_model("모델 답입니다."))
    assert result["calls"], result["answer"]
    assert (result["pose_context"] or {}).get("scope") != "arm_pose" and pose_query.ANGLE_NOTE not in result["answer"]


def test_angle_note_is_only_for_joint_angles() -> None:
    assert pose_query.angle_note("팔꿈치 각도가 몇 도 차이 나?")
    assert pose_query.angle_note("오른손으로 잡은 건 각도는?") == ""
    assert pose_query.angle_note("오른팔은 어떻게 고쳐?") == ""
