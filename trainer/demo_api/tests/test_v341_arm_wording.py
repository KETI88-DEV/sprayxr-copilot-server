"""팔 자세 답의 표현(운영 2026-11-14, 전형기 학습모드 4 · 4_1000768_1 · "오른손을 어떻게 고쳐?").

운영 답: "채점한 분사 24회 중 4회에서 오른팔 교정이 나왔습니다. 예(SEQ-005): 스트로크 0~100% 구간에서 오른쪽 팔꿈치가
숙련자 평균보다 22 cm 가까이 있었습니다. 이 가운데 15회는 입력 품질 제한이 있어 참고용 채점입니다."
- "가까이"의 기준이 없다. 반경 교정은 어깨 중점에서 관절까지의 거리다(arm_reference.Correction) → 몸 쪽으로 붙었다.
- 15회는 모두 frame_interval_irregular(프레임 간격 고르지 않음)뿐이고 채점에서 빠진 것이 없다 → 말하지 않는다.
- 스트로크 구간 같은 용어는 그대로 둔다(2026-11-14 지시: 바꾸는 것은 문체이지 용어가 아니다).
- 오른손을 물었는데 팔꿈치를 골랐다. SEQ-025에 오른쪽 손목 교정이 있었다.
교정 문장은 MCP 자세 엔진이 만든다(그 이미지는 다시 배포하지 않는다). agent-api에서 바꿔 쓴다.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from trainer.demo_api import pose_query, session_analysis as sa
from trainer.demo_api.pose_query import PoseQueryResult
from trainer.demo_api.viewer_arm_pose import arm_pose_answer

ELBOW = "스트로크 0~100% 구간에서 오른쪽 팔꿈치가 숙련자 평균보다 22 cm 가까이 있었습니다."
WRIST = "스트로크 20~80% 구간에서 오른쪽 손목이 숙련자 평균보다 9 cm 멀리 있었습니다."
HEADING = "왼쪽 위팔이 향한 방향이 숙련자 평균과 31° 달랐습니다."


def _sequence(number: int, corrections: list[str], joints: list[str], *, reasons: list[str] | None = None) -> dict:
    return {"sequence_id": f"SEQ-{number:03d}", "zone_id": "3", "start_seconds": float(number), "end_seconds": number + 0.5,
            "similarity_percent": 70, "correction": corrections[0] if corrections else None, "corrections": corrections,
            "joints": joints, "confidence": "limited" if reasons else None,
            "reason_code": reasons[0] if reasons else None, "reason_codes": reasons or []}


def _arm(sequences: list[dict]) -> dict:
    return {"status": "available", "average_similarity_percent": 70, "evaluated_sequence_count": len(sequences),
            "sequence_count": len(sequences), "guideline": "", "zones": {}, "priority_corrections": [], "sequences": sequences}


def test_radius_corrections_name_the_body_and_give_an_action() -> None:
    assert pose_query.friendly_correction(ELBOW) == (
        "스트로크 내내 오른쪽 팔꿈치가 숙련자 평균보다 몸 쪽으로 22cm 더 붙어 있었습니다. 팔꿈치를 몸에서 더 떼어 주세요.")
    assert pose_query.friendly_correction(WRIST) == (
        "스트로크 20~80% 구간에서 오른쪽 손목이 숙련자 평균보다 몸 밖으로 9cm 더 뻗어 있었습니다. 손목을 몸 쪽으로 당겨 주세요.")
    assert pose_query.friendly_correction(HEADING) == HEADING


def test_production_right_hand_answer() -> None:
    # 24회 중 15회는 프레임 간격만 고르지 않았다. 오른팔 교정 4회(팔꿈치 3, 손목 1).
    sequences = [_sequence(n, [], [], reasons=["frame_interval_irregular"]) for n in range(1, 16)]
    sequences += [_sequence(n, [], []) for n in range(16, 21)]
    sequences += [_sequence(5 + 100 * k, [ELBOW], ["RightForeArm"]) for k in range(1, 4)]
    sequences.append(_sequence(25, [WRIST], ["RightHand"]))
    answer = arm_pose_answer(_arm(sequences), current=False, playback=None, zone_id=None, side="right", prefer_hand=True)
    assert "입력 품질" not in answer and "참고로만" not in answer
    assert "오른쪽 손목" in answer and "SEQ-025" in answer and "몸 밖으로 9cm 더 뻗어" in answer
    assert "cm 가까이" not in answer


def test_right_hand_without_a_wrist_correction_says_so_and_uses_the_arm() -> None:
    sequences = [_sequence(5, [ELBOW], ["RightForeArm"])]
    answer = arm_pose_answer(_arm(sequences), current=False, playback=None, zone_id=None, side="right", prefer_hand=True)
    assert "손목 교정은 없었습니다" in answer and "팔꿈치를 몸에서 더 떼어 주세요" in answer


def test_other_limit_reasons_are_said_once_in_plain_words() -> None:
    sequences = [_sequence(1, [], [], reasons=["skeleton_joint_frozen", "frame_interval_irregular"]),
                 _sequence(2, [], [], reasons=["skeleton_joint_frozen"]), _sequence(3, [], [], reasons=["frame_interval_irregular"])]
    answer = arm_pose_answer(_arm(sequences), current=False, playback=None, zone_id=None, side="left")
    assert answer.count("참고로만") == 1 and "관절 일부가 멈춘 채로 기록된 분사 2회는 참고로만 봐 주세요" in answer
    assert "입력 품질" not in answer


def test_viewer_sequences_carry_friendly_text_and_the_engine_reason() -> None:
    assert pose_query.limited_note([{"confidence": "limited", "reason_codes": ["frame_interval_irregular"]}]) == ""
    item = {"confidence": "limited", "reason_code": "skeleton_bone_length_inconsistent"}
    assert pose_query.limited_note([item]) == "관절 위치가 흔들려 기록된 분사 1회는 참고로만 봐 주세요."


def test_tsb_sequence_answer_prefers_the_wrist_for_a_hand_question() -> None:
    corrections = [{"joint": "RightForeArm", "label": ELBOW}, {"joint": "RightHand", "label": WRIST}]
    pose = PoseQueryResult("available", None, "SEQ-1", 0.7, "joint_only", corrections, "요약")
    answer = pose_query.answer_sequence_question(pose, prompt="오른손을 어떻게 고쳐?")
    assert answer.startswith("스트로크 20~80% 구간에서 오른쪽 손목") and "팔꿈치" not in answer


def test_session_guideline_does_not_blame_input_quality_for_frame_intervals() -> None:
    timing = PoseQueryResult("available", "frame_interval_irregular", "S1", 0.7, "joint_only", [], None, confidence="limited",
                             input_quality={"reason_codes": ["frame_interval_irregular"]})
    frozen = PoseQueryResult("available", "skeleton_joint_frozen", "S2", 0.6, "joint_only", [], None, confidence="limited",
                             input_quality={"reason_codes": ["skeleton_joint_frozen"]})
    assert "입력 품질" not in sa._guideline([timing], 1, 0) and "참고로만" not in sa._guideline([timing], 1, 0)
    assert "관절 일부가 멈춘 채로 기록된 분사 1회는 참고로만 봐 주세요" in sa._guideline([timing, frozen], 2, 0)


def test_priority_correction_names_the_body_reference() -> None:
    scored = [("9", PoseQueryResult("available", None, None, 0.5, "joint_only", [{"joint": "RightForeArm"}], None,
                                    joint_distances={"RightForeArm": 0.35}, joint_measures={"RightForeArm": ("radius", -0.22)}))]
    text = sa.priority_corrections(sa.zone_summary(scored), measures=sa.joint_measures(scored))[0]["correction_guidance"]
    assert "몸 쪽으로 평균 22cm 더 붙어" in text and "떼어 주세요" in text


def test_viewer_hud_uses_the_server_limited_note() -> None:
    static = Path(pose_query.__file__).with_name("static")
    for name in ("motion_viewer_v2.js", "motion_viewer_pose.js"):
        text = (static / name).read_text(encoding="utf-8")
        assert "입력 품질 제한" not in text and "limited_note" in text, name


@pytest.mark.parametrize("label", [ELBOW, WRIST])
def test_rewrite_keeps_the_side_and_joint_words_for_matching(label: str) -> None:
    rewritten = pose_query.friendly_correction(label)
    assert ("오른쪽 팔꿈치" in rewritten) or ("오른쪽 손목" in rewritten)
