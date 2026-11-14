"""도막 조절 후보를 실측과 맞춰 쓴다(2026-11-14 운영 q19·q20·q33·q42, 이슈 #19).

PR #8 규칙은 저도막 Zone마다 "속도↓·거리↓"를 낸다. 이 기록의 저도막 8곳 가운데 6곳은 이미 전문가보다 느리고
가까웠다. 규칙(`trainer/zone_engine`, `coating_assessment`)은 고치지 않고, 답과 모델 자료에서 쓰는 쪽만 고친다.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api import coating_query as cq
from trainer.demo_api.copilot_router import create_copilot_router
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.tests.viewer_record import AT_3S, RecordingPipeline, ask, context, payload

Q19 = ("도막 두께가 목표 160µm보다 낮아 저도막 등급으로 분류된 이유는 작업 시 이격거리가 251mm로 전문가 기준 550mm보다 약 54% 짧았고, "
       "이동 속도가 2,230mm/s로 전문가 기준 3,514mm/s보다 약 37% 낮아 도막이 얇게 형성되었기 때문입니다.")
Q42 = ("천천히 뿌리는 방식은 도막 두께에 영향을 줄 수 있지만, 이 경우 속도와 이격거리가 전문가 기준보다 낮아 도막이 얇게 형성된 것으로 보입니다. "
       "속도가 느려지면 분사량이 줄어들 수 있으므로, 일정한 속도를 유지하면서도 가까이서 뿌리는 방식을 조정해 보는 것이 좋습니다.")
Q33 = "과도막은 스프레이 작업 시 과도한 분사 거리 또는 속도로 인해 도막이 두꺼워져 발생합니다."


def _measures() -> dict:
    return cq.zone_measures(payload()["dialogue_evaluation"])


def test_measured_comparison_decides_whether_a_candidate_is_advice() -> None:
    measures = _measures()
    assert cq.measured_relation(measures["8"], "speed_mm_s") == "lower"
    assert cq.measured_relation(measures["8"], "distance_mm") == "lower"
    assert cq.measured_relation(measures["4"], "distance_mm") == "higher"
    assert cq.measured_relation(measures["7"], "speed_mm_s") == "similar"  # 1,976 vs 1,964 mm/s
    decrease = {"variable": "distance_mm", "direction": "decrease", "priority": 2}
    assert cq.contradicted(decrease, measures["8"]) and not cq.contradicted(decrease, measures["4"])
    # 전문가 속도는 이전 장비로 잰 값이라(훈련생이 약 0.36배) 속도 후보는 실측과 엇갈린다고 판정하지 않는다(v3.4.1).
    assert not cq.contradicted({"variable": "speed_mm_s", "direction": "decrease", "priority": 1}, measures["8"])


def test_distribution_cause_states_measurements_instead_of_contradicted_candidates() -> None:
    coating = payload()["coating_evaluation"]
    answer = cq.answer_distribution_cause(coating, detailed=True, measures=_measures())
    zone8 = next(line for line in answer.split("\n") if line.startswith("Zone 8은"))
    # 속도는 측정값으로만 적고 전문가와 견주지 않는다. 거리만 견줘 후보를 걸러 낸다.
    assert zone8 == ("Zone 8은 저도막입니다(칠한 곳 평균 77µm). Zone 8 실측값은 이동 속도 2,230 mm/s, 이격거리 251 mm입니다"
                     "(전문가 550 mm보다 54% 가까움). 가까이 칠하면 도막은 보통 두꺼워집니다. 그래서 얇게 나온 이유를 거리로는 "
                     "설명하기 어렵습니다. 조절 후보는 이동 속도를 낮추는 쪽입니다.")
    # 실측과 엇갈리는 후보(거리를 줄이는 쪽)는 답 어디에도 말하지 않는다(2026-11-14 사용자 결정).
    assert "이격거리를 줄이는 쪽" not in zone8 and "검토 중" not in answer and "규칙의 조절 후보" not in answer
    zone4 = next(line for line in answer.split("\n") if line.startswith("Zone 4는"))
    assert "조절 후보는 이동 속도를 낮추는 쪽, 이격거리를 줄이는 쪽입니다." in zone4 and "먼" not in zone4
    # 실측이 없으면(완료 리포트만) 규칙 후보를 그대로 적는다.
    assert "조절 후보는 이동 속도를 낮추는 쪽, 이격거리를 줄이는 쪽입니다." in cq.answer_distribution_cause(coating, detailed=True)


def test_error_analysis_query_uses_the_signed_measurements() -> None:
    data = payload()
    app = FastAPI()
    app.include_router(create_copilot_router(None, store=SequenceStore(":memory:")))
    answer = TestClient(app).post("/api/v1/copilot/query", json={
        "user_id": "u", "session_id": data["selected_session_id"], "task_type": "error_analysis", "prompt": "왜 얇아?",
        "response_mode": "detailed", "context": context("knowledge_search", data, AT_3S)}).json()["data"]["answer"]
    zone8 = next(line for line in answer.split("\n") if line.startswith("Zone 8은"))
    assert "조절 후보는 이동 속도를 낮추는 쪽입니다." in zone8 and "검토 중" not in zone8 and "이격거리를 줄이는 쪽" not in zone8


def test_model_facts_carry_measurements_and_drop_contradicted_candidates() -> None:
    data = payload()
    facts = cq.target_advice_facts({**data["coating_evaluation"], "playback_context": {"map_num": "1", "zone_id": "8"}},
                                   measures=_measures())
    zone8 = facts["zones"][0]
    assert zone8["zone_id"] == "8" and [item["variable"] for item in zone8["adjustments"]] == ["speed_mm_s"]
    # 엇갈리는 후보는 모델 자료에도 싣지 않는다(모델이 말하지 못하게).
    assert "adjustments_under_review" not in zone8 and "distance_mm" not in {item["variable"] for item in zone8["adjustments"]}
    # 속도는 측정값만 싣는다. 전문가 속도와의 관계는 모델에 넘기지 않는다.
    assert zone8["measured"]["speed_mm_s"] == {"record": 2230.0}
    assert zone8["measured"]["distance_mm"] == {"record": 250.8, "expert_median": 550.0, "relation": "lower"}


@pytest.mark.parametrize("sentence", [
    "거리가 251mm로 짧았고 속도가 낮아 도막이 얇게 형성되었습니다.",
    "속도가 느려지면 분사량이 줄어들 수 있습니다.",
    "과도한 분사 거리 또는 속도로 인해 도막이 두꺼워져 발생합니다.",
    "거리가 멀어서 도막이 두꺼워졌습니다.",
])
def test_physically_inverted_causes_are_detected(sentence: str) -> None:
    assert cq.inverted_cause(sentence)


@pytest.mark.parametrize("sentence", [
    "가까이·천천히 칠하면 도막이 두꺼워집니다.", "속도가 빨라서 도막이 얇아졌습니다.",
    "Zone 8은 전문가보다 가깝고 느리게 칠했는데도 얇게 나왔습니다.", "저도막은 목표보다 얇은 도막입니다.",
])
def test_correct_statements_are_kept(sentence: str) -> None:
    assert not cq.inverted_cause(sentence)


def test_advice_that_contradicts_the_measured_direction_is_dropped() -> None:
    measures = _measures()
    kept = cq.drop_contradicted_causes("Zone 8에서는 이격거리를 더 줄이세요. 패스를 50% 겹치세요.", measures, default_zone="8")
    assert kept == "패스를 50% 겹치세요."
    assert cq.drop_contradicted_causes("이격거리를 줄이세요.", measures, default_zone="4") == "이격거리를 줄이세요."


def test_why_thin_answer_does_not_pass_an_inverted_cause() -> None:
    # 운영에서는 계획 모델이 문서 답으로 보냈다(답출처 documents).
    pipeline = RecordingPipeline(plan={"kind": "coating_knowledge", "intent": "knowledge", "scope": "zone"}, document="[답변] " + Q19)
    result = ask("왜 이렇게 얇게 나왔어?", tab="coating", pipeline=pipeline)
    answer = result["answer"]
    # v3.5.0 설계 표 v19: 계획 모델·문서 답 없이 서명된 기록 행으로 답한다.
    assert result["calls"] == [] and result["source"] == "analysis"
    assert "얇게 형성되었기 때문" not in answer
    # 대신 실측 거리 비교를 말하고, 가까운 쪽은 보통 두꺼워진다는 점을 밝힌다.
    assert "251 mm" in answer and "550 mm" in answer and "가까이 칠하면 도막은 보통 두꺼워집니다." in answer
    assert "거리로는 설명하기 어렵" in answer


def test_correction_follow_up_does_not_repeat_the_inverted_cause() -> None:
    pipeline = RecordingPipeline(plan={"kind": "correction", "intent": "coating_status", "scope": "zone"}, converse=Q42)
    answer = ask("근데 가까이서 천천히 뿌리면 오히려 두꺼워지는 거 아냐? 아까 답 이상한데", tab="coating", pipeline=pipeline)["answer"]
    assert "얇게 형성" not in answer and "분사량이 줄" not in answer


def test_inverted_physics_is_checked_only_against_a_viewer_record() -> None:
    pipeline = RecordingPipeline(plan={"kind": "coating_knowledge", "intent": "knowledge", "scope": "unspecified"}, document="[답변] " + Q33)
    # 기록 문맥 없는 일반 지식 답에는 손대지 않는다(독립 검토 2026-11-14).
    assert ask("과도막은 왜 생겨?", tab="knowledge", pipeline=pipeline, raw_context="")["answer"] == Q33
    viewer = ask("과도막은 왜 생겨? 내 기록에서도 과도막 난 데 있어?", tab="knowledge", pipeline=pipeline)["answer"]
    assert "과도한 분사 거리" not in viewer and "두꺼워" in viewer


@pytest.mark.parametrize("sentence", [
    "거리가 멀어지면 도막이 얇아지고, 가까워지면 두꺼워집니다.",
    "속도가 느려지면 두꺼워지고 빨라지면 얇아집니다.",
    "도막 두께가 목표 160µm보다 낮아 저도막 등급입니다.",
])
def test_correct_two_way_physics_is_kept(sentence: str) -> None:
    assert not cq.inverted_cause(sentence)
    pipeline = RecordingPipeline(plan={"kind": "coating_knowledge", "intent": "knowledge", "scope": "unspecified"}, document="[답변] " + sentence)
    assert ask("거리와 속도가 도막에 어떤 영향을 줘?", tab="knowledge", pipeline=pipeline)["answer"] == sentence


def test_a_dropped_body_never_leaves_only_labels() -> None:
    answer = cq.drop_contradicted_causes("[답변] 거리가 멀어서 도막이 두꺼워졌습니다.\n[근거] - 도장 매뉴얼", _measures(), default_zone=None)
    assert "[답변]" not in answer and "두꺼워지" in answer


def test_general_advice_is_not_judged_against_the_playhead_zone() -> None:
    # 재생 위치가 Zone 8(전문가보다 가까움)이어도 일반 질문의 "거리를 줄이세요"는 그 Zone 조언이 아니다.
    pipeline = RecordingPipeline(plan={"kind": "coating_knowledge", "intent": "knowledge", "scope": "unspecified"},
                                 document="[답변] 도막이 두꺼우면 이격거리를 줄이세요.")
    assert ask("도막이 흘러내리면 어떻게 해?", tab="knowledge", pipeline=pipeline)["answer"] == "도막이 두꺼우면 이격거리를 줄이세요."
    zone = RecordingPipeline(plan={"kind": "coating_knowledge", "intent": "knowledge", "scope": "zone"},
                             document="[답변] 이격거리를 더 줄이세요.")
    answer = ask("이 구역 도막을 두껍게 하려면 어떻게 해?", tab="coating", pipeline=zone)["answer"]
    assert "더 줄이세요" not in answer and "Zone 8은 251 mm 거리에서 칠했습니다" in answer
