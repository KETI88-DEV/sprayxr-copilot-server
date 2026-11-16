"""v3.4.0 운영 질문 시험(2026-11-14)에서 남은 결함: 기록과 어긋난 원인 문장, 측정값 짧은 답의 가로채기."""
from __future__ import annotations

import pytest

from trainer.demo_api import coating_query as cq

from trainer.demo_api.tests.viewer_record import RecordingPipeline, ask, payload

Q19 = "분사 거리가 너무 멀어 도막 두께가 얇아졌습니다."
Q42 = ("천천히 뿌리는 방식은 도막 두께에 영향을 줄 수 있지만, 이 기록에서는 분사 거리가 너무 멀어 도막이 얇아졌다는 분석입니다. "
       "분사 속도와 거리 조절이 도막 두께에 직접적인 영향을 미칩니다.")
Q40 = ("안녕하세요 저는 이번에 처음 도장 훈련을 받는 사람인데요, 오늘 시뮬레이터에서 스프레이로 칠해 봤는데 화면에 저도막이라고 빨갛게 "
       "많이 뜨더라고요. 선배는 건이 너무 가까우면 오히려 흘러내린다고 하고 다른 선배는 멀리 떨어지면 건조 분무가 생긴다고 해서 헷갈립니다. "
       "결국 제가 지금 제일 먼저 신경 써야 하는 게 거리인지 속도인지 겹침인지, 그리고 그 이유가 뭔지 하나만 콕 집어서 알려주시면 좋겠습니다.")
Q22 = "Zone 6 속도랑 거리 어땠어? 뭐부터 고치면 돼?"


def _documents(answer: str) -> RecordingPipeline:
    return RecordingPipeline(plan={"kind": "coating_knowledge", "intent": "knowledge", "scope": "zone"}, document="[답변] " + answer)


def test_q19_far_distance_cause_is_replaced_by_the_measured_distance() -> None:
    # Zone 8은 251 mm로 전문가 550 mm보다 가까웠다. "멀어 얇아졌다"는 이 기록과 반대다.
    result = ask("왜 이렇게 얇게 나왔어?", tab="coating", pipeline=_documents(Q19), mode="detailed")  # 상세 분석의 전체 답. 빠른 안내는 한 줄 60자다(2026-11-16)
    answer = result["answer"]
    assert result["source"] == "analysis"  # v3.5.0 설계 표 v19: 문서 답 없이 기록 행으로 답한다
    assert "너무 멀어" not in answer
    assert "251 mm" in answer and "550 mm" in answer and "가까" in answer
    assert "거리로는 설명하기 어렵" in answer


def test_q42_conversation_follow_up_does_not_repeat_the_far_distance_cause() -> None:
    pipeline = RecordingPipeline(plan={"kind": "correction", "intent": "coating_status", "scope": "zone"}, converse=Q42)
    answer = ask("근데 가까이서 천천히 뿌리면 오히려 두꺼워지는 거 아냐? 아까 답 이상한데", tab="coating", pipeline=pipeline)["answer"]
    assert pipeline.calls[-1] == "converse"
    assert "너무 멀어" not in answer and "251 mm" in answer and "가까" in answer


def test_record_cause_check_uses_distance_only() -> None:
    measures = cq.zone_measures(payload()["dialogue_evaluation"])
    # 속도 원인 문장은 기록과 견주지 않는다(전문가 속도는 다른 장비로 잰 값이다).
    speed = "속도가 너무 빨라 도막이 얇아졌습니다."
    assert cq.drop_contradicted_causes(speed, measures, default_zone="8") == speed
    # 이 기록 Zone 4는 전문가보다 멀었다(407 vs 330 mm). "멀어 얇아졌다"는 맞으니 둔다.
    assert cq.drop_contradicted_causes(Q19, measures, default_zone="4") == Q19


@pytest.mark.parametrize("answer", ["거리가 멀어지면 도막이 얇아지고, 가까워지면 두꺼워집니다.", "분사 거리가 멀면 도막이 얇아집니다."])
def test_general_physics_in_a_viewer_knowledge_answer_is_untouched(answer: str) -> None:
    pipeline = RecordingPipeline(plan={"kind": "coating_knowledge", "intent": "knowledge", "scope": "unspecified"}, document="[답변] " + answer)
    assert ask("거리와 도막 두께는 어떤 관계야?", tab="knowledge", pipeline=pipeline)["answer"] == answer
