"""v3.6.2 자세 낱말 라우팅(v360-review.md Important 1·2).

자세 낱말만으로는 도장 질문이 아니다. 도장 낱말과 함께이거나 자세 문맥(자세 탭·서명된 자세 기록)일 때만 센다.
자세 문맥에서도 "종류·유형·뜻·의미·정의"를 묻는 개념 질문만 문서로 보낸다(966a3d0의 의도). 기록 질문은 4c5da92처럼 답한다.
"""
from __future__ import annotations

import pytest

from trainer.demo_api.copilot_contracts import CopilotQueryRequest
from trainer.demo_api.copilot_intent import classify_question_intent, fallback_intent, is_domain_question, is_general_painting_knowledge
from trainer.demo_api.copilot_router import _pose_engine_question
from trainer.demo_api.tests.viewer_record import RecordingPipeline, ask

RECORD = ["자세 유사도는 얼마야?", "자세 점수가 왜 낮아?", "팔꿈치는 왜 지적됐어?", "전문가와 자세 차이는?", "무릎 자세는 어떻게 고쳐?"]
OFF_TOPIC = ["요리 도구 종류는 뭐야?", "요가 자세 종류는?", "스쿼트 동작 방법 알려줘", "허리 디스크 예방 방법은?"]
CONCEPT = ["자세 종류로는 어떤게있어?", "도장 작업 자세에는 어떤 종류가 있나요?", "무릎을 굽힌 자세의 의미는?"]

# 4c5da92에서 서명된 뷰어 기록으로 물었을 때의 경로(answer source, 부른 모델 단계)와 답 머리.
BASE = {
    ("pose", "자세 유사도는 얼마야?"): ("analysis", [], "분사 2회 중 2회의 자세를 숙련 작업자와 비교했습니다."),
    ("pose", "자세 점수가 왜 낮아?"): ("analysis", [], "분사 2회 중 2회의 자세를 숙련 작업자와 비교했습니다."),
    ("pose", "팔꿈치는 왜 지적됐어?"): ("analysis", [], "채점한 분사 2회 가운데 1회에서 팔꿈치 교정이 나왔습니다."),
    ("pose", "전문가와 자세 차이는?"): ("analysis", [], "분사 2회 중 2회의 자세를 숙련 작업자와 비교했습니다."),
    ("pose", "무릎 자세는 어떻게 고쳐?"): ("analysis", [], "질문하신 내용(무릎)은 채점 결과로는 확인할 수 없습니다."),
    ("order", "자세 유사도는 얼마야?"): ("conversation", ["plan", "converse"], ""),
    ("order", "자세 점수가 왜 낮아?"): ("conversation", ["plan", "converse"], ""),
    ("order", "팔꿈치는 왜 지적됐어?"): ("analysis", [], "채점한 분사 2회 가운데 1회에서 팔꿈치 교정이 나왔습니다."),
    ("order", "전문가와 자세 차이는?"): ("conversation", ["plan", "converse"], ""),
    ("order", "무릎 자세는 어떻게 고쳐?"): ("conversation", ["plan", "converse"], ""),
}


@pytest.mark.parametrize("tab,prompt", list(BASE), ids=[f"{t}-{p}" for t, p in BASE])
def test_signed_pose_record_questions_answer_as_before_v360(tab: str, prompt: str) -> None:
    source, calls, head = BASE[(tab, prompt)]
    result = ask(prompt, tab=tab)
    assert (result["source"], result["calls"]) == (source, calls), result
    assert result["answer"].startswith(head), result["answer"]


def _query(prompt: str, task: str = "pose_recomm") -> CopilotQueryRequest:
    return CopilotQueryRequest.model_validate({"user_id": "u", "session_id": "S1", "scenario_id": "3", "task_type": task, "prompt": prompt})


@pytest.mark.parametrize("prompt", RECORD)
def test_tsb_session_pose_questions_go_to_the_scoring_engine(prompt: str) -> None:
    assert _pose_engine_question(_query(prompt))


@pytest.mark.parametrize("prompt", CONCEPT)
def test_pose_concept_questions_skip_the_scoring_engine(prompt: str) -> None:
    assert not _pose_engine_question(_query(prompt))


@pytest.mark.parametrize("tab", ["knowledge", "order"])
@pytest.mark.parametrize("prompt", OFF_TOPIC)
@pytest.mark.parametrize("viewer", [True, False])
def test_off_topic_questions_stay_on_the_conversation_route(prompt: str, tab: str, viewer: bool) -> None:
    result = ask(prompt, tab=tab, **({} if viewer else {"raw_context": ""}))
    assert (result["source"], result["calls"]) == ("conversation", ["plan", "converse"]), result


@pytest.mark.parametrize("prompt", OFF_TOPIC)
def test_pose_words_alone_are_not_painting_words(prompt: str) -> None:
    assert not is_domain_question(prompt) and not is_general_painting_knowledge(prompt)


@pytest.mark.parametrize("prompt", CONCEPT)
def test_pose_concept_questions_use_documents_in_a_pose_context(prompt: str) -> None:
    for kwargs in ({"tab": "pose"}, {"tab": "pose", "raw_context": ""}):
        pipeline = RecordingPipeline()
        result = ask(prompt, pipeline=pipeline, **kwargs)
        assert result["source"] == "documents" and "plan" not in pipeline.calls, (kwargs, result)
    assert is_general_painting_knowledge(prompt, pose=True)
    assert fallback_intent("pose_recomm", prompt) == "knowledge"
    assert classify_question_intent("pose_recomm", prompt, lambda *_: "pose_overall") == "knowledge"


@pytest.mark.parametrize("prompt", RECORD)
def test_pose_record_questions_are_not_concept_questions_even_in_a_pose_context(prompt: str) -> None:
    assert not is_general_painting_knowledge(prompt, pose=True)
