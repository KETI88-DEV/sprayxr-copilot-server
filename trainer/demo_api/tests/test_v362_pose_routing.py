"""v3.6.2 자세 낱말 라우팅(v360-review.md Important 1·2, 조정 라운드 1 판정 R1).

자세 낱말(자세·동작·팔꿈치·무릎 등)은 도장 질문의 낱말이다(966a3d0). 다만 도장이 아닌 활동·분야를 가리키는 낱말
(요리·요가·스쿼트·디스크 등)이 있고 도장 낱말이 없으면 주제 밖 질문이라 어느 탭이든 대화 경로에 둔다.
자세 낱말만 있는 질문은 종류·유형·뜻·의미·정의를 물을 때만 문서로 보낸다. 기록 질문은 4c5da92처럼 답한다.
"""
from __future__ import annotations

import pytest

from trainer.demo_api.copilot_contracts import CopilotQueryRequest
from trainer.demo_api.copilot_intent import is_domain_question, is_general_painting_knowledge
from trainer.demo_api.copilot_router import _pose_engine_question
from trainer.demo_api.tests.document_evidence import DocumentRecordingPipeline
from trainer.demo_api.tests.viewer_record import ask

RECORD = ["자세 유사도는 얼마야?", "자세 점수가 왜 낮아?", "팔꿈치는 왜 지적됐어?", "전문가와 자세 차이는?", "무릎 자세는 어떻게 고쳐?"]
OFF_TOPIC = ["요리 도구 종류는 뭐야?", "요가 자세 종류는?", "요가 동작 종류 알려줘", "스쿼트 동작 방법 알려줘", "허리 디스크 예방 방법은?"]
CONCEPT = ["자세 종류로는 어떤 게 있어?", "자세 종류로는 어떤게있어?", "도장 작업 자세에는 어떤 종류가 있나요?",
           "무릎을 굽힌 자세의 의미는?", "스프레이 자세 종류는?"]
TABS = ["knowledge", "order", "pose"]

# 4c5da92에서 서명된 뷰어 기록으로 물었을 때의 경로(answer source, 부른 모델 단계)와 답 머리.
BASE = {
    ("pose", "자세 유사도는 얼마야?"): ("analysis", [], "분사 2회 중 2회의 자세를 숙련 작업자와 비교했습니다."),
    ("pose", "자세 점수가 왜 낮아?"): ("analysis", [], "분사 2회 중 2회의 자세를 숙련 작업자와 비교했습니다."),
    ("pose", "팔꿈치는 왜 지적됐어?"): ("analysis", [], "채점한 분사 2회 중 1회에서 팔꿈치 교정:"),
    ("pose", "전문가와 자세 차이는?"): ("analysis", [], "분사 2회 중 2회의 자세를 숙련 작업자와 비교했습니다."),
    ("pose", "무릎 자세는 어떻게 고쳐?"): ("analysis", [], "질문하신 내용(무릎)은 채점 결과로는 확인할 수 없습니다."),
    ("order", "자세 유사도는 얼마야?"): ("conversation", ["plan", "converse"], ""),
    ("order", "자세 점수가 왜 낮아?"): ("conversation", ["plan", "converse"], ""),
    ("order", "팔꿈치는 왜 지적됐어?"): ("analysis", [], "채점한 분사 2회 중 1회에서 팔꿈치 교정:"),
    ("order", "전문가와 자세 차이는?"): ("conversation", ["plan", "converse"], ""),
    ("order", "무릎 자세는 어떻게 고쳐?"): ("conversation", ["plan", "converse"], ""),
}


@pytest.mark.parametrize("tab,prompt", list(BASE), ids=[f"{t}-{p}" for t, p in BASE])
def test_signed_pose_record_questions_answer_as_before_v360(tab: str, prompt: str) -> None:
    source, calls, head = BASE[(tab, prompt)]
    result = ask(prompt, tab=tab)
    assert (result["source"], result["calls"]) == (source, calls), result
    # 답 머리는 상세 분석에서 견준다. 빠른 안내는 한 줄 60자다(2026-11-16).
    assert ask(prompt, tab=tab, mode="detailed")["answer"].startswith(head) and "\n" not in result["answer"]


def _query(prompt: str, task: str = "pose_recomm") -> CopilotQueryRequest:
    return CopilotQueryRequest.model_validate({"user_id": "u", "session_id": "S1", "scenario_id": "3", "task_type": task, "prompt": prompt})


@pytest.mark.parametrize("prompt", RECORD)
def test_tsb_session_pose_questions_go_to_the_scoring_engine(prompt: str) -> None:
    assert _pose_engine_question(_query(prompt))
    assert not is_general_painting_knowledge(prompt)


@pytest.mark.parametrize("prompt", CONCEPT + OFF_TOPIC)
def test_concept_and_off_topic_questions_skip_the_scoring_engine(prompt: str) -> None:
    assert not _pose_engine_question(_query(prompt))


@pytest.mark.parametrize("tab", TABS)
@pytest.mark.parametrize("prompt", OFF_TOPIC)
@pytest.mark.parametrize("viewer", [True, False])
def test_off_topic_questions_stay_on_the_conversation_route_on_every_tab(prompt: str, tab: str, viewer: bool) -> None:
    result = ask(prompt, tab=tab, **({} if viewer else {"raw_context": ""}))
    assert (result["source"], result["calls"]) == ("conversation", ["plan", "converse"]), result


@pytest.mark.parametrize("prompt", OFF_TOPIC)
def test_off_topic_questions_are_not_painting_questions(prompt: str) -> None:
    assert not is_domain_question(prompt) and not is_general_painting_knowledge(prompt)


@pytest.mark.parametrize("tab", TABS)
@pytest.mark.parametrize("prompt", CONCEPT)
@pytest.mark.parametrize("viewer", [True, False])
def test_pose_concept_questions_use_documents_on_every_tab(prompt: str, tab: str, viewer: bool) -> None:
    pipeline = DocumentRecordingPipeline()
    result = ask(prompt, tab=tab, pipeline=pipeline, **({} if viewer else {"raw_context": ""}))
    assert result["source"] == "documents" and "plan" not in pipeline.calls, result
    assert is_general_painting_knowledge(prompt)


def test_query_chat_posture_types_question_uses_documents() -> None:
    # /query-chat(query_chat.js)은 knowledge_search, 문맥 없이 보낸다.
    result = ask("자세 종류로는 어떤 게 있어?", tab="knowledge", raw_context="", pipeline=DocumentRecordingPipeline())
    assert result["source"] == "documents", result


def test_bare_pose_concept_question_on_the_pose_tab_uses_documents() -> None:
    assert ask("허리 자세 의미가 뭐야?", tab="pose", pipeline=DocumentRecordingPipeline())["source"] == "documents"


def test_painting_word_overrides_the_off_topic_lexicon() -> None:
    assert is_general_painting_knowledge("도장할 때 요가처럼 허리를 굽히는 자세의 의미는?")
