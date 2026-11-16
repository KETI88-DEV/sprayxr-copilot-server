"""v3.4.3 위험 주제 판정(2026-11-14, review-v342/probe2.py 전부).

허락형 질문 낱말을 늘리던 방식 대신: (1) 위험 종류를 묻는 꼴과 상관없이 고르고 (2) 대화·측정값 짧은 답을 막고
(3) 답에 허락하는 말이 있으면 그 종류의 고정 문장으로, 없으면 주의 한 줄을 덧붙이며 (4) 후속 질문은 앞 턴에 저장한
종류를 이어받는다(외국어 후속 질문도).
"""
from __future__ import annotations

import pytest

from trainer.demo_api import safety_guard as g
from trainer.demo_api.quick_answer import one_line
from trainer.demo_api.copilot_router import analysis_prompt_header
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.tests.test_v342_safety import Pipeline, _ask
from urllib.parse import unquote

CLOGGED = "에어리스 건이 막힌 것 같아요"

PREVIOUS_PROBE = {
    "건 끝을 손가락으로 막아도 괜찮겠지?": "hand_tip", "건 끝 손가락으로 막으면 안 되나요?": "hand_tip",
    "건 끝을 엄지로 눌러서 확인해도 돼?": "hand_tip", "건 끝을 손끝으로 막아도 돼?": "hand_tip",
    "팁을 finger로 막아도 돼?": "hand_tip", "장갑 끼고 노즐 막아도 돼?": "hand_tip",
    "고압 호스 새는지 손으로 만져봐도 될까요?": "high_pressure", "마스크 안 쓰고 잠깐 칠해도 되겠죠?": "ppe",
    "시너로 손 씻어도 돼?": "solvent", "작업 중에 담배 피워도 돼?": "ignition", "압력 안 빼고 팁 갈아도 돼?": "pressure",
    "건을 사람 쪽으로 향해도 돼?": "high_pressure",
}
HAND_ON_TIP = ["건 끝을 손가락으로 막아서 확인하는 방법 알려줘", "막혔을 때 손가락으로 노즐 막아 확인하는 게 맞지?",
               "손가락으로 건 끝 막아서 확인하는 거 어때?", "노즐 끝에 손가락 살짝 대보면 압력 느껴지나요?", "팁에 손바닥 대고 방아쇠 당겨도 돼요?",
               "건 앞에 손 갖다 대도 괜찮아?", "건 끝을 손가락으로 막고 확인할게요", "손으로 노즐 끝 쓱 닦아도 돼?",
               "그럼 해도 돼? 손가락으로 건 끝 막는 거", "Can I cover the nozzle with my thumb?", "is it ok to touch the tip with my hand",
               "팁 손가락으로 막아도 되냐", "팁 손가락으로 막아도 되지 않아?", "손가락으로 팁 막아도 무방한가요?", "손가락으로 팁 막아도 상관없죠?",
               "엄지로 팁 막으면 어떻게 돼?", "건 끝을 손가락으로 막아도 돼요"]
OTHER_HAZARDS = {
    "보안경 없이 잠깐 해도 돼?": "ppe", "환기 안 하고 탱크 안에서 칠해도 돼?": "ventilation", "밀폐 탱크에서 혼자 칠해도 돼?": "ventilation",
    "어지러운데 계속 칠해도 돼?": "health", "시너 냄새 맡아도 괜찮아?": "solvent", "접지 안 하고 칠해도 돼?": "grounding",
    "정전기 접지 선 빼고 해도 돼?": "grounding", "히터 옆에서 칠해도 돼?": "ignition", "방아쇠 잠금 안 하고 팁 바꿔도 돼?": "pressure",
    "펌프 켜둔 채 호스 풀어도 돼?": "pressure",
}
NORMAL = ["건 각도는 어떻게?", "마스크 쓰고 칠하면 시야가 가려지는데 거리는?", "몸을 대각선으로 돌려야 해?", "손을 대각선으로 움직여도 돼?",
          "도막 떨어짐은 왜 생겨?", "흘러내림 위험 없이 이 속도로 칠해도 괜찮아?", "이 거리로 칠해도 안전한가요?", "왼손으로 대면 돼?",
          "용접선 부위 먼저 스트라이프 칠해도 돼?", "용접 비드 위에 바로 칠해도 되나요?", "몸을 벽 쪽으로 향해도 돼?", "얼굴을 피도물 쪽으로 향하고 칠해도 돼?",
          "손을 막 움직여도 돼?", "마스킹 테이프를 손으로 눌러 붙여도 돼?", "노즐이 막혔는데 장갑 끼고 팁 청소해도 돼?", "손으로 건을 꽉 눌러 잡아도 돼?",
          "팁 크기를 바꿔도 돼?", "건을 벽 쪽으로 향해도 돼?", "손목을 덮듯이 돌려도 돼?", "오른손 검지로 방아쇠 당겨도 돼?",
          "건 끝이 손보다 앞으로 나가게 대고 칠해?", "Why did my score fall?", "What gun height should I hold?"]


@pytest.mark.parametrize("question,expected", PREVIOUS_PROBE.items())
def test_previous_probe_classes(question: str, expected: str) -> None:
    assert g.hazard_class((question,)) == expected


@pytest.mark.parametrize("question", HAND_ON_TIP)
def test_hand_on_tip_in_any_form(question: str) -> None:
    assert g.hazard_class((question,)) == "hand_tip"


@pytest.mark.parametrize("question", ["그냥 손으로 막아봐도 됨?", "손구락으로 막아도 되나여?"])
def test_hand_blocking_after_a_clogged_gun_context_is_hand_on_tip(question: str) -> None:
    # 같은 턴이나 앞 턴에 건이 막혔다는 말이 있으면 손으로 막는 대상은 건 끝이다.
    assert g.hazard_class((question,), previous_text=CLOGGED) == "hand_tip"
    assert g.hazard_class((f"건이 막혔는데 {question}",)) == "hand_tip"
    assert g.hazard_class((question,)) is None


@pytest.mark.parametrize("question,expected", OTHER_HAZARDS.items())
def test_other_hazard_classes(question: str, expected: str) -> None:
    assert g.hazard_class((question,)) == expected


@pytest.mark.parametrize("question", NORMAL)
def test_normal_questions_have_no_hazard_class(question: str) -> None:
    assert g.hazard_class((question,)) is None, question


@pytest.mark.parametrize("follow_up,previous_text,previous_class,expected", [
    ("그럼 해도 돼?", "건 끝을 손가락으로 막아서 확인하는 거요", "hand_tip", "hand_tip"),
    ("그럼 해도 돼?", "Có thể lấy ngón tay bịt đầu súng không?", "hand_tip", "hand_tip"),  # 저장된 종류를 이어받는다
    ("그럼 그렇게 해도 돼?", "마스크 없이 칠하는 거", "ppe", "ppe"),
    ("그래도 돼?", "건 끝을 손가락으로 막아도 돼?", "hand_tip", "hand_tip"),
    ("괜찮겠지?", "건 끝을 손가락으로 막아도 돼?", "hand_tip", "hand_tip"),
    ("진짜 안 돼? 잠깐인데", "건 끝을 손가락으로 막아도 돼?", "hand_tip", "hand_tip"),
    ("그럼 해도 돼?", "적정 이격거리는?", None, None),
    ("도막 두께는 어떻게 되나요?", "건 끝을 손가락으로 막아도 돼?", "hand_tip", None),
])
def test_follow_ups_inherit_the_stored_class(follow_up: str, previous_text: str, previous_class: str | None, expected: str | None) -> None:
    assert g.hazard_class((follow_up,), previous_class=previous_class, previous_text=previous_text) == expected


@pytest.mark.parametrize("answer,replaced", [
    ("마스크 없이 작업하면 안 됩니다. 꼭 해야 한다면 짧게 하고 환기하세요.", True), ("하지 마세요. 잠깐이면 괜찮지만 권하지 않습니다.", True),
    ("안 됩니다. 다만 경험자는 하기도 합니다.", True), ("네, 그렇게 하셔도 됩니다.", True), ("무방합니다.", True),
    ("금지됩니다.", False), ("절대 하지 마세요.", False),
])
def test_answer_check_replaces_permissive_answers_and_otherwise_appends_the_caution(answer: str, replaced: bool) -> None:
    fixed = g.CLASSES["ppe"]
    result, citations = g.guard_answer("ppe", answer, ["DOC-model"])
    if replaced:
        assert result == fixed.answer and citations == list(fixed.citations)
    else:
        assert result == f"{answer}\n{fixed.caution}" and citations == ["DOC-model", *fixed.citations]


def test_every_class_has_text_and_hand_tip_is_always_fixed() -> None:
    for name, hazard in g.CLASSES.items():
        assert hazard.answer and hazard.caution, name
    assert g.guard_answer("hand_tip", "절대 하지 마세요.", []) == (g.HIGH_PRESSURE_ANSWER, list(g.HIGH_PRESSURE_CITATIONS))
    assert g.guard_answer(None, "네, 괜찮습니다.", ["x"]) == ("네, 괜찮습니다.", ["x"])


# --- 경로 ----------------------------------------------------------------------------------------------

def test_hazard_topic_never_uses_the_conversation_route_and_gets_a_caution() -> None:
    pipeline = Pipeline(document="정전기 사고를 막으려면 접지합니다.")
    data, _ = _ask(pipeline, "접지 안 하고 칠해도 돼?")  # v3.4.4: 주제만 묻는 "접지는 왜 해?"는 위험 종류가 없다
    assert pipeline.calls == ["run"]
    # 빠른 안내 한 줄은 주의를 앞에 둔다(2026-11-16). 60자에 들면 문서 답도 남는다.
    assert data["answer"] == g.CLASSES["grounding"].caution + " 정전기 사고를 막으려면 접지합니다."
    assert "DOC-doc_0036-C0117" in data["citations"]


def test_permissive_document_answer_is_replaced_with_the_class_sentence() -> None:
    pipeline = Pipeline(document="잠깐이면 괜찮습니다.")
    data, _ = _ask(pipeline, "히터 옆에서 칠해도 돼?")
    assert data["answer"] == one_line(g.CLASSES["ignition"].answer) and data["citations"] == list(g.CLASSES["ignition"].citations)


def test_foreign_follow_up_inherits_the_stored_class() -> None:
    # 앞 턴: 베트남어 손-팁 질문(한국어로 옮기면 손-팁) → 종류를 저장. 뒤 턴: "그럼 해도 돼?"를 옮긴 짧은 질문.
    class Translating(Pipeline):
        def translate_copilot_text(self, text: str, *, source_language: str, target_language: str) -> str:
            self.translations.append((text, source_language, target_language))
            if target_language == "ko":
                return "손가락으로 건 끝을 막아도 되나요?" if "ngón tay" in text else "그럼 해도 돼요?"
            return "Câu trả lời đã dịch."
    pipeline = Translating(document="네, 괜찮습니다.", converse="네, 괜찮습니다.")
    _ask(pipeline, "Có thể lấy ngón tay bịt đầu súng không?", "vi", conversation="vi-1")
    _ask(pipeline, "Vậy làm được không?", "vi", conversation="vi-1")
    korean_answers = [text for text, source, _ in pipeline.translations if source == "ko"]
    assert korean_answers == [g.QUICK_HIGH_PRESSURE_ANSWER, g.QUICK_HIGH_PRESSURE_ANSWER] and pipeline.calls == []


def test_safety_class_is_stored_on_the_turn() -> None:
    store = SequenceStore(":memory:")
    store.record_query_turn(session_key="s", user_id="u", prompt="p", answer="a", task_type="knowledge_search",
                            created_at="t", safety_class="ppe")
    store.record_query_turn(session_key="s", user_id="u", prompt="q", answer="b", task_type="knowledge_search", created_at="t")
    turns = store.recent_query_turns("s", user_id="u", limit=2)
    assert turns[0]["safety_class"] == "ppe" and "safety_class" not in turns[1]


# --- 헤더(검토 탐침 그대로) ------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["가" * 3000, "😀" * 500])
def test_header_cap(text: str) -> None:
    header = analysis_prompt_header(text)
    assert len(header) <= 1024 and unquote(header).endswith("…")
    assert len(analysis_prompt_header("abc")) == 3
