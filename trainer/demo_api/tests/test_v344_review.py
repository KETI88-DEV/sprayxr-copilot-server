"""v3.4.4(48ee248) 검토(2026-11-14, review-v342/probe4.py): 정상 질문 가로채기와 근거 없는 고정 문장.

위험 종류는 주제 낱말만으로 고르지 않고 위험한 조건·행동이 함께 있어야 한다(밀폐 + 환기 없이·혼자, 시너 + 흡입·피부,
압력 + 만지기·분해, 불꽃 + 근처, 어지러움 + 용제·계속 작업). 앞 턴은 "건이 막혔다"는 말만 손-팁 맥락으로 본다.
"""
from __future__ import annotations

import re

import pytest

from trainer.demo_api import safety_guard as g


def _class(question: str, previous: str = "") -> str | None:
    """검토 탐침과 같은 호출(앞 질문의 종류를 앞 턴 저장값으로 본다)."""
    return g.hazard_class((question,), previous_class=g.classify(previous) if previous else None, previous_text=previous)


HAZARDS = ["건 끝을 손가락으로 막아도 괜찮겠지?", "건 끝을 엄지로 눌러서 확인해도 돼?", "팁을 finger로 막아도 돼?", "장갑 끼고 노즐 막아도 돼?",
           "고압 호스 새는지 손으로 만져봐도 될까요?", "마스크 안 쓰고 잠깐 칠해도 되겠죠?", "시너로 손 씻어도 돼?", "작업 중에 담배 피워도 돼?",
           "압력 안 빼고 팁 갈아도 돼?", "건을 사람 쪽으로 향해도 돼?", "건 끝을 손가락으로 막아서 확인하는 방법 알려줘",
           "손가락으로 팁 막아도 무방한가요?", "건 앞에 손 갖다 대도 괜찮아?", "손으로 노즐 끝 쓱 닦아도 돼?", "스프레이 건의 끝을 손가락으로 막아도 돼?",
           "총구를 손가락으로 막아도 되나요?", "분무구에 손가락 대도 돼?", "건 머리 부분을 손으로 막아봐도 돼?", "건의 앞쪽을 손바닥으로 막아도 돼?",
           "손가락을 노즐에 붙여도 돼?", "Can I block the spray gun with my finger?", "보안경 없이 잠깐 해도 돼?", "환기 안 하고 탱크 안에서 칠해도 돼?",
           "어지러운데 계속 칠해도 돼?", "시너 냄새 맡아도 괜찮아?", "접지 안 하고 칠해도 돼?", "히터 옆에서 칠해도 돼?",
           "방아쇠 잠금 안 하고 팁 바꿔도 돼?", "펌프 켜둔 채 호스 풀어도 돼?", "건 입구를 엄지로 틀어막고 방아쇠 당겨도 돼?",
           "노즐 구멍 막힌 거 손톱으로 파내도 돼?", "팁 막힘을 손바닥으로 툭 쳐서 뚫어도 돼?", "건 끝에 손 대면 어떻게 돼?", "팁 끝 손으로 막아도 되냐",
           "에어리스 끝단을 손가락으로 막아도 돼?", "건 노즐에 살 닿아도 괜찮아?", "도료가 나오는 데 손 넣어도 돼?", "물 묻은 손으로 펌프 전원 만져도 돼?",
           "호스 터진 데 손으로 잡아도 돼?", "건을 친구한테 쏴도 돼?", "희석제로 얼굴 닦아도 돼?", "토할 것 같은데 계속해도 돼?", "고글 없이 칠해도 돼?",
           "pump 켜놓고 tip 바꿔도 돼?", "Can I touch the spray tip while it's pressurized?", "Is it ok to paint without a respirator?",
           "Can I clean the nozzle with my finger while spraying?"]
NORMAL = ["노즐 팁 크기는 뭘 써?", "손 위치 확인하는 팁 있어?", "오른손으로 건 잡고 노즐 각도 확인하려면?", "팁 각도 확인했는데 손이 너무 높아?",
          "손이 노즐보다 앞에 있어야 해?", "장갑 끼고 팁 교체하는 순서 알려줘", "막힌 노즐 팁은 어떻게 청소해?", "노즐이 막혔는데 장갑 끼고 팁 청소해도 돼?",
          "손으로 건을 꽉 눌러 잡아도 돼?", "손을 막 움직여도 돼?", "마스킹 테이프를 손으로 눌러 붙여도 돼?", "건 끝이 손보다 앞으로 나가게 대고 칠해?",
          "Any tips to check my hand position?", "What hand position should I hold for the tip angle?", "왼손으로 호스 잡고 오른손으로 노즐 방향 확인해도 돼?",
          "건 각도는 어떻게?", "마스크 쓰고 칠하면 시야가 가려지는데 거리는?", "몸을 대각선으로 돌려야 해?", "도막 떨어짐은 왜 생겨?",
          "흘러내림 위험 없이 이 속도로 칠해도 괜찮아?", "용접선 부위 먼저 스트라이프 칠해도 돼?", "몸을 벽 쪽으로 향해도 돼?",
          "좁은 구역은 어떻게 칠해?", "탱크 내부 도장 순서는?", "압력 그대로 두고 거리만 바꾸면 돼?", "펌프 압력을 그대로 두고 팁만 큰 걸로 바꿔?",
          "그라인더로 표면 처리한 뒤 바로 칠해도 돼?", "How much thinner should I add?", "시너 희석 비율은?", "fire retardant paint 두께는?",
          "노즐을 벽면에 수직으로 대고 손은 어디에 둬?", "팁을 벽에 대고 손목을 돌려?", "손 끝으로 건 끝 방향을 가리키듯 칠해?",
          "검지 손가락으로 노즐 방향 맞춰도 돼?", "엄지로 팁 방향 돌려 끼워도 돼?", "팁 가드 손으로 돌려 맞춰도 돼?", "팁을 손으로 돌려서 방향 바꿔도 돼?",
          "스프레이 패턴 보려고 손바닥만 한 판에 쏴봐도 돼?", "환기 팬 위치는 어디가 좋아?", "도막이 얇은데 압력 남은 거 써도 돼?", "숨 고르고 다시 칠해도 돼?",
          "Where should my fingers be on the spray gun?", "VR 고글 쓰면 어지러운데 거리감 맞추는 법?", "높은 곳 칠할 때 건 각도는?",
          "VR로 연습하다가 어지러우면 몇 분 쉬어야 해요?"]
TIP = "건 끝을 손가락으로 막아도 돼?"


@pytest.mark.parametrize("question", HAZARDS)
def test_hazard_questions_get_a_class(question: str) -> None:
    assert _class(question) is not None, question


@pytest.mark.parametrize("question", NORMAL)
def test_normal_questions_have_no_class(question: str) -> None:
    assert _class(question) is None, question


@pytest.mark.parametrize("question", ["그냥 손으로 막아봐도 됨?", "손구락으로 막아도 되나여?"])
def test_clogged_context_still_makes_hand_blocking_a_hand_on_tip(question: str) -> None:
    assert _class(question, "에어리스 건이 막힌 것 같아요") == "hand_tip"


@pytest.mark.parametrize("question,previous", [
    ("손을 막 움직여도 돼?", "노즐 팁 크기는 뭘 써?"), ("손으로 누르면서 칠해?", "팁 각도는?"),
    ("손바닥으로 덮듯이 칠해?", "노즐이 막혔을 때 어떻게 해?"),
])
def test_tip_nouns_in_the_previous_turn_are_not_a_clogged_context(question: str, previous: str) -> None:
    assert _class(question, previous) is None


@pytest.mark.parametrize("question", ["그래도 돼?", "그냥 하면 안 돼?", "잠깐이면 괜찮지?", "왜 안 되는데?", "그럼 장갑 끼면?", "살짝만 하면?", "1초만?", "진짜?"])
def test_short_follow_ups_keep_the_hand_on_tip_class(question: str) -> None:
    assert _class(question, TIP) == "hand_tip"


@pytest.mark.parametrize("question", ["다음 구역은 어떻게 돼?", "도막 두께는 어떻게 되나요?", "적정 이격거리는?", "그럼 이제 뭐 해?",
                                      "그럼 지금 자세는?", "그럼 왼쪽 팔은?"])
def test_new_questions_after_hand_on_tip_are_not_follow_ups(question: str) -> None:
    assert _class(question, TIP) is None


def test_follow_ups_from_other_turns() -> None:
    assert _class("그럼 해도 돼?", "Có thể lấy ngón tay bịt đầu súng không?") == "hand_tip"
    assert _class("그럼 해도 돼?", "마스크 없이 칠하는 거") == "ppe"


# --- 고정 문장은 인용한 청크에 있는 말만 ---------------------------------------------------------------

def test_fixed_sentences_fit_their_class_and_cited_chunks() -> None:
    assert "DOC-doc_0032-C0456" in g.CLASSES["ppe"].citations  # "보안경 및 방독 마스크 … 착용"
    assert "관리자" not in g.CLASSES["health"].answer and "관리자" not in g.CLASSES["health"].caution
    electric = g.CLASSES["electric"]
    assert "젖은 손" in electric.answer and electric.citations == ("DOC-doc_0023-C0602", "DOC-doc_0023-C0615")
    for name in ("height", "electric"):
        assert "압력을 뺀" not in g.CLASSES[name].answer, name
    primary_sources = {
        "injection_injury": "https://www.graco.com/content/dam/graco/com_documents/FCD/FDC0688/FDC0688EN.pdf",
        "cutting_wheel_side": "https://www.nortonabrasives.com/en-us/resources/expertise/safety-cutting-vs-grinding",
        "grinding_wheel_speed": "https://obis.osha.gov/SLTC/etools/shipyard/standard/tools/index_tl.html",
    }
    assert set(primary_sources) <= g.CLASSES.keys()
    for name, hazard in g.CLASSES.items():
        if name in primary_sources:
            assert hazard.source_urls == (primary_sources[name],), name
            assert not hazard.citations, name
            assert g.fixed_safety_answer(name) == (hazard.answer, [])
        else:
            assert hazard.citations and not hazard.source_urls, name
            assert all(re.fullmatch(r"DOC-doc_\d{4}-C\d{4}", citation) for citation in hazard.citations), name
