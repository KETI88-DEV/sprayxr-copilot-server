"""안전 질문의 경로와 답 검사(2026-11-14 훈련생 질문 시험 T15).

"에어리스 건 끝을 손가락으로 막아서 확인해도 되나요?"에 외국어 답 10개 중 8개가 위험한 행동을 허락했다.
7개는 검색 없는 대화 경로로 갔고, 문서 경로 답도 인용 청크에 없는 지시를 지어냈다.

- 안전 질문(원문과 한국어로 옮긴 질문 둘 다로 판정)은 대화 경로로 보내지 않고 문서 검색으로 답한다.
- 위험한 행동을 해도 되는지 묻는 질문에 답이 분명히 말리지 않거나 그 행동을 허락하면 고정 문장으로 바꾼다.
  답을 요청 언어로 옮기기 전(한국어 쪽)에서 바꾸므로 바꾼 문장이 기존 번역 경로를 그대로 지난다.
"""
from __future__ import annotations

import re
from typing import Final

CONSERVATIVE_ANSWER: Final = "안전과 관련된 작업이라 확인되지 않은 방법은 하지 마세요. 장비를 멈추고 압력을 뺀 뒤 작업 지침이나 관리자에게 확인하세요."
# 근거: DOC-doc_0219-C0084 "(라) 도료의 분출 압력이 높으므로 건 선단에 손을 내밀면 안 된다.",
#       DOC-doc_0036-C0062 "에어리스 스프레이의 고압 분사에 의하여 도료 및 시너 등이 눈, 피부 등에 침투되는 위험에 주의한다."
HIGH_PRESSURE_ANSWER: Final = ("건 끝을 손가락이나 손으로 막아서 확인하지 마세요. 에어리스 건은 분출 압력이 높아 도료와 시너가 "
                               "피부나 눈에 침투할 수 있습니다. 장비를 멈추고 압력을 뺀 뒤 작업 지침이나 관리자에게 확인하세요.")
HIGH_PRESSURE_CITATIONS: Final = ("DOC-doc_0219-C0084", "DOC-doc_0036-C0062")

_SAFETY: Final = re.compile(
    r"고압\s*(?:분사|주입|침투)|피부\s*(?:침투|주입)|피부에.{0,10}(?:묻|닿|들어|튀)|주입\s*사고|손가락|손바닥|맨손|"
    r"(?:손|몸)\s*(?:으로|을)?\s*(?:막|대|덮|만지)|눈에.{0,10}(?:들어|튀|묻)|보호구|보호\s*장비|호흡\s*보호|방독|송기|마스크|"
    r"(?:시너|용제|유기\s*용제|증기|냄새).{0,12}(?:흡입|마시|맡|들이|어지)|흡입|화재|폭발|불꽃|인화|정전기|접지|"
    r"밀폐|산소\s*결핍|질식|탱크\s*(?:안|내부)|사다리|고소\s*작업|비계|추락|떨어짐|감전|누전|전기\s*(?:충격|안전|위험)|안전|위험|"
    r"\b(?:finger|palm|bare\s+hand|injection|skin|eyes?|respirator|mask|ppe|thinner|solvent|fumes?|inhal\w*|fire|explosi\w*|"
    r"static|ground(?:ing|ed)?|confined|ladder|height|fall|electric\w*|shock|safe(?:ty)?|danger\w*)\b",
    re.I)
_TIP: Final = r"건\s*(?:끝|선단|앞|입구)|팁|노즐|분사구|토출구|\b(?:tip|nozzle|gun)\b"
_HAND: Final = r"손가락|손바닥|맨손|손으로|손을|\b(?:finger|palm|hand)s?\b"
_BLOCK: Final = r"막|대|덮|누르|block|cover|put|plug|press|hold"
_PERMISSION: Final = re.compile(
    r"해도\s*(?:되|돼|괜찮)|하면\s*(?:되|돼|안\s*되|괜찮)|괜찮(?:나요|을까|아요|은가|습니까|아\?|지)|되나요|될까요|돼요\?|돼\?|되지\?|"
    r"써도\s*(?:되|돼)|가능한가요|가능할까요|안전한가요|문제\s*없(?:나요|을까|죠)|\b(?:can|may)\s+i\b|is\s+it\s+(?:ok|okay|safe|fine)",
    re.I)
_REFUSE: Final = re.compile(r"하지\s*마|안\s*됩니다|안\s*돼요|하면\s*안|금지|위험|권장되지\s*않|삼가|피하세요|피해야|절대|말아야")
# 그 행동을 허락하는 말("네", "해도 됩니다", "확인할 수 있습니다", "일반적인 방법"). "침투할 수 있습니다"는 허락이 아니다.
_AFFIRM: Final = re.compile(
    r"^\s*(?:네|예)(?:[,.\s]|$)|해도\s*(?:됩니다|돼요|괜찮)|괜찮습니다|가능합니다|"
    r"(?:확인|점검|검사|사용|작업|칠|시험)할\s*수\s*있|막(?:을|아서|고)\s*(?:수\s*있|확인|흐름|점검)|쓸\s*수\s*있|"
    r"일반적(?:인|으로)\s*(?:점검|확인|검사)?\s*(?:방법|쓸|사용)|확인해야\s*합니다")


def is_safety_question(text: str) -> bool:
    """고압 분사·손으로 팁 막기·피부·눈·보호구·용제 흡입·화재·폭발·정전기·접지·밀폐·사다리·고소·감전 질문."""
    return bool(text and _SAFETY.search(text))


def blocks_tip_by_hand(text: str) -> bool:
    """건 끝·팁·노즐을 손(가락)으로 막는 행동을 말하는가."""
    return bool(re.search(_TIP, text, re.I) and re.search(_HAND, text, re.I) and re.search(_BLOCK, text, re.I))


def asks_permission(text: str) -> bool:
    """그 행동을 해도 되는지 묻는가(해도 되나요·괜찮나요·써도 돼·can I)."""
    return bool(_PERMISSION.search(text))


def needs_replacement(answer: str) -> bool:
    """허락을 묻는 안전 질문의 답이 그 행동을 허락하거나 분명히 말리지 않는가."""
    return bool(_AFFIRM.search(answer)) or not _REFUSE.search(answer)


def guard_answer(questions: tuple[str, ...], answer: str, citations: list[str]) -> tuple[str, list[str]]:
    """안전 허락 질문의 한국어 답을 검사한다. 바꿔야 하면 고정 문장으로, 아니면 그대로 돌려준다."""
    texts = [text for text in questions if text]
    if not any(is_safety_question(text) for text in texts) or not any(asks_permission(text) for text in texts):
        return answer, citations
    if not needs_replacement(answer):
        return answer, citations
    if any(blocks_tip_by_hand(text) for text in texts):
        return HIGH_PRESSURE_ANSWER, list(HIGH_PRESSURE_CITATIONS)
    return CONSERVATIVE_ANSWER, citations
