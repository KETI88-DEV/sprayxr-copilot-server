"""안전 질문의 경로와 답 검사(2026-11-14 훈련생 질문 시험 T15, v3.4.2 독립 검토).

"에어리스 건 끝을 손가락으로 막아서 확인해도 되나요?"에 외국어 답 10개 중 8개가 위험한 행동을 허락했다.
7개는 검색 없는 대화 경로로 갔고, 문서 경로 답도 인용 청크에 없는 지시를 지어냈다.

두 단계로 본다.
- 안전 주제(`is_safety_topic`): 고압 분사·피부 침투·보호구·용제 흡입·화재·폭발·정전기·접지·밀폐·사다리·고소·감전.
  지식 질문(knowledge_search)이면 대화 경로로 보내지 않고 문서 검색으로 답한다. 뷰어 탭 경로는 바꾸지 않는다.
- 위험 행동(`is_hazard_act`): 사람에게 위험한 그 행동을 해도 되는지 묻는 질문. 문서 검색으로 답하고 답을 검사한다.
  손으로 건 끝을 막는 경우는 모델 답과 무관하게 고정 답, 그 밖은 분명히 말리고 허락하는 말이 없을 때만 모델 답을 둔다.
  답을 요청 언어로 옮기기 전(한국어 쪽)에서 바꾸므로 바꾼 문장도 기존 번역 경로를 그대로 지난다.
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


def _en(words: str) -> str:
    """영어 낱말. 한글 조사가 바로 붙어도("finger로") 잡는다 — \\b는 한글을 낱말 글자로 본다."""
    return rf"(?<![A-Za-z])(?:{words})(?![A-Za-z])"


# 안전 주제. "안전·위험·마스크·떨어짐·height·fall" 같은 낱말 하나로는 고르지 않는다(검토: "도막 떨어짐", "score fall").
_TOPIC: Final = re.compile(
    r"고압\s*(?:분사|주입|침투)|피부\s*(?:침투|주입)|주입\s*사고|(?:눈|피부)에.{0,10}(?:들어|튀|묻|닿)|보호구|호흡\s*보호|방독|송기\s*마스크|"
    r"(?:시너|용제|유기\s*용제|증기).{0,12}(?:흡입|마시|맡|들이)|흡입\s*(?:사고|중독)|중독|화재|폭발|인화|정전기|접지|"
    r"밀폐\s*(?:공간|구역|된)|산소\s*결핍|질식|사다리|고소\s*작업|추락|감전|누전|"
    + _en(r"respirator|ppe|explosion|explosive|static\s+electricity|grounding|confined\s+space|ladder|electric\s+shock|injection\s+injury"),
    re.I)
_TIP: Final = r"건\s*(?:끝|선단|앞|입구)|팁|노즐|분사구|토출구|" + _en(r"tip|nozzle|gun")
_HAND: Final = r"손가락|손구락|손끝|엄지|검지|손바닥|맨손|손으로|손을|장갑|" + _en(r"fingers?|thumb|palm|hands?|gloves?")
_HAND_BLOCK: Final = r"막|덮|누르|눌러|눌러서|" + _en(r"block|cover|plug|press")
_TIP_TOUCH: Final = _HAND_BLOCK + r"|대(?:도|면|고|서)|" + _en(r"put|hold|touch")
# 사람에게 위험한 행동 그 자체(위험 행동 질문의 대상).
_HAZARD_ACT: Final = re.compile(
    r"고압.{0,15}(?:손|몸|피부).{0,6}(?:만지|만져|대|닿)|(?:손|몸|피부).{0,6}(?:만지|만져|대|닿).{0,15}고압|"
    r"(?:마스크|보호구|방독|송기|장갑|보안경|호흡\s*보호).{0,6}(?:안\s*(?:쓰|끼|착용)|없이|벗|빼고)|"
    r"(?:시너|용제|희석제).{0,8}(?:손|피부|얼굴|몸).{0,6}(?:씻|닦|묻|바르)|(?:손|피부|얼굴|몸).{0,6}(?:시너|용제)로|"
    r"담배|흡연|라이터|성냥|불\s*(?:붙|피우)|용접|화기|"
    r"압력.{0,6}(?:안\s*빼|빼지\s*않|안\s*낮추|그대로).{0,15}(?:팁|노즐|건|호스|필터|분해|교체|갈|청소)|"
    r"(?:사람|몸|얼굴|동료|눈).{0,4}(?:쪽|에게|에|을)?\s*(?:으로|로)?\s*(?:향|겨누|쏘|뿌)|"
    r"(?:안전\s*장치|방아쇠\s*잠금).{0,6}(?:풀|빼|없이|안\s*걸)|밀폐.{0,12}(?:혼자|환기\s*없이|마스크\s*없이)|"
    r"사다리.{0,12}(?:혼자|한\s*손|뛰|올라서)",
    re.I)
_PERMISSION: Final = re.compile(
    r"(?:해|하|막아|막아봐|대|써|칠해|갈아|피워|씻어|만져봐|향해)도\s*(?:되|돼|괜찮|됨)|하면\s*(?:되|돼|안\s*되|괜찮)|"
    r"(?:막|대|하|쓰)면\s*안\s*되나|괜찮(?:나요|을까|아요|은가|습니까|겠|아\?|지)|되나요|되나여|될까요|되겠|돼요\?|돼여|돼\?|되지\?|됨\?|"
    r"가능한가요|가능할까요|안전한가요|문제\s*없(?:나요|을까|죠)|"
    + _en(r"can\s+i|may\s+i|is\s+it\s+(?:ok|okay|safe|fine)"),
    re.I)
_FOLLOW_UP: Final = re.compile(r"^\s*(?:그럼|그러면|그래도|그냥)?\s*(?:그거|그렇게|이렇게)?\s*(?:해도|하면)\s*(?:돼|되|괜찮)", re.I)

# 분명히 말리는 말. "위험"만으로는 말린 것이 아니다(검토: "위험하니 주의하세요. … 막아 확인하면 됩니다.").
_REFUSE: Final = re.compile(r"하지\s*마|안\s*됩니다|안\s*돼요|안\s*돼(?:[.!,]|$)|금지|해서는\s*안|하면\s*안|삼가|말아야|^\s*아니요")
# 그 행동을 허락하는 말. "안 됩니다"의 "됩니다"는 허락이 아니다.
_AFFIRM: Final = re.compile(
    r"(?:해|막아|써|칠해|대)도\s*(?:됩니다|돼요|괜찮)|(?<!안 )(?<!안)됩니다|괜찮(?:습니다|아요)|하면\s*됩니다|위험하지\s*않|"
    r"문제\s*(?:없|가\s*없)|안전합니다|(?:확인|점검|검사|사용|작업|칠|시험)할\s*수\s*있|막(?:을|아서)\s*수\s*있|"
    r"(?:막아|만져)\s*(?:보세요|확인하세요)|막아서\s*확인하세요|^\s*(?:네|예)(?:[,.\s]|$)|규정은\s*없|"
    r"일반적(?:인|으로)\s*(?:점검|확인|검사)?\s*(?:방법|쓸|사용)|확인해야\s*합니다")


def is_safety_topic(text: str) -> bool:
    """고압 분사·피부 침투·보호구·용제 흡입·화재·폭발·정전기·접지·밀폐·사다리·고소·감전 질문."""
    return bool(text and _TOPIC.search(text))


def blocks_tip_by_hand(text: str) -> bool:
    """건 끝·팁·노즐을 손(가락)·장갑으로 막거나 대는 행동, 또는 손으로 무엇을 막는 행동."""
    if not text:
        return False
    tip, hand = re.search(_TIP, text, re.I), re.search(_HAND, text, re.I)
    return bool((tip and hand and re.search(_TIP_TOUCH, text, re.I))
                or (hand and re.search(_HAND_BLOCK, text, re.I) and not re.search(r"손목|손잡이|잡고", text)))


def asks_permission(text: str) -> bool:
    """그 행동을 해도 되는지 묻는가(해도 되나요·괜찮겠지·막아봐도 됨?·can I)."""
    return bool(text and _PERMISSION.search(text))


def is_hazard_act(text: str, *, previous: str = "") -> bool:
    """사람에게 위험한 행동을 해도 되는지 묻는가. "그럼 해도 돼?"는 같은 대화의 앞 질문으로 본다."""
    if not text:
        return False
    if _FOLLOW_UP.search(text) and previous and previous != text:
        return is_hazard_act(previous)
    return asks_permission(text) and (blocks_tip_by_hand(text) or bool(_HAZARD_ACT.search(text)))


# 이전 판의 이름. 위험 행동 판정이 경로를 정한다.
def is_safety_question(text: str) -> bool:
    return is_safety_topic(text) or is_hazard_act(text)


def needs_replacement(answer: str) -> bool:
    """분명히 말리고(하지 마세요·안 됩니다·금지…) 허락하는 말이 하나도 없을 때만 모델 답을 둔다."""
    return not (_REFUSE.search(answer) and not _AFFIRM.search(answer))


def guard_answer(questions: tuple[str, ...], answer: str, citations: list[str], *, previous: str = "") -> tuple[str, list[str]]:
    """위험 행동 질문의 한국어 답을 검사한다. 위험 행동 질문이 아니면 그대로 돌려준다."""
    texts = [text for text in questions if text]
    if not any(is_hazard_act(text, previous=previous) for text in texts):
        return answer, citations
    acts = [text for text in texts if blocks_tip_by_hand(text)] or ([previous] if previous and blocks_tip_by_hand(previous) else [])
    if acts:
        # 손으로 건 끝을 막는 행동은 모델 답이 무엇이든 고정 답이다(검토 2026-11-14: "막아도 됩니다. 다만 위험할 수 …").
        return HIGH_PRESSURE_ANSWER, list(HIGH_PRESSURE_CITATIONS)
    if needs_replacement(answer):
        return CONSERVATIVE_ANSWER, citations
    return answer, citations
