"""Resolve bounded UI-copy, label and hypothetical questions without session inference."""
from __future__ import annotations

import re


_ZONE_COMPLETION_COPY = re.compile(
    r'''["'“‘`]\s*(?:Zone\s*(?P<english>\d+)|(?P<korean>\d+)\s*번\s*구역)(?:을|를)?\s*'''
    r'''(?:작업(?:을|를)?\s*)?완료\s*했(?:습니다|어요|다)[.!]?["'”’`]''',
    re.I,
)
_COPY_REQUEST = re.compile(
    r"(?:문구|문장|라벨|텍스트|철자|맞춤법|표현).*(?:다듬어|고쳐|바꿔|교정)", re.S,
)
_EN_NOT_SPRAYED = re.compile(
    r"\b(?:what\s+does|what\s+is|meaning\s+of)\s+['\"‘“]?"
    r"(?:not\s+yet\s+sprayed|not\s+sprayed|uncoated)\b",
    re.I | re.S,
)
_KO_NOT_SPRAYED = re.compile(r"미\s*도장.{0,28}(?:뜻|의미|뭐|무엇|어떤\s*상태)", re.S)
_CHART_CONTEXT = re.compile(r"\b(?:chart|legend|heat\s*map)\b|히트맵|범례|차트|도막\s*분포", re.I)
_VI_UNMEASURED = re.compile(
    r"\bnếu\b.{0,28}\bchưa\s+phun\b.{0,64}\b(?:biết|đo|xác\s+định)\b.{0,28}"
    r"\b(?:độ\s+dày|dày\s+bao\s+nhiêu)\b", re.I | re.S,
)
_EN_UNMEASURED = re.compile(
    r"\b(?:if|before)\b.{0,32}\b(?:not\s+spray|not\s+sprayed|haven['’]t\s+sprayed|spraying|"
    r"no\s+paint\s+has\s+been\s+applied)\b.{0,48}\b(?:measure|know|estimate|determine|tell|check)\b"
    r".{0,52}\b(?:thickness|how\s+thick)\b", re.I | re.S,
)
_KO_UNMEASURED = re.compile(
    r"(?:아직\s*)?(?:분사|도포|도장)(?:를|을)?\s*(?:안\s*했|하지\s*않았|하기\s*전)"
    r".{0,48}(?:두께|도막).{0,30}(?:측정|알\s*수|알고|아나요)", re.S,
)
_SETPOINT = re.compile(r"(?:목표|설정)\s*(?:도막\s*)?두께|(?:target|setpoint)\s*(?:film\s*)?thickness|độ\s+dày.{0,24}mục\s+tiêu", re.I)


def narrow_language_answer(prompt: str) -> str | None:
    """Return Korean source prose; the Query wrapper localizes the final answer."""
    copy = _ZONE_COMPLETION_COPY.search(prompt)
    if copy and _COPY_REQUEST.search(prompt[copy.end():]):
        number = copy.group("english") or copy.group("korean")
        return f"‘{number}번 구역 작업 완료’로 쓰면 자연스럽습니다."
    if _CHART_CONTEXT.search(prompt) and (_EN_NOT_SPRAYED.search(prompt) or _KO_NOT_SPRAYED.search(prompt)):
        return "도막 차트의 ‘미도장’은 해당 영역에 도료가 도포되지 않은 상태를 뜻합니다."
    if not _SETPOINT.search(prompt) and any(
        pattern.search(prompt) for pattern in (_VI_UNMEASURED, _EN_UNMEASURED, _KO_UNMEASURED)
    ):
        return "아직 분사하지 않았다면 실제 도막 두께를 측정할 수 없습니다. 목표 두께는 실측값과 별개의 설정값입니다."
    return None


_REFRIGERATION_DEFINITION = re.compile(
    r"냉장\s*보관(?:이|은|의)?\s*(?:무슨\s*뜻|무엇|뭐(?:예요|야)|뜻|의미)", re.S,
)


def unrelated_definition(prompt: str) -> str | None:
    """Answer an explicit ordinary-language definition without coating context."""
    if _REFRIGERATION_DEFINITION.search(prompt):
        return "냉장 보관은 음식이 쉽게 상하지 않도록 낮은 온도에서 차갑게 보관하는 것을 말합니다."
    return None
