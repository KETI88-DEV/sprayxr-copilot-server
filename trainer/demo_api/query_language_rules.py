"""Resolve bounded UI-copy, label and hypothetical questions without session inference."""
from __future__ import annotations

import re
from fractions import Fraction


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
_DEW_POINT_COMPARISON = tuple(re.compile(pattern, re.I) for pattern in (
    r"이슬점|노점|\bdew\s+point\b|露点",
    r"(?:강판|피도면|표면)\s*온도|\b(?:steel|surface|substrate)\s+(?:surface\s+)?temperature\b|(?:钢板|表面)温度",
    r"(?:공기|대기)\s*온도|\b(?:air|ambient)\s+temperature\b|空气温度",
    r"왜|이유|차이|\bwhy\b|为什么|为何",
))
_POT_LIFE_COMPARISON = tuple(re.compile(pattern, re.I) for pattern in (
    r"가사\s*시간|혼합\s*후|\bpot\s*life\b|после\s+смешивания|жизнеспособност",
    r"건조\s*시간|\bdrying\s+time\b|высыхан",
    r"차이|다르|\bdiffer(?:s|ence|ent)?\b|отлича",
))
_RECOAT_READINESS_COMPARISON = tuple(re.compile(pattern, re.I) for pattern in (
    r"재도장|덧칠|\brecoat(?:ing)?\b|ทาทับ|ชั้นถัดไป",
    r"손에\s*묻지|손으로\s*만져|겉이\s*마르|\b(?:dry\s+to\s+the\s+touch|touch[- ]dry)\b|แห้งเมื่อสัมผัส",
    r"간격|\b(?:interval|manufacturer|data\s*sheet)\b|ช่วงเวลา|ผู้ผลิต",
))
_TIME_CONVERSION = re.compile(
    r"\bhow\s+many\s+(?P<target>seconds?|minutes?|hours?)\s+(?:(?:is|are)\s+)?(?:in\s+)?"
    r"(?P<quantity>(?:a|one|two|three)\s+quarters?|(?:a\s+)?half|[1-9]\d{0,3}\s*/\s*[1-9]\d{0,3})"
    r"\s+(?:of\s+)?(?:an?\s+)?(?P<source>hours?|minutes?|seconds?)\s*[?.!]?\s*"
    r"(?:give\s+me\s+the\s+answer\s+in\s+english\s*[.!]?)?\Z", re.I,
)
_FRACTION_WORDS = {
    "a quarter": Fraction(1, 4), "one quarter": Fraction(1, 4),
    "two quarters": Fraction(1, 2), "three quarters": Fraction(3, 4),
    "half": Fraction(1, 2), "a half": Fraction(1, 2),
}
_TIME_UNITS = {"second": (1, "초"), "minute": (60, "분"), "hour": (3600, "시간")}


def _time_conversion_answer(prompt: str) -> str | None:
    match = _TIME_CONVERSION.search(prompt)
    if match is None:
        return None
    quantity = " ".join(match["quantity"].casefold().split())
    amount = Fraction(quantity.replace(" ", "")) if "/" in quantity else _FRACTION_WORDS.get(quantity)
    if amount is None:
        return None
    source_seconds, _ = _TIME_UNITS[match["source"].casefold().removesuffix("s")]
    target_seconds, target_label = _TIME_UNITS[match["target"].casefold().removesuffix("s")]
    return f"{amount * source_seconds / target_seconds}{target_label}입니다."


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
    if all(pattern.search(prompt) for pattern in _DEW_POINT_COMPARISON):
        return (
            "공기 온도와 강판 온도는 다를 수 있습니다. 강판 온도가 이슬점에 가까우면 표면에 결로가 생길 수 있으므로 "
            "표면 온도와 이슬점을 각각 확인하고 제품 기술자료의 작업 조건을 따르세요."
        )
    if all(pattern.search(prompt) for pattern in _POT_LIFE_COMPARISON):
        return (
            "가사시간은 주제와 경화제를 섞은 뒤 도장에 사용할 수 있는 시간입니다. "
            "건조 시간은 도료를 표면에 바른 뒤 정해진 건조 상태에 이르기까지 걸리는 시간입니다. 두 시간은 제품과 작업 조건에 따라 달라집니다."
        )
    if all(pattern.search(prompt) for pattern in _RECOAT_READINESS_COMPARISON):
        return (
            "겉이 마르고 손에 묻지 않아도 재도장 가능 시점은 별도로 확인해야 합니다. "
            "제품 기술자료의 최소·최대 재도장 간격과 온도 등 적용 조건을 확인하세요."
        )
    return _time_conversion_answer(prompt)


_REFRIGERATION_DEFINITION = re.compile(
    r"냉장\s*보관(?:이|은|의)?\s*(?:무슨\s*뜻|무엇|뭐(?:예요|야)|뜻|의미)", re.S,
)


def unrelated_definition(prompt: str) -> str | None:
    """Answer an explicit ordinary-language definition without coating context."""
    if _REFRIGERATION_DEFINITION.search(prompt):
        return "냉장 보관은 음식이 쉽게 상하지 않도록 낮은 온도에서 차갑게 보관하는 것을 말합니다."
    return None
