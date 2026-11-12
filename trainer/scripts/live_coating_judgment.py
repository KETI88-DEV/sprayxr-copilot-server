"""내부 채팅이 질문에 적힌 DFT 측정값을 모델 없이 dft_judge(PR #8 기준)로 판정한다.

2026-11-12 카나리: 27B 가 도구를 부르지 않고 fast 재답이 RAG 문서의 옛 90% 규칙을 옮겼다.
측정값을 결정적으로 뽑을 수 있으면 판정·평균·6구간은 도구 결과로만 쓰고, 원인은 지어내지 않는다.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Final, Protocol

from trainer.demo_api import coating_assessment as ca

logger = logging.getLogger(__name__)

_NUMBER: Final = r"\d+(?:\.\d+)?"
_UNIT: Final = r"\s*(?:μm|um|마이크로(?:미터)?|마이크론)"  # NFKC 뒤라 µ(U+00B5)·㎛(U+33DB)는 μm 다
_VALUE: Final = rf"{_NUMBER}(?:{_UNIT})?"
_SEP: Final = r"\s*[,、]\s*"  # NFKC 뒤라 전각 쉼표는 , 다
_PARTICLE: Final = r"(?:가|이|은|는|의|에서|:|=)?"
_MARKER: Final = rf"(?:측정(?:값)?|readings?){_PARTICLE}"
# 목표는 "목표 160"·"목표 DFT 200µm" 또는 "160µm 목표에". 뒤에 쉼표 목록이 이어지면 목표가 아니다("규격이 150, 200").
_TARGET_RE: Final = re.compile(
    rf"(?:목표|기준|규격|지정)(?:\s*(?:두께|dft))?(?:값|치)?\s*(?:는|은|가|이|:)?\s*({_NUMBER})(?:{_UNIT})?(?![\d.]|{_SEP}\d)")
_TARGET_AFTER_RE: Final = re.compile(rf"(?<![\d.,、])(?<![,、]\s)({_NUMBER})(?:{_UNIT})?\s*목표")
# "기준 3번 Zone"·"목표 2코트"·"기준 30분"의 수는 목표 두께가 아니다. "목표 160일 때"의 "일"은 서술격이라 받는다.
_TARGET_TAIL_RE: Final = re.compile(r"\s*(?:번|코트|분|초|시간|개|겹)|일(?!\s*때)")
# "200µm 넘으면"·"150 이상이면"은 문턱값을 묻는 말이지 측정값이 아니다. "이상 탐지"는 받는다.
_THRESHOLD_RE: Final = re.compile(rf"{_NUMBER}(?:{_UNIT})?\s*(?:이상(?!\s*탐지)|이하|미만|초과|넘)")
_ID_RE: Final = re.compile(r"\b(?:tr-l|seq-)\d+")  # 레벨·시퀀스 ID 의 숫자는 측정값이 아니다
_ZONE_RE: Final = re.compile(rf"{_NUMBER}\s*번(?:\s*(?:zone|구역))?|{_NUMBER}\s*구역|(?:zone|구역)\s*{_NUMBER}")
_RANGE_RE: Final = re.compile(rf"{_NUMBER}(?:{_UNIT})?\s*[~∼〜-]\s*{_NUMBER}")
# "1,500"·"150,000"은 천 단위 수다. "150,158,171"은 띄어쓰기 없는 목록으로 본다.
_THOUSANDS_RE: Final = re.compile(r"(?<![\d.,])\d{1,2},\d{3}(?!\d)|,000(?!\d)")
# 측정값은 머리말 바로 뒤의 수(쉼표 목록 또는 하나)다. 머리말은 도막 낱말(뒤에 "측정값" 가능) 또는
# 문장 첫머리·구두점·지금/현재/방금/이번·Zone 표기(@) 뒤의 "측정(값)"이다. "이격 300, 250 인데 도막"처럼
# 다른 낱말 뒤의 수는 그 낱말의 값이라 받지 않는다.
# 맨 "두께"는 첫머리·Zone 표기 뒤에서만 머리말이다("철판 두께 10, 12"는 소지 두께). 쉼표 뒤 "측정값"은 앞 구절이 다른 양일 수 있어 따로 본다.
# "건조막 두께"·"막두께"도 도막 낱말이다(4차 카나리 B2″). 공백 목록("DFT 150 158 171")은 도막 낱말 뒤에서만 받는다.
_VALUES_RE: Final = re.compile(
    rf"(?:(?P<coat>(?:(?:dft|도막|막\s*두께)(?:{_PARTICLE}\s*두께)?|(?:^|(?<=@))\s*두께){_PARTICLE}\s*(?:{_MARKER}\s*)?)"
    rf"|(?P<mark>(?:^|(?P<comma>,)|[.;@]|지금|현재|방금|이번)\s*{_PARTICLE}\s*{_MARKER}\s*))"
    rf"(?P<values>(?<![\d.]){_VALUE}(?:(?:{_SEP}|\s+){_VALUE})*)(?![\d.])")
_SPACED_RE: Final = re.compile(r"(?<![,、\s])\s+(?=\d)")  # 쉼표 없이 띄어쓰기로만 나눈 자리
# Zone ID 는 "1"~"39"(AGENTS.md). 단위 없는 공백 목록에 이 범위 수가 있으면 Zone 번호·횟수·날짜일 수 있어 모델로 보낸다.
_MAX_ZONE_NUMBER: Final = 39
# 단위 없는 마지막 수에 바로 붙은 말("2, 3코트"·"11, 12일")이나 띄운 영문 단위("80, 90 ku")는 다른 양이다. "171이야"·"150 이면"은 받는다.
_FOREIGN_TAIL_RE: Final = re.compile(r"(?![이인])[가-힣a-z°%/]|\s+[a-z°%/]")
# 길이·각도·횟수·층 단위가 붙은 수는 두께가 아니다. "도막"의 "도"는 제외한다.
_OTHER_UNIT_RE: Final = re.compile(rf"{_NUMBER}\s*(?:mm|cm|m/s|%|도(?!막)|회|층|차)")
# shortcut: 두께가 아닌 양·습막(WFT)·규격·다음 Zone 진행을 가리키는 낱말은 직접 정한 목록이다. 다른 말투는 모델로 간다.
_NOT_MEASUREMENT: Final = (
    # 습막 계열. "도장 직후·건조 전"은 WFT 정의(.taskmaster/docs/ship_painting_optimal_prd_final.md 용어집)다.
    "wft", "wet", "웻", "웨트", "습막", "습도막", "습윤", "습식", "젖은", "미건조", "건조 전", "건조전", "도장 직후",
    "자세", "각도", "거리", "속도", "규정", "권장", "사이", "최소", "최대", "범위", "다음")
_COATING_WORDS: Final = ("dft", "도막", "두께", "μm")  # "막두께"·"건조막 두께"는 "두께"로 잡힌다
# shortcut: 판정 의도 낱말은 직접 정한 목록이다. 다른 말투는 모델로 간다.
_JUDGE_WORDS: Final = ("판정", "합격", "괜찮", "통과", "어때", "정상")
_WEAK_JUDGE_WORDS: Final = ("맞아",)  # "DFT 150 맞아?"는 목표를 묻는 말일 수 있어 수 2개 이상일 때만 의도로 본다
_CAUSE_WORDS: Final = ("원인", "진단", "이상 탐지", "이상탐지", "왜")


class ToolAdapter(Protocol):
    async def dispatch_tool_call(
        self, tool_name: str, arguments: dict[str, Any], session_id: str | None = None) -> dict[str, Any]: ...


def parse_dft_question(instruction: str, require_intent: bool = False) -> tuple[list[float], float | None] | None:
    """(측정값, 목표 또는 None). 도막 측정값 질문이 아니면 None.

    1) 제외 낱말·범위·천 단위·다른 단위·문턱값 말투가 있으면 None. 2) 목표와 Zone 번호를 지운다.
    3) 머리말 바로 뒤의 수를 측정값으로 보고, 그 밖에 수가 남으면 None.
    4) 도막 낱말 머리말이면 µm·"측정" 표지·판정 의도 중 하나가, "측정값" 머리말만 있으면 도막 낱말 또는
       판정 의도와 수 2개 이상이 있어야 한다(내부 채팅의 기본 문맥은 도막, 명세 R4a-2 (라)).
    require_intent: 라우터가 판정 경로가 아닌 곳으로 보낸 질문("200µm 내려면 몇 회 칠해?")은 판정 의도나 목표가 있어야 한다.
    공백 목록은 도막 낱말 머리말 뒤에서만, 단위가 없으면 수가 모두 Zone 번호 범위(1~39)보다 클 때만 받는다.
    shortcut: "150과 158"·머리말 없는 수("150, 158 µm 판정")·"150/158" 은 모델로 보낸다.
    """
    text = _ID_RE.sub(" ", unicodedata.normalize("NFKC", instruction).casefold().strip())
    if any(word in text for word in _NOT_MEASUREMENT):
        return None
    if (_RANGE_RE.search(text) or _THOUSANDS_RE.search(text) or _OTHER_UNIT_RE.search(text)
            or _THRESHOLD_RE.search(text)):
        return None
    target_match = _TARGET_RE.search(text)
    if target_match is not None and _TARGET_TAIL_RE.match(text, target_match.end()):
        return None
    target_match = target_match or _TARGET_AFTER_RE.search(text)
    target =float(target_match.group(1)) if target_match else None
    if target is not None and target <= 0:
        return None
    rest = text if target_match is None else text[:target_match.start()] + " ; " + text[target_match.end():]
    rest = _ZONE_RE.sub(" @ ", rest)
    found = _VALUES_RE.search(rest)
    if found is None or re.search(r"\d", rest[:found.start("values")] + rest[found.end():]):
        return None
    picked = found.group("values")
    values = [float(v) for v in re.findall(_NUMBER, picked)]
    has_unit = bool(re.search(rf"{_UNIT}$", picked))
    if _SPACED_RE.search(picked) and (found.group("coat") is None
                                      or not has_unit and min(values) <= _MAX_ZONE_NUMBER):
        return None
    if not has_unit and _FOREIGN_TAIL_RE.match(rest, found.end()):
        return None
    intent = (any(word in text for word in _JUDGE_WORDS)
              or len(values) >= 2 and any(word in text for word in _WEAK_JUDGE_WORDS))
    coating_word = any(word in text for word in _COATING_WORDS)
    if found.group("coat") is not None:
        if not (has_unit or "측정" in found.group("coat") or intent):
            return None
    elif found.group("comma") is not None and not coating_word:
        return None  # "온도는 높고, 측정값 23, 25"
    elif not (len(values) >= 2 and intent or coating_word):
        return None
    if require_intent and not intent and target is None:
        return None
    return values, target


def _answer(instruction: str, values: list[float], judged: dict[str, Any]) -> str:
    target = float(judged["nominal"])
    upper = float(judged["normal_band_um"][1])
    goal = (f"기본 목표 {target:g}µm(질문에 목표가 없어 기본값으로 판정)" if judged["target_basis"] == "default_target"
            else f"목표 {target:g}µm")
    grade = judged["grade"]
    if grade is None:
        verdict = "[판정] 측정값이 모두 1µm 미만이라 칠한 곳이 없어 등급을 정하지 않았습니다."
    else:
        reason = {"below": f"{goal}에 못 미칩니다", "normal": f"{goal} 이상, {upper:g}µm 미만입니다",
                  "above": f"{goal}의 1.5배({upper:g}µm) 이상입니다"}[grade]
        verdict = f"[판정] {ca.GRADE_WORDS[grade]}입니다. 평균이 {reason}."
    counts = [judged["six_band_counts"].get(key, 0) for key in ca.SIX_BAND_KEYS]
    bands = ", ".join(f"{label} {n}개" for label, n in zip(ca.BAND_LABELS, counts) if n)
    mean = "" if judged["mean"] is None else f"의 평균은 {ca.mean_text(float(judged['mean']), target)}µm입니다"
    lines = [
        f"[분석] 측정값 {len(values)}개({', '.join(f'{v:g}' for v in values)}µm){mean}. {ca.RULE_NOTE}"
        + (f" 측정값 분포(목표 대비): {bands}." if bands else ""),
        verdict,
    ]
    if any(word in instruction for word in _CAUSE_WORDS):
        lines.append("[원인] 측정값만으로는 원인을 정할 수 없습니다. 분사 속도·이격거리 기록이 있어야 조절 후보를 낼 수 있습니다.")
    return "\n".join(lines)


def judge_dft_question(
    instruction: str, adapter: ToolAdapter | None, session_id: str | None, run_sync: Any,
    require_intent: bool = False,
) -> tuple[str, list[dict[str, Any]]] | None:
    """(답, 도구 호출 목록) 또는 None(도막 측정값 질문이 아님). MCP 가 실패하면 같은 규칙을 프로세스 안에서 부른다."""
    parsed = parse_dft_question(instruction, require_intent)
    if parsed is None:
        return None
    values, target = parsed
    arguments: dict[str, Any] = {"measurements": values}
    if target is not None:  # nominal=None 은 스키마 검증(number)이 막는다
        arguments["spec"] = {"nominal": target}
    reply: dict[str, Any] = {}
    if adapter is not None:
        try:
            reply = run_sync(adapter.dispatch_tool_call("dft_judge", arguments, session_id))
        except Exception:  # MCP 연결 불가는 오류 봉투가 아니라 예외로 온다
            logger.warning("dft_judge call raised; judging in-process", exc_info=True)
    judged = reply.get("result") if reply.get("status") == "success" else None
    if isinstance(judged, dict):
        return _answer(instruction, values, judged), [{"name": "dft_judge", "arguments": arguments, "result": reply}]
    if adapter is not None:
        logger.warning("dft_judge returned no result (%s); judging in-process", reply.get("status"))
    local = ca.assess_readings(values, target)
    judged = {"grade": local["grade"], "mean": local["painted_mean_um"], "nominal": local["target_thickness_um"],
              "target_basis": local["target_basis"], "normal_band_um": local["normal_band_um"],
              "six_band_counts": local["six_band_counts"]}
    return _answer(instruction, values, judged), []
