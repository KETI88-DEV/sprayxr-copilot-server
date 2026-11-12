"""내부 채팅이 질문에 적힌 DFT 측정값을 모델 없이 dft_judge(PR #8 기준)로 판정한다.

2026-11-12 카나리: 27B 가 도구를 부르지 않고 fast 재답이 RAG 문서의 옛 90% 규칙을 옮겼다.
측정값을 결정적으로 뽑을 수 있으면 판정·평균·6구간은 도구 결과로만 쓰고, 원인은 지어내지 않는다.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Final, Protocol

from trainer.demo_api import coating_assessment as ca

logger = logging.getLogger(__name__)

_NUMBER: Final = r"\d+(?:\.\d+)?"
_UNIT: Final = r"\s*(?:μm|um|마이크로미터|마이크론)"  # casefold 뒤라 µ(U+00B5)는 μ(U+03BC)다
_TARGET_RE: Final = re.compile(
    rf"(?:목표|기준|규격|지정)(?:\s*두께)?(?:값|치)?\s*(?:는|은|가|이|:)?\s*({_NUMBER})(?:{_UNIT})?")
_ZONE_RE: Final = re.compile(rf"{_NUMBER}\s*번|(?:zone|구역)\s*{_NUMBER}")
_RANGE_RE: Final = re.compile(rf"{_NUMBER}(?:{_UNIT})?\s*[~∼〜-]\s*{_NUMBER}")
_THOUSANDS_RE: Final = re.compile(r"\d,\d{3}(?!\d)")
# 쉼표로 이은 수 2개 이상.
_LIST_RE: Final = re.compile(rf"(?<![\d.]){_NUMBER}(?:{_UNIT})?(?:\s*[,，、]\s*{_NUMBER}(?:{_UNIT})?)+(?![\d.])")
_UNIT_NUMBER_RE: Final = re.compile(rf"(?<![\d.])({_NUMBER}){_UNIT}")
# 길이·각도·횟수·층 단위가 붙은 수는 두께가 아니다. "도막"의 "도"는 제외한다.
_OTHER_UNIT_RE: Final = re.compile(rf"{_NUMBER}\s*(?:mm|cm|m/s|%|도(?!막)|회|층|차)")
_NOT_MEASUREMENT: Final = ("wft", "자세", "각도", "거리", "속도", "규정", "권장", "사이", "최소", "최대", "범위")
_MEASURE_MARKERS: Final = ("측정값", "측정")
# 도막 낱말 없는 길: 문장 첫머리·지금/현재/방금/이번·Zone 표기 바로 뒤의 "측정(값)" 다음에 쉼표 목록이 와야 한다("온도 측정값 23, 25"는 제외).
_BARE_MARKER_RE: Final = re.compile(
    rf"(?:^|(?:지금|현재|방금|이번|(?:zone|구역)\s*{_NUMBER}|{_NUMBER}\s*번)(?:의|에서|은|는)?)\s*측정(?:값)?\s*[은는이가:]?\s*{_NUMBER}(?:{_UNIT})?\s*[,，、]")
_COATING_WORDS: Final = ("dft", "도막", "두께", "μm")
# shortcut: 판정 의도 낱말은 직접 정한 목록이다. 다른 말투는 모델로 간다.
_JUDGE_WORDS: Final = ("판정", "합격", "괜찮", "통과", "어때")
_CAUSE_WORDS: Final = ("원인", "진단", "이상 탐지", "이상탐지", "왜")


class ToolAdapter(Protocol):
    async def dispatch_tool_call(
        self, tool_name: str, arguments: dict[str, Any], session_id: str | None = None) -> dict[str, Any]: ...


def parse_dft_question(instruction: str) -> tuple[list[float], float | None] | None:
    """(측정값, 목표 또는 None). 도막 질문이 아니거나 측정값을 못 뽑으면 None.

    shortcut: 쉼표 목록(µm 또는 "측정값" 표지) 또는 질문의 유일한 µm 붙은 수만 측정값으로 본다. 그 밖의 말투("150과 158", 공백 목록)는 모델로 보낸다.
    """
    text = instruction.casefold()
    # 내부 채팅의 기본 문맥은 도막이라 "측정값" 표지 + 판정 의도면 도막 낱말이 없어도 받는다(명세 R4a-2 (라) 확인 문장).
    asked = (bool(_BARE_MARKER_RE.search(text.strip())) and "다음" not in text
             and any(word in text for word in _JUDGE_WORDS))
    if not (asked or any(word in text for word in _COATING_WORDS)) or any(word in text for word in _NOT_MEASUREMENT):
        return None
    if _RANGE_RE.search(text) or _THOUSANDS_RE.search(text) or _OTHER_UNIT_RE.search(text):
        return None
    target_match = _TARGET_RE.search(text)
    target = float(target_match.group(1)) if target_match else None
    if target is not None and target <= 0:
        return None
    rest = _ZONE_RE.sub(" ", _TARGET_RE.sub(" ", text))
    listed = _LIST_RE.search(rest)
    if listed:
        if not (re.search(_UNIT, listed.group(0)) or any(word in text for word in _MEASURE_MARKERS)):
            return None
        picked, left = listed.group(0), rest.replace(listed.group(0), " ")
    else:
        picked, left = rest, ""
        if len(re.findall(_NUMBER, rest)) != 1 or not _UNIT_NUMBER_RE.search(rest):
            return None
    if listed and re.search(_NUMBER, left):
        return None
    return [float(v) for v in re.findall(_NUMBER, picked)], target


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
) -> tuple[str, list[dict[str, Any]]] | None:
    """(답, 도구 호출 목록) 또는 None(도막 측정값 질문이 아님). MCP 가 실패하면 같은 규칙을 프로세스 안에서 부른다."""
    parsed = parse_dft_question(instruction)
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
