"""Semantic intent labels for the pose and Zone-order chat tabs."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

logger = logging.getLogger(__name__)

POSE_INTENTS = {"pose_overall", "pose_worst_zone", "pose_zone", "pose_right", "pose_left", "knowledge"}
ORDER_INTENTS = {"order_reference", "order_actual", "order_missing", "order_adherence", "order_direction", "knowledge"}


def is_coating_advice_question(prompt: str) -> bool:
    return bool(re.search(r"도막|두께|과도막|저도막|coating|thickness", prompt, re.I)
                and re.search(r"뭐|무엇|뜻|의미|정의|원인|이유|왜|방법|어떻게|예방|방지|원리|관계|영향|중요|하려면|개선|해야|고치|맞추|조절", prompt))


def is_coating_status_question(prompt: str) -> bool:
    subject = re.search(r"도막|두께|과도막|저도막|coating|thickness", prompt, re.I)
    if not subject or is_coating_advice_question(prompt):
        return False
    specific = re.search(r"영역|구역|zone|존|부위|현재|지금|이번|내[ 가]|어디|어느", prompt, re.I)
    definition = re.search(r"뭐|무엇|뜻|의미|정의|원인|예방|방지", prompt)
    return bool(specific or (not definition and re.search(r"과하|모자|부족|이상|상태|어때|분석|평가|얇|두껍|과다|괜찮|충분|양호|균일|불량|칠했|잘됐", prompt)))


def preset_intent(task_type: str, prompt: str) -> str | None:
    """Skip model classification only for exact, unambiguous UI questions."""
    normalized = re.sub(r"[\s?!.]", "", prompt).lower()
    presets = {
        "pose_recomm": {
            "전체자세유사도": "pose_overall", "전체자세유사도는": "pose_overall",
            "가장문제가큰zone": "pose_worst_zone", "가장문제가큰zone은": "pose_worst_zone",
            "오른손을어떻게고쳐": "pose_right", "왼손을어떻게고쳐": "pose_left",
        },
        "trajectory_recomm": {
            "권장zone순서": "order_reference", "권장zone순서는": "order_reference",
            "권장순서는": "order_reference", "실제수행순서는": "order_actual",
            "누락한zone은": "order_missing", "순서준수율은": "order_adherence",
            "도포방향은": "order_direction",
        },
    }
    return presets.get(task_type, {}).get(normalized)


def fallback_intent(task_type: str, prompt: str) -> str:
    if is_coating_advice_question(prompt):
        return "knowledge"
    lowered = prompt.lower()
    if task_type == "pose_recomm":
        if "오른" in lowered or "right" in lowered:
            return "pose_right"
        if "왼" in lowered or "left" in lowered:
            return "pose_left"
        if "zone" in lowered or "구역" in lowered:
            return "pose_worst_zone" if any(word in lowered for word in ("문제", "최악", "낮")) else "pose_zone"
        if any(word in lowered for word in ("자세", "유사도", "점수", "평균", "전체")):
            return "pose_overall"
        return "knowledge"
    if any(word in lowered for word in ("권장", "추천", "정석")):
        return "order_reference"
    if any(word in lowered for word in ("실제", "수행 순서", "칠한 순서")):
        return "order_actual"
    if any(word in lowered for word in ("누락", "안 칠", "빠진", "미수행")):
        return "order_missing"
    if "방향" in lowered:
        return "order_direction"
    if any(word in lowered for word in ("준수", "유사", "몇 퍼센트", "점수")):
        return "order_adherence"
    return "knowledge"


def classify_question_intent(
    task_type: str,
    prompt: str,
    llm_classifier: Callable[[str, str], str] | None,
) -> str:
    allowed = POSE_INTENTS if task_type == "pose_recomm" else ORDER_INTENTS
    if llm_classifier is not None:
        try:
            result = str(llm_classifier(task_type, prompt)).strip().lower()
            if result in allowed:
                # A tab is a UI preference. A confidently different industrial
                # subject must not become a Zone-order answer on a bad label.
                if (is_coating_advice_question(prompt) or (result != "knowledge" and fallback_intent(task_type, prompt) == "knowledge"
                        and re.search(r"도막|두께|이격|속도|노즐|압력|안전|보호구|오버랩", prompt))):
                    return "knowledge"
                return result
            logger.warning("copilot intent classifier returned invalid label task=%s label=%s", task_type, result)
        except Exception:  # noqa: BROAD_EXCEPT_OK - deterministic fallback keeps chat available
            logger.exception("copilot intent classification failed task=%s", task_type)
    return fallback_intent(task_type, prompt)
