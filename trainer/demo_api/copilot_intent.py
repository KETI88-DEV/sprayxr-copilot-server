"""Semantic intent labels for the pose and Zone-order chat tabs."""

from __future__ import annotations

import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)

POSE_INTENTS = {"pose_overall", "pose_worst_zone", "pose_zone", "pose_right", "pose_left", "knowledge"}
ORDER_INTENTS = {"order_reference", "order_actual", "order_missing", "order_adherence", "order_direction", "knowledge"}


def fallback_intent(task_type: str, prompt: str) -> str:
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
                return result
            logger.warning("copilot intent classifier returned invalid label task=%s label=%s", task_type, result)
        except Exception:  # noqa: BROAD_EXCEPT_OK - deterministic fallback keeps chat available
            logger.exception("copilot intent classification failed task=%s", task_type)
    return fallback_intent(task_type, prompt)
