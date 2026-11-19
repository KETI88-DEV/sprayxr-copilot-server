"""질문 유형별 맥락 프롬프트와 근거 묶음(2026-11-18). 목록은 configs/context_prompts.json.

'저도막 원인'처럼 답이 여러 조각에 흩어지거나, '충전전로 이격거리'처럼 검색이 다른 뜻의 조각을 가져오는
질문은 골라 둔 원문 조각을 검색 결과 앞에 붙이고(agent_pipeline ``pinned_chunk_ids``) 유형 규칙을 맥락에 넣는다.
조각 본문은 색인 원문을 그대로 쓰므로 인용이 실제 문서를 가리킨다.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from trainer.demo_api.term_rules import named_terms

_TYPES = json.loads((Path(__file__).resolve().parents[1] / "configs" / "context_prompts.json").read_text(encoding="utf-8"))["질문 유형"]
_SPACE = re.compile(r"\s+")


def _has(text: str, words: Iterable[str]) -> bool:
    flat = _SPACE.sub("", text)
    return any(_SPACE.sub("", word) in flat for word in words)


def _subjects(text: str) -> set[str]:
    return {entry["용어"] for entry in _TYPES if _has(text, entry["대상"])}


def _other_topic(text: str) -> bool:
    """결함 대상은 없고 다른 교육 용어(이격거리·분사 속도 등)가 있는 질문 = 주제가 바뀐 곳. 더 거슬러 이어받지 않는다."""
    return bool(named_terms(text) - {entry["용어"] for entry in _TYPES if entry["이어받기"]})


def matched_prompt(prompt: str, previous_prompts: Iterable[str] = ()) -> dict | None:
    """처음 맞는 질문 유형. 질문에 대상이 없으면 대상이 적힌 가장 가까운 이전 질문에서 이어받는다.
    대상이 둘 이상인 비교·정정 질문('저도막 말고 과도막 원인')은 한 묶음으로 좁히지 않는다.
    사이에 다른 교육 용어를 물은 질문이 있으면 그 앞으로는 거슬러 가지 않는다."""
    subjects = _subjects(prompt)
    if subjects:
        return None if len(subjects) > 1 else next(
            (entry for entry in _TYPES if _has(prompt, entry["질문"]) and _has(prompt, entry["대상"])), None)
    if _other_topic(prompt):
        return None
    for previous in reversed(list(previous_prompts)):
        subjects = _subjects(previous)
        if not subjects:
            if _other_topic(previous):
                return None
            continue
        return None if len(subjects) > 1 else next(
            (entry for entry in _TYPES
             if entry["이어받기"] and _has(prompt, entry["질문"]) and _has(previous, entry["대상"])), None)
    return None


def recent_subject(prompts: Iterable[str]) -> str | None:
    """대상이 적힌 가장 가까운 질문의 결함 용어('저도막' 등). 둘 이상이면 None(비교 질문은 한쪽으로 좁히지 않는다)."""
    for text in reversed(list(prompts)):
        subjects = _subjects(text)
        if subjects:
            return next(iter(subjects)) if len(subjects) == 1 else None
        if _other_topic(text):
            return None
    return None


def context_text(entry: dict) -> str:
    return "\n".join(entry["규칙"])
