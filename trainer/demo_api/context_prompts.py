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

_CONFIG = json.loads((Path(__file__).resolve().parents[1] / "configs" / "context_prompts.json").read_text(encoding="utf-8"))
_SPACE = re.compile(r"\s+")


def _filled(entry: dict) -> dict:
    """'종류'를 적은 유형은 그 종류의 질문 낱말·질문 제외·규칙 틀을 받는다. 틀의 {용어}·{표현}·{다른 결함}·{안내}는 유형의 값으로 채우고
    '덧붙임' 줄을 뒤에 붙인다. '질문'이 없는 종류(안내)는 대상만 맞으면 쓴다."""
    kind = entry.get("종류")
    if kind is None:
        return {"질문 제외": [], **entry}
    return {"질문": _CONFIG["질문 낱말"].get(kind, [""]), "질문 제외": _CONFIG["질문 제외"].get(kind, []),
            "규칙": [line.format(**entry) for line in _CONFIG["규칙 틀"][kind]] + list(entry.get("덧붙임", ())), **entry}


_TYPES = [_filled(entry) for entry in _CONFIG["질문 유형"]]
_INHERITED = {entry["용어"] for entry in _TYPES if entry["이어받기"]}


def _has(text: str, words: Iterable[str]) -> bool:
    flat = _SPACE.sub("", text).casefold()
    return any(_SPACE.sub("", word).casefold() in flat for word in words)


def _about(text: str, entry: dict) -> bool:
    """글에 유형의 대상이 있고 다른 뜻이라는 낱말('제외': '어깨 처짐', '백화점', 블라스팅 거리)이 없다."""
    return _has(text, entry["대상"]) and not _has(text, entry.get("제외", ()))


def _asks(prompt: str, entry: dict) -> bool:
    """유형의 질문 낱말이 있고 그 종류가 아니라는 낱말('질문 제외')이 없다."""
    return _has(prompt, entry["질문"]) and not _has(prompt, entry["질문 제외"])


def _subjects(text: str) -> set[str]:
    """글에 적힌 결함(이어받는 유형의 용어). 둘 이상이면 비교·정정 질문이라 한 묶음으로 좁히지 않는다."""
    return {entry["용어"] for entry in _TYPES if entry["이어받기"] and _about(text, entry)}


def _other_topic(text: str) -> bool:
    """결함 대상은 없고 다른 주제(교육 용어, 안내 유형의 대상)를 말한 질문 = 주제가 바뀐 곳. 더 거슬러 이어받지 않는다."""
    return bool(named_terms(text) - _INHERITED) or any(_about(text, entry) for entry in _TYPES if not entry["이어받기"])


def matched_prompt(prompt: str, previous_prompts: Iterable[str] = ()) -> dict | None:
    """처음 맞는 질문 유형. 질문에 대상이 없으면 대상이 적힌 가장 가까운 이전 질문에서 이어받는다.
    결함이 둘 이상인 비교·정정 질문('저도막 말고 과도막 원인')은 한 묶음으로 좁히지 않는다.
    사이에 다른 주제를 물은 질문이 있으면 그 앞으로는 거슬러 가지 않는다."""
    subjects = _subjects(prompt)
    if len(subjects) > 1:
        return None
    found = next((entry for entry in _TYPES if _asks(prompt, entry) and _about(prompt, entry)), None)
    if found is not None or subjects or _other_topic(prompt):
        return found
    for previous in reversed(list(previous_prompts)):
        subjects = _subjects(previous)
        if not subjects:
            if _other_topic(previous):
                return None
            continue
        return None if len(subjects) > 1 else next(
            (entry for entry in _TYPES if entry["이어받기"] and _asks(prompt, entry) and _about(previous, entry)), None)
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
