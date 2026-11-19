"""교육 용어 ↔ 문서 표현 대응(2026-11-18 '이격거리' 상세 분석 기권).

문서는 '이격거리'를 전기 작업 안전거리로만 쓰고 도장 거리는 '분사 거리'로 쓴다. 27B는 근거를 받고도
두 말을 같은 뜻으로 보지 않아 기권했다. 검색어 뒤에 문서 표현을 덧붙이고(용어는 그대로 둔다),
답변 맥락에 대응 한 줄을 넣는다. 목록은 configs/term_rules.json.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

_RULES = json.loads((Path(__file__).resolve().parents[1] / "configs" / "term_rules.json").read_text(encoding="utf-8"))["용어"]
_SPACE = re.compile(r"\s+")


def _flat(text: str) -> str:
    return _SPACE.sub("", text)


def _found(text: str, around: str) -> list[dict]:
    flat, near = _flat(text), _flat(around)
    return [rule for rule in _RULES
            if any(_flat(alias) in flat for alias in rule["별칭"])
            and not any(_flat(word) in near for word in rule["제외"])]


def matched_terms(prompt: str, previous_prompts: Iterable[str] = ()) -> list[dict]:
    """질문의 용어 규칙. 질문에 용어가 없으면 가장 가까운 이전 질문의 용어를 이어받는다.
    제외어는 이전 질문까지 본다: 블라스팅 이야기 뒤의 "이격거리는 적절했는데?"는 노즐 거리다."""
    previous = list(previous_prompts)
    around = " ".join([prompt, *previous])
    for text in [prompt, *reversed(previous)]:
        found = _found(text, around)
        if found:
            return found
    return []


def named_terms(text: str) -> set[str]:
    """글에 별칭이 나온 교육 용어 이름(제외어는 보지 않는다). 꼬리물기 이어받기의 주제 경계로 쓴다."""
    flat = _flat(text)
    return {rule["용어"] for rule in _RULES if any(_flat(alias) in flat for alias in rule["별칭"])}


def means_otherwise(term: str, text: str) -> bool:
    """용어가 이 글에서 다른 뜻인가(예: '충전전로 최소 이격거리'). 빠른 안내의 용어 풀이 표도 이걸 따른다."""
    flat = _flat(text)
    return any(_flat(term) in map(_flat, rule["별칭"]) and any(_flat(word) in flat for word in rule["제외"]) for rule in _RULES)


def with_document_terms(query: str, rules: list[dict]) -> str:
    """검색어 뒤에 문서 표현을 덧붙인다. 용어를 바꾸면 그 낱말이 그대로 적힌 조각이 가려진다(2026-11-17)."""
    extra = [expr for rule in rules for expr in rule["문서 표현"] if _flat(expr) not in _flat(query)]
    return f"{query} ({', '.join(dict.fromkeys(extra))})" if extra else query


def term_context(rules: list[dict]) -> str:
    lines = [f"- {rule['용어']}: 문서에서는 {', '.join(repr(e) for e in rule['문서 표현'])}(으)로 씁니다. 뜻은 {rule['뜻']}입니다. "
             f"같은 뜻이므로 그 표현으로 적힌 근거로 답하세요." + (f" {rule['주의']}" if rule["주의"] else "")
             for rule in rules]
    return "[용어 대응]\n" + "\n".join(lines)
