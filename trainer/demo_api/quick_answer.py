"""시퀀스 분석의 실시간 text_guide를 한 줄 60자로 맞추는 유틸리티.

Query 답변 본문에는 적용하지 않는다. 실시간 안내의 내용은 해당 분석 엔진이 정한다.
문장 중간은 자르지 않는다(v3.8.2 원칙). Zone 순서처럼 화살표로 이은 목록만 앞부분과 개수로 줄인다.
"""
from __future__ import annotations

import re
from typing import Final

QUICK_LIMIT: Final = 60
_BULLET: Final = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_SENTENCE_END: Final = re.compile(r"(?<=[.!?。？！])\s+")
_ARROW: Final = re.compile(r"\s*→\s*")
_CHAIN: Final = re.compile(r"\d+(?:→\d+){5,}")  # Zone 번호는 숫자 문자열이다(AGENTS.md)


def _chains(text: str, head: int) -> str:
    """'7→4→8→5→6→3→2→1→9' → '7→4→8…(9개)'."""
    def short(match: re.Match[str]) -> str:
        items = match.group(0).split("→")
        return "→".join(items[:head]) + f"…({len(items)}개)"
    return _CHAIN.sub(short, text)


def _sentences(text: str) -> list[str]:
    lines = (_BULLET.sub("", line).strip() for line in text.splitlines())
    return [part.strip() for line in lines if line for part in _SENTENCE_END.split(line) if part.strip()]


def _prefix(sentences: list[str]) -> str:
    kept = sentences[0] if sentences else ""
    for sentence in sentences[1:]:
        if len(kept) + 1 + len(sentence) > QUICK_LIMIT:
            break
        kept += " " + sentence
    return kept


def one_line(answer: str) -> str:
    """줄바꿈·목록 기호를 없애고 앞에서부터 문장 단위로 60자까지만 남긴다. 첫 문장이 60자를 넘으면 그 문장을 그대로 둔다."""
    sentences = [_ARROW.sub("→", sentence) for sentence in _sentences(answer)]
    kept = _prefix(sentences)
    for head in (5, 3, 2):
        if len(kept) <= QUICK_LIMIT:
            break
        kept = _prefix([_chains(sentence, head) for sentence in sentences])
    return kept
