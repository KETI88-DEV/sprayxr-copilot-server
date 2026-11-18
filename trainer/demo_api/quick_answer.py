"""시퀀스 분석의 실시간 text_guide를 한 줄 60자로 맞추는 유틸리티.

one_line은 실시간 안내 전용이며 내용은 해당 분석 엔진이 정한다.
문장 중간은 자르지 않는다(v3.8.2 원칙). Zone 순서처럼 화살표로 이은 목록만 앞부분과 개수로 줄인다.
Query의 선택적 요약은 별도 summarize_query_answer에서 안전·수치·순서를 보존한다.
"""
from __future__ import annotations

import re
from typing import Final

QUICK_LIMIT: Final = 60
QUERY_SUMMARY_INSTRUCTION: Final = (
    "핵심 요약: 질문에 직접 답하는 내용만 짧고 명확하게 전달하세요. "
    "가능하면 한두 문장으로 답하되, 요청한 원인·방법·절차·목록의 항목은 빠뜨리지 마세요. "
    "도입·배경 설명과 반복은 생략하고, 필요한 항목은 짧은 문장이나 목록으로 묶으세요. "
    "별도 제목이나 Markdown 강조 없이 일반 텍스트로 답하되 코드·수학 연산자는 그대로 유지하세요. "
    "안전 경고·판정 한계·확인된 수치와 단위·적용 조건·전체 Zone 순서는 유지하세요. "
    "필수 내용을 담기 위해 필요한 문장은 남기며, 없는 사실이나 현재 상태를 추측하지 마세요."
)
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


def summarize_query_answer(answer: str, *, safety: bool = False, precomposed: bool = False) -> str:
    """Format an already concise generation without deleting approved content.

    The first generation owns semantic summarization. This final gate changes
    whitespace presentation only, never selects or deletes sentences.
    Safety and question-specific engine/record answers stay intact: their facts,
    comparison basis and recovery actions are already the composed core.
    No sentence, condition, decimal, unit or arrow chain is cut in half.
    The source inference result/cache is never modified.
    """
    if safety or precomposed:
        return answer
    return " ".join(answer.split())
