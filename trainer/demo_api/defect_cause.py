"""결함 원인 질문에 붙일 원문 근거 묶음과 규칙(2026-11-18).

'저도막 원인'처럼 원인이 여러 조각에 흩어진 질문은 검색 8칸에 다 들어오지 않고, NCS 학습 목표 반복 문장이
자리를 채운다. 묶음 조각을 검색 결과 앞에 붙이고(agent_pipeline ``pinned_chunk_ids``) 인과 방향 규칙을 맥락에 넣는다.
조각 본문은 색인 원문을 그대로 쓰므로 인용이 실제 문서를 가리킨다.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

_DATA = json.loads((Path(__file__).resolve().parents[1] / "configs" / "defect_cause_evidence.json").read_text(encoding="utf-8"))["결함"]
_CAUSE = re.compile(r"원인|왜|이유|생기|생겨|생깁|생긴|생겼|때문")
_SPACE = re.compile(r"\s+")


def _defect_in(text: str) -> str | None:
    flat = _SPACE.sub("", text)
    for name, entry in _DATA.items():
        if any(_SPACE.sub("", alias) in flat for alias in entry["별칭"]):
            return name
    return None


def defect_cause_topic(prompt: str, previous_prompts: Iterable[str] = ()) -> str | None:
    """원인을 묻는 질문이면 결함 이름. 결함 낱말이 없으면 직전 질문들에서 가장 가까운 결함을 이어받는다."""
    if not _CAUSE.search(prompt):
        return None
    found = _defect_in(prompt)
    if found is not None:
        return found
    for previous in reversed(list(previous_prompts)):
        found = _defect_in(previous)
        if found is not None:
            return found
    return None


def pinned_chunks(name: str) -> list[str]:
    return list(_DATA[name]["조각"])


def cause_context(name: str) -> str:
    return (
        f"[결함 원인 질문]\n질문한 결함: {name}\n"
        f"- 문서 근거에 '~하면 ~이 발생한다'로 적힌 {name}의 원인만 말하세요. 원인마다 해당 조각을 인용하세요.\n"
        "- 원인과 결과의 방향을 바꾸지 마세요. 'A 또는 B로 C가 생긴다'는 A와 B가 각각 C의 원인이라는 뜻이지, B가 A의 원인이라는 뜻이 아닙니다.\n"
        f"- 다른 결함(처짐·흘러내림·링클링·도막 불균일 등)의 원인을 {name}의 원인으로 옮기지 마세요. 한 문장에 반대 조건이 함께 있으면 {name} 쪽 조건만 쓰세요.\n"
        "- 근거에 없는 원인이나 수치는 덧붙이지 마세요."
    )
