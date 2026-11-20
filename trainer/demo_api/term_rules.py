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

_CONFIG = json.loads((Path(__file__).resolve().parents[1] / "configs" / "term_rules.json").read_text(encoding="utf-8"))
_RULES = _CONFIG["용어"]
_SPACE = re.compile(r"\s+")
# 오타 → 바른 용어. 목록(term_rules.json '오타')에 적힌 글자열만 바꾼다. 앞에 한글이 붙어 있으면 다른 낱말의 일부라 두고,
# 긴 것부터 맞춘다. 비슷한 글자를 어림으로 맞추지 않는다: '분사 강도'가 '분사 각도'로, '저도 많이'가 '저도막'으로 바뀐다.
_TYPOS = {typo: right for right, wrong in _CONFIG.get("오타", {}).items() for typo in wrong}
_TYPO = re.compile("(?<![가-힣])(?:" + "|".join(map(re.escape, sorted(_TYPOS, key=len, reverse=True))) + ")") if _TYPOS else None


def _flat(text: str) -> str:
    return _SPACE.sub("", text).casefold()  # 'dft 뭐임'도 DFT다


def _names(rule: dict) -> list[str]:
    """용어를 찾는 말: 별칭과 구어(입말 표현)."""
    return [*rule["별칭"], *rule.get("구어", ())]


def _found(text: str, around: str, *, formal: bool = False) -> list[dict]:
    flat, near = _flat(text), _flat(around)
    return [rule for rule in _RULES
            if any(_flat(alias) in flat for alias in (rule["별칭"] if formal else _names(rule)))
            and not any(_flat(word) in near for word in rule["제외"])]


def routed_terms(prompt: str) -> list[dict]:
    """질문을 도장 질문(문서 경로)으로 보게 하는 용어: '경로'가 참이고 '별칭'(구어 제외)이 질문에 있으며 제외어가 없다.

    구어('가까이서 뿌', '안 말라')는 일상어·기록 대화와 겹쳐 경로 판정에 쓰지 않는다. 2026-11-18의 여덟 용어는 '경로'가 없어
    예전 경로 그대로다(순서 탭 '도포 방향은 어떻게 돼?'가 문서로 새지 않는다)."""
    return [rule for rule in _found(prompt, prompt, formal=True) if rule.get("경로")]


# 단위의 뜻·환산을 묻는 말. 기록 수치를 묻는 질문("이격거리 몇 mm였어?")에 단위 표를 붙이지 않으려고 본다.
_UNIT_ASK = re.compile(r"뭐|무엇|뜻|의미|단위|환산|바꾸|얼마나|차이|같[아은]|몇\s*(?:[A-Za-zµμ㎛㎡℃°%]|마이크로|미크론|밀리|센치|센티|미터|인치|피트|리터|갤런|도\b|초\b)")


def unit_facts(prompt: str) -> dict[str, str]:
    """질문이 뜻·환산을 묻는 단위의 뜻(일반 지식). 문서 근거 없이 답할 때 답 모델과 검토가 '마이크로미터'를 측정 공구로 읽지 않게 준다."""
    if not _UNIT_ASK.search(prompt):
        return {}
    return {unit["단위"]: unit["뜻"] for unit in _CONFIG.get("단위", ())
            if any(re.search(_unit_pattern(alias), prompt, re.I) for alias in unit["별칭"])}


def _unit_pattern(alias: str) -> str:
    if alias.startswith("(?"):  # 앞 글자 조건을 직접 적은 별칭('마이크로미터' 안의 '미터'는 미터가 아니다)
        return alias
    # 영문 기호는 낱말 안에 들어 있을 때('similar'의 mil, 'µm'의 m) 맞지 않게 한다.
    return (r"(?<![A-Za-zµμ/])" + re.escape(alias) + r"(?![A-Za-z/²2])") if alias.isascii() and alias.isalpha() else re.escape(alias)


def fix_typos(text: str) -> str:
    """등록된 용어 오타를 바른 용어로 바꾼다('저도맥이 뭐야?' → '저도막이 뭐야?'). 질문 분석·검색·이어받기가 모두 바른 용어를 본다."""
    return _TYPO.sub(lambda found: _TYPOS[found.group()], text) if _TYPO else text


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


_MEANING = re.compile(r"뭐|머[야에예임]|모야|무엇|뜻|의미|정의|이?란\s*[?？]?$|설명|차이|다른가|다르|어떤\s*(?:상태|것|거|현상)")
_NOT_MEANING = re.compile(r"왜|원인|이유|어떻게\s*(?:해|하|칠|재|써|쓰|막|고)|어떡|방법|얼마|몇|언제|어디|기준|적정|순서|문제|영향|결과|위험")


def meaning_evidence(prompt: str, rules: list[dict]) -> list[str]:
    """뜻·차이만 묻는 질문('습도막이 뭐야?', 낱말만 적은 '저도막', '습도막과 건조도막 차이')이면 그 용어의 근거 조각.

    검색은 낱말이 닮은 다른 뜻의 조각을 가져온다(습도막→상대습도, 드라이 스프레이→에어리스 장비, 2026-11-20).
    뜻을 적은 조각을 검색 결과 앞에 붙여 두면 답이 그 조각을 인용한다. 용어가 셋 이상이면 좁히지 않는다."""
    formal = {rule["용어"] for rule in _found(prompt, prompt, formal=True)}
    rules = [rule for rule in rules if rule["용어"] in formal]  # 구어로만 걸린 용어('곰보빵'의 곰보)는 뜻 근거를 붙이지 않는다
    if not rules or len(rules) > 2:
        return []
    bare = _flat(prompt).strip("?？!.") in {_flat(alias) for rule in rules for alias in [rule["용어"], *rule["별칭"]]}
    if not bare and not (_MEANING.search(prompt) and not _NOT_MEANING.search(prompt)):
        return []
    return list(dict.fromkeys(chunk for rule in rules for chunk in rule["근거"][:3 if len(rules) == 1 else 2]))


def named_terms(text: str) -> set[str]:
    """글에 별칭이 나온 교육 용어 이름(제외어는 보지 않는다). 꼬리물기 이어받기의 주제 경계로 쓴다."""
    flat = _flat(text)
    return {rule["용어"] for rule in _RULES if any(_flat(alias) in flat for alias in _names(rule))}


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
