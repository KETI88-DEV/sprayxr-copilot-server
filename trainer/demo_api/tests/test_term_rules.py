"""교육 용어 ↔ 문서 표현 대응(2026-11-18 상세 분석 '이격거리' 기권)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from trainer.demo_api.context_prompts import matched_prompt
from trainer.demo_api.term_rules import _RULES, fix_typos, matched_terms, same_word_lines, term_context, with_document_terms
from trainer.demo_api.tests.test_query_contextual_dialogue import DialoguePipeline, ask, client

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


@pytest.mark.parametrize("prompt, expected", [
    ("이격거리는 얼마가 적당해?", ["이격거리"]),
    ("적정 분사 속도는?", ["분사 속도"]),
    ("오버랩은 몇 퍼센트로 해야 돼?", ["중첩률"]),
    ("도포 방향은?", ["도포 방향"]),
    ("분사 각도는 몇 도가 적당해?", ["분사 각도"]),
    ("저도막이 뭐야?", ["저도막"]),
    ("과도막이 생기면 어떤 문제가 있어?", ["과도막"]),
    ("충전전로 최소 이격거리는?", []),           # 전기 안전거리는 다른 뜻
    ("블라스팅 연마재 분사속도는?", []),         # 연마재 분사속도는 다른 뜻
    ("후도막형 도료는 왜 써?", ["후도막형 도료"]),  # 도료 종류이지 과도막 결함이 아니다
    ("가사시간이 뭐야?", ["가사시간"]),          # 2026-11-20 등록
    ("점심 메뉴 추천해 줘", []),
])
def test_terms_are_found_and_other_meanings_left_alone(prompt, expected):
    assert [rule["용어"] for rule in matched_terms(prompt)] == expected


def test_follow_up_inherits_the_term_of_the_previous_question():
    assert [r["용어"] for r in matched_terms("그럼 얼마가 적당해?", ["이격거리가 뭐야?"])] == ["이격거리"]


def test_high_build_paint_is_not_an_excess_film_cause_question():
    assert matched_prompt("후도막형 도료는 왜 써?") is None


def test_document_terms_are_appended_and_the_user_term_kept():
    query = with_document_terms("이격거리는 얼마가 적당해?", matched_terms("이격거리는 얼마가 적당해?"))
    assert query.startswith("이격거리는 얼마가 적당해?") and "분사 거리" in query
    assert with_document_terms("가사시간이 뭐야?", []) == "가사시간이 뭐야?"


def test_every_rule_has_meaning_and_evidence_outside_the_exclusion_list():
    excluded = set(json.loads((CONFIGS / "evidence_exclusions.json").read_text(encoding="utf-8"))["조각"])
    for rule in _RULES:
        assert rule["별칭"] and rule["문서 표현"] and rule["뜻"] and rule["근거"], rule["용어"]
        assert not excluded & set(rule["근거"]), rule["용어"]
    assert "'분사 거리'" in term_context(matched_terms("이격거리"))


@pytest.mark.parametrize("mode", ["quick", "detailed"])
def test_router_adds_the_rule_and_the_document_terms(mode):
    pipeline = DialoguePipeline()
    ask(client(pipeline), "분사 각도는 몇 도가 적당해?", response_mode=mode)  # 이격거리·속도·중첩 적정값은 2026-11-20부터 근거 묶음(질문 유형)이 맡는다
    call = pipeline.calls[-1]
    assert "[용어 대응]" in (call.get("generation_instruction") or call["instruction"])
    assert "분사 각도" in call["retrieval_instruction"] and "피도면과의 각도 직각" in call["retrieval_instruction"]


def test_router_leaves_record_questions_alone():
    pipeline = DialoguePipeline()
    ask(client(pipeline), "Zone 3 이격거리는 얼마였나요?", response_mode="detailed")
    assert all("[용어 대응]" not in str(call.get("instruction", "")) for call in pipeline.calls)


def test_blasting_conversation_keeps_its_own_distance():
    assert matched_terms("이격거리는 적절했는데?", ["블라스팅 노즐에서 분진이 많이 생기는 원인은?"]) == []


def test_quick_term_table_leaves_the_electrical_clearance_alone():
    from trainer.demo_api.query_runtime import retrieval_question
    assert "분사 거리" not in retrieval_question("충전전로 최소 이격거리는?")
    assert "분사 거리" in retrieval_question("이격거리는 얼마가 적당해?")


def test_cause_question_keeps_only_the_cause_rule_for_its_defect():
    pipeline = DialoguePipeline()
    ask(client(pipeline), "과도막은 왜 생겨?", response_mode="quick")
    text = pipeline.calls[-1].get("generation_instruction") or pipeline.calls[-1]["instruction"]
    assert "[결함 원인 질문]" in text and "[용어 대응]" not in text


def test_a_question_that_calls_the_term_by_another_word_gets_the_link_line():
    # v8이 '노점이 뭐야?'에 이슬점 근거를 받고도 '해당 정보가 없습니다'로 답했다(2026-11-21 운영).
    context = term_context(matched_terms("노점이 뭐야?"), "노점이 뭐야?")
    assert "질문의 '노점'은(는) 이슬점과(와) 같은 말입니다" in context
    assert "같은 말입니다" not in term_context(matched_terms("이슬점이 뭐야?"), "이슬점이 뭐야?")
    assert same_word_lines("더스트 스프레이는 왜 생겨?", ["드라이 스프레이"]) and not same_word_lines("팁 막힘은 왜 생겨?", ["스프레이 팁"])
    assert fix_typos("저도막 워인이 뭐야?") == "저도막 원인이 뭐야?"
    for rule in _RULES:
        assert set(rule.get("같은 말", ())) <= set(rule["별칭"]), rule["용어"]
