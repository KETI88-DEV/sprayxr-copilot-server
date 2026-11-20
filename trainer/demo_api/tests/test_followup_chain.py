"""결함 꼬리물기(2026-11-19): 뜻 → 원인 → 대책 → 내 기록과 비교. 대상은 대상이 적힌 가장 가까운 질문에서 이어받는다."""
from __future__ import annotations

import pytest

from trainer.demo_api.context_prompts import _TYPES, context_text, matched_prompt, recent_subject
from trainer.demo_api.record_query import _MINE_COMPARE, answer_record_question
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.tests.test_query_contextual_dialogue import DialoguePipeline, ask, client
from trainer.demo_api.tests.viewer_record import AT_3S, AT_24S, context, payload

FOLLOWUP = {"kind": "coating_knowledge", "intent": "knowledge", "scope": "unspecified", "context_relation": "followup"}


def _named(name):
    return next(entry for entry in _TYPES if entry["이름"] == name)


@pytest.mark.parametrize("prompt, previous, expected", [
    ("그러면 어떻게 해야 하는데?", ["저도막이 뭐야?", "원인이 뭔데?"], "저도막 대책"),   # 대상은 두 턴 앞에 있다
    ("그러면 어떻게 해야 해?", ["과도막이 뭐야?", "그럼 왜 생겨?"], "과도막 대책"),
    ("그럼 어떤 문제가 생겨?", ["과도막이 뭐야?", "그럼 왜 생겨?", "그러면 어떻게 해야 해?"], "과도막 영향"),
    ("저도막을 막으려면 어떻게 해야 해?", [], "저도막 대책"),
    ("저도막 문제를 해결하려면?", [], "저도막 대책"),                  # '문제'가 있어도 대책 질문이다
    ("과도막이 안 생기게 하려면?", [], "과도막 대책"),                 # '생기'가 있어도 대책 질문이다
    ("미도장은 왜 생겨?", [], "미도장 원인"),
    ("그럼 어떻게 보완해?", ["미도장이 뭐야?"], "미도장 대책"),
    ("그러면 어떻게 해야 하는데?", ["저도막이 뭐야?", "이격거리는 얼마가 적당해?"], None),  # 가장 가까운 질문의 주제를 따른다
    ("그러면 어떻게 해야 하는데?", ["저도막과 과도막 차이는?", "원인이 뭔데?"], None),     # 대상이 둘이면 좁히지 않는다
])
def test_the_chain_keeps_the_nearest_defect(prompt, previous, expected):
    found = matched_prompt(prompt, previous)
    assert (found["이름"] if found else None) == expected


def test_remedy_rules_name_the_defect_its_document_wording_and_no_borrowing():
    for name, defect in (("저도막 대책", "저도막"), ("과도막 대책", "과도막"), ("미도장 대책", "미도장")):
        rule = context_text(_named(name))
        assert "[결함 대책 질문]" in rule and f"질문한 결함: {defect}" in rule and "같은 뜻이므로" in rule
        assert "대책으로 옮기지 마세요" in rule and "근거에 없는 방법이나 수치는" in rule


def test_every_remedy_rule_caps_the_list_at_five_methods():
    # Qwen3.8 원본이 대책을 33개 넘게 되풀이했다(2026-11-20 운영 교체 시험).
    remedies = [entry for entry in _TYPES if entry["이름"].endswith("대책")]
    assert len(remedies) >= 16 and all("5개 이내로" in context_text(entry) for entry in remedies)


def test_router_inherits_the_defect_across_two_followups():
    store = SequenceStore(":memory:")
    for at, prompt in (("05:05:01", "저도막이 뭐야?"), ("05:05:20", "원인이 뭔데?")):
        store.record_query_turn(session_key="CHAT:U", user_id="U", prompt=prompt, answer="이전 답", task_type="knowledge_search",
                                created_at=f"2026-11-19T{at}Z", answer_source="documents")
    for mode in ("quick", "detailed"):
        pipeline = DialoguePipeline(FOLLOWUP)
        ask(client(pipeline, store), "그러면 어떻게 해야 하는데?", response_mode=mode)
        assert pipeline.calls[-1]["pinned_chunk_ids"] == _named("저도막 대책")["조각"], mode
        assert "[결함 대책 질문]" in pipeline.calls[-1]["instruction"], mode


@pytest.mark.parametrize("prompt", ["지금 내가 한 거랑 비교하면?", "내 기록이랑 비교해 줘", "제가 칠한 것과 비교하면 어때요?", "숙련자랑 비교하면?"])
def test_mine_compare_phrases(prompt):
    assert _MINE_COMPARE.search(prompt)


@pytest.mark.parametrize("prompt", ["저도막과 과도막을 비교해 줘", "저도막과 비교하면 과도막은?", "에어리스와 에어 스프레이 비교"])
def test_knowledge_comparisons_are_not_record_comparisons(prompt):
    assert not _MINE_COMPARE.search(prompt)


def _chain(*prompts):
    return [{"prompt": prompt, "answer": "이전 답", "answer_source": "documents"} for prompt in prompts]


def test_compare_lists_the_defect_zones_playback_first_then_worst():
    found = answer_record_question(context("knowledge_search", payload(), AT_24S), "지금 내가 한 거랑 비교하면?", task_type="knowledge_search",
                                   recent_turns=_chain("저도막이 뭐야?", "원인이 뭔데?", "그러면 어떻게 해야 하는데?"))
    assert found is not None and found.kind == "cause" and found.subject["zone_ids"] == ["8", "2", "5"]  # 재생 위치 Zone 1은 정상
    text = found.text
    assert text.index("Zone 8") < text.index("Zone 2") < text.index("Zone 5")
    assert "77" in text and "거리로는 설명하기 어렵" in text and "그 밖의 저도막 Zone" in text and "세션이 끝났을 때" in text


def test_compare_puts_the_playback_zone_first_when_it_has_the_defect():
    found = answer_record_question(context("knowledge_search", payload(), AT_3S), "지금 내가 한 거랑 비교하면?", task_type="knowledge_search",
                                   recent_turns=_chain("저도막이 뭐야?"))
    assert found.subject["zone_ids"][0] == "8"


def test_compare_says_when_the_record_has_no_zone_of_that_defect():
    found = answer_record_question(context("knowledge_search", payload(), AT_3S), "내 기록이랑 비교하면?", task_type="knowledge_search",
                                   recent_turns=_chain("과도막이 뭐야?", "그럼 왜 생겨?"))
    assert found.text.startswith("이 기록에는 과도막 Zone이 없습니다.") and found.subject["zone_ids"] == []


def test_compare_without_a_defect_uses_the_playback_zone():
    found = answer_record_question(context("knowledge_search", payload(), AT_3S), "지금 내가 한 거랑 비교하면?", task_type="knowledge_search",
                                   recent_turns=_chain("이격거리는 얼마가 적당해?"))
    assert found.subject["zone_ids"] == ["8"]


def test_compare_needs_a_signed_record():
    assert answer_record_question(None, "지금 내가 한 거랑 비교하면?", task_type="knowledge_search", recent_turns=_chain("저도막이 뭐야?")) is None


def test_recent_subject_takes_the_nearest_single_defect():
    assert recent_subject(["저도막이 뭐야?", "원인이 뭔데?"]) == "저도막"
    assert recent_subject(["저도막이 뭐야?", "과도막은?"]) == "과도막"
    assert recent_subject(["저도막과 과도막 차이"]) is None
    assert recent_subject(["이격거리는?"]) is None


def test_detailed_viewer_type_question_gets_no_record_numbers_but_other_questions_keep_them():
    # '과도막' 낱말이 든 질문은 뷰어에서 도막 조언 경로로 간다. 기권한 것은 낱말 없는 꼬리 질문이었다.
    viewer = context("knowledge_search", payload(), AT_3S)
    store = SequenceStore(":memory:")
    store.record_query_turn(session_key="CHAT:U", user_id="U", prompt="과도막이 뭐야?", answer="이전 답", task_type="knowledge_search",
                            created_at="2026-11-19T05:05:01Z", answer_source="documents")
    typed = DialoguePipeline(FOLLOWUP)
    ask(client(typed, store), "그럼 왜 생겨?", response_mode="detailed", context=viewer)
    assert "[결함 원인 질문]" in typed.calls[-1]["instruction"] and "\"distance_mm\"" not in typed.calls[-1]["instruction"]
    plain = DialoguePipeline()
    ask(client(plain), "도장 품질은 어떻게 관리해?", response_mode="detailed", context=viewer)
    assert "\"distance_mm\"" in plain.calls[-1]["instruction"]  # 유형 밖 질문은 지금처럼 기록 수치를 받는다
