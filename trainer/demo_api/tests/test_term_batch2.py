"""용어 규칙·질문 유형 2차(2026-11-20): 오타 교정, 등록 용어의 문서 경로, 뜻 질문 근거, 종류별 규칙 틀.

운영 3.13.2가 '오렌지필'을 과일 껍질, 'DFT'를 푸리에 변환, '습도막'을 습도가 높은 도막이라고 답했고
'제도막'·'핀흘' 같은 오타에는 근거 묶음을 쓰지 못했다."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from trainer.demo_api.context_prompts import _CONFIG, _TYPES, context_text, matched_prompt
from trainer.demo_api.copilot_intent import is_coating_status_question, is_domain_question, is_general_painting_knowledge
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.term_rules import _CONFIG as TERMS, _RULES, fix_typos, matched_terms, meaning_evidence
from trainer.demo_api.tests.test_query_contextual_dialogue import DialoguePipeline, ask, client

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
CHAT = {"kind": "chat", "intent": "knowledge", "scope": "unspecified"}
FOLLOWUP = {"kind": "coating_knowledge", "intent": "knowledge", "scope": "unspecified", "context_relation": "followup"}


def _named(name):
    return next(entry for entry in _TYPES if entry["이름"] == name)


def _flat(text):
    return re.sub(r"\s+", "", text).casefold()


@pytest.mark.parametrize("typed, fixed", [
    ("저도맥은 왜 생겨?", "저도막은 왜 생겨?"),
    ("제도막 두께 어때?", "제도막 두께 어때?"),                 # '제 도막'을 붙여 쓴 말일 수 있어 고치지 않는다
    ("제도막은 왜 생겨?", "저도막은 왜 생겨?"),                 # 뜻이 분명한 구절은 고친다
    ("과도먹 생기면 어떡해", "과도막 생기면 어떡해"),
    ("적정 이겨거리는?", "적정 이격거리는?"),
    ("오버렙은 얼마나 해야 해?", "오버랩은 얼마나 해야 해?"),
    ("중첩율은 몇 퍼센트가 좋아?", "중첩률은 몇 퍼센트가 좋아?"),
    ("핀흘이랑 오랜지필", "핀홀이랑 오렌지필"),
    ("스프래이 건 에어레스 희석재", "스프레이 건 에어리스 희석제"),
    ("두께가도막에 주는 영향", "두께가도막에 주는 영향"),       # 앞에 한글이 붙은 '가도막'은 다른 낱말의 일부
    ("분사 강도는 어때? 저도 많이 뿌렸어", "분사 강도는 어때? 저도 많이 뿌렸어"),  # 목록에 없는 닮은 말은 건드리지 않는다
    ("저도막이 뭐야?", "저도막이 뭐야?"),
])
def test_listed_typos_are_fixed_and_lookalikes_left_alone(typed, fixed):
    assert fix_typos(typed) == fixed


def test_no_typo_is_itself_a_registered_word():
    words = {_flat(alias) for rule in _RULES for alias in [rule["용어"], *rule["별칭"]]} | {_flat(right) for right in TERMS["오타"]}
    seen = {}
    for right, wrong in TERMS["오타"].items():
        for typo in wrong:
            assert _flat(typo) not in words, typo
            assert seen.setdefault(typo, right) == right, typo


@pytest.mark.parametrize("prompt, expected", [
    ("습도막이 뭐야?", ["습도막"]),
    ("dft 뭐임", ["건조도막"]),                       # 영문 대소문자 무시
    ("신나는 얼마나 넣어?", ["희석제"]),
    ("신나는 노래 틀어줘", []),
    ("노점이 뭐야?", ["이슬점"]),
    ("노점상은 어디 있어?", []),
    ("어깨 처짐이 심해요", []),                        # 자세 이야기의 처짐은 흘림이 아니다
    ("집중도가 뭐야?", []),                           # '중도'·'조도'가 낱말 속에 든 말
    ("습도막 두께와 건조도막 두께는 어떻게 다른가요?", ["습도막", "건조도막"]),  # '건조도막' 속 '조도'는 표면 조도가 아니다
    ("가까이 대고 쏘면 빨리 끝나잖아?", ["이격거리"]),
])
def test_new_terms_and_colloquial_aliases(prompt, expected):
    assert [rule["용어"] for rule in matched_terms(prompt)] == expected


@pytest.mark.parametrize("prompt, pinned", [
    ("습도막이 뭐야?", 3), ("핀홀", 3), ("DFT가 뭐야?", 3), ("미도장 부위는 정확히 어떤 상태야?", 3),
    ("이격거리는 얼마가 적당해?", 0), ("흘림은 왜 생겨?", 0), ("신나는 얼마나 넣어?", 0),
])
def test_meaning_questions_get_the_terms_evidence(prompt, pinned):
    assert len(meaning_evidence(prompt, matched_terms(prompt))) == pinned


@pytest.mark.parametrize("prompt, previous, expected", [
    ("흘림은 왜 생겨?", [], "흘림 원인"),
    ("도료가 흘러내리는 이유는?", [], "흘림 원인"),
    ("핀홀은 어떻게 막아?", [], "핀홀 대책"),
    ("그럼 어떻게 막아?", ["핀홀이 뭐야?", "원인이 뭔데?"], "핀홀 대책"),          # 새 결함도 꼬리물기를 잇는다
    ("핀홀이 생기면 어떤 문제가 있어?", [], None),                              # 영향 질문은 원인 묶음을 쓰지 않는다
    ("핀홀과 기포는 왜 생겨?", [], None),                                     # 결함 둘은 좁히지 않는다
    ("습도막 게이지는 어떻게 써?", [], "도막 두께 측정"),
    ("습도막 두께로 건조도막 두께를 어떻게 계산해?", [], "습도막·건도막 환산"),
    ("습도막이 뭐야?", [], None),                                            # 뜻 질문은 용어 근거가 맡는다
    ("방아쇠는 언제 당기고 언제 놓아야 해?", [], "분사 시작·멈춤"),
    ("가까이 대고 쏘면 빨리 끝나잖아?", [], "분사 거리 영향"),
    ("가까이서 보면 어때?", [], None),                                        # 도장 낱말이 없으면 거리 유형이 아니다
    ("어깨 처짐 원인은?", [], None),                                          # 유형도 용어의 제외어를 따른다
    ("백화점 왜 이렇게 붐벼?", [], None),
    ("용접 기포 원인은?", [], None),
    ("손 씻는 건 왜 해야 해?", [], None),                                     # '것'의 준말 '건'은 스프레이 건이 아니다
    ("겨울에 감기 걸리는 건 왜야?", [], None),
    ("골프 초보 요령 알려줘", [], None),
    ("비 와도 그냥 칠하면 안 돼?", [], "우천 도장"),
    ("오늘 비 와?", [], None),
    ("두껍게 칠하면 더 좋은 거 아냐?", [], "과도막 영향"),
    ("왜 자꾸 얇게 나와?", [], "저도막 원인"),
    ("저도막 생기면 어떡해?", [], "저도막 대책"),                               # '어떡해'도 대책을 묻는 말
    ("신나는 얼마나 넣어?", [], "희석"),
    ("대충 칠해도 티 안 나지 않아?", [], "기본 요령"),
    ("저도막을 막으려면 이격거리를 어떻게 해야 해?", [], "저도막 대책"),            # 결함 유형이 안내 유형보다 먼저다
    ("그러면 어떻게 해야 해?", ["저도막이 뭐야?", "비 와도 칠해도 돼?"], None),     # 안내 유형의 주제도 이어받기의 경계다
    ("이격거리는 얼마가 적당해?", [], "적정 분사 거리"),
    ("몇 센치 띄워?", [], "적정 분사 거리"),
    ("중첩은 몇 퍼센트로 해야 돼요?", [], "적정 중첩률"),
    ("마스크 안 써도 되지?", [], "방독 마스크"),
    ("이격거리는 가까울수록 좋지?", [], "분사 거리 영향"),
    ("다 마른 건지 어떻게 알아?", [], "건조 확인"),
    ("몇 번 칠해야 돼?", [], "칠하는 순서·방향"),
    ("건 이동 속도는 얼마가 적당해?", [], "적정 이동 속도"),
    ("블라스팅 연마재 분사속도는 얼마야?", [], None),
    ("충전전로 최소 이격거리는?", [], "충전전로 최소 이격거리"),
    ("블라스팅 노즐 이격거리는 얼마가 적당해?", [], None),                         # 블라스팅 거리는 분사 거리 묶음을 쓰지 않는다
    ("젖은 도막 두께는 왜 재?", [], "습도막 측정 이유"),
    ("도막 표면이 거칠게 나오는 이유는?", [], "표면 거칠음"),
    ("건은 어떻게 잡아?", [], "건 운행"),
    ("뭐부터 해야 돼?", [], None),                                            # 첫 순서 질문은 검색에 맡긴다
    ("도장 자격에 나이 제한이 있어?", [], "도장 자격 나이"),                       # 일반 설명이 검토에서 거절되어 503이던 질문(2026-11-23)
    ("도장 자격증이 꼭 필요해?", [], "도장 자격증"),
    ("우리 팀장님 나이가 몇 살이야?", [], None),
    ("추운 날 도장하면 어떻게 돼?", [], "저온 도장"),                            # 저온만 물으면 고온 결함을 섞지 않게 따로 받는다
    ("더운 날 도장하면 어떻게 돼?", [], "저온·고온 도장"),
    ("여름이랑 겨울 중 언제 칠하기 좋아?", [], "저온·고온 도장"),
])
def test_new_question_types(prompt, previous, expected):
    found = matched_prompt(prompt, previous)
    assert (found["이름"] if found else None) == expected


def test_config_text_has_no_hanzi():
    # 맥락에 한자가 있으면 모델이 한자를 따라 쓰고, 답의 그 문장이 통째로 빠진다(dialogue.without_hanzi).
    for name in ("term_rules.json", "context_prompts.json"):
        assert not re.search(r"[\u3400-\u9fff]", (CONFIGS / name).read_text(encoding="utf-8")), name


def test_every_type_has_filled_rules_and_real_distinct_chunks():
    corrected = set(json.loads((CONFIGS / "evidence_corrections.json").read_text(encoding="utf-8"))["조각"])
    assert set(_CONFIG["규칙 틀"]) == {"원인", "대책", "안내"}
    for entry in _TYPES:
        rule = context_text(entry)
        assert "{" not in rule and "}" not in rule, entry["이름"]
        assert len(entry["조각"]) == len(set(entry["조각"])) and all(re.fullmatch(r"DOC-doc_\d{4}-C\d{4}", c) for c in entry["조각"]), entry["이름"]
    assert "질문한 결함: 핀홀" in context_text(_named("핀홀 원인")) and "문서별로 구분해" in context_text(_named("오렌지필 원인"))
    assert "피부를 시너로 닦지 말라" in context_text(_named("응급조치"))
    assert "한 문장입니다" in context_text(_named("흘림 대책")) and "답을 거절하지 마세요" in context_text(_named("핀홀 대책"))
    assert corrected  # 고친 조각은 서버가 고친 본문으로 붙인다(test_context_prompts 참고)


@pytest.mark.parametrize("prompt, domain", [
    ("오렌지필이 뭐야?", True), ("DFT가 뭐야?", True), ("노점이 뭐야?", True), ("신너는 얼마나 넣어야 해?", True), ("핀홀은 왜 생겨?", True),
    # 구어와 안내 유형은 경로 판정에 쓰지 않는다. 뷰어의 기록 대화와 잡담이 문서 경로로 새지 않게 한다.
    ("비 와도 그냥 칠하면 안 돼?", False), ("가까이 대고 쏘면 빨리 끝나잖아?", False), ("빨래가 안 말라 왜?", False), ("안경이 뿌옇게 되는 이유", False),
    ("오늘 비 와?", False), ("신나는 노래 틀어줘", False), ("어깨 처짐이 심해요", False), ("점심 뭐 먹지?", False), ("자세 유사도는 얼마야?", False),
    ("백화점 왜 이렇게 붐벼?", False), ("징크스가 뭐야?", False), ("버블티가 뭐야?", False), ("트리거 포인트가 뭐야?", False),
    ("도포 방향은 어떻게 돼?", False),                # 2026-11-18의 용어는 예전 경로 그대로(순서 탭 질문)
])
def test_registered_formal_terms_are_painting_questions(prompt, domain):
    assert is_domain_question(prompt) is domain


@pytest.mark.parametrize("prompt, pinned", [("징크스가 뭐야?", 0), ("곰보빵이 뭐야?", 0), ("버블티가 뭐야?", 0), ("흘림 생기면 뭐가 문제야?", 0)])
def test_everyday_words_get_no_meaning_evidence(prompt, pinned):
    assert len(meaning_evidence(prompt, matched_terms(prompt))) == pinned


@pytest.mark.parametrize("prompt, back", [
    ("가까이 대고 쏘면 빨리 끝나잖아?", True), ("비 와도 그냥 칠하면 안 돼?", True), ("건은 어떻게 잡아?", True), ("페인트 피부에 묻으면 어떡해?", True),
    ("오렌지필이 뭐야?", True),
    ("빨래가 안 말라 왜?", False), ("안경이 뿌옇게 되는 이유", False), ("맥주 거품 왜 생겨?", False), ("오늘 비 와?", False), ("가까이서 보면 어때?", False),
])
def test_chat_decisions_are_sent_back_to_documents_only_with_a_painting_word(prompt, back):
    from trainer.demo_api.copilot_router import _registered_knowledge
    assert _registered_knowledge(prompt) is back


def test_definition_of_a_registered_term_skips_the_planner_and_status_lookup():
    assert is_general_painting_knowledge("오렌지필이 뭐야?") and is_general_painting_knowledge("DFT가 뭐야?")
    assert not is_coating_status_question("미도장 부위는 정확히 어떤 상태야?")  # 뜻을 묻는 말이지 내 기록의 상태가 아니다
    assert is_coating_status_question("미도장 부위가 어디야?")


@pytest.mark.parametrize("mode", ["quick", "detailed"])
def test_router_answers_a_registered_term_from_documents_even_when_the_planner_says_chat(mode):
    pipeline = DialoguePipeline(CHAT)
    result = ask(client(pipeline), "오렌지필이 뭐야?", response_mode=mode)
    call = pipeline.calls[-1]
    assert result["retrieval_used"] is True and not pipeline.conversations
    assert call["pinned_chunk_ids"] == next(rule["근거"][:3] for rule in _RULES if rule["용어"] == "오렌지필")
    assert "과일 껍질 이야기가 아니라" in (call.get("generation_instruction") or call["instruction"])
    assert "Orange Peel" in call["retrieval_instruction"]


@pytest.mark.parametrize("mode", ["quick", "detailed"])
def test_router_fixes_the_typo_for_analysis_and_keeps_the_typed_words_in_history(mode):
    store = SequenceStore(":memory:")
    pipeline = DialoguePipeline()
    ask(client(pipeline, store), "저도맥은 왜 생겨?", response_mode=mode)
    call = pipeline.calls[-1]
    assert call["pinned_chunk_ids"] == _named("저도막 원인")["조각"] and "저도막은 왜 생겨?" in call["instruction"]
    assert store.recent_query_turns("CHAT:U", user_id="U")[-1]["prompt"] == "저도맥은 왜 생겨?"


def test_router_inherits_the_defect_from_a_mistyped_earlier_question():
    store = SequenceStore(":memory:")
    store.record_query_turn(session_key="CHAT:U", user_id="U", prompt="핀흘이 뭐야?", answer="이전 답", task_type="knowledge_search",
                            created_at="2026-11-20T05:05:01Z", answer_source="documents")
    pipeline = DialoguePipeline(FOLLOWUP)
    ask(client(pipeline, store), "원인이 뭔데?", response_mode="detailed")
    assert pipeline.calls[-1]["pinned_chunk_ids"] == _named("핀홀 원인")["조각"]


def test_router_grounds_a_naive_premise_question_the_planner_calls_chat():
    pipeline = DialoguePipeline(CHAT)
    pipeline.config = {"confidence": {"abstain_below_rerank_score": 0.1, "quick_abstain_below_rerank_score": 0.1}}  # 근거 확인 단계가 켜진 운영 설정
    ask(client(pipeline), "가까이 대고 쏘면 빨리 끝나잖아?", response_mode="quick")
    call = pipeline.calls[-1]
    assert call["pinned_chunk_ids"] == _named("분사 거리 영향")["조각"] and not pipeline.conversations
    assert "바로잡으세요" in (call.get("generation_instruction") or call["instruction"])


def test_router_leaves_small_talk_to_the_conversation_model():
    pipeline = DialoguePipeline(CHAT)
    ask(client(pipeline), "오늘 비 와?", response_mode="quick")
    assert pipeline.conversations and not pipeline.calls


def _answered(store, prompt, source="documents"):
    store.record_query_turn(session_key="CHAT:U", user_id="U", prompt=prompt, answer="이전 답", task_type="knowledge_search",
                            created_at="2026-11-20T05:05:01Z", answer_source=source)


@pytest.mark.parametrize("earlier, prompt, asked", [
    (["저도막이 뭐야?"], "쉽게 다시 설명해줘", "저도막이 뭐야? (쉽게 다시 설명해줘)"),
    (["저도막이 뭐야?", "쉽게 다시 설명해줘"], "한 줄로 말해줘", "저도막이 뭐야? (한 줄로 말해줘)"),   # 되물음 위의 질문까지 거슬러 간다
    (["흘림이 뭐야?"], "뭔 소리야", "흘림이 뭐야? (쉽게 다시 설명해 주세요)"),
    (["이격거리는 얼마가 적당해?"], "그게 몇 센치야?", "이격거리는 얼마가 적당해? (그게 몇 센치야?)"),
    (["습도막 두께와 건조도막 두께는 어떻게 다른가요?"], "둘은 어떻게 재?", "습도막 두께와 건조도막 두께는 어떻게 다른가요? (둘은 어떻게 재?)"),
])
def test_a_request_to_explain_again_reasks_the_document_question(earlier, prompt, asked):
    store = SequenceStore(":memory:")
    for text in earlier:
        _answered(store, text)
    pipeline = DialoguePipeline(CHAT)
    ask(client(pipeline, store), prompt, response_mode="quick")
    assert pipeline.calls and asked in pipeline.calls[-1]["instruction"] and not pipeline.conversations
    assert store.recent_query_turns("CHAT:U", user_id="U")[-1]["prompt"] == prompt


@pytest.mark.parametrize("earlier, source, prompt", [
    ([], "documents", "쉽게 설명해줘"),                          # 앞 질문이 없다
    (["점심 뭐 먹지?"], "conversation", "쉽게 다시 설명해줘"),      # 앞 답이 문서 답이 아니다
    (["저도막이 뭐야?"], "documents", "다시 칠하려면 어떻게 해?"),   # 새 질문이다
    (["저도막이 뭐야?"], "documents", "그거 말고 과도막은?"),
    (["저도막이 뭐야?"], "documents", "그건 그렇고 오늘 점심 뭐야?"),  # 화제 전환
    (["저도막이 뭐야?"], "documents", "그거 고마워"),
    (["저도막이 뭐야?"], "documents", "둘 다 틀렸어"),
    (["저도막이 뭐야?"], "documents", "그게 이번에 몇 cm였어?"),      # 내 기록을 묻는 말
])
def test_other_followups_are_not_restated(earlier, source, prompt):
    from trainer.demo_api.copilot_router import _restated_followup
    assert _restated_followup(prompt, [{"prompt": text, "answer_source": source} for text in earlier]) is None


def test_viewer_question_about_my_own_run_keeps_the_record_numbers():
    from trainer.demo_api.tests.viewer_record import AT_3S, context, payload
    viewer = context("knowledge_search", payload(), AT_3S)
    general = DialoguePipeline()
    ask(client(general), "건을 너무 빨리 움직이면 어떻게 돼?", response_mode="detailed", context=viewer)
    assert "[도장 지식 질문: 이동 속도 영향]" in general.calls[-1]["instruction"] and "\"distance_mm\"" not in general.calls[-1]["instruction"]
    mine = DialoguePipeline()
    ask(client(mine), "지금 너무 빨리 움직였어?", response_mode="detailed", context=viewer)
    assert all("[도장 지식 질문" not in str(call.get("instruction", "")) for call in mine.calls)


@pytest.mark.parametrize("prompt", ["마이크로미터가 뭐야?", "마이크론이 뭐야?", "µm가 뭐야?"])
def test_the_thickness_unit_is_explained_from_the_documents(prompt):
    # 운영 3.13.3 상세 분석이 측정 공구로 답했다가 검토에서 거절돼 503 이 났다.
    rules = matched_terms(prompt)
    assert [rule["용어"] for rule in rules] == ["마이크로미터"] and "단위" in rules[0]["뜻"]
    assert meaning_evidence(prompt, rules) == ["DOC-doc_0190-C0145", "DOC-doc_0045-C0180"]


def test_a_viewer_thickness_question_with_the_unit_stays_a_record_question():
    from trainer.demo_api.tests.viewer_record import AT_3S, context, payload
    viewer = context("knowledge_search", payload(), AT_3S)
    with_unit, plain = DialoguePipeline(), DialoguePipeline()
    first = ask(client(with_unit), "Zone 8 두께 몇 마이크로미터야?", response_mode="quick", context=viewer)
    second = ask(client(plain), "Zone 8 두께 몇이야?", response_mode="quick", context=viewer)
    assert first["answer"] == second["answer"] and "µm" in first["answer"]
    assert len(with_unit.calls) == len(plain.calls) and not any(call.get("pinned_chunk_ids") for call in with_unit.calls)


# TSB 연동 단위 표(2026-08): 위치 m, 이격거리 mm, 속도 mm/s, 각도 °, 도막 µm, 압력 bar, 넓이 m², 도료량 L, 비율 %, 온도 ℃, 텍스처 px, 누적 시간 s, 간격 ms
@pytest.mark.parametrize("prompt, unit", [
    ("m가 뭐야?", "미터(m)"), ("mm가 뭐야?", "밀리미터(mm)"), ("mm/s가 뭐야?", "mm/s"), ("°가 무슨 뜻이야?", "도(°)"),
    ("마이크로미터가 뭐야?", "마이크로미터(µm)"), ("bar가 뭐야?", "바(bar)"), ("m2가 뭐야?", "제곱미터(m²)"), ("L가 뭐야?", "리터(L)"),
    ("%가 뭐야?", "퍼센트(%)"), ("℃가 뭐야?", "섭씨(℃)"), ("px가 뭐야?", "픽셀(px)"), ("s가 무슨 단위야?", "초(s)"), ("ms가 뭐야?", "밀리초(ms)"),
    ("150 bar는 몇 MPa야?", "메가파스칼(MPa)"), ("1마이크론은 몇 mm야?", "마이크로미터(µm)"),
])
def test_every_unit_of_the_simulator_table_has_a_meaning(prompt, unit):
    from trainer.demo_api.term_rules import unit_facts
    assert unit in unit_facts(prompt)


@pytest.mark.parametrize("prompt", ["이격거리 300mm 유지해", "Zone 8 두께 몇이야?", "similar family milk items", "저도막이 뭐야?", "몇 도로 기울여야 해?"])
def test_unit_meanings_are_given_only_when_a_unit_is_asked_about(prompt):
    from trainer.demo_api.term_rules import unit_facts
    assert unit_facts(prompt) == {}


def test_a_general_explanation_gets_the_unit_meaning_as_a_fact():
    # 운영 3.13.3: "1마이크론은 몇 mm야?" 의 답이 근거 없는 수치로 지워져 "세션 기록으로는 확인할 수 없습니다."만 남았다.
    pipeline = DialoguePipeline(CHAT)
    ask(client(pipeline), "bar가 뭐야?", response_mode="quick")
    assert any("0.1MPa" in json.dumps(kwargs.get("facts"), ensure_ascii=False) for _, kwargs in pipeline.conversations)
    mine = DialoguePipeline(CHAT)
    ask(client(mine), "내가 몇 bar로 했어?", response_mode="quick")
    assert not any("0.1MPa" in json.dumps(kwargs.get("facts"), ensure_ascii=False) for _, kwargs in mine.conversations)
