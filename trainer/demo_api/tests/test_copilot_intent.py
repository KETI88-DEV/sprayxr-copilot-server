from __future__ import annotations

from trainer.demo_api.copilot_intent import classify_question_intent
from trainer.demo_api.pose_query import answer_order_question


def test_llm_can_route_a_pose_tab_question_to_knowledge() -> None:
    calls: list[tuple[str, str]] = []

    def classify(task_type: str, prompt: str) -> str:
        calls.append((task_type, prompt))
        return "knowledge"

    result = classify_question_intent("pose_recomm", "저도막이 무엇이야?", classify)

    assert result == "knowledge"
    assert calls == [("pose_recomm", "저도막이 무엇이야?")]


def test_classifier_failure_falls_back_to_structured_order_intent() -> None:
    def broken(_task_type: str, _prompt: str) -> str:
        raise RuntimeError("model unavailable")

    assert classify_question_intent("trajectory_recomm", "권장 Zone 순서는?", broken) == "order_reference"


def test_recommended_order_answer_prioritizes_action_over_raw_rank_probabilities() -> None:
    report = {
        "analysis_status": "available",
        "recommended_flow": ["7", "4", "8"],
        "first_visit_flow": ["7", "8", "4"],
        "evaluated_visit_count": 3,
        "on_track_visit_count": 3,
        "adherence_rate": 0.95,
        "reference_session_count": 6,
        "best_reference_similarity": 1.0,
        "position_similarity": 0.833,
        "position_distribution": [
            [{"zone_id": "7", "count": 6, "probability": 1.0}],
            [
                {"zone_id": "4", "count": 5, "probability": 0.833},
                {"zone_id": "8", "count": 1, "probability": 0.167},
            ],
            [{"zone_id": "8", "count": 4, "probability": 0.667}],
        ],
        "zone_summary": [],
    }
    found = ("RPT-test", {"sequence_report": report})

    result = answer_order_question(found, prompt="권장 Zone 순서는?", intent="order_reference")

    assert result.answer.startswith("권장 작업 순서는 Zone 7 → 4 → 8입니다.")
    assert "전문가 6회 기준" in result.answer
    assert result.answer.count("Zone 7") == 1
    assert "순번별 주요 분포" not in result.answer
    assert result.context["position_distribution"] == report["position_distribution"]


def test_single_zone_reference_is_not_described_as_a_broken_one_item_order() -> None:
    report = {
        "analysis_status": "available",
        "recommended_flow": ["1"],
        "first_visit_flow": ["1"],
        "evaluated_visit_count": 1,
        "on_track_visit_count": 1,
        "adherence_rate": 1.0,
        "reference_session_count": 5,
        "best_reference_similarity": 1.0,
        "position_similarity": 1.0,
        "position_distribution": [[{"zone_id": "1", "count": 5, "probability": 1.0}]],
        "zone_summary": [],
    }

    result = answer_order_question(("RPT-mode1", {"sequence_report": report}), prompt="권장 순서는?", intent="order_reference")

    assert result.answer == "권장 작업 순서는 Zone 1입니다. 전문가 5회 기준입니다."



def test_domain_knowledge_is_not_forced_into_order_by_a_wrong_classifier_label():
    for question in ("저도막이 뭐야?", "적정 이격거리는?", "과도막의 원인은?"):
        assert classify_question_intent("trajectory_recomm", question, lambda *_: "order_reference") == "knowledge"



def test_coating_advice_keeps_its_question_meaning_even_when_a_zone_is_mentioned():
    from trainer.demo_api.copilot_intent import is_coating_status_question
    for question in ("Zone별 도막을 균일하게 만드는 방법은?", "이 구역에서 과도막이 생기는 원인은 뭐야?",
                     "도막 두께와 이격거리 중 어느 것이 더 중요해?", "지금 도막이 두꺼운데 어떻게 고쳐?"):
        assert not is_coating_status_question(question)
        for task in ("pose_recomm", "trajectory_recomm"):
            bad = "pose_zone" if task == "pose_recomm" else "order_reference"
            assert classify_question_intent(task, question, lambda *_: bad) == "knowledge"


def test_ui_overall_work_similarity_preset_uses_pose_overall_without_model():
    from trainer.demo_api.copilot_intent import preset_intent
    assert preset_intent("pose_recomm", "전체 작업 유사도는?")=="pose_overall"


def test_posture_taxonomy_is_knowledge_even_when_a_pose_label_is_returned():
    from trainer.demo_api.copilot_intent import is_general_painting_knowledge, fallback_intent
    for question in ("자세 종류로는 어떤게있어?", "도장 작업 자세에는 어떤 종류가 있나요?", "무릎을 굽힌 자세의 의미는?"):
        assert is_general_painting_knowledge(question), question
        for task in ("pose_recomm", "trajectory_recomm"):
            assert fallback_intent(task, question) == "knowledge", question
            assert classify_question_intent(task, question, lambda *_: "pose_overall") == "knowledge", question
    for question in ("지금 자세는 어때?", "내 자세 종류는 뭐야?", "Zone 3 자세는 어때?", "전체 자세 유사도는?"):
        assert not is_general_painting_knowledge(question), question


def test_unpainted_and_distribution_questions_are_status_questions():
    from trainer.demo_api.copilot_intent import is_coating_advice_question, is_coating_status_question
    for question in ("미도장 영역이 있어?", "미도장은?", "도막 분포는?", "도막 등급은?", "도막 등급은 어때?"):
        assert is_coating_status_question(question), question
    for question in ("미도장이 뭐야?", "도막 분포 이상 원인은?"):
        assert not is_coating_status_question(question), question
        assert is_coating_advice_question(question), question
    # "정상"은 상태 낱말이 아니다 — 지식 질문이 상태 질문으로 넘어가지 않는다.
    assert not is_coating_status_question("정상 도막 두께는 얼마야?")
    assert not is_coating_advice_question("정상 도막 두께는 얼마야?")


def test_widened_status_words_do_not_capture_rule_and_procedure_questions():
    """넓힌 낱말(미도장·분포·등급)로만 걸리는 규정·절차 질문은 문서 검색에 남는다."""
    from trainer.demo_api.copilot_intent import is_coating_status_question
    for question in ("미도장 부위 보수 절차 알려줘", "ISO 12944 도막 등급은?", "미도장 발생 시 재도장 기준",
                     "도막 두께 분포 규정은?", "도막 두께 등급 기준 알려줘", "도막 두께 분포 기준은?",
                     "미도장 판정 기준은?", "도막 등급 규정"):
        assert not is_coating_status_question(question), question
    # 넓히기 전에도 상태 질문이던 것은 규정 낱말이 있어도 그대로다.
    for question in ("도막 두께가 기준보다 얇은 구역은 어디야?", "지금 도막 상태가 기준에 맞아?"):
        assert is_coating_status_question(question), question


def test_coating_symptoms_and_causal_hypotheses_are_explanations():
    from trainer.demo_api.copilot_intent import is_coating_advice_question, is_coating_status_question

    for question in ("칠한 게 자꾸 눈물처럼 아래로 흘러. 이건 도막이 부족한 거야?",
                     "도막이 아래로 흘러내리는데 이게 저도막이라는 뜻이야?",
                     "도막 얇다고 무조건 희석제 많이 넣어서 그런 거 맞지?"):
        assert is_coating_advice_question(question), question
        assert not is_coating_status_question(question), question

    assert is_coating_status_question("도막이 흘러내린 3번 구역의 실제 도막 상태를 리포트로 알려줘.")


def test_non_spray_process_and_observed_defect_skip_even_mistaken_coating_lookup():
    from trainer.demo_api.tests.test_query_evidence_fallback import DOC, Pipeline, ask, client

    class WrongLookup(Pipeline):
        def plan_copilot_dialogue(self, prompt, **kwargs):
            return {"kind": "lookup", "intent": "coating_status", "scope": "zone"}

    for prompt, required in (
        ("칠한 게 자꾸 눈물처럼 아래로 흘러. 이건 도막이 부족한 거야?", ("처짐", "측정")),
        ("블라스팅 중인데 얇은 도막이면 더 쏴서 160까지 도료를 쌓아야 돼?", ("표면 준비", "적용하지")),
    ):
        pipeline = WrongLookup(response=f"[답변] 문서로 확인한 일반 공정 설명입니다.\n[근거] {DOC}")
        response = ask(client(pipeline), prompt=prompt, mode="quick")
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["coating_context"] is None
        assert not data["retrieval_used"]
        assert all(word in data["answer"] for word in required), data["answer"]
        assert not pipeline.runs and not pipeline.conversations


def test_grinder_alias_and_process_corrections_choose_the_actual_operation():
    from trainer.demo_api.copilot_intent import is_domain_question, is_off_topic
    from trainer.demo_api.query_runtime import explicit_work_process, retrieval_question
    from trainer.demo_api.simulator_context import wants_simulator_context

    question = "아니, 스프레이 말고 그라인더 디스크를 한곳에 멈추고 누르는 얘기야."
    assert explicit_work_process(question) == "grinding"
    assert retrieval_question(question).startswith("선박 표면 처리 그라인딩 작업 ")
    assert not wants_simulator_context(question)
    assert explicit_work_process("그라인딩이 아니라 스프레이로 칠하는 거야.") == "spray"
    assert is_domain_question("그라인더 디스크를 너무 세게 누르면 어떻게 돼?")
    assert not is_off_topic("그라인더 디스크를 너무 세게 누르면 어떻게 돼?")
    assert is_off_topic("허리 디스크 운동은 어떤 게 있어?")


def test_low_film_typo_keeps_the_term_and_its_definition_question():
    from trainer.demo_api.copilot_intent import is_coating_status_question
    from trainer.demo_api.term_rules import fix_typos, matched_terms, meaning_evidence

    question = fix_typos("저도마악이라는 게 두껍게 발랐다는 말이야? 현장 용어 맞아?")
    assert question.startswith("저도막이라는 게")
    assert not is_coating_status_question(question)
    assert meaning_evidence(question, matched_terms(question))


def test_film_stage_comparison_pins_definition_and_measurement_evidence():
    from trainer.demo_api.copilot_intent import is_domain_question
    from trainer.demo_api.term_rules import matched_terms, meaning_evidence

    for question in ("습도막 두께랑 건도막 두께 차이를, 언제 재는지도 같이 알려 줘.",
                     "방금 칠하고 잰 160마이크로랑 마른 뒤 160마이크로랑 같은 판정이야?"):
        assert is_domain_question(question)
        terms = matched_terms(question)
        assert {term["용어"] for term in terms} == {"습도막", "건조도막"}
        assert "DOC-doc_0033-C0020" in meaning_evidence(question, terms)
    assert matched_terms("방금 칠하고 나갔다가 마른 뒤 다시 올게.") == []
