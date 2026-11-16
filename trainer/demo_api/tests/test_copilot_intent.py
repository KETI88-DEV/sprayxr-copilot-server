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
    assert "우선 Zone 7에서 시작한 뒤 Zone 4로 이동하세요." in result.answer
    assert "이 시작 순서는 전문가 작업의 약 83%에서 동일하게 나타났습니다." in result.answer
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

    assert result.answer == "학습모드 1은 Zone 1 한 곳으로 구성되어 있으며, 전형기 5회 모두 Zone 1부터 작업했습니다."



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
