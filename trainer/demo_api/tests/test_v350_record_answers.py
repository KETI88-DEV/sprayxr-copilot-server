import re

import pytest

from trainer.demo_api.dialogue import authenticated_facts
from trainer.demo_api.record_query import RecordFacts, RecordPart, answer_part, one_priority
from trainer.demo_api.tests.viewer_record import AT_3S, AT_24S, AT_35S, context, payload
from trainer.demo_api.viewer_context import sign_context, viewer_signatures
from trainer.demo_api.viewer_progress import order_progress_facts, viewer_progress_answer


def facts(playback=AT_3S, data=None):
    return RecordFacts.from_context(context("knowledge_search", data or payload(), playback))


def part(code, zones=(), operator=None, metric=None, cardinality="any"):
    return RecordPart(text="", code=code, zones=zones, zone_source="asked" if zones else "none", operator=operator,
                      metric=metric, cardinality=cardinality, premises=(), knowledge=False)


ANSWERS = [  # (이름, part, playback, 꼭 있어야 할 낱말, 없어야 할 정규식)
    ("R1 argmin", part("coating_extreme", operator="argmin"), AT_35S, ["Zone 8", "77.2", "160", "Zone 2", "84.3", "Zone 5", "89.6"], r"Zone 9"),
    ("R1 argmax", part("coating_extreme", operator="argmax"), AT_3S, ["Zone 1", "161.7"], None),
    ("R2", part("coating_over"), AT_24S, ["0곳", "Zone 1", "10.1%", "Zone 2", "1.3%", "상한(240µm)"], r"1\.5배"),  # 기록의 범위(160~240µm)를 읽는다
    ("R3 all", part("completion"), AT_35S, ["9", "95%", "0곳"], r"완료된 상태|끝났습니다"),
    ("R3 zone 1", part("completion", ("1",)), AT_24S, ["Zone 1", "162µm", "160~240µm", "안이지만", "47.9%", "95%", "보완"], r"정상(?!\s*범위)"),
    ("R3 zone 8", part("completion", ("8",)), AT_35S, ["Zone 8", "1.9%", "95%"], None),
    ("R4", part("distance_compare", ("8",)), AT_3S, ["Zone 8", "251 mm", "550 mm", "54%", "가까운"], None),
    ("R4 similar", part("distance_compare", ("7",)), AT_3S, ["Zone 7", "비슷"], None),
    ("R5 far", part("distance_filter", operator="higher"), AT_24S, ["Zone 4", "330"], r"없습니다"),
    ("R6", part("speed", ("8",)), AT_3S, ["2,230 mm/s", "3,514", "이전 장비", "더 얇아"], r"올리세요|높이세요|맞추세요"),
    ("R7", part("work_metric", metric="tool_angle"), AT_3S, ["9개 Zone 모두", "100%", "5.2°", "3.1°"], None),
    ("R8", part("pose_judgement"), AT_3S, ["54%", "59%", "50%", "참고용", "70%"], r"나쁜|정상|보통 수준|잘하고"),
    ("R9 start", part("order_shape", operator="start"), AT_3S, ["Zone 7", "6회", "Zone 9", "3회", "Zone 1", "2회", "Zone 3", "1회"], None),
    ("R9 allowed", part("order_shape", operator="allowed"), AT_3S, ["Zone 7", "6회 모두", "Zone 9", "45%"], None),
    ("R9 adherence", part("order_shape", operator="adherence"), AT_3S, ["45.1%", "55.6%", "20.7%", "6회"], None),
    ("R9 agreement", part("order_shape", operator="agreement"), AT_3S, ["기록에 없", "6회 중 6회", "6회 중 5회"], None),
    ("R10", part("progress_status", ("10", "11")), AT_3S, ["Zone 10", "11", "1~9"], r"안 칠"),
    ("R11", part("external_standard"), AT_35S, ["판정할 수 없", "160", "DFT", "8곳", "0곳"], r"합격입니다"),
    ("C zone 8", part("cause", ("8",)), AT_3S, ["Zone 8", "77", "251 mm", "550 mm", "가까", "거리로는 설명하기 어렵", "2,230 mm/s"], r"멀어|멀리서"),
]


@pytest.mark.parametrize("name,record_part,playback,needed,forbidden", ANSWERS, ids=[a[0] for a in ANSWERS])
def test_deterministic_record_answers(name, record_part, playback, needed, forbidden) -> None:
    text, _ = answer_part(facts(playback), record_part, task_type="knowledge_search")
    for word in needed:
        assert word in text, (word, text)
    assert forbidden is None or not re.search(forbidden, text), text


def test_one_priority_picks_the_lowest_coverage_zone_only() -> None:
    single = one_priority(facts(), single=True)
    assert "Zone 3" in single and "95%" in single
    assert not any(word in single for word in ("팔꿈치", "Zone 6", "Zone 8")), single
    both = one_priority(facts(), single=False)
    assert both.index("Zone 3") < both.index("Zone 8"), both


def test_progress_codes_reuse_viewer_progress() -> None:
    value = context("trajectory_recomm", payload(), AT_3S)
    record = RecordFacts.from_context(value)
    for code, kind, prompt in (("progress_next", "next", "다음엔 어디 칠해야 돼?"), ("progress_status", "status", "빼먹은 데 있어?")):
        assert answer_part(record, part(code), task_type="trajectory_recomm") == viewer_progress_answer(value, prompt, kind)


def test_coating_answers_without_a_target_say_the_data_is_missing() -> None:
    data = payload()
    data["coating_evaluation"]["target"] = {"analysis_status": "not_available", "reason_code": "height_map_unavailable"}
    data["query_context_signatures"] = viewer_signatures(data)
    data["query_context_signatures"]["arm_pose"] = sign_context("arm_pose", data["arm_pose_evaluation"])
    record = facts(data=data)
    for record_part in (part("coating_extreme", operator="argmin"), part("completion"), part("completion", ("8",)),
                        part("coating_over"), part("external_standard"), part("cause", ("8",))):
        text, _ = answer_part(record, record_part, task_type="knowledge_search")
        assert not re.search(r"\d\s*µm", text), text
        assert "판정 자료가 없" in text, text


def test_order_progress_facts_carry_distribution_but_model_facts_do_not() -> None:
    data = payload()
    carried = order_progress_facts(data["reference_order_evaluation"])
    assert "position_distribution" in carried and carried["reference_session_count"] == 6
    assert "position_distribution" not in authenticated_facts(context("knowledge_search", data, AT_3S))


def test_zone_sentence_uses_the_production_grade_phrase() -> None:
    from trainer.demo_api.coating_assessment import grade_phrase
    record = facts()
    for zone, codes in (("1", ("cause",)), ("8", ("completion", "cause"))):  # Zone 1 완료 답은 정상+보완 모순을 피해 대비문이다
        row = next(r for r in record.target_rows if str(r["zone_id"]) == zone)
        for code in codes:
            text = answer_part(record, part(code, (zone,)), task_type="knowledge_search")[0]
            assert grade_phrase(row) in text, (grade_phrase(row), text)


def test_normal_grade_with_needs_more_gate_reads_as_a_contrast() -> None:
    text, _ = answer_part(facts(AT_24S), part("completion", ("1",)), task_type="knowledge_search")
    assert not re.search(r"정상(?!\s*범위)", text) and "안이지만" in text and "보완(더 칠하기)이 필요" in text, text


def test_missing_zone_uses_the_right_particle() -> None:
    assert "Zone 10, 11은" in answer_part(facts(), part("progress_status", ("10", "11")), task_type="knowledge_search")[0]
    assert "Zone 12는" in answer_part(facts(), part("completion", ("12",)), task_type="knowledge_search")[0]


def _resigned(mutate):
    data = payload()
    mutate(data)
    data["query_context_signatures"] = viewer_signatures(data)
    data["query_context_signatures"]["arm_pose"] = sign_context("arm_pose", data["arm_pose_evaluation"])
    return facts(data=data)


def _all_complete(data) -> None:
    for row in data["coating_evaluation"]["target"]["zones"]:
        row["gate_status"] = "complete"


def test_one_priority_falls_back_to_the_arm_pose_zone() -> None:
    text = one_priority(_resigned(_all_complete), single=True)
    assert "Zone 9" in text and "팔 자세" in text and "50%" in text and "팔꿈치" in text, text


def test_one_priority_falls_back_to_work_similarity() -> None:
    def mutate(data) -> None:
        _all_complete(data)
        data["arm_pose_evaluation"]["zones"] = {}
    record = _resigned(mutate)
    lowest = min(record.claims["zones"].values(), key=lambda zone: zone["work_similarity_percent"])
    text = one_priority(record, single=True)
    assert f"Zone {lowest['zone_id']}" in text and f"{lowest['work_similarity_percent']:.0f}%" in text and "작업 유사도" in text, text
