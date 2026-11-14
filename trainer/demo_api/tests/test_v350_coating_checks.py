from trainer.demo_api.coating_query import drop_contradicted_causes, measured_relation, zone_measures
from trainer.demo_api.tests.viewer_record import payload

MEASURES = zone_measures(payload()["dialogue_evaluation"])


def test_similar_band_is_ten_percent() -> None:
    assert measured_relation(MEASURES["7"], "distance_mm") == "similar"   # 334.0 / 366.66 = -8.9%
    assert measured_relation(MEASURES["2"], "distance_mm") == "lower"     # 346.3 / 403.786 = -14.2%
    assert measured_relation(MEASURES["4"], "distance_mm") == "higher"    # 406.9 / 329.98 = +23.3%


def test_thick_effect_on_a_below_zone_is_dropped() -> None:
    answer = "분사 거리가 과도하게 가까워 도료가 과도하게 부착된 것으로 보입니다. 덧칠이 필요합니다."
    kept = drop_contradicted_causes(answer, MEASURES, default_zone="8")
    assert "과도하게 부착" not in kept and "덧칠이 필요합니다" in kept


def test_speed_prescription_toward_the_expert_is_dropped() -> None:
    answer = "속도를 숙련자 수준으로 조정하면 작업 효율이 향상될 수 있습니다. 건 이동 속도는 일정하게 유지하세요."
    kept = drop_contradicted_causes(answer, MEASURES, default_zone="8")
    assert "숙련자 수준" not in kept and "일정하게 유지하세요" in kept
