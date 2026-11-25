"""v3.6.2 새 도막 기준(정상 범위 하한~상한, 기본 160~200µm) 문구와 모자람의 방향(v360-review.md §6).

완료 진행률은 하한에서 화소별 기여를 제한한다. 상한 초과 면적은 더 칠하지 말라고 말한다. 수치는 기록 행의 값만 쓴다.
"""
from __future__ import annotations

import re

from trainer.demo_api.record_query import RecordFacts, RecordPart, answer_part, answer_record_question, one_priority
from trainer.demo_api.simulator_context import SIMULATOR_CONTEXT
from trainer.demo_api.tests.viewer_record import AT_3S, context, payload, payload_range
from trainer.demo_api.viewer_progress import gate_lines

OLD = r"1\.5배|240|목표는 160"


def facts(data=None, playback=AT_3S):
    return RecordFacts.from_context(context("knowledge_search", data or payload_range(), playback))


def part(code, zones=(), operator=None):
    return RecordPart(text="", code=code, zones=zones, zone_source="asked" if zones else "none", operator=operator,
                      metric=None, cardinality="any", premises=(), knowledge=False)


def text(code, zones=(), operator=None, data=None):
    return answer_part(facts(data), part(code, zones, operator), task_type="knowledge_search")[0]


def test_mixed_zone_with_overfilm_area_counts_it_toward_completion() -> None:
    out = text("completion", ("2",))
    for word in ("진행률", "100%", "85% 이상", "완료"):
        assert word in out, (word, out)
    assert not re.search(OLD + r"|보완|더 칠", out), out


def test_all_above_zone_counts_overfilm_area_as_target_or_more() -> None:
    out = text("completion", ("1",))
    assert "과도막" in out and "진행률" in out and "100%" in out, out
    assert not re.search(OLD + r"|보완", out), out


def test_overfilm_counts_zones_above_the_upper_bound() -> None:
    out = text("coating_over")
    assert "상한(200µm)을 넘는 Zone은 1곳" in out and "상한을 넘는 면적" in out, out
    assert "Zone 1(100%)" in out and "Zone 2(25%)" in out, out
    assert not re.search(OLD, out), out


def test_target_lines_state_the_range() -> None:
    for out in (text("coating_extreme", operator="argmax"), text("cause", ("2",))):
        assert "정상 범위는 160~200µm입니다" in out, out
        assert not re.search(OLD, out), out


def test_all_zone_completion_counts_use_the_range() -> None:
    out = text("completion")
    assert "2곳 중 2곳" in out and "160~200µm" in out, out
    assert not re.search(OLD + r"|보완이 필요", out), out


def test_original_viewer_record_keeps_its_own_completion_rule() -> None:
    out = text("completion", ("1",), data=payload())
    assert "기록 당시 정상 범위 내 면적 95%" in out and "85% 이상" not in out, out


def test_completion_reference_sentence_states_the_current_progress_rule() -> None:
    found = answer_record_question(context("knowledge_search", payload_range(), AT_3S),
                                   "이 기록 도막을 PSPC 90/10 규칙으로 판정하면 합격이에요?", task_type="knowledge_search", recent_turns=[])
    assert found is not None
    assert "진행률 85% 이상" in found.text and "평균이 목표 이상" not in found.text, found.text
    assert "하한 미만 Zone" in found.text and not re.search(OLD, found.text), found.text


def test_priority_skips_a_complete_overfilm_zone() -> None:
    out = one_priority(facts(), single=True)
    assert "Zone 2" not in out and "Zone 1" not in out and "상한 초과" not in out, out
    assert not re.search(OLD + r"|보완", out), out


def test_shortfall_below_the_lower_bound_still_says_touch_up() -> None:
    # 2026-11-14 기록 Zone 1: 평균 정상, 범위 안 47.9%, 상한 초과 10.1%, 나머지는 하한 미만이다.
    out = text("completion", ("1",), data=payload())
    assert "하한(160µm) 미만(미도장 포함)" in out and "보완(더 칠하기)이 필요하고" in out, out
    assert "상한(240µm)을 넘은 10.1%는 더 칠하지 마세요" in out, out
    assert "Zone 3부터 보완하십시오." in one_priority(facts(payload()), single=True)


def test_viewer_gate_lines_split_overfilm_shortfall_from_touch_up() -> None:
    rows = facts().target_rows
    lines, groups = gate_lines(list(rows), order=["1", "2"])
    joined = "\n".join(lines)
    assert groups["touch_up_zone_ids"] == [] and groups["over_band_zone_ids"] == [], groups
    assert groups["overpaint_zone_ids"] == [] and groups["complete_zone_ids"] == ["1", "2"], groups
    assert "보완" not in joined, joined
    under, _ = gate_lines(list(facts(payload()).target_rows), order=[])
    assert "가장 부족한 곳은 Zone 3(목표 이상 0%)" in "\n".join(under), under


def test_simulator_block_states_the_default_range_only() -> None:
    assert "기본 정상 범위는 160~200µm" in SIMULATOR_CONTEXT
    assert "진행률" in SIMULATOR_CONTEXT and "85%" in SIMULATOR_CONTEXT
    assert "평균 도막 등급은 품질 안내에 사용합니다" in SIMULATOR_CONTEXT
    assert not re.search(r"320|1차|2차|저도막이 아니면", SIMULATOR_CONTEXT), SIMULATOR_CONTEXT


def test_range_is_read_from_the_record_not_assumed() -> None:
    # 2026-11-14 기록은 옛 판정 범위(160~240µm)를 담고 있다. 문장은 기록의 범위를 그대로 쓴다.
    out = text("coating_over", data=payload())
    assert "상한(240µm)" in out and "200µm" not in out, out


def test_one_answer_uses_the_row_range_not_a_different_target_block_range() -> None:
    # 세션 범위 150~250µm로 판정한 행인데 target 블록은 목표만으로 구한 150~187.5µm를 담은 경우(조정 라운드 1 지적 a).
    from trainer.demo_api import coating_assessment as ca
    from trainer.demo_api.viewer_context import sign_context, viewer_signatures
    data = payload()
    target = data["coating_evaluation"]["target"]
    target.update(target_thickness_um=150.0, normal_band_um=[150.0, 187.5], zones=[
        ca.zone_row(1, "1", ca.assess_readings([270.0] * 4, 150.0, max_thickness_um=250.0)),
        ca.zone_row(1, "2", ca.assess_readings([160.0, 160.0, 160.0, 260.0], 150.0, max_thickness_um=250.0))])
    data["query_context_signatures"] = viewer_signatures(data)
    data["query_context_signatures"]["arm_pose"] = sign_context("arm_pose", data["arm_pose_evaluation"])
    for code, zones, operator in (("completion", ("2",), None), ("completion", (), None), ("coating_over", (), None),
                                  ("cause", ("2",), None), ("coating_extreme", (), "argmax"), ("external_standard", (), None)):
        out = text(code, zones, operator, data=data)
        assert "187.5" not in out, (code, out)
        if code != "completion":
            assert "150~250µm" in out or "상한(250µm)" in out, (code, out)


def test_below_grade_zone_never_says_mostly_above() -> None:
    # 평균은 하한 미만(저도막)인데 범위 밖 면적은 상한 초과가 더 많은 Zone(조정 라운드 1 지적 b).
    from trainer.demo_api import coating_assessment as ca
    from trainer.demo_api.viewer_progress import shortfall_text
    row = ca.zone_row(1, "4", ca.assess_readings([1.5, 1.5, 210.0, 210.0, 210.0]))
    assert row["grade"] == "below" and row["gate_status"] == "needs_more", row
    out = shortfall_text(row)
    assert "주로" not in out, out
    assert "하한(160µm) 미만(미도장 포함) 40%" in out and "상한(200µm)을 넘은 60%는 더 칠하지 마세요" in out, out
