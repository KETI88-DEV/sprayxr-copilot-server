"""누락·보완·다음 Zone 질문은 서명된 방문 순서와 최종 도막 게이트로만 답한다(2026-11-14 운영 q11·q12·q15·q16)."""
from __future__ import annotations

import json
import re

import pytest

from trainer.demo_api.pose_query import answer_order_question
from trainer.demo_api.reference_order_query import actual_order_intent, is_current_next_question
from trainer.demo_api.tests.viewer_record import AT_3S, AT_24S, RecordingPipeline, ask, context, payload
from trainer.demo_api.viewer_progress import first_visit_seconds, transition_counts

# 모델이 운영에서 지어낸 답. 이 문장이 나오면 모델 경로를 탄 것이다.
INVENTED = "권장 순서와 실제 작업 순서를 비교했을 때, Zone 3, 2, 1, 6이 빠진 순서로 진행되었습니다."


def _model() -> RecordingPipeline:
    return RecordingPipeline(plan={"kind": "chat", "intent": "order_missing", "scope": "record"}, converse=INVENTED)


@pytest.mark.parametrize("tab", ["order", "knowledge", "coating", "pose"])
@pytest.mark.parametrize("prompt", ["빼먹은 존 있어?", "누락한 Zone은?", "다시 칠해야 할 Zone은?", "아직 안 칠한 데 있어?"])
@pytest.mark.parametrize("mode", ["quick", "detailed"])
def test_missing_and_touch_up_questions_use_visits_and_the_final_gate(tab: str, prompt: str, mode: str) -> None:
    result = ask(prompt, tab=tab, mode=mode, pipeline=_model())
    answer = result["answer"]
    assert result["calls"] == [] and result["source"] == "analysis", answer
    # 9개 Zone 모두 최종 게이트가 needs_more다. v3.4.1: 목록을 다 늘어놓지 않고 정상 범위 안 면적이 가장 낮은 3곳만,
    # 등급 낱말(정상·저도막)과 평균 µm 없이, 줄을 나눠 적는다.
    assert "9개 Zone을 모두 칠했지만 9개 모두 보완이 필요합니다." in answer
    assert "가장 부족한 곳은 Zone 3(정상 범위 안 0%), Zone 8(1.9%), Zone 5(2.8%)입니다." in answer
    assert not re.search(r"정상(?!\s*범위)", answer) and "저도막" not in answer and "µm" not in answer and "방문하지 않은" not in answer
    assert answer.endswith("최종 도막 기준이며 재생 시점의 도막은 아닙니다.") and "\n" in answer
    assert "빠진" not in answer


def test_progress_scope_lists_each_category() -> None:
    scope = ask("다시 칠해야 할 Zone은?", pipeline=_model())["sequence_context"]
    assert scope["scope"] == "viewer_progress" and scope["unvisited_zone_ids"] == []
    assert scope["touch_up_zone_ids"] == ["7", "4", "8", "5", "6", "3", "2", "1", "9"] and scope["complete_zone_ids"] == []


@pytest.mark.parametrize("tab", ["order", "knowledge", "pose"])
@pytest.mark.parametrize("prompt", ["그럼 지금 다음에 어디 칠해야 돼?", "지금 재생하는 곳 다음에는 어디를 칠해?", "다음 구역은?"])
def test_next_zone_uses_the_playhead_and_expert_transitions(tab: str, prompt: str) -> None:
    result = ask(prompt, tab=tab, playback=AT_3S, pipeline=_model())
    answer = result["answer"]
    assert result["calls"] == [], answer
    # 3.0초까지 Zone 9·8을 칠했다. 전문가 6회 중 2회씩 Zone 8 다음에 Zone 5와 6으로 갔다 → 동률이라 둘 다 든다.
    assert answer.startswith("다음은 Zone 5나 Zone 6을 권합니다") and "현재 Zone 8" in answer
    assert "전문가 6회 중 각 2회로 동률" in answer
    assert "최종 도막만 있어 현재 완료 여부는 알 수 없습니다" in answer
    assert result["sequence_context"]["next_zone_candidates"] == ["5", "6"]
    assert "Zone 7" not in answer  # 권장 다음 구역만 물었으므로 이 기록의 실제 다음 방문은 나열하지 않는다.
    assert result["sequence_context"]["next_zone_id"] == "5"
    assert result["sequence_context"]["visited_zone_ids"] == ["9", "8"]


def test_when_every_zone_is_visited_the_final_gate_lists_touch_ups() -> None:
    result = ask("지금 위치에서 다음 Zone은 어디야?", playback=AT_24S, mode="detailed", pipeline=_model())
    answer = result["answer"]
    assert result["calls"] == [] and "확인할 수 없습니다" not in answer
    assert "모두 한 번 이상 칠했습니다" in answer and "최종 도막만 있어 현재 완료 여부는 알 수 없습니다." in answer
    assert "보완" in answer and "Zone 8" in answer
    assert result["sequence_context"]["next_zone_id"] is None


def test_older_viewer_payload_without_visit_times_still_answers_from_the_first_visit_order() -> None:
    result = ask("다음 구역은?", playback=AT_3S, data=payload(progress=False), pipeline=_model())
    assert result["calls"] == [] and "확인할 수 없습니다" not in result["answer"]
    # 전이 기록이 없으면 권장 순서에서 아직 안 칠한 첫 Zone.
    assert result["sequence_context"]["next_zone_id"] == "7"


def test_forged_progress_facts_are_ignored() -> None:
    data = payload()
    signed = context("knowledge_search", data, AT_3S)
    forged = json.loads(signed.split(":", 1)[1])
    forged["dialogue_context"]["actual_order"] = ["9"]
    raw = "motion_viewer_playback:" + json.dumps(forged, ensure_ascii=False)
    result = ask("빼먹은 존 있어?", tab="knowledge", raw_context=raw, pipeline=_model())
    assert "방문하지 않은 Zone" not in result["answer"]


def test_visit_times_and_transitions_are_built_server_side() -> None:
    trace = [{"t": 1.0, "zone_id": "9", "map_num": "1"}, {"t": 1.2, "zone_id": None, "map_num": None},
             {"t": 2.5, "zone_id": "8", "map_num": "1"}, {"t": 4.0, "zone_id": "9", "map_num": "1"}]
    assert first_visit_seconds(trace, offset=1.0) == {"9": 0.0, "8": 1.5}
    assert transition_counts([["7", "4", "8"], ["7", "4", "5"]]) == [
        {"from": "4", "to": "5", "count": 1}, {"from": "4", "to": "8", "count": 1}, {"from": "7", "to": "4", "count": 2}]


@pytest.mark.parametrize("prompt", ["빼먹은 존 있어?", "다시 칠해야 할 Zone은?", "아직 안 칠한 데 있어?", "덧칠할 곳 있어?"])
def test_tsb_order_questions_are_missing_intents(prompt: str) -> None:
    assert actual_order_intent(prompt) == "order_missing"


@pytest.mark.parametrize("prompt", ["지금 재생하는 곳 다음에는 어디를 칠해?", "그럼 지금 다음에 어디 칠해야 돼?", "다음 구역은?"])
def test_next_zone_phrasings(prompt: str) -> None:
    assert is_current_next_question(prompt)


def test_saved_report_missing_answer_adds_the_final_gate() -> None:
    report = {"analysis_status": "available", "recommended_flow": ["1", "2", "3"], "first_visit_flow": ["1", "2"],
              "evaluated_visit_count": 2, "on_track_visit_count": 2, "adherence_rate": 1.0}
    rows = [{"zone_id": "1", "map_num": 1, "grade": "below", "thickness_um": 90.0, "target_thickness_um": 160.0,
             "unpainted_fraction": 0.0, "coverage_percent": 10.0, "overpaint_percent": 0.0, "gate_status": "needs_more",
             "six_band_counts": {"lt_0_9t": 90, "from_0_9t_to_lt_1_0t": 0, "from_1_0t_to_lt_1_5t": 10,
                                 "from_1_5t_to_lt_2_0t": 0, "from_2_0t_to_lt_2_5t": 0, "gte_2_5t": 0}},
            {"zone_id": "2", "map_num": 1, "grade": "normal", "thickness_um": 170.0, "target_thickness_um": 160.0,
             "unpainted_fraction": 0.0, "coverage_percent": 97.0, "gate_status": "complete"}]
    answer = answer_order_question(("r", {"sequence_report": report, "coating_zones": rows}),
                                   prompt="다시 칠해야 할 Zone은?", intent="order_missing").answer
    assert "3" in answer and "Zone 1" in answer and "보완" in answer and "Zone 2" in answer


# --- 독립 검토 지적(2026-11-14) ---------------------------------------------------------------------------

KNOWLEDGE = ["재도장 간격은 얼마인가요?", "다시 칠할 때 주의할 점은?", "보완 도장 절차 알려줘",
             "다음에는 어디를 조심해야 해?", "다음엔 어느 부분을 고쳐야 할까요?"]


@pytest.mark.parametrize("prompt", KNOWLEDGE)
def test_knowledge_questions_with_progress_words_are_not_progress_questions(prompt: str) -> None:
    from trainer.demo_api.viewer_progress import progress_question
    assert progress_question(prompt) is None and actual_order_intent(prompt) is None
    assert not is_current_next_question(prompt)
    # 진행 답(viewer_progress)이 아니다. "고쳐야"는 기존의 서명된 고칠 점 답으로 갈 수 있다.
    result = ask(prompt, tab="knowledge", pipeline=_model())
    assert (result["sequence_context"] or {}).get("scope") != "viewer_progress" and "방문하지 않은 Zone" not in result["answer"]


def _claims(**extra: object) -> dict:
    return {"reference_order": ["2", "1", "3"], "actual_order": ["2", "1", "3"],
            "zone_first_visit_seconds": {"2": 5.0, "1": 8.0, "3": 12.0}, **extra}


def test_the_zone_under_the_playhead_counts_as_visited() -> None:
    from trainer.demo_api.viewer_progress import next_answer
    answer, scope = next_answer(_claims(), None, {"zone_id": "2", "playhead_seconds": 4.9}, prompt="다음 권장 Zone과 실제 기록의 다음 Zone은?")
    assert "2" in scope["visited_zone_ids"] and scope["next_zone_id"] == "1"
    assert "이 기록에서는 실제로 Zone 1을 다음에 처음 칠했습니다." in answer


def test_moving_playhead_is_not_called_zone_minus_one() -> None:
    from trainer.demo_api.viewer_progress import next_answer
    answer, scope = next_answer(_claims(), None, {"zone_id": "-1", "playhead_seconds": 9.0})
    assert "Zone -1" not in answer and "이동 중" in answer and scope["current_zone_id"] is None


def test_without_a_reference_flow_nothing_is_counted_against_zero_zones() -> None:
    from trainer.demo_api.viewer_progress import next_answer, status_answer
    claims = _claims(reference_order=[])
    status, _ = status_answer(claims, None)
    assert "0개" not in status and "권장 기준" in status
    answer, scope = next_answer(claims, None, {"zone_id": "2", "playhead_seconds": 6.0})
    assert "모두 한 번 이상 칠했습니다" not in answer and scope["next_zone_id"] is None


# --- 운영 답(2026-11-14, Zone 13개): 전부 글로 늘어놓아 읽을 수 없었다 ---------------------------------------

PRODUCTION = ("권장 기준 13개 Zone 가운데 방문하지 않은 Zone은 없습니다. 방문했지만 보완이 필요한 Zone(완료 기준: 목표 두께 이상 면적 95%, "
              "저도막 아님): Zone 8(정상 165µm, 목표 이상 면적 61.9%), Zone 12(저도막 127µm, …) … Zone 1(저도막 109µm, 목표 이상 면적 9.9%). "
              "완료한 Zone은 없습니다. 최종 높이맵 기준이며 재생 시점의 도막이 아닙니다.")


def _rows(areas: dict[str, float], *, complete: tuple[str, ...] = ()) -> dict:
    zones = [{"zone_id": zone, "map_num": 1, "grade": "normal" if zone == "8" else "below", "thickness_um": 120.0,
              "target_thickness_um": 160.0, "unpainted_fraction": 0.0, "coverage_percent": area, "overpaint_percent": 0.0,
              "six_band_counts": {"lt_0_9t": round(100 - area), "from_0_9t_to_lt_1_0t": 0, "from_1_0t_to_lt_1_5t": round(area),
                                  "from_1_5t_to_lt_2_0t": 0, "from_2_0t_to_lt_2_5t": 0, "gte_2_5t": 0},
              "gate_status": "complete" if zone in complete else "needs_more"} for zone, area in areas.items()]
    return {"target": {"analysis_status": "available", "zones": zones}}


def test_thirteen_zone_touch_up_lists_the_three_lowest_areas_only() -> None:
    from trainer.demo_api.viewer_progress import status_answer
    areas = {str(n): float(area) for n, area in zip(range(1, 14), (9.9, 30.0, 0.0, 12.5, 4.2, 50.0, 20.0, 61.9, 40.0, 33.0, 15.0, 25.0, 45.0))}
    order = [str(n) for n in range(1, 14)]
    answer, scope = status_answer({"reference_order": order, "actual_order": order}, _rows(areas))
    assert answer != PRODUCTION
    lines = answer.split("\n")
    assert lines[0] == "13개 Zone을 모두 칠했지만 13개 모두 보완이 필요합니다."
    assert "가장 부족한 곳은 Zone 3(정상 범위 안 0%), Zone 5(4.2%), Zone 1(9.9%)입니다." in lines
    assert answer.count("Zone ") == 3 + 0 and not re.search(r"정상(?!\s*범위)", answer) and "µm" not in answer
    assert len(scope["touch_up_zone_ids"]) == 13


def test_partial_completion_and_unvisited_zones_are_counted() -> None:
    from trainer.demo_api.viewer_progress import status_answer
    rows = _rows({"1": 10.0, "2": 98.0, "3": 50.0}, complete=("2",))
    answer, _ = status_answer({"reference_order": ["1", "2", "3", "4"], "actual_order": ["1", "2", "3"]}, rows)
    lines = answer.split("\n")
    assert lines[0] == "아직 칠하지 않은 구역: Zone 4."
    assert "지금까지 칠한 Zone 3개 가운데 2개는 보완이 필요합니다." in lines and "완료: Zone 2." in lines


# --- 확정 문구(2026-11-14 사용자 승인)와 동률 규칙 ---------------------------------------------------------

def _next(transitions: dict[tuple[str, str], int], *, current: str = "8") -> tuple[str, dict]:
    from trainer.demo_api.viewer_progress import next_answer
    claims = {"reference_order": ["7", "4", "8", "5", "6", "3"], "actual_order": ["9", "8", "7", "4", "5", "6", "3"],
              "zone_first_visit_seconds": {"9": 0.0, "8": 0.2, "7": 4.5}, "expert_transitions": [
                  {"from": a, "to": b, "count": n} for (a, b), n in transitions.items()]}
    return next_answer(claims, None, {"zone_id": current, "playhead_seconds": 3.0})


def test_a_single_best_transition_names_one_zone() -> None:
    answer, scope = _next({("8", "5"): 2, ("8", "6"): 1})
    assert answer.split("\n")[0] == "다음은 Zone 5를 권합니다(현재 Zone 8, 전문가 3회 중 2회)."
    assert scope["next_zone_candidates"] == ["5"]


def test_three_way_tie_names_three_zones() -> None:
    answer, scope = _next({("8", "5"): 1, ("8", "6"): 1, ("8", "3"): 1})
    assert "다음은 Zone 5, Zone 6이나 Zone 3을 권합니다(현재 Zone 8, 전문가 3회 중 각 1회로 동률)." in answer
    assert scope["next_zone_candidates"] == ["5", "6", "3"]


def test_three_or_more_painted_zones_are_joined_with_dots_and_long_lists_are_cut() -> None:
    from trainer.demo_api.viewer_progress import _painted, status_answer
    assert _painted(["9", "8", "5"]) == "Zone 9·8·5를" and _painted(["7", "1"]) == "Zone 7과 1을"
    order = ["1", "2", "3", "4", "5", "6", "7"]
    answer, _ = status_answer({"reference_order": order, "actual_order": ["1", "2"]}, None)
    assert answer.split("\n")[0] == "아직 칠하지 않은 구역: Zone 3·4·5 외 2곳."
