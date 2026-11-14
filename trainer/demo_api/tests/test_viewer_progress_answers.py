"""누락·보완·다음 Zone 질문은 서명된 방문 순서와 최종 도막 게이트로만 답한다(2026-11-14 운영 q11·q12·q15·q16)."""
from __future__ import annotations

import json

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
    assert "방문하지 않은 Zone은 없습니다" in answer
    # 9개 Zone 모두 최종 게이트가 needs_more다. 정상 등급인 Zone 1도 면적(47.9%)이 모자라 보완 대상이다.
    assert "보완" in answer and "Zone 1" in answer and "47.9%" in answer and "77µm" in answer
    assert "완료한 Zone은 없습니다" in answer
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
    # 3.0초까지 Zone 9·8을 칠했다. 전문가 6회 중 2회가 Zone 8 다음에 Zone 5로 갔다(5와 6이 2회로 같으면 권장 순서가 앞선 5).
    assert "Zone 5" in answer and "9·8" in answer and "확인할 수 없습니다" not in answer
    assert "Zone 7" in answer  # 이 기록이 실제로 다음에 칠한 곳
    assert result["sequence_context"]["next_zone_id"] == "5"
    assert result["sequence_context"]["visited_zone_ids"] == ["9", "8"]


def test_when_every_zone_is_visited_the_final_gate_lists_touch_ups() -> None:
    result = ask("지금 위치에서 다음 Zone은 어디야?", playback=AT_24S, mode="detailed", pipeline=_model())
    answer = result["answer"]
    assert result["calls"] == [] and "확인할 수 없습니다" not in answer
    assert "모든 Zone" in answer and "재생 시점의 도막 완료 여부는 알 수 없" in answer
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
             "unpainted_fraction": 0.0, "coverage_percent": 10.0, "gate_status": "needs_more"},
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
    answer, scope = next_answer(_claims(), None, {"zone_id": "2", "playhead_seconds": 4.9})
    assert "2" in scope["visited_zone_ids"] and scope["next_zone_id"] == "1"
    assert "실제로 다음에 처음 칠한 곳은 Zone 1" in answer


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
    assert "모든 Zone" not in answer and scope["next_zone_id"] is None
