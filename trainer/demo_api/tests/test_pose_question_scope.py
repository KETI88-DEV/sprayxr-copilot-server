"""자세 질문은 붙인 범위(세션·구역·시퀀스) 안에서 질문이 묻는 것에 답한다(2026-11-11 지시).

범위는 "어디를 볼지"만 정한다. 이전에는 시퀀스나 구역을 붙이면 어떤 질문이든 그 범위의 요약 한 문장만 나왔다.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from trainer.demo_api import coating_query, copilot_live_analysis, pose_query
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.tests.test_pose_recomm import QUERY, TELEMETRY, _MCP_RESULT, _app, _query_body, _sequence_body

_TAIL = " 착탄을 몸이 따라간 기울기 +0.181(시퀀스 232개)."
_LEFT = {"joint": "LeftUpperArm", "label": "스트로크 0~100% 구간에서 왼쪽 위팔이 향한 방향이 숙련자 평균과 33° 달랐습니다.",
         "target_validated": False, "angle_delta_deg": None, "direction": None}


def _pose(corrections: list[dict[str, Any]], summary: str) -> pose_query.PoseQueryResult:
    return pose_query.PoseQueryResult("available", None, "SEQ-003", 0.2, "joint_only", corrections, summary + _TAIL)


def test_sequence_answer_drops_the_engine_session_tracking_sentence() -> None:
    pose = _pose([], "숙련자 팔 자세 대비 유사도 80% — 숙련자 75백분위. 여섯 관절 모두 숙련자 통상 범위 안입니다.")

    answer = pose_query.answer_sequence_question(pose, prompt="지금 자세는 어때?")

    assert answer == "숙련자 팔 자세 대비 유사도 80% — 숙련자 75백분위. 여섯 관절 모두 숙련자 통상 범위 안입니다."


@pytest.mark.parametrize(("prompt", "intent"), [("왼손을 어떻게 고쳐?", None), ("이쪽은 어때?", "pose_left")])
def test_sequence_answer_quotes_the_engine_correction_for_the_asked_arm(prompt: str, intent: str | None) -> None:
    answer = pose_query.answer_sequence_question(_pose([_LEFT], "숙련자 팔 자세 대비 유사도 20%."), prompt=prompt, intent=intent)

    assert _LEFT["label"] in answer
    assert "기울기" not in answer


def test_sequence_answer_says_the_other_arm_is_not_the_correction_target() -> None:
    answer = pose_query.answer_sequence_question(_pose([_LEFT], "숙련자 팔 자세 대비 유사도 20%."), prompt="오른손을 어떻게 고쳐?")

    assert "오른팔" in answer and "교정 대상이 아닙니다" in answer
    assert "왼쪽 위팔" in answer  # 엔진이 실제로 짚은 관절은 알려 준다


def test_sequence_answer_without_corrections_reports_the_arm_in_range() -> None:
    pose = _pose([], "숙련자 팔 자세 대비 유사도 80%. 여섯 관절 모두 숙련자 통상 범위 안입니다.")

    assert pose_query.answer_sequence_question(pose, prompt="오른손을 어떻게 고쳐?") == "이 분사에서 오른팔은 숙련자 통상 범위 안입니다."


@pytest.mark.parametrize("prompt", ["팔꿈치를 더 펴야 해, 굽혀야 해?", "손목 각도는 어때?", "어깨를 더 올려야 해?"])
def test_parts_the_engine_does_not_score_are_declined_not_invented(prompt: str) -> None:
    answer = pose_query.unsupported_pose_part(prompt)

    assert answer is not None and "확인할 수 없습니다" in answer


@pytest.mark.parametrize("prompt", ["오른손을 어떻게 고쳐?", "지금 자세는 어때?", "전체 작업 유사도는?"])
def test_scored_questions_are_not_declined(prompt: str) -> None:
    assert pose_query.unsupported_pose_part(prompt) is None


def _report() -> tuple[str, dict[str, Any]]:
    return "RPT-1", {"pose_report": {"average_similarity_score": 0.66, "evaluated_sequence_count": 8, "zone_summary": [
        {"zone_id": "5", "average_similarity_score": 0.79, "evaluated_sequence_count": 5, "problem_joints": []},
        {"zone_id": "9", "average_similarity_score": 0.19, "evaluated_sequence_count": 3, "problem_joints": [
            {"joint": "RightUpperArm", "occurrence_count": 1, "occurrence_rate": 1 / 3, "average_distance": 0.21}]},
    ]}}


def test_zone_scope_answers_the_asked_arm_within_that_zone() -> None:
    right = pose_query.answer_pose_question(_report(), prompt="오른손을 어떻게 고쳐?", context="9")
    left = pose_query.answer_pose_question(_report(), prompt="왼손을 어떻게 고쳐?", context="9")
    other_zone = pose_query.answer_pose_question(_report(), prompt="오른손을 어떻게 고쳐?", context="5")

    assert "9번 구역" in right.answer and "오른쪽 위팔" in right.answer and "21cm" in right.answer
    assert "9번 구역" in left.answer and "왼쪽" in left.answer and "발견되지 않았습니다" in left.answer
    assert "5번 구역" in other_zone.answer and "발견되지 않았습니다" in other_zone.answer
    assert right.context["zone_id"] == "9" and other_zone.context["zone_id"] == "5"


def test_zone_scope_without_an_arm_keeps_the_zone_summary() -> None:
    assert pose_query.answer_pose_question(_report(), prompt="지금 자세는 어때?", context="9").answer.startswith("9번 구역 결과입니다.")


def test_question_scoring_does_not_touch_the_real_engine_session() -> None:
    seen: list[str] = []

    def mcp(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content)["session_id"])
        return httpx.Response(200, json={"status": "success", "result": dict(_MCP_RESULT), "error": None})

    body = _sequence_body()
    for _ in range(2):
        pose_query.score_body(body, sequence_id="SEQ-012", mcp_base_url="http://mcp", transport=httpx.MockTransport(mcp), isolated=True)
    pose_query.score_body(body, sequence_id="SEQ-012", mcp_base_url="http://mcp", transport=httpx.MockTransport(mcp))

    assert body["session_id"] not in seen[:2] and seen[0] != seen[1]  # 질문마다 새 엔진 세션
    assert seen[2] == body["session_id"]  # 완료 분석 등 기존 호출은 그대로
    assert body["session_id"] == "1_1000053_1"


def _live_store(zones: list[str]) -> SequenceStore:
    store = SequenceStore(":memory:")
    for index, zone in enumerate(zones):
        body = {"session_id": "S-LIVE", "sequence_id": f"SEQ-{index + 1:03d}", "scenario_id": "3", "zone_id": zone,
                "sequence_setting": {"tool_type": "spray_gun"}, "sequence_info": []}
        store.record(session_key="S-LIVE", sequence_id=body["sequence_id"], body_sha256=str(index), frames=0,
                     received_at="now", body=json.dumps(body))
    return store


def test_live_zone_analysis_scores_every_spray_of_that_zone(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _live_store(["5", "9", "9", "9", "9", "5", "9"])
    calls: list[tuple[str, bool]] = []

    def score(body: dict, **kwargs: object) -> pose_query.PoseQueryResult:
        calls.append((body["sequence_id"], bool(kwargs.get("isolated"))))
        return pose_query.PoseQueryResult("available", None, body["sequence_id"], 0.5, None, [], "summary")

    monkeypatch.setattr(copilot_live_analysis, "score_body", score)
    found = copilot_live_analysis.live_analysis(store, session_id="S-LIVE", scenario_id=None, task_type="pose_recomm",
                                                sequence_engine=None, mcp_base_url="http://mcp-zone", zone_id="9")

    assert sorted(name for name, _ in calls) == ["SEQ-002", "SEQ-003", "SEQ-004", "SEQ-005", "SEQ-007"]
    assert all(isolated for _, isolated in calls)
    assert found is not None
    assert found[1]["live_pose_context"]["scope"] == "zone"
    assert found[1]["live_pose_context"]["evaluated_sequence_count"] == 5
    assert copilot_live_analysis.current_zone(store, "S-LIVE") == "9"


@pytest.mark.parametrize("detailed", [False, True])
def test_coating_answers_carry_no_section_title(detailed: bool) -> None:
    saved = {"analysis_status": "available", "comparison_basis": "saved_coating_report", "zones": [
        {"map_num": 1, "zone_id": "3", "status": "under_app"}, {"map_num": 1, "zone_id": "5", "status": "over_app"}]}
    compared = {"analysis_status": "available", "zones": [
        {"map_num": 1, "zone_id": "3", "actual_um": 80.0, "reference_um": 120.0}, {"map_num": 1, "zone_id": "5", "actual_um": 180.0, "reference_um": 120.0}]}

    for context in (saved, compared):
        answer = coating_query.answer_coating_question(context, "도막이 과하거나 모자란 영역이 있어?", detailed=detailed)
        assert "[" not in answer
        assert "Zone 3" in answer and "Zone 5" in answer


def _client_with_two_arms() -> Any:
    result = dict(_MCP_RESULT, summary=_MCP_RESULT["summary"] + _TAIL)
    client, _ = _app(lambda _request: httpx.Response(200, json={"status": "success", "result": result, "error": None}))
    _ = client.post(TELEMETRY, json=_sequence_body())
    return client


def test_sequence_context_answers_the_question_not_only_the_summary() -> None:
    client = _client_with_two_arms()

    overall = client.post(QUERY, json=_query_body(context="SEQ-012", prompt="지금 자세는 어때?")).json()["data"]
    right = client.post(QUERY, json=_query_body(context="SEQ-012", prompt="오른손을 어떻게 고쳐?")).json()["data"]
    left = client.post(QUERY, json=_query_body(context="SEQ-012", prompt="왼손을 어떻게 고쳐?")).json()["data"]

    assert overall["answer"] == _MCP_RESULT["summary"]
    assert "오른팔" in right["answer"] and "교정 대상이 아닙니다" in right["answer"]
    assert "왼쪽 손목이 숙련자 평균보다 31 cm 멀리 있었습니다." in left["answer"]
    assert len({overall["answer"], right["answer"], left["answer"]}) == 3
    assert right["pose_context"]["sequence_id"] == "SEQ-012"


def test_pose_tab_question_about_an_unscored_part_never_reaches_the_model() -> None:
    client = _client_with_two_arms()

    data = client.post(QUERY, json=_query_body(context="SEQ-012", prompt="팔꿈치를 더 펴야 해, 굽혀야 해?")).json()["data"]

    assert "확인할 수 없습니다" in data["answer"]
    assert data["retrieval_used"] is False
