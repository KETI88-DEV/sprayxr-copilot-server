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

    assert pose_query.friendly_correction(_LEFT["label"]) in answer  # v3.4.1: "스트로크 0~100% 구간에서" → "스트로크 내내"
    assert "기울기" not in answer


def test_sequence_answer_says_the_other_arm_is_not_the_correction_target() -> None:
    answer = pose_query.answer_sequence_question(_pose([_LEFT], "숙련자 팔 자세 대비 유사도 20%."), prompt="오른손을 어떻게 고쳐?")

    assert "오른팔" in answer and "교정 대상이 아닙니다" in answer
    assert "왼쪽 위팔" in answer  # 엔진이 실제로 짚은 관절은 알려 준다


def test_sequence_answer_without_corrections_does_not_call_the_arm_normal() -> None:
    pose = _pose([], "숙련자 팔 자세 대비 유사도 80%. 여섯 관절 모두 숙련자 통상 범위 안입니다.")

    # 2026-11-14: 교정이 없다는 것을 "통상 범위 안"으로 단정하지 않는다(엔진 안내문과 같은 단서를 붙인다).
    assert pose_query.answer_sequence_question(pose, prompt="오른손을 어떻게 고쳐?") == (
        "이 분사에서 오른팔 교정 항목은 나오지 않았습니다. 교정이 없어도 자세가 정상이라는 뜻은 아닙니다.")


@pytest.mark.parametrize("prompt", ["어깨를 더 올려야 해?", "무릎을 굽혀야 해?"])
def test_parts_the_engine_does_not_score_are_declined_not_invented(prompt: str) -> None:
    answer = pose_query.unsupported_pose_part(prompt)

    assert answer is not None and "확인할 수 없습니다" in answer


# 팔꿈치(ForeArm)·손목(Hand)은 엔진이 위치로 잰다. 각도를 물어도 거절하지 않고 위치 결과와 각도가 없는 까닭을 답한다.
@pytest.mark.parametrize("prompt", ["오른손을 어떻게 고쳐?", "지금 자세는 어때?", "전체 작업 유사도는?",
                                    "팔꿈치를 더 펴야 해, 굽혀야 해?", "손목 각도는 어때?"])
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
    answer = pose_query.answer_pose_question(found, prompt="지금 자세는 어때?", context="9").answer
    assert "9번 구역" in answer and "세션 평균" not in answer  # 구역만 채점한 값을 세션 평균이라 부르지 않는다
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


def _graded(zone_id: str, grade: str | None, thickness_um: float | None, unpainted: float, *counts: int,
            adjustments: tuple[tuple[str, str, int], ...] = ()) -> dict[str, Any]:
    """완료 리포트에 저장되는 새 도막 행(coating_assessment.zone_row 모양) 가운데 답이 읽는 열쇠."""
    from trainer.zone_engine.numerical import SIX_BAND_KEYS

    return {"map_num": 1, "zone_id": zone_id, "grade": grade,
            "status": {"below": "under_app", "normal": "normal", "above": "over_app"}.get(grade or ""),
            "thickness_um": thickness_um, "target_thickness_um": 160.0, "target_basis": "default_target",
            "unpainted_fraction": unpainted, "six_band_counts": dict(zip(SIX_BAND_KEYS, counts or (0,) * 6)),
            "adjustments": [{"variable": variable, "direction": direction, "priority": priority,
                             "evidence_kind": "quality_rule_candidate"} for variable, direction, priority in adjustments]}


def _saved(*rows: dict[str, Any]) -> dict[str, Any]:
    return coating_query.report_coating_context(("RPT-1", {"coating_zones": list(rows)}))


_GRADED = (_graded("3", "below", 98.0, 0.12, 80, 8, 0, 0, 0, 0), _graded("5", "above", 301.0, 0.03, 0, 0, 0, 50, 47, 0),
           _graded("7", None, None, 1.0), _graded("1", "normal", 171.0, 0.0, 0, 0, 100, 0, 0, 0))


@pytest.mark.parametrize("detailed", [False, True])
def test_report_answer_names_grades_means_and_unpainted_share(detailed: bool) -> None:
    """새 리포트 행은 등급·칠한 곳 평균·미도장 비율로 답한다. 칠한 곳 없는 Zone도 빠지지 않는다.

    깨뜨리는 법: 인정하는 행을 status 세 값으로만 두면 칠한 곳 없는 7번(status None)이 답에서 사라진다.
    """
    answer = coating_query.answer_coating_question(_saved(*_GRADED), "도막이 과하거나 모자란 영역이 있어?", detailed=detailed)

    assert answer.split("\n") == [
        "기본 목표 160µm 기준으로 Zone별 도막을 판정했습니다. "
        # 966a3d0 정상 범위 기준(기본 160~200µm). 문장은 coating_assessment.RULE_NOTE 그대로다.
        "칠한 곳 평균 기준: 하한 미만 저도막·하한~상한 정상(경계 포함)·상한 초과 과도막. 기본 160~200µm.",
        "저도막: 1개 — Map 1 · Zone 3(칠한 곳 평균 98µm, 미도장 12%)",
        "과도막: 1개 — Map 1 · Zone 5(칠한 곳 평균 301µm, 미도장 3%)",
        "정상: 1개",
        "칠한 곳 없음: 1개 — Map 1 · Zone 7",
    ]
    assert not any(word in answer for word in ("[", "픽셀", "화소", "마스크"))


def test_report_answer_orders_by_distance_from_the_target_and_shortens_the_quick_answer() -> None:
    rows = [_graded(str(zone), "below", thickness, 0.0) for zone, thickness in ((1, 150.0), (2, 40.0), (3, 120.0), (4, 90.0))]

    quick = coating_query.answer_coating_question(_saved(*rows), "도막 상태는 어때?", detailed=False)
    full = coating_query.answer_coating_question(_saved(*rows), "도막 상태는 어때?", detailed=True)

    assert "저도막: 4개 — Map 1 · Zone 2(칠한 곳 평균 40µm, 미도장 0%), Map 1 · Zone 4(" in quick
    assert "Zone 3(칠한 곳 평균 120µm, 미도장 0%) 외 1곳\n과도막: 없음\n정상: 0개" in quick
    assert "Zone 1(칠한 곳 평균 150µm, 미도장 0%)\n과도막: 없음" in full and "외 " not in full


def test_a_named_zone_gets_only_its_own_grade() -> None:
    answer = coating_query.answer_coating_question(_saved(*_GRADED), "Zone 5 도막은 어때?", detailed=False)

    assert answer.split("\n")[1:] == ["Map 1 · Zone 5: 과도막(칠한 곳 평균 301µm, 미도장 3%)"]
    assert "Zone 3" not in answer


def test_distribution_and_unpainted_questions_get_numbers() -> None:
    """분포는 칠한 면적의 6구간 비율, 미도장은 Zone별 숫자만 준다(합격 기준이 없다)."""
    context = _saved(*_GRADED)

    distribution = coating_query.answer_coating_question(context, "도막 분포는 어때?", detailed=False)
    unpainted = coating_query.answer_coating_question(context, "미도장 영역이 있어?", detailed=False)

    # 칠한 화소 285개: 80·8·100·50·47·0
    # 966a3d0부터 분포는 저도막·정상·과도막 세 묶음이다(coating_assessment.distribution_sentence). 88·100·97
    assert distribution.endswith("칠한 면적의 두께 분포: 저도막 31%, 정상 35%, 과도막 34%.")
    assert "미도장 비율이 큰 곳" not in distribution
    assert unpainted.endswith("미도장 비율이 큰 곳: Zone 7(100%), Zone 3(12%), Zone 5(3%). 미도장 비율에는 합격 기준이 없어 숫자만 알려 드립니다.")
    assert "두께 분포" not in unpainted
    # 0%인 Zone은 "큰 곳"에 넣지 않는다.
    one = coating_query.answer_coating_question(_saved(*_GRADED[2:], _graded("2", "normal", 171.0, 0.004)), "미도장 영역이 있어?", detailed=False)
    assert one.endswith("미도장 비율이 큰 곳: Zone 7(100%). 미도장 비율에는 합격 기준이 없어 숫자만 알려 드립니다.")
    none = coating_query.answer_coating_question(_saved(_GRADED[3]), "미도장 영역이 있어?", detailed=False)
    assert none.endswith("판정한 Zone의 미도장 비율은 모두 1% 미만입니다. 미도장 비율에는 합격 기준이 없어 숫자만 알려 드립니다.")


def test_old_saved_rows_keep_the_old_answer() -> None:
    """2026-11 이전에 저장된 리포트 행에는 grade가 없다. 그 행은 예전 문장 그대로 답한다."""
    answer = coating_query.answer_coating_question(
        _saved({"map_num": 1, "zone_id": "3", "status": "under_app"}), "도막이 과하거나 모자란 영역이 있어?", detailed=False)

    assert "도막 부족: Map 1 · Zone 3\n도막 과다: 판정된 영역 없음" in answer


def test_distribution_cause_names_candidates_without_claiming_a_cause() -> None:
    """error_analysis 답: 분포와 저도막·과도막 Zone의 조절 후보. 후보는 원인이 아니라고 밝힌다."""
    rows = (_graded("3", "below", 98.0, 0.12, 88, adjustments=(("distance_mm", "decrease", 2), ("speed_mm_s", "decrease", 1))),
            _graded("5", "above", 301.0, 0.03, 0, 0, 0, 97), _graded("1", "normal", 171.0, 0.0, 0, 0, 100))

    answer = coating_query.answer_distribution_cause(_saved(*rows), detailed=False)

    assert answer is not None and answer.split("\n") == [
        "기본 목표 160µm 기준으로 본 도막 분포입니다. "
        "칠한 면적의 두께 분포: 저도막 31%, 정상 35%, 과도막 34%.",
        "Zone 3은 저도막입니다(칠한 곳 평균 98µm). 조절 후보는 이동 속도를 낮추는 쪽, 이격거리를 줄이는 쪽입니다.",
        "Zone 5는 과도막입니다(칠한 곳 평균 301µm). 속도·이격거리 기록이 없어 조절 후보를 내지 못했습니다.",
        "조절 후보는 등급에 따른 일반적인 방향이며 원인으로 확정된 것은 아닙니다.",
    ]
    only_normal = coating_query.answer_distribution_cause(_saved(rows[2]), detailed=True)
    assert only_normal is not None and only_normal.split("\n")[1:] == ["저도막·과도막으로 판정된 Zone이 없습니다."]
    # 옛 행만 있는 완료 세션에 "완료 분석이 없다"고 답하면 사실과 다르다.
    assert coating_query.answer_distribution_cause(_saved({"map_num": 1, "zone_id": "3", "status": "under_app"}), detailed=False) == (
        "이 세션의 완료 분석은 이전 기준으로 저장되어 있어 도막 분포와 조절 후보를 낼 수 없습니다.")
    # 자료 없음은 답하지 않는다(라우터가 자료 없음 안내로 떨어진다).
    assert coating_query.answer_distribution_cause(_saved(), detailed=False) is None


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
    assert "왼쪽 손목이 숙련자 평균보다 몸 밖으로 31cm 더 뻗어 있었습니다." in left["answer"]  # v3.4.1 몸 기준 표현
    assert len({overall["answer"], right["answer"], left["answer"]}) == 3
    assert right["pose_context"]["sequence_id"] == "SEQ-012"


def test_pose_tab_elbow_question_reads_the_forearm_and_never_reaches_the_model() -> None:
    client = _client_with_two_arms()

    data = client.post(QUERY, json=_query_body(context="SEQ-012", prompt="팔꿈치를 더 펴야 해, 굽혀야 해?")).json()["data"]

    assert "팔꿈치 교정 항목은 나오지 않았습니다" in data["answer"] and "자세가 정상이라는 뜻은 아닙니다" in data["answer"]
    declined = client.post(QUERY, json=_query_body(context="SEQ-012", prompt="어깨를 더 올려야 해?")).json()["data"]
    assert "확인할 수 없습니다" in declined["answer"]
    assert data["retrieval_used"] is False
