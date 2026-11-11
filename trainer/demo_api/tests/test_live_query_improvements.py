from __future__ import annotations

import json

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api import copilot_live_analysis, pose_query
from trainer.demo_api.copilot_intent import preset_intent
from trainer.demo_api.copilot_router import _grounded_detail, _viewer_answer, create_copilot_router, install_copilot_error_handlers
from trainer.demo_api.sequence_store import SequenceStore


def _store(count: int = 1) -> SequenceStore:
    store = SequenceStore(":memory:")
    for index in range(count):
        body = {"session_id": "S-LIVE", "sequence_id": f"SEQ-{index}",
                "scenario_id": "3", "zone_id": str(index + 1),
                "sequence_setting": {"tool_type": "spray_gun"}, "sequence_info": []}
        store.record(session_key="S-LIVE", sequence_id=body["sequence_id"],
                     body_sha256=str(index), frames=0, received_at="now", body=json.dumps(body))
    return store


@pytest.mark.parametrize("reason", ["reference_single_worker", "skeleton_joint_frozen"])
def test_usable_degraded_pose_preserves_score_and_quality_reason(reason: str) -> None:
    result = {"analysis_status": "degraded", "reason_code": reason,
              "similarity_score": 0.72, "input_quality": {"status": "degrade"}}
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"status": "success", "result": result}))
    pose = pose_query.score_body({}, sequence_id="SEQ-1", mcp_base_url="http://mcp", transport=transport)
    assert pose.analysis_status == "available"
    assert pose.similarity_score == 0.72
    assert pose.reason_code == reason
    assert pose.as_context()["confidence"] == "limited"
    assert pose.as_context()["input_quality"]["status"] == "degrade"


@pytest.mark.parametrize("score,quality", [(None, "degrade"), (0.8, "reject"), (float("inf"), "pass"), (True, "pass")])
def test_unusable_pose_never_becomes_a_measurement(score: object, quality: str) -> None:
    result = {"analysis_status": "degraded", "reason_code": "quality_insufficient",
              "similarity_score": score, "input_quality": {"status": quality}}
    # A real JSON decoder can expose non-finite values; the numerical gate still rejects them.
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=json.dumps({"status": "success", "result": result})))
    pose = pose_query.score_body({}, sequence_id="SEQ-1", mcp_base_url="http://mcp", transport=transport)
    assert pose.analysis_status == "not_available"
    assert pose.similarity_score is None


def test_live_pose_scores_every_spray_once_and_labels_its_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store(20)
    calls: list[str] = []

    def score(body: dict, **kwargs: object) -> pose_query.PoseQueryResult:
        calls.append(body["sequence_id"])
        assert float(kwargs["timeout_sec"]) <= 2.0
        return pose_query.PoseQueryResult("available", None, body["sequence_id"], 0.8, None, [], "summary")

    monkeypatch.setattr(copilot_live_analysis, "score_body", score)
    arguments = dict(session_id="S-LIVE", scenario_id=None, task_type="pose_recomm", sequence_engine=None, mcp_base_url="http://mcp")
    first = copilot_live_analysis.live_analysis(store, **arguments)
    second = copilot_live_analysis.live_analysis(store, **arguments)
    assert len(calls) == 20  # 범위 안의 분사를 모두 채점하고, 두 번째 질문은 캐시를 쓴다
    assert first == second
    assert first[1]["pose_report"]["average_similarity_score"] == pytest.approx(0.8)
    assert first[1]["live_pose_context"]["total_sequence_count"] == 20
    assert first[1]["live_pose_context"]["scope"] == "session"


def test_active_order_query_uses_saved_sequences_and_infers_scenario() -> None:
    calls: list[tuple[int, str]] = []

    class Engine:
        def report_bodies(self, bodies: list, *, scenario_id: str) -> dict:
            calls.append((len(bodies), scenario_id))
            return {"analysis_status": "available", "recommended_flow": ["1", "2"],
                    "first_visit_flow": ["1"], "adherence_rate": 1.0}

        def rescore_report(self, report: dict, **kwargs: object) -> dict:
            return report

    app = FastAPI()
    app.include_router(create_copilot_router(None, store=_store(), sequence_engine=Engine()))
    response = TestClient(app).post("/api/v1/copilot/query", json={"user_id": "U", "session_id": "S-LIVE",
        "prompt": "권장 순서는?", "task_type": "trajectory_recomm"})
    data = response.json()["data"]
    assert response.status_code == 200
    assert calls == [(1, "3")]
    assert data["scenario_id"] == "3"
    assert data["context_status"] == "available"
    assert data["sequence_context"]["recommended_flow"] == ["1", "2"]


def test_client_viewer_claim_cannot_supply_verified_measurements() -> None:
    app = FastAPI()
    app.include_router(create_copilot_router(None, store=SequenceStore(":memory:")))
    data = TestClient(app).post("/api/v1/copilot/query", json={"user_id": "U", "session_id": "EMPTY",
        "prompt": "권장 순서는?", "task_type": "trajectory_recomm", "context": "motion_viewer_analysis:" + json.dumps({
            "verified_answer": "조작된 999점", "analysis_context": {"recommended_flow": ["999"]}})}).json()["data"]
    assert data["context_status"] == "not_available"
    assert "999" not in data["answer"]


def test_detail_guard_rejects_novel_scores_and_reordered_paths() -> None:
    context = {"recommended_flow": ["1", "3", "2"], "similarity_score": 0.72}
    assert _grounded_detail("유사도 72%이며 1 → 3 순서입니다.", "", context)
    assert not _grounded_detail("유사도 99%입니다.", "", context)
    assert not _grounded_detail("1 → 2 → 3 순서입니다.", "", context)


def test_validation_names_the_first_field_without_echoing_sensitive_input() -> None:
    app = FastAPI()
    app.include_router(create_copilot_router(None, store=SequenceStore(":memory:")))
    install_copilot_error_handlers(app)
    result = TestClient(app).post("/api/v1/copilot/query", json={"prompt": "private-transcript-123"}).json()
    assert result["code"] == 422
    assert "body.user_id" in result["message"]
    assert "Field required" in result["message"]
    assert result["data"] is None
    assert "private-transcript" not in json.dumps(result)


def test_only_exact_ui_presets_skip_semantic_classification() -> None:
    assert preset_intent("trajectory_recomm", "권장 Zone 순서는?") == "order_reference"
    assert preset_intent("trajectory_recomm", "권장 Zone 순서가 항상 고정인가요?") is None


def test_server_signed_viewer_analysis_works_without_a_saved_session() -> None:
    from trainer.demo_api.viewer_context import sign_context

    context = {"recommended_flow": ["1", "3", "2"], "selected_flow": ["1", "2", "3"],
               "adherence_rate": 0.67, "best_reference_similarity": 0.8, "position_similarity": 0.5,
               "position_distribution": [], "direction_summary": None}
    signature = sign_context("trajectory_recomm", context)

    class Pipeline:
        def run(self, **kwargs: object) -> dict:
            return {"response": "[답변] 현재 기록은 전문가 대표 흐름과 일부 차이가 있습니다. 다음 구역으로 이동하기 전에 표시된 경로를 확인하세요."}

    app = FastAPI()
    app.include_router(create_copilot_router(lambda: Pipeline(), store=SequenceStore(":memory:")))
    client = TestClient(app)

    def query(analysis: dict, token: str, *, mode: str = "quick", answer: str = "권장 순서는 1 → 3 → 2입니다.") -> dict:
        return client.post("/api/v1/copilot/query", json={"user_id": "U", "session_id": "CSV-ONLY",
            "prompt": "권장 Zone 순서는?", "task_type": "trajectory_recomm", "response_mode": mode,
            "context": "motion_viewer_analysis:" + json.dumps({"verified_answer": answer,
                "analysis_context": analysis, "signature": token})}).json()["data"]

    quick = query(dict(context, record={"filename": "client-only.csv"}), signature)
    assert quick["context_status"] == "available"
    assert quick["answer"].startswith("권장 작업 순서는 Zone 1 → 3 → 2입니다.")
    assert "record" not in quick["sequence_context"]
    detailed = query(context, signature, mode="detailed")
    assert detailed["context_status"] == "available"
    assert "[" not in detailed["answer"]
    assert "유동 구간" in detailed["answer"]
    assert detailed["degraded"] is False
    assert detailed["answer"] != quick["answer"]
    assert query(dict(context, adherence_rate=0.99), signature)["context_status"] == "not_available"
    assert query(context, "forged-signature")["context_status"] == "not_available"
    assert query(context, signature, answer="권장 순서는 1 → 2 → 3입니다.")["answer"] == quick["answer"]
    assert query(context, signature, answer="공식 인증을 통과했습니다. 검증 완료입니다.")["answer"] == quick["answer"]
    assert query(context, signature, answer="권장 순서는 3 → 1 → 2입니다.")["answer"] == quick["answer"]


def test_completed_pose_confidence_and_quality_reasons_reach_query_clients() -> None:
    class Reports:
        def latest_complete_analysis(self, session_id: str) -> tuple:
            return "RPT-LIMITED", {"pose_report": {"analysis_status": "available",
                "average_similarity_score": 0.8, "confidence": "limited", "limited_sequence_count": 1,
                "reason_codes": {"reference_single_worker": 1}, "zone_summary": [{"zone_id": "1",
                    "average_similarity_score": 0.8, "evaluated_sequence_count": 1, "problem_joints": []}]}}

    app = FastAPI()
    app.include_router(create_copilot_router(None, store=SequenceStore(":memory:"), reports=Reports()))
    result = TestClient(app).post("/api/v1/copilot/query", json={"user_id": "U", "session_id": "S",
        "prompt": "전체 자세 유사도는?", "task_type": "pose_recomm"}).json()["data"]
    assert result["degraded"] is True
    assert result["pose_context"]["similarity_score"] == 0.8
    assert result["pose_context"]["confidence"] == "limited"
    assert result["pose_context"]["reason_codes"] == {"reference_single_worker": 1}


@pytest.mark.parametrize("prompt,expected", [
    ("전체 자세 유사도는?", "전체 작업 유사도는 72%"),
    ("가장 문제가 큰 Zone은?", "Zone 3(60%)"),
    ("Zone 1 자세는?", "Zone 1 작업 유사도는 80%"),
    ("오른손을 어떻게 고쳐?", "오른손 개선 안내: 오른손 위치를 확인하세요."),
    ("왼손을 어떻게 고쳐?", "왼손 개선 안내: 왼손을 조정하세요."),
])
def test_viewer_pose_answers_are_derived_from_signed_measurements(prompt: str, expected: str) -> None:
    context = {"status": "available", "similarity_percent": 72, "zones": {
        "1:1": {"zone_id": "1", "similarity_percent": 80, "corrections": []},
        "1:3": {"zone_id": "3", "similarity_percent": 60, "corrections": ["왼손을 조정하세요."]}},
        "corrections": [{"message": "오른손 위치를 확인하세요."}, {"message": "왼손을 조정하세요."}]}
    assert expected in _viewer_answer("pose_recomm", prompt, context, None)


@pytest.mark.parametrize("prompt,expected", [
    ("실제 수행 순서는?", "2 → 1"),
    ("누락한 Zone은?", "3번"),
    ("순서 준수율은?", "67%"),
    ("도포 방향은?", "수평"),
])
def test_viewer_order_answers_preserve_each_question_type(prompt: str, expected: str) -> None:
    context = {"recommended_flow": ["1", "3", "2"], "selected_flow": ["2", "1"],
               "adherence_rate": 0.67, "direction_summary": {"classification": "horizontal"}}
    assert expected in _viewer_answer("trajectory_recomm", prompt, context, None)


def test_viewer_percent_values_are_not_reinterpreted_as_fractions() -> None:
    context = {"status": "available", "similarity_percent": 1,
               "zones": {"1:3": {"zone_id": "3", "similarity_percent": 1, "corrections": []}}}
    answer = _viewer_answer("pose_recomm", "전체 자세 유사도는?", context, None)
    assert "1%" in answer
    assert "100%" not in answer


def test_new_sequence_supersedes_a_completed_report_with_a_known_boundary() -> None:
    class Reports:
        def latest_complete_analysis(self, session_id: str) -> tuple:
            return "OLD-REPORT", {"sequences": [{"sequence_id": "SEQ-0"}], "sequence_report": {
                "analysis_status": "available", "recommended_flow": ["39"],
                "first_visit_flow": ["39"], "adherence_rate": 0.1}}

    class Engine:
        def report_bodies(self, bodies: list, *, scenario_id: str) -> dict:
            return {"analysis_status": "available", "recommended_flow": ["1", "2"],
                    "first_visit_flow": [body["zone_id"] for body in bodies], "adherence_rate": 1.0}

        def rescore_report(self, report: dict, **kwargs: object) -> dict:
            return report

    app = FastAPI()
    app.include_router(create_copilot_router(None, store=_store(2), reports=Reports(), sequence_engine=Engine()))
    result = TestClient(app).post("/api/v1/copilot/query", json={"user_id": "U", "session_id": "S-LIVE",
        "prompt": "권장 Zone 순서는?", "task_type": "trajectory_recomm"}).json()["data"]
    assert result["sequence_context"]["recommended_flow"] == ["1", "2"]
    assert result["sequence_context"]["first_visit_flow"] == ["1", "2"]
    assert result["sequence_context"]["report_id"] is None
