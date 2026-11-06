"""Questions retain their meaning across tabs and detailed formatting."""
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.copilot_router import create_copilot_router
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.viewer_context import sign_context


class Pipeline:
    def __init__(self):
        self.calls = []
    def classify_copilot_intent(self, task, prompt):
        return "order_reference"  # Reproduces the real classifier's incorrect choice.
    def run(self, **kwargs):
        self.calls.append(kwargs)
        return {"response": "[답변] 권장 작업 순서는 Zone 1 → 2입니다.", "rag_chunks": []}


def context(task="trajectory_recomm", coating=True):
    analysis = {"recommended_flow": ["1", "2"], "selected_flow": ["1"],
                "adherence_rate": 0.8, "reference_session_count": 6,
                "direction_summary": {"classification": "horizontal"}, "zone_keys": ["1:1", "1:2"]}
    if task == "pose_recomm":
        analysis = {"status": "available", "similarity_percent": 90,
                    "zones": {"1:1": {"zone_id": "1", "similarity_percent": 90}}, "corrections": []}
    payload = {"analysis_context": analysis, "signature": sign_context(task, analysis),
               "playback_context": {"playhead_seconds": 1, "source_index": 0, "map_num": "1", "zone_id": "1"}}
    if coating:
        claims = {"analysis_status": "available", "scope": "final_record", "comparison_basis": "expert_zone_median",
                  "zones": [{"map_num": "1", "zone_id": "1", "actual_um": 180, "reference_um": 120},
                            {"map_num": "1", "zone_id": "2", "actual_um": 80, "reference_um": 120}]}
        payload.update(coating_context=claims, coating_signature=sign_context("coating_status", claims))
    return "motion_viewer_analysis:" + json.dumps(payload)


def client(pipeline):
    app = FastAPI()
    app.include_router(create_copilot_router(lambda: pipeline, store=SequenceStore(":memory:")))
    return TestClient(app)


@pytest.mark.parametrize("task", ["knowledge_search", "pose_recomm", "trajectory_recomm"])
@pytest.mark.parametrize("mode", ["quick", "detailed"])
def test_coating_question_answers_signed_coating_values_in_every_tab(task, mode):
    pipeline = Pipeline()
    response = client(pipeline).post("/api/v1/copilot/query", json={
        "user_id": "routing-test", "task_type": task, "response_mode": mode,
        "prompt": "도막이 과하거나 모자란 영역이 있어?", "context": context(task)})
    assert response.status_code == 200
    data = response.json()["data"]
    assert "최종" in data["answer"] and "두꺼운" in data["answer"] and "얇은" in data["answer"]
    assert "180" in data["answer"] and "80" in data["answer"] and "120" in data["answer"]
    assert "권장 Zone 순서" not in data["answer"] and "1 → 2" not in data["answer"]
    assert data["coating_context"]["scope"] == "final_record"
    assert data["retrieval_used"] is False and pipeline.calls == []


@pytest.mark.parametrize("tamper", [False, True])
def test_missing_or_forged_coating_values_do_not_become_order_or_normal_answers(tamper):
    value = context(coating=tamper)
    if tamper:
        value = value.replace('"actual_um": 180', '"actual_um": 9999')
    data = client(Pipeline()).post("/api/v1/copilot/query", json={
        "user_id": "routing-test", "task_type": "trajectory_recomm", "response_mode": "detailed",
        "prompt": "도막 두께가 이상한 존이 있어?", "context": value}).json()["data"]
    assert data["coating_context"]["analysis_status"] == "not_available"
    assert "판단" in data["answer"] and "도막" in data["answer"]
    assert "권장" not in data["answer"] and "9999" not in data["answer"]


@pytest.mark.parametrize("question,expected", [("누락한 Zone은?", "누락"), ("실제 수행 순서는?", "실제"),
                                               ("순서 준수율은?", "유사도"), ("도포 방향은?", "방향")])
def test_detailed_order_answer_preserves_the_specific_question(question, expected):
    data = client(Pipeline()).post("/api/v1/copilot/query", json={
        "user_id": "routing-test", "task_type": "trajectory_recomm", "response_mode": "detailed",
        "prompt": question, "context": context()}).json()["data"]
    assert expected in data["answer"]
    assert "[권장 Zone 순서]" not in data["answer"]
    if "누락" in question:
        assert "2번" in data["answer"] and "모든 구역을 칠했습니다" not in data["answer"]



def test_viewer_coating_uses_only_available_physical_thickness_metrics():
    from trainer.demo_api.coating_query import viewer_coating_context, report_coating_context
    payload = {"pose_evaluation": {"zones": {
        "1:1": {"map_num": 1, "zone_id": "1", "metrics": [{"key": "coating_thickness", "available": True, "actual": 0, "reference": 120}]},
        "1:2": {"map_num": 1, "zone_id": "2", "metrics": [{"key": "coating_thickness", "available": False, "actual": 150, "reference": 120}]},
        "2:1": {"map_num": 2, "zone_id": "1", "metrics": [{"key": "coating_thickness", "available": True, "actual": 180, "reference": 120}]},
    }}}
    result = viewer_coating_context(payload)
    assert {(z["map_num"], z["zone_id"]) for z in result["zones"]} == {("1", "1"), ("2", "1")}
    assert result["zones"][0]["actual_um"] == 0
    assert report_coating_context(("R", {"coating_zones": [{"zone_id": "1", "status": "unavailable"}]}))["analysis_status"] == "not_available"



def test_equal_coating_values_are_reported_as_measurements():
    from trainer.demo_api.coating_query import answer_coating_question
    text = answer_coating_question({"analysis_status": "available", "comparison_basis": "expert_zone_median",
        "zones": [{"map_num": "1", "zone_id": "1", "actual_um": 120, "reference_um": 120}]}, "Zone 1 도막 두께는?", detailed=True)
    assert "120µm" in text and "같은" in text


def test_coating_selection_respects_map_number_when_zone_ids_repeat():
    from trainer.demo_api.coating_query import answer_coating_question
    text = answer_coating_question({"analysis_status": "available", "comparison_basis": "expert_zone_median",
        "zones": [{"map_num": "1", "zone_id": "1", "actual_um": 180, "reference_um": 120},
                  {"map_num": "2", "zone_id": "1", "actual_um": 80, "reference_um": 120}]}, "Map 2 Zone 1 도막 두께는?", detailed=True)
    assert "Map 2" in text and "80µm" in text and "180µm" not in text



@pytest.mark.parametrize("task", ["pose_recomm", "trajectory_recomm"])
@pytest.mark.parametrize("mode", ["quick", "detailed"])
def test_coating_how_to_question_uses_rag_and_verified_data_instead_of_fixed_status(task, mode):
    class Advice(Pipeline):
        def run(self, **kwargs):
            self.calls.append(kwargs)
            return {"response": "[답변] 균일한 도막을 위해 분사 거리와 이동 속도를 일정하게 유지하세요.", "rag_chunks": []}
    pipeline = Advice()
    response = client(pipeline).post("/api/v1/copilot/query", json={
        "user_id": "routing-test", "task_type": task, "response_mode": mode,
        "prompt": "이 Zone에서 도막을 균일하게 만드는 방법은?", "context": context(task)})
    data = response.json()["data"]
    assert response.status_code == 200 and data["retrieval_used"] is True
    assert "일정하게" in data["answer"] and "[최종 도막 두께 비교]" not in data["answer"]
    assert data["coating_context"]["playback_context"] == {"map_num": "1", "zone_id": "1"}
    assert len(pipeline.calls) == 1
    assert "검증된 최종 도막 비교 자료" in pipeline.calls[0]["instruction"]
    assert "180" in pipeline.calls[0]["instruction"] and "120" in pipeline.calls[0]["instruction"]
