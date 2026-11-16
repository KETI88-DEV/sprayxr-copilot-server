"""Playback selects a signed Zone; it never supplies its own evaluation."""
from __future__ import annotations

import copy
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.copilot_router import create_copilot_router
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.viewer_context import sign_context, viewer_signatures


@pytest.fixture
def pose_context() -> dict:
    return {"status": "available", "similarity_percent": 70, "zones": {
        "1:2": {"map_num": "1", "zone_id": "2", "similarity_percent": 88,
                  "corrections": ["오른손의 현재 높이를 유지하세요."]},
        "2:2": {"map_num": "2", "zone_id": "2", "similarity_percent": 66,
                  "corrections": ["왼손을 조정하세요."]},
        "1:3": {"map_num": "1", "zone_id": "3", "similarity_percent": 40,
                  "corrections": ["전역 최저 구역 교정입니다."]}},
        "corrections": [{"message": "전역 최저 구역 교정입니다."}]}


def _client(*, classify_as: str | None = None) -> TestClient:
    class Pipeline:
        def classify_copilot_intent(self, task: str, prompt: str) -> str:
            return classify_as or "pose_overall"

        def run(self, **kwargs: object) -> dict:
            # A detailed response cannot import the global worst-zone facts.
            return {"response": "[답변] 현재 자세를 확인하고 해당 구역에 제시된 교정을 적용하세요."}

    app = FastAPI()
    app.include_router(create_copilot_router(lambda: Pipeline(), store=SequenceStore(":memory:")))
    return TestClient(app)


def _query(client: TestClient, context: dict, *, prompt: str = "지금 자세는 어때?",
           mode: str = "quick", playback: object = None, include_playback: bool = True,
           signature: str | None = None) -> object:
    payload = {"analysis_context": context, "signature": signature or sign_context("pose_recomm", context),
               "verified_answer": "위조된 999점 평가입니다."}
    if include_playback:
        payload["playback_context"] = playback if playback is not None else {
            "playhead_seconds": 5.25, "source_index": 42, "map_num": "1", "zone_id": "2"}
    return client.post("/api/v1/copilot/query", json={"user_id": "U", "session_id": "CSV-ONLY",
        "task_type": "pose_recomm", "prompt": prompt, "response_mode": mode,
        "use_rag": "Y", "context": "motion_viewer_analysis:" + json.dumps(payload)})


@pytest.mark.parametrize("mode", ["quick", "detailed"])
@pytest.mark.parametrize("prompt", ["지금 자세는 어때?", "현재 자세에서 오른손을 어떻게 고쳐?", "이 Zone 자세는?"])
def test_current_pose_uses_only_the_playback_maps_signed_zone(pose_context: dict, mode: str, prompt: str) -> None:
    result = _query(_client(), pose_context, mode=mode, prompt=prompt)
    assert result.status_code == 200
    data = result.json()["data"]
    if mode == "quick" and "오른손" in prompt:  # 팔 채점이 없다는 말이 한 줄의 앞이다(2026-11-16)
        assert data["answer"].startswith("팔 자세 채점 결과가 없어") and "40%" not in data["answer"]
        return
    assert "88%" in data["answer"]
    assert "40%" not in data["answer"] and "66%" not in data["answer"]
    assert "전역 최저" not in data["answer"] and "999" not in data["answer"]
    assert data["pose_context"]["scope"] == "current_zone"
    assert set(data["pose_context"]["zones"]) == {"1:2"}
    assert data["pose_context"]["playback_context"]["playhead_seconds"] == 5.25
    assert data["retrieval_used"] is False


@pytest.mark.parametrize("prompt,expected", [("전체 자세 유사도는?", "70%"),
    ("현재 기록의 전체 자세 유사도는?", "70%"), ("가장 문제가 큰 Zone은?", "40%")])
@pytest.mark.parametrize("mode", ["quick", "detailed"])
def test_explicit_global_questions_ignore_the_playback_selection(pose_context: dict, prompt: str, expected: str, mode: str) -> None:
    data = _query(_client(), pose_context, prompt=prompt, mode=mode).json()["data"]
    assert expected in data["answer"]
    # 뷰어 점수는 작업 유사도다. "자세 유사도"는 전문가 자세(팔 관절 엔진)와의 유사도에만 쓴다.
    assert "자세 유사도" not in data["answer"]
    if "유사도" in data["answer"]:
        assert "작업 유사도" in data["answer"]
    assert len(data["pose_context"]["zones"]) == 3
    assert data["pose_context"].get("scope") != "current_zone"


def test_current_pose_stays_structured_when_semantic_classifier_says_knowledge(pose_context: dict) -> None:
    data = _query(_client(classify_as="knowledge"), pose_context).json()["data"]
    assert "88%" in data["answer"]
    assert data["retrieval_used"] is False


def test_current_zone_number_uses_the_playback_map_when_zone_numbers_repeat(pose_context: dict) -> None:
    data = _query(_client(), pose_context, prompt="현재 Zone 2 자세는?", playback={
        "map_num": "2", "zone_id": "2", "playhead_seconds": 8}).json()["data"]
    assert "66%" in data["answer"]
    assert "88%" not in data["answer"]
    assert set(data["pose_context"]["zones"]) == {"2:2"}


@pytest.mark.parametrize("mode", ["quick", "detailed"])
def test_current_pose_without_a_zone_does_not_fall_back_to_global_worst(pose_context: dict, mode: str) -> None:
    data = _query(_client(), pose_context, mode=mode, playback={"playhead_seconds": 0, "source_index": 0,
        "map_num": None, "zone_id": None}).json()["data"]
    assert data["context_status"] == "not_available"
    assert "Zone ID가 없는 구간" in data["answer"]
    assert "70%" not in data["answer"] and "40%" not in data["answer"]


@pytest.mark.parametrize("prompt", ["전체 자세 유사도는?", "지금 자세는 어때?"])
def test_legacy_signed_context_without_playback_preserves_global_answer(pose_context: dict, prompt: str) -> None:
    data = _query(_client(), pose_context, prompt=prompt, include_playback=False).json()["data"]
    assert data["context_status"] == "available"
    assert "70%" in data["answer"] and "40%" in data["answer"]


@pytest.mark.parametrize("playback", [
    {"map_num": "9", "zone_id": "2", "playhead_seconds": 1},
    {"map_num": "1", "zone_id": "999", "playhead_seconds": 1},
    {"map_num": "1", "zone_id": None, "playhead_seconds": 1},
    {"map_num": "1", "zone_id": "2", "playhead_seconds": -1},
    {"map_num": "1", "zone_id": "2", "playhead_seconds": "nan"},
    {"map_num": "1", "zone_id": "2", "playhead_seconds": True},
    {"map_num": "1", "zone_id": "2", "playhead_seconds": 10 ** 400},
    {"map_num": "1", "zone_id": "2", "playhead_seconds": 1, "similarity_percent": 999},
])
def test_invalid_playback_selector_is_rejected_instead_of_overriding_signed_evaluation(pose_context: dict, playback: dict) -> None:
    response = _query(_client(), pose_context, playback=playback)
    assert response.status_code == 422
    assert response.json()["status"] == "ERROR"
    assert response.json()["data"] is None


def test_modified_evaluation_cannot_be_authenticated_by_valid_playback(pose_context: dict) -> None:
    original_signature = sign_context("pose_recomm", pose_context)
    forged = copy.deepcopy(pose_context)
    forged["zones"]["1:2"]["similarity_percent"] = 999
    data = _query(_client(), forged, signature=original_signature).json()["data"]
    assert data["context_status"] == "not_available"
    assert "999" not in data["answer"]


def test_order_playback_membership_is_also_signed(pose_context: dict) -> None:
    order = {"recommended_flow": ["2", "3"], "selected_flow": ["2"], "adherence_rate": 0.5,
        "best_reference_similarity": 0.5, "position_similarity": 0.5,
        "position_distribution": [], "reference_session_count": 1}
    payload = {"pose_evaluation": pose_context, "reference_order_evaluation": order, "direction_summary": None}
    analysis = dict(order, direction_summary=None, zone_keys=["1:2", "1:3", "2:2"])
    signed = {"analysis_context": analysis, "signature": viewer_signatures(payload)["trajectory_recomm"],
              "playback_context": {"map_num": "1", "zone_id": "2", "playhead_seconds": 5}}
    client = _client()
    query = {"user_id": "U", "task_type": "trajectory_recomm", "prompt": "권장 Zone 순서는?",
             "context": "motion_viewer_analysis:" + json.dumps(signed)}
    valid = client.post("/api/v1/copilot/query", json=query)
    assert valid.status_code == 200
    assert "Zone 2→3" in valid.json()["data"]["answer"]  # 빠른 안내는 화살표를 붙여 쓴다
    signed["playback_context"]["map_num"] = "9"
    query["context"] = "motion_viewer_analysis:" + json.dumps(signed)
    assert client.post("/api/v1/copilot/query", json=query).status_code == 422
