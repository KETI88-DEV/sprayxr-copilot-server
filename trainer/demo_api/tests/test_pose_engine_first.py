"""자세 질문은 채점 엔진이 답한다 — 대화 계획·자유 대화가 가로채지 않는다.

2026-11-11 재현: `pose_recomm`에 "지금 자세 어때?"를 물으면 시퀀스를 지정해도 자유 대화가 답했고,
같은 자료에 "교정 필요"와 "기준 범위 내"가 번갈아 나왔다. 채점 결과(`pose_context`)는 비어 있었다.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api import viewer_arm_pose
from trainer.demo_api.copilot_router import create_copilot_router
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.viewer_context import sign_context

QUERY = "/api/v1/copilot/query"
SAMPLE_CSV = Path("/data/sources/new_format/sample_2026-10-08/3_11000346_1.csv.gz")
SUMMARY = "숙련자 팔 자세 대비 유사도 53% — 왼쪽 손목이 31 cm 멀리 있었습니다."


def _mcp(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"status": "success", "error": None, "result": {
        "analysis_status": "available", "reason_code": None, "similarity_score": 0.5296, "correction_mode": "joint_only",
        "summary": SUMMARY,
        "corrections": [{"joint": "LeftHand", "label": "왼쪽 손목", "correction_text": "왼쪽 손목이 숙련자 평균보다 31 cm 멀리 있었습니다.",
                         "evidence": {"distance_m": 0.31}}],
    }})


class _HijackingPipeline:
    """대화 계획이 자세 질문을 잡담으로 분류하고 자유 대화가 수치를 지어내는 모델."""

    def __init__(self) -> None:
        self.plans: list[str] = []

    def plan_copilot_dialogue(self, prompt: str, **_: Any) -> dict[str, str]:
        self.plans.append(prompt)
        return {"kind": "chat", "intent": "knowledge", "scope": "unspecified"}

    def converse_copilot(self, prompt: str, **_: Any) -> str:
        return "속도 845 m/s, 기준 범위 내입니다."

    def classify_copilot_intent(self, task_type: str, prompt: str) -> str:
        return "knowledge"

    def run(self, **_: Any) -> dict[str, Any]:
        return {"response": "[답변] 일반 지식 답변", "rag_chunks": []}


def _client(pipeline: object, store: SequenceStore) -> TestClient:
    app = FastAPI()
    app.include_router(create_copilot_router(
        lambda: pipeline, store=store, mcp_base_url="http://mcp.test:8100", transport=httpx.MockTransport(_mcp)))
    return TestClient(app)


def _stored_sequence(store: SequenceStore) -> None:
    body = {
        "session_id": "S1", "user_id": "W-017", "scenario_id": "3", "zone_id": "12", "sequence_id": "SEQ-012",
        "sequence_setting": {"tool_type": "spray_gun"},
        "sequence_info": [{"index": 1, "time": "2026-09-21T00:08:07.377Z",
                           "tool_pos_rot": {"pos": [0.98, 5.14, 0.35], "rot": [270.0, 270.0, 0.0]},
                           "skeleton_joints": {"pelvis": [0.37, 3.82, 0.44], "righthand": [0.81, 4.83, 0.39]},
                           "gun_metrics": {"speed": 0.0, "distance": 372.06, "spraytime": 0}}],
        "sequence_result": {},
    }
    store.record(session_key="S1", sequence_id="SEQ-012", body_sha256="x", frames=1,
                 received_at="2026-10-08T00:00:00Z", body=json.dumps(body, ensure_ascii=False))


def test_natural_pose_question_on_a_named_sequence_is_scored_not_chatted() -> None:
    """깨뜨리는 법: 대화 계획 조건에서 `_pose_engine_question`을 빼면 "845 m/s" 잡담이 답이 된다."""
    store = SequenceStore(":memory:")
    _stored_sequence(store)
    pipeline = _HijackingPipeline()
    for prompt in ("지금 자세 어때?", "지금 자세는 어때?", "이 시퀀스 자세를 어떻게 고쳐?"):
        data = _client(pipeline, store).post(QUERY, json={
            "user_id": "W-017", "session_id": "S1", "task_type": "pose_recomm", "prompt": prompt, "context": "SEQ-012",
        }).json()["data"]
        assert data["answer"] == SUMMARY, prompt
        assert data["pose_context"]["sequence_id"] == "SEQ-012" and data["pose_context"]["similarity_score"] == 0.5296
    assert pipeline.plans == []


def test_unrelated_talk_in_the_pose_tab_still_reaches_the_dialogue_planner() -> None:
    pipeline = _HijackingPipeline()
    data = _client(pipeline, SequenceStore(":memory:")).post(QUERY, json={
        "user_id": "W-017", "task_type": "pose_recomm", "prompt": "야"}).json()["data"]
    assert pipeline.plans == ["야"] and data["answer"] == "속도 845 m/s, 기준 범위 내입니다."


ARM = {
    "status": "available", "average_similarity_percent": 70, "evaluated_sequence_count": 2, "sequence_count": 2,
    "guideline": "분사 2회 중 2회의 자세를 숙련 작업자와 비교했습니다. 평균 유사도 70%.",
    "zones": {"8": {"similarity_percent": 80, "evaluated_sequence_count": 1, "guidance": None},
              "9": {"similarity_percent": 60, "evaluated_sequence_count": 1, "guidance": "9번 구역 분사 1회 중 1회에서 오른쪽 위팔 위치가 숙련자 범위를 벗어났습니다."}},
    "priority_corrections": ["9번 구역 분사 1회 중 1회에서 오른쪽 위팔 위치가 숙련자 범위를 벗어났습니다."],
    "sequences": [
        {"sequence_id": "SEQ-001", "zone_id": "9", "start_seconds": 10.0, "end_seconds": 20.0, "similarity_percent": 60,
         "correction": "오른쪽 위팔이 향한 방향이 숙련자 평균과 65° 달랐습니다.", "confidence": None},
        {"sequence_id": "SEQ-002", "zone_id": "8", "start_seconds": 30.0, "end_seconds": 40.0, "similarity_percent": 80,
         "correction": None, "confidence": None},
    ],
}


def test_arm_pose_answer_follows_the_playhead_the_zone_and_the_record() -> None:
    at = lambda seconds: viewer_arm_pose.arm_pose_answer(  # noqa: E731
        ARM, current=True, playback={"playhead_seconds": seconds, "zone_id": "8"}, zone_id=None)
    assert "SEQ-001" in at(15.0) and "60%" in at(15.0) and "65°" in at(15.0)
    assert "SEQ-001" in at(25.0)  # 이동 구간은 직전에 끝난 분사
    assert "SEQ-002" in at(35.0) and "교정할 관절은 없습니다" in at(35.0)
    assert "Zone 8의 자세 유사도는 평균 80%" in at(5.0)  # 첫 분사 전에는 선택 Zone
    zone = viewer_arm_pose.arm_pose_answer(ARM, current=False, playback=None, zone_id="9")
    assert "Zone 9의 자세 유사도는 평균 60%" in zone and "오른쪽 위팔" in zone
    whole = viewer_arm_pose.arm_pose_answer(ARM, current=False, playback=None, zone_id=None)
    assert "평균 유사도 70%" in whole and "우선 교정" in whole
    missing = viewer_arm_pose.arm_pose_answer(
        {"status": "unavailable", "reason_code": "no_world_joint_sequences"}, current=True, playback=None, zone_id=None)
    assert "관절 좌표가 없어" in missing and "%" not in missing


def _viewer_context(arm_signature: str) -> str:
    pose = {"status": "available", "similarity_percent": 92, "corrections": [],
            "zones": {"1:8": {"map_num": "1", "zone_id": "8", "similarity_percent": 93, "corrections": []}}}
    return "motion_viewer_analysis:" + json.dumps({
        "analysis_context": pose, "signature": sign_context("pose_recomm", pose),
        "arm_pose_context": ARM, "arm_pose_signature": arm_signature,
        "playback_context": {"map_num": "1", "zone_id": "8", "source_index": 3, "playhead_seconds": 15.0},
    })


def test_viewer_pose_question_answers_arm_pose_first_then_work_similarity() -> None:
    """뷰어 자세 탭: 시퀀스별 자세 유사도(팔 관절) → 전문가 기록 대비 작업 유사도 순."""
    pipeline = _HijackingPipeline()
    data = _client(pipeline, SequenceStore(":memory:")).post(QUERY, json={
        "user_id": "COPILOT-DEMO-USER", "task_type": "pose_recomm", "prompt": "지금 자세 어때?",
        "context": _viewer_context(sign_context("arm_pose", ARM)),
    }).json()["data"]
    arm_line, work_line = data["answer"].split("\n")
    assert "SEQ-001" in arm_line and "자세 유사도 60%" in arm_line
    assert "작업 유사도는 93%" in work_line
    assert data["pose_context"]["arm_pose"]["average_similarity_percent"] == 70
    assert "sequences" not in data["pose_context"]["arm_pose"]
    assert pipeline.plans == []


def test_forged_arm_pose_numbers_are_ignored() -> None:
    data = _client(_HijackingPipeline(), SequenceStore(":memory:")).post(QUERY, json={
        "user_id": "COPILOT-DEMO-USER", "task_type": "pose_recomm", "prompt": "지금 자세 어때?",
        "context": _viewer_context("0" * 64),
    }).json()["data"]
    assert "자세 유사도" not in data["answer"] and "작업 유사도는 93%" in data["answer"]
    assert "arm_pose" not in data["pose_context"]


def test_viewer_record_is_scored_sequence_by_sequence_from_its_csv() -> None:
    """실제 샘플 기록(분사 32회)을 구간마다 채점해 시퀀스·구역·재생 시각을 낸다. 자료가 없으면 실패한다."""
    viewer_arm_pose._CACHE.clear()
    result = viewer_arm_pose.build_arm_pose_evaluation(
        SAMPLE_CSV, record_id="3_11000346_1", mode_id=3, mcp_base_url="http://mcp.test:8100",
        transport=httpx.MockTransport(_mcp))
    assert result["status"] == "available" and result["average_similarity_percent"] == 53
    assert result["sequence_count"] == 32 and 0 < result["evaluated_sequence_count"] <= 32
    assert len(result["sequences"]) == result["evaluated_sequence_count"]
    first = result["sequences"][0]
    assert first["sequence_id"].startswith("SEQ-") and 0 <= first["start_seconds"] < first["end_seconds"]
    assert first["correction"] == "왼쪽 손목이 숙련자 평균보다 31 cm 멀리 있었습니다."
    assert set(result["zones"]) == {s["zone_id"] for s in result["sequences"]}
