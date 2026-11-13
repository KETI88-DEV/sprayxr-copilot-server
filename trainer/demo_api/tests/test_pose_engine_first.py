"""자세 질문은 채점 엔진이 답한다 — 대화 계획·자유 대화가 가로채지 않는다.

2026-11-11 재현: `pose_recomm`에 "지금 자세 어때?"를 물으면 시퀀스를 지정해도 자유 대화가 답했고,
같은 자료에 "교정 필요"와 "기준 범위 내"가 번갈아 나왔다. 채점 결과(`pose_context`)는 비어 있었다.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api import viewer_arm_pose
from trainer.demo_api.copilot_router import create_copilot_router
from trainer.demo_api.motion_viewer import create_motion_viewer_router
from trainer.demo_api.session_analysis import spray_runs
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.viewer_context import sign_context, verify_context
from trainer.scripts import replay_telemetry
from trainer.scripts.build_zone_guidance_profiles import build_profiles

QUERY = "/api/v1/copilot/query"
SAMPLE_CSV = Path("/data/sources/new_format/sample_2026-10-08/3_11000346_1.csv.gz")
# 2025년 숙련자 녹화(예전 형식: 관절 로컬 회전만, 월드 열 없음). Spark 뷰어 기록이 이 형식에 `Zone ID`를 더한 것이다.
LEGACY_CSV = Path("/data/sources/professionals/김경호_20250831/김경호_20250831(Sequences)/3_1000391_1.csv")
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
    # 대화 경로까지 간다. 모델이 지어낸 측정값(자료에 없는 845 m/s)은 답에 실리지 않는다.
    assert pipeline.plans == ["야"] and "845" not in data["answer"]


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
    assert "직전 분사(SEQ-001)" in at(25.0)  # 이동 구간은 직전에 끝난 분사
    assert "Zone 9" not in at(15.0)  # 분사 하나가 여러 Zone에 걸칠 수 있어 분사 문장에는 Zone을 적지 않는다
    assert "SEQ-002" in at(35.0) and "교정 관절 결과는 없습니다" in at(35.0) and "교정할 관절" not in at(35.0)
    limited = {**ARM, "sequences": [{**ARM["sequences"][1], "confidence": "limited"}]}
    reference_only = viewer_arm_pose.arm_pose_answer(
        limited, current=True, playback={"playhead_seconds": 35.0}, zone_id=None)
    assert "참고용" in reference_only and "교정 관절 결과는 없습니다" not in reference_only and "  " not in reference_only
    assert "Zone 8의 자세 유사도는 평균 80%" in at(5.0)  # 첫 분사 전에는 선택 Zone
    zone = viewer_arm_pose.arm_pose_answer(ARM, current=False, playback=None, zone_id="9")
    assert "Zone 9의 자세 유사도는 평균 60%" in zone and "오른쪽 위팔" in zone
    whole = viewer_arm_pose.arm_pose_answer(ARM, current=False, playback=None, zone_id=None)
    assert "평균 자세 유사도 70%" in whole and "우선 교정" in whole  # 작업 유사도와 나란히 나오므로 이름을 붙인다
    missing = viewer_arm_pose.arm_pose_answer(
        {"status": "unavailable", "reason_code": "no_joint_sequences"}, current=True, playback=None, zone_id=None)
    assert "채점할 분사 구간이 없습니다" in missing and "%" not in missing


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
    viewer_arm_pose._evaluate.cache_clear()
    result = viewer_arm_pose.build_arm_pose_evaluation(
        SAMPLE_CSV, record_id="3_11000346_1", process="spray", mode_id=3, mcp_base_url="http://mcp.test:8100",
        transport=httpx.MockTransport(_mcp))
    assert result["status"] == "available" and result["average_similarity_percent"] == 53
    assert result["joint_source"] == "world"
    assert result["sequence_count"] == 32 and 0 < result["evaluated_sequence_count"] <= 32
    assert len(result["sequences"]) == result["evaluated_sequence_count"]
    first = result["sequences"][0]
    assert first["sequence_id"].startswith("SEQ-") and 0 <= first["start_seconds"] < first["end_seconds"]
    assert first["correction"] == "왼쪽 손목이 숙련자 평균보다 31 cm 멀리 있었습니다."
    assert set(result["zones"]) == {s["zone_id"] for s in result["sequences"]}


def _legacy_with_zone(target: Path) -> Path:
    """예전 형식 녹화에 `Zone ID` 열을 더한 사본(원본은 고치지 않는다). 데이터 행 끝의 빈 칸은 머리글에 맞춰 버린다."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with LEGACY_CSV.open(encoding="utf-8-sig", newline="") as source, target.open("w", encoding="utf-8-sig", newline="") as out:
        reader, writer = csv.reader(source), csv.writer(out)
        header = next(reader)
        writer.writerow([*header, "Zone ID"])
        for row in reader:
            writer.writerow([*row[:len(header)], "1"])
    return target


def test_old_format_record_is_scored_with_the_fk_the_expert_reference_was_built_with(tmp_path: Path) -> None:
    """Spark 뷰어 기록은 관절 월드 열이 없다. 깨뜨리는 법: `_engine_bodies`의 예전 형식 분기를 빼면 `no_joint_sequences`."""
    seen: list[dict[str, Any]] = []

    def mcp(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return _mcp(request)

    path = _legacy_with_zone(tmp_path / "3_1000391_1.csv")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        runs = spray_runs(list(csv.DictReader(handle)))
    viewer_arm_pose._evaluate.cache_clear()
    result = viewer_arm_pose.build_arm_pose_evaluation(
        path, record_id="3_1000391_1", process="spray", mode_id=3, mcp_base_url="http://mcp.test:8100",
        transport=httpx.MockTransport(mcp))
    assert result["status"] == "available" and result["joint_source"] == "fk"
    assert len(seen) == result["sequence_count"] == len(runs) > 0
    body = seen[0]
    # 엔진의 세션 상태를 실제 훈련 세션과 섞지 않고, 한 번 채점 안에서는 분사 순서대로 같은 세션을 쓴다.
    assert body["session_id"].startswith("motion-viewer:3_1000391_1:")
    assert {item["session_id"] for item in seen} == {body["session_id"]}
    # 다시 채점하면 새 세션이다. 같은 이름이면 엔진에 남은 굳은 관절이 결과를 바꾼다.
    viewer_arm_pose._evaluate.cache_clear()
    again = viewer_arm_pose.build_arm_pose_evaluation(
        path, record_id="3_1000391_1", process="spray", mode_id=3, mcp_base_url="http://mcp.test:8100",
        transport=httpx.MockTransport(mcp))
    assert again == result and seen[-1]["session_id"] != body["session_id"]
    frame = body["sequence_info"][0]
    assert len(frame["skeleton_joints"]) == 20
    gap = math.dist(frame["skeleton_joints"]["righthand"], frame["tool_pos_rot"]["pos"])
    assert gap == pytest.approx(replay_telemetry.HAND_TOOL_GAP_M)  # 건 위치는 손에서 합성한다


def test_blasting_records_are_not_scored_against_the_spray_reference(tmp_path: Path) -> None:
    calls: list[httpx.Request] = []
    result = viewer_arm_pose.build_arm_pose_evaluation(
        tmp_path / "absent.csv", record_id="x", process="blasting", mode_id=1, mcp_base_url="http://mcp.test:8100",
        transport=httpx.MockTransport(lambda request: calls.append(request) or httpx.Response(500)))
    assert result == {"status": "unavailable", "reason_code": "process_not_supported"} and calls == []
    answer = viewer_arm_pose.arm_pose_answer(result, current=True, playback=None, zone_id=None)
    assert "블라스팅" in answer and "%" not in answer


def test_viewer_route_returns_a_signed_arm_pose_evaluation(tmp_path: Path) -> None:
    """패널과 채팅이 쓰는 경로. 서명이 이 서버의 것이어야 채팅이 수치를 받는다."""
    _legacy_with_zone(tmp_path / "raw/spray/learning/mode_03/3_1000391_1.csv")
    assert build_profiles(tmp_path, process="spray", legacy_expert_zone_ids=False)
    app = FastAPI()
    app.include_router(create_motion_viewer_router(
        tmp_path, mcp_base_url="http://mcp.test:8100", transport=httpx.MockTransport(_mcp)))
    viewer_arm_pose._evaluate.cache_clear()
    data = TestClient(app).get("/motion-viewer/api/arm-pose", params={
        "process": "spray", "mode_id": 3, "session_id": "representative"}).json()
    assert data["selected_session_id"] == "3_1000391_1"
    assert data["arm_pose_evaluation"]["status"] == "available"
    assert verify_context("arm_pose", data["arm_pose_evaluation"], data["signature"])


@pytest.mark.parametrize("prompt", [
    "적정 이격거리를 자세히 알려줘",
    "적정 이격거리를 자세하게 설명해 줘",
    "적정 이격거리에 대한 자세한 설명을 해 줘",
])
def test_detailed_knowledge_word_is_not_a_pose_question(prompt: str) -> None:
    pipeline = _HijackingPipeline()
    response = _client(pipeline, SequenceStore(":memory:")).post(QUERY, json={
        "user_id": "COPILOT-AI-TEST", "task_type": "pose_recomm", "prompt": prompt,
    })
    assert response.status_code == 200
    assert pipeline.plans == [prompt]
    assert not response.json()["data"].get("pose_context")


@pytest.mark.parametrize("partial_failure", [False, True])
def test_viewer_retries_transient_engine_failures_without_caching_them(partial_failure: bool) -> None:
    viewer_arm_pose._evaluate.cache_clear()
    failing, calls = True, 0

    def mcp(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if failing and (not partial_failure or calls == 1):
            return httpx.Response(503)
        return _mcp(request)

    transport = httpx.MockTransport(mcp)
    def evaluate() -> dict[str, Any]:
        return viewer_arm_pose.build_arm_pose_evaluation(
            SAMPLE_CSV, record_id="3_11000346_1", process="spray", mode_id=3,
            mcp_base_url="http://mcp.test:8100", transport=transport)

    first = evaluate()
    failed_calls = calls
    assert failed_calls > 0
    failing = False
    recovered = evaluate()
    assert calls > failed_calls, "Recovered engine must be called again for the unchanged CSV"
    assert recovered["status"] == "available"
    if partial_failure:
        assert recovered["evaluated_sequence_count"] > first["evaluated_sequence_count"]
    successful_calls = calls
    assert evaluate() == recovered
    assert calls == successful_calls, "Successful evaluations should remain cached"
