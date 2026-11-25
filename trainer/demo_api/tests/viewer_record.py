"""2026-11-14 운영 질문 시험 기록(학습모드 3 스프레이)을 뷰어와 같은 모양의 서명된 문맥으로 만든다.

`fixtures/viewer_record_20261114.json`은 운영 `/motion-viewer/api/motion`·`arm-pose` 응답에서 질의에 쓰는
키만 남긴 사본이다(작업자 이름은 지웠다). 서명은 이 프로세스의 키로 다시 한다.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.copilot_router import create_copilot_router
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.viewer_context import sign_context, viewer_signatures

_FIXTURE = Path(__file__).with_name("fixtures") / "viewer_record_20261114.json"
TASKS = {"pose": "pose_recomm", "order": "trajectory_recomm", "coating": "knowledge_search", "knowledge": "knowledge_search"}


def payload(*, progress: bool = True) -> dict[str, Any]:
    """`progress=False`면 Zone 첫 방문 시각·전문가 전이가 없던 v3.3.1 응답 모양이다."""
    data = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    first, flows = data.pop("paint_trace_first_seconds"), data.pop("expert_flows")
    if progress:
        from trainer.demo_api.viewer_progress import order_progress_facts, transition_counts
        # 서버(motion_data)가 하는 일: 순서 평가에 싣고, 서명되는 대화 자료로 옮긴다.
        data["reference_order_evaluation"].update(zone_first_visit_seconds=first, expert_transitions=transition_counts(flows))
        data["dialogue_evaluation"].update(order_progress_facts(data["reference_order_evaluation"]))
    data["query_context_signatures"] = viewer_signatures(data)
    data["query_context_signatures"]["arm_pose"] = sign_context("arm_pose", data["arm_pose_evaluation"])
    return data


def payload_range() -> dict[str, Any]:
    """v3.6.2 정상 범위(160~200µm) 판정 행 두 개로 바꾼 기록. 운영 판정 함수로 만든다.

    Zone 1: 보정 220µm 일색(상한 초과, 범위 안 0%). Zone 2: 보정 170·170·170·240(평균 187.5 정상, 범위 안 75%, 상한 초과 25%).
    """
    from trainer.demo_api import coating_assessment as ca
    data = payload()
    target = data["coating_evaluation"]["target"]
    target.update(criterion_version=ca.ASSESSMENT_VERSION, normal_band_um=[160.0, 200.0], zones=[
        ca.zone_row(1, "1", ca.assess_readings([220.0] * 4)), ca.zone_row(1, "2", ca.assess_readings([170.0, 170.0, 170.0, 240.0]))])
    data["query_context_signatures"] = viewer_signatures(data)
    data["query_context_signatures"]["arm_pose"] = sign_context("arm_pose", data["arm_pose_evaluation"])
    return data


def context(task: str, data: dict[str, Any], playback: dict[str, Any]) -> str:
    """motion_viewer_v2.js `context()`와 같은 문자열."""
    sigs = data["query_context_signatures"]
    shared = {
        "frame_direction_context": None, "frame_direction_signature": None,
        "dialogue_context": data.get("dialogue_evaluation"), "dialogue_signature": sigs.get("dialogue"),
        "coating_context": data.get("coating_evaluation"), "coating_signature": sigs.get("coating_status"),
        "arm_pose_context": data.get("arm_pose_evaluation"), "arm_pose_signature": sigs.get("arm_pose"),
    }
    if task == "knowledge_search":
        return "motion_viewer_playback:" + json.dumps({"record": "훈련생 · 스프레이 · 학습모드 3", "playback_context": playback, **shared}, ensure_ascii=False)
    pose, order = data["pose_evaluation"], data["reference_order_evaluation"]
    if task == "pose_recomm":
        analysis = {"record": "r", "status": pose.get("status"), "similarity_percent": pose.get("similarity_percent"),
                    "zones": pose.get("zones"), "corrections": pose.get("corrections")}
    else:
        analysis = {"record": "r", **{key: order.get(key) for key in (
            "recommended_flow", "selected_flow", "adherence_rate", "best_reference_similarity", "position_similarity",
            "position_distribution", "reference_session_count")},
            "direction_summary": data.get("direction_summary"), "zone_keys": sorted(pose.get("zones") or {})}
    return "motion_viewer_analysis:" + json.dumps({**shared, "analysis_context": analysis, "signature": sigs.get(task),
                                                   "playback_context": playback}, ensure_ascii=False)


AT_3S = {"map_num": "1", "zone_id": "8", "source_index": 20, "playhead_seconds": 3.011151}
AT_24S = {"map_num": "1", "zone_id": "1", "source_index": 160, "playhead_seconds": 24.054651}
AT_28S = {"map_num": "1", "zone_id": "9", "source_index": 190, "playhead_seconds": 28.566211}
AT_35S = {"map_num": "1", "zone_id": "9", "source_index": 239, "playhead_seconds": 35.933611}


class RecordingPipeline:
    """계획 모델·대화 모델·문서 답을 흉내 낸다. 무엇을 불렀는지 남긴다."""

    def __init__(self, *, plan: dict[str, str] | None = None, converse: str = "대화 답입니다.",
                 document: str = "[답변] 문서 답입니다.", chunks: list[dict[str, Any]] | None = None) -> None:
        self.plan, self.reply, self.document, self.chunks = plan, converse, document, chunks or []
        self.calls: list[str] = []
        self.kwargs: list[tuple[str, dict[str, Any]]] = []
        self.config: dict[str, Any] = {}

    def plan_copilot_dialogue(self, prompt: str, **kwargs: object) -> dict[str, str]:
        self.calls.append("plan")
        self.kwargs.append(("plan", dict(kwargs)))
        return self.plan or {"kind": "chat", "intent": "knowledge", "scope": "unspecified"}

    def converse_copilot(self, prompt: str, **kwargs: object) -> str:
        self.calls.append("converse")
        self.kwargs.append(("converse", dict(kwargs)))
        return self.reply

    def run(self, **kwargs: object) -> dict[str, Any]:
        self.calls.append("run")
        self.kwargs.append(("run", dict(kwargs)))
        return {"response": self.document, "rag_chunks": self.chunks}


def ask(prompt: str, *, tab: str = "order", playback: dict[str, Any] = AT_3S, mode: str = "quick",
        pipeline: RecordingPipeline | None = None, data: dict[str, Any] | None = None,
        raw_context: str | None = None, store: SequenceStore | None = None, offline: bool = False,
        summarize: bool = False) -> dict[str, Any]:
    """`offline=True`면 모델 서버가 없는 라우터(pipeline_provider=None)다."""
    pipeline = pipeline or RecordingPipeline()
    data = data or payload()
    task = TASKS[tab]
    app = FastAPI()
    app.include_router(create_copilot_router(None if offline else lambda: pipeline, store=store or SequenceStore(":memory:")))
    response = TestClient(app).post("/api/v1/copilot/query", json={
        "user_id": "COPILOT-AI-TEST", "session_id": data["selected_session_id"], "scenario_id": str(data["mode_id"]),
        "task_type": task, "prompt": prompt, "response_mode": mode, "summarize": summarize,
        "context": raw_context if raw_context is not None else context(task, copy.deepcopy(data), playback)})
    assert response.status_code == 200, response.text
    return {**response.json()["data"], "source": response.headers["x-copilot-answer-source"], "calls": pipeline.calls,
            "headers": {key.lower(): value for key, value in response.headers.items()}}
