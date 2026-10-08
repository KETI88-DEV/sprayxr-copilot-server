"""`POST /api/v1/copilot/query` + `task_type=pose_recomm` — 저장된 시퀀스의 자세 채점.

명세서 p.9가 `pose_recomm`을 "전문가 작업 자세 유사도 비교"로, `context`를 "가리키는 대상
구역/시퀀스 식별자 (예: SEQ-012)"로 정한다. 채점 대상은 `telemetry/sequence`가 이미
데이터베이스에 올린 시퀀스다 — 텔레메트리는 저장만 하고 채점하지 않는다(p.4).
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.copilot_router import create_copilot_router, install_copilot_error_handlers
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.telemetry_router import create_telemetry_router

QUERY = "/api/v1/copilot/query"
TELEMETRY = "/api/v1/telemetry/sequence"

_MCP_RESULT: dict[str, Any] = {
    "analysis_status": "available",
    "reason_code": None,
    "similarity_score": 0.5296,
    "correction_mode": "joint_only",
    "summary": "숙련자 팔 자세 대비 유사도 53% — 왼쪽 손목이 31 cm 멀리 있었습니다.",
    "corrections": [
        {
            "joint": "LeftHand", "label": "왼쪽 손목",
            "correction_text": "왼쪽 손목이 숙련자 평균보다 31 cm 멀리 있었습니다.",
            "severity": "medium", "angle_delta_deg": None, "direction": None,
            "evidence": {"sigma": 2.561},
            "target": {"requires_ik": True, "per_frame": [{"index": 0}]},
        }
    ],
}


def _frame(index: int = 1) -> dict[str, Any]:
    return {
        "index": index,
        "time": "2026-09-21T00:08:07.377Z",
        "tool_pos_rot": {"pos": [0.98, 5.14, 0.35], "rot": [270.0, 270.0, 0.0]},
        "skeleton_joints": {"pelvis": [0.37, 3.82, 0.44], "righthand": [0.81, 4.83, 0.39]},
        "gun_metrics": {"speed": 0.0, "distance": 372.06, "spraytime": 0},
    }


def _sequence_body(sequence_id: str = "SEQ-012") -> dict[str, Any]:
    return {
        "session_id": "1_1000053_1", "user_id": "W-017", "scenario_id": "1-A", "zone_id": "12",
        "sequence_id": sequence_id,
        "sequence_setting": {"tool_type": "spray_gun"},
        "sequence_info": [_frame()],
        "sequence_result": {"avg_speed_mms": 0.0},
    }


def _query_body(**over: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "user_id": "W-017",
        "session_id": "1_1000053_1",
        "prompt": "내 자세가 숙련자랑 얼마나 다른가?",
        "task_type": "pose_recomm",
    }
    body.update(over)
    return body


def _app(mcp: Any = None, *, store: SequenceStore | None = None) -> tuple[TestClient, SequenceStore]:
    sequences = store if store is not None else SequenceStore(":memory:")

    def _ok(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "success", "result": dict(_MCP_RESULT), "error": None})

    app = FastAPI()
    app.include_router(create_telemetry_router(store=sequences))
    app.include_router(
        create_copilot_router(
            None, store=sequences, mcp_base_url="http://mcp.test:8100",
            transport=httpx.MockTransport(mcp or _ok),
        )
    )
    install_copilot_error_handlers(app)
    return TestClient(app), sequences


# --- 저장 → 채점 ----------------------------------------------------------


def test_telemetry_stores_and_the_query_scores_it() -> None:
    """Given an uploaded sequence, When pose_recomm asks, Then the stored sequence is scored.

    이것이 이 변경의 요점이다 — 텔레메트리는 올리고, 질의가 채점한다(p.4 vs p.9).
    """
    client, _ = _app()
    telemetry = client.post(TELEMETRY, json=_sequence_body()).json()["data"]

    data = client.post(QUERY, json=_query_body(context="SEQ-012")).json()["data"]

    assert telemetry["sequence_data_ack"]["stored"] is True
    assert "pose_feedback" not in telemetry
    assert data["context_status"] == "available"
    assert data["pose_context"]["analysis_status"] == "available"
    assert data["pose_context"]["similarity_score"] == 0.5296
    assert data["pose_context"]["sequence_id"] == "SEQ-012"
    assert data["answer"] == _MCP_RESULT["summary"]


def test_without_context_the_latest_sequence_of_the_session_is_used() -> None:
    """Given no context, When asked, Then the session's most recent sequence is scored."""
    client, _ = _app()
    _ = client.post(TELEMETRY, json=_sequence_body("SEQ-001"))
    _ = client.post(TELEMETRY, json=_sequence_body("SEQ-002"))

    data = client.post(QUERY, json=_query_body()).json()["data"]

    assert data["pose_context"]["sequence_id"] == "SEQ-002"


def test_corrections_expose_only_the_contract_fields() -> None:
    """Given an engine result, When projected, Then internal evidence stays internal."""
    client, _ = _app()
    _ = client.post(TELEMETRY, json=_sequence_body())

    pose = client.post(QUERY, json=_query_body(context="SEQ-012")).json()["data"]["pose_context"]

    assert pose["corrections"] == [
        {
            "joint": "LeftHand",
            "label": "왼쪽 손목이 숙련자 평균보다 31 cm 멀리 있었습니다.",
            "target_validated": True,
            "angle_delta_deg": None,
            "direction": None,
        }
    ]
    assert "evidence" not in json.dumps(pose)
    assert "per_frame" not in json.dumps(pose)


def test_empty_per_frame_means_the_target_is_not_validated() -> None:
    """Given a correction with no coordinates, When projected, Then it is not called validated."""
    result = dict(_MCP_RESULT)
    result["corrections"] = [{"joint": "LeftHand", "label": "왼쪽 손목", "target": {"per_frame": []}}]
    client, _ = _app(lambda _r: httpx.Response(200, json={"status": "success", "result": result}))
    _ = client.post(TELEMETRY, json=_sequence_body())

    pose = client.post(QUERY, json=_query_body(context="SEQ-012")).json()["data"]["pose_context"]

    assert pose["corrections"][0]["target_validated"] is False


# --- 대상을 찾지 못하는 경우 ----------------------------------------------


@pytest.mark.parametrize(
    ("over", "reason"),
    [
        ({"session_id": None}, "session_id_missing"),
        ({}, "no_sequence_for_session"),
        ({"context": "SEQ-999"}, "sequence_not_found"),
    ],
    ids=["no-session", "no-sequence", "unknown-context"],
)
def test_missing_target_is_200_with_a_stable_reason(over: dict[str, Any], reason: str) -> None:
    """Given no scorable target, When asked, Then 200 with a reason — 점수를 지어내지 않는다."""
    client, _ = _app()
    body = _query_body(**over)
    if body.get("session_id") is None:
        body.pop("session_id", None)

    response = client.post(QUERY, json=body)

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["context_status"] == "not_available"
    assert data["pose_context"]["reason_code"] == reason
    assert data["pose_context"]["similarity_score"] is None
    assert data["citations"] == []


def test_engine_failure_is_200_not_available() -> None:
    """Given an engine fault, When asked, Then the query still answers and says why."""
    client, _ = _app(lambda _r: httpx.Response(500, text="boom"))
    _ = client.post(TELEMETRY, json=_sequence_body())

    data = client.post(QUERY, json=_query_body(context="SEQ-012")).json()["data"]

    assert data["pose_context"]["reason_code"] == "engine_unavailable"
    assert "엔진" in data["answer"]


def test_engine_error_envelope_is_not_read_as_success() -> None:
    """Given MCP's HTTP 200 error envelope, When read, Then status decides, not the code.

    MCP 는 오류도 HTTP 200 으로 답한다.
    """
    client, _ = _app(lambda _r: httpx.Response(200, json={"status": "error", "result": None}))
    _ = client.post(TELEMETRY, json=_sequence_body())

    pose = client.post(QUERY, json=_query_body(context="SEQ-012")).json()["data"]["pose_context"]

    assert pose["reason_code"] == "engine_unavailable"


def test_engine_not_available_reason_is_passed_through() -> None:
    """Given the gate's own reason, When unavailable, Then it reaches the caller unchanged."""
    client, _ = _app(
        lambda _r: httpx.Response(
            200,
            json={
                "status": "success",
                "result": {"analysis_status": "not_available", "reason_code": "skeleton_joint_frozen"},
            },
        )
    )
    _ = client.post(TELEMETRY, json=_sequence_body())

    pose = client.post(QUERY, json=_query_body(context="SEQ-012")).json()["data"]["pose_context"]

    assert pose["reason_code"] == "skeleton_joint_frozen"


def test_a_duplicate_upload_is_still_scorable() -> None:
    """Given a re-sent sequence, When asked, Then the stored body is still there to score.

    중복이라 다시 저장하지 않지만 채점 대상은 남아 있다.
    """
    client, _ = _app()
    _ = client.post(TELEMETRY, json=_sequence_body())
    second = client.post(TELEMETRY, json=_sequence_body()).json()["data"]

    pose = client.post(QUERY, json=_query_body(context="SEQ-012")).json()["data"]["pose_context"]

    assert second["sequence_data_ack"]["duplicate"] is True
    assert pose["analysis_status"] == "available"


# --- analyze-sequence 경로 → 실제 MCP 요청 계약 (2026-10-08 샘플 재생에서 발견) -----------------


def _contract_checking_mcp(request: httpx.Request) -> httpx.Response:
    """실제 MCP의 요청 모델(`SequenceRequest`, 최상위 `extra="forbid"`)로 본문을 검증하는 대역.

    검증에 걸리면 실제 MCP처럼 422를 낸다 — 대역이 무엇이든 받아 주면 계약 어긋남을 못 잡는다.
    """
    from pydantic import ValidationError

    from mcp_servers.pose_estimation import SequenceRequest

    try:
        SequenceRequest.model_validate(json.loads(request.content))
    except ValidationError as error:
        return httpx.Response(422, json={"detail": error.errors(include_url=False)})
    return httpx.Response(200, json={"status": "success", "result": dict(_MCP_RESULT), "error": None})


def test_a_sequence_received_through_analyze_sequence_is_scorable() -> None:
    """벤더는 `copilot/analyze-sequence`로만 보낸다(2026-10-08 샘플 노트). 그 경로는 `mode_id`를 붙여 저장하는데
    MCP 자세 엔진은 모르는 최상위 필드를 거절해, 실제 세션의 pose_recomm 32건이 모두 `engine_unavailable`이었다.

    깨뜨리는 법: `pose_query`가 저장 본문을 그대로 MCP에 넘기면 422 → `engine_unavailable`로 실패한다.
    """
    client, _ = _app(_contract_checking_mcp)
    uploaded = client.post("/api/v1/copilot/analyze-sequence", json=dict(_sequence_body(), mode_id=1))
    assert uploaded.status_code == 200
    response = client.post(QUERY, json=_query_body(context="SEQ-012"))
    context = response.json()["data"]["pose_context"]
    assert context["analysis_status"] == "available", context
    assert context["sequence_id"] == "SEQ-012"
