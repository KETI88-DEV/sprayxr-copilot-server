"""v2.5 POST /api/v1/telemetry/sequence: 수신 확인, 자세 위임, 미구현 엔진, 별칭."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.copilot_router import install_copilot_error_handlers
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.telemetry_router import (
    ANALYZE_SEQUENCE_ALIAS,
    TELEMETRY_SEQUENCE_PATH,
    create_telemetry_router,
)

URL = TELEMETRY_SEQUENCE_PATH
# 계약 §7.3 의 12키와 정확히 같아야 한다. 한 개라도 더하거나 빼면 이 시험이 실패한다.
_DATA_KEYS = {
    "request_id", "session_id", "user_id", "scenario_id", "sequence_id",
    "generated_at", "latency_ms", "sequence_data_ack",
    "coating_feedback", "sequence_feedback", "pose_feedback", "text_guide",
}

# 실물 샘플의 관절 표기: 소문자 키에 [x,y,z] 배열. 계약 예시의 {pos, rot} 형태도 받는다.
_JOINTS = {"pelvis": [0.3716, 3.8185, 0.4379], "lefthand": [0.5772, 3.6674, 0.7986]}

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


def _frame(index: int = 1, **extra: Any) -> dict[str, Any]:
    frame = {
        "index": index,
        "time": "2026-09-21T00:08:07.377Z",
        "tool_pos_rot": {"pos": [0.9834, 5.138, 0.3487], "rot": [270.0, 270.0, 0.0]},
        "skeleton_joints": dict(_JOINTS),
        "gun_metrics": {"speed": 0.0, "distance": 372.06, "spraytime": 0},
    }
    frame.update(extra)
    return frame


def _body(**over: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "sequence_id": "SEQ-002",
        "session_id": "1_1001642_1",
        "user_id": "1001642",
        "scenario_id": "1",
        "zone_id": "1",
        "sequence_info": [_frame()],
    }
    body.update(over)
    return body


def _client(mcp: Any = None, *, store: SequenceStore | None = None) -> TestClient:
    def _ok(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "success", "result": dict(_MCP_RESULT), "error": None})

    app = FastAPI()
    app.include_router(
        create_telemetry_router(
            store=store if store is not None else SequenceStore(":memory:"),
            mcp_base_url="http://mcp.test:8100",
            transport=httpx.MockTransport(mcp or _ok),
        )
    )
    install_copilot_error_handlers(app)
    return TestClient(app)


# --- 수신 확인 -------------------------------------------------------------


def test_new_sequence_is_stored_and_acked() -> None:
    """Given a new sequence, When posted, Then the ack reports what was observed."""
    data = _client().post(URL, json=_body()).json()["data"]

    assert set(data) == _DATA_KEYS
    assert data["sequence_data_ack"] == {
        "stored": True, "duplicate": False, "frames_received": 1, "frames_accepted": 1,
    }
    assert data["request_id"] == "sequence-SEQ-002"


def test_same_body_twice_is_a_duplicate_and_counts_no_frames() -> None:
    """Given the same body again, When posted, Then it is a duplicate, not a second ingest."""
    client = _client()
    first = client.post(URL, json=_body()).json()["data"]
    second = client.post(URL, json=_body()).json()["data"]

    assert first["sequence_data_ack"]["stored"] is True
    assert second["sequence_data_ack"] == {
        "stored": False, "duplicate": True, "frames_received": 1, "frames_accepted": 0,
    }
    # 두 번째는 다시 채점하지 않는다 — 같은 시퀀스를 두 번 누적하면 안 된다.
    assert second["pose_feedback"]["reason_code"] == "duplicate_sequence"


def test_alias_is_the_same_endpoint_and_the_second_send_is_a_duplicate() -> None:
    """Given the vendor's off-contract path, When it repeats a body, Then the store identifies it.

    벤더 클라이언트는 같은 본문을 telemetry/sequence 와 copilot/analyze-sequence 로 각각 보낸다.
    """
    client = _client()
    canonical = client.post(URL, json=_body()).json()["data"]
    alias = client.post(ANALYZE_SEQUENCE_ALIAS, json=_body()).json()["data"]

    assert canonical["sequence_data_ack"]["stored"] is True
    assert alias["sequence_data_ack"]["duplicate"] is True
    assert set(alias) == _DATA_KEYS


def test_same_key_with_a_different_body_is_reported_not_overwritten() -> None:
    """Given a conflicting body under one key, When posted, Then it is degraded, not silently kept."""
    client = _client()
    _ = client.post(URL, json=_body())
    conflict = client.post(URL, json=_body(sequence_info=[_frame(), _frame(2)])).json()["data"]

    # 충돌은 ack 의 stored=false + duplicate=false 조합이 구별해 준다 — 계약에 없는
    # 필드를 더하지 않는다.
    assert conflict["sequence_data_ack"] == {
        "stored": False, "duplicate": False, "frames_received": 2, "frames_accepted": 0,
    }


# --- 엄격성 정책 ----------------------------------------------------------


def test_top_level_typo_is_rejected_with_the_v25_error_envelope() -> None:
    """Given a misspelled top-level key, When posted, Then it fails loudly."""
    body = _body()
    body["sesion_id"] = body.pop("session_id")

    response = _client().post(URL, json=body)

    assert response.status_code == 422
    assert response.json() == {"status": "ERROR", "code": 422, "message": "Invalid request", "data": None}


def test_unknown_frame_key_is_accepted_and_reported() -> None:
    """Given a new key inside a frame, When posted, Then the sequence survives and is reported.

    현행 정책을 프레임까지 적용하면 실물 샘플의 정상 시퀀스 100%가 422다.
    """
    data = _client().post(URL, json=_body(sequence_info=[_frame(heatmap_crop={"png": "..."})])).json()["data"]

    assert data["sequence_data_ack"]["stored"] is True
    # 미지 키는 응답이 아니라 로그로 보고한다(계약 §7.3 은 12키다).
    assert set(data) == _DATA_KEYS


def test_unknown_frame_fields_are_reported_once_per_request(caplog: pytest.LogCaptureFixture) -> None:
    """Given many frames with a new key, When posted, Then one warning carries the count.

    120FPS 세션에서 프레임마다 찍으면 실제 경고가 묻힌다.
    """
    frames = [_frame(i, extra_key=i) for i in range(60)]

    with caplog.at_level("WARNING", logger="trainer.demo_api.telemetry_router"):
        data = _client().post(URL, json=_body(sequence_info=frames)).json()["data"]

    warnings = [r for r in caplog.records if "unknown frame fields" in r.getMessage()]
    assert len(warnings) == 1
    assert "count=50" in warnings[0].getMessage()
    assert set(data) == _DATA_KEYS


@pytest.mark.parametrize(
    "joints",
    [
        {"pelvis": [0.0, 1.0, 0.0]},                                  # 실물
        {"pelvis": {"pos": [0.0, 1.0, 0.0], "rot": [0.0, 0.0, 0.0]}},  # 계약 예시
    ],
    ids=["real-sample", "contract-example"],
)
def test_both_joint_shapes_are_accepted(joints: dict[str, Any]) -> None:
    """Given either skeleton shape, When posted, Then validation passes; the engine interprets."""
    data = _client().post(URL, json=_body(sequence_info=[_frame(skeleton_joints=joints)])).json()["data"]

    assert data["sequence_data_ack"]["stored"] is True


# --- 자세 위임 ------------------------------------------------------------


def test_pose_feedback_projects_only_the_contract_fields() -> None:
    """Given an engine result, When projected, Then internal evidence stays internal."""
    data = _client().post(URL, json=_body()).json()["data"]
    pose = data["pose_feedback"]

    assert pose["analysis_status"] == "available"
    assert pose["similarity_score"] == 0.5296
    assert pose["correction_mode"] == "joint_only"
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
    assert data["text_guide"] == _MCP_RESULT["summary"]


def test_empty_per_frame_means_the_target_is_not_validated() -> None:
    """Given a correction with no coordinates, When projected, Then it is not called validated.

    적용할 좌표가 없으면 교정은 말뿐이다.
    """
    result = dict(_MCP_RESULT)
    result["corrections"] = [{"joint": "LeftHand", "label": "왼쪽 손목", "target": {"per_frame": []}}]

    def _handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "success", "result": result})

    data = _client(_handler).post(URL, json=_body()).json()["data"]

    assert data["pose_feedback"]["corrections"][0]["target_validated"] is False


def test_missing_session_id_stores_but_does_not_score() -> None:
    """Given no session_id, When posted, Then it is stored and pose says why it was skipped.

    없는 세션을 지어내면 자세 누적이 엉뚱한 곳에 쌓인다.
    """
    body = _body()
    del body["session_id"]

    data = _client().post(URL, json=body).json()["data"]

    assert data["sequence_data_ack"]["stored"] is True
    assert data["session_id"] is None
    assert data["pose_feedback"]["reason_code"] == "session_id_missing"
    assert data["pose_feedback"]["analysis_status"] == "not_available"


def test_no_frames_is_not_scored() -> None:
    """Given an empty frame list, When posted, Then nothing is invented."""
    data = _client().post(URL, json=_body(sequence_info=[])).json()["data"]

    assert data["pose_feedback"]["reason_code"] == "no_frames"


@pytest.mark.parametrize(
    "handler",
    [
        lambda _r: httpx.Response(500, text="boom"),
        lambda _r: httpx.Response(200, json={"status": "error", "error": {"code": "x"}, "result": None}),
    ],
    ids=["http-500", "mcp-error-envelope"],
)
def test_engine_failure_is_200_and_not_available(handler: Any) -> None:
    """Given an engine fault, When posted, Then ingest still succeeds and pose says why.

    MCP는 오류도 HTTP 200으로 답하므로 status 필드가 성공을 결정한다.
    """
    response = _client(handler).post(URL, json=_body())

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["sequence_data_ack"]["stored"] is True
    assert data["pose_feedback"]["reason_code"] == "engine_unavailable"


# --- 미구현 엔진 ----------------------------------------------------------


def test_unimplemented_engines_say_so_rather_than_inventing_values() -> None:
    """Given no coating or sequence engine, When posted, Then both report not_available."""
    data = _client().post(URL, json=_body()).json()["data"]

    for key in ("coating_feedback", "sequence_feedback"):
        assert data[key] == {"analysis_status": "not_available", "reason_code": "engine_not_implemented"}


# --- 모니터 기록 ----------------------------------------------------------


def _monitored() -> tuple[TestClient, Any]:
    from trainer.demo_api.monitor import MonitorRecorder

    recorder = MonitorRecorder()

    def _ok(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "success", "result": dict(_MCP_RESULT), "error": None})

    app = FastAPI()
    app.include_router(
        create_telemetry_router(
            store=SequenceStore(":memory:"),
            mcp_base_url="http://mcp.test:8100",
            transport=httpx.MockTransport(_ok),
            recorder=recorder,
        )
    )
    install_copilot_error_handlers(app)
    return TestClient(app), recorder


def test_one_input_and_one_output_event_per_call() -> None:
    """Given a recorder, When a sequence is posted, Then the monitor sees both directions."""
    client, recorder = _monitored()

    _ = client.post(URL, json=_body())
    events = recorder.snapshot(cursor=0, limit=50).events

    assert [(e.phase, e.direction) for e in events] == [("runtime", "input"), ("runtime", "output")]
    assert [e.session_id for e in events] == ["1_1001642_1", "1_1001642_1"]


def test_the_output_event_carries_the_response_body() -> None:
    """Given a recorded call, When the output event is read, Then it is the response data."""
    client, recorder = _monitored()

    posted = client.post(URL, json=_body()).json()["data"]
    output = recorder.snapshot(cursor=0, limit=50).events[1]

    assert output.payload["data"] == posted


def test_the_input_event_precedes_the_stored_state() -> None:
    """Given a failing engine, When posted, Then the input is still on the monitor.

    처리 전에 남기므로 이후 단계가 실패해도 무엇이 들어왔는지는 남는다.
    """
    from trainer.demo_api.monitor import MonitorRecorder

    recorder = MonitorRecorder()
    app = FastAPI()
    app.include_router(
        create_telemetry_router(
            store=SequenceStore(":memory:"),
            mcp_base_url="http://mcp.test:8100",
            transport=httpx.MockTransport(lambda _r: httpx.Response(500, text="boom")),
            recorder=recorder,
        )
    )
    _ = TestClient(app).post(URL, json=_body())

    events = recorder.snapshot(cursor=0, limit=50).events
    assert len(events) == 2
    assert events[1].payload["data"]["pose_feedback"]["reason_code"] == "engine_unavailable"


def test_unattributed_sequence_is_still_visible_on_the_monitor() -> None:
    """Given no session_id, When posted, Then the monitor still shows the exchange."""
    client, recorder = _monitored()
    body = _body()
    del body["session_id"]

    _ = client.post(URL, json=body)

    assert {e.session_id for e in recorder.snapshot(cursor=0, limit=50).events} == {"(unattributed)"}


def test_no_recorder_means_no_recording_and_no_failure() -> None:
    """Given no recorder, When posted, Then the endpoint works unchanged."""
    assert _client().post(URL, json=_body()).status_code == 200
