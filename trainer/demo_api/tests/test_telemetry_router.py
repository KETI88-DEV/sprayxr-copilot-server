"""v2.5 POST /api/v1/telemetry/sequence: 수신 확인, 자세 위임, 미구현 엔진, 별칭."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.copilot_router import install_copilot_error_handlers
from trainer.demo_api.expert_guidance import GuideItem, SequenceGuide
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.anomaly_stream import AnomalyBroadcaster
from trainer.demo_api.telemetry_contracts import EngineFeedback
from trainer.demo_api.telemetry_router import (
    ANALYZE_SEQUENCE_PATH,
    TELEMETRY_SEQUENCE_PATH,
    create_telemetry_router,
)

URL = TELEMETRY_SEQUENCE_PATH
# 계약 §7.3 의 12키와 정확히 같아야 한다. 한 개라도 더하거나 빼면 이 시험이 실패한다.
# 자세는 여기 없다 — copilot/query 의 pose_recomm 이 채점한다(명세서 p.4 vs p.9).
_DATA_KEYS = {
    "request_id", "session_id", "user_id", "scenario_id", "sequence_id",
    "generated_at", "latency_ms", "sequence_data_ack",
    "coating_feedback", "sequence_feedback", "text_guide",
}
_ANALYZE_DATA_KEYS = _DATA_KEYS | {"guide"}

# 실물 샘플의 관절 표기: 소문자 키에 [x,y,z] 배열. 계약 예시의 {pos, rot} 형태도 받는다.
_JOINTS = {"pelvis": [0.3716, 3.8185, 0.4379], "lefthand": [0.5772, 3.6674, 0.7986]}


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
    """명세서 p.5 의 최상위 8키. 전부 필수다."""
    body: dict[str, Any] = {
        "session_id": "1_1001642_1",
        "user_id": "1001642",
        "scenario_id": "1",
        "zone_id": "1",
        "sequence_id": "SEQ-002",
        "sequence_setting": {"tool_type": "spray_gun", "pressure": 150, "month": 9},
        "sequence_info": [_frame()],
        "sequence_result": {"avg_speed_mms": 0.0, "avg_distance_mm": 372.06},
    }
    body.update(over)
    return body


def _client(
    _unused: Any = None,
    *,
    store: SequenceStore | None = None,
    guidance_engine: Any = None,
    sequence_engine: Any = None,
    broadcaster: AnomalyBroadcaster | None = None,
) -> TestClient:
    """자세는 여기서 채점하지 않으므로 MCP 트랜스포트가 필요 없다(명세서 p.4)."""
    app = FastAPI()
    app.include_router(
        create_telemetry_router(
            store=store if store is not None else SequenceStore(":memory:"),
            guidance_engine=guidance_engine,
            sequence_engine=sequence_engine,
            broadcaster=broadcaster,
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


def test_analyze_sequence_uses_the_same_eight_keys_as_the_latest_tsb_sample() -> None:
    """The 2026-10-08 TSB body omits mode_id; scenario_id is the mode source."""
    client = _client()
    first = client.post(URL, json=_body()).json()["data"]
    second = client.post(ANALYZE_SEQUENCE_PATH, json=_body()).json()["data"]

    assert first["sequence_data_ack"]["stored"] is True
    assert second["sequence_data_ack"]["duplicate"] is True
    assert set(second) == _ANALYZE_DATA_KEYS


def test_legacy_mode_id_must_not_conflict_with_scenario_id() -> None:
    assert _client().post(ANALYZE_SEQUENCE_PATH, json=_body(mode_id=2)).status_code == 422
    assert _client().post(
        ANALYZE_SEQUENCE_PATH,
        json=_body(scenario_id="TR-L01", mode_id=2),
    ).status_code == 422


class _GuidanceStub:
    def __init__(self, guide: SequenceGuide) -> None:
        self.guide = guide
        self.calls = 0

    def evaluate(self, _request: Any) -> SequenceGuide:
        self.calls += 1
        return self.guide


def _guide(speed: str = "slow", distance: str = "far") -> SequenceGuide:
    return SequenceGuide(
        speed=GuideItem(status=speed, message="speed-message", available=True, reason=None),
        distance=GuideItem(status=distance, message="distance-message", available=True, reason=None),
    )


def test_analyze_sequence_always_returns_structured_speed_and_distance_guide() -> None:
    stub = _GuidanceStub(_guide())

    data = _client(guidance_engine=stub).post(
        ANALYZE_SEQUENCE_PATH,
        json=_body(scenario_id="3"),
    ).json()["data"]

    assert set(data) == _ANALYZE_DATA_KEYS
    assert data["guide"] == {
        "speed": {"status": "slow", "message": "speed-message", "available": True, "reason": None},
        "distance": {"status": "far", "message": "distance-message", "available": True, "reason": None},
    }
    assert stub.calls == 1


def test_telemetry_sequence_contract_does_not_gain_the_analyze_guide() -> None:
    data = _client(guidance_engine=_GuidanceStub(_guide())).post(URL, json=_body()).json()["data"]

    assert set(data) == _DATA_KEYS
    assert "guide" not in data


def test_analyze_without_reference_returns_explicitly_unavailable_guide() -> None:
    data = _client().post(ANALYZE_SEQUENCE_PATH, json=_body(scenario_id="3")).json()["data"]

    assert data["guide"]["speed"] == {
        "status": None,
        "message": None,
        "available": False,
        "reason": "reference_unavailable",
    }
    assert data["guide"]["distance"]["reason"] == "reference_unavailable"


def test_duplicate_analyze_returns_cached_guide_without_advancing_smoothing_state() -> None:
    stub = _GuidanceStub(_guide(speed="ok", distance="ok"))
    client = _client(guidance_engine=stub)

    first = client.post(ANALYZE_SEQUENCE_PATH, json=_body(scenario_id="3")).json()["data"]
    second = client.post(ANALYZE_SEQUENCE_PATH, json=_body(scenario_id="3")).json()["data"]

    assert first["guide"] == second["guide"]
    assert second["sequence_data_ack"]["duplicate"] is True
    assert stub.calls == 1


@pytest.mark.parametrize("mode_id", [0, 11, "x"])
def test_mode_id_outside_the_documented_range_is_rejected(mode_id: Any) -> None:
    """Given a mode_id outside 1~10, When posted, Then it is 422."""
    assert _client().post(ANALYZE_SEQUENCE_PATH, json=_body(mode_id=mode_id)).status_code == 422


def test_telemetry_sequence_does_not_take_mode_id() -> None:
    """Given mode_id on the telemetry path, When posted, Then it is rejected.

    mode_id 는 analyze-sequence 에만 있다(명세서 p.5 vs p.7).
    """
    assert _client().post(URL, json=_body(mode_id=1)).status_code == 422


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




@pytest.mark.parametrize(
    "missing",
    ["session_id", "user_id", "scenario_id", "zone_id", "sequence_id",
     "sequence_setting", "sequence_info", "sequence_result"],
)
def test_every_documented_required_field_is_required(missing: str) -> None:
    """Given a missing required field, When posted, Then it is 422.

    명세서 p.5 가 최상위 8키를 모두 `(필수)`로 적는다.
    """
    body = _body()
    del body[missing]

    assert _client().post(URL, json=body).status_code == 422




# --- 미구현 엔진 ----------------------------------------------------------


def test_unconnected_engines_report_distinct_stable_reasons() -> None:
    data = _client().post(URL, json=_body()).json()["data"]

    assert data["coating_feedback"] == {
        "analysis_status": "not_available", "reason_code": "coating_engine_pending_integration",
    }
    assert data["sequence_feedback"] == {
        "analysis_status": "not_available", "reason_code": "sequence_reference_unavailable",
    }


class _SequenceDecision:
    feedback = EngineFeedback(analysis_status="available", reason_code="zone_skipped")
    text_guide = "2번 구역을 먼저 도장하세요."
    anomaly_type = "sequence_order_deviation"


class _SequenceStub:
    def evaluate(self, _request: Any, _store: SequenceStore) -> _SequenceDecision:
        return _SequenceDecision()


def test_analyze_sequence_returns_feedback_and_publishes_the_same_anomaly() -> None:
    bus = AnomalyBroadcaster()
    data = _client(sequence_engine=_SequenceStub(), broadcaster=bus).post(
        ANALYZE_SEQUENCE_PATH, json=_body(scenario_id="3"),
    ).json()["data"]

    assert data["sequence_feedback"] == {"analysis_status": "available", "reason_code": "zone_skipped"}
    assert data["text_guide"] == "2번 구역을 먼저 도장하세요."
    events = bus.since("1_1001642_1", last_event_id=0)
    assert len(events) == 1
    assert events[0].anomaly_type == "sequence_order_deviation"


# --- 모니터 기록 ----------------------------------------------------------


def _monitored() -> tuple[TestClient, Any]:
    from trainer.demo_api.monitor import MonitorRecorder

    recorder = MonitorRecorder()
    app = FastAPI()
    app.include_router(
        create_telemetry_router(store=SequenceStore(":memory:"), recorder=recorder)
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


def test_a_conflicting_body_is_still_recorded_both_ways() -> None:
    """Given a body conflict, When posted, Then the monitor shows input and output.

    처리 전에 입력을 남기므로 이후 단계 결과와 무관하게 무엇이 들어왔는지는 남는다.
    """
    client, recorder = _monitored()
    _ = client.post(URL, json=_body())

    _ = client.post(URL, json=_body(sequence_info=[_frame(), _frame(2)]))

    events = recorder.snapshot(cursor=0, limit=50).events
    assert [(e.phase, e.direction) for e in events[2:]] == [("runtime", "input"), ("runtime", "output")]
    assert events[3].payload["data"]["sequence_data_ack"]["stored"] is False


def test_the_monitor_groups_by_session_id() -> None:
    """Given a sequence, When recorded, Then both events carry its session."""
    client, recorder = _monitored()

    _ = client.post(URL, json=_body())

    assert {e.session_id for e in recorder.snapshot(cursor=0, limit=50).events} == {"1_1001642_1"}


def test_no_recorder_means_no_recording_and_no_failure() -> None:
    """Given no recorder, When posted, Then the endpoint works unchanged."""
    assert _client().post(URL, json=_body()).status_code == 200
