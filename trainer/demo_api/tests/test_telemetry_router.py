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
from trainer.demo_api.telemetry_contracts import CoatingProgress
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
_ANALYZE_DATA_KEYS = _DATA_KEYS | {"guide", "coating_progress"}

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
    coating_progress_engine: Any = None,
) -> TestClient:
    """자세는 여기서 채점하지 않으므로 MCP 트랜스포트가 필요 없다(명세서 p.4)."""
    app = FastAPI()
    app.include_router(
        create_telemetry_router(
            store=store if store is not None else SequenceStore(":memory:"),
            guidance_engine=guidance_engine,
            sequence_engine=sequence_engine,
            broadcaster=broadcaster,
            coating_progress_engine=coating_progress_engine,
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


def test_raytexuv_uses_tsb_pixel_coordinates_and_accepts_a_miss() -> None:
    body = _body()
    body["sequence_info"] = [
        _frame(index=1, gun_metrics={"raytexuv": [2048, 0]}),
        _frame(index=2, gun_metrics={"raytexuv": None}),
    ]

    response = _client().post(ANALYZE_SEQUENCE_PATH, json=body)

    assert response.status_code == 200


def test_raytexuv_rejects_out_of_range_pixel_coordinates() -> None:
    body = _body()
    body["sequence_info"] = [_frame(gun_metrics={"raytexuv": [2048.1, 0]})]

    assert _client().post(ANALYZE_SEQUENCE_PATH, json=body).status_code == 422


def test_practical_scenario_id_is_not_coerced_to_learning_mode_one() -> None:
    from trainer.demo_api.telemetry_contracts import AnalyzeSequenceRequest

    request = AnalyzeSequenceRequest.model_validate(_body(scenario_id="1-R-ABC"))

    assert request.resolved_mode_id is None
    assert request.scenario_id == "1-R-ABC"


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


class _ProgressStub:
    def evaluate(self, request: Any) -> CoatingProgress:
        return CoatingProgress(
            zone_id=request.zone_id,
            available=True,
            status="needs_more",
            recommendation="continue_current_zone",
            completion_percent=62.5,
            coverage_percent=50.0,
            overpaint_percent=0.0,
            next_zone_id=None,
            message="Zone 1을 추가 도장하세요.",
            reason=None,
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


def test_analyze_sequence_returns_current_zone_coating_progress() -> None:
    data = _client(coating_progress_engine=_ProgressStub()).post(
        ANALYZE_SEQUENCE_PATH,
        json=_body(scenario_id="3"),
    ).json()["data"]

    assert data["coating_progress"] == {
        "zone_id": "1",
        "available": True,
        "status": "needs_more",
        "recommendation": "continue_current_zone",
        "completion_percent": 62.5,
        "coverage_percent": 50.0,
        "normal_band_percent": None,
        "overpaint_percent": 0.0,
        "next_zone_id": None,
        "message": "Zone 1을 추가 도장하세요.",
        "reason": None,
        "comparison_basis": "expert_reference",
        "target_thickness_um": None,
        "expert_completion_percent": None,
        "target_basis": None,
    }


_ZONE_ROW = {
    "zone_id": "1", "map_num": 1, "status": "under_app", "grade": "below", "thickness_um": 98.0,
    "target_thickness_um": 160.0, "target_basis": "default_target",
    # 실제 값(160, 240)이 아니다 — 라우터가 정상 범위를 다시 계산하지 않고 행의 값을 옮기는지 본다.
    "normal_band_um": [1.0, 2.0], "unpainted_fraction": 0.12,
    "six_band_counts": {"lt_0_9t": 88, "from_0_9t_to_lt_1_0t": 0, "from_1_0t_to_lt_1_5t": 0,
                        "from_1_5t_to_lt_2_0t": 0, "from_2_0t_to_lt_2_5t": 0, "gte_2_5t": 0},
    "coverage_percent": 0.0, "overpaint_percent": 0.0, "completion_percent": 53.9, "gate_status": "needs_more",
    "adjustments": [{"variable": "speed_mm_s", "direction": "decrease", "priority": 1,
                     "evidence_kind": "quality_rule_candidate"}],
    "adjustment_reason": None,
}


class _AssessedProgressStub:
    def evaluate(self, request: Any) -> CoatingProgress:
        return _ProgressStub().evaluate(request).model_copy(update={"assessment": dict(_ZONE_ROW)})


def test_analyze_sequence_fills_coating_feedback_from_the_assessment() -> None:
    client = _client(coating_progress_engine=_AssessedProgressStub())

    data = client.post(ANALYZE_SEQUENCE_PATH, json=_body(scenario_id="3")).json()["data"]
    feedback = data["coating_feedback"]

    assert set(data) == _ANALYZE_DATA_KEYS
    assert "assessment" not in data["coating_progress"]
    assert feedback["analysis_status"] == "available"
    assert feedback["reason_code"] is None
    for key in ("zone_id", "map_num", "status", "grade", "thickness_um", "target_thickness_um", "target_basis",
                "normal_band_um", "unpainted_fraction", "six_band_counts", "adjustments"):
        assert feedback[key] == _ZONE_ROW[key], key
    assert feedback["message"] == (
        "1구역: 저도막(칠한 곳 평균 98µm, 미도장 12%) "
        "조절 후보: 이동 속도를 낮추는 쪽(원인 미확정)."
    )
    assert feedback["algorithm_version"].startswith("coating-v1-adapter/")
    # 재전송은 저장된 분석을 돌려준다.
    assert client.post(ANALYZE_SEQUENCE_PATH, json=_body(scenario_id="3")).json()["data"]["coating_feedback"] == feedback


def _assessed(grade: str, *directions: tuple[str, str]) -> Any:
    row = {**_ZONE_ROW, "grade": grade, "adjustments": [
        {"variable": var, "direction": d, "priority": i + 1, "evidence_kind": "quality_rule_candidate"}
        for i, (var, d) in enumerate(directions)]}

    class _Stub:
        def evaluate(self, request: Any) -> CoatingProgress:
            return _ProgressStub().evaluate(request).model_copy(update={"assessment": dict(row)})

    return _Stub()


_BOTH = (("speed_mm_s", "decrease"), ("distance_mm", "decrease"))
_WITHHELD = {"status": None, "message": None, "available": False, "reason": "coating_adjustment_priority"}


def _analyze(guide: SequenceGuide, progress: Any) -> dict[str, Any]:
    client = _client(guidance_engine=_GuidanceStub(guide), coating_progress_engine=progress)
    return client.post(ANALYZE_SEQUENCE_PATH, json=_body(scenario_id="3")).json()["data"]


def test_opposite_expert_guide_is_withheld_when_coating_adjustment_leads() -> None:
    data = _analyze(_guide(speed="slow", distance="far"), _assessed("below", *_BOTH))

    assert data["guide"]["speed"] == _WITHHELD
    assert data["guide"]["distance"]["status"] == "far"
    assert [a["variable"] for a in data["coating_feedback"]["adjustments"]] == ["speed_mm_s", "distance_mm"]
    assert "이동 속도를 낮추는 쪽" in data["coating_feedback"]["message"]


def test_ok_guide_is_withheld_when_a_coating_adjustment_exists() -> None:
    data = _analyze(_guide(speed="ok", distance="far"), _assessed("below", *_BOTH))

    assert data["guide"]["speed"]["reason"] == "coating_adjustment_priority"


def test_agreeing_guide_and_adjustment_are_both_kept() -> None:
    data = _analyze(_guide(speed="fast", distance="far"), _assessed("below", *_BOTH))

    assert data["guide"] == _guide(speed="fast", distance="far").model_dump(mode="json")
    assert len(data["coating_feedback"]["adjustments"]) == 2


def test_expert_priority_drops_the_conflicting_adjustment_instead(monkeypatch: pytest.MonkeyPatch) -> None:
    from trainer.demo_api import telemetry_router

    monkeypatch.setattr(telemetry_router, "ADVICE_PRIORITY", "expert_guidance")
    data = _analyze(_guide(speed="slow", distance="far"), _assessed("below", *_BOTH))

    assert data["guide"] == _guide(speed="slow", distance="far").model_dump(mode="json")
    assert [a["variable"] for a in data["coating_feedback"]["adjustments"]] == ["distance_mm"]
    assert "이동 속도" not in data["coating_feedback"]["message"]


@pytest.mark.parametrize("grade,directions", [
    ("below", _BOTH),
    ("above", (("speed_mm_s", "increase"), ("distance_mm", "increase"))),
])
def test_no_response_carries_contradicting_speed_or_distance_advice(
    grade: str, directions: tuple[tuple[str, str], ...],
) -> None:
    from trainer.demo_api.telemetry_router import _CONFLICTING_GUIDE

    for speed in ("slow", "ok", "fast"):
        for distance in ("near", "ok", "far"):
            data = _analyze(_guide(speed=speed, distance=distance), _assessed(grade, *directions))
            for item in data["coating_feedback"]["adjustments"]:
                field = "speed" if item["variable"] == "speed_mm_s" else "distance"
                assert data["guide"][field]["status"] not in _CONFLICTING_GUIDE[(item["variable"], item["direction"])]


def test_analyze_sequence_reports_a_zone_with_nothing_painted_as_not_available() -> None:
    row = {**_ZONE_ROW, "status": None, "grade": None, "thickness_um": None, "unpainted_fraction": 1.0,
           "adjustments": [], "adjustment_reason": "no_valid_painted_pixels"}

    class _Unpainted:
        def evaluate(self, request: Any) -> CoatingProgress:
            return _ProgressStub().evaluate(request).model_copy(update={"assessment": row})

    feedback = _client(coating_progress_engine=_Unpainted()).post(
        ANALYZE_SEQUENCE_PATH, json=_body(scenario_id="3"),
    ).json()["data"]["coating_feedback"]

    assert (feedback["analysis_status"], feedback["reason_code"]) == ("not_available", "no_valid_painted_pixels")
    assert feedback["unpainted_fraction"] == 1.0
    assert feedback["message"] == "1구역: 칠한 곳 없음(미도장 100%)"


_NO_FEEDBACK = {
    "zone_id": None, "map_num": None, "status": None, "grade": None, "thickness_um": None,
    "target_thickness_um": None, "target_basis": None, "normal_band_um": None, "unpainted_fraction": None,
    "six_band_counts": None, "adjustments": [], "message": None, "algorithm_version": None,
}


def test_analyze_sequence_without_an_assessment_reports_why() -> None:
    stubbed = _client(coating_progress_engine=_ProgressStub()).post(
        ANALYZE_SEQUENCE_PATH, json=_body(scenario_id="3"),
    ).json()["data"]
    no_engine = _client().post(ANALYZE_SEQUENCE_PATH, json=_body(scenario_id="3")).json()["data"]

    assert stubbed["coating_feedback"] == {
        "analysis_status": "not_available", "reason_code": "coating_assessment_unavailable", **_NO_FEEDBACK}
    # 도막을 못 잰 까닭이 있으면 그 까닭을 그대로 싣는다.
    assert no_engine["coating_feedback"] == {
        "analysis_status": "not_available", "reason_code": "coating_reference_unavailable", **_NO_FEEDBACK}


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


def test_conflict_is_rejected_before_any_engine_or_anomaly_changes() -> None:
    stub = _GuidanceStub(_guide())
    bus = AnomalyBroadcaster()
    client = _client(guidance_engine=stub, sequence_engine=_SequenceStub(), broadcaster=bus)
    assert client.post(URL, json=_body()).status_code == 200
    response = client.post(ANALYZE_SEQUENCE_PATH, json=_body(zone_id="2"))
    assert response.status_code == 409
    assert stub.calls == 0
    assert bus.since(_body()["session_id"], last_event_id=0) == []


def test_late_new_sequence_is_rejected_before_storage_or_analysis() -> None:
    store = SequenceStore(":memory:")
    stub = _GuidanceStub(_guide())
    client = _client(store=store, guidance_engine=stub)
    assert client.post(URL, json=_body(sequence_id="SEQ-002")).status_code == 200
    response = client.post(ANALYZE_SEQUENCE_PATH, json=_body(sequence_id="SEQ-001"))
    assert response.status_code == 409
    assert "sequence_out_of_order" in response.json()["message"]
    assert store.load_body(session_key=_body()["session_id"], sequence_id="SEQ-001") is None
    assert stub.calls == 0


def test_duplicate_analysis_survives_router_restart_without_reapplying_engines() -> None:
    store = SequenceStore(":memory:")
    first_stub = _GuidanceStub(_guide())
    first = _client(store=store, guidance_engine=first_stub).post(ANALYZE_SEQUENCE_PATH, json=_body())
    second_stub = _GuidanceStub(_guide(speed="fast"))
    second = _client(store=store, guidance_engine=second_stub).post(ANALYZE_SEQUENCE_PATH, json=_body())
    assert second.status_code == 200
    assert second.json()["data"]["guide"] == first.json()["data"]["guide"]
    assert second_stub.calls == 0


def test_crop_placement_revision_refreshes_old_result_then_deduplicates_retries(monkeypatch: Any) -> None:
    from trainer.demo_api import telemetry_router as module
    store = SequenceStore(":memory:")
    bus = AnomalyBroadcaster()
    monkeypatch.setattr(module, "CROP_PLACEMENT_VERSION", "old-sparse-placement", raising=False)
    previous = _client(store=store, guidance_engine=_GuidanceStub(_guide(speed="slow")),
                       sequence_engine=_SequenceStub(), broadcaster=bus)
    assert previous.post(ANALYZE_SEQUENCE_PATH, json=_body()).status_code == 200

    monkeypatch.setattr(module, "CROP_PLACEMENT_VERSION", "new-nearest-placement", raising=False)
    guidance = _GuidanceStub(_guide(speed="fast"))
    updated = _client(store=store, guidance_engine=guidance, sequence_engine=_SequenceStub(), broadcaster=bus)
    refreshed = updated.post(ANALYZE_SEQUENCE_PATH, json=_body())
    repeated = updated.post(ANALYZE_SEQUENCE_PATH, json=_body())
    assert refreshed.status_code == repeated.status_code == 200
    assert refreshed.json()["data"]["guide"]["speed"]["status"] == "fast"
    assert refreshed.json()["data"]["sequence_data_ack"]["duplicate"] is True
    assert refreshed.json()["data"]["guide"] == repeated.json()["data"]["guide"]
    assert guidance.calls == 1
    assert len(bus.since(_body()["session_id"], last_event_id=0)) == 1


def test_pre_revision_historical_retry_returns_saved_result_without_replaying_engines() -> None:
    class PreRevisionStore(SequenceStore):
        def save_analysis(self, **kwargs: Any) -> None:
            key = json.loads(kwargs["analysis_key"])
            kwargs["analysis_key"] = json.dumps([key[0], *key[2:]])
            super().save_analysis(**kwargs)

    store = PreRevisionStore(":memory:")
    bus = AnomalyBroadcaster()
    previous = _client(store=store, guidance_engine=_GuidanceStub(_guide(speed="slow")),
                       sequence_engine=_SequenceStub(), broadcaster=bus)
    first = _body(sequence_id="SEQ-001")
    assert previous.post(ANALYZE_SEQUENCE_PATH, json=first).status_code == 200
    assert previous.post(URL, json=_body(sequence_id="SEQ-002")).status_code == 200
    assert previous.post(URL, json=_body(sequence_id="SEQ-003")).status_code == 200
    guidance = _GuidanceStub(_guide(speed="fast"))
    updated = _client(store=store, guidance_engine=guidance, sequence_engine=_SequenceStub(), broadcaster=bus)
    repeated = updated.post(ANALYZE_SEQUENCE_PATH, json=first)
    assert repeated.status_code == 200
    assert repeated.json()["data"]["guide"]["speed"]["status"] == "slow"
    assert repeated.json()["data"]["sequence_data_ack"]["duplicate"] is True
    assert updated.post(ANALYZE_SEQUENCE_PATH, json=_body(sequence_id="SEQ-002")).status_code == 409
    assert guidance.calls == 0
    assert len(bus.since(_body()["session_id"], last_event_id=0)) == 1


def test_historical_first_analysis_does_not_rewind_guidance_smoothing() -> None:
    store = SequenceStore(":memory:")
    stub = _GuidanceStub(_guide())
    client = _client(store=store, guidance_engine=stub)
    client.post(URL, json=_body(sequence_id="SEQ-001"))
    client.post(ANALYZE_SEQUENCE_PATH, json=_body(sequence_id="SEQ-002"))
    response = client.post(ANALYZE_SEQUENCE_PATH, json=_body(sequence_id="SEQ-001"))
    assert response.status_code == 409
    assert stub.calls == 1


def test_concurrent_duplicate_analysis_has_one_engine_evaluation() -> None:
    from concurrent.futures import ThreadPoolExecutor
    stub = _GuidanceStub(_guide())
    client = _client(guidance_engine=stub)
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(lambda _: client.post(ANALYZE_SEQUENCE_PATH, json=_body()), range(8)))
    assert all(response.status_code == 200 for response in responses)
    assert sum(response.json()["data"]["sequence_data_ack"]["stored"] for response in responses) == 1
    assert stub.calls == 1


def test_synchronous_engines_execute_off_the_asgi_event_loop() -> None:
    import threading
    import anyio

    class ThreadGuide(_GuidanceStub):
        thread_id: int | None = None
        def evaluate(self, request: Any) -> SequenceGuide:
            self.thread_id = threading.get_ident()
            return super().evaluate(request)

    async def run() -> None:
        stub = ThreadGuide(_guide())
        app = FastAPI()
        app.include_router(create_telemetry_router(store=SequenceStore(":memory:"), guidance_engine=stub))
        event_loop_thread = threading.get_ident()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(ANALYZE_SEQUENCE_PATH, json=_body())
        assert response.status_code == 200
        assert stub.thread_id != event_loop_thread

    anyio.run(run)


def test_coating_replay_receives_prior_telemetry_and_session_target() -> None:
    from types import SimpleNamespace

    class ReplayCoating:
        def evaluate_replay(self, request: Any, prior: Any, *, target_thickness_um: Any) -> CoatingProgress:
            assert request.sequence_id == "SEQ-002"
            assert [item.sequence_id for item in prior] == ["SEQ-001"]
            assert target_thickness_um == 175
            return CoatingProgress(
                zone_id="1", available=False, status=None, recommendation=None,
                completion_percent=None, coverage_percent=None, overpaint_percent=None,
                next_zone_id=None, message=None, reason="test",
            )

    class Sessions:
        def get_session(self, _session_id: str) -> Any:
            return SimpleNamespace(target_thickness_um=175)

    app = FastAPI()
    app.include_router(create_telemetry_router(
        store=SequenceStore(":memory:"), coating_progress_engine=ReplayCoating(), session_store=Sessions(),
    ))
    client = TestClient(app)
    assert client.post(URL, json=_body(sequence_id="SEQ-001")).status_code == 200
    assert client.post(ANALYZE_SEQUENCE_PATH, json=_body()).status_code == 200


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
    """A conflicting payload cannot be acknowledged as successfully evaluated."""
    client = _client()
    _ = client.post(URL, json=_body())
    conflict = client.post(URL, json=_body(sequence_info=[_frame(), _frame(2)]))
    assert conflict.status_code == 409
    assert conflict.json() == {
        "status": "ERROR", "code": 409,
        "message": "sequence_body_conflict: sequence_id already has different content", "data": None,
    }


# --- 엄격성 정책 ----------------------------------------------------------


def test_top_level_typo_is_rejected_with_the_v25_error_envelope() -> None:
    """Given a misspelled top-level key, When posted, Then it fails loudly."""
    body = _body()
    body["sesion_id"] = body.pop("session_id")

    response = _client().post(URL, json=body)

    assert response.status_code == 422
    assert response.json() == {
        "status": "ERROR", "code": 422,
        "message": "Invalid request: body.session_id (Field required)", "data": None,
    }


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
    assert "현재 1구역" in data["text_guide"]
    assert "2번" not in data["text_guide"]
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
    assert events[3].payload["status"] == "ERROR"
    assert events[3].payload["code"] == 409


def test_the_monitor_groups_by_session_id() -> None:
    """Given a sequence, When recorded, Then both events carry its session."""
    client, recorder = _monitored()

    _ = client.post(URL, json=_body())

    assert {e.session_id for e in recorder.snapshot(cursor=0, limit=50).events} == {"1_1001642_1"}


def test_no_recorder_means_no_recording_and_no_failure() -> None:
    """Given no recorder, When posted, Then the endpoint works unchanged."""
    assert _client().post(URL, json=_body()).status_code == 200
