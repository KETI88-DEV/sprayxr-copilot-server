from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.monitor import MonitorRecorder
from trainer.demo_api.sim_session import SimSessionStore
from trainer.demo_api.simulator import create_simulator_router
from trainer.demo_api.zone_resolver import NullZoneResolver


class _FixedResolver:
    def __init__(self, zone_id: str | None) -> None:
        self._zone_id = zone_id

    async def resolve(self, coordinates: list[float]) -> str | None:
        _ = coordinates
        return self._zone_id


def _store_client(
    store: SimSessionStore, resolver: object = None
) -> TestClient:
    app = FastAPI()
    app.include_router(
        create_simulator_router(
            MonitorRecorder(), store, resolver if resolver is not None else NullZoneResolver()
        )
    )
    return TestClient(app)


def _client() -> tuple[TestClient, MonitorRecorder]:
    recorder = MonitorRecorder()
    app = FastAPI()
    app.include_router(
        create_simulator_router(recorder, SimSessionStore(), NullZoneResolver())
    )
    return TestClient(app), recorder


def _start_body() -> dict[str, object]:
    return {
        "session_id": "s1",
        "level_id": "SP0035-L01",
        "worker": {"trainee_id": "t", "skill_level": "beginner", "handedness": "right"},
        "spec": {"nominal_um": 100.0},
    }


def _complete_body() -> dict[str, object]:
    return {
        "session_id": "s1",
        "scenario_id": "sc",
        "mode": "live",
        "zone_id": "A-2",
        "zone_type": "panel",
        "metrics": {
            "avg_dft": 100.0,
            "dft_values": [100.0],
            "avg_speed": 0.8,
            "speed_variance": 0.05,
            "avg_distance": 0.3,
            "distance_variance": 0.05,
            "avg_angle": 70.0,
            "angle_consistency": 0.9,
            "sequence_correct": False,
        },
    }


def test_start_returns_recommended_flow_and_records_input_output() -> None:
    client, recorder = _client()

    response = client.post("/api/sim/sessions/start", json=_start_body())

    assert response.status_code == 200
    assert response.json()["recommended_flow"] == ["A-1", "A-2", "A-3"]
    snapshot = recorder.snapshot(cursor=0, limit=10)
    assert [(event.phase, event.direction) for event in snapshot.events] == [
        ("start", "input"),
        ("start", "output"),
    ]
    assert snapshot.events[0].payload["level_id"] == "SP0035-L01"
    assert snapshot.events[1].payload["flow_source"] == "rule"


def test_event_anomaly_round_trip_records_pair() -> None:
    client, recorder = _client()
    body = {
        "event_id": "e1",
        "telemetry": {"gun_metrics": {"speed_m_per_s": 0.8, "standoff_mm": 350.0, "angle_deg": 55.0}},
    }

    response = client.post("/api/sim/sessions/s1/events", json=body)

    assert response.status_code == 200
    assert response.json()["kind"] == "anomaly"
    snapshot = recorder.snapshot(cursor=0, limit=10)
    assert [(event.phase, event.direction) for event in snapshot.events] == [
        ("runtime", "input"),
        ("runtime", "output"),
    ]
    assert snapshot.events[0].session_id == "s1"


def test_invalid_input_returns_422_and_records_error_output() -> None:
    client, recorder = _client()

    response = client.post("/api/sim/sessions/start", json={"session_id": "s1"})

    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"
    snapshot = recorder.snapshot(cursor=0, limit=10)
    assert snapshot.events[0].direction == "input"
    assert snapshot.events[1].payload["error"] == "validation_error"


def test_path_body_session_id_mismatch_rejected() -> None:
    client, recorder = _client()

    response = client.post("/api/sim/sessions/s1/events", json={"session_id": "OTHER", "event_id": "e1"})

    assert response.status_code == 422
    assert response.json()["error"] == "session_id_mismatch"
    snapshot = recorder.snapshot(cursor=0, limit=10)
    assert snapshot.events[1].payload["error"] == "session_id_mismatch"


def test_malformed_json_is_rejected_and_recorded() -> None:
    client, recorder = _client()

    response = client.post(
        "/api/sim/sessions/s1/events",
        content=b"{not json",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422
    assert response.json()["error"] == "invalid_json"
    snapshot = recorder.snapshot(cursor=0, limit=10)
    assert [event.payload["error"] for event in snapshot.events] == ["invalid_json", "invalid_json"]


def test_complete_review_uses_risk_flags_and_records_pair() -> None:
    client, recorder = _client()

    response = client.post("/api/sim/sessions/s1/complete", json=_complete_body())

    assert response.status_code == 200
    assert any(flag["flag"] == "sequence_violation" for flag in response.json()["risk_flags"])
    snapshot = recorder.snapshot(cursor=0, limit=10)
    assert [(event.phase, event.direction) for event in snapshot.events] == [
        ("session_end", "input"),
        ("session_end", "output"),
    ]


def test_full_session_records_two_events_per_call() -> None:
    client, recorder = _client()

    assert client.post("/api/sim/sessions/start", json=_start_body()).status_code == 200
    assert client.post("/api/sim/sessions/s1/events", json={"event_id": "e1"}).status_code == 200
    assert client.post("/api/sim/sessions/s1/complete", json=_complete_body()).status_code == 200

    snapshot = recorder.snapshot(cursor=0, limit=50)
    assert [(event.phase, event.direction) for event in snapshot.events] == [
        ("start", "input"),
        ("start", "output"),
        ("runtime", "input"),
        ("runtime", "output"),
        ("session_end", "input"),
        ("session_end", "output"),
    ]
    assert [event.seq for event in snapshot.events] == [1, 2, 3, 4, 5, 6]


START_BODY: dict[str, object] = {
    "session_id": "s-1",
    "level_id": "SP0035-L01",
    "worker": {"trainee_id": "t-1", "skill_level": "beginner", "handedness": "right"},
    "spec": {"nominal_um": 100.0},
}


def test_start_populates_store_with_flow() -> None:
    store = SimSessionStore()
    response = _store_client(store).post("/api/sim/sessions/start", json=START_BODY)
    assert response.status_code == 200
    state = store.current()
    assert state is not None
    assert state.recommended_flow == ("A-1", "A-2", "A-3")
    assert state.current_zone_id == "A-1"
    assert state.phase == "start"


def test_runtime_zone_id_moves_current_position() -> None:
    store = SimSessionStore()
    client = _store_client(store)
    _ = client.post("/api/sim/sessions/start", json=START_BODY)
    response = client.post(
        "/api/sim/sessions/s-1/events", json={"event_id": "ev-0", "zone_id": "A-2"}
    )
    assert response.status_code == 200
    state = store.current()
    assert state is not None
    assert state.current_zone_id == "A-2"
    assert state.completed_zone_ids == ("A-1",)


def test_runtime_falls_back_to_resolver_when_zone_id_absent() -> None:
    store = SimSessionStore()
    client = _store_client(store, _FixedResolver("A-3"))
    _ = client.post("/api/sim/sessions/start", json=START_BODY)
    _ = client.post(
        "/api/sim/sessions/s-1/events",
        json={"event_id": "ev-0", "telemetry": {"coordinates": [1.0, 2.0, 3.0]}},
    )
    state = store.current()
    assert state is not None
    assert state.current_zone_id == "A-3"


def test_runtime_records_unresolved_zone_outside_flow() -> None:
    store = SimSessionStore()
    client = _store_client(store, _FixedResolver("B-3"))
    _ = client.post("/api/sim/sessions/start", json=START_BODY)
    _ = client.post(
        "/api/sim/sessions/s-1/events",
        json={"event_id": "ev-0", "telemetry": {"coordinates": [1.0, 2.0, 3.0]}},
    )
    state = store.current()
    assert state is not None
    assert state.current_zone_id == "A-1"
    assert state.unresolved_zone_id == "B-3"


def test_runtime_anomaly_output_increments_counter() -> None:
    store = SimSessionStore()
    client = _store_client(store)
    _ = client.post("/api/sim/sessions/start", json=START_BODY)
    _ = client.post(
        "/api/sim/sessions/s-1/events",
        json={
            "event_id": "ev-0",
            "telemetry": {
                "gun_metrics": {"speed_m_per_s": 0.8, "standoff_mm": 350.0, "angle_deg": 55.0}
            },
        },
    )
    state = store.current()
    assert state is not None
    assert state.anomaly_count == 1


def test_complete_marks_session_end_in_store() -> None:
    store = SimSessionStore()
    client = _store_client(store)
    _ = client.post("/api/sim/sessions/start", json=START_BODY)
    body = {**_complete_body(), "session_id": "s-1"}
    response = client.post("/api/sim/sessions/s-1/complete", json=body)
    assert response.status_code == 200
    state = store.current()
    assert state is not None
    assert state.phase == "session_end"


def test_rejected_payload_leaves_store_untouched() -> None:
    store = SimSessionStore()
    client = _store_client(store)
    _ = client.post("/api/sim/sessions/start", json=START_BODY)
    response = client.post("/api/sim/sessions/s-1/events", json={"zone_id": "A-2"})
    assert response.status_code == 422
    state = store.current()
    assert state is not None
    assert state.phase == "start"
    assert state.current_zone_id == "A-1"


def test_session_endpoint_returns_null_before_any_start() -> None:
    response = _store_client(SimSessionStore()).get("/api/sim/session")
    assert response.status_code == 200
    assert response.json() == {"session": None}


def test_session_endpoint_exposes_current_state() -> None:
    store = SimSessionStore()
    client = _store_client(store)
    _ = client.post("/api/sim/sessions/start", json=START_BODY)
    body = client.get("/api/sim/session").json()
    assert body["session"]["session_id"] == "s-1"
    assert body["session"]["phase"] == "start"
    assert body["session"]["recommended_flow"] == ["A-1", "A-2", "A-3"]
    assert body["session"]["unresolved_zone_id"] is None


def test_store_is_updated_before_the_output_event_is_published() -> None:
    """구독자는 output 이벤트를 재조회 신호로 쓴다. 그 시점에 상태가 이미 보여야 한다."""
    store = SimSessionStore()
    observed: list[str | None] = []

    class _OrderingRecorder(MonitorRecorder):
        def record(self, session_id, phase, direction, payload):  # type: ignore[no-untyped-def]
            if direction == "output":
                state = store.current()
                observed.append(None if state is None else state.current_zone_id)
            return super().record(session_id, phase, direction, payload)

    app = FastAPI()
    app.include_router(create_simulator_router(_OrderingRecorder(), store, NullZoneResolver()))
    client = TestClient(app)

    _ = client.post("/api/sim/sessions/start", json=START_BODY)
    _ = client.post("/api/sim/sessions/s-1/events", json={"event_id": "ev-0", "zone_id": "A-3"})

    assert observed == ["A-1", "A-3"]


def test_rejected_payloads_leave_the_whole_state_identical_and_record_two_events() -> None:
    store = SimSessionStore()
    recorder = MonitorRecorder()
    app = FastAPI()
    app.include_router(create_simulator_router(recorder, store, NullZoneResolver()))
    client = TestClient(app)

    _ = client.post("/api/sim/sessions/start", json=START_BODY)
    before = store.current()
    assert before is not None
    baseline = recorder.snapshot(cursor=0, limit=1000).next_cursor

    rejects: list[tuple[str, object]] = [
        ("/api/sim/sessions/s-1/events", {"zone_id": "A-2"}),
        ("/api/sim/sessions/s-1/events", {"event_id": "ev-0", "session_id": "other"}),
        ("/api/sim/sessions/s-1/complete", {"session_id": "s-1"}),
    ]
    for path, body in rejects:
        response = client.post(path, json=body)
        assert response.status_code == 422, (path, response.text)

    malformed = client.post(
        "/api/sim/sessions/s-1/events",
        content=b"{not json",
        headers={"Content-Type": "application/json"},
    )
    assert malformed.status_code == 422

    assert store.current() == before
    after = recorder.snapshot(cursor=baseline, limit=1000)
    assert len(after.events) == 2 * (len(rejects) + 1)
