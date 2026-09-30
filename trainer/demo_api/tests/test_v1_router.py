"""v2.5 공개 API 12개: 세션·사용자·리포트·이상 알림 SSE·테스트.

정본은 대화형 API 명세서다. 쪽 번호를 각 시험에 적는다.
"""

from __future__ import annotations

from typing import Any

import anyio
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.anomaly_stream import AnomalyBroadcaster
from trainer.demo_api.v1_router import _sse_body, create_v1_router
from trainer.demo_api.v1_store import V1Store

_ENVELOPE = {"status", "code", "message", "data"}


def _app(*, broadcaster: AnomalyBroadcaster | None = None) -> tuple[TestClient, AnomalyBroadcaster]:
    bus = broadcaster if broadcaster is not None else AnomalyBroadcaster()
    app = FastAPI()
    app.include_router(create_v1_router(store=V1Store(":memory:"), broadcaster=bus))
    return TestClient(app), bus


def _start_body(**over: Any) -> dict[str, Any]:
    body = {
        "session_id": "1_1000053_1",
        "user_id": "W-017",
        "scenario_id": "1-A",
        "started_at": "2026-08-07T14:00:00Z",
        "worker": {"skill_level": "novice", "handedness": "right"},
    }
    body.update(over)
    return body


# --- 1. 세션 관리 (p.1-3) --------------------------------------------------


def test_every_response_uses_the_documented_envelope() -> None:
    """Given any call, When answered, Then the p.25 APIResponse wrapper is used."""
    client, _ = _app()

    body = client.post("/api/v1/sessions/start", json=_start_body()).json()

    assert set(body) == _ENVELOPE
    assert (body["status"], body["code"]) == ("SUCCESS", 200)


def test_start_creates_the_user_then_updates_it() -> None:
    """Given a new user_id, When a session starts, Then it is created; a second start updates.

    p.1: "입력된 user_id 가 DB에 없을 경우 신규 유저로 자동 생성하고, 기존 유저일 경우
    숙련도 및 파지 손 정보를 최신 상태로 업데이트합니다."
    """
    client, _ = _app()

    first = client.post("/api/v1/sessions/start", json=_start_body()).json()["data"]
    second = client.post(
        "/api/v1/sessions/start",
        json=_start_body(session_id="1_1000053_2", worker={"skill_level": "expert", "handedness": "left"}),
    ).json()["data"]

    assert first["user_created"] is True
    assert first["session_status"] == "ACTIVE"
    assert second["user_created"] is False
    profile = client.get("/api/v1/users/W-017").json()["data"]
    assert (profile["skill_level"], profile["dominant_hand"]) == ("expert", "left")


def test_end_marks_the_session_completed_and_keeps_started_at() -> None:
    """Given a started session, When ended, Then it is COMPLETED with both timestamps (p.3)."""
    client, _ = _app()
    _ = client.post("/api/v1/sessions/start", json=_start_body())

    data = client.post(
        "/api/v1/sessions/end",
        json={
            "session_id": "1_1000053_1", "user_id": "W-017",
            "scenario_id": "1-A", "ended_at": "2026-08-07T14:30:00Z",
        },
    ).json()["data"]

    assert data["session_status"] == "COMPLETED"
    assert data["started_at"] == "2026-08-07T14:00:00Z"
    assert data["ended_at"] == "2026-08-07T14:30:00Z"


def test_end_without_a_prior_start_is_still_recorded() -> None:
    """Given no start, When ended, Then the notice is not lost; started_at stays unknown."""
    client, _ = _app()

    data = client.post(
        "/api/v1/sessions/end",
        json={"session_id": "S-none", "user_id": "W-1", "scenario_id": "1-A", "ended_at": "t"},
    ).json()["data"]

    assert data["session_status"] == "COMPLETED"
    assert data["started_at"] is None


@pytest.mark.parametrize(
    "missing", ["session_id", "user_id", "scenario_id", "started_at", "worker"]
)
def test_start_requires_every_documented_field(missing: str) -> None:
    """Given a missing required field, When posted, Then it is 422 (p.1)."""
    client, _ = _app()
    body = _start_body()
    del body[missing]

    assert client.post("/api/v1/sessions/start", json=body).status_code == 422


def test_worker_is_optional_on_end_but_required_on_start() -> None:
    """Given the two endpoints, When worker is omitted, Then only end accepts it (p.1 vs p.3)."""
    client, _ = _app()

    start = client.post("/api/v1/sessions/start", json={k: v for k, v in _start_body().items() if k != "worker"})
    end = client.post(
        "/api/v1/sessions/end",
        json={"session_id": "S1", "user_id": "W-1", "scenario_id": "1-A", "ended_at": "t"},
    )

    assert (start.status_code, end.status_code) == (422, 200)


# --- 5. 리포트 (p.11-16) --------------------------------------------------


def _complete(client: TestClient, **over: Any):
    form = {
        "session_id": "1_1000053_1", "user_id": "W-017", "scenario_id": "1-A",
        "vr_score": "88", "is_pass": "true", "avg_thickness_um": "125.5",
        "avg_speed_mms": "2750", "avg_distance_mm": "300",
    }
    form.update({k: str(v) for k, v in over.items()})
    return client.post(
        "/api/v1/sessions/complete",
        data=form,
        files={"telemetry_csv": ("log.csv", b"index,time\n1,0\n", "text/csv")},
    )


def test_complete_assigns_a_report_id_and_starts_pending() -> None:
    """Given a completion, When requested, Then a report_id is issued as pending.

    p.12: "실제 분석 작업이 연결되기 전에는 complete 로 표시하지 않습니다."
    """
    client, _ = _app()

    data = _complete(client).json()["data"]

    assert data["report_status"] == "pending"
    assert data["report_id"].startswith("RPT-")
    assert data["telemetry_csv_bytes"] == len(b"index,time\n1,0\n")
    assert data["height_map_image_bytes"] is None


def test_complete_rejects_a_vr_score_outside_the_documented_range() -> None:
    """Given vr_score outside 0~100, When posted, Then it is 422 (p.12)."""
    client, _ = _app()

    assert _complete(client, vr_score=101).status_code == 422


def test_complete_requires_the_telemetry_csv() -> None:
    """Given no CSV, When posted, Then it is 422 — p.12 marks the file 필수."""
    client, _ = _app()

    response = client.post(
        "/api/v1/sessions/complete",
        data={
            "session_id": "S1", "user_id": "W-1", "scenario_id": "1-A", "vr_score": "88",
            "is_pass": "true", "avg_thickness_um": "1", "avg_speed_mms": "1", "avg_distance_mm": "1",
        },
    )

    assert response.status_code == 422


def test_report_polling_finds_the_report_and_rejects_a_wrong_session() -> None:
    """Given a report_id, When polled, Then the status is returned only under its own session (p.14)."""
    client, _ = _app()
    report_id = _complete(client).json()["data"]["report_id"]

    ok = client.get(f"/api/v1/sessions/1_1000053_1/reports/{report_id}")
    wrong = client.get(f"/api/v1/sessions/other/reports/{report_id}")

    assert ok.json()["data"]["report_status"] == "pending"
    assert wrong.status_code == 404
    assert wrong.json()["status"] == "ERROR"


def test_report_detail_returns_unavailable_areas_without_inventing_numbers() -> None:
    """Given no analysis engine, When the report is read, Then every area says unavailable.

    p.15: "실제 분석이 아직 없으면 상태는 pending, 각 분석 영역은 unavailable 로 반환됩니다."
    """
    client, _ = _app()
    report_id = _complete(client).json()["data"]["report_id"]

    data = client.get(f"/api/v1/reports/{report_id}").json()["data"]

    assert data["report_status"] == "pending"
    assert data["llm_briefing"] is None
    for area in ("coating_report", "sequence_report", "pose_report"):
        assert data[area]["analysis_status"] == "unavailable"
    assert data["coating_report"]["coating_distribution_index"] is None
    assert data["sequence_report"]["adherence_rate"] is None
    assert data["pose_report"]["average_similarity_score"] is None


def test_unknown_report_is_404() -> None:
    """Given an unknown report_id, When read, Then it is 404, not an empty report."""
    client, _ = _app()

    assert client.get("/api/v1/reports/RPT-nope").status_code == 404


# --- 6. 사용자 관리 (p.17-20) ---------------------------------------------


def test_user_create_and_get_round_trip() -> None:
    """Given a new user, When created, Then only user_id was required (p.18)."""
    client, _ = _app()

    created = client.post("/api/v1/users", json={"user_id": "W-900"}).json()["data"]
    fetched = client.get("/api/v1/users/W-900").json()["data"]

    assert created == fetched
    assert (created["skill_level"], created["dominant_hand"]) == ("novice", "right")


def test_creating_an_existing_user_updates_and_says_so() -> None:
    """Given an existing user_id, When posted again, Then the message reports the update."""
    client, _ = _app()
    _ = client.post("/api/v1/users", json={"user_id": "W-900", "skill_level": "novice"})

    body = client.post("/api/v1/users", json={"user_id": "W-900", "skill_level": "expert"}).json()

    assert "already existed" in body["message"]
    assert body["data"]["skill_level"] == "expert"


def test_user_list_honours_skip_and_limit() -> None:
    """Given many users, When listed, Then pagination follows p.17 (skip 0, limit 100)."""
    client, _ = _app()
    for index in range(5):
        _ = client.post("/api/v1/users", json={"user_id": f"W-{index}"})

    everyone = client.get("/api/v1/users").json()["data"]
    page = client.get("/api/v1/users", params={"skip": 2, "limit": 2}).json()["data"]

    assert len(everyone) == 5
    assert [row["user_id"] for row in page] == ["W-2", "W-3"]


@pytest.mark.parametrize(("skip", "limit"), [(-1, 100), (0, 0), (0, 1001)])
def test_pagination_outside_the_documented_bounds_is_rejected(skip: int, limit: int) -> None:
    """Given out-of-range paging, When listed, Then it is 422 (p.17: limit 1~1000, skip ≥ 0)."""
    client, _ = _app()

    assert client.get("/api/v1/users", params={"skip": skip, "limit": limit}).status_code == 422


def test_unknown_user_is_404() -> None:
    """Given an unknown user_id, When read, Then it is 404."""
    client, _ = _app()

    assert client.get("/api/v1/users/nobody").status_code == 404


# --- 7. 테스트 전용 (p.21-24) ---------------------------------------------


def test_normal_case_publishes_no_anomaly() -> None:
    """Given the normal pass, When simulated, Then nothing is broadcast — 이상만 보낸다 (p.10)."""
    client, bus = _app()

    data = client.post("/api/v1/test/sequence/normal").json()["data"]

    assert data["anomaly_published"] is False
    assert bus.since("1_1000053_1", last_event_id=0) == []


@pytest.mark.parametrize(
    ("case", "anomaly_type"),
    [("abnormal-vertical", "direction_deviation"), ("abnormal-horizontal", "distance_deviation")],
)
def test_abnormal_cases_broadcast_to_the_subscribed_session(case: str, anomaly_type: str) -> None:
    """Given an abnormal pass, When simulated, Then the session's stream gets one event (p.22-24)."""
    client, bus = _app()

    data = client.post(f"/api/v1/test/sequence/{case}", params={"session_id": "S-1"}).json()["data"]
    events = bus.since("S-1", last_event_id=0)

    assert data["anomaly_published"] is True
    assert [event.anomaly_type for event in events] == [anomaly_type]
    assert events[0].payload["message"] == data["coach_message"]


def test_the_same_anomaly_is_not_published_twice_for_one_sequence() -> None:
    """Given a repeat, When simulated again, Then it is suppressed — 계약 §8.1."""
    client, bus = _app()

    first = client.post("/api/v1/test/sequence/abnormal-vertical", params={"session_id": "S-2"}).json()["data"]
    second = client.post("/api/v1/test/sequence/abnormal-vertical", params={"session_id": "S-2"}).json()["data"]

    assert (first["anomaly_published"], second["anomaly_published"]) == (True, False)
    assert len(bus.since("S-2", last_event_id=0)) == 1


def test_starting_a_session_clears_its_earlier_anomalies() -> None:
    """Given a reused session_id, When it starts again, Then earlier suppression is cleared."""
    client, bus = _app()
    _ = client.post("/api/v1/test/sequence/abnormal-vertical", params={"session_id": "1_1000053_1"})

    _ = client.post("/api/v1/sessions/start", json=_start_body())
    again = client.post("/api/v1/test/sequence/abnormal-vertical", params={"session_id": "1_1000053_1"}).json()

    assert again["data"]["anomaly_published"] is True
    assert len(bus.since("1_1000053_1", last_event_id=0)) == 1


# --- 4. 이상 알림 SSE (p.10) ----------------------------------------------
#
# 비종료 SSE 라우트를 TestClient 로 몰면 멈추지 않는다(AGENTS.md). 순수 생성기를 anyio 로
# 직접 시험한다.


def _first_frames(bus: AnomalyBroadcaster, session_id: str, count: int, *, cursor: int = 0) -> list[bytes]:
    async def _drain() -> list[bytes]:
        frames: list[bytes] = []
        agen = _sse_body(bus, session_id, cursor)
        # 취소 범위는 async 안에서만 쓸 수 있다. heartbeat 간격이 15초라 상한을 둔다.
        with anyio.move_on_after(5):
            try:
                async for frame in agen:
                    frames.append(frame)
                    if len(frames) >= count:
                        break
            finally:
                await agen.aclose()
        return frames

    return anyio.run(_drain)


def test_the_stream_sends_the_event_then_a_heartbeat() -> None:
    """Given one anomaly, When subscribed, Then the event arrives with an id, then keep-alive."""
    bus = AnomalyBroadcaster()
    _ = bus.publish(session_id="S-3", sequence_id="Q1", anomaly_type="direction_deviation", payload={"m": 1})

    frames = _first_frames(bus, "S-3", 2)

    assert b"event: anomaly_detection" in frames[0]
    assert b"id: 1" in frames[0]
    assert b'"anomaly_type": "direction_deviation"' in frames[0]
    assert frames[1] == b": keep-alive\n\n"


def test_a_quiet_session_only_gets_heartbeats() -> None:
    """Given no anomaly, When subscribed, Then only keep-alive is sent — 이상만 보낸다."""
    frames = _first_frames(AnomalyBroadcaster(), "S-quiet", 1)

    assert frames == [b": keep-alive\n\n"]


def test_last_event_id_resumes_after_the_cursor() -> None:
    """Given Last-Event-ID, When reconnected, Then earlier events are not replayed."""
    bus = AnomalyBroadcaster()
    _ = bus.publish(session_id="S-4", sequence_id="Q1", anomaly_type="a", payload={})
    _ = bus.publish(session_id="S-4", sequence_id="Q2", anomaly_type="a", payload={})

    frames = _first_frames(bus, "S-4", 1, cursor=1)

    assert b"id: 2" in frames[0]
    assert b"id: 1" not in frames[0]
