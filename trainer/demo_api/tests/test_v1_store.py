"""v2.5 공개 API 상태 저장소: 사용자 upsert, 세션 전이, 리포트 발급."""

from __future__ import annotations

from pathlib import Path

from trainer.demo_api.v1_store import V1Store


def _store() -> V1Store:
    return V1Store(":memory:")


def test_upsert_creates_then_updates_only_the_documented_fields() -> None:
    """Given an existing user, When upserted, Then only skill_level and hand change (p.1)."""
    store = _store()
    _ = store.upsert_user(
        user_id="W-1", skill_level="novice", dominant_hand="right",
        name="이름", age=40, gender="M", created_at="t",
    )

    updated = store.upsert_user(
        user_id="W-1", skill_level="expert", dominant_hand="left", created_at="t2"
    )
    row = store.get_user("W-1")

    assert updated is False
    assert row is not None
    assert (row.skill_level, row.dominant_hand) == ("expert", "left")
    # 이름·나이·성별은 /sessions/start 가 보내지 않으므로 지워지면 안 된다.
    assert (row.name, row.age, row.gender) == ("이름", 40, "M")


def test_restarting_a_session_clears_the_end_time() -> None:
    """Given an ended session, When started again, Then it is ACTIVE with no ended_at."""
    store = _store()
    _ = store.start_session(session_id="S1", user_id="W-1", scenario_id="1-A", started_at="t0")
    _ = store.end_session(session_id="S1", user_id="W-1", scenario_id="1-A", ended_at="t1")

    row = store.start_session(session_id="S1", user_id="W-1", scenario_id="1-A", started_at="t2")

    assert (row.session_status, row.started_at, row.ended_at) == ("ACTIVE", "t2", None)


def test_session_target_thickness_round_trips_and_updates() -> None:
    store = _store()

    first = store.start_session(
        session_id="S1", user_id="W-1", scenario_id="3", started_at="t0",
        target_thickness_um=150.0,
    )
    second = store.start_session(
        session_id="S1", user_id="W-1", scenario_id="3", started_at="t1",
        target_thickness_um=175.5,
    )

    assert first.target_thickness_um == 150.0
    assert second.target_thickness_um == 175.5
    assert store.get_session("S1").target_thickness_um == 175.5  # type: ignore[union-attr]


def test_report_ids_are_unique_per_request() -> None:
    """Given two completions of one session, When requested, Then each gets its own report_id."""
    store = _store()

    first = store.create_completion(session_id="S1", requested_at="t", payload={})
    second = store.create_completion(session_id="S1", requested_at="t", payload={})

    assert first.report_id != second.report_id
    assert store.get_completion(first.report_id) is not None


def test_completion_payload_round_trips() -> None:
    """Given a completion payload, When read back, Then the stored values are intact."""
    store = _store()
    payload = {"vr_score": 88, "is_pass": True, "avg_thickness_um": 125.5}

    report_id = store.create_completion(session_id="S1", requested_at="t", payload=payload).report_id

    row = store.get_completion(report_id)
    assert row is not None
    assert row.payload == payload


def test_state_survives_reopen(tmp_path: Path) -> None:
    """Given a file-backed store, When reopened, Then users and sessions are still there."""
    path = str(tmp_path / "state.db")
    first = V1Store(path)
    _ = first.upsert_user(user_id="W-1", skill_level="expert", dominant_hand="right", created_at="t")
    _ = first.start_session(session_id="S1", user_id="W-1", scenario_id="1-A", started_at="t0")
    first.close()

    reopened = V1Store(path)

    assert reopened.get_user("W-1") is not None
    assert reopened.get_session("S1") is not None


def test_unknown_rows_are_none_not_empty() -> None:
    """Given unknown ids, When read, Then None comes back — 빈 레코드를 지어내지 않는다."""
    store = _store()

    assert store.get_user("nobody") is None
    assert store.get_session("nothing") is None
    assert store.get_completion("RPT-nope") is None
