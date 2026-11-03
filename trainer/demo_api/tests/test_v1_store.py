"""v2.5 공개 API 상태 저장소: 사용자 upsert, 세션 전이, 리포트 발급."""

from __future__ import annotations

from pathlib import Path
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from trainer.demo_api.v1_store import SessionConflict, V1Store


def _store() -> V1Store:
    return V1Store(":memory:")


def test_server_creation_timestamp_is_recorded_once_without_inventing_legacy_times(tmp_path) -> None:
    path = tmp_path / "created.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY,user_id TEXT,scenario_id TEXT,session_status TEXT,started_at TEXT,ended_at TEXT)")
        conn.execute("INSERT INTO sessions VALUES ('OLD','W','3','ACTIVE','2026-10-01T00:00:00Z',NULL)")
    store = V1Store(str(path))
    store.start_session(session_id="NEW",user_id="W",scenario_id="3",started_at="2026-11-03T00:00:00Z")
    with sqlite3.connect(path) as conn:
        stamp = conn.execute("SELECT created_at FROM sessions WHERE session_id='NEW'").fetchone()[0]
        assert stamp and stamp.endswith("Z")
        assert conn.execute("SELECT created_at FROM sessions WHERE session_id='OLD'").fetchone()[0] is None
    store.start_session(session_id="NEW",user_id="W",scenario_id="3",started_at="2026-11-03T00:00:00Z")
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT created_at FROM sessions WHERE session_id='NEW'").fetchone()[0] == stamp


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


def test_start_retry_does_not_reopen_an_ended_session() -> None:
    store = _store()
    _ = store.start_session(session_id="S1", user_id="W-1", scenario_id="1-A", started_at="t0")
    _ = store.end_session(session_id="S1", user_id="W-1", scenario_id="1-A", ended_at="t1")

    row = store.start_session(session_id="S1", user_id="W-1", scenario_id="1-A", started_at="t0")

    assert (row.session_status, row.started_at, row.ended_at) == ("COMPLETED", "t0", "t1")


def test_session_target_thickness_is_immutable_for_a_session_id() -> None:
    store = _store()

    first = store.start_session(
        session_id="S1", user_id="W-1", scenario_id="3", started_at="t0",
        target_thickness_um=150.0,
    )
    with pytest.raises(SessionConflict):
        store.start_session(
            session_id="S1", user_id="W-1", scenario_id="3", started_at="t0",
            target_thickness_um=175.5,
        )

    assert first.target_thickness_um == 150.0
    assert store.get_session("S1").target_thickness_um == 150.0  # type: ignore[union-attr]


@pytest.mark.parametrize("field,value", [("user_id", "other"), ("scenario_id", "other"), ("started_at", "t1")])
def test_start_conflict_preserves_the_original_session(field: str, value: str) -> None:
    store = _store()
    args = dict(session_id="S1", user_id="W-1", scenario_id="3", started_at="t0")
    original = store.start_session(**args)
    with pytest.raises(SessionConflict):
        store.start_session(**(args | {field: value}))
    assert store.get_session("S1") == original


def test_end_retries_keep_the_first_timestamp_and_reject_wrong_owner() -> None:
    store = _store()
    store.start_session(session_id="S1", user_id="W-1", scenario_id="3", started_at="t0")
    first = store.end_session(session_id="S1", user_id="W-1", scenario_id="3", ended_at="t1")
    assert store.end_session(session_id="S1", user_id="W-1", scenario_id="3", ended_at="t2") == first
    with pytest.raises(SessionConflict):
        store.end_session(session_id="S1", user_id="other", scenario_id="3", ended_at="t3")
    assert store.get_session("S1") == first


def test_completion_fingerprint_is_atomic_across_store_connections(tmp_path: Path) -> None:
    stores = [V1Store(str(tmp_path / "state.db")) for _ in range(2)]
    def create(index: int):
        return stores[index].create_completion(session_id="S1", requested_at="t", payload={"x": 1}, fingerprint="same")
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(create, range(2)))
    assert rows[0].report_id == rows[1].report_id
    stores[0].finish_completion(rows[0].report_id, status="complete", analysis={"evidence": 1})
    stores[1].finish_completion(rows[0].report_id, status="failed", analysis={"error": "late"})
    assert stores[0].get_completion(rows[0].report_id).report_status == "complete"


def test_concurrent_starts_have_only_one_creation_owner(tmp_path: Path) -> None:
    stores = [V1Store(str(tmp_path / "state.db")) for _ in range(2)]
    def start(index: int):
        return stores[index].start_session_with_created(
            session_id="S1", user_id="W1", scenario_id="3", started_at="t0",
        )
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(start, range(2)))
    assert rows[0][0] == rows[1][0]
    assert sorted(row[1] for row in rows) == [False, True]


def test_legacy_completion_migration_preserves_existing_rows(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE completions (report_id TEXT PRIMARY KEY, session_id TEXT, report_status TEXT, requested_at TEXT, payload TEXT)")
        conn.execute("INSERT INTO completions VALUES ('RPT-old', 'S1', 'complete', 'old', '{\"value\":42}')")
    store = V1Store(str(path))
    assert store.get_completion("RPT-old").payload == {"value": 42}
    assert store.create_completion(session_id="S2", requested_at="t", payload={}, fingerprint="new")


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
