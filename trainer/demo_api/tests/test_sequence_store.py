"""시퀀스 수신 기록: 중복과 본문 충돌을 구분한다."""

from __future__ import annotations

import json
from pathlib import Path

from trainer.demo_api.sequence_store import SequenceStore


def _rec(store: SequenceStore, *, key: str = "S1", seq: str = "Q1", sha: str = "h1", frames: int = 11):
    return store.record(session_key=key, sequence_id=seq, body_sha256=sha, frames=frames, received_at="t")


def test_first_write_is_stored() -> None:
    """Given an unseen key, When recorded, Then it is stored."""
    result = _rec(SequenceStore(":memory:"))

    assert (result.stored, result.duplicate, result.conflict) == (True, False, False)


def test_same_key_same_body_is_a_duplicate() -> None:
    """Given a repeat of the same body, When recorded, Then it is a duplicate, not a new ingest."""
    store = SequenceStore(":memory:")
    _ = _rec(store)

    result = _rec(store)

    assert (result.stored, result.duplicate, result.conflict) == (False, True, False)


def test_same_key_different_body_is_a_conflict() -> None:
    """Given a different body under one key, When recorded, Then it is a conflict.

    조용히 덮어쓰면 어느 쪽이 채점된 것인지 알 수 없게 된다.
    """
    store = SequenceStore(":memory:")
    _ = _rec(store)

    result = _rec(store, sha="h2")

    assert (result.stored, result.duplicate, result.conflict) == (False, False, True)


def test_different_sessions_do_not_collide() -> None:
    """Given the same sequence_id in two sessions, When recorded, Then both are stored."""
    store = SequenceStore(":memory:")

    assert _rec(store, key="S1").stored is True
    assert _rec(store, key="S2").stored is True


def test_missing_session_key_is_its_own_bucket() -> None:
    """Given an unattributed sequence, When recorded, Then it does not shadow a real session."""
    store = SequenceStore(":memory:")

    assert _rec(store, key="").stored is True
    assert _rec(store, key="S1").stored is True


def test_records_survive_reopen(tmp_path: Path) -> None:
    """Given a file-backed store, When reopened, Then the earlier record is still known."""
    path = str(tmp_path / "sequences.db")
    first = SequenceStore(path)
    _ = _rec(first)
    first.close()

    result = _rec(SequenceStore(path))

    assert result.duplicate is True


def test_unusable_path_falls_back_to_memory(tmp_path: Path) -> None:
    """Given an unwritable path, When constructed, Then ingest still works.

    저장 실패가 수신을 막지 않는다.
    """
    blocker = tmp_path / "file"
    blocker.write_text("not a directory", encoding="utf-8")

    store = SequenceStore(str(blocker / "nested" / "sequences.db"))

    assert _rec(store).stored is True


def _legacy_db(path: Path) -> None:
    """`body` 열이 추가되기 전 스키마로 DB를 만든다.

    실물 `/output/eval_results/sequences.db`(2026-09-30)가 이 모양이었고, `/output`은
    마운트 볼륨이라 이미지를 다시 빌드해도 남는다. 즉 배포는 반드시 이 파일을 만난다.
    """
    import sqlite3

    conn = sqlite3.connect(str(path))
    with conn:
        conn.execute(
            """
            CREATE TABLE sequences (
                session_key TEXT NOT NULL,
                sequence_id TEXT NOT NULL,
                body_sha256 TEXT NOT NULL,
                frames      INTEGER NOT NULL,
                received_at TEXT NOT NULL,
                PRIMARY KEY (session_key, sequence_id)
            )
            """
        )
    conn.close()


def test_legacy_db_without_body_column_is_upgraded(tmp_path: Path) -> None:
    """Given a pre-`body` DB, When opened, Then recording a body still works.

    이행이 없으면 `sqlite3.OperationalError: table sequences has no column named body`로
    POST /api/v1/telemetry/sequence 가 500이 된다.
    """
    path = tmp_path / "sequences.db"
    _legacy_db(path)
    store = SequenceStore(str(path))

    result = store.record(
        session_key="S1",
        sequence_id="Q1",
        body_sha256="h1",
        frames=11,
        received_at="t",
        body='{"a": 1}',
    )

    assert result.stored is True
    assert store.load_body(session_key="S1", sequence_id="Q1") == {"a": 1}


def test_legacy_db_keeps_rows_written_before_the_upgrade(tmp_path: Path) -> None:
    """Given legacy rows, When upgraded, Then they survive with an empty body."""
    import sqlite3

    path = tmp_path / "sequences.db"
    _legacy_db(path)
    conn = sqlite3.connect(str(path))
    with conn:
        conn.execute(
            "INSERT INTO sequences VALUES ('S0', 'Q0', 'old-sha', 7, 'then')",
        )
    conn.close()

    store = SequenceStore(str(path))

    assert store.load_body(session_key="S0", sequence_id="Q0") is None
    assert store.latest_sequence_id("S0") == "Q0"


def test_session_bodies_keep_receive_order_even_when_ids_do_not_sort() -> None:
    store = SequenceStore(":memory:")
    for sequence_id in ("SEQ-10", "SEQ-2"):
        store.record(
            session_key="S1", sequence_id=sequence_id, body_sha256=sequence_id,
            frames=1, received_at="2026-10-13T00:00:00Z",
            body=json.dumps({"sequence_id": sequence_id}),
        )

    assert [body["sequence_id"] for body in store.session_bodies("S1")] == ["SEQ-10", "SEQ-2"]
    assert [body["sequence_id"] for body in store.session_bodies_through("S1", "SEQ-10")] == ["SEQ-10"]


def _catalog_record(
    store: SequenceStore,
    *,
    session_key: str,
    sequence_id: str,
    received_at: str,
    frames: int = 1,
) -> None:
    store.record(
        session_key=session_key,
        sequence_id=sequence_id,
        body_sha256=f"{session_key}:{sequence_id}",
        frames=frames,
        received_at=received_at,
        body=json.dumps({"sequence_id": sequence_id}),
    )


def test_recent_sequence_catalog_orders_sessions_and_sequences_by_receive_time() -> None:
    store = SequenceStore(":memory:")
    _catalog_record(
        store, session_key="older", sequence_id="SEQ-10",
        received_at="2026-10-14T01:00:00Z", frames=10,
    )
    _catalog_record(
        store, session_key="older", sequence_id="SEQ-2",
        received_at="2026-10-14T01:01:00Z", frames=20,
    )
    _catalog_record(
        store, session_key="newer", sequence_id="SEQ-1",
        received_at="2026-10-14T02:00:00Z", frames=30,
    )

    result = store.recent_sequence_catalog()

    assert [item["session_id"] for item in result] == ["newer", "older"]
    assert result[0]["sequences"] == [
        {"sequence_id": "SEQ-1", "received_at": "2026-10-14T02:00:00Z", "frames": 30}
    ]
    assert [item["sequence_id"] for item in result[1]["sequences"]] == [
        "SEQ-10", "SEQ-2",
    ]
    assert all("body" not in sequence for item in result for sequence in item["sequences"])


def test_recent_sequence_catalog_respects_session_and_sequence_limits() -> None:
    store = SequenceStore(":memory:")
    for session_index in range(3):
        for sequence_index in range(4):
            _catalog_record(
                store,
                session_key=f"S{session_index}",
                sequence_id=f"SEQ-{sequence_index}",
                received_at=f"2026-10-14T0{session_index}:0{sequence_index}:00Z",
            )

    result = store.recent_sequence_catalog(session_limit=2, sequence_limit=3)

    assert len(result) == 2
    assert sum(len(item["sequences"]) for item in result) == 3
    assert [item["session_id"] for item in result] == ["S2", "S1"]
