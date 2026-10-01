"""시퀀스 수신 기록: 중복과 본문 충돌을 구분한다."""

from __future__ import annotations

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
