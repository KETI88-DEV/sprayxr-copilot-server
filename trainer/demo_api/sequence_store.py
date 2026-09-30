"""시퀀스 수신 기록 (계획 E7: SQLite, 표준 라이브러리, WAL, 단일 워커).

`POST /api/v1/telemetry/sequence`의 `sequence_data_ack`가 **관측된 사실**을 말하도록 하는
최소 저장소다. 같은 `(session_id, sequence_id)`가 같은 본문으로 다시 오면 `duplicate=true`이고
프레임을 두 번 세지 않는다 — 벤더 클라이언트가 같은 시퀀스를 두 경로로 보내기 때문이다
(`docs/sprayxr_v25_sample_delta_2026-09-28.md` B18).

경로는 `SEQUENCE_DB_PATH`다. 기본값은 agent-api가 쓸 수 있는 유일한 마운트 아래이며,
전용 `STATE_DIR` 볼륨이 이 상태 파일의 제 자리다(후속 작업).
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)

DB_PATH_ENV: Final[str] = "SEQUENCE_DB_PATH"
DEFAULT_DB_PATH: Final[str] = "/output/eval_results/sequences.db"

_SCHEMA: Final[str] = """
CREATE TABLE IF NOT EXISTS sequences (
    session_key TEXT NOT NULL,
    sequence_id TEXT NOT NULL,
    body_sha256 TEXT NOT NULL,
    frames      INTEGER NOT NULL,
    received_at TEXT NOT NULL,
    PRIMARY KEY (session_key, sequence_id)
)
"""


@dataclass(frozen=True)
class StoreResult:
    """``duplicate``은 같은 키가 **같은 본문으로** 다시 온 경우만이다.

    같은 키에 다른 본문이 오면 ``conflict``다 — 조용히 덮어쓰면 어느 쪽이 채점된 것인지
    알 수 없게 된다. 그때 기존 기록을 유지하고 저장하지 않는다.
    """

    stored: bool
    duplicate: bool
    conflict: bool


class SequenceStore:
    """단일 워커 전제. 쓰기는 프로세스 내 락으로 직렬화한다."""

    def __init__(self, db_path: str | None = None) -> None:
        raw = db_path if db_path is not None else self._default_path()
        self._guard = threading.Lock()
        self._conn = self._connect(raw)
        with self._conn:
            self._conn.execute(_SCHEMA)

    @staticmethod
    def _default_path() -> str:
        """`SEQUENCE_DB_PATH` 또는 기본값. `V1Store`가 같은 DB를 쓰도록 공개한다."""
        return os.environ.get(DB_PATH_ENV, DEFAULT_DB_PATH)

    @staticmethod
    def _connect(raw: str) -> sqlite3.Connection:
        """경로를 쓸 수 없으면 메모리로 내려간다 — 저장 실패가 수신을 막지 않는다."""
        if raw != ":memory:":
            try:
                Path(raw).parent.mkdir(parents=True, exist_ok=True)
                conn = sqlite3.connect(raw, check_same_thread=False)
                conn.execute("PRAGMA journal_mode=WAL")
                return conn
            except (OSError, sqlite3.Error):
                logger.warning("sequence store: %s unusable, falling back to memory", raw, exc_info=True)
        return sqlite3.connect(":memory:", check_same_thread=False)

    def record(self, *, session_key: str, sequence_id: str, body_sha256: str, frames: int, received_at: str) -> StoreResult:
        with self._guard, self._conn:
            row = self._conn.execute(
                "SELECT body_sha256 FROM sequences WHERE session_key = ? AND sequence_id = ?",
                (session_key, sequence_id),
            ).fetchone()
            if row is not None:
                same = str(row[0]) == body_sha256
                if not same:
                    logger.warning(
                        "sequence store: same key with a different body session=%s sequence=%s",
                        session_key,
                        sequence_id,
                    )
                return StoreResult(stored=False, duplicate=same, conflict=not same)
            self._conn.execute(
                "INSERT INTO sequences (session_key, sequence_id, body_sha256, frames, received_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (session_key, sequence_id, body_sha256, int(frames), received_at),
            )
            return StoreResult(stored=True, duplicate=False, conflict=False)

    def close(self) -> None:
        with self._guard:
            self._conn.close()
