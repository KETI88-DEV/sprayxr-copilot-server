"""시퀀스 저장소 (계획 E7: SQLite, 표준 라이브러리, WAL, 단일 워커).

`POST /api/v1/telemetry/sequence`는 **데이터베이스에 시퀀스를 올리는** 엔드포인트다
(명세서 p.4). 채점은 나중에 `POST /api/v1/copilot/query`의 `pose_recomm`이 `context`로
시퀀스를 가리켜 수행한다(p.9). **그래서 본문을 보관한다** — 해시만 남기면 나중에 채점할
대상이 없다. 같은 `(session_id, sequence_id)`가 같은 본문으로 다시 오면 `duplicate=true`이고
프레임을 두 번 세지 않는다 — 벤더 클라이언트가 같은 시퀀스를 두 경로로 보내기 때문이다
(`docs/sprayxr_v25_sample_delta_2026-09-28.md` B18).

경로는 `SEQUENCE_DB_PATH`다. 기본값은 agent-api가 쓸 수 있는 유일한 마운트 아래이며,
전용 `STATE_DIR` 볼륨이 이 상태 파일의 제 자리다(후속 작업).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

logger = logging.getLogger(__name__)

# 시퀀스 **내용**이 아니라 분석 파라미터인 최상위 필드. `copilot/analyze-sequence`만 붙인다(명세서 p.7, `mode_id`).
# 같은 시퀀스를 두 경로로 보낼 때 해시가 갈리지 않게 빼고(`telemetry_router`), MCP 자세 엔진에 넘길 때도 뺀다
# (`pose_query` — MCP `SequenceRequest`는 최상위 `extra="forbid"`다).
NON_CONTENT_FIELDS: Final[frozenset[str]] = frozenset({"mode_id"})

DB_PATH_ENV: Final[str] = "SEQUENCE_DB_PATH"
DEFAULT_DB_PATH: Final[str] = "/output/eval_results/sequences.db"

_SCHEMA: Final[str] = """
CREATE TABLE IF NOT EXISTS sequences (
    session_key TEXT NOT NULL,
    sequence_id TEXT NOT NULL,
    body_sha256 TEXT NOT NULL,
    frames      INTEGER NOT NULL,
    received_at TEXT NOT NULL,
    body        TEXT,
    PRIMARY KEY (session_key, sequence_id)
)
"""


# `CREATE TABLE IF NOT EXISTS`는 이미 있는 테이블을 그대로 둔다. `/output`은 마운트
# 볼륨이라 이미지를 다시 빌드해도 DB 파일이 남으므로, 나중에 추가한 열은 직접 넣어야
# 한다. 2026-10-01 배포에서 실물 DB(2026-09-30 생성)에 `body`가 없어
# `POST /api/v1/telemetry/sequence`가 500이 됐다. 열 추가는 되돌릴 필요가 없으므로
# 버전 표가 아니라 "없으면 넣는다"로 충분하다 — 멱등이고 기존 행을 건드리지 않는다.
_ADDED_COLUMNS: Final[tuple[tuple[str, str], ...]] = (("body", "TEXT"),)


def _add_missing_columns(conn: sqlite3.Connection) -> None:
    present = {row[1] for row in conn.execute("PRAGMA table_info(sequences)")}
    for name, ddl in _ADDED_COLUMNS:
        if name not in present:
            conn.execute(f"ALTER TABLE sequences ADD COLUMN {name} {ddl}")


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
            _add_missing_columns(self._conn)

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

    def record(
        self,
        *,
        session_key: str,
        sequence_id: str,
        body_sha256: str,
        frames: int,
        received_at: str,
        body: str | None = None,
    ) -> StoreResult:
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
                "INSERT INTO sequences (session_key, sequence_id, body_sha256, frames, received_at, body)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (session_key, sequence_id, body_sha256, int(frames), received_at, body),
            )
            return StoreResult(stored=True, duplicate=False, conflict=False)

    def load_body(self, *, session_key: str, sequence_id: str) -> dict[str, Any] | None:
        """저장된 시퀀스 본문. `pose_recomm` 질의가 채점 대상을 찾는 경로다."""
        with self._guard:
            row = self._conn.execute(
                "SELECT body FROM sequences WHERE session_key = ? AND sequence_id = ?",
                (session_key, sequence_id),
            ).fetchone()
        if row is None or row[0] is None:
            return None
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            logger.warning("stored sequence body is not JSON session=%s sequence=%s", session_key, sequence_id)
            return None

    def latest_sequence_id(self, session_key: str) -> str | None:
        """`context`가 없을 때 쓸 가장 최근 시퀀스. 없으면 `None` — 지어내지 않는다."""
        with self._guard:
            row = self._conn.execute(
                "SELECT sequence_id FROM sequences WHERE session_key = ?"
                " ORDER BY received_at DESC, sequence_id DESC LIMIT 1",
                (session_key,),
            ).fetchone()
        return str(row[0]) if row is not None else None

    def close(self) -> None:
        with self._guard:
            self._conn.close()
