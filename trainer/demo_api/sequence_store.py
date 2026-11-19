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
import re
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from collections.abc import Mapping
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

_QUERY_TURNS_SCHEMA: Final[str] = """
CREATE TABLE IF NOT EXISTS query_turns (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_key TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    prompt      TEXT NOT NULL,
    answer      TEXT NOT NULL,
    task_type   TEXT NOT NULL,
    created_at  TEXT NOT NULL
)
"""

_QUERY_TURNS_INDEX: Final[str] = """
CREATE INDEX IF NOT EXISTS idx_query_turns_session_id
ON query_turns (session_key, id)
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
    out_of_order: bool = False


def sequence_order_key(sequence_id: str) -> tuple[tuple[int, Any], ...]:
    """Natural identifier order, including unpadded SEQ-2 before SEQ-10."""
    return tuple((1, int(part)) if part.isdigit() else (0, part)
                 for part in re.split(r"(\d+)", sequence_id))


def _json_object(raw: object) -> dict[str, Any] | None:
    try:
        value = json.loads(raw) if isinstance(raw, str) else None
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


class SequenceStore:
    """단일 워커 전제. 쓰기는 프로세스 내 락으로 직렬화한다."""

    def __init__(self, db_path: str | None = None) -> None:
        raw = db_path if db_path is not None else self._default_path()
        self._guard = threading.Lock()
        self._conn = self._connect(raw)
        with self._conn:
            self._conn.execute(_SCHEMA)
            self._conn.execute(_QUERY_TURNS_SCHEMA)
            columns = {row[1] for row in self._conn.execute("PRAGMA table_info(query_turns)")}
            if "conversation_key" not in columns:
                self._conn.execute("ALTER TABLE query_turns ADD COLUMN conversation_key TEXT")
            if "safety_class" not in columns:
                # 위험 주제 질문의 종류(safety_guard). 후속 질문이 다시 판정하지 않고 이어받는다(v3.4.3).
                self._conn.execute("ALTER TABLE query_turns ADD COLUMN safety_class TEXT")
            for column in ("answer_source", "subject", "query_context"):  # sources, subject and bounded feedback/training context
                if column not in columns:
                    self._conn.execute(f"ALTER TABLE query_turns ADD COLUMN {column} TEXT")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_query_dialogue ON query_turns (session_key, user_id, conversation_key, id)")
            self._conn.execute(_QUERY_TURNS_INDEX)
            self._conn.execute("""CREATE TABLE IF NOT EXISTS sequence_analysis_results (
                session_key TEXT NOT NULL, sequence_id TEXT NOT NULL,
                analysis_key TEXT NOT NULL, result TEXT NOT NULL, created_at TEXT,
                PRIMARY KEY (session_key, sequence_id, analysis_key)
            )""")
            analysis_columns = {row[1] for row in self._conn.execute('PRAGMA table_info(sequence_analysis_results)')}
            if 'created_at' not in analysis_columns:
                # Legacy creation times were never recorded; keep them NULL.
                self._conn.execute('ALTER TABLE sequence_analysis_results ADD COLUMN created_at TEXT')
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
                conn = sqlite3.connect(raw, check_same_thread=False, timeout=30)
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
        enforce_order: bool = False,
    ) -> StoreResult:
        with self._guard, self._conn:
            # Lock before reading: separate store connections must not both observe
            # a missing key and then race while upgrading a deferred transaction.
            self._conn.execute("BEGIN IMMEDIATE")
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
            if enforce_order:
                ids = self._conn.execute(
                    "SELECT sequence_id FROM sequences WHERE session_key = ?", (session_key,),
                ).fetchall()
                if any(sequence_order_key(str(item[0])) >= sequence_order_key(sequence_id) for item in ids):
                    return StoreResult(False, False, False, out_of_order=True)
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

    def sequence_ids(self, session_key: str) -> list[str]:
        """세션의 저장된 시퀀스 id 전부(이름순). 세션 종합 분석이 쓴다."""
        with self._guard:
            rows = self._conn.execute(
                "SELECT sequence_id FROM sequences WHERE session_key = ? ORDER BY sequence_id",
                (session_key,),
            ).fetchall()
        return sorted((str(row[0]) for row in rows), key=sequence_order_key)

    def session_bodies(self, session_key: str) -> list[dict[str, Any]]:
        """Return valid bodies in natural sequence order, independent of arrival."""
        with self._guard:
            rows = self._conn.execute(
                "SELECT sequence_id, body FROM sequences WHERE session_key = ? AND body IS NOT NULL",
                (session_key,),
            ).fetchall()
        rows.sort(key=lambda row: sequence_order_key(str(row[0])))
        return self._decode_bodies([(row[1],) for row in rows], session_key=session_key)

    def session_bodies_through(self, session_key: str, sequence_id: str) -> list[dict[str, Any]]:
        """Return the natural-order prefix ending at an existing sequence."""
        bodies = self.session_bodies(session_key)
        if not any(body.get("sequence_id") == sequence_id for body in bodies):
            return []
        target = sequence_order_key(sequence_id)
        return [body for body in bodies if sequence_order_key(str(body.get("sequence_id", ""))) <= target]

    @staticmethod
    def _decode_bodies(rows: list[tuple[Any, ...]], *, session_key: str) -> list[dict[str, Any]]:
        bodies: list[dict[str, Any]] = []
        for row in rows:
            try:
                body = json.loads(row[0])
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(body, dict):
                bodies.append(body)
        return bodies

    def latest_sequence_id(self, session_key: str) -> str | None:
        """`context`가 없을 때 쓸 가장 최근 시퀀스. 없으면 `None` — 지어내지 않는다."""
        ids = self.sequence_ids(session_key)
        return ids[-1] if ids else None

    def load_analysis(self, *, session_key: str, sequence_id: str, analysis_key: str) -> dict[str, Any] | None:
        """Persisted feedback makes retries harmless after restart or cache eviction."""
        with self._guard:
            row = self._conn.execute(
                "SELECT result FROM sequence_analysis_results WHERE session_key=? AND sequence_id=? AND analysis_key=?",
                (session_key, sequence_id, analysis_key),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def save_analysis(self, *, session_key: str, sequence_id: str, analysis_key: str, result: dict[str, Any]) -> None:
        with self._guard, self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO sequence_analysis_results (session_key,sequence_id,analysis_key,result,created_at) VALUES (?, ?, ?, ?, ?)",
                (session_key, sequence_id, analysis_key, json.dumps(result, ensure_ascii=False), datetime.now(timezone.utc).isoformat()),
            )

    def record_query_turn(
        self,
        *,
        session_key: str,
        user_id: str,
        prompt: str,
        answer: str,
        task_type: str,
        created_at: str,
        conversation_key: str | None = None,
        safety_class: str | None = None,
        answer_source: str | None = None,
        subject: Mapping[str, Any] | None = None,
        query_context: Mapping[str, Any] | None = None,
    ) -> int | None:
        """Persist an exchange and return its receipt for answer feedback."""
        if not session_key:
            return
        serialized_context = None
        if query_context is not None:
            serialized_context = json.dumps(dict(query_context), ensure_ascii=False, separators=(",", ":"))
            original_bytes = len(serialized_context.encode("utf-8"))
            if original_bytes > 131_072:
                # Preserve a complete JSON marker, never a broken/partial fact
                # snapshot that could silently become a training example.
                serialized_context = json.dumps({"context_truncated": True, "original_bytes": original_bytes})
        with self._guard, self._conn:
            cursor = self._conn.execute(
                "INSERT INTO query_turns (session_key, user_id, prompt, answer, task_type, created_at, conversation_key, safety_class, answer_source, subject, query_context)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (session_key, user_id, prompt, answer, task_type, created_at, conversation_key, safety_class, answer_source,
                 json.dumps(subject, ensure_ascii=False) if subject else None, serialized_context),
            )
            return int(cursor.lastrowid)

    def recent_query_turns(self, session_key: str, *, limit: int = 4, user_id: str | None = None, conversation_key: str | None = None) -> list[dict[str, Any]]:
        """Return a bounded chronological window without leaking another session."""
        bounded = max(1, min(int(limit), 12))
        where, params = "session_key = ?", [session_key]
        if user_id is not None:
            where += " AND user_id = ? AND conversation_key IS ?"
            params.extend([user_id, conversation_key])
        with self._guard:
            rows = self._conn.execute(
                "SELECT user_id, prompt, answer, task_type, created_at, safety_class, answer_source, subject FROM query_turns WHERE " + where + " ORDER BY id DESC LIMIT ?",
                [*params, bounded],
            ).fetchall()
        return [
            {
                "user_id": str(user_id),
                "prompt": str(prompt),
                "answer": str(answer),
                "task_type": str(task_type),
                "created_at": str(created_at),
                # 위험 종류가 있는 턴에만 싣는다(모델에 넘기는 최근 대화 모양을 바꾸지 않으려고).
                **({"safety_class": str(safety_class)} if safety_class else {}),
                **({"answer_source": str(answer_source)} if answer_source else {}),
                **({"subject": parsed} if (parsed := _json_object(subject)) else {}),
            }
            for user_id, prompt, answer, task_type, created_at, safety_class, answer_source, subject in reversed(rows)
        ]

    def recent_sequence_catalog(
        self, *, session_limit: int = 50, sequence_limit: int = 200
    ) -> list[dict[str, Any]]:
        """최근 수신 세션과 시퀀스의 제한된 뷰어용 목록.

        원문 ``body``는 목록 응답에 싣지 않는다. 세션은 마지막 수신 순서의
        역순이고, 각 세션 안의 시퀀스는 실제 수신 순서를 유지한다.
        """
        bounded_sessions = max(1, min(int(session_limit), 200))
        bounded_sequences = max(1, min(int(sequence_limit), 1000))
        with self._guard:
            session_rows = self._conn.execute(
                "SELECT session_key, MAX(rowid), MAX(received_at)"
                " FROM sequences GROUP BY session_key"
                " ORDER BY MAX(rowid) DESC LIMIT ?",
                (bounded_sessions,),
            ).fetchall()
            if not session_rows:
                return []
            session_ids = [str(row[0]) for row in session_rows]
            placeholders = ",".join("?" for _ in session_ids)
            priority = " ".join(
                f"WHEN ? THEN {index}" for index, _ in enumerate(session_ids)
            )
            sequence_rows = self._conn.execute(
                "SELECT session_key, sequence_id, received_at, frames"
                f" FROM sequences WHERE session_key IN ({placeholders})"
                f" ORDER BY CASE session_key {priority} END, rowid ASC LIMIT ?",
                (*session_ids, *session_ids, bounded_sequences),
            ).fetchall()

        catalog = [
            {
                "session_id": str(session_id),
                "received_at": str(received_at),
                "sequences": [],
            }
            for session_id, _rowid, received_at in session_rows
        ]
        by_session = {item["session_id"]: item for item in catalog}
        for session_id, sequence_id, received_at, frames in sequence_rows:
            by_session[str(session_id)]["sequences"].append(
                {
                    "sequence_id": str(sequence_id),
                    "received_at": str(received_at),
                    "frames": int(frames),
                }
            )
        return catalog

    def close(self) -> None:
        with self._guard:
            self._conn.close()
