"""v2.5 공개 API의 영속 상태 — 사용자·세션·리포트 (계획 E7).

SQLite, 표준 라이브러리, WAL, 단일 워커. `SequenceStore`와 같은 DB 파일을 쓴다
(`SEQUENCE_DB_PATH`). 시퀀스 수신 기록은 그쪽이, 나머지 상태는 여기가 소유한다.

명세서가 상태 전이를 규정한다.
* `/sessions/start` (p.1): `user_id`가 없으면 신규 생성, 있으면 숙련도·파지손 갱신
* `/sessions/end` (p.3): `COMPLETED`로 전환하고 종료 시각 기록
* `/sessions/complete` (p.11-12): `report_id` 부여, `pending`으로 시작.
  "실제 분석 작업이 연결되기 전에는 `complete`로 표시하지 않습니다."
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import uuid
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from trainer.demo_api.sequence_store import SequenceStore

logger = logging.getLogger(__name__)


class SessionConflict(ValueError):
    """An existing session identifier belongs to different immutable input."""

_SCHEMA: Final[tuple[str, ...]] = (
    """
    CREATE TABLE IF NOT EXISTS users (
        user_id       TEXT PRIMARY KEY,
        name          TEXT,
        skill_level   TEXT NOT NULL,
        dominant_hand TEXT NOT NULL,
        age           INTEGER,
        gender        TEXT,
        created_at    TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sessions (
        session_id     TEXT PRIMARY KEY,
        user_id        TEXT NOT NULL,
        scenario_id    TEXT NOT NULL,
        session_status TEXT NOT NULL,
        started_at     TEXT,
        ended_at       TEXT,
        target_thickness_um REAL,
        created_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS completions (
        report_id     TEXT PRIMARY KEY,
        session_id    TEXT NOT NULL,
        report_status TEXT NOT NULL,
        requested_at  TEXT NOT NULL,
        payload       TEXT NOT NULL
    )
    """,
)


@dataclass(frozen=True)
class UserRow:
    user_id: str
    name: str | None
    skill_level: str
    dominant_hand: str
    age: int | None
    gender: str | None


@dataclass(frozen=True)
class SessionRow:
    session_id: str
    user_id: str
    scenario_id: str
    session_status: str
    started_at: str | None
    ended_at: str | None
    target_thickness_um: float | None


@dataclass(frozen=True)
class CompletionRow:
    report_id: str
    session_id: str
    report_status: str
    requested_at: str
    payload: dict[str, Any]


class V1Store:
    """단일 워커 전제. 쓰기는 프로세스 내 락으로 직렬화한다."""

    def __init__(self, db_path: str | None = None) -> None:
        self._guard = threading.Lock()
        configured_path = db_path if db_path is not None else SequenceStore._default_path()
        # 연결 규칙(경로 불가 시 메모리로 내려감, WAL)을 SequenceStore 와 공유한다.
        self._conn = SequenceStore._connect(  # noqa: SLF001 - 같은 DB, 같은 연결 규칙
            configured_path
        )
        with self._conn:
            for statement in _SCHEMA:
                self._conn.execute(statement)
            session_columns = {
                str(row[1]) for row in self._conn.execute("PRAGMA table_info(sessions)")
            }
            if "target_thickness_um" not in session_columns:
                self._conn.execute(
                    "ALTER TABLE sessions ADD COLUMN target_thickness_um REAL"
                )
            if "created_at" not in session_columns:
                self._conn.execute("ALTER TABLE sessions ADD COLUMN created_at TEXT")
            # Nullable metadata: migration cannot establish the origin of old rows.
            for name in ("creation_origin", "creation_path"):
                if name not in session_columns:
                    self._conn.execute(f"ALTER TABLE sessions ADD COLUMN {name} TEXT")
            columns = {str(row[1]) for row in self._conn.execute("PRAGMA table_info(completions)")}
            if "fingerprint" not in columns:
                self._conn.execute("ALTER TABLE completions ADD COLUMN fingerprint TEXT")
            for name in ("processing_started_at", "processing_updated_at", "processing_finished_at", "processing_reason"):
                if name not in columns:
                    self._conn.execute(f"ALTER TABLE completions ADD COLUMN {name} TEXT")
            self._conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS completions_fingerprint ON completions(fingerprint) WHERE fingerprint IS NOT NULL"
            )
        actual_path = self._conn.execute("PRAGMA database_list").fetchone()[2]
        # Explicit in-memory stores are for isolated callers/tests. A failed production
        # database must not acknowledge a report as durable after silently falling back.
        self.report_storage_available = bool(actual_path) or configured_path == ":memory:"
        self._artifact_tmp = tempfile.TemporaryDirectory(prefix="sprayxr-reports-") if not actual_path else None
        self.artifact_dir = (
            Path(actual_path).resolve().parent / (Path(actual_path).name + ".reports")
            if actual_path else Path(self._artifact_tmp.name)
        )

    # --- 사용자 -----------------------------------------------------------

    def upsert_user(
        self,
        *,
        user_id: str,
        skill_level: str,
        dominant_hand: str,
        name: str | None = None,
        age: int | None = None,
        gender: str | None = None,
        created_at: str,
    ) -> bool:
        """``True``면 신규 생성. 기존이면 숙련도·파지손만 갱신한다(p.1)."""
        with self._guard, self._conn:
            row = self._conn.execute(
                "SELECT user_id FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO users (user_id, name, skill_level, dominant_hand, age, gender, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (user_id, name, skill_level, dominant_hand, age, gender, created_at),
                )
                return True
            self._conn.execute(
                "UPDATE users SET skill_level = ?, dominant_hand = ? WHERE user_id = ?",
                (skill_level, dominant_hand, user_id),
            )
            return False

    def get_user(self, user_id: str) -> UserRow | None:
        with self._guard:
            row = self._conn.execute(
                "SELECT user_id, name, skill_level, dominant_hand, age, gender FROM users WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return UserRow(*row) if row is not None else None

    def list_users(self, *, skip: int, limit: int) -> list[UserRow]:
        with self._guard:
            rows = self._conn.execute(
                "SELECT user_id, name, skill_level, dominant_hand, age, gender FROM users"
                " ORDER BY created_at, user_id LIMIT ? OFFSET ?",
                (int(limit), int(skip)),
            ).fetchall()
        return [UserRow(*row) for row in rows]

    # --- 세션 -------------------------------------------------------------

    def start_session(
        self, *, session_id: str, user_id: str, scenario_id: str, started_at: str,
        target_thickness_um: float | None = None,
        creation_origin: str | None = None, creation_path: str | None = None,
    ) -> SessionRow:
        return self.start_session_with_created(
            session_id=session_id, user_id=user_id, scenario_id=scenario_id,
            started_at=started_at, target_thickness_um=target_thickness_um,
            creation_origin=creation_origin, creation_path=creation_path,
        )[0]

    def start_session_with_created(
        self, *, session_id: str, user_id: str, scenario_id: str, started_at: str,
        target_thickness_um: float | None = None,
        creation_origin: str | None = None, creation_path: str | None = None,
    ) -> tuple[SessionRow, bool]:
        """Return creation ownership atomically so retries cannot reset event history."""
        if creation_origin not in {None, "real", "test"}:
            raise ValueError("unknown session creation origin")
        with self._guard, self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            existing = self._conn.execute(
                "SELECT session_id, user_id, scenario_id, session_status, started_at, ended_at, target_thickness_um FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if existing is not None:
                row = SessionRow(*existing)
                if (row.user_id, row.scenario_id, row.started_at, row.target_thickness_um) != (user_id, scenario_id, started_at, target_thickness_um):
                    raise SessionConflict("session_id already exists with different start parameters")
                return row, False
            self._conn.execute(
                "INSERT INTO sessions (session_id, user_id, scenario_id, session_status, started_at, ended_at, target_thickness_um, created_at, creation_origin, creation_path)"
                " VALUES (?, ?, ?, 'ACTIVE', ?, NULL, ?, strftime('%Y-%m-%dT%H:%M:%fZ','now'), ?, ?)",
                (session_id, user_id, scenario_id, started_at, target_thickness_um, creation_origin, creation_path),
            )
        return SessionRow(
            session_id, user_id, scenario_id, "ACTIVE", started_at, None,
            target_thickness_um,
        ), True

    def end_session(
        self, *, session_id: str, user_id: str, scenario_id: str, ended_at: str
    ) -> SessionRow:
        """세션이 없어도 기록한다 — 종료 통지를 잃지 않는다. 시작 시각은 모르는 채 남는다."""
        with self._guard, self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            existing = self._conn.execute(
                "SELECT session_id, user_id, scenario_id, session_status, started_at, ended_at, target_thickness_um FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if existing is not None:
                row = SessionRow(*existing)
                if (row.user_id, row.scenario_id) != (user_id, scenario_id):
                    raise SessionConflict("session user_id or scenario_id does not match")
                if row.session_status == "COMPLETED":
                    return row
            self._conn.execute(
                "INSERT INTO sessions (session_id, user_id, scenario_id, session_status, started_at, ended_at, target_thickness_um, created_at)"
                " VALUES (?, ?, ?, 'COMPLETED', NULL, ?, NULL, strftime('%Y-%m-%dT%H:%M:%fZ','now'))"
                " ON CONFLICT(session_id) DO UPDATE SET"
                "   session_status = 'COMPLETED', ended_at = excluded.ended_at",
                (session_id, user_id, scenario_id, ended_at),
            )
            row = self._conn.execute(
                "SELECT session_id, user_id, scenario_id, session_status, started_at, ended_at, target_thickness_um"
                " FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return SessionRow(*row)

    def get_session(self, session_id: str) -> SessionRow | None:
        with self._guard:
            row = self._conn.execute(
                "SELECT session_id, user_id, scenario_id, session_status, started_at, ended_at, target_thickness_um"
                " FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return SessionRow(*row) if row is not None else None

    # --- 리포트 -----------------------------------------------------------

    def create_completion(self, *, session_id: str, requested_at: str, payload: dict[str, Any], fingerprint: str | None = None) -> CompletionRow:
        """`report_id`를 부여하고 `pending`으로 저장한다(p.12)."""
        import json

        report_id = f"RPT-{uuid.uuid4().hex[:12]}"
        with self._guard, self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            if fingerprint is not None:
                old = self._conn.execute(
                    "SELECT report_id, session_id, report_status, requested_at, payload FROM completions WHERE fingerprint = ?",
                    (fingerprint,),
                ).fetchone()
                if old is not None:
                    return CompletionRow(*old[:4], json.loads(old[4]))
            self._conn.execute(
                "INSERT INTO completions (report_id, session_id, report_status, requested_at, payload, fingerprint, processing_updated_at, processing_reason)"
                " VALUES (?, ?, 'pending', ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ','now'), 'queued')",
                (report_id, session_id, requested_at, json.dumps(payload, ensure_ascii=False), fingerprint),
            )
        return CompletionRow(report_id, session_id, "pending", requested_at, payload)

    def mark_completion_processing(self, report_id: str) -> None:
        """Record an observed worker start; never called by migration or DB reads."""
        with self._guard, self._conn:
            self._conn.execute(
                "UPDATE completions SET processing_started_at=COALESCE(processing_started_at,strftime('%Y-%m-%dT%H:%M:%fZ','now')),"
                " processing_updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now'),processing_reason='processing'"
                " WHERE report_id=? AND report_status='pending'", (report_id,),
            )

    def finish_completion(self, report_id: str, *, status: str, analysis: dict[str, Any]) -> None:
        """분석이 끝나면 상태(`complete`/`failed`)와 결과를 남긴다. 결과는 payload의 `analysis` 아래에 둔다."""
        import json

        with self._guard, self._conn:
            row = self._conn.execute(
                "SELECT payload FROM completions WHERE report_id = ?", (report_id,)
            ).fetchone()
            if row is None:
                return
            payload = json.loads(row[0]) | {"analysis": analysis}
            self._conn.execute(
                "UPDATE completions SET report_status = ?, payload = ?,"
                " processing_updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now'),"
                " processing_finished_at=strftime('%Y-%m-%dT%H:%M:%fZ','now'),processing_reason=?"
                " WHERE report_id = ? AND report_status = 'pending'",
                (status, json.dumps(payload, ensure_ascii=False), str(analysis.get('error') or status), report_id),
            )

    def pending_completions(self) -> list[CompletionRow]:
        import json

        with self._guard:
            rows = self._conn.execute(
                "SELECT report_id, session_id, report_status, requested_at, payload FROM completions WHERE report_status = 'pending' ORDER BY rowid"
            ).fetchall()
        return [CompletionRow(*row[:4], json.loads(row[4])) for row in rows]

    def get_completion(self, report_id: str) -> CompletionRow | None:
        import json

        with self._guard:
            row = self._conn.execute(
                "SELECT report_id, session_id, report_status, requested_at, payload"
                " FROM completions WHERE report_id = ?",
                (report_id,),
            ).fetchone()
        if row is None:
            return None
        return CompletionRow(row[0], row[1], row[2], row[3], json.loads(row[4]))

    def complete_sessions(self, *, limit: int = 50) -> list[dict[str, Any]]:
        """`complete` 리포트가 있는 세션, 최근 것부터(세션마다 최신 리포트 하나)."""
        import json

        with self._guard:
            rows = self._conn.execute(
                "SELECT report_id, session_id, requested_at, payload FROM completions WHERE report_status = 'complete'"
                " ORDER BY requested_at DESC, rowid DESC"
            ).fetchall()
        sessions: dict[str, dict[str, Any]] = {}
        for report_id, session_id, requested_at, payload in rows:
            if session_id in sessions:
                continue
            data = json.loads(payload)
            sessions[session_id] = {
                "session_id": session_id, "report_id": report_id, "requested_at": requested_at,
                "user_id": data.get("user_id"), "scenario_id": data.get("scenario_id"),
            }
            if len(sessions) >= limit:
                break
        return list(sessions.values())

    def latest_complete_analysis(self, session_id: str) -> tuple[str, dict[str, Any]] | None:
        """세션의 가장 최근 `complete` 리포트의 (report_id, analysis). 없으면 None."""
        import json

        with self._guard:
            row = self._conn.execute(
                "SELECT report_id, payload FROM completions WHERE session_id = ? AND report_status = 'complete'"
                " ORDER BY requested_at DESC, rowid DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        analysis = json.loads(row[1]).get("analysis")
        return (row[0], analysis) if isinstance(analysis, dict) else None

    def close(self) -> None:
        with self._guard:
            self._conn.close()
