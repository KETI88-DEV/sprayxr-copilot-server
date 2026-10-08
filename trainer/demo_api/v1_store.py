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
from dataclasses import dataclass
from typing import Any, Final

from trainer.demo_api.sequence_store import SequenceStore

logger = logging.getLogger(__name__)

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
        ended_at       TEXT
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
        # 연결 규칙(경로 불가 시 메모리로 내려감, WAL)을 SequenceStore 와 공유한다.
        self._conn = SequenceStore._connect(  # noqa: SLF001 - 같은 DB, 같은 연결 규칙
            db_path if db_path is not None else SequenceStore._default_path()
        )
        with self._conn:
            for statement in _SCHEMA:
                self._conn.execute(statement)

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
        self, *, session_id: str, user_id: str, scenario_id: str, started_at: str
    ) -> SessionRow:
        with self._guard, self._conn:
            self._conn.execute(
                "INSERT INTO sessions (session_id, user_id, scenario_id, session_status, started_at, ended_at)"
                " VALUES (?, ?, ?, 'ACTIVE', ?, NULL)"
                " ON CONFLICT(session_id) DO UPDATE SET"
                "   user_id = excluded.user_id, scenario_id = excluded.scenario_id,"
                "   session_status = 'ACTIVE', started_at = excluded.started_at, ended_at = NULL",
                (session_id, user_id, scenario_id, started_at),
            )
        return SessionRow(session_id, user_id, scenario_id, "ACTIVE", started_at, None)

    def end_session(
        self, *, session_id: str, user_id: str, scenario_id: str, ended_at: str
    ) -> SessionRow:
        """세션이 없어도 기록한다 — 종료 통지를 잃지 않는다. 시작 시각은 모르는 채 남는다."""
        with self._guard, self._conn:
            self._conn.execute(
                "INSERT INTO sessions (session_id, user_id, scenario_id, session_status, started_at, ended_at)"
                " VALUES (?, ?, ?, 'COMPLETED', NULL, ?)"
                " ON CONFLICT(session_id) DO UPDATE SET"
                "   session_status = 'COMPLETED', ended_at = excluded.ended_at",
                (session_id, user_id, scenario_id, ended_at),
            )
            row = self._conn.execute(
                "SELECT session_id, user_id, scenario_id, session_status, started_at, ended_at"
                " FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return SessionRow(*row)

    def get_session(self, session_id: str) -> SessionRow | None:
        with self._guard:
            row = self._conn.execute(
                "SELECT session_id, user_id, scenario_id, session_status, started_at, ended_at"
                " FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return SessionRow(*row) if row is not None else None

    # --- 리포트 -----------------------------------------------------------

    def create_completion(self, *, session_id: str, requested_at: str, payload: dict[str, Any]) -> CompletionRow:
        """`report_id`를 부여하고 `pending`으로 저장한다(p.12)."""
        import json

        report_id = f"RPT-{uuid.uuid4().hex[:12]}"
        with self._guard, self._conn:
            self._conn.execute(
                "INSERT INTO completions (report_id, session_id, report_status, requested_at, payload)"
                " VALUES (?, ?, 'pending', ?, ?)",
                (report_id, session_id, requested_at, json.dumps(payload, ensure_ascii=False)),
            )
        return CompletionRow(report_id, session_id, "pending", requested_at, payload)

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
                "UPDATE completions SET report_status = ?, payload = ? WHERE report_id = ?",
                (status, json.dumps(payload, ensure_ascii=False), report_id),
            )

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

    def close(self) -> None:
        with self._guard:
            self._conn.close()
