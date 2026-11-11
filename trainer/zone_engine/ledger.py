"""Durable SQLite event and assessment ledger."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .errors import EngineError


SCHEMA_VERSION = 1


class SQLiteLedger:
    """Small synchronous store with one atomic event/state/result transaction."""

    def __init__(self, path: Path) -> None:
        raw_path = str(path)
        if raw_path == ":memory:" or raw_path.startswith("file::memory:"):
            raise EngineError(
                "persistent_ledger_required",
                "corrected_zone_v2 refuses an in-memory engine ledger",
                status_code=503,
            )
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    def _initialize(self) -> None:
        try:
            with self._connect() as connection:
                connection.execute("PRAGMA journal_mode = WAL")
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS metadata (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        source_instance_id TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        scenario_id TEXT NOT NULL,
                        started_at TEXT NOT NULL,
                        evaluation_mode TEXT NOT NULL,
                        contract_version TEXT NOT NULL,
                        target_thickness_um REAL,
                        target_source TEXT,
                        provider_model_key TEXT NOT NULL,
                        work_id TEXT NOT NULL,
                        session_family_id TEXT NOT NULL,
                        round_index INTEGER NOT NULL,
                        round_id TEXT NOT NULL,
                        geometry_model_key TEXT NOT NULL,
                        manifest_id TEXT NOT NULL,
                        manifest_sha256 TEXT NOT NULL,
                        map_nums_json TEXT NOT NULL,
                        expected_zones_json TEXT NOT NULL,
                        ended_at TEXT,
                        completed_at TEXT,
                        provider_summary_json TEXT,
                        last_event_order INTEGER NOT NULL DEFAULT 0,
                        last_event_id TEXT,
                        last_event_type TEXT,
                        latest_assessment_id TEXT,
                        latest_assessment_revision INTEGER,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        UNIQUE(source_instance_id, session_family_id, round_index)
                    );

                    CREATE TABLE IF NOT EXISTS face_maps (
                        session_id TEXT NOT NULL REFERENCES sessions(session_id),
                        face_id TEXT NOT NULL,
                        map_num INTEGER NOT NULL,
                        PRIMARY KEY(session_id, face_id),
                        UNIQUE(session_id, map_num)
                    );

                    CREATE TABLE IF NOT EXISTS events (
                        event_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL REFERENCES sessions(session_id),
                        event_type TEXT NOT NULL,
                        body_json TEXT NOT NULL,
                        body_sha256 TEXT NOT NULL,
                        order_index INTEGER NOT NULL,
                        captured_at TEXT NOT NULL,
                        response_json TEXT,
                        created_at TEXT NOT NULL,
                        UNIQUE(session_id, order_index)
                    );
                    CREATE INDEX IF NOT EXISTS idx_events_session_type_order
                        ON events(session_id, event_type, order_index);

                    CREATE TABLE IF NOT EXISTS assessments (
                        assessment_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL REFERENCES sessions(session_id),
                        revision INTEGER NOT NULL,
                        assessment_type TEXT NOT NULL,
                        model_key TEXT,
                        face_id TEXT,
                        map_num INTEGER,
                        zone_id TEXT,
                        snapshot_id TEXT,
                        snapshot_revision INTEGER,
                        captured_at TEXT,
                        input_manifest_sha256 TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        UNIQUE(
                            session_id, assessment_type, model_key, face_id,
                            map_num, zone_id, revision
                        )
                    );
                    CREATE INDEX IF NOT EXISTS idx_assessments_zone_latest
                        ON assessments(
                            session_id, assessment_type, model_key, face_id,
                            map_num, zone_id, revision DESC
                        );
                    """
                )
                existing = connection.execute(
                    "SELECT value FROM metadata WHERE key = 'schema_version'"
                ).fetchone()
                if existing is None:
                    connection.execute(
                        "INSERT INTO metadata(key, value) VALUES ('schema_version', ?)",
                        (str(SCHEMA_VERSION),),
                    )
                elif int(existing["value"]) != SCHEMA_VERSION:
                    raise EngineError(
                        "ledger_schema_mismatch",
                        "engine ledger schema is not supported by this code",
                        status_code=503,
                    )
        except sqlite3.Error as exc:
            raise EngineError(
                "persistent_ledger_unavailable",
                f"cannot initialize persistent engine ledger: {exc}",
                status_code=503,
            ) from exc

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    def health(self) -> dict[str, object]:
        try:
            with self.read() as connection:
                check = connection.execute("PRAGMA quick_check").fetchone()[0]
                schema = connection.execute(
                    "SELECT value FROM metadata WHERE key = 'schema_version'"
                ).fetchone()[0]
        except sqlite3.Error as exc:
            raise EngineError(
                "persistent_ledger_unavailable",
                f"engine ledger health check failed: {exc}",
                status_code=503,
            ) from exc
        return {
            "persistent": True,
            "path": str(self.path),
            "quick_check": check,
            "schema_version": int(schema),
        }
