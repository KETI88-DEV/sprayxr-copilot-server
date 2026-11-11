"""Durable corrected-v2 source ledger, outboxes, assessments and explanations.

The API opens this store in a dedicated SQLite file, separate from its legacy
``SequenceStore`` database.  It owns ``corrected_`` tables and a namespaced row
in its private ``sequences`` mirror for each v2 sequence.  Source input and its
engine outbox row commit in one ``BEGIN IMMEDIATE`` transaction.  Unlike the
legacy stores, corrected v2 never falls back to an in-memory database.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from trainer.demo_api.engine_contracts import (
    CONTRACT_VERSION,
    EVALUATION_MODE,
    CorrectedRequest,
    EngineAssessment,
    EngineEventAck,
    EngineEventEnvelope,
    ExplanationResponse,
    ProviderOfficialSummary,
    SequenceRequest,
    SessionCompleteRequest,
    SessionEndRequest,
    SessionStartRequest,
    ZoneObservationRequest,
    canonical_json,
    canonical_sha256,
    captured_at,
    natural_key,
    request_logical_body,
)


class CorrectedStoreError(RuntimeError):
    pass


class CorrectedStoreConflict(CorrectedStoreError):
    pass


class CorrectedStoreOrderError(CorrectedStoreError):
    pass


class CorrectedStoreNotFound(CorrectedStoreError):
    pass


@dataclass(frozen=True)
class EventReceipt:
    accepted: bool
    duplicate: bool
    event_id: str
    event_type: str
    session_id: str
    source_cursor: int
    outbox_status: str
    attempts: int
    last_error: str | None
    last_confirmed_engine_cursor: int
    session_revision: int
    assessment_id: str | None
    assessment_revision: int | None
    engine_state: dict[str, Any] | None


@dataclass(frozen=True)
class EngineOutboxItem:
    event: EngineEventEnvelope
    source_cursor: int
    attempts: int


@dataclass(frozen=True)
class ExplanationOutboxItem:
    assessment: EngineAssessment
    kind: str
    attempts: int


_SCHEMA = """
CREATE TABLE IF NOT EXISTS corrected_sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    scenario_id TEXT NOT NULL,
    evaluation_mode TEXT NOT NULL,
    contract_version TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    completed_at TEXT,
    target_thickness_um REAL,
    target_source TEXT,
    model_mapping TEXT NOT NULL,
    zone_manifest TEXT NOT NULL,
    expected_zones TEXT,
    lifecycle_status TEXT NOT NULL,
    source_cursor INTEGER NOT NULL DEFAULT 0,
    last_source_event_id TEXT,
    last_source_event_type TEXT,
    last_sequence_captured_at TEXT,
    last_engine_cursor INTEGER NOT NULL DEFAULT 0,
    last_engine_event_id TEXT,
    last_engine_event_type TEXT,
    session_revision INTEGER NOT NULL DEFAULT 0,
    latest_assessment_id TEXT,
    latest_assessment_revision INTEGER,
    official_summary TEXT,
    engine_session_state TEXT,
    CHECK (evaluation_mode = 'corrected_zone_v2'),
    CHECK (lifecycle_status IN ('active','ended','complete'))
);

CREATE TABLE IF NOT EXISTS corrected_source_events (
    event_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    natural_key TEXT NOT NULL,
    request_sha256 TEXT NOT NULL,
    logical_sha256 TEXT NOT NULL,
    request_body TEXT NOT NULL,
    engine_payload TEXT NOT NULL,
    captured_at TEXT,
    source_cursor INTEGER NOT NULL,
    assessment_id TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(session_id, event_type, natural_key),
    UNIQUE(session_id, source_cursor),
    FOREIGN KEY(session_id) REFERENCES corrected_sessions(session_id)
);

CREATE TABLE IF NOT EXISTS corrected_engine_outbox (
    event_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    source_cursor INTEGER NOT NULL,
    payload TEXT NOT NULL,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    next_attempt_at REAL NOT NULL DEFAULT 0,
    last_error TEXT,
    ack TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    delivered_at TEXT,
    CHECK (status IN ('pending','processing','failed_retryable','delivered','rejected')),
    FOREIGN KEY(event_id) REFERENCES corrected_source_events(event_id)
);
CREATE INDEX IF NOT EXISTS corrected_engine_outbox_ready
ON corrected_engine_outbox(status, next_attempt_at, source_cursor);

CREATE TABLE IF NOT EXISTS corrected_assessments (
    assessment_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    assessment_type TEXT NOT NULL,
    status TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL,
    received_at TEXT NOT NULL,
    CHECK (assessment_type IN ('zone','session')),
    FOREIGN KEY(session_id) REFERENCES corrected_sessions(session_id)
);
CREATE INDEX IF NOT EXISTS corrected_assessments_session
ON corrected_assessments(session_id, assessment_type, revision);

CREATE TABLE IF NOT EXISTS corrected_explanation_outbox (
    assessment_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    next_attempt_at REAL NOT NULL DEFAULT 0,
    last_error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    delivered_at TEXT,
    CHECK (kind IN ('zone','session')),
    CHECK (status IN ('pending','processing','failed_retryable','delivered','rejected')),
    FOREIGN KEY(assessment_id) REFERENCES corrected_assessments(assessment_id)
);
CREATE INDEX IF NOT EXISTS corrected_explanation_outbox_ready
ON corrected_explanation_outbox(status, next_attempt_at, revision);

CREATE TABLE IF NOT EXISTS corrected_explanations (
    assessment_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    status TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(assessment_id) REFERENCES corrected_assessments(assessment_id)
);

CREATE TABLE IF NOT EXISTS corrected_zone_heads (
    session_id TEXT NOT NULL,
    face_id TEXT NOT NULL,
    map_num INTEGER NOT NULL,
    zone_id TEXT NOT NULL,
    snapshot_id TEXT NOT NULL,
    snapshot_revision INTEGER NOT NULL,
    captured_at TEXT NOT NULL,
    event_id TEXT NOT NULL,
    PRIMARY KEY(session_id, face_id, map_num, zone_id),
    FOREIGN KEY(event_id) REFERENCES corrected_source_events(event_id)
);

CREATE TABLE IF NOT EXISTS corrected_query_turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    assessment_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    answer_status TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sequences (
    session_key TEXT NOT NULL,
    sequence_id TEXT NOT NULL,
    body_sha256 TEXT NOT NULL,
    frames INTEGER NOT NULL,
    received_at TEXT NOT NULL,
    body TEXT,
    PRIMARY KEY (session_key, sequence_id)
);
"""


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _epoch(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


class CorrectedStore:
    def __init__(self, db_path: str) -> None:
        if not db_path or db_path == ":memory:":
            raise CorrectedStoreError("corrected v2 requires a persistent SQLite path")
        path = Path(db_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=30000")
            conn.executescript(_SCHEMA)
            session_columns = {
                str(row[1]) for row in conn.execute("PRAGMA table_info(corrected_sessions)")
            }
            if "engine_session_state" not in session_columns:
                conn.execute(
                    "ALTER TABLE corrected_sessions ADD COLUMN engine_session_state TEXT"
                )
        except (OSError, sqlite3.Error) as exc:
            raise CorrectedStoreError(f"persistent corrected store unavailable: {path}") from exc
        actual = str(conn.execute("PRAGMA database_list").fetchone()[2] or "")
        if not actual:
            conn.close()
            raise CorrectedStoreError("corrected v2 refuses an in-memory SQLite connection")
        self.path = Path(actual).resolve()
        self._conn = conn
        self._guard = threading.Lock()
        # A process may stop after claiming work but before the HTTP result is
        # committed.  Replaying the same immutable event/assessment is safe.
        with self._conn:
            self._conn.execute(
                "UPDATE corrected_engine_outbox SET status='failed_retryable',"
                " next_attempt_at=0,last_error='worker_restarted',updated_at=?"
                " WHERE status='processing'",
                (_now(),),
            )
            self._conn.execute(
                "UPDATE corrected_explanation_outbox SET status='failed_retryable',"
                " next_attempt_at=0,last_error='worker_restarted',updated_at=?"
                " WHERE status='processing'",
                (_now(),),
            )

    @staticmethod
    def _event_type(body: CorrectedRequest) -> str:
        if isinstance(body, SessionStartRequest):
            return "session_start"
        if isinstance(body, SequenceRequest):
            return "sequence"
        if isinstance(body, ZoneObservationRequest):
            return "zone_observation"
        if isinstance(body, SessionEndRequest):
            return "session_end"
        if isinstance(body, SessionCompleteRequest):
            return "session_complete"
        raise TypeError(type(body).__name__)

    def accept(self, body: CorrectedRequest, event: EngineEventEnvelope) -> EventReceipt:
        event_type = self._event_type(body)
        if (event.event_id, event.event_type, event.session_id) != (
            body.event_id,
            event_type,
            body.session_id,
        ):
            raise CorrectedStoreConflict("internal event projection identity mismatch")
        request_dict = body.model_dump(mode="json")
        request_text = canonical_json(request_dict)
        request_hash = canonical_sha256(request_dict)
        logical_hash = canonical_sha256(request_logical_body(body))
        natural = natural_key(body)
        engine_text = canonical_json(event.model_dump(mode="json"))
        stamp = captured_at(body)
        created = _now()

        with self._guard, self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            existing = self._conn.execute(
                "SELECT request_sha256 FROM corrected_source_events WHERE event_id=?",
                (body.event_id,),
            ).fetchone()
            if existing is not None:
                if str(existing[0]) != request_hash:
                    raise CorrectedStoreConflict("event_id already exists with a different body")
                return self._receipt_locked(body.event_id, accepted=False, duplicate=True)

            same_natural = self._conn.execute(
                "SELECT event_id,logical_sha256 FROM corrected_source_events"
                " WHERE session_id=? AND event_type=? AND natural_key=?",
                (body.session_id, event_type, natural),
            ).fetchone()
            if same_natural is not None:
                if str(same_natural[1]) != logical_hash:
                    raise CorrectedStoreConflict(
                        "session event identity already exists with a different body"
                    )
                return self._receipt_locked(str(same_natural[0]), accepted=False, duplicate=True)

            session = self._conn.execute(
                "SELECT evaluation_mode,contract_version,started_at,ended_at,completed_at,"
                " lifecycle_status,source_cursor,last_sequence_captured_at"
                " FROM corrected_sessions WHERE session_id=?",
                (body.session_id,),
            ).fetchone()
            if isinstance(body, SessionStartRequest):
                if session is not None:
                    raise CorrectedStoreConflict("corrected session already exists without matching start")
                self._conn.execute(
                    "INSERT INTO corrected_sessions(session_id,user_id,scenario_id,evaluation_mode,"
                    " contract_version,started_at,target_thickness_um,target_source,model_mapping,"
                    " zone_manifest,expected_zones,lifecycle_status) VALUES(?,?,?,?,?,?,?,?,?,?,?,'active')",
                    (
                        body.session_id,
                        body.user_id,
                        body.scenario_id,
                        body.evaluation_mode,
                        body.contract_version,
                        body.started_at,
                        body.target_thickness_um,
                        body.target_source,
                        canonical_json(body.model_mapping.model_dump(mode="json")),
                        canonical_json(body.zone_manifest.model_dump(mode="json")),
                        canonical_json([item.model_dump(mode="json") for item in body.expected_zones])
                        if body.expected_zones is not None
                        else None,
                    ),
                )
                source_cursor = 1
            else:
                if session is None:
                    raise CorrectedStoreNotFound("corrected session must be started first")
                if session[0] != EVALUATION_MODE or session[1] != body.contract_version:
                    raise CorrectedStoreConflict("session mode or contract_version mismatch")
                started_at, ended_at, completed_at = session[2], session[3], session[4]
                lifecycle = str(session[5])
                if completed_at is not None or lifecycle == "complete":
                    raise CorrectedStoreOrderError("session is already complete")
                if stamp is not None and _epoch(stamp) < _epoch(str(started_at)):
                    raise CorrectedStoreOrderError("event timestamp precedes session start")
                if isinstance(body, SessionEndRequest):
                    if lifecycle != "active":
                        raise CorrectedStoreOrderError("session end is out of order")
                elif isinstance(body, SessionCompleteRequest):
                    if lifecycle != "ended" or ended_at is None:
                        raise CorrectedStoreOrderError("session must be ended before complete")
                    if _epoch(body.completed_at) < _epoch(str(ended_at)):
                        raise CorrectedStoreOrderError("completed_at precedes ended_at")
                elif ended_at is not None and stamp is not None and _epoch(stamp) > _epoch(str(ended_at)):
                    raise CorrectedStoreOrderError("late data was captured after session end")
                if isinstance(body, SequenceRequest):
                    last_sequence = session[7]
                    if lifecycle == "active" and last_sequence is not None and _epoch(body.captured_at) < _epoch(str(last_sequence)):
                        raise CorrectedStoreOrderError("sequence captured_at is out of order")
                if isinstance(body, ZoneObservationRequest):
                    head = self._conn.execute(
                        "SELECT snapshot_id,snapshot_revision,captured_at FROM corrected_zone_heads"
                        " WHERE session_id=? AND face_id=? AND map_num=? AND zone_id=?",
                        (body.session_id, body.face_id, body.map_num, body.zone_id),
                    ).fetchone()
                    if head is not None:
                        if body.snapshot_id == str(head[0]):
                            if body.snapshot_revision <= int(head[1]):
                                raise CorrectedStoreOrderError(
                                    "zone observation snapshot_revision must increase"
                                )
                        elif body.snapshot_revision != 1:
                            raise CorrectedStoreOrderError(
                                "a new snapshot_id must start at revision 1"
                            )
                        if _epoch(body.captured_at) < _epoch(str(head[2])):
                            raise CorrectedStoreOrderError("zone observation captured_at is stale")
                source_cursor = int(session[6]) + 1

            self._conn.execute(
                "INSERT INTO corrected_source_events(event_id,session_id,event_type,natural_key,"
                " request_sha256,logical_sha256,request_body,engine_payload,captured_at,source_cursor,created_at)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    body.event_id,
                    body.session_id,
                    event_type,
                    natural,
                    request_hash,
                    logical_hash,
                    request_text,
                    engine_text,
                    stamp,
                    source_cursor,
                    created,
                ),
            )
            self._conn.execute(
                "INSERT INTO corrected_engine_outbox(event_id,session_id,event_type,source_cursor,"
                " payload,status,created_at,updated_at) VALUES(?,?,?,?,?,'pending',?,?)",
                (body.event_id, body.session_id, event_type, source_cursor, engine_text, created, created),
            )

            if isinstance(body, SequenceRequest):
                sequence_key = f"{EVALUATION_MODE}:{body.session_id}"
                row = self._conn.execute(
                    "SELECT body_sha256 FROM sequences WHERE session_key=? AND sequence_id=?",
                    (sequence_key, body.sequence_id),
                ).fetchone()
                if row is not None:
                    if str(row[0]) != logical_hash:
                        raise CorrectedStoreConflict(
                            "sequence_id already exists with a different corrected-v2 body"
                        )
                else:
                    self._conn.execute(
                        "INSERT INTO sequences(session_key,sequence_id,body_sha256,frames,received_at,body)"
                        " VALUES(?,?,?,?,?,?)",
                        (
                            sequence_key,
                            body.sequence_id,
                            logical_hash,
                            len(body.frames),
                            created,
                            request_text,
                        ),
                    )
                self._conn.execute(
                    "UPDATE corrected_sessions SET last_sequence_captured_at=? WHERE session_id=?",
                    (body.captured_at, body.session_id),
                )
            elif isinstance(body, ZoneObservationRequest):
                self._conn.execute(
                    "INSERT INTO corrected_zone_heads(session_id,face_id,map_num,zone_id,snapshot_id,"
                    " snapshot_revision,captured_at,event_id) VALUES(?,?,?,?,?,?,?,?)"
                    " ON CONFLICT(session_id,face_id,map_num,zone_id) DO UPDATE SET"
                    " snapshot_id=excluded.snapshot_id,snapshot_revision=excluded.snapshot_revision,"
                    " captured_at=excluded.captured_at,event_id=excluded.event_id",
                    (
                        body.session_id,
                        body.face_id,
                        body.map_num,
                        body.zone_id,
                        body.snapshot_id,
                        body.snapshot_revision,
                        body.captured_at,
                        body.event_id,
                    ),
                )
            elif isinstance(body, SessionEndRequest):
                self._conn.execute(
                    "UPDATE corrected_sessions SET ended_at=?,lifecycle_status='ended' WHERE session_id=?",
                    (body.ended_at, body.session_id),
                )
            elif isinstance(body, SessionCompleteRequest):
                self._conn.execute(
                    "UPDATE corrected_sessions SET completed_at=?,lifecycle_status='complete',official_summary=?"
                    " WHERE session_id=?",
                    (
                        body.completed_at,
                        canonical_json(body.provider_official_summary.model_dump(mode="json")),
                        body.session_id,
                    ),
                )

            self._conn.execute(
                "UPDATE corrected_sessions SET source_cursor=?,last_source_event_id=?,"
                " last_source_event_type=? WHERE session_id=?",
                (source_cursor, body.event_id, event_type, body.session_id),
            )
            return self._receipt_locked(body.event_id, accepted=True, duplicate=False)

    def _receipt_locked(self, event_id: str, *, accepted: bool, duplicate: bool) -> EventReceipt:
        row = self._conn.execute(
            "SELECT s.event_id,s.event_type,s.session_id,s.source_cursor,o.status,o.attempts,o.last_error,"
            " c.last_engine_cursor,c.session_revision,s.assessment_id,a.revision,c.engine_session_state,"
            " EXISTS(SELECT 1 FROM corrected_engine_outbox prior WHERE prior.session_id=s.session_id"
            "  AND prior.source_cursor<s.source_cursor AND prior.status='rejected')"
            " FROM corrected_source_events s JOIN corrected_engine_outbox o ON o.event_id=s.event_id"
            " JOIN corrected_sessions c ON c.session_id=s.session_id"
            " LEFT JOIN corrected_assessments a ON a.assessment_id=s.assessment_id"
            " WHERE s.event_id=?",
            (event_id,),
        ).fetchone()
        if row is None:
            raise CorrectedStoreNotFound("event not found")
        status = "blocked" if bool(row[12]) and row[4] != "delivered" else str(row[4])
        return EventReceipt(
            accepted=accepted,
            duplicate=duplicate,
            event_id=str(row[0]),
            event_type=str(row[1]),
            session_id=str(row[2]),
            source_cursor=int(row[3]),
            outbox_status=status,
            attempts=int(row[5]),
            last_error=str(row[6]) if row[6] is not None else None,
            last_confirmed_engine_cursor=int(row[7]),
            session_revision=int(row[8]),
            assessment_id=str(row[9]) if row[9] is not None else None,
            assessment_revision=int(row[10]) if row[10] is not None else None,
            engine_state=json.loads(row[11]) if row[11] is not None else None,
        )

    def event_receipt(self, event_id: str) -> EventReceipt:
        with self._guard:
            return self._receipt_locked(event_id, accepted=False, duplicate=True)

    def claim_engine_event(self) -> EngineOutboxItem | None:
        now = time.time()
        with self._guard, self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            row = self._conn.execute(
                "SELECT o.event_id,o.payload,o.source_cursor,o.attempts FROM corrected_engine_outbox o"
                " WHERE o.status IN ('pending','failed_retryable') AND o.next_attempt_at<=?"
                " AND NOT EXISTS(SELECT 1 FROM corrected_engine_outbox prior"
                "  WHERE prior.session_id=o.session_id AND prior.source_cursor<o.source_cursor"
                "  AND prior.status<>'delivered')"
                " ORDER BY o.created_at,o.source_cursor LIMIT 1",
                (now,),
            ).fetchone()
            if row is None:
                return None
            attempts = int(row[3]) + 1
            self._conn.execute(
                "UPDATE corrected_engine_outbox SET status='processing',attempts=?,updated_at=?"
                " WHERE event_id=?",
                (attempts, _now(), row[0]),
            )
            return EngineOutboxItem(
                event=EngineEventEnvelope.model_validate(json.loads(row[1])),
                source_cursor=int(row[2]),
                attempts=attempts,
            )

    def mark_engine_failure(self, event_id: str, *, error: str, retryable: bool) -> None:
        bounded = error[:500]
        with self._guard, self._conn:
            row = self._conn.execute(
                "SELECT attempts FROM corrected_engine_outbox WHERE event_id=?", (event_id,)
            ).fetchone()
            if row is None:
                return
            attempts = int(row[0])
            status = "failed_retryable" if retryable else "rejected"
            delay = min(30.0, float(2 ** min(attempts, 5))) if retryable else 0.0
            self._conn.execute(
                "UPDATE corrected_engine_outbox SET status=?,next_attempt_at=?,last_error=?,updated_at=?"
                " WHERE event_id=?",
                (status, time.time() + delay, bounded, _now(), event_id),
            )

    def commit_engine_success(
        self,
        event_id: str,
        *,
        ack: EngineEventAck,
        assessment: EngineAssessment | None,
        engine_session: dict[str, Any],
    ) -> None:
        ack_text = canonical_json(ack.model_dump(mode="json"))
        engine_session_text = canonical_json(engine_session)
        now = _now()
        with self._guard, self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            row = self._conn.execute(
                "SELECT session_id,event_type,source_cursor,payload FROM corrected_engine_outbox WHERE event_id=?",
                (event_id,),
            ).fetchone()
            if row is None:
                raise CorrectedStoreNotFound("outbox event not found")
            if (ack.event_id, ack.session_id, ack.event_type) != (event_id, row[0], row[1]):
                raise CorrectedStoreConflict("engine acknowledgement does not match outbox event")
            if ack.payload_hash != canonical_sha256(json.loads(row[3])):
                raise CorrectedStoreConflict("engine acknowledgement payload hash mismatch")
            if ack.processing_status != "processed":
                raise CorrectedStoreConflict("engine acknowledgement is not processed")

            event_type = str(row[1])
            expected_assessment_type = {
                "zone_observation": "zone",
                "session_complete": "session",
            }.get(event_type)
            if expected_assessment_type is None:
                if ack.assessment_id is not None or assessment is not None:
                    raise CorrectedStoreConflict(
                        f"{event_type} must not produce an assessment"
                    )
                if ack.evidence_ids:
                    raise CorrectedStoreConflict(
                        f"{event_type} acknowledgement must not contain evidence_ids"
                    )
            else:
                if ack.assessment_id is None or assessment is None:
                    raise CorrectedStoreConflict(
                        f"{event_type} requires a {expected_assessment_type} assessment"
                    )
                if assessment.assessment_type != expected_assessment_type:
                    raise CorrectedStoreConflict(
                        f"{event_type} assessment type mismatch"
                    )
                if ack.session_revision != assessment.revision:
                    raise CorrectedStoreConflict("engine assessment revision mismatch")
                if ack.evidence_ids != assessment.evidence_ids:
                    raise CorrectedStoreConflict("engine assessment evidence_ids mismatch")
            if assessment is not None:
                if ack.assessment_id != assessment.assessment_id:
                    raise CorrectedStoreConflict("engine assessment_id mismatch")
                if assessment.session_id != str(row[0]):
                    raise CorrectedStoreConflict("assessment belongs to another session")
                payload = assessment.model_dump(mode="json")
                payload_text = canonical_json(payload)
                payload_hash = canonical_sha256(payload)
                old = self._conn.execute(
                    "SELECT payload_sha256 FROM corrected_assessments WHERE assessment_id=?",
                    (assessment.assessment_id,),
                ).fetchone()
                if old is not None and str(old[0]) != payload_hash:
                    raise CorrectedStoreConflict("assessment_id changed immutable content")
                self._conn.execute(
                    "INSERT OR IGNORE INTO corrected_assessments(assessment_id,session_id,revision,"
                    " assessment_type,status,payload_sha256,payload,created_at,received_at)"
                    " VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        assessment.assessment_id,
                        assessment.session_id,
                        assessment.revision,
                        assessment.assessment_type,
                        assessment.status,
                        payload_hash,
                        payload_text,
                        assessment.created_at,
                        now,
                    ),
                )
                self._conn.execute(
                    "INSERT OR IGNORE INTO corrected_explanation_outbox(assessment_id,session_id,revision,kind,"
                    " status,created_at,updated_at) VALUES(?,?,?,?,'pending',?,?)",
                    (
                        assessment.assessment_id,
                        assessment.session_id,
                        assessment.revision,
                        assessment.assessment_type,
                        now,
                        now,
                    ),
                )
                self._conn.execute(
                    "UPDATE corrected_source_events SET assessment_id=? WHERE event_id=?",
                    (assessment.assessment_id, event_id),
                )
                # Revisions are immutable within one Zone (and within the final
                # session assessment), but they are not a global session clock.
                # Engine outbox events are committed in source-cursor order, so
                # the assessment produced by the current event is the latest
                # even when its local revision is numerically lower.
                self._conn.execute(
                    "UPDATE corrected_sessions SET latest_assessment_id=?,latest_assessment_revision=?"
                    " WHERE session_id=?",
                    (
                        assessment.assessment_id,
                        assessment.revision,
                        assessment.session_id,
                    ),
                )
            self._conn.execute(
                "UPDATE corrected_engine_outbox SET status='delivered',ack=?,last_error=NULL,"
                " delivered_at=?,updated_at=? WHERE event_id=?",
                (ack_text, now, now, event_id),
            )
            self._conn.execute(
                "UPDATE corrected_sessions SET last_engine_cursor=?,last_engine_event_id=?,"
                " last_engine_event_type=?,session_revision=?,engine_session_state=?"
                " WHERE session_id=?",
                (
                    int(row[2]),
                    event_id,
                    str(row[1]),
                    ack.session_revision,
                    engine_session_text,
                    str(row[0]),
                ),
            )

    def claim_explanation(self) -> ExplanationOutboxItem | None:
        now = time.time()
        with self._guard, self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            row = self._conn.execute(
                "SELECT o.assessment_id,o.kind,o.attempts,a.payload"
                " FROM corrected_explanation_outbox o JOIN corrected_assessments a"
                " ON a.assessment_id=o.assessment_id"
                " WHERE o.status IN ('pending','failed_retryable') AND o.next_attempt_at<=?"
                " ORDER BY o.created_at,o.revision LIMIT 1",
                (now,),
            ).fetchone()
            if row is None:
                return None
            attempts = int(row[2]) + 1
            self._conn.execute(
                "UPDATE corrected_explanation_outbox SET status='processing',attempts=?,updated_at=?"
                " WHERE assessment_id=?",
                (attempts, _now(), row[0]),
            )
            return ExplanationOutboxItem(
                assessment=EngineAssessment.model_validate(json.loads(row[3])),
                kind=str(row[1]),
                attempts=attempts,
            )

    def mark_explanation_failure(
        self, assessment_id: str, *, error: str, retryable: bool
    ) -> None:
        with self._guard, self._conn:
            row = self._conn.execute(
                "SELECT attempts FROM corrected_explanation_outbox WHERE assessment_id=?",
                (assessment_id,),
            ).fetchone()
            if row is None:
                return
            attempts = int(row[0])
            status = "failed_retryable" if retryable else "rejected"
            delay = min(30.0, float(2 ** min(attempts, 5))) if retryable else 0.0
            self._conn.execute(
                "UPDATE corrected_explanation_outbox SET status=?,next_attempt_at=?,last_error=?,updated_at=?"
                " WHERE assessment_id=?",
                (status, time.time() + delay, error[:500], _now(), assessment_id),
            )

    def commit_explanation(
        self, assessment_id: str, explanation: ExplanationResponse
    ) -> None:
        payload = explanation.model_dump(mode="json")
        payload_text = canonical_json(payload)
        payload_hash = canonical_sha256(payload)
        now = _now()
        with self._guard, self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            row = self._conn.execute(
                "SELECT session_id,revision FROM corrected_explanation_outbox WHERE assessment_id=?",
                (assessment_id,),
            ).fetchone()
            if row is None:
                raise CorrectedStoreNotFound("explanation job not found")
            old = self._conn.execute(
                "SELECT payload_sha256 FROM corrected_explanations WHERE assessment_id=?",
                (assessment_id,),
            ).fetchone()
            if old is not None and str(old[0]) != payload_hash:
                raise CorrectedStoreConflict("assessment explanation changed immutable content")
            self._conn.execute(
                "INSERT OR IGNORE INTO corrected_explanations(assessment_id,session_id,revision,status,"
                " payload_sha256,payload,created_at) VALUES(?,?,?,?,?,?,?)",
                (assessment_id, row[0], int(row[1]), explanation.status, payload_hash, payload_text, now),
            )
            self._conn.execute(
                "UPDATE corrected_explanation_outbox SET status='delivered',last_error=NULL,"
                " delivered_at=?,updated_at=? WHERE assessment_id=?",
                (now, now, assessment_id),
            )

    def get_assessment(self, assessment_id: str) -> EngineAssessment | None:
        with self._guard:
            row = self._conn.execute(
                "SELECT payload FROM corrected_assessments WHERE assessment_id=?",
                (assessment_id,),
            ).fetchone()
        return EngineAssessment.model_validate(json.loads(row[0])) if row else None

    def get_explanation(self, assessment_id: str) -> ExplanationResponse | None:
        with self._guard:
            row = self._conn.execute(
                "SELECT payload FROM corrected_explanations WHERE assessment_id=?",
                (assessment_id,),
            ).fetchone()
        return ExplanationResponse.model_validate(json.loads(row[0])) if row else None

    def explanation_status(self, assessment_id: str) -> str:
        with self._guard:
            row = self._conn.execute(
                "SELECT status FROM corrected_explanation_outbox WHERE assessment_id=?",
                (assessment_id,),
            ).fetchone()
        if row is None:
            raise CorrectedStoreNotFound("assessment explanation job not found")
        return str(row[0])

    def session_status(self, session_id: str) -> dict[str, Any] | None:
        with self._guard:
            row = self._conn.execute(
                "SELECT session_id,evaluation_mode,contract_version,lifecycle_status,source_cursor,"
                " last_source_event_id,last_source_event_type,last_engine_cursor,last_engine_event_id,"
                " last_engine_event_type,session_revision,latest_assessment_id,latest_assessment_revision,"
                " engine_session_state"
                " FROM corrected_sessions WHERE session_id=?",
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            engine_counts = dict(
                self._conn.execute(
                    "SELECT status,COUNT(*) FROM corrected_engine_outbox WHERE session_id=? GROUP BY status",
                    (session_id,),
                ).fetchall()
            )
            explanation_counts = dict(
                self._conn.execute(
                    "SELECT status,COUNT(*) FROM corrected_explanation_outbox WHERE session_id=? GROUP BY status",
                    (session_id,),
                ).fetchall()
            )
        return {
            "session_id": str(row[0]),
            "evaluation_mode": str(row[1]),
            "contract_version": str(row[2]),
            "lifecycle_status": str(row[3]),
            "source_cursor": int(row[4]),
            "last_source_event_id": row[5],
            "last_source_event_type": row[6],
            "last_confirmed_engine_cursor": int(row[7]),
            "last_engine_event_id": row[8],
            "last_engine_event_type": row[9],
            "session_revision": int(row[10]),
            "latest_assessment_id": row[11],
            "latest_assessment_revision": int(row[12]) if row[12] is not None else None,
            "engine_outbox": {str(key): int(value) for key, value in engine_counts.items()},
            "explanation_outbox": {
                str(key): int(value) for key, value in explanation_counts.items()
            },
            "engine_state": json.loads(row[13]) if row[13] is not None else None,
        }

    def session_report(self, session_id: str) -> dict[str, Any] | None:
        with self._guard:
            session = self._conn.execute(
                "SELECT evaluation_mode,contract_version,official_summary FROM corrected_sessions"
                " WHERE session_id=?",
                (session_id,),
            ).fetchone()
            if session is None:
                return None
            assessment_row = self._conn.execute(
                "SELECT assessment_id,payload FROM corrected_assessments"
                " WHERE session_id=? AND assessment_type='session' ORDER BY revision DESC LIMIT 1",
                (session_id,),
            ).fetchone()
            explanation_row = None
            explanation_status = "pending"
            if assessment_row is not None:
                explanation_row = self._conn.execute(
                    "SELECT payload FROM corrected_explanations WHERE assessment_id=?",
                    (assessment_row[0],),
                ).fetchone()
                status_row = self._conn.execute(
                    "SELECT status FROM corrected_explanation_outbox WHERE assessment_id=?",
                    (assessment_row[0],),
                ).fetchone()
                explanation_status = str(status_row[0]) if status_row else "pending"
        assessment = (
            EngineAssessment.model_validate(json.loads(assessment_row[1]))
            if assessment_row is not None
            else None
        )
        explanation = (
            ExplanationResponse.model_validate(json.loads(explanation_row[0]))
            if explanation_row is not None
            else None
        )
        if explanation is not None:
            report_status = "available"
        elif explanation_status in {"failed_retryable", "processing", "pending"}:
            report_status = "failed_retryable" if explanation_status == "failed_retryable" else "pending"
        else:
            report_status = "rejected"
        return {
            "session_id": session_id,
            "evaluation_mode": str(session[0]),
            "contract_version": str(session[1]),
            "official_summary": ProviderOfficialSummary.model_validate(json.loads(session[2]))
            if session[2]
            else None,
            "assessment": assessment,
            "explanation": explanation,
            "report_status": report_status,
        }

    def record_query(
        self,
        *,
        session_id: str,
        assessment_id: str,
        revision: int,
        question: str,
        answer: str,
        answer_status: str,
    ) -> None:
        with self._guard, self._conn:
            self._conn.execute(
                "INSERT INTO corrected_query_turns(session_id,assessment_id,revision,question,answer,"
                " answer_status,created_at) VALUES(?,?,?,?,?,?,?)",
                (session_id, assessment_id, revision, question, answer, answer_status, _now()),
            )

    def close(self) -> None:
        with self._guard:
            self._conn.close()
