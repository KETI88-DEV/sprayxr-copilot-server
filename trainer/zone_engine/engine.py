"""Transactional orchestration for corrected Zone events and assessments."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from .assets import ZoneAssetCatalog, ZoneMask
from .behavior import COACHING_RULE_VERSION, build_coaching
from .contracts import (
    CONTRACT_VERSION,
    EventRequest,
    HeightPng,
    SequencePayload,
    SessionCompletePayload,
    SessionEndPayload,
    SessionStartPayload,
    StrictModel,
    ZoneObservationPayload,
    parse_payload,
    parse_utc,
)
from .errors import ConflictError, EngineError, NotFoundError
from .ledger import SQLiteLedger
from .numerical import (
    Surface,
    assess_zone,
    combine_current_rounds,
    decode_height_png,
    overwrite_cumulative_crops,
    place_crop,
)


ENGINE_VERSION = "sprayxr-corrected-zone-engine/1.0.0"
CRITERION_VERSION = "tsb-round-thickness-20261021+painted-zone-mean/1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def utc_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class CorrectedZoneEngine:
    """Synchronous CPU engine whose public state is fully reconstructible."""

    def __init__(
        self,
        db_path: Path,
        *,
        assets: ZoneAssetCatalog | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.ledger = SQLiteLedger(db_path)
        self.assets = assets or ZoneAssetCatalog.default()
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._lock = threading.RLock()

    def _now(self) -> str:
        current = self._clock()
        if current.tzinfo is None or current.utcoffset() is None:
            raise EngineError("clock_not_utc_aware", "engine clock must be timezone-aware")
        if current.utcoffset() != timedelta(0):
            raise EngineError("clock_not_utc", "engine clock must use UTC")
        return utc_iso(current)

    def process_event(self, request: EventRequest) -> dict[str, Any]:
        payload = parse_payload(request)
        body_json = canonical_json(request.model_dump(mode="json"))
        body_sha256 = sha256_text(body_json)
        with self._lock, self.ledger.transaction() as connection:
            existing = connection.execute(
                "SELECT body_sha256, response_json FROM events WHERE event_id = ?",
                (request.event_id,),
            ).fetchone()
            if existing is not None:
                if existing["body_sha256"] != body_sha256:
                    raise ConflictError(
                        "event_id_payload_conflict",
                        "event_id already exists with different canonical JSON",
                    )
                if existing["response_json"] is None:
                    raise EngineError(
                        "event_result_unavailable",
                        "committed event has no result",
                        status_code=503,
                    )
                response = json.loads(existing["response_json"])
                response["status"] = "duplicate"
                return response

            now = self._now()
            if request.event_type == "session_start":
                assert isinstance(payload, SessionStartPayload)
                response = self._process_start(
                    connection, request, payload, body_json, body_sha256, now
                )
            else:
                response = self._process_existing_session_event(
                    connection, request, payload, body_json, body_sha256, now
                )
            connection.execute(
                "UPDATE events SET response_json = ? WHERE event_id = ?",
                (canonical_json(response), request.event_id),
            )
            return response

    def _process_start(
        self,
        connection: sqlite3.Connection,
        request: EventRequest,
        payload: SessionStartPayload,
        body_json: str,
        body_sha256: str,
        now: str,
    ) -> dict[str, Any]:
        if payload.contract_version != CONTRACT_VERSION:
            raise EngineError("contract_version_unsupported", "unsupported corrected contract")
        if connection.execute(
            "SELECT 1 FROM sessions WHERE session_id = ?", (request.session_id,)
        ).fetchone():
            raise ConflictError(
                "session_already_started",
                "session already has a different start event",
            )
        self.assets.verify_session_reference(
            manifest_id=payload.zone_manifest.manifest_id,
            manifest_sha256=payload.zone_manifest.sha256,
            model_key=payload.zone_manifest.model_key,
            map_nums=payload.zone_manifest.map_nums,
        )
        if payload.model_mapping.geometry_model_key != payload.zone_manifest.model_key:
            raise ConflictError(
                "geometry_manifest_model_conflict",
                "explicit geometry model does not match the Zone manifest model",
            )
        family_rows = connection.execute(
            """
            SELECT * FROM sessions
            WHERE source_instance_id = ? AND session_family_id = ?
            """,
            (payload.source_instance_id, payload.model_mapping.session_family_id),
        ).fetchall()
        for row in family_rows:
            if int(row["round_index"]) == payload.model_mapping.round_index:
                raise ConflictError(
                    "round_index_conflict",
                    "round_index already exists in this session family",
                )
            if row["round_id"] == payload.model_mapping.round_id:
                raise ConflictError(
                    "round_id_conflict",
                    "round_id already exists in this session family",
                )
            immutable = (
                row["provider_model_key"],
                row["work_id"],
                row["geometry_model_key"],
                row["manifest_id"],
                row["manifest_sha256"],
            )
            proposed = (
                payload.model_mapping.provider_model_key,
                payload.model_mapping.work_id,
                payload.model_mapping.geometry_model_key,
                payload.zone_manifest.manifest_id,
                payload.zone_manifest.sha256,
            )
            if immutable != proposed:
                raise ConflictError(
                    "round_family_identity_conflict",
                    "round family immutable identity differs from an earlier round",
                )

        for expected in payload.expected_zones:
            if expected.map_num not in payload.zone_manifest.map_nums:
                raise ConflictError(
                    "expected_zone_face_not_declared",
                    "expected Zone references an undeclared face",
                )
            for zone_id in expected.zone_ids:
                self.assets.zone_mask(
                    payload.model_mapping.geometry_model_key, expected.map_num, zone_id
                )

        connection.execute(
            """
            INSERT INTO sessions(
                session_id, source_instance_id, user_id, scenario_id, started_at,
                evaluation_mode, contract_version, target_thickness_um, target_source,
                provider_model_key, work_id, session_family_id, round_index, round_id,
                geometry_model_key, manifest_id, manifest_sha256, map_nums_json,
                expected_zones_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request.session_id,
                payload.source_instance_id,
                payload.user_id,
                payload.scenario_id,
                payload.started_at,
                payload.evaluation_mode,
                payload.contract_version,
                payload.target_thickness_um,
                payload.target_source,
                payload.model_mapping.provider_model_key,
                payload.model_mapping.work_id,
                payload.model_mapping.session_family_id,
                payload.model_mapping.round_index,
                payload.model_mapping.round_id,
                payload.model_mapping.geometry_model_key,
                payload.zone_manifest.manifest_id,
                payload.zone_manifest.sha256,
                canonical_json(payload.zone_manifest.map_nums),
                canonical_json([item.model_dump(mode="json") for item in payload.expected_zones]),
                now,
                now,
            ),
        )
        for expected in payload.expected_zones:
            try:
                connection.execute(
                    "INSERT INTO face_maps(session_id, face_id, map_num) VALUES (?, ?, ?)",
                    (request.session_id, expected.face_id, expected.map_num),
                )
            except sqlite3.IntegrityError as exc:
                raise ConflictError(
                    "expected_face_mapping_conflict",
                    "expected face IDs and map numbers must be one-to-one",
                ) from exc
        self._insert_event(
            connection,
            request=request,
            body_json=body_json,
            body_sha256=body_sha256,
            captured_at=payload.started_at,
            now=now,
        )
        return self._event_response(
            request=request,
            body_sha256=body_sha256,
            assessment=None,
            revision=0,
            evidence_ids=[],
        )

    def _process_existing_session_event(
        self,
        connection: sqlite3.Connection,
        request: EventRequest,
        payload: StrictModel,
        body_json: str,
        body_sha256: str,
        now: str,
    ) -> dict[str, Any]:
        session = connection.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (request.session_id,)
        ).fetchone()
        if session is None:
            raise ConflictError("session_not_started", "session_start must be accepted first")
        if session["completed_at"] is not None:
            raise ConflictError("session_already_completed", "completed session is sealed")

        assessment: dict[str, Any] | None = None
        captured_at: str
        if request.event_type == "sequence":
            assert isinstance(payload, SequencePayload)
            captured_at = payload.captured_at
            self._validate_physical_time(session, captured_at)
            self._validate_sequence(connection, session, payload)
            self._ensure_face_map(connection, request.session_id, payload.face_id, payload.map_num)
            if payload.cumulative_crop is not None:
                self._validate_height_geometry(session, payload.map_num, payload.cumulative_crop.height_png)
                decode_height_png(payload.cumulative_crop.height_png)
                self._validate_cumulative_snapshot(connection, session, payload)
        elif request.event_type == "zone_observation":
            assert isinstance(payload, ZoneObservationPayload)
            captured_at = payload.captured_at
            self._validate_physical_time(session, captured_at)
            self._validate_observation(connection, session, payload)
            self._ensure_face_map(connection, request.session_id, payload.face_id, payload.map_num)
        elif request.event_type == "session_end":
            assert isinstance(payload, SessionEndPayload)
            captured_at = payload.ended_at
            if session["ended_at"] is not None:
                raise ConflictError("session_already_ended", "session end is immutable")
            if parse_utc(payload.ended_at, field_name="ended_at") < parse_utc(
                session["started_at"], field_name="started_at"
            ):
                raise ConflictError("end_before_start", "ended_at precedes started_at")
        elif request.event_type == "session_complete":
            assert isinstance(payload, SessionCompletePayload)
            captured_at = payload.completed_at
            if session["ended_at"] is None:
                raise ConflictError("session_not_ended", "session_complete requires session_end")
            if parse_utc(payload.completed_at, field_name="completed_at") < parse_utc(
                session["ended_at"], field_name="ended_at"
            ):
                raise ConflictError("complete_before_end", "completed_at precedes ended_at")
        else:  # protected by EventType
            raise EngineError("event_type_unsupported", "unsupported event type")

        self._insert_event(
            connection,
            request=request,
            body_json=body_json,
            body_sha256=body_sha256,
            captured_at=captured_at,
            now=now,
        )

        if request.event_type == "zone_observation":
            assert isinstance(payload, ZoneObservationPayload)
            assessment = self._build_zone_assessment(
                connection, session, request.event_id, payload, now
            )
            self._insert_assessment(connection, assessment)
        elif request.event_type == "session_end":
            connection.execute(
                "UPDATE sessions SET ended_at = ?, updated_at = ? WHERE session_id = ?",
                (captured_at, now, request.session_id),
            )
        elif request.event_type == "session_complete":
            assert isinstance(payload, SessionCompletePayload)
            connection.execute(
                """
                UPDATE sessions
                SET completed_at = ?, provider_summary_json = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (
                    payload.completed_at,
                    canonical_json(payload.provider_official_summary.model_dump(mode="json")),
                    now,
                    request.session_id,
                ),
            )
            session = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (request.session_id,)
            ).fetchone()
            assessment = self._build_session_assessment(
                connection, session, request.event_id, now
            )
            self._insert_assessment(connection, assessment)

        if assessment is not None:
            connection.execute(
                """
                UPDATE sessions SET latest_assessment_id = ?,
                    latest_assessment_revision = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (
                    assessment["assessment_id"],
                    assessment["revision"],
                    now,
                    request.session_id,
                ),
            )
        if assessment is not None:
            response_revision = int(assessment["revision"])
        else:
            current_revision = connection.execute(
                "SELECT latest_assessment_revision FROM sessions WHERE session_id = ?",
                (request.session_id,),
            ).fetchone()["latest_assessment_revision"]
            response_revision = int(current_revision) if current_revision is not None else 0
        return self._event_response(
            request=request,
            body_sha256=body_sha256,
            assessment=assessment,
            revision=response_revision,
            evidence_ids=assessment["evidence_ids"] if assessment is not None else [],
        )

    def _insert_event(
        self,
        connection: sqlite3.Connection,
        *,
        request: EventRequest,
        body_json: str,
        body_sha256: str,
        captured_at: str,
        now: str,
    ) -> int:
        session = connection.execute(
            "SELECT last_event_order FROM sessions WHERE session_id = ?", (request.session_id,)
        ).fetchone()
        order_index = int(session["last_event_order"]) + 1
        connection.execute(
            """
            INSERT INTO events(
                event_id, session_id, event_type, body_json, body_sha256,
                order_index, captured_at, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request.event_id,
                request.session_id,
                request.event_type,
                body_json,
                body_sha256,
                order_index,
                captured_at,
                now,
            ),
        )
        connection.execute(
            """
            UPDATE sessions SET last_event_order = ?, last_event_id = ?,
                last_event_type = ?, updated_at = ? WHERE session_id = ?
            """,
            (order_index, request.event_id, request.event_type, now, request.session_id),
        )
        return order_index

    @staticmethod
    def _event_response(
        *,
        request: EventRequest,
        body_sha256: str,
        assessment: dict[str, Any] | None,
        revision: int,
        evidence_ids: list[str],
    ) -> dict[str, Any]:
        return {
            "status": "accepted",
            "event_id": request.event_id,
            "event_type": request.event_type,
            "session_id": request.session_id,
            "payload_hash": body_sha256,
            "assessment_id": assessment["assessment_id"] if assessment else None,
            "session_revision": revision,
            "processing_status": "processed",
            "evidence_ids": evidence_ids,
        }

    def _validate_physical_time(self, session: sqlite3.Row, captured_at: str) -> None:
        captured = parse_utc(captured_at, field_name="captured_at")
        started = parse_utc(session["started_at"], field_name="started_at")
        if captured < started:
            raise ConflictError("capture_before_start", "captured_at precedes session start")
        if session["ended_at"] is not None:
            ended = parse_utc(session["ended_at"], field_name="ended_at")
            if captured > ended:
                raise ConflictError(
                    "capture_after_end",
                    "post-end capture is rejected; only delayed pre-end capture is allowed",
                )

    def _validate_sequence(
        self,
        connection: sqlite3.Connection,
        session: sqlite3.Row,
        payload: SequencePayload,
    ) -> None:
        self._validate_round_and_face(session, payload.round_id, payload.map_num)
        captured = parse_utc(payload.captured_at, field_name="captured_at")
        anchor = (
            parse_utc(payload.sequence_started_at, field_name="sequence_started_at")
            if payload.sequence_started_at is not None
            else None
        )
        for frame in payload.frames:
            if frame.timestamp is not None:
                frame_time = parse_utc(frame.timestamp, field_name="frame.timestamp")
            else:
                assert anchor is not None and frame.relative_time_s is not None
                frame_time = anchor + timedelta(seconds=frame.relative_time_s)
            if frame_time > captured:
                raise ConflictError("frame_after_capture", "frame timestamp exceeds captured_at")
            if frame_time < parse_utc(session["started_at"], field_name="started_at"):
                raise ConflictError("frame_before_session", "frame timestamp precedes session")
            if frame.raytexmapnum is not None and frame.raytexmapnum != payload.map_num:
                raise ConflictError("frame_map_conflict", "frame map differs from sequence face")
            if frame.zone_id is not None:
                self.assets.zone_mask(
                    session["geometry_model_key"], payload.map_num, frame.zone_id
                )
            if frame.raytexuv is not None:
                resolved = self.assets.zone_at_uv(
                    session["geometry_model_key"],
                    payload.map_num,
                    frame.raytexuv.x,
                    frame.raytexuv.y,
                )
                if frame.zone_id is not None and resolved != frame.zone_id:
                    raise ConflictError(
                        "frame_zone_uv_conflict",
                        "explicit frame Zone conflicts with exact-color UV lookup",
                    )

    def _validate_observation(
        self,
        connection: sqlite3.Connection,
        session: sqlite3.Row,
        payload: ZoneObservationPayload,
    ) -> None:
        self._validate_round_and_face(session, payload.round_id, payload.map_num)
        self._validate_height_geometry(session, payload.map_num, payload.height_png)
        self.assets.zone_mask(session["geometry_model_key"], payload.map_num, payload.zone_id)
        expected = json.loads(session["expected_zones_json"])
        if expected:
            allowed = {
                (item["face_id"], int(item["map_num"]), zone_id)
                for item in expected
                for zone_id in item["zone_ids"]
            }
            if (payload.face_id, payload.map_num, payload.zone_id) not in allowed:
                raise ConflictError(
                    "zone_outside_declared_task",
                    "Zone observation is outside the session target set",
                )
        previous = self._zone_observation_events(
            connection,
            session_id=session["session_id"],
            face_id=payload.face_id,
            map_num=payload.map_num,
            zone_id=payload.zone_id,
        )
        same_snapshot = [
            item for item in previous if item[1].snapshot_id == payload.snapshot_id
        ]
        if same_snapshot:
            maximum = max(item[1].snapshot_revision for item in same_snapshot)
            if payload.snapshot_revision <= maximum:
                raise ConflictError(
                    "snapshot_revision_not_monotonic",
                    "snapshot revision must increase for the same snapshot_id",
                )
        elif payload.snapshot_revision != 1:
            raise ConflictError(
                "snapshot_revision_must_start_at_one",
                "a new snapshot_id must start at revision 1",
            )

    def _validate_cumulative_snapshot(
        self,
        connection: sqlite3.Connection,
        session: sqlite3.Row,
        payload: SequencePayload,
    ) -> None:
        assert payload.cumulative_crop is not None
        prior = connection.execute(
            """
            SELECT body_json FROM events
            WHERE session_id = ? AND event_type = 'sequence'
            ORDER BY order_index
            """,
            (session["session_id"],),
        ).fetchall()
        revisions: list[int] = []
        for row in prior:
            request = EventRequest.model_validate(json.loads(row["body_json"]))
            previous = parse_payload(request)
            assert isinstance(previous, SequencePayload)
            crop = previous.cumulative_crop
            if (
                crop is not None
                and previous.face_id == payload.face_id
                and previous.map_num == payload.map_num
                and crop.snapshot_id == payload.cumulative_crop.snapshot_id
            ):
                revisions.append(crop.snapshot_revision)
        if revisions:
            if payload.cumulative_crop.snapshot_revision <= max(revisions):
                raise ConflictError(
                    "cumulative_revision_not_monotonic",
                    "cumulative snapshot revision must increase",
                )
        elif payload.cumulative_crop.snapshot_revision != 1:
            raise ConflictError(
                "cumulative_revision_must_start_at_one",
                "a new cumulative snapshot_id must start at revision 1",
            )

    @staticmethod
    def _validate_round_and_face(
        session: sqlite3.Row, round_id: str, map_num: int
    ) -> None:
        if round_id != session["round_id"]:
            raise ConflictError("round_id_conflict", "event round does not match session round")
        if map_num not in json.loads(session["map_nums_json"]):
            raise ConflictError("map_not_declared", "event face is absent from Zone manifest ref")

    def _validate_height_geometry(
        self, session: sqlite3.Row, map_num: int, height_png: HeightPng
    ) -> None:
        map_spec = self.assets.map_spec(session["geometry_model_key"], map_num)
        expected = (int(map_spec["width"]), int(map_spec["height"]))
        actual = (height_png.crop.atlas_width, height_png.crop.atlas_height)
        if actual != expected:
            raise ConflictError(
                "height_atlas_geometry_conflict",
                "Height atlas dimensions must exactly match the pinned Zone label map",
            )

    @staticmethod
    def _ensure_face_map(
        connection: sqlite3.Connection, session_id: str, face_id: str, map_num: int
    ) -> None:
        row = connection.execute(
            "SELECT map_num FROM face_maps WHERE session_id = ? AND face_id = ?",
            (session_id, face_id),
        ).fetchone()
        if row is not None:
            if int(row["map_num"]) != map_num:
                raise ConflictError(
                    "face_map_conflict", "face_id was already bound to another map_num"
                )
            return
        try:
            connection.execute(
                "INSERT INTO face_maps(session_id, face_id, map_num) VALUES (?, ?, ?)",
                (session_id, face_id, map_num),
            )
        except sqlite3.IntegrityError as exc:
            raise ConflictError(
                "map_face_conflict", "map_num was already bound to another face_id"
            ) from exc

    def _zone_observation_events(
        self,
        connection: sqlite3.Connection,
        *,
        session_id: str,
        face_id: str,
        map_num: int,
        zone_id: str,
    ) -> list[tuple[sqlite3.Row, ZoneObservationPayload]]:
        rows = connection.execute(
            """
            SELECT * FROM events
            WHERE session_id = ? AND event_type = 'zone_observation'
            ORDER BY order_index
            """,
            (session_id,),
        ).fetchall()
        output: list[tuple[sqlite3.Row, ZoneObservationPayload]] = []
        for row in rows:
            request = EventRequest.model_validate(json.loads(row["body_json"]))
            payload = parse_payload(request)
            assert isinstance(payload, ZoneObservationPayload)
            if (
                payload.face_id == face_id
                and payload.map_num == map_num
                and payload.zone_id == zone_id
            ):
                output.append((row, payload))
        return output

    def _surface_for_observation(
        self,
        connection: sqlite3.Connection,
        session: sqlite3.Row,
        payload: ZoneObservationPayload,
    ) -> tuple[Surface, list[sqlite3.Row], list[str]]:
        current_row = connection.execute(
            "SELECT * FROM events WHERE event_id = ?",
            (
                connection.execute(
                    "SELECT last_event_id FROM sessions WHERE session_id = ?",
                    (session["session_id"],),
                ).fetchone()["last_event_id"],
            ),
        ).fetchone()
        current_surface = place_crop(decode_height_png(payload.height_png), payload.height_png)
        if payload.height_png.basis == "cumulative_crop":
            return current_surface, [current_row], []

        family = connection.execute(
            """
            SELECT * FROM sessions
            WHERE source_instance_id = ? AND session_family_id = ? AND round_index <= ?
            ORDER BY round_index
            """,
            (
                session["source_instance_id"],
                session["session_family_id"],
                session["round_index"],
            ),
        ).fetchall()
        required = list(range(1, int(session["round_index"]) + 1))
        observed_indices = [int(item["round_index"]) for item in family]
        missing_reasons: list[str] = []
        if observed_indices != required:
            missing_reasons.append("round_session_lineage_incomplete")

        rows: list[sqlite3.Row] = []
        surfaces: list[Surface] = []
        for round_session in family:
            events = self._zone_observation_events(
                connection,
                session_id=round_session["session_id"],
                face_id=payload.face_id,
                map_num=payload.map_num,
                zone_id=payload.zone_id,
            )
            if not events:
                missing_reasons.append(
                    f"round_{round_session['round_index']}_zone_observation_missing"
                )
                continue
            row, latest = events[-1]
            if latest.height_png.basis != "current_round":
                missing_reasons.append(
                    f"round_{round_session['round_index']}_height_basis_not_current_round"
                )
                continue
            rows.append(row)
            surfaces.append(place_crop(decode_height_png(latest.height_png), latest.height_png))
        if missing_reasons or len(surfaces) != len(required):
            return current_surface, [current_row], sorted(set(missing_reasons))
        return combine_current_rounds(surfaces), rows, []

    def _behavior_samples(
        self,
        connection: sqlite3.Connection,
        session: sqlite3.Row,
        *,
        face_id: str,
        map_num: int,
        zone_id: str,
        cutoff: datetime,
    ) -> tuple[dict[str, list[float]], dict[str, list[str]]]:
        values = {"speed_mm_s": [], "distance_mm": []}
        source_ids = {"speed_mm_s": [], "distance_mm": []}
        rows = connection.execute(
            """
            SELECT * FROM events
            WHERE session_id = ? AND event_type = 'sequence'
            ORDER BY order_index
            """,
            (session["session_id"],),
        ).fetchall()
        for row in rows:
            request = EventRequest.model_validate(json.loads(row["body_json"]))
            sequence = parse_payload(request)
            assert isinstance(sequence, SequencePayload)
            if sequence.face_id != face_id or sequence.map_num != map_num:
                continue
            anchor = (
                parse_utc(sequence.sequence_started_at, field_name="sequence_started_at")
                if sequence.sequence_started_at is not None
                else None
            )
            for frame in sequence.frames:
                if frame.timestamp is not None:
                    frame_time = parse_utc(frame.timestamp, field_name="frame.timestamp")
                else:
                    assert anchor is not None and frame.relative_time_s is not None
                    frame_time = anchor + timedelta(seconds=frame.relative_time_s)
                if frame_time > cutoff or not frame.spray_on:
                    continue
                resolved_zone = frame.zone_id
                if frame.raytexuv is not None:
                    resolved_zone = self.assets.zone_at_uv(
                        session["geometry_model_key"],
                        map_num,
                        frame.raytexuv.x,
                        frame.raytexuv.y,
                    )
                if resolved_zone != zone_id:
                    continue
                if frame.speed_mm_s is not None:
                    values["speed_mm_s"].append(frame.speed_mm_s)
                    source_ids["speed_mm_s"].append(row["event_id"])
                if frame.distance_mm is not None:
                    values["distance_mm"].append(frame.distance_mm)
                    source_ids["distance_mm"].append(row["event_id"])
        return values, source_ids

    def _build_zone_assessment(
        self,
        connection: sqlite3.Connection,
        session: sqlite3.Row,
        observation_event_id: str,
        payload: ZoneObservationPayload,
        now: str,
    ) -> dict[str, Any]:
        mask: ZoneMask = self.assets.zone_mask(
            session["geometry_model_key"], payload.map_num, payload.zone_id
        )
        surface, source_rows, lineage_reasons = self._surface_for_observation(
            connection, session, payload
        )
        measured = assess_zone(surface, mask.values, session["target_thickness_um"])
        if lineage_reasons:
            measured["quality"].update(
                {
                    "analysis_status": "unavailable",
                    "reason_codes": ["round_lineage_incomplete", *lineage_reasons],
                    "grade": None,
                    "painted_mean_um": None,
                    "painted_pixel_count": 0,
                    "unpainted_fraction": None,
                    "six_band_counts": None,
                    "statistics_scope": "none",
                }
            )
            measured["metrics"]["observed_statistics"] = None

        revision_row = connection.execute(
            """
            SELECT COALESCE(MAX(revision), 0) AS value FROM assessments
            WHERE session_id = ? AND assessment_type = 'zone' AND model_key = ?
                AND face_id = ? AND map_num = ? AND zone_id = ?
            """,
            (
                session["session_id"],
                session["geometry_model_key"],
                payload.face_id,
                payload.map_num,
                payload.zone_id,
            ),
        ).fetchone()
        revision = int(revision_row["value"]) + 1
        behavior_values, behavior_source_ids = self._behavior_samples(
            connection,
            session,
            face_id=payload.face_id,
            map_num=payload.map_num,
            zone_id=payload.zone_id,
            cutoff=parse_utc(payload.captured_at, field_name="captured_at"),
        )
        behavior_event_ids = sorted(
            set(behavior_source_ids["speed_mm_s"] + behavior_source_ids["distance_mm"])
        )
        source_rows_by_id = {row["event_id"]: row for row in source_rows}
        source_rows_by_id[observation_event_id] = connection.execute(
            "SELECT * FROM events WHERE event_id = ?", (observation_event_id,)
        ).fetchone()
        for event_id in behavior_event_ids:
            source_rows_by_id[event_id] = connection.execute(
                "SELECT * FROM events WHERE event_id = ?", (event_id,)
            ).fetchone()
        ordered_rows = sorted(source_rows_by_id.values(), key=lambda row: row["created_at"])
        input_manifest = {
            "schema_version": "sprayxr-assessment-input/v1",
            "events": [
                {"event_id": row["event_id"], "sha256": row["body_sha256"]}
                for row in ordered_rows
            ],
            "height_artifact_sha256": payload.height_png.sha256,
            "snapshot_id": payload.snapshot_id,
            "snapshot_revision": payload.snapshot_revision,
            "session_family_id": session["session_family_id"],
            "round_index": session["round_index"],
            "face_id": payload.face_id,
            "map_num": payload.map_num,
            "zone_id": payload.zone_id,
            "target_thickness_um": session["target_thickness_um"],
            "contract_version": session["contract_version"],
            "criterion_version": CRITERION_VERSION,
            "engine_version": ENGINE_VERSION,
            "zone_asset_manifest_sha256": session["manifest_sha256"],
            "mask_sha256": mask.sha256,
        }
        input_sha256 = sha256_text(canonical_json(input_manifest))
        assessment_id = f"za-{input_sha256[:32]}"
        coaching, evidence, behavior_stats = build_coaching(
            namespace=assessment_id,
            quality=measured["quality"],
            behavior_values=behavior_values,
            behavior_source_ids=behavior_source_ids,
            observation_event_id=observation_event_id,
            contract_version=session["contract_version"],
            revision=revision,
        )
        lineage_evidence_id = f"ev-lineage-{input_sha256[:24]}"
        evidence.append(
            {
                "evidence_id": lineage_evidence_id,
                "evidence_type": "lineage",
                "statement": (
                    "순서가 고정된 원천 이벤트와 변경 불가 평가 입력 명세의 "
                    "SHA-256 해시입니다."
                ),
                "value": input_sha256,
                "unit": None,
                "source_event_ids": [row["event_id"] for row in ordered_rows],
                "contract_version": session["contract_version"],
                "revision": revision,
            }
        )
        measured["metrics"]["behavior_statistics"] = behavior_stats
        source_event_ids = [row["event_id"] for row in ordered_rows]
        provenance = {
            "engine_version": ENGINE_VERSION,
            "criterion_version": CRITERION_VERSION,
            "coaching_rule_version": COACHING_RULE_VERSION,
            "contract_version": session["contract_version"],
            "manifest_id": session["manifest_id"],
            "zone_asset_manifest_sha256": session["manifest_sha256"],
            "label_map_sha256": mask.label_map_sha256,
            "mask_sha256": mask.sha256,
            "source_event_ids": source_event_ids,
            "assessment_input_manifest_sha256": input_sha256,
            "snapshot_id": payload.snapshot_id,
            "snapshot_revision": payload.snapshot_revision,
            "captured_at": payload.captured_at,
        }
        return {
            "assessment_id": assessment_id,
            "session_id": session["session_id"],
            "revision": revision,
            "assessment_type": "zone",
            "status": measured["quality"]["analysis_status"],
            "zone_key": {
                "model_key": session["geometry_model_key"],
                "face_id": payload.face_id,
                "map_num": payload.map_num,
                "zone_id": payload.zone_id,
            },
            "quality": measured["quality"],
            "metrics": measured["metrics"],
            "coaching": coaching,
            "evidence": evidence,
            "evidence_ids": [item["evidence_id"] for item in evidence],
            "created_at": now,
            "provenance": provenance,
            "_storage": {
                "snapshot_id": payload.snapshot_id,
                "snapshot_revision": payload.snapshot_revision,
                "captured_at": payload.captured_at,
                "input_manifest_sha256": input_sha256,
            },
        }

    def _latest_zone_assessments(
        self, connection: sqlite3.Connection, session_id: str
    ) -> list[dict[str, Any]]:
        rows = connection.execute(
            """
            SELECT payload_json FROM assessments
            WHERE session_id = ? AND assessment_type = 'zone'
            ORDER BY revision DESC, created_at DESC
            """,
            (session_id,),
        ).fetchall()
        latest: dict[tuple[str, str, int, str], dict[str, Any]] = {}
        for row in rows:
            assessment = json.loads(row["payload_json"])
            key_data = assessment["zone_key"]
            key = (
                key_data["model_key"],
                key_data["face_id"],
                int(key_data["map_num"]),
                key_data["zone_id"],
            )
            latest.setdefault(key, assessment)
        return sorted(
            latest.values(),
            key=lambda item: (
                item["zone_key"]["face_id"],
                item["zone_key"]["map_num"],
                int(item["zone_key"]["zone_id"]),
            ),
        )

    def _build_session_assessment(
        self,
        connection: sqlite3.Connection,
        session: sqlite3.Row,
        complete_event_id: str,
        now: str,
    ) -> dict[str, Any]:
        zones = self._latest_zone_assessments(connection, session["session_id"])
        latest_by_key = {
            (
                item["zone_key"]["face_id"],
                int(item["zone_key"]["map_num"]),
                item["zone_key"]["zone_id"],
            ): item
            for item in zones
        }
        expected_spec = json.loads(session["expected_zones_json"])
        expected_keys = [
            (item["face_id"], int(item["map_num"]), zone_id)
            for item in expected_spec
            for zone_id in item["zone_ids"]
        ]
        missing = [
            {"face_id": face, "map_num": map_num, "zone_id": zone_id}
            for face, map_num, zone_id in expected_keys
            if (face, map_num, zone_id) not in latest_by_key
        ]
        pending = [
            {
                "face_id": key[0],
                "map_num": key[1],
                "zone_id": key[2],
                "assessment_id": latest_by_key[key]["assessment_id"],
                "status": latest_by_key[key]["status"],
            }
            for key in expected_keys
            if key in latest_by_key and latest_by_key[key]["status"] != "available"
        ]

        painted_sum = 0.0
        painted_count = 0
        unpainted_count = 0
        observed_zone_count = 0
        known_zone_count = 0
        band_totals: dict[str, int] | None = None
        for assessment in zones:
            coverage = assessment["metrics"]["coverage"]
            known_zone_count += int(coverage["zone_pixel_count"])
            observed_zone_count += int(coverage["observed_zone_pixel_count"])
            observed = assessment["metrics"].get("observed_statistics")
            if observed is None:
                continue
            painted_sum += float(observed["painted_sum_um"])
            painted_count += int(observed["painted_pixel_count"])
            unpainted_count += int(observed["unpainted_pixel_count"])
            counts = observed.get("six_band_counts")
            if counts is not None:
                if band_totals is None:
                    band_totals = {key: 0 for key in counts}
                for key, value in counts.items():
                    band_totals[key] += int(value)
        observed_mean = painted_sum / painted_count if painted_count else None

        if expected_keys:
            total_zone_count = 0
            for face_id, map_num, zone_id in expected_keys:
                del face_id
                mask = self.assets.zone_mask(
                    session["geometry_model_key"], map_num, zone_id
                )
                total_zone_count += int(mask.values.sum())
            full_scope = not missing and not pending and all(
                latest_by_key[key]["metrics"]["coverage"]["full_zone"]
                for key in expected_keys
            )
        else:
            total_zone_count = known_zone_count
            full_scope = False

        reason_codes: list[str] = []
        if not expected_keys:
            reason_codes.append("expected_zone_set_missing")
        if missing:
            reason_codes.append("expected_zones_missing")
        if pending:
            reason_codes.append("latest_zone_revision_pending")
        if session["target_thickness_um"] is None:
            reason_codes.append("target_thickness_missing")
        if not zones:
            reason_codes.append("zone_assessments_missing")

        if not zones:
            analysis_status = "unavailable"
        elif session["target_thickness_um"] is None:
            analysis_status = "unavailable"
        elif full_scope:
            analysis_status = "available"
        else:
            analysis_status = "partial"
        quality = {
            "analysis_status": analysis_status,
            "reason_codes": reason_codes,
            "grade": None,
            "target_thickness_um": session["target_thickness_um"],
            "painted_mean_um": observed_mean,
            "painted_pixel_count": painted_count,
            "zone_pixel_count": total_zone_count,
            "unpainted_fraction": (
                unpainted_count / total_zone_count if full_scope and total_zone_count else None
            ),
            "six_band_counts": band_totals if full_scope else None,
            "statistics_scope": "whole_zone" if full_scope else ("observed_only" if zones else "none"),
        }
        official_summary = (
            json.loads(session["provider_summary_json"])
            if session["provider_summary_json"] is not None
            else None
        )
        metrics = {
            "observed_statistics": (
                {
                    "painted_mean_um": observed_mean,
                    "painted_sum_um": painted_sum,
                    "painted_pixel_count": painted_count,
                    "unpainted_pixel_count": unpainted_count,
                    "six_band_counts": band_totals,
                }
                if zones
                else None
            ),
            "coverage": {
                "zone_pixel_count": total_zone_count,
                "observed_zone_pixel_count": observed_zone_count,
                "fraction": (
                    observed_zone_count / total_zone_count if total_zone_count else 0.0
                ),
                "full_zone": full_scope,
            },
            "provider_official_summary": official_summary,
        }
        revision_row = connection.execute(
            """
            SELECT COALESCE(MAX(revision), 0) AS value FROM assessments
            WHERE session_id = ? AND assessment_type = 'session'
            """,
            (session["session_id"],),
        ).fetchone()
        revision = int(revision_row["value"]) + 1
        complete_row = connection.execute(
            "SELECT * FROM events WHERE event_id = ?", (complete_event_id,)
        ).fetchone()
        input_manifest = {
            "schema_version": "sprayxr-session-assessment-input/v1",
            "complete_event": {
                "event_id": complete_event_id,
                "sha256": complete_row["body_sha256"],
            },
            "zone_assessments": [
                {
                    "assessment_id": item["assessment_id"],
                    "revision": item["revision"],
                    "input_sha256": item["provenance"]["assessment_input_manifest_sha256"],
                }
                for item in zones
            ],
            "expected_zones": expected_spec,
            "provider_official_summary": official_summary,
            "contract_version": session["contract_version"],
            "criterion_version": CRITERION_VERSION,
            "engine_version": ENGINE_VERSION,
        }
        input_sha256 = sha256_text(canonical_json(input_manifest))
        assessment_id = f"sa-{input_sha256[:32]}"

        source_event_ids = sorted(
            {complete_event_id}
            | {
                event_id
                for item in zones
                for event_id in item["provenance"]["source_event_ids"]
            }
        )
        evidence_by_id: dict[str, dict[str, Any]] = {}
        aggregate_id = f"ev-session-aggregate-{input_sha256[:24]}"
        aggregate_statement = (
            "세션 최신 Zone revision의 유효 도포 픽셀 "
            f"{painted_count}개를 픽셀 수로 가중한 평균은 {observed_mean:g} µm입니다. "
            "Zone별 평균의 단순 평균이 아닙니다."
            if observed_mean is not None
            else "세션 최신 Zone revision에 유효 도포 픽셀이 없어 도포부 평균을 산정하지 않았습니다."
        )
        evidence_by_id[aggregate_id] = {
            "evidence_id": aggregate_id,
            "evidence_type": "quality_measurement",
            "statement": aggregate_statement,
            "value": observed_mean,
            "unit": "um",
            "source_event_ids": source_event_ids,
            "contract_version": session["contract_version"],
            "revision": revision,
        }
        if official_summary is not None and official_summary["avg_thickness_um"] is not None:
            official_id = f"ev-provider-official-{input_sha256[:24]}"
            evidence_by_id[official_id] = {
                "evidence_id": official_id,
                "evidence_type": "quality_measurement",
                "statement": (
                    "공급자가 별도 공식 세션 평균으로 전달한 값은 "
                    f"{official_summary['avg_thickness_um']:g} {official_summary['unit']}입니다. "
                    "Zone 도포부 평균의 재현값이 아닙니다."
                ),
                "value": official_summary["avg_thickness_um"],
                "unit": official_summary["unit"],
                "source_event_ids": [complete_event_id],
                "contract_version": session["contract_version"],
                "revision": revision,
            }
        lineage_id = f"ev-lineage-{input_sha256[:24]}"
        evidence_by_id[lineage_id] = {
            "evidence_id": lineage_id,
            "evidence_type": "lineage",
            "statement": (
                "세션 집계는 각 Zone의 최신 revision만 사용하며, "
                "유효 도포 픽셀 수로 가중한 결과입니다."
            ),
            "value": input_sha256,
            "unit": None,
            "source_event_ids": source_event_ids,
            "contract_version": session["contract_version"],
            "revision": revision,
        }
        causes: list[dict[str, Any]] = []
        actions: list[dict[str, Any]] = []
        for item in zones:
            zone_key = item["zone_key"]
            zone_scope = (
                f"면 {zone_key['face_id']}, map {zone_key['map_num']}, "
                f"Zone {zone_key['zone_id']} 관측: "
            )
            for evidence in item["evidence"]:
                evidence_by_id[evidence["evidence_id"]] = {
                    **evidence,
                    "statement": f"{zone_scope}{evidence['statement']}",
                }
            causes.extend(
                {
                    **cause,
                    "statement": f"{zone_scope}{cause['statement']}",
                }
                for cause in item["coaching"]["causes"]
            )
            actions.extend(
                {
                    **action,
                    "statement": f"{zone_scope}{action['statement']}",
                }
                for action in item["coaching"]["actions"]
            )
        actions.sort(key=lambda item: (item["priority"], item["action_id"]))
        if analysis_status == "available":
            coaching_status = "available"
            coaching_reason = None if actions else "no_supported_corrective_action"
        else:
            coaching_status = "withheld"
            coaching_reason = reason_codes[0] if reason_codes else "session_quality_withheld"
            causes = []
            actions = []
        coaching = {
            "analysis_status": coaching_status,
            "reason_code": coaching_reason,
            "causes": causes,
            "actions": actions,
            "primary_action_id": actions[0]["action_id"] if actions else None,
        }
        provenance = {
            "engine_version": ENGINE_VERSION,
            "criterion_version": CRITERION_VERSION,
            "coaching_rule_version": COACHING_RULE_VERSION,
            "contract_version": session["contract_version"],
            "manifest_id": session["manifest_id"],
            "zone_asset_manifest_sha256": session["manifest_sha256"],
            "label_map_sha256": None,
            "mask_sha256": None,
            "source_event_ids": source_event_ids,
            "assessment_input_manifest_sha256": input_sha256,
            "snapshot_id": None,
            "snapshot_revision": None,
            "captured_at": session["completed_at"],
        }
        evidence = list(evidence_by_id.values())
        return {
            "assessment_id": assessment_id,
            "session_id": session["session_id"],
            "revision": revision,
            "assessment_type": "session",
            "status": analysis_status,
            "zone_key": None,
            "quality": quality,
            "metrics": metrics,
            "coaching": coaching,
            "evidence": evidence,
            "evidence_ids": [item["evidence_id"] for item in evidence],
            "created_at": now,
            "provenance": provenance,
            "zone_assessments": [
                {
                    "assessment_id": item["assessment_id"],
                    "revision": item["revision"],
                    "zone_key": item["zone_key"],
                    "status": item["status"],
                    "quality": item["quality"],
                }
                for item in zones
            ],
            "missing_or_pending_zones": {"missing": missing, "pending": pending},
            "_storage": {
                "snapshot_id": None,
                "snapshot_revision": None,
                "captured_at": session["completed_at"],
                "input_manifest_sha256": input_sha256,
            },
        }

    @staticmethod
    def _insert_assessment(
        connection: sqlite3.Connection, assessment: dict[str, Any]
    ) -> None:
        storage = assessment.pop("_storage")
        zone_key = assessment["zone_key"] or {}
        connection.execute(
            """
            INSERT INTO assessments(
                assessment_id, session_id, revision, assessment_type,
                model_key, face_id, map_num, zone_id, snapshot_id,
                snapshot_revision, captured_at, input_manifest_sha256,
                payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                assessment["assessment_id"],
                assessment["session_id"],
                assessment["revision"],
                assessment["assessment_type"],
                zone_key.get("model_key"),
                zone_key.get("face_id"),
                zone_key.get("map_num"),
                zone_key.get("zone_id"),
                storage["snapshot_id"],
                storage["snapshot_revision"],
                storage["captured_at"],
                storage["input_manifest_sha256"],
                canonical_json(assessment),
                assessment["created_at"],
            ),
        )

    def get_assessment(self, assessment_id: str) -> dict[str, Any]:
        with self.ledger.read() as connection:
            row = connection.execute(
                "SELECT payload_json FROM assessments WHERE assessment_id = ?",
                (assessment_id,),
            ).fetchone()
        if row is None:
            raise NotFoundError("assessment_not_found", "assessment does not exist")
        return json.loads(row["payload_json"])

    def _cumulative_crop_states(
        self, connection: sqlite3.Connection, session: sqlite3.Row
    ) -> list[dict[str, Any]]:
        rows = connection.execute(
            """
            SELECT * FROM events WHERE session_id = ? AND event_type = 'sequence'
            ORDER BY order_index
            """,
            (session["session_id"],),
        ).fetchall()
        grouped: dict[tuple[str, int], list[Surface]] = {}
        source_ids: dict[tuple[str, int], list[str]] = {}
        for row in rows:
            request = EventRequest.model_validate(json.loads(row["body_json"]))
            payload = parse_payload(request)
            assert isinstance(payload, SequencePayload)
            if payload.cumulative_crop is None:
                continue
            spec = payload.cumulative_crop.height_png
            key = (payload.face_id, payload.map_num)
            grouped.setdefault(key, []).append(place_crop(decode_height_png(spec), spec))
            source_ids.setdefault(key, []).append(row["event_id"])
        output: list[dict[str, Any]] = []
        for (face_id, map_num), surfaces in sorted(grouped.items()):
            state = overwrite_cumulative_crops(surfaces)
            output.append(
                {
                    "face_id": face_id,
                    "map_num": map_num,
                    "known_pixel_count": int(state.known.sum()),
                    "raw_sum_um": float(state.raw_um[state.known].sum()),
                    "source_event_ids": source_ids[(face_id, map_num)],
                    "semantics": "absolute_cumulative_overwrite",
                }
            )
        return output

    def get_session(self, session_id: str) -> dict[str, Any]:
        with self._lock, self.ledger.read() as connection:
            session = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if session is None:
                raise NotFoundError("session_not_found", "session does not exist")
            counts = {
                row["event_type"]: int(row["count"])
                for row in connection.execute(
                    """
                    SELECT event_type, COUNT(*) AS count FROM events
                    WHERE session_id = ? GROUP BY event_type
                    """,
                    (session_id,),
                ).fetchall()
            }
            latest_zones = self._latest_zone_assessments(connection, session_id)
            latest_assessment = None
            if session["latest_assessment_id"] is not None:
                row = connection.execute(
                    "SELECT payload_json FROM assessments WHERE assessment_id = ?",
                    (session["latest_assessment_id"],),
                ).fetchone()
                latest_assessment = json.loads(row["payload_json"])
            cumulative_states = self._cumulative_crop_states(connection, session)
            sequence_rows = connection.execute(
                """
                SELECT body_json FROM events
                WHERE session_id = ? AND event_type = 'sequence' ORDER BY order_index
                """,
                (session_id,),
            ).fetchall()
            last_spray_at: datetime | None = None
            last_observation_at: datetime | None = None
            for row in sequence_rows:
                request = EventRequest.model_validate(json.loads(row["body_json"]))
                sequence = parse_payload(request)
                assert isinstance(sequence, SequencePayload)
                anchor = (
                    parse_utc(sequence.sequence_started_at, field_name="sequence_started_at")
                    if sequence.sequence_started_at is not None
                    else None
                )
                for frame in sequence.frames:
                    frame_time = (
                        parse_utc(frame.timestamp, field_name="frame.timestamp")
                        if frame.timestamp is not None
                        else anchor + timedelta(seconds=frame.relative_time_s)  # type: ignore[arg-type,operator]
                    )
                    if last_observation_at is None or frame_time > last_observation_at:
                        last_observation_at = frame_time
                    if frame.spray_on and (
                        last_spray_at is None or frame_time > last_spray_at
                    ):
                        last_spray_at = frame_time

        zone_revision_index = [
            {
                "zone_key": item["zone_key"],
                "assessment_id": item["assessment_id"],
                "revision": item["revision"],
                "status": item["status"],
            }
            for item in latest_zones
        ]
        return {
            "session_id": session_id,
            "evaluation_mode": session["evaluation_mode"],
            "contract_version": session["contract_version"],
            "started_at": session["started_at"],
            "ended_at": session["ended_at"],
            "completed_at": session["completed_at"],
            "last_event_id": session["last_event_id"],
            "last_event_type": session["last_event_type"],
            "last_event_order": int(session["last_event_order"]),
            "latest_assessment_id": session["latest_assessment_id"],
            "latest_assessment_revision": session["latest_assessment_revision"],
            "event_counts": counts,
            "zone_revision_index": zone_revision_index,
            "latest_assessment": latest_assessment,
            "cumulative_crop_states": cumulative_states,
            "idle_policy": {
                "automatic_idle_completion": False,
                "requires_explicit_no_spray_tick": True,
                "last_spray_at": utc_iso(last_spray_at) if last_spray_at else None,
                "last_observation_at": (
                    utc_iso(last_observation_at) if last_observation_at else None
                ),
            },
            "provider_official_summary": (
                json.loads(session["provider_summary_json"])
                if session["provider_summary_json"] is not None
                else None
            ),
        }

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "corrected-zone-engine",
            "engine_version": ENGINE_VERSION,
            "criterion_version": CRITERION_VERSION,
            "coaching_rule_version": COACHING_RULE_VERSION,
            "contract_version": CONTRACT_VERSION,
            "zone_manifest": {
                "manifest_id": self.assets.manifest_id,
                "sha256": self.assets.manifest_sha256,
            },
            "ledger": self.ledger.health(),
        }
