from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from trainer.demo_api.engine_contracts import (
    CONTRACT_VERSION,
    EVALUATION_MODE,
    EngineEventAck,
    SequenceRequest,
    SessionCompleteRequest,
    SessionEndRequest,
    SessionStartRequest,
    canonical_sha256,
    engine_event_from_request,
)
from trainer.demo_api.engine_store import (
    CorrectedStore,
    CorrectedStoreConflict,
    CorrectedStoreError,
    CorrectedStoreOrderError,
)


SOURCE = "api-instance-1"


def _start(event_id: str = "evt-start") -> SessionStartRequest:
    return SessionStartRequest.model_validate(
        {
            "event_id": event_id,
            "session_id": "session-1",
            "contract_version": CONTRACT_VERSION,
            "user_id": "user-1",
            "scenario_id": "scenario-1",
            "started_at": "2026-11-06T00:00:00Z",
            "evaluation_mode": EVALUATION_MODE,
            "target_thickness_um": 100.0,
            "target_source": "simulator_session_target",
            "model_mapping": {
                "provider_model_key": "provider-model",
                "work_id": "work-1",
                "session_family_id": "family-1",
                "round_index": 1,
                "round_id": "round-1",
                "geometry_model_key": "geometry-1",
            },
            "zone_manifest": {
                "manifest_id": "tsb-zones-20261015-v1",
                "sha256": "a" * 64,
                "model_key": "geometry-1",
                "map_nums": [1],
            },
        }
    )


def _sequence(
    sequence_id: str,
    captured_at: str,
    *,
    event_id: str | None = None,
    speed: float = 400.0,
) -> SequenceRequest:
    return SequenceRequest.model_validate(
        {
            "event_id": event_id or f"evt-{sequence_id}",
            "session_id": "session-1",
            "contract_version": CONTRACT_VERSION,
            "sequence_id": sequence_id,
            "face_id": "face-1",
            "map_num": 1,
            "round_id": "round-1",
            "captured_at": captured_at,
            "frames": [
                {
                    "timestamp": captured_at,
                    "spray_on": True,
                    "speed_mm_s": speed,
                    "distance_mm": 250.0,
                    "zone_id": "1",
                }
            ],
        }
    )


def _accept(store: CorrectedStore, body: object):
    return store.accept(body, engine_event_from_request(body, source_instance_id=SOURCE))  # type: ignore[arg-type]


def test_corrected_store_refuses_memory() -> None:
    with pytest.raises(CorrectedStoreError, match="persistent"):
        CorrectedStore(":memory:")


def test_source_and_outbox_are_atomic_and_sequence_uses_v2_namespace(tmp_path: Path) -> None:
    path = tmp_path / "state" / "sequences.db"
    store = CorrectedStore(str(path))

    start = _accept(store, _start())
    sequence = _accept(store, _sequence("SEQ-001", "2026-11-06T00:00:01Z"))

    assert start.source_cursor == 1
    assert sequence.source_cursor == 2
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM corrected_source_events").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM corrected_engine_outbox").fetchone()[0] == 2
        row = conn.execute(
            "SELECT session_key,sequence_id FROM sequences"
        ).fetchone()
        assert row == ("corrected_zone_v2:session-1", "SEQ-001")
    store.close()


def test_same_natural_sequence_is_duplicate_but_changed_body_conflicts(tmp_path: Path) -> None:
    store = CorrectedStore(str(tmp_path / "sequences.db"))
    _accept(store, _start())
    original = _sequence("SEQ-001", "2026-11-06T00:00:01Z", event_id="evt-a")
    _accept(store, original)

    retry = _sequence("SEQ-001", "2026-11-06T00:00:01Z", event_id="evt-b")
    receipt = _accept(store, retry)
    assert receipt.duplicate is True
    assert receipt.event_id == "evt-a"

    changed = _sequence(
        "SEQ-001", "2026-11-06T00:00:01Z", event_id="evt-c", speed=999.0
    )
    with pytest.raises(CorrectedStoreConflict, match="different body"):
        _accept(store, changed)
    store.close()


def test_order_late_data_and_complete_boundaries(tmp_path: Path) -> None:
    store = CorrectedStore(str(tmp_path / "sequences.db"))
    _accept(store, _start())
    _accept(store, _sequence("SEQ-002", "2026-11-06T00:00:04Z"))
    with pytest.raises(CorrectedStoreOrderError, match="out of order"):
        _accept(store, _sequence("SEQ-001", "2026-11-06T00:00:03Z"))

    end = SessionEndRequest(
        event_id="evt-end",
        session_id="session-1",
        contract_version=CONTRACT_VERSION,
        ended_at="2026-11-06T00:00:10Z",
    )
    _accept(store, end)
    # Delayed transport is allowed after end when the source capture preceded end.
    _accept(store, _sequence("SEQ-003", "2026-11-06T00:00:02Z"))
    with pytest.raises(CorrectedStoreOrderError, match="after session end"):
        _accept(store, _sequence("SEQ-004", "2026-11-06T00:00:11Z"))

    complete = SessionCompleteRequest.model_validate(
        {
            "event_id": "evt-complete",
            "session_id": "session-1",
            "contract_version": CONTRACT_VERSION,
            "completed_at": "2026-11-06T00:00:12Z",
            "provider_official_summary": {
                "avg_thickness_um": 123.4,
                "source": "simulator",
                "unit": "um",
                "progress_value": 100.0,
                "progress_unit": "percent_0_100",
            },
        }
    )
    _accept(store, complete)
    with pytest.raises(CorrectedStoreOrderError, match="already complete"):
        _accept(store, _sequence("SEQ-005", "2026-11-06T00:00:09Z"))
    store.close()


def test_processing_claim_is_recovered_after_restart(tmp_path: Path) -> None:
    path = tmp_path / "sequences.db"
    store = CorrectedStore(str(path))
    _accept(store, _start())
    claimed = store.claim_engine_event()
    assert claimed is not None and claimed.attempts == 1
    store.close()

    reopened = CorrectedStore(str(path))
    claimed_again = reopened.claim_engine_event()
    assert claimed_again is not None
    assert claimed_again.event.event_id == "evt-start"
    assert claimed_again.attempts == 2
    reopened.close()


def test_store_defensively_rejects_wrong_acknowledgement_hash(tmp_path: Path) -> None:
    store = CorrectedStore(str(tmp_path / "sequences.db"))
    _accept(store, _start())
    claimed = store.claim_engine_event()
    assert claimed is not None
    ack = EngineEventAck(
        status="accepted",
        event_id=claimed.event.event_id,
        event_type=claimed.event.event_type,
        session_id=claimed.event.session_id,
        payload_hash="0" * 64,
        assessment_id=None,
        session_revision=0,
        processing_status="processed",
        evidence_ids=[],
    )
    with pytest.raises(CorrectedStoreConflict, match="payload hash mismatch"):
        store.commit_engine_success(
            claimed.event.event_id,
            ack=ack,
            assessment=None,
            engine_session={"session_id": "session-1"},
        )
    store.close()


def test_store_requires_processed_acknowledgement(tmp_path: Path) -> None:
    store = CorrectedStore(str(tmp_path / "sequences.db"))
    _accept(store, _start())
    claimed = store.claim_engine_event()
    assert claimed is not None
    ack = EngineEventAck(
        status="accepted",
        event_id=claimed.event.event_id,
        event_type=claimed.event.event_type,
        session_id=claimed.event.session_id,
        payload_hash=canonical_sha256(claimed.event.model_dump(mode="json")),
        assessment_id=None,
        session_revision=0,
        processing_status="queued",
        evidence_ids=[],
    )
    with pytest.raises(CorrectedStoreConflict, match="not processed"):
        store.commit_engine_success(
            claimed.event.event_id,
            ack=ack,
            assessment=None,
            engine_session={"session_id": "session-1"},
        )
    store.close()
