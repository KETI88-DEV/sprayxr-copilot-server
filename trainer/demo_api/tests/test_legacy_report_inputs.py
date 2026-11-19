"""Legacy input provenance is a read-time finding, never a report state transition."""
import json
import sqlite3

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.v1_router import create_v1_router
from trainer.demo_api.v1_store import V1Store


def client_for(store):
    app = FastAPI()
    app.include_router(create_v1_router(store=store, sequences=SequenceStore(":memory:")))
    return TestClient(app, raise_server_exceptions=False)


def migrated_store(tmp_path, *, status="pending", payload=None):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE completions (report_id TEXT PRIMARY KEY, session_id TEXT, "
                           "report_status TEXT, requested_at TEXT, payload TEXT)")
        connection.execute("INSERT INTO completions VALUES (?, ?, ?, ?, ?)",
                           ("RPT-old", "S-old", status, "2026-10-01T00:00:00Z", json.dumps(payload or {})))
    return V1Store(str(path))


@pytest.mark.parametrize("status", ["pending", "PENDING"])
def test_legacy_pending_explains_unreferenced_inputs_without_changing_database_status(tmp_path, status):
    store = migrated_store(tmp_path, status=status, payload={"telemetry_csv_bytes": 123})
    client = client_for(store)

    response = client.get("/api/v1/reports/RPT-old")
    poll = client.get("/api/v1/sessions/S-old/reports/RPT-old")

    assert response.status_code == poll.status_code == 200
    report = response.json()["data"]["report"]
    assert report["report_status"] == poll.json()["data"]["report_status"] == "pending"
    for area in ("coating_report", "pose_report", "sequence_report"):
        assert report[area]["analysis_status"] == "unavailable"
        assert report[area]["reason_code"] == "legacy_report_inputs_unavailable"
        assert "입력 자료 참조 확인 불가" in report[area]["guideline"]
        assert "재업로드 권장" in report[area]["guideline"]
    assert report["coating_report"]["distribution"] is None
    assert store.get_completion("RPT-old").report_status == status
    assert store.get_completion("RPT-old").payload == {"telemetry_csv_bytes": 123}


def test_new_queued_pending_without_input_references_is_not_called_legacy():
    store = V1Store(":memory:")
    # Even an old requested_at cannot override observed modern queue metadata.
    row = store.create_completion(session_id="S-new", requested_at="2020-01-01T00:00:00Z", payload={})

    response = client_for(store).get(f"/api/v1/reports/{row.report_id}")

    assert response.status_code == 200
    report = response.json()["data"]["report"]
    assert report["report_status"] == "pending"
    assert report["coating_report"]["reason_code"] == "report_pending"
    assert "재업로드" not in report["coating_report"]["guideline"]


@pytest.mark.parametrize("marker", ["artifacts", "analysis", "analysis_requested"])
def test_payload_evidence_of_a_durable_input_or_analysis_request_prevents_legacy_inference(tmp_path, marker):
    store = migrated_store(tmp_path, payload={marker: {}})

    response = client_for(store).get("/api/v1/reports/RPT-old")

    assert response.status_code == 200
    assert response.json()["data"]["report"]["coating_report"]["reason_code"] == "report_pending"


@pytest.mark.parametrize("marker", ["fingerprint", "processing_started_at", "processing_updated_at",
                                   "processing_finished_at", "processing_reason"])
def test_any_recorded_queue_or_processing_evidence_prevents_legacy_inference(tmp_path, marker):
    store = migrated_store(tmp_path)
    with store._conn:
        store._conn.execute(f"UPDATE completions SET {marker}=? WHERE report_id='RPT-old'", ("observed",))

    response = client_for(store).get("/api/v1/reports/RPT-old")

    assert response.status_code == 200
    assert response.json()["data"]["report"]["coating_report"]["reason_code"] == "report_pending"
