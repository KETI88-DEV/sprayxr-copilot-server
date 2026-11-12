from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from trainer.zone_engine.assets import ZoneAssetCatalog
from trainer.zone_engine.engine import CorrectedZoneEngine
from trainer.zone_engine.service import create_app

from .conftest import event, start_event
from .test_engine_ledger import _observation_payload


def _client(tmp_path: Path, catalog: ZoneAssetCatalog) -> TestClient:
    engine = CorrectedZoneEngine(
        tmp_path / "service.db",
        assets=catalog,
        clock=lambda: datetime(2026, 11, 6, 1, 0, tzinfo=timezone.utc),
    )
    return TestClient(create_app(engine))


def test_http_health_event_assessment_and_conflict_contract(
    tmp_path: Path, asset_catalog: ZoneAssetCatalog
) -> None:
    client = _client(tmp_path, asset_catalog)
    health = client.get("/internal/v1/health")
    assert health.status_code == 200
    assert health.json()["ledger"]["persistent"] is True

    start = start_event(asset_catalog, expected_zone_ids=["1"])
    accepted = client.post(
        "/internal/v1/events", json=start.model_dump(mode="json")
    )
    assert accepted.status_code == 200
    assert accepted.json()["session_revision"] == 0
    duplicate = client.post(
        "/internal/v1/events", json=start.model_dump(mode="json")
    )
    assert duplicate.status_code == 200
    assert duplicate.json()["status"] == "duplicate"

    changed = start.model_dump(mode="json")
    changed["payload"]["user_id"] = "another-user"
    conflict = client.post("/internal/v1/events", json=changed)
    assert conflict.status_code == 409
    assert conflict.json()["error"]["reason_code"] == "event_id_payload_conflict"

    observation = event(
        "event-zone", "zone_observation", _observation_payload()
    )
    result = client.post(
        "/internal/v1/events", json=observation.model_dump(mode="json")
    )
    assessment = client.get(
        f"/internal/v1/assessments/{result.json()['assessment_id']}"
    )
    assert assessment.status_code == 200
    assert "_storage" not in assessment.json()
    session = client.get("/internal/v1/sessions/session-1")
    assert session.status_code == 200
    assert session.json()["latest_assessment_revision"] == 1


def test_http_rejects_oversized_body_before_json_parsing(
    tmp_path: Path, asset_catalog: ZoneAssetCatalog, monkeypatch
) -> None:
    monkeypatch.setenv("ZONE_ENGINE_MAX_BODY_BYTES", "32")
    client = _client(tmp_path, asset_catalog)
    response = client.post(
        "/internal/v1/events",
        content=b"{" + b"x" * 100 + b"}",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 413
    assert response.json()["error"]["reason_code"] == "request_body_too_large"
