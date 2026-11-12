import json
import os
import sqlite3
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.app import create_app
from trainer.demo_api.chat_service import ChatService
from trainer.demo_api.engine_contracts import CONTRACT_VERSION, EVALUATION_MODE


API_KEY = "test-only-key"


def _offline_transport() -> httpx.MockTransport:
    return httpx.MockTransport(
        lambda request: httpx.Response(503, request=request)
    )


def _isolate_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name in (
        "COPILOT_API_KEY",
        "CORRECTED_DB_PATH",
        "CORRECTED_ENGINE_URL",
        "CORRECTED_EXPLANATION_URL",
        "CORRECTED_ENGINE_TIMEOUT_S",
        "CORRECTED_EXPLANATION_TIMEOUT_S",
        "CORRECTED_SOURCE_INSTANCE_ID",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DEMO_LIVE_MODE", "off")
    monkeypatch.setenv("COPILOT_PROTECT_ADMIN", "0")
    monkeypatch.setenv("SEQUENCE_DB_PATH", str(tmp_path / "sequences.db"))


def _create_isolated_app() -> FastAPI:
    return create_app(
        ChatService(),
        probe_transport=_offline_transport(),
        corrected_engine_transport=_offline_transport(),
        corrected_explanation_transport=_offline_transport(),
    )


def _configure_corrected_services(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CORRECTED_ENGINE_URL", "http://engine.test")
    monkeypatch.setenv("CORRECTED_EXPLANATION_URL", "http://explanation.test")
    monkeypatch.setenv("CORRECTED_DB_PATH", str(tmp_path / "corrected.db"))
    monkeypatch.setenv("CORRECTED_SOURCE_INSTANCE_ID", "pr-test-api")


def _start_body() -> dict[str, object]:
    return {
        "event_id": "evt-private-start",
        "session_id": "private-session",
        "contract_version": CONTRACT_VERSION,
        "user_id": "private-user-marker",
        "scenario_id": "scenario-1",
        "started_at": "2026-11-11T00:00:00Z",
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
            "manifest_id": "zone-manifest-1",
            "sha256": "a" * 64,
            "model_key": "geometry-1",
            "map_nums": [1],
        },
        "expected_zones": [
            {"face_id": "face-1", "map_num": 1, "zone_ids": ["1"]}
        ],
    }


def _sequence_body() -> dict[str, object]:
    return {
        "event_id": "evt-private-sequence",
        "session_id": "private-session",
        "contract_version": CONTRACT_VERSION,
        "sequence_id": "private-sequence",
        "face_id": "face-1",
        "map_num": 1,
        "round_id": "round-1",
        "captured_at": "2026-11-11T00:00:01Z",
        "sequence_started_at": "2026-11-11T00:00:00Z",
        "frames": [
            {
                "relative_time_s": 0.5,
                "spray_on": True,
                "speed_mm_s": 400.0,
                "distance_mm": 300.0,
                "raytexmapnum": 1,
                "raytexuv": {"x": 10.0, "y": 20.0},
            }
        ],
        "cumulative_crop": None,
    }


def test_default_factory_omits_v2_and_preserves_v1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_environment(monkeypatch, tmp_path)

    app = _create_isolated_app()
    paths = app.openapi()["paths"]

    assert "/api/v1/sessions/start" in paths
    assert not any(path.startswith("/api/v2/") for path in paths)
    assert app.state.corrected_store is None
    assert app.state.corrected_worker is None


def test_factory_requires_explanation_service_with_corrected_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("CORRECTED_ENGINE_URL", "http://engine.test")

    with pytest.raises(RuntimeError, match="CORRECTED_EXPLANATION_URL is required"):
        _create_isolated_app()


def test_factory_requires_dedicated_corrected_database_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_environment(monkeypatch, tmp_path)
    _configure_corrected_services(monkeypatch, tmp_path)
    monkeypatch.delenv("CORRECTED_DB_PATH")

    with pytest.raises(RuntimeError, match="CORRECTED_DB_PATH"):
        _create_isolated_app()


@pytest.mark.parametrize("source_instance_id", [None, "   "])
def test_factory_requires_nonblank_corrected_source_instance_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source_instance_id: str | None,
) -> None:
    _isolate_environment(monkeypatch, tmp_path)
    _configure_corrected_services(monkeypatch, tmp_path)
    if source_instance_id is None:
        monkeypatch.delenv("CORRECTED_SOURCE_INSTANCE_ID")
    else:
        monkeypatch.setenv("CORRECTED_SOURCE_INSTANCE_ID", source_instance_id)

    with pytest.raises(RuntimeError, match="CORRECTED_SOURCE_INSTANCE_ID"):
        _create_isolated_app()


def test_factory_rejects_legacy_database_as_corrected_database(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_environment(monkeypatch, tmp_path)
    _configure_corrected_services(monkeypatch, tmp_path)
    monkeypatch.setenv("CORRECTED_DB_PATH", str(tmp_path / "sequences.db"))

    with pytest.raises(RuntimeError, match="CORRECTED_DB_PATH"):
        _create_isolated_app()


def test_factory_rejects_hardlink_alias_of_legacy_database(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_environment(monkeypatch, tmp_path)
    legacy_path = tmp_path / "sequences.db"
    sqlite3.connect(legacy_path).close()
    alias_path = tmp_path / "corrected-hardlink.db"
    os.link(legacy_path, alias_path)
    _configure_corrected_services(monkeypatch, tmp_path)
    monkeypatch.setenv("CORRECTED_DB_PATH", str(alias_path))

    with pytest.raises(RuntimeError, match="CORRECTED_DB_PATH"):
        _create_isolated_app()


def test_configured_factory_exposes_v2_with_matching_openapi_security(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_environment(monkeypatch, tmp_path)
    _configure_corrected_services(monkeypatch, tmp_path)
    monkeypatch.setenv("COPILOT_API_KEY", API_KEY)

    app = _create_isolated_app()
    with TestClient(app):
        schema = app.openapi()
        paths = schema["paths"]

        assert "/api/v2/sessions/start" in paths
        assert "/api/v2/sessions/{session_id}/status" in paths
        expected_security = [{"TSBApiKey": []}]
        assert paths["/api/v1/sessions/start"]["post"]["security"] == expected_security
        assert paths["/api/v2/sessions/start"]["post"]["security"] == expected_security
        assert (
            paths["/api/v2/sessions/{session_id}/status"]["get"]["security"]
            == expected_security
        )
        assert schema["components"]["securitySchemes"]["TSBApiKey"] == {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": (
                "서버에서 API 키를 활성화한 경우 사용합니다. "
                "JSON 본문은 변경하지 않습니다."
            ),
        }


def test_configured_v2_http_requires_key_before_unknown_session_lookup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_environment(monkeypatch, tmp_path)
    _configure_corrected_services(monkeypatch, tmp_path)
    monkeypatch.setenv("COPILOT_API_KEY", API_KEY)

    with TestClient(_create_isolated_app()) as client:
        path = "/api/v2/sessions/missing-session/status"
        assert client.get(path).status_code == 401
        assert client.get(path, headers={"X-API-Key": "wrong-key"}).status_code == 401
        response = client.get(path, headers={"X-API-Key": API_KEY})

    assert response.status_code == 404
    assert response.json()["detail"] == "corrected session not found"


def test_accepted_v2_start_is_persisted_only_in_private_database(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_environment(monkeypatch, tmp_path)
    _configure_corrected_services(monkeypatch, tmp_path)
    monkeypatch.setenv("COPILOT_API_KEY", API_KEY)
    body = _start_body()

    app = _create_isolated_app()
    with TestClient(app) as client:
        response = client.post(
            "/api/v2/sessions/start",
            json=body,
            headers={"X-API-Key": API_KEY},
        )
        sequence_response = client.post(
            "/api/v2/telemetry/sequence",
            json=_sequence_body(),
            headers={"X-API-Key": API_KEY},
        )
        public_detail = client.get(
            "/admin/api/record",
            params={"table": "corrected_source_events", "rowid": 1},
        )
        assert app.state.corrected_store.path == (tmp_path / "corrected.db").resolve()

    assert response.status_code == 202
    assert response.json()["status"] == "accepted"
    assert sequence_response.status_code == 202
    assert sequence_response.json()["status"] == "accepted"
    assert public_detail.status_code == 404

    corrected_path = tmp_path / "corrected.db"
    with sqlite3.connect(corrected_path) as connection:
        private_tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        stored_body = connection.execute(
            "SELECT request_body FROM corrected_source_events WHERE event_id=?",
            (body["event_id"],),
        ).fetchone()
        private_sequence_count = connection.execute(
            "SELECT COUNT(*) FROM sequences WHERE session_key=?",
            (f"{EVALUATION_MODE}:{body['session_id']}",),
        ).fetchone()[0]
    assert "corrected_source_events" in private_tables
    assert stored_body is not None
    assert json.loads(stored_body[0]) == body
    assert private_sequence_count == 1

    legacy_path = tmp_path / "sequences.db"
    with sqlite3.connect(legacy_path) as connection:
        legacy_tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        mirrored_sequences = connection.execute(
            "SELECT COUNT(*) FROM sequences"
        ).fetchone()[0]
    assert not any(name.startswith("corrected_") for name in legacy_tables)
    assert mirrored_sequences == 0
