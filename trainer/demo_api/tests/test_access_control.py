import base64
import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.access_control import AccessControlMiddleware, OperatorCredentials


def _client(tmp_path, *, api_key=None):
    credentials = OperatorCredentials(tmp_path / "operator_credentials.json")
    app = FastAPI()
    app.add_middleware(AccessControlMiddleware, credentials=credentials, protect_admin=True, api_key=api_key)
    @app.get("/admin")
    def admin():
        return {"readonly": True}
    @app.get("/api/monitor/events")
    def events():
        return {"events": []}
    @app.get("/api/v1/users")
    def users():
        return {"users": []}
    @app.get("/api/ready")
    def ready():
        return {"ready": True}
    return TestClient(app), credentials


def test_monitor_requires_auth_and_persistent_credentials_work_after_restart(tmp_path):
    client, credentials = _client(tmp_path)
    assert client.get("/api/monitor/events").status_code == 401
    credentials.ensure()
    saved = json.loads(credentials.path.read_text())
    assert client.get("/api/monitor/events", auth=(saved["username"], saved["password"])).status_code == 200
    another, _ = _client(tmp_path)
    assert another.get("/api/monitor/events", auth=(saved["username"], saved["password"])).status_code == 200
    assert client.get("/api/monitor/events", auth=("admin", "bad")).status_code == 401


def test_optional_tsb_key_does_not_change_json_or_block_health_checks(tmp_path):
    client, _ = _client(tmp_path, api_key="test-only-key")
    assert client.get("/api/v1/users").status_code == 401
    assert client.get("/api/v1/users", headers={"X-API-Key": "test-only-key"}).status_code == 200
    assert client.get("/api/ready").status_code == 200
    assert client.get("/admin").status_code == 200
    assert client.get("/api/monitor/events", headers={"X-API-Key": "test-only-key"}).status_code == 401


def test_legacy_simulator_remains_connected_until_api_key_is_configured(tmp_path):
    client, _ = _client(tmp_path)
    assert client.get("/api/v1/users").status_code == 200
