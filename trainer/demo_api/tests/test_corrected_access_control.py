from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.access_control import AccessControlMiddleware, OperatorCredentials


API_KEY = "test-only-key"


def _client(tmp_path) -> TestClient:
    app = FastAPI()
    app.add_middleware(
        AccessControlMiddleware,
        credentials=OperatorCredentials(tmp_path / "operator_credentials.json"),
        protect_admin=True,
        api_key=API_KEY,
    )

    @app.get("/api/v1/users")
    def users() -> dict[str, list[object]]:
        return {"users": []}

    @app.post("/api/v2/sessions/start")
    def corrected_session_start() -> dict[str, str]:
        return {"status": "accepted"}

    @app.get("/api/ready")
    def ready() -> dict[str, bool]:
        return {"ready": True}

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return TestClient(app)


def test_v2_requires_configured_api_key(tmp_path) -> None:
    client = _client(tmp_path)

    assert client.post("/api/v2/sessions/start").status_code == 401
    assert (
        client.post(
            "/api/v2/sessions/start", headers={"X-API-Key": "wrong-key"}
        ).status_code
        == 401
    )
    assert (
        client.post(
            "/api/v2/sessions/start", headers={"X-API-Key": API_KEY}
        ).status_code
        == 200
    )


def test_v2_accepts_bearer_api_key(tmp_path) -> None:
    client = _client(tmp_path)

    response = client.post(
        "/api/v2/sessions/start", headers={"Authorization": f"Bearer {API_KEY}"}
    )

    assert response.status_code == 200


def test_v1_api_key_behavior_is_preserved(tmp_path) -> None:
    client = _client(tmp_path)

    assert client.get("/api/v1/users").status_code == 401
    assert (
        client.get("/api/v1/users", headers={"X-API-Key": API_KEY}).status_code
        == 200
    )


def test_health_and_preflight_options_remain_open(tmp_path) -> None:
    client = _client(tmp_path)

    assert client.get("/api/ready").status_code == 200
    assert client.get("/health").status_code == 200
    assert client.options("/api/v2/sessions/start").status_code != 401
