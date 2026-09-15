"""GET /api/ready: live readiness of the model server, RAG index and (informational) MCP."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient

from trainer.demo_api import chat_service as chat_service_mod
from trainer.demo_api.app import create_app


def _openai_config() -> dict[str, Any]:
    return {
        "llm": {
            "backend": "openai",
            "base_url": "http://vllm.test:8000/v1",
            "served_model_name": "shipcoat-v6-1-fp8",
        },
        "generation": {"temperature": 0.3, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.15},
    }


class _Index:
    def __init__(self, ntotal: int) -> None:
        self.ntotal = ntotal


class _LoadedPipeline:
    def __init__(self, config: dict[str, Any], ntotal: int = 10829) -> None:
        self.config = config
        self.index = _Index(ntotal)


class _Upstreams:
    def __init__(self, *, health: int = 200, models: tuple[str, ...] = ("shipcoat-v6-1-fp8",), mcp_up: bool = False) -> None:
        self.health = health
        self.models = models
        self.mcp_up = mcp_up
        self.hosts: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.hosts.append(request.url.host)
        if request.url.host == "vllm.test":
            if request.url.path == "/health":
                return httpx.Response(self.health)
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"object": "list", "data": [{"id": m} for m in self.models]})
        if request.url.host == "mcp.test" and request.url.path == "/health":
            if self.mcp_up:
                return httpx.Response(200, json={"status": "ok", "servers": 7})
            raise httpx.ConnectError("refused")
        return httpx.Response(404)


def _client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    pipeline: object | None,
    upstreams: _Upstreams,
    config: dict[str, Any] | None = None,
    live: bool = True,
) -> TestClient:
    config_path = tmp_path / "rag_config.yaml"
    config_path.write_text(yaml.safe_dump(config or _openai_config()), encoding="utf-8")
    monkeypatch.setenv("DEMO_LIVE_MODE", "on" if live else "off")
    monkeypatch.setenv("CONFIG_PATH", str(config_path))
    monkeypatch.setenv("MCP_BASE_URL", "http://mcp.test:8100")
    monkeypatch.delenv("LIVE_EAGER_INIT", raising=False)
    monkeypatch.setattr(chat_service_mod, "_pipeline_singleton", pipeline)
    return TestClient(create_app(probe_transport=httpx.MockTransport(upstreams)))


def test_ready_when_model_is_served_and_index_is_loaded_even_if_mcp_is_down(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    upstreams = _Upstreams(mcp_up=False)
    client = _client(monkeypatch, tmp_path, pipeline=_LoadedPipeline(_openai_config()), upstreams=upstreams)

    response = client.get("/api/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert body["mode"] == "live"
    assert body["checks"]["llm"] == {"ok": True, "backend": "openai", "health": True, "model_served": True}
    assert body["checks"]["pipeline"] == {"ok": True, "loaded": True, "rag_vectors": 10829}
    assert body["checks"]["mcp"] == {"ok": False, "required": False}
    assert "mcp.test" in upstreams.hosts


def test_not_ready_when_served_model_name_is_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    upstreams = _Upstreams(models=("Qwen/Qwen3.6-27B-FP8",))
    client = _client(monkeypatch, tmp_path, pipeline=_LoadedPipeline(_openai_config()), upstreams=upstreams)

    response = client.get("/api/ready")

    assert response.status_code == 503
    assert response.json()["ready"] is False
    assert response.json()["checks"]["llm"]["model_served"] is False


def test_not_ready_when_vllm_health_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client = _client(
        monkeypatch, tmp_path, pipeline=_LoadedPipeline(_openai_config()), upstreams=_Upstreams(health=503)
    )

    response = client.get("/api/ready")

    assert response.status_code == 503
    assert response.json()["checks"]["llm"]["health"] is False


def test_not_ready_before_pipeline_is_loaded_but_model_server_is_still_checked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    upstreams = _Upstreams()
    client = _client(monkeypatch, tmp_path, pipeline=None, upstreams=upstreams)

    response = client.get("/api/ready")

    assert response.status_code == 503
    checks = response.json()["checks"]
    assert checks["pipeline"] == {"ok": False, "loaded": False, "rag_vectors": None}
    assert checks["llm"]["ok"] is True
    assert "vllm.test" in upstreams.hosts


def test_not_ready_with_an_empty_rag_index(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client = _client(
        monkeypatch, tmp_path, pipeline=_LoadedPipeline(_openai_config(), ntotal=0), upstreams=_Upstreams()
    )

    response = client.get("/api/ready")

    assert response.status_code == 503
    assert response.json()["checks"]["pipeline"]["rag_vectors"] == 0


def test_hf_backend_readiness_does_not_contact_a_model_server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = {"llm": {"backend": "hf", "model_path": "/models/Qwen3.6-27B"}}
    upstreams = _Upstreams()
    client = _client(monkeypatch, tmp_path, pipeline=_LoadedPipeline(config), upstreams=upstreams, config=config)

    response = client.get("/api/ready")

    assert response.status_code == 200
    assert response.json()["checks"]["llm"] == {"ok": True, "backend": "hf"}
    assert "vllm.test" not in upstreams.hosts


def test_demo_mode_is_ready_without_checks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    upstreams = _Upstreams()
    client = _client(monkeypatch, tmp_path, pipeline=None, upstreams=upstreams, live=False)

    response = client.get("/api/ready")

    assert response.status_code == 200
    assert response.json()["ready"] is True
    assert response.json()["mode"] == "demo"
    assert upstreams.hosts == []


def test_health_contract_is_unchanged_when_not_ready(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client = _client(monkeypatch, tmp_path, pipeline=None, upstreams=_Upstreams(health=503))

    assert client.get("/api/ready").status_code == 503
    health = client.get("/api/health")
    assert health.status_code == 200
    assert set(health.json()) == {"status", "servers", "timestamp"}


def test_live_eager_init_builds_the_pipeline_once_at_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[object] = []

    def _construct() -> object:
        pipeline = _LoadedPipeline(_openai_config())
        built.append(pipeline)
        return pipeline

    monkeypatch.setenv("DEMO_LIVE_MODE", "on")
    monkeypatch.setenv("LIVE_EAGER_INIT", "1")
    monkeypatch.setattr(chat_service_mod, "_pipeline_singleton", None)
    monkeypatch.setattr(chat_service_mod, "_construct_default_pipeline", _construct)
    app = create_app()

    with TestClient(app):
        thread = app.state.eager_init_thread
        assert thread is not None
        thread.join(timeout=5)

    assert len(built) == 1
    assert chat_service_mod._pipeline_singleton is built[0]


def test_without_eager_init_startup_does_not_build_the_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[object] = []
    monkeypatch.setenv("DEMO_LIVE_MODE", "on")
    monkeypatch.delenv("LIVE_EAGER_INIT", raising=False)
    monkeypatch.setattr(chat_service_mod, "_pipeline_singleton", None)
    monkeypatch.setattr(chat_service_mod, "_construct_default_pipeline", lambda: built.append(object()))
    app = create_app()

    with TestClient(app):
        assert app.state.eager_init_thread is None

    assert built == []


def test_eager_init_failure_is_logged_and_does_not_crash_startup(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def _boom() -> object:
        raise RuntimeError("model load failed")

    monkeypatch.setenv("DEMO_LIVE_MODE", "on")
    monkeypatch.setenv("LIVE_EAGER_INIT", "1")
    monkeypatch.setattr(chat_service_mod, "_pipeline_singleton", None)
    monkeypatch.setattr(chat_service_mod, "_construct_default_pipeline", _boom)
    app = create_app()

    with TestClient(app) as client:
        app.state.eager_init_thread.join(timeout=5)
        assert client.get("/api/health").status_code == 200

    assert chat_service_mod._pipeline_singleton is None
    assert "model load failed" in caplog.text
