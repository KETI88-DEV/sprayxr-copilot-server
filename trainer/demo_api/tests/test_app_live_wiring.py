from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from trainer.demo_api.app import create_app
from trainer.demo_api.v1_router import create_v1_router


def test_create_app_uses_demo_mode_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('DEMO_LIVE_MODE', raising=False)

    app = create_app()

    assert app.state.chat_service.live_runner is None


def test_create_app_attaches_live_runner_without_loading_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('DEMO_LIVE_MODE', 'on')

    app = create_app()

    assert app.state.chat_service.live_runner is not None


def test_create_app_rejects_invalid_live_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('DEMO_LIVE_MODE', 'bogus')

    with pytest.raises(RuntimeError, match='DEMO_LIVE_MODE'):
        create_app()


def test_create_app_wires_the_expert_reference_root_from_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("EXPERT_REFERENCE_ROOT", str(tmp_path))

    app = create_app()

    assert app.state.expert_guidance.reference_root == tmp_path


def test_create_app_preserves_guided_store_and_session_analysis_mcp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEMO_LIVE_MODE", "off")
    monkeypatch.setenv("MCP_BASE_URL", "http://integration-mcp.test:8100")

    with patch("trainer.demo_api.app.create_v1_router", wraps=create_v1_router) as factory:
        app = create_app()

    factory.assert_called_once()
    assert factory.call_args.kwargs["store"] is app.state.guided_workflow._store
    assert factory.call_args.kwargs["mcp_base_url"] == "http://integration-mcp.test:8100"
