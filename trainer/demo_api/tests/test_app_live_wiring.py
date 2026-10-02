from __future__ import annotations

from pathlib import Path

import pytest

from trainer.demo_api.app import create_app


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
