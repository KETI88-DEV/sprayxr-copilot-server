"""Live readiness for ``GET /api/ready``.

``/api/health`` only says the process is up. Readiness says a live request can
be answered now: the model is reachable (a remote OpenAI-compatible server
serves ``llm.served_model_name``, or the in-process HF model is loaded) and the
pipeline singleton is loaded with a non-empty RAG index. MCP is probed for
information only, because ``knowledge_search`` never calls it.

The body names checks and booleans only — no URLs, paths or exception text.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import httpx
import yaml

from trainer.scripts.openai_backend import OpenAIBackendSettings, probe_openai_server

logger = logging.getLogger(__name__)

_MCP_PROBE_TIMEOUT_SEC: Final[float] = 2.0


def _load_config(config_path: str) -> Mapping[str, Any] | None:
    try:
        with Path(config_path).open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError):
        logger.warning("readiness: config %s unreadable", config_path, exc_info=True)
        return None
    return data if isinstance(data, dict) else None


def _pipeline_check(pipeline: object | None) -> dict[str, Any]:
    if pipeline is None:
        return {"ok": False, "loaded": False, "rag_vectors": None}
    ntotal = getattr(getattr(pipeline, "index", None), "ntotal", None)
    vectors = int(ntotal) if isinstance(ntotal, int) else None
    return {"ok": bool(vectors), "loaded": True, "rag_vectors": vectors}


async def _llm_check(
    config: Mapping[str, Any] | None,
    pipeline_loaded: bool,
    transport: httpx.AsyncBaseTransport | None,
) -> dict[str, Any]:
    if config is None:
        return {"ok": False, "backend": None}
    backend = str(config.get("llm", {}).get("backend", "hf"))
    if backend != "openai":
        # The model lives in-process: it is ready exactly when the pipeline is.
        return {"ok": pipeline_loaded, "backend": backend}
    try:
        settings = OpenAIBackendSettings.from_config(config)
    except ValueError:
        logger.warning("readiness: invalid openai backend config", exc_info=True)
        return {"ok": False, "backend": backend, "health": False, "model_served": False}
    probe = await probe_openai_server(settings, transport=transport)
    return {
        "ok": probe.health_ok and probe.model_served,
        "backend": backend,
        "health": probe.health_ok,
        "model_served": probe.model_served,
    }


async def _mcp_check(mcp_base_url: str, transport: httpx.AsyncBaseTransport | None) -> dict[str, Any]:
    ok = False
    try:
        async with httpx.AsyncClient(timeout=_MCP_PROBE_TIMEOUT_SEC, transport=transport) as client:
            ok = (await client.get(f"{mcp_base_url.rstrip('/')}/health")).status_code == 200
    except httpx.HTTPError:
        logger.info("readiness: MCP /health unreachable")
    return {"ok": ok, "required": False}


async def evaluate_live_readiness(
    *,
    pipeline: object | None,
    config_path: str | None = None,
    mcp_base_url: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Returns ``(ready, checks)``; only required checks decide ``ready``."""
    config = getattr(pipeline, "config", None) if pipeline is not None else None
    if not isinstance(config, Mapping):
        config = _load_config(config_path or os.environ.get("CONFIG_PATH", ""))

    pipeline_check = _pipeline_check(pipeline)
    checks: dict[str, dict[str, Any]] = {
        "pipeline": pipeline_check,
        "llm": await _llm_check(config, pipeline_check["loaded"], transport),
        "mcp": await _mcp_check(mcp_base_url or "http://localhost:8100", transport),
    }
    ready = all(check["ok"] for check in checks.values() if check.get("required", True))
    return ready, checks
