"""Bounded caller-channel hints; never authentication or trainee identity.

IP comes only from the server's ASGI client address. Forwarded headers are the
ASGI server's responsibility and require its explicit trusted-proxy settings.
"""
from __future__ import annotations

from ipaddress import ip_address
from typing import Any
from urllib.parse import urlsplit, urlunsplit

CALLER_LABELS = {"motion-viewer": "3D VIEWER", "assistant-test": "TEST", "query-chat": "QUERY CHAT",
                 "test-api": "TEST API", "api-docs": "API DOCS", "simulator-api": "SIMULATOR/API"}
PROVENANCE_COLUMNS = ("client_ip", "caller_source", "caller_origin", "caller_referer", "source_evidence")


def _header(headers: list, name: bytes) -> str:
    for key, value in headers:
        if key.lower() == name:
            return value.decode("latin-1")[:2048]
    return ""


def valid_ip(value: Any) -> str | None:
    try:
        return str(ip_address(str(value))) if value else None
    except ValueError:
        return None


def _safe_url(value: str, *, origin: bool = False) -> str | None:
    """Retain useful browser context without URL credentials or query secrets."""
    try:
        if any(ord(char) < 32 for char in value):
            return None
        parts = urlsplit(value)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            return None
        host = parts.hostname
        if ":" in host:
            host = "[" + host + "]"
        authority = host + (f":{parts.port}" if parts.port is not None else "")
        return urlunsplit((parts.scheme, authority, "" if origin else parts.path[:512], "", ""))[:768]
    except ValueError:
        return None


def capture_provenance(scope: dict[str, Any]) -> dict[str, Any]:
    headers = scope.get("headers", [])
    supplied = _header(headers, b"x-copilot-source").strip().lower()
    referer = _safe_url(_header(headers, b"referer"))
    if supplied in {"motion-viewer", "assistant-test", "query-chat"}:
        source, evidence = supplied, "header"
    elif scope.get("path", "").startswith("/api/v1/test/"):
        source, evidence = "test-api", "path"
    elif referer and urlsplit(referer).path.rstrip("/") in {"/docs", "/redoc"}:
        source, evidence = "api-docs", "referer"
    else:
        source, evidence = "simulator-api", "fallback"
    peer = scope.get("client")
    return {"client_ip": valid_ip(peer[0]) if isinstance(peer, (tuple, list)) and peer else None,
            "caller_source": source, "source_evidence": evidence,
            "caller_origin": _safe_url(_header(headers, b"origin"), origin=True), "caller_referer": referer}


def caller_label(data: dict[str, Any]) -> str:
    source = data.get("caller_source")
    if source in CALLER_LABELS:
        return CALLER_LABELS[source]
    # Historical synthetic route is observable evidence, unlike generic "http".
    if data.get("source") == "test" or str(data.get("path") or "").startswith("/api/v1/test/"):
        return CALLER_LABELS["test-api"]
    return "미기록"
