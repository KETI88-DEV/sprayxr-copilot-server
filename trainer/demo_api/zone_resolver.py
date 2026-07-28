from __future__ import annotations

import logging
from typing import Protocol

import httpx

logger = logging.getLogger(__name__)

_COORDINATE_LENGTH = 3


class ZoneResolver(Protocol):
    async def resolve(self, coordinates: list[float]) -> str | None: ...


class NullZoneResolver:
    async def resolve(self, coordinates: list[float]) -> str | None:
        _ = coordinates
        return None


class McpZoneResolver:
    """Resolves a zone through the MCP location_tracking tool.

    Every failure returns None instead of raising: this is an observability
    fallback for telemetry that carries no explicit zone, and a missing or slow
    MCP must never break the simulator interface contract.
    """

    def __init__(
        self,
        base_url: str,
        timeout_sec: float = 2.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_sec = timeout_sec
        self._transport = transport

    async def resolve(self, coordinates: list[float]) -> str | None:
        if len(coordinates) < _COORDINATE_LENGTH:
            return None

        payload = {
            "coordinates": [
                {"x": coordinates[0], "y": coordinates[1], "z": coordinates[2]}
            ]
        }

        try:
            async with httpx.AsyncClient(
                timeout=self._timeout_sec, transport=self._transport
            ) as client:
                response = await client.post(
                    f"{self._base_url}/location_tracking/identify", json=payload
                )
                if response.status_code != 200:
                    return None
                body: object = response.json()
        except Exception:  # noqa: BLE001
            logger.debug("zone resolver: location_tracking unavailable", exc_info=True)
            return None

        if not isinstance(body, dict):
            return None
        result = body.get("result")
        if not isinstance(result, dict):
            return None
        zone_id = result.get("zone_id")
        return zone_id if isinstance(zone_id, str) and zone_id else None
