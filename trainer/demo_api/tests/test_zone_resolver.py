from __future__ import annotations

import anyio
import httpx

from trainer.demo_api.zone_resolver import McpZoneResolver, NullZoneResolver


def test_null_resolver_returns_none() -> None:
    async def scenario() -> str | None:
        return await NullZoneResolver().resolve([1.0, 2.0, 3.0])

    assert anyio.run(scenario) is None


def test_mcp_resolver_returns_zone_id_from_success_envelope() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        return httpx.Response(200, json={"status": "success", "result": {"zone_id": "B-3"}})

    async def scenario() -> str | None:
        resolver = McpZoneResolver("http://mcp.test", transport=httpx.MockTransport(handler))
        return await resolver.resolve([1.0, 2.0, 3.0])

    assert anyio.run(scenario) == "B-3"
    assert seen["path"] == "/location_tracking/identify"


def test_mcp_resolver_sends_first_triplet_as_coordinate() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.content.decode("utf-8")
        return httpx.Response(200, json={"status": "success", "result": {"zone_id": "B-3"}})

    async def scenario() -> str | None:
        resolver = McpZoneResolver("http://mcp.test", transport=httpx.MockTransport(handler))
        return await resolver.resolve([1.5, 2.5, 3.5])

    _ = anyio.run(scenario)
    body = str(seen["body"])
    assert '"x": 1.5' in body or '"x":1.5' in body


def test_mcp_resolver_returns_none_on_error_status() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"status": "error"})

    async def scenario() -> str | None:
        resolver = McpZoneResolver("http://mcp.test", transport=httpx.MockTransport(handler))
        return await resolver.resolve([1.0, 2.0, 3.0])

    assert anyio.run(scenario) is None


def test_mcp_resolver_returns_none_when_transport_raises() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("unreachable")

    async def scenario() -> str | None:
        resolver = McpZoneResolver("http://mcp.test", transport=httpx.MockTransport(handler))
        return await resolver.resolve([1.0, 2.0, 3.0])

    assert anyio.run(scenario) is None


def test_mcp_resolver_returns_none_when_zone_id_missing() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "success", "result": {}})

    async def scenario() -> str | None:
        resolver = McpZoneResolver("http://mcp.test", transport=httpx.MockTransport(handler))
        return await resolver.resolve([1.0, 2.0, 3.0])

    assert anyio.run(scenario) is None


def test_mcp_resolver_ignores_coordinates_that_are_not_triplets() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("resolver must not call MCP for an incomplete coordinate")

    async def scenario() -> str | None:
        resolver = McpZoneResolver("http://mcp.test", transport=httpx.MockTransport(handler))
        return await resolver.resolve([1.0, 2.0])

    assert anyio.run(scenario) is None
