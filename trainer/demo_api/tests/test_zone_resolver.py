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


def test_mcp_resolver_returns_none_for_http_200_error_envelope() -> None:
    """MCP는 오류도 200으로 돌려준다. status를 보지 않으면 오류를 성공으로 오인한다."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"status": "error", "error": {"code": "SERVER_ERROR"}, "result": {"zone_id": "B-3"}},
        )

    async def scenario() -> str | None:
        resolver = McpZoneResolver("http://mcp.test", transport=httpx.MockTransport(handler))
        return await resolver.resolve([1.0, 2.0, 3.0])

    assert anyio.run(scenario) is None


def test_mcp_resolver_returns_none_for_malformed_json() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"{not json")

    async def scenario() -> str | None:
        resolver = McpZoneResolver("http://mcp.test", transport=httpx.MockTransport(handler))
        return await resolver.resolve([1.0, 2.0, 3.0])

    assert anyio.run(scenario) is None


def test_mcp_resolver_accepts_the_real_success_envelope() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "status": "success",
                "result": {"zone_id": "B-3", "coverage_pct": 0.62},
                "error": None,
                "timestamp": "2026-07-28T00:00:00+00:00",
            },
        )

    async def scenario() -> str | None:
        resolver = McpZoneResolver("http://mcp.test", transport=httpx.MockTransport(handler))
        return await resolver.resolve([1.0, 2.0, 3.0])

    assert anyio.run(scenario) == "B-3"
