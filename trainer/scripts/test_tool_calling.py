#!/usr/bin/env python3

from __future__ import annotations

import json
import importlib
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx

adapter_module = importlib.import_module("tool_calling_adapter")
ToolCallingAdapter = adapter_module.ToolCallingAdapter
ToolCallingError = adapter_module.ToolCallingError
ToolValidationError = adapter_module.ToolValidationError


class DummyResponse:
    def __init__(self, status_code: int = 200, json_data: dict[str, Any] | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text
        self.request = httpx.Request("POST", "http://localhost/mock")

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                message=f"HTTP {self.status_code}",
                request=self.request,
                response=httpx.Response(self.status_code, text=self.text, request=self.request),
            )

    def json(self) -> dict[str, Any]:
        return self._json_data


class FakeModel:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses

    def generate_response(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
        return self.responses.pop(0)


class ToolCallingAdapterTest(unittest.IsolatedAsyncioTestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.adapter = ToolCallingAdapter(
            schema_dir=Path("/workspace/trainer/configs/tool_schemas"),
            protocol_path=Path("/workspace/trainer/configs/tool_calling_protocol.json"),
            mcp_base_url="http://localhost:8100",
        )

    def _build_client_class(self, replies: list[Any], calls: list[dict[str, Any]]) -> type:
        class _Client:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            async def __aenter__(self) -> _Client:
                return self

            async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
                return False

            async def post(self, url: str, json: dict[str, Any]) -> DummyResponse:
                calls.append({"url": url, "json": json})
                item = replies.pop(0)
                if isinstance(item, Exception):
                    raise item
                return item

        return _Client

    def test_parse_tool_calls_single(self) -> None:
        raw = '<tool_call>{"name":"dft_judge","arguments":{"measurements":[10]}}</tool_call>'
        parsed = self.adapter.parse_tool_calls(raw)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["name"], "dft_judge")
        self.assertEqual(parsed[0]["arguments"]["measurements"], [10])
        self.assertIn("call_id", parsed[0])

    def test_parse_tool_calls_multiple(self) -> None:
        raw = (
            '<tool_call>{"name":"dft_judge","arguments":{"measurements":[10]}}</tool_call>'
            '<tool_call>{"name":"timeseries","arguments":{"action":"query","session_id":"s1"}}</tool_call>'
        )
        parsed = self.adapter.parse_tool_calls(raw)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[1]["name"], "timeseries")

    def test_parse_tool_calls_with_think_block(self) -> None:
        raw = (
            "<think>reasoning</think>"
            '<tool_call>{"name":"location_tracking","arguments":{"coordinates":[{"x":0,"y":0,"z":0}]}}</tool_call>'
        )
        parsed = self.adapter.parse_tool_calls(raw)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["name"], "location_tracking")

    def test_parse_tool_calls_malformed_json_raises(self) -> None:
        raw = '<tool_call>{"name":"dft_judge","arguments":{"x":1,}}</tool_call>'
        with self.assertRaises(ToolValidationError):
            self.adapter.parse_tool_calls(raw)

    def test_format_tool_results(self) -> None:
        results = [{"call_id": "abc", "status": "success", "result": {"ok": True}}]
        messages = self.adapter.format_tool_results(results)
        self.assertEqual(messages[0]["role"], "tool")
        self.assertEqual(messages[0]["tool_call_id"], "abc")
        self.assertEqual(json.loads(messages[0]["content"])["status"], "success")

    async def test_dispatch_dft_judge_normal(self) -> None:
        calls: list[dict[str, Any]] = []
        replies = [DummyResponse(json_data={"status": "success", "result": {"pass": True}})]
        client_cls = self._build_client_class(replies, calls)
        with patch("tool_calling_adapter.httpx.AsyncClient", new=client_cls):
            result = await self.adapter.dispatch_tool_call(
                "dft_judge",
                {"measurements": [95, 100], "spec": {"nominal": 100, "tolerance_rule": "90/10"}},
            )
        self.assertEqual(result["status"], "success")
        self.assertTrue(calls[0]["url"].endswith("/dft_judge/evaluate"))

    async def test_dispatch_dft_judge_edge_empty_measurements(self) -> None:
        calls: list[dict[str, Any]] = []
        replies = [DummyResponse(json_data={"status": "success", "result": {"pass": False}})]
        client_cls = self._build_client_class(replies, calls)
        with patch("tool_calling_adapter.httpx.AsyncClient", new=client_cls):
            result = await self.adapter.dispatch_tool_call(
                "dft_judge",
                {"measurements": [], "spec": {"nominal": 100, "tolerance_rule": "90/10"}},
            )
        self.assertEqual(result["status"], "success")

    async def test_dispatch_anomaly_detection_normal(self) -> None:
        calls: list[dict[str, Any]] = []
        replies = [DummyResponse(json_data={"status": "success", "result": {"anomaly": "low"}})]
        client_cls = self._build_client_class(replies, calls)
        with patch("tool_calling_adapter.httpx.AsyncClient", new=client_cls):
            result = await self.adapter.dispatch_tool_call(
                "anomaly_detection",
                {
                    "image_context": "left area thin film",
                    "dft_measurements": [90, 88],
                    "env": {"temperature": 23, "humidity": 55},
                },
            )
        self.assertEqual(result["status"], "success")
        self.assertTrue(calls[0]["url"].endswith("/anomaly_detection/detect"))

    async def test_dispatch_anomaly_detection_edge_boundary_env(self) -> None:
        calls: list[dict[str, Any]] = []
        replies = [DummyResponse(json_data={"status": "success", "result": {"anomaly": "none"}})]
        client_cls = self._build_client_class(replies, calls)
        with patch("tool_calling_adapter.httpx.AsyncClient", new=client_cls):
            result = await self.adapter.dispatch_tool_call(
                "anomaly_detection",
                {
                    "image_context": "stable",
                    "dft_measurements": [100],
                    "env": {"temperature": 0, "humidity": 0},
                },
            )
        self.assertEqual(result["status"], "success")

    async def test_dispatch_pose_estimation_normal(self) -> None:
        calls: list[dict[str, Any]] = []
        replies = [DummyResponse(json_data={"status": "success", "result": {"score": 0.9}})]
        client_cls = self._build_client_class(replies, calls)
        args = {
            "current_pose": {"joints": {"elbow": {"x": 1, "y": 2, "z": 3}}},
            "reference_pose": {"joints": {"elbow": {"x": 1, "y": 1.5, "z": 3}}},
            "imu_data": {"accel": [0.1, 0.2, 0.3], "gyro": [1, 2, 3]},
        }
        with patch("tool_calling_adapter.httpx.AsyncClient", new=client_cls):
            result = await self.adapter.dispatch_tool_call("pose_estimation", args)
        self.assertEqual(result["status"], "success")
        self.assertTrue(calls[0]["url"].endswith("/pose_estimation/compare"))

    async def test_dispatch_pose_estimation_edge_without_imu(self) -> None:
        calls: list[dict[str, Any]] = []
        replies = [DummyResponse(json_data={"status": "success", "result": {"score": 0.7}})]
        client_cls = self._build_client_class(replies, calls)
        args = {
            "current_pose": {"joints": {"wrist": {"x": 0, "y": 0, "z": 0}}},
            "reference_pose": {"joints": {"wrist": {"x": 0, "y": 0.5, "z": 0}}},
        }
        with patch("tool_calling_adapter.httpx.AsyncClient", new=client_cls):
            result = await self.adapter.dispatch_tool_call("pose_estimation", args)
        self.assertEqual(result["status"], "success")

    async def test_dispatch_sequence_validation_normal(self) -> None:
        calls: list[dict[str, Any]] = []
        replies = [DummyResponse(json_data={"status": "success", "result": {"valid": True}})]
        client_cls = self._build_client_class(replies, calls)
        args = {
            "trajectory": [{"x": 0, "y": 1, "z": 2, "timestamp": 1.0}, {"x": 0, "y": 0, "z": 2, "timestamp": 2.0}],
            "zone_id": "B-3",
            "zone_type": "수직 패널",
            "reference_pattern": "top_to_bottom",
        }
        with patch("tool_calling_adapter.httpx.AsyncClient", new=client_cls):
            result = await self.adapter.dispatch_tool_call("sequence_validation", args)
        self.assertEqual(result["status"], "success")
        self.assertTrue(calls[0]["url"].endswith("/sequence_validation/validate"))

    async def test_dispatch_sequence_validation_edge_single_point(self) -> None:
        calls: list[dict[str, Any]] = []
        replies = [DummyResponse(json_data={"status": "success", "result": {"valid": False}})]
        client_cls = self._build_client_class(replies, calls)
        args = {
            "trajectory": [{"x": 0, "y": 0, "z": 0, "timestamp": 1.0}],
            "zone_id": "A-1",
            "zone_type": "수평 패널",
            "reference_pattern": "left_to_right",
        }
        with patch("tool_calling_adapter.httpx.AsyncClient", new=client_cls):
            result = await self.adapter.dispatch_tool_call("sequence_validation", args)
        self.assertEqual(result["status"], "success")

    async def test_dispatch_timeseries_normal_update(self) -> None:
        calls: list[dict[str, Any]] = []
        replies = [DummyResponse(json_data={"status": "success", "result": {"updated": True}})]
        client_cls = self._build_client_class(replies, calls)
        args = {
            "action": "update",
            "session_id": "s-1",
            "data": {"timestamp": 1700, "speed": 0.1, "distance": 2.0, "angle": 10.0},
        }
        with patch("tool_calling_adapter.httpx.AsyncClient", new=client_cls):
            result = await self.adapter.dispatch_tool_call("timeseries", args)
        self.assertEqual(result["status"], "success")
        self.assertTrue(calls[0]["url"].endswith("/timeseries/update"))

    async def test_dispatch_timeseries_edge_query_without_data(self) -> None:
        calls: list[dict[str, Any]] = []
        replies = [DummyResponse(json_data={"status": "success", "result": {"points": 0}})]
        client_cls = self._build_client_class(replies, calls)
        args = {"action": "query", "session_id": "s-2"}
        with patch("tool_calling_adapter.httpx.AsyncClient", new=client_cls):
            result = await self.adapter.dispatch_tool_call("timeseries", args)
        self.assertEqual(result["status"], "success")
        self.assertTrue(calls[0]["url"].endswith("/timeseries/query"))

    async def test_dispatch_location_tracking_normal(self) -> None:
        calls: list[dict[str, Any]] = []
        replies = [DummyResponse(json_data={"status": "success", "result": {"zone_id": "B-2"}})]
        client_cls = self._build_client_class(replies, calls)
        args = {
            "coordinates": [{"x": 1, "y": 2, "z": 3, "timestamp": 10.0}],
            "ship_model_id": "ship-v1",
        }
        with patch("tool_calling_adapter.httpx.AsyncClient", new=client_cls):
            result = await self.adapter.dispatch_tool_call("location_tracking", args)
        self.assertEqual(result["status"], "success")
        self.assertTrue(calls[0]["url"].endswith("/location_tracking/identify"))

    async def test_dispatch_location_tracking_edge_without_model_id(self) -> None:
        calls: list[dict[str, Any]] = []
        replies = [DummyResponse(json_data={"status": "success", "result": {"zone_id": "UNK"}})]
        client_cls = self._build_client_class(replies, calls)
        args = {"coordinates": [{"x": 1, "y": 2, "z": 3}]}
        with patch("tool_calling_adapter.httpx.AsyncClient", new=client_cls):
            result = await self.adapter.dispatch_tool_call("location_tracking", args)
        self.assertEqual(result["status"], "success")

    async def test_dispatch_invalid_tool_name_raises(self) -> None:
        with self.assertRaises(ToolValidationError):
            await self.adapter.dispatch_tool_call("unknown_tool", {"x": 1})

    async def test_dispatch_timeout_returns_protocol_error(self) -> None:
        calls: list[dict[str, Any]] = []
        replies = [httpx.TimeoutException("timed out"), httpx.TimeoutException("timed out")]
        client_cls = self._build_client_class(replies, calls)
        with patch("tool_calling_adapter.httpx.AsyncClient", new=client_cls):
            result = await self.adapter.dispatch_tool_call(
                "dft_judge",
                {"measurements": [100], "spec": {"nominal": 100, "tolerance_rule": "90/10"}},
            )
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "TIMEOUT")

    async def test_run_tool_loop_calls_tool_and_returns_final_text(self) -> None:
        model = FakeModel(
            responses=[
                '<tool_call>{"name":"timeseries","arguments":{"action":"query","session_id":"s1"}}</tool_call>',
                "Final answer after tool",
            ]
        )
        tokenizer: Any = object()
        with patch.object(
            self.adapter,
            "dispatch_tool_call",
            new=AsyncMock(return_value={"status": "success", "result": {"points": 10}}),
        ):
            output = await self.adapter.run_tool_loop(
                messages=[{"role": "user", "content": "status?"}],
                model=model,
                tokenizer=tokenizer,
                max_rounds=3,
            )
        self.assertEqual(output["final_response"], "Final answer after tool")
        self.assertGreaterEqual(len(output["tool_results"]), 1)

    async def test_run_tool_loop_exceeds_rounds_raises(self) -> None:
        model = FakeModel(
            responses=['<tool_call>{"name":"timeseries","arguments":{"action":"query","session_id":"s1"}}</tool_call>'] * 3
        )
        tokenizer: Any = object()
        with patch.object(
            self.adapter,
            "dispatch_tool_call",
            new=AsyncMock(return_value={"status": "success", "result": {"points": 10}}),
        ):
            with self.assertRaises(ToolCallingError):
                await self.adapter.run_tool_loop(
                    messages=[{"role": "user", "content": "status?"}],
                    model=model,
                    tokenizer=tokenizer,
                    max_rounds=2,
                )


if __name__ == "__main__":
    unittest.main()
