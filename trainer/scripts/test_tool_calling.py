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

    def generate_response(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> str:
        return self.responses.pop(0)


class _FakeIds:
    shape = (1, 4)


class _FakeInputs(dict[str, Any]):
    def __init__(self) -> None:
        super().__init__()
        self.input_ids = _FakeIds()
        self["input_ids"] = self.input_ids

    def to(self, device: Any) -> "_FakeInputs":
        return self


class RecordingTokenizer:
    eos_token_id = 0

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.decode_count = 0

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
        return "PROMPT"

    def __call__(self, texts: Any, **kwargs: Any) -> _FakeInputs:
        return _FakeInputs()

    def decode(self, ids: Any, skip_special_tokens: bool = True) -> str:
        out = self.outputs[self.decode_count]
        self.decode_count += 1
        return out


class RecordingModel:
    device = "cpu"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        return [[0] * 12]


class ScriptedClock:
    def __init__(self, *values: float) -> None:
        self.values = list(values)
        self.index = 0

    def monotonic(self) -> float:
        if self.index >= len(self.values):
            return self.values[-1]
        value = self.values[self.index]
        if self.index < len(self.values) - 1:
            self.index += 1
        return value


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

    def test_parse_tool_calls_nested_args_no_closing_tag(self) -> None:
        raw = '<tool_call>{"name": "location_tracking", "arguments": {"coordinates": [{"x": 910, "y": -45, "z": 250}], "ship_model_id": "default"}}'
        parsed = self.adapter.parse_tool_calls(raw)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["name"], "location_tracking")
        self.assertEqual(parsed[0]["arguments"]["coordinates"][0]["x"], 910)

    def test_parse_tool_calls_multiple_nested_no_closing(self) -> None:
        raw = (
            '<tool_call>{"name": "location_tracking", "arguments": {"coordinates": [{"x": 0}]}}\n'
            '<tool_call>{"name": "sequence_validation", "arguments": {"current_zone_id": "P-2", "zone_map": [{"zone_id":"A-2"}]}}'
        )
        parsed = self.adapter.parse_tool_calls(raw)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["name"], "location_tracking")
        self.assertEqual(parsed[1]["name"], "sequence_validation")

    def test_parse_tool_calls_malformed_json_skips(self) -> None:
        raw = '<tool_call>{"name":"dft_judge","arguments":{"x":1,}}</tool_call>'
        parsed = self.adapter.parse_tool_calls(raw)
        self.assertEqual(len(parsed), 0)

    def test_parse_tool_calls_mixed_valid_and_malformed(self) -> None:
        raw = (
            '<tool_call>{"name":"dft_judge","arguments":{"x":1,}}</tool_call>'
            '<tool_call>{"name":"anomaly_detection","arguments":{"image_context":"test"}}</tool_call>'
        )
        parsed = self.adapter.parse_tool_calls(raw)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["name"], "anomaly_detection")

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
        replies = [DummyResponse(json_data={"status": "success", "result": {"anomaly_detected": True, "anomalies": [{"type": "저도막"}]}})]
        client_cls = self._build_client_class(replies, calls)
        with patch("tool_calling_adapter.httpx.AsyncClient", new=client_cls):
            result = await self.adapter.dispatch_tool_call(
                "anomaly_detection",
                {
                    "image_context": "left area thin film",
                    "zone_id": "B-3",
                    "spec": {"nominal_um": 100, "tolerance_rule": "90/10"},
                    "dft_measurements": [
                        {"value_um": 90, "zone_id": "B-3", "position": "left"},
                        {"value_um": 88, "zone_id": "B-3", "position": "left"}
                    ],
                },
            )
        self.assertEqual(result["status"], "success")
        self.assertTrue(calls[0]["url"].endswith("/anomaly_detection/detect"))

    async def test_dispatch_anomaly_detection_edge_boundary_env(self) -> None:
        calls: list[dict[str, Any]] = []
        replies = [DummyResponse(json_data={"status": "success", "result": {"anomaly_detected": False, "anomalies": []}})]
        client_cls = self._build_client_class(replies, calls)
        with patch("tool_calling_adapter.httpx.AsyncClient", new=client_cls):
            result = await self.adapter.dispatch_tool_call(
                "anomaly_detection",
                {
                    "image_context": "stable",
                    "zone_id": "B-3",
                    "spec": {"nominal_um": 100, "tolerance_rule": "90/10"},
                    "dft_measurements": [{"value_um": 100, "zone_id": "B-3", "position": "center"}],
                },
            )
        self.assertEqual(result["status"], "success")

    async def test_dispatch_pose_estimation_normal(self) -> None:
        calls: list[dict[str, Any]] = []
        replies = [DummyResponse(json_data={"status": "success", "result": {"similarity_score": 0.9}})]
        client_cls = self._build_client_class(replies, calls)
        args = {
            "mode": "compare",
            "block_id": "BLK-A",
            "current_pose": {"joints": {"elbow_r": {"x": 1, "y": 2, "z": 3}}},
            "expert_pose_id": "EP-001",
        }
        with patch("tool_calling_adapter.httpx.AsyncClient", new=client_cls):
            result = await self.adapter.dispatch_tool_call("pose_estimation", args)
        self.assertEqual(result["status"], "success")
        self.assertTrue(calls[0]["url"].endswith("/pose_estimation/compare"))

    async def test_dispatch_pose_estimation_edge_without_imu(self) -> None:
        calls: list[dict[str, Any]] = []
        replies = [DummyResponse(json_data={"status": "success", "result": {"similarity_score": 0.7}})]
        client_cls = self._build_client_class(replies, calls)
        args = {
            "mode": "compare",
            "current_pose": {"joints": {"wrist_r": {"x": 0, "y": 0, "z": 0}}},
            "block_id": "BLK-A",
        }
        with patch("tool_calling_adapter.httpx.AsyncClient", new=client_cls):
            result = await self.adapter.dispatch_tool_call("pose_estimation", args)
        self.assertEqual(result["status"], "success")

    async def test_dispatch_sequence_validation_normal(self) -> None:
        calls: list[dict[str, Any]] = []
        replies = [DummyResponse(json_data={"status": "success", "result": {"next_zone": "B-3", "block_sequence": ["BLK-A"]}})]
        client_cls = self._build_client_class(replies, calls)
        args = {
            "structure_type": "block",
            "zone_map": [
                {"zone_id": "B-3", "block_id": "BLK-A", "zone_type": "수직 패널", "position": "upper", "is_painted": False, "area_m2": 8.0},
                {"zone_id": "B-4", "block_id": "BLK-A", "zone_type": "수직 패널", "position": "middle", "is_painted": False, "area_m2": 6.0}
            ],
            "current_zone_id": "B-3",
        }
        with patch("tool_calling_adapter.httpx.AsyncClient", new=client_cls):
            result = await self.adapter.dispatch_tool_call("sequence_validation", args)
        self.assertEqual(result["status"], "success")
        self.assertTrue(calls[0]["url"].endswith("/sequence_validation/validate"))

    async def test_dispatch_sequence_validation_edge_single_point(self) -> None:
        calls: list[dict[str, Any]] = []
        replies = [DummyResponse(json_data={"status": "success", "result": {"next_zone": "A-1", "block_sequence": ["BLK-A"]}})]
        client_cls = self._build_client_class(replies, calls)
        args = {
            "structure_type": "block",
            "zone_map": [
                {"zone_id": "A-1", "block_id": "BLK-A", "zone_type": "수평 패널", "position": "left", "is_painted": False, "area_m2": 4.0}
            ],
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

    async def test_dispatch_invalid_tool_name_returns_error(self) -> None:
        result = await self.adapter.dispatch_tool_call("unknown_tool", {"x": 1})
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "TOOL_NOT_FOUND")
        self.assertIn("unknown_tool", result["error"]["message"])

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

    async def test_run_tool_loop_exceeds_rounds_graceful_fallback(self) -> None:
        tool_call = '<tool_call>{"name":"timeseries","arguments":{"action":"query","session_id":"s1"}}</tool_call>'
        model = FakeModel(
            responses=[tool_call, tool_call, "[분석] 종합 결과입니다. [판정] 정상. [근거] DOC-doc_0001-C0001"]
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
                max_rounds=2,
            )
        self.assertEqual(output["rounds"], 2)
        self.assertTrue(output.get("truncated", False))
        self.assertIsInstance(output["final_response"], str)
        self.assertNotIn("<tool_call>", output["final_response"])
        self.assertTrue(output.get("synthesized", False))
        self.assertGreaterEqual(len(output["tool_results"]), 1)


class ToolLoopDeadlineTest(unittest.IsolatedAsyncioTestCase):
    adapter: ToolCallingAdapter

    def setUp(self) -> None:
        self.adapter = ToolCallingAdapter(
            schema_dir=Path("/workspace/trainer/configs/tool_schemas"),
            protocol_path=Path("/workspace/trainer/configs/tool_calling_protocol.json"),
            mcp_base_url="http://localhost:8100",
        )

    async def test_deadline_stops_further_rounds_and_forces_synthesis(self) -> None:
        call = '<tool_call>{"name": "dft_judge", "arguments": {"measurements": [95], "spec": {"nominal": 100}}}</tool_call>'
        model = FakeModel([call, call, "SYNTHESIZED"])
        clock = ScriptedClock(0.0, 100.0, 200.0, 215.0, 216.0)

        with patch.object(adapter_module, "time", clock), \
             patch.object(self.adapter, "dispatch_tool_call", new=AsyncMock(return_value={"status": "ok"})):
            result = await self.adapter.run_tool_loop(
                messages=[{"role": "user", "content": "q"}],
                model=model,
                tokenizer=object(),
                max_rounds=5,
                deadline_sec=150.0,
            )

        self.assertTrue(result["deadline_hit"])
        self.assertEqual(result["rounds"], 2)
        self.assertEqual(result["final_response"], "SYNTHESIZED")
        self.assertEqual(result["synthesis_ms"], 1000)

    async def test_no_deadline_hit_when_within_budget(self) -> None:
        call = '<tool_call>{"name": "timeseries", "arguments": {"action": "query", "session_id": "s1"}}</tool_call>'
        model = FakeModel([call, "plain answer"])
        clock = ScriptedClock(0.0, 50.0, 100.0)
        with patch.object(adapter_module, "time", clock), \
             patch.object(
                 self.adapter,
                 "dispatch_tool_call",
                 new=AsyncMock(return_value={"status": "success", "result": {"points": 10}}),
             ):
            result = await self.adapter.run_tool_loop(
                messages=[{"role": "user", "content": "q"}],
                model=model,
                tokenizer=object(),
                max_rounds=5,
                deadline_sec=150.0,
            )
        self.assertFalse(result["deadline_hit"])
        self.assertEqual(result["rounds"], 2)
        self.assertEqual(result["final_response"], "plain answer")

    async def test_deadline_never_fires_before_any_tool_result(self) -> None:
        model = FakeModel(["plain answer"])
        clock = ScriptedClock(0.0, 200.0)
        with patch.object(adapter_module, "time", clock):
            result = await self.adapter.run_tool_loop(
                messages=[{"role": "user", "content": "q"}],
                model=model,
                tokenizer=object(),
                max_rounds=5,
                deadline_sec=1.0,
            )
        self.assertFalse(result["deadline_hit"])
        self.assertEqual(result["rounds"], 1)
        self.assertEqual(result["final_response"], "plain answer")

    async def test_empty_post_tool_response_reports_timed_synthesis_ms(self) -> None:
        call = '<tool_call>{"name": "timeseries", "arguments": {"action": "query", "session_id": "s1"}}</tool_call>'
        model = FakeModel([call, "", "SYNTHESIZED"])
        clock = ScriptedClock(0.0, 10.0, 20.0, 25.0, 26.0)

        with patch.object(adapter_module, "time", clock), \
             patch.object(
                 self.adapter,
                 "dispatch_tool_call",
                 new=AsyncMock(return_value={"status": "success", "result": {"points": 10}}),
             ):
            result = await self.adapter.run_tool_loop(
                messages=[{"role": "user", "content": "q"}],
                model=model,
                tokenizer=object(),
                max_rounds=5,
                deadline_sec=150.0,
            )

        self.assertFalse(result["deadline_hit"])
        self.assertEqual(result["rounds"], 2)
        self.assertEqual(result["final_response"], "SYNTHESIZED")
        self.assertEqual(result["synthesis_ms"], 5000)


if __name__ == "__main__":
    unittest.main()
