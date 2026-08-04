#!/usr/bin/env python3
# pyright: reportAny=false, reportImplicitOverride=false

from __future__ import annotations

import importlib
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

bench_latency = importlib.import_module("bench_latency")


class _ChunkStream(httpx.SyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks: list[bytes] = chunks
        self.consumed: list[int] = []

    def __iter__(self) -> Iterator[bytes]:
        for index, chunk in enumerate(self.chunks):
            self.consumed.append(index)
            yield chunk


def _frame(kind: str, data: str) -> bytes:
    return f'event: {kind}\ndata: {{"event":"{kind}","data":{data}}}\n\n'.encode()


def test_measure_question_stops_at_complete_and_counts_structured_timeout() -> None:
    stream = _ChunkStream(
        [
            _frame("message_delta", '{"text":"첫 토큰"}'),
            _frame(
                "tool_result",
                '{"name":"dft_judge","arguments":{},"result":{"status":"error","error":{"code":"TIMEOUT"}}}',
            ),
            _frame(
                "complete",
                '{"tool_call_count":1,"rag_chunk_count":2,"ttft_ms":12,"synthesis_ms":34,"total_ms":56,"deadline_hit":false,"degraded":false}',
            ),
            _frame("status", '{"stage":"must_not_be_consumed","chunks":0}'),
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        return httpx.Response(200, stream=stream)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        row = bench_latency.measure_question(
            client,
            "http://test",
            {"q_id": "T1", "question": "DFT", "tool_calls_required": ["dft_judge"]},
            5.0,
        )

    assert row["path"] == "tools"
    assert row["ttft_ms"] is not None
    assert row["mcp_timeout_count"] == 1
    assert row["degraded"] is False
    assert stream.consumed == [0, 1, 2]


def test_measure_question_preserves_terminal_error_payload() -> None:
    stream = _ChunkStream(
        [_frame("error", '{"message":"streaming pipeline failed","retryable":true,"ttft_ms":null,"total_ms":42}')]
    )

    with httpx.Client(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, stream=stream))
    ) as client:
        row = bench_latency.measure_question(
            client,
            "http://test",
            {"q_id": "R1", "question": "질문", "tool_calls_required": []},
            5.0,
        )

    assert row["path"] is None
    assert row["error"] == "streaming pipeline failed"
    assert row["backend_total_ms"] == 42.0


def test_measure_question_rejects_malformed_complete_payload() -> None:
    stream = _ChunkStream(
        [_frame("complete", '{"tool_call_count":-1,"rag_chunk_count":2,"ttft_ms":12}')]
    )

    with httpx.Client(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, stream=stream))
    ) as client:
        row = bench_latency.measure_question(
            client,
            "http://test",
            {"q_id": "R1", "question": "질문", "tool_calls_required": []},
            5.0,
        )

    assert row["path"] is None
    assert row["error"] is not None
    assert "complete" in row["error"]


def test_measure_question_rejects_malformed_message_delta() -> None:
    stream = _ChunkStream(
        [
            _frame("message_delta", "{}"),
            _frame(
                "complete",
                '{"tool_call_count":0,"rag_chunk_count":1,"ttft_ms":12,"synthesis_ms":34,"total_ms":56,"deadline_hit":false}',
            ),
        ]
    )

    with httpx.Client(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, stream=stream))
    ) as client:
        row = bench_latency.measure_question(
            client,
            "http://test",
            {"q_id": "R1", "question": "질문", "tool_calls_required": []},
            5.0,
        )

    assert row["path"] is None
    assert "message_delta" in str(row["error"])


def test_measure_question_rejects_malformed_second_message_delta() -> None:
    stream = _ChunkStream(
        [
            _frame("message_delta", '{"text":"valid"}'),
            _frame("message_delta", "{}"),
            _frame(
                "complete",
                '{"tool_call_count":0,"rag_chunk_count":1,"ttft_ms":12,"synthesis_ms":34,"total_ms":56,"deadline_hit":false}',
            ),
        ]
    )

    with httpx.Client(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, stream=stream))
    ) as client:
        row = bench_latency.measure_question(
            client,
            "http://test",
            {"q_id": "R1", "question": "질문", "tool_calls_required": []},
            5.0,
        )

    assert row["path"] is None
    assert "message_delta" in str(row["error"])


def test_measure_question_rejects_malformed_nested_tool_result() -> None:
    stream = _ChunkStream(
        [
            _frame(
                "tool_result",
                '{"name":"dft_judge","arguments":{},"result":{"status":1}}',
            ),
            _frame("message_delta", '{"text":"answer"}'),
            _frame(
                "complete",
                '{"tool_call_count":1,"rag_chunk_count":1,"ttft_ms":12,"synthesis_ms":34,"total_ms":56,"deadline_hit":false}',
            ),
        ]
    )

    with httpx.Client(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, stream=stream))
    ) as client:
        row = bench_latency.measure_question(
            client,
            "http://test",
            {"q_id": "T1", "question": "도구", "tool_calls_required": ["dft_judge"]},
            5.0,
        )

    assert row["path"] is None
    assert "tool_result" in str(row["error"])


def test_measure_question_rejects_contradictory_tool_outcomes() -> None:
    outcomes = (
        '{"status":"success","result":{},"error":{"code":"TIMEOUT"}}',
        '{"status":"error","result":{},"error":{"code":"TIMEOUT"}}',
    )
    for outcome in outcomes:
        stream = _ChunkStream(
            [
                _frame(
                    "tool_result",
                    f'{{"name":"dft_judge","arguments":{{}},"result":{outcome}}}',
                ),
                _frame("message_delta", '{"text":"answer"}'),
                _frame(
                    "complete",
                    '{"tool_call_count":1,"rag_chunk_count":1,"ttft_ms":12,"synthesis_ms":34,"total_ms":56,"deadline_hit":false}',
                ),
            ]
        )
        with httpx.Client(
            transport=httpx.MockTransport(lambda _request: httpx.Response(200, stream=stream))
        ) as client:
            row = bench_latency.measure_question(
                client,
                "http://test",
                {"q_id": "T1", "question": "도구", "tool_calls_required": ["dft_judge"]},
                5.0,
            )

        assert row["path"] is None
        assert "tool_result" in str(row["error"])


def test_measure_question_rejects_complete_without_answer_delta() -> None:
    stream = _ChunkStream(
        [
            _frame(
                "complete",
                '{"tool_call_count":1,"rag_chunk_count":1,"ttft_ms":12,"synthesis_ms":34,"total_ms":56,"deadline_hit":false}',
            )
        ]
    )

    with httpx.Client(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, stream=stream))
    ) as client:
        row = bench_latency.measure_question(
            client,
            "http://test",
            {"q_id": "T1", "question": "도구", "tool_calls_required": ["dft_judge"]},
            5.0,
        )

    assert row["path"] is None
    assert row["error"] == "complete event arrived before a valid message_delta"


def test_measure_question_rejects_coercive_and_non_finite_metrics() -> None:
    for total_ms in ('"56"', "1e999"):
        stream = _ChunkStream(
            [
                _frame("message_delta", '{"text":"answer"}'),
                _frame(
                    "complete",
                    (
                        '{"tool_call_count":0,"rag_chunk_count":1,"ttft_ms":12,'
                        f'"synthesis_ms":34,"total_ms":{total_ms},"deadline_hit":false}}'
                    ),
                ),
            ]
        )
        with httpx.Client(
            transport=httpx.MockTransport(lambda _request: httpx.Response(200, stream=stream))
        ) as client:
            row = bench_latency.measure_question(
                client,
                "http://test",
                {"q_id": "R1", "question": "질문", "tool_calls_required": []},
                5.0,
            )

        assert row["path"] is None
        assert "complete" in str(row["error"])


class _HeartbeatStream(httpx.SyncByteStream):
    def __iter__(self) -> Iterator[bytes]:
        while True:
            time.sleep(0.01)
            yield _frame("status", '{"stage":"heartbeat","chunks":0}')


def test_measure_question_enforces_overall_deadline_during_heartbeats() -> None:
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, stream=_HeartbeatStream())
        )
    ) as client:
        started = time.monotonic()
        row = bench_latency.measure_question(
            client,
            "http://test",
            {"q_id": "R1", "question": "질문", "tool_calls_required": []},
            0.025,
        )

    assert time.monotonic() - started < 0.2
    assert row["path"] is None
    assert row["error"] == "overall sample timeout after 0.025 seconds"
