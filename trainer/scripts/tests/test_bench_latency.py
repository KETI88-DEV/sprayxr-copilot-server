#!/usr/bin/env python3
# pyright: reportAny=false

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

bench_latency = importlib.import_module("bench_latency")


def test_classify_path_uses_actual_tool_count() -> None:
    assert bench_latency.classify_path(0) == "rag"
    assert bench_latency.classify_path(1) == "tools"
    assert bench_latency.classify_path(3) == "tools"


def test_percentiles_use_inclusive_upper_index() -> None:
    got = bench_latency.percentiles([10.0, 20.0, 30.0, 40.0])

    assert got == {
        "n": 4,
        "p50": 30.0,
        "p90": 40.0,
        "p95": 40.0,
        "p99": 40.0,
        "max": 40.0,
    }


def test_percentiles_return_no_data_without_samples() -> None:
    assert bench_latency.percentiles([]) == {
        "n": 0,
        "p50": None,
        "p90": None,
        "p95": None,
        "p99": None,
        "max": None,
    }


def test_summarize_uses_path_specific_budget_metrics() -> None:
    rows = [
        {"q_id": "A1", "path": "rag", "ttft_ms": 12_000.0, "total_ms": 20_000.0},
        {"q_id": "A2", "path": "rag", "ttft_ms": 41_000.0, "total_ms": 60_000.0},
        {"q_id": "A3", "path": "tools", "ttft_ms": 5_000.0, "total_ms": 200_000.0},
    ]

    got = bench_latency.summarize(rows)

    assert got["rag"]["metric"] == "ttft_ms"
    assert got["rag"]["breaches"] == ["A2"]
    assert got["tools"]["metric"] == "total_ms"
    assert got["tools"]["breaches"] == ["A3"]


def test_summarize_marks_missing_path_as_no_data() -> None:
    got = bench_latency.summarize(
        [{"q_id": "A1", "path": "rag", "ttft_ms": 12_000.0, "total_ms": 20_000.0}]
    )

    assert got["tools"]["n"] == 0
    assert got["tools"]["measured_n"] == 0
    assert got["tools"]["compliance_rate"] is None
    assert got["tools"]["gate_status"] == "no_data"


def test_parse_sse_frames_handles_backend_envelope() -> None:
    frames = bench_latency.parse_sse_frames(
        [
            'event: message_delta\ndata: {"event":"message_delta","data":{"text":"첫 토큰"}}\n\n',
            'event: complete\ndata: {"event":"complete","data":{"tool_call_count":1,"rag_chunk_count":2,"ttft_ms":12,"synthesis_ms":34,"total_ms":56,"deadline_hit":false}}\n\n',
        ]
    )

    assert frames[0] == ("message_delta", {"text": "첫 토큰"})
    assert frames[1][0] == "complete"
    assert frames[1][1]["tool_call_count"] == 1


def test_build_report_includes_required_global_metrics() -> None:
    rows = [
        {
            "q_id": "R1",
            "path": "rag",
            "expected_path": "rag",
            "ttft_ms": 10_000.0,
            "total_ms": 20_000.0,
            "synthesis_ms": 0.0,
            "tool_call_count": 0,
            "rag_chunk_count": 2,
            "deadline_hit": False,
            "degraded": False,
            "error": None,
            "mcp_timeout_count": 0,
        },
        {
            "q_id": "T1",
            "path": "tools",
            "expected_path": "tools",
            "ttft_ms": 30_000.0,
            "total_ms": 170_000.0,
            "synthesis_ms": 20_000.0,
            "tool_call_count": 1,
            "rag_chunk_count": 3,
            "deadline_hit": True,
            "degraded": True,
            "error": None,
            "mcp_timeout_count": 0,
        },
    ]

    report = bench_latency.build_report(rows)

    assert report["ttft_ms"]["p50"] == 30_000.0
    assert report["total_ms"]["max"] == 170_000.0
    assert report["synthesis_ms"]["p95"] == 20_000.0
    assert report["deadline_hit_count"] == 1
    assert report["degraded_count"] == 1
    assert report["by_path"]["rag"]["breaches"] == []
    assert report["by_path"]["rag"]["compliance_rate"] == 1.0
    assert "| R1 | rag | rag |" in bench_latency.render_summary(report)


def test_select_questions_uses_expected_tool_requirements() -> None:
    questions: list[dict[str, str | list[str]]] = [
        {"q_id": "R1", "question": "지식 질문", "tool_calls_required": []},
        {"q_id": "T1", "question": "도구 질문", "tool_calls_required": ["dft_judge"]},
    ]

    assert [item["q_id"] for item in bench_latency.select_questions(questions, "rag")] == ["R1"]
    assert [item["q_id"] for item in bench_latency.select_questions(questions, "tools")] == ["T1"]
    assert len(bench_latency.select_questions(questions, "mixed")) == 2


def test_report_excludes_missing_and_timeout_measurements_from_gates() -> None:
    rows = [
        {
            "q_id": "R1",
            "path": "rag",
            "ttft_ms": None,
            "total_ms": 20_000.0,
            "synthesis_ms": 0.0,
            "tool_call_count": 0,
            "mcp_timeout_count": 0,
            "error": None,
        },
        {
            "q_id": "T1",
            "path": "tools",
            "ttft_ms": 10_000.0,
            "total_ms": 100_000.0,
            "synthesis_ms": 5_000.0,
            "tool_call_count": 1,
            "mcp_timeout_count": 1,
            "error": None,
        },
    ]

    report = bench_latency.build_report(rows)

    assert report["ttft_ms"]["n"] == 0
    assert report["by_path"]["rag"]["measured_n"] == 0
    assert report["by_path"]["rag"]["gate_status"] == "invalid"
    assert report["by_path"]["tools"]["n"] == 1
    assert report["by_path"]["tools"]["measured_n"] == 0
    assert report["by_path"]["tools"]["invalid_n"] == 1
    assert report["invalid_measurement_count"] == 2


def test_render_summary_keeps_multiline_errors_inside_one_table_row() -> None:
    rows = [
        {
            "q_id": "R1",
            "path": None,
            "expected_path": "rag",
            "ttft_ms": None,
            "total_ms": 10.0,
            "synthesis_ms": None,
            "tool_call_count": 0,
            "mcp_timeout_count": 0,
            "error": "invalid complete\nfield required | malformed",
        }
    ]

    summary = bench_latency.render_summary(bench_latency.build_report(rows))

    assert "invalid complete<br>field required \\| malformed" in summary


def test_report_marks_error_contaminated_path_and_run_not_ready() -> None:
    rows = [
        {
            "q_id": "T1",
            "path": "tools",
            "expected_path": "tools",
            "ttft_ms": 10.0,
            "total_ms": 100.0,
            "synthesis_ms": 20.0,
            "tool_call_count": 1,
            "mcp_timeout_count": 0,
            "error": None,
        },
        {
            "q_id": "T2",
            "path": None,
            "expected_path": "tools",
            "ttft_ms": None,
            "total_ms": 50.0,
            "synthesis_ms": None,
            "tool_call_count": 0,
            "mcp_timeout_count": 0,
            "error": "missing terminal event",
        },
    ]

    report = bench_latency.build_report(rows)

    assert report["unclassified_invalid_count"] == 1
    assert report["gate_ready"] is False
    assert report["by_path"]["tools"]["n"] == 2
    assert report["by_path"]["tools"]["invalid_n"] == 1
    assert report["by_path"]["tools"]["gate_status"] == "invalid"


def test_report_requires_measurement_for_every_expected_path() -> None:
    rows = [
        {
            "q_id": "R1",
            "path": "rag",
            "expected_path": "rag",
            "ttft_ms": 10.0,
            "total_ms": 20.0,
            "synthesis_ms": 5.0,
            "tool_call_count": 0,
            "mcp_timeout_count": 0,
            "error": None,
        },
        {
            "q_id": "T1",
            "path": "rag",
            "expected_path": "tools",
            "ttft_ms": 11.0,
            "total_ms": 21.0,
            "synthesis_ms": 5.0,
            "tool_call_count": 0,
            "mcp_timeout_count": 0,
            "error": None,
        },
    ]

    report = bench_latency.build_report(rows)

    assert report["invalid_measurement_count"] == 0
    assert report["by_path"]["tools"]["gate_status"] == "no_data"
    assert report["gate_ready"] is False


def test_measure_question_defaults_to_normal_and_ignores_answer_timeout_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        return httpx.Response(
            200,
            text=(
                'event: message_delta\ndata: {"event":"message_delta","data":{"text":"timeout 안내"}}\n\n'
                'event: complete\ndata: {"event":"complete","data":{"tool_call_count":0,"rag_chunk_count":2,"ttft_ms":12,"synthesis_ms":34,"total_ms":56,"deadline_hit":false}}\n\n'
            ),
        )

    question: dict[str, str | list[str]] = {
        "q_id": "R1",
        "question": "지식 질문",
        "tool_calls_required": [],
    }
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        row = bench_latency.measure_question(client, "http://test", question, 5.0)

    assert row["degraded"] is False
    assert row["mcp_timeout_count"] == 0
