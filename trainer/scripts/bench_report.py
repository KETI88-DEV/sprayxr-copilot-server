#!/usr/bin/env python3
"""Pure latency aggregation and report rendering for the SSE benchmark."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, Literal, NotRequired, TypedDict

RAG_BUDGET_MS: Final[int] = 30_000
TOOL_BUDGET_MS: Final[int] = 180_000
PathName = Literal["rag", "tools"]
Profile = Literal["rag", "tools", "mixed"]
JsonObject = dict[str, object]
MetricName = Literal["ttft_ms", "total_ms"]


class BenchRow(TypedDict):
    q_id: str
    path: PathName | None
    ttft_ms: float | None
    total_ms: float
    synthesis_ms: float | None
    tool_call_count: int
    expected_path: NotRequired[PathName]
    backend_ttft_ms: NotRequired[float | None]
    backend_total_ms: NotRequired[float | None]
    rag_chunk_count: NotRequired[int]
    deadline_hit: NotRequired[bool]
    degraded: NotRequired[bool]
    mcp_timeout_count: NotRequired[int]
    error: NotRequired[str | None]


class PercentileSummary(TypedDict):
    n: int
    p50: float | None
    p90: float | None
    p95: float | None
    p99: float | None
    max: float | None


class PathSummary(TypedDict):
    n: int
    measured_n: int
    invalid_n: int
    metric: str
    budget_ms: int
    p50: float | None
    p90: float | None
    p95: float | None
    p99: float | None
    max: float | None
    within_budget: int
    compliance_rate: float | None
    gate_status: Literal["no_data", "pass", "fail", "invalid"]
    breaches: list[str]


SummaryMap = dict[PathName, PathSummary]


class BenchReport(TypedDict):
    sample_count: int
    valid_measurement_count: int
    invalid_measurement_count: int
    unclassified_invalid_count: int
    gate_ready: bool
    ttft_ms: PercentileSummary
    total_ms: PercentileSummary
    synthesis_ms: PercentileSummary
    by_path: SummaryMap
    deadline_hit_count: int
    degraded_count: int
    mcp_timeout_count: int
    error_count: int
    rows: Sequence[BenchRow]


_PATH_BUDGETS: Final[tuple[tuple[PathName, int, MetricName], ...]] = (
    ("rag", RAG_BUDGET_MS, "ttft_ms"),
    ("tools", TOOL_BUDGET_MS, "total_ms"),
)


def percentiles(values: Sequence[float]) -> PercentileSummary:
    """Use the plan's conservative inclusive upper-index percentile rule."""
    if not values:
        return {"n": 0, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    ordered = sorted(values)

    def at(quantile: float) -> float:
        return ordered[min(int(len(ordered) * quantile), len(ordered) - 1)]

    return {
        "n": len(ordered),
        "p50": at(0.50),
        "p90": at(0.90),
        "p95": at(0.95),
        "p99": at(0.99),
        "max": ordered[-1],
    }


def _metric_value(row: BenchRow, metric: MetricName) -> float | None:
    return row["ttft_ms"] if metric == "ttft_ms" else row["total_ms"]


def _gate_eligible(row: BenchRow) -> bool:
    if row.get("error") is not None or row.get("mcp_timeout_count", 0) > 0:
        return False
    path = row["path"]
    if path == "rag":
        return row["ttft_ms"] is not None
    return path == "tools"


def summarize(rows: Sequence[BenchRow]) -> SummaryMap:
    """Apply the RAG TTFT and tool completion budgets independently."""
    output: SummaryMap = {}
    for path, budget, metric in _PATH_BUDGETS:
        subset = [
            row
            for row in rows
            if row["path"] == path
            or (row["path"] is None and row.get("expected_path") == path)
        ]
        measured = [row for row in subset if _gate_eligible(row)]
        values = [value for row in measured if (value := _metric_value(row, metric)) is not None]
        within_budget = sum(value <= budget for value in values)
        measured_n = len(values)
        breaches = [
            row["q_id"]
            for row in measured
            if (value := _metric_value(row, metric)) is not None and value > budget
        ]
        distribution = percentiles(values)
        output[path] = {
            "n": len(subset),
            "measured_n": measured_n,
            "invalid_n": len(subset) - measured_n,
            "metric": metric,
            "budget_ms": budget,
            "p50": distribution["p50"],
            "p90": distribution["p90"],
            "p95": distribution["p95"],
            "p99": distribution["p99"],
            "max": distribution["max"],
            "within_budget": within_budget,
            "compliance_rate": within_budget / measured_n if measured_n else None,
            "gate_status": "invalid"
            if len(subset) > measured_n
            else "no_data"
            if measured_n == 0
            else "fail"
            if breaches
            else "pass",
            "breaches": breaches,
        }
    return output


def build_report(rows: Sequence[BenchRow]) -> BenchReport:
    """Build all metrics required for deadline recalibration and gate judgment."""
    valid_rows = [row for row in rows if _gate_eligible(row)]
    invalid_count = len(rows) - len(valid_rows)
    by_path = summarize(rows)
    required_paths: set[PathName] = set()
    for row in rows:
        expected_path = row.get("expected_path")
        if expected_path is not None:
            required_paths.add(expected_path)
        elif row["path"] is not None:
            required_paths.add(row["path"])
    has_required_coverage = all(by_path[path]["measured_n"] > 0 for path in required_paths)
    return {
        "sample_count": len(rows),
        "valid_measurement_count": len(valid_rows),
        "invalid_measurement_count": invalid_count,
        "unclassified_invalid_count": sum(row["path"] is None for row in rows),
        "gate_ready": bool(valid_rows) and invalid_count == 0 and has_required_coverage,
        "ttft_ms": percentiles(
            [row["ttft_ms"] for row in valid_rows if row["ttft_ms"] is not None]
        ),
        "total_ms": percentiles([row["total_ms"] for row in valid_rows]),
        "synthesis_ms": percentiles(
            [row["synthesis_ms"] for row in valid_rows if row["synthesis_ms"] is not None]
        ),
        "by_path": by_path,
        "deadline_hit_count": sum(row.get("deadline_hit", False) for row in rows),
        "degraded_count": sum(row.get("degraded", False) for row in rows),
        "mcp_timeout_count": sum(row.get("mcp_timeout_count", 0) for row in rows),
        "error_count": sum(row.get("error") is not None for row in rows),
        "rows": rows,
    }


def render_summary(report: BenchReport) -> str:
    lines = [
        "# Latency Bench Summary",
        "",
        f"Samples: {report['sample_count']} total / {report['valid_measurement_count']} valid / {report['invalid_measurement_count']} invalid",
        f"Gate ready: {report['gate_ready']} / unclassified invalid: {report['unclassified_invalid_count']}",
        "",
        "| Path | Total N | Measured N | Invalid N | Metric | Budget ms | Gate | Compliance | p50 | p90 | p99 | Breaches |",
        "|---|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for path in ("rag", "tools"):
        item = report["by_path"][path]
        lines.append(
            f"| {path} | {item['n']} | {item['measured_n']} | {item['invalid_n']} | "
            + f"{item['metric']} | {item['budget_ms']} | {item['gate_status']} | "
            + f"{item.get('compliance_rate', 'n/a')} | "
            + f"{item.get('p50', 'n/a')} | {item.get('p90', 'n/a')} | {item.get('p99', 'n/a')} | "
            + f"{item.get('breaches', [])} |"
        )
    lines.extend(
        [
            "",
            f"Deadline hits: {report['deadline_hit_count']}",
            f"Degraded streams: {report['degraded_count']}",
            f"MCP timeouts: {report['mcp_timeout_count']}",
            f"Errors: {report['error_count']}",
            f"Synthesis p95: {report['synthesis_ms']['p95']} ms",
        ]
    )
    lines.extend(
        [
            "",
            "| q_id | Expected | Actual | Tools | TTFT ms | Synthesis ms | Total ms | Deadline | Degraded | Error |",
            "|---|---|---|---:|---:|---:|---:|---|---|---|",
        ]
    )
    for row in report["rows"]:
        error = str(row.get("error") or "").replace("|", "\\|").replace("\n", "<br>")
        lines.append(
            f"| {row['q_id']} | {row.get('expected_path', 'n/a')} | {row['path'] or 'invalid'} | "
            + f"{row['tool_call_count']} | {row['ttft_ms']} | {row['synthesis_ms']} | "
            + f"{row['total_ms']} | {row.get('deadline_hit', False)} | "
            + f"{row.get('degraded', False)} | {error} |"
        )
    return "\n".join(lines) + "\n"
