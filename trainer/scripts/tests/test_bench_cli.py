#!/usr/bin/env python3
# pyright: reportAny=false

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from math import nan
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

bench_latency = importlib.import_module("bench_latency")
SCRIPT = Path(__file__).resolve().parents[1] / "bench_latency.py"


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def _write_eval(path: Path) -> None:
    _ = path.write_text(
        json.dumps({"q_id": "R1", "question": "질문", "tool_calls_required": []}) + "\n",
        encoding="utf-8",
    )


def test_cli_reports_missing_eval_set_without_traceback(tmp_path: Path) -> None:
    result = _run_cli(
        "--eval-set",
        str(tmp_path / "missing.jsonl"),
        "--output-root",
        str(tmp_path / "output"),
    )

    assert result.returncode == 2
    assert "eval set not found" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_rejects_non_positive_repeat(tmp_path: Path) -> None:
    eval_set = tmp_path / "eval.jsonl"
    _write_eval(eval_set)

    result = _run_cli("--eval-set", str(eval_set), "--repeat", "0")

    assert result.returncode == 2
    assert "positive integer" in result.stderr


def test_cli_rejects_profile_without_questions_before_creating_artifacts(tmp_path: Path) -> None:
    eval_set = tmp_path / "eval.jsonl"
    output_root = tmp_path / "output"
    _write_eval(eval_set)

    result = _run_cli(
        "--eval-set",
        str(eval_set),
        "--profile",
        "tools",
        "--output-root",
        str(output_root),
    )

    assert result.returncode == 2
    assert "no questions selected" in result.stderr
    assert not output_root.exists()


def test_write_report_creates_json_and_markdown_artifacts(tmp_path: Path) -> None:
    report = bench_latency.build_report([])

    bench_latency.write_report(tmp_path, report)

    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))["sample_count"] == 0
    assert "0 total / 0 valid / 0 invalid" in (tmp_path / "SUMMARY.md").read_text(
        encoding="utf-8"
    )


def test_cli_rejects_non_finite_timeout(tmp_path: Path) -> None:
    eval_set = tmp_path / "eval.jsonl"
    _write_eval(eval_set)

    for value in ("nan", "inf"):
        result = _run_cli("--eval-set", str(eval_set), "--timeout", value)
        assert result.returncode == 2
        assert "finite positive number" in result.stderr


def test_write_report_rejects_non_standard_json_numbers(tmp_path: Path) -> None:
    report = bench_latency.build_report([])
    report["ttft_ms"]["p50"] = nan

    try:
        bench_latency.write_report(tmp_path, report)
    except ValueError as exc:
        assert "Out of range float values" in str(exc)
    else:
        raise AssertionError("write_report accepted NaN")

    assert not (tmp_path / "report.json").exists()
