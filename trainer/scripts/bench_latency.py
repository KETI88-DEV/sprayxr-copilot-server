#!/usr/bin/env python3
# pyright: reportAny=false, reportExplicitAny=false
"""Measure live demo latency through the user-visible HTTP/SSE boundary."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent))

_report_module = importlib.import_module("bench_report")
_sse_module = importlib.import_module("bench_sse")
build_report = _report_module.build_report
percentiles = _report_module.percentiles
render_summary = _report_module.render_summary
summarize = _report_module.summarize
BenchRow = _sse_module.BenchRow
EvalQuestion = _sse_module.EvalQuestion
classify_path = _sse_module.classify_path
measure_question = _sse_module.measure_question
parse_sse_frames = _sse_module.parse_sse_frames
BenchReport = _report_module.BenchReport

Profile = Literal["rag", "tools", "mixed"]
JsonObject = dict[str, Any]

logger = logging.getLogger(__name__)


def select_questions(questions: Sequence[EvalQuestion], profile: Profile) -> list[EvalQuestion]:
    """Filter by expected tools while preserving measured post-run classification."""
    if profile == "mixed":
        return list(questions)
    expects_tools = profile == "tools"
    return [question for question in questions if bool(question["tool_calls_required"]) is expects_tools]


class _EvalQuestionModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")
    q_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    tool_calls_required: list[str]
    context_data: JsonObject = Field(default_factory=dict)


def _load_questions(path: Path) -> list[EvalQuestion]:
    questions: list[EvalQuestion] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = _EvalQuestionModel.model_validate_json(line)
        questions.append(cast(EvalQuestion, payload.model_dump()))
    return questions


def run_benchmark(
    api_url: str,
    eval_set: Path,
    profile: Profile,
    repeat: int,
    output_root: Path,
    timeout_sec: float,
) -> Path:
    """Run sequentially so measurements represent the one-user latency gate."""
    questions = select_questions(_load_questions(eval_set), profile)
    if not questions:
        raise ValueError(f"no questions selected for profile {profile}")
    rows: list[BenchRow] = []
    with httpx.Client() as client:
        for repeat_index in range(repeat):
            for question in questions:
                logger.info("bench.question", extra={"q_id": question["q_id"], "repeat": repeat_index})
                row = measure_question(client, api_url, question, timeout_sec)
                rows.append(row)
                if row["error"] is not None:
                    logger.warning("bench.question_failed", extra={"q_id": row["q_id"], "error": row["error"]})

    output_dir = output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir.mkdir(parents=True, exist_ok=False)
    report = build_report(rows)
    write_report(output_dir, report)
    return output_dir


def write_report(output_dir: Path, report: BenchReport) -> None:
    report_json = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    _ = (output_dir / "report.json").write_text(report_json, encoding="utf-8")
    _ = (output_dir / "SUMMARY.md").write_text(render_summary(report), encoding="utf-8")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("expected a finite positive number")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--api-url", default="http://127.0.0.1:8130")
    _ = parser.add_argument("--eval-set", type=Path, default=Path("/data/eval/eval_set_agentic.jsonl"))
    _ = parser.add_argument("--profile", choices=("rag", "tools", "mixed"), default="mixed")
    _ = parser.add_argument("--repeat", type=_positive_int, default=1)
    _ = parser.add_argument("--output-root", type=Path, default=Path("/output/eval_results/latency_bench"))
    _ = parser.add_argument("--timeout", type=_positive_float, default=300.0)
    _ = parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO"
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    try:
        output_dir = run_benchmark(
            args.api_url, args.eval_set, args.profile, args.repeat, args.output_root, args.timeout
        )
    except (FileNotFoundError, ValidationError, ValueError) as exc:
        message = f"eval set not found: {args.eval_set}" if isinstance(exc, FileNotFoundError) else str(exc)
        logger.error(message)
        return 2
    logger.info("bench.complete", extra={"output_dir": str(output_dir)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
