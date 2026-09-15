#!/usr/bin/env python3
"""FP8 품질 게이트 판정 (계획 D4, 표준 라이브러리만).

후보(Spark vLLM FP8) 반복 실행 평균이 기준선(A40 HF NF4) 평균 − Δ 이상이면 합격이다.

입력은 run_agentic_eval.py가 쓰는 ``agentic_eval_results.csv``다. 루트 아래 ``seed_<n>/`` 디렉터리 하나가
실행 한 번이고, 그 아래(깊이 무관)에 CSV가 정확히 하나 있어야 한다. 실행 점수는 ``total`` 열의 평균으로,
run_agentic_eval의 average_total_score와 같은 계산이다(누락 응답은 0점으로 포함).

    python3 quality_gate.py --baseline-root /output/eval_results/qwen36_27b_v6_1_nf4_baseline_965549e \\
        --candidate-root results/qwen36_27b_v6_1_fp8_spark_<commit>

종료 코드: 0 합격, 1 불합격, 2 입력 불완전(실행 수·문항 수·문항 집합·CSV 개수).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

RESULTS_CSV = "agentic_eval_results.csv"
_EPSILON = 1e-9


def load_run(csv_path: Path) -> dict[str, Any]:
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    totals = {row["q_id"]: int(row["total"]) for row in rows}
    return {
        "csv": str(csv_path),
        "rows": len(rows),
        "question_ids": sorted(totals),
        "totals": totals,
        "mean": statistics.fmean(totals.values()) if totals else 0.0,
        "missing_responses": sum(1 for row in rows if row.get("note") == "missing_response"),
        "duplicate_ids": len(rows) != len(totals),
    }


def discover_runs(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    problems: list[str] = []
    runs: list[dict[str, Any]] = []
    if not root.is_dir():
        return runs, [f"{root}: not a directory"]
    for seed_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("seed_")):
        found = sorted(seed_dir.rglob(RESULTS_CSV))
        if len(found) != 1:
            problems.append(f"{seed_dir}: expected exactly one {RESULTS_CSV}, found {len(found)}")
            continue
        run = load_run(found[0])
        run["seed"] = seed_dir.name[len("seed_") :]
        runs.append(run)
    return runs, problems


def _side_summary(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    means = [run["mean"] for run in runs]
    return {
        "runs": [
            {"seed": r["seed"], "mean": round(r["mean"], 4), "rows": r["rows"], "missing_responses": r["missing_responses"]}
            for r in runs
        ],
        "mean": statistics.fmean(means) if means else 0.0,
        "sd": statistics.stdev(means) if len(means) > 1 else 0.0,
    }


def evaluate(
    baseline_root: Path,
    candidate_root: Path,
    *,
    delta: float,
    expect_runs: int,
    expect_questions: int,
) -> tuple[int, dict[str, Any]]:
    baseline, problems = discover_runs(baseline_root)
    candidate, candidate_problems = discover_runs(candidate_root)
    problems += candidate_problems
    for label, runs in (("baseline", baseline), ("candidate", candidate)):
        if len(runs) != expect_runs:
            problems.append(f"{label}: {len(runs)} runs, expected {expect_runs}")
        for run in runs:
            if run["rows"] != expect_questions or run["duplicate_ids"]:
                problems.append(f"{label} seed_{run['seed']}: {run['rows']} rows, expected {expect_questions} unique questions")
    question_sets = {tuple(run["question_ids"]) for run in (*baseline, *candidate)}
    if len(question_sets) > 1:
        problems.append("question ids differ between runs")

    report: dict[str, Any] = {
        "baseline_root": str(baseline_root),
        "candidate_root": str(candidate_root),
        "delta": delta,
        "problems": problems,
    }
    if problems:
        report["verdict"] = "incomplete"
        return 2, report

    base = _side_summary(baseline)
    cand = _side_summary(candidate)
    threshold = base["mean"] - delta
    passed = cand["mean"] >= threshold - _EPSILON
    per_question = []
    for q_id in baseline[0]["question_ids"]:
        b = statistics.fmean(run["totals"][q_id] for run in baseline)
        c = statistics.fmean(run["totals"][q_id] for run in candidate)
        per_question.append({"q_id": q_id, "baseline_mean": round(b, 3), "candidate_mean": round(c, 3), "diff": round(c - b, 3)})
    per_question.sort(key=lambda item: item["diff"])
    report.update(
        baseline=base,
        candidate=cand,
        threshold=threshold,
        difference=cand["mean"] - base["mean"],
        verdict="pass" if passed else "fail",
        per_question=per_question,
    )
    return (0 if passed else 1), report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--delta", type=float, default=0.2)
    parser.add_argument("--expect-runs", type=int, default=8)
    parser.add_argument("--expect-questions", type=int, default=30)
    parser.add_argument("--output", type=Path, default=None, help="also write the JSON report here")
    args = parser.parse_args(argv)
    code, report = evaluate(
        args.baseline_root,
        args.candidate_root,
        delta=args.delta,
        expect_runs=args.expect_runs,
        expect_questions=args.expect_questions,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
