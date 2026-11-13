#!/usr/bin/env python3
"""FP8 품질 게이트 판정 (계획 D4, 표준 라이브러리만).

사전등록한 단계적 신뢰구간 규칙(2026-09-15, 결과를 보기 전에 고정)으로 후보(Spark vLLM FP8)를
기준선(A40 HF NF4)과 비교한다.

- 차이 d = 후보 실행 평균들의 평균 − 기준선 실행 평균들의 평균.
- Welch 표준오차 se = sqrt(s_c²/n_c + s_b²/n_b), Welch–Satterthwaite 자유도, 양측 95% 신뢰구간 d ± t·se.
- pass: CI 하한 > −Δ (exit 0).  fail: CI 상한 < −Δ (exit 1).  그 외 inconclusive (exit 3).
- 단계: 1단계 각 ≥8회. inconclusive 이면 양쪽을 각 18회(--target-runs)로 늘려(1단계 실행 포함) 같은 규칙으로
  한 번 더 판정하고 그 결과로 확정한다. 2단계에서도 inconclusive 면 "판정 불가 — 사용자 결정"이다.
- 한계: 같은 규칙을 두 번 확인하므로 1종 오류가 조금 늘어난다. 두 번뿐이라 작다고 보고 보정하지 않는다.

입력은 run_agentic_eval.py가 쓰는 ``agentic_eval_results.csv``다. 루트 아래 ``seed_<n>/`` 디렉터리 하나가
실행 한 번이고, 그 아래(깊이 무관)에 CSV가 정확히 하나 있어야 한다. 실행 점수는 ``total`` 열의 평균으로,
run_agentic_eval의 average_total_score와 같은 계산이다(누락 응답은 0점으로 포함).

    python3 quality_gate.py --baseline-root /output/eval_results/qwen36_27b_v6_1_nf4_baseline_965549e \\
        --candidate-root results/qwen36_27b_v6_1_fp8_spark_<commit>

종료 코드: 0 pass, 1 fail, 2 입력 불완전(실행 수·문항 수·문항 집합·CSV 개수), 3 inconclusive.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

RESULTS_CSV = "agentic_eval_results.csv"
CONFIDENCE = 0.95
Z_ALPHA_TWO_SIDED = 1.96  # α = 0.05 양측
Z_POWER = 0.84  # 검정력 0.8
EXIT_PASS, EXIT_FAIL, EXIT_INCOMPLETE, EXIT_INCONCLUSIVE = 0, 1, 2, 3
LIMITATION = (
    "사전등록한 단계 규칙은 같은 신뢰구간 규칙을 최대 두 번(1단계, 2단계) 확인한다. 반복 확인에 따른 "
    "1종 오류 증가는 두 번뿐이라 작다고 보고 보정하지 않았다."
)


def _continued_fraction_beta(a: float, b: float, x: float) -> float:
    """정규화 불완전 베타 함수의 연분수 부분(Lentz 방법)."""
    tiny = 1e-300
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        for numerator in (
            m * (b - m) * x / ((a + m2 - 1.0) * (a + m2)),
            -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1.0)),
        ):
            d = 1.0 + numerator * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + numerator / c
            c = c if abs(c) > tiny else tiny
            step = c * d
            h *= step
        if abs(step - 1.0) < 1e-15:
            break
    return h


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _continued_fraction_beta(a, b, x) / a
    return 1.0 - front * _continued_fraction_beta(b, a, 1.0 - x) / b


def t_two_sided_tail(t: float, df: float) -> float:
    """P(|T| > t), T ~ Student t(df)."""
    return regularized_incomplete_beta(df / 2.0, 0.5, df / (df + t * t))


def t_quantile_975(df: float) -> float:
    """Student t 분포의 0.975 분위수(양측 95%). df는 실수(Welch 자유도)이며 math.inf는 정규분포."""
    if df <= 0:
        raise ValueError(f"degrees of freedom must be positive, got {df}")
    if math.isinf(df):
        return statistics.NormalDist().inv_cdf(0.975)
    tail = 1.0 - CONFIDENCE
    low, high = 0.0, 1.0
    while t_two_sided_tail(high, df) > tail:
        high *= 2.0
    for _ in range(200):
        mid = (low + high) / 2.0
        if t_two_sided_tail(mid, df) > tail:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


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


def welch_interval(baseline: Sequence[float], candidate: Sequence[float]) -> dict[str, float | None]:
    n_b, n_c = len(baseline), len(candidate)
    w_b = statistics.variance(baseline) / n_b
    w_c = statistics.variance(candidate) / n_c
    difference = statistics.fmean(candidate) - statistics.fmean(baseline)
    se = math.sqrt(w_b + w_c)
    denominator = w_b**2 / (n_b - 1) + w_c**2 / (n_c - 1)
    df = (w_b + w_c) ** 2 / denominator if denominator > 0 else math.inf
    t_crit = t_quantile_975(df)
    return {
        "difference": difference,
        "se": se,
        "df": None if math.isinf(df) else df,  # JSON에는 무한대가 없다. null = 정규 근사
        "t_crit": t_crit,
        "ci_low": difference - t_crit * se,
        "ci_high": difference + t_crit * se,
    }


def required_runs(baseline: Sequence[float], candidate: Sequence[float], delta: float) -> tuple[int, float]:
    """현재 산포에서 Δ를 α=0.05(양측)·검정력 0.8로 가려내는 팔당 반복 수 n ≈ 2·(1.96+0.84)²·s_pooled²/Δ²."""
    n_b, n_c = len(baseline), len(candidate)
    pooled = ((n_b - 1) * statistics.variance(baseline) + (n_c - 1) * statistics.variance(candidate)) / (n_b + n_c - 2)
    raw = 2 * (Z_ALPHA_TWO_SIDED + Z_POWER) ** 2 * pooled / delta**2
    return max(2, math.ceil(raw)), raw


def evaluate(
    baseline_root: Path,
    candidate_root: Path,
    *,
    delta: float,
    min_runs: int,
    target_runs: int,
    expect_questions: int,
) -> tuple[int, dict[str, Any]]:
    if min_runs < 2:
        raise ValueError("min_runs must be at least 2 (a standard deviation needs two runs)")
    baseline, problems = discover_runs(baseline_root)
    candidate, candidate_problems = discover_runs(candidate_root)
    problems += candidate_problems
    for label, runs in (("baseline", baseline), ("candidate", candidate)):
        if len(runs) < min_runs:
            problems.append(f"{label}: {len(runs)} runs, expected at least {min_runs}")
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
        return EXIT_INCOMPLETE, report

    base_means = [run["mean"] for run in baseline]
    cand_means = [run["mean"] for run in candidate]
    interval = welch_interval(base_means, cand_means)
    stage = 2 if min(len(baseline), len(candidate)) >= target_runs else 1
    runs_needed, runs_needed_raw = required_runs(base_means, cand_means, delta)

    if interval["ci_low"] > -delta:
        verdict, code = "pass", EXIT_PASS
        decision = f"{stage}단계 pass 확정: CI 하한 {interval['ci_low']:.3f} > −Δ = {-delta:.3f}"
    elif interval["ci_high"] < -delta:
        verdict, code = "fail", EXIT_FAIL
        decision = f"{stage}단계 fail 확정: CI 상한 {interval['ci_high']:.3f} < −Δ = {-delta:.3f}"
    else:
        verdict, code = "inconclusive", EXIT_INCONCLUSIVE
        span = f"CI [{interval['ci_low']:.3f}, {interval['ci_high']:.3f}]가 −Δ = {-delta:.3f}를 걸친다"
        if stage == 1:
            decision = (
                f"1단계 inconclusive: {span}. 양쪽을 팔당 {target_runs}회(--target-runs)로 늘려 같은 규칙으로 재판정한다"
                f"(1단계 실행 포함, 추가분만 실행). 현재 SD 기준 필요 반복 수 n ≈ {runs_needed_raw:.1f} → {runs_needed}."
            )
        else:
            decision = f"2단계 inconclusive: {span}. 판정 불가 — 사용자 결정."

    per_question = []
    for q_id in baseline[0]["question_ids"]:
        b = statistics.fmean(run["totals"][q_id] for run in baseline)
        c = statistics.fmean(run["totals"][q_id] for run in candidate)
        per_question.append({"q_id": q_id, "baseline_mean": round(b, 3), "candidate_mean": round(c, 3), "diff": round(c - b, 3)})
    per_question.sort(key=lambda item: item["diff"])
    report.update(
        stage=stage,
        target_runs=target_runs,
        n_baseline=len(baseline),
        n_candidate=len(candidate),
        baseline=_side_summary(baseline),
        candidate=_side_summary(candidate),
        **interval,
        verdict=verdict,
        decision=decision,
        required_runs_estimate=runs_needed,
        required_runs_raw=round(runs_needed_raw, 2),
        limitation=LIMITATION,
        per_question=per_question,
    )
    return code, report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--delta", type=float, default=0.2)
    parser.add_argument("--min-runs", type=int, default=8, help="each side needs at least this many runs (stage 1)")
    parser.add_argument("--target-runs", type=int, default=18, help="runs per side for stage 2")
    parser.add_argument("--expect-questions", type=int, default=30)
    parser.add_argument("--output", type=Path, default=None, help="also write the JSON report here")
    args = parser.parse_args(argv)
    if args.min_runs < 2:
        parser.error("--min-runs must be at least 2")
    code, report = evaluate(
        args.baseline_root,
        args.candidate_root,
        delta=args.delta,
        min_runs=args.min_runs,
        target_runs=args.target_runs,
        expect_questions=args.expect_questions,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
