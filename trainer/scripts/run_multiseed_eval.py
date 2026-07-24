#!/usr/bin/env python3
"""Run N-seed agentic eval and report per-seed + mean/std for reproducibility verification."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _score_summary(csv_path: Path) -> dict[str, float]:
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    if not rows:
        return {}
    total_col = next((c for c in rows[0] if c.lower() in ("total", "total_score")), "total")
    dims = ["tool_selection_accuracy", "parameter_correctness", "reasoning_depth",
            "context_integration", "response_structure"]
    return {
        "total": statistics.mean(float(r[total_col]) for r in rows),
        **{d: statistics.mean(float(r[d]) for r in rows) for d in dims},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run agentic eval with multiple seeds and report reproducibility.")
    parser.add_argument("--eval-set", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--mcp-url", type=str, default="http://localhost:8100")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tool-schemas-dir", type=Path,
                        default=Path("/workspace/trainer/configs/tool_schemas"))
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    args.output_root.mkdir(parents=True, exist_ok=True)
    per_seed_scores: dict[int, dict[str, float]] = {}

    for seed in args.seeds:
        seed_dir = args.output_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        resp = seed_dir / "agentic_responses.jsonl"

        logger.info("=== SEED %d ===", seed)
        gen_cmd = [
            sys.executable, "trainer/scripts/generate_agentic_responses.py",
            "--eval-set", str(args.eval_set),
            "--config", str(args.config),
            "--adapter", str(args.adapter),
            "--mcp-url", args.mcp_url,
            "--output", str(resp),
            "--seed", str(seed),
        ]
        rc = subprocess.call(gen_cmd)
        if rc != 0:
            logger.error("Gen failed for seed %d (rc=%d)", seed, rc)
            continue

        score_cmd = [
            sys.executable, "trainer/scripts/run_agentic_eval.py",
            "--responses", str(resp),
            "--output-dir", str(seed_dir),
            "--tool-schemas-dir", str(args.tool_schemas_dir),
        ]
        rc = subprocess.call(score_cmd)
        if rc != 0:
            logger.error("Score failed for seed %d (rc=%d)", seed, rc)
            continue

        summary = _score_summary(seed_dir / "agentic_eval_results.csv")
        per_seed_scores[seed] = summary
        logger.info("seed=%d summary=%s", seed, summary)

    if not per_seed_scores:
        logger.error("No successful seed runs")
        return 1

    report = {"per_seed": per_seed_scores, "aggregate": {}}
    dims = ["total", "tool_selection_accuracy", "parameter_correctness", "reasoning_depth",
            "context_integration", "response_structure"]
    for d in dims:
        vals = [s[d] for s in per_seed_scores.values() if d in s]
        report["aggregate"][d] = {
            "mean": round(statistics.mean(vals), 4),
            "std": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
        }

    (args.output_root / "multiseed_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    print("=" * 70)
    print(f"MULTI-SEED AGENTIC EVAL — {len(per_seed_scores)} seeds")
    print("=" * 70)
    print(f"{'dim':30s} {'mean':>7s} {'std':>7s} {'min':>7s} {'max':>7s}")
    for d in dims:
        a = report["aggregate"][d]
        print(f"{d:30s} {a['mean']:>7.3f} {a['std']:>7.3f} {a['min']:>7.3f} {a['max']:>7.3f}")
    print("=" * 70)
    for seed, s in per_seed_scores.items():
        print(f"seed {seed}: total={s.get('total', 0):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
