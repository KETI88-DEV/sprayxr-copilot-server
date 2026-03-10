#!/usr/bin/env python3
"""Quality review for generated SFT dataset."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

logger = logging.getLogger(__name__)

RUBRIC_DIMENSIONS = [
    "technical_accuracy",
    "faithfulness",
    "terminology",
    "completeness",
    "korean_quality",
]

DOMAIN_TERMS = [
    "DFT", "WFT", "Sa 2.5", "SSPC", "ISO", "IMO", "PSPC", "도막", "표면처리",
    "블라스팅", "결함", "부식", "에폭시", "프라이머", "환기", "안전", "규정",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review SFT quality with 5-dim rubric")
    parser.add_argument("--train", type=Path, default=Path("/data/sft/sft_train.jsonl"))
    parser.add_argument("--val", type=Path, default=Path("/data/sft/sft_val.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("/data/sft/quality_report.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-score", type=float, default=1.5)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping invalid JSON %s:%d", path, line_no)
    return rows


def validate_record(record: Dict[str, Any], index: int) -> None:
    if "category" not in record or "messages" not in record:
        raise ValueError(f"Missing required field at row {index}")
    messages = record["messages"]
    if len(messages) != 3:
        raise ValueError(f"Invalid messages length at row {index}: {len(messages)}")
    for role, row in zip(["system", "user", "assistant"], messages):
        if row.get("role") != role:
            raise ValueError(f"Role mismatch at row {index}: expected {role}")
        if not str(row.get("content", "")).strip():
            raise ValueError(f"Empty content at row {index}, role={role}")


def score_technical_accuracy(answer: str) -> int:
    spec_count = len(re.findall(r"(?:ISO|IMO|SSPC|KS|NACE|ASTM|Sa\s*2\.5)", answer))
    number_count = len(re.findall(r"\d+(?:\.\d+)?\s*(?:µm|um|mm|°C|%|kPa)?", answer))
    term_count = sum(1 for term in DOMAIN_TERMS if term in answer)
    if spec_count >= 1 and number_count >= 1:
        return 2
    if spec_count >= 1 or number_count >= 1 or term_count >= 2:
        return 1
    return 0


def score_faithfulness(answer: str) -> int:
    if "근거 문서 ID:" not in answer:
        return 0
    ids = re.findall(r"DOC-[^\s,]+", answer)
    if len(ids) >= 1:
        return 2
    return 1


def score_terminology(answer: str) -> int:
    term_count = sum(1 for term in DOMAIN_TERMS if term in answer)
    if term_count >= 2:
        return 2
    if term_count >= 1:
        return 1
    return 0


def score_completeness(answer: str) -> int:
    answer_length = len(answer)
    has_safety = "안전" in answer
    if answer_length >= 140 and has_safety:
        return 2
    if answer_length >= 80:
        return 1
    return 0


def score_korean_quality(answer: str) -> int:
    if not answer:
        return 0
    korean_chars = sum(1 for char in answer if "\uac00" <= char <= "\ud7a3")
    ratio = korean_chars / max(len(answer), 1)
    if ratio >= 0.45:
        return 2
    if ratio >= 0.3:
        return 1
    return 0


def score_record(record: Dict[str, Any]) -> Dict[str, Any]:
    answer = record["messages"][2]["content"]
    dimension_scores = {
        "technical_accuracy": score_technical_accuracy(answer),
        "faithfulness": score_faithfulness(answer),
        "terminology": score_terminology(answer),
        "completeness": score_completeness(answer),
        "korean_quality": score_korean_quality(answer),
    }
    mean_score = sum(dimension_scores.values()) / len(RUBRIC_DIMENSIONS)
    return {
        "category": record.get("category", ""),
        "dimension_scores": dimension_scores,
        "mean_score": round(mean_score, 4),
    }


def compute_duplicate_rate(records: Sequence[Dict[str, Any]]) -> float:
    answers = [record["messages"][2]["content"].strip() for record in records]
    if not answers:
        return 0.0
    counts = Counter(answers)
    duplicate_count = sum(count - 1 for count in counts.values() if count > 1)
    return round(duplicate_count / len(answers), 4)


def aggregate_scores(scored_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not scored_rows:
        return {
            "overall_mean": 0.0,
            "dimension_mean": {key: 0.0 for key in RUBRIC_DIMENSIONS},
            "category_mean": {},
        }

    dimension_totals = {key: 0.0 for key in RUBRIC_DIMENSIONS}
    category_bucket: Dict[str, List[float]] = defaultdict(list)

    for scored in scored_rows:
        for key in RUBRIC_DIMENSIONS:
            dimension_totals[key] += scored["dimension_scores"][key]
        category_bucket[scored["category"]].append(scored["mean_score"])

    count = len(scored_rows)
    dimension_mean = {key: round(dimension_totals[key] / count, 4) for key in RUBRIC_DIMENSIONS}
    category_mean = {
        key: round(sum(values) / len(values), 4)
        for key, values in sorted(category_bucket.items())
    }
    overall_mean = round(sum(row["mean_score"] for row in scored_rows) / count, 4)
    return {
        "overall_mean": overall_mean,
        "dimension_mean": dimension_mean,
        "category_mean": category_mean,
    }


def sample_records(records: Sequence[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    if not records:
        return []
    sample_size = max(100, int(math.ceil(len(records) * 0.3)))
    sample_size = min(sample_size, len(records))
    rng = random.Random(seed)
    selected_indices = rng.sample(list(range(len(records))), sample_size)
    return [records[idx] for idx in selected_indices]


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    train_rows = load_jsonl(args.train)
    val_rows = load_jsonl(args.val)
    all_rows = train_rows + val_rows
    for idx, row in enumerate(all_rows):
        validate_record(row, idx)

    sampled = sample_records(train_rows, args.seed)
    scored = [score_record(row) for row in sampled]
    aggregates = aggregate_scores(scored)
    duplicate_rate = compute_duplicate_rate(train_rows)

    report = {
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "total_count": len(all_rows),
        "sample_count": len(sampled),
        "sample_ratio": round(len(sampled) / max(len(train_rows), 1), 4),
        "scores": aggregates,
        "duplicate_rate": duplicate_rate,
        "pass_gate": aggregates["overall_mean"] >= args.min_score,
        "min_required_score": args.min_score,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    logger.info("Quality report written to %s", args.report)
    logger.info(
        "Quality mean=%.3f gate=%s sample=%d",
        aggregates["overall_mean"],
        "PASS" if report["pass_gate"] else "FAIL",
        len(sampled),
    )

    if not report["pass_gate"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
