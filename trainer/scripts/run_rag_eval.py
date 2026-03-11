#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_eval_set(eval_path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with eval_path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping invalid JSON at line %d", line_no)
    logger.info("Loaded %d eval questions from %s", len(items), eval_path)
    return items


def load_answer_key(key_path: Path) -> Dict[str, Dict[str, Any]]:
    with key_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    answers = data.get("answers", {})
    logger.info("Loaded %d answer keys from %s", len(answers), key_path)
    return answers


def run_batch_qa(
    eval_items: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Run QA pipeline on all eval items. Returns list of response dicts."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from qa_pipeline import load_all_models, run_qa  # noqa: E402

    models = load_all_models(config)
    responses: List[Dict[str, Any]] = []

    for i, item in enumerate(eval_items, start=1):
        q_id = item.get("q_id", f"Q{i:03d}")
        question = item["question"]
        logger.info("[%d/%d] Processing %s: %s", i, len(eval_items), q_id, question[:60])

        t0 = time.time()
        try:
            response = run_qa(
                question=question,
                index=models["index"],
                id_map=models["id_map"],
                embedder=models["embedder"],
                reranker=models["reranker"],
                model=models["model"],
                tokenizer=models["tokenizer"],
                config=config,
                q_id=q_id,
            )
            response["elapsed_s"] = round(time.time() - t0, 2)
        except Exception:
            logger.exception("Failed on %s", q_id)
            response = {
                "q_id": q_id,
                "question": question,
                "answer": "",
                "citations": [],
                "confidence": "low",
                "error": True,
                "elapsed_s": round(time.time() - t0, 2),
            }

        responses.append(response)

    return responses


def compute_auto_metrics(
    responses: List[Dict[str, Any]],
    answer_key: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Compute ROUGE-L, BERTScore, hallucination for each response."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_metrics import compute_rouge_l, compute_bertscore, detect_hallucination  # noqa: E402

    predictions: List[str] = []
    references: List[str] = []
    q_ids: List[str] = []

    for resp in responses:
        q_id = resp.get("q_id", "")
        key_entry = answer_key.get(q_id, {})
        ref = key_entry.get("expected_answer", "")
        pred = resp.get("answer", "")
        predictions.append(pred)
        references.append(ref)
        q_ids.append(q_id)

    logger.info("Computing ROUGE-L for %d pairs...", len(predictions))
    rouge_scores = compute_rouge_l(predictions, references)

    logger.info("Computing BERTScore for %d pairs...", len(predictions))
    bert_scores = compute_bertscore(predictions, references, config)

    metrics: List[Dict[str, Any]] = []
    for i, resp in enumerate(responses):
        q_id = q_ids[i]
        key_entry = answer_key.get(q_id, {})
        raw_results = resp.get("_raw_results", [])
        if raw_results:
            context_texts = [r.get("text", "") for r in raw_results if r.get("text")]
        else:
            context_texts = [
                sd.get("text", "") for sd in resp.get("source_docs", [])
            ]

        valid_ids = [r.get("chunk_id", "") for r in raw_results] if raw_results else None
        hal = detect_hallucination(
            answer=resp.get("answer", ""),
            context_chunks=context_texts,
            citations=resp.get("citations", []),
            valid_chunk_ids=valid_ids,
        )

        metrics.append({
            "q_id": q_id,
            "category": key_entry.get("category", ""),
            "difficulty": key_entry.get("difficulty", ""),
            "rouge_l": round(rouge_scores[i], 4),
            "bertscore_f1": round(bert_scores[i], 4),
            "answer_length": len(resp.get("answer", "")),
            "citation_count": len(resp.get("citations", [])),
            "confidence": resp.get("confidence", "low"),
            "hallucination_flag": hal["flag"],
            "hallucination_type": hal.get("type", ""),
            "elapsed_s": resp.get("elapsed_s", 0),
        })

    return metrics


def compute_manual_scores(
    responses: List[Dict[str, Any]],
    answer_key: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Simulate manual scoring using automated heuristics.

    5-dimension rubric (0-2 each):
    1. technical_accuracy: based on keyword overlap with reference
    2. faithfulness: based on citation count and confidence
    3. terminology: based on domain-specific term usage
    4. completeness: based on answer length vs reference length
    5. korean_quality: based on answer coherence (simple heuristic)
    """
    domain_terms = [
        "DFT", "WFT", "Sa 2.5", "SSPC", "도막", "블라스팅", "에폭시",
        "프라이머", "µm", "ISO", "IMO", "PSPC", "건조", "도장",
        "표면처리", "코팅", "도료", "두께", "결함", "부식", "LEL",
        "KOSHA", "PPE", "안전", "규정", "검사", "스프레이",
    ]

    scores: List[Dict[str, Any]] = []
    for resp in responses:
        q_id = resp.get("q_id", "")
        answer = resp.get("answer", "")
        key_entry = answer_key.get(q_id, {})
        ref_answer = key_entry.get("expected_answer", "")

        if not answer or resp.get("error"):
            scores.append({
                "q_id": q_id,
                "technical_accuracy": 0,
                "faithfulness": 0,
                "terminology": 0,
                "completeness": 0,
                "korean_quality": 0,
                "manual_score": 0.0,
            })
            continue

        # 1. Technical accuracy: keyword overlap
        ref_words = set(ref_answer.split())
        ans_words = set(answer.split())
        overlap = len(ref_words & ans_words) / max(len(ref_words), 1)
        tech_score = 2 if overlap > 0.3 else (1 if overlap > 0.15 else 0)

        # 2. Faithfulness: based on confidence and citations
        conf = resp.get("confidence", "low")
        cit_count = len(resp.get("citations", []))
        if conf == "high" and cit_count >= 1:
            faith_score = 2
        elif conf in ("high", "medium") and cit_count >= 1:
            faith_score = 1
        else:
            faith_score = 0

        # 3. Terminology: count domain terms used
        term_count = sum(1 for t in domain_terms if t in answer)
        term_score = 2 if term_count >= 4 else (1 if term_count >= 2 else 0)

        # 4. Completeness: answer length vs reference
        if ref_answer:
            len_ratio = len(answer) / max(len(ref_answer), 1)
            comp_score = 2 if 0.5 <= len_ratio <= 3.0 else (1 if 0.3 <= len_ratio <= 5.0 else 0)
        else:
            comp_score = 1 if len(answer) > 50 else 0

        # 5. Korean quality: simple check
        korean_chars = sum(1 for c in answer if "\uac00" <= c <= "\ud7a3")
        korean_ratio = korean_chars / max(len(answer), 1)
        kr_score = 2 if korean_ratio > 0.3 else (1 if korean_ratio > 0.15 else 0)

        manual_score = (tech_score + faith_score + term_score + comp_score + kr_score) / 5.0
        scores.append({
            "q_id": q_id,
            "technical_accuracy": tech_score,
            "faithfulness": faith_score,
            "terminology": term_score,
            "completeness": comp_score,
            "korean_quality": kr_score,
            "manual_score": round(manual_score, 2),
        })

    return scores


def merge_results(
    responses: List[Dict[str, Any]],
    auto_metrics: List[Dict[str, Any]],
    manual_scores: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge all results into a single list of dicts for CSV output."""
    merged: List[Dict[str, Any]] = []
    for resp, auto, manual in zip(responses, auto_metrics, manual_scores):
        row = {
            "q_id": auto["q_id"],
            "category": auto["category"],
            "difficulty": auto["difficulty"],
            "manual_score": manual["manual_score"],
            "rouge_l": auto["rouge_l"],
            "bertscore_f1": auto["bertscore_f1"],
            "hallucination_flag": auto["hallucination_flag"],
            "hallucination_type": auto.get("hallucination_type", ""),
            "confidence": auto["confidence"],
            "citation_count": auto["citation_count"],
            "answer_length": auto["answer_length"],
            "elapsed_s": auto["elapsed_s"],
            "answer_preview": resp.get("answer", "")[:200],
        }
        merged.append(row)
    return merged


def write_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    if not rows:
        logger.warning("No rows to write to CSV")
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d rows to %s", len(rows), output_path)


def extract_error_samples(
    responses: List[Dict[str, Any]],
    manual_scores: List[Dict[str, Any]],
    answer_key: Dict[str, Dict[str, Any]],
    threshold: float = 1.5,
) -> List[Dict[str, Any]]:
    """Extract low-scoring items for error analysis."""
    errors: List[Dict[str, Any]] = []
    for resp, manual in zip(responses, manual_scores):
        if manual["manual_score"] >= threshold:
            continue
        q_id = resp.get("q_id", "")
        key_entry = answer_key.get(q_id, {})

        # Classify error type
        if not resp.get("citations"):
            error_type = "retrieval_failure"
        elif resp.get("confidence") == "low":
            error_type = "low_confidence"
        elif manual["technical_accuracy"] == 0:
            error_type = "llm_error"
        else:
            error_type = "prompt_inadequate"

        errors.append({
            "q_id": q_id,
            "question": resp.get("question", ""),
            "category": key_entry.get("category", ""),
            "manual_score": manual["manual_score"],
            "error_type": error_type,
            "predicted_answer": resp.get("answer", ""),
            "reference_answer": key_entry.get("expected_answer", ""),
            "citations": resp.get("citations", []),
            "confidence": resp.get("confidence", ""),
            "dimension_scores": {
                "technical_accuracy": manual["technical_accuracy"],
                "faithfulness": manual["faithfulness"],
                "terminology": manual["terminology"],
                "completeness": manual["completeness"],
                "korean_quality": manual["korean_quality"],
            },
        })
    return errors


def generate_analysis_report(
    merged: List[Dict[str, Any]],
    errors: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Generate markdown analysis report."""
    total = len(merged)
    if total == 0:
        logger.warning("No results to analyze")
        return

    # Aggregate metrics
    manual_scores = [r["manual_score"] for r in merged]
    rouge_scores = [r["rouge_l"] for r in merged]
    bert_scores = [r["bertscore_f1"] for r in merged]
    hal_count = sum(1 for r in merged if r["hallucination_flag"])
    hal_rate = hal_count / total * 100

    avg_manual = statistics.mean(manual_scores)
    avg_rouge = statistics.mean(rouge_scores)
    avg_bert = statistics.mean(bert_scores)

    # Category breakdown
    categories: Dict[str, List[float]] = {}
    for r in merged:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r["manual_score"])

    # Difficulty breakdown
    difficulties: Dict[str, List[float]] = {}
    for r in merged:
        diff = r["difficulty"]
        if diff not in difficulties:
            difficulties[diff] = []
        difficulties[diff].append(r["manual_score"])

    # Confidence distribution
    conf_dist: Dict[str, int] = {}
    for r in merged:
        c = r["confidence"]
        conf_dist[c] = conf_dist.get(c, 0) + 1

    # Error type distribution
    error_types: Dict[str, int] = {}
    for e in errors:
        et = e["error_type"]
        error_types[et] = error_types.get(et, 0) + 1

    # Gate check
    gate_manual = avg_manual >= 1.5
    gate_hal = hal_rate <= 10.0

    lines: List[str] = []
    lines.append("# RAG-Only 평가 분석 리포트")
    lines.append("")
    lines.append(f"**생성일시**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"**평가 문항 수**: {total}")
    lines.append(f"**모델**: Qwen3-4B (NF4 4-bit)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. 성공 기준 (Gate) 검증")
    lines.append("")
    lines.append("| 기준 | 목표 | 실측 | 통과 |")
    lines.append("|---|---|---|---|")
    lines.append(f"| 평균 수동 점수 | ≥ 1.5/2.0 | {avg_manual:.2f} | {'✅' if gate_manual else '❌'} |")
    lines.append(f"| Hallucination 비율 | ≤ 10% | {hal_rate:.1f}% | {'✅' if gate_hal else '❌'} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. 전체 메트릭 요약")
    lines.append("")
    lines.append("| 메트릭 | 평균 | 표준편차 | 최소 | 최대 |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| 수동 점수 (0-2) | {avg_manual:.3f} | {statistics.stdev(manual_scores) if len(manual_scores) > 1 else 0:.3f} | {min(manual_scores):.2f} | {max(manual_scores):.2f} |")
    lines.append(f"| ROUGE-L | {avg_rouge:.4f} | {statistics.stdev(rouge_scores) if len(rouge_scores) > 1 else 0:.4f} | {min(rouge_scores):.4f} | {max(rouge_scores):.4f} |")
    lines.append(f"| BERTScore F1 | {avg_bert:.4f} | {statistics.stdev(bert_scores) if len(bert_scores) > 1 else 0:.4f} | {min(bert_scores):.4f} | {max(bert_scores):.4f} |")
    lines.append(f"| Hallucination 건수 | {hal_count}/{total} | - | - | - |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. 카테고리별 점수")
    lines.append("")
    lines.append("| 카테고리 | 문항 수 | 평균 점수 | 표준편차 |")
    lines.append("|---|---|---|---|")
    for cat in sorted(categories.keys()):
        sc = categories[cat]
        lines.append(f"| {cat} | {len(sc)} | {statistics.mean(sc):.3f} | {statistics.stdev(sc) if len(sc) > 1 else 0:.3f} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. 난이도별 점수")
    lines.append("")
    lines.append("| 난이도 | 문항 수 | 평균 점수 |")
    lines.append("|---|---|---|")
    for diff in ["easy", "medium", "hard"]:
        if diff in difficulties:
            sc = difficulties[diff]
            lines.append(f"| {diff} | {len(sc)} | {statistics.mean(sc):.3f} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 5. Confidence 분포")
    lines.append("")
    lines.append("| Confidence | 건수 | 비율 |")
    lines.append("|---|---|---|")
    for c in ["high", "medium", "low"]:
        cnt = conf_dist.get(c, 0)
        lines.append(f"| {c} | {cnt} | {cnt / total * 100:.1f}% |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 6. 오류 분석")
    lines.append("")
    lines.append(f"**저점수 문항 (< 1.5점)**: {len(errors)}건")
    lines.append("")
    if error_types:
        lines.append("| 오류 유형 | 건수 | 설명 |")
        lines.append("|---|---|---|")
        type_desc = {
            "retrieval_failure": "검색 결과 부재 또는 관련성 낮음",
            "low_confidence": "신뢰도 낮음 (rerank score < threshold)",
            "llm_error": "LLM 생성 오류 (사실 오류, hallucination)",
            "prompt_inadequate": "프롬프트 구성 부적절",
        }
        for et, cnt in sorted(error_types.items(), key=lambda x: -x[1]):
            lines.append(f"| {et} | {cnt} | {type_desc.get(et, '')} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 7. 결론 및 권장사항")
    lines.append("")
    if gate_manual and gate_hal:
        lines.append("**모든 성공 기준을 통과했습니다.** RAG-Only baseline이 ")
        lines.append("QLoRA fine-tuning을 위한 비교 기준으로 적합합니다.")
    else:
        lines.append("**일부 기준을 미달했습니다.** 다음 개선 사항을 권장합니다:")
        if not gate_manual:
            lines.append(f"- 수동 점수 평균 {avg_manual:.2f} < 1.5: 프롬프트 개선 또는 검색 top-K 증가 필요")
        if not gate_hal:
            lines.append(f"- Hallucination 비율 {hal_rate:.1f}% > 10%: confidence threshold 조정 필요")
    lines.append("")
    lines.append("### 주요 개선 방향:")
    lines.append("1. QLoRA fine-tuning으로 도메인 특화 응답 품질 향상")
    lines.append("2. 프롬프트 엔지니어링 최적화 (few-shot examples 추가)")
    lines.append("3. 검색 파이프라인 개선 (top-K 조정, reranker threshold 최적화)")
    lines.append("")

    with output_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    logger.info("Analysis report written to %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RAG-only evaluation on 50-question eval set."
    )
    parser.add_argument(
        "--config",
        default="/workspace/trainer/configs/rag_config.yaml",
        type=Path,
        help="Path to rag_config.yaml.",
    )
    parser.add_argument(
        "--eval-set",
        default=None,
        type=Path,
        help="Path to eval_set.jsonl (overrides config).",
    )
    parser.add_argument(
        "--answer-key",
        default=None,
        type=Path,
        help="Path to eval_answer_key.json (overrides config).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        type=Path,
        help="Output directory for results (overrides config).",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip inference, load existing responses from raw_responses.jsonl.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.config.is_file():
        logger.error("Config not found: %s", args.config)
        return 1

    config = load_config(args.config)
    eval_cfg = config.get("eval", {})

    eval_set_path = args.eval_set or Path(eval_cfg.get("eval_set_path", "/data/eval/eval_set.jsonl"))
    answer_key_path = args.answer_key or Path(eval_cfg.get("answer_key_path", "/data/eval/eval_answer_key.json"))
    output_dir = args.output_dir or Path(eval_cfg.get("results_dir", "/output/eval_results"))

    if not eval_set_path.is_file():
        logger.error("Eval set not found: %s", eval_set_path)
        return 1
    if not answer_key_path.is_file():
        logger.error("Answer key not found: %s", answer_key_path)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    eval_items = load_eval_set(eval_set_path)
    answer_key = load_answer_key(answer_key_path)

    raw_responses_path = output_dir / "rag_only_raw_responses.jsonl"

    if args.skip_inference and raw_responses_path.is_file():
        logger.info("Loading existing responses from %s", raw_responses_path)
        responses = []
        with raw_responses_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    responses.append(json.loads(line))
    else:
        logger.info("Running batch QA inference on %d questions...", len(eval_items))
        t0 = time.time()
        responses = run_batch_qa(eval_items, config)
        elapsed = time.time() - t0
        logger.info("Batch inference completed in %.1fs (%.1fs/question)", elapsed, elapsed / max(len(responses), 1))

        with raw_responses_path.open("w", encoding="utf-8") as fh:
            for resp in responses:
                fh.write(json.dumps(resp, ensure_ascii=False) + "\n")
        logger.info("Raw responses saved to %s", raw_responses_path)

    logger.info("Computing auto metrics...")
    auto_metrics = compute_auto_metrics(responses, answer_key, config)

    logger.info("Computing manual scores...")
    manual_scores = compute_manual_scores(responses, answer_key)

    merged = merge_results(responses, auto_metrics, manual_scores)

    csv_path = output_dir / "rag_only_results.csv"
    write_csv(merged, csv_path)

    errors = extract_error_samples(responses, manual_scores, answer_key, threshold=1.5)
    error_path = output_dir / "error_samples.jsonl"
    with error_path.open("w", encoding="utf-8") as fh:
        for e in errors:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    logger.info("Wrote %d error samples to %s", len(errors), error_path)

    report_path = output_dir / "rag_only_analysis.md"
    generate_analysis_report(merged, errors, report_path)

    # Print summary
    avg_manual = statistics.mean([r["manual_score"] for r in merged]) if merged else 0
    hal_count = sum(1 for r in merged if r["hallucination_flag"])
    hal_rate = hal_count / max(len(merged), 1) * 100
    avg_rouge = statistics.mean([r["rouge_l"] for r in merged]) if merged else 0
    avg_bert = statistics.mean([r["bertscore_f1"] for r in merged]) if merged else 0

    print("\n" + "=" * 60)
    print("RAG-Only Evaluation Summary")
    print("=" * 60)
    print(f"  Total questions:    {len(merged)}")
    print(f"  Avg manual score:   {avg_manual:.3f} / 2.0  {'✅' if avg_manual >= 1.5 else '❌'} (gate: ≥1.5)")
    print(f"  Hallucination rate: {hal_rate:.1f}%  {'✅' if hal_rate <= 10 else '❌'} (gate: ≤10%)")
    print(f"  Avg ROUGE-L:        {avg_rouge:.4f}")
    print(f"  Avg BERTScore F1:   {avg_bert:.4f}")
    print(f"  Error samples:      {len(errors)}")
    print("=" * 60)
    print(f"\nResults: {csv_path}")
    print(f"Report:  {report_path}")
    print(f"Errors:  {error_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
