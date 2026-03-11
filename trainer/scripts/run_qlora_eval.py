#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import PeftModel

logger = logging.getLogger(__name__)

SFT_SYSTEM_PROMPT = "너는 선박 도장 분야 전문 AI 어시스턴트이다. 제공된 문서를 근거로 정확한 기술 정보를 제공하고, 답변 마지막에 근거 문서 ID를 반드시 명시한다. 근거가 부족하면 솔직히 '제공된 문서에 해당 정보가 없습니다'라고 답한다. 안전 관련 질문에는 항상 관련 규정을 함께 안내한다."

SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

run_rag_eval_mod = importlib.import_module("run_rag_eval")
qa_pipeline_mod = importlib.import_module("qa_pipeline")
eval_metrics_mod = importlib.import_module("eval_metrics")

load_config = run_rag_eval_mod.load_config
load_eval_set = run_rag_eval_mod.load_eval_set
load_answer_key = run_rag_eval_mod.load_answer_key
compute_auto_metrics = run_rag_eval_mod.compute_auto_metrics
compute_manual_scores = run_rag_eval_mod.compute_manual_scores
merge_results = run_rag_eval_mod.merge_results
write_csv = run_rag_eval_mod.write_csv
extract_error_samples = run_rag_eval_mod.extract_error_samples
generate_analysis_report = run_rag_eval_mod.generate_analysis_report

load_all_models = qa_pipeline_mod.load_all_models
run_qa = qa_pipeline_mod.run_qa
generate_answer = qa_pipeline_mod.generate_answer
load_llm = qa_pipeline_mod.load_llm
_patch_torch_set_submodule = qa_pipeline_mod._patch_torch_set_submodule

compute_rouge_l = eval_metrics_mod.compute_rouge_l
compute_bertscore = eval_metrics_mod.compute_bertscore
detect_hallucination = eval_metrics_mod.detect_hallucination


def _load_qlora_system_prompt(config_path: Path) -> str:
    if not config_path.is_file():
        return SFT_SYSTEM_PROMPT
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    return str(cfg.get("data", {}).get("system_prompt", SFT_SYSTEM_PROMPT))


def _save_raw_responses(responses: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as fh:
        for resp in responses:
            fh.write(json.dumps(resp, ensure_ascii=False) + "\n")
    logger.info("Raw responses saved to %s", output_path)


def _load_raw_responses(raw_path: Path) -> list[dict[str, Any]]:
    responses: list[dict[str, Any]] = []
    with raw_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                responses.append(json.loads(line))
    return responses


def _strip_think_blocks(text: str) -> str:
    import re
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _generate_qlora_only_answer(
    question: str,
    model: Any,
    tokenizer: Any,
    config: dict[str, Any],
    system_prompt: str,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    enable_thinking = config["generation"].get("enable_thinking", False)
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    inputs = tokenizer([text], return_tensors="pt", return_attention_mask=True).to(model.device)

    gen_kwargs: dict[str, Any] = {
        "do_sample": bool(config["generation"]["do_sample"]),
        "temperature": float(config["generation"]["temperature"]),
        "max_new_tokens": int(config["generation"]["max_new_tokens"]),
        "pad_token_id": tokenizer.eos_token_id,
    }
    if "top_p" in config["generation"]:
        gen_kwargs["top_p"] = float(config["generation"]["top_p"])
    if "top_k" in config["generation"]:
        gen_kwargs["top_k"] = int(config["generation"]["top_k"])

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)
    generated = output_ids[0][inputs.input_ids.shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True)
    return _strip_think_blocks(answer)


def run_qlora_only_batch(
    eval_items: list[dict[str, Any]],
    config: dict[str, Any],
    adapter_dir: Path,
) -> list[dict[str, Any]]:
    _patch_torch_set_submodule()
    base_model, tokenizer = load_llm(config)
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()

    system_prompt = _load_qlora_system_prompt(Path("/workspace/trainer/configs/qlora_config.yaml"))
    responses: list[dict[str, Any]] = []

    for i, item in enumerate(eval_items, start=1):
        q_id = item.get("q_id", f"Q{i:03d}")
        question = item["question"]
        logger.info("[%d/%d] Processing %s: %s", i, len(eval_items), q_id, question[:60])
        t0 = time.time()
        try:
            answer = _generate_qlora_only_answer(question, model, tokenizer, config, system_prompt)
            response = {
                "q_id": q_id,
                "question": question,
                "answer": answer,
                "citations": [],
                "confidence": "low",
                "error": False,
                "elapsed_s": round(time.time() - t0, 2),
                "source_docs": [],
                "_raw_results": [],
            }
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
                "source_docs": [],
                "_raw_results": [],
            }
        responses.append(response)

    return responses


def run_rag_qlora_batch(
    eval_items: list[dict[str, Any]],
    config: dict[str, Any],
    adapter_dir: Path,
) -> list[dict[str, Any]]:
    _patch_torch_set_submodule()
    models = load_all_models(config)
    models["model"] = PeftModel.from_pretrained(models["model"], str(adapter_dir))
    models["model"].eval()

    responses: list[dict[str, Any]] = []
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
            response["error"] = False
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
                "source_docs": [],
                "_raw_results": [],
            }
        responses.append(response)

    return responses


def _run_eval_flow(
    responses: list[dict[str, Any]],
    answer_key: dict[str, dict[str, Any]],
    config: dict[str, Any],
    output_csv: Path,
) -> list[dict[str, Any]]:
    auto_metrics = compute_auto_metrics(responses, answer_key, config)
    manual_scores = compute_manual_scores(responses, answer_key)
    merged = merge_results(responses, auto_metrics, manual_scores)
    write_csv(merged, output_csv)
    return merged


def _safe_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _safe_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def _read_csv_by_qid(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            q_id = row.get("q_id", "")
            if q_id:
                rows[q_id] = row
    return rows


def run_compare_mode(output_dir: Path) -> int:
    rag_only_path = output_dir / "rag_only_results.csv"
    qlora_only_path = output_dir / "qlora_only_results.csv"
    ragqlora_path = output_dir / "ragqlora_results.csv"

    for p in [rag_only_path, qlora_only_path, ragqlora_path]:
        if not p.is_file():
            logger.error("Missing required file for compare mode: %s", p)
            return 1

    rag_rows = _read_csv_by_qid(rag_only_path)
    qlora_rows = _read_csv_by_qid(qlora_only_path)
    ragqlora_rows = _read_csv_by_qid(ragqlora_path)
    q_ids = sorted(set(rag_rows.keys()) | set(qlora_rows.keys()) | set(ragqlora_rows.keys()))

    comparison_rows: list[dict[str, Any]] = []
    for q_id in q_ids:
        r = rag_rows.get(q_id, {})
        q = qlora_rows.get(q_id, {})
        rq = ragqlora_rows.get(q_id, {})
        comparison_rows.append({
            "q_id": q_id,
            "category": rq.get("category") or r.get("category") or q.get("category", ""),
            "difficulty": rq.get("difficulty") or r.get("difficulty") or q.get("difficulty", ""),
            "rag_only_score": _safe_float(r.get("manual_score", "0")),
            "qlora_only_score": _safe_float(q.get("manual_score", "0")),
            "ragqlora_score": _safe_float(rq.get("manual_score", "0")),
            "rag_only_rouge": _safe_float(r.get("rouge_l", "0")),
            "qlora_only_rouge": _safe_float(q.get("rouge_l", "0")),
            "ragqlora_rouge": _safe_float(rq.get("rouge_l", "0")),
            "rag_only_bert": _safe_float(r.get("bertscore_f1", "0")),
            "qlora_only_bert": _safe_float(q.get("bertscore_f1", "0")),
            "ragqlora_bert": _safe_float(rq.get("bertscore_f1", "0")),
            "rag_only_hal": _safe_bool(r.get("hallucination_flag", "False")),
            "qlora_only_hal": _safe_bool(q.get("hallucination_flag", "False")),
            "ragqlora_hal": _safe_bool(rq.get("hallucination_flag", "False")),
        })

    comparison_csv = output_dir / "comparison.csv"
    write_csv(comparison_rows, comparison_csv)

    def agg(rows: list[dict[str, Any]], score_key: str, rouge_key: str, bert_key: str, hal_key: str) -> dict[str, float]:
        total = len(rows)
        if total == 0:
            return {"manual": 0.0, "rouge": 0.0, "bert": 0.0, "hal_rate": 0.0}
        return {
            "manual": statistics.mean([float(r[score_key]) for r in rows]),
            "rouge": statistics.mean([float(r[rouge_key]) for r in rows]),
            "bert": statistics.mean([float(r[bert_key]) for r in rows]),
            "hal_rate": (sum(1 for r in rows if r[hal_key]) / total) * 100,
        }

    rag_stats = agg(comparison_rows, "rag_only_score", "rag_only_rouge", "rag_only_bert", "rag_only_hal")
    qlora_stats = agg(comparison_rows, "qlora_only_score", "qlora_only_rouge", "qlora_only_bert", "qlora_only_hal")
    ragqlora_stats = agg(comparison_rows, "ragqlora_score", "ragqlora_rouge", "ragqlora_bert", "ragqlora_hal")

    by_category = sorted(set(r["category"] for r in comparison_rows if r.get("category")))
    by_difficulty = sorted(set(r["difficulty"] for r in comparison_rows if r.get("difficulty")))

    def section_stats(
        rows: list[dict[str, Any]],
        field: str,
        value: str,
        score_key: str,
        rouge_key: str,
        bert_key: str,
        hal_key: str,
    ) -> dict[str, float]:
        filtered = [r for r in rows if r.get(field) == value]
        return agg(filtered, score_key, rouge_key, bert_key, hal_key)

    delta_qlora_vs_rag = qlora_stats["manual"] - rag_stats["manual"]
    delta_ragqlora_vs_rag = ragqlora_stats["manual"] - rag_stats["manual"]
    keep_adapter = (
        ragqlora_stats["manual"] >= rag_stats["manual"] + 0.15
        and ragqlora_stats["hal_rate"] <= rag_stats["hal_rate"]
    )

    sorted_by_delta = sorted(
        comparison_rows,
        key=lambda x: float(x["ragqlora_score"]) - float(x["rag_only_score"]),
        reverse=True,
    )
    top_improved = sorted_by_delta[:5]
    top_degraded = sorted_by_delta[-5:]

    lines: list[str] = []
    lines.append("# QLoRA/RAG 비교 분석 리포트")
    lines.append("")
    lines.append(f"**생성일시**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"**평가 문항 수**: {len(comparison_rows)}")
    lines.append("")
    lines.append("## 1. 3-way Summary")
    lines.append("")
    lines.append("| Config | Avg manual_score | Avg ROUGE-L | Avg BERTScore F1 | Hallucination rate |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(f"| RAG-Only | {rag_stats['manual']:.3f} | {rag_stats['rouge']:.4f} | {rag_stats['bert']:.4f} | {rag_stats['hal_rate']:.1f}% |")
    lines.append(f"| QLoRA-Only | {qlora_stats['manual']:.3f} | {qlora_stats['rouge']:.4f} | {qlora_stats['bert']:.4f} | {qlora_stats['hal_rate']:.1f}% |")
    lines.append(f"| RAG+QLoRA | {ragqlora_stats['manual']:.3f} | {ragqlora_stats['rouge']:.4f} | {ragqlora_stats['bert']:.4f} | {ragqlora_stats['hal_rate']:.1f}% |")
    lines.append("")
    lines.append("## 2. Category-wise Breakdown")
    lines.append("")
    lines.append("| Category | RAG-Only | QLoRA-Only | RAG+QLoRA |")
    lines.append("|---|---:|---:|---:|")
    for cat in by_category:
        rs = section_stats(comparison_rows, "category", cat, "rag_only_score", "rag_only_rouge", "rag_only_bert", "rag_only_hal")
        qs = section_stats(comparison_rows, "category", cat, "qlora_only_score", "qlora_only_rouge", "qlora_only_bert", "qlora_only_hal")
        rqs = section_stats(comparison_rows, "category", cat, "ragqlora_score", "ragqlora_rouge", "ragqlora_bert", "ragqlora_hal")
        lines.append(f"| {cat} | {rs['manual']:.3f} | {qs['manual']:.3f} | {rqs['manual']:.3f} |")
    lines.append("")
    lines.append("## 3. Difficulty-wise Breakdown")
    lines.append("")
    lines.append("| Difficulty | RAG-Only | QLoRA-Only | RAG+QLoRA |")
    lines.append("|---|---:|---:|---:|")
    for diff in by_difficulty:
        rs = section_stats(comparison_rows, "difficulty", diff, "rag_only_score", "rag_only_rouge", "rag_only_bert", "rag_only_hal")
        qs = section_stats(comparison_rows, "difficulty", diff, "qlora_only_score", "qlora_only_rouge", "qlora_only_bert", "qlora_only_hal")
        rqs = section_stats(comparison_rows, "difficulty", diff, "ragqlora_score", "ragqlora_rouge", "ragqlora_bert", "ragqlora_hal")
        lines.append(f"| {diff} | {rs['manual']:.3f} | {qs['manual']:.3f} | {rqs['manual']:.3f} |")
    lines.append("")
    lines.append("## 4. Delta Analysis")
    lines.append("")
    lines.append(f"- QLoRA-Only vs RAG-Only (manual_score delta): {delta_qlora_vs_rag:+.3f}")
    lines.append(f"- RAG+QLoRA vs RAG-Only (manual_score delta): {delta_ragqlora_vs_rag:+.3f}")
    lines.append("")
    lines.append("## 5. Keep/Drop Decision")
    lines.append("")
    lines.append("기준: RAG+QLoRA avg_manual_score >= RAG-Only avg + 0.15 AND RAG+QLoRA hallucination_rate <= RAG-Only hallucination_rate")
    lines.append("")
    lines.append(f"- RAG+QLoRA avg_manual_score: {ragqlora_stats['manual']:.3f}")
    lines.append(f"- RAG-Only avg_manual_score + 0.15: {rag_stats['manual'] + 0.15:.3f}")
    lines.append(f"- RAG+QLoRA hallucination_rate: {ragqlora_stats['hal_rate']:.1f}%")
    lines.append(f"- RAG-Only hallucination_rate: {rag_stats['hal_rate']:.1f}%")
    lines.append("")
    lines.append(f"**Decision: {'KEEP' if keep_adapter else 'DROP'} adapter**")
    lines.append("")
    lines.append("## 6. Top 5 Improved Questions")
    lines.append("")
    lines.append("| q_id | category | difficulty | rag_only_score | ragqlora_score | delta |")
    lines.append("|---|---|---|---:|---:|---:|")
    for row in top_improved:
        delta = float(row["ragqlora_score"]) - float(row["rag_only_score"])
        lines.append(
            f"| {row['q_id']} | {row['category']} | {row['difficulty']} | {row['rag_only_score']:.2f} | {row['ragqlora_score']:.2f} | {delta:+.2f} |"
        )
    lines.append("")
    lines.append("## 7. Top 5 Degraded Questions")
    lines.append("")
    lines.append("| q_id | category | difficulty | rag_only_score | ragqlora_score | delta |")
    lines.append("|---|---|---|---:|---:|---:|")
    for row in top_degraded:
        delta = float(row["ragqlora_score"]) - float(row["rag_only_score"])
        lines.append(
            f"| {row['q_id']} | {row['category']} | {row['difficulty']} | {row['rag_only_score']:.2f} | {row['ragqlora_score']:.2f} | {delta:+.2f} |"
        )

    analysis_path = output_dir / "comparison_analysis.md"
    with analysis_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    logger.info("Wrote comparison analysis to %s", analysis_path)

    print("\n" + "=" * 60)
    print("Comparison Summary")
    print("=" * 60)
    print(f"  Total questions:     {len(comparison_rows)}")
    print(f"  RAG-Only avg score:  {rag_stats['manual']:.3f}")
    print(f"  QLoRA-Only avg score:{qlora_stats['manual']:.3f}")
    print(f"  RAG+QLoRA avg score: {ragqlora_stats['manual']:.3f}")
    print(f"  Decision:            {'KEEP' if keep_adapter else 'DROP'}")
    print("=" * 60)
    print(f"\nComparison CSV: {comparison_csv}")
    print(f"Analysis MD:     {analysis_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run QLoRA-only, RAG+QLoRA evaluation and compare results."
    )
    parser.add_argument(
        "--config",
        default="/workspace/trainer/configs/rag_config.yaml",
        type=Path,
        help="Path to rag_config.yaml.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["qlora-only", "rag-qlora", "compare"],
        help="Evaluation mode.",
    )
    parser.add_argument(
        "--adapter-dir",
        default="/output/adapters/qwen3_4b_shipcoat_lora",
        type=Path,
        help="Path to QLoRA adapter directory.",
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
        help="Skip inference and load existing raw responses.",
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

    _patch_torch_set_submodule()
    config = load_config(args.config)
    eval_cfg = config.get("eval", {})
    eval_set_path = Path(eval_cfg.get("eval_set_path", "/data/eval/eval_set.jsonl"))
    answer_key_path = Path(eval_cfg.get("answer_key_path", "/data/eval/eval_answer_key.json"))
    output_dir = args.output_dir or Path(eval_cfg.get("results_dir", "/output/eval_results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "compare":
        return run_compare_mode(output_dir)

    if not eval_set_path.is_file():
        logger.error("Eval set not found: %s", eval_set_path)
        return 1
    if not answer_key_path.is_file():
        logger.error("Answer key not found: %s", answer_key_path)
        return 1
    if not args.adapter_dir.is_dir():
        logger.error("Adapter dir not found: %s", args.adapter_dir)
        return 1

    eval_items = load_eval_set(eval_set_path)
    answer_key = load_answer_key(answer_key_path)

    if args.mode == "qlora-only":
        raw_path = output_dir / "qlora_only_raw_responses.jsonl"
        csv_path = output_dir / "qlora_only_results.csv"
        if args.skip_inference and raw_path.is_file():
            logger.info("Loading existing responses from %s", raw_path)
            responses = _load_raw_responses(raw_path)
        else:
            logger.info("Running QLoRA-only inference on %d questions...", len(eval_items))
            t0 = time.time()
            responses = run_qlora_only_batch(eval_items, config, args.adapter_dir)
            elapsed = time.time() - t0
            logger.info("QLoRA-only inference completed in %.1fs (%.1fs/question)", elapsed, elapsed / max(len(responses), 1))
            _save_raw_responses(responses, raw_path)

        merged = _run_eval_flow(responses, answer_key, config, csv_path)
        avg_manual = statistics.mean([r["manual_score"] for r in merged]) if merged else 0.0
        hal_rate = (sum(1 for r in merged if r["hallucination_flag"]) / max(len(merged), 1)) * 100
        print("\n" + "=" * 60)
        print("QLoRA-Only Evaluation Summary")
        print("=" * 60)
        print(f"  Total questions:    {len(merged)}")
        print(f"  Avg manual score:   {avg_manual:.3f}")
        print(f"  Hallucination rate: {hal_rate:.1f}%")
        print("=" * 60)
        print(f"\nRaw:    {raw_path}")
        print(f"Result: {csv_path}")
        return 0

    raw_path = output_dir / "ragqlora_raw_responses.jsonl"
    csv_path = output_dir / "ragqlora_results.csv"
    if args.skip_inference and raw_path.is_file():
        logger.info("Loading existing responses from %s", raw_path)
        responses = _load_raw_responses(raw_path)
    else:
        logger.info("Running RAG+QLoRA inference on %d questions...", len(eval_items))
        t0 = time.time()
        responses = run_rag_qlora_batch(eval_items, config, args.adapter_dir)
        elapsed = time.time() - t0
        logger.info("RAG+QLoRA inference completed in %.1fs (%.1fs/question)", elapsed, elapsed / max(len(responses), 1))
        _save_raw_responses(responses, raw_path)

    merged = _run_eval_flow(responses, answer_key, config, csv_path)
    avg_manual = statistics.mean([r["manual_score"] for r in merged]) if merged else 0.0
    hal_rate = (sum(1 for r in merged if r["hallucination_flag"]) / max(len(merged), 1)) * 100
    print("\n" + "=" * 60)
    print("RAG+QLoRA Evaluation Summary")
    print("=" * 60)
    print(f"  Total questions:    {len(merged)}")
    print(f"  Avg manual score:   {avg_manual:.3f}")
    print(f"  Hallucination rate: {hal_rate:.1f}%")
    print("=" * 60)
    print(f"\nRaw:    {raw_path}")
    print(f"Result: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
