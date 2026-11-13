#!/usr/bin/env python3
"""Retrieval-only RAG recall eval: measures recall@k of gold required_chunks."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_eval(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG retrieval recall eval (no LLM).")
    parser.add_argument("--config", type=Path, default=Path("/workspace/trainer/configs/rag_config.yaml"))
    parser.add_argument("--eval-set", type=Path, default=Path("/data/eval/eval_set.jsonl"))
    parser.add_argument("--retrieve-k", type=int, default=10)
    parser.add_argument("--rerank-k", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("/output/eval_results/rag_retrieval_recall.json"))
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from search_index import (
        load_config,
        load_embedder,
        load_faiss_index,
        load_metadata,
        load_reranker,
        search_and_rerank,
    )

    config = load_config(args.config)
    index = load_faiss_index(config["paths"]["faiss_index"], config["index"]["ef_search"])
    metadata = load_metadata(config["paths"]["metadata_json"], config["embedding"]["model_name"])
    id_map = metadata["id_map"]
    embedder = load_embedder(
        config["embedding"]["model_name"],
        config["embedding"].get("device", "cuda"),
        config["embedding"].get("max_seq_length", 1024),
    )
    reranker = load_reranker(
        config["reranker"]["model_name"],
        config["reranker"].get("device", "cuda"),
        config["reranker"].get("max_length", 1024),
    )

    items = load_eval(args.eval_set)
    items = [it for it in items if it.get("required_chunks")]
    logger.info("Evaluating retrieval recall on %d items with gold chunks", len(items))

    retrieve_hits = 0
    rerank_hits = 0
    gold_total = 0
    retrieve_gold_found = 0
    rerank_gold_found = 0
    per_item: list[dict[str, Any]] = []

    for it in items:
        gold = set(it["required_chunks"])
        gold_total += len(gold)

        # Retrieve a wide net (retrieve-k), then rerank to rerank-k.
        retrieved, _ = search_and_rerank(
            it["question"], index, id_map, embedder, reranker, config,
            top_k_override=args.retrieve_k, skip_rerank=True,
        )
        retrieved_ids = [c.get("chunk_id") for c in retrieved]
        reranked, _ = search_and_rerank(
            it["question"], index, id_map, embedder, reranker, config,
            top_k_override=args.retrieve_k, rerank_top_k_override=args.rerank_k,
        )
        reranked_ids = [c.get("chunk_id") for c in reranked]

        r_found = gold & set(retrieved_ids)
        rr_found = gold & set(reranked_ids)
        retrieve_gold_found += len(r_found)
        rerank_gold_found += len(rr_found)
        if r_found:
            retrieve_hits += 1
        if rr_found:
            rerank_hits += 1

        per_item.append({
            "q_id": it.get("q_id"),
            "gold": sorted(gold),
            "retrieve_hit": bool(r_found),
            "rerank_hit": bool(rr_found),
            "reranked_ids": reranked_ids,
        })

    n = len(items)
    summary = {
        "items": n,
        "retrieve_k": args.retrieve_k,
        "rerank_k": args.rerank_k,
        "any_gold_recall_at_retrieve_k": round(retrieve_hits / n, 4),
        "any_gold_recall_at_rerank_k": round(rerank_hits / n, 4),
        "micro_recall_retrieve": round(retrieve_gold_found / gold_total, 4),
        "micro_recall_rerank": round(rerank_gold_found / gold_total, 4),
    }
    logger.info("Recall summary: %s", json.dumps(summary, ensure_ascii=False))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "per_item": per_item}, fh, ensure_ascii=False, indent=2)
    logger.info("Wrote %s", args.output)

    print("=" * 60)
    print("RAG Retrieval Recall (retrieval-only, no LLM)")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
