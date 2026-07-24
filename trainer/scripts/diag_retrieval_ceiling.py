#!/usr/bin/env python3
"""Diagnose which gold chunks are unreachable by dense retrieval (recall ceiling)."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from search_index import load_config, load_embedder, load_faiss_index, load_metadata, retrieve

CONFIG = Path("/workspace/trainer/configs/rag_config.yaml")
EVAL = Path("/data/eval/eval_set.jsonl")


def main() -> int:
    config = load_config(CONFIG)
    index = load_faiss_index(config["paths"]["faiss_index"], 256)
    metadata = load_metadata(config["paths"]["metadata_json"])
    id_map = metadata["id_map"]
    embedder = load_embedder(config["embedding"]["model_name"], "cuda", 1024)

    items = [json.loads(l) for l in open(EVAL) if json.loads(l).get("required_chunks")]
    by_id = {x["chunk_id"]: x for x in id_map}

    gold_total = 0
    misses = []
    for it in items:
        cids = [x["chunk_id"] for x in retrieve(it["question"], index, id_map, embedder, top_k=30)]
        for g in it["required_chunks"]:
            gold_total += 1
            if g not in cids:
                gt = by_id.get(g, {})
                misses.append((it["q_id"], it["category"], it["question"][:45], g,
                               gt.get("tokens"), (gt.get("text", "") or "")[:80]))

    print(f"TOTAL GOLD: {gold_total} | MISSES (not in top30): {len(misses)} "
          f"| retrieve ceiling micro: {round(1 - len(misses)/gold_total, 3)}")
    print("misses by category:", dict(Counter(m[1] for m in misses)))
    for m in misses:
        print(f"  MISS {m[0]} [{m[1]}] q='{m[2]}' gold={m[3]} tok={m[4]} text='{m[5]}'")
    print("CEILING_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
