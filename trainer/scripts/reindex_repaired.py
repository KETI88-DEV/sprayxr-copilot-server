#!/usr/bin/env python3
"""Re-embed repaired chunks into a shadow FAISS index, preserving chunk_id order from current metadata."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild FAISS index from repaired chunk text.")
    parser.add_argument("--config", type=Path, default=Path("/workspace/trainer/configs/rag_config.yaml"))
    parser.add_argument("--repaired", type=Path, default=Path("/data/chunked/chunks_repaired.jsonl"))
    parser.add_argument("--current-metadata", type=Path, default=Path("/output/indexes/metadata.json"))
    parser.add_argument("--out-index", type=Path, default=Path("/output/indexes/faiss_repaired.index"))
    parser.add_argument("--out-metadata", type=Path, default=Path("/output/indexes/metadata_repaired.json"))
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import faiss
    from build_index import build_faiss_index, embed_chunks
    from search_index import load_config

    config = load_config(args.config)

    current = json.load(args.current_metadata.open("r", encoding="utf-8"))
    id_map = current["id_map"]

    repaired = {json.loads(l)["chunk_id"]: json.loads(l)
                for l in args.repaired.open("r", encoding="utf-8") if l.strip()}

    new_id_map = []
    texts = []
    repaired_count = 0
    for entry in id_map:
        cid = entry["chunk_id"]
        rep = repaired.get(cid)
        new_entry = dict(entry)
        if rep is not None:
            new_entry["text"] = rep["text"]
            if rep.get("spacing_repaired"):
                new_entry["spacing_repaired"] = True
                repaired_count += 1
        new_id_map.append(new_entry)
        texts.append(new_entry["text"])

    logger.info("Preserved %d chunk_ids, %d had repaired text", len(new_id_map), repaired_count)

    embeddings = embed_chunks(texts, config)
    index = build_faiss_index(embeddings, config)

    args.out_index.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(args.out_index))
    out_meta = dict(current)
    out_meta["id_map"] = new_id_map
    json.dump(out_meta, args.out_metadata.open("w", encoding="utf-8"), ensure_ascii=False)

    logger.info("Wrote %s (%d vecs) and %s", args.out_index, index.ntotal, args.out_metadata)
    print(f"reindex done: vectors={index.ntotal} repaired_texts={repaired_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
