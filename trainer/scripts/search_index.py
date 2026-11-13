#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_faiss_index(index_path: str, ef_search: int) -> Any:
    import faiss

    if not Path(index_path).is_file():
        logger.error("FAISS index not found: %s", index_path)
        raise FileNotFoundError(f"FAISS index not found: {index_path}")

    index = faiss.read_index(index_path)
    index.hnsw.efSearch = ef_search
    logger.info("Loaded FAISS index: %d vectors, efSearch=%d", index.ntotal, ef_search)
    return index


def load_metadata(metadata_path: str, embedding_model: Optional[str] = None) -> Dict[str, Any]:
    path = Path(metadata_path)
    if not path.is_file():
        logger.error("Metadata not found: %s", metadata_path)
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    with path.open("r", encoding="utf-8") as fh:
        metadata = json.load(fh)
    if embedding_model is not None:
        # Builds record a hub id ("BAAI/bge-m3") while serving loads a local copy
        # ("/models/bge-m3"); the last path part names the model in both.
        built_with = str(metadata.get("embedding_model") or "")
        if Path(built_with).name.lower() != Path(embedding_model).name.lower():
            raise ValueError(
                f"Index {metadata_path} was built with embedding model {built_with!r}, "
                f"but the config loads {embedding_model!r}"
            )
    logger.info("Loaded metadata: %d entries", len(metadata.get("id_map", [])))
    return metadata


def load_embedder(model_name: str, device: str, max_seq_length: int) -> Any:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        model_name,
        device=device,
        model_kwargs={"use_safetensors": True},
    )
    model.max_seq_length = max_seq_length
    logger.info("Loaded embedder: %s on %s", model_name, device)
    return model


def load_reranker(model_name: str, device: str, max_length: int) -> Any:
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(
        model_name,
        device=device,
        max_length=max_length,
        model_kwargs={"use_safetensors": True},
    )
    logger.info("Loaded reranker: %s on %s", model_name, device)
    return model


def retrieve(
    query: str,
    index: Any,
    id_map: List[Dict[str, Any]],
    embedder: Any,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    # encode_query applies the model's query prompt (KURE-v3: "query: "); bge-m3 has none.
    query_vec = embedder.encode_query(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).reshape(1, -1).astype(np.float32)

    distances, indices = index.search(query_vec, top_k)

    results: List[Dict[str, Any]] = []
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
        if idx == -1 or idx >= len(id_map):
            continue
        entry = id_map[int(idx)]
        results.append({
            "rank": rank,
            "chunk_id": entry["chunk_id"],
            "doc_id": entry["doc_id"],
            "text": entry["text"],
            "tokens": entry["tokens"],
            "source_type": entry.get("source_type", ""),
            "domain_category": entry.get("domain_category", ""),
            "has_table": entry.get("has_table", False),
            "has_procedure": entry.get("has_procedure", False),
            "faiss_score": float(dist),
        })
    return results


def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    reranker: Any,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []

    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(
        pairs,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    for i, r in enumerate(ranked, start=1):
        r["rank"] = i

    return ranked[:top_k]


def search_and_rerank(
    query: str,
    index: Any,
    id_map: List[Dict[str, Any]],
    embedder: Any,
    reranker: Optional[Any],
    config: Dict[str, Any],
    top_k_override: Optional[int] = None,
    rerank_top_k_override: Optional[int] = None,
    skip_rerank: bool = False,
) -> tuple[List[Dict[str, Any]], Dict[str, float]]:
    top_k = top_k_override or config["reranker"]["top_k_retrieve"]
    rerank_top_k = rerank_top_k_override or config["reranker"]["top_k_rerank"]

    timing: Dict[str, float] = {}

    t0 = time.time()
    candidates = retrieve(query, index, id_map, embedder, top_k=top_k)
    timing["retrieve_ms"] = (time.time() - t0) * 1000

    if skip_rerank or reranker is None:
        timing["total_ms"] = timing["retrieve_ms"]
        return candidates, timing

    t0 = time.time()
    results = rerank(query, candidates, reranker, top_k=rerank_top_k)
    timing["rerank_ms"] = (time.time() - t0) * 1000
    timing["total_ms"] = timing["retrieve_ms"] + timing["rerank_ms"]

    return results, timing


def format_results(
    query: str,
    results: List[Dict[str, Any]],
    timing: Dict[str, float],
    use_rerank: bool,
) -> str:
    lines: List[str] = []
    lines.append(f"Query: {query}")
    if use_rerank:
        lines.append(f"Retrieve top-{len(results)}, reranked")
    else:
        lines.append(f"Retrieved top-{len(results)} (no rerank)")
    lines.append("")

    for r in results:
        score_parts = [f"faiss: {r['faiss_score']:.4f}"]
        if "rerank_score" in r:
            score_parts.append(f"rerank: {r['rerank_score']:.4f}")
        score_str = ", ".join(score_parts)

        lines.append(f"[{r['rank']}] {r['chunk_id']} ({score_str})")
        lines.append(
            f"    doc_id: {r['doc_id']} | tokens: {r['tokens']} | "
            f"domain: {r['domain_category']}"
        )
        preview = r["text"][:200].replace("\n", " ")
        lines.append(f"    {preview}...")
        lines.append("")

    timing_parts = []
    if "retrieve_ms" in timing:
        timing_parts.append(f"Search: {timing['retrieve_ms']:.0f}ms")
    if "rerank_ms" in timing:
        timing_parts.append(f"Rerank: {timing['rerank_ms']:.0f}ms")
    timing_parts.append(f"Total: {timing['total_ms']:.0f}ms")
    lines.append(" | ".join(timing_parts))

    return "\n".join(lines)


def format_results_json(
    query: str,
    results: List[Dict[str, Any]],
    timing: Dict[str, float],
) -> str:
    output = {
        "query": query,
        "results": results,
        "timing": timing,
    }
    return json.dumps(output, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search FAISS index with BGE-M3 embeddings and optional reranking."
    )
    parser.add_argument(
        "--config",
        default="/workspace/trainer/configs/rag_config.yaml",
        type=Path,
        help="Path to rag_config.yaml.",
    )
    parser.add_argument(
        "--query",
        required=True,
        type=str,
        help="Search query string.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override top-K retrieval count from config.",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=None,
        help="Override rerank top-K count from config.",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip reranking, return raw FAISS results.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output results as JSON.",
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

    index = load_faiss_index(
        config["paths"]["faiss_index"],
        config["index"]["ef_search"],
    )
    metadata = load_metadata(config["paths"]["metadata_json"], config["embedding"]["model_name"])
    id_map = metadata["id_map"]

    embedder = load_embedder(
        config["embedding"]["model_name"],
        config["embedding"]["device"],
        config["embedding"]["max_seq_length"],
    )

    reranker = None
    if not args.no_rerank:
        reranker = load_reranker(
            config["reranker"]["model_name"],
            config["reranker"]["device"],
            config["reranker"]["max_length"],
        )

    results, timing = search_and_rerank(
        args.query,
        index,
        id_map,
        embedder,
        reranker,
        config,
        top_k_override=args.top_k,
        rerank_top_k_override=args.rerank_top_k,
        skip_rerank=args.no_rerank,
    )

    if not results:
        logger.warning("No results found for query: %s", args.query)

    if args.json_output:
        print(format_results_json(args.query, results, timing))
    else:
        print(format_results(args.query, results, timing, not args.no_rerank))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
