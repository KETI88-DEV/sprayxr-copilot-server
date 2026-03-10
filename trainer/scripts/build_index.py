#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)

SECTION_RE = re.compile(
    r"(?:^|\n)(?="
    r"#{1,4}\s"
    r"|[A-Z\uAC00-\uD7A3][A-Z\uAC00-\uD7A3 \t]{4,}\n"
    r"|제\s*\d+\s*[장절조항]"
    r"|[0-9]+[.\)]\s"
    r")",
    re.MULTILINE,
)

HEADING_RE = re.compile(
    r"^(?:#{1,4}\s+.+|[A-Z][A-Z \t]{4,}|제\s*\d+\s*[장절조항].*)$",
    re.MULTILINE,
)

PROCEDURE_LINE_RE = re.compile(
    r"^\s*(?:\d+[.\)]\s|[•\-\*▪]\s)",
    re.MULTILINE,
)

DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "coating": ["도장", "도료", "코팅", "paint", "coating", "도막"],
    "blasting": ["블라스팅", "블라스트", "blast", "연마", "abrasive"],
    "safety": ["안전", "safety", "보호구", "위험", "hazard"],
    "defects": ["결함", "defect", "불량", "핀홀", "기포", "크레이터"],
    "thickness": ["두께", "thickness", "DFT", "WFT", "도막두께", "mil"],
}


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_tokenizer(tokenizer_path: str) -> Any:
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(tokenizer_path)


def count_tokens(text: str, tokenizer: Any) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def detect_domain(text: str) -> str:
    text_lower = text.lower()
    scores: Dict[str, int] = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        scores[domain] = sum(1 for kw in keywords if kw.lower() in text_lower)
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else "general"


def detect_procedure(text: str) -> bool:
    return bool(PROCEDURE_LINE_RE.search(text))


def split_sections(content: str) -> List[str]:
    parts = SECTION_RE.split(content)
    sections = [p.strip() for p in parts if p and p.strip()]
    return sections if sections else [content.strip()]


def split_paragraphs(text: str) -> List[str]:
    parts = re.split(r"\n{2,}", text)
    return [p.strip() for p in parts if p.strip()]


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?。])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text_by_tokens(
    text: str,
    tokenizer: Any,
    min_tokens: int,
    max_tokens: int,
    overlap_fraction: float,
    min_chunk_tokens: int,
) -> List[Dict[str, Any]]:
    token_count = count_tokens(text, tokenizer)
    if token_count <= max_tokens:
        if token_count < min_chunk_tokens:
            return []
        return [{"text": text, "tokens": token_count}]

    paragraphs = split_paragraphs(text)
    if len(paragraphs) <= 1:
        paragraphs = split_sentences(text)

    chunks: List[Dict[str, Any]] = []
    current_parts: List[str] = []
    current_tokens = 0
    overlap_tokens = int(max_tokens * overlap_fraction)

    for para in paragraphs:
        para_tokens = count_tokens(para, tokenizer)
        if para_tokens == 0:
            continue

        if para_tokens > max_tokens:
            if current_parts:
                combined = "\n\n".join(current_parts)
                combined_tokens = count_tokens(combined, tokenizer)
                if combined_tokens >= min_chunk_tokens:
                    chunks.append({"text": combined, "tokens": combined_tokens})
                current_parts = []
                current_tokens = 0

            sentences = split_sentences(para)
            sub_parts: List[str] = []
            sub_tokens = 0
            for sent in sentences:
                sent_tokens = count_tokens(sent, tokenizer)
                if sub_tokens + sent_tokens > max_tokens and sub_parts:
                    combined = " ".join(sub_parts)
                    combined_tokens = count_tokens(combined, tokenizer)
                    if combined_tokens >= min_chunk_tokens:
                        chunks.append({"text": combined, "tokens": combined_tokens})
                    keep_count = max(1, len(sub_parts) // 3)
                    sub_parts = sub_parts[-keep_count:]
                    sub_tokens = count_tokens(" ".join(sub_parts), tokenizer)
                sub_parts.append(sent)
                sub_tokens += sent_tokens
            if sub_parts:
                combined = " ".join(sub_parts)
                combined_tokens = count_tokens(combined, tokenizer)
                if combined_tokens >= min_chunk_tokens:
                    chunks.append({"text": combined, "tokens": combined_tokens})
            continue

        if current_tokens + para_tokens > max_tokens and current_parts:
            combined = "\n\n".join(current_parts)
            combined_tokens = count_tokens(combined, tokenizer)
            if combined_tokens >= min_chunk_tokens:
                chunks.append({"text": combined, "tokens": combined_tokens})
            keep_count = 0
            overlap_acc = 0
            for i in range(len(current_parts) - 1, -1, -1):
                pt = count_tokens(current_parts[i], tokenizer)
                if overlap_acc + pt > overlap_tokens:
                    break
                overlap_acc += pt
                keep_count += 1
            current_parts = current_parts[-keep_count:] if keep_count > 0 else []
            current_tokens = overlap_acc if keep_count > 0 else 0

        current_parts.append(para)
        current_tokens += para_tokens

    if current_parts:
        combined = "\n\n".join(current_parts)
        combined_tokens = count_tokens(combined, tokenizer)
        if combined_tokens >= min_chunk_tokens:
            chunks.append({"text": combined, "tokens": combined_tokens})

    return chunks


def chunk_table(
    table_text: str,
    tokenizer: Any,
    max_tokens: int,
    min_chunk_tokens: int,
) -> List[Dict[str, Any]]:
    token_count = count_tokens(table_text, tokenizer)
    if token_count < min_chunk_tokens:
        return []
    if token_count <= max_tokens:
        return [{"text": table_text, "tokens": token_count}]

    lines = table_text.split("\n")
    header_lines: List[str] = []
    data_lines: List[str] = []
    for i, line in enumerate(lines):
        if i < 2 or (i == 2 and re.match(r"^\|[\s\-|]+\|$", line)):
            header_lines.append(line)
        else:
            data_lines.append(line)

    header = "\n".join(header_lines)
    chunks: List[Dict[str, Any]] = []
    batch: List[str] = []

    for dline in data_lines:
        batch.append(dline)
        candidate = header + "\n" + "\n".join(batch) if header else "\n".join(batch)
        if count_tokens(candidate, tokenizer) > max_tokens and len(batch) > 1:
            prev = header + "\n" + "\n".join(batch[:-1]) if header else "\n".join(batch[:-1])
            pt = count_tokens(prev, tokenizer)
            if pt >= min_chunk_tokens:
                chunks.append({"text": prev, "tokens": pt})
            batch = [dline]

    if batch:
        final = header + "\n" + "\n".join(batch) if header else "\n".join(batch)
        ft = count_tokens(final, tokenizer)
        if ft >= min_chunk_tokens:
            chunks.append({"text": final, "tokens": ft})

    return chunks


def force_split_oversized(
    chunks: List[Dict[str, Any]],
    tokenizer: Any,
    max_tokens: int,
    min_chunk_tokens: int,
) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for ch in chunks:
        if ch["tokens"] <= max_tokens:
            result.append(ch)
            continue
        text = ch["text"]
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        start = 0
        while start < len(token_ids):
            end = min(start + max_tokens, len(token_ids))
            sub_ids = token_ids[start:end]
            sub_text = tokenizer.decode(sub_ids, skip_special_tokens=True).strip()
            sub_tokens = len(sub_ids)
            if sub_tokens >= min_chunk_tokens and sub_text:
                new_ch = dict(ch)
                new_ch["text"] = sub_text
                new_ch["tokens"] = sub_tokens
                result.append(new_ch)
            start = end
    return result


def chunk_document(
    doc: Dict[str, Any],
    tokenizer: Any,
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    chunking_cfg = config["chunking"]
    min_tokens = chunking_cfg["min_tokens"]
    max_tokens = chunking_cfg["max_tokens"]
    overlap_fraction = chunking_cfg["overlap_fraction"]
    min_chunk_tokens = chunking_cfg["min_chunk_tokens"]

    doc_id = doc["doc_id"]
    content = doc.get("content", "")
    tables = doc.get("tables", [])
    source_type = doc.get("format", "txt")

    all_chunks: List[Dict[str, Any]] = []

    if content and isinstance(content, str) and content.strip():
        sections = split_sections(content)
        for section in sections:
            section_chunks = chunk_text_by_tokens(
                section, tokenizer, min_tokens, max_tokens,
                overlap_fraction, min_chunk_tokens,
            )
            for ch in section_chunks:
                ch["has_table"] = False
                ch["has_procedure"] = detect_procedure(ch["text"])
                ch["domain_category"] = detect_domain(ch["text"])
                ch["source_type"] = source_type
                all_chunks.append(ch)

    for tbl in tables:
        if not isinstance(tbl, str) or not tbl.strip():
            continue
        table_chunks = chunk_table(tbl, tokenizer, max_tokens, min_chunk_tokens)
        for ch in table_chunks:
            ch["has_table"] = True
            ch["has_procedure"] = False
            ch["domain_category"] = detect_domain(ch["text"])
            ch["source_type"] = source_type
            all_chunks.append(ch)

    all_chunks = force_split_oversized(all_chunks, tokenizer, max_tokens, min_chunk_tokens)

    for seq, ch in enumerate(all_chunks, start=1):
        ch["chunk_id"] = f"DOC-{doc_id}-C{seq:04d}"
        ch["doc_id"] = doc_id

    return all_chunks


def load_documents(input_dir: Path) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for path in sorted(input_dir.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data.get("content"), str):
                logger.debug("Skipping non-document file: %s", path.name)
                continue
            docs.append(data)
        except Exception as exc:
            logger.warning("Failed to load %s: %s", path.name, exc)
    return docs


def embed_chunks(
    texts: List[str],
    config: Dict[str, Any],
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    emb_cfg = config["embedding"]
    logger.info("Loading embedding model: %s", emb_cfg["model_name"])
    model = SentenceTransformer(
        emb_cfg["model_name"],
        device=emb_cfg["device"],
        model_kwargs={"use_safetensors": True},
    )
    model.max_seq_length = emb_cfg["max_seq_length"]

    logger.info("Encoding %d chunks (batch_size=%d)...", len(texts), emb_cfg["batch_size"])
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=emb_cfg["batch_size"],
        normalize_embeddings=emb_cfg.get("normalize", True),
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    elapsed = time.time() - t0
    logger.info(
        "Embedding complete: shape=%s, dtype=%s, time=%.1fs",
        embeddings.shape, embeddings.dtype, elapsed,
    )
    return embeddings


def build_faiss_index(
    embeddings: np.ndarray,
    config: Dict[str, Any],
) -> Any:
    import faiss

    idx_cfg = config["index"]
    d = embeddings.shape[1]
    m = idx_cfg["hnsw_m"]

    logger.info("Building FAISS HNSW index: d=%d, M=%d", d, m)
    index = faiss.IndexHNSWFlat(d, m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = idx_cfg["ef_construction"]
    index.hnsw.efSearch = idx_cfg["ef_search"]

    vecs = np.ascontiguousarray(embeddings, dtype=np.float32)
    t0 = time.time()
    index.add(vecs)
    elapsed = time.time() - t0

    logger.info("Index built: %d vectors, time=%.1fs", index.ntotal, elapsed)
    return index


def save_outputs(
    chunks: List[Dict[str, Any]],
    embeddings: Optional[np.ndarray],
    index: Any,
    config: Dict[str, Any],
) -> None:
    import faiss

    paths_cfg = config["paths"]

    jsonl_path = Path(paths_cfg["chunks_jsonl"])
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for ch in chunks:
            row = {
                "chunk_id": ch["chunk_id"],
                "text": ch["text"],
                "doc_id": ch["doc_id"],
                "tokens": ch["tokens"],
                "source_type": ch["source_type"],
                "domain_category": ch["domain_category"],
                "has_table": ch["has_table"],
                "has_procedure": ch["has_procedure"],
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Saved chunks.jsonl: %d lines → %s", len(chunks), jsonl_path)

    if index is not None:
        index_path = Path(paths_cfg["faiss_index"])
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_path))
        size_mb = index_path.stat().st_size / 1e6
        logger.info("Saved FAISS index: %s (%.1f MB)", index_path, size_mb)

    if index is not None:
        metadata = {
            "index_type": config["index"]["type"],
            "metric": config["index"]["metric"],
            "dimension": config["embedding"]["dimension"],
            "hnsw_m": config["index"]["hnsw_m"],
            "ef_construction": config["index"]["ef_construction"],
            "total_vectors": len(chunks),
            "embedding_model": config["embedding"]["model_name"],
            "tokenizer": str(config["chunking"]["tokenizer_path"]),
            "chunk_size_range": [
                config["chunking"]["min_tokens"],
                config["chunking"]["max_tokens"],
            ],
            "overlap_fraction": config["chunking"]["overlap_fraction"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "id_map": [],
        }
        for i, ch in enumerate(chunks):
            metadata["id_map"].append({
                "faiss_id": i,
                "chunk_id": ch["chunk_id"],
                "doc_id": ch["doc_id"],
                "text": ch["text"],
                "tokens": ch["tokens"],
                "source_type": ch["source_type"],
                "domain_category": ch["domain_category"],
                "has_table": ch["has_table"],
                "has_procedure": ch["has_procedure"],
            })

        meta_path = Path(paths_cfg["metadata_json"])
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, ensure_ascii=False, indent=2)
        logger.info("Saved metadata: %s", meta_path)


def print_summary(chunks: List[Dict[str, Any]], elapsed: float) -> None:
    if not chunks:
        logger.info("No chunks produced.")
        return

    token_counts = [c["tokens"] for c in chunks]
    domain_dist: Dict[str, int] = {}
    tables = 0
    procedures = 0
    for c in chunks:
        domain_dist[c["domain_category"]] = domain_dist.get(c["domain_category"], 0) + 1
        if c["has_table"]:
            tables += 1
        if c["has_procedure"]:
            procedures += 1

    doc_ids = set(c["doc_id"] for c in chunks)

    logger.info("=== Chunking Summary ===")
    logger.info("  total_docs:        %d", len(doc_ids))
    logger.info("  total_chunks:      %d", len(chunks))
    logger.info("  avg_tokens:        %.1f", sum(token_counts) / len(token_counts))
    logger.info("  min_tokens:        %d", min(token_counts))
    logger.info("  max_tokens:        %d", max(token_counts))
    logger.info("  chunks_with_table: %d (%.1f%%)", tables, 100 * tables / len(chunks))
    logger.info("  chunks_with_proc:  %d (%.1f%%)", procedures, 100 * procedures / len(chunks))
    logger.info("  domain_distribution:")
    for domain in sorted(domain_dist, key=lambda d: domain_dist[d], reverse=True):
        logger.info("    %-12s %d (%.1f%%)", domain, domain_dist[domain], 100 * domain_dist[domain] / len(chunks))
    logger.info("  elapsed:           %.1fs", elapsed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build RAG index: token-based chunking → BGE-M3 embedding → FAISS HNSW index."
    )
    parser.add_argument(
        "--config",
        default="/workspace/trainer/configs/rag_config.yaml",
        type=Path,
        help="Path to rag_config.yaml.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chunk documents only, skip embedding and index building.",
    )
    parser.add_argument(
        "--skip-embed",
        action="store_true",
        help="Chunk and save chunks.jsonl but skip embedding and index.",
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
    input_dir = Path(config["paths"]["input_dir"])
    if not input_dir.is_dir():
        logger.error("Input directory not found: %s", input_dir)
        return 1

    logger.info("Loading Qwen2.5 tokenizer from %s", config["chunking"]["tokenizer_path"])
    tokenizer = load_tokenizer(config["chunking"]["tokenizer_path"])

    logger.info("Loading documents from %s", input_dir)
    docs = load_documents(input_dir)
    logger.info("Loaded %d documents", len(docs))

    t0 = time.time()
    all_chunks: List[Dict[str, Any]] = []
    for doc in docs:
        doc_chunks = chunk_document(doc, tokenizer, config)
        all_chunks.extend(doc_chunks)
        logger.debug("  %s → %d chunks", doc["doc_id"], len(doc_chunks))

    chunk_elapsed = time.time() - t0
    print_summary(all_chunks, chunk_elapsed)

    if args.dry_run:
        logger.info("Dry run complete. No files written.")
        return 0

    if args.skip_embed:
        save_outputs(all_chunks, None, None, config)
        logger.info("Chunks saved. Embedding/index skipped (--skip-embed).")
        return 0

    texts = [c["text"] for c in all_chunks]
    embeddings = embed_chunks(texts, config)

    if embeddings.shape[0] != len(all_chunks):
        logger.error(
            "Embedding count mismatch: %d embeddings vs %d chunks",
            embeddings.shape[0], len(all_chunks),
        )
        return 1

    index = build_faiss_index(embeddings, config)
    save_outputs(all_chunks, embeddings, index, config)

    logger.info("Build complete. Total time: %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
