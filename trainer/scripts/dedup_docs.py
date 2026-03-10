#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


logger = logging.getLogger(__name__)


def load_datasketch() -> Tuple[Any, Any]:
    try:
        module = importlib.import_module("datasketch")
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasketch"])
        module = importlib.import_module("datasketch")
    return module.MinHash, module.MinHashLSH


MinHash, MinHashLSH = load_datasketch()


@dataclass
class DocRecord:
    path: Path
    payload: Dict[str, Any]


def normalize_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def compute_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def generate_char_ngrams(content: str, n: int = 5) -> Set[str]:
    if len(content) < n:
        return {content} if content else set()
    return {content[idx : idx + n] for idx in range(len(content) - n + 1)}


def build_minhash(content: str, num_perm: int = 128) -> Any:
    minhash = MinHash(num_perm=num_perm)
    shingles = generate_char_ngrams(content, n=5)
    if not shingles and content:
        shingles = {content}
    for shingle in shingles:
        minhash.update(shingle.encode("utf-8"))
    return minhash


def discover_files(input_dir: Path) -> List[Path]:
    return sorted(input_dir.glob("*.json"), key=lambda p: p.name)


def load_documents(files: List[Path]) -> List[DocRecord]:
    records: List[DocRecord] = []
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Skipping unreadable JSON %s: %s", path.name, exc)
            continue

        doc_id = str(payload.get("doc_id") or "").strip()
        if not doc_id:
            logger.warning("Skipping JSON without doc_id: %s", path.name)
            continue

        payload["content"] = normalize_content(payload.get("content", ""))
        payload.setdefault("tables", [])
        payload.setdefault("cleaning_log", {})
        payload["dedup_log"] = {
            "stage_1_exact": {"removed": False, "duplicate_of": None},
            "stage_2_near": {"similarity": 0.0, "duplicate_of": None},
            "stage_3_para": {"paragraphs_removed": 0, "paragraphs_total": 0},
        }
        records.append(DocRecord(path=path, payload=payload))

    records.sort(key=lambda record: str(record.payload["doc_id"]))
    return records


def stage_exact_dedup(records: List[DocRecord]) -> Tuple[List[DocRecord], int]:
    kept: List[DocRecord] = []
    content_hash_to_doc_id: Dict[str, str] = {}
    duplicates = 0

    for record in records:
        doc_id = str(record.payload["doc_id"])
        content = record.payload["content"]
        digest = compute_sha256(content)

        original_doc_id = content_hash_to_doc_id.get(digest)
        if original_doc_id:
            duplicates += 1
            record.payload["dedup_log"]["stage_1_exact"]["removed"] = True
            record.payload["dedup_log"]["stage_1_exact"]["duplicate_of"] = original_doc_id
            logger.info("Stage 1 exact duplicate: %s -> %s", doc_id, original_doc_id)
            continue

        content_hash_to_doc_id[digest] = doc_id
        kept.append(record)

    return kept, duplicates


def stage_near_dedup(records: List[DocRecord], jaccard_threshold: float) -> Tuple[List[DocRecord], int]:
    if not records:
        return records, 0

    ordered = sorted(records, key=lambda record: str(record.payload["doc_id"]))
    lsh = MinHashLSH(threshold=jaccard_threshold, num_perm=128)
    minhash_map: Dict[str, Any] = {}
    active_map: Dict[str, DocRecord] = {}
    near_duplicates = 0

    for record in ordered:
        doc_id = str(record.payload["doc_id"])
        content = record.payload["content"]
        minhash = build_minhash(content, num_perm=128)
        candidates = [candidate for candidate in lsh.query(minhash) if candidate in active_map]

        best_candidate: Optional[str] = None
        best_similarity = 0.0
        for candidate in candidates:
            similarity = float(minhash.jaccard(minhash_map[candidate]))
            if similarity > best_similarity:
                best_similarity = similarity
                best_candidate = candidate

        if best_candidate and best_similarity >= jaccard_threshold:
            candidate_record = active_map[best_candidate]
            candidate_content = candidate_record.payload["content"]
            candidate_length = len(candidate_content)
            current_length = len(content)

            if current_length > candidate_length:
                near_duplicates += 1
                candidate_record.payload["dedup_log"]["stage_2_near"]["similarity"] = best_similarity
                candidate_record.payload["dedup_log"]["stage_2_near"]["duplicate_of"] = doc_id
                logger.info(
                    "Stage 2 near-duplicate: replacing shorter %s with %s (sim=%.4f)",
                    best_candidate,
                    doc_id,
                    best_similarity,
                )
                lsh.remove(best_candidate)
                active_map.pop(best_candidate, None)
                minhash_map.pop(best_candidate, None)

                active_map[doc_id] = record
                minhash_map[doc_id] = minhash
                lsh.insert(doc_id, minhash)
            else:
                near_duplicates += 1
                record.payload["dedup_log"]["stage_2_near"]["similarity"] = best_similarity
                record.payload["dedup_log"]["stage_2_near"]["duplicate_of"] = best_candidate
                logger.info(
                    "Stage 2 near-duplicate: dropping %s, keeping %s (sim=%.4f)",
                    doc_id,
                    best_candidate,
                    best_similarity,
                )
            continue

        active_map[doc_id] = record
        minhash_map[doc_id] = minhash
        lsh.insert(doc_id, minhash)

    kept = [active_map[key] for key in sorted(active_map.keys())]
    return kept, near_duplicates


def stage_paragraph_dedup(records: List[DocRecord], min_para_len: int) -> int:
    paragraph_hash_owner: Dict[str, str] = {}
    total_removed = 0

    for record in sorted(records, key=lambda item: str(item.payload["doc_id"])):
        doc_id = str(record.payload["doc_id"])
        content = record.payload["content"]
        paragraphs = content.split("\n\n") if content else []
        total_paragraphs = len(paragraphs)

        record.payload["dedup_log"]["stage_3_para"]["paragraphs_total"] = total_paragraphs

        removable_indexes: Set[int] = set()
        for idx, paragraph in enumerate(paragraphs):
            normalized = paragraph.strip()
            if len(normalized) < min_para_len:
                continue

            para_hash = compute_sha256(normalized)
            owner = paragraph_hash_owner.get(para_hash)
            if owner is not None:
                removable_indexes.add(idx)
            else:
                paragraph_hash_owner[para_hash] = doc_id

        if not removable_indexes:
            continue

        filtered_paragraphs = [p for idx, p in enumerate(paragraphs) if idx not in removable_indexes]
        updated_content = "\n\n".join(filtered_paragraphs).strip()

        original_length = len(content)
        remaining_ratio = 1.0 if original_length == 0 else (len(updated_content) / original_length)

        if remaining_ratio < 0.3:
            logger.info(
                "Stage 3 paragraph dedup skipped for %s: remaining ratio %.4f below threshold",
                doc_id,
                remaining_ratio,
            )
            continue

        removed_count = len(removable_indexes)
        record.payload["content"] = updated_content
        record.payload["dedup_log"]["stage_3_para"]["paragraphs_removed"] = removed_count
        total_removed += removed_count
        logger.info("Stage 3 paragraph dedup: removed %s paragraphs from %s", removed_count, doc_id)

    return total_removed


def write_outputs(records: List[DocRecord], output_dir: Path, dry_run: bool) -> None:
    if dry_run:
        for record in records:
            logger.info("Dry run: would write %s", output_dir / f"{record.payload['doc_id']}.json")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        output_path = output_dir / f"{record.payload['doc_id']}.json"
        output_path.write_text(
            json.dumps(record.payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def write_report(
    output_dir: Path,
    total_input: int,
    exact_duplicates: int,
    near_duplicates: int,
    paragraph_dedup_count: int,
    total_output: int,
    dry_run: bool,
) -> Dict[str, Any]:
    dedup_rate = 0.0
    if total_input > 0:
        dedup_rate = (total_input - total_output) / total_input

    report: Dict[str, Any] = {
        "total_input": total_input,
        "exact_duplicates": exact_duplicates,
        "near_duplicates": near_duplicates,
        "paragraph_dedup_count": paragraph_dedup_count,
        "total_output": total_output,
        "dedup_rate": dedup_rate,
    }

    report_path = output_dir / "dedup_report.json"
    if dry_run:
        logger.info("Dry run: would write %s", report_path)
        return report

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deduplicate cleaned JSON documents using exact, near, and paragraph-level matching."
    )
    parser.add_argument(
        "--input-dir",
        default="/data/cleaned",
        type=Path,
        help="Directory containing cleaned JSON documents.",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/deduplicated",
        type=Path,
        help="Directory where deduplicated JSON files are written.",
    )
    parser.add_argument(
        "--jaccard-threshold",
        default=0.8,
        type=float,
        help="Jaccard similarity threshold for MinHash near-duplicate detection.",
    )
    parser.add_argument(
        "--min-para-len",
        default=512,
        type=int,
        help="Minimum paragraph length to consider for paragraph-level deduplication.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run deduplication and logging without writing output files.",
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

    if not args.input_dir.is_dir():
        logger.error("Input directory not found: %s", args.input_dir)
        return 1

    files = discover_files(args.input_dir)
    if not files:
        logger.warning("No JSON files found in %s", args.input_dir)

    records = load_documents(files)
    total_input = len(records)
    logger.info("Loaded %s input documents", total_input)

    stage1_records, exact_duplicates = stage_exact_dedup(records)
    logger.info("Stage 1 complete: kept=%s exact_duplicates=%s", len(stage1_records), exact_duplicates)

    stage2_records, near_duplicates = stage_near_dedup(stage1_records, args.jaccard_threshold)
    logger.info("Stage 2 complete: kept=%s near_duplicates=%s", len(stage2_records), near_duplicates)

    paragraph_dedup_count = stage_paragraph_dedup(stage2_records, args.min_para_len)
    logger.info("Stage 3 complete: paragraph_dedup_count=%s", paragraph_dedup_count)

    write_outputs(stage2_records, args.output_dir, args.dry_run)

    report = write_report(
        output_dir=args.output_dir,
        total_input=total_input,
        exact_duplicates=exact_duplicates,
        near_duplicates=near_duplicates,
        paragraph_dedup_count=paragraph_dedup_count,
        total_output=len(stage2_records),
        dry_run=args.dry_run,
    )

    logger.info("Finished. report=%s", json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
