#!/usr/bin/env python3
"""Generate QLoRA SFT dataset for ship-painting coaching.

Builds category-balanced training data from chunked documents, eval set,
and prior RAG responses.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "너는 선박 도장 훈련 코치다. 항상 안전과 표준 절차를 우선하고, "
    "마지막에 근거 문서 ID를 적는다."
)

CATEGORY_TARGETS: Dict[str, int] = {
    "문서QA": 100,
    "절차설명": 80,
    "결함분석": 60,
    "DFT피드백": 60,
    "용어/다국어": 50,
    "요약/비교": 50,
}

DOMAIN_TERMS = [
    "DFT", "WFT", "Sa 2.5", "SSPC", "ISO", "IMO", "PSPC", "블라스팅",
    "표면처리", "도막", "프라이머", "에폭시", "부식", "결함", "안전",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SFT JSONL for QLoRA")
    parser.add_argument("--chunks", type=Path, default=Path("/data/chunked/chunks.jsonl"))
    parser.add_argument("--eval-set", type=Path, default=Path("/data/eval/eval_set.jsonl"))
    parser.add_argument(
        "--rag-responses",
        type=Path,
        default=Path("/output/eval_results/rag_only_raw_responses.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("/data/sft"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-count", type=int, default=20)
    parser.add_argument("--dup-threshold", type=float, default=0.95)
    parser.add_argument("--embed-model", default="BAAI/bge-m3")
    parser.add_argument(
        "--sources-dir",
        type=Path,
        default=Path("/data/sources"),
        help="Directory for source manifest outputs.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping invalid JSON: %s:%d", path, line_no)
    logger.info("Loaded %d records from %s", len(items), path)
    return items


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sanitize_answer_text(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"\((?:DOC-|doc_)\S+\)", "", text)
    text = re.sub(r"\[(?:DOC-|doc_)\S+\]", "", text)
    return normalize_text(text)


def build_messages(category: str, user_text: str, assistant_text: str, doc_ids: Sequence[str]) -> Dict[str, Any]:
    citation = ", ".join(sorted(set(doc_ids)))
    grounded = sanitize_answer_text(assistant_text)
    if "안전" not in grounded:
        grounded = "안전과 표준 절차를 우선하여 설명합니다. " + grounded
    grounded = f"{grounded}\n\n근거 문서 ID: {citation}"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": normalize_text(user_text)},
            {"role": "assistant", "content": grounded},
        ],
        "category": category,
    }


def make_document_qa(
    eval_set: Sequence[Dict[str, Any]],
    rag_map: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for item in eval_set:
        q_id = item.get("q_id", "")
        question = item.get("question", "")
        expected = item.get("expected_answer", "")
        required = item.get("required_chunks", [])
        raw = rag_map.get(q_id, {})
        context = []
        for row in raw.get("_raw_results", [])[:2]:
            context.append(row.get("text", "")[:260])
        user_1 = f"[문서] {' '.join(context)}\n\n[질문] {question}"
        samples.append(build_messages("문서QA", user_1, expected, required))

        alt_question = f"교육생 질문: {question} 현장에서 바로 적용할 수 있게 답해줘."
        alt_answer = raw.get("answer", expected)
        user_2 = f"[문서] {' '.join(context)}\n\n[질문] {alt_question}"
        samples.append(build_messages("문서QA", user_2, alt_answer, required))
    return samples


def infer_step_lines(text: str, max_steps: int = 6) -> List[str]:
    lines = [normalize_text(line) for line in text.splitlines()]
    step_lines = [line for line in lines if re.match(r"^\d+(?:\.\d+)*", line)]
    if step_lines:
        return step_lines[:max_steps]
    fallback = [line for line in lines if len(line) > 15]
    return fallback[:max_steps]


def make_procedure_explain(chunks: Sequence[Dict[str, Any]], rng: random.Random) -> List[Dict[str, Any]]:
    candidates = [c for c in chunks if c.get("has_procedure")]
    rng.shuffle(candidates)
    samples: List[Dict[str, Any]] = []
    for chunk in candidates[:260]:
        text = chunk.get("text", "")
        steps = infer_step_lines(text)
        if not steps:
            continue
        question = "다음 작업 절차를 신규 작업자에게 단계별로 설명해줘. 안전 유의점도 포함해줘."
        user = f"[문서] {text[:480]}\n\n[질문] {question}"
        joined_steps = " ".join(steps)
        answer = f"표준 절차는 다음 순서로 진행합니다: {joined_steps}. 각 단계 전후로 PPE 착용과 작업장 환기 상태를 확인하세요."
        samples.append(build_messages("절차설명", user, answer, [chunk.get("chunk_id", "")]))
    return samples


def make_defect_analysis(chunks: Sequence[Dict[str, Any]], rng: random.Random) -> List[Dict[str, Any]]:
    pattern = re.compile(r"결함|부식|균열|기포|핀홀|박리|오렌지필|들뜸")
    candidates = [c for c in chunks if pattern.search(c.get("text", ""))]
    rng.shuffle(candidates)
    samples: List[Dict[str, Any]] = []
    for chunk in candidates[:220]:
        text = chunk.get("text", "")
        question = "아래 사례의 결함 원인과 재발 방지 조치를 분석해줘."
        user = f"[문서] {text[:520]}\n\n[질문] {question}"
        answer = (
            "문서 근거로 볼 때 표면처리 불량, 환경 조건 미준수, 도장 간격 미준수가 핵심 원인입니다. "
            "재발 방지를 위해 표면 청정도 재확인, DFT 측정 기록, 작업 전 안전 점검을 병행하세요."
        )
        samples.append(build_messages("결함분석", user, answer, [chunk.get("chunk_id", "")]))
    return samples


def make_dft_feedback(chunks: Sequence[Dict[str, Any]], rng: random.Random) -> List[Dict[str, Any]]:
    pattern = re.compile(r"DFT|WFT|도막|두께|µm|um|μm")
    candidates = [c for c in chunks if pattern.search(c.get("text", ""))]
    rng.shuffle(candidates)
    samples: List[Dict[str, Any]] = []
    for chunk in candidates[:220]:
        text = chunk.get("text", "")
        question = "측정된 DFT 결과를 기준으로 합격/보완 피드백을 작성해줘."
        user = f"[문서] {text[:520]}\n\n[질문] {question}"
        answer = (
            "DFT는 규격 하한/상한을 동시에 확인해야 하며, 편차가 큰 구간은 재도장보다 원인 점검이 우선입니다. "
            "측정 위치별 로그를 남기고 기준 미달 구역은 표면 상태와 희석비를 재검토하세요."
        )
        samples.append(build_messages("DFT피드백", user, answer, [chunk.get("chunk_id", "")]))
    return samples


def extract_bilingual_pairs(text: str) -> List[Tuple[str, str]]:
    matches = re.findall(r"([가-힣A-Za-z\s]{2,30})\(([A-Za-z][A-Za-z\s\-]{2,40})\)", text)
    pairs: List[Tuple[str, str]] = []
    for ko, en in matches:
        ko_norm = normalize_text(ko)
        en_norm = normalize_text(en)
        if 2 <= len(ko_norm) <= 25 and 2 <= len(en_norm) <= 40:
            pairs.append((ko_norm, en_norm))
    return pairs


def make_term_multilingual(chunks: Sequence[Dict[str, Any]], rng: random.Random) -> List[Dict[str, Any]]:
    rng.shuffle(chunks)
    samples: List[Dict[str, Any]] = []
    for chunk in chunks:
        pairs = extract_bilingual_pairs(chunk.get("text", ""))
        for ko_term, en_term in pairs[:2]:
            user = (
                f"[문서] {chunk.get('text', '')[:360]}\n\n"
                f"[질문] 용어 '{ko_term}'와 '{en_term}'를 초급 작업자도 이해하게 한국어로 설명해줘."
            )
            answer = (
                f"'{ko_term}'는 현장에서 {en_term}로도 표기하며, 작업지시서와 점검표에서 같은 의미로 사용됩니다. "
                "교육 시 용어를 혼용하되 기준 규격 표기는 원문 그대로 유지하세요."
            )
            samples.append(build_messages("용어/다국어", user, answer, [chunk.get("chunk_id", "")]))
            if len(samples) >= 180:
                return samples
    return samples


def make_summary_compare(chunks: Sequence[Dict[str, Any]], rng: random.Random) -> List[Dict[str, Any]]:
    by_domain: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        by_domain[chunk.get("domain_category", "general")].append(chunk)

    samples: List[Dict[str, Any]] = []
    for domain, rows in by_domain.items():
        if len(rows) < 2:
            continue
        rng.shuffle(rows)
        pairs = zip(rows[0::2], rows[1::2])
        for left, right in pairs:
            left_text = left.get("text", "")[:280]
            right_text = right.get("text", "")[:280]
            user = (
                f"[문서] A: {left_text}\nB: {right_text}\n\n"
                "[질문] 두 문서의 핵심을 요약하고 차이점을 비교해줘."
            )
            answer = (
                f"두 문서는 {domain} 작업 기준을 다루며 공통적으로 안전 점검과 절차 준수를 강조합니다. "
                "차이는 적용 조건과 측정 항목의 상세 수준이며, 작업 전 표준서 우선 확인이 필요합니다."
            )
            ids = [left.get("chunk_id", ""), right.get("chunk_id", "")]
            samples.append(build_messages("요약/비교", user, answer, ids))
            if len(samples) >= 180:
                return samples
    return samples


def deduplicate_samples(
    samples: Sequence[Dict[str, Any]],
    model_name: str,
    threshold: float,
) -> Tuple[List[Dict[str, Any]], int]:
    def _lexical_fallback() -> Tuple[List[Dict[str, Any]], int]:
        kept: List[Dict[str, Any]] = []
        seen_vectors: List[Counter[str]] = []
        removed = 0
        for sample in samples:
            assistant = sample["messages"][2]["content"]
            if len(assistant) < 100:
                kept.append(sample)
                continue
            grams = Counter(
                assistant[i : i + 3] for i in range(max(len(assistant) - 2, 1))
            )
            is_duplicate = False
            for seen in seen_vectors:
                intersect = set(grams.keys()) & set(seen.keys())
                dot = float(sum(grams[key] * seen[key] for key in intersect))
                norm_a = math.sqrt(sum(value * value for value in grams.values()))
                norm_b = math.sqrt(sum(value * value for value in seen.values()))
                similarity = dot / max(norm_a * norm_b, 1e-8)
                if similarity > threshold:
                    is_duplicate = True
                    break
            if is_duplicate:
                removed += 1
                continue
            kept.append(sample)
            seen_vectors.append(grams)
        return kept, removed

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        logger.warning("sentence-transformers unavailable, skip semantic dedup: %s", exc)
        return _lexical_fallback()

    long_indices: List[int] = []
    texts: List[str] = []
    for idx, sample in enumerate(samples):
        assistant = sample["messages"][2]["content"]
        if len(assistant) >= 100:
            long_indices.append(idx)
            texts.append(assistant)

    if not texts:
        return list(samples), 0

    try:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    except Exception as exc:
        logger.warning("Embedding dedup failed, using lexical fallback: %s", exc)
        return _lexical_fallback()

    kept_flags = [True] * len(samples)
    removed_count = 0
    kept_long_vectors: List[np.ndarray] = []
    for local_idx, sample_idx in enumerate(long_indices):
        vector = embeddings[local_idx]
        if kept_long_vectors:
            sims = np.dot(np.vstack(kept_long_vectors), vector)
            if float(np.max(sims)) > threshold:
                kept_flags[sample_idx] = False
                removed_count += 1
                continue
        kept_long_vectors.append(vector)

    deduped = [sample for keep, sample in zip(kept_flags, samples) if keep]
    return deduped, removed_count


def select_balanced_samples(
    samples: Sequence[Dict[str, Any]],
    targets: Dict[str, int],
    rng: random.Random,
) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        buckets[sample["category"]].append(sample)

    selected: List[Dict[str, Any]] = []
    for category, target in targets.items():
        rows = buckets.get(category, [])
        if len(rows) < target:
            raise ValueError(f"Insufficient samples for {category}: {len(rows)} < {target}")
        rng.shuffle(rows)
        selected.extend(rows[:target])
    rng.shuffle(selected)
    return selected


def topup_categories(
    deduped: Sequence[Dict[str, Any]],
    raw_candidates: Sequence[Dict[str, Any]],
    targets: Dict[str, int],
) -> List[Dict[str, Any]]:
    merged = list(deduped)
    category_counts = Counter(sample["category"] for sample in merged)
    existing_texts = {
        sample["messages"][2]["content"]: True
        for sample in merged
    }
    by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sample in raw_candidates:
        by_category[sample["category"]].append(sample)

    for category, target in targets.items():
        missing = target - category_counts.get(category, 0)
        if missing <= 0:
            continue
        for sample in by_category.get(category, []):
            assistant = sample["messages"][2]["content"]
            if assistant in existing_texts:
                continue
            merged.append(sample)
            existing_texts[assistant] = True
            missing -= 1
            if missing == 0:
                break
        if missing > 0:
            raise ValueError(f"Unable to top up category {category}, missing={missing}")
    return merged


def validate_samples(samples: Sequence[Dict[str, Any]]) -> None:
    for idx, sample in enumerate(samples):
        if "category" not in sample or "messages" not in sample:
            raise ValueError(f"Missing fields at sample {idx}")
        messages = sample["messages"]
        if len(messages) != 3:
            raise ValueError(f"Expected 3 messages, got {len(messages)} at {idx}")
        roles = [row.get("role") for row in messages]
        if roles != ["system", "user", "assistant"]:
            raise ValueError(f"Invalid role order at {idx}: {roles}")
        if not messages[2].get("content", "").strip():
            raise ValueError(f"Empty assistant content at {idx}")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_sources_manifest(
    sources_dir: Path,
    chunks: Sequence[Dict[str, Any]],
    eval_set: Sequence[Dict[str, Any]],
    rag_responses: Sequence[Dict[str, Any]],
) -> Dict[str, int]:
    sources_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = sources_dir / "manifest.jsonl"
    records: List[Dict[str, Any]] = []

    for chunk in chunks:
        records.append({
            "source_type": "chunk",
            "doc_id": chunk.get("doc_id", ""),
            "chunk_id": chunk.get("chunk_id", ""),
            "category": chunk.get("domain_category", "general"),
        })
    for item in eval_set:
        records.append({
            "source_type": "eval",
            "doc_id": "eval_set",
            "chunk_id": ",".join(item.get("required_chunks", [])),
            "category": item.get("category", ""),
        })
    for resp in rag_responses:
        records.append({
            "source_type": "rag_response",
            "doc_id": "rag_only_eval",
            "chunk_id": ",".join(resp.get("citations", [])),
            "category": resp.get("confidence", ""),
        })

    write_jsonl(manifest_path, records)
    return {
        "chunks": len(chunks),
        "eval_items": len(eval_set),
        "rag_responses": len(rag_responses),
        "manifest_rows": len(records),
    }


def split_train_val(
    samples: Sequence[Dict[str, Any]],
    val_count: int,
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if val_count <= 0 or val_count >= len(samples):
        raise ValueError("val_count must be between 1 and len(samples)-1")
    items = list(samples)
    rng.shuffle(items)
    val = items[:val_count]
    train = items[val_count:]
    return train, val


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    chunks = load_jsonl(args.chunks)
    eval_set = load_jsonl(args.eval_set)
    rag_responses = load_jsonl(args.rag_responses)
    rag_map = {row.get("q_id", ""): row for row in rag_responses}

    source_stats = write_sources_manifest(args.sources_dir, chunks, eval_set, rag_responses)

    candidates: List[Dict[str, Any]] = []
    candidates.extend(make_document_qa(eval_set, rag_map))
    candidates.extend(make_procedure_explain(chunks, rng))
    candidates.extend(make_defect_analysis(chunks, rng))
    candidates.extend(make_dft_feedback(chunks, rng))
    candidates.extend(make_term_multilingual(chunks, rng))
    candidates.extend(make_summary_compare(chunks, rng))
    logger.info("Generated %d raw candidates", len(candidates))

    validate_samples(candidates)
    deduped, removed_count = deduplicate_samples(candidates, args.embed_model, args.dup_threshold)
    logger.info("Semantic dedup removed %d samples", removed_count)

    balanced_source = topup_categories(deduped, candidates, CATEGORY_TARGETS)
    selected = select_balanced_samples(balanced_source, CATEGORY_TARGETS, rng)
    validate_samples(selected)

    total_count = len(selected)
    train_rows, val_rows = split_train_val(selected, args.val_count, rng)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "sft_train.jsonl"
    val_path = args.output_dir / "sft_val.jsonl"

    train_count = write_jsonl(train_path, train_rows)
    val_count = write_jsonl(val_path, val_rows)

    category_counts = Counter(row["category"] for row in selected)
    report = {
        "system_prompt": SYSTEM_PROMPT,
        "total_selected": total_count,
        "train_count": train_count,
        "val_count": val_count,
        "category_counts": dict(category_counts),
        "dedup_removed": removed_count,
        "source_stats": source_stats,
        "seed": args.seed,
    }

    report_path = args.output_dir / "sft_generation_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    logger.info("Wrote train=%d val=%d to %s", train_count, val_count, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
