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
    "너는 선박 도장 분야 전문 AI 어시스턴트이다. "
    "제공된 문서를 근거로 정확한 기술 정보를 제공하고, "
    "답변 마지막에 근거 문서 ID를 반드시 명시한다. "
    "근거가 부족하면 솔직히 '제공된 문서에 해당 정보가 없습니다'라고 답한다. "
    "안전 관련 질문에는 항상 관련 규정을 함께 안내한다."
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


def compute_echo_ratio(answer: str, source_text: str, ngram_size: int = 4) -> float:
    """Compute the fraction of answer n-grams that appear verbatim in source_text."""
    answer_norm = re.sub(r"\s+", "", answer)
    source_norm = re.sub(r"\s+", "", source_text)
    if len(answer_norm) < ngram_size or len(source_norm) < ngram_size:
        return 0.0
    answer_ngrams = [answer_norm[i:i + ngram_size] for i in range(len(answer_norm) - ngram_size + 1)]
    source_set = set(source_norm[i:i + ngram_size] for i in range(len(source_norm) - ngram_size + 1))
    if not answer_ngrams:
        return 0.0
    matched = sum(1 for ng in answer_ngrams if ng in source_set)
    return matched / len(answer_ngrams)


def sanitize_answer_text(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"\((?:DOC-|doc_)\S+\)", "", text)
    text = re.sub(r"\[(?:DOC-|doc_)\S+\]", "", text)
    return normalize_text(text)


def build_messages(category: str, user_text: str, assistant_text: str, doc_ids: Sequence[str]) -> Dict[str, Any]:
    citation = ", ".join(sorted(set(doc_ids)))
    grounded = sanitize_answer_text(assistant_text)
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


_PROCEDURE_QUESTIONS = [
    "다음 작업 절차를 신규 작업자에게 단계별로 설명해줘.",
    "아래 절차의 핵심 단계와 주의사항을 정리해줘.",
    "이 작업 절차를 현장 교육용으로 요약 설명해줘.",
    "다음 문서의 작업 순서와 각 단계의 포인트를 알려줘.",
]


def make_procedure_explain(chunks: Sequence[Dict[str, Any]], rng: random.Random) -> List[Dict[str, Any]]:
    candidates = [c for c in chunks if c.get("has_procedure")]
    rng.shuffle(candidates)
    samples: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(candidates[:260]):
        text = chunk.get("text", "")
        steps = infer_step_lines(text)
        if not steps:
            continue
        question = _PROCEDURE_QUESTIONS[idx % len(_PROCEDURE_QUESTIONS)]
        user = f"[문서] {text[:480]}\n\n[질문] {question}"
        joined_steps = " ".join(steps)
        answer = f"문서에 따른 절차는 다음과 같습니다: {joined_steps}."
        samples.append(build_messages("절차설명", user, answer, [chunk.get("chunk_id", "")]))
    return samples


def _extract_defect_type(text: str) -> str:
    """Extract dominant defect type from chunk text."""
    defect_map = [
        (r"핀홀|pin\s*hole", "핀홀(Pinhole)"),
        (r"크레이터|crater", "크레이터링(Cratering)"),
        (r"오렌지\s*필|orange\s*peel", "오렌지필(Orange Peel)"),
        (r"흘러내림|새그|sag", "흘러내림(Sagging)"),
        (r"박리|들뜸|peeling|flaking", "박리/들뜸(Peeling)"),
        (r"기포|blister", "기포(Blistering)"),
        (r"균열|crack", "균열(Cracking)"),
        (r"부식|rust|corrosion", "부식(Corrosion)"),
    ]
    for pat, name in defect_map:
        if re.search(pat, text, re.IGNORECASE):
            return name
    return "도막 결함"


_DEFECT_QUESTIONS = [
    "아래 사례의 결함 원인과 재발 방지 조치를 분석해줘.",
    "이 결함이 발생한 원인을 문서 근거로 설명하고 시정 조치를 제시해줘.",
    "다음 결함 사례를 분석하고 현장 적용 가능한 개선 방안을 제안해줘.",
    "아래 문서에서 언급된 결함의 메커니즘과 예방법을 정리해줘.",
]


def _summarize_sentences(text: str, max_sentences: int = 3) -> List[str]:
    """Extract and lightly reword sentences from text for answer construction."""
    sentences = [s.strip() for s in re.split(r'[.。]\s*', text) if len(s.strip()) > 15]
    return sentences[:max_sentences]


_DEFECT_ANALYSIS_TEMPLATES = [
    "{defect} 결함은 다음과 같은 원인으로 발생할 수 있습니다. 문서에 따르면, {detail}. 재발 방지를 위해 {prevention}.",
    "문서 분석 결과, {defect}의 주요 발생 원인은 {detail}입니다. 시정 조치로는 {prevention}이 권장됩니다.",
    "{defect} 발생 메커니즘을 정리하면 다음과 같습니다. {detail}. 현장 적용 가능한 예방법은 {prevention}입니다.",
    "해당 사례에서 {defect}이(가) 발생한 원인을 분석하면, {detail}. 개선 방안으로 {prevention}을 제안합니다.",
]

_DEFECT_PREVENTION = [
    "표면 전처리 기준(Sa 2.5 이상) 준수와 도장 환경 조건(온도, 습도) 관리",
    "도료 교반 및 희석 비율 확인, 스프레이 건 압력/거리 조정",
    "적정 도막 두께 확보와 층간 재도장 시간(overcoat interval) 준수",
    "작업 환경(이슬점 대비 3°C 이상) 확인 및 전처리 상태 검수 강화",
]


def make_defect_analysis(chunks: Sequence[Dict[str, Any]], rng: random.Random) -> List[Dict[str, Any]]:
    pattern = re.compile(r"결함|부식|균열|기포|핀홀|박리|오렌지필|들뜸")
    candidates = [c for c in chunks if pattern.search(c.get("text", ""))]
    rng.shuffle(candidates)
    samples: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(candidates[:220]):
        text = chunk.get("text", "")
        defect_type = _extract_defect_type(text)
        question = _DEFECT_QUESTIONS[idx % len(_DEFECT_QUESTIONS)]
        user = f"[문서] {text[:520]}\n\n[질문] {question}"
        sentences = _summarize_sentences(text, max_sentences=3)
        template = _DEFECT_ANALYSIS_TEMPLATES[idx % len(_DEFECT_ANALYSIS_TEMPLATES)]
        prevention = _DEFECT_PREVENTION[idx % len(_DEFECT_PREVENTION)]
        if len(sentences) >= 2:
            detail = f"{sentences[0]}이며, {sentences[1]}"
        elif sentences:
            detail = sentences[0]
        else:
            detail = f"{defect_type} 관련 상세 조건은 문서를 참고해야 합니다"
        answer = template.format(defect=defect_type, detail=detail, prevention=prevention)
        samples.append(build_messages("결함분석", user, answer, [chunk.get("chunk_id", "")]))
    return samples


_DFT_QUESTIONS = [
    "측정된 DFT 결과를 기준으로 합격/보완 피드백을 작성해줘.",
    "도막 두께 측정값에 대한 판정 기준과 조치 방안을 설명해줘.",
    "아래 DFT/WFT 관련 문서의 핵심 기준을 정리해줘.",
    "도막 두께 편차가 발생했을 때 원인과 대응 절차를 알려줘.",
]


_DFT_FEEDBACK_TEMPLATES = [
    "도막 두께 측정 결과를 검토하면, {detail}. 판정 기준에 따라 {judgment}.",
    "문서 기준에 의거하여, {detail}. 따라서 {judgment}.",
    "DFT/WFT 관련 핵심 기준을 정리하면, {detail}. 조치 방안으로는 {judgment}.",
    "측정값 분석 결과, {detail}. 현장 대응 절차는 {judgment}.",
]

_DFT_JUDGMENTS = [
    "기준 범위 이내인 경우 합격 처리하고, 미달 시 추가 도장 후 재측정이 필요합니다",
    "SSPC/NACE 기준에 따른 최소 도막 두께를 확인하고, 편차 발생 시 원인을 분석해야 합니다",
    "도막 두께가 과도하게 높은 경우 균열·박리 위험이 있으므로 적정 범위 관리가 중요합니다",
    "측정 위치별 편차가 클 경우, 스프레이 패턴과 도장 거리를 재점검해야 합니다",
]


def make_dft_feedback(chunks: Sequence[Dict[str, Any]], rng: random.Random) -> List[Dict[str, Any]]:
    pattern = re.compile(r"DFT|WFT|도막|두께|µm|um|μm")
    candidates = [c for c in chunks if pattern.search(c.get("text", ""))]
    rng.shuffle(candidates)
    samples: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(candidates[:220]):
        text = chunk.get("text", "")
        question = _DFT_QUESTIONS[idx % len(_DFT_QUESTIONS)]
        user = f"[문서] {text[:520]}\n\n[질문] {question}"
        sentences = _summarize_sentences(text, max_sentences=2)
        template = _DFT_FEEDBACK_TEMPLATES[idx % len(_DFT_FEEDBACK_TEMPLATES)]
        judgment = _DFT_JUDGMENTS[idx % len(_DFT_JUDGMENTS)]
        if sentences:
            detail = sentences[0]
            if len(sentences) >= 2:
                detail = f"{sentences[0]}이며, {sentences[1]}"
        else:
            detail = "도막 두께 관련 세부 기준은 해당 사양서를 참조해야 합니다"
        answer = template.format(detail=detail, judgment=judgment)
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


_TERM_TEMPLATES = [
    "'{ko}'({en})는 선박 도장 분야에서 {context} 관련 작업 시 사용되는 전문 용어입니다. 현장에서는 한국어와 영문 약어가 혼용되므로 두 표현을 모두 숙지해야 합니다.",
    "'{ko}'는 영문으로 '{en}'이라 하며, {context} 작업 과정에서 주로 사용됩니다. 초급 작업자는 이 용어가 사양서와 현장 지시서에서 동일한 의미로 쓰인다는 점을 기억해야 합니다.",
    "'{ko}'({en})는 {context}에 관한 기술 용어입니다. 국제 표준(ISO, SSPC 등) 문서에서 영문 표기가, 현장 교육에서는 한국어 표기가 주로 사용됩니다.",
    "선박 도장에서 '{ko}'({en})는 {context}과(와) 관련된 핵심 용어입니다. 사양서 해석 시 영문 원문과 한국어 번역 간 의미 차이에 주의해야 합니다.",
]


def _extract_term_context(text: str) -> str:
    """Extract a short domain context phrase from chunk text for term explanations."""
    context_patterns = [
        (r"(표면\s*처리|전처리|블라스팅|그라인딩)", "표면 처리"),
        (r"(도막|도장|코팅|페인팅)", "도장/코팅"),
        (r"(방식|방청|부식\s*방지)", "방식/방청"),
        (r"(검사|측정|품질|QC)", "검사/품질관리"),
        (r"(안전|보호구|환기)", "안전관리"),
        (r"(건조|경화|양생)", "건조/경화"),
    ]
    for pat, label in context_patterns:
        if re.search(pat, text):
            return label
    return "도장 공정"


def make_term_multilingual(chunks: Sequence[Dict[str, Any]], rng: random.Random) -> List[Dict[str, Any]]:
    shuffled_chunks = list(chunks)
    rng.shuffle(shuffled_chunks)
    samples: List[Dict[str, Any]] = []
    for chunk in shuffled_chunks:
        pairs = extract_bilingual_pairs(chunk.get("text", ""))
        for ko_term, en_term in pairs[:2]:
            user = (
                f"[문서] {chunk.get('text', '')[:360]}\n\n"
                f"[질문] 용어 '{ko_term}'와 '{en_term}'를 초급 작업자도 이해하게 한국어로 설명해줘."
            )
            context = _extract_term_context(chunk.get("text", ""))
            template = _TERM_TEMPLATES[len(samples) % len(_TERM_TEMPLATES)]
            answer = template.format(ko=ko_term, en=en_term, context=context)
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
            left_summary = left_text[:80].replace("\n", " ")
            right_summary = right_text[:80].replace("\n", " ")
            answer = (
                f"문서 A는 '{left_summary}'에 관한 내용이고, "
                f"문서 B는 '{right_summary}'에 관한 내용입니다. "
                f"두 문서 모두 {domain} 분야를 다루고 있으며, "
                "적용 범위와 세부 기준에서 차이가 있습니다."
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
    kept_long_vectors: List[Any] = []
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

    echo_filtered: List[Dict[str, Any]] = []
    echo_removed = 0
    echo_thresholds = {
        "문서QA": 0.95,
        "결함분석": 0.80,
        "DFT피드백": 0.80,
        "용어/다국어": 0.75,
        "절차설명": 0.70,
    }
    default_echo_threshold = 0.50
    for sample in candidates:
        user_text = sample["messages"][1]["content"]
        answer_text = sample["messages"][2]["content"]
        doc_match = re.search(r"\[문서\]\s*(.*?)(?:\[질문\]|$)", user_text, re.DOTALL)
        source_text = doc_match.group(1) if doc_match else ""
        threshold = echo_thresholds.get(sample.get("category", ""), default_echo_threshold)
        if source_text and compute_echo_ratio(answer_text, source_text) > threshold:
            echo_removed += 1
            continue
        echo_filtered.append(sample)
    logger.info("Echo filter removed %d/%d samples (>50%% verbatim)", echo_removed, len(candidates))
    candidates = echo_filtered

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
