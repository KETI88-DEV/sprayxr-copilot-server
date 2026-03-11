#!/usr/bin/env python3
"""Automated evaluation metrics for Korean RAG QA.

Provides ROUGE-L (Korean-aware), BERTScore, and hallucination detection.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Repetition loop detection
# ---------------------------------------------------------------------------

_SENT_SPLIT_RE = re.compile(r"[.。!?\n]+")


def compute_repetition_score(text: str, min_ngram: int = 4, min_repeat: int = 3) -> Dict[str, Any]:
    """Detect degenerate repetition loops in generated text.

    Returns a dict with:
        score : float  – 0.0 (no repetition) to 1.0 (entirely repeated)
        is_degenerate : bool – True when score >= 0.5 (major repetition)
        repeated_phrase : str | None – the most repeated n-gram (if any)
        repeat_count : int – how many times the top phrase repeats
    """
    if not text or len(text) < 40:
        return {"score": 0.0, "is_degenerate": False, "repeated_phrase": None, "repeat_count": 0}

    # Strategy 1: sentence-level exact repetition
    sentences = [s.strip() for s in _SENT_SPLIT_RE.split(text) if len(s.strip()) > 10]
    if sentences:
        sent_counts = Counter(sentences)
        top_sent, top_count = sent_counts.most_common(1)[0]
        if top_count >= min_repeat:
            ratio = top_count / len(sentences)
            return {
                "score": round(min(ratio, 1.0), 4),
                "is_degenerate": ratio >= 0.5,
                "repeated_phrase": top_sent[:80],
                "repeat_count": top_count,
            }

    # Strategy 2: character n-gram repetition (catches partial loops)
    text_norm = re.sub(r"\s+", " ", text)
    best_ratio = 0.0
    best_phrase = None
    best_count = 0

    for n in (min_ngram, min_ngram + 2, min_ngram + 4):
        if len(text_norm) < n * 2:
            continue
        ngrams = [text_norm[i:i + n] for i in range(len(text_norm) - n + 1)]
        ngram_counts = Counter(ngrams)
        top_ngram, count = ngram_counts.most_common(1)[0]
        # expected count for uniform text ≈ len/vocab → high ratio = repetition
        expected = max(len(text_norm) / max(len(set(text_norm)), 1), 1.0)
        ratio = count / max(len(ngrams), 1)
        if ratio > best_ratio:
            best_ratio = ratio
            best_phrase = top_ngram
            best_count = count

    # Normalize: for random Korean text, 4-gram repetition ratio is ~0.01-0.05
    # Degenerate loops push this above 0.3
    score = min(best_ratio * 5.0, 1.0)  # scale so 0.2 raw → 1.0 score
    return {
        "score": round(score, 4),
        "is_degenerate": score >= 0.5,
        "repeated_phrase": best_phrase[:80] if best_phrase else None,
        "repeat_count": best_count,
    }


# ---------------------------------------------------------------------------
# Language contamination detection
# ---------------------------------------------------------------------------

# CJK Unified Ideographs (Chinese characters)
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
# Japanese-specific (Hiragana + Katakana)
_JPN_RE = re.compile(r"[\u3040-\u309f\u30a0-\u30ff]")
# Emoji (broad range)
_EMOJI_RE = re.compile(
    r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    r"\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001F900-\U0001F9FF"
    r"\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF]"
)


def detect_language_contamination(text: str) -> Dict[str, Any]:
    """Detect non-Korean language contamination in text.

    Returns a dict with:
        is_contaminated : bool – True if meaningful contamination detected
        contamination_type : str | None – 'chinese', 'japanese', 'emoji', or None
        contamination_ratio : float – ratio of contaminated chars to total
        sample_chars : str | None – up to 10 contaminated characters
    """
    if not text:
        return {"is_contaminated": False, "contamination_type": None,
                "contamination_ratio": 0.0, "sample_chars": None}

    total_chars = len(text)

    cjk_chars = _CJK_RE.findall(text)
    cjk_count = len(cjk_chars)

    jpn_chars = _JPN_RE.findall(text)
    jpn_count = len(jpn_chars)

    emoji_chars = _EMOJI_RE.findall(text)
    emoji_count = len(emoji_chars)

    if cjk_count >= 4:
        ratio = cjk_count / max(total_chars, 1)
        return {
            "is_contaminated": True,
            "contamination_type": "chinese",
            "contamination_ratio": round(ratio, 4),
            "sample_chars": "".join(cjk_chars[:10]),
        }
    if jpn_count >= 3:
        ratio = jpn_count / max(total_chars, 1)
        return {
            "is_contaminated": True,
            "contamination_type": "japanese",
            "contamination_ratio": round(ratio, 4),
            "sample_chars": "".join(jpn_chars[:10]),
        }
    if emoji_count >= 3:
        ratio = emoji_count / max(total_chars, 1)
        return {
            "is_contaminated": True,
            "contamination_type": "emoji",
            "contamination_ratio": round(ratio, 4),
            "sample_chars": "".join(emoji_chars[:10]),
        }

    return {"is_contaminated": False, "contamination_type": None,
            "contamination_ratio": 0.0, "sample_chars": None}


# ---------------------------------------------------------------------------
# Korean ROUGE-L
# ---------------------------------------------------------------------------


def compute_rouge_l(predictions: List[str], references: List[str]) -> List[float]:
    """Compute ROUGE-L F1 for Korean text using korouge_score."""
    from korouge_score import rouge_scorer as ko_scorer

    scorer = ko_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores: List[float] = []
    for pred, ref in zip(predictions, references):
        if not pred.strip() or not ref.strip():
            scores.append(0.0)
            continue
        result = scorer.score(ref, pred)
        scores.append(result["rougeL"].fmeasure)
    return scores


# ---------------------------------------------------------------------------
# BERTScore
# ---------------------------------------------------------------------------


def compute_bertscore(
    predictions: List[str],
    references: List[str],
    config: Dict[str, Any],
) -> List[float]:
    """Compute BERTScore F1 for Korean text."""
    from bert_score import score as bert_score_fn

    eval_cfg = config.get("eval", {})
    model_type = eval_cfg.get("bertscore_model", "bert-base-multilingual-cased")
    num_layers = int(eval_cfg.get("bertscore_layers", 9))
    batch_size = int(eval_cfg.get("bertscore_batch_size", 16))

    if not predictions or not references:
        return []

    _P, _R, F1 = bert_score_fn(
        predictions,
        references,
        model_type=model_type,
        num_layers=num_layers,
        batch_size=batch_size,
        lang="ko",
        verbose=False,
    )
    return [round(float(f), 4) for f in F1]


# ---------------------------------------------------------------------------
# Hallucination detection
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?(?:\s*(?:mm|µm|um|μm|℃|°C|%|m/s|kg|MPa|bar|psi))?")
_SPEC_RE = re.compile(
    r"(?:ISO|KS|IMO|SSPC|NACE|ASTM|Sa)\s*[\d.\-]+(?:\s*[A-Za-z]*)?",
    re.IGNORECASE,
)


def detect_hallucination(
    answer: str,
    context_chunks: List[str],
    citations: List[str],
    valid_chunk_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Detect potential hallucination in an answer.

    Parameters
    ----------
    answer : str
    context_chunks : list of str – text of retrieved chunks
    citations : list of str – chunk_ids cited by the model
    valid_chunk_ids : list of str | None – known valid chunk_ids from retrieval.
        When provided, used for fabricated_source check instead of text extraction.

    Returns
    -------
    dict with keys:
        flag : bool          – True if hallucination suspected
        type : str | None    – hallucination category
        reason : str | None  – human-readable explanation
    """
    if not answer.strip():
        return {"flag": False, "type": None, "reason": None}

    combined_context = "\n".join(context_chunks)

    if citations:
        if valid_chunk_ids is not None:
            provided_ids = set(valid_chunk_ids)
        else:
            provided_ids = set()
            for chunk in context_chunks:
                ids_in_chunk = re.findall(r"DOC-\S+-C\d+", chunk)
                provided_ids.update(ids_in_chunk)
            for chunk in context_chunks:
                bracket_ids = re.findall(r"\[([^\]]+)\]", chunk)
                provided_ids.update(bracket_ids)

        answer_cited = re.findall(r"DOC-\S+-C\d+", answer)
        if answer_cited:
            fabricated = [c for c in answer_cited if c not in provided_ids]
            if fabricated:
                return {
                    "flag": True,
                    "type": "fabricated_source",
                    "reason": "답변이 제공되지 않은 문서를 인용함: %s" % ", ".join(fabricated[:3]),
                }

    answer_clean = re.sub(r"DOC-\S+-C\d+", "", answer)
    answer_numbers = set(_NUMBER_RE.findall(answer_clean))
    context_numbers = set(_NUMBER_RE.findall(combined_context))
    unsupported_numbers = answer_numbers - context_numbers
    unsupported_numbers = {n for n in unsupported_numbers if len(n) > 1}

    answer_specs = set(_SPEC_RE.findall(answer_clean))
    context_specs = set(s.lower() for s in _SPEC_RE.findall(combined_context))
    unsupported_specs = {s for s in answer_specs if s.lower() not in context_specs}

    if len(unsupported_numbers) >= 3 or len(unsupported_specs) >= 2:
        items = list(unsupported_specs) + list(unsupported_numbers)
        return {
            "flag": True,
            "type": "unsupported_claim",
            "reason": "문서에 없는 수치/규격이 답변에 포함됨: %s" % ", ".join(items[:5]),
        }

    # 3. Contradiction — check if answer directly contradicts context
    # Simple heuristic: if answer says "아닙니다" or negates while context affirms
    # This is a lightweight check; full NLI would be better
    negation_words = ["아닙니다", "불가능합니다", "없습니다", "하지 않습니다", "안 됩니다"]
    has_negation = any(w in answer for w in negation_words)
    affirmation_words = ["입니다", "있습니다", "합니다", "됩니다", "해야"]
    context_has_affirmation = any(w in combined_context for w in affirmation_words)

    if has_negation and context_has_affirmation and len(answer) < 100:
        # Short negation answers when context affirms may be contradictions
        return {
            "flag": True,
            "type": "contradiction",
            "reason": "답변이 문서 내용과 상반되는 부정 표현을 포함",
        }

    return {"flag": False, "type": None, "reason": None}


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


def compute_all_metrics(
    eval_results: List[Dict[str, Any]],
    eval_set: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute all metrics given raw eval results and eval set.

    Parameters
    ----------
    eval_results : list of dicts with at least 'q_id', 'answer', 'citations',
                   and optionally 'source_docs' (list of chunk dicts with 'text')
    eval_set : list of dicts with 'q_id', 'expected_answer'
    config : rag_config dict

    Returns
    -------
    dict with per-question scores and aggregates
    """
    # Build lookup
    ref_map: Dict[str, str] = {}
    for item in eval_set:
        ref_map[item["q_id"]] = item.get("expected_answer", "")

    predictions: List[str] = []
    references: List[str] = []
    q_ids: List[str] = []

    for result in eval_results:
        qid = result.get("q_id", "")
        pred = result.get("answer", "")
        ref = ref_map.get(qid, "")
        predictions.append(pred)
        references.append(ref)
        q_ids.append(qid)

    # Compute metrics
    logger.info("Computing ROUGE-L for %d items", len(predictions))
    rouge_scores = compute_rouge_l(predictions, references)

    logger.info("Computing BERTScore for %d items", len(predictions))
    bert_scores = compute_bertscore(predictions, references, config)

    logger.info("Running hallucination detection for %d items", len(eval_results))
    hallucination_results: List[Dict[str, Any]] = []
    for result in eval_results:
        context_chunks = []
        for doc in result.get("source_docs", []):
            text = doc.get("text", "")
            if text:
                context_chunks.append(text)
        citations = result.get("citations", [])
        hal = detect_hallucination(result.get("answer", ""), context_chunks, citations)
        hallucination_results.append(hal)

    logger.info("Running repetition + language contamination detection")
    repetition_results: List[Dict[str, Any]] = []
    contamination_results: List[Dict[str, Any]] = []
    for result in eval_results:
        answer = result.get("answer", "")
        repetition_results.append(compute_repetition_score(answer))
        contamination_results.append(detect_language_contamination(answer))

    per_question: List[Dict[str, Any]] = []
    for i, qid in enumerate(q_ids):
        per_question.append({
            "q_id": qid,
            "rouge_l": rouge_scores[i] if i < len(rouge_scores) else 0.0,
            "bertscore_f1": bert_scores[i] if i < len(bert_scores) else 0.0,
            "hallucination_flag": hallucination_results[i]["flag"],
            "hallucination_type": hallucination_results[i]["type"],
            "hallucination_reason": hallucination_results[i]["reason"],
            "repetition_score": repetition_results[i]["score"],
            "repetition_degenerate": repetition_results[i]["is_degenerate"],
            "lang_contaminated": contamination_results[i]["is_contaminated"],
            "lang_contamination_type": contamination_results[i]["contamination_type"],
        })

    n = len(per_question)
    avg_rouge = sum(q["rouge_l"] for q in per_question) / max(n, 1)
    avg_bert = sum(q["bertscore_f1"] for q in per_question) / max(n, 1)
    hal_count = sum(1 for q in per_question if q["hallucination_flag"])
    hal_rate = hal_count / max(n, 1)
    rep_degenerate_count = sum(1 for q in per_question if q["repetition_degenerate"])
    rep_degenerate_rate = rep_degenerate_count / max(n, 1)
    lang_contaminated_count = sum(1 for q in per_question if q["lang_contaminated"])
    lang_contaminated_rate = lang_contaminated_count / max(n, 1)

    return {
        "per_question": per_question,
        "aggregates": {
            "count": n,
            "rouge_l_mean": round(avg_rouge, 4),
            "bertscore_f1_mean": round(avg_bert, 4),
            "hallucination_count": hal_count,
            "hallucination_rate": round(hal_rate, 4),
            "repetition_degenerate_count": rep_degenerate_count,
            "repetition_degenerate_rate": round(rep_degenerate_rate, 4),
            "lang_contaminated_count": lang_contaminated_count,
            "lang_contaminated_rate": round(lang_contaminated_rate, 4),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute evaluation metrics for RAG QA results."
    )
    parser.add_argument(
        "--eval-results",
        type=Path,
        required=True,
        help="Path to eval results JSONL (one response per line).",
    )
    parser.add_argument(
        "--eval-set",
        type=Path,
        required=True,
        help="Path to eval_set.jsonl with expected answers.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/workspace/trainer/configs/rag_config.yaml"),
        help="Path to rag_config.yaml.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write metrics JSON output.",
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

    # Load config
    if not args.config.is_file():
        logger.error("Config not found: %s", args.config)
        return 1

    with args.config.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    # Load eval results
    if not args.eval_results.is_file():
        logger.error("Eval results not found: %s", args.eval_results)
        return 1

    eval_results: List[Dict[str, Any]] = []
    with args.eval_results.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                eval_results.append(json.loads(line))
    logger.info("Loaded %d eval results", len(eval_results))

    # Load eval set
    if not args.eval_set.is_file():
        logger.error("Eval set not found: %s", args.eval_set)
        return 1

    eval_set: List[Dict[str, Any]] = []
    with args.eval_set.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                eval_set.append(json.loads(line))
    logger.info("Loaded %d eval set items", len(eval_set))

    # Compute
    metrics = compute_all_metrics(eval_results, eval_set, config)

    # Output
    output_text = json.dumps(metrics, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as fh:
            fh.write(output_text + "\n")
        logger.info("Metrics written to %s", args.output)
    else:
        print(output_text)

    # Summary
    agg = metrics["aggregates"]
    logger.info(
        "Summary: ROUGE-L=%.4f  BERTScore=%.4f  Hallucination=%d/%d (%.1f%%)"
        "  RepetitionLoop=%d/%d (%.1f%%)  LangContam=%d/%d (%.1f%%)",
        agg["rouge_l_mean"],
        agg["bertscore_f1_mean"],
        agg["hallucination_count"],
        agg["count"],
        agg["hallucination_rate"] * 100,
        agg["repetition_degenerate_count"],
        agg["count"],
        agg["repetition_degenerate_rate"] * 100,
        agg["lang_contaminated_count"],
        agg["count"],
        agg["lang_contaminated_rate"] * 100,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
