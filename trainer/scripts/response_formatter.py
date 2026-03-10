from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen2.5-7B-Instruct"


@dataclass
class SourceDocMeta:
    chunk_id: str
    doc_id: str
    domain_category: str
    rerank_score: float


@dataclass
class FormattedResponse:
    question: str
    answer: str
    citations: List[str]
    confidence: str
    model_name: str
    timestamp: str
    source_docs: List[Dict[str, Any]]
    q_id: Optional[str] = None


def extract_citations(results: List[Dict[str, Any]]) -> List[str]:
    citations: List[str] = []
    seen: set[str] = set()

    for result in results:
        chunk_id = str(result.get("chunk_id", "")).strip()
        if not chunk_id or chunk_id in seen:
            continue
        citations.append(chunk_id)
        seen.add(chunk_id)

    return citations


def compute_confidence(results: List[Dict[str, Any]], config: Dict[str, Any]) -> str:
    conf_cfg = config.get("confidence", {})
    min_results = int(conf_cfg.get("min_results", 1))
    high_threshold = float(conf_cfg.get("high_threshold", 0.7))
    medium_threshold = float(conf_cfg.get("medium_threshold", 0.4))

    if len(results) < min_results:
        return "low"

    max_score = max(float(r.get("rerank_score", 0.0)) for r in results)
    if max_score >= high_threshold:
        return "high"
    if max_score >= medium_threshold:
        return "medium"
    return "low"


def format_context(results: List[Dict[str, Any]]) -> str:
    blocks: List[str] = []
    for result in results:
        chunk_id = str(result.get("chunk_id", "UNKNOWN"))
        text = str(result.get("text", "")).strip()
        if not text:
            continue
        blocks.append(f"[{chunk_id}]\n{text}")
    return "\n\n".join(blocks)


def _build_source_docs(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    source_docs: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for result in results:
        chunk_id = str(result.get("chunk_id", "")).strip()
        if not chunk_id or chunk_id in seen:
            continue
        seen.add(chunk_id)

        source = SourceDocMeta(
            chunk_id=chunk_id,
            doc_id=str(result.get("doc_id", "")),
            domain_category=str(result.get("domain_category", "")),
            rerank_score=float(result.get("rerank_score", 0.0)),
        )
        source_docs.append(asdict(source))

    return source_docs


def format_response(
    question: str,
    answer: str,
    results: List[Dict[str, Any]],
    config: Dict[str, Any],
    q_id: Optional[str] = None,
) -> Dict[str, Any]:
    citations = extract_citations(results)
    source_docs = _build_source_docs(results)
    confidence = compute_confidence(results, config)
    timestamp = datetime.now(timezone.utc).isoformat()

    response = FormattedResponse(
        q_id=q_id,
        question=question,
        answer=answer,
        citations=citations,
        confidence=confidence,
        model_name=MODEL_NAME,
        timestamp=timestamp,
        source_docs=source_docs,
    )

    payload = asdict(response)
    if q_id is None:
        payload.pop("q_id", None)

    logger.debug("Formatted response with %d citations", len(citations))
    return payload
