from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen2.5-7B-Instruct"

RESPONSE_SCHEMAS: Dict[str, Dict[str, List[str]]] = {
    "fast": {
        "required": ["text_guide", "citations", "metadata"],
        "optional": ["steps"],
    },
    "training_realtime": {
        "required": ["text_guide", "citations", "metadata"],
        "optional": ["dft_result", "next_path", "anomaly", "steps"],
    },
    "training_review": {
        "required": ["text_guide", "citations", "metadata"],
        "optional": ["dft_result", "pose_comparison", "anomaly", "steps"],
    },
}

_TEXT_GUIDE_REQUIRED_FIELDS = ["language", "detail_level", "summary", "body"]
_METADATA_REQUIRED_FIELDS = ["latency_ms", "tools_used", "rounds", "route"]
_UNQUOTED_KEY_PATTERN = re.compile(r'([,{]\s*)([A-Za-z_][A-Za-z0-9_-]*)(\s*:)')
_TRAILING_COMMA_PATTERN = re.compile(r",\s*([}\]])")
_KEY_VALUE_PATTERN = re.compile(
    r'"(?P<dkey>[^"\\]+)"\s*:\s*(?P<dval>"(?:\\.|[^"\\])*"|-?\d+(?:\.\d+)?|true|false|null)|'
    + r"'(?P<skey>[^'\\]+)'\s*:\s*(?P<sval>'(?:\\.|[^'\\])*'|-?\d+(?:\.\d+)?|true|false|null)|"
    + r"(?P<ukey>[A-Za-z_][A-Za-z0-9_-]*)\s*:\s*(?P<uval>\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|-?\d+(?:\.\d+)?|true|false|null)"
)


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


def validate_response_structure(response: dict[str, Any], route: str) -> dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    schema = RESPONSE_SCHEMAS.get(route)
    if schema is None:
        errors.append(f"unsupported route: {route}")
        logger.warning("Response validation failed: unsupported route '%s'", route)
        return {"is_valid": False, "errors": errors, "warnings": warnings}

    for key in schema["required"]:
        if key not in response:
            errors.append(f"missing required field: {key}")

    allowed_keys = set(schema["required"] + schema["optional"])
    for key in response:
        if key not in allowed_keys:
            warnings.append(f"unexpected field for route {route}: {key}")

    text_guide = response.get("text_guide")
    if "text_guide" in response and not isinstance(text_guide, dict):
        errors.append("text_guide must be a dict")
    elif isinstance(text_guide, dict):
        for field in _TEXT_GUIDE_REQUIRED_FIELDS:
            value = text_guide.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"text_guide.{field} must be a non-empty string")

    metadata = response.get("metadata")
    if "metadata" in response and not isinstance(metadata, dict):
        errors.append("metadata must be a dict")
    elif isinstance(metadata, dict):
        for field in _METADATA_REQUIRED_FIELDS:
            if field not in metadata:
                errors.append(f"metadata missing required field: {field}")
        if "route" in metadata and metadata.get("route") != route:
            warnings.append(
                f"metadata.route mismatch: expected {route}, got {metadata.get('route')}"
            )

    citations = response.get("citations")
    if "citations" in response and not isinstance(citations, list):
        errors.append("citations must be a list")

    is_valid = len(errors) == 0
    if not is_valid:
        logger.warning("Response validation failed for route %s: %s", route, errors)
    elif warnings:
        logger.info("Response validation warnings for route %s: %s", route, warnings)

    return {"is_valid": is_valid, "errors": errors, "warnings": warnings}


def ensure_citations(
    response: dict[str, Any],
    rag_chunks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    citations = response.get("citations")
    if isinstance(citations, list) and citations:
        return response

    if not rag_chunks:
        response["citations"] = citations if isinstance(citations, list) else []
        return response

    extracted: List[str] = []
    seen: set[str] = set()
    for chunk in rag_chunks:
        if not isinstance(chunk, dict):
            continue
        metadata = chunk.get("metadata")
        source = None
        chunk_id = None

        if isinstance(metadata, dict):
            source = metadata.get("source") or metadata.get("doc_id") or metadata.get("source_id")
            chunk_id = metadata.get("chunk_id") or metadata.get("chunk_seq") or metadata.get("chunk_index")

        source = source or chunk.get("source") or chunk.get("doc_id")
        chunk_id = chunk_id or chunk.get("chunk_id") or chunk.get("chunk_seq") or chunk.get("chunk_index")

        if chunk_id is None:
            continue

        chunk_text = str(chunk_id).strip()
        if re.fullmatch(r"DOC-[^-]+-C.+", chunk_text):
            normalized = chunk_text
        else:
            if source is None:
                continue
            source_text = str(source).strip()
            chunk_suffix = chunk_text if chunk_text.startswith("C") else f"C{chunk_text}"
            normalized = f"DOC-{source_text}-{chunk_suffix}"

        if normalized not in seen:
            seen.add(normalized)
            extracted.append(normalized)

    response["citations"] = extracted
    return response


def repair_json_structure(raw_text: str) -> dict[str, Any]:
    if not raw_text.strip():
        return {"_repair_error": "empty_input"}

    def _parse_json_object(text: str) -> dict[str, Any] | None:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed, "_repair_warning": "top_level_non_dict"}

    candidate = raw_text.strip()
    direct = _parse_json_object(candidate)
    if direct is not None:
        return direct

    block = candidate
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and start < end:
        block = candidate[start : end + 1]

    repaired = _TRAILING_COMMA_PATTERN.sub(r"\1", block)
    repaired = _UNQUOTED_KEY_PATTERN.sub(r'\1"\2"\3', repaired)

    open_count = repaired.count("{")
    close_count = repaired.count("}")
    if open_count > close_count:
        repaired = repaired + ("}" * (open_count - close_count))
    elif close_count > open_count:
        overflow = close_count - open_count
        while overflow > 0 and repaired.endswith("}"):
            repaired = repaired[:-1]
            overflow -= 1

    parsed_repaired = _parse_json_object(repaired)
    if parsed_repaired is not None:
        parsed_repaired.setdefault("_repair_applied", True)
        return parsed_repaired

    result: Dict[str, Any] = {}
    for match in _KEY_VALUE_PATTERN.finditer(repaired):
        key = match.group("dkey") or match.group("skey") or match.group("ukey")
        raw_value = match.group("dval") or match.group("sval") or match.group("uval")
        if not key or raw_value is None:
            continue

        value_text = raw_value.strip()
        parsed_value: Any
        if value_text.startswith("'") and value_text.endswith("'"):
            parsed_value = value_text[1:-1].replace("\\'", "'")
        elif value_text.startswith('"') and value_text.endswith('"'):
            try:
                parsed_value = json.loads(value_text)
            except json.JSONDecodeError:
                parsed_value = value_text[1:-1]
        elif value_text in {"true", "false", "null"}:
            parsed_value = {"true": True, "false": False, "null": None}[value_text]
        else:
            try:
                parsed_value = int(value_text) if "." not in value_text else float(value_text)
            except ValueError:
                parsed_value = value_text

        result[key] = parsed_value

    if result:
        result["_repair_applied"] = True
        result["_repair_warning"] = "regex_extraction"
        return result

    logger.warning("Failed to repair malformed JSON text")
    return {"_repair_error": "json_parse_failed"}
