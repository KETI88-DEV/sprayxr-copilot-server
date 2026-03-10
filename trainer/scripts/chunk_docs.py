#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import TypedDict, cast


logger = logging.getLogger(__name__)


DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "coating": ["도장", "도료", "코팅", "paint", "coating", "도막"],
    "blasting": ["블라스팅", "블라스트", "blast", "연마", "abrasive"],
    "safety": ["안전", "safety", "보호구", "위험", "hazard"],
    "defects": ["결함", "defect", "불량", "핀홀", "기포", "크레이터"],
    "thickness": ["두께", "thickness", "dft", "wft", "도막두께", "mil"],
}

PROCESS_KEYWORDS: dict[str, list[str]] = {
    "preparation": ["전처리", "표면처리", "surface prep", "세척", "cleaning"],
    "application": ["도장작업", "도포", "application", "spray", "스프레이", "붓칠"],
    "inspection": ["검사", "inspection", "측정", "measurement", "qc"],
    "qc": ["품질", "quality", "합격", "판정"],
}

SUBTASK_KEYWORDS: list[tuple[str, list[str]]] = [
    ("paint_mixing", ["혼합", "mix", "교반", "희석", "배합"]),
    ("thickness_measurement", ["두께", "dft", "wft", "도막두께", "측정", "mil"]),
    ("surface_prep", ["전처리", "표면처리", "surface prep", "세척", "blast", "블라스팅"]),
    ("spray_application", ["spray", "스프레이", "도포", "도장작업"]),
    ("defect_inspection", ["결함", "defect", "검사", "inspection", "불량", "핀홀", "기포"]),
]

SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?。！？])\s+")
PROCEDURE_NUMBERED_RE = re.compile(r"(?m)^\s*\d+\.\s+")
PROCEDURE_BULLET_RE = re.compile(r"(?m)^\s*[•\-*]\s+")
TABLE_PAGE_RE = re.compile(r"<!--\s*page\s*(\d+)", re.IGNORECASE)
LOG_LEVELS: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}


class ChunkPayload(TypedDict):
    doc_id: str
    chunk_id: str
    text: str
    source_type: str
    domain_category: str
    process_stage: str
    subtask: str
    language: str
    revision: str
    quality_score: float
    has_table: bool
    has_procedure: bool
    page_num: int | None
    offset: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk deduplicated JSON documents into chunk-level metadata JSON files."
    )
    _ = parser.add_argument(
        "--input-dir",
        default="/data/deduplicated",
        type=Path,
        help="Directory containing deduplicated JSON files.",
    )
    _ = parser.add_argument(
        "--output-dir",
        default="/data/chunked",
        type=Path,
        help="Directory where chunked JSON files are written.",
    )
    _ = parser.add_argument(
        "--schema",
        default="/workspace/trainer/configs/metadata_schema.json",
        type=Path,
        help="Chunk metadata schema JSON path.",
    )
    _ = parser.add_argument(
        "--min-chunk-size",
        default=50,
        type=int,
        help="Minimum chunk size in characters.",
    )
    _ = parser.add_argument(
        "--max-chunk-size",
        default=1024,
        type=int,
        help="Maximum chunk size in characters.",
    )
    _ = parser.add_argument(
        "--overlap",
        default=50,
        type=int,
        help="Character overlap between consecutive chunks.",
    )
    _ = parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run chunking without writing output files.",
    )
    _ = parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def split_semantic_sections(text: str) -> list[str]:
    if not text.strip():
        return []

    sections: list[str] = []
    current: list[str] = []

    for raw_line in text.splitlines(keepends=True):
        line = raw_line.rstrip("\n")
        stripped = line.strip()

        is_all_caps = bool(stripped) and any(ch.isalpha() for ch in stripped) and stripped.upper() == stripped
        is_header = stripped.startswith("#") or stripped.endswith(":") or is_all_caps

        if is_header and current:
            section_text = "".join(current).strip()
            if section_text:
                sections.append(section_text)
            current = []

        current.append(raw_line)

    if current:
        section_text = "".join(current).strip()
        if section_text:
            sections.append(section_text)

    return sections


def split_by_paragraph(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def split_by_sentence(text: str) -> list[str]:
    parts = [part.strip() for part in SENTENCE_BOUNDARY_RE.split(text) if part.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def hard_split(text: str, max_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    step = max(1, max_size - max(0, overlap))

    while start < len(text):
        end = min(len(text), start + max_size)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start += step

    return chunks


def split_to_units(text: str, max_size: int, overlap: int) -> list[str]:
    units: list[str] = []
    for section in split_semantic_sections(text):
        if len(section) <= max_size:
            units.append(section)
            continue

        for paragraph in split_by_paragraph(section):
            if len(paragraph) <= max_size:
                units.append(paragraph)
                continue

            for sentence in split_by_sentence(paragraph):
                if len(sentence) <= max_size:
                    units.append(sentence)
                else:
                    units.extend(hard_split(sentence, max_size, overlap))

    return units


def assemble_chunks(
    units: list[str],
    target_size: int,
    min_size: int,
    max_size: int,
    overlap: int,
) -> list[str]:
    if not units:
        return []

    assembled: list[str] = []
    current = ""

    for unit in units:
        if not current:
            current = unit
            continue

        separator = "\n\n" if not current.endswith("\n") else "\n"
        candidate = current + separator + unit
        if len(candidate) <= max_size and len(current) < target_size:
            current = candidate
        else:
            assembled.append(current.strip())
            current = unit

    if current.strip():
        assembled.append(current.strip())

    merged: list[str] = []
    for chunk in assembled:
        if merged and len(chunk) < min_size:
            join_candidate = merged[-1] + "\n\n" + chunk
            if len(join_candidate) <= max_size:
                merged[-1] = join_candidate
                continue
        merged.append(chunk)

    with_overlap: list[str] = []
    for idx, chunk in enumerate(merged):
        if idx == 0 or overlap <= 0:
            with_overlap.append(chunk)
            continue

        prefix = merged[idx - 1][-overlap:] if overlap < len(merged[idx - 1]) else merged[idx - 1]
        candidate = (prefix + "\n" + chunk).strip()
        if len(candidate) > max_size:
            candidate = candidate[:max_size].rstrip()
        with_overlap.append(candidate)

    return with_overlap


def detect_domain_category(text: str) -> tuple[str, bool]:
    lowered = text.lower()
    for category in ["coating", "blasting", "safety", "defects", "thickness"]:
        if any(keyword.lower() in lowered for keyword in DOMAIN_KEYWORDS[category]):
            return category, True
    return "general", False


def detect_process_stage(text: str) -> str:
    lowered = text.lower()
    for stage in ["preparation", "application", "inspection", "qc"]:
        if any(keyword.lower() in lowered for keyword in PROCESS_KEYWORDS[stage]):
            return stage
    return "general"


def detect_subtask(text: str) -> str:
    lowered = text.lower()
    for label, keywords in SUBTASK_KEYWORDS:
        if any(keyword.lower() in lowered for keyword in keywords):
            return label
    return "general"


def detect_has_procedure(text: str) -> bool:
    return bool(PROCEDURE_NUMBERED_RE.search(text) or PROCEDURE_BULLET_RE.search(text))


def parse_table_page(table_text: str) -> int | None:
    match = TABLE_PAGE_RE.search(table_text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def quality_score_for_text(text: str, has_domain_keyword: bool) -> float:
    size = len(text)
    if size > 200 and has_domain_keyword:
        return 1.0
    if size > 100:
        return 0.7
    return 0.3


def load_schema_required_fields(schema_path: Path) -> list[str]:
    with schema_path.open("r", encoding="utf-8") as handle:
        loaded = cast(object, json.load(handle))

    if not isinstance(loaded, dict):
        raise ValueError("Schema file root must be a JSON object")

    schema_data = cast(dict[str, object], loaded)

    required = schema_data.get("required")
    if not isinstance(required, list):
        raise ValueError("Schema file missing valid 'required' field list")

    required_list = cast(list[object], required)
    required_fields: list[str] = []
    for value in required_list:
        if not isinstance(value, str):
            raise ValueError("Schema file has non-string item in 'required' field list")
        required_fields.append(value)
    return required_fields


def validate_chunk(chunk: ChunkPayload, required_fields: list[str]) -> None:
    missing = [field for field in required_fields if field not in chunk]
    if missing:
        raise ValueError(f"Chunk {chunk.get('chunk_id', '<unknown>')} missing required fields: {missing}")


def resolve_offsets(content: str, chunks: list[str]) -> list[int]:
    offsets: list[int] = []
    cursor = 0

    for chunk in chunks:
        normalized = chunk.strip()
        if not normalized:
            offsets.append(max(0, cursor))
            continue

        found = content.find(normalized, cursor)
        if found == -1:
            found = content.find(normalized)
        if found == -1:
            found = max(0, cursor)

        offsets.append(found)
        cursor = max(cursor, found + len(normalized))

    return offsets


def chunk_document_content(content: str, min_size: int, max_size: int, overlap: int) -> tuple[list[str], list[int]]:
    if not content.strip():
        return [], []

    units = split_to_units(content, max_size=max_size, overlap=overlap)
    target_size = max(512, min(768, max_size)) if max_size >= 512 else max_size
    chunks = assemble_chunks(
        units,
        target_size=target_size,
        min_size=min_size,
        max_size=max_size,
        overlap=overlap,
    )

    offsets = resolve_offsets(content, chunks)
    return chunks, offsets


def build_chunk_object(
    doc_id: str,
    chunk_index: int,
    text: str,
    source_type: str,
    offset: int,
    page_num: int | None,
    has_table: bool,
) -> ChunkPayload:
    domain_category, has_domain_keyword = detect_domain_category(text)
    process_stage = detect_process_stage(text)

    return {
        "doc_id": doc_id,
        "chunk_id": f"{doc_id}_chunk_{chunk_index:03d}",
        "text": text,
        "source_type": source_type,
        "domain_category": domain_category,
        "process_stage": process_stage,
        "subtask": detect_subtask(text),
        "language": "ko",
        "revision": "v1.0",
        "quality_score": quality_score_for_text(text, has_domain_keyword),
        "has_table": has_table,
        "has_procedure": detect_has_procedure(text),
        "page_num": page_num,
        "offset": max(0, int(offset)),
    }


def process_document(
    input_path: Path,
    output_dir: Path,
    required_fields: list[str],
    min_chunk_size: int,
    max_chunk_size: int,
    overlap: int,
    dry_run: bool,
) -> list[ChunkPayload]:
    with input_path.open("r", encoding="utf-8") as handle:
        loaded = cast(object, json.load(handle))

    if not isinstance(loaded, dict):
        raise ValueError(f"Document {input_path.name} is not a JSON object")

    payload = cast(dict[str, object], loaded)

    doc_id = str(payload.get("doc_id") or input_path.stem)
    source_type = str(payload.get("format") or "txt").lower()
    if source_type not in {"pdf", "html", "txt"}:
        logger.warning("Unknown source type '%s' for %s, defaulting to txt", source_type, input_path.name)
        source_type = "txt"

    content = payload.get("content")
    if not isinstance(content, str):
        raise ValueError(f"Document {input_path.name} has non-string content")

    tables_obj = payload.get("tables")
    tables: list[object] = cast(list[object], tables_obj) if isinstance(tables_obj, list) else []

    text_chunks, offsets = chunk_document_content(
        content=content,
        min_size=min_chunk_size,
        max_size=max_chunk_size,
        overlap=overlap,
    )

    chunk_objects: list[ChunkPayload] = []
    chunk_index = 1

    for text, offset in zip(text_chunks, offsets):
        chunk = build_chunk_object(
            doc_id=doc_id,
            chunk_index=chunk_index,
            text=text,
            source_type=source_type,
            offset=offset,
            page_num=None,
            has_table=False,
        )
        validate_chunk(chunk, required_fields)
        chunk_objects.append(chunk)
        chunk_index += 1

    for table in tables:
        if not isinstance(table, str) or not table.strip():
            continue

        table_text = table.strip()
        chunk = build_chunk_object(
            doc_id=doc_id,
            chunk_index=chunk_index,
            text=table_text,
            source_type=source_type,
            offset=0,
            page_num=parse_table_page(table_text),
            has_table=True,
        )
        validate_chunk(chunk, required_fields)
        chunk_objects.append(chunk)
        chunk_index += 1

    if dry_run:
        logger.info("Dry run: %s -> %s chunks", input_path.name, len(chunk_objects))
        return chunk_objects

    output_path = output_dir / f"{doc_id}_chunks.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(chunk_objects, handle, ensure_ascii=False, indent=2)

    logger.info("Wrote chunked document: %s (%s chunks)", output_path, len(chunk_objects))
    return chunk_objects


def build_summary(all_chunks: list[ChunkPayload], total_docs: int) -> dict[str, object]:
    total_chunks = len(all_chunks)
    avg_chunk_size = 0.0
    if total_chunks > 0:
        avg_chunk_size = sum(len(str(chunk.get("text") or "")) for chunk in all_chunks) / total_chunks

    domain_distribution: dict[str, int] = {}
    chunks_with_tables = 0
    chunks_with_procedures = 0

    for chunk in all_chunks:
        domain = str(chunk.get("domain_category") or "general")
        domain_distribution[domain] = domain_distribution.get(domain, 0) + 1
        if bool(chunk.get("has_table")):
            chunks_with_tables += 1
        if bool(chunk.get("has_procedure")):
            chunks_with_procedures += 1

    return {
        "total_docs": total_docs,
        "total_chunks": total_chunks,
        "avg_chunk_size": round(avg_chunk_size, 2),
        "domain_distribution": domain_distribution,
        "chunks_with_tables": chunks_with_tables,
        "chunks_with_procedures": chunks_with_procedures,
    }


def main() -> int:
    args = parse_args()
    log_level = cast(str, args.log_level)
    logging.basicConfig(
        level=LOG_LEVELS.get(log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    input_dir = cast(Path, args.input_dir)
    output_dir = cast(Path, args.output_dir)
    schema_path = cast(Path, args.schema)
    min_chunk_size = cast(int, args.min_chunk_size)
    max_chunk_size = cast(int, args.max_chunk_size)
    overlap = cast(int, args.overlap)
    dry_run = cast(bool, args.dry_run)
    logging.getLogger().setLevel(LOG_LEVELS.get(log_level, logging.INFO))

    if min_chunk_size <= 0:
        logger.error("--min-chunk-size must be > 0")
        return 1
    if max_chunk_size < min_chunk_size:
        logger.error("--max-chunk-size must be >= --min-chunk-size")
        return 1
    if overlap < 0:
        logger.error("--overlap must be >= 0")
        return 1

    if not input_dir.is_dir():
        logger.error("Input directory not found: %s", input_dir)
        return 1
    if not schema_path.is_file():
        logger.error("Schema not found: %s", schema_path)
        return 1

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    try:
        required_fields = load_schema_required_fields(schema_path)
    except Exception as exc:
        logger.error("Failed to load schema %s: %s", schema_path, exc)
        return 1

    input_files = sorted(input_dir.glob("*.json"))
    logger.info("Discovered %s deduplicated JSON files", len(input_files))

    all_chunks: list[ChunkPayload] = []
    success_count = 0
    failure_count = 0

    for input_path in input_files:
        try:
            chunks = process_document(
                input_path=input_path,
                output_dir=output_dir,
                required_fields=required_fields,
                min_chunk_size=min_chunk_size,
                max_chunk_size=max_chunk_size,
                overlap=overlap,
                dry_run=dry_run,
            )
            all_chunks.extend(chunks)
            success_count += 1
        except Exception as exc:
            failure_count += 1
            logger.exception("Failed to chunk %s: %s", input_path.name, exc)

    summary = build_summary(all_chunks, total_docs=success_count)
    logger.info("Chunk summary: %s", json.dumps(summary, ensure_ascii=False))

    if not dry_run:
        summary_path = output_dir / "chunk_summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        logger.info("Wrote chunk summary: %s", summary_path)

    logger.info("Finished chunking. success=%s failure=%s", success_count, failure_count)
    return 0 if failure_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
