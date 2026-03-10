#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pymupdf
from bs4 import BeautifulSoup


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocTask:
    doc_id: str
    path: Path
    source_format: str


def normalize_name(name: str) -> str:
    return unicodedata.normalize("NFC", name)


def load_manifest(manifest_path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            filename = normalize_name((row.get("filename") or "").strip())
            doc_id = (row.get("doc_id") or "").strip()
            if not filename or not doc_id:
                continue
            if filename in mapping and mapping[filename] != doc_id:
                logger.warning(
                    "Duplicate manifest filename '%s' (%s, %s). Keeping first.",
                    filename,
                    mapping[filename],
                    doc_id,
                )
                continue
            mapping[filename] = doc_id
    return mapping


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def markdown_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").strip()


def rows_to_markdown(rows: List[List[str]]) -> str:
    if not rows:
        return ""

    width = max(len(row) for row in rows)
    normalized_rows = [row + [""] * (width - len(row)) for row in rows]

    header = normalized_rows[0]
    separator = ["---"] * width
    body = normalized_rows[1:] if len(normalized_rows) > 1 else []

    lines = [
        "| " + " | ".join(markdown_escape(cell) for cell in header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(markdown_escape(cell) for cell in row) + " |")
    return "\n".join(lines)


def html_table_to_markdown(table) -> str:
    rows: List[List[str]] = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if not cells:
            continue
        rows.append([cell.get_text(" ", strip=True) for cell in cells])
    return rows_to_markdown(rows)


def extract_pdf_tables(page, page_number: int) -> List[str]:
    extracted: List[str] = []
    if not hasattr(page, "find_tables"):
        return extracted

    try:
        finder = page.find_tables()
    except Exception as exc:
        logger.warning("Table detection failed on page %s: %s", page_number, exc)
        return extracted

    tables = getattr(finder, "tables", []) if finder is not None else []
    for idx, table in enumerate(tables, start=1):
        table_markdown = ""
        try:
            if hasattr(table, "to_markdown"):
                table_markdown = table.to_markdown() or ""
            elif hasattr(table, "extract"):
                raw_rows = table.extract() or []
                norm_rows: List[List[str]] = []
                for row in raw_rows:
                    if row is None:
                        continue
                    norm_rows.append(["" if cell is None else str(cell) for cell in row])
                table_markdown = rows_to_markdown(norm_rows)
        except Exception as exc:
            logger.warning("Table extraction failed on page %s table %s: %s", page_number, idx, exc)

        if table_markdown.strip():
            extracted.append(f"<!-- page {page_number}, table {idx} -->\n{table_markdown.strip()}")

    return extracted


def parse_pdf(path: Path) -> tuple[int, str, List[str]]:
    pages_text: List[str] = []
    tables: List[str] = []
    with pymupdf.open(path) as doc:
        page_count = len(doc)
        for page_index in range(page_count):
            page = doc.load_page(page_index)
            display_page = page_index + 1
            raw_text = page.get_text("text")
            page_text = clean_text(raw_text if isinstance(raw_text, str) else str(raw_text or ""))
            if page_text:
                pages_text.append(page_text)
            tables.extend(extract_pdf_tables(page, display_page))

    content = clean_text("\n\n".join(pages_text))
    return page_count, content, tables


def parse_html(path: Path) -> tuple[int, str, List[str]]:
    html = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    tables: List[str] = []
    for table in soup.find_all("table"):
        markdown = html_table_to_markdown(table)
        if markdown:
            tables.append(markdown)
        table.decompose()

    main = soup.find("main") or soup.find("article") or soup.body or soup
    content = clean_text(main.get_text("\n", strip=True))
    return 1, content, tables


def parse_txt(path: Path) -> tuple[int, str, List[str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return 1, clean_text(text), []


def discover_files(input_dir: Path) -> List[Path]:
    pdf_dir = input_dir / "pdf"
    html_dir = input_dir / "html"

    files: List[Path] = []
    if pdf_dir.is_dir():
        files.extend(sorted(pdf_dir.glob("*.pdf"), key=lambda p: normalize_name(p.name)))

    if html_dir.is_dir():
        for item in sorted(html_dir.iterdir(), key=lambda p: normalize_name(p.name)):
            if not item.is_file():
                continue
            if item.suffix.lower() in {".html", ".txt"}:
                files.append(item)

    return files


def select_doc_id(path: Path, mapping: Dict[str, str]) -> Optional[str]:
    file_name = normalize_name(path.name)
    if file_name in mapping:
        return mapping[file_name]

    if path.suffix.lower() == ".txt":
        html_name = normalize_name(f"{path.stem}.html")
        if html_name in mapping:
            return mapping[html_name]

    return None


def build_tasks(
    files: List[Path],
    mapping: Dict[str, str],
    filenames_filter: Optional[set[str]] = None,
) -> List[DocTask]:
    by_doc_id: Dict[str, DocTask] = {}

    for path in files:
        if filenames_filter is not None and normalize_name(path.name) not in filenames_filter:
            continue

        doc_id = select_doc_id(path, mapping)
        if not doc_id:
            logger.error("Manifest doc_id not found for '%s'", path.name)
            continue

        source_format = path.suffix.lower().lstrip(".")
        candidate = DocTask(doc_id=doc_id, path=path, source_format=source_format)

        existing = by_doc_id.get(doc_id)
        if existing is None:
            by_doc_id[doc_id] = candidate
            continue

        if existing.source_format == "html" and source_format == "txt":
            logger.info(
                "Preferring TXT over HTML for doc_id %s (%s over %s)",
                doc_id,
                path.name,
                existing.path.name,
            )
            by_doc_id[doc_id] = candidate
        else:
            logger.warning(
                "Duplicate source for doc_id %s: keeping %s, skipping %s",
                doc_id,
                existing.path.name,
                path.name,
            )

    return [by_doc_id[key] for key in sorted(by_doc_id.keys())]


def process_task(task: DocTask, output_dir: Path, dry_run: bool) -> bool:
    logger.info("Processing %s (%s)", task.path.name, task.source_format)
    if dry_run:
        logger.info("Dry run: would write %s", output_dir / f"{task.doc_id}.json")
        return True

    if task.source_format == "pdf":
        pages, content, tables = parse_pdf(task.path)
    elif task.source_format == "html":
        pages, content, tables = parse_html(task.path)
    elif task.source_format == "txt":
        pages, content, tables = parse_txt(task.path)
    else:
        logger.error("Unsupported format for file %s", task.path.name)
        return False

    payload = {
        "doc_id": task.doc_id,
        "filename": task.path.name,
        "format": task.source_format,
        "pages": pages,
        "content": content,
        "tables": tables,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
    }

    output_path = output_dir / f"{task.doc_id}.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest raw PDF/HTML/TXT documents into parsed JSON files."
    )
    parser.add_argument(
        "--input-dir",
        default="/data/raw",
        type=Path,
        help="Root input directory containing pdf/ and html/ folders.",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/parsed",
        type=Path,
        help="Directory where parsed JSON files are written.",
    )
    parser.add_argument(
        "--manifest",
        default="/data/manifest.csv",
        type=Path,
        help="Manifest CSV path containing filename -> doc_id mappings.",
    )
    parser.add_argument(
        "--filenames",
        nargs="*",
        help="Optional specific filenames to process (exact basename match).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which files would be processed without writing output files.",
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

    if not args.manifest.is_file():
        logger.error("Manifest not found: %s", args.manifest)
        return 1
    if not args.input_dir.is_dir():
        logger.error("Input directory not found: %s", args.input_dir)
        return 1

    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_map = load_manifest(args.manifest)
    files = discover_files(args.input_dir)

    filenames_filter = None
    if args.filenames:
        filenames_filter = {normalize_name(name) for name in args.filenames}

    tasks = build_tasks(files, manifest_map, filenames_filter=filenames_filter)
    logger.info("Discovered %s candidate files", len(files))
    logger.info("Prepared %s processing tasks", len(tasks))

    success_count = 0
    failure_count = 0

    for task in tasks:
        try:
            ok = process_task(task, args.output_dir, args.dry_run)
            if ok:
                success_count += 1
            else:
                failure_count += 1
        except Exception as exc:
            failure_count += 1
            logger.exception("Failed to process %s: %s", task.path.name, exc)

    logger.info("Finished. success=%s failure=%s", success_count, failure_count)
    return 0 if failure_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
