#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


logger = logging.getLogger(__name__)


STAGE_ORDER: List[str] = [
    "unicode_nfkc_normalization",
    "control_character_removal",
    "non_breaking_space_normalization",
    "line_ending_normalization",
    "header_footer_removal",
    "page_number_removal",
    "unit_normalization",
    "term_replacement",
    "trailing_whitespace_trimming",
    "collapse_blank_lines",
]


@dataclass(frozen=True)
class RegexReplacement:
    pattern: re.Pattern[str]
    replacement: str


@dataclass(frozen=True)
class NormalizationRules:
    control_char_rule: RegexReplacement
    non_breaking_space_rule: RegexReplacement
    line_ending_rule: RegexReplacement
    trailing_whitespace_rule: RegexReplacement
    collapse_blank_lines_rule: RegexReplacement
    header_footer_patterns: List[re.Pattern[str]]
    page_number_patterns: List[re.Pattern[str]]
    unit_mappings: List[RegexReplacement]
    term_replacements: List[RegexReplacement]


def parse_regex_flags(flags: str | None) -> int:
    if not flags:
        return 0

    value = 0
    for part in flags.split("|"):
        name = part.strip().upper()
        if not name:
            continue
        flag = getattr(re, name, None)
        if not isinstance(flag, re.RegexFlag):
            raise ValueError(f"Unsupported regex flag '{name}'")
        value |= int(flag)
    return value


def compile_rule_entry(entry: dict) -> RegexReplacement:
    pattern = str(entry.get("pattern") or "")
    replacement = str(entry.get("replacement") or "")
    flags = parse_regex_flags(entry.get("flags"))
    return RegexReplacement(pattern=re.compile(pattern, flags), replacement=replacement)


def build_special_char_rules(entries: List[dict]) -> Dict[str, RegexReplacement]:
    compiled: Dict[str, RegexReplacement] = {}
    for entry in entries:
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        compiled[name] = compile_rule_entry(entry)
    return compiled


def load_rules(rules_path: Path) -> NormalizationRules:
    with rules_path.open("r", encoding="utf-8") as handle:
        raw_rules = yaml.safe_load(handle) or {}

    special_char_raw = raw_rules.get("special_char_rules") or []
    special_char_rules = build_special_char_rules(special_char_raw)

    required_special_rules = {
        "remove_control_chars",
        "normalize_non_breaking_spaces",
        "normalize_line_endings",
        "trim_trailing_space",
        "collapse_many_blank_lines",
    }
    missing = sorted(required_special_rules - set(special_char_rules.keys()))
    if missing:
        raise ValueError(f"Missing required special_char_rules entries: {', '.join(missing)}")

    header_footer_patterns = [re.compile(pattern) for pattern in (raw_rules.get("header_footer_patterns") or [])]
    page_number_patterns = [re.compile(pattern) for pattern in (raw_rules.get("page_number_patterns") or [])]
    unit_mappings = [compile_rule_entry(entry) for entry in (raw_rules.get("unit_mappings") or [])]
    term_replacements = [compile_rule_entry(entry) for entry in (raw_rules.get("term_replacements") or [])]

    return NormalizationRules(
        control_char_rule=special_char_rules["remove_control_chars"],
        non_breaking_space_rule=special_char_rules["normalize_non_breaking_spaces"],
        line_ending_rule=special_char_rules["normalize_line_endings"],
        trailing_whitespace_rule=special_char_rules["trim_trailing_space"],
        collapse_blank_lines_rule=special_char_rules["collapse_many_blank_lines"],
        header_footer_patterns=header_footer_patterns,
        page_number_patterns=page_number_patterns,
        unit_mappings=unit_mappings,
        term_replacements=term_replacements,
    )


def count_unicode_changes(before: str, after: str) -> int:
    if before == after:
        return 0

    count = 0
    shared = min(len(before), len(after))
    for index in range(shared):
        if before[index] != after[index]:
            count += 1
    count += abs(len(before) - len(after))
    return count


def apply_regex_sequence(text: str, rules: List[RegexReplacement]) -> Tuple[str, int]:
    updated = text
    replacements = 0
    for rule in rules:
        updated, count = rule.pattern.subn(rule.replacement, updated)
        replacements += count
    return updated, replacements


def apply_pattern_sequence(text: str, patterns: List[re.Pattern[str]]) -> Tuple[str, int]:
    updated = text
    replacements = 0
    for pattern in patterns:
        updated, count = pattern.subn("", updated)
        replacements += count
    return updated, replacements


def normalize_text(text: str, rules: NormalizationRules) -> Tuple[str, Dict[str, int]]:
    counts: Dict[str, int] = {stage: 0 for stage in STAGE_ORDER}
    current = text

    normalized = unicodedata.normalize("NFKC", current)
    counts["unicode_nfkc_normalization"] += count_unicode_changes(current, normalized)
    current = normalized

    current, count = rules.control_char_rule.pattern.subn(rules.control_char_rule.replacement, current)
    counts["control_character_removal"] += count

    current, count = rules.non_breaking_space_rule.pattern.subn(rules.non_breaking_space_rule.replacement, current)
    counts["non_breaking_space_normalization"] += count

    current, count = rules.line_ending_rule.pattern.subn(rules.line_ending_rule.replacement, current)
    counts["line_ending_normalization"] += count

    current, count = apply_pattern_sequence(current, rules.header_footer_patterns)
    counts["header_footer_removal"] += count

    current, count = apply_pattern_sequence(current, rules.page_number_patterns)
    counts["page_number_removal"] += count

    current, count = apply_regex_sequence(current, rules.unit_mappings)
    counts["unit_normalization"] += count

    current, count = apply_regex_sequence(current, rules.term_replacements)
    counts["term_replacement"] += count

    current, count = rules.trailing_whitespace_rule.pattern.subn(rules.trailing_whitespace_rule.replacement, current)
    counts["trailing_whitespace_trimming"] += count

    current, count = rules.collapse_blank_lines_rule.pattern.subn(rules.collapse_blank_lines_rule.replacement, current)
    counts["collapse_blank_lines"] += count

    return current, counts


def merge_counts(accumulator: Dict[str, int], counts: Dict[str, int]) -> None:
    for stage in STAGE_ORDER:
        accumulator[stage] += counts.get(stage, 0)


def normalize_document(payload: dict, rules: NormalizationRules) -> Tuple[dict, Dict[str, int]]:
    stage_counts: Dict[str, int] = {stage: 0 for stage in STAGE_ORDER}

    content = payload.get("content")
    if not isinstance(content, str):
        content = "" if content is None else str(content)
    cleaned_content, content_counts = normalize_text(content, rules)
    merge_counts(stage_counts, content_counts)

    tables_raw = payload.get("tables")
    normalized_tables: List[str] = []
    if isinstance(tables_raw, list):
        for table in tables_raw:
            table_text = table if isinstance(table, str) else str(table)
            cleaned_table, table_counts = normalize_text(table_text, rules)
            normalized_tables.append(cleaned_table)
            merge_counts(stage_counts, table_counts)
    else:
        normalized_tables = []

    cleaned_payload = dict(payload)
    cleaned_payload["content"] = cleaned_content
    cleaned_payload["tables"] = normalized_tables
    cleaned_payload["cleaning_log"] = {stage: stage_counts[stage] for stage in STAGE_ORDER}

    return cleaned_payload, stage_counts


def discover_input_files(input_dir: Path) -> List[Path]:
    return sorted(input_dir.glob("*.json"), key=lambda path: path.name)


def process_file(input_path: Path, output_dir: Path, rules: NormalizationRules, dry_run: bool) -> Tuple[bool, Dict[str, int]]:
    stage_counts: Dict[str, int] = {stage: 0 for stage in STAGE_ORDER}

    try:
        with input_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        logger.exception("Failed to read JSON %s: %s", input_path, exc)
        return False, stage_counts

    if not isinstance(payload, dict):
        logger.error("Invalid JSON schema in %s: top-level object expected", input_path)
        return False, stage_counts

    cleaned_payload, stage_counts = normalize_document(payload, rules)

    output_path = output_dir / input_path.name
    if dry_run:
        logger.info("Dry run: would write %s", output_path)
        return True, stage_counts

    try:
        output_path.write_text(json.dumps(cleaned_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.exception("Failed to write JSON %s: %s", output_path, exc)
        return False, {stage: 0 for stage in STAGE_ORDER}

    return True, stage_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize parsed document JSON files into cleaned JSON outputs."
    )
    parser.add_argument(
        "--input-dir",
        default="/data/parsed",
        type=Path,
        help="Directory containing parsed JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/cleaned",
        type=Path,
        help="Directory where cleaned JSON files are written.",
    )
    parser.add_argument(
        "--rules",
        default="/workspace/trainer/configs/normalization_rules.yaml",
        type=Path,
        help="YAML config file containing normalization rules.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process files and report changes without writing outputs.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def print_summary(total_docs: int, stage_totals: Dict[str, int]) -> None:
    print("Cleaning summary")
    print(f"- total_docs_processed: {total_docs}")
    print("- per_stage_replacements:")
    for stage in STAGE_ORDER:
        print(f"  - {stage}: {stage_totals[stage]}")


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.input_dir.is_dir():
        logger.error("Input directory not found: %s", args.input_dir)
        return 1
    if not args.rules.is_file():
        logger.error("Rules file not found: %s", args.rules)
        return 1

    try:
        rules = load_rules(args.rules)
    except Exception as exc:
        logger.exception("Failed to load rules from %s: %s", args.rules, exc)
        return 1

    files = discover_input_files(args.input_dir)
    logger.info("Discovered %s JSON files", len(files))

    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    failures = 0
    stage_totals: Dict[str, int] = {stage: 0 for stage in STAGE_ORDER}

    for input_path in files:
        try:
            ok, doc_counts = process_file(input_path, args.output_dir, rules, args.dry_run)
            if ok:
                processed += 1
                merge_counts(stage_totals, doc_counts)
            else:
                failures += 1
        except Exception as exc:
            failures += 1
            logger.exception("Failed to process %s: %s", input_path, exc)

    logger.info("Finished. processed=%s failures=%s", processed, failures)
    print_summary(processed, stage_totals)
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
