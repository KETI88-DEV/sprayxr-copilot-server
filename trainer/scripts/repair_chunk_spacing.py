#!/usr/bin/env python3
"""Repair Korean word-spacing in space-stripped chunks using kiwipiepy, preserving chunk_ids."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


import re

_HANGUL_RUN = re.compile(r"[\uac00-\ud7a3]+")


def space_ratio(text: str) -> float:
    return text.count(" ") / max(len(text), 1) if text else 1.0


def max_hangul_run(text: str) -> int:
    runs = _HANGUL_RUN.findall(text)
    return max((len(r) for r in runs), default=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-space stripped Korean chunks.")
    parser.add_argument("--chunks", type=Path, default=Path("/data/chunked/chunks.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("/data/chunked/chunks_repaired.jsonl"))
    parser.add_argument("--max-run", type=int, default=6,
                        help="Repair chunks whose longest unbroken Hangul run exceeds this (glued words).")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    from kiwipiepy import Kiwi
    kiwi = Kiwi()

    records = [json.loads(l) for l in args.chunks.open("r", encoding="utf-8") if l.strip()]
    repaired = 0
    with args.output.open("w", encoding="utf-8") as fh:
        for rec in records:
            text = rec.get("text", "")
            if text and max_hangul_run(text) > args.max_run:
                respaced = kiwi.space(text)
                if max_hangul_run(respaced) < max_hangul_run(text):
                    rec["text"] = respaced
                    rec["spacing_repaired"] = True
                    repaired += 1
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info("Repaired %d/%d chunks (max_run>%d) -> %s", repaired, len(records), args.max_run, args.output)
    print(f"repaired={repaired}/{len(records)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
