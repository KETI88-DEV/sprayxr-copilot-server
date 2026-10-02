#!/usr/bin/env python3
"""Benchmark voice Query latency by language and report host-memory high water."""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import psutil

LANGUAGES = ("ko", "vi", "id", "th")


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction))))
    return ordered[position]


def summarize(latencies_ms: list[float]) -> dict[str, float | int]:
    return {
        "count": len(latencies_ms),
        "mean_ms": round(statistics.fmean(latencies_ms), 2) if latencies_ms else 0.0,
        "p50_ms": round(percentile(latencies_ms, 0.50), 2),
        "p95_ms": round(percentile(latencies_ms, 0.95), 2),
        "max_ms": round(max(latencies_ms), 2) if latencies_ms else 0.0,
    }


def run_benchmark(
    base_url: str,
    samples: dict[str, Path],
    *,
    repetitions: int,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    peak_memory = psutil.virtual_memory().used
    running = True

    def sample_memory() -> None:
        nonlocal peak_memory
        while running:
            peak_memory = max(peak_memory, psutil.virtual_memory().used)
            time.sleep(0.05)

    watcher = threading.Thread(target=sample_memory, daemon=True)
    watcher.start()
    results: dict[str, Any] = {}
    try:
        with httpx.Client(base_url=base_url, timeout=timeout_seconds) as client:
            for language in LANGUAGES:
                audio = samples[language].read_bytes()
                latencies: list[float] = []
                transcripts: list[str] = []
                for index in range(repetitions):
                    started = time.monotonic()
                    response = client.post(
                        "/api/v1/copilot/query-voice",
                        data={
                            "user_id": "BENCH",
                            "language": language,
                            "request_id": f"BENCH-{language}-{index}",
                            "use_rag": "Y",
                        },
                        files={"audio": (f"{language}.wav", audio, "audio/wav")},
                    )
                    response.raise_for_status()
                    latencies.append((time.monotonic() - started) * 1000)
                    transcripts.append(str(response.json()["data"]["transcript"]))
                results[language] = summarize(latencies) | {"transcripts": transcripts}
    finally:
        running = False
        watcher.join(timeout=1)
    return {
        "languages": results,
        "host_memory_peak_bytes": peak_memory,
        "host_memory_available_bytes": psutil.virtual_memory().available,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:58000")
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    samples = {language: args.sample_dir / f"{language}.wav" for language in LANGUAGES}
    missing = [str(path) for path in samples.values() if not path.is_file()]
    if missing:
        parser.error("missing samples: " + ", ".join(missing))
    report = run_benchmark(args.base_url, samples, repetitions=max(1, args.repetitions))
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
