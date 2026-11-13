#!/usr/bin/env python3
"""Offline STT candidate load/latency/memory probe for DGX Spark."""

from __future__ import annotations

import argparse
import io
import json
import math
import resource
import struct
import time
import wave

from trainer.demo_api.speech_to_text import TransformersWhisperSpeechToText


def probe_wav(seconds: float = 3.0) -> bytes:
    stream = io.BytesIO()
    with wave.open(stream, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        frames = (
            struct.pack("<h", int(500 * math.sin(2 * math.pi * 220 * index / 16_000)))
            for index in range(int(seconds * 16_000))
        )
        output.writeframes(b"".join(frames))
    return stream.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.monotonic()
    engine = TransformersWhisperSpeechToText(args.model, device=args.device)
    loaded = time.monotonic()
    audio = probe_wav()
    languages: dict[str, object] = {}
    for language in ("ko", "vi", "id", "th"):
        run_started = time.monotonic()
        result = engine.transcribe(audio, language)  # type: ignore[arg-type]
        languages[language] = {
            "latency_ms": round((time.monotonic() - run_started) * 1000, 2),
            "transcript": result.text,
        }
    print(json.dumps({
        "model": args.model,
        "device": args.device,
        "load_ms": round((loaded - started) * 1000, 2),
        "languages": languages,
        "process_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
