#!/usr/bin/env python3
"""8-bit quantized inference smoke test (T16 proxy: A40 bnb INT8, Spark FP8)."""
from __future__ import annotations

import argparse
import logging
import os
import time

os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = (
    "선박 도장에서 DFT 90/10 규칙이 무엇인지 설명하고, 목표 DFT 100µm에서 "
    "측정값 [82, 88, 140]µm이 나왔을 때 합격 여부를 판단해줘."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a model in 8-bit (bitsandbytes INT8) and run one generation."
    )
    parser.add_argument("--model-path", default="/models/Qwen3.6-27B")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", default=256, type=int)
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    logger.info("Loading tokenizer from %s", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)

    logger.info("Loading model in 8-bit (bnb INT8) from %s", args.model_path)
    t0 = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map=args.device_map,
            dtype=torch.bfloat16,
        )
    except Exception:
        logger.exception(
            "Model load failed (VRAM allocated at failure: %.2f GB, reserved: %.2f GB)",
            torch.cuda.memory_allocated() / 1e9,
            torch.cuda.memory_reserved() / 1e9,
        )
        return 1
    load_s = time.time() - t0
    logger.info(
        "Model loaded in %.1fs, VRAM allocated: %.2f GB",
        load_s,
        torch.cuda.memory_allocated() / 1e9,
    )

    messages = [{"role": "user", "content": args.prompt}]
    encoded = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)

    t1 = time.time()
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_s = time.time() - t1

    new_tokens = output[0][input_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    n_tokens = int(new_tokens.shape[0])

    print("=" * 60)
    print("8-bit Inference Smoke Summary")
    print("=" * 60)
    print(f"  Model:            {args.model_path}")
    print(f"  Load time:        {load_s:.1f} s")
    print(f"  New tokens:       {n_tokens} ({n_tokens / gen_s:.2f} tok/s)")
    print(f"  Peak VRAM:        {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print("=" * 60)
    print("=== GENERATION ===")
    print(text)
    print("=== END ===")

    if not text.strip():
        logger.error("Empty generation - smoke failed")
        return 1
    logger.info("8-bit inference smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
