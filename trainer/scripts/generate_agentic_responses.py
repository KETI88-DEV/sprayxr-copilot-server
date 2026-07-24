#!/usr/bin/env python3

from __future__ import annotations

import os
import argparse
import importlib
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def _ensure_scripts_path() -> None:
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)


def _resolve_adapter_path(adapter_arg: Path | None, qlora_config_path: Path) -> str | None:
    if adapter_arg is not None:
        return str(adapter_arg)
    if not qlora_config_path.is_file():
        return None

    with qlora_config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    adapter_dir = cfg.get("output", {}).get("adapter_dir")
    if isinstance(adapter_dir, str) and adapter_dir:
        return adapter_dir
    return None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            payload = line.strip()
            if not payload:
                continue
            try:
                item = json.loads(payload)
            except json.JSONDecodeError:
                logger.warning("Skipping invalid JSONL line %d", line_no)
                continue
            if not isinstance(item, dict):
                logger.warning("Skipping non-object JSON at line %d", line_no)
                continue
            items.append(item)
    return items


def _should_enable_thinking(question: dict[str, Any]) -> bool:
    force = os.environ.get("EVAL_FORCE_THINKING")
    if force == "0":
        return False
    if force == "1":
        return True
    reasoning_steps = int(question.get("reasoning_steps", 3))
    required_tools = question.get("tool_calls_required", [])
    if reasoning_steps <= 2 and isinstance(required_tools, list) and len(required_tools) <= 1:
        return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate responses JSONL for agentic evaluation.")
    parser.add_argument(
        "--eval-set",
        type=Path,
        default=Path("/data/eval/eval_set_agentic.jsonl"),
        help="Path to eval_set_agentic.jsonl.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/workspace/trainer/configs/rag_config.yaml"),
        help="Path to rag_config.yaml.",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="Optional QLoRA adapter path.",
    )
    parser.add_argument(
        "--qlora-config",
        type=Path,
        default=Path("/workspace/trainer/configs/qlora_config.yaml"),
        help="Path to qlora_config.yaml for adapter fallback.",
    )
    parser.add_argument(
        "--mcp-url",
        type=str,
        default="http://localhost:8100",
        help="MCP server base URL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/output/eval_results/agentic_responses.jsonl"),
        help="Path to write generated responses JSONL.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-question timeout in seconds (default: 600).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible generation (default: 42).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.eval_set.is_file():
        logger.error("Eval set not found: %s", args.eval_set)
        return 1
    if not args.config.is_file():
        logger.error("Config not found: %s", args.config)
        return 1

    import random
    import torch
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    logger.info("Reproducibility seed set: %d", args.seed)

    adapter_path = _resolve_adapter_path(args.adapter, args.qlora_config)

    _ensure_scripts_path()
    agent_mod = importlib.import_module("agent_pipeline")
    AgentPipeline = agent_mod.AgentPipeline

    try:
        pipeline = AgentPipeline(
            config_path=args.config,
            adapter_path=adapter_path,
            mcp_base_url=args.mcp_url,
        )
    except Exception:
        logger.exception("Failed to initialize AgentPipeline")
        return 1

    questions = _load_jsonl(args.eval_set)
    logger.info("Loaded %d eval questions", len(questions))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    response_count = 0
    tool_call_count = 0
    total_answer_chars = 0

    # Per-question timeout (seconds). Prevents infinite hanging.
    question_timeout = int(args.timeout) if hasattr(args, "timeout") and args.timeout else 600

    class _QuestionTimeout(Exception):
        pass

    def _timeout_handler(signum: int, frame: Any) -> None:
        raise _QuestionTimeout(f"Question exceeded {question_timeout}s timeout")

    with args.output.open("w", encoding="utf-8") as out_fh:
        for idx, item in enumerate(questions, start=1):
            q_id = str(item.get("q_id", f"AQ{idx:03d}"))
            instruction = str(item.get("question", "")).strip()
            environment = item.get("context_data", {})

            if not instruction:
                logger.warning("Skipping %s due to empty question", q_id)
                continue
            if not isinstance(environment, dict):
                environment = {}

            enable_thinking = _should_enable_thinking(item)
            logger.info("[%d/%d] %s thinking=%s", idx, len(questions), q_id, enable_thinking)

            q_start = time.monotonic()
            try:
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(question_timeout)

                result = pipeline.run(
                    instruction=instruction,
                    environment=environment,
                    session_id=q_id,
                    enable_thinking=enable_thinking,
                )
                signal.alarm(0)

                text_guide = result.get("text_guide", {})
                final_text = str(text_guide.get("body", "")) if text_guide else str(result.get("response", ""))
                tool_calls = result.get("tool_calls", [])
                normalized_tools = [
                    {
                        "name": call.get("name", ""),
                        "arguments": call.get("arguments", {}),
                    }
                    for call in tool_calls
                    if isinstance(call, dict) and call.get("name")
                ]
                metadata = result.get("metadata", {})
                thinking = str(metadata.get("thinking", ""))
                citations = result.get("citations", []) or []
            except _QuestionTimeout:
                signal.alarm(0)
                logger.error("TIMEOUT: %s exceeded %ds — skipping", q_id, question_timeout)
                final_text = "[TIMEOUT]"
                normalized_tools = []
                thinking = ""
                citations = []
            except Exception:
                signal.alarm(0)
                logger.exception("Generation failed for %s", q_id)
                final_text = ""
                normalized_tools = []
                thinking = ""
                citations = []

            elapsed = time.monotonic() - q_start
            logger.info("[%d/%d] %s completed in %.1fs, tools=%d, len=%d",
                        idx, len(questions), q_id, elapsed, len(normalized_tools), len(final_text))

            output_item = {
                "q_id": q_id,
                "tool_calls": normalized_tools,
                "answer": final_text,
                "response": final_text,
                "thinking": thinking,
                "citations": citations,
            }
            out_fh.write(json.dumps(output_item, ensure_ascii=False) + "\n")
            out_fh.flush()  # flush after each question to prevent data loss

            response_count += 1
            tool_call_count += len(normalized_tools)
            total_answer_chars += len(final_text)

    avg_response_length = total_answer_chars / max(response_count, 1)
    print(f"total_questions={response_count}")
    print(f"tool_calls_made={tool_call_count}")
    print(f"avg_response_length={avg_response_length:.2f}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
