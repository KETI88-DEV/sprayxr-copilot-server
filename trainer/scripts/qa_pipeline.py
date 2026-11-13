#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _strip_think_blocks(text: str) -> str:
    import re
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _ensure_scripts_path() -> None:
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)


def _patch_torch_set_submodule() -> None:
    import torch
    if hasattr(torch.nn.Module, "set_submodule"):
        return

    def _set_submodule(self: torch.nn.Module, target: str, module: torch.nn.Module) -> None:
        atoms = target.split(".")
        mod = self
        for item in atoms[:-1]:
            mod = getattr(mod, item)
        setattr(mod, atoms[-1], module)

    torch.nn.Module.set_submodule = _set_submodule  # type: ignore[attr-defined]
    logger.info("Patched torch.nn.Module.set_submodule for torch %s", torch.__version__)


def load_llm(config: Dict[str, Any]) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    _patch_torch_set_submodule()

    model_path = config["llm"]["model_path"]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        use_safetensors=True,
        dtype=torch.bfloat16,
    )
    model.eval()
    return model, tokenizer


def generate_answer(
    question: str,
    context: str,
    model: Any,
    tokenizer: Any,
    config: Dict[str, Any],
) -> str:
    import torch

    system_prompt = config["prompt"]["system"]
    context_template = config["prompt"]["context_template"]
    user_template = config["prompt"]["user_template"]

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": context_template.format(context=context) + "\n\n" + user_template.format(question=question),
        },
    ]
    enable_thinking = config["generation"].get("enable_thinking", False)
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    inputs = tokenizer([text], return_tensors="pt", return_attention_mask=True).to(model.device)

    gen_kwargs: dict[str, Any] = {
        "do_sample": config["generation"]["do_sample"],
        "temperature": config["generation"]["temperature"],
        "max_new_tokens": config["generation"]["max_new_tokens"],
        "pad_token_id": tokenizer.eos_token_id,
    }
    if "top_p" in config["generation"]:
        gen_kwargs["top_p"] = config["generation"]["top_p"]
    if "top_k" in config["generation"]:
        gen_kwargs["top_k"] = config["generation"]["top_k"]

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)
    generated = output_ids[0][inputs.input_ids.shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True)
    answer = _strip_think_blocks(answer)
    return answer


def run_qa(
    question: str,
    index: Any,
    id_map: List[Dict[str, Any]],
    embedder: Any,
    reranker: Any,
    model: Any,
    tokenizer: Any,
    config: Dict[str, Any],
    q_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Full RAG QA pipeline: retrieve -> rerank -> generate -> format."""
    search_mod = importlib.import_module("search_index")
    formatter_mod = importlib.import_module("response_formatter")
    search_and_rerank = search_mod.search_and_rerank
    format_context = formatter_mod.format_context
    format_response = formatter_mod.format_response

    total_start = time.time()

    results, timing = search_and_rerank(question, index, id_map, embedder, reranker, config)
    context = format_context(results)

    gen_start = time.time()
    try:
        answer = generate_answer(question, context, model, tokenizer, config)
    except Exception:
        logger.exception("Answer generation failed")
        answer = "제공된 문서에 해당 정보가 없습니다"
    timing["generate_ms"] = (time.time() - gen_start) * 1000
    timing["total_pipeline_ms"] = (time.time() - total_start) * 1000

    response = format_response(question, answer, results, config, q_id=q_id)
    response["timing"] = timing
    response["_raw_results"] = results
    return response


def load_all_models(config: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_scripts_path()
    search_mod = importlib.import_module("search_index")
    load_faiss_index = search_mod.load_faiss_index
    load_metadata = search_mod.load_metadata
    load_embedder = search_mod.load_embedder
    load_reranker = search_mod.load_reranker

    t0 = time.time()
    index = load_faiss_index(config["paths"]["faiss_index"], config["index"]["ef_search"])
    metadata = load_metadata(config["paths"]["metadata_json"], config["embedding"]["model_name"])
    id_map = metadata["id_map"]
    logger.info("Loaded index and metadata in %.2fs", time.time() - t0)

    t0 = time.time()
    embedder = load_embedder(
        config["embedding"]["model_name"],
        config["embedding"]["device"],
        config["embedding"]["max_seq_length"],
    )
    logger.info("Loaded embedder in %.2fs", time.time() - t0)

    t0 = time.time()
    reranker = load_reranker(
        config["reranker"]["model_name"],
        config["reranker"]["device"],
        config["reranker"]["max_length"],
    )
    logger.info("Loaded reranker in %.2fs", time.time() - t0)

    t0 = time.time()
    model, tokenizer = load_llm(config)
    logger.info("Loaded LLM in %.2fs", time.time() - t0)

    return {
        "index": index,
        "id_map": id_map,
        "embedder": embedder,
        "reranker": reranker,
        "model": model,
        "tokenizer": tokenizer,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAG QA pipeline with FAISS retrieval and Qwen3 generation."
    )
    parser.add_argument(
        "--config",
        default="/workspace/trainer/configs/rag_config.yaml",
        type=Path,
        help="Path to rag_config.yaml.",
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--query",
        type=str,
        help="Single-question mode.",
    )
    mode_group.add_argument(
        "--input",
        type=Path,
        help="Batch mode input JSONL with at least a 'question' field.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Batch mode output JSONL path. Optional for single query mode.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def _load_batch_items(input_path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping invalid JSON at line %d", line_no)
                continue
            if "question" not in item:
                logger.warning("Skipping item without 'question' at line %d", line_no)
                continue
            items.append(item)
    return items


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.config.is_file():
        logger.error("Config not found: %s", args.config)
        return 1

    _ensure_scripts_path()

    try:
        search_mod = importlib.import_module("search_index")
        load_config = search_mod.load_config
    except Exception:
        logger.exception("Failed to import load_config from search_index.py")
        return 1

    config = load_config(args.config)

    try:
        models = load_all_models(config)
    except Exception:
        logger.exception("Failed to load retrieval or generation models")
        return 1

    if args.query is not None:
        started = time.time()
        try:
            response = run_qa(
                question=args.query,
                index=models["index"],
                id_map=models["id_map"],
                embedder=models["embedder"],
                reranker=models["reranker"],
                model=models["model"],
                tokenizer=models["tokenizer"],
                config=config,
            )
            response["elapsed_s"] = round(time.time() - started, 2)
        except Exception:
            logger.exception("Single-query inference failed")
            return 1

        output_text = json.dumps(response, ensure_ascii=False)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="utf-8") as fh:
                fh.write(output_text + "\n")
        print(output_text)
        return 0

    if args.input is None:
        logger.error("Batch mode requires --input")
        return 1
    if not args.input.is_file():
        logger.error("Input JSONL not found: %s", args.input)
        return 1
    if args.output is None:
        logger.error("Batch mode requires --output")
        return 1

    items = _load_batch_items(args.input)
    logger.info("Loaded %d batch items", len(items))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    success_count = 0

    with args.output.open("w", encoding="utf-8") as out_fh:
        for i, item in enumerate(items, start=1):
            question = item["question"]
            q_id = item.get("q_id", f"Q{i:03d}")
            started = time.time()
            try:
                response = run_qa(
                    question=question,
                    index=models["index"],
                    id_map=models["id_map"],
                    embedder=models["embedder"],
                    reranker=models["reranker"],
                    model=models["model"],
                    tokenizer=models["tokenizer"],
                    config=config,
                    q_id=q_id,
                )
                response["elapsed_s"] = round(time.time() - started, 2)
                response["error"] = False
                success_count += 1
            except Exception:
                logger.exception("Failed batch item %s", q_id)
                response = {
                    "q_id": q_id,
                    "question": question,
                    "answer": "",
                    "citations": [],
                    "confidence": "low",
                    "error": True,
                    "elapsed_s": round(time.time() - started, 2),
                    "timing": {},
                }

            out_fh.write(json.dumps(response, ensure_ascii=False) + "\n")

    logger.info("Batch completed: %d/%d successful", success_count, len(items))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
