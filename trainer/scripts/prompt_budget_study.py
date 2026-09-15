#!/usr/bin/env python3
"""Offline study: legacy ``len(text)//2`` prompt trimming vs exact token counting.

GPU-free and read-only. RAG context is rebuilt from saved evaluation outputs
instead of re-running retrieval:

* agentic set (30 items): every run in ``--agentic-glob`` stores ``citations``,
  which ``AgentPipeline._extract_citations`` fills with the IDs cited in the
  answer followed by every retrieved chunk ID. The retrieved set is the
  intersection across runs; its order is the most common order across runs
  (answers move cited IDs to the front, so this approximates rerank order).
  Rerank scores are not saved and are rendered as a fixed 4-decimal
  placeholder of the same width.
* RAG-only set: ``source_docs`` of a docqa run gives chunk IDs, ranks and
  scores. Texts come from the current ``metadata.json``.
* fast-route stress: the agentic chunks under ``prompt.fast_system`` with no
  environment, i.e. an 8-chunk knowledge question.

For each item it records chunks kept by the legacy estimator at the configured
``max_model_len`` and by the current ``AgentPipeline._build_messages`` with the
real tokenizer, the exact rendered token count of both prompts, and the
smallest ``max_model_len`` that keeps the legacy chunk counts everywhere.

With ``--replay-tools`` the saved tool calls are replayed through the MCP app
in-process (FastAPI ASGI transport; no live server is contacted) to measure how
far tool rounds grow the prompt.

Run inside the trainer container:

    nice -n 19 python3 -m trainer.scripts.prompt_budget_study --replay-tools \\
        --output /tmp/prompt_budget_study.json
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _path in (REPO_ROOT, SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from trainer.scripts.agent_pipeline import (  # noqa: E402
    ROUTE_MAX_ROUNDS,
    ROUTE_MAX_TOKENS,
    AgentPipeline,
    SemanticRouter,
    compress_evidence,
)
from trainer.scripts.prompt_budget import TokenizerTokenCounter  # noqa: E402

LEGACY_TEMPLATE_RESERVE = 64
PLACEHOLDER_SCORE = 0.5
SWEEP_STEP = 64


def _quantiles(values: list[int]) -> dict[str, int]:
    ordered = sorted(values)
    if not ordered:
        return {}

    def pick(p: float) -> int:
        return ordered[min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))]

    return {"n": len(ordered), "p50": pick(0.5), "p95": pick(0.95), "max": ordered[-1], "min": ordered[0]}


def legacy_build(
    config: dict[str, Any],
    instruction: str,
    environment: dict[str, Any] | None,
    rag_chunks: list[dict[str, Any]],
    system_prompt: str,
) -> tuple[list[dict[str, str]], int]:
    """The pre-change ``_build_messages`` trimming, frozen for comparison."""
    context_template = str(config.get("prompt", {}).get("context_template", "{context}"))
    max_model_len = int(config.get("llm", {}).get("max_model_len", 4096))
    max_gen_tokens = int(config.get("generation", {}).get("max_new_tokens", 768))
    prompt_budget = max_model_len - max_gen_tokens - LEGACY_TEMPLATE_RESERVE

    lines: list[str] = []
    for chunk in rag_chunks:
        score = chunk.get("rerank_score")
        score_text = f" (score={score:.4f})" if isinstance(score, (int, float)) else ""
        lines.append(f"[{chunk.get('chunk_id', '')}]{score_text}\n{chunk.get('text', '')}")

    def user_content(kept: list[str]) -> str:
        parts: list[str] = []
        rag_context = context_template.format(context="\n\n".join(kept)) if kept else ""
        if rag_context:
            parts.append("[RAG 문맥]\n" + rag_context)
        if environment:
            parts.append("[환경 데이터]\n" + json.dumps(environment, ensure_ascii=False, indent=2))
        parts.append("[사용자 요청]\n" + instruction)
        return "\n\n".join(parts)

    content = user_content(lines)
    while lines and len(system_prompt + content) // 2 > prompt_budget:
        lines.pop()
        content = user_content(lines)
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}], len(lines)


def legacy_compresses(config: dict[str, Any], rag_chunks: list[dict[str, Any]]) -> bool:
    max_model_len = int(config.get("llm", {}).get("max_model_len", 4096))
    max_gen = int(config.get("generation", {}).get("max_new_tokens", 768))
    budget = max_model_len - max_gen - LEGACY_TEMPLATE_RESERVE
    return sum(len(c.get("text", "")) for c in rag_chunks) // 2 > budget


def kept_chunks(messages: list[dict[str, str]], rag_chunks: list[dict[str, Any]]) -> int:
    content = messages[1]["content"]
    kept = 0
    for chunk in rag_chunks:
        if f"[{chunk['chunk_id']}]" not in content:
            break
        kept += 1
    return kept


def _pipeline(config: dict[str, Any], counter: TokenizerTokenCounter) -> AgentPipeline:
    pipe = AgentPipeline.__new__(AgentPipeline)
    pipe.config = config
    pipe._token_counter = counter
    return pipe


def _with_window(config: dict[str, Any], max_model_len: int) -> dict[str, Any]:
    return {**config, "llm": {**config.get("llm", {}), "max_model_len": max_model_len}}


def load_agentic_items(eval_set: Path, pattern: str, id_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"No agentic runs match {pattern}")
    runs = []
    for path in files:
        with open(path, encoding="utf-8") as fh:
            runs.append({row["q_id"]: row for row in map(json.loads, fh)})

    items: list[dict[str, Any]] = []
    with eval_set.open(encoding="utf-8") as fh:
        for question in map(json.loads, fh):
            q_id = question["q_id"]
            cited = [r[q_id]["citations"] for r in runs if q_id in r]
            shared = set(cited[0]).intersection(*map(set, cited[1:])) if cited else set()
            shared = {cid for cid in shared if cid in id_map}
            orders = collections.Counter(tuple(c for c in seq if c in shared) for seq in cited)
            order = list(orders.most_common(1)[0][0]) if orders else []
            items.append(
                {
                    "q_id": q_id,
                    "question": question["question"],
                    "environment": question.get("context_data") or {},
                    "chunks": [
                        {"chunk_id": cid, "text": id_map[cid]["text"], "rerank_score": PLACEHOLDER_SCORE}
                        for cid in order
                    ],
                    "order_agreement": orders.most_common(1)[0][1] / len(cited) if cited else 0.0,
                    "tool_runs": [
                        {"run": files[i], "tool_calls": r[q_id]["tool_calls"]}
                        for i, r in enumerate(runs)
                        if q_id in r and r[q_id]["tool_calls"]
                    ],
                }
            )
    return items


def load_docqa_items(path: Path, id_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for row in map(json.loads, fh):
            docs = [d for d in row.get("source_docs", []) if d.get("chunk_id") in id_map]
            items.append(
                {
                    "q_id": row["q_id"],
                    "question": row["question"],
                    "environment": {},
                    "chunks": [
                        {
                            "chunk_id": d["chunk_id"],
                            "text": id_map[d["chunk_id"]]["text"],
                            "rerank_score": d.get("rerank_score"),
                        }
                        for d in docs
                    ],
                }
            )
    return items


def evaluate_item(
    item: dict[str, Any],
    config: dict[str, Any],
    counter: TokenizerTokenCounter,
    system_key: str,
    sweep: list[int],
) -> dict[str, Any]:
    system_prompt = str(config["prompt"][system_key]).strip()
    override = system_prompt if system_key == "fast_system" else None
    chunks = item["chunks"]
    environment = item["environment"] or None

    max_gen = int(config["generation"]["max_new_tokens"])
    legacy_compressed = legacy_compresses(config, chunks)
    legacy_evidence = compress_evidence(item["question"], chunks) if legacy_compressed else chunks
    legacy_messages, legacy_kept = legacy_build(config, item["question"], environment, legacy_evidence, system_prompt)

    def build(window: int, evidence: list[dict[str, Any]]) -> list[dict[str, str]]:
        return _pipeline(_with_window(config, window), counter)._build_messages(
            instruction=item["question"],
            environment=environment,
            rag_chunks=evidence,
            session_context_summary="",
            system_prompt=override,
        )

    # Exact tokens of the prompt holding the first k chunks. The new trimming keeps
    # the largest k whose prompt fits window - max_new_tokens, so these decide every window.
    unbounded = 1 << 30
    prefix_tokens = [counter.count_messages(build(unbounded, chunks[:k])) for k in range(len(chunks) + 1)]
    evidence_tokens = sum(counter.count_text(c["text"]) for c in chunks)

    def new_kept_at(window: int) -> tuple[int, bool]:
        budget = window - max_gen
        compressed = evidence_tokens > budget
        if compressed:
            evidence = compress_evidence(item["question"], chunks)
            return kept_chunks(build(window, evidence), evidence), True
        return max(k for k, tokens in enumerate(prefix_tokens) if tokens <= budget or k == 0), False

    configured = int(config["llm"]["max_model_len"])
    new_messages = build(configured, chunks if evidence_tokens <= configured - max_gen else compress_evidence(item["question"], chunks))
    new_kept, new_compressed = new_kept_at(configured)
    assert new_kept == kept_chunks(new_messages, chunks) or new_compressed, "prefix model disagrees with _build_messages"
    matching = [w for w in sweep if new_kept_at(w) == (legacy_kept, legacy_compressed)]
    return {
        "q_id": item["q_id"],
        "chunks_in": len(chunks),
        "legacy_compressed": legacy_compressed,
        "legacy_kept": legacy_kept,
        "legacy_prompt_tokens": counter.count_messages(legacy_messages),
        "legacy_estimate": len(legacy_messages[0]["content"] + legacy_messages[1]["content"]) // 2,
        "new_compressed": new_compressed,
        "new_kept": new_kept,
        "new_prompt_tokens": counter.count_messages(new_messages),
        "full_prompt_tokens": prefix_tokens[-1],
        "min_window_matching_legacy": min(matching) if matching else None,
        "max_window_matching_legacy": max(matching) if matching else None,
        "order_agreement": item.get("order_agreement"),
        "_messages": legacy_messages,
    }


async def replay_tool_growth(
    items: list[dict[str, Any]],
    rows: dict[str, dict[str, Any]],
    counter: TokenizerTokenCounter,
    tool_max_tokens: int,
) -> list[dict[str, Any]]:
    import httpx

    import tool_calling_adapter as adapter_mod
    from mcp_servers.main import app

    real_client = httpx.AsyncClient

    def asgi_client(*_args: Any, **kwargs: Any) -> httpx.AsyncClient:
        return real_client(transport=httpx.ASGITransport(app=app), timeout=kwargs.get("timeout"))

    adapter_mod.httpx.AsyncClient = asgi_client  # type: ignore[misc]
    adapter = adapter_mod.ToolCallingAdapter(
        schema_dir=REPO_ROOT / "trainer/configs/tool_schemas",
        protocol_path=REPO_ROOT / "trainer/configs/tool_calling_protocol.json",
        mcp_base_url="http://mcp.inprocess",
        pass_native_tools=False,
    )
    router = SemanticRouter()
    growth: list[dict[str, Any]] = []
    try:
        for item in items:
            base = rows[item["q_id"]]["_messages"]
            route = router.classify(item["question"], item["environment"])
            synth_tokens = ROUTE_MAX_TOKENS.get(route, 768)
            for run in item["tool_runs"]:
                conversation = list(base)
                peak_round = 0
                calls = run["tool_calls"][: ROUTE_MAX_ROUNDS.get(route, 5)]
                for call in calls:
                    block = json.dumps({"name": call["name"], "arguments": call["arguments"]}, ensure_ascii=False)
                    conversation.append({"role": "assistant", "content": f"<tool_call>{block}</tool_call>"})
                    result = await adapter.dispatch_tool_call(call["name"], call["arguments"])
                    result.setdefault("tool_name", call["name"])
                    conversation.extend(adapter.format_tool_results([result]))
                    peak_round = max(peak_round, counter.count_messages(conversation))
                synthesis = counter.count_messages(
                    [*conversation, {"role": "user", "content": adapter_mod.SYNTHESIS_INSTRUCTION}]
                )
                # Each assistant round may carry up to tool_max_tokens of prose besides the call.
                prose_bound = len(calls) * tool_max_tokens
                growth.append(
                    {
                        "q_id": item["q_id"],
                        "run": run["run"],
                        "route": route,
                        "tool_calls": len(calls),
                        "base_prompt_tokens": counter.count_messages(base),
                        "last_round_prompt_tokens": peak_round,
                        "synthesis_prompt_tokens": synthesis,
                        "synthesis_window_needed": synthesis + synth_tokens,
                        "synthesis_window_needed_with_prose_bound": synthesis + prose_bound + synth_tokens,
                    }
                )
    finally:
        adapter_mod.httpx.AsyncClient = real_client  # type: ignore[misc]
    return growth


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "trainer/configs/rag_config_fewshot_27b.yaml")
    parser.add_argument("--legacy-max-model-len", type=int, default=8192)
    parser.add_argument("--eval-set", type=Path, default=Path("/data/eval/eval_set_agentic.jsonl"))
    parser.add_argument(
        "--agentic-glob",
        default="/output/eval_results/qwen36_27b_v6_1_multiseed/**/agentic_responses.jsonl",
    )
    parser.add_argument(
        "--docqa-responses",
        type=Path,
        default=Path("/output/eval_results/docqa_eval_iter17/rag_only_raw_responses.jsonl"),
    )
    parser.add_argument("--metadata", type=Path, default=None, help="defaults to paths.metadata_json")
    parser.add_argument("--replay-tools", action="store_true")
    parser.add_argument("--sweep-max", type=int, default=16384)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for forbidden in ("/output", "/data", "/models"):
        if str(args.output.resolve()).startswith(forbidden):
            parser.error(f"--output must not be under {forbidden}")

    import torch
    from transformers import AutoTokenizer

    torch.set_num_threads(min(8, os.cpu_count() or 1))
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config = _with_window(config, args.legacy_max_model_len)
    tokenizer = AutoTokenizer.from_pretrained(config["llm"]["model_path"], trust_remote_code=True)
    counter = TokenizerTokenCounter(tokenizer)

    metadata_path = args.metadata or Path(config["paths"]["metadata_json"])
    id_map = {row["chunk_id"]: row for row in json.loads(metadata_path.read_text(encoding="utf-8"))["id_map"]}
    agentic = load_agentic_items(args.eval_set, args.agentic_glob, id_map)
    docqa = load_docqa_items(args.docqa_responses, id_map)

    max_gen = int(config["generation"]["max_new_tokens"])
    sweep = list(range(max_gen + SWEEP_STEP, args.sweep_max + 1, SWEEP_STEP))
    datasets = {
        "agentic_tool_prompt": (agentic, "system"),
        "docqa_fast_prompt": (docqa, "fast_system"),
        "agentic_chunks_fast_prompt": (
            [{**item, "environment": {}} for item in agentic],
            "fast_system",
        ),
    }

    report: dict[str, Any] = {
        "config": str(args.config),
        "legacy_max_model_len": args.legacy_max_model_len,
        "max_new_tokens": max_gen,
        "sources": {"agentic_glob": args.agentic_glob, "docqa": str(args.docqa_responses)},
        "datasets": {},
    }
    agentic_rows: dict[str, dict[str, Any]] = {}
    for name, (items, system_key) in datasets.items():
        rows = [evaluate_item(item, config, counter, system_key, sweep) for item in items]
        if name == "agentic_tool_prompt":
            agentic_rows = {row["q_id"]: row for row in rows}
        windows = [row["min_window_matching_legacy"] for row in rows]
        report["datasets"][name] = {
            "items": len(rows),
            "kept_changed_at_legacy_window": sum(1 for r in rows if r["new_kept"] != r["legacy_kept"]),
            "legacy_trimmed_items": sum(1 for r in rows if r["legacy_kept"] < r["chunks_in"]),
            "new_trimmed_items_at_legacy_window": sum(1 for r in rows if r["new_kept"] < r["chunks_in"]),
            "legacy_prompt_tokens": _quantiles([r["legacy_prompt_tokens"] for r in rows]),
            "new_prompt_tokens_at_legacy_window": _quantiles([r["new_prompt_tokens"] for r in rows]),
            "legacy_over_window_items": sum(
                1 for r in rows if r["legacy_prompt_tokens"] + max_gen > args.legacy_max_model_len
            ),
            "min_window_matching_legacy_all": None if None in windows else max(windows),
            "rows": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
        }

    if args.replay_tools:
        growth = asyncio.run(
            replay_tool_growth(agentic, agentic_rows, counter, int(config["generation"].get("max_new_tokens_tool", 256)))
        )
        report["tool_growth"] = {
            "replayed_runs": len(growth),
            "synthesis_prompt_tokens": _quantiles([g["synthesis_prompt_tokens"] for g in growth]),
            "synthesis_window_needed": _quantiles([g["synthesis_window_needed"] for g in growth]),
            "synthesis_window_needed_with_prose_bound": _quantiles(
                [g["synthesis_window_needed_with_prose_bound"] for g in growth]
            ),
            "rows": growth,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    for name, summary in report["datasets"].items():
        print(
            f"{name}: items={summary['items']} kept_changed={summary['kept_changed_at_legacy_window']} "
            f"legacy_trimmed={summary['legacy_trimmed_items']} new_trimmed={summary['new_trimmed_items_at_legacy_window']} "
            f"legacy_tokens={summary['legacy_prompt_tokens']} over_window={summary['legacy_over_window_items']} "
            f"min_window_matching_legacy={summary['min_window_matching_legacy_all']}"
        )
    if "tool_growth" in report:
        tg = report["tool_growth"]
        print(
            f"tool_growth: runs={tg['replayed_runs']} synthesis_prompt={tg['synthesis_prompt_tokens']} "
            f"window_needed={tg['synthesis_window_needed']} with_prose_bound={tg['synthesis_window_needed_with_prose_bound']}"
        )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
