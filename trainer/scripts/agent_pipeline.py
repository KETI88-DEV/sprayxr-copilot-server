#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
import os
import re
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

# vLLM requires 'spawn' multiprocessing when CUDA is already initialized
# (e.g., by embedding models). Must be set before any vLLM import.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    import vllm as _vllm_check  # noqa: F401
    _VLLM_AVAILABLE = True
    del _vllm_check
except ImportError:
    _VLLM_AVAILABLE = False

logger = logging.getLogger(__name__)

THINK_BLOCK_REGEX = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL)
TOOL_CALL_BLOCK_REGEX = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", flags=re.DOTALL)
XR_REQUIRED_KEYS = {"xr_type", "steps"}
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.?!。\n])\s+")
_WORD_RE = re.compile(r"\w+")


# ---------------------------------------------------------------------------
# Semantic Router — keyword-based Fast/Deep classification
# ---------------------------------------------------------------------------

# Deep path keywords: questions that need tool calling, sensor analysis, real-time data
_DEEP_KEYWORDS: set[str] = {
    # Tool-related (measurement, judgement)
    "측정", "수치", "dft", "도막두께", "합격", "불합격", "판정",
    "이상", "탐지", "anomaly", "센서", "데이터",
    # Real-time / environment
    "현재", "상태", "확인해", "어때", "코칭", "실시간",
    # Tool-specific actions
    "순서", "궤적", "자세", "포즈", "시계열", "추이",
    "위치", "좌표", "블록", "구역",
    # XR content
    "보여줘", "가이드", "xr", "시연", "시각화",
    # Comparative / analytical
    "비교", "분석", "점검", "검사", "진단",
}


class SemanticRouter:
    """Classifies questions as 'fast' (docQA only) or 'deep' (agentic tool loop).

    Fast path skips the tool loop entirely for ~3x latency reduction on simple QA.
    """

    def __init__(self, deep_keywords: set[str] | None = None) -> None:
        self.deep_keywords = deep_keywords or _DEEP_KEYWORDS

    def classify(self, instruction: str, environment: dict[str, Any] | None = None) -> str:
        """Return 'fast' or 'deep' based on instruction content and environment data."""
        if environment and len(environment) > 0:
            return "deep"

        text_lower = instruction.lower()
        for keyword in self.deep_keywords:
            if keyword.lower() in text_lower:
                return "deep"
        return "fast"


# ---------------------------------------------------------------------------
# Evidence Compression — extract query-relevant sentences from RAG chunks
# ---------------------------------------------------------------------------

def compress_evidence(
    query: str,
    chunks: list[dict[str, Any]],
    max_sentences_per_chunk: int = 3,
    min_sentence_len: int = 10,
) -> list[dict[str, Any]]:
    """Extract only sentences that overlap with the query, reducing prompt by ~80%.

    Falls back to original chunks if compression yields nothing.
    """
    query_tokens = set(_WORD_RE.findall(query.lower()))
    if not query_tokens:
        return chunks

    compressed: list[dict[str, Any]] = []
    for chunk in chunks:
        text = chunk.get("text", "")
        sentences = _SENTENCE_SPLIT_RE.split(text)

        scored: list[tuple[int, str]] = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < min_sentence_len:
                continue
            sent_tokens = set(_WORD_RE.findall(sent.lower()))
            overlap = len(query_tokens & sent_tokens)
            if overlap > 0:
                scored.append((overlap, sent))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_sents = [s for _, s in scored[:max_sentences_per_chunk]]

        if top_sents:
            compressed.append({
                "chunk_id": chunk.get("chunk_id", ""),
                "text": " ".join(top_sents),
                "rerank_score": chunk.get("rerank_score"),
            })

    return compressed if compressed else chunks


def _ensure_scripts_path() -> None:
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)


def _patch_torch_set_submodule() -> None:
    if hasattr(torch.nn.Module, "set_submodule"):
        return

    def _set_submodule(self: torch.nn.Module, target: str, module: torch.nn.Module) -> None:
        atoms = target.split(".")
        current = self
        for item in atoms[:-1]:
            current = getattr(current, item)
        setattr(current, atoms[-1], module)

    setattr(torch.nn.Module, "set_submodule", _set_submodule)


def _run_coro_sync(coro: Any) -> Any:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    if not loop.is_running():
        return loop.run_until_complete(coro)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, coro)
        return future.result()


class SessionContext:
    def __init__(self, max_turns: int = 3):
        self.max_turns = max_turns
        self.sessions: dict[str, list[dict[str, Any]]] = {}

    def get_context(self, session_id: str) -> list[dict[str, Any]]:
        turns = self.sessions.get(session_id, [])
        return turns[-self.max_turns :]

    def update(self, session_id: str, turn_data: dict[str, Any]) -> None:
        turns = self.sessions.setdefault(session_id, [])
        turns.append(turn_data)
        if len(turns) > self.max_turns:
            self.sessions[session_id] = turns[-self.max_turns :]

    def summarize_previous(self, session_id: str) -> str:
        turns = self.get_context(session_id)
        if not turns:
            return ""

        lines: list[str] = []
        for idx, turn in enumerate(turns, start=1):
            instruction = str(turn.get("instruction", "")).strip().replace("\n", " ")
            response = str(turn.get("response", "")).strip().replace("\n", " ")
            tools = turn.get("tool_calls", [])
            tool_names = ", ".join(str(t.get("name", "")) for t in tools if isinstance(t, dict) and t.get("name"))
            preview = response[:180]
            line = f"Turn {idx} | instruction: {instruction[:80]} | response: {preview}"
            if tool_names:
                line += f" | tools: {tool_names}"
            lines.append(line)
        return "\n".join(lines)


class VLLMWrapper:
    """Makes vLLM LLM compatible with tool_calling_adapter's generate_response interface."""

    def __init__(
        self,
        llm: Any,
        sampling_params: Any,
        lora_request: Any | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> None:
        self.llm = llm
        self.sampling_params = sampling_params
        self.lora_request = lora_request
        self.tools = tools

    def generate_response(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> str:
        effective_tools = tools if tools is not None else self.tools
        kwargs: dict[str, Any] = {}
        if self.lora_request:
            kwargs["lora_request"] = self.lora_request
        if effective_tools:
            kwargs["tools"] = effective_tools
        outputs = self.llm.chat(
            messages=[messages],
            sampling_params=self.sampling_params,
            **kwargs,
        )
        return outputs[0].outputs[0].text


class AgentPipeline:
    def __init__(
        self,
        config_path: str | Path,
        adapter_path: str | Path | None = None,
        mcp_base_url: str = "http://localhost:8100",
    ) -> None:
        _ensure_scripts_path()
        _patch_torch_set_submodule()

        self.config_path = Path(config_path)
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        self.config = self._load_yaml(self.config_path)
        self.session_context = SessionContext(max_turns=3)
        self.router = SemanticRouter()
        self.backend = str(self.config.get("llm", {}).get("backend", "hf"))

        self.model, self.tokenizer = self._load_model_and_tokenizer(adapter_path)
        self.tool_adapter = self._load_tool_adapter(mcp_base_url)
        self.index, self.id_map, self.embedder, self.reranker = self._load_rag_components()

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            raise ValueError(f"Invalid YAML object: {path}")
        return data

    def _load_model_and_tokenizer(self, adapter_path: str | Path | None) -> tuple[Any, Any]:
        llm_cfg = self.config.get("llm", {})
        model_path = llm_cfg.get("model_path")
        if not isinstance(model_path, str) or not model_path:
            raise ValueError("Missing llm.model_path in config")

        if self.backend == "vllm":
            return self._load_vllm_backend(model_path, adapter_path, llm_cfg)

        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=bool(llm_cfg.get("load_in_4bit", True)),
            bnb_4bit_quant_type=str(llm_cfg.get("bnb_4bit_quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(llm_cfg.get("bnb_4bit_use_double_quant", True)),
            bnb_4bit_compute_dtype=getattr(torch, str(llm_cfg.get("bnb_4bit_compute_dtype", "bfloat16")), torch.bfloat16),
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=llm_cfg.get("device_map", "auto"),
            torch_dtype=getattr(torch, str(llm_cfg.get("torch_dtype", "bfloat16")), torch.bfloat16),
            quantization_config=quant_cfg,
            use_safetensors=bool(llm_cfg.get("use_safetensors", True)),
            trust_remote_code=True,
        )

        if adapter_path:
            adapter_dir = str(adapter_path)
            logger.info("Loading QLoRA adapter from %s", adapter_dir)
            model = PeftModel.from_pretrained(model, adapter_dir)

        model.eval()
        return model, tokenizer

    def _load_vllm_backend(
        self,
        model_path: str,
        adapter_path: str | Path | None,
        llm_cfg: dict[str, Any],
    ) -> tuple[Any, Any]:
        if not _VLLM_AVAILABLE:
            raise ImportError("vLLM is not installed. pip install vllm")

        from vllm import LLM as _LLM, SamplingParams as _SP
        from vllm.lora.request import LoRARequest as _LR

        gen_cfg = self.config.get("generation", {})
        max_lora_rank = int(llm_cfg.get("max_lora_rank", 64))

        llm_kwargs: dict[str, Any] = {
            "model": model_path,
            "enable_prefix_caching": True,
            "gpu_memory_utilization": float(llm_cfg.get("gpu_memory_utilization", 0.85)),
            "max_model_len": int(llm_cfg.get("max_model_len", 4096)),
            "trust_remote_code": True,
            "dtype": "bfloat16",
        }

        lora_request = None
        if adapter_path:
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_lora_rank"] = max_lora_rank
            lora_request = _LR("qlora_adapter", 1, str(adapter_path))
            logger.info("vLLM: LoRA adapter from %s (rank≤%d)", adapter_path, max_lora_rank)

        logger.info("vLLM: loading %s with APC enabled", model_path)
        llm = _LLM(**llm_kwargs)

        sampling_params = _SP(
            temperature=float(gen_cfg.get("temperature", 0.7)),
            max_tokens=int(gen_cfg.get("max_new_tokens", 1024)),
            top_p=float(gen_cfg.get("top_p", 0.8)),
            top_k=int(gen_cfg.get("top_k", 20)),
        )

        wrapper = VLLMWrapper(
            llm=llm,
            sampling_params=sampling_params,
            lora_request=lora_request,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return wrapper, tokenizer

    def _load_tool_adapter(self, mcp_base_url: str) -> Any:
        adapter_mod = importlib.import_module("tool_calling_adapter")
        return adapter_mod.ToolCallingAdapter(
            schema_dir=Path("/workspace/trainer/configs/tool_schemas"),
            protocol_path=Path("/workspace/trainer/configs/tool_calling_protocol.json"),
            mcp_base_url=mcp_base_url,
        )

    def _load_rag_components(self) -> tuple[Any, list[dict[str, Any]], Any, Any]:
        search_mod = importlib.import_module("search_index")
        index = search_mod.load_faiss_index(
            self.config["paths"]["faiss_index"],
            int(self.config["index"]["ef_search"]),
        )
        metadata = search_mod.load_metadata(self.config["paths"]["metadata_json"])
        id_map = metadata["id_map"]
        embedder = search_mod.load_embedder(
            self.config["embedding"]["model_name"],
            self.config["embedding"]["device"],
            int(self.config["embedding"]["max_seq_length"]),
        )
        reranker = search_mod.load_reranker(
            self.config["reranker"]["model_name"],
            self.config["reranker"]["device"],
            int(self.config["reranker"]["max_length"]),
        )
        return index, id_map, embedder, reranker

    @staticmethod
    def _strip_think_blocks(text: str) -> tuple[str, str]:
        thinking_parts = [m.group(1).strip() for m in THINK_BLOCK_REGEX.finditer(text)]
        stripped = THINK_BLOCK_REGEX.sub("", text).strip()
        thinking = "\n\n".join(part for part in thinking_parts if part)
        return stripped, thinking

    @staticmethod
    def _extract_tool_calls(
        messages: list[dict[str, Any]],
        tool_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        parsed_calls: list[dict[str, Any]] = []
        for message in messages:
            if message.get("role") != "assistant":
                continue
            content = str(message.get("content", ""))
            for match in TOOL_CALL_BLOCK_REGEX.finditer(content):
                try:
                    payload = json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                name = payload.get("name")
                arguments = payload.get("arguments", {})
                if isinstance(name, str) and isinstance(arguments, dict):
                    parsed_calls.append({"name": name, "arguments": arguments})

        calls: list[dict[str, Any]] = []
        for idx, parsed in enumerate(parsed_calls):
            result = tool_results[idx] if idx < len(tool_results) else {}
            calls.append(
                {
                    "name": parsed["name"],
                    "arguments": parsed["arguments"],
                    "result": result,
                }
            )
        return calls

    @staticmethod
    def _iter_json_blocks(text: str) -> list[str]:
        blocks: list[str] = []
        stack: list[int] = []
        start = -1
        for idx, char in enumerate(text):
            if char == "{":
                if not stack:
                    start = idx
                stack.append(idx)
            elif char == "}" and stack:
                stack.pop()
                if not stack and start != -1:
                    blocks.append(text[start : idx + 1])
                    start = -1
        return blocks

    @staticmethod
    def _detect_xr_content(response_text: str) -> tuple[str, dict[str, Any] | None]:
        for block in AgentPipeline._iter_json_blocks(response_text):
            try:
                payload = json.loads(block)
            except json.JSONDecodeError:
                continue

            if isinstance(payload, dict) and XR_REQUIRED_KEYS.intersection(payload.keys()):
                text_only = response_text.replace(block, "", 1).strip()
                if not text_only:
                    text_only = "XR 가이드를 생성했습니다."
                return text_only, payload
        return response_text.strip(), None

    def _build_messages(
        self,
        instruction: str,
        environment: dict[str, Any] | None,
        rag_chunks: list[dict[str, Any]],
        session_context_summary: str,
    ) -> list[dict[str, str]]:
        system_prompt = str(self.config.get("prompt", {}).get("system", ""))
        context_template = str(
            self.config.get("prompt", {}).get(
                "context_template",
                "아래는 관련 문서입니다. 이 문서만을 근거로 답변하세요.\n\n{context}",
            )
        )

        # Token budget: reserve space for generation output within max_model_len
        max_model_len = int(self.config.get("llm", {}).get("max_model_len", 4096))
        max_gen_tokens = int(self.config.get("generation", {}).get("max_new_tokens", 768))
        prompt_budget = max_model_len - max_gen_tokens - 64  # 64 tokens for chat template overhead

        context_lines: list[str] = []
        for chunk in rag_chunks:
            chunk_id = chunk.get("chunk_id", "")
            score = chunk.get("rerank_score")
            score_text = f" (score={score:.4f})" if isinstance(score, (int, float)) else ""
            context_lines.append(f"[{chunk_id}]{score_text}\n{chunk.get('text', '')}")

        rag_context = context_template.format(context="\n\n".join(context_lines)) if context_lines else ""

        user_parts: list[str] = []
        if session_context_summary:
            user_parts.append("[세션 이전 요약]\n" + session_context_summary)
        if rag_context:
            user_parts.append("[RAG 문맥]\n" + rag_context)
        if environment:
            user_parts.append("[환경 데이터]\n" + json.dumps(environment, ensure_ascii=False, indent=2))
        user_parts.append("[사용자 요청]\n" + instruction)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ]

        # Truncate RAG context if prompt exceeds budget
        total_text = system_prompt + "\n\n".join(user_parts)
        estimated_tokens = len(total_text) // 2  # rough estimate: 1 token ≈ 2 chars for multilingual
        if estimated_tokens > prompt_budget and context_lines:
            # Progressively remove context chunks until within budget
            while context_lines and estimated_tokens > prompt_budget:
                context_lines.pop()
                rag_context = context_template.format(context="\n\n".join(context_lines)) if context_lines else ""
                user_parts_trimmed = []
                if session_context_summary:
                    user_parts_trimmed.append("[세션 이전 요약]\n" + session_context_summary)
                if rag_context:
                    user_parts_trimmed.append("[RAG 문맥]\n" + rag_context)
                if environment:
                    user_parts_trimmed.append("[환경 데이터]\n" + json.dumps(environment, ensure_ascii=False, indent=2))
                user_parts_trimmed.append("[사용자 요청]\n" + instruction)
                total_text = system_prompt + "\n\n".join(user_parts_trimmed)
                estimated_tokens = len(total_text) // 2
                messages[1]["content"] = "\n\n".join(user_parts_trimmed)
            if not context_lines:
                logger.warning("All RAG context dropped to fit within token budget (%d)", prompt_budget)
            else:
                logger.info("Trimmed to %d context chunks to fit token budget", len(context_lines))

        return messages

    def _generate_direct(
        self,
        messages: list[dict[str, str]],
        enable_thinking: bool = False,
    ) -> str:
        if isinstance(self.model, VLLMWrapper):
            return self.model.generate_response(messages=messages, tools=None)

        gen_cfg = self.config.get("generation", {})
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        inputs = self.tokenizer([prompt_text], return_tensors="pt", return_attention_mask=True)
        if hasattr(inputs, "to") and hasattr(self.model, "device"):
            inputs = inputs.to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=int(gen_cfg.get("max_new_tokens", 768)),
                do_sample=bool(gen_cfg.get("do_sample", True)),
                temperature=float(gen_cfg.get("temperature", 0.7)),
                top_p=float(gen_cfg.get("top_p", 0.8)),
                top_k=int(gen_cfg.get("top_k", 20)),
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def run(
        self,
        instruction: str,
        environment: dict[str, Any] | None = None,
        session_id: str | None = None,
        enable_thinking: bool = True,
        force_route: str | None = None,
    ) -> dict[str, Any]:
        if not instruction.strip():
            raise ValueError("instruction must not be empty")

        search_mod = importlib.import_module("search_index")
        active_session_id = session_id or str(uuid.uuid4())

        route = force_route or self.router.classify(instruction, environment)
        logger.info("Router: %s → %s", instruction[:60], route)

        rag_results, rag_timing = search_mod.search_and_rerank(
            instruction,
            self.index,
            self.id_map,
            self.embedder,
            self.reranker,
            self.config,
        )
        rag_chunks = [
            {
                "chunk_id": r.get("chunk_id", ""),
                "text": r.get("text", ""),
                "rerank_score": r.get("rerank_score"),
            }
            for r in rag_results
        ]

        summary = self.session_context.summarize_previous(active_session_id)

        # Adaptive compression: only compress when full context exceeds budget
        max_model_len = int(self.config.get("llm", {}).get("max_model_len", 4096))
        max_gen = int(self.config.get("generation", {}).get("max_new_tokens", 768))
        token_budget = max_model_len - max_gen - 64
        full_text_len = sum(len(c.get("text", "")) for c in rag_chunks)
        estimated_total = full_text_len // 2  # rough: 1 token ≈ 2 chars for multilingual
        if estimated_total > token_budget:
            evidence = compress_evidence(instruction, rag_chunks)
            logger.info("Compressed evidence: %d chunks, budget=%d", len(evidence), token_budget)
        else:
            evidence = rag_chunks

        if route == "fast":
            messages = self._build_messages(
                instruction=instruction,
                environment=None,
                rag_chunks=evidence,
                session_context_summary=summary,
            )
            raw_response = self._generate_direct(messages, enable_thinking=False)
            final_no_think, thinking = self._strip_think_blocks(raw_response)
            response_text = final_no_think.strip()
            tool_calls: list[dict[str, Any]] = []
            xr_content = None
            rounds = 0
        else:
            messages = self._build_messages(
                instruction=instruction,
                environment=environment,
                rag_chunks=evidence,
                session_context_summary=summary,
            )
            logger.debug("Running tool loop with %d initial messages", len(messages))
            loop_output = _run_coro_sync(
                self.tool_adapter.run_tool_loop(
                    messages=messages,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    enable_thinking=enable_thinking,
                    max_rounds=5,
                )
            )
            raw_final = str(loop_output.get("final_response", ""))
            final_no_think, thinking = self._strip_think_blocks(raw_final)
            response_text, xr_content = self._detect_xr_content(final_no_think)
            tool_calls = self._extract_tool_calls(
                loop_output.get("messages", []),
                loop_output.get("tool_results", []),
            )
            rounds = int(loop_output.get("rounds", 0))

        result = {
            "session_id": active_session_id,
            "instruction": instruction,
            "response": response_text,
            "xr_content": xr_content,
            "tool_calls": tool_calls,
            "rag_chunks": rag_chunks,
            "thinking": thinking,
            "rounds": rounds,
            "environment": environment or {},
            "timing": rag_timing,
            "route": route,
        }

        self.session_context.update(
            active_session_id,
            {
                "instruction": instruction,
                "response": response_text,
                "tool_calls": tool_calls,
            },
        )
        return result


def _load_qlora_adapter_from_config(qlora_config_path: Path) -> str | None:
    if not qlora_config_path.is_file():
        return None
    with qlora_config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    adapter = cfg.get("output", {}).get("adapter_dir")
    if isinstance(adapter, str) and adapter:
        return adapter
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end agent orchestration pipeline.")
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
        help="Optional QLoRA config for default adapter lookup.",
    )
    parser.add_argument(
        "--mcp-url",
        type=str,
        default="http://localhost:8100",
        help="MCP server base URL.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        required=True,
        help="Instruction to run through the pipeline.",
    )
    parser.add_argument(
        "--environment",
        type=str,
        default="{}",
        help="JSON string for environment context.",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Optional multi-turn session ID.",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Disable Qwen3 thinking mode.",
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

    if not args.config.is_file():
        logger.error("Config not found: %s", args.config)
        return 1

    adapter_path = args.adapter
    if adapter_path is None:
        configured_adapter = _load_qlora_adapter_from_config(args.qlora_config)
        if configured_adapter:
            adapter_path = Path(configured_adapter)

    environment: dict[str, Any]
    try:
        parsed_env = json.loads(args.environment)
        if not isinstance(parsed_env, dict):
            logger.error("--environment must be a JSON object")
            return 1
        environment = parsed_env
    except json.JSONDecodeError:
        logger.error("Invalid --environment JSON")
        return 1

    try:
        pipeline = AgentPipeline(
            config_path=args.config,
            adapter_path=adapter_path,
            mcp_base_url=args.mcp_url,
        )
        result = pipeline.run(
            instruction=args.instruction,
            environment=environment,
            session_id=args.session_id,
            enable_thinking=not args.disable_thinking,
        )
    except Exception:
        logger.exception("Agent pipeline execution failed")
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
