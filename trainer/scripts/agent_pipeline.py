#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import contextvars
import importlib
import json
import logging
import os
import re
import sys
import threading
import uuid
from collections.abc import AsyncIterator, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Final

# vLLM requires 'spawn' multiprocessing when CUDA is already initialized
# (e.g., by embedding models). Must be set before any vLLM import.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

import time
from dataclasses import dataclass, field

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 스크립트를 경로로 실행하면 sys.path[0]이 trainer/scripts라 `agent_pipeline`으로 불리고 루트가 없다.
if __package__ in {None, ""} and str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trainer.scripts.live_response_contract import parse_knowledge_answer, retrieved_chunk_ids
from trainer.scripts.live_sequence_fallback import (
    LiveSequenceRequest,
    enforce_live_sequence_validation,
)
from trainer.scripts.generation_slot import GENERATION_SLOT
from trainer.scripts.openai_backend import (
    ContextLengthExceededError,
    GenerationCancelledError,
    OpenAIBackendSettings,
    OpenAIChatBackend,
    generation_cancel_scope,
)
from trainer.scripts.prompt_budget import (
    PromptTokenCounter,
    context_window,
    load_token_counter,
    prompt_token_budget,
    token_counter_for,
)

try:
    import vllm as _vllm_check  # noqa: F401
    _vllm_available = True
    del _vllm_check
except ImportError:
    _vllm_available = False

logger = logging.getLogger(__name__)

THINK_BLOCK_REGEX = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL)
TOOL_CALL_BLOCK_REGEX = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", flags=re.DOTALL)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.?!。\n])\s+")
_WORD_RE = re.compile(r"\w+")
_DOC_ID_RE = re.compile(r"DOC-[a-zA-Z0-9_-]+-C\d+")

# Skill level → detail level mapping for text guide personalization
_SKILL_DETAIL_MAP: dict[str, str] = {
    "beginner": "detailed",
    "intermediate": "standard",
    "advanced": "minimal",
}


@dataclass
class TraineeProfile:
    """Trainee metadata from simulator request."""
    trainee_id: str = ""
    language: str = "ko"
    skill_level: str = "beginner"

    @property
    def detail_level(self) -> str:
        return _SKILL_DETAIL_MAP.get(self.skill_level, "standard")


# ---------------------------------------------------------------------------
# Semantic Router — keyword-based 3-way classification
# Routes: training_realtime | training_review | fast
#   training_realtime: 순서/이상탐지 (< 60s, max_rounds=2)
#   training_review: 자세/분포분석 (no time limit, max_rounds=5)
#   fast: simple QA, no tools (single LLM round)
# ---------------------------------------------------------------------------

# Training-realtime keywords: sequence, anomaly, DFT judgment — must respond < 60s
_TRAINING_REALTIME_KEYWORDS: set[str] = {
    "순서", "다음", "칠하", "깎아", "순번",
    "이상", "탐지", "anomaly", "과도막", "저도막",
    "합격", "불합격", "판정", "합부",
    "dft", "도막두께", "측정", "수치",
    "실시간", "현재", "지금",
}

# Training-review keywords: posture, distribution, coaching — no strict time limit
_TRAINING_REVIEW_KEYWORDS: set[str] = {
    "자세", "포즈", "숙련자", "교정", "pose",
    "분포", "분석", "원인", "추이", "시계열", "timeseries",
    "코칭", "피드백", "세션", "리포트",
    "위치", "좌표", "궤적", "블록", "구역", "location",
}

# General deep keywords that don't fall into training_realtime or training_review
_DEEP_KEYWORDS: set[str] = {
    "센서", "데이터", "상태", "확인해", "어때",
    "보여줘", "가이드", "xr", "시연", "시각화",
    "비교", "점검", "검사", "진단",
}

# Route → max_rounds mapping
ROUTE_MAX_ROUNDS: dict[str, int] = {
    "training_realtime": 2,
    "training_review": 5,
    "fast": 1,
}

ROUTE_MAX_TOKENS: dict[str, int] = {
    "training_realtime": 256,
    "training_review": 1024,
    "fast": 256,
}

_PROGRESS_INTERVAL_SEC: Final[float] = 3.0

# Acquired inside the worker thread, so async cancellation cannot free the slot
# while an orphaned generation is still occupying the GPU. Shared with
# POST /api/v1/copilot/query through trainer.scripts.generation_slot.
_GENERATION_SLOT: Final[threading.Semaphore] = GENERATION_SLOT


class SemanticRouter:
    """Classifies questions into training_realtime, training_review, or fast.

    - training_realtime: 순서/이상탐지 tools, max_rounds=2, target < 60s
    - training_review: 자세/분포분석 tools, max_rounds=5, no time limit
    - fast: docQA only, no tool loop
    """

    def __init__(
        self,
        realtime_keywords: set[str] | None = None,
        review_keywords: set[str] | None = None,
        deep_keywords: set[str] | None = None,
    ) -> None:
        self.realtime_keywords = realtime_keywords or _TRAINING_REALTIME_KEYWORDS
        self.review_keywords = review_keywords or _TRAINING_REVIEW_KEYWORDS
        self.deep_keywords = deep_keywords or _DEEP_KEYWORDS

    def classify(self, instruction: str, environment: dict[str, Any] | None = None) -> str:
        """Return route: 'training_realtime', 'training_review', or 'fast'."""
        text_lower = instruction.lower()

        # Environment data always implies tool-needing deep path
        if environment and len(environment) > 0:
            # Check if environment relates to realtime or review
            realtime_score = sum(1 for kw in self.realtime_keywords if kw.lower() in text_lower)
            review_score = sum(1 for kw in self.review_keywords if kw.lower() in text_lower)
            if realtime_score >= review_score:
                return "training_realtime"
            return "training_review"

        # Score each route by keyword matches
        realtime_score = sum(1 for kw in self.realtime_keywords if kw.lower() in text_lower)
        review_score = sum(1 for kw in self.review_keywords if kw.lower() in text_lower)
        deep_score = sum(1 for kw in self.deep_keywords if kw.lower() in text_lower)

        # Highest scoring route wins; ties go to realtime (faster response needed)
        if realtime_score > 0 and realtime_score >= review_score and realtime_score >= deep_score:
            return "training_realtime"
        if review_score > 0 and review_score >= deep_score:
            return "training_review"
        if deep_score > 0:
            return "training_review"  # general deep → review (lenient timing)
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


def _drain_orphaned_run(task: asyncio.Future[Any]) -> None:
    """Retrieves the outcome of a run whose consumer left, so it is logged once, not as 'never retrieved'."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None and not isinstance(exc, GenerationCancelledError):
        logger.warning("run_streaming: abandoned run() ended with %s", type(exc).__name__)


def _run_coro_sync(coro: Any) -> Any:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    if not loop.is_running():
        return loop.run_until_complete(coro)

    with ThreadPoolExecutor(max_workers=1) as executor:
        # Carry context variables (the generation cancel scope) into the helper thread.
        future = executor.submit(contextvars.copy_context().run, asyncio.run, coro)
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
    """Makes vLLM LLM compatible with tool_calling_adapter's generate_response interface.

    Known defects (in-process ``llm.backend: vllm``, kept only for reproduction):
    it passes native ``tools`` even with ``pass_native_tools: false``, and it
    drops ``enable_thinking``, per-round ``max_tokens`` and
    ``repetition_penalty``. Serve through ``llm.backend: openai``
    (``trainer/scripts/openai_backend.py``) instead.
    """

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
        if self.backend == "openai":
            return self._load_openai_backend(adapter_path)

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
            dtype=getattr(torch, str(llm_cfg.get("torch_dtype", "bfloat16")), torch.bfloat16),
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

    def _load_openai_backend(self, adapter_path: str | Path | None) -> tuple[Any, Any]:
        """HTTP client for an external OpenAI-compatible server; no local weights, no tokenizer model.

        The served checkpoint already has the LoRA merged, so ``adapter_path``
        is ignored. Prompt accounting loads only ``llm.tokenizer_path``
        (see ``_prompt_token_counter``).
        """
        settings = OpenAIBackendSettings.from_config(self.config)
        if adapter_path:
            logger.info("openai backend: adapter %s ignored (served model is merged)", adapter_path)
        logger.info("openai backend: model=%s", settings.served_model_name)
        return OpenAIChatBackend(settings), None

    def _load_vllm_backend(
        self,
        model_path: str,
        adapter_path: str | Path | None,
        llm_cfg: dict[str, Any],
    ) -> tuple[Any, Any]:
        if not _vllm_available:
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
        llm_cfg = self.config.get("llm", {})
        return adapter_mod.ToolCallingAdapter(
            schema_dir=Path("/workspace/trainer/configs/tool_schemas"),
            protocol_path=Path("/workspace/trainer/configs/tool_calling_protocol.json"),
            mcp_base_url=mcp_base_url,
            pass_native_tools=bool(llm_cfg.get("pass_native_tools", True)),
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
        adapter_mod = importlib.import_module("tool_calling_adapter")

        parsed_calls: list[dict[str, Any]] = []
        for message in messages:
            if message.get("role") != "assistant":
                continue
            content = str(message.get("content", ""))
            for block in adapter_mod.ToolCallingAdapter._extract_tool_call_blocks(content):
                try:
                    payload = json.loads(block)
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
    def _assemble_tool_sections(
        tool_calls: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Map MCP tool results into dedicated response sections."""
        sections: dict[str, Any] = {
            "pose_comparison": None,
            "next_path": None,
            "dft_result": None,
            "anomaly": None,
        }
        for call in tool_calls:
            name = call.get("name", "")
            envelope = call.get("result", {})
            if not isinstance(envelope, dict) or envelope.get("status") == "error":
                continue

            # Unwrap MCP envelope: business payload is nested under "result" (common.py success_response).
            result = envelope.get("result")
            if not isinstance(result, dict):
                result = {}

            if name == "pose_estimation":
                sections["pose_comparison"] = {
                    "similarity_score": result.get("similarity_score"),
                    "corrections": result.get("corrections", result.get("joint_deviations", [])),
                }
            elif name == "sequence_validation":
                sections["next_path"] = {
                    "zone_id": result.get("zone_id", ""),
                    "direction": result.get("recommended_direction", result.get("reference_pattern", "")),
                    "reference_pattern": result.get("reference_pattern", ""),
                    "waypoints": result.get("waypoints", []),
                    "violation": result.get("violation", result.get("sequence_violation", False)),
                }
            elif name == "dft_judge":
                sections["dft_result"] = {
                    "verdict": result.get("verdict", ""),
                    "nominal_um": result.get("nominal", result.get("spec", {}).get("nominal")),
                    "measurements_um": result.get("measurements", []),
                    "mean_um": result.get("mean"),
                    "rule": result.get("rule", "90/10"),
                    "out_of_spec_count": result.get("out_of_spec_count", 0),
                }
            elif name == "anomaly_detection":
                sections["anomaly"] = {
                    "detected": result.get("detected", result.get("anomaly_detected", False)),
                    "type": result.get("anomaly_type", ""),
                    "zone": result.get("zone", result.get("affected_zone", "")),
                    "cause": result.get("cause", result.get("root_cause", "")),
                    "severity": result.get("severity", "warning"),
                }
        return sections

    @staticmethod
    def _extract_citations(text: str, rag_chunks: list[dict[str, Any]]) -> list[str]:
        """Extract DOC-xxx-Cxxx citations from text and RAG chunk IDs."""
        citations = list(dict.fromkeys(_DOC_ID_RE.findall(text)))
        for chunk in rag_chunks:
            cid = chunk.get("chunk_id", "")
            if cid and cid not in citations:
                citations.append(cid)
        return citations

    @staticmethod
    def _build_text_guide(
        response_text: str,
        trainee: "TraineeProfile",
    ) -> dict[str, Any]:
        """Build text_guide section with personalization."""
        lines = response_text.strip().split("\n")
        summary = lines[0][:200] if lines else response_text[:200]
        return {
            "language": trainee.language,
            "detail_level": trainee.detail_level,
            "summary": summary,
            "body": response_text,
        }

    @staticmethod
    def _append_citations(response_text: str, citations: list[str]) -> str:
        if not citations:
            return response_text
        if _DOC_ID_RE.search(response_text):
            return response_text
        cited = ", ".join(citations[:3])
        return f"{response_text.rstrip()}\n[근거] 참조 문서 ID: {cited}"

    @staticmethod
    def _extract_steps(response_text: str) -> list[dict[str, Any]] | None:
        """Try to extract structured steps from response JSON blocks."""
        for block_str in AgentPipeline._iter_json_blocks(response_text):
            try:
                payload = json.loads(block_str)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and "steps" in payload:
                steps = payload["steps"]
                if isinstance(steps, list):
                    return steps
        return None

    def _build_messages(
        self,
        instruction: str,
        environment: dict[str, Any] | None,
        rag_chunks: list[dict[str, Any]],
        session_context_summary: str,
        *,
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, str]]:
        if system_prompt is None:
            system_prompt = str(self.config.get("prompt", {}).get("system", ""))
        context_template = str(
            self.config.get("prompt", {}).get(
                "context_template",
                "아래는 관련 문서입니다. 이 문서만을 근거로 답변하세요.\n\n{context}",
            )
        )

        # The rendered prompt must leave generation.max_new_tokens inside
        # llm.max_model_len; the count covers the chat template, not just text.
        counter = self._prompt_token_counter()
        prompt_budget = prompt_token_budget(self.config)

        context_lines: list[str] = []
        for chunk in rag_chunks:
            chunk_id = chunk.get("chunk_id", "")
            score = chunk.get("rerank_score")
            score_text = f" (score={score:.4f})" if isinstance(score, (int, float)) else ""
            context_lines.append(f"[{chunk_id}]{score_text}\n{chunk.get('text', '')}")

        def _messages_with(lines: list[str]) -> list[dict[str, str]]:
            rag_context = context_template.format(context="\n\n".join(lines)) if lines else ""
            user_parts: list[str] = []
            if session_context_summary:
                user_parts.append("[세션 이전 요약]\n" + session_context_summary)
            if rag_context:
                user_parts.append("[RAG 문맥]\n" + rag_context)
            if environment:
                user_parts.append("[환경 데이터]\n" + json.dumps(environment, ensure_ascii=False, indent=2))
            user_parts.append("[사용자 요청]\n" + instruction)
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "\n\n".join(user_parts)},
            ]

        messages = _messages_with(context_lines)
        prompt_tokens = counter.count_messages(messages, tools=tools)
        if prompt_tokens > prompt_budget and context_lines:
            # Drop the lowest-ranked chunks first until the prompt fits.
            while context_lines and prompt_tokens > prompt_budget:
                context_lines.pop()
                messages = _messages_with(context_lines)
                prompt_tokens = counter.count_messages(messages, tools=tools)
            if not context_lines:
                logger.warning("All RAG context dropped to fit within token budget (%d)", prompt_budget)
            else:
                logger.info("Trimmed to %d context chunks to fit token budget", len(context_lines))
        if prompt_tokens > prompt_budget:
            logger.warning(
                "Prompt is %d tokens with no RAG context left; budget is %d (exact=%s)",
                prompt_tokens,
                prompt_budget,
                counter.exact,
            )

        return messages

    def _prompt_token_counter(self) -> PromptTokenCounter:
        """Exact counter from the loaded tokenizer; else ``llm.tokenizer_path``; else the estimate."""
        cached = getattr(self, "_token_counter", None)
        if cached is not None:
            return cached
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is not None:
            counter = token_counter_for(tokenizer)
        else:
            counter = load_token_counter(self.config.get("llm", {}).get("tokenizer_path"))
        self._token_counter = counter
        return counter

    def _generate_direct(
        self,
        messages: list[dict[str, str]],
        enable_thinking: bool = False,
    ) -> str:
        if isinstance(self.model, VLLMWrapper):
            return self.model.generate_response(messages=messages, tools=None)

        gen_cfg = self.config.get("generation", {})
        if isinstance(self.model, OpenAIChatBackend):
            return self.model.complete_chat(
                messages,
                max_tokens=min(int(gen_cfg.get("max_new_tokens", 768)), ROUTE_MAX_TOKENS["fast"]),
                enable_thinking=enable_thinking,
            )

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
                max_new_tokens=min(int(gen_cfg.get("max_new_tokens", 768)), ROUTE_MAX_TOKENS["fast"]),
                do_sample=bool(gen_cfg.get("do_sample", True)),
                temperature=float(gen_cfg.get("temperature", 0.7)),
                top_p=float(gen_cfg.get("top_p", 0.8)),
                top_k=int(gen_cfg.get("top_k", 20)),
                repetition_penalty=float(gen_cfg.get("repetition_penalty", 1.15)),
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _stream_generate(
        self,
        prompt_text: str,
        gen_cfg: dict[str, Any],
        max_new_tokens: int,
    ) -> Iterator[str]:
        """Yields decoded text chunks as the model produces them.

        ``model.generate`` runs on a worker thread. If it raises, the worker
        records the error and enqueues the streamer stop signal so the consumer
        loop terminates instead of blocking forever on the unbounded
        ``queue.get`` (TextIteratorStreamer uses ``timeout=None``). The worker
        is always joined, then the captured exception is re-raised to the caller.
        """
        from threading import Thread

        from transformers import TextIteratorStreamer

        inputs = self.tokenizer([prompt_text], return_tensors="pt", return_attention_mask=True)
        if hasattr(inputs, "to") and hasattr(self.model, "device"):
            inputs = inputs.to(self.model.device)

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        do_sample = bool(gen_cfg.get("do_sample", True))
        kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": float(gen_cfg.get("repetition_penalty", 1.15)),
            "do_sample": do_sample,
            "streamer": streamer,
        }
        if do_sample:
            kwargs["temperature"] = float(gen_cfg.get("temperature", 0.3))
            kwargs["top_p"] = float(gen_cfg.get("top_p", 0.8))
            kwargs["top_k"] = int(gen_cfg.get("top_k", 20))

        worker_error: list[Exception] = []

        def _run_generation() -> None:
            try:
                self.model.generate(**kwargs)
            except Exception as exc:
                worker_error.append(exc)
                streamer.on_finalized_text("", stream_end=True)

        thread = Thread(target=_run_generation, daemon=True)
        thread.start()
        try:
            for chunk in streamer:
                if chunk:
                    yield chunk
        finally:
            thread.join()
        if worker_error:
            raise worker_error[0]

    async def run_streaming(
        self,
        instruction: str,
        environment: dict[str, Any] | None = None,
        session_id: str | None = None,
        enable_thinking: bool = False,
        request_id: str | None = None,
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """Yields (kind, payload) where kind is status|tool_result|delta|complete|error.

        Staged buffered streaming: this awaits the full non-streaming run() in a
        worker thread, then re-emits the finished answer as fixed-width (24-char)
        compatibility chunks so callers can consume a uniform delta stream today.
        It does NOT yet reduce time-to-first-token; true token-level streaming
        via _stream_generate is deferred to a later task. While run() is in
        flight, a ``generating`` status carrying elapsed seconds is emitted every
        _PROGRESS_INTERVAL_SEC so callers can show progress, and _GENERATION_SLOT
        keeps concurrent runs from competing for the GPU.

        On failure no partial events are emitted: a blank instruction is rejected
        up front as a single non-retryable error without invoking run(), and any
        exception escaping run() is logged and surfaced as a single generic
        retryable error without leaking its message. A prompt that exceeds the
        model context window is the one non-retryable run() failure. Every error
        event carries ttft_ms=None and a measured total_ms.

        If the consumer leaves (aclose or task cancellation) while run() is in
        flight, a cancel event installed in the worker's context stops the
        remote (openai) backend between streamed tokens and closes its request;
        a run still waiting for _GENERATION_SLOT is skipped. The HF backend
        cannot be interrupted and finishes in the background.
        """
        t0 = time.monotonic()
        ttft_ms: int | None = None

        if not instruction.strip():
            yield "error", {
                "message": "instruction must not be empty",
                "retryable": False,
                "ttft_ms": None,
                "total_ms": int((time.monotonic() - t0) * 1000),
            }
            return

        # Ambient context rides on every live turn, so passing it here would make
        # classify() read tool intent from state; run() keeps its rule for evaluation.
        live_route = self.router.classify(instruction, None)
        cancel_event = threading.Event()

        def _guarded_run() -> dict[str, Any]:
            with _GENERATION_SLOT:
                if cancel_event.is_set():
                    raise GenerationCancelledError("consumer left before generation started")
                # Runs in the worker's copied context, so the scope never leaks to the caller.
                with generation_cancel_scope(cancel_event):
                    return self.run(
                        instruction=instruction,
                        environment=environment,
                        session_id=session_id,
                        enable_thinking=enable_thinking,
                        request_id=request_id,
                        force_route=live_route,
                    )

        run_task = asyncio.ensure_future(asyncio.to_thread(_guarded_run))
        try:
            while True:
                finished, _pending = await asyncio.wait({run_task}, timeout=_PROGRESS_INTERVAL_SEC)
                if finished:
                    break
                yield "status", {
                    "stage": "generating",
                    "elapsed_sec": int(time.monotonic() - t0),
                }
            result = run_task.result()
        except ContextLengthExceededError:
            logger.exception("run_streaming: prompt exceeds the model context window")
            yield "error", {
                "message": "prompt exceeds the model context window",
                "retryable": False,
                "ttft_ms": None,
                "total_ms": int((time.monotonic() - t0) * 1000),
            }
            return
        except Exception:
            logger.exception("run_streaming: unexpected failure in run()")
            yield "error", {
                "message": "streaming pipeline failed",
                "retryable": True,
                "ttft_ms": None,
                "total_ms": int((time.monotonic() - t0) * 1000),
            }
            return
        finally:
            if not run_task.done():
                logger.info("run_streaming: consumer left; cancelling generation session=%s", session_id)
                cancel_event.set()
                run_task.add_done_callback(_drain_orphaned_run)

        logger.info(
            "run_streaming: run() finished session=%s route=%s elapsed_ms=%d",
            session_id,
            result.get("route", ""),
            int((time.monotonic() - t0) * 1000),
        )

        sequence_result = await enforce_live_sequence_validation(
            LiveSequenceRequest(
                instruction=instruction,
                route=str(result.get("route", "")),
                tool_calls=list(result.get("tool_calls", [])),
                response=str(result.get("response", "")),
                environment=environment,
                session_id=session_id,
            ),
            getattr(self, "tool_adapter", None),
        )
        tool_calls = sequence_result.tool_calls
        text = sequence_result.response

        yield "status", {"stage": "rag", "chunks": len(result.get("rag_chunks", []))}

        if result.get("deadline_hit"):
            yield "status", {
                "stage": "deadline_forced_synthesis",
                "elapsed_sec": int((time.monotonic() - t0)),
            }

        for call in tool_calls:
            yield "tool_result", {
                "name": call.get("name", ""),
                "arguments": call.get("arguments", {}) or {},
                "result": call.get("result", {}) or {},
            }

        if sequence_result.error_retryable is not None:
            yield "error", {
                "message": "sequence validation failed",
                "retryable": sequence_result.error_retryable,
                "ttft_ms": None,
                "total_ms": int((time.monotonic() - t0) * 1000),
            }
            return

        if str(result.get("route", "")) == "fast" and not tool_calls:
            # Citations are reconciled against what this request actually retrieved, so an
            # id left in the streamed text is one the model was shown.
            text = parse_knowledge_answer(text, retrieved_chunk_ids(result.get("rag_chunks", []))).answer

        for index in range(0, len(text), 24):
            if ttft_ms is None:
                ttft_ms = int((time.monotonic() - t0) * 1000)
            yield "delta", {"text": text[index : index + 24]}

        yield "complete", {
            "ttft_ms": ttft_ms,
            "synthesis_ms": int(result.get("synthesis_ms", 0)),
            "total_ms": int((time.monotonic() - t0) * 1000),
            "tool_call_count": len(tool_calls),
            "rag_chunk_count": len(result.get("rag_chunks", [])),
            "deadline_hit": bool(result.get("deadline_hit", False)),
        }

    def run(
        self,
        instruction: str,
        environment: dict[str, Any] | None = None,
        session_id: str | None = None,
        enable_thinking: bool = True,
        force_route: str | None = None,
        mode: str | None = None,
        trainee: dict[str, Any] | None = None,
        request_id: str | None = None,
        use_rag: bool = True,
    ) -> dict[str, Any]:
        """``use_rag=False`` skips retrieval: no RAG context, no chunk citations. The default is unchanged."""
        t0 = time.monotonic()
        if not instruction.strip():
            raise ValueError("instruction must not be empty")

        search_mod = importlib.import_module("search_index")
        active_request_id = request_id or str(uuid.uuid4())
        active_session_id = session_id or str(uuid.uuid4())
        trainee_profile = TraineeProfile(**(trainee or {}))

        route = force_route or mode or self.router.classify(instruction, environment)
        logger.info("Router: %s → %s", instruction[:60], route)

        if use_rag:
            rag_results, rag_timing = search_mod.search_and_rerank(
                instruction,
                self.index,
                self.id_map,
                self.embedder,
                self.reranker,
                self.config,
            )
        else:
            rag_results, rag_timing = [], {"total_ms": 0.0}
        rag_chunks = [
            {
                "chunk_id": r.get("chunk_id", ""),
                "text": r.get("text", ""),
                "rerank_score": r.get("rerank_score"),
            }
            for r in rag_results
        ]

        summary = self.session_context.summarize_previous(active_session_id)

        # Adaptive compression: only compress when the chunk text alone exceeds the prompt budget
        counter = self._prompt_token_counter()
        token_budget = prompt_token_budget(self.config)
        evidence_tokens = sum(counter.count_text(str(c.get("text", ""))) for c in rag_chunks)
        if evidence_tokens > token_budget:
            evidence = compress_evidence(instruction, rag_chunks)
            logger.info("Compressed evidence: %d chunks, budget=%d", len(evidence), token_budget)
        else:
            evidence = rag_chunks

        deadline_hit = False
        synthesis_ms = 0
        if route == "fast":
            messages = self._build_messages(
                instruction=instruction,
                environment=None,
                rag_chunks=evidence,
                session_context_summary=summary,
                system_prompt=str(self.config.get("prompt", {}).get("fast_system", "")).strip() or None,
            )
            fast_prompt_tokens = counter.count_messages(messages)
            fast_max_tokens = min(
                int(self.config.get("generation", {}).get("max_new_tokens", 768)), ROUTE_MAX_TOKENS["fast"]
            )
            max_model_len = context_window(self.config)
            context_report: dict[str, Any] | None = {
                "max_model_len": max_model_len,
                "exact": counter.exact,
                "calls": [
                    {"stage": "direct", "prompt_tokens": fast_prompt_tokens, "max_new_tokens": fast_max_tokens}
                ],
                "overflow": fast_prompt_tokens + fast_max_tokens > max_model_len,
            }
            raw_response = self._generate_direct(messages, enable_thinking=False)
            final_no_think, thinking = self._strip_think_blocks(raw_response)
            response_text = final_no_think.strip()
            tool_calls: list[dict[str, Any]] = []
            rounds = 0
        else:
            # The template renders native tool declarations into the prompt, so they count too.
            native_tools = None
            if bool(getattr(self.tool_adapter, "pass_native_tools", False)) or isinstance(self.model, VLLMWrapper):
                native_tools = self.tool_adapter.get_tools_param()
            messages = self._build_messages(
                instruction=instruction,
                environment=environment,
                rag_chunks=evidence,
                session_context_summary=summary,
                tools=native_tools,
            )
            logger.debug("Running tool loop with %d initial messages", len(messages))
            route_max_tokens = ROUTE_MAX_TOKENS.get(route, 768)
            tool_thinking = enable_thinking
            if bool(self.config.get("llm", {}).get("thinking_off_for_tools", False)):
                tool_thinking = False
            gen_cfg = self.config.get("generation", {})
            gen_params = {
                "do_sample": bool(gen_cfg.get("do_sample", True)),
                "temperature": float(gen_cfg.get("temperature", 0.3)),
                "top_p": float(gen_cfg.get("top_p", 0.8)),
                "top_k": int(gen_cfg.get("top_k", 20)),
                "repetition_penalty": float(gen_cfg.get("repetition_penalty", 1.15)),
            }
            loop_output = _run_coro_sync(
                self.tool_adapter.run_tool_loop(
                    messages=messages,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    enable_thinking=tool_thinking,
                    max_rounds=ROUTE_MAX_ROUNDS.get(route, 5),
                    max_tokens=route_max_tokens,
                    gen_params=gen_params,
                    deadline_sec=float(self.config.get("llm", {}).get("tool_deadline_sec", 150)),
                    tool_max_tokens=int(gen_cfg.get("max_new_tokens_tool", 256)),
                    token_counter=counter,
                    max_model_len=context_window(self.config),
                )
            )
            raw_final = str(loop_output.get("final_response", ""))
            final_no_think, thinking = self._strip_think_blocks(raw_final)
            response_text = final_no_think.strip()
            tool_calls = self._extract_tool_calls(
                loop_output.get("messages", []),
                loop_output.get("tool_results", []),
            )
            rounds = int(loop_output.get("rounds", 0))
            context_report = loop_output.get("context")
            deadline_hit = bool(loop_output.get("deadline_hit", False))
            synthesis_ms = int(loop_output.get("synthesis_ms", 0))

        latency_ms = int((time.monotonic() - t0) * 1000)
        tool_sections = self._assemble_tool_sections(tool_calls)
        citations = self._extract_citations(response_text, rag_chunks)
        if bool(self.config.get("llm", {}).get("append_citations", False)):
            response_text = self._append_citations(response_text, citations)
        text_guide = self._build_text_guide(response_text, trainee_profile)
        tools_used = list(dict.fromkeys(c.get("name", "") for c in tool_calls if c.get("name")))

        result = {
            "request_id": active_request_id,
            "session_id": active_session_id,
            "mode": route,
            "text_guide": text_guide,
            "pose_comparison": tool_sections["pose_comparison"],
            "next_path": tool_sections["next_path"],
            "dft_result": tool_sections["dft_result"],
            "anomaly": tool_sections["anomaly"],
            "steps": self._extract_steps(response_text),
            "citations": citations,
            "metadata": {
                "latency_ms": latency_ms,
                "tools_used": tools_used,
                "rounds": rounds,
                "route": route,
                "thinking": thinking,
                "context": context_report,
            },
            "response": response_text,
            "tool_calls": tool_calls,
            "rag_chunks": rag_chunks,
            "instruction": instruction,
            "environment": environment or {},
            "timing": rag_timing,
            "deadline_hit": deadline_hit,
            "synthesis_ms": synthesis_ms,
            "route": route,
            "xr_content": None,
            "_legacy_keys": [
                "response",
                "tool_calls",
                "rag_chunks",
                "instruction",
                "environment",
                "timing",
                "route",
                "xr_content",
            ],
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
        "--mode",
        type=str,
        default=None,
        choices=["training_realtime", "training_review"],
        help="Explicit mode override.",
    )
    parser.add_argument(
        "--trainee",
        type=str,
        default="{}",
        help="JSON string for trainee profile.",
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

    trainee_data: dict[str, Any] | None
    try:
        parsed_trainee = json.loads(args.trainee)
        if isinstance(parsed_trainee, dict):
            trainee_data = parsed_trainee
        else:
            trainee_data = None
    except json.JSONDecodeError:
        logger.error("Invalid --trainee JSON")
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
            mode=args.mode,
            trainee=trainee_data,
        )
    except Exception:
        logger.exception("Agent pipeline execution failed")
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
