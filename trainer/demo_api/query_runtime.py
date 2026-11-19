"""Opt-in quick generator; retrieval and persistent state stay on the shared pipeline."""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, replace
from typing import Any, Mapping

from trainer.scripts.generation_slot import GENERATION_SLOT, QUICK_GENERATION_SLOT


def safe_model_name(value: Any) -> str | None:
    """Only a short served-model identifier may cross the HTTP header boundary."""
    if (isinstance(value, str) and len(value) <= 128
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]*(?:/[A-Za-z0-9][A-Za-z0-9._:-]*)?", value)):
        return value
    return None


def query_model_name(pipeline: Any) -> str | None:
    settings = getattr(getattr(pipeline, "model", None), "settings", None)
    return safe_model_name(getattr(settings, "served_model_name", None))


@dataclass(frozen=True)
class QuickRuntimeConfig:
    base_url: str
    model_name: str
    tokenizer_path: str
    context_limit: int = 8192
    cache_enabled: bool = True

    @classmethod
    def from_environ(cls, env: Mapping[str, str]) -> QuickRuntimeConfig | None:
        keys = ("COPILOT_QUICK_BASE_URL", "COPILOT_QUICK_MODEL", "COPILOT_QUICK_TOKENIZER_PATH")
        if not any(env.get(k) for k in keys):
            return None
        if not all(env.get(k, "").strip() for k in keys):
            raise ValueError("Quick backend requires URL, model and tokenizer path together")
        limit = int(env.get("COPILOT_QUICK_CONTEXT_LIMIT", "8192"))
        if not 2048 <= limit <= 32768:
            raise ValueError("Quick context limit must be between2048 and32768 tokens")
        cache = env.get("COPILOT_QUICK_CACHE_ENABLED", "1")
        if cache not in ("0", "1"):
            raise ValueError("Quick cache flag must be0 or1")
        return cls(env[keys[0]].rstrip("/"), env[keys[1]], env[keys[2]], limit, cache == "1")


@dataclass(frozen=True)
class QuickQueryRuntime:
    backend: Any
    token_counter: Any
    context_limit: int
    cache_enabled: bool = True


class _QuickPipeline:
    def __init__(self, parent: Any, runtime: QuickQueryRuntime):
        self._parent = parent
        self._runtime = runtime

    def __getattr__(self, name: str) -> Any:
        return getattr(self._parent, name)

    @property
    def model(self) -> Any:
        return self._runtime.backend

    @property
    def query_generation_slot(self) -> Any:
        return QUICK_GENERATION_SLOT

    @property
    def _translation_cache(self) -> Any:
        return getattr(self._parent, "_translation_cache", None) if self._runtime.cache_enabled else None

    def _options(self) -> dict[str, Any]:
        return {"inference_backend": self.model, "inference_counter": self._runtime.token_counter,
                "inference_context_limit": self._runtime.context_limit}

    def run(self, **kwargs: Any) -> Any:
        if not self._runtime.cache_enabled:
            kwargs["quick_cache_scope"] = None
        if kwargs.get("retrieval_instruction"):
            kwargs["retrieval_instruction"] = retrieval_question(
                kwargs["retrieval_instruction"], context=kwargs.get("instruction", ""),
            )
        return self._parent.run(**kwargs, **self._options())

    def translate_copilot_text(self, text: str, **kwargs: Any) -> str:
        return self._parent.translate_copilot_text(text, **kwargs, **self._options())

    def classify_copilot_intent(self, task: str, prompt: str) -> str:
        return self._parent.classify_copilot_intent(task, prompt, **self._options())

    @property
    def supports_copilot_dialogue(self) -> bool:
        return callable(getattr(self._parent, "plan_copilot_dialogue", None))

    def plan_copilot_dialogue(self, prompt: str, **kwargs: Any) -> Any:
        return self._parent.plan_copilot_dialogue(prompt, **kwargs, **self._options())

    def converse_copilot(self, prompt: str, **kwargs: Any) -> str:
        return self._parent.converse_copilot(prompt, **kwargs, **self._options())


def select_query_pipeline(pipeline: Any, response_mode: str) -> Any:
    if isinstance(pipeline, _QuickPipeline):
        return pipeline
    runtime = getattr(pipeline, "quick_query_runtime", None)
    return _QuickPipeline(pipeline, runtime) if response_mode == "quick" and isinstance(runtime, QuickQueryRuntime) else pipeline


def generation_slot_for(pipeline: Any) -> Any:
    return getattr(pipeline, "query_generation_slot", GENERATION_SLOT)


def translation_pipeline_for(pipeline: Any, source: str, target: str) -> Any:
    # Translation is independent of analysis mode: the tuned 27B analysis model
    # can emit Korean instead of the requested language for detailed answers.
    # Use the validated multilingual backend without switching the analysis model.
    parent = pipeline._parent if isinstance(pipeline, _QuickPipeline) else pipeline
    if {source, target} & {"ne", "my", "si", "km"}:
        return parent
    return pipeline if isinstance(pipeline, _QuickPipeline) else select_query_pipeline(parent, "quick")


def retrieval_question(question: str, *, context: str = "", expand_terms: bool = True) -> str:
    """Expand training terms to the vocabulary in manuals; never add numeric advice."""
    # The router renders model advice as one JSON-quoted physical line. Remove
    # that data block before reading relation, history or process selectors;
    # an interpretation must not change those through its words or headings.
    context = re.sub(r"(?m)^\[후속 질문 해석[^\n]*\]\r?\n[^\n]*(?:\r?\n모델의 임시 해석[^\n]*)?", "", context)
    query = question
    # The detailed (27B) route passes expand_terms=False: the KURE index contains these terms
    # verbatim ("저도막 결함의 원인 중 하나는 중첩 부족"), and replacing them hid those passages
    # (2026-11-17). Quick guidance re-applies this function in _QuickPipeline.run and keeps the table.
    terms = {
        "저도막": "도막 두께 부족 규정 두께 미달 얇은 도막",
        "과도막": "도막 두께 과다 과도하게 두꺼운 도막",
        "이격거리": "분사 거리 피도체와 건 사이 거리",
        "오버랩": "중첩 overlap 겹침",
    } if expand_terms else {}
    for term, expansion in terms.items():
        query = query.replace(term, expansion)
    # Carry the user's technical subject across an elliptical extra condition.
    # Previous AI text is deliberately excluded from search: it may be the very
    # explanation being corrected. An explicitly new subject starts fresh.
    def follows_up(text: str) -> bool:
        return bool(re.match(r"그럼|그때|그거|그러면|왜\s*그래|그런데|아니|그래도", text)
                    or re.search(r"했는데|한데|인데|일텐데|다는데|맞았|적절했|일정했|맞지\s*않", text))
    # A self-contained topic followed by its own question is a topic switch,
    # including topics unknown to this application. Do not enumerate domain or
    # general-knowledge nouns here.
    explicit_topic = bool(re.match(
        r"(?:(?:그럼|그러면|그런데)\s*)?[^?!.]{2,50}?(?:은|는|이란|란|에\s*대해)\s*"
        r"(?:뭐|무엇|어떤|어떻게|왜|어디|언제|누구|설명|알려|[?？]\s*$)", question))
    semantic_followup = '[문서 후속 질문]' in context
    new_topic = '[새 주제 질문]' in context or '[새주제질문]' in context
    previous: list[str] = []
    # An interpretation is unverified model advice, not an independent source
    # of the user's subject. Its presence must never suppress original turns.
    if not new_topic and (semantic_followup or (follows_up(question) and not explicit_topic)) and "[최근 대화]" in context:
        previous = re.findall(r"사용자: (.+)", context.split("[최근 대화]", 1)[1])
        if previous:
            # Keep bounded user exchanges in time order instead of guessing
            # which condition is the subject or trusting a rewritten subject.
            # This retains the original problem and the intermediate exclusions.
            prefix = " ".join(value[:200] for value in previous[-6:])
            query = prefix + " " + query
    current_context = context.split("[최근 대화]", 1)[0]
    def explicit_process(text: str) -> str | None:
        if re.search(r"블라스팅|blasting", text, re.I):
            return "blasting"
        return "spray" if re.search(r"스프레이|spray", text, re.I) else None
    # Current user wording wins; an actual same-topic user exchange can supply
    # an omitted process. Only then consult current context with advice removed.
    process = explicit_process(question)
    if process is None and previous:
        process = next((found for text in reversed(previous) if (found := explicit_process(text))), None)
    process = process or explicit_process(current_context)
    if process == "blasting":
        return "선박 블라스팅 노즐 작업 " + query
    return "선박 도장 스프레이 건 작업 " + query


def warm_quick_backend(backend: Any, *, timeout: float = 900.0,
                       clock: Any = time.monotonic, sleep: Any = time.sleep) -> None:
    """Readiness waits for the resident engine, including after a host restart."""
    from trainer.scripts.answer_grammar import quick_answer_regex
    from trainer.scripts.openai_backend import OpenAIBackendUnavailableError
    deadline = clock() + timeout
    while True:
        try:
            backend.complete_chat(
                [{"role": "user", "content": (
                    "다음은 기동 확인용 문구입니다. 준비가 완료됐다고 짧게 답하세요.\n"
                    + "안전한 도장 훈련 응답 준비를 확인합니다. " * 120
                )}],
                max_tokens=16, enable_thinking=False,
                answer_regex=quick_answer_regex(has_evidence=True),
            )
            return
        except OpenAIBackendUnavailableError:
            if clock() >= deadline:
                raise
            sleep(min(2.0, max(0.0, deadline - clock())))


def install_quick_runtime(pipeline: Any, env: Mapping[str, str] | None = None) -> None:
    config = QuickRuntimeConfig.from_environ(os.environ if env is None else env)
    if config is None:
        return
    from trainer.scripts.openai_backend import OpenAIChatBackend
    from trainer.scripts.prompt_budget import load_token_counter

    if not isinstance(pipeline.model, OpenAIChatBackend):
        raise ValueError("A separate quick server requires the OpenAI-compatible live backend")
    counter = load_token_counter(config.tokenizer_path, trust_remote_code=False)
    if not counter.exact:
        raise ValueError("Quick backend requires its own exact tokenizer")
    settings = replace(pipeline.model.settings, base_url=config.base_url, served_model_name=config.model_name,
                       api_key=None, request_timeout_sec=30.0,
                       sampling=replace(pipeline.model.settings.sampling, temperature=0.0))
    # Compile the decoding grammar and initial kernels before readiness becomes true.
    # Exercise a real prefill batch and decode steps; one token does not warm decoding.
    # The synthetic completion is discarded: no user/session/cache record is created.
    warmer = OpenAIChatBackend(replace(settings, request_timeout_sec=90.0))
    try:
        warm_quick_backend(warmer)
    finally:
        warmer.close()
    pipeline.quick_query_runtime = QuickQueryRuntime(OpenAIChatBackend(settings), counter,
                                                    config.context_limit, config.cache_enabled)
