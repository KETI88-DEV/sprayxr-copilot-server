"""Quick routing uses a resident small backend without cloning RAG or mutating27B."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from trainer.demo_api.query_runtime import (
    QuickRuntimeConfig, QuickQueryRuntime, select_query_pipeline, generation_slot_for,
)
from trainer.scripts.generation_slot import GENERATION_SLOT


def test_fast_runtime_is_explicitly_opt_in_and_config_must_be_complete():
    assert QuickRuntimeConfig.from_environ({}) is None
    with pytest.raises(ValueError):
        QuickRuntimeConfig.from_environ({"COPILOT_QUICK_BASE_URL": "http://quick:8000/v1"})
    config = QuickRuntimeConfig.from_environ({
        "COPILOT_QUICK_BASE_URL": "http://quick:8000/v1/",
        "COPILOT_QUICK_MODEL": "shipcoat-quick-8b",
        "COPILOT_QUICK_TOKENIZER_PATH": "/quick-model",
        "COPILOT_QUICK_CACHE_ENABLED": "0",
    })
    assert config.model_name == "shipcoat-quick-8b"
    assert config.base_url == "http://quick:8000/v1"
    assert config.cache_enabled is False


class Parent:
    def __init__(self):
        self.model = SimpleNamespace(name="27B")
        self.index = object()
        self._translation_cache = object()
    def run(self, **kwargs):
        assert self.model.name == "27B"
        return kwargs
    def translate_copilot_text(self, text, **kwargs):
        return text, kwargs
    def classify_copilot_intent(self, task, prompt, **kwargs):
        return task, prompt, kwargs


def configured_parent(cache_enabled=True):
    parent = Parent()
    backend = SimpleNamespace(name="8B")
    counter = object()
    parent.quick_query_runtime = QuickQueryRuntime(
        backend=backend, token_counter=counter, context_limit=8192, cache_enabled=cache_enabled,
    )
    return parent, backend, counter


def test_mode_selects_only_the_generator_and_keeps_shared_data_and_legacy_default():
    parent, backend, counter = configured_parent()
    fast = select_query_pipeline(parent, "quick")
    assert select_query_pipeline(parent, "detailed") is parent
    assert select_query_pipeline(Parent(), "quick").model.name == "27B"
    assert fast.model is backend and parent.model.name == "27B"
    assert fast.index is parent.index
    args = fast.run(instruction="현재 이격거리300 mm", quick_cache_scope="u")
    assert args["inference_backend"] is backend
    assert args["inference_counter"] is counter
    assert args["inference_context_limit"] == 8192
    assert args["instruction"] == "현재 이격거리300 mm"


def test_intent_and_translation_use_the_same_quick_backend():
    parent, backend, counter = configured_parent()
    fast = select_query_pipeline(parent, "quick")
    text, kwargs = fast.translate_copilot_text("300 mm를 유지하세요.", source_language="ko", target_language="en")
    assert text == "300 mm를 유지하세요." and kwargs["inference_backend"] is backend
    assert kwargs["inference_counter"] is counter
    assert fast.classify_copilot_intent("pose_recomm", "현재 자세는?")[2]["inference_backend"] is backend
    assert fast._translation_cache is parent._translation_cache


def test_quick_can_run_while_the_detailed_generation_slot_is_occupied():
    parent, backend, counter = configured_parent()
    fast = select_query_pipeline(parent, "quick")
    with GENERATION_SLOT:
        slot = generation_slot_for(fast)
        assert slot.acquire(timeout=.1)
        slot.release()
    assert generation_slot_for(parent) is GENERATION_SLOT


def test_mode_bindings_do_not_change_shared_model_during_simultaneous_requests():
    parent, backend, counter = configured_parent()
    fast = select_query_pipeline(parent, "quick")
    with ThreadPoolExecutor(max_workers=2) as pool:
        quick = pool.submit(fast.run, instruction="quick")
        detailed = pool.submit(parent.run, instruction="detailed")
        assert quick.result()["inference_backend"] is backend
        assert "inference_backend" not in detailed.result()
    assert parent.model.name == "27B"


def test_cache_can_be_disabled_for_uncached_benchmarks_without_changing_public_parameters():
    parent, backend, counter = configured_parent(cache_enabled=False)
    fast = select_query_pipeline(parent, "quick")
    assert fast.run(instruction="첫 질문", quick_cache_scope="user")["quick_cache_scope"] is None
    assert fast._translation_cache is None


def test_shared_history_updates_are_atomic_across_the_two_generators():
    from trainer.scripts.agent_pipeline import SessionContext
    import threading
    context = SessionContext(max_turns=3)
    entered = threading.Event()
    done = threading.Event()
    def update():
        entered.set()
        context.update("same-session", {"instruction": "new"})
        done.set()
    with context._lock:
        thread = threading.Thread(target=update)
        thread.start()
        assert entered.wait(1)
        assert not done.wait(.05)
    thread.join(1)
    assert done.is_set()
    assert context.get_context("same-session") == [{"instruction": "new"}]


def test_router_selects_quick_backend_without_altering_public_request(monkeypatch):
    from trainer.demo_api.tests.test_copilot_router import _FakePipeline, _client, _body, URL
    parent = _FakePipeline()
    backend = SimpleNamespace(name="8B")
    parent.quick_query_runtime = QuickQueryRuntime(backend, object(), 8192)
    client = _client(monkeypatch, parent)
    assert client.post(URL, json=_body(response_mode="quick")).status_code == 200
    assert parent.calls[-1]["inference_backend"] is backend
    assert client.post(URL, json=_body(response_mode="detailed")).status_code == 200
    assert "inference_backend" not in parent.calls[-1]


def test_quick_generation_uses_its_own_budget_and_only_visible_evidence(monkeypatch):
    from trainer.demo_api.tests.test_query_quick_latency import _pipeline
    pipeline, generated, searched, chunks = _pipeline(monkeypatch)
    counter = SimpleNamespace(count_text=lambda text: len(text), exact=True,
                              count_messages=lambda messages, **kw: sum(len(m["content"]) for m in messages))
    chunks[:] = [{"chunk_id": "DOC-a-C0001", "text": "300 mm", "rerank_score": 1.0},
                 {"chunk_id": "DOC-b-C0002", "text": "x" * 3000, "rerank_score": .1}]
    backend = object()
    result = pipeline.run(instruction="거리?", force_route="fast", inference_backend=backend,
                          inference_counter=counter, inference_context_limit=1500, max_tokens=128)
    assert generated[-1][1]["inference_backend"] is backend
    assert [c["chunk_id"] for c in result["rag_chunks"]] == ["DOC-a-C0001"]
    assert result["metadata"]["context"]["max_model_len"] == 1500
    assert not result["metadata"]["context"]["overflow"]


def test_configured_backend_does_not_change_detailed_model_or_sampling(monkeypatch):
    from trainer.demo_api.query_runtime import install_quick_runtime
    from trainer.scripts.openai_backend import OpenAIBackendSettings, OpenAIChatBackend
    import trainer.scripts.prompt_budget as budget
    settings = OpenAIBackendSettings.from_config({
        "llm": {"base_url": "http://large/v1", "served_model_name": "27B"},
        "generation": {"temperature": .3, "top_p": .8, "top_k": 20, "repetition_penalty": 1.1},
    })
    parent = SimpleNamespace(model=OpenAIChatBackend(settings))
    warmups = []
    monkeypatch.setattr(OpenAIChatBackend, "complete_chat", lambda self, *a, **kw: warmups.append(kw) or "[")
    monkeypatch.setattr(budget, "load_token_counter", lambda *a, **kw: SimpleNamespace(exact=True))
    install_quick_runtime(parent, {"COPILOT_QUICK_BASE_URL": "http://small/v1", "COPILOT_QUICK_MODEL": "8B",
                                   "COPILOT_QUICK_TOKENIZER_PATH": "/quick"})
    try:
        assert parent.model.settings is settings
        assert parent.quick_query_runtime.backend.settings.served_model_name == "8B"
        assert parent.quick_query_runtime.backend.settings.sampling.temperature == 0
        assert warmups[0]["max_tokens"] == 16
    finally:
        parent.model.close()
        parent.quick_query_runtime.backend.close()


def test_training_terms_and_process_context_select_relevant_manual_vocabulary():
    from trainer.demo_api.query_runtime import retrieval_question
    assert "두께 미달" in retrieval_question("저도막이 뭐야?")
    assert "분사 거리" in retrieval_question("적정 이격거리는?")
    assert "블라스팅 노즐" in retrieval_question("적정 거리는?", context='{"process":"blasting"}')
    assert "스프레이" in retrieval_question("스프레이 속도는?", context='{"process":"blasting"}')
    assert "스프레이" in retrieval_question("적정 거리는?", context='{"process":"spray"}\n[최근 대화]\n사용자: blasting도 설명해줘')
    assert "이격거리" in retrieval_question("그럼 더 가까이 가야 해?", context="[최근 대화]\n사용자: 이격거리는?\nCo-Pilot:300mm입니다.")


def test_short_small_model_answer_limits_visible_passages_without_changing_default(monkeypatch):
    from trainer.demo_api.tests.test_query_quick_latency import _pipeline
    pipeline, generated, searched, chunks = _pipeline(monkeypatch)
    chunks[:] = [{"chunk_id": f"DOC-a-C{i:04}", "text": "도장 자료", "rerank_score": 1.0} for i in range(6)]
    full = pipeline.run(instruction="질문", force_route="fast")
    quick = pipeline.run(instruction="질문", force_route="fast", inference_backend=object(), concise_answer=True)
    assert len(full["rag_chunks"]) == 6
    assert len(quick["rag_chunks"]) == 4


def test_quick_body_rejects_control_tags_and_no_rag_does_not_invent_citations():
    from trainer.scripts.answer_grammar import quick_answer_regex
    import re
    assert re.fullmatch(quick_answer_regex(has_evidence=False), "[답변] 200 mm 줄이세요.")
    assert not re.fullmatch(quick_answer_regex(has_evidence=False), "[답변] 200 mm 줄이세요.\n<근거>1_20240618")
    assert not re.fullmatch(quick_answer_regex(has_evidence=False), "[답변] 답변.</think>")
    assert not re.fullmatch(quick_answer_regex(has_evidence=False), "[답변] 답변.\n[근거] DOC-a-C0001")
    assert re.fullmatch(quick_answer_regex(has_evidence=True), "[답변] 답변.\n[근거] DOC-a-C0001")


def test_startup_warmup_retries_an_unavailable_engine_but_has_a_deadline():
    from trainer.demo_api.query_runtime import warm_quick_backend
    from trainer.scripts.openai_backend import OpenAIBackendUnavailableError
    clock = [0.0]
    calls = []
    def complete(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise OpenAIBackendUnavailableError("starting")
        return "[답변] 준비 완료."
    warm_quick_backend(SimpleNamespace(complete_chat=complete), clock=lambda: clock[0],
                       sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    assert len(calls) == 2 and clock[0] == 2
    def unavailable(*args, **kwargs):
        raise OpenAIBackendUnavailableError("not reachable")
    with pytest.raises(OpenAIBackendUnavailableError):
        warm_quick_backend(SimpleNamespace(complete_chat=unavailable), timeout=1,
                           clock=lambda: clock[0], sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds))


def test_low_resource_translations_keep_the_existing_backend_and_its_own_slot():
    from trainer.demo_api.query_runtime import translation_pipeline_for
    parent, backend, counter = configured_parent()
    quick = select_query_pipeline(parent, "quick")
    assert translation_pipeline_for(quick, "ko", "vi") is quick
    assert translation_pipeline_for(quick, "en", "ko") is quick
    for language in ("ne", "my", "si", "km"):
        selected = translation_pipeline_for(quick, "ko", language)
        assert selected is parent and generation_slot_for(selected) is GENERATION_SLOT


@pytest.mark.parametrize("language", ["en", "vi", "id", "th", "uz", "ru", "zh"])
def test_detailed_analysis_keeps_27b_but_translation_uses_the_multilingual_backend(language):
    from trainer.demo_api.query_runtime import translation_pipeline_for
    from trainer.scripts.generation_slot import QUICK_GENERATION_SLOT
    parent, backend, counter = configured_parent()
    detailed = select_query_pipeline(parent, "detailed")
    translator = translation_pipeline_for(detailed, "ko", language)
    assert translator.model is backend
    assert generation_slot_for(translator) is QUICK_GENERATION_SLOT
    text, kwargs = translator.translate_copilot_text("300 mm를 유지하세요.", source_language="ko", target_language=language)
    assert kwargs["inference_backend"] is backend and kwargs["inference_counter"] is counter
    assert detailed is parent and detailed.model.name == "27B"
    assert "inference_backend" not in detailed.run(instruction="상세 분석")


def test_translation_preserves_the_legacy_and_low_resource_backends():
    from trainer.demo_api.query_runtime import translation_pipeline_for
    parent, _, _ = configured_parent()
    legacy = Parent()
    assert translation_pipeline_for(legacy, "ko", "vi") is legacy
    for language in ("ne", "my", "si", "km"):
        assert translation_pipeline_for(parent, "ko", language) is parent
