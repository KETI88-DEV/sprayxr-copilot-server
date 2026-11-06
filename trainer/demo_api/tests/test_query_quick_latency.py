"""Quick repeats reuse inference only while all evidence and context match."""
from types import SimpleNamespace

import pytest

from trainer.scripts.quick_answer import TextCache


def test_cache_has_a_fixed_expiry_capacity_and_does_not_keep_large_values():
    clock = [0.0]
    cache = TextCache(capacity=2, ttl=10, clock=lambda: clock[0])
    cache.put("a", "first")
    clock[0] = 5
    cache.put("b", "second")
    assert cache.get("a") == "first"
    clock[0] = 10
    assert cache.get("a") is None  # Reads do not extend lifetime.
    cache.put("c", "third")
    cache.put("d", "fourth")
    assert cache.get("b") is None
    cache.put("large", "x" * 24001)
    assert cache.get("large") is None


def _pipeline(monkeypatch):
    from trainer.scripts import agent_pipeline as module
    module._ensure_scripts_path()
    import search_index

    pipeline = object.__new__(module.AgentPipeline)
    pipeline.config = {"llm": {"max_model_len": 8192}, "generation": {"max_new_tokens": 1024}}
    pipeline.session_context = module.SessionContext()
    pipeline.index = pipeline.id_map = pipeline.embedder = pipeline.reranker = None
    pipeline._token_counter = SimpleNamespace(
        count_messages=lambda messages, **kw: sum(len(m["content"]) // 2 for m in messages),
        count_text=lambda text: len(text) // 2, exact=False,
    )
    pipeline.model = None
    generated, searched = [], []
    chunks = [{"chunk_id": "DOC-a-C0001", "text": "이격거리는 300 mm입니다.", "rerank_score": 1.0}]

    def search(query, *args):
        searched.append(query)
        return [dict(chunk) for chunk in chunks], {"total_ms": 1}

    def generate(messages, **kwargs):
        generated.append((messages, kwargs))
        return "[답변] 거리를 300 mm로 유지하세요.\n[근거] DOC-a-C0001"

    monkeypatch.setattr(search_index, "search_and_rerank", search)
    pipeline._generate_direct = generate
    return pipeline, generated, searched, chunks


def test_quick_repeats_skip_decoding_but_search_again_and_refresh_request_metadata(monkeypatch):
    pipeline, generated, searched, chunks = _pipeline(monkeypatch)
    kw = dict(instruction="질문과 안내 지침", force_route="fast", quick_cache_scope="user-1")
    first = pipeline.run(**kw, request_id="first")
    second = pipeline.run(**kw, request_id="second")
    assert len(generated) == 1
    assert searched == ["질문과 안내 지침"] * 2
    assert second["request_id"] == "second"
    assert first["response"] == second["response"]
    assert second["response"].endswith("[근거] DOC-a-C0001")
    assert second["metadata"]["quick_cache_hit"] is True
    # An updated source or user must cause fresh inference.
    chunks[0]["text"] = "새 문서: 350 mm입니다."
    pipeline.run(**kw)
    pipeline.run(**dict(kw, quick_cache_scope="user-2"))
    assert len(generated) == 3


def test_session_follow_up_and_detailed_evaluation_keep_fresh_generation(monkeypatch):
    pipeline, generated, searched, chunks = _pipeline(monkeypatch)
    kw = dict(instruction="현재 상황은?", force_route="fast", session_id="session-1", quick_cache_scope="user-1")
    pipeline.run(**kw)
    pipeline.run(**kw)
    assert len(generated) == 2  # Previous turns change the complete prompt.
    for _ in range(2):
        pipeline.run(instruction="평가용 기본 호출", force_route="fast")
    assert len(generated) == 4
    assert "문서 번호" not in generated[-1][0][0]["content"]


@pytest.mark.parametrize("raw", [
    "도구를 300",
    "[답변] 거리를 300 mm로 유지하세요.\n[근거] DOC-a-C",
    "[답변] 거리를 300 mm로 유지하세요.\n[근거] ",
    "[답변] 거리를 300 mm로 유지하세요.\n[근거] DOC-unknown-C0001",
])
def test_invalid_generation_is_not_cached(monkeypatch, raw):
    pipeline, generated, searched, chunks = _pipeline(monkeypatch)
    attempts = []
    def truncated(*args, **kwargs):
        attempts.append(True)
        return raw
    pipeline._generate_direct = truncated
    for _ in range(2):
        result = pipeline.run(instruction="질문", force_route="fast", quick_cache_scope="u")
        assert not result["metadata"]["quick_cache_hit"]
    assert len(attempts) == 2


def test_translation_reuse_is_language_specific_and_only_keeps_validated_text():
    from trainer.demo_api.query_language import translate_text, TranslationUnavailableError
    calls = []
    def translate(text, *, source_language, target_language):
        calls.append((source_language, target_language, text))
        return "Keep the distance at 300 mm." if target_language == "en" else "잘못된 한국어 답변입니다."
    pipeline = SimpleNamespace(_translation_cache=TextCache(capacity=4, ttl=60), translate_copilot_text=translate)
    for _ in range(2):
        assert translate_text(pipeline, "거리를 300 mm로 유지하세요.", source_language="ko", target_language="en") == "Keep the distance at 300 mm."
    assert len(calls) == 1
    for _ in range(2):
        with pytest.raises(TranslationUnavailableError):
            translate_text(pipeline, "거리를 300 mm로 유지하세요.", source_language="ko", target_language="vi")
    assert len(calls) == 3  # A mislabeled/invalid translation never enters the cache.
