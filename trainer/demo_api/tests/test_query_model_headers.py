"""Model labels follow answer generation, including mode selection and cache misses."""
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.copilot_router import create_copilot_router
from trainer.demo_api.query_runtime import QuickQueryRuntime
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.tests.test_copilot_router import _FakePipeline, _body, URL, _DATA_KEYS
from trainer.scripts.quick_answer import TextCache


DETAILED = "qwen36-27b-shipcoat-v6-1-fp8"
QUICK = "Qwen3-8B-AWQ"


class GeneratingPipeline(_FakePipeline):
    def __init__(self, *, cached=False):
        super().__init__()
        self.model = SimpleNamespace(settings=SimpleNamespace(served_model_name=DETAILED))
        backend = SimpleNamespace(settings=SimpleNamespace(served_model_name=QUICK))
        self.quick_query_runtime = QuickQueryRuntime(backend, object(), 8192)
        self.cached = cached
        self._translation_cache = TextCache(capacity=8, ttl=60)

    def run(self, **kwargs):
        result = super().run(**kwargs)
        backend = kwargs.get("inference_backend", self.model)
        result["metadata"] = {"generation_model": backend.settings.served_model_name,
                              "quick_cache_hit": self.cached, "private_config": "must-stay-private"}
        return result

    def translate_copilot_text(self, text, **kwargs):
        return "Have a good day."


def client_for(pipeline):
    app = FastAPI()
    app.include_router(create_copilot_router(lambda: pipeline, store=SequenceStore(":memory:")))
    return TestClient(app)


@pytest.mark.parametrize("mode,expected", [("quick", QUICK), ("detailed", DETAILED)])
def test_model_header_reports_the_selected_generator_without_changing_json(mode, expected):
    response = client_for(GeneratingPipeline()).post(URL, json=_body(response_mode=mode))
    assert response.status_code == 200
    assert response.headers.get("X-Copilot-Model") == expected
    assert set(response.json()["data"]) == _DATA_KEYS
    assert "must-stay-private" not in str(response.headers)


def test_cached_generation_and_verified_glossary_do_not_claim_a_model_call():
    client = client_for(GeneratingPipeline(cached=True))
    cached = client.post(URL, json=_body())
    glossary = client.post(URL, json=_body(prompt="노즐을 영어로 번역해줘"))
    assert cached.status_code == glossary.status_code == 200
    assert "X-Copilot-Model" not in cached.headers
    assert "X-Copilot-Model" not in glossary.headers
    assert "X-Copilot-Translation-Model" not in glossary.headers


def test_analyzed_rule_answers_do_not_inherit_the_planner_model():
    response = client_for(GeneratingPipeline()).post(URL, json=_body(task_type="error_analysis"))
    assert response.status_code == 200
    assert "X-Copilot-Model" not in response.headers


@pytest.mark.parametrize("unsafe", ["bad\r\nInjected: value", "/models/private/model", "x" * 129, "한글모델"])
def test_unsafe_model_identifiers_are_not_exposed(unsafe):
    pipeline = GeneratingPipeline()
    pipeline.model.settings.served_model_name = unsafe
    response = client_for(pipeline).post(URL, json=_body(response_mode="detailed"))
    assert response.status_code == 200
    assert "X-Copilot-Model" not in response.headers


def test_direct_translation_reports_its_actual_generator_but_cached_translation_does_not():
    client = client_for(GeneratingPipeline())
    body = _body(prompt='"좋은 하루 보내세요."를 영어로 번역해줘', response_mode="detailed")
    first = client.post(URL, json=body)
    second = client.post(URL, json=body)
    assert first.status_code == second.status_code == 200
    assert first.headers.get("X-Copilot-Model") == QUICK
    assert "X-Copilot-Model" not in second.headers


def test_translated_detailed_answer_distinguishes_the_answer_and_translation_models():
    pipeline = GeneratingPipeline()
    pipeline.response = "[답변] 도장 작업을 준비하세요."
    pipeline.translate_copilot_text = lambda *args, **kwargs: "Prepare for painting."
    response = client_for(pipeline).post(URL, json=_body(response_mode="detailed", response_language="en"))
    assert response.status_code == 200, response.text
    assert response.headers.get("X-Copilot-Model") == DETAILED
    assert response.headers.get("X-Copilot-Translation-Model") == QUICK


@pytest.mark.parametrize("fallback,expected", [(False, DETAILED), (True, QUICK)])
def test_conversation_reports_the_backend_that_supplied_the_final_answer(fallback, expected):
    class DialoguePipeline(GeneratingPipeline):
        def plan_copilot_dialogue(self, *args, **kwargs):
            return {"kind": "chat", "intent": "knowledge", "scope": "unspecified"}

        def converse_copilot(self, *args, **kwargs):
            if fallback and "inference_backend" not in kwargs:
                return "[분석] 대화 안내입니다."
            return "안녕하세요."

    response = client_for(DialoguePipeline()).post(URL, json=_body(prompt="안녕", response_mode="detailed"))
    assert response.status_code == 200, response.text
    assert response.headers.get("X-Copilot-Model") == expected


@pytest.mark.parametrize("override,expected", [(False, DETAILED), (True, QUICK)])
def test_native_generation_metadata_preserves_the_backend_and_omits_cache_hits(monkeypatch, override, expected):
    from trainer.demo_api.tests.test_query_quick_latency import _pipeline
    pipeline, _, _, _ = _pipeline(monkeypatch)
    pipeline.model = SimpleNamespace(settings=SimpleNamespace(served_model_name=DETAILED))
    kwargs = dict(instruction="도장 분사 거리?", force_route="fast", quick_cache_scope="model-test")
    if override:
        kwargs["inference_backend"] = SimpleNamespace(settings=SimpleNamespace(served_model_name=QUICK))
    first = pipeline.run(**kwargs)
    cached = pipeline.run(**kwargs)
    assert first["metadata"].get("generation_model") == expected
    assert "generation_model" not in cached["metadata"]


def test_evaluation_pipeline_without_a_model_attribute_keeps_its_response(monkeypatch):
    from trainer.demo_api.tests.test_query_quick_latency import _pipeline
    pipeline, _, _, _ = _pipeline(monkeypatch)
    del pipeline.model

    result = pipeline.run(instruction="평가용 도장 분사 거리?", force_route="fast")

    assert result["response"].startswith("[답변]")
    assert "generation_model" not in result["metadata"]
