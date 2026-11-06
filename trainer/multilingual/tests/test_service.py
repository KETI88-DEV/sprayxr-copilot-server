from __future__ import annotations

import io
import wave
from contextlib import nullcontext
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from trainer.multilingual.service import Settings, Whisper, create_app, protect, validate_wav


def audio_payload(*, rate: int = 16000, truncated: bool = False) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        stream.writeframes(b"\x10\x01" * rate)
    payload = output.getvalue()
    return payload[:-10] if truncated else payload


def query(output_language: str = "en") -> dict:
    return {"query": {"user_id": "USER-1", "prompt": "저도막이 뭐야?", "task_type": "knowledge_search"}, "input_language": "ko", "output_language": output_language}


def upstream(answer: str = "건조도막두께는 80 µm입니다. A-1을 확인하세요.") -> dict:
    return {"status": "SUCCESS", "code": 200, "message": "ok", "data": {"request_id": None, "session_id": None, "user_id": "USER-1", "scenario_id": None, "generated_at": "2026-10-06T00:00:00Z", "latency_ms": 100, "degraded": False, "answer": answer, "resolved_intents": ["knowledge_search"], "context_status": "not_available", "sequence_context": None, "coating_context": None, "pose_context": None, "retrieval_used": True, "citations": ["DOC-1"]}}


def client_for(*, translation: str | None = "Dry film thickness is 80 µm. Check A-1.", finish_reason: str = "stop", calls: list | None = None) -> TestClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request)
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "test-model"}]})
        if request.url.path.endswith("/api/ready"):
            return httpx.Response(200, json={"ready": True})
        if request.url.path.endswith("/query"):
            return httpx.Response(200, json=upstream())
        return httpx.Response(200, json={"choices": [{"message": {"content": translation}, "finish_reason": finish_reason}]})
    settings = Settings(model="test-model", stt_model_path="/missing", stt_device="cpu")
    return TestClient(create_app(settings, transport=httpx.MockTransport(handler), transcriber=lambda audio, language: "저도막이 뭐야?"))


@pytest.mark.parametrize("language", ["vi", "uz", "my", "ne", "id", "th"])
def test_unreleased_languages_rejected_before_inference(language: str) -> None:
    calls = []
    with client_for(calls=calls) as client:
        response = client.post("/query", json=query(language))
    assert response.status_code == 422
    assert response.json()["data"] is None
    assert calls == []


def test_translation_preserves_canonical_data() -> None:
    with client_for() as client:
        response = client.post("/query", json=query())
    assert response.status_code == 200
    result = response.json()["data"]
    assert result["copilot"] == upstream()["data"]
    assert result["localization"]["status"] == "translated"
    assert result["localization"]["answer"].startswith("Dry film")


@pytest.mark.parametrize("translation", ["Dry film thickness is 90 µm. Check A-1.", "Dry film thickness is 80 mm. Check A-1.", "", None, "건조도막두께는 80 µm입니다. A-1을 확인하세요.", "The film is 80 µm thick. Check A-1."])
def test_invalid_translation_falls_back_without_losing_analysis(translation: str | None) -> None:
    with client_for(translation=translation) as client:
        response = client.post("/query", json=query())
    assert response.status_code == 200
    result = response.json()["data"]
    assert result["copilot"] == upstream()["data"]
    assert result["localization"]["status"] == "fallback_original"
    assert result["localization"]["answer"] is None


def test_truncated_generation_is_not_a_translation() -> None:
    with client_for(finish_reason="length") as client:
        response = client.post("/query", json=query())
    assert response.json()["data"]["localization"]["status"] == "fallback_original"


def test_korean_skips_translation() -> None:
    calls = []
    with client_for(calls=calls) as client:
        response = client.post("/query", json=query("ko"))
    assert response.json()["data"]["localization"]["status"] == "unchanged"
    assert len(calls) == 1


def test_bad_input_translation_never_runs_query() -> None:
    calls = []
    body = query()
    body.update(input_language="en")
    body["query"]["prompt"] = "Check 80 µm in A-1"
    with client_for(translation="Check 90 µm in A-1", calls=calls) as client:
        response = client.post("/query", json=body)
    assert response.status_code == 502
    assert all(not request.url.path.endswith("/query") for request in calls)


def test_transcription_requires_confirmation_and_never_calls_query() -> None:
    calls = []
    with client_for(calls=calls) as client:
        response = client.post("/transcriptions", data={"language": "ko"}, files={"audio": ("test.wav", audio_payload(), "audio/wav")})
    assert response.status_code == 200
    assert response.json()["data"]["confirmation_required"] is True
    assert calls == []


@pytest.mark.parametrize("payload", [b"not wav", audio_payload(rate=8000), audio_payload(truncated=True)])
def test_invalid_audio(payload: bytes) -> None:
    with pytest.raises(ValueError):
        validate_wav(payload)


def test_protected_numbers_units_and_identifiers() -> None:
    assert protect("A-1: 80 µm / TR-L01 / 50%") != protect("A-2: 80 µm / TR-L01 / 50%")
    assert protect("A-1: 80 µm / TR-L01 / 50%") != protect("A-1: 80 mm / TR-L01 / 50%")
    assert protect("80 µm 이상: ≥80") != protect("80 µm 이하: ≤80")
    assert protect("A-1을 확인하세요. 80 µm") == protect("Check A-1. 80 μm")


def test_capabilities_have_explicit_deferred_languages() -> None:
    with client_for() as client:
        body = client.get("/languages").json()
    assert body["enabled"] == ["ko", "en", "zh"]
    assert body["deferred"] == ["vi", "uz", "my", "ne"]


def test_readiness_requires_dependencies() -> None:
    with client_for() as client:
        response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["ready"] is True


def test_non_knowledge_task_rejected() -> None:
    body = query()
    body["query"]["task_type"] = "pose_recomm"
    with client_for() as client:
        response = client.post("/query", json=body)
    assert response.status_code == 422


def test_whisper_uses_raw_generation_to_check_end_token() -> None:
    class Tensor:
        def to(self, *args, **kwargs):
            return self

    class Processor:
        def __call__(self, *args, **kwargs):
            return SimpleNamespace(input_features=Tensor(), attention_mask=Tensor())

        def batch_decode(self, tokens, **kwargs):
            return ["Roses are red."]

    def generate(*args, **kwargs):
        assert kwargs["return_dict_in_generate"] is True
        return SimpleNamespace(sequences=[[123, 50257]])

    engine = Whisper.__new__(Whisper)
    engine.processor = Processor()
    engine.model = SimpleNamespace(generate=generate, generation_config=SimpleNamespace(eos_token_id=50257))
    engine.torch = SimpleNamespace(inference_mode=nullcontext)
    engine.device = "cpu"
    engine.dtype = "float32"
    assert engine(validate_wav(audio_payload()), "en") == "Roses are red."
