from __future__ import annotations

import io
import logging
import wave
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.copilot_router import install_copilot_error_handlers
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.speech_to_text import SpeechToTextUnavailableError, Transcript
from trainer.demo_api.voice_router import VOICE_QUERY_PATH, create_voice_router
from trainer.scripts.openai_backend import OpenAIBackendUnavailableError


def _wav(seconds: float = 0.1) -> bytes:
    stream = io.BytesIO()
    with wave.open(stream, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(b"\x00" * int(seconds * 16_000) * 2)
    return stream.getvalue()


class _Stt:
    def __init__(self, text: str = "저도막이 무엇인지 설명해줘", exc: Exception | None = None) -> None:
        self.text = text
        self.exc = exc
        self.calls: list[tuple[int, str]] = []

    def transcribe(self, audio: bytes, language: str) -> Transcript:
        self.calls.append((len(audio), language))
        if self.exc is not None:
            raise self.exc
        return Transcript(text=self.text, language=language)


class _Pipeline:
    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = exc
        self.calls: list[dict[str, Any]] = []

    def run(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.exc is not None:
            raise self.exc
        return {
            "response": "[답변] 기준보다 얇은 도막입니다.\n[근거] DOC-doc_0001-C0001",
            "rag_chunks": [{"chunk_id": "DOC-doc_0001-C0001", "text": ""}],
        }


TEST_VOICE_QUERY_PATH = "/api/v1/test/query-voice"
TEST_VOICE_ASSET_PATH = "/api/v1/test/assets/query-voice-ko.wav"


def _client(stt: _Stt, pipeline: _Pipeline) -> TestClient:
    app = FastAPI()
    app.include_router(
        create_voice_router(
            lambda: pipeline,
            stt_provider=lambda: stt,
            store=SequenceStore(":memory:"),
            mcp_base_url="http://127.0.0.1:9",
        )
    )
    install_copilot_error_handlers(app)
    return TestClient(app)


def _post(client: TestClient, **fields: str) -> Any:
    data = {"user_id": "W-017", "language": "ko", "use_rag": "Y"} | fields
    return client.post(
        VOICE_QUERY_PATH,
        data=data,
        files={"audio": ("question.wav", _wav(), "audio/wav")},
    )


def test_voice_query_returns_transcript_and_the_existing_query_answer_fields() -> None:
    stt = _Stt()
    pipeline = _Pipeline()

    response = _post(_client(stt, pipeline), session_id="S-1", request_id="REQ-V-1")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["transcript"] == "저도막이 무엇인지 설명해줘"
    assert data["language"] == "ko"
    assert isinstance(data["stt_latency_ms"], int) and data["stt_latency_ms"] >= 0
    assert data["answer"] == "기준보다 얇은 도막입니다."
    assert data["citations"] == ["DOC-doc_0001-C0001"]
    assert pipeline.calls[0]["instruction"] == data["transcript"]
    assert pipeline.calls[0]["force_route"] == "fast"
    assert pipeline.calls[0]["enable_thinking"] is False


def test_blank_transcript_is_rejected_without_calling_the_model() -> None:
    pipeline = _Pipeline()

    response = _post(_client(_Stt(text="   "), pipeline))

    assert response.status_code == 422
    assert response.json()["data"] is None
    assert pipeline.calls == []


def test_stt_unavailable_and_model_unavailable_are_reported_as_503() -> None:
    stt_failure = _post(_client(_Stt(exc=SpeechToTextUnavailableError()), _Pipeline()))
    model_failure = _post(_client(_Stt(), _Pipeline(exc=OpenAIBackendUnavailableError())))

    assert stt_failure.status_code == 503
    assert stt_failure.json()["message"] == "Speech recognition unavailable"
    assert model_failure.status_code == 503
    assert model_failure.json()["message"] == "Model server unavailable"


def test_invalid_audio_and_unsupported_language_use_the_v25_error_envelope() -> None:
    client = _client(_Stt(), _Pipeline())
    invalid_audio = client.post(
        VOICE_QUERY_PATH,
        data={"user_id": "W-017", "language": "ko"},
        files={"audio": ("bad.wav", b"bad", "audio/wav")},
    )
    unsupported = _post(client, language="en")

    assert invalid_audio.status_code == 422
    assert invalid_audio.json() == {"status": "ERROR", "code": 422, "message": "Invalid audio", "data": None}
    assert unsupported.status_code == 422
    assert unsupported.json()["data"] is None


def test_transcript_and_internal_stt_exception_are_not_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "민감한 음성 질문 원문"
    with caplog.at_level(logging.INFO):
        response = _post(_client(_Stt(text=secret), _Pipeline()))

    assert response.status_code == 200
    assert secret not in caplog.text

    caplog.clear()
    with caplog.at_level(logging.INFO):
        failed = _post(_client(_Stt(exc=RuntimeError("private-engine-detail")), _Pipeline()))
    assert failed.status_code == 503
    assert "private-engine-detail" not in caplog.text


def test_input_free_voice_test_runs_the_real_stt_and_query_path() -> None:
    """Catches a demo route that returns canned JSON instead of exercising STT/query."""
    stt = _Stt(text="스프레이 도장 시 적정 이격거리를 알려주세요")
    pipeline = _Pipeline()
    client = _client(stt, pipeline)

    response = client.post(TEST_VOICE_QUERY_PATH)

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["transcript"] == "스프레이 도장 시 적정 이격거리를 알려주세요"
    assert data["user_id"] == "TSB-VOICE-DEMO-001"
    assert data["session_id"] == "SWAGGER-VOICE-DEMO"
    assert pipeline.calls[0]["instruction"] == data["transcript"]
    assert stt.calls[0][1] == "ko"


def test_voice_sample_asset_is_downloadable_valid_16khz_mono_pcm16_wav() -> None:
    """Catches missing or incompatible sample audio offered to Swagger users."""
    client = _client(_Stt(), _Pipeline())

    response = client.get(TEST_VOICE_ASSET_PATH)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert "query-voice-ko.wav" in response.headers["content-disposition"]
    with wave.open(io.BytesIO(response.content), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 16_000
