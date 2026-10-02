from __future__ import annotations

import io
import sys
import wave
from types import SimpleNamespace

import numpy as np
import pytest

from trainer.demo_api.speech_to_text import (
    LazySpeechToText,
    SpeechToTextUnavailableError,
    TransformersWhisperSpeechToText,
)


def _wav() -> bytes:
    stream = io.BytesIO()
    with wave.open(stream, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(np.full(1600, 1000, dtype=np.int16).tobytes())
    return stream.getvalue()


class _Tensor:
    def to(self, **_: object) -> "_Tensor":
        return self


class _Processor:
    last_model_path: str | None = None

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs: object) -> "_Processor":
        cls.last_model_path = model_path
        assert kwargs["local_files_only"] is True
        return cls()

    def __call__(self, samples: object, **kwargs: object) -> SimpleNamespace:
        assert kwargs == {
            "sampling_rate": 16_000,
            "return_tensors": "pt",
            "return_attention_mask": True,
        }
        assert len(samples) == 1600  # type: ignore[arg-type]
        return SimpleNamespace(input_features=_Tensor(), attention_mask=_Tensor())

    def batch_decode(self, tokens: object, **kwargs: object) -> list[str]:
        assert kwargs["skip_special_tokens"] is True
        return ["  인식된 질문입니다  "]


class _Model:
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs: object) -> "_Model":
        assert model_path == _Processor.last_model_path
        assert kwargs["local_files_only"] is True
        return cls()

    def to(self, device: str) -> "_Model":
        assert device == "cuda"
        return self

    def eval(self) -> "_Model":
        return self

    def generate(self, _: object, **kwargs: object) -> list[list[int]]:
        assert kwargs["language"] == "ko"
        assert kwargs["task"] == "transcribe"
        assert kwargs["attention_mask"].__class__ is _Tensor
        return [[1, 2]]


def test_transformers_whisper_uses_only_the_mounted_local_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: object,
) -> None:
    fake_transformers = SimpleNamespace(
        AutoModelForSpeechSeq2Seq=_Model,
        AutoProcessor=_Processor,
    )
    fake_torch = SimpleNamespace(float16="float16", float32="float32")
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    model_path = str(tmp_path) + "/whisper"
    from pathlib import Path

    Path(model_path).mkdir()
    engine = TransformersWhisperSpeechToText(model_path, device="cuda")
    result = engine.transcribe(_wav(), "ko")

    assert result.text == "인식된 질문입니다"
    assert result.language == "ko"


def test_missing_model_directory_is_reported_as_unavailable(tmp_path: object) -> None:
    with pytest.raises(SpeechToTextUnavailableError):
        TransformersWhisperSpeechToText(str(tmp_path) + "/missing")


def test_lazy_provider_exposes_preload_readiness_and_failure() -> None:
    provider = LazySpeechToText(lambda: object())  # type: ignore[arg-type]
    assert provider.ready is False
    provider.preload()
    assert provider.ready is True
    assert provider.failed is False

    failed = LazySpeechToText(lambda: (_ for _ in ()).throw(RuntimeError("load failed")))
    with pytest.raises(SpeechToTextUnavailableError):
        failed.preload()
    assert failed.ready is False
    assert failed.failed is True
