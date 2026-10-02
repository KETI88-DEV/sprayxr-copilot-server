"""Validated WAV input and lazy speech-to-text provider contracts."""

from __future__ import annotations

import io
import threading
import wave
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Protocol


VoiceLanguage = Literal["ko", "vi", "id", "th"]


class InvalidAudioError(ValueError):
    pass


class SpeechToTextUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class ValidatedWav:
    audio: bytes
    channels: int
    sample_rate: int
    sample_width: int
    frame_count: int
    duration_seconds: float


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str


class SpeechToText(Protocol):
    def transcribe(self, audio: bytes, language: VoiceLanguage) -> Transcript: ...


def validate_wav(audio: bytes) -> ValidatedWav:
    try:
        with wave.open(io.BytesIO(audio), "rb") as source:
            channels = source.getnchannels()
            sample_rate = source.getframerate()
            sample_width = source.getsampwidth()
            frame_count = source.getnframes()
            compression = source.getcomptype()
    except (EOFError, wave.Error):
        raise InvalidAudioError("invalid_wav") from None

    if channels != 1:
        raise InvalidAudioError("audio_must_be_mono")
    if sample_rate != 16_000:
        raise InvalidAudioError("audio_must_be_16khz")
    if sample_width != 2 or compression != "NONE":
        raise InvalidAudioError("audio_must_be_pcm16")
    if frame_count <= 0:
        raise InvalidAudioError("audio_is_empty")
    duration = frame_count / sample_rate
    if duration > 15.0:
        raise InvalidAudioError("audio_too_long")
    return ValidatedWav(
        audio=audio,
        channels=channels,
        sample_rate=sample_rate,
        sample_width=sample_width,
        frame_count=frame_count,
        duration_seconds=duration,
    )


class LazySpeechToText:
    def __init__(self, factory: Callable[[], SpeechToText] | None = None) -> None:
        self._factory = factory
        self._instance: SpeechToText | None = None
        self._failed = False
        self._guard = threading.Lock()

    def _get(self) -> SpeechToText:
        if self._instance is not None:
            return self._instance
        if self._factory is None:
            raise SpeechToTextUnavailableError()
        with self._guard:
            if self._instance is None:
                try:
                    self._instance = self._factory()
                except Exception:
                    self._failed = True
                    raise SpeechToTextUnavailableError() from None
        return self._instance

    @property
    def ready(self) -> bool:
        return self._instance is not None

    @property
    def failed(self) -> bool:
        return self._failed

    def preload(self) -> None:
        self._get()

    def transcribe(self, audio: bytes, language: VoiceLanguage) -> Transcript:
        return self._get().transcribe(audio, language)


class TransformersWhisperSpeechToText:
    """Offline Whisper runtime backed only by a mounted local model directory."""

    def __init__(self, model_path: str, *, device: str = "cuda") -> None:
        if not Path(model_path).is_dir():
            raise SpeechToTextUnavailableError()
        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

            dtype = torch.float16 if device.startswith("cuda") else torch.float32
            self._processor = AutoProcessor.from_pretrained(
                model_path,
                local_files_only=True,
            )
            self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_path,
                local_files_only=True,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            ).to(device).eval()
            self._torch = torch
            self._device = device
            self._dtype = dtype
        except Exception:
            raise SpeechToTextUnavailableError() from None

    def transcribe(self, audio: bytes, language: VoiceLanguage) -> Transcript:
        import numpy as np

        try:
            with wave.open(io.BytesIO(audio), "rb") as source:
                samples = np.frombuffer(source.readframes(source.getnframes()), dtype=np.int16)
            normalized = samples.astype(np.float32) / 32768.0
            if float(np.sqrt(np.mean(np.square(normalized)))) < 0.0015:
                return Transcript(text="", language=language)
            inputs = self._processor(
                normalized,
                sampling_rate=16_000,
                return_tensors="pt",
                return_attention_mask=True,
            )
            features = inputs.input_features.to(device=self._device, dtype=self._dtype)
            attention_mask = inputs.attention_mask.to(device=self._device)
            inference_mode = getattr(self._torch, "inference_mode", None)
            context = inference_mode() if inference_mode is not None else nullcontext()
            with context:
                tokens = self._model.generate(
                    features,
                    attention_mask=attention_mask,
                    language=language,
                    task="transcribe",
                    max_new_tokens=64,
                    do_sample=False,
                )
            text = self._processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()
            return Transcript(text=text, language=language)
        except SpeechToTextUnavailableError:
            raise
        except Exception:
            raise SpeechToTextUnavailableError() from None
