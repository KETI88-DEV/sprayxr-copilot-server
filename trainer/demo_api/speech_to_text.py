"""Validated WAV input and lazy speech-to-text provider contracts."""

from __future__ import annotations

import io
import threading
import wave
from dataclasses import dataclass
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
                    raise SpeechToTextUnavailableError() from None
        return self._instance

    def transcribe(self, audio: bytes, language: VoiceLanguage) -> Transcript:
        return self._get().transcribe(audio, language)
