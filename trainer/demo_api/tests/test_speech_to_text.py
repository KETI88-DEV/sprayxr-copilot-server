from __future__ import annotations

import io
import wave

import pytest

from trainer.demo_api.speech_to_text import InvalidAudioError, validate_wav


def _wav(*, seconds: float = 0.25, channels: int = 1, sample_rate: int = 16_000, width: int = 2) -> bytes:
    frames = int(seconds * sample_rate)
    stream = io.BytesIO()
    with wave.open(stream, "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(width)
        output.setframerate(sample_rate)
        output.writeframes(b"\x00" * frames * channels * width)
    return stream.getvalue()


def test_accepts_pcm_16khz_mono_wav_up_to_fifteen_seconds() -> None:
    validated = validate_wav(_wav(seconds=15.0))

    assert validated.sample_rate == 16_000
    assert validated.channels == 1
    assert validated.sample_width == 2
    assert validated.duration_seconds == pytest.approx(15.0)


@pytest.mark.parametrize(
    ("audio", "reason"),
    [
        (_wav(channels=2), "audio_must_be_mono"),
        (_wav(sample_rate=8_000), "audio_must_be_16khz"),
        (_wav(width=1), "audio_must_be_pcm16"),
        (_wav(seconds=15.01), "audio_too_long"),
        (b"not-a-wave", "invalid_wav"),
    ],
)
def test_rejects_audio_outside_the_public_contract(audio: bytes, reason: str) -> None:
    with pytest.raises(InvalidAudioError, match=reason):
        validate_wav(audio)
