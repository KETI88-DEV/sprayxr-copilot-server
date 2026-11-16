"""Validated WAV input and lazy speech-to-text provider contracts."""

from __future__ import annotations

import io
import math
import struct
import threading
import wave
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from trainer.demo_api.query_language import QueryLanguage


VoiceLanguage = QueryLanguage
MIN_SAMPLE_RATE = 8_000
MAX_SAMPLE_RATE = 192_000
STT_SAMPLE_RATE = 16_000
MAX_WAV_BYTES = 6 * 1024 * 1024


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


def _validate_wav_metadata(audio: bytes, *, channels: int, sample_rate: int,
                           sample_width: int, frame_count: int) -> None:
    """Reject contradictory RIFF/fmt/data sizes rather than repairing the input."""
    if len(audio)<12 or audio[:4]!=b'RIFF' or audio[8:12]!=b'WAVE':
        raise InvalidAudioError('invalid_wav')
    declared_end=struct.unpack_from('<I',audio,4)[0]+8
    if declared_end>len(audio):
        raise InvalidAudioError('audio_is_truncated')
    if declared_end!=len(audio):
        raise InvalidAudioError('invalid_wav_metadata')
    position=12
    fmt=None
    data_size=None
    while position<declared_end:
        if position+8>declared_end:
            raise InvalidAudioError('invalid_wav_metadata')
        kind=audio[position:position+4]
        size=struct.unpack_from('<I',audio,position+4)[0]
        start=position+8
        end=start+size
        if end>declared_end:
            raise InvalidAudioError('audio_is_truncated')
        if kind==b'fmt ':
            if fmt is not None or size<16:
                raise InvalidAudioError('invalid_wav_metadata')
            fmt=struct.unpack_from('<HHIIHH',audio,start)
        elif kind==b'data':
            if data_size is not None:
                raise InvalidAudioError('invalid_wav_metadata')
            data_size=size
        position=end+(size&1)
        if position>declared_end:
            raise InvalidAudioError('invalid_wav_metadata')
    if fmt is None or data_size is None:
        raise InvalidAudioError('invalid_wav_metadata')
    encoding,fmt_channels,fmt_rate,byte_rate,block_align,bits=fmt
    if encoding!=1 or bits!=16:
        raise InvalidAudioError('audio_must_be_pcm16')
    frame_size=channels*sample_width
    if (fmt_channels!=channels or fmt_rate!=sample_rate or block_align!=frame_size
            or byte_rate!=sample_rate*frame_size or data_size%frame_size
            or data_size!=frame_count*frame_size):
        raise InvalidAudioError('invalid_wav_metadata')


def validate_wav(audio: bytes) -> ValidatedWav:
    if not isinstance(audio,bytes):
        raise InvalidAudioError('invalid_wav')
    if len(audio)>MAX_WAV_BYTES:
        raise InvalidAudioError('audio_too_large')
    try:
        with wave.open(io.BytesIO(audio), "rb") as source:
            channels = source.getnchannels()
            sample_rate = source.getframerate()
            sample_width = source.getsampwidth()
            frame_count = source.getnframes()
            compression = source.getcomptype()
            frame_bytes = source.readframes(frame_count)
    except (EOFError, wave.Error):
        raise InvalidAudioError("invalid_wav") from None

    if channels != 1:
        raise InvalidAudioError("audio_must_be_mono")
    if not MIN_SAMPLE_RATE<=sample_rate<=MAX_SAMPLE_RATE:
        raise InvalidAudioError("audio_sample_rate_out_of_range")
    if sample_width != 2 or compression != "NONE":
        raise InvalidAudioError("audio_must_be_pcm16")
    if frame_count <= 0:
        raise InvalidAudioError("audio_is_empty")
    if len(frame_bytes) != frame_count * channels * sample_width:
        raise InvalidAudioError("audio_is_truncated")
    _validate_wav_metadata(audio,channels=channels,sample_rate=sample_rate,
                           sample_width=sample_width,frame_count=frame_count)
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


def normalize_wav(validated: ValidatedWav) -> ValidatedWav:
    """Resample already validated mono PCM16 input for Whisper; preserve 16k bytes."""
    if validated.sample_rate==STT_SAMPLE_RATE:
        return validated
    try:
        import numpy as np
        from scipy.signal import resample_poly

        with wave.open(io.BytesIO(validated.audio),'rb') as source:
            frame_bytes=source.readframes(validated.frame_count)
        samples=np.frombuffer(frame_bytes,dtype='<i2').astype(np.float64)/32768.0
        if samples.size!=validated.frame_count:
            raise ValueError('inconsistent validated frames')
        divisor=math.gcd(STT_SAMPLE_RATE,validated.sample_rate)
        converted=np.asarray(resample_poly(samples,STT_SAMPLE_RATE//divisor,
                                          validated.sample_rate//divisor),dtype=np.float64)
        count=(validated.frame_count*STT_SAMPLE_RATE+validated.sample_rate-1)//validated.sample_rate
        if converted.ndim!=1 or len(converted)!=count or not np.isfinite(converted).all():
            raise ValueError('invalid resampler result')
        pcm=np.rint(np.clip(converted,-1.0,32767.0/32768.0)*32768.0).astype('<i2')
        stream=io.BytesIO()
        with wave.open(stream,'wb') as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(STT_SAMPLE_RATE)
            output.writeframes(pcm.tobytes())
        return ValidatedWav(audio=stream.getvalue(),channels=1,sample_rate=STT_SAMPLE_RATE,
                            sample_width=2,frame_count=count,duration_seconds=count/STT_SAMPLE_RATE)
    except Exception:
        raise SpeechToTextUnavailableError() from None


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
                samples = np.frombuffer(source.readframes(source.getnframes()), dtype='<i2')
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
