from __future__ import annotations

import io
import struct
import wave

import numpy as np
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
        (_wav(sample_rate=4_000), "audio_sample_rate_out_of_range"),
        (_wav(width=1), "audio_must_be_pcm16"),
        (_wav(seconds=15.01), "audio_too_long"),
        (b"not-a-wave", "invalid_wav"),
    ],
)
def test_rejects_audio_outside_the_public_contract(audio: bytes, reason: str) -> None:
    with pytest.raises(InvalidAudioError, match=reason):
        validate_wav(audio)


def test_rejects_wav_with_truncated_pcm_data_even_when_header_is_valid() -> None:
    audio = _wav()
    with pytest.raises(InvalidAudioError, match="audio_is_truncated"):
        validate_wav(audio[:-20])


def _sample_wav(samples, sample_rate):
    stream = io.BytesIO()
    with wave.open(stream,'wb') as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(np.asarray(samples,dtype='<i2').tobytes())
    return stream.getvalue()


@pytest.mark.parametrize('rate',[8000,11025,12345,22050,32000,44100,48000,96000,192000])
def test_original_validation_accepts_other_rates_without_altering_audio(rate):
    audio = _sample_wav(np.arange(101),rate)
    validated = validate_wav(audio)
    assert validated.audio is audio and validated.sample_rate==rate
    assert validated.frame_count==101 and validated.duration_seconds==101/rate


@pytest.mark.parametrize('rate',[7999,192001,0])
def test_input_sample_rate_has_a_bounded_supported_range(rate):
    audio = bytearray(_wav())
    struct.pack_into('<I',audio,24,rate)
    with pytest.raises(InvalidAudioError,match='audio_sample_rate_out_of_range'):
        validate_wav(bytes(audio))


def test_normalization_keeps_the_same_object_and_bytes_for_sixteen_khz():
    from trainer.demo_api.speech_to_text import normalize_wav
    validated = validate_wav(_wav())
    assert normalize_wav(validated) is validated


@pytest.mark.parametrize('rate',[8000,11025,12345,22050,32000,44100,48000,96000,192000])
def test_normalized_frame_count_and_duration_use_ceil_without_truncating_the_input(rate):
    from trainer.demo_api.speech_to_text import normalize_wav
    count=rate//10+1
    original = validate_wav(_sample_wav(np.zeros(count),rate))
    result = normalize_wav(original)
    expected=(count*16000+rate-1)//rate
    assert result.sample_rate==16000 and result.channels==1 and result.sample_width==2
    assert result.frame_count==expected and result.duration_seconds==expected/16000
    assert original.frame_count==count and original.duration_seconds==count/rate
    with wave.open(io.BytesIO(result.audio),'rb') as output:
        assert output.getframerate()==16000 and output.getnframes()==expected
        assert len(output.readframes(expected))==expected*2
    assert 0<=result.duration_seconds-original.duration_seconds<1/16000


@pytest.mark.parametrize('rate',[8000,44100,48000,192000])
def test_normalization_preserves_a_speech_band_tone_and_its_amplitude(rate):
    from trainer.demo_api.speech_to_text import normalize_wav
    samples=np.rint(12000*np.sin(2*np.pi*1000*np.arange(rate//2)/rate))
    result=normalize_wav(validate_wav(_sample_wav(samples,rate)))
    with wave.open(io.BytesIO(result.audio),'rb') as output:
        converted=np.frombuffer(output.readframes(output.getnframes()),dtype='<i2').astype(float)
    steady=converted[160:-160]
    frequencies=np.fft.rfftfreq(len(steady),d=1/16000)
    assert frequencies[np.argmax(abs(np.fft.rfft(steady)))]==pytest.approx(1000,abs=5)
    assert np.sqrt(np.mean(steady**2))==pytest.approx(12000/np.sqrt(2),rel=.02)


def test_downsampling_attenuates_above_nyquist_audio_instead_of_aliasing_it():
    from trainer.demo_api.speech_to_text import normalize_wav
    def converted_rms(frequency):
        samples=np.rint(12000*np.sin(2*np.pi*frequency*np.arange(24000)/48000))
        result=normalize_wav(validate_wav(_sample_wav(samples,48000)))
        with wave.open(io.BytesIO(result.audio),'rb') as output:
            converted=np.frombuffer(output.readframes(output.getnframes()),dtype='<i2').astype(float)[160:-160]
        return np.sqrt(np.mean(converted**2))
    assert converted_rms(12000)<.02*converted_rms(4000)


def test_resampling_clips_overshoot_before_pcm16_conversion(monkeypatch):
    from trainer.demo_api.speech_to_text import normalize_wav
    monkeypatch.setattr('scipy.signal.resample_poly',lambda *args,**kwargs:np.array([1.5,-1.5,.5,-.5]))
    result=normalize_wav(validate_wav(_sample_wav([32767,-32768],8000)))
    with wave.open(io.BytesIO(result.audio),'rb') as output:
        converted=np.frombuffer(output.readframes(output.getnframes()),dtype='<i2')
    np.testing.assert_array_equal(converted,[32767,-32768,16384,-16384])


@pytest.mark.parametrize('converted',[np.array([np.nan,0,0,0]),np.array([np.inf,0,0,0]),np.zeros(3),np.zeros((2,2))])
def test_invalid_resampler_results_are_stt_unavailable_not_corrupted_wav(monkeypatch,converted):
    from trainer.demo_api.speech_to_text import normalize_wav,SpeechToTextUnavailableError
    monkeypatch.setattr('scipy.signal.resample_poly',lambda *args,**kwargs:converted)
    with pytest.raises(SpeechToTextUnavailableError):
        normalize_wav(validate_wav(_sample_wav([1,2],8000)))


def test_duration_is_checked_before_rounding_or_resampling():
    audio=_sample_wav(np.zeros(44100*15+1),44100)
    with pytest.raises(InvalidAudioError,match='audio_too_long'):
        validate_wav(audio)


@pytest.mark.parametrize('offset,format_,value',[(28,'<I',1),(32,'<H',4),(4,'<I',1)])
def test_inconsistent_riff_or_pcm_metadata_is_rejected(offset,format_,value):
    audio=bytearray(_wav())
    struct.pack_into(format_,audio,offset,value)
    with pytest.raises(InvalidAudioError):
        validate_wav(bytes(audio))


def test_a_partial_pcm16_frame_is_not_silently_discarded():
    audio=bytearray(_wav()+b'\x00')
    struct.pack_into('<I',audio,4,len(audio)-8)
    struct.pack_into('<I',audio,40,len(audio)-44)
    with pytest.raises(InvalidAudioError,match='invalid_wav_metadata'):
        validate_wav(bytes(audio))


def test_input_byte_limit_is_enforced_before_reading_a_claimed_wav():
    from trainer.demo_api.speech_to_text import MAX_WAV_BYTES
    assert MAX_WAV_BYTES==6*1024*1024
    with pytest.raises(InvalidAudioError,match='audio_too_large'):
        validate_wav(b'\x00'*(MAX_WAV_BYTES+1))
