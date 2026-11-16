"""The voice HTTP boundary sends real 16kHz samples to STT, preserving its response contract."""
from __future__ import annotations

import io
import math
import wave

import pytest

from trainer.demo_api.speech_to_text import Transcript
from trainer.demo_api.tests.test_voice_router import _Pipeline, _client
from trainer.demo_api.voice_router import VOICE_QUERY_PATH


def _audio(rate: int, seconds: float = .125, *, channels: int = 1, width: int = 2) -> bytes:
    stream = io.BytesIO()
    with wave.open(stream, 'wb') as output:
        output.setnchannels(channels)
        output.setsampwidth(width)
        output.setframerate(rate)
        output.writeframes(b'\0' * (int(rate * seconds) * channels * width))
    return stream.getvalue()


class RecordingStt:
    def __init__(self):
        self.audio: list[bytes] = []

    def transcribe(self, audio: bytes, language: str) -> Transcript:
        self.audio.append(audio)
        return Transcript(text='저도막이 무엇인지 설명해줘', language=language)


def _upload(client, audio):
    return client.post(VOICE_QUERY_PATH,
        headers={'X-Copilot-Source': 'assistant-test'},
        data={'user_id': 'COPILOT-AI-TEST', 'language': 'ko'},
        files={'audio': ('question.wav', audio, 'audio/wav')})


@pytest.mark.parametrize('rate', [8_000, 22_050, 32_000, 44_100, 48_000, 96_000, 192_000])
def test_uploads_are_resampled_before_the_stt_provider_receives_them(rate):
    stt = RecordingStt()
    response = _upload(_client(stt, _Pipeline()), _audio(rate))
    assert response.status_code == 200, response.text
    assert response.json()['data']['transcript'] == '저도막이 무엇인지 설명해줘'
    with wave.open(io.BytesIO(stt.audio[0]), 'rb') as actual:
        assert actual.getframerate() == 16_000
        assert actual.getnchannels() == 1 and actual.getsampwidth() == 2
        expected = math.ceil(int(rate * .125) * 16_000 / rate)
        assert actual.getnframes() == expected
        assert len(actual.readframes(expected)) == expected * 2


def test_the_largest_supported_recording_fits_the_bounded_http_upload():
    stt = RecordingStt()
    original = _audio(192_000, 15.0)
    assert len(original) > 1_048_576
    response = _upload(_client(stt, _Pipeline()), original)
    assert response.status_code == 200, response.text
    with wave.open(io.BytesIO(stt.audio[0]), 'rb') as actual:
        assert actual.getframerate() == 16_000 and actual.getnframes() == 240_000


def test_existing_16khz_bytes_and_response_fields_are_preserved():
    stt = RecordingStt()
    client = _client(stt, _Pipeline())
    original = _audio(16_000)
    baseline = _upload(client, original)
    converted = _upload(client, _audio(48_000))
    assert baseline.status_code == converted.status_code == 200
    assert stt.audio[0] == original
    assert baseline.json()['data'].keys() == converted.json()['data'].keys()


@pytest.mark.parametrize('rate', [4_000, 192_001])
def test_out_of_range_rates_are_422_before_stt_or_query(rate):
    stt, pipeline = RecordingStt(), _Pipeline()
    response = _upload(_client(stt, pipeline), _audio(rate))
    assert response.status_code == 422
    assert '8000' in response.json()['message'] and '192000' in response.json()['message']
    assert not stt.audio and not pipeline.calls


def test_oversized_uploads_remain_413_without_starting_stt():
    stt, pipeline = RecordingStt(), _Pipeline()
    response = _upload(_client(stt, pipeline), b'x' * (6 * 1024 * 1024 + 1))
    assert response.status_code == 413
    assert not stt.audio and not pipeline.calls


def test_swagger_explains_input_rate_conversion_without_a_new_required_field():
    schema = _client(RecordingStt(), _Pipeline()).get('/openapi.json').json()
    operation = schema['paths'][VOICE_QUERY_PATH]['post']
    description = operation['description']
    assert '8~192kHz' in description and '16kHz' in description and '자동 변환' in description
    assert 'PCM16' in description and 'mono' in description and '15초' in description and '6MiB' in description
    ref = operation['requestBody']['content']['multipart/form-data']['schema']['$ref']
    body = schema['components']['schemas'][ref.rsplit('/', 1)[-1]]
    assert set(body['required']) == {'audio', 'user_id', 'language'}


def test_normalization_failure_is_503_without_calling_stt_or_disclosing_internals(monkeypatch):
    from trainer.demo_api import voice_router
    from trainer.demo_api.speech_to_text import SpeechToTextUnavailableError

    def unavailable(validated):
        raise SpeechToTextUnavailableError('private decoder internals')

    monkeypatch.setattr(voice_router, 'normalize_wav', unavailable)
    stt, pipeline = RecordingStt(), _Pipeline()
    response = _upload(_client(stt, pipeline), _audio(16_000))
    assert response.status_code == 503 and response.json()['data'] is None
    assert 'private decoder internals' not in response.text
    assert not stt.audio and not pipeline.calls


def test_reported_stt_time_includes_audio_normalization(monkeypatch):
    import time
    from trainer.demo_api import voice_router
    original = voice_router.normalize_wav

    def delayed(validated):
        time.sleep(.025)
        return original(validated)

    monkeypatch.setattr(voice_router, 'normalize_wav', delayed)
    response = _upload(_client(RecordingStt(), _Pipeline()), _audio(16_000))
    assert response.status_code == 200
    assert response.json()['data']['stt_latency_ms'] >= 20
