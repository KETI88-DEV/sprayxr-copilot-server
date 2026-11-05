"""Multilingual transport preserves the Korean analysis and the original dialogue."""
from __future__ import annotations

import io
import json
import wave

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.copilot_router import create_copilot_router, install_copilot_error_handlers
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.speech_to_text import Transcript
from trainer.demo_api.viewer_context import sign_context
from trainer.demo_api.voice_router import create_voice_router

URL = "/api/v1/copilot/query"
QUESTION = "What is the recommended distance?"
KOREAN_QUESTION = "적정 이격거리는?"
ANSWER = "이격거리를 300 mm로 유지하세요."
ENGLISH = "Keep the distance at 300 mm."


class Pipeline:
    def __init__(self, *, translated_answer: str = ENGLISH) -> None:
        self.translated_answer = translated_answer
        self.calls: list[dict] = []
        self.translations: list[tuple[str, str, str]] = []

    def translate_copilot_text(self, text: str, *, source_language: str, target_language: str) -> str:
        self.translations.append((text, source_language, target_language))
        if target_language == "ko":
            return "지금 자세는 어때?" if text == "How is my current posture?" else KOREAN_QUESTION
        return self.translated_answer

    def run(self, **kwargs: object) -> dict:
        self.calls.append(kwargs)
        return {"response": "[답변] " + ANSWER + "\n[근거] DOC-distance-C0001",
                "rag_chunks": [{"chunk_id": "DOC-distance-C0001", "text": "distance"}]}

    def classify_copilot_intent(self, task_type: str, prompt: str) -> str:
        return "pose_overall"


def client(pipeline: object | None, store: SequenceStore | None = None) -> TestClient:
    app = FastAPI()
    app.include_router(create_copilot_router(
        (lambda: pipeline) if pipeline is not None else None,
        store=store if store is not None else SequenceStore(":memory:"),
    ))
    install_copilot_error_handlers(app)
    return TestClient(app)


def body(**fields: object) -> dict:
    return {"user_id": "MULTILINGUAL-TEST", "prompt": QUESTION,
            "task_type": "knowledge_search", **fields}


def test_english_question_uses_korean_retrieval_and_returns_english_with_citations() -> None:
    pipeline = Pipeline()
    response = client(pipeline).post(URL, json=body(language="en"))
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["answer"] == ENGLISH
    assert data["language"] == data["response_language"] == "en"
    assert data["retrieval_used"] is True
    assert data["citations"] == ["DOC-distance-C0001"]
    assert KOREAN_QUESTION in pipeline.calls[0]["instruction"]


def test_cross_language_response_keeps_original_question_and_final_answer_in_memory() -> None:
    store = SequenceStore(":memory:")
    question = "Khoảng cách phun phù hợp là bao nhiêu?"
    response = client(Pipeline(), store).post(URL, json=body(
        prompt=question, language="vi", response_language="en", session_id="MULTI-SESSION"))
    assert response.status_code == 200
    data = response.json()["data"]
    assert (data["language"], data["response_language"]) == ("vi", "en")
    turns = store.recent_query_turns("MULTI-SESSION", limit=4)
    assert len(turns) == 1
    assert turns[0]["prompt"] == question
    assert turns[0]["answer"] == ENGLISH


def test_legacy_korean_request_does_not_add_translation_calls() -> None:
    pipeline = Pipeline()
    data = client(pipeline).post(URL, json=body(prompt=KOREAN_QUESTION)).json()["data"]
    assert data["answer"] == ANSWER
    assert "language" not in data and "response_language" not in data
    assert pipeline.translations == []
    assert len(pipeline.calls) == 1


@pytest.mark.parametrize("language", ["ko", "en", "vi", "id", "th", "ne", "my", "uz", "si", "ru", "zh", "km"])
def test_supported_question_languages_can_request_a_korean_answer(language: str) -> None:
    response = client(Pipeline()).post(URL, json=body(language=language, response_language="ko"))
    assert response.status_code == 200
    assert response.json()["data"]["language"] == language
    assert response.json()["data"]["response_language"] == "ko"
    assert response.json()["data"]["answer"] == ANSWER


@pytest.mark.parametrize("fields", [{"language": "xx"}, {"response_language": "xx"}, {"language": ""}])
def test_unsupported_languages_return_the_existing_validation_error_wrapper(fields: dict) -> None:
    response = client(Pipeline()).post(URL, json=body(**fields))
    assert response.status_code == 422
    assert response.json()["status"] == "ERROR"
    assert response.json()["data"] is None


def test_regional_language_tags_are_normalized() -> None:
    data = client(Pipeline()).post(URL, json=body(language=" EN-us ", response_language="KO_kr")).json()["data"]
    assert data["language"] == "en"
    assert data["response_language"] == "ko"


def test_foreign_posture_question_keeps_signed_playback_scope_and_numeric_measurements() -> None:
    context = {"status": "available", "similarity_percent": 70, "zones": {
        "1:2": {"map_num": "1", "zone_id": "2", "similarity_percent": 88,
                "corrections": []},
        "1:3": {"map_num": "1", "zone_id": "3", "similarity_percent": 40,
                "corrections": []}}, "corrections": []}
    envelope = {"analysis_context": context, "signature": sign_context("pose_recomm", context),
                "playback_context": {"map_num": "1", "zone_id": "2", "playhead_seconds": 5}}
    pipeline = Pipeline(translated_answer="The work similarity in Map 1, Zone 2 is 88%.")
    response = client(pipeline).post(URL, json=body(
        prompt="How is my current posture?", language="en", task_type="pose_recomm",
        context="motion_viewer_analysis:" + json.dumps(envelope)))
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["pose_context"]["scope"] == "current_zone"
    assert set(data["pose_context"]["zones"]) == {"1:2"}
    assert data["answer"] == "The work similarity in Map 1, Zone 2 is 88%."
    assert data["retrieval_used"] is False


def test_translation_cannot_change_verified_numeric_values_or_store_a_false_answer() -> None:
    store = SequenceStore(":memory:")
    response = client(Pipeline(translated_answer="Keep the distance at 999 mm."), store).post(
        URL, json=body(language="en", session_id="BAD-TRANSLATION"))
    assert response.status_code == 503
    assert response.json()["status"] == "ERROR"
    assert store.recent_query_turns("BAD-TRANSLATION", limit=4) == []


def test_model_unavailable_is_not_misrepresented_as_a_successful_foreign_answer() -> None:
    response = client(None).post(URL, json=body(language="vi", task_type="error_analysis"))
    assert response.status_code == 503
    assert response.json()["data"] is None


def test_voice_language_reaches_text_query_and_response_language_can_differ() -> None:
    class Stt:
        def transcribe(self, audio: bytes, language: str) -> Transcript:
            return Transcript(text=QUESTION, language=language)

    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000)
        wav.writeframes(b"\x00" * 3200)
    app = FastAPI()
    app.include_router(create_voice_router(lambda: Pipeline(), stt_provider=lambda: Stt(),
                                          store=SequenceStore(":memory:")))
    install_copilot_error_handlers(app)
    response = TestClient(app).post("/api/v1/copilot/query-voice",
        data={"user_id": "MULTI", "language": "en", "response_language": "ko"},
        files={"audio": ("test.wav", stream.getvalue(), "audio/wav")})
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["transcript"] == QUESTION
    assert data["language"] == "en" and data["response_language"] == "ko"
    assert data["answer"] == ANSWER


def test_test_query_exposes_language_and_accurate_input_echo() -> None:
    app = FastAPI()
    app.include_router(create_voice_router(lambda: Pipeline(), stt_provider=lambda: None,
                                          store=SequenceStore(":memory:")))
    install_copilot_error_handlers(app)
    response = TestClient(app).post("/api/v1/test/query", params={
        "prompt": QUESTION, "language": "en", "response_language": "en"})
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["answer"] == ENGLISH
    assert data["test_context"]["input_used"]["query_parameters"]["language"] == "en"
    assert data["test_context"]["input_used"]["query_parameters"]["response_language"] == "en"


@pytest.mark.parametrize("answer", ["Keep the distance at 300 cm.", "Keep the distance at 300.", ANSWER])
def test_translation_preserves_units_and_does_not_return_unchanged_korean(answer: str) -> None:
    response = client(Pipeline(translated_answer=answer)).post(URL, json=body(language="en"))
    assert response.status_code == 503
    assert response.json()["data"] is None


def test_translated_posture_cannot_drop_the_percentage_symbol() -> None:
    context = {"status": "available", "similarity_percent": 88, "zones": {
        "1:2": {"zone_id": "2", "similarity_percent": 88, "corrections": []}}, "corrections": []}
    envelope = {"analysis_context": context, "signature": sign_context("pose_recomm", context)}
    response = client(Pipeline(translated_answer="The overall work similarity is 88. Zone 2 is the lowest at 88.")).post(
        URL, json=body(prompt="전체 작업 유사도는?", response_language="en", task_type="pose_recomm",
                       context="motion_viewer_analysis:" + json.dumps(envelope)))
    assert response.status_code == 503


@pytest.mark.parametrize("source,target,original,localized", [
    ("en", "ko", "Check SEQ-012.", "SEQ-012를 확인하세요."),
    ("ko", "en", "SEQ-012를 확인하세요.", "Check SEQ-012."),
])
def test_translation_preserves_ids_attached_to_korean_particles(source: str, target: str, original: str, localized: str) -> None:
    from trainer.demo_api.query_language import translate_text
    class Translator:
        def translate_copilot_text(self, text: str, **kwargs: str) -> str:
            return localized
    assert translate_text(Translator(), original, source_language=source, target_language=target) == localized


@pytest.mark.parametrize("original,localized", [
    ("이격거리 300 mm, 도막 80 µm", "Distance 300 µm, thickness 80 mm"),
    ("이격거리를 300 mm로 유지하세요.", "이격거리를 300 mm로 유지하세요!"),
    ("DFT ≥ 150 µm.", "DFT ≤ 150 µm."),
    ("Zone 1 → 3.", "Zone 1 ← 3."),
])
def test_translation_rejects_changed_measurement_associations_or_wrong_language(original: str, localized: str) -> None:
    from trainer.demo_api.query_language import TranslationUnavailableError, translate_text
    class Translator:
        def translate_copilot_text(self, text: str, **kwargs: str) -> str:
            return localized
    with pytest.raises(TranslationUnavailableError):
        translate_text(Translator(), original, source_language="ko", target_language="en")


def test_equivalent_micro_symbols_and_retained_korean_names_are_allowed() -> None:
    from trainer.demo_api.query_language import translate_text
    class Translator:
        def translate_copilot_text(self, text: str, **kwargs: str) -> str:
            return "전형기: keep the coating at 80 μm."
    assert translate_text(Translator(), "전형기: 도막은 80 µm입니다.", source_language="ko", target_language="en") == "전형기: keep the coating at 80 μm."



def test_vietnamese_output_cannot_be_an_english_translation() -> None:
    response = client(Pipeline(translated_answer=ENGLISH)).post(URL, json=body(language="vi"))
    assert response.status_code == 503



def test_translation_placeholders_restore_measurements_identifiers_and_directions() -> None:
    from trainer.demo_api.query_language import protect_translation, restore_translation
    original = "Zone 3: 150 µm, SEQ-012 → 3_11000346_1"
    masked, prefix, tokens = protect_translation(original)
    assert "150 µm" not in masked and "SEQ-012" not in masked
    localized = "Zona " + prefix + "0__: " + prefix + "1__, " + prefix + "2__ " + prefix + "3__ " + prefix + "4__"
    assert restore_translation(localized, prefix, tokens) == "Zona 3: 150 µm, SEQ-012 → 3_11000346_1"


@pytest.mark.parametrize("indexes", [[1, 0], [0], [0, 0, 1]])
def test_translation_cannot_drop_duplicate_or_reorder_protected_values(indexes: list[int]) -> None:
    from trainer.demo_api.query_language import TranslationUnavailableError, protect_translation, restore_translation
    masked, prefix, tokens = protect_translation("150 µm, 300 mm")
    with pytest.raises(TranslationUnavailableError):
        restore_translation(" ".join(prefix + str(i) + "__" for i in indexes), prefix, tokens)


def test_literal_placeholder_text_from_a_user_does_not_collide_with_protection() -> None:
    from trainer.demo_api.query_language import protect_translation, restore_translation
    original = "__CP_KEEP_0__ and Zone 3"
    masked, prefix, tokens = protect_translation(original)
    assert prefix != "__CP_KEEP_"
    assert restore_translation(masked, prefix, tokens) == original



def test_an_entire_zone_flow_can_be_inserted_without_retranslating_its_numbers() -> None:
    from trainer.demo_api.query_language import protect_translation, restore_translation
    masked, prefix, tokens = protect_translation("권장 순서: Zone 7 → 5 → 3 → 1")
    assert restore_translation("Recommended order: " + prefix + "0__", prefix, tokens) == "Recommended order: Zone 7 → 5 → 3 → 1"


def test_constrained_translation_requires_all_protected_values_once_in_order() -> None:
    import re
    from trainer.demo_api.query_language import translation_regex
    pattern = translation_regex("__CP_KEEP_", ["150 µm", "300 mm"])
    assert re.fullmatch(pattern, json.dumps({"text": "Thickness __CP_KEEP_0__, distance __CP_KEEP_1__."}))
    for text in ["Missing __CP_KEEP_0__.", "Reordered __CP_KEEP_1__ __CP_KEEP_0__", "Duplicate __CP_KEEP_0__ __CP_KEEP_0__ __CP_KEEP_1__"]:
        assert re.fullmatch(pattern, json.dumps({"text": text})) is None
