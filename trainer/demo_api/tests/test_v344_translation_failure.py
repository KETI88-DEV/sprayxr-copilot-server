"""질문 번역이 실패해도 원문이 손-팁 질문이면 고정 안전 답을 낸다(v3.4.4).

외국어 질문은 한국어로 옮겨 분석하는데, 그 번역이 실패하면 예전에는 503("Query translation unavailable")이었다.
원문만으로 손-팁 질문임을 알 수 있으면 503 대신 고정 답을 기존 답 번역으로 옮겨 낸다. 답 번역까지 실패하면
지금까지와 같이 503이다 — 번역 실패를 요청 언어 답처럼 보이게 하지 않는다(execute_copilot_query의 기존 규칙).
"""
from __future__ import annotations

from trainer.demo_api import safety_guard as g
from trainer.demo_api.query_language import TranslationUnavailableError
from trainer.demo_api.tests.test_v342_safety import Pipeline, _ask

VI = "Súng airless hình như bị tắc, em lấy ngón tay bịt đầu súng để kiểm tra có được không ạ?"


class FailingQuestion(Pipeline):
    """질문(→ko) 번역은 실패하고, 답(ko→외국어) 번역은 `answer_ok`일 때만 된다."""

    def __init__(self, *, answer_ok: bool = True) -> None:
        super().__init__(document="네, 괜찮습니다.", converse="네, 괜찮습니다.")
        self.answer_ok = answer_ok

    def translate_copilot_text(self, text: str, *, source_language: str, target_language: str) -> str:
        self.translations.append((text, source_language, target_language))
        if target_language == "ko" or not self.answer_ok:
            raise TranslationUnavailableError("translation failed")
        return "Đừng dùng ngón tay bịt đầu súng."


def _status(pipeline: Pipeline, prompt: str) -> tuple[int, dict]:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from trainer.demo_api.copilot_router import create_copilot_router
    from trainer.demo_api.sequence_store import SequenceStore
    app = FastAPI()
    app.include_router(create_copilot_router(lambda: pipeline, store=SequenceStore(":memory:")))
    response = TestClient(app).post("/api/v1/copilot/query", json={
        "user_id": "COPILOT-AI-TEST", "task_type": "knowledge_search", "response_mode": "quick", "prompt": prompt,
        "language": "vi", "response_language": "vi"})
    return response.status_code, response.json()


def test_failed_question_translation_still_gives_the_fixed_answer_for_hand_on_tip() -> None:
    pipeline = FailingQuestion()
    data, _ = _ask(pipeline, VI, "vi")
    assert data["answer"] == "Đừng dùng ngón tay bịt đầu súng." and data["citations"] == list(g.HIGH_PRESSURE_CITATIONS)
    assert (g.HIGH_PRESSURE_ANSWER, "ko", "vi") in pipeline.translations
    assert pipeline.calls == []  # 모델은 부르지 않았다


def test_when_the_answer_translation_also_fails_the_prepared_translation_is_used() -> None:
    # v3.4.5: 운영 우즈베크어에서 고정 답의 답 번역이 늘 실패했다. 고정 답에만 미리 옮겨 둔 문장을 쓴다.
    status, body = _status(FailingQuestion(answer_ok=False), VI)
    assert status == 200 and body["data"]["answer"] == g.fixed_answer_translation(g.HIGH_PRESSURE_ANSWER, "vi")


class FailingAnswer(Pipeline):
    """질문 번역은 되고 답 번역만 실패한다."""

    def __init__(self) -> None:
        super().__init__(document="이격거리는 300~400mm가 적정합니다.", converse="이격거리는 300~400mm가 적정합니다.")

    def translate_copilot_text(self, text: str, *, source_language: str, target_language: str) -> str:
        self.translations.append((text, source_language, target_language))
        if target_language == "ko":
            return "적정 분사 거리는 얼마인가요?"
        raise TranslationUnavailableError("translation failed")


def test_ordinary_answer_translation_failure_stays_503() -> None:
    status, body = _status(FailingAnswer(), "Khoảng cách phun phù hợp là bao nhiêu?")
    assert status == 503 and body["status"] == "ERROR"


def test_other_questions_keep_the_503_on_question_translation_failure() -> None:
    status, _ = _status(FailingQuestion(), "Khoảng cách phun phù hợp là bao nhiêu?")
    assert status == 503
