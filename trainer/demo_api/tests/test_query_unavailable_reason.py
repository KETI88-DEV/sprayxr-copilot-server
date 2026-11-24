"""Unavailable query answers report the actual limitation across text and voice."""
from types import SimpleNamespace

import pytest

from trainer.demo_api.tests.test_scenario_reference_query import profile
from trainer.demo_api.tests.test_tsb_query_session_context import _client
from trainer.demo_api.tests.test_voice_router import _wav
from trainer.demo_api.tests.test_voice_session_routing import _store
from trainer.demo_api.tests.test_zone_completion_gate import engine, request


_REASON = "X-Copilot-Unavailable-Reason"
_COMPLETION = "현재 구역은 완료됐나요?"
_PLAN = "이 피도면의 작업 순서를 구역 단위로 알려주세요."


def _query(client, prompt, *, voice=False, **selectors):
    data = {"user_id": "COPILOT-AI-TEST", "task_type": "knowledge_search", **selectors}
    headers = {"X-Copilot-Source": "assistant-test"}
    if voice:
        return client.post(
            "/api/v1/copilot/query-voice", data={**data, "language": "ko"},
            files={"audio": ("test-query.wav", _wav(), "audio/wav")}, headers=headers,
        )
    return client.post("/api/v1/copilot/query", json={**data, "prompt": prompt}, headers=headers)


def _unavailable(response, reason, *phrases):
    assert response.status_code == 200
    assert response.headers[_REASON] == reason
    data = response.json()["data"]
    assert data["context_status"] == "not_available"
    assert "unavailable_reason" not in data
    assert all(phrase in data["answer"] for phrase in phrases)
    return data


@pytest.mark.parametrize("voice", [False, True])
def test_current_progress_names_missing_session_in_text_and_voice(tmp_path, voice):
    response = _query(_client(tmp_path, prompt=_COMPLETION), _COMPLETION, voice=voice)
    data = _unavailable(response, "session_required", "세션 ID", "session_id", "확인할 수 없습니다")
    assert bool(data.get("transcript")) is voice


def test_completion_without_sequence_explains_missing_measurement_even_with_start_plan(tmp_path):
    profile(tmp_path, "spray", ["2", "3", "1"])
    client = _client(tmp_path)
    response = _query(client, _COMPLETION, session_id="gate-test", scenario_id="3")
    _unavailable(response, "sequence_missing", "분사 기록", "완료 여부")
    start_guide = _query(client, "다음 순서는 뭐지?", session_id="gate-test", scenario_id="3")
    assert start_guide.status_code == 200 and _REASON not in start_guide.headers
    assert start_guide.json()["data"]["sequence_context"]["next_zone_id"] == "2"


@pytest.mark.parametrize(("case", "reason", "phrase"), [
    ("zone", "zone_unresolved", "구역 번호"),
    ("heatmap", "heatmap_unavailable", "히트맵"),
    ("scenario", "scenario_mismatch", "콘텐츠"),
    ("engine", "analysis_unavailable", "도막 분석"),
])
def test_current_progress_explains_unavailable_inputs_and_engine(tmp_path, case, reason, phrase):
    body = request()
    if case == "zone":
        body.zone_id = "-1"
    elif case == "heatmap":
        body.sequence_result.cur_sequence_heatmap = None
    client = _client(tmp_path, store=_store(body), coating=None if case == "engine" else engine(tmp_path))
    response = _query(
        client, _COMPLETION, session_id="gate-test", scenario_id="4" if case == "scenario" else "3",
    )
    _unavailable(response, reason, phrase, "완료")


def test_reference_order_names_absent_registered_plan(tmp_path):
    response = _query(_client(tmp_path), _PLAN, scenario_id="3")
    _unavailable(response, "reference_unavailable", "등록", "순서")


def test_review_names_absent_completed_report(tmp_path):
    reports = SimpleNamespace(latest_complete_analysis=lambda _: None)
    response = _query(
        _client(tmp_path, reports=reports), "이번 훈련에서 잘한 점과 고칠 점 알려줘.", session_id="started-session",
    )
    _unavailable(response, "report_unavailable", "리포트", "완료")


def test_analysis_failure_is_not_reported_as_translation_failure(tmp_path):
    def broken_engine(*_args, **_kwargs):
        raise RuntimeError("internal engine detail")

    client = _client(tmp_path, store=_store(request()), coating=SimpleNamespace(evaluate_replay=broken_engine))
    response = _query(client, _COMPLETION, session_id="gate-test", scenario_id="3")
    assert response.status_code == 500
    assert response.json() == {
        "status": "ERROR", "code": 500, "message": "Copilot query failed", "data": None,
    }
    assert "internal engine detail" not in response.text
