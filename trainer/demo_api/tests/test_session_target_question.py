"""Ask for the saved session target without confusing it with a measured coating."""

from types import SimpleNamespace

import pytest

from trainer.demo_api.copilot_session_context import session_coating_target_answer
from trainer.demo_api.tests.test_voice_router import _Pipeline, _Stt, _client, _post
from trainer.demo_api.v1_store import V1Store


def _session(tool_type: str = "spray_gun") -> SimpleNamespace:
    return SimpleNamespace(
        user_id="W-017",
        scenario_id="3",
        tool_type=tool_type,
        target_thickness_um=145,
        target_thickness_min_um=145,
        target_thickness_max_um=190,
        target_basis="session_target",
    )


@pytest.mark.parametrize("prompt", [
    "주막 두께 몇으로 해야 돼",
    "도막 두께를 얼마로 해야 하나요?",
    "이 훈련에서 두께를 몇으로 설정하나요?",
    "도막 두께를 어느 수치로 맞춰야 하나요?",
    "도막 두께는 몇 마이크론으로 맞춰?",
    "몇 마이크론까지 칠해야 해요?",
    "이번 스프레이 훈련은 몇 미크론을 맞추면 돼요?",
])
def test_spoken_setting_questions_use_the_saved_target_and_normal_band(prompt: str) -> None:
    result = session_coating_target_answer(prompt, _session())
    assert result is not None
    answer, scope = result
    assert "목표 도막 두께는 145µm" in answer
    assert "정상 범위는 145~190µm" in answer
    assert scope["target_thickness_um"] == 145
    assert scope["normal_band_um"] == [145, 190]


@pytest.mark.parametrize(("prompt", "expected"), [
    ("이 세션 도막은 몇에서 몇까지가 정상이에요?", "정상 도막 두께 범위는 145~190µm"),
    ("도막 정상 범위의 위쪽 끝은 몇 마이크론이에요?", "정상 도막 두께 상한은 190µm"),
    ("도막 정상 범위의 아래쪽 끝은 몇 마이크론이에요?", "정상 도막 두께 하한은 145µm"),
])
def test_spoken_setting_questions_distinguish_bounds_from_the_target(prompt: str, expected: str) -> None:
    result = session_coating_target_answer(prompt, _session())
    assert result is not None
    answer, scope = result
    assert expected in answer
    assert scope["normal_band_um"] == [145, 190]


@pytest.mark.parametrize("prompt", [
    "지금 구역의 도막 두께는 몇이야?",
    "지금 칠한 도막은 몇 마이크론이야?",
    "2번 구역에서 몇 마이크론까지 칠했어요?",
    "철판은 몇 마이크론까지 칠해야 해요?",
    "PSPC에서는 도막을 몇 마이크론까지 칠해야 해요?",
    "지금 거리는 몇 밀리미터를 맞추면 돼요?",
    "2번 구역의 목표 도막 두께는 얼마예요?",
    "다음 구역의 목표 도막 두께는 얼마예요?",
    "최종 리포트에 적힌 목표 도막 두께는 얼마예요?",
    "180µm를 측정했는데 목표 도막 두께에 맞아요?",
    "현장 도막 두께를 몇으로 정해야 해요?",
    "블라스팅 후 도장 목표 두께는 얼마예요?",
    "철판 두께를 몇으로 해야 돼?",
    "가정으로 도막 두께를 몇으로 해야 해?",
])
def test_coating_lookups_do_not_replace_measurements_reports_or_industrial_specifications(prompt: str) -> None:
    assert session_coating_target_answer(prompt, _session()) is None


@pytest.mark.parametrize("prompt", [
    "이 세션의 도막 두께를 얼마로 해야 해요?",
    "몇 마이크론까지 칠해야 해요?",
])
def test_the_current_blasting_session_has_no_spray_coating_target(prompt: str) -> None:
    result = session_coating_target_answer(prompt, _session("blasting_gun"))
    assert result is not None
    answer, scope = result
    assert "블라스팅" in answer
    assert "적용하지 않습니다" in answer
    assert "µm" not in answer
    assert scope["analysis_status"] == "not_available"
    assert "target_thickness_um" not in scope


def test_voice_query_uses_session_target_without_rewriting_the_stt_transcript() -> None:
    utterance = "주막 두께 몇으로 해야 돼"
    pipeline = _Pipeline()
    reports = SimpleNamespace(get_session=lambda _: _session())
    response = _post(_client(_Stt(utterance), pipeline, reports=reports), session_id="S-1", summarize="false")
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["transcript"] == utterance
    assert "목표 도막 두께는 145µm" in data["answer"]
    assert "정상 범위는 145~190µm" in data["answer"]
    assert data["citations"] == [] and not data["retrieval_used"]
    assert pipeline.calls == []


@pytest.mark.parametrize(("session_id", "user_id", "scenario_id", "stored", "reason"), [
    (None, "W-017", "3", False, "session_required"),
    ("S-1", "W-017", "3", False, "session_required"),
    ("S-1", "W-999", "3", True, "session_mismatch"),
    ("S-1", "W-017", "4", True, "scenario_mismatch"),
])
def test_voice_query_explains_why_the_requested_session_target_cannot_be_found(
    session_id: str | None, user_id: str, scenario_id: str, stored: bool, reason: str,
) -> None:
    prompt = "이번 훈련의 목표 도막 두께를 알려줘."
    pipeline = _Pipeline()
    reports = V1Store(":memory:")
    if stored:
        reports.start_session(session_id="S-1", user_id="W-017", scenario_id="3",
                              started_at="2026-11-25T00:00:00Z", target_thickness_um=145)
    fields = {"user_id": user_id, "scenario_id": scenario_id, "summarize": "false"}
    if session_id is not None:
        fields["session_id"] = session_id
    response = _post(_client(_Stt(prompt), pipeline, reports=reports), **fields)
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["context_status"] == "not_available"
    assert "세션" in data["answer"] and "확인" in data["answer"]
    assert "µm" not in data["answer"] and data["coating_context"] is None
    assert response.headers["x-copilot-unavailable-reason"] == reason
    assert pipeline.calls == []
