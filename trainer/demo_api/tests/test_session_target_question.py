"""Ask for the saved session target without confusing it with a measured coating."""

from types import SimpleNamespace

import pytest

from trainer.demo_api.copilot_session_context import session_coating_target_answer
from trainer.demo_api.tests.test_voice_router import _Pipeline, _Stt, _client, _post


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
])
def test_spoken_setting_questions_use_the_saved_target_and_normal_band(prompt: str) -> None:
    result = session_coating_target_answer(prompt, _session())
    assert result is not None
    answer, scope = result
    assert "목표 도막 두께는 145µm" in answer
    assert "정상 범위는 145~190µm" in answer
    assert scope["target_thickness_um"] == 145
    assert scope["normal_band_um"] == [145, 190]


@pytest.mark.parametrize("prompt", [
    "지금 구역의 도막 두께는 몇이야?",
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


def test_the_current_blasting_session_has_no_spray_coating_target() -> None:
    result = session_coating_target_answer("이 세션의 도막 두께를 얼마로 해야 해요?", _session("blasting_gun"))
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
