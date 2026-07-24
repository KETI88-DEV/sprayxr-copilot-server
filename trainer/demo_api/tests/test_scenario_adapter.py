from __future__ import annotations

from trainer.demo_api.scenario_adapter import classify_question, get_demo_scenario


def test_sequence_question_returns_scene_source_data() -> None:
    assert classify_question("다음에 어떤 구역을 칠해야 해?") == "sequence"
    scenario = get_demo_scenario("다음에 어떤 구역을 칠해야 해?")
    assert scenario["tool_calls"][0]["name"] == "sequence_validation"
    assert scenario["xr_commands"]["next_path"]["current_zone"] == "A-1"
    assert scenario["xr_commands"]["next_path"]["next_zone"] == "A-2"


def test_unknown_question_uses_general_scenario() -> None:
    scenario = get_demo_scenario("지원하지 않는 자유 질문")
    assert scenario["tool_calls"] == []
    assert "구체적인 정보" in scenario["response"]
