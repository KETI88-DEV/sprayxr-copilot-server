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


def _walk_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)


def test_coating_demo_results_equal_the_real_tool_outputs() -> None:
    from fastapi.testclient import TestClient

    from mcp_servers.main import app

    client = TestClient(app)
    routes = {"dft_judge": "/dft_judge/evaluate", "anomaly_detection": "/anomaly_detection/detect"}
    for question in ("현재 구역 도막 두께 합부 판정해줘", "현재 구역에서 도막 이상 패턴이 감지됐어. 원인 분석해줘"):
        call = get_demo_scenario(question)["tool_calls"][0]
        response = client.post(routes[call["name"]], json=call["args"])
        assert response.status_code == 200
        assert response.json()["result"] == call["result"], call["name"]


def test_coating_demo_texts_use_current_terms() -> None:
    banned = ("B-3", "nominal 100", "±20%", "+/- 20%", "SSPC", "90/10", "confidence", "신뢰도", "근본 원인")
    for question in ("현재 구역 도막 두께 합부 판정해줘", "현재 구역에서 도막 이상 패턴이 감지됐어. 원인 분석해줘"):
        for text in _walk_strings(get_demo_scenario(question)):
            for word in banned:
                assert word not in text, (word, text)
