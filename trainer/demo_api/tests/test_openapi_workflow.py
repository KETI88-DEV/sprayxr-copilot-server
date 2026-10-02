from fastapi.testclient import TestClient

from trainer.demo_api.app import create_app


def test_openapi_describes_tsb_workflow_and_db_viewer() -> None:
    client = TestClient(create_app())
    schema = client.get("/openapi.json").json()

    assert schema["info"]["title"] == "XR Painting Simulator AI Co-Pilot API"
    assert "/admin" in schema["info"]["description"]
    assert "누적 히트맵" in schema["info"]["description"]
    assert [tag["name"] for tag in schema["tags"]][:4] == [
        "0. Swagger 모의 연동",
        "1. 세션 관리",
        "2. 시퀀스 수집·피드백",
        "3. AI Co-Pilot 질의",
    ]
    assert client.get("/admin").status_code == 200


def test_query_voice_and_sequence_operations_have_examples_and_ordered_tags() -> None:
    schema = TestClient(create_app()).get("/openapi.json").json()
    paths = schema["paths"]

    text = paths["/api/v1/copilot/query"]["post"]
    voice = paths["/api/v1/copilot/query-voice"]["post"]
    analyze = paths["/api/v1/copilot/analyze-sequence"]["post"]
    assert text["tags"] == ["3. AI Co-Pilot 질의"]
    assert voice["tags"] == ["3. AI Co-Pilot 질의"]
    assert analyze["tags"] == ["2. 시퀀스 수집·피드백"]
    assert text["summary"].startswith("3-1")
    assert voice["summary"].startswith("3-2")

    voice_schema = voice["requestBody"]["content"]["multipart/form-data"]["schema"]
    assert "$ref" in voice_schema
    body_name = voice_schema["$ref"].rsplit("/", 1)[-1]
    properties = schema["components"]["schemas"][body_name]["properties"]
    assert {"audio", "user_id", "language", "use_rag"}.issubset(properties)

    query_example = schema["components"]["schemas"]["CopilotQueryRequest"]["example"]
    assert query_example["prompt"]
    analyze_example = schema["components"]["schemas"]["AnalyzeSequenceRequest"]["example"]
    assert analyze_example["sequence_info"]
