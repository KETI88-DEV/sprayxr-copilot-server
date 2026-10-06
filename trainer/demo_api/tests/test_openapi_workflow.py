from fastapi.testclient import TestClient

from trainer.demo_api.app import create_app


def test_openapi_describes_tsb_workflow_and_db_viewer() -> None:
    client = TestClient(create_app())
    schema = client.get("/openapi.json").json()

    assert schema["info"]["title"] == "XR Painting Simulator AI Co-Pilot API"
    assert "http://keti-xr-lab.iptime.org:58000/admin" in schema["info"]["description"]
    assert "처음 보는 개발자" in schema["info"]["description"]
    assert "실제 연동 API" in schema["info"]["description"]
    assert "누적 히트맵" in schema["info"]["description"]
    assert [tag["name"] for tag in schema["tags"]][:4] == [
        "0. Swagger 모의 연동",
        "1. 세션 관리",
        "2. 시퀀스 수집·피드백",
        "3. AI Co-Pilot 질의",
    ]
    assert client.get("/admin").status_code == 200


def test_guided_workflow_has_five_input_free_ordered_operations() -> None:
    schema = TestClient(create_app()).get("/openapi.json").json()
    expected = [
        "/api/v1/test/workflow/0-reset",
        "/api/v1/test/workflow/1-start",
        "/api/v1/test/workflow/2-sequence",
        "/api/v1/test/workflow/3-complete",
        "/api/v1/test/workflow/4-report",
    ]

    for order, path in enumerate(expected):
        operation = schema["paths"][path]["post"]
        assert operation["tags"] == ["0. Swagger 모의 연동"]
        assert operation["summary"].startswith(f"0-{order}")
        assert "requestBody" not in operation
        assert operation.get("parameters", []) == []
        assert len(operation["description"]) >= 80


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


def test_every_public_operation_explains_when_and_how_to_use_it() -> None:
    schema = TestClient(create_app()).get("/openapi.json").json()

    missing = []
    for path, path_item in schema["paths"].items():
        for method, operation in path_item.items():
            if method not in {"get", "post"}:
                continue
            if len(operation.get("description", "").strip()) < 50:
                missing.append(f"{method.upper()} {path}")

    assert missing == []
