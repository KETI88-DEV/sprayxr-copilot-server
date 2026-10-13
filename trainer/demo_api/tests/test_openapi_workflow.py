from fastapi.testclient import TestClient

from trainer.demo_api.app import create_app


def test_openapi_describes_tsb_workflow_and_db_viewer() -> None:
    client = TestClient(create_app())
    schema = client.get("/openapi.json").json()

    assert schema["info"]["title"] == "XR Painting Simulator AI Co-Pilot API"
    assert "http://keti-xr-lab.iptime.org:58000/admin" in schema["info"]["description"]
    assert "http://keti-xr-lab.iptime.org:58000/motion-viewer" in schema["info"]["description"]
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
    assert client.get("/motion-viewer").status_code == 200


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
        example = operation["responses"]["200"]["content"]["application/json"]["example"]
        context = example["data"]["test_context"]
        assert context["input_used"]
        assert context["input_summary"]
        assert context["evaluation_focus"]


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
    assert analyze_example["scenario_id"] == "3"
    assert "mode_id" not in analyze_example


def test_latest_tsb_fields_are_not_required_by_the_public_contract() -> None:
    schema = TestClient(create_app()).get("/openapi.json").json()
    analyze = schema["components"]["schemas"]["AnalyzeSequenceRequest"]
    assert "mode_id" not in analyze.get("required", [])

    complete_ref = schema["paths"]["/api/v1/sessions/complete"]["post"]["requestBody"]["content"][
        "multipart/form-data"
    ]["schema"]["$ref"]
    complete = schema["components"]["schemas"][complete_ref.rsplit("/", 1)[-1]]
    assert "avg_speed_mms" not in complete["required"]
    assert "avg_distance_mm" not in complete["required"]


def test_voice_demo_is_input_free_and_links_the_downloadable_sample() -> None:
    """Catches a voice demo that still requires manual multipart form entry."""
    schema = TestClient(create_app()).get("/openapi.json").json()
    operation = schema["paths"]["/api/v1/test/query-voice"]["post"]
    asset = schema["paths"]["/api/v1/test/assets/query-voice-ko.wav"]["get"]

    assert operation["tags"] == ["0. Swagger 모의 연동"]
    assert operation["summary"].startswith("0-5")
    assert "requestBody" not in operation
    assert operation.get("parameters", []) == []
    assert "/api/v1/test/assets/query-voice-ko.wav" in operation["description"]
    voice_example = operation["responses"]["200"]["content"]["application/json"]["example"]
    voice_context = voice_example["data"]["test_context"]
    assert voice_context["input_used"]["audio"]["spoken_text"]
    assert voice_context["input_used"]["form_fields"]["language"] == "ko"
    assert voice_context["input_summary"]
    assert voice_context["evaluation_focus"]
    assert asset["summary"].startswith("0-6")
    ordered_paths = list(schema["paths"])
    expected_order = [
        "/api/v1/test/workflow/0-reset",
        "/api/v1/test/workflow/1-start",
        "/api/v1/test/workflow/2-sequence",
        "/api/v1/test/workflow/3-complete",
        "/api/v1/test/workflow/4-report",
        "/api/v1/test/query-voice",
        "/api/v1/test/assets/query-voice-ko.wav",
    ]
    assert sorted(expected_order, key=ordered_paths.index) == expected_order


def test_every_json_request_has_a_swagger_sample_and_forms_have_field_examples() -> None:
    """Catches empty Try-it-out editors that leave first-time integrators guessing."""
    schema = TestClient(create_app()).get("/openapi.json").json()
    json_paths = [
        ("/api/v1/sessions/start", "post"),
        ("/api/v1/sessions/end", "post"),
        ("/api/v1/telemetry/sequence", "post"),
        ("/api/v1/copilot/analyze-sequence", "post"),
        ("/api/v1/copilot/query", "post"),
        ("/api/v1/users", "post"),
    ]
    for path, method in json_paths:
        body = schema["paths"][path][method]["requestBody"]["content"]["application/json"]["schema"]
        model_name = body["$ref"].rsplit("/", 1)[-1]
        assert schema["components"]["schemas"][model_name].get("example"), path

    form_paths = [
        "/api/v1/copilot/query-voice",
        "/api/v1/sessions/complete",
    ]
    for path in form_paths:
        body = schema["paths"][path]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
        model_name = body["$ref"].rsplit("/", 1)[-1]
        properties = schema["components"]["schemas"][model_name]["properties"]
        for name, prop in properties.items():
            if name in {"audio", "telemetry_csv", "height_map_image"}:
                continue
            assert prop.get("examples") or "default" in prop, f"{path} {name}"


def test_every_public_url_parameter_has_an_example_or_default() -> None:
    """Catches blank path/query fields in Try-it-out for first-time users."""
    schema = TestClient(create_app()).get("/openapi.json").json()
    missing = []
    for path, path_item in schema["paths"].items():
        for method, operation in path_item.items():
            if method not in {"get", "post"}:
                continue
            for parameter in operation.get("parameters", []):
                value_schema = parameter["schema"]
                if (
                    not parameter.get("examples")
                    and "example" not in parameter
                    and not value_schema.get("examples")
                    and "example" not in value_schema
                    and "default" not in value_schema
                ):
                    missing.append(f"{method.upper()} {path}: {parameter['name']}")
    assert missing == []


def test_final_report_operation_shows_the_actual_nested_contract() -> None:
    schema = TestClient(create_app()).get("/openapi.json").json()
    example = schema["paths"]["/api/v1/reports/{report_id}"]["get"]["responses"]["200"]["content"]["application/json"]["example"]

    assert set(example["data"]["report"]) == {
        "report_id", "report_status", "coating_report", "sequence_report",
        "pose_report", "llm_briefing",
    }
    assert example["data"]["report"]["sequence_report"]["analysis_status"] == "available"


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
