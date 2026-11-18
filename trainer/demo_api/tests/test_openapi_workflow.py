from typing import Any

from fastapi.testclient import TestClient

from trainer.demo_api.app import create_app


def test_openapi_describes_tsb_workflow_and_db_viewer() -> None:
    client = TestClient(create_app())
    schema = client.get("/openapi.json").json()

    assert schema["info"]["title"] == "XR Painting Simulator AI Co-Pilot API"
    assert "상단 **DB 뷰어** 메뉴" in schema["info"]["description"]
    assert "DB 뷰어 바로가기" not in schema["info"]["description"]
    assert "전문가 3D 모션 뷰어" not in schema["info"]["description"]
    assert "처음 보는 개발자" in schema["info"]["description"]
    assert "실제 연동 API" in schema["info"]["description"]
    assert "누적 히트맵" in schema["info"]["description"]
    assert [tag["name"] for tag in schema["tags"]][:4] == [
        "0. 모의 연동",
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
        assert operation["tags"] == ["0. 모의 연동"]
        assert operation["summary"].startswith(f"0-{order}")
        assert "requestBody" not in operation
        assert operation.get("parameters", []) == []
        assert len(operation["description"]) >= 80
        example = operation["responses"]["200"]["content"]["application/json"]["example"]
        context = example["data"]["test_context"]
        assert context["input_used"]
        assert context["input_summary"]
        assert context["evaluation_focus"]


def test_all_demo_swagger_examples_use_the_same_trainee() -> None:
    """Copying any demo example must preserve the same account across the walkthrough."""
    schema = TestClient(create_app()).get("/openapi.json").json()
    paths = [
        "/api/v1/test/workflow/0-reset", "/api/v1/test/workflow/1-start",
        "/api/v1/test/workflow/2-sequence", "/api/v1/test/workflow/3-complete",
        "/api/v1/test/workflow/4-report", "/api/v1/test/query", "/api/v1/test/query-voice",
    ]

    def user_ids(value: Any) -> list[str]:
        if isinstance(value, dict):
            return [child for key, child in value.items() if key == "user_id"] + [
                found for child in value.values() for found in user_ids(child)
            ]
        if isinstance(value, list):
            return [found for child in value for found in user_ids(child)]
        return []

    for path in paths:
        operation = schema["paths"][path]["post"]
        example = operation["responses"]["200"]["content"]["application/json"]["example"]
        assert set(user_ids(example)) == {"COPILOT-DEMO-USER"}, path
    start = schema["paths"]["/api/v1/test/workflow/1-start"]["post"]
    registration = start["responses"]["200"]["content"]["application/json"]["example"]["data"]["test_context"]["input_used"]["user_registration"]
    assert registration["name"] == "모의훈련자"


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
    gun_metrics = schema["components"]["schemas"]["GunMetricsIn"]
    raytexuv = gun_metrics["properties"]["raytexuv"]
    assert "2048x2048" in raytexuv["description"]
    assert "좌하단" in raytexuv["description"]
    scenario = schema["components"]["schemas"]["AnalyzeSequenceRequest"]["properties"]["scenario_id"]
    assert {"3", "1-R-A", "2-ABCDE", "Unit12"}.issubset(scenario["examples"])
    assert "전체 누적 두께" in analyze["description"]


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


def test_public_swagger_examples_use_the_latest_tsb_session_identity() -> None:
    """All real integration steps should be copyable as one coherent TSB session."""
    schema = TestClient(create_app()).get("/openapi.json").json()
    components = schema["components"]["schemas"]
    expected = {
        "session_id": "3_11000346_1",
        "user_id": "1234",
        "scenario_id": "3",
    }

    for model_name in (
        "SessionStartRequest",
        "SessionEndRequest",
        "TelemetrySequenceRequest",
        "AnalyzeSequenceRequest",
        "CopilotQueryRequest",
    ):
        example = components[model_name]["example"]
        assert {key: example[key] for key in expected} == expected, model_name

    for path in ("/api/v1/sessions/complete", "/api/v1/copilot/query-voice"):
        body = schema["paths"][path]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
        form = components[body["$ref"].rsplit("/", 1)[-1]]["properties"]
        assert form["session_id"]["examples"] == [expected["session_id"]]
        assert form["user_id"]["examples"] == [expected["user_id"]]
        assert form["scenario_id"]["examples"] == [expected["scenario_id"]]

    session_field = components["SessionStartRequest"]["properties"]["session_id"]
    assert "CSV" in session_field["description"]
    assert expected["session_id"] in session_field["examples"]


def test_session_start_documents_target_coating_thickness() -> None:
    schema = TestClient(create_app()).get("/openapi.json").json()
    start = schema["components"]["schemas"]["SessionStartRequest"]
    target = start["properties"]["target_thickness_um"]

    assert "target_thickness_um" not in start["required"]
    assert target["anyOf"][0]["exclusiveMinimum"] == 0
    assert target["anyOf"][0]["maximum"] == 20000
    assert target["examples"] == [160.0]
    assert "µm" in target["description"]
    assert start["example"]["target_thickness_um"] == 160.0


def test_every_public_input_parameter_has_a_detailed_description() -> None:
    """Swagger should explain every editable request, form, path and query field."""
    schema = TestClient(create_app()).get("/openapi.json").json()
    components = schema["components"]["schemas"]
    missing: list[str] = []
    visited: set[str] = set()

    def inspect_schema(node: dict[str, Any], location: str) -> None:
        ref = node.get("$ref")
        if ref:
            name = ref.rsplit("/", 1)[-1]
            if name in visited:
                return
            visited.add(name)
            inspect_schema(components[name], name)
            return
        for branch in node.get("anyOf", []):
            if isinstance(branch, dict):
                inspect_schema(branch, location)
        items = node.get("items")
        if isinstance(items, dict):
            inspect_schema(items, f"{location}[]")
        for name, prop in node.get("properties", {}).items():
            field = f"{location}.{name}"
            if len(prop.get("description", "").strip()) < 12:
                missing.append(field)
            inspect_schema(prop, field)

    for path, path_item in schema["paths"].items():
        for method, operation in path_item.items():
            if method not in {"get", "post"}:
                continue
            for parameter in operation.get("parameters", []):
                if len(parameter.get("description", "").strip()) < 12:
                    missing.append(f"{method.upper()} {path}.{parameter['name']}")
            request_body = operation.get("requestBody", {})
            for media in request_body.get("content", {}).values():
                inspect_schema(media.get("schema", {}), f"{method.upper()} {path}")

    assert missing == []


def test_voice_demo_is_input_free_and_links_the_downloadable_sample() -> None:
    """Catches a voice demo that still requires manual multipart form entry."""
    schema = TestClient(create_app()).get("/openapi.json").json()
    operation = schema["paths"]["/api/v1/test/query-voice"]["post"]
    text_operation = schema["paths"]["/api/v1/test/query"]["post"]
    asset = schema["paths"]["/api/v1/test/assets/query-voice-ko.wav"]["get"]

    assert text_operation["tags"] == ["0. 모의 연동"]
    assert text_operation["summary"].startswith("0-5")
    assert "requestBody" not in text_operation
    text_parameters = {item["name"]: item for item in text_operation["parameters"]}
    assert text_parameters["prompt"]["schema"]["default"]
    assert text_parameters["response_mode"]["schema"]["default"] == "quick"
    assert text_parameters["use_rag"]["schema"]["default"] == "Y"
    text_example = text_operation["responses"]["200"]["content"]["application/json"]["example"]
    assert text_example["data"]["test_context"]["input_used"]["query_parameters"]["prompt"]
    assert text_example["data"]["answer"]

    assert operation["tags"] == ["0. 모의 연동"]
    assert operation["summary"].startswith("0-6")
    assert "requestBody" not in operation
    assert all(not parameter.get("required", False) for parameter in operation.get("parameters", []))
    assert "/api/v1/test/assets/query-voice-ko.wav" in operation["description"]
    voice_example = operation["responses"]["200"]["content"]["application/json"]["example"]
    voice_context = voice_example["data"]["test_context"]
    assert voice_context["input_used"]["audio"]["spoken_text"]
    assert voice_context["input_used"]["form_fields"]["language"] == "ko"
    assert voice_context["input_summary"]
    assert voice_context["evaluation_focus"]
    assert asset["summary"].startswith("0-7")
    ordered_paths = list(schema["paths"])
    expected_order = [
        "/api/v1/test/workflow/0-reset",
        "/api/v1/test/workflow/1-start",
        "/api/v1/test/workflow/2-sequence",
        "/api/v1/test/workflow/3-complete",
        "/api/v1/test/workflow/4-report",
        "/api/v1/test/query",
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


def test_every_public_operation_shows_a_bulleted_input_parameter_overview() -> None:
    """Each Swagger operation should summarize editable inputs without opening Models."""
    schema = TestClient(create_app()).get("/openapi.json").json()

    missing: list[str] = []
    for path, path_item in schema["paths"].items():
        for method, operation in path_item.items():
            if method not in {"get", "post"}:
                continue
            if "### 입력 파라미터" not in operation.get("description", ""):
                missing.append(f"{method.upper()} {path}")
    assert missing == []

    start = schema["paths"]["/api/v1/sessions/start"]["post"]["description"]
    assert "- `scenario_id(string)` :" in start
    assert "- `session_id(string)` :" in start
    assert "- `target_thickness_um(number)` :" in start
    assert "[필수]" in start
    assert "[선택]" in start

    voice = schema["paths"]["/api/v1/copilot/query-voice"]["post"]["description"]
    assert "- `audio(file)` :" in voice
    assert "- `language(string)` :" in voice

    query = schema["paths"]["/api/v1/copilot/query"]["post"]["description"]
    assert "- `response_mode(string)` :" in query
    assert "    - 기본값: `quick`" in query
    assert "    - `quick`: Qwen3-8B 빠른 안내 모델입니다. 답변 분량은 summarize로 선택합니다." in query
    assert "    - `detailed`: Qwen3.6-27B 상세 분석 모델입니다. 답변 분량은 summarize로 선택합니다." in query
    assert "- `summarize(boolean)` :" in query
    assert "    - 기본값: `true`" in query
    assert "    - `true`: 핵심 답변을 요약합니다." in query
    assert "    - `false`: 문단·목록을 유지한 전체 답변을 제공합니다." in query
    assert "    - `trajectory_recomm`: Zone 순서와 도장 경로를 안내합니다." in query
    assert "    - 생략 시: Y로 처리하여 관련 지식 검색(RAG)을 사용합니다." in query
    assert "    - `Y`: 관련 지식과 현재 세션 문맥을 검색해 답변에 활용합니다." in query

    use_rag_schema = schema["components"]["schemas"]["CopilotQueryRequest"]["properties"]["use_rag"]
    assert use_rag_schema["default"] == "Y"

    assert "    - `worker.skill_level(string)` :" in start
    assert "    - `worker.handedness(string)` :" in start

    analyze = schema["paths"]["/api/v1/copilot/analyze-sequence"]["post"]["description"]
    assert "    - `sequence_setting.tool_type(string)` :" in analyze
    assert "    - `sequence_info[].gun_metrics.speed(number)` :" in analyze
    assert "    - `sequence_info[].gun_metrics.raytexuv(array<number>)` :" in analyze
    assert "    - `sequence_result.avg_speed_mms(number)` :" in analyze
    assert "    - `sequence_result.cur_sequence_heatmap[].map_num(integer)` :" in analyze
    assert "    - `sequence_result.cur_sequence_heatmap[].data(string)` :" in analyze

    report = schema["paths"]["/api/v1/reports/{report_id}"]["get"]["description"]
    assert "- `report_id(string)` :" in report

    no_input = schema["paths"]["/api/v1/test/workflow/0-reset"]["post"]["description"]
    assert "### 입력 파라미터\n\n- 없음" in no_input


def test_report_detail_swagger_shows_one_complete_final_report() -> None:
    """TSB 2026-10-12: 리포트 응답 예시가 없어 Replay 리포트 UI 를 설계할 수 없었다."""
    schema = TestClient(create_app()).get("/openapi.json").json()
    operation = schema["paths"]["/api/v1/reports/{report_id}"]["get"]

    example = operation["responses"]["200"]["content"]["application/json"]["example"]

    report = example["data"]["report"]
    assert report["report_status"] == "complete"
    for area in ("coating_report", "sequence_report", "pose_report"):
        assert report[area]["analysis_status"] == "available"
    assert report["llm_briefing"]
    assert example["data"]["session_id"] == "3_11000346_1"


def test_sequence_result_schema_exposes_tsb_cumulative_coating_input() -> None:
    client = TestClient(create_app())
    schema = client.get("/openapi.json").json()
    components = schema["components"]["schemas"]

    for request_name in ("TelemetrySequenceRequest", "AnalyzeSequenceRequest"):
        request_schema = components[request_name]
        result_ref = request_schema["properties"]["sequence_result"]["$ref"]
        result_schema = components[result_ref.rsplit("/", 1)[-1]]
        assert {
            "avg_speed_mms", "avg_distance_mm", "avg_thickness_um", "cur_sequence_heatmap",
        }.issubset(result_schema["properties"])
        assert "전체 누적 도막" in result_schema["properties"]["cur_sequence_heatmap"]["description"]

        heatmap = result_schema["properties"]["cur_sequence_heatmap"]["anyOf"][0]["items"]
        crop_schema = components[heatmap["$ref"].rsplit("/", 1)[-1]]
        assert {
            "map_num", "x0", "y0", "width", "height",
            "img_width", "img_height", "format", "encoding", "data",
        } == set(crop_schema["required"])

        example = request_schema["example"]["sequence_result"]
        assert example["avg_thickness_um"] == 138.91
        assert example["cur_sequence_heatmap"][0]["encoding"] == "png;base64"

    start = components["SessionStartRequest"]["example"]
    assert client.post("/api/v1/sessions/start", json=start).status_code == 200
    example = components["AnalyzeSequenceRequest"]["example"]
    response = client.post("/api/v1/copilot/analyze-sequence", json=example)
    assert response.status_code == 200, response.text
