"""Swagger output descriptions follow actual contracts without changing requests."""

from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

from trainer.demo_api.app import create_app


@pytest.fixture(scope="module")
def schema():
    return TestClient(create_app()).get("/openapi.json").json()


def _operation(schema, path, method="post"):
    return schema["paths"][path][method]


def test_every_visible_operation_has_output_fields_and_an_appropriate_response_example(schema):
    for path, operations in schema["paths"].items():
        for method, operation in operations.items():
            if method not in {"get", "post", "put", "patch", "delete"}:
                continue
            assert "### 출력 파라미터" in operation["description"], (method, path)
            assert "### 응답 예시" in operation["description"], (method, path)


def test_start_and_report_descriptions_include_deep_output_fields_and_nullable_values(schema):
    start = _operation(schema, "/api/v1/sessions/start")["description"]
    assert "data.first_zone_guidance.zone_id(string | null)" in start
    assert "data.first_zone_guidance.recommended_flow(array<string>)" in start
    assert "data.recommended_flow(array<string>)" in start
    assert "실전" in start and "null" in start
    report = _operation(schema, "/api/v1/reports/{report_id}", "get")["description"]
    assert "data.report.pose_report.priority_corrections[].correction_guidance(string)" in report
    assert "data.report.coating_report.worst_zones[].thickness_um(number | null)" in report


def test_order_alias_is_only_in_the_start_response_example(schema):
    start = _operation(schema, "/api/v1/sessions/start")["responses"]["200"]["content"]["application/json"]["example"]
    end = _operation(schema, "/api/v1/sessions/end")["responses"]["200"]["content"]["application/json"]["example"]
    assert start["data"]["recommended_flow"] == start["data"]["first_zone_guidance"]["recommended_flow"]
    assert "recommended_flow" not in end["data"]


def test_query_documents_fallbacks_dynamic_context_and_evidence_without_inventing_context_keys(schema):
    text = _operation(schema, "/api/v1/copilot/query")["description"]
    for field in ("data.answer(string)", "data.degraded(boolean)", "data.context_status(string)",
                  "data.citations(array<string>)", "data.pose_context(object | null)"):
        assert field in text
    output = text.split("### 출력 파라미터", 1)[1]
    assert "SUCCESS" in output and "분석 가능" in output
    assert "가변" in output and "degraded=true" in output
    assert "null" in output and "0점" in output


def test_sse_and_wav_outputs_are_not_described_as_json_success_responses(schema):
    stream = _operation(schema, "/api/v1/stream/anomalies/{session_id}", "get")
    output = stream["description"].split("### 출력 파라미터", 1)[1]
    assert "text/event-stream" in output and "anomaly_detection" in output and "keep-alive" in output
    assert "event: anomaly_detection" in output and "id:" in output
    assert "application/json" not in stream["responses"]["200"]["content"]
    wav = _operation(schema, "/api/v1/test/assets/query-voice-ko.wav", "get")
    assert "audio/wav" in wav["responses"]["200"]["content"]
    assert "바이너리" in wav["description"].split("### 출력 파라미터", 1)[1]
    assert '"status": "SUCCESS"' not in wav["description"].split("### 출력 파라미터", 1)[1]


@pytest.mark.parametrize("path,model", [
    ("/api/v1/copilot/query", "CopilotResponse"),
    ("/api/v1/copilot/query-voice", "VoiceCopilotResponse"),
    ("/api/v1/telemetry/sequence", "TelemetrySequenceResponse"),
    ("/api/v1/copilot/analyze-sequence", "AnalyzeSequenceResponse"),
    ("/api/v1/sessions/start", "SessionResponse"),
])
def test_new_json_examples_validate_against_the_real_response_models(schema, path, model):
    from trainer.demo_api import copilot_contracts, telemetry_contracts, v1_contracts, voice_contracts
    modules = (copilot_contracts, telemetry_contracts, v1_contracts, voice_contracts)
    contract = next(getattr(module, model) for module in modules if hasattr(module, model))
    media = _operation(schema, path)["responses"]["200"]["content"]["application/json"]
    example = media["example"]
    contract.model_validate(example)
    if "query" in path:
        assert example["data"]["retrieval_used"] is False and example["data"]["citations"] == []


def test_generic_v1_response_docs_expose_concrete_payloads_and_list_items(schema):
    complete = _operation(schema, "/api/v1/sessions/complete")["description"]
    assert "data.report_id(string)" in complete and "data.report_status(string)" in complete
    users = _operation(schema, "/api/v1/users", "get")["description"]
    assert "data(array<object>)" in users and "data[].user_id(string)" in users
    status = _operation(schema, "/api/v1/sessions/{session_id}/reports/{report_id}", "get")["description"]
    assert "pending" in status and "complete" in status and "failed" in status


def test_nullable_status_unions_explain_all_allowed_progress_and_guide_values(schema):
    output = _operation(schema, "/api/v1/copilot/analyze-sequence")["description"].split("### 출력 파라미터", 1)[1]
    assert "data.coating_progress.normal_band_percent(number | null)" in output
    fields = output.split("### 응답 예시", 1)[0]
    for value in ("needs_more", "complete", "overpaint_risk", "continue_current_zone",
                  "move_next_zone", "stop_current_zone", "slow", "fast", "near", "far"):
        assert f"`{value}`" in fields
    assert "160~200µm" in fields and "95%" in fields


def test_expert_percentages_use_valid_reference_cells_and_never_the_target_area_gate(schema):
    report = _operation(schema, "/api/v1/reports/{report_id}", "get")["description"]
    fields = report.split("### 출력 파라미터", 1)[1].split("### 응답 예시", 1)[0]
    for name, boundary in (("coverage_percent", "q1"), ("overpaint_percent", "q3"), ("completion_percent", "q1")):
        line = next(line for line in fields.splitlines() if f"expert_comparison.zones[].{name}(" in line)
        assert "유효 전문가 격자" in line and boundary in line and "보정 전" in line
        assert "공식 Zone 면적" not in line and "이동 게이트와 무관" in line
    current = _operation(schema, "/api/v1/copilot/analyze-sequence")["description"]
    current_fields = current.split("### 출력 파라미터", 1)[1].split("### 응답 예시", 1)[0]
    target_line = next(line for line in current_fields.splitlines() if "data.coating_progress.coverage_percent(" in line)
    assert "공식 Zone 면적" in target_line and "목표 두께" in target_line and "과도막" in target_line
    expert_line = next(line for line in current_fields.splitlines() if "data.coating_progress.expert_completion_percent(" in line)
    assert "유효 전문가 격자" in expert_line and "보정 전" in expert_line


def test_output_enrichment_is_idempotent_keeps_existing_examples_and_all_request_parameters():
    from trainer.demo_api.openapi_output_docs import add_openapi_output_overviews
    original = {"requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
                "parameters": [{"name": "limit", "in": "query", "schema": {"type": "integer"}}],
                "description": "기존 설명", "responses": {"200": {"content": {"application/json": {
                    "schema": {"type": "object"}, "example": {"data": {"test_context": {"evaluation_mode": "synthetic_event"}}},
                }}}}}
    document = {"paths": {"/api/v1/test/custom": {"post": deepcopy(original)}}}
    add_openapi_output_overviews(document)
    operation = document["paths"]["/api/v1/test/custom"]["post"]
    assert operation["requestBody"] == original["requestBody"] and operation["parameters"] == original["parameters"]
    assert operation["responses"]["200"]["content"]["application/json"]["example"] == original["responses"]["200"]["content"]["application/json"]["example"]
    assert "모의" in operation["description"] and "실측" in operation["description"]
    once = deepcopy(document)
    add_openapi_output_overviews(document)
    assert document == once


def test_recursive_and_untyped_schemas_are_explained_honestly_without_unbounded_expansion():
    from trainer.demo_api.openapi_output_docs import add_openapi_output_overviews
    document = {"components": {"schemas": {"Node": {"type": "object", "properties": {
        "child": {"anyOf": [{"$ref": "#/components/schemas/Node"}, {"type": "null"}]},
        "context": {"type": "object", "additionalProperties": True},
    }}}}, "paths": {"/recursive": {"get": {"responses": {"200": {"content": {"application/json": {
        "schema": {"$ref": "#/components/schemas/Node"},
    }}}}}}}}
    add_openapi_output_overviews(document)
    description = document["paths"]["/recursive"]["get"]["description"]
    assert "child(object | null)" in description and "가변" in description
    assert len(description) < 16000
