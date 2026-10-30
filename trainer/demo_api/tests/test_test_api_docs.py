from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.expert_guidance import ExpertGuidanceEngine
from trainer.demo_api.guided_workflow import create_guided_workflow_router
from trainer.demo_api.test_api_docs import add_test_input_overviews
from trainer.demo_api.v1_store import V1Store


def test_descriptions_show_nested_typed_inputs_and_literal_json_without_request_body() -> None:
    used = {"body": {"worker": {"skill_level": "novice"}, "target_thickness_um": 150.0,
                     "sequence_info": [{"index": 0, "gun_metrics": {"speed": 300}}]}}
    schema = {"paths": {"/api/v1/test/workflow/1-start": {"post": {
        "description": "Existing description", "responses": {"200": {"content": {"application/json": {
            "example": {"data": {"test_context": {"input_used": used, "input_summary": "자동 구성"}}}
        }}}}
    }}}}
    add_test_input_overviews(schema)
    operation = schema["paths"]["/api/v1/test/workflow/1-start"]["post"]
    text = operation["description"]
    assert "POST /api/v1/test/workflow/1-start" in text
    assert 'body.worker.skill_level (string)' in text
    assert 'body.sequence_info[0].gun_metrics.speed (integer)' in text
    assert "```json\n" + json.dumps(used, ensure_ascii=False, indent=2) in text
    assert "requestBody" not in operation
    assert "실제 요청 본문" in text and "런타임" in text
    add_test_input_overviews(schema)
    assert operation["description"] == text


def test_workflow_echoes_actual_engine_input_and_reuses_actual_start_time(tmp_path: Any, monkeypatch: Any) -> None:
    import trainer.demo_api.guided_workflow as module
    ticks = iter([f"2026-10-30T00:00:{index:02d}Z" for index in range(40)])
    monkeypatch.setattr(module, "_now", lambda: next(ticks))
    store = V1Store(":memory:")

    class CaptureGuidance(ExpertGuidanceEngine):
        request: Any = None
        def evaluate(self, request: Any) -> Any:
            self.request = request
            return super().evaluate(request)

    guidance = CaptureGuidance(tmp_path)
    app = FastAPI()
    router, _ = create_guided_workflow_router(store, guidance)
    app.include_router(router)
    client = TestClient(app)
    start = client.post("/api/v1/test/workflow/1-start").json()["data"]
    data = client.post("/api/v1/test/workflow/2-sequence").json()["data"]
    echo = dict(data["test_context"]["input_used"])
    echo.pop("equivalent_real_api")
    assert echo == guidance.request.model_dump(mode="json")
    row = store.get_session(data["session_id"])
    assert start["test_context"]["input_used"]["body"]["started_at"] == row.started_at
    assert data["test_context"]["evaluation_mode"] == "mock_fallback"
    assert client.post("/api/v1/test/workflow/3-complete").status_code == 200
    assert client.post("/api/v1/test/workflow/4-report").status_code == 200


def test_every_visible_test_api_has_pre_execution_inputs_and_preserves_transport() -> None:
    from trainer.demo_api.app import create_app
    schema = create_app().openapi()
    add_test_input_overviews(schema)
    visible = [(path, method, operation) for path, methods in schema["paths"].items()
               if path.startswith("/api/v1/test/") for method, operation in methods.items()
               if method in {"get", "post"}]
    assert len(visible) >= 11
    for path, method, operation in visible:
        text = operation["description"]
        assert f"{method.upper()} {path}" in text
        assert "```json" in text
        assert "(string)" in text
        assert "requestBody" not in operation
    sequence = schema["paths"]["/api/v1/test/workflow/2-sequence"]["post"]["description"]
    assert "sequence_info[0].time (string)" in sequence
    assert "sequence_result.cur_sequence_heatmap[0].encoding (string)" in sequence
    start = schema["paths"]["/api/v1/test/workflow/1-start"]["post"]["description"]
    assert "body.started_at (string)" in start


def test_synthetic_sequence_echoes_custom_session_and_fixed_event_inputs() -> None:
    from trainer.demo_api.v1_router import create_v1_router
    app = FastAPI()
    app.include_router(create_v1_router(store=V1Store(":memory:")))
    data = TestClient(app).post("/api/v1/test/sequence/abnormal-vertical?session_id=DOCS-TEST").json()["data"]
    used = data["test_context"]["input_used"]
    assert used["query_parameters"]["session_id"] == data["session_id"] == "DOCS-TEST"
    assert used["sequence_id"] == data["sequence_id"] == "TEST-abnormal-vertical"
    assert used["anomaly_type"] == data["anomaly_type"] == "direction_deviation"
    assert used["payload"]["message"] == data["coach_message"]
    assert data["test_context"]["evaluation_mode"] == "synthetic_event"
