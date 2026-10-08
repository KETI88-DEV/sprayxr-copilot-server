from fastapi.testclient import TestClient

from trainer.demo_api.app import create_app


def test_guided_workflow_runs_end_to_end_without_parameters() -> None:
    client = TestClient(create_app())
    paths = [
        "/api/v1/test/workflow/0-reset",
        "/api/v1/test/workflow/1-start",
        "/api/v1/test/workflow/2-sequence",
        "/api/v1/test/workflow/3-complete",
        "/api/v1/test/workflow/4-report",
    ]

    payloads = []
    for path in paths:
        response = client.post(path)
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "SUCCESS"
        assert body["data"]["what_happened"]
        assert body["data"]["beginner_tip"]
        context = body["data"]["test_context"]
        assert context["input_used"]
        assert context["input_summary"]
        assert context["evaluation_focus"]
        payloads.append(body["data"])

    assert payloads[0]["next_endpoint"].endswith("/1-start")
    assert payloads[1]["session_status"] == "ACTIVE"
    assert payloads[2]["guide"]["speed"]["status"] in {"slow", "ok", "fast"}
    assert payloads[2]["guide"]["distance"]["status"] in {"near", "ok", "far"}
    assert payloads[2]["test_context"]["input_used"]["sequence_result"] == {
        "avg_speed_mms": 300,
        "avg_distance_mm": 250,
    }
    assert payloads[2]["test_context"]["input_used"]["sequence_setting"] == {
        "tool_type": "spray_gun",
        "target_speed_mms": 500,
        "target_distance_mm": 300,
    }
    assert payloads[3]["report_id"].startswith("RPT-")
    assert payloads[4]["report_id"] == payloads[3]["report_id"]
    assert len(payloads[4]["narrative_evaluation"]) >= 40


def test_each_later_step_recovers_prerequisites_for_a_beginner() -> None:
    client = TestClient(create_app())

    response = client.post("/api/v1/test/workflow/4-report")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["report_id"].startswith("RPT-")
    assert data["workflow_status"] == "complete"
