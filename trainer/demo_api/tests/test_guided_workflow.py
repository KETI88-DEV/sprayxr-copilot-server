from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.app import create_app
from trainer.demo_api.expert_guidance import ExpertGuidanceEngine
from trainer.demo_api.guided_workflow import create_guided_workflow_router
from trainer.demo_api.v1_contracts import ReportDetail
from trainer.demo_api.v1_store import V1Store


def test_workflow_resets_reuse_one_demo_trainee_with_new_sessions(tmp_path: Path) -> None:
    """Reset must start another run without creating another user or changing old records."""
    store = V1Store(":memory:")
    store.upsert_user(
        user_id="real-worker", name="실제 훈련자", skill_level="expert",
        dominant_hand="left", created_at="2000-01-01T00:00:00Z",
    )
    app = FastAPI()
    router, _ = create_guided_workflow_router(store, ExpertGuidanceEngine(tmp_path))
    app.include_router(router)
    client = TestClient(app)

    first = client.post("/api/v1/test/workflow/1-start").json()["data"]
    initial_created_at = store._conn.execute(
        "SELECT created_at FROM users WHERE user_id = ?", (first["user_id"],)
    ).fetchone()[0]
    reset = client.post("/api/v1/test/workflow/0-reset").json()["data"]
    second = client.post("/api/v1/test/workflow/1-start").json()["data"]

    assert first["user_id"] == reset["user_id"] == second["user_id"] == "COPILOT-DEMO-USER"
    assert first["session_id"] != second["session_id"] == reset["session_id"]
    users = {user.user_id: user for user in store.list_users(skip=0, limit=10)}
    assert set(users) == {"real-worker", "COPILOT-DEMO-USER"}
    assert users["COPILOT-DEMO-USER"].name == "모의훈련자"
    assert users["real-worker"].name == "실제 훈련자"
    assert users["real-worker"].skill_level == "expert"
    assert store._conn.execute(
        "SELECT created_at FROM users WHERE user_id = 'COPILOT-DEMO-USER'"
    ).fetchone()[0] == initial_created_at
    assert store._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 2


def test_workflow_preserves_existing_demo_account_creation_time(tmp_path: Path) -> None:
    store = V1Store(":memory:")
    store.upsert_user(
        user_id="COPILOT-DEMO-USER", name="모의훈련자", skill_level="novice",
        dominant_hand="right", created_at="2000-01-01T00:00:00Z",
    )
    router, workflow = create_guided_workflow_router(store, ExpertGuidanceEngine(tmp_path))
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    start = client.post("/api/v1/test/workflow/1-start").json()["data"]
    workflow.reset()
    workflow.start()

    assert start["user_id"] == "COPILOT-DEMO-USER"
    registration = start["test_context"]["input_used"]["user_registration"]
    assert registration["name"] == "모의훈련자"
    assert registration.get("created_at") in {None, "2000-01-01T00:00:00Z"}
    assert store._conn.execute(
        "SELECT created_at FROM users WHERE user_id = 'COPILOT-DEMO-USER'"
    ).fetchone()[0] == "2000-01-01T00:00:00Z"


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
    assert payloads[1]["test_context"]["input_used"]["body"]["target_thickness_um"] == 150.0
    assert payloads[2]["guide"]["speed"]["status"] in {"slow", "ok", "fast"}
    assert payloads[2]["guide"]["distance"]["status"] in {"near", "ok", "far"}
    sequence_result = payloads[2]["test_context"]["input_used"]["sequence_result"]
    assert sequence_result["avg_speed_mms"] == 300
    assert sequence_result["avg_distance_mm"] == 250
    assert sequence_result["avg_thickness_um"] == 138.91
    assert sequence_result["cur_sequence_heatmap"][0]["encoding"] == "png;base64"
    assert payloads[2]["test_context"]["input_used"]["sequence_setting"] == {
        "tool_type": "spray_gun",
        "pressure": 0.55,
        "nozzle_size": 0.021,
        "thinning_ratio": 10.0,
        "viscosity": 65.0,
        "month": 10,
    }
    assert payloads[3]["report_id"].startswith("RPT-")
    assert payloads[4]["report_id"] == payloads[3]["report_id"]
    # 모의 리포트는 정식 리포트(GET /reports/{id})와 같은 모양이다 — 안내용 키만 덧붙는다.
    guide_keys = {"test_context", "workflow_status", "step", "what_happened", "beginner_tip", "next_endpoint", "report_id"}
    detail = ReportDetail.model_validate({k: v for k, v in payloads[4].items() if k not in guide_keys})
    assert detail.report.report_id == payloads[3]["report_id"]
    assert detail.session_id == payloads[3]["session_id"]
    assert detail.report.llm_briefing
    assert "narrative_evaluation" not in payloads[4]


def test_each_later_step_recovers_prerequisites_for_a_beginner() -> None:
    client = TestClient(create_app())

    response = client.post("/api/v1/test/workflow/4-report")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["report_id"].startswith("RPT-")
    assert data["workflow_status"] == "complete"
