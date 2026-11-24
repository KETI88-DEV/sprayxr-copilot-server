"""Start exposes the same registered process-specific order used by Query."""
from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.sequence_analysis import SequenceAnalysisEngine
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.v1_router import create_v1_router
from trainer.demo_api.v1_store import V1Store


def _profile(root, *, mode, process, flow):
    path = root / "generated" / "profiles" / process / "learning" / f"mode_{mode:02d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"aggregate": {"zones": {}, "macro_orders": [
        {"source": "expert_zone_orders", "order": [{"zone_id": "99"}], "session_count": 99},
        {"order": [{"map_num": 1, "zone_id": zone} for zone in flow], "session_count": 3},
    ]}}), encoding="utf-8")


def _client(root):
    state = V1Store(":memory:")
    app = FastAPI()
    app.include_router(create_v1_router(
        store=state, sequences=SequenceStore(":memory:"), sequence_engine=SequenceAnalysisEngine(root),
    ))
    return TestClient(app), state


def _body(scenario="2", **overrides):
    return {
        "session_id": f"{scenario}_AI_START_ORDER_1", "user_id": "COPILOT-AI-TEST",
        "scenario_id": scenario, "started_at": "2026-11-24T01:00:00Z",
        "worker": {"skill_level": "novice"}, **overrides,
    }


@pytest.mark.parametrize("tool,process,flow", [
    (None, "spray", ["2", "3", "1"]),
    ("spray_gun", "spray", ["2", "3", "1"]),
    ("blasting_gun", "blasting", ["1", "2", "3"]),
])
def test_start_exposes_full_registered_order_without_trainee_data(tmp_path, tool, process, flow):
    _profile(tmp_path, mode=2, process="spray", flow=["2", "3", "1"])
    _profile(tmp_path, mode=2, process="blasting", flow=["1", "2", "3"])
    client, state = _client(tmp_path)
    body = _body(**({"tool_type": tool} if tool is not None else {}))
    result = client.post("/api/v1/sessions/start", json=body).json()
    assert set(result) == {"status", "code", "message", "data"}
    assert result["status"] == "SUCCESS"
    data = result["data"]
    assert data["recommended_flow"] == flow
    assert data["first_zone_guidance"]["recommended_flow"] == flow
    assert data["first_zone_id"] == data["first_zone_guidance"]["zone_id"] == flow[0]
    assert data["first_zone_guidance"]["process"] == process
    assert data["first_zone_guidance"]["reference_basis"] == "measured_expert_runs"
    assert "→".join(flow) in data["first_zone_guidance"]["message"]
    assert "→".join(flow) in result["message"]
    assert state.get_session(body["session_id"]).tool_type == (tool or "spray_gun")
    retry = client.post("/api/v1/sessions/start", json=body).json()
    assert retry["data"]["recommended_flow"] == flow


@pytest.mark.parametrize("scenario", [*[str(mode) for mode in range(1, 11)], "TR-L10", "learning-mode-10"])
def test_learning_scenarios_select_their_registered_mode_without_coercing_ids(tmp_path, scenario):
    mode = int(scenario) if scenario.isdigit() else 10
    flow = [str(mode), "1"] if mode != 1 else ["1"]
    _profile(tmp_path, mode=mode, process="spray", flow=flow)
    client, _ = _client(tmp_path)
    data = client.post("/api/v1/sessions/start", json=_body(scenario)).json()["data"]
    assert data["scenario_id"] == scenario
    assert data["recommended_flow"] == flow


@pytest.mark.parametrize("scenario", ["11", "1-R-A", "2-ABCDE", "Unit12"])
def test_unregistered_scenarios_do_not_borrow_a_learning_order(tmp_path, scenario):
    _profile(tmp_path, mode=1, process="spray", flow=["1", "2"])
    client, _ = _client(tmp_path)
    result = client.post("/api/v1/sessions/start", json=_body(scenario)).json()
    assert result["status"] == "SUCCESS"
    assert result["data"]["recommended_flow"] == []
    assert result["data"]["first_zone_id"] is None
    assert result["data"]["first_zone_guidance"]["recommended_flow"] == []
    assert result["data"]["first_zone_guidance"]["reason_code"] == "sequence_reference_unavailable"


def test_start_does_not_substitute_other_process_when_only_one_profile_exists(tmp_path):
    _profile(tmp_path, mode=2, process="blasting", flow=["1", "2", "3"])
    client, _ = _client(tmp_path)
    data = client.post("/api/v1/sessions/start", json=_body()).json()["data"]
    assert data["recommended_flow"] == []
    assert data["first_zone_guidance"]["process"] == "spray"


def test_long_order_keeps_every_id_and_a_short_display_message(tmp_path):
    flow = [str(zone) for zone in range(1, 40)]
    _profile(tmp_path, mode=10, process="spray", flow=flow)
    client, _ = _client(tmp_path)
    result = client.post("/api/v1/sessions/start", json=_body("10")).json()
    assert result["data"]["recommended_flow"] == flow
    assert result["data"]["first_zone_guidance"]["recommended_flow"] == flow
    assert "39" in result["message"] and "recommended_flow" in result["message"]
    assert len(result["message"]) <= 60


def test_changed_start_parameters_stay_a_conflict_not_a_silent_order_replacement(tmp_path):
    _profile(tmp_path, mode=2, process="spray", flow=["2", "3", "1"])
    client, state = _client(tmp_path)
    body = _body()
    assert client.post("/api/v1/sessions/start", json=body).status_code == 200
    response = client.post("/api/v1/sessions/start", json=body | {"tool_type": "blasting_gun"})
    assert response.status_code == 409
    assert response.json()["status"] == "ERROR"
    assert state.get_session(body["session_id"]).tool_type == "spray_gun"


def test_start_openapi_documents_the_direct_order_field(tmp_path):
    client, _ = _client(tmp_path)
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    field = schemas["SessionStateData"]["properties"]["recommended_flow"]
    assert field["items"]["type"] == "string"
    assert field["description"]


def test_end_response_keeps_its_existing_wire_fields(tmp_path):
    _profile(tmp_path, mode=2, process="spray", flow=["2", "3", "1"])
    client, _ = _client(tmp_path)
    body = _body()
    start = client.post("/api/v1/sessions/start", json=body).json()["data"]
    end = client.post("/api/v1/sessions/end", json={
        "session_id": body["session_id"], "user_id": body["user_id"], "scenario_id": body["scenario_id"],
        "ended_at": "2026-11-24T01:01:00Z",
    }).json()["data"]
    assert start["recommended_flow"] == ["2", "3", "1"]
    assert "recommended_flow" not in end
    assert end["session_status"] == "COMPLETED"
    assert end["first_zone_guidance"] is None
