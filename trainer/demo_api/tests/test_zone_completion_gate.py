from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from trainer.demo_api.anomaly_stream import AnomalyBroadcaster
from trainer.demo_api.coating_progress import CoatingProgressEngine
from trainer.demo_api.sequence_analysis import SequenceAnalysisEngine
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.telemetry_contracts import AnalyzeSequenceRequest
from trainer.demo_api.telemetry_router import create_telemetry_router


def fixture(root: Path) -> tuple[Path, Path]:
    profile = root / "generated/profiles/spray/learning/mode_03.json"
    profile.parent.mkdir(parents=True)
    profile.write_text(json.dumps({"aggregate": {
        "zones": {"1:1": {"zone_id": "1", "map_num": 1,
            "expected_direction": "horizontal", "coating": {
                "analysis_status": "available", "atlas_width": 10, "atlas_height": 10,
                "grid_size": 10, "mask_source": "tsb_color_label_map",
                "cells": {"0:0": {"q1_um": 5, "q3_um": 300}}}}},
        "macro_orders": [{"session_count": 2, "order": [
            {"map_num": 1, "zone_id": "1"}, {"map_num": 1, "zone_id": "2"}]}]
    }}), encoding="utf-8")
    catalog = root / "official"
    (catalog / "color").mkdir(parents=True)
    (catalog / "palette.csv").write_text("zone_id,hex,r,g,b\n1,#FF0000,255,0,0\n", encoding="utf-8")
    (catalog / "scenario_map.csv").write_text("scenario_id,model_key,note\n3,Learn03,test\n", encoding="utf-8")
    (catalog / "zone_rects.csv").write_text(
        "model_key,map_num,zone_id,rect_x,rect_y,rect_w,rect_h,px_x,px_y,px_w,px_h,u_min,v_min,u_max,v_max,area_percent\n"
        "Learn03,01,1,0,0,1,1,0,0,10,10,0,0,1,1,100\n", encoding="utf-8")
    Image.new("RGB", (10, 10), (255, 0, 0)).save(catalog / "color/Learn03_01.png")
    return root, catalog


def request(seq: int = 1, *, value: float = 160, width: int = 10, x0: int = 0,
            valid_count: int | None = None) -> AnalyzeSequenceRequest:
    raw = round(value / 20000 * 65025)
    image = Image.new("RGB", (width, 10), (raw // 255, raw % 255, 255))
    if valid_count is not None:
        for i in range(valid_count, width * 10): image.putpixel((i % width, i // width), (0, 0, 0))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return AnalyzeSequenceRequest.model_validate({
        "session_id": "gate-test", "user_id": "test", "scenario_id": "3", "zone_id": "1",
        "sequence_id": f"SEQ-{seq:03d}", "sequence_setting": {"tool_type": "spray_gun"},
        "sequence_info": [], "sequence_result": {"cur_sequence_heatmap": [{
            "map_num": 1, "x0": x0, "y0": 0, "width": width, "height": 10,
            "img_width": width, "img_height": 10, "encoding": "png;base64",
            "format": "v2;thickness=RG16;alpha=B8;scale_um=20000",
            "data": base64.b64encode(stream.getvalue()).decode("ascii")}]} })


def engine(root: Path) -> CoatingProgressEngine:
    ref, catalog = fixture(root)
    return CoatingProgressEngine(ref, zone_reference_root=catalog)


def test_tiny_expert_cell_cannot_unlock_official_zone(tmp_path: Path) -> None:
    result = engine(tmp_path).evaluate(request(width=1))
    assert result.status == "needs_more"
    assert result.coverage_percent == 10
    assert result.next_zone_id is None
    assert result.target_thickness_um == 160


@pytest.mark.parametrize(("count", "status"), [(94, "needs_more"), (95, "complete"), (100, "complete")])
def test_completion_requires_95_percent_of_all_official_pixels(tmp_path: Path, count: int, status: str) -> None:
    result = engine(tmp_path).evaluate(request(valid_count=count))
    assert result.status == status
    assert result.coverage_percent == count
    assert (result.next_zone_id == "2") == (status == "complete")


def test_exact_target_is_required_and_explicit_session_goal_is_respected(tmp_path: Path) -> None:
    progress = engine(tmp_path)
    below = progress.evaluate(request(value=150))
    assert below.status == "needs_more"
    assert below.coverage_percent == 0
    explicit = progress.evaluate(request(2, value=180), target_thickness_um=200)
    assert explicit.status == "needs_more"
    assert explicit.target_thickness_um == 200
    complete = progress.evaluate(request(3, value=200), target_thickness_um=200)
    assert complete.status == "complete"


def test_unknown_official_mask_fails_closed_even_with_complete_expert_cell(tmp_path: Path) -> None:
    progress = engine(tmp_path)
    progress._zone_catalog = None
    result = progress.evaluate(request())
    assert result.available is False
    assert result.next_zone_id is None


def test_replay_merges_absolute_crops_without_doubling_or_rewinding(tmp_path: Path) -> None:
    progress = engine(tmp_path)
    left = request(1, width=5, value=80)
    right = request(2, width=5, x0=5)
    latest_left = request(3, width=5)
    assert progress.evaluate_replay(right, [left]).status == "needs_more"
    result = progress.evaluate_replay(latest_left, [right, left])
    assert result.status == "complete"
    assert progress.evaluate_replay(latest_left, [left, right]) == result


def test_spray_sequence_engine_without_completion_evidence_never_recommends_next_zone(tmp_path: Path) -> None:
    fixture(tmp_path)
    store = SequenceStore(":memory:")
    req = request()
    store.record(session_key=req.session_id, sequence_id=req.sequence_id, body_sha256="fixture",
                 frames=0, received_at="now", body=req.model_dump_json())
    decision = SequenceAnalysisEngine(tmp_path).evaluate(req, store)
    assert "다음" not in (decision.text_guide or "")
    assert "현재" in decision.text_guide


def test_analyze_sequence_and_sse_use_same_gate_for_order_deviation(tmp_path: Path) -> None:
    progress = engine(tmp_path)
    sequence = SequenceAnalysisEngine(tmp_path)
    store = SequenceStore(":memory:")
    bus = AnomalyBroadcaster()
    app = FastAPI()
    app.include_router(create_telemetry_router(store=store, sequence_engine=sequence,
        coating_progress_engine=progress, broadcaster=bus))
    client = TestClient(app)
    first = request(width=1)
    data = client.post("/api/v1/copilot/analyze-sequence", json=first.model_dump(mode="json")).json()["data"]
    assert "다음" not in (data["text_guide"] or "")
    assert data["coating_progress"]["next_zone_id"] is None
    complete = request(2)
    data = client.post("/api/v1/copilot/analyze-sequence", json=complete.model_dump(mode="json")).json()["data"]
    assert data["coating_progress"]["next_zone_id"] == "2"
    assert "2" in data["text_guide"]
    duplicate = client.post("/api/v1/copilot/analyze-sequence", json=first.model_dump(mode="json")).json()["data"]
    assert duplicate["coating_progress"]["next_zone_id"] is None
    late = request(0)
    assert client.post("/api/v1/copilot/analyze-sequence", json=late.model_dump(mode="json")).status_code == 409
    assert all("다음" not in (event.payload.get("message") or "") for event in bus.since(first.session_id, last_event_id=0))


def test_missing_heatmap_suppresses_sequence_transition(tmp_path: Path) -> None:
    progress = engine(tmp_path)
    app = FastAPI()
    app.include_router(create_telemetry_router(store=SequenceStore(":memory:"),
        sequence_engine=SequenceAnalysisEngine(tmp_path), coating_progress_engine=progress))
    req = request()
    req.sequence_result.cur_sequence_heatmap = None
    data = TestClient(app).post("/api/v1/copilot/analyze-sequence", json=req.model_dump(mode="json")).json()["data"]
    assert data["coating_progress"]["available"] is False
    assert "다음" not in (data["text_guide"] or "")


def test_reference_plan_exposes_measured_flow_without_trainee_sequences(tmp_path: Path) -> None:
    fixture(tmp_path)
    plan = SequenceAnalysisEngine(tmp_path).reference_plan(scenario_id="3")
    assert plan["flow"] == ["1", "2"]
    assert plan["reference_session_count"] == 2
    assert plan["analysis_status"] == "available"
    assert SequenceAnalysisEngine(tmp_path).reference_plan(scenario_id="1-R-A") is None


def test_missing_older_replay_crop_does_not_rewind_newer_absolute_pixels(tmp_path: Path) -> None:
    progress = engine(tmp_path)
    old = request(1, value=30)
    newer = request(2)
    progress.evaluate(newer)
    current = request(3, value=160, width=1)
    result = progress.evaluate_replay(current, [newer, old])
    assert result.status == "complete"


def test_wrong_map_and_unobserved_pixels_never_unlock(tmp_path: Path) -> None:
    progress = engine(tmp_path)
    wrong = request()
    wrong.sequence_result.cur_sequence_heatmap[0].map_num = 2
    result = progress.evaluate(wrong)
    assert result.available is False
    assert result.reason == "heatmap_map_unavailable"
    unknown = progress.evaluate(request(2, valid_count=0))
    assert unknown.status == "needs_more"
    assert unknown.coverage_percent == 0


def test_overcoat_warning_targets_thick_areas_and_does_not_unlock(tmp_path: Path) -> None:
    result = engine(tmp_path).evaluate(request(value=300, width=5))
    assert result.status == "overpaint_risk"
    assert result.next_zone_id is None
    assert "과도장 영역" in result.message
    assert "미달" in result.message


def test_order_deviation_sse_has_no_conflicting_transition_when_incomplete(tmp_path: Path) -> None:
    progress = engine(tmp_path)
    profile_path = tmp_path / "generated/profiles/spray/learning/mode_03.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["aggregate"]["macro_orders"][0]["order"].insert(0, {"map_num": 1, "zone_id": "9"})
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    bus = AnomalyBroadcaster()
    app = FastAPI()
    app.include_router(create_telemetry_router(store=SequenceStore(":memory:"),
        sequence_engine=SequenceAnalysisEngine(tmp_path), coating_progress_engine=progress, broadcaster=bus))
    req = request(width=1)
    client = TestClient(app)
    data = client.post("/api/v1/copilot/analyze-sequence", json=req.model_dump(mode="json")).json()["data"]
    assert data["sequence_feedback"]["reason_code"] == "zone_skipped"
    assert "다음" not in data["text_guide"]
    events = bus.since(req.session_id, last_event_id=0)
    assert len(events) == 1
    assert events[0].payload["message"] == data["text_guide"]
    client.post("/api/v1/copilot/analyze-sequence", json=req.model_dump(mode="json"))
    assert len(bus.since(req.session_id, last_event_id=0)) == 1


def test_blasting_keeps_reference_order_without_spray_thickness_gate(tmp_path: Path) -> None:
    fixture(tmp_path)
    source = tmp_path / "generated/profiles/spray/learning/mode_03.json"
    dest = tmp_path / "generated/profiles/blasting/learning/mode_03.json"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(source.read_bytes())
    req = request()
    req.sequence_setting["tool_type"] = "blasting_gun"
    store = SequenceStore(":memory:")
    store.record(session_key=req.session_id, sequence_id=req.sequence_id, body_sha256="blasting",
                 frames=0, received_at="now", body=req.model_dump_json())
    decision = SequenceAnalysisEngine(tmp_path).evaluate(req, store)
    assert "다음 권장 구역은 2" in decision.text_guide
    assert CoatingProgressEngine(tmp_path).evaluate(req).reason == "coating_not_applicable"


def test_completed_zone_direction_guidance_has_no_stale_unknown_gate_message(tmp_path: Path) -> None:
    progress = engine(tmp_path)
    app = FastAPI()
    app.include_router(create_telemetry_router(store=SequenceStore(":memory:"),
        sequence_engine=SequenceAnalysisEngine(tmp_path), coating_progress_engine=progress))
    req = request()
    req.sequence_info = AnalyzeSequenceRequest.model_validate({**req.model_dump(mode="json"),
        "sequence_info": [{"index": 1, "gun_metrics": {"raytexuv": [0, 0]}},
                          {"index": 2, "gun_metrics": {"raytexuv": [0, 100]}}]}).sequence_info
    data = TestClient(app).post("/api/v1/copilot/analyze-sequence", json=req.model_dump(mode="json")).json()["data"]
    assert data["sequence_feedback"]["reason_code"] == "direction_deviation"
    assert "horizontal" in data["text_guide"]
    assert "확인할 수 없습니다" not in data["text_guide"]
    assert data["coating_progress"]["status"] == "complete"


def multimap_engine(root: Path) -> CoatingProgressEngine:
    ref, catalog = fixture(root)
    path = ref / "generated/profiles/spray/learning/mode_03.json"
    profile = json.loads(path.read_text(encoding="utf-8"))
    other = json.loads(json.dumps(profile["aggregate"]["zones"]["1:1"]))
    other.update(zone_id="2", map_num=2)
    profile["aggregate"]["zones"]["2:2"] = other
    path.write_text(json.dumps(profile), encoding="utf-8")
    with (catalog / "palette.csv").open("a", encoding="utf-8") as stream:
        stream.write("2,#00FF00,0,255,0\n")
    with (catalog / "zone_rects.csv").open("a", encoding="utf-8") as stream:
        stream.write("Learn03,02,2,0,0,1,1,0,0,10,10,0,0,1,1,100\n")
    Image.new("RGB", (10, 10), (0, 255, 0)).save(catalog / "color/Learn03_02.png")
    return CoatingProgressEngine(ref, zone_reference_root=catalog)


def test_all_maps_in_a_sequence_survive_warm_and_cold_replay(tmp_path: Path) -> None:
    warm = multimap_engine(tmp_path)
    first = request()
    second_map = first.sequence_result.cur_sequence_heatmap[0].model_copy(deep=True)
    second_map.map_num = 2
    first.sequence_result.cur_sequence_heatmap.append(second_map)
    second = request(2, width=1)
    second.zone_id = "2"
    second.sequence_result.cur_sequence_heatmap[0].map_num = 2
    assert warm.evaluate_replay(first, []).status == "complete"
    warm_result = warm.evaluate_replay(second, [first])
    cold = CoatingProgressEngine(tmp_path, zone_reference_root=tmp_path / "official")
    cold_result = cold.evaluate_replay(second, [first])
    assert warm_result.status == cold_result.status == "complete"
    assert warm_result.coverage_percent == cold_result.coverage_percent == 100


def test_replay_with_missing_current_heatmap_enforces_session_cap_and_evicts_metadata(tmp_path: Path) -> None:
    progress = engine(tmp_path)
    progress.evaluate(request())
    for i in range(20):
        previous = request()
        previous.session_id = f"replay-memory-{i}"
        current = request(2)
        current.session_id = previous.session_id
        current.sequence_result.cur_sequence_heatmap = None
        assert progress.evaluate_replay(current, [previous]).reason == "heatmap_unavailable"
    assert len(progress._sessions) == 16
    assert set(progress._applied) == set(progress._sessions)
    assert all(key[:3] in progress._sessions for key in progress._results)
    assert all(key[0] != "gate-test" for key in progress._results)


def test_requests_without_valid_crops_allocate_no_session_or_applied_metadata(tmp_path: Path) -> None:
    progress = engine(tmp_path)
    for i in range(20):
        previous = request()
        previous.session_id = f"empty-memory-{i}"
        previous.sequence_result.cur_sequence_heatmap[0].format = "unknown"
        current = request(2)
        current.session_id = previous.session_id
        current.sequence_result.cur_sequence_heatmap = None
        progress.evaluate_replay(current, [previous])
    assert progress._sessions == {}
    assert progress._applied == {}
    assert progress._results == {}


def test_report_session_target_uses_same_strict_zone_gate_as_live(tmp_path: Path) -> None:
    progress = engine(tmp_path)
    surface = np.full((10, 10), 150, dtype=np.float32)
    live = progress.evaluate(request(value=150), target_thickness_um=160)
    judgments, reason = progress.judge_zones(scenario_id="3", process="spray", mode_id=3,
        surfaces={1: surface}, target_thickness_um=160)
    assert reason is None
    assert live.status == judgments[0].coverage.status == "needs_more"
    assert judgments[0].coverage.coverage_percent == live.coverage_percent == 0
    historical, _ = progress.judge_zones(scenario_id="3", process="spray", mode_id=3,
        surfaces={1: surface})
    assert historical[0].coverage.status == "complete"


def test_report_session_target_cannot_complete_without_official_mask(tmp_path: Path) -> None:
    progress = engine(tmp_path)
    progress._zone_catalog = None
    judgments, reason = progress.judge_zones(scenario_id="3", process="spray", mode_id=3,
        surfaces={1: np.full((10, 10), 160, dtype=np.float32)}, target_thickness_um=160)
    assert judgments == []
    assert reason is not None
