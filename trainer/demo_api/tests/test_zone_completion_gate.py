from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from trainer.demo_api import coating_assessment
from trainer.demo_api.anomaly_stream import AnomalyBroadcaster
from trainer.demo_api.coating_progress import CoatingProgressEngine
from trainer.demo_api.sequence_analysis import SequenceAnalysisEngine
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.telemetry_contracts import AnalyzeSequenceRequest
from trainer.demo_api.telemetry_router import create_telemetry_router
from trainer.demo_api.v1_store import V1Store
from trainer.zone_engine import numerical
from trainer.zone_engine.contracts import HeightPng


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


def request(seq: int = 1, *, value: float = 171, width: int = 10, x0: int = 0,
            valid_count: int | None = None, rest: float | None = None) -> AnalyzeSequenceRequest:
    """`valid_count` 뒤의 화소는 미도장이다. `rest`를 주면 그 두께(원시 µm)로 칠한다."""
    def pixel(um: float) -> tuple[int, int, int]:
        raw = round(um / 20000 * 65025)
        return raw // 255, raw % 255, 255
    image = Image.new("RGB", (width, 10), pixel(value))
    if valid_count is not None:
        for i in range(valid_count, width * 10):
            image.putpixel((i % width, i // width), (0, 0, 0) if rest is None else pixel(rest))
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


def analyze(root: Path, req: AnalyzeSequenceRequest, session_store: Any = None) -> dict[str, Any]:
    progress = engine(root)
    app = FastAPI()
    app.include_router(create_telemetry_router(store=SequenceStore(":memory:"),
        sequence_engine=SequenceAnalysisEngine(root), coating_progress_engine=progress,
        session_store=session_store))
    return TestClient(app).post("/api/v1/copilot/analyze-sequence", json=req.model_dump(mode="json")).json()["data"]


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
    complete = progress.evaluate(request(3, value=211), target_thickness_um=200)
    assert complete.status == "complete"
    # 원시 160µm는 보정하면 150µm다.
    assert progress.evaluate(request(4, value=160)).status == "needs_more"


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
    current = request(3, value=171, width=1)
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


def test_overcoat_warning_prompts_a_temporary_pause_without_unlocking_the_next_zone(tmp_path: Path) -> None:
    result = engine(tmp_path).evaluate(request(value=300, width=5))
    assert result.status == "overpaint_risk"
    assert result.next_zone_id is None
    assert "과도막" in result.message and "잠시 분사를 멈추고 도막 분포를 확인" in result.message
    assert "부족한 구간만 보완" in result.message and "160µm 이상 면적 50.0%" in result.message


def test_overpaint_risk_follows_the_painted_mean_not_a_pixel_share(tmp_path: Path) -> None:
    progress = engine(tmp_path)
    # 30%가 두껍지만 칠한 곳 평균(190µm)은 과도막 경계 아래다.
    thick_part = progress.evaluate(request(value=200, valid_count=30))
    assert thick_part.status == "needs_more"
    assert progress.evaluate(request(2, value=251, valid_count=50)).status == "overpaint_risk"
    # 하한 이상 면적이 95% 이상이면 과도막도 완료다(2026-11-24).
    done = engine(tmp_path / "full").evaluate(request(2, value=251, valid_count=95))
    assert (done.status, done.coverage_percent) == ("complete", 95)


def test_live_crop_path_matches_the_engine_cumulative_crop_path(tmp_path: Path) -> None:
    """두 복호·누적 경로에 같은 공개 API 정상범위(160~200µm)를 적용하면 같은 판정이 나와야 한다."""
    rng = np.random.default_rng(20261112)
    code = rng.integers(0, 2200, size=(10, 10))
    pixels = np.stack([code // 255, code % 255, np.where(rng.random((10, 10)) < 0.2, 0, 255)], axis=-1)
    stream = io.BytesIO()
    Image.fromarray(pixels.astype(np.uint8), "RGB").save(stream, format="PNG")
    png = stream.getvalue()
    encoded = base64.b64encode(png).decode("ascii")
    req = request()
    req.sequence_result.cur_sequence_heatmap[0].data = encoded

    row = engine(tmp_path).evaluate(req).assessment

    spec = HeightPng.model_validate({
        "data_base64": encoded, "sha256": hashlib.sha256(png).hexdigest(), "encoding": "rg16_b8",
        "basis": "cumulative_crop", "width": 10, "height": 10, "origin": "top_left",
        "crop": {"x0": 0, "y0": 0, "width": 10, "height": 10, "atlas_width": 10, "atlas_height": 10}})
    surface = numerical.overwrite_cumulative_crops([numerical.place_crop(numerical.decode_height_png(spec), spec)])
    # PR #8 assess_zone은 상한 인자를 받지 않고 1.5배를 쓴다. 공개 API의 범위 정책은
    # 두 경로 모두 같은 어댑터에 맡겨 복호·누적 경로만 비교한다.
    quality = coating_assessment.assess_zone(
        np.where(surface.source_painted, surface.raw_um, 0), np.ones((10, 10), dtype=bool),
        160.0, min_thickness_um=160.0, max_thickness_um=200.0,
    )
    assert quality is not None
    assert quality["normal_band_um"] == [160, 200]
    assert 0 < quality["painted_pixel_count"] < 100
    assert row["grade"] == quality["grade"]
    assert row["six_band_counts"] == quality["six_band_counts"]
    assert row["unpainted_fraction"] == pytest.approx(quality["unpainted_fraction"])
    assert row["thickness_um"] == pytest.approx(quality["painted_mean_um"], abs=0.05)


@pytest.mark.parametrize(("stored_basis", "target_basis"), [
    ("default_target", "default_target"),
    ("session_target", "session_target"),
    (None, "session_target"),  # target_basis 열이 생기기 전에 만든 세션
])
def test_session_started_without_a_target_is_judged_as_the_default_target(
    tmp_path: Path, stored_basis: str | None, target_basis: str,
) -> None:
    """목표를 생략한 세션도 160.0이 저장되므로 저장된 기준으로만 가른다."""
    req = request()
    sessions = V1Store(":memory:")
    sessions.start_session_with_created(session_id=req.session_id, user_id="test", scenario_id="3",
        started_at="2026-11-12T00:00:00Z", target_thickness_um=160.0, target_basis=stored_basis)
    data = analyze(tmp_path, req, sessions)
    assert data["coating_progress"]["target_basis"] == target_basis
    assert data["coating_feedback"]["target_basis"] == target_basis
    assert data["coating_progress"]["target_thickness_um"] == 160
    assert data["coating_feedback"]["normal_band_um"] == [160, 200]
    assert "정상범위 160~200µm" in data["coating_progress"]["message"]
    assert len(data["text_guide"]) <= 60


@pytest.mark.parametrize(("sessions", "basis", "target", "band"), [
    (None, "default_target", 160, [160, 200]),
    (SimpleNamespace(get_session=lambda _: SimpleNamespace(target_thickness_um=200)), "session_target", 200, [200, 250]),
])
def test_target_basis_and_range_are_preserved_outside_the_short_guide(
    tmp_path: Path, sessions: Any, basis: str, target: int, band: list[int],
) -> None:
    data = analyze(tmp_path, request(), sessions)
    assert data["coating_progress"]["target_basis"] == basis
    assert data["coating_feedback"]["target_basis"] == basis
    assert data["coating_progress"]["target_thickness_um"] == target
    assert data["coating_feedback"]["normal_band_um"] == band
    assert f"정상범위 {band[0]}~{band[1]}µm" in data["coating_progress"]["message"]
    assert len(data["text_guide"]) <= 60


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
    assert "다음 권장 2구역" in decision.text_guide
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
    assert "1구역: 수평 도포 권장." in data["text_guide"] and "horizontal" not in data["text_guide"]
    assert "160~200µm" in data["coating_progress"]["message"]
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


def alert_client(progress: CoatingProgressEngine, root: Path) -> tuple[TestClient, AnomalyBroadcaster]:
    bus = AnomalyBroadcaster()
    app = FastAPI()
    app.include_router(create_telemetry_router(store=SequenceStore(":memory:"),
        sequence_engine=SequenceAnalysisEngine(root), coating_progress_engine=progress, broadcaster=bus))
    return TestClient(app), bus


def post(client: TestClient, req: AnalyzeSequenceRequest) -> None:
    assert client.post("/api/v1/copilot/analyze-sequence", json=req.model_dump(mode="json")).status_code == 200


def zone2(seq: int, **kwargs: Any) -> AnalyzeSequenceRequest:
    req = request(seq, **kwargs)
    req.zone_id = "2"
    req.sequence_result.cur_sequence_heatmap[0].map_num = 2
    return req


def coating_alerts(bus: AnomalyBroadcaster) -> list[Any]:
    # 순서·방향 이탈 알림이 같은 통로로 나오므로 종류로 거른다.
    return [event for event in bus.since("gate-test", last_event_id=0) if event.anomaly_type.startswith("coating_")]


def test_overcoat_alert_is_published_once_per_zone(tmp_path: Path) -> None:
    client, bus = alert_client(engine(tmp_path), tmp_path)
    post(client, request(value=300))
    alerts = coating_alerts(bus)
    assert [(alert.anomaly_type, alert.sequence_id) for alert in alerts] == [("coating_over_app", "SEQ-001")]
    assert alerts[0].payload == {
        "reason_code": "zone_grade_above",
        "message": "기본 목표 160µm 기준: 1구역 칠한 곳 평균 290µm(상한 초과·과도막). 추가 분사 중지. 2구역으로 이동하세요.",
        "zone_id": "1", "map_num": 1, "status": "over_app", "grade": "above",
        "thickness_um": 289.9, "target_thickness_um": 160.0, "target_basis": "default_target",
    }
    post(client, request(2, value=300))
    assert len(coating_alerts(bus)) == 1


def test_local_overcoat_alert_requests_temporarily_pausing_and_checking(tmp_path: Path) -> None:
    client, bus = alert_client(engine(tmp_path), tmp_path)
    response = client.post("/api/v1/copilot/analyze-sequence", json=request(value=300, width=5).model_dump(mode="json"))
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["coating_progress"]["status"] == "overpaint_risk"
    assert data["coating_progress"]["next_zone_id"] is None
    assert "일시 중지" in data["text_guide"] and "부족한 곳만 보완" in data["text_guide"]
    assert len(data["text_guide"]) <= 60
    alert, = coating_alerts(bus)
    assert alert.anomaly_type == "coating_over_app" and alert.payload["zone_id"] == "1"
    assert "잠시 분사를 멈추고 도막 분포를 확인" in alert.payload["message"]
    assert "부족한 구간만 보완" in alert.payload["message"] and "이동" not in alert.payload["message"]


def test_undercoat_alert_fires_when_the_trainee_leaves_a_below_zone(tmp_path: Path) -> None:
    client, bus = alert_client(multimap_engine(tmp_path), tmp_path)
    post(client, request(value=100))
    assert coating_alerts(bus) == []
    post(client, zone2(2))
    alerts = coating_alerts(bus)
    assert [(alert.anomaly_type, alert.sequence_id) for alert in alerts] == [("coating_under_app", "SEQ-002")]
    # 본문의 Zone은 지금 Zone(2)이 아니라 떠난 Zone(1)이다.
    assert alerts[0].payload == {
        "reason_code": "zone_grade_below",
        "message": "기본 목표 160µm 기준: 방금 떠난 1구역 칠한 곳 평균 90µm(하한 미만·저도막). "
                   "돌아가 도막 분포 확인 후 보완하세요.",
        "zone_id": "1", "map_num": 1, "status": "under_app", "grade": "below",
        # 원시 100µm는 PNG 양자화로 칠한 곳 평균 89.96µm가 된다. 저도막 행은 0.1 단위로 내린다.
        "thickness_um": 89.9, "target_thickness_um": 160.0, "target_basis": "default_target",
    }
    post(client, request(3, value=100))
    post(client, zone2(4))
    assert len(coating_alerts(bus)) == 1


def test_normal_and_unpainted_zones_publish_no_coating_alert(tmp_path: Path) -> None:
    client, bus = alert_client(multimap_engine(tmp_path), tmp_path)
    post(client, request(valid_count=0))     # Zone 1에 칠한 곳이 없다
    post(client, zone2(2))                   # 칠한 곳 없는 Zone 1을 떠난다. Zone 2는 정상
    post(client, request(3, valid_count=0))  # 정상인 Zone 2를 떠난다
    assert coating_alerts(bus) == []


def test_retried_sequence_does_not_publish_the_coating_alert_again(tmp_path: Path) -> None:
    client, bus = alert_client(engine(tmp_path), tmp_path)
    over = request(value=300)
    post(client, over)
    post(client, over)
    assert len(coating_alerts(bus)) == 1


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
    assert live.status == judgments[0].assessment["gate_status"] == "needs_more"
    assert judgments[0].assessment["coverage_percent"] == live.coverage_percent == 0
    # 목표 없이 불러도 전문가 칸이 아니라 기본 목표 160µm로 같은 게이트를 쓴다.
    unset, _ = progress.judge_zones(scenario_id="3", process="spray", mode_id=3,
        surfaces={1: surface})
    assert unset[0].assessment["target_basis"] == "default_target"
    assert unset[0].assessment["gate_status"] == "needs_more"


def test_report_session_target_cannot_complete_without_official_mask(tmp_path: Path) -> None:
    progress = engine(tmp_path)
    progress._zone_catalog = None
    judgments, reason = progress.judge_zones(scenario_id="3", process="spray", mode_id=3,
        surfaces={1: np.full((10, 10), 160, dtype=np.float32)}, target_thickness_um=160)
    assert judgments == []
    assert reason is not None


# 2026-11-12 책임자 결정: 면적 95%를 채워도 칠한 곳 평균이 목표 미만(저도막)이면 다음 Zone으로 보내지 않는다.
THIN_MEAN_GUIDE = ("정상범위 160~200µm: 1구역 충족 면적 95.0%, 칠한 곳 평균 "
                   "153µm(하한 미만·저도막). 도막 분포 확인 후 보완하세요.")


def thin_mean(seq: int = 1) -> AnalyzeSequenceRequest:
    # 95%는 보정 161µm, 나머지 5%는 보정 10µm로 칠했다 → 칠한 곳 평균 153.45µm(표기 153).
    return request(seq, valid_count=95, rest=20)


def test_area_met_but_below_mean_keeps_the_trainee_in_the_zone(tmp_path: Path) -> None:
    result = engine(tmp_path).evaluate(thin_mean())
    assert result.coverage_percent == 95
    assert result.assessment["grade"] == "below"
    assert result.assessment["gate_status"] == "needs_more"
    assert (result.status, result.recommendation, result.next_zone_id) == (
        "needs_more", "continue_current_zone", None)
    assert result.message == THIN_MEAN_GUIDE


@pytest.mark.parametrize("req", [request(valid_count=94), request(value=100)])
def test_area_shortfall_keeps_the_area_guide(tmp_path: Path, req: AnalyzeSequenceRequest) -> None:
    result = engine(tmp_path).evaluate(req)
    assert result.status == "needs_more"
    assert result.message == (
        f"정상범위 160~200µm: 1구역 160µm 이상 면적 {result.coverage_percent:.1f}%. "
        "도막 분포 확인 후 보완하세요.")


def test_area_and_mean_met_moves_to_the_next_zone(tmp_path: Path) -> None:
    result = engine(tmp_path).evaluate(request(valid_count=95))
    assert result.assessment["grade"] == "normal"
    assert (result.status, result.next_zone_id) == ("complete", "2")
    assert result.message == ("정상범위 160~200µm: 1구역 공식 면적 95% 이상 충족. "
                              "2구역으로 이동하세요.")


def test_thin_zone_guide_and_leaving_alert_agree(tmp_path: Path) -> None:
    client, bus = alert_client(multimap_engine(tmp_path), tmp_path)
    data = client.post("/api/v1/copilot/analyze-sequence", json=thin_mean().model_dump(mode="json")).json()["data"]
    assert data["coating_progress"]["status"] == "needs_more"
    assert data["coating_progress"]["next_zone_id"] is None
    assert "저도막" in data["text_guide"] and "도막 분포 확인 후 보완" in data["text_guide"]
    assert "다음" not in data["text_guide"]
    assert coating_alerts(bus) == []
    post(client, zone2(2))
    post(client, thin_mean(3))
    post(client, zone2(4))
    alerts = coating_alerts(bus)
    assert [(alert.anomaly_type, alert.sequence_id) for alert in alerts] == [("coating_under_app", "SEQ-002")]
    assert alerts[0].payload["message"] == ("기본 목표 160µm 기준: 방금 떠난 1구역 칠한 곳 평균 153µm"
                                            "(하한 미만·저도막). 돌아가 도막 분포 확인 후 보완하세요.")


@pytest.mark.parametrize(("req", "status", "next_zone"), [(thin_mean(), "needs_more", None), (request(), "complete", "2")])
def test_current_next_query_uses_the_same_gate(
    tmp_path: Path, req: AnalyzeSequenceRequest, status: str, next_zone: str | None,
) -> None:
    from trainer.demo_api.reference_order_query import current_next_answer

    progress = engine(tmp_path)
    store = SequenceStore(":memory:")
    store.record(session_key=req.session_id, sequence_id=req.sequence_id, body_sha256="query",
                 frames=0, received_at="now", body=req.model_dump_json())
    query = SimpleNamespace(session_id=req.session_id, context=None, scenario_id=None, prompt="다음 구역은?")
    text, scope, payload = current_next_answer(
        query, store, sequence_engine=SequenceAnalysisEngine(tmp_path), coating_engine=progress)
    assert (scope["status"], scope["next_zone_id"]) == (status, next_zone)
    assert payload["message"] in text
    if status == "needs_more":
        assert text.endswith(THIN_MEAN_GUIDE)


def test_report_rows_use_the_same_thin_mean_gate(tmp_path: Path) -> None:
    surface = np.full((10, 10), 171, dtype=np.float32)
    surface.flat[95:] = 20
    judgments, _ = engine(tmp_path).judge_zones(scenario_id="3", process="spray", mode_id=3, surfaces={1: surface})
    assert (judgments[0].assessment["grade"], judgments[0].assessment["gate_status"]) == ("below", "needs_more")


def test_mean_just_under_the_target_reads_the_same_below_target_number_everywhere(tmp_path: Path) -> None:
    # 99%는 원시 170µm, 1화소는 원시 158µm → 칠한 곳 평균 약 159.97µm. 반올림하면 "160µm로 목표 미만"이 된다.
    near = request(value=170, valid_count=99, rest=158)
    client, bus = alert_client(multimap_engine(tmp_path), tmp_path)
    data = client.post("/api/v1/copilot/analyze-sequence", json=near.model_dump(mode="json")).json()["data"]
    progress, feedback = data["coating_progress"], data["coating_feedback"]
    assert (progress["status"], progress["next_zone_id"], feedback["grade"]) == ("needs_more", None, "below")
    assert feedback["thickness_um"] == 159.9
    assert "칠한 곳 평균 159.9µm(하한 미만·저도막)." in progress["message"]
    assert "칠한 곳 평균 159.9µm" in feedback["message"]
    post(client, zone2(2))
    assert "칠한 곳 평균 159.9µm(하한 미만·저도막)." in coating_alerts(bus)[0].payload["message"]


def test_blasting_direction_guide_is_nominal_without_changing_direction_evidence(tmp_path: Path) -> None:
    fixture(tmp_path)
    source = tmp_path / "generated/profiles/spray/learning/mode_03.json"
    target = tmp_path / "generated/profiles/blasting/learning/mode_03.json"
    target.parent.mkdir(parents=True)
    target.write_bytes(source.read_bytes())
    req = request()
    req.sequence_setting["tool_type"] = "blasting_gun"
    req.sequence_info = AnalyzeSequenceRequest.model_validate({**req.model_dump(mode="json"),
        "sequence_info": [{"index": 1, "gun_metrics": {"raytexuv": [0, 0]}},
                          {"index": 2, "gun_metrics": {"raytexuv": [0, 100]}}]}).sequence_info
    store = SequenceStore(":memory:")
    store.record(session_key=req.session_id, sequence_id=req.sequence_id, body_sha256="blasting-direction",
                 frames=0, received_at="now", body=req.model_dump_json())
    decision = SequenceAnalysisEngine(tmp_path).evaluate(req, store)
    assert decision.feedback.reason_code == "direction_deviation"
    assert decision.text_guide == "1구역: 수평 도포 권장."
    assert decision.directions == ("horizontal", "vertical")
    assert decision.direction_guide == "1구역: 수평 방향으로 도장하세요."
