"""complete 의 세션 도막 분석과, 그 결과로 답하는 pose_recomm 구역 질문.

구역 경계는 저장소에 있는 **실제 TSB 라벨맵**(Learn03 = scenario 3)이다. 두께와 프로필은 합성한다 —
Spark 의 전문가 프로필은 이 저장소에 없고, 정답을 아는 입력이어야 판정이 맞는지 볼 수 있다.
두께는 원시 값이다. 판정은 10µm를 뺀 보정 두께로 한다(원시 180 = 보정 170).
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from trainer.demo_api.coating_progress import CoatingProgressEngine
from trainer.demo_api.copilot_router import create_copilot_router
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.session_analysis import csv_sequences
from trainer.demo_api.session_coating import analyze_session_coating
from trainer.demo_api.v1_router import create_v1_router
from trainer.demo_api.v1_store import V1Store
from trainer.zone_reference import ZoneReferenceCatalog

SIZE = 2048
GRID = 64
# 정답(기본 목표 160µm): 1번 구역은 정상(보정 170), 2번은 안 칠함(등급 없음), 3번은 과도막(보정 390).
THICKNESS = {"1": 180.0, "2": 0.0, "3": 400.0}
SPRAY = {"sequence_setting": {"tool_type": "spray_gun"}}


def _mask(zone_id: str) -> np.ndarray:
    """좌상단 원점(높이맵 PNG와 같은 방향)."""
    image = ZoneReferenceCatalog.default().zone_mask("Learn03", 1, zone_id)
    assert image is not None
    return np.asarray(image) > 0


def _cells(zone_id: str) -> dict[str, dict[str, float]]:
    """구역과 겹치는 64격자 셀마다 전문가 기준 q1=80·q3=120µm(셀 좌표는 좌하단 원점)."""
    step = SIZE // GRID
    touched = np.flipud(_mask(zone_id)).reshape(GRID, step, GRID, step).any(axis=(1, 3))
    return {
        f"{gx}:{gy}": {"q1_um": 80.0, "median_um": 100.0, "q3_um": 120.0}
        for gy, gx in zip(*np.nonzero(touched))
    }


def _profile(root: Path) -> None:
    target = root / "generated" / "profiles" / "spray" / "learning"
    target.mkdir(parents=True)
    zones = {
        f"1:{zone_id}": {
            "map_num": 1,
            "zone_id": zone_id,
            "coating": {
                "analysis_status": "available",
                "mask_source": "tsb_color_label_map",
                "atlas_width": SIZE,
                "atlas_height": SIZE,
                "grid_size": GRID,
                "cells": _cells(zone_id),
            },
        }
        for zone_id in THICKNESS
    }
    (target / "mode_03.json").write_text(json.dumps({"aggregate": {"zones": zones}}), encoding="utf-8")


def _thickness(values: dict[str, float]) -> np.ndarray:
    surface = np.zeros((SIZE, SIZE), dtype=np.float64)
    for zone_id, value in values.items():
        surface[_mask(zone_id)] = value
    return surface


def _png(thickness: np.ndarray, *, valid: np.ndarray | None = None) -> bytes:
    """TSB RG16 인코딩(B는 유효 화소)."""
    raw = np.round(thickness / 20000.0 * 65025.0).astype(np.int64)
    rgba = np.zeros((*thickness.shape, 4), dtype=np.uint8)
    rgba[..., 0] = raw // 255
    rgba[..., 1] = raw % 255
    rgba[..., 2] = 255 if valid is None else np.where(valid, 255, 0)
    rgba[..., 3] = 255
    stream = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(stream, format="PNG")
    return stream.getvalue()


def _crop(png: bytes, *, x0: int, y0: int, width: int, height: int) -> dict[str, Any]:
    return {
        "map_num": 1, "x0": x0, "y0": y0, "width": width, "height": height,
        "img_width": width, "img_height": height,
        "format": "v2;thickness=RG16;alpha=B8;scale_um=20000",
        "encoding": "png;base64", "data": base64.b64encode(png).decode("ascii"),
    }


@pytest.fixture(scope="module")
def height_map() -> bytes:
    return _png(_thickness(THICKNESS))


@pytest.fixture
def engine(tmp_path: Path) -> CoatingProgressEngine:
    _profile(tmp_path)
    return CoatingProgressEngine(tmp_path)


def test_height_map_zones_are_graded_against_the_target(engine: CoatingProgressEngine, height_map: bytes) -> None:
    """정상·칠한 곳 없음·과도막이 라벨맵 구역별로 갈린다. 분포는 칠한 화소 수다.

    깨뜨리는 법: 높이맵을 뒤집지 않으면(좌상단↔좌하단) 마스크와 두께가 어긋나 1번이 정상이 아니게 된다.
    """
    result = analyze_session_coating(
        scenario_id="3", bodies=[SPRAY], engine=engine, height_map=height_map, height_map_name="Map01_x_Height.png"
    )

    zones = {zone["zone_id"]: zone for zone in result.zones}
    assert {zone_id: zone["status"] for zone_id, zone in zones.items()} == {
        "1": "normal", "2": None, "3": "over_app", "4": None, "5": None,
        "6": None, "7": None, "8": None, "9": None,
    }
    assert zones["1"]["thickness_um"] == pytest.approx(170.0, abs=0.5)
    assert zones["3"]["thickness_um"] == pytest.approx(390.0, abs=0.5)
    report = result.report
    assert report["analysis_status"] == "available"
    assert report["source_artifact_id"] == "Map01_x_Height.png"
    assert report["comparison_basis"] == "session_target"
    assert (report["target_thickness_um"], report["target_basis"]) == (160.0, "default_target")
    assert report["algorithm_version"].startswith("coating-session-v2/")
    pixels = {zone_id: int(_mask(zone_id).sum()) for zone_id in THICKNESS}
    assert report["distribution"] == {"below": 0, "in_band": pixels["1"], "above": pixels["3"]}
    # 보정 390µm는 목표의 2.0~2.5배 구간이다.
    assert report["zone_summary"]["over2"] > 0
    assert sum(report["zone_summary"].values()) == pytest.approx(1.0)
    assert [zone["zone_id"] for zone in report["worst_zones"]] == ["3"]
    assert report["worst_zones"][0]["thickness_um"] == pytest.approx(390.0, abs=0.5)
    for phrase in ("기본 목표 160µm 기준", "과도막 1개(3번)", "칠한 곳 없음 7개(2·4·5·6·7·8·9번)"):
        assert phrase in report["guideline"]


def test_explicit_session_target_changes_the_grade_and_the_basis(engine: CoatingProgressEngine, height_map: bytes) -> None:
    result = analyze_session_coating(
        scenario_id="3", bodies=[SPRAY], engine=engine, height_map=height_map, target_thickness_um=400.0
    )

    zone3 = next(zone for zone in result.zones if zone["zone_id"] == "3")
    assert zone3["status"] == "under_app"
    assert result.report["target_basis"] == "session_target"
    assert "세션 목표 400µm 기준" in result.report["guideline"]


def test_zone_rows_carry_the_area_gate_next_to_the_grade(engine: CoatingProgressEngine, height_map: bytes) -> None:
    result = analyze_session_coating(scenario_id="3", bodies=[SPRAY], engine=engine, height_map=height_map)

    zone1 = next(zone for zone in result.zones if zone["zone_id"] == "1")
    assert (zone1["coverage_percent"], zone1["gate_status"], zone1["grade"]) == (100.0, "complete", "normal")


def test_report_evaluates_painted_and_unpainted_official_zones_missing_from_the_expert_profile(
    engine: CoatingProgressEngine,
) -> None:
    """프로필은 1~3번뿐이어도 최종 높이맵의 공식 1~9번을 빠짐없이 평가한다."""
    result = analyze_session_coating(
        scenario_id="3", bodies=[SPRAY], engine=engine, height_map=_png(_thickness({"9": 180.0}))
    )
    zones = {zone["zone_id"]: zone for zone in result.zones}

    assert set(zones) == {str(zone) for zone in range(1, 10)}
    assert (zones["9"]["grade"], zones["9"]["gate_status"]) == ("normal", "complete")
    assert zones["9"]["thickness_um"] == pytest.approx(170.0, abs=0.5)
    assert all(zones[str(zone)]["unpainted_fraction"] == 1.0 for zone in range(1, 9))
    assert all(zones[str(zone)]["expert"] is None for zone in range(4, 10))
    assert result.report["expert_comparison"]["zone_count"] == 3


@pytest.mark.parametrize("scenario", ["3", "TR-L03", "learning-mode-3"])
def test_target_assessment_needs_official_zones_and_height_but_not_expert_samples(
    tmp_path: Path, height_map: bytes, scenario: str
) -> None:
    result = analyze_session_coating(
        scenario_id=scenario, bodies=[SPRAY], engine=CoatingProgressEngine(tmp_path), height_map=height_map
    )

    assert result.report["analysis_status"] == "available"
    assert result.report["comparison_basis"] == "session_target"
    assert len(result.zones) == 9
    assert result.report["expert_comparison"]["analysis_status"] == "unavailable"
    assert result.report["expert_comparison"]["zone_count"] == 0
    assert all(zone["expert"] is None for zone in result.zones)


def test_final_assessment_does_not_invent_unpainted_data_for_a_map_without_a_surface(tmp_path: Path) -> None:
    from trainer.demo_api.tests.test_zone_completion_gate import multimap_engine

    judgments, reason = multimap_engine(tmp_path).judge_zones(
        scenario_id="3", process="spray", mode_id=3, surfaces={1: np.full((10, 10), 180.0)}
    )

    assert reason is None
    assert [(zone.map_num, zone.zone_id) for zone in judgments] == [(1, "1")]


def test_height_map_pixels_without_blue_support_are_unpainted(engine: CoatingProgressEngine) -> None:
    """높이맵의 B 채널이 0인 화소는 RG에 값이 있어도 칠한 자리가 아니다.

    깨뜨리는 법: B를 보지 않고 RG만 읽으면 3번이 과도막으로 나온다.
    """
    height_map = _png(_thickness(THICKNESS), valid=_mask("1"))

    result = analyze_session_coating(scenario_id="3", bodies=[SPRAY], engine=engine, height_map=height_map)

    zones = {zone["zone_id"]: zone for zone in result.zones}
    assert zones["1"]["status"] == "normal"
    for zone_id in ("2", "3"):
        assert zones[zone_id]["status"] is None and zones[zone_id]["unpainted_fraction"] == 1.0


def test_worst_zone_carries_adjustment_candidates_without_claiming_a_cause(
    engine: CoatingProgressEngine, height_map: bytes
) -> None:
    bodies = [{**SPRAY, "zone_id": "3", "sequence_result": {"avg_speed_mms": 400}}]

    result = analyze_session_coating(scenario_id="3", bodies=bodies, engine=engine, height_map=height_map)

    worst = result.report["worst_zones"][0]
    assert worst["zone_id"] == "3"
    assert "이동 속도를 높이는 쪽" in worst["action"] and "원인 미확정" in worst["action"]
    assert worst["cause"] == "원인 후보: 이동 속도 (확정 아님)"
    zone3 = next(zone for zone in result.zones if zone["zone_id"] == "3")
    assert zone3["adjustments"][0]["evidence_kind"] == "quality_rule_candidate"


def test_csv_restored_sequences_keep_measured_behavior_for_coating_advice(
    engine: CoatingProgressEngine, height_map: bytes
) -> None:
    """CSV 분사 속도·거리 실측이 있어도 복원 본문의 요약을 비우면 도막 조절 후보가 사라진다."""
    rows = [
        {"분사": "On", "Zone ID": "3", "Pelvis Position X (World)": "0",
         "한길숏건 Position X": str(index), "한길숏건 Position Y": "0", "한길숏건 Position Z": "0",
         "Speed (Real)": speed, "Distance (Real)": distance, "Ray Texture Map Number": "nan"}
        for index, (speed, distance) in enumerate((("100", "200"), ("300", "400"), ("nan", "")))
    ]
    rows.append({**rows[0], "분사": "Off", "Speed (Real)": "9999", "Distance (Real)": "9999"})
    bodies = csv_sequences(rows, session_id="3_CSV_BEHAVIOR_1", user_id="u", scenario_id="3")

    result = analyze_session_coating(scenario_id="3", bodies=bodies, engine=engine, height_map=height_map)

    assert bodies[0]["sequence_result"] == {"avg_speed_mms": 200.0, "avg_distance_mm": 300.0}
    assert bodies[0]["sequence_info"][2]["gun_metrics"]["speed"] is None
    assert bodies[0]["sequence_info"][0]["gun_metrics"]["raytexmapnum"] is None
    zone3 = next(zone for zone in result.zones if zone["zone_id"] == "3")
    assert {item["variable"] for item in zone3["adjustments"]} == {"speed_mm_s", "distance_mm"}
    assert all(item["evidence_kind"] == "quality_rule_candidate" for item in zone3["adjustments"])
    assert "확정 아님" in result.report["worst_zones"][0]["cause"]


def test_without_a_height_map_later_crops_overwrite_earlier_ones(engine: CoatingProgressEngine, height_map: bytes) -> None:
    """높이맵이 없으면 시퀀스 크롭을 순서대로 덮어쓴다. 크롭 값은 누적 두께라 더하지 않는다.

    깨뜨리는 법: 더하면 3번 구역이 400+180=580µm 로 여전히 과도막이다.
    """
    zone3 = _mask("3")
    rows, columns = np.nonzero(zone3)
    top, bottom, left, right = rows.min(), rows.max() + 1, columns.min(), columns.max() + 1
    fix = _png(np.full((bottom - top, right - left), 180.0), valid=zone3[top:bottom, left:right])
    bodies = [
        {**SPRAY, "sequence_result": {"cur_sequence_heatmap": [_crop(height_map, x0=0, y0=0, width=SIZE, height=SIZE)]}},
        {**SPRAY, "sequence_result": {"cur_sequence_heatmap": [
            # 크롭 y0 는 좌하단 원점이다.
            _crop(fix, x0=int(left), y0=int(SIZE - bottom), width=int(right - left), height=int(bottom - top))
        ]}},
    ]

    result = analyze_session_coating(scenario_id="3", bodies=bodies, engine=engine)

    assert {zone["zone_id"]: zone["status"] for zone in result.zones} == {
        "1": "normal", "2": None, "3": "normal", "4": None, "5": None,
        "6": None, "7": None, "8": None, "9": None,
    }
    assert result.report["source_artifact_id"] == "cur_sequence_heatmap"


@pytest.mark.parametrize(
    ("scenario", "bodies", "with_profile", "reason"),
    [
        ("Unit3", [SPRAY], True, "reference_context_unavailable"),
        ("3", [{"sequence_setting": {"tool_type": "blasting"}}], True, "coating_not_applicable"),
        ("3", [SPRAY], False, "heatmap_unavailable"),
        ("3", [SPRAY], True, "heatmap_unavailable"),
    ],
)
def test_missing_inputs_are_unavailable_without_numbers(
    tmp_path: Path, scenario: str, bodies: list[dict[str, Any]], with_profile: bool, reason: str
) -> None:
    if with_profile:
        _profile(tmp_path)
    result = analyze_session_coating(
        scenario_id=scenario, bodies=bodies, engine=CoatingProgressEngine(tmp_path)
    )

    assert result.report["analysis_status"] == "unavailable"
    assert result.report["reason_code"] == reason
    assert "distribution" not in result.report and result.zones == []


# --- complete → report → query --------------------------------------------------------------

SESSION = "3_COATING_1"
SCORES = {"1": 0.9, "3": 0.4}


def _fake_mcp(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    zone_id = str(body["zone_id"])
    corrections = [] if zone_id == "1" else [{"joint": "RightUpperArm", "evidence": {"distance_m": 0.12}}]
    result = {"analysis_status": "available", "similarity_score": SCORES[zone_id], "corrections": corrections}
    return httpx.Response(200, json={"status": "success", "result": result})


def _client(engine: CoatingProgressEngine, reports: V1Store | None = None) -> TestClient:
    sequences = SequenceStore(":memory:")
    reports = reports if reports is not None else V1Store(":memory:")
    for number, zone_id in ((1, "1"), (2, "3"), (3, "3")):
        body = {
            **SPRAY, "session_id": SESSION, "user_id": "u", "scenario_id": "3", "zone_id": zone_id,
            "sequence_id": f"SEQ-{number:03d}", "sequence_info": [{"index": 1}], "sequence_result": {},
        }
        sequences.record(
            session_key=SESSION, sequence_id=body["sequence_id"], body_sha256=body["sequence_id"],
            frames=1, received_at="2026-10-15T00:00:00Z", body=json.dumps(body),
        )
    transport = httpx.MockTransport(_fake_mcp)
    app = FastAPI()
    app.include_router(create_v1_router(
        store=reports, sequences=sequences, mcp_base_url="http://mcp", transport=transport, coating_engine=engine,
    ))
    app.include_router(create_copilot_router(
        None, store=sequences, reports=reports, mcp_base_url="http://mcp", transport=transport,
    ))
    return TestClient(app)


def _ask(client: TestClient, prompt: str, *, session_id: str = SESSION, context: str | None = None, mode: str = "quick") -> dict[str, Any]:
    body = {"user_id": "u", "session_id": session_id, "scenario_id": "3", "task_type": "pose_recomm", "prompt": prompt,
            "response_mode": mode}
    if context is not None:
        body["context"] = context
    response = client.post("/api/v1/copilot/query", json=body)
    assert response.status_code == 200, response.text
    return response.json()["data"]


def test_after_complete_pose_recomm_names_the_most_problematic_zone(engine: CoatingProgressEngine, height_map: bytes) -> None:
    """complete 가 자세·도막을 구역별로 남기고, pose_recomm 이 그것으로 "가장 문제 있는 구역"에 답한다.

    깨뜨리는 법: 구역 질문을 시퀀스 채점으로 보내면 답이 마지막 시퀀스 이야기가 되고 3번 구역을 말하지 않는다.
    """
    client = _client(engine)
    before = _ask(client, "가장 문제 있는 구역이 어디야?")
    assert before["context_status"] == "available"
    assert before["pose_context"]["source"] == "live_sequences"
    assert before["pose_context"]["scope"] == "session"  # 세션 전체를 묻는 질문은 올라온 분사 전체를 본다
    assert before["answer"].startswith("지금까지 올라온 분사 기준입니다.")

    response = client.post(
        "/api/v1/sessions/complete",
        data={"session_id": SESSION, "user_id": "u", "scenario_id": "3", "vr_score": "60", "is_pass": "true",
              "avg_thickness_um": "120"},
        files={
            "telemetry_csv": ("t.csv", b"index,time\n1,0\n", "text/csv"),
            "height_map_image": ("Map01_3_COATING_1_Height.png", height_map, "image/png"),
        },
    )
    assert response.status_code == 200, response.text
    report_id = response.json()["data"]["report_id"]
    report = client.get(f"/api/v1/reports/{report_id}").json()["data"]["report"]
    assert report["coating_report"]["analysis_status"] == "available"
    worst = report["coating_report"]["worst_zones"][0]
    assert f"Zone {worst['zone_id']}" in report["llm_briefing"]
    assert f"{worst['thickness_um']:g}µm" in report["llm_briefing"]
    assert "160~200µm" in report["llm_briefing"]
    assert report["coating_report"]["guideline"] not in report["llm_briefing"]
    assert [zone["zone_id"] for zone in report["pose_report"]["zone_summary"]] == ["1", "3"]

    quick = _ask(client, "가장 문제 있는 구역이 어디야?")["answer"]
    worst = _ask(client, "가장 문제 있는 구역이 어디야?", mode="detailed")

    assert worst["context_status"] == "available"
    assert worst["answer"].startswith("가장 문제가 큰 구역은 3번 구역입니다.")
    assert quick == worst["answer"]
    assert "평균 자세 유사도는 40%로 세션 평균 57%보다 낮습니다" in worst["answer"]
    assert "오른쪽 위팔 위치가 숙련자 범위를 벗어난 분사가 2회 중 2회였습니다(평균 12cm)" in worst["answer"]
    assert "과도막" not in worst["answer"]
    pose_context = worst["pose_context"]
    assert pose_context["scope"] == "zone" and pose_context["report_id"] == report_id and pose_context["zone_id"] == "3"
    saved_coating = pose_context["coating"]
    assert saved_coating["thickness_um"] == pytest.approx(390, abs=.3)
    assert saved_coating["thickness_um"] == report["coating_report"]["worst_zones"][0]["thickness_um"]
    assert (saved_coating["status"], saved_coating["target_basis"]) == ("over_app", "default_target")

    zone1 = _ask(client, "이 구역 자세 어땠어?", context="1", mode="detailed")
    assert zone1["answer"].startswith("1번 구역 자세 유사도는 ")
    assert "숙련자 범위를 벗어난 관절은 없었습니다" in zone1["answer"]
    assert "두께" not in zone1["answer"]

    # 시퀀스를 가리키면 예전처럼 그 시퀀스를 채점한다.
    sequence = _ask(client, "이 분사 어땠어?", context="SEQ-002")
    assert sequence["pose_context"]["sequence_id"] == "SEQ-002"
    assert "scope" not in sequence["pose_context"]


@pytest.mark.parametrize(
    ("prompt", "context", "zone"),
    [
        ("가장 문제 있는 구역이 어디야?", None, True),
        ("어느 쪽 자세가 제일 안 좋았어?", None, True),
        ("방금 분사 자세 어땠어?", None, False),
        ("기존 자세랑 비교해 줘", None, False),  # "존" 단독은 구역 질문이 아니다
        ("이 구역 자세 어땠어?", "SEQ-012", False),  # 시퀀스를 가리키면 시퀀스 질문
        ("자세 어땠어?", "7", True),  # 구역 번호
    ],
)
def test_zone_questions_are_recognized(prompt: str, context: str | None, zone: bool) -> None:
    from trainer.demo_api.pose_query import asks_about_zones

    assert asks_about_zones(prompt, context) is zone


def _analysis(*zones: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    return "RPT-1", {"pose_report": {"average_similarity_score": 0.6, "zone_summary": list(zones)}}


def _zone(zone_id: str, score: float, rate: float | None = None) -> dict[str, Any]:
    joints = [] if rate is None else [{
        "joint": "LeftHand", "occurrence_count": 1, "occurrence_rate": rate, "average_distance": None, "angle_issues": [],
    }]
    return {"zone_id": zone_id, "evaluated_sequence_count": 2, "average_similarity_score": score, "problem_joints": joints}


def test_equal_similarity_is_broken_by_correction_rate_then_zone_order() -> None:
    from trainer.demo_api.pose_query import answer_zone_question

    tied = answer_zone_question(_analysis(_zone("2", 0.5), _zone("5", 0.5, rate=0.5), _zone("1", 0.7)), context=None)
    assert tied.context["zone_id"] == "5"
    assert "왼쪽 손목 위치가 숙련자 범위를 벗어난 분사가 2회 중 1회였습니다." in tied.answer

    plain = answer_zone_question(_analysis(_zone("4", 0.5), _zone("2", 0.5)), context=None)
    assert plain.context["zone_id"] == "2"


def test_a_zone_without_scored_sequences_is_not_answered_with_numbers() -> None:
    from trainer.demo_api.pose_query import answer_zone_question

    result = answer_zone_question(_analysis(_zone("1", 0.7)), context="12")

    assert result.analysis_status == "not_available" and result.reason_code == "zone_not_evaluated"
    assert result.answer == "12번 구역은 이 세션에서 자세를 채점한 분사가 없습니다."
    assert "similarity_score" not in result.context


def test_without_an_expert_reference_the_briefing_still_uses_official_target_grades(tmp_path: Path, height_map: bytes) -> None:
    """전문가 표본이 없어도 공식 Zone·높이맵의 목표 판정과 자세 결과는 보존한다."""
    client = _client(CoatingProgressEngine(tmp_path))

    response = client.post(
        "/api/v1/sessions/complete",
        data={"session_id": SESSION, "user_id": "u", "scenario_id": "3", "vr_score": "60", "is_pass": "true",
              "avg_thickness_um": "120"},
        files={
            "telemetry_csv": ("t.csv", b"index,time\n1,0\n", "text/csv"),
            "height_map_image": ("Map01_x_Height.png", height_map, "image/png"),
        },
    )
    report = client.get(f"/api/v1/reports/{response.json()['data']['report_id']}").json()["data"]["report"]

    assert report["coating_report"]["analysis_status"] == "available"
    assert report["coating_report"]["expert_comparison"]["analysis_status"] == "unavailable"
    assert "도막 Zone 3 과도막" in report["llm_briefing"]
    assert report["pose_report"]["analysis_status"] == "available"


def _complete_session(client: TestClient, height_map: bytes) -> str:
    response = client.post(
        "/api/v1/sessions/complete",
        data={"session_id": SESSION, "user_id": "u", "scenario_id": "3", "vr_score": "60", "is_pass": "true",
              "avg_thickness_um": "120"},
        files={
            "telemetry_csv": ("t.csv", b"index,time\n1,0\n", "text/csv"),
            "height_map_image": ("Map01_x_Height.png", height_map, "image/png"),
        },
    )
    assert response.status_code == 200, response.text
    return str(response.json()["data"]["report_id"])


@pytest.mark.parametrize(
    ("target", "basis", "phrase"),
    [({}, "default_target", "기본 목표 160µm 기준"), ({"target_thickness_um": 160.0}, "session_target", "세션 목표 160µm 기준")],
)
def test_session_started_without_a_target_is_reported_as_the_default_target(
    engine: CoatingProgressEngine, height_map: bytes, target: dict[str, float], basis: str, phrase: str
) -> None:
    """시작 때 목표를 생략해도 160.0이 저장된다. 리포트는 세션이 목표를 명시했는지를 구분해 말한다.

    깨뜨리는 법: complete가 저장된 목표를 그대로 넘기면 생략한 세션도 "세션 목표"로 나온다.
    """
    client = _client(engine)
    started = client.post("/api/v1/sessions/start", json={
        "session_id": SESSION, "user_id": "u", "scenario_id": "3", "started_at": "2026-10-15T00:00:00Z",
        "worker": {"skill_level": "novice", "handedness": "right"}, **target,
    })
    assert started.status_code == 200, started.text

    report_id = _complete_session(client, height_map)

    coating = client.get(f"/api/v1/reports/{report_id}").json()["data"]["report"]["coating_report"]
    assert (coating["target_basis"], coating["target_thickness_um"]) == (basis, 160.0)
    assert phrase in coating["guideline"]


@pytest.mark.parametrize("tool_type", ["spray_gun", "blasting_gun"])
def test_complete_uses_the_session_process_when_pose_sequences_are_missing(
    engine: CoatingProgressEngine, height_map: bytes, tool_type: str
) -> None:
    """관절 CSV나 저장 시퀀스가 없어도 시작 때 확정한 공정과 최종 높이맵은 도막 판정에 쓴다."""
    state = V1Store(":memory:")
    app = FastAPI()
    app.include_router(create_v1_router(
        store=state, sequences=SequenceStore(":memory:"), coating_engine=engine,
    ))
    client = TestClient(app)
    started = client.post("/api/v1/sessions/start", json={
        "session_id": SESSION, "user_id": "u", "scenario_id": "3",
        "started_at": "2026-10-15T00:00:00Z", "tool_type": tool_type,
        "worker": {"skill_level": "novice", "handedness": "right"},
    })
    assert started.status_code == 200, started.text

    report_id = _complete_session(client, height_map)
    report = client.get(f"/api/v1/reports/{report_id}").json()["data"]["report"]

    assert report["report_status"] == "complete"
    assert report["pose_report"]["analysis_status"] == "unavailable"
    coating = report["coating_report"]
    if tool_type == "spray_gun":
        assert coating["analysis_status"] == "available"
        assert coating["target_basis"] == "default_target"
        assert coating["worst_zones"][0]["zone_id"] == "3"
    else:
        assert coating["analysis_status"] == "unavailable"
        assert coating["reason_code"] == "coating_not_applicable"
        assert coating["distribution"] is None


def test_error_analysis_answers_from_the_completed_report(engine: CoatingProgressEngine, height_map: bytes) -> None:
    """error_analysis는 완료 리포트의 도막 평가로 답한다. 리포트가 없으면 자료가 없다고 말한다.

    깨뜨리는 법: error_analysis를 예전처럼 고정 문장으로 돌려보내면 과도막 3번 구역이 답에 없다.
    """
    client = _client(engine)
    body = {"user_id": "u", "session_id": SESSION, "scenario_id": "3", "task_type": "error_analysis",
            "prompt": "도막 분포 이상 원인은?"}
    before = client.post("/api/v1/copilot/query", json=body).json()["data"]
    assert "완료 분석의 도막 결과가 없어" in before["answer"]
    assert (before["context_status"], before["coating_context"]) == ("not_available", None)

    _complete_session(client, height_map)
    quick = client.post("/api/v1/copilot/query", json=body).json()["data"]["answer"]
    assert "Zone 3은 과도막입니다(칠한 곳 평균 390µm)" in quick and quick.endswith("원인으로 확정된 것은 아닙니다.")
    data = client.post("/api/v1/copilot/query", json={**body, "response_mode": "detailed"}).json()["data"]

    assert data["answer"].startswith("기본 목표 160µm 기준으로 본 도막 분포입니다. 칠한 면적의 두께 분포: ")
    assert "Zone 3은 과도막입니다(칠한 곳 평균 390µm). 속도·이격거리 기록이 없어 조절 후보를 내지 못했습니다." in data["answer"]
    assert "저도막" not in data["answer"]
    assert data["answer"].endswith("원인으로 확정된 것은 아닙니다.")
    assert data["coating_context"]["comparison_basis"] == "saved_coating_report"
    assert (data["context_status"], data["retrieval_used"], data["citations"]) == ("available", False, [])


def test_coating_status_question_after_complete_lists_grades(engine: CoatingProgressEngine, height_map: bytes) -> None:
    client = _client(engine)
    _complete_session(client, height_map)

    quick = _ask(client, "도막이 과하거나 모자란 영역이 있어?")["answer"]
    data = _ask(client, "도막이 과하거나 모자란 영역이 있어?", mode="detailed")

    assert "저도막: 없음\n과도막: 1개 — Map 1 · Zone 3(칠한 곳 평균 390µm, 미도장 0%)\n정상: 1개\n칠한 곳 없음: 7개 — Map 1 · Zone 2" in data["answer"]
    assert "Map 1 · Zone 9" in data["answer"]
    assert quick == data["answer"]
    assert [row["zone_id"] for row in data["coating_context"]["zones"]] == ["1", "2", "3", "4", "5", "6", "7", "8", "9"]


def test_after_complete_chip_questions_use_the_session_report(engine: CoatingProgressEngine, height_map: bytes) -> None:
    """뷰어 채팅은 칩으로 유형만 고른다. complete 뒤에는 '구역' 없는 자세 질문도 세션 결과로 답하고, 순서 칩은 순서 리포트로 답한다.

    깨뜨리는 법: 키워드가 있어야만 구역 답을 내면 '자세 어땠어?'가 마지막 시퀀스 채점으로 간다.
    """
    reports = V1Store(":memory:")
    client = _client(engine, reports)
    report_id = _complete_session(client, height_map)

    pose = _ask(client, "자세 어땠어?")
    assert pose["answer"].startswith("세션 전체 자세 유사도는 57%입니다.")
    assert "가장 낮은 구역은 3번(40%)" in pose["answer"]

    body = {"user_id": "u", "session_id": SESSION, "scenario_id": "3", "task_type": "trajectory_recomm", "prompt": "순서 잘 지켰어?"}
    order = client.post("/api/v1/copilot/query", json=body).json()["data"]
    # 이 시험 라우터에는 순서 엔진이 없다 → 순서 리포트가 unavailable 이고 그 사유로 답한다.
    assert order["answer"].startswith("이 세션은 순서를 분석하지 못했습니다.")
    assert order["sequence_context"]["report_id"] == report_id

    from trainer.demo_api.motion_viewer import create_motion_viewer_router

    viewer = FastAPI()
    viewer.include_router(create_motion_viewer_router(Path("/nonexistent"), reports=reports))
    listed = TestClient(viewer).get("/motion-viewer/api/completed").json()["sessions"]
    assert [(item["session_id"], item["report_id"], item["scenario_id"]) for item in listed] == [(SESSION, report_id, "3")]
    script = TestClient(viewer).get("/motion-viewer/assets/motion_viewer_chat.js")
    assert script.status_code == 200 and "trajectory_recomm" in script.text


def test_order_answer_reads_the_sequence_report() -> None:
    from trainer.demo_api.pose_query import answer_order_question

    analysis = {"sequence_report": {
        "analysis_status": "available", "recommended_flow": ["7", "4", "8", "9"], "first_visit_flow": ["9", "7", "4"],
        "evaluated_visit_count": 5, "on_track_visit_count": 3, "adherence_rate": 0.6, "deviation_count": 2,
        "direction_adherence_rate": 0.75,
        "zone_summary": [{"zone_id": "9", "deviation_count": 1}, {"zone_id": "4", "deviation_count": 2}],
    }}

    result = answer_order_question(("RPT-1", analysis))

    assert result.answer == (
        "권장 순서는 7 → 4 → 8 → 9입니다. 이번 세션에서 처음 칠한 순서는 9 → 7 → 4입니다. "
        "구역 방문 5회 중 3회가 순서에 맞아 순서 준수율은 60%입니다. 칠하지 않은 구역은 8번입니다. "
        "순서 이탈이 가장 많은 구역은 4번(2회)입니다. 도포 방향 준수율은 75%입니다."
    )
    assert answer_order_question(None).reason_code == "session_report_unavailable"


def test_pose_answers_follow_the_question_instead_of_repeating_the_worst_zone() -> None:
    """깨뜨리는 법: prompt를 무시하고 항상 최악 Zone만 답하면 전체 점수와 오른손 질문이 같은 문장이 된다.
    """
    from trainer.demo_api.pose_query import answer_pose_question

    found = (
        "RPT-POSE",
        {
            "pose_report": {
                "analysis_status": "available",
                "average_similarity_score": 0.74,
                "evaluated_sequence_count": 12,
                "zone_summary": [
                    _zone("1", 0.82),
                    {
                        **_zone("3", 0.51),
                        "problem_joints": [
                            {"joint": "RightHand", "occurrence_count": 2, "occurrence_rate": 0.5, "average_distance": 0.11},
                            {"joint": "LeftHand", "occurrence_count": 1, "occurrence_rate": 0.25, "average_distance": 0.07},
                        ],
                    },
                ],
            }
        },
    )

    overall = answer_pose_question(found, prompt="전체 자세 유사도는 몇 퍼센트야?", context=None)
    right = answer_pose_question(found, prompt="오른손을 어떻게 고쳐야 해?", context=None)

    assert "전체 자세 유사도는 74%" in overall.answer
    assert "오른쪽 손목" in right.answer and "11cm" in right.answer
    assert overall.answer != right.answer


def test_pose_question_can_select_a_zone_from_natural_language() -> None:
    from trainer.demo_api.pose_query import answer_pose_question

    result = answer_pose_question(
        _analysis(_zone("1", 0.81), _zone("3", 0.52)),
        prompt="3번 Zone 자세는 어떠어?",
        context=None,
    )

    assert result.context["zone_id"] == "3"
    assert result.answer.startswith("3번 구역 자세 유사도는 52%입니다.")


def test_order_answers_follow_recommended_actual_missing_and_direction_questions() -> None:
    """깨뜨리는 법: 질문을 무시하고 전체 순서 요약을 매번 반환하면 네 답변이 동일해진다.
    """
    from trainer.demo_api.pose_query import answer_order_question

    found = ("RPT-ORDER", {"sequence_report": {
        "analysis_status": "available", "recommended_flow": ["7", "4", "8", "9"],
        "first_visit_flow": ["9", "7", "4"], "evaluated_visit_count": 5,
        "on_track_visit_count": 3, "adherence_rate": 0.6, "deviation_count": 2,
        "direction_adherence_rate": 0.75,
        "zone_summary": [{"zone_id": "9", "deviation_count": 1}, {"zone_id": "4", "deviation_count": 2}],
    }})

    recommended = answer_order_question(found, prompt="권장 순서를 알려줘")
    actual = answer_order_question(found, prompt="실제로 어떤 순서로 칠했어?")
    missing = answer_order_question(found, prompt="누락한 Zone이 있어?")
    direction = answer_order_question(found, prompt="도포 방향은 잘 지켰어?")

    assert "7 → 4 → 8 → 9" in recommended.answer
    assert "Zone 7 →" in recommended.answer
    assert actual.answer == "이번 세션에서 처음 칠한 순서는 9 → 7 → 4입니다."
    assert missing.answer == "권장 순서 중 칠하지 않은 구역은 8번입니다."
    assert direction.answer == "전문가 기준 도포 방향 준수율은 75%입니다."
    assert len({recommended.answer, actual.answer, missing.answer, direction.answer}) == 4


def test_pose_unknown_question_guides_the_user_instead_of_repeating_overview() -> None:
    """깨뜨리는 법: 알 수 없는 질문을 전체 자세 요약으로 처리하면 사용자는 계속 같은 답을 받는다."""
    from trainer.demo_api.pose_query import answer_pose_question

    found = _analysis(_zone("1", 0.81), _zone("3", 0.52))

    result = answer_pose_question(found, prompt="오늘 훈련 어땠어?", context=None)

    assert "자세 탭에서는" in result.answer
    assert "전체 자세 유사도" in result.answer
    assert result.context["scope"] == "help"


def test_order_adherence_and_unknown_questions_return_distinct_focused_answers() -> None:
    """깨뜨리는 법: 준수율이나 미분류 질문을 모두 종합 순서 문장으로 반환하면 같은 답처럼 보인다."""
    from trainer.demo_api.pose_query import answer_order_question

    found = ("RPT-ORDER", {"sequence_report": {
        "analysis_status": "available", "recommended_flow": ["7", "4", "8", "9"],
        "first_visit_flow": ["9", "7", "4"], "evaluated_visit_count": 5,
        "on_track_visit_count": 3, "adherence_rate": 0.6, "deviation_count": 2,
        "direction_adherence_rate": 0.75,
        "zone_summary": [{"zone_id": "9", "deviation_count": 1}, {"zone_id": "4", "deviation_count": 2}],
    }})

    adherence = answer_order_question(found, prompt="순서 준수율은 몇 퍼센트야?")
    unknown = answer_order_question(found, prompt="오늘 훈련 어땠어?")

    assert adherence.answer == "순서 준수율은 60%입니다."
    assert "순서 탭에서는" in unknown.answer
    assert unknown.context["scope"] == "help"
    assert adherence.answer != unknown.answer


# --- 참고 판정: 전문가 대비 (보정 전 두께, 전문가 칸 q1=80·q3=120) -------------------------
_EXPERT_THICKNESS = {"1": 100.0, "2": 0.0, "3": 400.0}


def _expert_report(engine: CoatingProgressEngine, **kwargs: Any) -> Any:
    return analyze_session_coating(
        scenario_id="3", bodies=[SPRAY], engine=engine, height_map=_png(_thickness(_EXPERT_THICKNESS)),
        height_map_name="Map01_x_Height.png", **kwargs,
    )


def test_report_carries_the_expert_comparison_beside_the_target_grade(engine: CoatingProgressEngine) -> None:
    """1번은 원시 100(보정 90) = 목표 대비 저도막이지만 전문가 칸 80 이상이라 전문가 대비는 충족이다."""
    result = _expert_report(engine)

    rows = {zone["zone_id"]: zone for zone in result.zones}
    assert {k: v["status"] for k, v in rows.items()} == {
        "1": "under_app", "2": None, "3": "over_app", "4": None, "5": None,
        "6": None, "7": None, "8": None, "9": None,
    }
    assert {k: v["expert"]["status"] for k, v in rows.items() if v["expert"] is not None} == {
        "1": "normal", "2": "under_app", "3": "over_app",
    }
    report = result.report
    assert report["comparison_basis"] == "session_target"
    comparison = report["expert_comparison"]
    assert comparison["comparison_basis"] == "expert_reference"
    assert (comparison["normal"], comparison["under_app"], comparison["over_app"]) == (1, 1, 1)
    assert comparison["zone_count"] == len(comparison["zones"]) == 3


def test_expert_sentence_follows_the_target_sentence(engine: CoatingProgressEngine) -> None:
    guideline = _expert_report(engine).report["guideline"]

    expert_label = "전문가 최종 도막 대비:"
    assert guideline.index("기본 목표 160µm 기준") < guideline.index(expert_label)
    assert guideline.endswith("(보정 전 두께 기준).") or guideline.endswith("입니다(보정 전 두께 기준).")
    expert_part = guideline[guideline.index(expert_label):]
    assert "저도막" not in expert_part and "과도막" not in expert_part
    assert "전문가보다 얇은 구역 1개(2번)" in expert_part and "전문가보다 두꺼운 구역 1개(3번)" in expert_part


def test_expert_comparison_ignores_the_session_target(engine: CoatingProgressEngine) -> None:
    assert (
        _expert_report(engine, target_thickness_um=400.0).report["expert_comparison"]
        == _expert_report(engine).report["expert_comparison"]
    )


def test_zone_without_expert_cells_in_its_mask_has_no_expert_row(tmp_path: Path) -> None:
    """칸 좌표가 마스크와 안 겹치면 전문가 행이 없고 구역 수에서 빠진다. 전부 없으면 unavailable."""
    _profile(tmp_path)
    path = tmp_path / "generated" / "profiles" / "spray" / "learning" / "mode_03.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for zone in data["aggregate"]["zones"].values():
        zone["coating"]["cells"] = {"0:0": {"q1_um": 80.0, "median_um": 100.0, "q3_um": 120.0}}
    path.write_text(json.dumps(data), encoding="utf-8")

    result = _expert_report(CoatingProgressEngine(tmp_path))

    assert all(zone["expert"] is None for zone in result.zones)
    comparison = result.report["expert_comparison"]
    assert (comparison["analysis_status"], comparison["reason_code"]) == ("unavailable", "expert_cells_unavailable")
    assert comparison["zone_count"] == 0
    assert "참고로 전문가" not in result.report["guideline"]


def test_coating_status_answer_after_complete_adds_the_expert_line_when_requested(engine: CoatingProgressEngine, height_map: bytes) -> None:
    client = _client(engine)
    _complete_session(client, height_map)

    direct = _ask_knowledge(client, "도막 상태 어때?")["answer"]
    assert "전문가 대비" not in direct
    answer = _ask_knowledge(client, "도막 상태와 전문가 대비 결과를 같이 알려줘.")["answer"]

    assert "참고(전문가 대비, 보정 전 두께):" in answer
    assert answer.index("저도막:") < answer.index("참고(전문가 대비, 보정 전 두께)")


def test_zone_question_appends_the_expert_word_when_requested(engine: CoatingProgressEngine, height_map: bytes) -> None:
    client = _client(engine)
    _complete_session(client, height_map)

    answer = _ask_knowledge(client, "Zone 3 도막과 전문가 대비 결과는?")["answer"]

    assert "· 전문가보다 두꺼운 구역" in answer


def test_rows_saved_before_round_two_keep_the_old_answer() -> None:
    from trainer.demo_api.coating_query import answer_coating_question

    row = {"map_num": 1, "zone_id": "3", "grade": "above", "thickness_um": 390.0, "target_thickness_um": 160.0,
           "target_basis": "default_target", "unpainted_fraction": 0.0, "adjustments": [], "six_band_counts": {}}
    context = {"analysis_status": "available", "comparison_basis": "saved_coating_report", "zones": [row]}

    assert "전문가 대비" not in answer_coating_question(context, "도막 상태 어때?", detailed=False)
    assert "전문가 대비" not in answer_coating_question(context, "Zone 3 도막은?", detailed=False)


def test_zone_answer_sentence_names_the_expert_comparison(engine: CoatingProgressEngine, height_map: bytes) -> None:
    from trainer.demo_api.pose_query import answer_pose_question

    reports = V1Store(":memory:")
    client = _client(engine, reports)
    _complete_session(client, height_map)

    result = answer_pose_question(
        reports.latest_complete_analysis(SESSION), prompt="가장 문제가 큰 구역의 자세와 도막은?", context=None,
    )

    assert "전문가 최종 도막과 견주면" in result.answer


def _ask_knowledge(client: TestClient, prompt: str, mode: str = "detailed") -> dict[str, Any]:
    """전체 답을 보는 시험이라 상세 분석으로 묻는다(빠른 안내는 한 줄 60자, 2026-11-16)."""
    body = {"user_id": "u", "session_id": SESSION, "scenario_id": "3", "task_type": "knowledge_search", "prompt": prompt,
            "response_mode": mode}
    return client.post("/api/v1/copilot/query", json=body).json()["data"]
