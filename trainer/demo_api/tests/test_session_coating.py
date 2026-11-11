"""complete 의 세션 도막 분석과, 그 결과로 답하는 pose_recomm 구역 질문.

구역 경계는 저장소에 있는 **실제 TSB 라벨맵**(Learn03 = scenario 3)이다. 두께와 전문가 기준은 합성한다 —
Spark 의 전문가 프로필은 이 저장소에 없고, 정답을 아는 입력이어야 판정이 맞는지 볼 수 있다.
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
from trainer.demo_api.session_coating import analyze_session_coating
from trainer.demo_api.v1_router import create_v1_router
from trainer.demo_api.v1_store import V1Store
from trainer.zone_reference import ZoneReferenceCatalog

SIZE = 2048
GRID = 64
# 정답: 1번 구역은 기준 안, 2번은 안 칠함(부족), 3번은 과도장.
THICKNESS = {"1": 100.0, "2": 0.0, "3": 400.0}
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


def test_height_map_zones_are_judged_against_the_expert_cells(engine: CoatingProgressEngine, height_map: bytes) -> None:
    """정상·부족·과도장이 라벨맵 구역별로 갈린다. 셀 수 합이 분포와 맞는다.

    깨뜨리는 법: 높이맵을 뒤집지 않으면(좌상단↔좌하단) 마스크와 두께가 어긋나 1번이 정상이 아니게 된다.
    """
    result = analyze_session_coating(
        scenario_id="3", bodies=[SPRAY], engine=engine, height_map=height_map, height_map_name="Map01_x_Height.png"
    )

    statuses = {zone["zone_id"]: zone["status"] for zone in result.zones}
    assert statuses == {"1": "normal", "2": "under_app", "3": "over_app"}
    report = result.report
    assert report["analysis_status"] == "available"
    assert report["source_artifact_id"] == "Map01_x_Height.png"
    cells = {zone_id: len(_cells(zone_id)) for zone_id in THICKNESS}
    assert report["distribution"] == {"below": cells["2"], "in_band": cells["1"], "above": cells["3"]}
    # 둘 다 셀 100%가 벗어났다 — 같으면 구역 순서.
    assert [zone["zone_id"] for zone in report["worst_zones"]] == ["2", "3"]
    assert report["worst_zones"][1]["thickness_um"] == pytest.approx(400.0, abs=0.5)
    assert "부족 1개(2번)" in report["guideline"] and "과도장 1개(3번)" in report["guideline"]


def test_without_a_height_map_later_crops_overwrite_earlier_ones(engine: CoatingProgressEngine, height_map: bytes) -> None:
    """높이맵이 없으면 시퀀스 크롭을 순서대로 덮어쓴다. 크롭 값은 누적 두께라 더하지 않는다.

    깨뜨리는 법: 더하면 3번 구역이 400+100=500µm 로 여전히 과도장이다.
    """
    zone3 = _mask("3")
    rows, columns = np.nonzero(zone3)
    top, bottom, left, right = rows.min(), rows.max() + 1, columns.min(), columns.max() + 1
    fix = _png(np.full((bottom - top, right - left), 100.0), valid=zone3[top:bottom, left:right])
    bodies = [
        {**SPRAY, "sequence_result": {"cur_sequence_heatmap": [_crop(height_map, x0=0, y0=0, width=SIZE, height=SIZE)]}},
        {**SPRAY, "sequence_result": {"cur_sequence_heatmap": [
            # 크롭 y0 는 좌하단 원점이다.
            _crop(fix, x0=int(left), y0=int(SIZE - bottom), width=int(right - left), height=int(bottom - top))
        ]}},
    ]

    result = analyze_session_coating(scenario_id="3", bodies=bodies, engine=engine)

    assert {zone["zone_id"]: zone["status"] for zone in result.zones} == {"1": "normal", "2": "under_app", "3": "normal"}
    assert result.report["source_artifact_id"] == "cur_sequence_heatmap"


@pytest.mark.parametrize(
    ("scenario", "bodies", "with_profile", "reason"),
    [
        ("Unit3", [SPRAY], True, "reference_context_unavailable"),
        ("3", [{"sequence_setting": {"tool_type": "blasting"}}], True, "coating_not_applicable"),
        ("3", [SPRAY], False, "reference_unavailable"),
        ("3", [SPRAY], True, "heatmap_unavailable"),
    ],
)
def test_missing_inputs_are_unavailable_without_numbers(
    tmp_path: Path, scenario: str, bodies: list[dict[str, Any]], with_profile: bool, reason: str
) -> None:
    if with_profile:
        _profile(tmp_path)
    height_map = _png(_thickness(THICKNESS)) if reason == "reference_unavailable" else None

    result = analyze_session_coating(
        scenario_id=scenario, bodies=bodies, engine=CoatingProgressEngine(tmp_path), height_map=height_map
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


def _ask(client: TestClient, prompt: str, *, session_id: str = SESSION, context: str | None = None) -> dict[str, Any]:
    body = {"user_id": "u", "session_id": session_id, "scenario_id": "3", "task_type": "pose_recomm", "prompt": prompt}
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
    assert report["coating_report"]["guideline"] in report["llm_briefing"]
    assert [zone["zone_id"] for zone in report["pose_report"]["zone_summary"]] == ["1", "3"]

    worst = _ask(client, "가장 문제 있는 구역이 어디야?")

    assert worst["context_status"] == "available"
    assert worst["answer"].startswith("가장 문제가 큰 구역은 3번 구역입니다.")
    assert "평균 자세 유사도는 40%로 세션 평균 57%보다 낮습니다" in worst["answer"]
    assert "오른쪽 위팔 위치가 숙련자 범위를 벗어난 분사가 2회 중 2회였습니다(평균 12cm)" in worst["answer"]
    assert "도막은 과도장입니다" in worst["answer"]
    assert worst["pose_context"]["scope"] == "zone" and worst["pose_context"]["report_id"] == report_id

    zone1 = _ask(client, "이 구역 자세 어땠어?", context="1")
    assert zone1["answer"].startswith("1번 구역 결과입니다.")
    assert "숙련자 범위를 벗어난 관절은 없었습니다" in zone1["answer"]
    assert "도막은 전문가 기준을 충족했습니다" in zone1["answer"]

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


def test_without_an_expert_reference_the_briefing_says_why_coating_is_missing(tmp_path: Path, height_map: bytes) -> None:
    """도막 엔진은 있는데 기준이 없으면 '외부 엔진 연동 후'가 아니라 실제 사유를 말한다. 자세 결과는 그대로 남는다."""
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

    assert report["coating_report"]["reason_code"] == "reference_unavailable"
    assert "도막은 분석하지 못했습니다: 이 콘텐츠의 전문가 도막 기준이 없습니다." in report["llm_briefing"]
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
    assert result.answer.startswith("3번 구역 결과입니다.")


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
    assert "Zone 7에서 시작" in recommended.answer
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

    assert adherence.answer == "순서 준수율은 60%입니다. 구역 방문 5회 중 3회가 권장 순서에 맞았습니다."
    assert "순서 탭에서는" in unknown.answer
    assert unknown.context["scope"] == "help"
    assert adherence.answer != unknown.answer
