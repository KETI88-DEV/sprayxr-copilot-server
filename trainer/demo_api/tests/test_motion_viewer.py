import csv
import json
import pytest
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.motion_viewer import (
    _build_zone_surface,
    _dense_paint_trace,
    _direction_summary,
    _joint_posture_state,
    _smooth_posture_states,
    create_motion_viewer_router,
    viewer_v2_document,
)
from trainer.demo_api.pose_query import PoseEngineResult
from trainer.demo_api.sequence_store import SequenceStore
from trainer.scripts.build_zone_guidance_profiles import build_profiles


def _write_csv(path: Path, *, offset: float, zone_ids: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "인덱스",
        "시간(초)",
        "분사",
        "한길숏건 Position X", "한길숏건 Position Y", "한길숏건 Position Z",
        "한길숏건 Rotation X", "한길숏건 Rotation Y", "한길숏건 Rotation Z",
        "플래쉬 Position X", "플래쉬 Position Y", "플래쉬 Position Z",
        "카메라 Position X", "카메라 Position Y", "카메라 Position Z",
        "카메라 Rotation X", "카메라 Rotation Y", "카메라 Rotation Z",
        "Ray point X", " Ray Point Y", " Ray Point Z",
        "Ray Texture Map Number", "Ray Texture UV X", "Ray Texture UV Y",
        "Zone ID",
        "Speed (Real)", "Distance (Real)", "Distance (Input)",
        "Pelvis Position X", "Pelvis Position Y", "Pelvis Position Z",
        "Pelvis Rotation X", "Pelvis Rotation Y", "Pelvis Rotation Z",
        "Spine Position X", "Spine Position Y", "Spine Position Z",
        "Spine Rotation X", "Spine Rotation Y", "Spine Rotation Z",
    ]
    joint_offsets = {
        "Chest": (0.0, 0.2, 0.0), "Head": (0.0, 0.2, 0.0),
        "Left Shoulder": (-0.2, 0.15, 0.0), "Left Upper Arm": (-0.2, 0.0, 0.0),
        "Left Fore Arm": (-0.25, 0.0, 0.0), "Left Hand": (-0.15, 0.0, 0.0),
        "Right Shoulder": (0.2, 0.15, 0.0), "Right Upper Arm": (0.2, 0.0, 0.0),
        "Right Fore Arm": (0.25, 0.0, 0.0), "Right Hand": (0.15, 0.0, 0.0),
        "Left Thigh": (-0.1, -0.1, 0.0), "Left Calf": (0.0, -0.4, 0.0),
        "Left Foot": (0.0, -0.4, 0.0), "Left Toe": (0.0, 0.0, 0.15),
        "Right Thigh": (0.1, -0.1, 0.0), "Right Calf": (0.0, -0.4, 0.0),
        "Right Foot": (0.0, -0.4, 0.0), "Right Toe": (0.0, 0.0, 0.15),
    }
    for joint in joint_offsets:
        fields.extend(f"{joint} {kind} {axis}" for kind in ("Position", "Rotation") for axis in "XYZ")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(5):
            row = {
                    "인덱스": index,
                    "시간(초)": index / 90,
                    "분사": "True",
                    "한길숏건 Position X": 0.3 + index * 0.01,
                    "한길숏건 Position Y": 1.2,
                    "한길숏건 Position Z": 0.2,
                    "한길숏건 Rotation X": 0,
                    "한길숏건 Rotation Y": 0,
                    "한길숏건 Rotation Z": 0,
                    "플래쉬 Position X": -0.3 + index * 0.01,
                    "플래쉬 Position Y": 1.1,
                    "플래쉬 Position Z": 0.1,
                    "카메라 Position X": index * 0.01,
                    "카메라 Position Y": 1.7,
                    "카메라 Position Z": 0,
                    "카메라 Rotation X": 0,
                    "카메라 Rotation Y": 0,
                    "카메라 Rotation Z": 0,
                    "Ray point X": offset + index * 0.2,
                    " Ray Point Y": 1.4,
                    " Ray Point Z": 2.0,
                    "Ray Texture Map Number": 1,
                    "Ray Texture UV X": (index % 2) * 32,
                    "Ray Texture UV Y": (index // 2) * 32,
                    "Zone ID": zone_ids[index] if zone_ids is not None else "1",
                    "Speed (Real)": 300,
                    "Distance (Real)": 250,
                    "Distance (Input)": 300,
                    "Pelvis Position X": offset + index * 0.01,
                    "Pelvis Position Y": 1,
                    "Pelvis Position Z": 0,
                    "Pelvis Rotation X": 0,
                    "Pelvis Rotation Y": 0,
                    "Pelvis Rotation Z": 0,
                    "Spine Position X": 0,
                    "Spine Position Y": 0.3,
                    "Spine Position Z": 0,
                    "Spine Rotation X": 0,
                    "Spine Rotation Y": 0,
                    "Spine Rotation Z": 0,
            }
            for joint, position in joint_offsets.items():
                for axis, value in zip("XYZ", position):
                    row[f"{joint} Position {axis}"] = value
                    row[f"{joint} Rotation {axis}"] = 0
            writer.writerow(row)


def _client(
    tmp_path: Path,
    *,
    store: SequenceStore | None = None,
    pose_scorer: Any = None,
    mesh_provider: Any = None,
    heightmaps: list[str] | None = None,
) -> TestClient:
    mode_root = tmp_path / "raw" / "spray" / "learning" / "mode_04"
    _write_csv(mode_root / "4_1001_1.csv", offset=0.0)
    _write_csv(mode_root / "4_1002_1.csv", offset=0.1)
    profile = {
        "schema_version": 1,
        "identity": {"process": "spray", "mode_id": 4, "content_type": "learning"},
        "quality": {"status": "multi_session", "session_count": 2},
        "aggregate": {
            "zones": {
                "1:1": {
                    "map_num": 1,
                    "zone_id": "1",
                    "speed_mms": {"median": 300},
                    "distance_mm": {"median": 250},
                    "expected_direction": "horizontal",
                    "coating": {
                        "analysis_status": "available",
                        "display_semantics": "physical_thickness_um",
                        "atlas_width": 2048,
                        "atlas_height": 2048,
                        "grid_size": 64,
                        "session_count": 2,
                        "cells": {
                            "10:20": {"sample_count": 2, "q1_um": 120, "median_um": 140, "q3_um": 160},
                            "11:20": {"sample_count": 2, "q1_um": 150, "median_um": 180, "q3_um": 210},
                        },
                    },
                }
            },
            "macro_orders": [
                {
                    "order": [{"map_num": 1, "zone_id": "1"}],
                    "session_count": 2,
                    "fraction": 1.0,
                }
            ],
            "transitions": [],
        },
        "sessions": [
            {
                "session_id": "4_1001_1",
                "sources": {"csv_path": "4_1001_1.csv"},
                "analysis": {
                    "macro_order": [{"map_num": 1, "zone_id": "1"}],
                    "zones": {"1:1": {"speed_mms": 300, "distance_mm": 250, "expected_direction": "horizontal"}},
                    "coating_cells": {
                        "1:1": {
                            "cells": {"10:20": 140, "11:20": 180},
                            "atlas_width": 2048, "atlas_height": 2048, "grid_size": 64,
                        }
                    },
                },
            },
            {
                "session_id": "4_1002_1",
                "sources": {"csv_path": "4_1002_1.csv"},
                "analysis": {
                    "macro_order": [{"map_num": 1, "zone_id": "1"}],
                    "zones": {"1:1": {"speed_mms": 360, "distance_mm": 280, "expected_direction": "horizontal"}},
                    "coating_cells": {
                        "1:1": {
                            "cells": {"10:20": 150, "11:20": 190},
                            "atlas_width": 2048, "atlas_height": 2048, "grid_size": 64,
                        }
                    },
                },
            },
        ],
    }
    if heightmaps is not None:
        profile["sessions"][1]["heightmaps"] = [{"source_path": item} for item in heightmaps]
    profile_path = tmp_path / "generated" / "profiles" / "spray" / "learning" / "mode_04.json"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    app = FastAPI()
    kwargs: dict[str, Any] = {"frame_count": 5}
    if store is not None:
        kwargs["sequence_store"] = store
    if pose_scorer is not None:
        kwargs["pose_scorer"] = pose_scorer
    if mesh_provider is not None:
        kwargs["mesh_provider"] = mesh_provider
    app.include_router(create_motion_viewer_router(tmp_path, **kwargs))
    return TestClient(app)


def _store_sequence(store: SequenceStore, *, sequence_id: str = "SEQ-020") -> dict[str, Any]:
    body = {
        "session_id": "3_11000346_1",
        "sequence_id": sequence_id,
        "scenario_id": "3",
        "zone_id": "7",
        "sequence_info": [
            {"index": 0, "skeleton_joints": {"Head": [0.0, 1.7, 0.0]}}
        ],
    }
    store.record(
        session_key=body["session_id"], sequence_id=sequence_id,
        body_sha256=sequence_id, frames=1, received_at="2026-10-14T00:00:00Z",
        body=json.dumps(body),
    )
    return body


def test_motion_viewer_page_explains_the_aggregated_expert_reference(tmp_path: Path) -> None:
    response = _client(tmp_path).get("/motion-viewer")

    assert response.status_code == 200
    assert "도장 경로와 작업 자세를 전문가 기준으로 비교합니다." in response.text
    assert "전체 기록 종합 기준" in response.text
    assert "대표 실측 모션" not in response.text
    assert "통계 범위" in response.text
    assert "페인트 누적 궤적" in response.text
    assert "주요 도장 방향" in response.text
    assert "머리·양손 실측 위치와 추정 상체" in response.text
    assert "worker.display_name" in response.text
    assert "function referencePointCloud" in response.text
    assert "function drawZoneFlowGuides" in response.text
    assert "function directionField" not in response.text
    assert "Zone별 전문가 표준 도장 패턴" in response.text
    assert "전문가 전체 Ray Point" in response.text
    assert 'id="pointSize"' in response.text
    assert "흰색 포인트 크기" in response.text
    assert "pointSizeValue" in response.text
    assert 'id="zoneSurface"' in response.text
    assert "Zone 영역 표시" in response.text
    assert "function drawZoneSurfaces" in response.text
    assert "137.508" not in response.text
    assert "`Zone ${zoneLabel(surface.zone_id)}`" in response.text
    assert 'id="loading"' in response.text
    assert "불러오는 중" in response.text
    assert "function setLoading" in response.text
    assert "data.paint_trace" in response.text
    assert 'id="currentZone"' in response.text
    assert 'id="zoneOrder"' in response.text
    assert 'id="zoneDetail"' in response.text
    assert "function updateZonePanel" in response.text
    assert "Zone 작업 순서" not in response.text
    assert "현재 작업 Zone" not in response.text
    assert "grid-template-columns:repeat(4,minmax(0,1fr))" in response.text
    assert "allButton.textContent='전체'" in response.text
    assert "expertDirectionMix" in response.text
    assert "현재 양손 위치 편차" not in response.text
    assert "trackedStats[marker]" not in response.text
    assert "function interpolatedFrame" in response.text
    assert "function resetViewToCamera" in response.text
    assert "viewCenter" in response.text
    assert "e.button===1" in response.text
    assert "drag.mode==='pan'" in response.text
    assert "canvas.onauxclick" in response.text
    assert 'id="viewerLegend"' in response.text
    assert response.text.count('id="viewerLegend"') == 1
    assert response.text.index('id="viewerLegend"') < response.text.index('id="loading"')
    assert 'id="cameraHome"' in response.text
    assert 'id="skeletonCanvas"' in response.text
    assert 'id="skeletonTurnLeft"' in response.text
    assert 'id="skeletonTurnRight"' in response.text
    assert 'id="skeletonReset"' in response.text
    assert 'id="skeletonPosture"' in response.text
    assert 'aria-label="스켈레톤 왼쪽 회전"' in response.text
    assert "function uprightSkeleton" in response.text
    assert "function drawSkeletonView" in response.text
    # 패널 0°는 정면에서 본 그림이다(몸 오른쪽 = 화면 왼쪽). 좌우·회전 방향은 호스트의 test_motion_viewer_chirality.js 가 잰다.
    assert "x=-dot(relative,across)" in response.text
    assert "정면 · 0°" in response.text
    assert "뒤에서 보기" not in response.text and "뒤에서 보기" not in viewer_v2_document()
    assert "CSV의 Pelvis·몸통·팔·다리 조인트" in response.text
    assert 'id="skeleton"' not in response.text
    assert "function alignSkeleton" not in response.text
    # 전신을 맞추는 보정은 없다. 오른쪽 전완·손만 실측 건 손잡이로 다시 세우고 나머지는 CSV 그대로다.
    assert "function panelSkeleton" in response.text
    assert "estimated:['Right Fore Arm','Right Hand']" in response.text
    assert "별도 보정 없이" not in response.text
    assert "playheadSeconds" in response.text
    assert "const fps=(data.frames.length-1)" not in response.text
    assert "<canvas" in response.text
    assert "function tick(now)" in response.text
    assert '<select id="speed"' in response.text
    for value in ("0.25", "0.5", "1", "1.5", "2", "3"):
        assert f'<option value="{value}"' in response.text
    assert 'id="speedValue"' not in response.text
    assert "function updateSpeedLabel" not in response.text
    assert "const ZONE_COLORS=[" in response.text
    assert "'#FF0000'" in response.text
    assert "'#BFBF40'" in response.text
    assert "function zoneColor(zoneId)" in response.text
    assert "function zoneLabel(zoneId)" in response.text
    assert "function averageCameraForward(payload)" in response.text
    assert "Math.atan2(forward[0],forward[2])" in response.text
    assert "4.5/(4.5+z)" in response.text
    assert "4.5/(4.5-z)" not in response.text
    assert "projected.sort((a,b)=>b.depth-a.depth)" in response.text
    assert "function zoneHue" not in response.text
    assert "let referencePreview=false" in response.text
    assert "function showReferencePreview(key)" in response.text
    assert "function startPlayback()" in response.text
    assert "button.onclick=()=>showReferencePreview(key)" in response.text
    assert "allButton.onclick=()=>showReferencePreview('all')" in response.text
    assert "if(!referencePreview){" in response.text


def test_mode_two_uses_corrected_expert_zone_ids_without_changing_raw_csv(tmp_path: Path) -> None:
    mode_root = tmp_path / "raw" / "spray" / "learning" / "mode_02"
    raw_zone_ids = ["1", "1", "2", "3", "3"]
    _write_csv(mode_root / "2_1001_1.csv", offset=0.0, zone_ids=raw_zone_ids)
    build_profiles(tmp_path, process="spray")
    app = FastAPI()
    app.include_router(create_motion_viewer_router(tmp_path, frame_count=5))

    response = TestClient(app).get(
        "/motion-viewer/api/motion",
        params={"process": "spray", "mode_id": 2, "session_id": "representative"},
    )

    assert response.status_code == 200
    data = response.json()
    assert [frame["zone_id"] for frame in data["frames"]] == ["3", "3", "2", "1", "1"]
    assert [item["zone_id"] for item in data["paint_trace"]] == ["3", "3", "2", "1", "1"]
    with mode_root.joinpath("2_1001_1.csv").open(encoding="utf-8-sig", newline="") as stream:
        assert [row["Zone ID"] for row in csv.DictReader(stream)] == raw_zone_ids


def test_catalog_lists_modes_and_individual_expert_sessions(tmp_path: Path) -> None:
    data = _client(tmp_path).get("/motion-viewer/api/catalog").json()

    assert data["workers"] == [
        {
            "worker_id": "jeon_hyeonggi",
            "display_name": "전형기",
            "role": "expert",
            "processes": data["workers"][0]["processes"],
        }
    ]
    mode = data["workers"][0]["processes"][0]["modes"][0]
    assert mode == {
        "mode_id": 4,
        "session_count": 2,
        "representative_session_id": "4_1001_1",
        "sessions": ["4_1001_1", "4_1002_1"],
    }


@pytest.mark.parametrize("legacy_worker", [False, True])
def test_worker_mode_two_keeps_source_zones_while_reference_expert_is_remapped(tmp_path: Path, legacy_worker: bool) -> None:
    expert_root = tmp_path / "raw/spray/learning/mode_02"
    worker_root = tmp_path / "workers/kim_kyungho"
    _write_csv(expert_root / "2_1001_1.csv", offset=0.0, zone_ids=["3", "3", "2", "1", "1"])
    _write_csv(expert_root / "2_1002_1.csv", offset=0.0, zone_ids=["3", "3", "2", "1", "1"])
    worker_zone_ids = ["3", "3", "2", "1", "1"] if legacy_worker else ["1", "1", "2", "3", "3"]
    _write_csv(worker_root / "raw/spray/learning/mode_02/2_2001_1.csv", offset=10.0, zone_ids=worker_zone_ids)
    (worker_root / "worker.json").write_text(json.dumps({"display_name": "김경호", "role": "worker"}), encoding="utf-8")
    build_profiles(tmp_path, process="spray")
    build_profiles(worker_root, process="spray", legacy_expert_zone_ids=legacy_worker)
    app = FastAPI()
    app.include_router(create_motion_viewer_router(tmp_path, frame_count=5))
    response = TestClient(app).get("/motion-viewer/api/motion", params={
        "worker_id": "kim_kyungho", "process": "spray", "mode_id": 2, "session_id": "2_2001_1",
    })
    assert response.status_code == 200
    data = response.json()
    assert [frame["zone_id"] for frame in data["frames"]] == ["1", "1", "2", "3", "3"]
    assert [item["zone_id"] for item in data["paint_trace"]] == ["1", "1", "2", "3", "3"]
    assert {item["zone_id"] for item in data["zone_surfaces"]} == {"1", "2", "3"}
    assert data["pose_evaluation"]["reference_session_count"] == 2
    assert data["reference_order_evaluation"]["reference_session_count"] == 2
    assert data["zone_statistics"]["all"]["expert_session_count"] == 2
    assert all(item["point"][0] < 1 for item in data["reference_surface"]["points"])
    assert all(item["point"][0] >= 10 for item in data["paint_trace"])
    assert all(point[0] < 1 for zone in data["zone_surfaces"] for point in zone["vertices"])


def test_catalog_discovers_future_worker_directory_without_code_change(tmp_path: Path) -> None:
    client = _client(tmp_path)
    worker_root = tmp_path / "workers" / "kim_gyeongho"
    (worker_root / "worker.json").parent.mkdir(parents=True, exist_ok=True)
    (worker_root / "worker.json").write_text(
        json.dumps({"display_name": "김경호", "role": "trainee"}), encoding="utf-8"
    )
    source_profile = tmp_path / "generated" / "profiles" / "spray" / "learning" / "mode_04.json"
    target_profile = worker_root / "generated" / "profiles" / "spray" / "learning" / "mode_04.json"
    target_profile.parent.mkdir(parents=True, exist_ok=True)
    target_profile.write_text(source_profile.read_text(encoding="utf-8"), encoding="utf-8")
    source_raw = tmp_path / "raw" / "spray" / "learning" / "mode_04"
    target_raw = worker_root / "raw" / "spray" / "learning" / "mode_04"
    target_raw.mkdir(parents=True, exist_ok=True)
    for source in source_raw.glob("*.csv"):
        target_raw.joinpath(source.name).write_bytes(source.read_bytes())

    payload = client.get("/motion-viewer/api/catalog").json()

    assert [worker["display_name"] for worker in payload["workers"]] == ["전형기", "김경호"]
    assert payload["workers"][1]["worker_id"] == "kim_gyeongho"


def test_motion_returns_real_representative_frames_and_cross_session_spread(tmp_path: Path) -> None:
    response = _client(tmp_path).get(
        "/motion-viewer/api/motion",
        params={"process": "spray", "mode_id": 4, "session_id": "representative"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["selected_session_id"] == "4_1001_1"
    assert data["representative_session_id"] == "4_1001_1"
    assert data["sampled_session_count"] == 2
    assert len(data["frames"]) == 5
    assert data["frames"][0]["joints"]["Pelvis"] == [0.0, 1.0, 0.0]
    assert data["frames"][0]["joints"]["Spine"] == [0.0, 1.3, 0.0]
    assert data["frames"][0]["camera_forward"] == [0.0, 0.0, 1.0]
    assert data["frames"][0]["tool_forward"] == [0.0, 1.0, 0.0]
    assert data["frames"][0]["posture"]["key"] == "standing"
    assert data["frames"][0]["posture"]["estimated"] is True
    assert data["frames"][0]["posture"]["source"] == "csv_skeleton_joints"
    assert data["worker"] == {
        "worker_id": "jeon_hyeonggi",
        "display_name": "전형기",
        "role": "expert",
    }
    assert data["statistics"][0]["spread"]["Pelvis"] > 0
    assert data["paint_trace"][0]["uv"] == [0.0, 0.0]
    assert data["zone_surfaces"][0]["reference_uv_cell_count"] == 5
    assert data["reference_order_evaluation"] == {
        "recommended_flow": ["1"],
        "selected_flow": ["1"],
        "reference_session_count": 2,
        "best_reference_similarity": 1.0,
        "position_similarity": 1.0,
        "adherence_rate": 1.0,
        "position_distribution": [[{"zone_id": "1", "count": 2, "probability": 1.0}]],
    }


def test_chat_is_bound_to_the_current_motion_record_instead_of_an_unrelated_complete_session(
    tmp_path: Path,
) -> None:
    script = _client(tmp_path).get("/motion-viewer/assets/motion_viewer_chat.js")

    assert script.status_code == 200
    assert "motion-viewer-data" in script.text
    assert "data.pose_evaluation" in script.text
    assert "data.reference_order_evaluation" in script.text
    assert "const answer = data.answer;" in script.text
    assert "? data.answer\n        : structuredAnswer(task.id, prompt, viewerData)" not in script.text
    assert "const buildViewerQueryContext" in script.text
    assert "context: sentContext" in script.text
    assert "viewerData?.selected_session_id" in script.text
    assert "sentData?.mode_id" in script.text
    assert "순번별 주요 분포" not in script.text
    assert "이 시작 순서는 전문가 작업의 약" in script.text
    assert "/motion-viewer/api/completed" not in script.text
    assert "loadSessions" not in script.text


def test_chat_offers_response_depth_and_sends_it_to_the_query_api(tmp_path: Path) -> None:
    script = _client(tmp_path).get("/motion-viewer/assets/motion_viewer_chat.js")

    assert script.status_code == 200
    assert "빠른 안내" in script.text
    assert "상세 분석" in script.text
    assert "response_mode: sentMode" in script.text
    assert "use_rag: 'Y'" in script.text
    assert "let responseMode = 'quick'" in script.text


def test_chat_flag_assets_are_local_svg_images_with_a_fixed_allowlist(tmp_path: Path) -> None:
    client = _client(tmp_path)
    for country in ("kr", "us", "vn", "id", "th", "np", "mm", "uz", "lk", "ru", "cn", "kh"):
        flag = client.get(f"/motion-viewer/assets/flags/{country}.svg")
        assert flag.status_code == 200
        assert flag.headers["content-type"].startswith("image/svg+xml")
        assert "<svg" in flag.text
    assert client.get("/motion-viewer/assets/flags/xx.svg").status_code == 404


def test_joint_posture_state_uses_leg_and_torso_geometry_instead_of_camera_height() -> None:
    standing = {
        "Pelvis": [0.0, 1.0, 0.0], "Head": [0.0, 1.7, 0.0],
        "Left Thigh": [-0.1, 0.9, 0.0], "Left Calf": [-0.1, 0.5, 0.0], "Left Foot": [-0.1, 0.1, 0.0],
        "Right Thigh": [0.1, 0.9, 0.0], "Right Calf": [0.1, 0.5, 0.0], "Right Foot": [0.1, 0.1, 0.0],
    }
    deeply_bent = {
        **standing,
        "Left Calf": [-0.5, 0.55, 0.0], "Left Foot": [-0.1, 0.45, 0.0],
        "Right Calf": [0.5, 0.55, 0.0], "Right Foot": [0.1, 0.45, 0.0],
    }

    upright = _joint_posture_state(standing)
    bent = _joint_posture_state(deeply_bent)

    assert upright["key"] == "standing"
    assert upright["source"] == "csv_skeleton_joints"
    assert upright["knee_angle_deg"] == 180.0
    assert upright["left_knee_angle_deg"] == 180.0
    assert upright["right_knee_angle_deg"] == 180.0
    assert bent["key"] == "deep_bend"
    assert 45.0 <= bent["knee_angle_deg"] <= 75.0
    assert _joint_posture_state({"Head": [0.0, 1.7, 0.0]})["key"] == "unavailable"


def test_joint_posture_state_does_not_let_one_bad_ik_leg_force_a_deep_bend() -> None:
    joints = {
        "Pelvis": [0.0, 1.0, 0.0], "Head": [0.0, 1.7, 0.0],
        "Left Thigh": [-0.1, 0.9, 0.0], "Left Calf": [-0.5, 0.65, 0.0], "Left Foot": [-0.1, 0.4, 0.0],
        "Right Thigh": [0.1, 0.9, 0.0], "Right Calf": [0.1, 0.5, 0.0], "Right Foot": [0.1, 0.1, 0.0],
    }

    posture = _joint_posture_state(joints)

    assert posture["left_knee_angle_deg"] < 75.0
    assert posture["right_knee_angle_deg"] == 180.0
    assert posture["key"] == "standing"
    assert posture["confidence"] == "low"


def test_posture_smoothing_removes_a_single_frame_joint_flicker() -> None:
    standing = {"key": "standing", "label": "서 있는 자세", "source": "csv_skeleton_joints"}
    bent = {"key": "deep_bend", "label": "깊게 굽힌 자세", "source": "csv_skeleton_joints"}
    frames = [{"posture": dict(state)} for state in (standing, standing, bent, standing, standing)]

    _smooth_posture_states(frames, radius=2)

    assert [frame["posture"]["key"] for frame in frames] == ["standing"] * 5
    assert all(frame["posture"]["smoothed"] is True for frame in frames)


def test_motion_returns_trusted_trackers_paint_trace_and_direction_statistics(tmp_path: Path) -> None:
    response = _client(tmp_path).get(
        "/motion-viewer/api/motion",
        params={"process": "spray", "mode_id": 4, "session_id": "representative"},
    )

    assert response.status_code == 200
    data = response.json()
    first = data["frames"][0]
    assert first["tracked"] == {
        "head": [0.0, 1.7, 0.0],
        "right_hand": [0.3, 1.2, 0.2],
        "left_hand": [-0.3, 1.1, 0.1],
    }
    assert first["ray_point"] == [0.0, 1.4, 2.0]
    assert first["spraying"] is True
    assert first["zone_id"] == "1"
    assert first["distance_mm"] == 300
    assert len(data["paint_trace"]) == 3
    assert data["direction_summary"]["classification"] == "horizontal"
    assert data["direction_summary"]["label"] == "수평 중심"
    assert data["direction_summary"]["arrow_count"] > 0
    assert data["statistics"][0]["tracked_mean"]["head"] == [0.0, 1.7, 0.0]
    assert data["statistics"][0]["ray_mean"] == [0.05, 1.4, 2.0]
    assert data["statistics"][0]["ray_spread"] > 0
    assert data["reference_surface"]["raw_point_count"] == 10
    assert len(data["reference_surface"]["points"]) == 10
    assert data["direction_field"]
    assert data["direction_distribution"]["horizontal"] == 1.0
    assert data["direction_distribution"]["vertical"] == 0.0
    assert len(data["zone_flow_guides"]) == 1
    guide = data["zone_flow_guides"][0]
    assert guide["map_num"] == "1"
    assert guide["zone_id"] == "1"
    assert guide["stroke_count"] == 1
    assert guide["strokes"][0]["order"] == 1
    assert guide["strokes"][0]["session_count"] == 2
    assert guide["strokes"][0]["confidence"] == 1.0
    assert len(guide["strokes"][0]["points"]) >= 2
    assert data["zone_surfaces"][0]["map_num"] == "1"
    assert data["zone_surfaces"][0]["zone_id"] == "1"
    assert data["zone_surfaces"][0]["vertices"]
    assert data["zone_order"] == [{"map_num": "1", "zone_id": "1", "order": 1}]
    zone = data["zone_statistics"]["1:1"]
    assert zone["expert_session_count"] == 2
    assert zone["median_speed_mms"] == 300
    assert zone["median_distance_mm"] == 300
    assert zone["selected_speed_mms"] == 300
    assert zone["selected_distance_mm"] == 300
    assert zone["selected_major_direction"] == "horizontal"
    assert zone["expert_direction_distribution"] == {
        "horizontal": 1.0,
        "vertical": 0.0,
        "diagonal": 0.0,
    }
    assert zone["observed_path_length_m"] > 0
    assert zone["observed_duration_seconds"] > 0
    overall = data["zone_statistics"]["all"]
    assert overall["zone_count"] == 1
    assert overall["expert_session_count"] == 2
    assert overall["selected_major_direction"] == "horizontal"
    assert overall["expert_direction_distribution"]["horizontal"] == 1.0
    coating = data["coating_reference"]["1:1"]
    assert coating["source"] == "expert_final_heightmap"
    assert coating["display_semantics"] == "physical_thickness_um"
    assert coating["unit"] == "um"
    assert coating["grid_size"] == 64
    assert coating["bounds"] == {"min_x": 10, "max_x": 11, "min_y": 20, "max_y": 20}
    assert coating["cells"] == [[10, 20, 140.0], [11, 20, 180.0]]


def test_representative_expert_motion_uses_other_sessions_as_the_reference_set(
    tmp_path: Path,
) -> None:
    response = _client(tmp_path).get(
        "/motion-viewer/api/motion",
        params={"process": "spray", "mode_id": 4, "session_id": "representative"},
    )

    assert response.status_code == 200
    evaluation = response.json()["pose_evaluation"]
    assert evaluation["status"] == "available"
    assert evaluation["reference_session_count"] == 1
    assert evaluation["similarity_percent"] >= 60
    assert evaluation["reference_session_id"] == "multi_session_expert_profile"


def test_zone_surface_keeps_non_planar_three_dimensional_vertices() -> None:
    samples = {
        (0, 0): {"sum": [0.0, 0.0, 0.0], "count": 1},
        (1, 0): {"sum": [1.0, 0.0, 0.0], "count": 1},
        (0, 1): {"sum": [0.0, -0.4, 1.0], "count": 1},
        (1, 1): {"sum": [1.0, -0.7, 1.0], "count": 1},
    }

    surface = _build_zone_surface("2", "7", samples)

    assert surface["map_num"] == "2"
    assert surface["zone_id"] == "7"
    assert surface["vertices"] == [
        [0.0, 0.0, 0.0],
        [0.0, -0.4, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, -0.7, 1.0],
    ]
    assert surface["triangles"] == [[0, 2, 1], [2, 3, 1]]
    assert sorted(surface["boundary_edges"]) == [[0, 1], [0, 2], [1, 3], [2, 3]]


def test_dense_paint_trace_keeps_long_sequences_above_motion_frame_limit(tmp_path: Path) -> None:
    path = tmp_path / "long.csv"
    fields = [
        "시간(초)", "분사", "Ray point X", "Ray Point Y", "Ray Point Z",
        "Ray Texture Map Number", "Ray Texture UV X", "Ray Texture UV Y", "Zone ID",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(1200):
            writer.writerow({
                "시간(초)": index / 120,
                "분사": "On",
                "Ray point X": index / 1000,
                "Ray Point Y": 1.0,
                "Ray Point Z": 2.0,
                "Ray Texture Map Number": 1,
                "Ray Texture UV X": index % 2048,
                "Ray Texture UV Y": 100,
                "Zone ID": 3,
            })

    trace = _dense_paint_trace(path, target_hz=30.0)

    assert 290 <= len(trace) <= 310
    assert len(trace) > 240
    assert trace[0]["segment"] == 0
    assert trace[-1]["t"] > 9.9


def test_back_and_forth_paint_stroke_is_classified_by_dominant_axis() -> None:
    trace = [
        {"point": [x, 1.4, 2.0]}
        for x in (0.0, 1.0, 0.0, 1.0, 0.1)
    ]

    summary = _direction_summary(trace)

    assert summary["classification"] == "horizontal"
    assert summary["label"] == "수평 중심"


def test_motion_rejects_unknown_session_without_reading_arbitrary_paths(tmp_path: Path) -> None:
    response = _client(tmp_path).get(
        "/motion-viewer/api/motion",
        params={"process": "spray", "mode_id": 4, "session_id": "../../secret"},
    )

    assert response.status_code == 404


def test_evaluation_catalog_lists_stored_tsb_sequences_without_bodies(tmp_path: Path) -> None:
    store = SequenceStore(":memory:")
    _store_sequence(store)

    response = _client(tmp_path, store=store).get("/motion-viewer/api/evaluations")

    assert response.status_code == 200
    payload = response.json()
    assert payload["sessions"][0]["session_id"] == "3_11000346_1"
    assert payload["sessions"][0]["sequences"][0]["sequence_id"] == "SEQ-020"
    assert "sequence_info" not in json.dumps(payload)


def test_evaluation_endpoint_uses_injected_store_and_product_scorer(tmp_path: Path) -> None:
    store = SequenceStore(":memory:")
    expected_body = _store_sequence(store)
    calls: list[tuple[dict[str, Any], str]] = []

    def scorer(body: dict[str, Any], *, sequence_id: str, **_kwargs: Any) -> PoseEngineResult:
        calls.append((body, sequence_id))
        return PoseEngineResult(
            "available", None, sequence_id,
            {
                "analysis_status": "available", "similarity_score": 0.91,
                "expert_percentile": 90, "reference_quality": "multi_worker",
                "support": 200, "corrections": [],
            },
        )

    response = _client(tmp_path, store=store, pose_scorer=scorer).get(
        "/motion-viewer/api/evaluation",
        params={"session_id": "3_11000346_1", "sequence_id": "SEQ-020"},
    )

    assert response.status_code == 200
    assert calls == [(expected_body, "SEQ-020")]
    assert response.json()["evaluation"]["similarity_score"] == 0.91


def test_evaluation_endpoint_returns_404_for_unknown_sequence_and_200_unavailable_for_engine_failure(
    tmp_path: Path,
) -> None:
    store = SequenceStore(":memory:")
    _store_sequence(store)

    def unavailable(_body: dict[str, Any], *, sequence_id: str, **_kwargs: Any) -> PoseEngineResult:
        return PoseEngineResult("not_available", "engine_unavailable", sequence_id, {})

    client = _client(tmp_path, store=store, pose_scorer=unavailable)
    missing = client.get(
        "/motion-viewer/api/evaluation",
        params={"session_id": "3_11000346_1", "sequence_id": "SEQ-999"},
    )
    unavailable_response = client.get(
        "/motion-viewer/api/evaluation",
        params={"session_id": "3_11000346_1", "sequence_id": "SEQ-020"},
    )

    assert missing.status_code == 404
    assert unavailable_response.status_code == 200
    assert unavailable_response.json()["evaluation"]["reason_code"] == "engine_unavailable"
    assert unavailable_response.json()["evaluation"]["similarity_score"] is None


def test_motion_endpoint_adds_verified_object_mesh_and_preserves_existing_fields(tmp_path: Path) -> None:
    def mesh(_mode: str) -> dict[str, Any]:
        return {
            "available": True,
            "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            "triangles": [[0, 1, 2]],
        }

    response = _client(tmp_path, mesh_provider=mesh).get(
        "/motion-viewer/api/motion",
        params={"process": "spray", "mode_id": 4, "session_id": "representative"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["frames"]
    assert payload["zone_surfaces"]
    assert payload["object_mesh"]["triangles"] == [[0, 1, 2]]
    assert payload["object_mesh_status"] == "verified"


def test_motion_endpoint_scores_selected_record_against_other_expert_zone_records(
    tmp_path: Path,
) -> None:
    response = _client(tmp_path).get(
        "/motion-viewer/api/motion",
        params={"process": "spray", "mode_id": 4, "session_id": "4_1002_1"},
    )

    assert response.status_code == 200
    payload = response.json()
    evaluation = payload["pose_evaluation"]
    assert 60 <= evaluation["similarity_percent"] <= 100
    assert 60 <= evaluation["order_similarity_percent"] <= 100
    assert evaluation["reference_session_count"] == 1
    assert evaluation["comparison_basis"] == "multi_session_zone_normalized"
    assert "duration" not in json.dumps(evaluation).casefold()

    zone = evaluation["zones"]["1:1"]
    assert zone["zone_id"] == "1"
    assert zone["reference_session_count"] == 1
    assert 60 <= zone["similarity_percent"] <= 100
    metrics = {item["key"]: item for item in zone["metrics"]}
    assert set(metrics) == {"tool_angle", "posture", "paint_pattern", "coating_thickness", "distance"}
    assert metrics["tool_angle"]["actual"] is not None
    assert metrics["tool_angle"]["reference"] is not None
    assert metrics["tool_angle"]["unit"] == "deg"
    assert metrics["posture"]["actual_label"] == "서 있는 자세"
    assert metrics["posture"]["reference_label"] == "서 있는 자세"
    assert metrics["paint_pattern"]["actual_label"] == "수평형"
    assert metrics["coating_thickness"]["actual"] == 170
    assert metrics["coating_thickness"]["reference"] == 160
    assert metrics["distance"]["actual"] == 300
    assert metrics["distance"]["reference"] == 300
    assert all(0 <= item["similarity_percent"] <= 100 for item in metrics.values())
    assert zone["corrections"] == []
    assert zone["predicted_similarity_percent"] == zone["similarity_percent"]


def test_motion_endpoint_uses_percentages_for_self_replay_without_self_reference_leakage(
    tmp_path: Path,
) -> None:
    payload = _client(tmp_path).get(
        "/motion-viewer/api/motion",
        params={"process": "spray", "mode_id": 4, "session_id": "4_1001_1"},
    ).json()

    assert payload["pose_evaluation"]["reference_session_count"] == 1
    assert payload["pose_evaluation"]["similarity_percent"] >= 60
    assert payload["pose_evaluation"]["similarity_score"] == (
        payload["pose_evaluation"]["similarity_percent"] / 100
    )


def test_trainee_motion_uses_all_expert_sessions_instead_of_trainee_repetitions(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)
    worker_root = tmp_path / "workers" / "kim_gyeongho"
    (worker_root / "worker.json").parent.mkdir(parents=True, exist_ok=True)
    (worker_root / "worker.json").write_text(
        json.dumps({"display_name": "김경호", "role": "trainee"}), encoding="utf-8"
    )
    expert_profile_path = tmp_path / "generated" / "profiles" / "spray" / "learning" / "mode_04.json"
    trainee_profile = json.loads(expert_profile_path.read_text(encoding="utf-8"))
    trainee_profile["sessions"] = [dict(trainee_profile["sessions"][0])]
    trainee_profile["sessions"][0]["session_id"] = "4_TRAINEE_1"
    trainee_profile["sessions"][0]["sources"] = {"csv_path": "4_TRAINEE_1.csv"}
    trainee_profile_path = worker_root / "generated" / "profiles" / "spray" / "learning" / "mode_04.json"
    trainee_profile_path.parent.mkdir(parents=True, exist_ok=True)
    trainee_profile_path.write_text(json.dumps(trainee_profile), encoding="utf-8")
    trainee_csv = worker_root / "raw" / "spray" / "learning" / "mode_04" / "4_TRAINEE_1.csv"
    _write_csv(trainee_csv, offset=0.05)

    payload = client.get(
        "/motion-viewer/api/motion",
        params={
            "worker_id": "kim_gyeongho", "process": "spray", "mode_id": 4,
            "session_id": "4_TRAINEE_1",
        },
    ).json()

    assert payload["pose_evaluation"]["reference_session_count"] == 2
    assert payload["pose_evaluation"]["comparison_basis"] == "multi_session_zone_normalized"
    assert payload["pose_evaluation"]["similarity_percent"] >= 60


def test_motion_viewer_page_exposes_pose_comparison_contract(tmp_path: Path) -> None:
    response = _client(tmp_path).get("/motion-viewer")

    assert response.status_code == 200
    for text in (
        "작업 자세 비교 평가",
        "도장 경로와 작업 자세를 전문가 기준으로 비교합니다.",
        '<a class="v2-switch" href="/motion-viewer/v2">v2 화면으로 전환',
        "작업자",
        "전문가 작업 유사도",
        "작업 개선 안내",
        "안내 적용 시 예상",
        "현재 Zone 작업 유사도",
        "현재 자세 상태",
        "Zone 진행 구간",
        "CSV 조인트 기반 전신 스켈레톤",
        "CSV 조인트 기반 자세 추정",
        "현재 Zone 도막 분포",
        "2D 분포",
        "3D 높이",
    ):
        assert text in response.text
    for removed in (
        "작업자 / TSB 세션", "선택 시퀀스 자세 평가",
        "머리·양손 기반 추정 상체 · 별도 화면",
        'id="evaluationSession"', 'id="evaluationSequence"',
    ):
        assert removed not in response.text
    assert 'id="worker"' in response.text
    assert response.text.index('id="worker"') < response.text.index('id="process"')
    assert response.text.count('id="timeline"') == 1
    assert 'id="zoneTimeline"' in response.text
    assert response.text.index('id="zoneTimeline"') < response.text.index('id="timeline"')
    assert response.text.index('id="timeline"') < response.text.index('id="play"')
    assert response.text.index('id="play"') < response.text.index('id="speed"')
    assert 'id="canvas"' in response.text
    assert 'id="poseCanvas"' not in response.text
    assert 'class="viewer-tab"' not in response.text
    assert 'id="cameraHome"' in response.text
    assert 'id="cameraAxisX"' in response.text
    assert 'id="cameraAxisY"' in response.text
    assert 'id="cameraAxisZ"' in response.text
    assert '/motion-viewer/assets/motion_viewer_pose.css' in response.text
    assert '/motion-viewer/assets/motion_viewer_pose.js' in response.text
    assert '/motion-viewer/assets/motion_viewer_camera.js' in response.text
    assert '/motion-viewer/assets/motion_viewer_coating.js' in response.text
    assert 'class="selection-panel"' in response.text
    assert 'class="selection-grid"' in response.text
    assert 'id="coatingCanvas"' in response.text
    assert 'id="coatingView2d"' in response.text
    assert 'id="coatingView3d"' in response.text
    assert "Number(zoneId)>0" in response.text


def test_pose_summary_uses_wide_zone_metric_rows_with_centered_label_and_progress(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)
    page = client.get("/motion-viewer")
    script = client.get("/motion-viewer/assets/motion_viewer_pose.js")
    style = client.get("/motion-viewer/assets/motion_viewer_pose.css")

    assert page.status_code == 200
    assert 'id="poseZoneCard"' in page.text
    assert 'id="poseZoneMetrics"' in page.text
    assert 'id="poseActionList"' in page.text
    assert "grid-template-columns:368px minmax(420px,1fr) 300px" in page.text
    assert "overflow-y:auto;overflow-x:hidden" in page.text
    assert "renderZoneEvaluation" in script.text
    assert "similarity_percent" in script.text
    assert "predicted_similarity_percent" in script.text
    assert "pose-zone-metric-values" in script.text
    assert "pose-zone-progress" in script.text
    assert "현재 :" in script.text
    assert "실제" not in script.text
    assert "전문가" in script.text
    assert ".pose-zone-metric" in style.text
    assert "align-items: center" in style.text
    assert "grid-template-columns: 68px minmax(92px, 1fr) 92px" in style.text
    assert ".evaluation-side { overflow-x:hidden; }" in style.text


def test_pose_guidance_is_scoped_to_current_zone_and_speed_select_is_readable(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)
    page = client.get("/motion-viewer")
    script = client.get("/motion-viewer/assets/motion_viewer_pose.js")

    assert page.status_code == 200
    assert 'id="poseActionTitle"' in page.text
    assert "Zone ${zoneLabel(zoneEvaluation.zone_id)} 작업 개선 안내" in script.text
    assert "renderCorrections(zoneEvaluation.corrections);" in script.text
    assert "zoneEvaluation.corrections?.length ? zoneEvaluation.corrections : evaluation?.corrections" not in script.text
    assert ".speed-control select{width:100%;font-weight:800;color:#e5f1ff;background:#14243a" in page.text
    assert ".speed-control select option{color:#e5f1ff;background:#14243a}" in page.text


def test_playback_status_exposes_a_persistent_colored_zone_badge(tmp_path: Path) -> None:
    client = _client(tmp_path)
    page = client.get("/motion-viewer")
    script = client.get("/motion-viewer/assets/motion_viewer_pose.js")

    assert page.status_code == 200
    assert 'id="playbackZoneBadge"' in page.text
    assert 'class="playback-zone-badge"' in page.text
    assert "let lastPlaybackZone = null;" in script.text
    assert "lastPlaybackZone = currentZone" in script.text
    assert "lastPlaybackZone = null" in script.text


def test_viewer_overlay_contains_a_matching_translucent_realtime_hud(tmp_path: Path) -> None:
    client = _client(tmp_path)
    page = client.get("/motion-viewer")
    script = client.get("/motion-viewer/assets/motion_viewer_pose.js")

    assert page.status_code == 200
    assert 'id="realtimeHud"' in page.text
    assert page.text.index('id="viewerLegend"') < page.text.index('id="realtimeHud"')
    for element_id in (
        "realtimeHudZone",
        "realtimeHudActivity",
        "realtimeHudSpeed",
        "realtimeHudDistance",
        "realtimeHudAxis",
        "realtimeHudDirection",
        "realtimeHudCoverage",
    ):
        assert f'id="{element_id}"' in page.text
    assert ".realtime-hud" in page.text
    assert "background:rgba(7,16,31,.42)" in page.text
    assert "updateRealtimeHud" in script.text
    assert "HUD_WINDOW_SECONDS = .5" in script.text
    assert "speed_mms" in script.text
    assert "distance_mm" in script.text
    assert "lastRealtimeZone" in script.text
    assert "value == null || value === ''" in script.text
    assert "buildRealtimeGuidance" in script.text
    assert "reference_uv_cell_count" in script.text
    assert "sample.uv" in script.text
    assert "도장 방향" in page.text
    assert "도포 범위" in page.text


def test_pose_assets_define_layers_and_interaction_states(tmp_path: Path) -> None:
    client = _client(tmp_path)
    script = client.get("/motion-viewer/assets/motion_viewer_pose.js")
    style = client.get("/motion-viewer/assets/motion_viewer_pose.css")
    camera = client.get("/motion-viewer/assets/motion_viewer_camera.js")
    coating = client.get("/motion-viewer/assets/motion_viewer_coating.js")

    assert script.status_code == 200
    assert style.status_code == 200
    assert camera.status_code == 200
    assert coating.status_code == 200
    assert script.headers["cache-control"] == "no-cache"
    assert style.headers["cache-control"] == "no-cache"
    for text in (
        "drawTrustedUpperBody",
        "drawOrientationGuides",
        "renderPoseSummary",
        "renderZoneTimeline",
        "camera_forward",
        "tool_forward",
        "ray_point",
        "timeline",
        "play",
    ):
        assert text in script.text
    for removed in ("evaluationLoadGeneration", "evaluationSession", "evaluationSequence", "loadEvaluation"):
        assert removed not in script.text
    assert "timelineBucketSeconds" in script.text
    assert "zoneCounts" in script.text
    assert "sample.segment" not in script.text
    assert "pairwise" not in script.text.lower()
    assert "viewer-tab" not in script.text
    for text in ("fit", "orbit", "pan", "dolly", "look", "fly", "focus", "axisView", "home"):
        assert text in camera.text
    assert "#38BDF8" in style.text
    assert "#22C55E" in style.text
    assert "#FACC15" in style.text
    assert "display:none" not in style.text.replace(" ", "").split("@media(max-width:1200px)", 1)[-1]
    assert "resize();" in script.text
    assert "data?.frames?.length) { resetViewToCamera(data); renderZoneTimeline(data); }" in script.text
    for text in (
        "coating_reference",
        "motion-viewer-frame",
        "renderCoatingZone",
        "drawTerrain",
        "expert_final_heightmap",
    ):
        assert text in coating.text


def test_skeleton_viewer_redraws_only_the_right_arm_from_the_measured_gun(tmp_path: Path) -> None:
    client = _client(tmp_path)
    page = client.get("/motion-viewer")
    script = client.get("/motion-viewer/assets/motion_viewer_pose.js")

    assert page.status_code == 200
    assert script.status_code == 200
    assert 'id="skeletonCorrection"' not in page.text
    assert "하반신 자세 보정" not in page.text
    assert "function correctLowerBody" not in script.text
    assert "function solveTwoBoneLeg" not in script.text
    assert "motionViewerCorrectLowerBody" not in page.text
    # 패널과 v2 교정 고리가 같은 함수를 쓴다. 다시 세운 팔의 위치와 선 모양은 test_motion_viewer_chirality.js 가 잰다.
    assert "const {joints}=panelSkeleton(frame)" in page.text
    v2 = client.get("/motion-viewer/assets/motion_viewer_v2.js")
    assert v2.status_code == 200
    assert "panelSkeleton(frame).joints" in v2.text
    assert "uprightSkeleton(" not in v2.text


def test_motion_viewer_keeps_the_last_spraying_zone_and_gives_skeleton_more_height(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)
    page = client.get("/motion-viewer")
    coating = client.get("/motion-viewer/assets/motion_viewer_coating.js")

    assert page.status_code == 200
    assert coating.status_code == 200
    assert (
        ".right{padding:12px;display:grid;"
        "grid-template-rows:auto minmax(360px,1fr)"
    ) in page.text
    assert ".skeleton-card{height:auto;display:block}" in page.text
    assert ".skeleton-screen{height:var(--skeleton-view-height,310px);min-height:var(--skeleton-view-height,310px);flex:none}" in page.text
    assert ".coating-screen{position:relative;min-height:100px;flex:1" in page.text
    assert (
        '<div id="coatingStatus" class="coating-status">도막 기준 준비 중</div>'
        '<div class="coating-screen"><canvas id="coatingCanvas"'
    ) in page.text
    assert ".coating-status{flex:none;margin:0 10px 8px" in page.text
    assert ".coating-status{position:absolute" not in page.text
    assert ".skeleton-screen{height:300px;min-height:300px}" in page.text
    assert "frame && frame.spraying && Number(frame.zone_id) > 0" in coating.text
    assert "if (!next || next === zoneKey) return;" in coating.text
    assert "zoneKey = next;" in coating.text


# --- 목표 대비 도막 평가(PR #8 기준) — 기록 옆 높이맵 PNG로 계산한다 ---------------------------

_TARGET_MODE_ROOT = Path("raw") / "spray" / "learning" / "mode_04"


def _zone_one_mask(size: int = 2048) -> Any:
    import numpy as np
    from trainer.zone_reference import ZoneReferenceCatalog

    return np.asarray(ZoneReferenceCatalog.default().zone_mask("Learn04", 1, "1", size=(size, size))) > 0


def _write_height_map(tmp_path: Path, name: str, raw_um: float, *, size: int = 2048) -> Path:
    """Learn04 1번 Zone 안만 원시 두께 raw_um(B>0)인 TSB RG16 높이맵."""
    import io
    import numpy as np
    from PIL import Image

    mask = _zone_one_mask(size)
    code = int(np.rint(raw_um / 20000.0 * 65025.0))
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[mask] = (code // 255, code % 255, 255, 255)
    stream = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(stream, format="PNG")
    path = tmp_path / _TARGET_MODE_ROOT / name
    path.write_bytes(stream.getvalue())
    return path


def _motion(client: TestClient) -> dict[str, Any]:
    response = client.get("/motion-viewer/api/motion", params={"process": "spray", "mode_id": 4, "session_id": "4_1002_1"})
    assert response.status_code == 200, response.text
    return response.json()


def test_motion_endpoint_adds_target_assessment_from_the_record_height_map(tmp_path: Path) -> None:
    from trainer.demo_api import coating_assessment

    client = _client(tmp_path, heightmaps=["Map01_1002_1_1_Height.png"])
    _write_height_map(tmp_path, "Map01_1002_1_1_Height.png", 181)
    coating = _motion(client)["coating_evaluation"]
    target = coating["target"]

    assert target["analysis_status"] == "available" and target["reason_code"] is None
    assert (target["target_thickness_um"], target["target_basis"]) == (160.0, "default_target")
    assert target["basis_label"] == "기본 목표 160µm 기준"
    assert target["criterion_version"] == coating_assessment.ASSESSMENT_VERSION
    assert target["normal_band_um"] == coating_assessment.normal_band_um(160.0)
    assert target["band_edges_um"] == coating_assessment.band_edges_um(160.0)
    assert target["band_labels"] == list(coating_assessment.BAND_LABELS)
    assert target["grade_labels"] == coating_assessment.GRADE_SHORT_LABELS
    row = next(item for item in target["zones"] if (item["map_num"], item["zone_id"]) == (1, "1"))
    assert row["grade"] == "normal" and row["status"] == "normal"
    assert row["thickness_um"] == pytest.approx(171, abs=0.5)
    assert row["unpainted_fraction"] == 0.0
    assert target["summary"]["zone_count"] == len(target["zones"])
    assert target["summary"]["grade_counts"]["normal"] == 1
    assert set(target["summary"]) == {"zone_count", "grade_counts", "painted_mean_um", "unpainted_fraction", "six_band_counts"}
    # 전문가 대비 자료(D5)는 그대로 있다.
    assert coating["comparison_basis"] == "expert_zone_median"
    assert coating["zones"][0]["reference_um"] > 0
    assert coating["analysis_status"] == "available"


def test_record_without_a_height_map_keeps_the_expert_comparison(tmp_path: Path) -> None:
    coating = _motion(_client(tmp_path))["coating_evaluation"]

    assert coating["target"] == {"analysis_status": "not_available", "reason_code": "height_map_unavailable"}
    assert coating["zones"]


def test_another_records_height_map_in_the_same_folder_is_not_used(tmp_path: Path) -> None:
    client = _client(tmp_path, heightmaps=["Map01_1001_1_1_Height.png"])
    _write_height_map(tmp_path, "Map01_1001_1_1_Height.png", 181)

    assert _motion(client)["coating_evaluation"]["target"]["reason_code"] == "height_map_unavailable"


def test_target_assessment_is_inside_the_signed_coating_claims(tmp_path: Path) -> None:
    from trainer.demo_api.viewer_context import verify_context

    client = _client(tmp_path, heightmaps=["Map01_1002_1_1_Height.png"])
    _write_height_map(tmp_path, "Map01_1002_1_1_Height.png", 181)
    payload = _motion(client)
    claims, signature = payload["coating_evaluation"], payload["query_context_signatures"]["coating_status"]

    assert verify_context("coating_status", claims, signature)
    claims["target"]["zones"][0]["grade"] = "above"
    assert not verify_context("coating_status", claims, signature)


def test_dialogue_facts_carry_the_target_grade_when_available(tmp_path: Path) -> None:
    client = _client(tmp_path, heightmaps=["Map01_1002_1_1_Height.png"])
    _write_height_map(tmp_path, "Map01_1002_1_1_Height.png", 181)
    facts = _motion(client)["dialogue_evaluation"]

    assert facts["zones"]["1:1"]["coating_target"]["grade"] == "normal"
    assert facts["zones"]["1:1"]["coating_target"]["label"].startswith("정상(칠한 곳 평균 ")
    assert facts["zones"]["1:1"]["coating_target"]["unpainted_percent"] == 0
    assert (facts["coating_target_um"], facts["coating_target_basis"]) == (160.0, "default_target")

    plain = _motion(_client(tmp_path / "plain"))["dialogue_evaluation"]
    assert "coating_target" not in plain["zones"]["1:1"]
    assert "coating_target_um" not in plain and "coating_target_basis" not in plain


def test_target_facts_do_not_unlock_the_reference_word_guard(tmp_path: Path) -> None:
    from trainer.demo_api import dialogue

    client = _client(tmp_path, heightmaps=["Map01_1002_1_1_Height.png"])
    _write_height_map(tmp_path, "Map01_1002_1_1_Height.png", 181)
    facts = _motion(client)["dialogue_evaluation"]

    assert "coating_target" in facts["zones"]["1:1"]
    assert "표준" in dialogue.ungrounded_measurements("속도는 표준 대비 높습니다.", facts=facts, prompt="")
    text = json.dumps(facts, ensure_ascii=False)
    assert "기준" not in text and "목표" not in text


def test_path_escape_in_heightmap_source_is_ignored(tmp_path: Path) -> None:
    client = _client(tmp_path, heightmaps=["../../x/Map01_1002_1_1_Height.png"])
    escaped = tmp_path / "raw" / "spray" / "x" / "Map01_1002_1_1_Height.png"
    escaped.parent.mkdir(parents=True)
    _write_height_map(tmp_path, "Map01_1002_1_1_Height.png", 181).rename(escaped)

    assert _motion(client)["coating_evaluation"]["target"]["reason_code"] == "height_map_unavailable"


def test_latest_run_number_is_compared_as_an_integer(tmp_path: Path) -> None:
    client = _client(tmp_path, heightmaps=["Map01_1002_1_10_Height.png", "Map01_1002_1_9_Height.png"])
    _write_height_map(tmp_path, "Map01_1002_1_9_Height.png", 100)
    _write_height_map(tmp_path, "Map01_1002_1_10_Height.png", 181)

    target = _motion(client)["coating_evaluation"]["target"]
    assert target["zones"][0]["grade"] == "normal"


def test_height_map_of_another_size_withholds_the_target_assessment(tmp_path: Path) -> None:
    client = _client(tmp_path, heightmaps=["Map01_1002_1_1_Height.png"])
    _write_height_map(tmp_path, "Map01_1002_1_1_Height.png", 181, size=1024)
    coating = _motion(client)["coating_evaluation"]

    assert coating["target"] == {"analysis_status": "not_available", "reason_code": "height_map_size_mismatch"}
    assert coating["zones"][0]["reference_um"] > 0


def test_zones_on_a_face_without_a_height_map_get_no_target_row(tmp_path: Path) -> None:
    from trainer.demo_api.motion_viewer import _target_coating

    (tmp_path / _TARGET_MODE_ROOT).mkdir(parents=True)
    path = _write_height_map(tmp_path, "Map01_1002_1_1_Height.png", 181)
    target = _target_coating("spray", 4, {1: path}, ["1:1", "2:1"], {})

    assert target["analysis_status"] == "available"
    assert [(row["map_num"], row["zone_id"]) for row in target["zones"]] == [(1, "1")]
    assert _target_coating("spray", 4, {1: path}, ["2:1"], {})["reason_code"] == "height_map_unavailable"
    assert _target_coating("blasting", 4, {1: path}, ["1:1"], {})["reason_code"] == "coating_not_applicable"


def test_target_assessment_is_cached_per_file_stamp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os
    from trainer.demo_api import coating_assessment
    from trainer.demo_api.motion_viewer import _target_coating

    calls: list[int] = []
    original = coating_assessment.painted_surface_from_png
    monkeypatch.setattr(coating_assessment, "painted_surface_from_png", lambda raw: calls.append(1) or original(raw))
    client = _client(tmp_path, heightmaps=["Map01_1002_1_1_Height.png"])
    path = _write_height_map(tmp_path, "Map01_1002_1_1_Height.png", 181)

    first = _motion(client)["coating_evaluation"]["target"]
    second = _motion(client)["coating_evaluation"]["target"]
    assert len(calls) == 1 and first == second

    _write_height_map(tmp_path, "Map01_1002_1_1_Height.png", 260)
    stamp = path.stat().st_mtime_ns + 1_000_000_000
    os.utime(path, ns=(stamp, stamp))
    third = _motion(client)["coating_evaluation"]["target"]
    assert len(calls) == 2 and third["zones"][0]["thickness_um"] != first["zones"][0]["thickness_um"]

    # 캐시된 결과를 응답 조립이 고쳐도 다음 응답은 그대로다(깊은 복사).
    direct = _target_coating("spray", 4, {1: path}, ["1:1"], {})
    direct["zones"][0]["grade"] = "forged"
    assert _target_coating("spray", 4, {1: path}, ["1:1"], {})["zones"][0]["grade"] != "forged"


def test_target_assessment_reads_one_face_at_a_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """면마다 표면을 읽고 평가한 뒤 놓는다. 다음 면을 읽을 때 앞 면의 2048² 표면이 남아 있지 않다."""
    import gc
    import weakref
    from trainer.demo_api import coating_assessment
    from trainer.demo_api.motion_viewer import _target_coating

    alive_at_read: list[int] = []
    surfaces: list[Any] = []
    original = coating_assessment.painted_surface_from_png

    def read(raw: bytes) -> Any:
        gc.collect()
        alive_at_read.append(sum(ref() is not None for ref in surfaces))
        surface = original(raw)
        surfaces.append(weakref.ref(surface))
        return surface

    monkeypatch.setattr(coating_assessment, "painted_surface_from_png", read)
    (tmp_path / _TARGET_MODE_ROOT).mkdir(parents=True)
    path = _write_height_map(tmp_path, "Map01_1002_1_1_Height.png", 181)
    copy = path.with_name("Map02_1002_1_1_Height.png")
    copy.write_bytes(path.read_bytes())
    target = _target_coating("spray", 4, {1: path, 2: copy}, ["1:1", "2:1"], {})

    assert alive_at_read == [0, 0]
    # 값은 한 면만 줄 때와 같다(2면에는 Learn04 Zone 마스크가 없다).
    assert target == _target_coating("spray", 4, {1: path}, ["1:1"], {})


def test_viewer_scripts_do_not_restate_the_band_rule() -> None:
    """등급 경계·칸 이름·등급 글자는 서버(target의 표시용 열쇠)에서만 온다."""
    static = Path(__file__).resolve().parents[1] / "static"
    for name in ("motion_viewer_coating.js", "motion_viewer_v2.js"):
        script = (static / name).read_text(encoding="utf-8")
        for word in ("1.5배", "90% 미만", "목표 미만"):
            assert word not in script, (name, word)


def test_coating_card_offers_both_bases_with_visible_labels(tmp_path: Path) -> None:
    page = _client(tmp_path).get("/motion-viewer").text

    assert '<button id="coatingBasisTarget" class="active" type="button" aria-pressed="true">목표 대비</button>' in page
    assert '<button id="coatingBasisExpert" type="button" aria-pressed="false">전문가 대비</button>' in page
    assert 'id="coatingTargetPanel"' in page
    assert "<strong>도막</strong>" in page


def test_v2_page_loads_the_layout_helpers_before_the_v2_script(tmp_path: Path) -> None:
    """화면 나누기·피드백·첨부 규칙은 motion_viewer_v2_layout.js 에 있고 v2 스크립트보다 먼저 실려야 한다."""
    client = _client(tmp_path)
    page = client.get("/motion-viewer/v2").text
    helpers = page.index('src="/motion-viewer/assets/motion_viewer_v2_layout.js"')
    assert helpers < page.index('src="/motion-viewer/assets/motion_viewer_v2.js"')
    script = client.get("/motion-viewer/assets/motion_viewer_v2_layout.js")
    assert script.status_code == 200
    assert script.headers["content-type"].startswith("text/javascript")
    assert "window.motionViewerV2Layout" in script.text
