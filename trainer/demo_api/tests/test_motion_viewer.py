import csv
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.motion_viewer import (
    _build_zone_surface,
    _dense_paint_trace,
    _direction_summary,
    create_motion_viewer_router,
)


def _write_csv(path: Path, *, offset: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "인덱스",
        "시간(초)",
        "분사",
        "한길숏건 Position X", "한길숏건 Position Y", "한길숏건 Position Z",
        "플래쉬 Position X", "플래쉬 Position Y", "플래쉬 Position Z",
        "카메라 Position X", "카메라 Position Y", "카메라 Position Z",
        "Ray point X", " Ray Point Y", " Ray Point Z",
        "Ray Texture Map Number", "Ray Texture UV X", "Ray Texture UV Y",
        "Zone ID",
        "Pelvis Position X", "Pelvis Position Y", "Pelvis Position Z",
        "Pelvis Rotation X", "Pelvis Rotation Y", "Pelvis Rotation Z",
        "Spine Position X", "Spine Position Y", "Spine Position Z",
        "Spine Rotation X", "Spine Rotation Y", "Spine Rotation Z",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(5):
            writer.writerow(
                {
                    "인덱스": index,
                    "시간(초)": index / 90,
                    "분사": "True",
                    "한길숏건 Position X": 0.3 + index * 0.01,
                    "한길숏건 Position Y": 1.2,
                    "한길숏건 Position Z": 0.2,
                    "플래쉬 Position X": -0.3 + index * 0.01,
                    "플래쉬 Position Y": 1.1,
                    "플래쉬 Position Z": 0.1,
                    "카메라 Position X": index * 0.01,
                    "카메라 Position Y": 1.7,
                    "카메라 Position Z": 0,
                    "Ray point X": offset + index * 0.2,
                    " Ray Point Y": 1.4,
                    " Ray Point Z": 2.0,
                    "Ray Texture Map Number": 1,
                    "Ray Texture UV X": (index % 2) * 32,
                    "Ray Texture UV Y": (index // 2) * 32,
                    "Zone ID": "1",
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
            )


def _client(tmp_path: Path) -> TestClient:
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
                    "zones": {"1:1": {"speed_mms": 300, "distance_mm": 250}},
                },
            },
            {
                "session_id": "4_1002_1",
                "sources": {"csv_path": "4_1002_1.csv"},
                "analysis": {
                    "macro_order": [{"map_num": 1, "zone_id": "1"}],
                    "zones": {"1:1": {"speed_mms": 360, "distance_mm": 280}},
                },
            },
        ],
    }
    profile_path = tmp_path / "generated" / "profiles" / "spray" / "learning" / "mode_04.json"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    app = FastAPI()
    app.include_router(create_motion_viewer_router(tmp_path, frame_count=5))
    return TestClient(app)


def test_motion_viewer_page_explains_representative_and_statistical_motion(tmp_path: Path) -> None:
    response = _client(tmp_path).get("/motion-viewer")

    assert response.status_code == 200
    assert "전문가 모션 뷰어" in response.text
    assert "대표 실측 모션" in response.text
    assert "통계 범위" in response.text
    assert "기준 전문가: 전형기 반장" in response.text
    assert "페인트 누적 궤적" in response.text
    assert "주요 도장 방향" in response.text
    assert "참고용 스켈레톤" in response.text
    assert "전형기 ·" in response.text
    assert "function referencePointCloud" in response.text
    assert "function directionField" in response.text
    assert "전문가 전체 Ray Point" in response.text
    assert 'id="pointSize"' in response.text
    assert "흰색 포인트 크기" in response.text
    assert "pointSizeValue" in response.text
    assert 'id="zoneSurface"' in response.text
    assert "Zone 영역 표시" in response.text
    assert "function drawZoneSurfaces" in response.text
    assert "137.508" in response.text
    assert "`Zone ${surface.zone_id}`" in response.text
    assert 'id="loading"' in response.text
    assert "불러오는 중" in response.text
    assert "function setLoading" in response.text
    assert "data.paint_trace" in response.text
    assert 'id="currentZone"' in response.text
    assert 'id="zoneOrder"' in response.text
    assert 'id="zoneDetail"' in response.text
    assert "function updateZonePanel" in response.text
    assert "Zone 작업 순서" in response.text
    assert "현재 작업 Zone" in response.text
    assert "trackedStats[marker]" not in response.text
    assert "function interpolatedFrame" in response.text
    assert "function alignSkeleton" in response.text
    assert "function solveArm" in response.text
    assert "실측 위치 정합" in response.text
    assert "playheadSeconds" in response.text
    assert "const fps=(data.frames.length-1)" not in response.text
    assert "<canvas" in response.text
    assert ":'자료 없음'}\nfunction tick" in response.text


def test_catalog_lists_modes_and_individual_expert_sessions(tmp_path: Path) -> None:
    data = _client(tmp_path).get("/motion-viewer/api/catalog").json()

    mode = data["processes"][0]["modes"][0]
    assert mode == {
        "mode_id": 4,
        "session_count": 2,
        "representative_session_id": "4_1001_1",
        "sessions": ["4_1001_1", "4_1002_1"],
    }


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
    assert data["statistics"][0]["spread"]["Pelvis"] > 0


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
    assert data["zone_surfaces"][0]["map_num"] == "1"
    assert data["zone_surfaces"][0]["zone_id"] == "1"
    assert data["zone_surfaces"][0]["vertices"]
    assert data["zone_order"] == [{"map_num": "1", "zone_id": "1", "order": 1}]
    zone = data["zone_statistics"]["1:1"]
    assert zone["expert_session_count"] == 2
    assert zone["median_speed_mms"] == 300
    assert zone["median_distance_mm"] == 250
    assert zone["selected_speed_mms"] == 300
    assert zone["selected_distance_mm"] == 250
    assert zone["observed_path_length_m"] > 0
    assert zone["observed_duration_seconds"] > 0


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
