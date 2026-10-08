import csv
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.motion_viewer import create_motion_viewer_router


def _write_csv(path: Path, *, offset: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "인덱스",
        "시간(초)",
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
    assert "<canvas" in response.text


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


def test_motion_rejects_unknown_session_without_reading_arbitrary_paths(tmp_path: Path) -> None:
    response = _client(tmp_path).get(
        "/motion-viewer/api/motion",
        params={"process": "spray", "mode_id": 4, "session_id": "../../secret"},
    )

    assert response.status_code == 404

