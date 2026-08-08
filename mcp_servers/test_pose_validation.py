import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from mcp_servers.main import app
from trainer.scripts.pose_similarity import normalize_pose

BUCKET = "horizontal-mid-+y"

REFERENCE_POSE: dict[str, tuple[float, float, float]] = {
    "Pelvis": (0.0, 0.0, 0.9),
    "Spine": (0.0, 0.0, 1.05),
    "Chest": (0.0, 0.0, 1.2),
    "Neck": (0.0, 0.0, 1.35),
    "Head": (0.0, 0.0, 1.45),
    "LeftShoulder": (-0.18, 0.0, 1.32),
    "RightShoulder": (0.18, 0.0, 1.32),
    "LeftUpperArm": (-0.22, 0.02, 1.28),
    "RightUpperArm": (0.22, 0.02, 1.28),
    "LeftForeArm": (-0.28, 0.18, 1.10),
    "RightForeArm": (0.28, 0.18, 1.10),
    "LeftHand": (-0.30, 0.34, 1.02),
    "RightHand": (0.30, 0.34, 1.02),
}


def _joints(pose: dict[str, tuple[float, float, float]]) -> dict[str, dict[str, float]]:
    return {name: {"x": x, "y": y, "z": z} for name, (x, y, z) in pose.items()}


def _write_artifact(tmp_path: Path, spread: float = 0.05) -> Path:
    """아티팩트의 mean은 빌더가 저장하는 대로 루트·요 정규화 공간의 좌표다."""
    normalized = normalize_pose(REFERENCE_POSE)
    payload: dict[str, Any] = {
        "10": {
            "level": "10",
            "status": "ok",
            "reference_quality": "single_worker",
            "worker_ids": ["전형기"],
            "buckets": {
                BUCKET: {
                    "n": 400,
                    "joints": {
                        name: {"n": 400, "mean": list(position), "spread": spread}
                        for name, position in normalized.items()
                    },
                }
            },
        }
    }
    path = tmp_path / "expert_pose_reference.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _compare(body: dict[str, Any]) -> dict[str, Any]:
    response = TestClient(app).post("/pose_estimation/compare", json=body)
    assert response.status_code == 200, response.text
    return response.json()["result"]


def test_pose_recommend_requires_block_context() -> None:
    response = TestClient(app).post(
        "/pose_estimation/compare",
        json={"mode": "recommend"},
    )

    assert response.status_code == 422


def test_pose_compare_requires_current_pose() -> None:
    response = TestClient(app).post(
        "/pose_estimation/compare",
        json={"mode": "compare", "block_id": "BLK-A"},
    )

    assert response.status_code == 422


def test_legacy_fixture_compare_also_carries_the_contract_status_fields() -> None:
    """계약 3.6은 두 경로 모두 analysis_status를 싣기를 요구한다."""
    result = _compare({
        "mode": "compare",
        "block_id": "BLK-A",
        "current_pose": {"joints": {"elbow_r": {"x": 0.28, "y": -0.1, "z": 0.35, "pitch": 12.0}}},
    })

    assert result["analysis_status"] == "available"
    assert result["reason_code"] is None
    assert result["expert_pose_id"] == "EP-001"


def test_bucket_compare_scores_the_reference_pose_as_a_perfect_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("POSE_REFERENCE_PATH", str(_write_artifact(tmp_path)))

    result = _compare({
        "mode": "compare",
        "level": "10",
        "bucket": BUCKET,
        "current_pose": {"joints": _joints(REFERENCE_POSE)},
    })

    assert result["analysis_status"] == "available"
    assert result["reason_code"] is None
    assert result["expert_pose_id"] == BUCKET
    assert result["similarity_score"] == pytest.approx(1.0, abs=1e-6)
    assert result["corrections"] == []


def test_bucket_compare_keeps_similarity_a_zero_to_one_float(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("POSE_REFERENCE_PATH", str(_write_artifact(tmp_path)))
    pose = dict(REFERENCE_POSE)
    pose["RightForeArm"] = (0.28, 0.18, 0.60)

    result = _compare({
        "mode": "compare",
        "level": "10",
        "bucket": BUCKET,
        "current_pose": {"joints": _joints(pose)},
    })

    similarity = result["similarity_score"]
    assert isinstance(similarity, float)
    assert 0.0 <= similarity < 1.0


def test_bucket_compare_names_the_body_part_in_korean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("POSE_REFERENCE_PATH", str(_write_artifact(tmp_path)))
    pose = dict(REFERENCE_POSE)
    pose["RightForeArm"] = (0.28, 0.18, 0.95)

    result = _compare({
        "mode": "compare",
        "level": "10",
        "bucket": BUCKET,
        "current_pose": {"joints": _joints(pose)},
    })

    top = result["corrections"][0]
    assert top["joint"] == "RightForeArm"
    assert top["label"] == "오른쪽 팔꿈치"
    assert "오른쪽 팔꿈치" in top["correction_text"]
    assert "오른쪽 팔꿈치" in result["summary"]


def test_bucket_without_scoring_joints_reports_unusable_instead_of_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """채점 가능한 관절이 없으면 유사도 0.0은 '많이 다르다'가 아니라 '못 잰다'는 뜻이다."""
    payload: dict[str, Any] = {
        "10": {
            "level": "10",
            "status": "ok",
            "buckets": {
                BUCKET: {"n": 400, "joints": {"LeftToe": {"mean": [0.0, 0.0, 0.0], "spread": 0.05}}}
            },
        }
    }
    path = tmp_path / "expert_pose_reference.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setenv("POSE_REFERENCE_PATH", str(path))

    result = _compare({
        "mode": "compare",
        "level": "10",
        "bucket": BUCKET,
        "current_pose": {"joints": _joints(REFERENCE_POSE)},
    })

    assert result["analysis_status"] == "not_available"
    assert result["reason_code"] == "quality_insufficient"
    assert "similarity_score" not in result


def test_bucket_compare_reports_how_much_data_backs_the_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """원점수만 내보내면 '63% 정답'으로 읽힌다. 근거량을 함께 실어야 한다."""
    monkeypatch.setenv("POSE_REFERENCE_PATH", str(_write_artifact(tmp_path)))

    result = _compare({
        "mode": "compare",
        "level": "10",
        "bucket": BUCKET,
        "current_pose": {"joints": _joints(REFERENCE_POSE)},
    })

    assert result["support"] == 400
    assert result["reference_quality"] == "single_worker"


def test_bucket_compare_reports_the_expert_percentile_when_calibrated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """숙련자 held-out 분포 대비 위치를 줘야 숫자가 과대 해석되지 않는다."""
    path = _write_artifact(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["10"]["expert_similarity"] = {
        "n": 2000,
        "p10": 0.30,
        "p25": 0.48,
        "p50": 0.66,
        "p75": 0.80,
        "p90": 0.90,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setenv("POSE_REFERENCE_PATH", str(path))

    result = _compare({
        "mode": "compare",
        "level": "10",
        "bucket": BUCKET,
        "current_pose": {"joints": _joints(REFERENCE_POSE)},
    })

    assert result["similarity_score"] == pytest.approx(1.0, abs=1e-6)
    assert result["expert_percentile"] == 90


def test_bucket_compare_omits_the_percentile_without_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("POSE_REFERENCE_PATH", str(_write_artifact(tmp_path)))

    result = _compare({
        "mode": "compare",
        "level": "10",
        "bucket": BUCKET,
        "current_pose": {"joints": _joints(REFERENCE_POSE)},
    })

    assert result["expert_percentile"] is None


def test_unknown_bucket_reports_missing_reference_instead_of_guessing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("POSE_REFERENCE_PATH", str(_write_artifact(tmp_path)))

    result = _compare({
        "mode": "compare",
        "level": "10",
        "bucket": "vertical-high--x",
        "current_pose": {"joints": _joints(REFERENCE_POSE)},
    })

    assert result["analysis_status"] == "not_available"
    assert result["reason_code"] == "reference_data_missing"
    assert "similarity_score" not in result


def test_absent_artifact_reports_missing_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("POSE_REFERENCE_PATH", str(tmp_path / "nope.json"))

    result = _compare({
        "mode": "compare",
        "level": "10",
        "bucket": BUCKET,
        "current_pose": {"joints": _joints(REFERENCE_POSE)},
    })

    assert result["analysis_status"] == "not_available"
    assert result["reason_code"] == "reference_data_missing"


def test_pose_without_the_root_joint_reports_missing_source_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("POSE_REFERENCE_PATH", str(_write_artifact(tmp_path)))
    pose = {name: value for name, value in REFERENCE_POSE.items() if name != "Pelvis"}

    result = _compare({
        "mode": "compare",
        "level": "10",
        "bucket": BUCKET,
        "current_pose": {"joints": _joints(pose)},
    })

    assert result["analysis_status"] == "not_available"
    assert result["reason_code"] == "source_data_missing"
