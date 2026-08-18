#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

eval_pose_reference = importlib.import_module("eval_pose_reference")
pose_similarity = importlib.import_module("pose_similarity")

BUCKET = "horizontal-mid-forward"


def _pose(lift: float = 0.0) -> dict[str, tuple[float, float, float]]:
    return {
        "Pelvis": (0.0, 0.0, 0.9),
        "Spine": (0.0, 0.0, 1.05),
        "Chest": (0.0, 0.0, 1.2),
        "LeftShoulder": (-0.18, 0.0, 1.32),
        "RightShoulder": (0.18, 0.0, 1.32),
        "RightUpperArm": (0.22, 0.02, 1.28),
        "RightForeArm": (0.28, 0.18, 1.10 + lift),
        "RightHand": (0.30, 0.34, 1.02),
    }


def _reference() -> dict[str, object]:
    normalized = pose_similarity.normalize_pose(_pose())
    return {
        "buckets": {
            BUCKET: {
                "n": 400,
                "joints": {
                    name: {"n": 400, "mean": list(position), "spread": 0.05}
                    for name, position in normalized.items()
                    if name in pose_similarity.SCORING_JOINTS
                },
            }
        }
    }


def test_score_samples_returns_one_similarity_per_scored_pose() -> None:
    samples = {BUCKET: [pose_similarity.normalize_pose(_pose()) for _ in range(3)]}

    result = eval_pose_reference.score_samples(samples, _reference())

    assert result.scored == 3
    assert result.matched == 3
    assert all(value == pytest.approx(1.0, abs=1e-6) for value in result.similarities)


def test_score_samples_counts_a_bucket_the_reference_lacks_as_unmatched() -> None:
    samples = {"vertical-high-left": [pose_similarity.normalize_pose(_pose())]}

    result = eval_pose_reference.score_samples(samples, _reference())

    assert result.scored == 1
    assert result.matched == 0
    assert result.similarities == []


def test_score_samples_drops_similarity_for_a_deviating_pose() -> None:
    samples = {BUCKET: [pose_similarity.normalize_pose(_pose(lift=-0.15))]}

    result = eval_pose_reference.score_samples(samples, _reference())

    assert result.matched == 1
    assert 0.0 <= result.similarities[0] < 1.0


def test_summarize_reports_match_rate_and_similarity_quartiles() -> None:
    summary = eval_pose_reference.summarize(
        [
            eval_pose_reference.FoldResult("10", "20250901", 4, 3, [0.2, 0.4, 0.6]),
            eval_pose_reference.FoldResult("10", "20250904", 6, 5, [0.5, 0.7, 0.9, 1.0, 0.8]),
        ]
    )

    assert summary["scored"] == 10
    assert summary["matched"] == 8
    assert summary["match_rate"] == pytest.approx(0.8)
    assert summary["p50"] == pytest.approx(0.65)


def _folds_with(similarities: list[float]) -> list[object]:
    return [eval_pose_reference.FoldResult("10", "d", len(similarities), len(similarities), similarities)]


def test_calibration_block_reports_the_percentile_ladder_the_mcp_reads() -> None:
    """MCP는 이 사다리로 질의 점수의 숙련자 대비 위치를 매긴다."""
    values = [index / 100.0 for index in range(100)]

    block = eval_pose_reference.calibration_block(_folds_with(values))

    assert block is not None
    assert block["n"] == 100
    assert block["p10"] == pytest.approx(0.10)
    assert block["p50"] == pytest.approx(0.50)
    assert block["p90"] == pytest.approx(0.90)
    assert list(block) == ["n", "p10", "p25", "p50", "p75", "p90"]


def test_write_calibration_stamps_which_reference_it_was_measured_against(
    tmp_path: Path,
) -> None:
    """레퍼런스를 다시 만들면 사다리는 무효다. 출처를 남겨야 낡은 백분위를 감지할 수 있다."""
    artifact = tmp_path / "expert_pose_reference.json"
    artifact.write_text(
        json.dumps({"10": {"status": "ok", "pose_count": 849, "buckets": {}}}),
        encoding="utf-8",
    )

    written = eval_pose_reference._write_calibration(
        artifact, {"10": {"n": 598.0, "p50": 0.64}}
    )

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert written == 1
    assert payload["10"]["expert_similarity"]["reference_pose_count"] == 849


def test_write_calibration_clears_a_level_it_could_not_measure(tmp_path: Path) -> None:
    artifact = tmp_path / "expert_pose_reference.json"
    artifact.write_text(
        json.dumps({"10": {"status": "ok", "pose_count": 5, "expert_similarity": {"p50": 0.5}}}),
        encoding="utf-8",
    )

    eval_pose_reference._write_calibration(artifact, {})

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert "expert_similarity" not in payload["10"]


def test_calibration_block_is_withheld_when_there_is_too_little_data() -> None:
    """근거가 얇으면 백분위를 만들지 않는다. 없는 게 틀린 것보다 낫다."""
    assert eval_pose_reference.calibration_block(_folds_with([0.5, 0.6])) is None


def test_summarize_handles_a_fold_set_with_nothing_scored() -> None:
    summary = eval_pose_reference.summarize([])

    assert summary["scored"] == 0
    assert summary["match_rate"] == 0.0
    assert summary["p50"] is None


def test_collect_csv_samples_groups_passes_by_bucket(tmp_path: Path) -> None:
    """납품 형식 CSV로도 폴드를 만들 수 있어야 백분위 보정이 활성화된다."""
    build_pose_reference = importlib.import_module("build_pose_reference")
    make_dummy_session = importlib.import_module("make_dummy_session")
    poses = []
    for index in range(240):
        sweep = 0.01 * (index % 40)
        lean = 0.2 * sweep
        poses.append({
            "Pelvis": (0.0, 0.0, 0.9),
            "Spine": (lean, 0.0, 1.05),
            "Chest": (lean, 0.0, 1.2),
            "LeftShoulder": (-0.18 + lean, 0.0, 1.32),
            "RightShoulder": (0.18 + lean, 0.0, 1.32),
            "LeftUpperArm": (-0.22 + lean, 0.02, 1.28),
            "RightUpperArm": (0.22 + lean, 0.02, 1.28),
            "LeftForeArm": (-0.28 + lean, 0.18, 1.10),
            "RightForeArm": (0.28 + sweep, 0.18, 1.10),
            "LeftHand": (-0.30 + lean, 0.34, 1.02),
            "RightHand": (0.30 + sweep, 0.34, 1.02),
            "Head": (lean, 0.0, 1.45),
        })
    path = make_dummy_session.write_session(
        tmp_path / "10_9000001_1.csv", poses, 0.011, sequence_frames=40
    )

    samples = eval_pose_reference.collect_csv_samples(path)

    assert samples == build_pose_reference.collect_csv_session(path)
    assert samples
