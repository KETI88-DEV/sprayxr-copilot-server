"""아바타(리그)별 숙련자 기준 고르기 — 2026-10-08 아바타 변경.

기준이 여럿이면 시퀀스 골격의 뼈 길이로 맞는 리그의 기준을 쓰고, 맞는 것이 없으면 쓰지 않는다(다른 몸의 기준에
대고 재지 않는다). 기준이 하나뿐이면 예전처럼 그것을 쓴다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mcp_servers import pose_estimation
from trainer.scripts import joint_names, rig_profile, skeleton_gate

SAMPLE = Path("/data/sources/new_format/sample_2026-10-08")
AVATAR_LENGTHS = {
    "Spine": 0.198457, "Chest": 0.105315, "Head": 0.343559, "LeftShoulder": 0.212527, "RightShoulder": 0.212527,
    "LeftThigh": 0.096502, "RightThigh": 0.096502, "LeftCalf": 0.422307, "RightCalf": 0.422307,
    "LeftFoot": 0.410989, "RightFoot": 0.410989, "LeftToe": 0.149667, "RightToe": 0.149667,
}


def _frames() -> list[skeleton_gate.GateFrame]:
    files = list(SAMPLE.glob("*.json"))
    if not files:
        pytest.fail(f"새 형식 샘플이 없다({SAMPLE}) — skip하지 않는다")
    doc = json.loads(files[0].read_text(encoding="utf-8"))
    body = next(r["body"] for r in doc["requests"] if r["body"].get("sequence_id") == "SEQ-020")
    return [
        skeleton_gate.GateFrame(joints=joint_names.positions(f["skeleton_joints"]), tool_pos=None, time_s=None)
        for f in body["sequence_info"]
    ]


def _write(tmp_path: Path, name: str, marker: int, rig: dict | None) -> Path:
    payload: dict = {"version": 1, "buckets": {"-0.10": {"n": marker}}}
    if rig is not None:
        payload["rig"] = rig
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_the_reference_of_the_matching_rig_is_used(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """새 아바타 시퀀스에는 새 아바타 기준을, 그 리그의 키 상한과 함께 쓴다.

    깨뜨리는 법: `_arm_buckets`가 리그를 보지 않고 첫 기준을 쓰면 legacy 버킷(n=1)이 나와 실패한다.
    """
    legacy = _write(tmp_path, "legacy.json", 1, None)
    avatar = _write(tmp_path, "avatar.json", 2, {"name": "avatar_2026-10", "bone_lengths_m": AVATAR_LENGTHS})
    monkeypatch.setenv(pose_estimation.ARM_REFERENCE_ENV, f"{legacy},{avatar}")
    frames = _frames()
    assert pose_estimation.matched_rig(frames) == "avatar_2026-10"
    assert pose_estimation._arm_buckets(frames)["-0.10"]["n"] == 2
    limit = pose_estimation.height_max_for(frames)
    assert limit == pytest.approx(rig_profile.chain_height_m(AVATAR_LENGTHS) * skeleton_gate.HEIGHT_CHAIN_SLACK)


def test_no_matching_rig_means_no_reference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """기준이 여럿인데 어느 리그와도 맞지 않으면 기준을 쓰지 않는다 — 다른 몸의 기준에 대고 재지 않는다."""
    legacy = _write(tmp_path, "legacy.json", 1, None)
    other = _write(tmp_path, "other.json", 3, {"name": "other", "bone_lengths_m": {k: v * 1.6 for k, v in AVATAR_LENGTHS.items()}})
    monkeypatch.setenv(pose_estimation.ARM_REFERENCE_ENV, f"{legacy},{other}")
    frames = _frames()
    assert pose_estimation.matched_rig(frames) is None
    assert pose_estimation._arm_buckets(frames) is None
    assert pose_estimation.height_max_for(frames) is None


def test_a_single_reference_is_used_as_before(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    legacy = _write(tmp_path, "legacy.json", 1, None)
    monkeypatch.setenv(pose_estimation.ARM_REFERENCE_ENV, str(legacy))
    assert pose_estimation._arm_buckets(_frames())["-0.10"]["n"] == 1
