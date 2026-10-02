"""기준 자세 복원 검증(2026-10-02) — 실데이터로 몸통 방향 보존, FK 해석, 배치의 강체성, "추정 자세" 판정을 본다.

외부 검증이 찾은 사례: 전형기 L8 1000772_1 구간 3. 레퍼런스 몸통이 착탄면을 보지 않았다.
- 몸통 비례 보정이 마디마다 다른 배율로 몸통을 13° 더 눕혔다 → 같은 배율로 고쳤다.
- 보정 전부터 몸통 정면이 HMD 정면과 106° 어긋난다. FK 회전 해석은 실측 강체와 맞으므로(아래) 원본 아바타 IK의 몸통이다
  → 해부학적으로 맞지 않는 몸으로 판정해 흐리게 그리고 "추정 자세"로 알린다.

실데이터가 없으면 **skip하지 않고 실패한다** — skip하면 실패할 수 없는 단언이 된다.
"""

from __future__ import annotations

import importlib
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from trainer.pose_lab import body_proportions as bp  # noqa: E402
from trainer.pose_lab import service  # noqa: E402

pose_fk_csv = importlib.import_module("pose_fk_csv")
measured_pose = importlib.import_module("measured_pose")
rotation_math = importlib.import_module("rotation_math")
simulator_csv = importlib.import_module("simulator_csv")

DATA_ROOT = Path("/data/sources/professionals")
CASE = service.PassRef(worker="전형기", date="20250908", level="8", session="1000772_1", pass_index=3)
UP = np.array([0.0, 1.0, 0.0])
_CACHE: dict[str, Any] = {}


def _raw() -> Any:
    if not DATA_ROOT.is_dir():
        pytest.fail(f"실데이터가 없다({DATA_ROOT}) — 이 검증은 skip하지 않는다")
    if "raw" not in _CACHE:
        from trainer.pose_lab import csv_source

        _CACHE["raw"] = csv_source.CsvPassSource(DATA_ROOT).load(CASE)
    return _CACHE["raw"]


def _torso_direction(joints: dict[str, Any]) -> np.ndarray:
    vector = np.array(joints["Chest"], dtype=float) - np.array(joints["Pelvis"], dtype=float)
    return vector / np.linalg.norm(vector)


def test_proportion_rescale_keeps_the_torso_direction_of_the_reported_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """보고된 구간에서 보정 후 몸통 방향이 보정 없는 FK 몸통 방향과 같다(옛 보정은 13° 더 눕혔다).

    깨뜨리는 법: `rig_offsets`를 마디마다 다른 배율로 되돌리면 실패한다.
    """
    raw = _raw()
    corrected = service.skeleton_block(raw)
    monkeypatch.setattr(bp, "rig_offsets", lambda offsets: dict(offsets))
    plain = service.skeleton_block(raw)
    cosine = float(_torso_direction(corrected.joints) @ _torso_direction(plain.joints))
    assert math.degrees(math.acos(min(1.0, cosine))) < 0.5


def test_the_reported_pass_is_marked_as_an_estimated_posture() -> None:
    """몸통 정면이 HMD 정면에서 80° 넘게 돌아간 몸은 "추정 자세"로 흐리게 그리고 3D 밖에 알린다."""
    result = service.skeleton_block(_raw(), who=service.LEG_NOTE_WHO["reference"])
    posture = result.fit["posture"]
    assert posture is not None and posture["head_torso_deg"] > service.POSTURE_NECK_LIMIT_DEG
    assert posture["plausible"] is False
    assert "추정 자세" in result.fit["posture_note"] and "기준" in result.fit["posture_note"]


def test_posture_check_accepts_a_torso_facing_where_the_head_looks() -> None:
    """정면을 보는 몸(머리 정면 = 몸통 정면)은 맞다고 판정한다 — 판정기가 늘 '추정'을 내지 않는지."""
    joints = {"RightUpperArm": (0.2, 1.4, 0.0), "LeftUpperArm": (-0.2, 1.4, 0.0)}
    row = {"카메라 Rotation X": "0", "카메라 Rotation Y": "0", "카메라 Rotation Z": "0"}
    posture = service.posture_check(joints, row)
    assert posture is not None and posture["head_torso_deg"] == pytest.approx(0.0, abs=1e-6)
    assert posture["plausible"] is True
    turned = {"카메라 Rotation X": "0", "카메라 Rotation Y": "120", "카메라 Rotation Z": "0"}
    assert service.posture_check(joints, turned)["plausible"] is False


def _local_chain(row: dict[str, str], offsets: dict[str, Any]) -> dict[str, np.ndarray]:
    _positions, world = pose_fk_csv.forward_kinematics_with_rotations(row, offsets)
    return {name: np.array(matrix) for name, matrix in world.items()}


def _as_world(row: dict[str, str], _offsets: dict[str, Any]) -> dict[str, np.ndarray]:
    out = {}
    for column in simulator_csv.JOINT_COLUMNS:
        try:
            angles = simulator_csv.vec3(row, column, "Rotation")
        except (KeyError, ValueError):
            continue
        out[simulator_csv.JOINT_KEYS[column]] = np.array(pose_fk_csv.euler_matrix(angles))
    return out


def _relative_angle(a: np.ndarray, b: np.ndarray) -> float:
    r = a.T @ b
    return math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(r)) - 1.0) / 2.0))))


def test_fk_rotation_convention_matches_the_rigid_bodies() -> None:
    """FK(부모 기준 로컬 회전의 합성)가 실측 강체(카메라·건·플래쉬)와 같은 상대회전을 낸다.

    두 프레임 사이 상대회전각은 상수 장착·루트 회전과 무관하므로 해석 규약만 가른다. 6명·18세션 실측(2026-10-02)에서
    로컬 합성 0.4–0.7°, 월드 해석 1.7–6.0°였다. 이 구간 하나로 같은 순서를 고정한다.
    깨뜨리는 법: FK가 관절 회전을 월드로 읽으면(합성하지 않으면) 오른손 차이가 커져 실패한다.
    """
    raw = _raw()
    rows = [dict(row) for row in raw.rows]
    gun = measured_pose.active_gun(rows)
    offsets = pose_fk_csv.bind_offsets(rows[0])
    rng = random.Random(20261002)
    gaps: dict[str, dict[str, list[float]]] = {"local": {}, "world": {}}
    for _ in range(60):
        r1, r2 = rng.sample(rows, 2)
        for name, convention in (("local", _local_chain), ("world", _as_world)):
            w1, w2 = convention(r1, offsets), convention(r2, offsets)
            for joint, group in (("Head", measured_pose.HEAD_GROUP), ("RightHand", gun), ("LeftHand", measured_pose.LEFT_GROUP)):
                try:
                    b1 = np.array(rotation_math.euler_matrix(measured_pose.EULER_ORDER, simulator_csv.vec3(r1, group, "Rotation"), measured_pose.EULER_SIGN))
                    b2 = np.array(rotation_math.euler_matrix(measured_pose.EULER_ORDER, simulator_csv.vec3(r2, group, "Rotation"), measured_pose.EULER_SIGN))
                except (KeyError, ValueError):
                    continue
                gaps[name].setdefault(joint, []).append(abs(_relative_angle(w1[joint], w2[joint]) - _relative_angle(b1, b2)))
    for joint, values in gaps["local"].items():
        assert statistics.median(values) < 2.0, joint
    assert statistics.median(gaps["local"]["RightHand"]) < statistics.median(gaps["world"]["RightHand"])


def test_placement_moves_the_fk_body_rigidly() -> None:
    """배치된 몸통·머리·왼팔 관절은 FK 관절(같은 오프셋·같은 행)의 강체 변환이다 — 관절 사이 거리가 모두 같다.

    표시 단계가 몸을 늘이거나 비틀면(옛 비례 보정처럼) 여기서 잡힌다. 오른팔은 건 손잡이로 다시 세우고 다리는 자리표시라 뺀다.
    """
    raw = _raw()
    result = service.skeleton_block(raw)
    offsets = bp.rig_offsets(pose_fk_csv.bind_offsets(raw.rows[0]))
    fk_joints = pose_fk_csv.forward_kinematics(dict(result.representative.row), offsets)
    names = ["Pelvis", "Spine", "Chest", "Head", "LeftShoulder", "LeftUpperArm", "LeftForeArm", "LeftHand"]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            placed = math.dist(result.joints[a], result.joints[b])
            assert placed == pytest.approx(math.dist(fk_joints[a], fk_joints[b]), abs=1e-6), (a, b)
