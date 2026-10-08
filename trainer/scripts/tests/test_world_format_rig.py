"""새 형식 CSV(관절 월드 위치·회전)와 아바타 리그 — 2026-10-08 샘플부터.

실데이터(`/data/sources/new_format/sample_2026-10-08`)가 없으면 **skip하지 않고 실패한다** — skip하면 실패할 수
없는 단언이 된다.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pose_fk_csv  # noqa: E402
import rig_profile  # noqa: E402
import simulator_csv  # noqa: E402
import skeleton_gate  # noqa: E402

SAMPLE = Path("/data/sources/new_format/sample_2026-10-08")
_ROWS: list[dict[str, str]] = []


def _rows() -> list[dict[str, str]]:
    path = SAMPLE / "3_11000346_1.csv.gz"
    if not path.is_file():
        pytest.fail(f"새 형식 샘플이 없다({path}) — 이 검증은 skip하지 않는다")
    if not _ROWS:
        _ROWS.extend(rig_profile.read_csv_rows(path))
    return _ROWS


def test_the_new_format_is_detected_and_the_old_one_is_not() -> None:
    assert pose_fk_csv.is_world_format(_rows()[0])
    assert not pose_fk_csv.is_world_format({"Pelvis Rotation X": "0"})


def test_fk_on_derived_offsets_reproduces_the_recorded_world_joints() -> None:
    """역산 오프셋 + 역산 로컬 회전으로 FK를 돌리면 기록된 월드 관절이 그대로 나온다(루트 원점 기준).

    FK 규약(YXZ, 부모 기준 합성)을 정답 좌표로 확인하는 시험이다. 깨뜨리는 법: 로컬 회전을 W_child로 두면
    (부모 회전을 빼지 않으면) 실패한다.
    """
    rows = _rows()
    for row in rows[:: len(rows) // 25]:
        offsets = pose_fk_csv.bind_offsets(row)
        assert len(offsets) == 19
        fk = pose_fk_csv.forward_kinematics(row, offsets)
        world = pose_fk_csv.world_positions(row)
        pelvis = world["Pelvis"]
        for name, position in fk.items():
            expected = tuple(world[name][i] - pelvis[i] for i in range(3))
            assert math.dist(position, expected) < 1e-4, name


def test_derived_offsets_are_constant_across_the_session() -> None:
    rows = _rows()
    spans = []
    for parent, child in simulator_csv.BONES:
        values = [pose_fk_csv.derived_offsets(row)[child] for row in rows[::400]]
        spans.append(max(max(v[i] for v in values) - min(v[i] for v in values) for i in range(3)))
    assert max(spans) < 1e-3


def test_rig_match_tells_the_two_avatars_apart() -> None:
    avatar = rig_profile.derive_from_rows(_rows(), name="avatar_2026-10").bone_lengths()
    candidates = {rig_profile.LEGACY_RIG: rig_profile.LEGACY_BONE_LENGTHS, "avatar_2026-10": avatar}
    assert rig_profile.match(avatar, candidates) == "avatar_2026-10"
    assert rig_profile.match(dict(rig_profile.LEGACY_BONE_LENGTHS), candidates) == rig_profile.LEGACY_RIG
    stranger = {name: value * 1.5 for name, value in avatar.items()}
    assert rig_profile.match(stranger, candidates) is None


def _sample_frames(sequence_id: str) -> list[skeleton_gate.GateFrame]:
    import joint_names

    doc = json.loads(next(SAMPLE.glob("*.json")).read_text(encoding="utf-8"))
    body = next(r["body"] for r in doc["requests"] if r["body"].get("sequence_id") == sequence_id)
    return [
        skeleton_gate.GateFrame(
            joints=joint_names.positions(frame["skeleton_joints"]),
            tool_pos=joint_names.parse_position(frame["tool_pos_rot"]["pos"]),
            time_s=None,
        )
        for frame in body["sequence_info"]
    ]


def _avatar_limit() -> float:
    lengths = rig_profile.derive_from_rows(_rows(), name="avatar_2026-10").bone_lengths()
    chain = rig_profile.chain_height_m(lengths)
    assert chain is not None
    return chain * skeleton_gate.HEIGHT_CHAIN_SLACK


def test_a_standing_new_avatar_passes_with_its_rig_limit() -> None:
    """새 아바타의 선 자세(머리↔발끝 1.61 m)는 그 리그의 상한(뼈 길이 합 1.73 m × 여유)으로는 통과하고, 예전 리그의
    고정 상한 1.50 m로는 거절된다(2026-10-08 샘플 20/32이 그렇게 거절됐다).
    """
    frames = _sample_frames("SEQ-020")
    assert "skeleton_height_out_of_range" in skeleton_gate.evaluate(frames).reason_codes
    result = skeleton_gate.evaluate(frames, height_max_m=_avatar_limit())
    assert "skeleton_height_out_of_range" not in result.reason_codes, result.reason_codes
    assert result.measurements["height_m"] > skeleton_gate.HEIGHT_MAX_M


def test_a_skeleton_larger_than_the_rig_is_still_rejected() -> None:
    """골반을 중심으로 1.2배 키운 골격(리그보다 큼)은 그 리그의 상한으로도 거절된다 — 상한을 없앤 것이 아니다.

    깨뜨리는 법: 상한을 같은 프레임에서 잰 뼈 길이 합으로 만들면 삼각부등식 때문에 늘 통과해 실패한다.
    """
    frames = _sample_frames("SEQ-020")
    scaled = []
    for f in frames:
        pelvis = f.joints["Pelvis"]
        joints = {k: tuple(pelvis[i] + (v[i] - pelvis[i]) * 1.2 for i in range(3)) for k, v in f.joints.items()}
        scaled.append(skeleton_gate.GateFrame(joints=joints, tool_pos=f.tool_pos, time_s=None))
    result = skeleton_gate.evaluate(scaled, height_max_m=_avatar_limit())
    assert "skeleton_height_out_of_range" in result.reason_codes
