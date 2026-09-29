#!/usr/bin/env python3
"""입력 품질 게이트 검사.

합성 픽스처로 임계 하나하나를 재고, **실물 샘플 두 시퀀스로 전체를 다시 잰다.**
합성만으로는 임계가 실제 데이터와 맞는지 알 수 없다 — 이 게이트의 가장 큰 위험이
"사람 해부 기준을 써서 정상 데이터를 100% 거부하는 것"이기 때문이다.
"""

from __future__ import annotations

import datetime as dt
import importlib
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
for path in (str(_REPO_ROOT), str(_SCRIPTS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

joint_names = importlib.import_module("joint_names")
skeleton_gate = importlib.import_module("skeleton_gate")

SAMPLE = _REPO_ROOT / "docs" / "SprayXR_CoPilot_샘플_2026-09-21.json"


def _rig(shoulder: float = 0.0567, height: float = 1.10) -> dict:
    """리그 기준 골격 하나. 어깨 사이와 키를 직접 준다."""
    half = shoulder / 2.0
    return {
        "Pelvis": (0.0, 0.0, 0.0),
        "Spine": (0.0, 0.15, 0.0),
        "Chest": (0.0, 0.30, 0.0),
        "Head": (0.0, height - 0.40, 0.0),
        "LeftShoulder": (-half, 0.30, 0.0),
        "RightShoulder": (half, 0.30, 0.0),
        "LeftUpperArm": (-half - 0.02, 0.28, 0.0),
        "RightUpperArm": (half + 0.02, 0.28, 0.0),
        "LeftForeArm": (-half - 0.02, 0.15, 0.0),
        "RightForeArm": (half + 0.02, 0.15, 0.0),
        "LeftHand": (-half - 0.02, 0.02, 0.0),
        "RightHand": (half + 0.02, 0.02, 0.0),
        "LeftThigh": (-0.05, -0.05, 0.0),
        "RightThigh": (0.05, -0.05, 0.0),
        "LeftCalf": (-0.05, -0.20, 0.0),
        "RightCalf": (0.05, -0.20, 0.0),
        "LeftFoot": (-0.05, -0.38, 0.0),
        "RightFoot": (0.05, -0.38, 0.0),
        "LeftToe": (-0.05, -0.40, 0.02),
        "RightToe": (0.05, -0.40, 0.02),
    }


def _moved(joints: dict, dx: float) -> dict:
    return {name: (x + dx, y, z) for name, (x, y, z) in joints.items()}


def _frames(count: int = 5, *, step: float = 0.01, tool_offset: float = 0.20, dt_s: float = 0.011):
    out = []
    for i in range(count):
        joints = _moved(_rig(), step * i)
        hand = joints["RightHand"]
        out.append(
            skeleton_gate.GateFrame(
                joints=joints,
                tool_pos=(hand[0] + tool_offset, hand[1], hand[2]),
                time_s=i * dt_s,
            )
        )
    return out


def test_a_normal_rig_sequence_passes() -> None:
    """정상 입력이 통과해야 한다. 이게 깨지면 나머지 검사는 의미가 없다."""
    result = skeleton_gate.evaluate(_frames())
    assert result.status == "pass", (result.status, result.reason_codes)
    assert result.reason_codes == ()
    assert result.scorable is True


def test_human_shoulder_width_would_reject_the_whole_corpus() -> None:
    """임계가 사람이 아니라 **리그**에 맞춰져 있다.

    실측 어깨 사이는 0.0567 m다(쇄골 기점). 사람 해부 기준 0.25~0.45 m를 쓰면
    코퍼스 100%가 거부된다.

    깨뜨리는 법: `SHOULDER_WIDTH_MIN_M`을 0.25로 올리면 정상 입력이 거부돼 실패한다.
    """
    assert skeleton_gate.SHOULDER_WIDTH_MIN_M <= 0.0567 <= skeleton_gate.SHOULDER_WIDTH_MAX_M
    assert skeleton_gate.evaluate(_frames()).status != "reject"


def test_shoulder_width_outside_the_rig_band_is_rejected() -> None:
    frames = [
        skeleton_gate.GateFrame(joints=_rig(shoulder=0.30), tool_pos=(0.17, 0.02, 0.0), time_s=0.0)
    ]
    result = skeleton_gate.evaluate(frames)
    assert result.status == "reject"
    assert "skeleton_shoulder_width_out_of_range" in result.reason_codes


def test_tool_far_from_the_hand_is_rejected() -> None:
    """건이 손에 없으면 그 시퀀스의 자세는 도장 자세가 아니다. 샘플 SEQ-001이 4.961 m다."""
    result = skeleton_gate.evaluate(_frames(tool_offset=4.961))
    assert result.status == "reject"
    assert "tool_not_held" in result.reason_codes
    assert result.measurements["hand_tool_gap_m"] > skeleton_gate.HAND_TOOL_MAX_M


def test_all_frozen_with_a_moving_tool_is_rejected() -> None:
    """골격이 얼었는데 도구가 움직이면 골격이 안 오는 것이다."""
    joints = _rig()
    frames = [
        skeleton_gate.GateFrame(
            joints=joints, tool_pos=(0.05 + 0.05 * i, 0.02, 0.0), time_s=0.011 * i
        )
        for i in range(5)
    ]
    result = skeleton_gate.evaluate(frames)
    assert result.status == "reject"
    assert "skeleton_all_joints_frozen" in result.reason_codes


def test_all_frozen_with_a_still_tool_is_not_rejected() -> None:
    """도구까지 정지했으면 그냥 멈춰 있는 시퀀스다 — 정상 케이스다(샘플 SEQ-002).

    깨뜨리는 법: 도구 변위 조건을 빼고 무조건 거부하면 이 검사가 실패한다.
    """
    joints = _rig()
    frames = [
        skeleton_gate.GateFrame(joints=joints, tool_pos=(0.25, 0.02, 0.0), time_s=0.011 * i)
        for i in range(5)
    ]
    result = skeleton_gate.evaluate(frames)
    assert result.status != "reject"
    assert "skeleton_all_joints_frozen" in result.info_codes


def test_one_frozen_joint_degrades_and_is_excluded() -> None:
    frames = _frames()
    pinned = []
    for frame in frames:
        joints = dict(frame.joints)
        joints["LeftHand"] = (-0.048, 0.02, 0.0)
        pinned.append(
            skeleton_gate.GateFrame(joints=joints, tool_pos=frame.tool_pos, time_s=frame.time_s)
        )
    result = skeleton_gate.evaluate(pinned)
    assert "skeleton_joint_frozen" in result.reason_codes
    assert "LeftHand" in result.excluded_joints


def test_a_joint_frozen_in_an_earlier_sequence_is_excluded_again() -> None:
    """R4c — 한 번은 우연이지만 두 번은 그 관절이 안 오는 것이다."""
    frames = _frames()
    pinned = []
    for frame in frames:
        joints = dict(frame.joints)
        joints["Head"] = (0.0, 0.70, 0.0)
        pinned.append(
            skeleton_gate.GateFrame(joints=joints, tool_pos=frame.tool_pos, time_s=frame.time_s)
        )
    result = skeleton_gate.evaluate(pinned, frozen_before=frozenset({"Head"}))
    assert "Head" in result.excluded_joints


def test_non_finite_coordinates_are_rejected() -> None:
    joints = dict(_rig())
    joints["Chest"] = (float("nan"), 0.3, 0.0)
    result = skeleton_gate.evaluate([skeleton_gate.GateFrame(joints=joints)])
    assert result.status == "reject"
    assert result.reason_codes == ("skeleton_coordinates_invalid",)


def test_missing_required_joint_is_rejected() -> None:
    joints = dict(_rig())
    del joints["Pelvis"]
    result = skeleton_gate.evaluate([skeleton_gate.GateFrame(joints=joints)])
    assert result.status == "reject"
    assert result.reason_codes == ("skeleton_required_joint_missing",)


def test_stretching_a_bone_within_a_sequence_degrades() -> None:
    """리그는 강체다. 시퀀스 안에서 뼈가 늘면 그 골격을 믿을 수 없다."""
    frames = list(_frames())
    last = frames[-1]
    joints = dict(last.joints)
    hand = joints["RightHand"]
    joints["RightHand"] = (hand[0], hand[1] - 0.50, hand[2])
    frames[-1] = skeleton_gate.GateFrame(
        joints=joints, tool_pos=last.tool_pos, time_s=last.time_s
    )
    result = skeleton_gate.evaluate(frames)
    assert "skeleton_bone_length_inconsistent" in result.reason_codes


def test_irregular_frame_interval_degrades_but_does_not_reject() -> None:
    frames = list(_frames(count=4))
    frames[2] = skeleton_gate.GateFrame(
        joints=frames[2].joints, tool_pos=frames[2].tool_pos, time_s=frames[1].time_s + 0.200
    )
    result = skeleton_gate.evaluate(frames)
    assert result.status == "degrade"
    assert "frame_interval_irregular" in result.reason_codes


def test_right_arm_degeneracy_is_info_not_rejection() -> None:
    """정상 데이터가 전부 이렇다(CSV-FK 신전 비율 p10 = p50 = 1.000). 거부하면 전부 죽는다."""
    base = dict(_rig())
    base["RightUpperArm"] = (0.05, 0.30, 0.0)
    base["RightForeArm"] = (0.05, 0.16, 0.0)
    base["RightHand"] = (0.05, 0.02, 0.0)
    frames = [
        skeleton_gate.GateFrame(
            joints=_moved(base, 0.01 * i), tool_pos=(0.25 + 0.01 * i, 0.02, 0.0), time_s=0.011 * i
        )
        for i in range(5)
    ]
    result = skeleton_gate.evaluate(frames)
    assert result.status != "reject"
    assert "right_arm_degenerate" in result.info_codes
    assert result.measurements["right_arm_extension"] >= skeleton_gate.ARM_EXTENSION_RATIO


def test_reason_codes_stay_inside_the_closed_set() -> None:
    """호출부가 문자열을 지어내면 소비자가 모르는 값을 받는다."""
    for frames in (_frames(), _frames(tool_offset=4.961)):
        result = skeleton_gate.evaluate(frames)
        assert set(result.reason_codes) <= skeleton_gate.REASON_CODES
        assert set(result.info_codes) <= skeleton_gate.REASON_CODES


def test_the_bone_chain_has_nineteen_bones() -> None:
    """실측 근거가 '19개 뼈 길이가 유일값 1개'다. 사슬 길이가 그 수와 맞아야 근거가 성립한다."""
    assert len(skeleton_gate.BONES) == 19
    children = {child for child, _ in skeleton_gate.BONES}
    assert children == set(joint_names.CODE_JOINTS) - {"Pelvis"}


def _sample_sequences() -> list[tuple[str, list]]:
    payload = json.loads(SAMPLE.read_text(encoding="utf-8"))
    out: list[tuple[str, list]] = []
    for request in payload["requests"]:
        body = request.get("body") or {}
        info = body.get("sequence_info")
        if not info or request["endpoint"] != "POST /telemetry/sequence":
            continue
        frames = []
        base = None
        for frame in info:
            stamp = dt.datetime.fromisoformat(frame["time"].replace("Z", "+00:00")).timestamp()
            base = stamp if base is None else base
            frames.append(
                skeleton_gate.GateFrame(
                    joints=joint_names.positions(frame.get("skeleton_joints") or {}),
                    tool_pos=joint_names.parse_position(
                        (frame.get("tool_pos_rot") or {}).get("pos")
                    ),
                    time_s=stamp - base,
                )
            )
        out.append((body["sequence_id"], frames))
    return out


def test_real_sample_sequence_one_is_rejected_for_an_unheld_tool() -> None:
    """SEQ-001은 건이 손에서 4.961 m 떨어져 있다. 설계 문서가 이 판정을 적어 두었다."""
    sequences = dict(_sample_sequences())
    assert "SEQ-001" in sequences, f"샘플에서 SEQ-001을 못 읽었다: {sorted(sequences)}"
    result = skeleton_gate.evaluate(sequences["SEQ-001"])
    assert result.status == "reject"
    assert "tool_not_held" in result.reason_codes
    assert result.measurements["hand_tool_gap_m"] > 4.0


def test_real_sample_head_is_frozen_in_every_sequence() -> None:
    """실물에서 `head`가 두 시퀀스 모두 정확히 0.0이다 — R4c가 겨냥한 결함이다."""
    sequences = dict(_sample_sequences())
    assert sequences, "샘플 시퀀스를 하나도 못 읽었다"
    for name, frames in sequences.items():
        points = [frame.joints["Head"] for frame in frames if "Head" in frame.joints]
        assert skeleton_gate._spread(points) < skeleton_gate.FROZEN_RANGE_M, name  # noqa: SLF001


def test_real_sample_passes_the_rig_thresholds() -> None:
    """어깨 너비·키·뼈 길이 임계가 실물을 거부하지 않는다 — 거부하면 임계가 틀린 것이다."""
    for name, frames in _sample_sequences():
        result = skeleton_gate.evaluate(frames)
        for code in (
            "skeleton_shoulder_width_out_of_range",
            "skeleton_height_out_of_range",
            "skeleton_bone_length_inconsistent",
            "skeleton_coordinates_invalid",
            "skeleton_required_joint_missing",
        ):
            assert code not in result.reason_codes, (name, code, dict(result.measurements))
