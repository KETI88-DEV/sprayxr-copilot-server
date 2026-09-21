#!/usr/bin/env python3
"""표시용 오른팔(오른어깨 → 건 손잡이 두 마디 IK)의 **위치**를 읽는 검사.

기존 `test_fk_chain_right_hand_tracks_gun_relative_rotation`은 관절 회전만 비교하고 위치를 한 번도 읽지
않는다 — 오른팔을 가슴 반대편으로 0.65~0.88m 옮겨도 통과했다. 여기 검사들은 표시되는 오른 팔꿈치·손의
**자리**를 본다: 손이 손잡이에 있는가(못 닿으면 손잡이 쪽으로 곧게 편 팔 끝인가), 뼈 길이가 리그 그대로인가,
그리고 실제 파일 여러 패스에서 표시 오른손이 몸 오른쪽에 있는 비율이 실측 건의 비율과 맞는가.
"""

from __future__ import annotations

import dataclasses
import importlib
import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

measured_pose = importlib.import_module("measured_pose")
pose_fk_csv = importlib.import_module("pose_fk_csv")
rotation_math = importlib.import_module("rotation_math")
world_axes = importlib.import_module("world_axes")

from trainer.pose_lab import body_proportions, right_arm, service, skeleton  # noqa: E402

DATA_ROOT = Path("/data/sources/professionals")

# 리그 바인드 오프셋(실측 CSV의 값). 상완·전완 둘 다 0.243m다 — 검사는 이 표가 아니라 **읽은 오프셋**과
# 대조한다. 합성 검사에서만 쓴다.
UPPER = (-0.2430, 0.0, 0.0)
FORE = (-0.2430, 0.0, 0.0)
OFFSETS = {"RightForeArm": UPPER, "RightHand": FORE}
TOL = 1e-9


def _gun_row(angles: tuple[float, float, float], gun: str = "한길숏건") -> dict[str, str]:
    return {f"{gun} Rotation {axis}": str(value) for axis, value in zip("XYZ", angles)}


# --- 손잡이 --------------------------------------------------------------------------------


def test_grip_sits_behind_the_tracker_on_the_measured_gun_axis() -> None:
    """손잡이 = 트래커 − 0.30 · 건 전방. 건 전방은 `measured_pose` 규약 행렬의 로컬 +Y 열이다(독립 계산)."""
    angles = (31.0, 72.5, -118.0)
    matrix = rotation_math.euler_matrix(measured_pose.EULER_ORDER, angles, measured_pose.EULER_SIGN)
    forward = tuple(matrix[i][1] for i in range(3))
    tracker = (0.4, 1.3, -0.2)
    joints = {"RightUpperArm": (0.2, 1.4, 0.0), "LeftUpperArm": (-0.2, 1.4, 0.0)}
    arm = right_arm.right_arm(joints, OFFSETS, _gun_row(angles), "한길숏건", tracker)
    assert arm is not None
    expected = tuple(tracker[i] - right_arm.GRIP_BEHIND_TRACKER_M * forward[i] for i in range(3))
    assert math.dist(arm.grip, expected) < TOL
    assert math.dist(arm.grip, tracker) == pytest.approx(right_arm.GRIP_BEHIND_TRACKER_M)
    # 추정값은 이름 있는 상수 하나다 — 사용자가 고른 0.30m.
    assert right_arm.GRIP_BEHIND_TRACKER_M == 0.30


# --- 위치 결함 검사(합성·실데이터가 같은 판정을 쓴다) -----------------------------------------------


def arm_position_faults(
    arm: dict[str, Any], joints: dict[str, Any], offsets: dict[str, Any]
) -> list[str]:
    """표시 오른팔의 **자리** 결함. `arm`은 `fit.right_arm`, `joints`는 화면이 그리는 관절이다."""
    faults: list[str] = []
    shoulder = np.asarray(joints["RightUpperArm"], dtype=float)
    elbow = np.asarray(joints["RightForeArm"], dtype=float)
    hand = np.asarray(joints["RightHand"], dtype=float)
    grip = np.asarray(arm["grip"], dtype=float)
    upper = math.hypot(*offsets["RightForeArm"])
    fore = math.hypot(*offsets["RightHand"])
    if abs(float(np.linalg.norm(elbow - shoulder)) - upper) > 1e-6:
        faults.append("upper_arm_is_not_the_rig_length")
    if abs(float(np.linalg.norm(hand - elbow)) - fore) > 1e-6:
        faults.append("forearm_is_not_the_rig_length")
    if math.dist(tuple(hand), tuple(arm["hand"])) > 1e-9 or math.dist(tuple(elbow), tuple(arm["elbow"])) > 1e-9:
        faults.append("drawn_joints_are_not_the_arm")
    reach = float(np.linalg.norm(grip - shoulder))
    if arm["reached"]:
        if float(np.linalg.norm(hand - grip)) > 1e-6:
            faults.append("hand_not_on_the_grip")
        if reach > upper + fore + 1e-9:
            faults.append("claims_reach_beyond_the_arm")
    else:
        direction = (grip - shoulder) / reach
        straight_end = shoulder + (upper + fore) * direction
        if float(np.linalg.norm(hand - straight_end)) > 1e-6:
            faults.append("unreached_hand_is_not_the_straight_arm_end")
        if reach < upper + fore - 1e-9:
            faults.append("claims_no_reach_within_the_arm")
        gap = arm.get("reach_gap") or {}
        if gap.get("from") != arm["hand"] or gap.get("to") != arm["grip"]:
            faults.append("reach_gap_is_not_hand_to_grip")
    return faults


def _synthetic(tracker: tuple[float, float, float]) -> tuple[dict[str, Any], dict[str, Any]]:
    joints = {"RightUpperArm": (0.2, 1.4, 0.0), "LeftUpperArm": (-0.2, 1.4, 0.0)}
    # Rotation X = 90°(YXZ 규약)는 건 로컬 +Y를 월드 +Z(앞)로 돌린다 — 손잡이는 트래커에서 0.30m 뒤(−Z)다.
    arm = right_arm.right_arm(joints, OFFSETS, _gun_row((90.0, 0.0, 0.0)), "한길숏건", tracker)
    assert arm is not None
    view = arm.as_view()
    drawn = dict(joints, RightForeArm=arm.elbow, RightHand=arm.hand)
    return view, drawn


def test_reachable_grip_puts_the_hand_on_it_with_rig_lengths() -> None:
    view, drawn = _synthetic((0.35, 1.15, 0.55))
    assert math.dist(view["grip"], (0.35, 1.15, 0.25)) < 1e-12
    assert view["reached"] is True and view["reach_gap"] is None
    assert arm_position_faults(view, drawn, OFFSETS) == []


def test_unreachable_grip_leaves_a_straight_arm_and_a_gap_to_the_grip() -> None:
    view, drawn = _synthetic((0.3, 1.3, 1.6))
    assert view["reached"] is False
    assert arm_position_faults(view, drawn, OFFSETS) == []
    # 늘리지 않는다 — 어깨→손은 두 뼈 길이의 합 그대로다.
    assert math.dist(drawn["RightUpperArm"], drawn["RightHand"]) == pytest.approx(0.486)


def test_elbow_bends_down_and_out() -> None:
    """폴 = 아래 + 바깥(반대편 어깨 → 이 어깨의 수평 방향). 팔꿈치는 어깨→손잡이 선보다 아래·바깥이다."""
    view, drawn = _synthetic((0.3, 1.2, 0.6))
    shoulder = np.asarray(drawn["RightUpperArm"])
    grip = np.asarray(view["grip"])
    elbow = np.asarray(view["elbow"])
    t = float(np.dot(elbow - shoulder, grip - shoulder) / np.dot(grip - shoulder, grip - shoulder))
    off_line = elbow - (shoulder + t * (grip - shoulder))
    up = np.asarray(world_axes.UP, dtype=float)
    out = np.asarray(world_axes.flatten(tuple(shoulder - np.asarray(drawn["LeftUpperArm"]))))
    assert float(np.dot(off_line, up)) < 0.0
    assert float(np.dot(off_line, out)) > 0.0


def test_arm_position_checks_can_fail() -> None:
    view, drawn = _synthetic((0.35, 1.15, 0.55))
    # 팔을 몸 반대편으로 옮기되(손·팔꿈치 자리를 좌우로 뒤집는다) 손 회전(건 전방)은 그대로 둔 변형.
    mirrored = {
        name: (-p[0], p[1], p[2]) if name in ("RightForeArm", "RightHand") else p for name, p in drawn.items()
    }
    moved = dict(view, hand=list(mirrored["RightHand"]), elbow=list(mirrored["RightForeArm"]))
    assert "hand_not_on_the_grip" in arm_position_faults(moved, mirrored, OFFSETS)
    stretched = dict(drawn, RightHand=tuple(np.asarray(drawn["RightHand"]) * 1.05))
    assert "forearm_is_not_the_rig_length" in arm_position_faults(dict(view, hand=list(stretched["RightHand"])), stretched, OFFSETS)
    fk = dict(drawn, RightHand=(0.0, 0.9, 0.1))
    assert "drawn_joints_are_not_the_arm" in arm_position_faults(view, fk, OFFSETS)
    far_view, far_drawn = _synthetic((0.3, 1.3, 1.6))
    snapped = dict(far_drawn, RightHand=tuple(far_view["grip"]))
    assert "unreached_hand_is_not_the_straight_arm_end" in arm_position_faults(
        dict(far_view, hand=far_view["grip"]), snapped, OFFSETS
    )


# --- 실데이터: 여러 패스의 모든 분사 프레임(홀수 프레임 건너뜀) -----------------------------------------
#
# 몸 좌표는 스켈레톤과 무관하게 잡는다: 정면 = HMD → 착탄점의 수평 성분, 오른쪽 = cross(UP, 정면)(유니티
# Y-up). 실측 건(트래커)이 몸 오른쪽에 있는 비율과 표시 오른손이 몸 오른쪽에 있는 비율이 맞아야 한다 —
# 사전등록 측정에서 실측 건 77.7%, CSV FK 오른손 29.5%였다.

SIDE_PASSES = (("김경호", "1"), ("김경호", "3"), ("김경호", "6"), ("전형기", "1"), ("전형기", "3"),
               ("전형기", "6"), ("손영석", "1"), ("손영석", "3"), ("손영석", "6"))
PASSES_PER_LEVEL = 3
SIDE_TOLERANCE = 0.05

_REAL: dict[str, Any] = {}


def _real_frames() -> list[dict[str, Any]]:
    """(프레임 재료, 배치, 오프셋) 목록. 한 번만 읽는다."""
    if "frames" in _REAL:
        return _REAL["frames"]
    if not DATA_ROOT.is_dir():
        pytest.fail(f"실데이터가 없다({DATA_ROOT}) — 이 검사는 skip하지 않는다")
    from trainer.pose_lab import csv_source

    source = csv_source.CsvPassSource(DATA_ROOT)
    table = service.load_mounts()
    out: list[dict[str, Any]] = []
    for worker, level in SIDE_PASSES:
        found, _total = source.passes(worker, level, limit=1)
        for pass_index, (item, _entry) in enumerate(found[:PASSES_PER_LEVEL]):
            raw = source.load(service.PassRef(item.worker, item.date, item.level, item.session, pass_index))
            representative = service.representative_frame(raw)
            if representative is None:
                continue
            offsets = body_proportions.rig_offsets(pose_fk_csv.bind_offsets(raw.rows[0]))
            mounts = skeleton.mounts_for(table.table, representative.gun)
            for frame, row in representative.spraying[::2]:
                sample = skeleton.frame_sample(frame, row, representative.gun, offsets)
                if sample is None:
                    continue
                placed = service.place_frame(sample, mounts)
                if placed is None:
                    continue
                here = dataclasses.replace(representative, frame=frame, row=row)
                out.append({"placed": placed, "offsets": offsets, "rep": here})
    _REAL["frames"] = out
    return out


def _right_of_body(point: Any, frame: Any) -> bool:
    head = np.asarray(frame.head, dtype=float)
    front = np.asarray(world_axes.flatten(tuple(np.asarray(frame.hit, dtype=float) - head)))
    front /= np.linalg.norm(front)
    right = np.cross(np.asarray(world_axes.UP, dtype=float), front)
    return float(np.dot(np.asarray(point, dtype=float) - head, right)) > 0.0


def side_faults(
    frames: list[dict[str, Any]], arm_for: Callable[..., Any]
) -> tuple[list[str], dict[str, float]]:
    """표시 오른손이 몸 오른쪽인 비율이 실측 건의 비율과 `SIDE_TOLERANCE` 안에서 맞는가, 그리고 모든
    프레임에서 표시 팔의 자리가 맞는가(`arm_position_faults`)."""
    faults: set[str] = set()
    hand_right = gun_right = reached = 0
    for item in frames:
        placed, offsets, rep = item["placed"], item["offsets"], item["rep"]
        arm = arm_for(placed, offsets, rep)
        if arm is None:
            faults.add("no_arm")
            continue
        joints = service.displayed_joints(placed, arm)
        faults.update(arm_position_faults(arm.as_view(), joints, offsets))
        hand_right += _right_of_body(joints["RightHand"], rep.frame)
        gun_right += _right_of_body(rep.frame.right, rep.frame)
        reached += arm.reached
    n = len(frames)
    stats = {"frames": float(n), "hand_right": hand_right / n, "gun_right": gun_right / n, "reached": reached / n}
    if abs(stats["hand_right"] - stats["gun_right"]) > SIDE_TOLERANCE:
        faults.add("hand_side_does_not_match_the_measured_gun")
    return sorted(faults), stats


def test_displayed_right_hand_is_on_the_body_side_the_measured_gun_is_on() -> None:
    frames = _real_frames()
    assert len(frames) > 500, len(frames)
    faults, stats = side_faults(frames, service.arm_for)
    assert faults == [], stats
    # 실측 건 비율(사전등록 측정 77.7%)과 같은 쪽이다. CSV FK 오른손은 29.5%였다.
    assert stats["gun_right"] > 0.7, stats
    # 대부분 손잡이에 닿는다(사전 확인 약 85%). 못 닿는 프레임도 위 자리 검사를 다 통과했다.
    assert 0.7 < stats["reached"] < 1.0, stats


def test_side_check_fails_when_the_arm_moves_to_the_other_side_keeping_the_hand_rotation() -> None:
    """변이: 팔(팔꿈치·손)을 몸 정중면 너머로 뒤집어 옮기되 손 회전(건 전방)은 그대로 둔다 — 옛 회전
    검사는 이런 팔을 통과시켰다. 이 검사는 자리와 좌우 비율 둘 다에서 잡는다."""
    frames = _real_frames()

    def mirrored(placed: Any, offsets: Any, rep: Any) -> Any:
        arm = service.arm_for(placed, offsets, rep)
        if arm is None:
            return None
        head = np.asarray(rep.frame.head, dtype=float)
        front = np.asarray(world_axes.flatten(tuple(np.asarray(rep.frame.hit, dtype=float) - head)))
        front /= np.linalg.norm(front)
        right = np.cross(np.asarray(world_axes.UP, dtype=float), front)

        def flip(p: Any) -> tuple[float, float, float]:
            v = np.asarray(p, dtype=float)
            out = v - 2.0 * float(np.dot(v - head, right)) * right
            return (float(out[0]), float(out[1]), float(out[2]))

        return dataclasses.replace(arm, elbow=flip(arm.elbow), hand=flip(arm.hand))

    faults, stats = side_faults(frames, mirrored)
    assert "hand_side_does_not_match_the_measured_gun" in faults, stats
    assert "hand_not_on_the_grip" in faults, stats
