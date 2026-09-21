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
    프레임에서 표시 팔의 자리가 맞는가(`arm_position_faults`).

    좌우 비율은 **화면이 건을 쥔 손으로 그리는 점**에서 잰다: 닿은 프레임은 오른손, 닿지 못한 프레임은 점선이
    끝나는 손잡이다. 닿지 못한 팔은 사용자 결정("늘리지 않는다")대로 어깨에서 손잡이 쪽으로 곧게 편 채 멈추므로
    그 손끝은 어깨(몸 오른쪽) 가까이에 남는다 — 그 점을 건 자리로 세면 좌우가 아니라 팔 길이를 재게 된다.
    닿은 프레임만의 손 비율도 따로 대조한다(`hand_side_does_not_match_the_measured_gun_where_reached`).
    `hand_right`(모든 프레임의 손끝)는 참고로 싣는다."""
    faults: set[str] = set()
    hand_right = shown_right = gun_right = reached = 0
    reached_hand_right = reached_gun_right = 0
    for item in frames:
        placed, offsets, rep = item["placed"], item["offsets"], item["rep"]
        arm = arm_for(placed, offsets, rep)
        if arm is None:
            faults.add("no_arm")
            continue
        joints = service.displayed_joints(placed, arm)
        faults.update(arm_position_faults(arm.as_view(), joints, offsets))
        hand_side = _right_of_body(joints["RightHand"], rep.frame)
        gun_side = _right_of_body(rep.frame.right, rep.frame)
        hand_right += hand_side
        gun_right += gun_side
        shown_right += hand_side if arm.reached else _right_of_body(arm.grip, rep.frame)
        if arm.reached:
            reached += 1
            reached_hand_right += hand_side
            reached_gun_right += gun_side
    n = len(frames)
    stats = {
        "frames": float(n),
        "shown_right": shown_right / n,
        "gun_right": gun_right / n,
        "hand_right": hand_right / n,
        "reached": reached / n,
        "reached_hand_right": reached_hand_right / max(reached, 1),
        "reached_gun_right": reached_gun_right / max(reached, 1),
    }
    if abs(stats["shown_right"] - stats["gun_right"]) > SIDE_TOLERANCE:
        faults.add("hand_side_does_not_match_the_measured_gun")
    if abs(stats["reached_hand_right"] - stats["reached_gun_right"]) > SIDE_TOLERANCE:
        faults.add("hand_side_does_not_match_the_measured_gun_where_reached")
    return sorted(faults), stats


def test_displayed_right_hand_is_on_the_body_side_the_measured_gun_is_on() -> None:
    frames = _real_frames()
    assert len(frames) > 500, len(frames)
    faults, stats = side_faults(frames, service.arm_for)
    assert faults == [], stats
    # 실측 건 비율(사전등록 측정 77.7%)과 같은 쪽이다. CSV FK 오른손은 29.5%였다.
    assert stats["gun_right"] > 0.7, stats
    # 과반이 손잡이에 닿는다. 몸을 머리·왼손으로 세운 뒤로(`service.BODY_FIT_POINTS`) 61%다 — 세 앵커로 세우던
    # 때(85%)는 CSV 오른팔이 몸을 건 쪽으로 끌어 준 몫이 섞여 있었다. 못 닿는 프레임도 위 자리 검사를 다 통과했다.
    assert 0.5 < stats["reached"] < 1.0, stats


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
    assert "hand_side_does_not_match_the_measured_gun_where_reached" in faults, stats
    assert "hand_not_on_the_grip" in faults, stats


# --- 몸 배치: 머리·왼손으로 세운다(오른손 앵커 = CSV FK 손 + 장착 오프셋은 쓰지 않는다) -----------------------
#
# 오른손 앵커로 몸을 끌면 오른어깨가 건 손잡이 쪽으로 딸려 가 팔이 접힌다. 실측(위 624프레임): 세 앵커로 풀면
# 머리·왼손 어긋남 중앙값 8.6·11.5cm, 두 앵커로 풀면 3.9·3.9cm.

BODY_MISS_MEDIAN_LIMIT_M = 0.06


def body_fit_faults(frames: list[dict[str, Any]]) -> tuple[list[str], dict[str, float]]:
    """몸이 머리·왼손 실측에 붙어 서는가(어긋남 중앙값), 오른손 앵커는 몸을 세우는 데 빠졌는가."""
    faults: set[str] = set()
    misses: dict[str, list[float]] = {"head": [], "left": []}
    for item in frames:
        placed = item["placed"]
        for fit in placed.anchors:
            if fit.point in misses:
                misses[fit.point].append(fit.miss_m)
        solved = [fit.miss_m for fit in placed.anchors if fit.point in service.BODY_FIT_POINTS]
        if len(solved) >= skeleton.MIN_ANCHORS:
            rms = math.sqrt(sum(m * m for m in solved) / len(solved))
            if abs(placed.residual_m - rms) > 1e-12:
                faults.add("residual_is_not_over_the_body_fit_anchors")
    stats = {point: float(np.median(values)) for point, values in misses.items()}
    for point, value in stats.items():
        if value > BODY_MISS_MEDIAN_LIMIT_M:
            faults.add(f"body_does_not_sit_on_the_{point}_marker")
    return sorted(faults), stats


def test_body_is_placed_from_the_head_and_left_hand() -> None:
    assert service.BODY_FIT_POINTS == ("head", "left")
    faults, stats = body_fit_faults(_real_frames())
    assert faults == [], stats


def test_body_fit_check_fails_when_the_right_anchor_pulls_the_body_again() -> None:
    """변이: 세 앵커 전부로 몸을 푼다(직전 라운드의 배치). 머리·왼손 어긋남이 커져 잡힌다."""
    frames = _real_frames()
    table = service.load_mounts()
    old: list[dict[str, Any]] = []
    for item in frames[::4]:
        rep = item["rep"]
        sample = skeleton.frame_sample(rep.frame, rep.row, rep.gun, item["offsets"])
        placed = skeleton.place_oriented(
            sample.local_pos, sample.local_rot, sample.measured, skeleton.mounts_for(table.table, rep.gun)
        )
        old.append(dict(item, placed=placed))
    faults, stats = body_fit_faults(old)
    assert "body_does_not_sit_on_the_left_marker" in faults, stats
    assert "residual_is_not_over_the_body_fit_anchors" in faults, stats


# --- 화면이 받는 바로 그 페이로드(`service.compare` → view.layers.*.joints) -------------------------------
#
# 위 검사들은 `service.arm_for`·`displayed_joints`를 직접 부른다. 그것만으로는 `skeleton_block`이 페이로드에
# 싣는 관절이 이 팔인지 모른다 — `displayed_joints(placed, None)`로 되돌려 CSV FK 오른팔을 다시 실어도 스위트
# 전체가 통과했다(검증 라운드 M2). 여기서는 `/api/compare`가 부르는 `service.compare`의 출력을 읽는다.

COMPARE_QUERIES = (("1", "1000389_1"), ("3", "1000391_1"), ("6", "1000394_1"))


# 사용자가 고른 손잡이 추정값(m). `right_arm.GRIP_BEHIND_TRACKER_M`을 읽지 않고 여기 따로 적는다 — 페이로드의
# 손잡이를 모듈 상수가 아니라 **사용자 결정**과 대조한다(모듈 상수는 `test_grip_sits_behind_…`가 0.30으로 고정).
USER_GRIP_BEHIND_TRACKER_M = 0.30
GRIP_TOL_M = 1e-9


def measured_gun_at(rows: list[dict[str, str]], frame_index: int) -> dict[str, Any] | None:
    """페이로드 프레임(`fit.frame_index`)의 **CSV** 건 트래커 위치와 분사 축. `right_arm`을 거치지 않는다 —
    `measured_pose`의 회전 규약(`forward_vector`, 건 로컬 `GUN_FORWARD`)으로 원본 행에서 바로 읽는다."""
    gun = measured_pose.active_gun(rows)
    for frame, row in measured_pose.measured_frames_with_rows(rows):
        if int(frame.index) != frame_index or frame.right is None:
            continue
        angles = tuple(float(row[f"{gun} Rotation {axis}"]) for axis in "XYZ")
        forward = np.asarray(measured_pose.forward_vector(angles, measured_pose.GUN_FORWARD), dtype=float)
        return {"tracker": np.asarray(frame.right, dtype=float), "forward": forward / np.linalg.norm(forward)}
    return None


def grip_faults(arm: dict[str, Any], right_anchor: dict[str, Any] | None, measured: dict[str, Any] | None) -> list[str]:
    """페이로드 손잡이 = CSV 트래커 − 0.30m · CSV 건 전방인가, 그리고 건 막대가 손잡이 → 트래커인가.

    자리만 보는 `arm_position_faults`는 **스스로 맞는** 틀린 손잡이(예: 몸 반대편으로 뒤집은 손잡이·팔)를
    통과시킨다 — 손이 그 손잡이에 붙어 있고 뼈 길이도 맞기 때문이다. 손잡이 자체를 실측에서 다시 짓는다."""
    if measured is None:
        return ["no_csv_frame_for_the_payload"]
    faults: list[str] = []
    tracker = measured["tracker"]
    expected = tracker - USER_GRIP_BEHIND_TRACKER_M * measured["forward"]
    if float(np.linalg.norm(np.asarray(arm["tracker"], dtype=float) - tracker)) > GRIP_TOL_M:
        faults.append("tracker_is_not_the_csv_gun")
    if float(np.linalg.norm(np.asarray(arm["grip"], dtype=float) - expected)) > GRIP_TOL_M:
        faults.append("grip_is_not_behind_the_measured_tracker")
    device = (right_anchor or {}).get("device") or {}
    # 막대의 손 쪽 끝 = 손잡이. `predicted_marker`(장착 상수 예측)나 FK 손으로 옮기면 여기서 잡힌다.
    if not device.get("from") or float(np.linalg.norm(np.asarray(device["from"], dtype=float) - expected)) > GRIP_TOL_M:
        faults.append("gun_bar_hand_end_is_not_the_grip")
    if not device.get("to") or float(np.linalg.norm(np.asarray(device["to"], dtype=float) - tracker)) > GRIP_TOL_M:
        faults.append("gun_bar_far_end_is_not_the_tracker")
    return faults


def payload_arm_faults(
    view: dict[str, Any], offsets_by_layer: dict[str, Any], rows_by_layer: dict[str, Any] | None = None
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for layer, fit_key in (("query", "fit"), ("reference", "reference_fit")):
        joints = (view.get("layers") or {}).get(layer, {}).get("joints")
        arm = (view.get(fit_key) or {}).get("right_arm")
        if not joints or not arm:
            out[layer] = ["no_arm_in_the_payload"]
            continue
        faults = arm_position_faults(arm, joints, offsets_by_layer[layer])
        right = next((a for a in view[fit_key]["anchors"] if a["point"] == "right"), None)
        if right is None or right["body_fit"] is not False or right["miss_display"] is not None:
            faults.append("right_anchor_is_reported_as_fitting_the_body")
        if any(a["body_fit"] is not True for a in view[fit_key]["anchors"] if a["point"] != "right"):
            faults.append("head_or_left_anchor_is_not_fitting_the_body")
        if rows_by_layer is not None:
            measured = measured_gun_at(rows_by_layer[layer], int(view[fit_key]["frame_index"]))
            faults.extend(grip_faults(arm, right, measured))
        out[layer] = faults
    return out


def _compare_views() -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    if "views" in _REAL:
        return _REAL["views"]
    from trainer.pose_lab.tests import test_scoring_isolation as isolation

    source, kwargs = isolation._source_for({"name": "real"})  # noqa: SLF001
    out = []
    for level, session in COMPARE_QUERIES:
        query = service.PassRef(worker="김경호", date="20250831", level=level, session=session, pass_index=0)
        candidates = service.reference_candidates(source, query)
        assert candidates["status"] == "ok" and candidates["candidates"], candidates
        first = candidates["candidates"][0]
        reference = service.PassRef(**{key: first[key] for key in ("worker", "date", "level", "session", "pass_index")})
        view = service.compare(source, query, reference, **kwargs)["view"]
        rows = {layer: source.load(ref).rows for layer, ref in (("query", query), ("reference", reference))}
        offsets = {
            layer: body_proportions.rig_offsets(pose_fk_csv.bind_offsets(layer_rows[0]))
            for layer, layer_rows in rows.items()
        }
        out.append((level, view, offsets, rows))
    _REAL["views"] = out
    return out


def test_compare_payload_draws_the_grip_arm_on_both_layers() -> None:
    for level, view, offsets, rows in _compare_views():
        assert payload_arm_faults(view, offsets, rows) == {"query": [], "reference": []}, level


def test_payload_arm_check_fails_when_the_payload_carries_the_csv_arm() -> None:
    """변이 M2의 모양: 페이로드 관절의 오른 팔꿈치·손을 CSV FK 자리(오른손 앵커 `joint_position`)로 되돌린다."""
    level, view, offsets, _rows = _compare_views()[1]
    right = next(a for a in view["fit"]["anchors"] if a["point"] == "right")
    joints = dict(view["layers"]["query"]["joints"], RightHand=right["joint_position"])
    mutated = dict(view, layers=dict(view["layers"], query=dict(view["layers"]["query"], joints=joints)))
    faults = payload_arm_faults(mutated, offsets)
    assert "drawn_joints_are_not_the_arm" in faults["query"], (level, faults)
    fitted = dict(right, body_fit=True, miss_display="12.0cm")
    anchors = [fitted if a["point"] == "right" else a for a in view["fit"]["anchors"]]
    faults = payload_arm_faults(dict(view, fit=dict(view["fit"], anchors=anchors)), offsets)
    assert faults["query"] == ["right_anchor_is_reported_as_fitting_the_body"], faults


def _flip_across_body(point: Any, head: np.ndarray, right: np.ndarray) -> list[float]:
    v = np.asarray(point, dtype=float)
    return [float(x) for x in v - 2.0 * float(np.dot(v - head, right)) * right]


def test_payload_grip_check_fails_for_a_self_consistent_mirrored_grip() -> None:
    """변이 MX2(검증 라운드): 손잡이를 몸 정중면 너머로 뒤집고 그 손잡이로 팔·점선·건 막대 손 쪽 끝을 **함께** 다시 짓는다.
    자리 검사(`arm_position_faults`)는 이 팔을 통과시킨다 — 손이 뒤집힌 손잡이에 붙어 있고 뼈 길이도 맞다.
    손잡이를 CSV에서 다시 짓는 검사가 잡는다. 막대 손 쪽 끝만 `predicted_marker`로 옮긴 변이도 잡는다."""
    level, view, offsets, rows = _compare_views()[0]
    fit = view["fit"]
    arm = fit["right_arm"]
    head_anchor = next(a for a in fit["anchors"] if a["point"] == "head")
    head = np.asarray(head_anchor["measured_marker"], dtype=float)
    front = np.asarray(world_axes.flatten(tuple(np.asarray(arm["tracker"], dtype=float) - head)))
    front /= np.linalg.norm(front)
    right = np.cross(np.asarray(world_axes.UP, dtype=float), front)
    # 뒤집은 손잡이로 **같은 어깨에서** 팔을 다시 푼다 — 뼈 길이·손 자리가 그 손잡이와 스스로 맞는 팔이다.
    grip = _flip_across_body(arm["grip"], head, right)
    joints_now = view["layers"]["query"]["joints"]
    elbow, hand, reached = right_arm.solve_two_bone(
        arm["shoulder"], grip, arm["upper_arm_m"], arm["forearm_m"], right_arm.pole_direction(joints_now)
    )
    flipped = {"grip": grip, "elbow": list(elbow), "hand": list(hand)}
    mirrored_arm = dict(arm, **flipped, reached=reached)
    mirrored_arm["reach_gap"] = None if reached else {"from": flipped["hand"], "to": flipped["grip"]}
    anchors = [
        dict(a, device=dict(a["device"], **{"from": flipped["grip"]})) if a["point"] == "right" else a
        for a in fit["anchors"]
    ]
    joints = dict(view["layers"]["query"]["joints"], RightForeArm=flipped["elbow"], RightHand=flipped["hand"])
    mutated = dict(
        view,
        fit=dict(fit, right_arm=mirrored_arm, anchors=anchors),
        layers=dict(view["layers"], query=dict(view["layers"]["query"], joints=joints)),
    )
    # 자리 검사만으로는 통과한다 — 이것이 검증 라운드가 짚은 공백이다.
    assert arm_position_faults(mirrored_arm, joints, offsets["query"]) == []
    faults = payload_arm_faults(mutated, offsets, rows)["query"]
    assert "grip_is_not_behind_the_measured_tracker" in faults, (level, faults)
    assert "gun_bar_hand_end_is_not_the_grip" in faults, (level, faults)

    right_anchor = next(a for a in fit["anchors"] if a["point"] == "right")
    predicted = [
        dict(a, device=dict(a["device"], **{"from": a["predicted_marker"]})) if a["point"] == "right" else a
        for a in fit["anchors"]
    ]
    assert right_anchor["predicted_marker"] != arm["grip"]
    faults = payload_arm_faults(dict(view, fit=dict(fit, anchors=predicted)), offsets, rows)["query"]
    assert faults == ["gun_bar_hand_end_is_not_the_grip"], (level, faults)
