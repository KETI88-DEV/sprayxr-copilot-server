#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false
"""관절 각도 추정표(`joint_angles.py`, D4).

값이 맞는지는 **합성 입력으로 분기를 직접 고정**해 가른다. 리그를 곧은 팔로 만들어 두면 전완 로컬
회전 θ(Z축)가 팔꿈치 각을 정확히 180−θ로 만든다 — 실데이터만 보는 단언은 그 관계를 못 잡는다.
품질 수치(0° 비율·급변 비율)도 같은 방식으로 프레임마다 회전을 심어 고정한다.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

simulator_csv = importlib.import_module("simulator_csv")

from trainer.pose_lab import joint_angles, service  # noqa: E402
from trainer.pose_lab.service import PassRef  # noqa: E402
from trainer.pose_lab.tests import test_scoring_isolation as isolation  # noqa: E402

# 곧은 팔·곧은 다리 리그(부모 기준 바인드 오프셋). 전완·손이 +X로 일직선이라 회전이 0이면 팔꿈치가
# 정확히 180°(편 팔)다. 전완 로컬 Z 회전 θ만 주면 손 오프셋이 +X에서 θ만큼 돌아 팔꿈치 각이 180−θ가 된다.
RIG: dict[str, tuple[float, float, float]] = {
    "Spine": (0.0, 0.0, 0.2),
    "Chest": (0.0, 0.0, 0.2),
    "Head": (0.0, 0.0, 0.2),
    "LeftShoulder": (0.0, 0.1, 0.15),
    "LeftUpperArm": (0.0, 0.15, 0.0),
    "LeftForeArm": (0.3, 0.0, 0.0),
    "LeftHand": (0.25, 0.0, 0.0),
    "RightShoulder": (0.0, -0.1, 0.15),
    "RightUpperArm": (0.0, -0.15, 0.0),
    "RightForeArm": (0.3, 0.0, 0.0),
    "RightHand": (0.25, 0.0, 0.0),
    "LeftThigh": (0.0, 0.1, 0.0),
    "LeftCalf": (0.0, 0.0, -0.4),
    "LeftFoot": (0.0, 0.0, -0.4),
    "LeftToe": (0.15, 0.0, 0.0),
    "RightThigh": (0.0, -0.1, 0.0),
    "RightCalf": (0.0, 0.0, -0.4),
    "RightFoot": (0.0, 0.0, -0.4),
    "RightToe": (0.15, 0.0, 0.0),
}


def _rows(rotations_by_frame: list[dict[str, tuple[float, float, float]]]) -> list[dict[str, str]]:
    """분사 중·착탄점이 있는 프레임 목록. 프레임마다 심을 관절 로컬 회전을 받는다."""
    rows: list[dict[str, str]] = []
    for i, rotations in enumerate(rotations_by_frame):
        row = {
            "인덱스": str(i),
            "시간(초)": f"{i * 0.01:.3f}",
            "분사": "On",
            "Ray Texture UV X": "1",
            "Ray Texture UV Y": "1",
        }
        for axis, value in zip("XYZ", (0.5, 1.6, 4.0 + 0.02 * i)):
            row[f"카메라 Position {axis}"] = str(value)
        for axis, value in zip("XYZ", (0.8, 1.3, 4.0 + 0.02 * i)):
            row[f"한길숏건 Position {axis}"] = str(value)
        for axis, value in zip("XYZ", (60.0, 40.0, 25.0)):
            row[f"한길숏건 Rotation {axis}"] = str(value)
        for name, value in zip(("Ray point X", " Ray Point Y", " Ray Point Z"), (0.8, 1.3, 5.0)):
            row[name] = str(value)
        for column in simulator_csv.JOINT_COLUMNS:
            key = simulator_csv.JOINT_KEYS[column]
            for axis, value in zip("XYZ", RIG.get(key, (0.0, 0.0, 0.0))):
                row[f"{column} Position {axis}"] = str(value)
            for axis, value in zip("XYZ", rotations.get(column, (0.0, 0.0, 0.0))):
                row[f"{column} Rotation {axis}"] = str(value)
        rows.append(row)
    return rows


def _stats(rows: list[dict[str, str]], key: str) -> joint_angles.AngleStats:
    stats = joint_angles.pass_angles(rows)[key]
    assert stats is not None
    return stats


def test_straight_arm_reads_180_degrees_and_says_the_rotation_is_zero() -> None:
    """전완 회전이 0°인 프레임에서 팔꿈치는 편 팔(180°)이고, 화면은 그 사실을 0° 비율로 말한다."""
    stats = _stats(_rows([{} for _ in range(10)]), "right_elbow")
    assert stats.mean_deg == pytest.approx(180.0, abs=1e-6)
    assert stats.spread_deg == pytest.approx(0.0, abs=1e-9)
    assert stats.zero_pct == 100.0
    assert stats.jump_pct == 0.0
    assert stats.samples == 10


@pytest.mark.parametrize("theta", [30.0, 50.0, 110.0])
def test_elbow_angle_is_180_minus_the_forearm_z_rotation(theta: float) -> None:
    """곧은 팔 리그에서 전완 로컬 Z 회전 θ는 팔꿈치 각을 정확히 180−θ로 만든다(손 전개, 규약 YXZ)."""
    rows = _rows([{"Right Fore Arm": (0.0, 0.0, theta)} for _ in range(8)])
    stats = _stats(rows, "right_elbow")
    assert stats.mean_deg == pytest.approx(180.0 - theta, abs=1e-6)
    assert stats.zero_pct == 0.0


def test_quality_counts_zero_frames_and_frame_to_frame_jumps() -> None:
    """프레임마다 0°↔50°를 번갈아 심으면 0° 비율 50%, 이웃 사이 급변 100%다."""
    frames = [{} if i % 2 == 0 else {"Right Fore Arm": (0.0, 0.0, 50.0)} for i in range(10)]
    stats = _stats(_rows(frames), "right_elbow")
    assert stats.zero_pct == pytest.approx(50.0)
    assert stats.jump_pct == pytest.approx(100.0)
    assert stats.spread_deg > 20.0  # 값이 상수가 아니라는 사실도 같이 보인다


def test_quality_does_not_call_a_small_wobble_a_jump() -> None:
    """문턱(30°) 아래 흔들림은 급변이 아니다 — 문턱이 실제로 분기를 가르는지 가른다."""
    frames = [{} if i % 2 == 0 else {"Right Fore Arm": (0.0, 0.0, 20.0)} for i in range(10)]
    stats = _stats(_rows(frames), "right_elbow")
    assert stats.jump_pct == 0.0
    assert stats.zero_pct == pytest.approx(50.0)


def test_knee_and_shoulder_use_their_own_driver_rotations() -> None:
    """무릎은 종아리 회전이, 어깨는 상완 회전이 움직인다 — 관절마다 다른 컬럼을 본다.

    다리는 −Z로 뻗으므로 무릎을 굽히는 것은 Z(비틀기)가 아니라 X 회전이다."""
    rows = _rows([{"Right Calf": (40.0, 0.0, 0.0)} for _ in range(6)])
    angles = joint_angles.pass_angles(rows)
    assert angles["right_knee"].zero_pct == 0.0
    assert angles["right_knee"].mean_deg == pytest.approx(140.0, abs=1e-6)
    # 같은 프레임에서 팔꿈치·어깨의 회전은 여전히 0°다.
    assert angles["right_elbow"].zero_pct == 100.0
    assert angles["right_shoulder"].zero_pct == 100.0
    assert angles["left_knee"].zero_pct == 100.0


def test_block_contract_and_wording() -> None:
    block = joint_angles.joint_angles_block(_rows([{} for _ in range(6)]), _rows([{} for _ in range(6)]))
    assert [row["key"] for row in block["rows"]] == [spec.key for spec in joint_angles.ANGLE_SPECS]
    elbow = next(row for row in block["rows"] if row["key"] == "right_elbow")
    assert elbow["display"] == {"query": "180°", "reference": "180°"}
    assert "0° 100%" in elbow["quality_display"]["query"]
    assert elbow["samples"] == {"query": 6, "reference": 6}
    assert elbow["reason"] is None
    blob = json.dumps(block, ensure_ascii=False)
    # 화면 문구는 "추정"이라고 쓴다. 금지어는 화면에 나가지 않는다.
    assert "추정" in blob
    for word in ("미사용", "사용하지 않", "unused", "not used", "표시하지 않습니다"):
        assert word not in blob


def test_block_reports_a_reason_instead_of_inventing_values() -> None:
    block = joint_angles.joint_angles_block([], [])
    for row in block["rows"]:
        assert row["query"] is None and row["reference"] is None
        assert row["reason"] == "no_spraying_frame"
        assert row["reason_text"] == joint_angles.REASON_TEXT["no_spraying_frame"]
        assert row["display"] == {"query": "—", "reference": "—"}


def test_real_pass_shows_a_constant_right_elbow_and_says_so() -> None:
    """실데이터 L1·L3: 오른 팔꿈치는 180°(편 팔)이고 전완 회전 0° 비율이 100%로 함께 실린다."""
    source, kwargs = isolation._source_for({"name": "real"})  # noqa: SLF001
    for level, session, ref_session, ref_index in (
        ("1", "1000389_1", "1000564_1", 17),
        ("3", "1000391_1", "1000444_1", 10),
    ):
        body = service.compare(
            source,
            PassRef(worker="김경호", date="20250831", level=level, session=session, pass_index=0),
            PassRef(
                worker="전형기",
                date="20250904" if level == "1" else "20250901",
                level=level,
                session=ref_session,
                pass_index=ref_index,
            ),
            **kwargs,
        )
        assert body["status"] == "ok"
        block = body["measured"]["joint_angles"]
        assert block is not None
        elbow = next(row for row in block["rows"] if row["key"] == "right_elbow")
        assert elbow["query"] == pytest.approx(180.0, abs=0.5), (level, elbow)
        assert elbow["quality"]["query"]["zero_pct"] == pytest.approx(100.0)
        assert "0° 100%" in elbow["quality_display"]["query"]
        # 왼 팔꿈치는 상수가 아니다 — 두 관절이 같은 문구를 달면 이 표가 아무것도 말하지 않는다.
        left = next(row for row in block["rows"] if row["key"] == "left_elbow")
        assert left["quality"]["query"]["zero_pct"] == pytest.approx(0.0)
        assert abs(left["query"] - 180.0) > 30.0
