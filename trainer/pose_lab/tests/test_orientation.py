#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false
"""건 각도 참고표(`orientation.py`)."""

from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

measured_pose = importlib.import_module("measured_pose")

from trainer.pose_lab import composite, orientation, service  # noqa: E402
from trainer.pose_lab.service import PassRef  # noqa: E402
from trainer.pose_lab.tests import test_scoring_isolation as isolation  # noqa: E402

ANGLES = (60.0, 40.0, 25.0)  # 건 Rotation (X, Y, Z)


def _plane(z: float = 5.0, flip: bool = False) -> composite.MeshGeometry:
    """z = const 평면(법선 ±Z) 큰 사각형 두 삼각형."""
    vertices = [(-10.0, -10.0, z), (10.0, -10.0, z), (10.0, 10.0, z), (-10.0, 10.0, z)]
    triangles = [(0, 1, 2), (0, 2, 3)]
    if flip:
        triangles = [(t[0], t[2], t[1]) for t in triangles]
    return composite.MeshGeometry.from_mesh(vertices, triangles)


def _rows(n: int = 12, angles=ANGLES) -> list[dict[str, str]]:  # noqa: ANN001
    rows = []
    for i in range(n):
        row = {"인덱스": str(i), "시간(초)": f"{i * 0.01:.3f}", "분사": "On", "Ray Texture UV X": "1", "Ray Texture UV Y": "1"}
        for axis, value in zip("XYZ", (0.5, 1.6, 4.0 + 0.02 * i)):
            row[f"카메라 Position {axis}"] = str(value)
        for axis, value in zip("XYZ", (0.8, 1.3, 4.0 + 0.02 * i)):
            row[f"한길숏건 Position {axis}"] = str(value)
        for axis, value in zip("XYZ", angles):
            row[f"한길숏건 Rotation {axis}"] = str(value)
        for axis, value in zip(("Ray point X", " Ray Point Y", " Ray Point Z"), (0.8, 1.3, 5.0)):
            row[axis] = str(value)
        rows.append(row)
    return rows


def _expected() -> tuple[float, float]:
    """YXZ(R = Ry·Rx·Rz), 전방 +Y = R의 둘째 열 = (−cb·sg + sb·sa·cg, ca·cg, sb·sg + cb·sa·cg). 손 전개."""
    a, b, g = (math.radians(v) for v in ANGLES)
    forward = (
        -math.cos(b) * math.sin(g) + math.sin(b) * math.sin(a) * math.cos(g),
        math.cos(a) * math.cos(g),
        math.sin(b) * math.sin(g) + math.cos(b) * math.sin(a) * math.cos(g),
    )
    pitch = math.degrees(math.asin(forward[1]))
    incidence = math.degrees(math.acos(abs(forward[2])))
    return pitch, incidence


def test_pitch_and_incidence_match_the_analytic_forward_axis() -> None:
    pitch, incidence = _expected()
    angles = orientation.pass_angles(_rows(), _plane())
    assert abs(angles["gun_pitch_deg"][0] - pitch) < 1e-6
    assert abs(angles["gun_incidence_deg"][0] - incidence) < 1e-6
    assert angles["gun_incidence_deg"][1] == 12


def test_incidence_is_the_same_when_the_triangle_winding_is_reflected() -> None:
    """반사 정렬(det<0)은 감김을 뒤집어 삼각형 법선 부호를 바꾼다. |s·n|이라 입사각은 같다."""
    straight = orientation.pass_angles(_rows(), _plane())["gun_incidence_deg"][0]
    reflected = orientation.pass_angles(_rows(), _plane(flip=True))["gun_incidence_deg"][0]
    assert straight == pytest.approx(reflected, abs=1e-9)
    assert straight < 90.0


def test_incidence_is_null_without_a_verified_surface() -> None:
    """메시가 없으면 스트로크 기저 법선으로 대체하지 않고 null과 사유를 낸다."""
    block = orientation.orientation_block(_rows(), _rows(), None)
    incidence = block["items"][0]
    assert incidence["key"] == "gun_incidence_deg"
    assert incidence["query"] is None and incidence["reference"] is None
    assert incidence["reason"] == "no_verified_surface"
    assert incidence["reason_text"] == orientation.REASON_TEXT["no_verified_surface"]
    assert incidence["display"] == {"query": "—", "reference": "—"}
    assert block["items"][1]["query"] is not None


def test_orientation_block_contract() -> None:
    block = orientation.orientation_block(_rows(), _rows(n=6, angles=(70.0, 0.0, 0.0)), _plane())
    assert [item["key"] for item in block["items"]] == ["gun_incidence_deg", "gun_pitch_deg"]
    pitch = block["items"][1]
    assert pitch["display"]["query"] == f"{pitch['query']:.0f}°"
    assert pitch["display"]["reference"] == f"{pitch['reference']:.0f}°"
    assert pitch["samples"] == {"query": 12, "reference": 6}
    for item in block["items"]:
        assert "difference" not in item
    for key in ("title", "header_note", "weight_note", "joint_angles_note", "excluded_note"):
        assert block[key]
    blob = json.dumps(block, ensure_ascii=False)
    for word in ("hmd_pitch", "gaze", "미사용", "사용하지 않음", "unused", "not used"):
        assert word not in blob


def test_missing_gun_rotation_is_reported() -> None:
    rows = _rows()
    for row in rows:
        for axis in "XYZ":
            del row[f"한길숏건 Rotation {axis}"]
    angles = orientation.pass_angles(rows, _plane())
    assert angles["gun_pitch_deg"] == (None, 0, "no_rotation")


def test_real_level2_incidence_is_null_and_level3_has_both_angles() -> None:
    source, kwargs = isolation._source_for({"name": "real"})  # noqa: SLF001
    level2 = service.compare(
        source,
        PassRef(worker="김경호", date="20250831", level="2", session="1000390_1", pass_index=2),
        PassRef(worker="전형기", date="20250901", level="2", session="1000443_1", pass_index=8),
        **kwargs,
    )
    assert level2["status"] == "ok"
    items = level2["measured"]["orientation"]["items"]
    assert items[0]["query"] is None and items[0]["reason"] == "no_verified_surface"
    assert items[0]["reason_text"]

    case = next(c for c in isolation._cases() if c["name"] == "real_L3_kkh_1000391_1_0")  # noqa: SLF001
    level3 = isolation._run(case, include_view=True)  # noqa: SLF001
    items = level3["measured"]["orientation"]["items"]
    assert len(items) == 2
    assert all(isinstance(item["query"], float) and isinstance(item["reference"], float) for item in items)
    assert all(item["display"]["query"] == f"{item['query']:.0f}°".replace("-", "\u2212") for item in items)
    assert all("-" not in item["display"]["query"] + item["display"]["reference"] for item in items)
    joint_axes = level3["measured"]["joint_axes"]
    head = joint_axes["rows"][0]
    assert head["key"] == "head_hmd_heading" and isinstance(head["query"], float) and isinstance(head["reference"], float)
    assert np.isfinite([item["query"] for item in items]).all()


def test_negative_degrees_use_the_same_minus_as_correction_figures() -> None:
    """음수 각은 U+2212로 낸다(교정 figure와 같은 부호). ASCII '-'가 섞이면 한 화면에 두 마이너스가 보인다."""
    assert orientation._deg_display(-33.4) == "\u221233°"  # noqa: SLF001
    assert orientation._deg_display(33.4) == "33°"  # noqa: SLF001
    assert orientation._deg_display(-0.2) == "0°"  # noqa: SLF001
    assert orientation._deg_display(None) == "—"  # noqa: SLF001
    down = orientation.orientation_block(_rows(angles=(0.0, 0.0, 120.0)), _rows(), None)
    pitch = down["items"][1]
    assert pitch["query"] < 0 and pitch["display"]["query"].startswith("\u2212")


def _heading_rows(yaw_offset_deg: float) -> list[dict[str, str]]:
    """머리→착탄 수평 방위에서 `yaw_offset_deg`만큼 돌린 HMD. HMD 회전 (0, yaw, 0) → 전방 +Z를 Y축 둘레로 돌린다."""
    rows = _rows()
    for row in rows:
        head = [float(row[f"카메라 Position {a}"]) for a in "XYZ"]
        hit = [float(row[k]) for k in ("Ray point X", " Ray Point Y", " Ray Point Z")]
        base = math.degrees(math.atan2(hit[0] - head[0], hit[2] - head[2]))
        for axis, value in zip("XYZ", (25.0, base + yaw_offset_deg, 0.0)):
            row[f"카메라 Rotation {axis}"] = str(value)
    return rows


def test_hmd_heading_to_hit_is_the_horizontal_angle_only() -> None:
    """HMD 앙각(Rotation X=25°)은 결과를 바꾸지 않고, 수평 방위 차만 잰다. 부호는 없다."""
    for offset in (0.0, 30.0, -30.0):
        value, n, reason = orientation.hmd_heading_to_hit_deg(_heading_rows(offset))
        assert reason is None and n == 12
        assert value == pytest.approx(abs(offset), abs=1e-6)


def test_joint_axes_block_measures_only_tracked_bodies() -> None:
    """측정 없는 관절은 값 없이 사유만 싣는다. 값이 있는 행은 실측 강체(HMD) 하나다."""
    block = orientation.joint_axes_block(_heading_rows(20.0), _heading_rows(5.0))
    rows = {row["key"]: row for row in block["rows"]}
    assert [row["key"] for row in block["rows"]] == [spec["key"] for spec in orientation.JOINT_AXIS_ROWS]
    head = rows["head_hmd_heading"]
    assert head["display"] == {"query": "20°", "reference": "5°"} and head["status"] == "measured"
    for key, row in rows.items():
        if key == "head_hmd_heading":
            continue
        assert row["query"] is None and row["reference"] is None and row["display"] is None
        assert row["reason_text"] and row["status_text"] == orientation.STATUS_TEXT[row["status"]]
    blob = json.dumps(block, ensure_ascii=False)
    for word in ("미사용", "사용하지 않음", "unused", "not used", "늘 0°"):
        assert word not in blob
    assert "늘 0°" not in orientation.JOINT_ANGLES_NOTE



def test_joint_axes_summary_counts_every_row_and_values_only_measured_ones() -> None:
    """첫 화면 요약(F2)은 전체 표와 같은 문자열만 쓰고, 값은 실측 강체가 붙은 관절에만 붙는다.

    요약이 전체 표와 어긋나면 첫 화면이 아래 표와 다른 말을 한다. 그래서 (1) 상태 수의 합이 행 수와
    같고, (2) 요약 조각(joint·status_text·값)이 전부 해당 행에서 온 것인지 본다.
    """
    block = orientation.joint_axes_block(_heading_rows(20.0), _heading_rows(5.0))
    rows = {row["key"]: row for row in block["rows"]}
    summary = block["summary"]
    assert summary["title"] == orientation.JOINT_AXES_SUMMARY_TITLE
    assert sum(summary["counts"].values()) == len(block["rows"])
    assert summary["counts"] == {"measured": 2, "not_evaluated": 1, "not_measured": 3}
    assert summary["counts_text"] == "관절 6곳 — 실측 2 · 평가 안 함 1 · 측정 없음 3"
    # 값이 붙는 칩은 실측 강체 관절뿐이고, 문자열은 전부 같은 행에서 온 것이다.
    assert [chip["key"] for chip in summary["chips"]] == ["head_hmd_heading", "right_gun_axis"]
    for chip in summary["chips"]:
        row = rows[chip["key"]]
        assert chip["joint"] == row["joint"] and chip["status_text"] == row["status_text"]
        assert chip["status"] in orientation.MEASURED_STATUSES
    head, gun = summary["chips"]
    assert head["value"] == "대상 20° · 기준 5°"
    assert gun["value"] is None


def test_joint_axes_summary_follows_the_rows_it_summarises() -> None:
    """요약이 행에서 나온다는 것을 확인한다 — 행을 바꾸면 요약도 바뀌어야 한다(고정 문자열이 아님)."""
    rows = orientation.joint_axes_block(_heading_rows(20.0), _heading_rows(5.0))["rows"]
    trimmed = [row for row in rows if row["status"] != "not_measured"]
    summary = orientation._joint_axes_summary(trimmed)  # noqa: SLF001
    assert summary["counts"] == {"measured": 2, "not_evaluated": 1, "not_measured": 0}
    assert summary["counts_text"] == "관절 3곳 — 실측 2 · 평가 안 함 1 · 측정 없음 0"
    # 실측 행에서 display를 떼면 값 칩도 사라진다.
    stripped = [dict(row, display=None) if row["key"] == "head_hmd_heading" else row for row in rows]
    assert orientation._joint_axes_summary(stripped)["chips"][0]["value"] is None  # noqa: SLF001
