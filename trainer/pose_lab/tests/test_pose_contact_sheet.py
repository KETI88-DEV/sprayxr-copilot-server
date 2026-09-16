#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false
"""컨택트시트 감사 도구(`trainer/scripts/pose_contact_sheet.py`, D5).

감사 도구의 값어치는 **무엇이 이상한지 짚어 준다**는 데 있다. 그래서 여기서 고정하는 것은
(1) 결정성 — 같은 입력이면 바이트가 같다, (2) 프레임 선택이 패스 전체를 훑는다,
(3) 칸마다 시각·지표가 실제로 글자로 들어간다, (4) 배치에 실패한 프레임을 조용히 **버리지 않고**
사유를 적는다, (5) 칸마다 축척이 같다 — 다르면 프레임끼리 비교할 수 없다.
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

pose_contact_sheet = importlib.import_module("pose_contact_sheet")

from trainer.pose_lab.service import PassRef, RawPass  # noqa: E402
from trainer.pose_lab.tests import test_scoring_isolation as isolation  # noqa: E402
from trainer.pose_lab.tests.test_service_view import (  # noqa: E402
    _rows_with_leg_poses,
    test_mounts,  # noqa: F401 — fixture
)

KKH = PassRef(worker="김경호", date="20250831", level="3", session="1000391_1", pass_index=0)


def test_pick_frames_spans_the_pass_and_is_deterministic() -> None:
    assert pose_contact_sheet.pick_frames(38, 12) == pose_contact_sheet.pick_frames(38, 12)
    picks = pose_contact_sheet.pick_frames(38, 12)
    assert picks[0] == 0 and picks[-1] == 37  # 첫 프레임과 끝 프레임을 다 본다
    assert picks == sorted(picks) and len(set(picks)) == len(picks)
    assert pose_contact_sheet.pick_frames(5, 12) == [0, 1, 2, 3, 4]
    assert pose_contact_sheet.pick_frames(0, 12) == []


def test_sheet_is_byte_identical_on_the_same_pass() -> None:
    first = pose_contact_sheet.sheet_for(isolation.DATA_ROOT, KKH, frames=6)
    second = pose_contact_sheet.sheet_for(isolation.DATA_ROOT, KKH, frames=6)
    assert first == second
    assert first.startswith("<svg") and first.rstrip().endswith("</svg>")


def test_every_cell_carries_its_time_and_the_metrics_that_name_the_odd_frame() -> None:
    svg = pose_contact_sheet.sheet_for(isolation.DATA_ROOT, KKH, frames=6)
    assert len(re.findall(r"#\d+ · \d+\.\d\ds", svg)) == 6
    for label in ("앵커 RMS", "다리 기울기", "오른 팔꿈치", "왼 팔꿈치", "IK 다리", "자리표시 다리"):
        assert svg.count(label) == 6, label
    # 시트가 스스로 표시 전용임을 밝힌다.
    assert "표시 전용(채점 아님)" in svg


def _stick_cell(index: int, length: float) -> "pose_contact_sheet.Cell":
    """가슴–머리 뼈 하나짜리 칸. 길이만 다르게 둬 축척이 공유되는지 가른다."""
    return pose_contact_sheet.Cell(
        index=index,
        time_sec=float(index),
        joints={"Chest": (0.0, 0.0), "Head": (0.0, length)},
        placeholder=None,
        markers=[],
        metrics=[],
        reason=None,
    )


def _first_bone_pixel_length(svg: str) -> float:
    match = re.search(
        r'<line x1="([-\d.]+)" y1="([-\d.]+)" x2="([-\d.]+)" y2="([-\d.]+)" stroke="'
        + re.escape(pose_contact_sheet.BONE),
        svg,
    )
    assert match, svg[:400]
    x1, y1, x2, y2 = (float(v) for v in match.groups())
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def test_cells_use_one_scale_so_frames_can_be_compared() -> None:
    """축척은 **모든 칸**의 경계에서 나와야 한다. 첫 칸(또는 칸마다)에서 뽑으면 같은 자세가 시트마다
    다른 크기로 그려져 프레임끼리 비교할 수 없다 — 작은 칸 옆에 큰 칸을 두면 작은 칸이 줄어야 한다."""
    alone = pose_contact_sheet.render_svg([_stick_cell(0, 0.4)], {}, "제목", columns=1)
    with_big = pose_contact_sheet.render_svg(
        [_stick_cell(0, 0.4), _stick_cell(1, 1.6)], {}, "제목", columns=1
    )
    small_alone = _first_bone_pixel_length(alone)
    small_beside_big = _first_bone_pixel_length(with_big)
    assert small_beside_big == pytest.approx(small_alone / 4.0, rel=1e-6), (small_alone, small_beside_big)


def test_real_cells_keep_rig_bone_lengths_across_frames() -> None:
    raw = isolation._source_for({"name": "real"})[0].load(KKH)  # noqa: SLF001
    cells, _summary = pose_contact_sheet.build_cells(raw, frames=6)
    assert len(cells) == 6 and all(cell.joints is not None for cell in cells)
    spans = [
        max(y for _x, y in cell.joints.values()) - min(y for _x, y in cell.joints.values())
        for cell in cells
    ]
    assert max(spans) / min(spans) < 2.0, spans


def test_failed_frames_are_shown_with_a_reason_not_dropped(monkeypatch, test_mounts) -> None:  # noqa: ANN001
    """세우지 못한 프레임을 조용히 빼면 시트가 '전 구간 정상'이라고 거짓말한다."""
    rows, _middle = _rows_with_leg_poses(21, ())
    raw = RawPass(rows=rows)
    ok_cells, _summary = pose_contact_sheet.build_cells(raw, frames=5, mounts_table=test_mounts)
    assert len(ok_cells) == 5 and all(cell.joints is not None for cell in ok_cells)

    calls = {"n": 0}
    real = pose_contact_sheet.skeleton.place_oriented

    def flaky(*args, **kwargs):  # noqa: ANN002, ANN003
        calls["n"] += 1
        return None if calls["n"] == 2 else real(*args, **kwargs)

    monkeypatch.setattr(pose_contact_sheet.skeleton, "place_oriented", flaky)
    cells, _summary = pose_contact_sheet.build_cells(raw, frames=5, mounts_table=test_mounts)
    assert len(cells) == 5  # 버리지 않는다
    broken = [cell for cell in cells if cell.joints is None]
    assert len(broken) == 1 and broken[0].reason
    svg = pose_contact_sheet.render_svg(cells, {}, "제목")
    assert broken[0].reason in svg


def test_placeholder_legs_follow_the_floor_when_the_mesh_gives_one(test_mounts) -> None:  # noqa: ANN001
    """시트가 화면과 같은 다리를 그린다(D3) — 바닥을 주면 배율이 1.0이 아니고 시트가 그렇게 적는다."""
    rows, _middle = _rows_with_leg_poses(21, ())
    raw = RawPass(rows=rows)
    loose, _s = pose_contact_sheet.build_cells(raw, frames=3, mounts_table=test_mounts)
    assert all(("자리표시 다리", "리그 길이", False) in cell.metrics for cell in loose)
    feet = min(y for cell in loose for name, (_x, y) in cell.joints.items() if name.endswith("Foot"))
    snapped, _s2 = pose_contact_sheet.build_cells(
        raw, frames=3, mounts_table=test_mounts, floor=feet + 0.4
    )
    labels = [dict((name, value) for name, value, _loud in cell.metrics)["자리표시 다리"] for cell in snapped]
    assert all(label.startswith("바닥 맞춤 ×") for label in labels), labels


def test_empty_pass_renders_a_reason_instead_of_a_blank_sheet() -> None:
    cells, summary = pose_contact_sheet.build_cells(RawPass(rows=[]), frames=4)
    assert cells == [] and "error" in summary
    svg = pose_contact_sheet.render_svg(cells, summary, "제목")
    assert summary["error"] in svg


def test_real_sheets_for_both_workers_report_the_same_shape() -> None:
    """전형기·김경호 각각 한 패스. 둘 다 12칸이 나오고 같은 지표 이름을 단다."""
    jhg = PassRef(worker="전형기", date="20250901", level="3", session="1000444_1", pass_index=10)
    for ref in (KKH, jhg):
        svg = pose_contact_sheet.sheet_for(isolation.DATA_ROOT, ref, frames=12)
        assert len(re.findall(r"#\d+ · \d+\.\d\ds", svg)) == 12, ref
        assert svg.count("앵커 RMS") == 12
        assert f"{ref.worker} · L{ref.level}" in svg
