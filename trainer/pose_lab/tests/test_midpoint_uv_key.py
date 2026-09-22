#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false
"""D1(키 접두어 m3): 동작 키의 UV 칸은 **원 프레임 호장 중점**의 UV다.

사전등록 `docs/measurements/2026-09-22-same-level-pairs-d1-prereg.md` §3.1·§5 성질 5.
결함은 `stroke_frame.resample_frames`가 모든 격자점에 첫 프레임 UV를 복사하는 데서 왔다 —
m1은 격자 가운데 점의 UV를 읽었으니 실제로는 시작점 칸이었다(9,571/9,571).

실데이터 검사는 skip하지 않는다 — 데이터가 없으면 실패한다. 중점은 여기서 따로 계산한다
(`motion_key.midpoint_uv`를 부르지 않는다). 같은 함수를 기댓값에 쓰면 실패할 수 없는 단언이 된다.
"""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

measured_pose = importlib.import_module("measured_pose")
motion_key = importlib.import_module("motion_key")
build_spread_reference = importlib.import_module("build_spread_reference")

from trainer.pose_lab import service  # noqa: E402
from trainer.pose_lab.service import PassRef  # noqa: E402

DATA_ROOT = Path("/data/sources/professionals")
PARAMS = motion_key.KeyParams()
# 김경호 20250831의 두 세션과 전형기 한 세션. 화면 예시로 쓰는 패스(L1 1000389_1 #0)가 들어 있다.
SESSIONS = (
    ("김경호", "20250831", "1", "1000389_1"),
    ("김경호", "20250831", "3", "1000391_1"),
    ("전형기", "20250901", "3", "1000444_1"),
)


def _independent_midpoint_uv(frames: list[object]) -> tuple[float, float] | None:
    """누적 호장이 전체의 절반에 가장 가까운 원 프레임의 UV(UV가 없으면 가장 가까운 UV 프레임, 같으면 앞쪽)."""
    hits = [frame.hit for frame in frames]
    cumulative = [0.0]
    for index in range(1, len(hits)):
        cumulative.append(cumulative[-1] + math.dist(hits[index - 1], hits[index]))
    total = cumulative[-1]
    half = total / 2.0
    middle = min(range(len(hits)), key=lambda index: abs(cumulative[index] - half))
    with_uv = [index for index, frame in enumerate(frames) if frame.uv is not None]
    if not with_uv:
        return None
    return frames[min(with_uv, key=lambda index: (abs(index - middle), index))].uv


def _cell(uv: tuple[float, float] | None) -> str:
    return "na" if uv is None else "{}x{}".format(*motion_key.uv_cell(uv, PARAMS))


def _without_prefix_and_cell(key: str) -> list[str]:
    parts = build_spread_reference.key_without_uv(key).split(":")
    return parts[1:]


def _synthetic(index: int) -> object:
    x = 0.1 * index
    return measured_pose.MeasuredFrame(
        index=index, time_sec=0.05 * index, head=(x, 5.0, -0.6), right=(x, 4.4, -0.3), left=None,
        hit=(x, 4.2, 0.0), uv=(100.0 + 800.0 * x, 10.0), spraying=True, yaw=0.3,
    )


def test_synthetic_key_uses_the_midpoint_cell_not_the_start_cell() -> None:
    """x로 2 m 가는 수평 패스. 시작 u=100(0칸), 호장 중점 u=900(3칸)."""
    raw = [_synthetic(index) for index in range(21)]
    frames = service.stroke_frame.resample_frames(raw, service.stroke_frame.DEFAULT_SAMPLES)
    # 결함의 근원은 그대로 있다: 격자점의 UV는 전부 첫 프레임 UV다. 키만 원 프레임에서 읽는다.
    assert {frame.uv for frame in frames} == {raw[0].uv}
    inputs = service.key_inputs(frames, raw_frames=raw)
    assert inputs is not None
    key = service.key_from(inputs, PARAMS)
    assert key.startswith("m3:")
    assert key.split(":")[3] == "3x0", f"키가 중점 칸(3x0)이 아니라 {key.split(':')[3]}를 쓴다"


def test_key_inputs_requires_the_raw_frames() -> None:
    """원 프레임을 빼먹으면 조용히 격자 UV(=첫 프레임)로 떨어지지 않고 호출이 실패한다."""
    raw = [_synthetic(index) for index in range(21)]
    frames = service.stroke_frame.resample_frames(raw, service.stroke_frame.DEFAULT_SAMPLES)
    with pytest.raises(TypeError):
        service.key_inputs(frames)  # type: ignore[call-arg]


def test_real_key_uv_is_the_arclength_midpoint_frame_uv() -> None:
    """실제 CSV: 색인 키와 비교 경로 키의 UV가 둘 다 **원 프레임 호장 중점의 UV**다.

    - 시작 칸과 중점 칸이 다른 패스가 이 세션들에 충분히 있어야 검사가 뜻을 갖는다.
    - 칸 성분과 접두어를 뺀 나머지(띠·방향·좌우·거리·높이)는 시작 UV로 만든 키와 같다(F9·F10).
    """
    if not DATA_ROOT.is_dir():
        pytest.fail(f"실데이터가 없다({DATA_ROOT}) — 이 검사는 skip하지 않는다")
    from trainer.pose_lab import csv_source

    source = csv_source.CsvPassSource(DATA_ROOT)
    checked = 0
    split = 0
    for worker, date, level, session in SESSIONS:
        item = next(
            s for s in source.sessions()
            if (s.worker, s.date, s.level, s.session) == (worker, date, level, session)
        )
        for entry in source.index_of(item):
            ref = PassRef(worker=worker, date=date, level=level, session=session, pass_index=entry.pass_index)
            raw = source.load(ref)
            spray = service.spray_frames(raw)
            expected = _independent_midpoint_uv(spray)
            start = spray[0].uv

            _frames, inputs = service.pass_key_inputs(raw)
            assert inputs is not None
            assert inputs.uv == expected, f"{ref}: 비교 경로의 키 UV {inputs.uv} ≠ 호장 중점 UV {expected}"
            assert entry.key_inputs.uv == expected, (
                f"{ref}: 색인 키 UV {entry.key_inputs.uv} ≠ 호장 중점 UV {expected}"
            )
            key = source.key_of(entry, PARAMS)
            assert key.split(":")[3] == _cell(expected), f"{ref}: 키 칸 {key} ≠ 중점 칸 {_cell(expected)}"

            start_key = motion_key.motion_key(
                displacement=inputs.displacement, yaw=service.yaw_of(inputs), head=inputs.head,
                hit=inputs.hit, uv=start, params=PARAMS,
            )
            assert _without_prefix_and_cell(key) == _without_prefix_and_cell(start_key), (
                f"{ref}: D1이 UV 말고 다른 성분도 바꿨다"
            )
            if _cell(start) == _cell(expected):
                assert key == start_key
            else:
                split += 1
            checked += 1
    assert checked >= 30, checked
    assert split >= 5, f"시작 칸과 중점 칸이 갈린 패스가 {split}개뿐 — 이 검사가 D1을 가려내지 못한다"
