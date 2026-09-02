#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import importlib
import math
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

stroke_frame = importlib.import_module("stroke_frame")
measured_pose = importlib.import_module("measured_pose")


def _line(count: int) -> list[tuple[float, float, float]]:
    return [(1.2, 4.0 + i * (1.0 / (count - 1)), 0.8) for i in range(count)]


def test_arclengths_are_normalized() -> None:
    values = stroke_frame.arclengths(_line(5))
    assert values[0] == 0.0
    assert values[-1] == 1.0
    assert all(a < b for a, b in zip(values, values[1:]))


def test_arclength_ignores_dwell_frames() -> None:
    """제자리에 머문 프레임은 호길이를 늘리지 않는다. 시간 위상과 다른 점이다."""
    points = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    assert stroke_frame.arclengths(points) == [0.0, 0.0, 1.0]


def test_resample_returns_requested_count_on_the_path() -> None:
    points = stroke_frame.resample(_line(11), 5)
    assert len(points) == 5
    assert points[0] == (1.2, 4.0, 0.8)
    assert math.isclose(points[-1][1], 5.0, abs_tol=1e-9)
    assert math.isclose(points[2][1], 4.5, abs_tol=1e-6)


def test_same_path_at_different_speed_resamples_identically() -> None:
    """속도만 다른 두 궤적은 호길이 정렬 후 같아야 한다. 시간 정렬이면 어긋난다."""
    slow = stroke_frame.resample(_line(41), 20)
    fast = stroke_frame.resample(_line(11), 20)
    assert len(slow) == 20
    for left, right in zip(slow, fast):
        for axis in range(3):
            assert math.isclose(left[axis], right[axis], abs_tol=1e-6)


def test_stroke_basis_is_orthonormal() -> None:
    basis = stroke_frame.stroke_basis(
        hit=(1.2, 4.5, 0.8), tangent=(0.0, 1.0, 0.0), gun=(0.6, 4.5, 0.8)
    )
    axes = (basis.normal, basis.tangent, basis.up)
    for axis in axes:
        assert math.isclose(math.dist(axis, (0.0, 0.0, 0.0)), 1.0, abs_tol=1e-9)
    for i in range(3):
        for j in range(i + 1, 3):
            dot = sum(axes[i][k] * axes[j][k] for k in range(3))
            assert math.isclose(dot, 0.0, abs_tol=1e-9)


def test_normal_points_from_hit_toward_the_worker() -> None:
    """건 거리를 양수로 읽으려면 법선이 작업자 쪽을 향해야 한다.

    tangent은 월드 Z(유니티 Y-up에서 수평)를 쓴다. tangent=(0,1,0)(월드 Y=위 축)을
    쓰면 forward가 world_axes.UP과 평행해 퇴화 분기(FALLBACK_TANGENT=(1,0,0))로
    떨어지고, 그 분기에서 나오는 normal=cross((0,1,0),(1,0,0))=(0,0,-1)은 X만
    어긋난 이 gun/hit 쌍의 to_worker=(-0.6,0,0)과 내적이 항상 0이라 "작업자 쪽"을
    검증할 수 없다. 그래서 이 못은 비퇴화 경로를 골라야 뜻이 있다.
    """
    hit = (1.2, 4.5, 0.8)
    gun = (0.6, 4.5, 0.8)
    basis = stroke_frame.stroke_basis(hit=hit, tangent=(0.0, 0.0, 1.0), gun=gun)
    to_worker = tuple(gun[axis] - hit[axis] for axis in range(3))
    assert sum(basis.normal[axis] * to_worker[axis] for axis in range(3)) > 0.0


def test_up_axis_is_world_vertical() -> None:
    """수직축이 위를 향해야 "15cm 올리세요"가 참이 된다(유니티 Y-up이므로 Y가 수직).

    tangent=(0,0,1)(월드 Z, 수평)로 forward를 잡으면 up=orthogonalized(world_axes.UP,
    forward)은 이미 forward와 직교한 world_axes.UP=(0,1,0) 그대로다(투영이 0이므로
    빼는 항이 없다) — 그래서 up[1]이 정확히 1.0이어야 한다.
    """
    basis = stroke_frame.stroke_basis(
        hit=(1.2, 4.5, 0.8), tangent=(0.0, 0.0, 1.0), gun=(0.6, 4.5, 0.8)
    )
    assert math.isclose(basis.up[1], 1.0, abs_tol=1e-9)


def test_gun_offsets_are_not_collapsed_onto_the_normal() -> None:
    """법선을 `건-착탄`으로 잡으면 건의 좌표가 정의상 (거리, 0, 0)이 된다.

    그러면 진행 방향과 수직 성분이 구조적으로 0이라 "두 동작이 일치해서 0"과 구별되지
    않는다. 가중 1.0인 오른손이 1차원으로 붕괴하므로 이 성질을 못으로 박는다. 축을
    건 방향으로 정의하는 구현에서는 아래 두 assert가 0.0을 받아 실패한다.

    tangent=(0,0,1)(월드 Z, 수평)이라 forward=(0,0,1), up=orthogonalized(world_axes.UP,
    forward)=(0,1,0)(이미 직교), normal=cross(forward,up)=(-1,0,0)이다(부호는
    `world_axes.sign_toward`가 gun 쪽으로 맞춘다). gun-hit=(-0.6,+0.15,+0.15)이므로
    local[0](법선 성분)=0.6, local[1](진행 성분, tangent=Z)=0.15, local[2](수직 성분,
    up=Y)=0.15 — 두 오프셋이 대칭(+0.15)이라 옛 값과 수치가 같다.
    """
    hit = (1.2, 4.5, 0.8)
    gun = (0.6, 4.65, 0.95)  # 벽에서 0.6 m, 진행 방향(Z) +0.15 m, 수직(Y) +0.15 m
    basis = stroke_frame.stroke_basis(hit=hit, tangent=(0.0, 0.0, 1.0), gun=gun)
    local = stroke_frame.to_local(basis, gun)
    assert math.isclose(local[0], 0.6, abs_tol=1e-9)
    assert math.isclose(local[1], 0.15, abs_tol=1e-9)
    assert math.isclose(local[2], 0.15, abs_tol=1e-9)


def test_basis_stays_orthonormal_for_a_vertical_stroke() -> None:
    """순수 수직 스트로크(접선이 world_axes.UP과 평행)는 위 축 직교화가 영벡터다.
    퇴화 대체축이 있어야 한다.

    tangent=(0,1,0)=world_axes.UP이라 forward∥UP → orthogonalized(UP,forward)=영벡터
    → 대체축 FALLBACK_TANGENT=(1,0,0)로 떨어진다(forward와 이미 직교이므로 그대로
    up=(1,0,0)). normal=cross(forward=(0,1,0), up=(1,0,0))=(0,0,-1). gun을
    hit보다 Z만 −0.1 낮춰 to_worker=(-0.6,0,-0.1)로 두면
    dot(normal,to_worker)=(-1)*(-0.1)=0.1≥0이라 부호가 뒤집히지 않고 normal[2]<0으로
    남는다(예전 코드에서 tangent=(0,0,1)일 때 normal[0]<0이던 것과 같은 자리다 —
    수직축이 Z→Y로 바뀌면서 퇴화 법선이 서는 축도 X→Z로 옮겨간다).
    """
    basis = stroke_frame.stroke_basis(
        hit=(1.2, 4.5, 0.8), tangent=(0.0, 1.0, 0.0), gun=(0.6, 4.5, 0.7)
    )
    axes = (basis.normal, basis.tangent, basis.up)
    for axis in axes:
        assert math.isclose(math.dist(axis, (0.0, 0.0, 0.0)), 1.0, abs_tol=1e-9)
    for i in range(3):
        for j in range(i + 1, 3):
            assert math.isclose(
                sum(axes[i][k] * axes[j][k] for k in range(3)), 0.0, abs_tol=1e-9
            )
    assert basis.normal[2] < 0.0


def test_up_axis_is_horizontal_on_a_vertical_stroke() -> None:
    """이름과 달리 `up`은 수직 스트로크에서 수평이다.

    접선이 이미 world_axes.UP이라 직교화가 영벡터가 되고 대체축(FALLBACK_TANGENT,
    수평)이 들어온다. 독스트링이 이 사실을 말해야 하고, 소비자는 성분을 축 이름이
    아니라 축 벡터로 읽어야 한다. tangent=(0,1,0)일 때 up=(1,0,0)(FALLBACK_TANGENT
    그대로, 이미 forward와 직교), tangent 자체가 곧 forward=(0,1,0)이라 수직 성분이
    1.0, normal=cross(forward,up)=(0,0,-1)이라 수직 성분은 0이다.
    """
    basis = stroke_frame.stroke_basis(
        hit=(1.2, 4.5, 0.8), tangent=(0.0, 1.0, 0.0), gun=(0.6, 4.5, 0.9)
    )
    assert math.isclose(basis.up[1], 0.0, abs_tol=1e-9)
    assert math.isclose(basis.tangent[1], 1.0, abs_tol=1e-9)
    # 법선은 어떤 스트로크에서도 수평이다. 벽 거리 문구가 그 위에 선다.
    assert math.isclose(basis.normal[1], 0.0, abs_tol=1e-9)


def test_basis_survives_a_dwell_frame_with_zero_tangent() -> None:
    """tangent=0이면 FALLBACK_TANGENT=(1,0,0)을 forward로 쓴다. forward와 이미
    직교이므로 up=orthogonalized(world_axes.UP,forward)는 그대로 (0,1,0)이다."""
    basis = stroke_frame.stroke_basis(
        hit=(1.2, 4.5, 0.8), tangent=(0.0, 0.0, 0.0), gun=(0.6, 4.5, 0.8)
    )
    assert basis.tangent == stroke_frame.FALLBACK_TANGENT
    assert math.isclose(basis.up[1], 1.0, abs_tol=1e-9)


def test_to_local_puts_hit_at_origin() -> None:
    basis = stroke_frame.stroke_basis(
        hit=(1.2, 4.5, 0.8), tangent=(0.0, 1.0, 0.0), gun=(0.6, 4.5, 0.8)
    )
    for axis in stroke_frame.to_local(basis, (1.2, 4.5, 0.8)):
        assert math.isclose(axis, 0.0, abs_tol=1e-9)


def test_resample_frames_keeps_three_points_aligned() -> None:
    frames = [
        measured_pose.MeasuredFrame(
            index=i,
            time_sec=i * 0.0111,
            head=(0.5, 4.0 + i * 0.1, 0.8),
            right=(0.85, 4.0 + i * 0.1, 0.65),
            left=(0.67, 4.0 + i * 0.1, 0.79),
            hit=(1.2, 4.0 + i * 0.1, 0.8),
            uv=(1024.0, 512.0),
            spraying=True,
        )
        for i in range(11)
    ]
    resampled = stroke_frame.resample_frames(frames, 5)
    assert len(resampled) == 5
    assert resampled[-1].hit is not None
    assert math.isclose(resampled[-1].hit[1], 5.0, abs_tol=1e-6)
    assert math.isclose(resampled[2].head[1], 4.5, abs_tol=1e-6)


def _frames(count: int) -> list[measured_pose.MeasuredFrame]:
    return [
        measured_pose.MeasuredFrame(
            index=i,
            time_sec=i * 0.0111,
            head=(0.5, 4.0 + i * 0.1, 0.8),
            right=(0.85, 4.0 + i * 0.1, 0.65),
            left=(0.67, 4.0 + i * 0.1, 0.79),
            hit=(1.2, 4.0 + i * 0.1, 0.8),
            uv=(1024.0, 512.0),
            spraying=True,
        )
        for i in range(count)
    ]


def test_one_missing_left_frame_does_not_drop_the_whole_pass() -> None:
    """설계 §5.4는 프레임 단위 마스킹이다. 실측 |머리-왼손| 최소가 0.000 m라
    패스 단위로 버리면 왼손 축이 상시 사라진다."""
    frames = _frames(11)
    frames[5] = replace(frames[5], left=None)
    resampled = stroke_frame.resample_frames(frames, 5)
    assert [frame.left is not None for frame in resampled] == [True] * 5
    assert resampled[2].left is not None
    assert math.isclose(resampled[2].left[1], 4.5, abs_tol=1e-6)


def test_left_is_none_outside_the_observed_left_span() -> None:
    """왼손이 없는 구간을 외삽으로 메우지 않는다."""
    frames = _frames(11)
    frames[0] = replace(frames[0], left=None)
    frames[1] = replace(frames[1], left=None)
    resampled = stroke_frame.resample_frames(frames, 5)
    assert resampled[0].left is None
    assert resampled[-1].left is not None


def test_resampled_yaw_is_interpolated_on_the_circle_not_on_the_number_line() -> None:
    """179°와 −179° 사이는 2°다. 스칼라로 섞으면 0°가 나와 뒤를 본다고 적게 된다."""
    frames = _frames(3)
    angles = (math.radians(179.0), math.radians(180.0), math.radians(-179.0))
    frames = [replace(frame, yaw=angle) for frame, angle in zip(frames, angles)]
    resampled = stroke_frame.resample_frames(frames, 3)
    middle = resampled[1].yaw
    assert middle is not None
    assert abs(abs(middle) - math.pi) < math.radians(1.0)


def test_resampled_yaw_is_none_when_no_frame_carried_one() -> None:
    resampled = stroke_frame.resample_frames(_frames(5), 3)
    assert all(frame.yaw is None for frame in resampled)
