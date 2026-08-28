#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import importlib
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

rotation_math = importlib.import_module("rotation_math")


# --- 오일러 ↔ 행렬 ↔ 쿼터니언 왕복 --------------------------------------------


def test_every_euler_order_round_trips_through_the_matrix() -> None:
    """규약을 데이터가 고르게 하려면 여섯 순서가 전부 정확해야 한다.

    한 순서만 틀려도 후보 비교가 그 순서에 불리하게 기울고, 이긴 규약이 "가장 잘 구현된
    규약"이 된다.
    """
    random.seed(0)
    for order in rotation_math.EULER_ORDERS:
        for _ in range(50):
            angles = tuple(random.uniform(-180.0, 180.0) for _ in range(3))
            matrix = rotation_math.euler_matrix(order, angles)
            again = rotation_math.euler_matrix(
                order, rotation_math.euler_from_matrix(order, matrix)
            )
            for row in range(3):
                for col in range(3):
                    assert math.isclose(matrix[row][col], again[row][col], abs_tol=1e-9)


def test_matrices_are_orthonormal() -> None:
    matrix = rotation_math.euler_matrix("YXZ", (17.0, -230.0, 91.0))
    product = rotation_math.matmul(matrix, rotation_math.transpose(matrix))
    for row in range(3):
        for col in range(3):
            assert math.isclose(product[row][col], 1.0 if row == col else 0.0, abs_tol=1e-12)


def test_quaternion_round_trips_and_is_sign_fixed() -> None:
    """`q`와 `−q`는 같은 회전이다. 부호를 고정하지 않으면 산포가 통째로 부푼다."""
    random.seed(1)
    for _ in range(200):
        angles = tuple(random.uniform(-180.0, 180.0) for _ in range(3))
        matrix = rotation_math.euler_matrix("YXZ", angles)
        quat = rotation_math.matrix_quaternion(matrix)
        assert quat[0] >= 0.0
        again = rotation_math.quaternion_matrix(quat)
        for row in range(3):
            for col in range(3):
                assert math.isclose(matrix[row][col], again[row][col], abs_tol=1e-9)


# --- 측지 거리 ----------------------------------------------------------------


def test_geodesic_is_the_rotation_angle_and_ignores_the_axis() -> None:
    """측지 각은 축이 어디를 가리키든 같은 회전각을 낸다. 이것이 오일러 축별과 다른 점이다."""
    for axis in "XYZ":
        near = rotation_math.matrix_quaternion(rotation_math.axis_matrix(axis, 0.0))
        far = rotation_math.matrix_quaternion(rotation_math.axis_matrix(axis, 40.0))
        assert math.isclose(rotation_math.geodesic_deg(near, far), 40.0, abs_tol=1e-9)


def test_geodesic_takes_the_short_way_round() -> None:
    """359°와 1°는 2° 떨어져 있다. 산술 차 358°는 wrap 인공물이다."""
    left = rotation_math.matrix_quaternion(rotation_math.axis_matrix("Z", 359.0))
    right = rotation_math.matrix_quaternion(rotation_math.axis_matrix("Z", 1.0))
    assert math.isclose(rotation_math.geodesic_deg(left, right), 2.0, abs_tol=1e-9)


def test_geodesic_never_exceeds_180() -> None:
    left = rotation_math.matrix_quaternion(rotation_math.axis_matrix("X", 0.0))
    right = rotation_math.matrix_quaternion(rotation_math.axis_matrix("X", 270.0))
    assert rotation_math.geodesic_deg(left, right) <= 180.0 + 1e-9


# --- 평균과 산포 --------------------------------------------------------------


def test_mean_of_a_symmetric_set_is_its_centre() -> None:
    quats = [
        rotation_math.matrix_quaternion(rotation_math.axis_matrix("Z", angle))
        for angle in (-20.0, -10.0, 0.0, 10.0, 20.0)
    ]
    mean = rotation_math.quaternion_mean(quats)
    assert mean is not None
    angles = rotation_math.euler_from_matrix("ZXY", rotation_math.quaternion_matrix(mean))
    assert math.isclose(angles[2], 0.0, abs_tol=1e-9)


def test_mean_crosses_the_wrap_without_flying_to_the_far_side() -> None:
    """`dataset_spec.md` §4.1이 기록한 결함의 회전공간판. 산술 평균이면 180°가 나온다."""
    quats = [
        rotation_math.matrix_quaternion(rotation_math.axis_matrix("Z", angle))
        for angle in (359.0, 1.0)
    ]
    mean = rotation_math.quaternion_mean(quats)
    assert mean is not None
    identity = rotation_math.matrix_quaternion(rotation_math.IDENTITY)
    assert rotation_math.geodesic_deg(mean, identity) < 1e-6


def test_spread_reports_mean_and_rms_and_rms_is_never_smaller() -> None:
    quats = [
        rotation_math.matrix_quaternion(rotation_math.axis_matrix("Y", angle))
        for angle in (-30.0, -10.0, 0.0, 10.0, 30.0)
    ]
    spread = rotation_math.geodesic_spread(quats)
    assert spread is not None
    average, rms = spread
    assert math.isclose(average, 16.0, abs_tol=1e-6)
    assert rms >= average


def test_spread_of_identical_rotations_is_zero() -> None:
    quats = [rotation_math.matrix_quaternion(rotation_math.axis_matrix("X", 12.0))] * 8
    spread = rotation_math.geodesic_spread(quats)
    assert spread is not None
    assert spread[0] < 1e-9 and spread[1] < 1e-9


def test_spread_needs_two_samples() -> None:
    single = [rotation_math.matrix_quaternion(rotation_math.IDENTITY)]
    assert rotation_math.geodesic_spread(single) is None
    assert rotation_math.geodesic_spread([]) is None


# --- 기저 ---------------------------------------------------------------------


def test_basis_matrix_transposed_reads_local_components() -> None:
    """스트로크 프레임으로 회전을 옮기는 경로가 이 한 줄 위에 선다."""
    basis = rotation_math.basis_matrix([(0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)])
    local = rotation_math.matvec(rotation_math.transpose(basis), (3.0, 5.0, 7.0))
    assert local == (5.0, 7.0, 3.0)
