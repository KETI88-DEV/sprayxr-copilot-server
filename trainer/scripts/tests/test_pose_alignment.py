#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pose_alignment = importlib.import_module("pose_alignment")


def test_dtw_path_recovers_a_known_stretch() -> None:
    """b는 a를 늘린 신호다. 정렬이 맞으면 대응쌍의 값이 정확히 같다."""
    a = [0.0, 1.0, 2.0, 3.0, 4.0]
    b = [0.0, 0.0, 1.0, 2.0, 2.0, 3.0, 4.0, 4.0]

    path = pose_alignment.dtw_path(a, b)

    assert path[0] == (0, 0)
    assert path[-1] == (len(a) - 1, len(b) - 1)
    assert all(a[i] == b[j] for i, j in path)


def test_align_session_maps_every_csv_row_to_one_bvh_frame() -> None:
    """빌더는 패스 구간(CSV 행 범위)으로 BVH 프레임을 뽑는다. 행마다 프레임이 있어야 한다."""
    bvh_signal = [0.0, 1.0, 2.0, 3.0, 4.0]
    csv_signal = [0.0, 0.0, 1.0, 2.0, 2.0, 3.0, 4.0, 4.0]

    alignment = pose_alignment.align_session(bvh_signal, csv_signal)

    assert len(alignment.frame_for_row) == len(csv_signal)
    assert alignment.frame_for_row[0] == 0
    assert alignment.frame_for_row[-1] == len(bvh_signal) - 1
    assert all(
        alignment.frame_for_row[i] <= alignment.frame_for_row[i + 1]
        for i in range(len(csv_signal) - 1)
    )


def test_align_session_reports_zero_residual_for_exact_stretch() -> None:
    bvh_signal = [0.0, 1.0, 2.0, 3.0, 4.0]
    csv_signal = [0.0, 0.0, 1.0, 2.0, 2.0, 3.0, 4.0, 4.0]

    alignment = pose_alignment.align_session(bvh_signal, csv_signal)

    assert alignment.residual == 0.0


def test_align_session_residual_grows_when_signals_disagree() -> None:
    bvh_signal = [0.0, 1.0, 2.0, 3.0, 4.0]
    csv_signal = [10.0, 11.0, 12.0, 13.0, 14.0]

    alignment = pose_alignment.align_session(bvh_signal, csv_signal)

    assert alignment.residual == 10.0


def test_co_indexed_alignment_is_the_identity_mapping() -> None:
    """런타임은 자세와 CSV가 같은 프레임으로 들어온다. 대응이 주어지므로 DTW를 쓰지 않는다."""
    alignment = pose_alignment.co_indexed_alignment(5)

    assert alignment.frame_for_row == (0, 1, 2, 3, 4)
    assert alignment.residual == 0.0


def test_co_indexed_alignment_handles_an_empty_session() -> None:
    alignment = pose_alignment.co_indexed_alignment(0)

    assert alignment.frame_for_row == ()
    assert alignment.residual == 0.0


def test_cross_validate_confirms_a_co_indexed_mapping_perfectly() -> None:
    """동일 인덱스 대응은 교차검증에서 r=1.0이 나온다. 게이트를 우회하는 게 아니라 실제로 통과한다."""
    signal = [0.0, 1.0, 4.0, 9.0, 16.0]
    alignment = pose_alignment.co_indexed_alignment(len(signal))

    score = pose_alignment.cross_validate(alignment.frame_for_row, signal, signal)

    assert score > 0.9999


def test_cross_validate_is_high_when_an_independent_signal_also_lines_up() -> None:
    """정합에 쓰지 않은 신호가 함께 맞아야 대응이 진짜다. 정렬 신호 자체의 잔차로는 알 수 없다."""
    bvh_check = [0.0, 1.0, 2.0, 3.0, 4.0]
    csv_check = [0.0, 0.0, 1.0, 2.0, 2.0, 3.0, 4.0, 4.0]
    frame_for_row = [0, 0, 1, 2, 2, 3, 4, 4]

    score = pose_alignment.cross_validate(frame_for_row, bvh_check, csv_check)

    assert score > 0.99


def test_cross_validate_is_near_zero_for_an_unrelated_mapping() -> None:
    bvh_check = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    csv_check = [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    frame_for_row = list(range(8))

    score = pose_alignment.cross_validate(frame_for_row, bvh_check, csv_check)

    assert score < 0.0


def test_cross_validate_returns_zero_for_a_flat_signal() -> None:
    """상수 신호는 정합을 검증할 수 없다. 높은 점수를 주면 안 된다."""
    score = pose_alignment.cross_validate([0, 1, 2], [1.0, 1.0, 1.0], [5.0, 5.0, 5.0])

    assert score == 0.0


def test_dtw_path_is_monotonic() -> None:
    a = [0.0, 3.0, 1.0, 4.0, 2.0]
    b = [0.0, 3.0, 3.0, 1.0, 4.0, 4.0, 2.0]

    path = pose_alignment.dtw_path(a, b)

    assert all(
        path[k + 1][0] >= path[k][0] and path[k + 1][1] >= path[k][1]
        for k in range(len(path) - 1)
    )
