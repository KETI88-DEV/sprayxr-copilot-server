#!/usr/bin/env python3
"""착탄 경로 DTW 정렬 검사.

DTW를 쓰기로 하면서 위험 둘을 적었다. 이 파일이 그 둘을 검사로 묶는다.

1. **워핑이 진짜 차이를 지운다** — 멈칫한 작업자를 안 멈춘 작업자에 맞춰 버린다.
   밴드로 묶고, 휜 양을 `warp_amount`로 내보내 지워지지 않게 한다.
2. **무게중심이 아무도 하지 않은 동작일 수 있다** — `barycenter_support`가
   그것을 잴 수 있어야 한다.
"""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
for path in (str(_REPO_ROOT), str(_SCRIPTS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

sequence_dtw = importlib.import_module("sequence_dtw")


def _line(count: int, length: float = 0.60) -> list[tuple[float, float, float]]:
    """위로 곧게 올라가는 착탄 경로."""
    return [(0.0, length * i / (count - 1), 0.0) for i in range(count)]


def _slow_then_fast(count: int, length: float = 0.60):
    """앞을 천천히, 뒤를 빠르게 — 같은 경로를 다른 속도 배분으로 지난다."""
    return [(0.0, length * (i / (count - 1)) ** 2, 0.0) for i in range(count)]


def test_the_same_path_at_different_lengths_aligns_end_to_end() -> None:
    """길이가 달라도 같은 스트로크면 처음과 끝이 맞아야 한다 — 이것이 '길이 정규화'다.

    비용이 정확히 0은 아니다. 30점과 12점은 같은 직선 위의 **다른 자리**를 밟으므로
    이산화 잔차가 남는다. 남아야 하는 것은 '길이가 다르다'가 아니라 그 잔차뿐이다.
    """
    path, cost = sequence_dtw.align(_line(30), _line(12))
    assert path[0] == (0, 0)
    assert path[-1] == (29, 11)
    # 12점으로 쪼갠 0.60 m 직선의 칸 간격이 0.0545 m다. 잔차는 그 절반 아래여야 한다.
    assert cost < 0.60 / 11 / 2

    # 대조 — 모양이 다르면 비용이 훨씬 크다. 그래야 위 임계가 공허하지 않다.
    bent = [(0.30 * (i / 19) ** 2, 0.60 * i / 19, 0.0) for i in range(20)]
    assert sequence_dtw.align(bent, _line(12))[1] > cost * 5


def test_a_different_speed_profile_shows_up_as_warp_not_as_zero_cost() -> None:
    """속도 배분이 다르면 **휜 양으로 드러나야** 한다.

    DTW가 비용을 0으로 만들며 조용히 삼키면 '멈칫'이 사라진다.

    깨뜨리는 법: `warp_amount`가 늘 0을 내면 실패한다.
    """
    straight = _line(24)
    uneven = _slow_then_fast(24)
    path, _ = sequence_dtw.align(uneven, straight)
    warp = sequence_dtw.warp_amount(path, len(uneven), len(straight))
    assert warp > 0.02, "속도 배분이 다른데 휘지 않았다고 나왔다"

    same, _ = sequence_dtw.align(straight, straight)
    assert sequence_dtw.warp_amount(same, 24, 24) == pytest.approx(0.0, abs=1e-9)


def test_the_band_stops_warping_from_swallowing_everything() -> None:
    """밴드가 없으면 DTW가 앞뒤를 통째로 끌어다 붙인다.

    극단적으로 어긋난 두 경로에서, 밴드가 좁을수록 비용이 **올라가야** 한다 —
    싸게 맞출 방법을 막기 때문이다.

    깨뜨리는 법: `align`이 `band_ratio`를 무시하면 두 비용이 같아져 실패한다.
    """
    a = _line(24)
    b = _slow_then_fast(24)
    wide = sequence_dtw.align(a, b, band_ratio=0.9)[1]
    narrow = sequence_dtw.align(a, b, band_ratio=0.05)[1]
    assert narrow > wide, "밴드를 좁혔는데 비용이 안 올랐다 — 밴드가 안 걸린다"


def test_translation_does_not_change_the_path_shape() -> None:
    """벽 어디를 칠했든 같은 모양의 스트로크는 같은 경로여야 한다."""
    base = _line(20)
    moved = [(p[0] + 3.1, p[1] - 0.7, p[2] + 12.0) for p in base]
    shifted = sequence_dtw.path_shape(moved)
    original = sequence_dtw.path_shape(base)
    assert shifted is not None and original is not None
    for a, b in zip(original, shifted):
        assert a == pytest.approx(b, abs=1e-9)


def test_a_path_that_goes_nowhere_has_no_shape() -> None:
    """건을 고정한 시퀀스에는 정렬할 모양이 없다 — 실물 샘플 SEQ-002가 그렇다.

    `None`을 내야 호출자가 위상 채점을 건너뛴다. 억지로 정렬하면 잡음에 맞추게 된다.
    """
    assert sequence_dtw.path_shape([(1.0, 2.0, 3.0)] * 10) is None
    assert sequence_dtw.path_shape([(1.0, 2.0, 3.0)]) is None
    assert sequence_dtw.path_shape(_line(10, length=0.005)) is None
    assert sequence_dtw.path_shape(_line(10, length=0.60)) is not None


def test_decimation_keeps_the_ends_and_bounds_the_cost() -> None:
    """비용을 묶되 시작과 끝은 남긴다 — 스트로크의 양 끝이 정렬의 고정점이다."""
    indices = sequence_dtw.decimate(_line(500))
    assert len(indices) == sequence_dtw.MAX_POINTS
    assert indices[0] == 0 and indices[-1] == 499
    assert indices == sorted(indices)
    assert sequence_dtw.decimate(_line(7)) == list(range(7)), "짧으면 그대로 둔다"


def test_decimation_is_uniform_in_index_not_in_arc_length() -> None:
    """호장 균등으로 솎으면 그 순간 대응이 정해져 DTW가 할 일이 없어진다.

    가속하는 경로를 솎았을 때 **간격이 고르지 않아야** 호장 균등이 아니다.
    """
    points = _slow_then_fast(100)
    picked = [points[i] for i in sequence_dtw.decimate(points)]
    steps = [math.dist(a, b) for a, b in zip(picked, picked[1:])]
    assert max(steps) > 3 * min(steps), "호장 균등으로 솎았다 — DTW가 정렬할 것이 없어진다"


def test_phase_means_average_the_frames_that_land_on_one_phase() -> None:
    """한 위상에 프레임이 여럿 붙으면 그 구간을 천천히 지난 것이다 — 평균이 대표다."""
    path = [(0, 0), (1, 0), (2, 1)]
    values = [[0.0], [2.0], [10.0]]
    assert sequence_dtw.phase_means(path, values, 2) == [[1.0], [10.0]]


def test_a_phase_with_no_frame_is_none_not_zero() -> None:
    """대응이 없으면 `None`이다. 0으로 채우면 원점에 선 자세를 지어내는 것이 된다."""
    assert sequence_dtw.phase_means([(0, 0)], [[5.0]], 3) == [[5.0], None, None]


def test_the_barycenter_stays_near_the_sequences_it_came_from() -> None:
    """**무게중심이 아무도 하지 않은 동작일 수 있다** — 그것을 잴 수 있어야 한다.

    서로 비슷한 시퀀스들의 무게중심은 그들 사이 거리보다 가까워야 한다.

    깨뜨리는 법: `barycenter`를 상수 시퀀스로 바꾸면 지지도가 무너져 실패한다.
    """
    series = [
        [(0.0, 0.60 * i / 19 + 0.01 * k, 0.0) for i in range(20)]
        for k in range(-3, 4)
    ]
    prototype = sequence_dtw.barycenter(series, phases=20)
    assert prototype is not None
    good, _ = sequence_dtw.barycenter_support(prototype, series)

    # 대조 — 실재하지 않는 프로토타입(정지 시퀀스)은 지지도가 무너져야 한다.
    # 이 대조가 없으면 `barycenter_support`가 늘 작은 값을 내도 검사가 통과한다.
    fake = [(0.0, 0.30, 0.0)] * 20
    bad, _ = sequence_dtw.barycenter_support(fake, series)
    assert bad > good * 5, (
        f"아무도 하지 않은 동작({bad:.4f})이 실제 무게중심({good:.4f})과 구별되지 않는다"
    )


def test_the_barycenter_starts_from_a_real_sequence() -> None:
    """초기값이 **메도이드**다 — 실재하는 시퀀스. 점별 평균에서 출발하면 첫 걸음부터
    아무도 하지 않은 동작이 된다.

    속도 배분이 서로 다른 세 시퀀스를 쓴다. 이들의 점별 평균은 **셋 중 어느 것도 아니다**
    — 평행선을 쓰면 평균이 우연히 실재 시퀀스와 같아져 검사가 공허해진다(실제로 그랬다).

    깨뜨리는 법: `barycenter`의 초기값을 점별 평균으로 바꾸면 실패한다.
    """
    length = 0.60
    count = 20
    series = [
        [(0.0, length * (i / (count - 1)) ** power, 0.0) for i in range(count)]
        for power in (0.5, 1.0, 2.0)
    ]
    mean = [
        tuple(sum(s[i][a] for s in series) / len(series) for a in range(3))
        for i in range(count)
    ]
    assert not any(
        all(x == pytest.approx(y, abs=1e-9) for x, y in zip(mean, item)) for item in series
    ), "점별 평균이 실재 시퀀스와 같다 — 이 검사가 공허하다"

    prototype = sequence_dtw.barycenter(series, phases=count, iterations=0)
    assert prototype is not None
    assert any(
        all(x == pytest.approx(y, abs=1e-9) for x, y in zip(prototype, item)) for item in series
    ), "반복 없이도 실재하지 않는 시퀀스가 나왔다 — 평균에서 출발했다"


def test_a_bent_stroke_is_not_matched_to_a_straight_one() -> None:
    """위로 칠하는 스트로크와 꺾이는 스트로크는 다른 동작이다.

    회전을 지우지 않는 이유다 — 지우면 둘이 같아진다.
    """
    straight = _line(20)
    bent = [(0.30 * max(0.0, i / 19 - 0.5) * 2, 0.60 * i / 19, 0.0) for i in range(20)]
    assert sequence_dtw.align(straight, bent)[1] > 0.02


def test_an_empty_sequence_is_refused_not_guessed() -> None:
    path, cost = sequence_dtw.align([], _line(10))
    assert path == [] and cost == math.inf
    assert sequence_dtw.barycenter([], phases=10) is None
    assert sequence_dtw.warp_amount([], 0, 0) == 0.0
