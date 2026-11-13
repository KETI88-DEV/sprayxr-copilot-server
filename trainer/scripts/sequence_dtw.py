#!/usr/bin/env python3
"""착탄 경로로 두 시퀀스를 정렬한다 — 무엇을 칠했는지로 맞추고, 어떻게 섰는지를 견준다.

## 왜 착탄으로 정렬하나

정렬 신호와 채점 대상이 같으면 순환이 된다. **착탄 경로로 맞추고 자세를 잰다.**
착탄은 과제 공간이라 "스트로크의 몇 %를 칠했을 때"라는 공통 시계를 준다.

경로는 **시작점을 뺀 변위**로 쓴다. 벽 어디를 칠했든 같은 모양의 스트로크가 같은 경로가
되어야 하기 때문이다(평행이동 불변). 회전은 빼지 않는다 — 위로 칠하는 스트로크와
옆으로 칠하는 스트로크는 다른 동작이고, 그 차이는 버킷이 아니라 정렬이 구분해야 한다.

## 워핑 폭을 제한한다 — 이게 없으면 DTW가 진짜 차이를 지운다

제약 없는 DTW는 중간에 멈칫한 작업자를 안 멈춘 작업자에 맞춰 버린다. **그 멈칫이
교정 대상일 수 있다.** Sakoe-Chiba 밴드로 대각선에서 `BAND_RATIO`만큼만 벗어나게 묶고,
**실제로 얼마나 휘었는지를 `warp`로 내보낸다.** 지워지지 않고 보고된다.

## 길이는 먼저 솎고 정렬은 DTW가 한다

비용을 묶으려고 프레임을 `MAX_POINTS`개로 솎는데, **시간 인덱스 균등**으로 솎는다.
호장 균등으로 솎으면 그 순간 대응이 정해져 버려서 DTW가 할 일이 없어진다 —
솎기는 해상도를 줄이는 일이고 정렬은 DTW의 일이다.

## 비용

솎은 뒤 `MAX_POINTS²`칸이다(기본 24² = 576). 런타임은 버킷 프로토타입 **하나**와만
정렬하므로 시퀀스당 DTW 1회다.
"""

from __future__ import annotations

import math
from typing import Final, Sequence

Vec3 = tuple[float, float, float]

# 솎은 뒤의 최대 점 수. DTW 비용이 이 값의 제곱이다.
# 분사 구간 실측이 p50 0.96초·115프레임(120FPS)이라 24점이면 40 ms 간격이다.
MAX_POINTS: Final = 24

# Sakoe-Chiba 밴드 폭(대각선 대비 비율). 0.25면 24점에서 ±6칸까지만 휜다.
# 제약이 없으면 DTW가 멈칫·정지 같은 실제 차이를 워핑으로 지운다.
BAND_RATIO: Final = 0.25

# 경로가 이보다 짧으면 방향이 정의되지 않는다 — 정렬할 모양이 없다.
MIN_PATH_LENGTH_M: Final = 0.02

# DBA 반복 횟수. 실측으로 3회면 이동량이 1 mm 아래로 떨어진다(`_burstvar` 계열 측정).
DBA_ITERATIONS: Final = 5


def decimate(points: Sequence[Vec3], limit: int = MAX_POINTS) -> list[int]:
    """시간 인덱스 균등으로 솎은 **인덱스**를 낸다. 값이 아니라 인덱스인 이유는
    같은 솎기를 자세 쪽에도 그대로 적용해야 하기 때문이다."""
    count = len(points)
    if count <= limit:
        return list(range(count))
    return [round(i * (count - 1) / (limit - 1)) for i in range(limit)]


def path_shape(points: Sequence[Vec3]) -> list[Vec3] | None:
    """착탄 경로를 시작점 기준 변위로. 너무 짧으면 `None`(정렬할 모양이 없다)."""
    if len(points) < 2:
        return None
    origin = points[0]
    shifted = [(p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]) for p in points]
    span = max(math.dist(a, origin) for a in points)
    return shifted if span >= MIN_PATH_LENGTH_M else None


def _cost(a: Vec3, b: Vec3) -> float:
    return math.dist(a, b)


def align(
    query: Sequence[Vec3], prototype: Sequence[Vec3], band_ratio: float = BAND_RATIO
) -> tuple[list[tuple[int, int]], float]:
    """DTW 정렬 경로와 평균 비용. 밴드 밖은 보지 않는다.

    돌려주는 경로는 `(query_index, prototype_index)` 쌍의 목록이고, 시작·끝이 고정된
    단조 경로다(`(0,0)`에서 `(n-1,m-1)`까지).
    """
    n, m = len(query), len(prototype)
    if n == 0 or m == 0:
        return [], math.inf
    band = max(1, int(round(band_ratio * max(n, m))))
    inf = math.inf
    # (n+1)×(m+1) 누적표. 밴드 밖은 inf로 둔다.
    acc = [[inf] * (m + 1) for _ in range(n + 1)]
    acc[0][0] = 0.0
    for i in range(1, n + 1):
        # i번째 query가 대응할 수 있는 prototype 범위. 대각선에서 band만큼.
        centre = (i - 1) * (m - 1) / (n - 1) if n > 1 else 0.0
        low = max(1, int(math.floor(centre - band)) + 1)
        high = min(m, int(math.ceil(centre + band)) + 1)
        for j in range(low, high + 1):
            best = min(acc[i - 1][j - 1], acc[i - 1][j], acc[i][j - 1])
            if best == inf:
                continue
            acc[i][j] = best + _cost(query[i - 1], prototype[j - 1])
    if acc[n][m] == inf:
        return [], math.inf

    # 역추적
    path: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = min(acc[i - 1][j - 1], acc[i - 1][j], acc[i][j - 1])
        if step == acc[i - 1][j - 1]:
            i, j = i - 1, j - 1
        elif step == acc[i - 1][j]:
            i -= 1
        else:
            j -= 1
    path.reverse()
    return path, acc[n][m] / len(path)


def warp_amount(path: Sequence[tuple[int, int]], n: int, m: int) -> float:
    """정렬이 대각선에서 얼마나 벗어났나(0이면 균등 대응).

    **지워지지 않고 보고되는 값이다.** 크면 두 시퀀스의 속도 배분이 다르다는 뜻이고,
    그 자체가 교정거리일 수 있다.
    """
    if not path or n < 2 or m < 2:
        return 0.0
    total = 0.0
    for qi, pi in path:
        expected = qi * (m - 1) / (n - 1)
        total += abs(pi - expected) / (m - 1)
    return total / len(path)


def phase_means(
    path: Sequence[tuple[int, int]], values: Sequence[Sequence[float]], phases: int
) -> list[list[float] | None]:
    """정렬 경로를 따라 위상마다 query 값의 평균. 대응이 없는 위상은 `None`.

    한 위상에 query 프레임이 여럿 붙으면 평균한다 — 그 구간을 천천히 지난 것이므로
    그 동안의 자세를 대표해야 한다.
    """
    buckets: list[list[Sequence[float]]] = [[] for _ in range(phases)]
    for query_index, phase in path:
        if 0 <= phase < phases and query_index < len(values):
            buckets[phase].append(values[query_index])
    out: list[list[float] | None] = []
    for slot in buckets:
        if not slot:
            out.append(None)
            continue
        width = len(slot[0])
        out.append([sum(item[axis] for item in slot) / len(slot) for axis in range(width)])
    return out


def barycenter(
    series: Sequence[Sequence[Vec3]], phases: int, iterations: int = DBA_ITERATIONS
) -> list[Vec3] | None:
    """DBA — DTW 평균 시퀀스. 초기값은 **메도이드**(실재하는 시퀀스)다.

    **무게중심은 아무도 하지 않은 동작일 수 있다.** 그래서 실재하는 시퀀스에서 출발하고,
    호출자가 `barycenter_support`로 그것이 실제 시퀀스들 근처에 있는지 확인해야 한다.
    """
    usable = [s for s in series if len(s) >= 2]
    if not usable:
        return None
    medoid = _medoid(usable)
    current = _resample_linear(medoid, phases)
    for _ in range(iterations):
        sums = [[0.0, 0.0, 0.0] for _ in range(phases)]
        counts = [0] * phases
        for item in usable:
            path, _ = align(item, current)
            for query_index, phase in path:
                for axis in range(3):
                    sums[phase][axis] += item[query_index][axis]
                counts[phase] += 1
        updated: list[Vec3] = []
        for phase in range(phases):
            if counts[phase] == 0:
                updated.append(current[phase])
            else:
                updated.append(
                    (
                        sums[phase][0] / counts[phase],
                        sums[phase][1] / counts[phase],
                        sums[phase][2] / counts[phase],
                    )
                )
        moved = max(math.dist(a, b) for a, b in zip(current, updated))
        current = updated
        if moved < 1e-4:
            break
    return current


def barycenter_support(
    prototype: Sequence[Vec3], series: Sequence[Sequence[Vec3]]
) -> tuple[float, float]:
    """프로토타입이 실제 시퀀스들 근처에 있는가 — (프로토타입까지 거리 p50, 서로 간 거리 p50).

    앞이 뒤보다 크면 **무게중심이 아무도 하지 않은 동작**이라는 신호다.
    """
    usable = [s for s in series if len(s) >= 2]
    if len(usable) < 2:
        return (math.inf, math.inf)
    to_prototype = sorted(align(item, prototype)[1] for item in usable)
    pairs = []
    for index in range(0, len(usable) - 1, 2):
        pairs.append(align(usable[index], usable[index + 1])[1])
    pairs.sort()
    return (to_prototype[len(to_prototype) // 2], pairs[len(pairs) // 2])


def _medoid(series: Sequence[Sequence[Vec3]]) -> Sequence[Vec3]:
    """서로에게 가장 가까운 **실재하는** 시퀀스. 표본이 많으면 앞쪽 일부로만 고른다."""
    sample = list(series[:: max(1, len(series) // 24)])[:24]
    best, best_cost = sample[0], math.inf
    for candidate in sample:
        total = sum(align(other, candidate)[1] for other in sample)
        if total < best_cost:
            best, best_cost = candidate, total
    return best


def _resample_linear(points: Sequence[Vec3], count: int) -> list[Vec3]:
    """인덱스 균등 리샘플. 프로토타입의 길이를 고정하기 위한 것이다."""
    if len(points) == count:
        return list(points)
    if len(points) == 1:
        return [points[0]] * count
    out: list[Vec3] = []
    for index in range(count):
        position = index * (len(points) - 1) / (count - 1)
        low = int(math.floor(position))
        high = min(low + 1, len(points) - 1)
        ratio = position - low
        out.append(
            tuple(  # type: ignore[arg-type]
                points[low][axis] * (1 - ratio) + points[high][axis] * ratio
                for axis in range(3)
            )
        )
    return out
