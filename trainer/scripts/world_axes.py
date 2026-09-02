#!/usr/bin/env python3
"""월드 축 규약. **여기가 유일한 출처다.**

캡처 CSV의 좌표는 유니티의 것이다 — Y가 위, 왼손 좌표계. `measured_pose`가 컬럼을
재배열 없이 읽으므로 그 규약이 그대로 파이프라인에 들어온다.

상수만 두지 않고 **연산**을 둔다. 이 모듈이 생긴 이유가 규약이 네 곳에 리터럴로
흩어져 있었기 때문이고(`world[2]`, `atan2(f[1], f[0])`, `(v[0], v[1], 0)`),
상수만 모으면 호출자가 그 리터럴을 다시 적을 수 있다. 연산을 쓰게 만들어야 막힌다.
"""

from __future__ import annotations

import math
from typing import Final, NamedTuple

Vec3 = tuple[float, float, float]

UP_INDEX: Final = 1
# UP과 HORIZONTAL_INDICES는 아래에 리터럴로 다시 적지 않는다 — 모듈 `__getattr__`
# (PEP 562)이 `world_axes.UP`, `world_axes.HORIZONTAL_INDICES`로 접근할 때마다
# UP_INDEX 하나에서 다시 계산한다. 손잡이는 UP_INDEX 하나뿐이다: 셋을 각자 리터럴로
# 두면 UP_INDEX만 몽키패치했을 때 나머지 둘이 따라오지 않는다 — 이 모듈이 막으려던
# 결함(규약이 네 곳에 흩어짐)의 축소판이 된다. `_up_vector`/`_horizontal_indices`
# 참고. 모듈 상수로 값을 한 번 복사해 굳히면(가져다 쓰는 쪽에서
# `WORLD_UP = world_axes.UP`처럼) 이후 UP_INDEX가 바뀌어도 그 값은 반응하지 않는다
# — 그래서 이 파일 안에서도 `UP`/`HORIZONTAL_INDICES`라는 이름을 직접 쓰지 않고
# 아래 헬퍼를 호출 시점에 부른다.
# 유니티는 왼손 좌표계다. 외적의 방향 감각이 오른손계와 반대이므로 새 외적을 쓸 때마다
# 부호를 기준점으로 다시 정해야 한다 — `sign_toward`가 그 일을 한다.
#
# 이 상수는 로직이 읽지 않는다. `sign_toward`(그리고 `frame_around`의 부호 미정 셋째
# 축)는 좌표계의 손대칭성과 무관하게 항상 기준점으로 부호를 다시 정하도록 설계됐다 —
# 그래서 "left"를 "right"로 바꿔도 어떤 계산도 달라지지 않는다. 이 값의 역할은
# 유니티가 왼손 좌표계라는 사실 — 그래서 외적의 방향 감각이 오른손 직관과 반대라는
# 사실 — 을 기록해 다음 사람이 그 직관을 믿고 놀라지 않게 하는 것뿐이다. 문서다.
HANDEDNESS: Final = "left"

# 수평 성분이 이보다 짧으면 방위가 정의되지 않는다(정확히 위/아래를 본 경우).
MIN_HORIZONTAL: Final = 1e-9


def _up_vector(up_index: int) -> Vec3:
    """UP_INDEX 하나에서 UP을 만든다. `UP = (0.0, 1.0, 0.0)`처럼 따로 적지 않는다."""
    axes = [0.0, 0.0, 0.0]
    axes[up_index] = 1.0
    return (axes[0], axes[1], axes[2])


def _horizontal_indices(up_index: int) -> tuple[int, int]:
    """UP_INDEX 하나에서 수평 축 인덱스 쌍을 만든다."""
    remaining = tuple(index for index in range(3) if index != up_index)
    return (remaining[0], remaining[1])


def __getattr__(name: str) -> object:
    """`world_axes.UP`, `world_axes.HORIZONTAL_INDICES`를 접근 시점마다 다시 계산한다.

    PEP 562 모듈 훅이다. 값을 임포트 시점에 굳히면 이후 `UP_INDEX`를 몽키패치해도
    이 둘이 따라오지 않는다 — 뒤따르는 태스크가 정확히 이 패턴(UP_INDEX 패치 후
    소비자 출력이 움직이는지 확인)으로 회귀를 잡는다.
    """
    if name == "UP":
        return _up_vector(UP_INDEX)
    if name == "HORIZONTAL_INDICES":
        return _horizontal_indices(UP_INDEX)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def vertical_of(v: Vec3) -> float:
    return v[UP_INDEX]


def horizontal_of(v: Vec3) -> tuple[float, float]:
    a, b = _horizontal_indices(UP_INDEX)
    return (v[a], v[b])


def flatten(v: Vec3) -> Vec3:
    out = list(v)
    out[UP_INDEX] = 0.0
    return (out[0], out[1], out[2])


def yaw_of(forward: Vec3) -> float | None:
    """수평면 방위. 수평 성분이 없으면 None이다 — 0으로 지어내지 않는다.

    `motion_key`가 전방 성분을 `cos(yaw)·h0 + sin(yaw)·h1`로 쓰므로 여기서 쓰는
    수평 축 쌍과 그쪽이 같아야 한다. 어긋나면 몸 기준 축이 조용히 월드 방위각으로
    돌아간다.
    """
    h0, h1 = horizontal_of(forward)
    if math.hypot(h0, h1) < MIN_HORIZONTAL:
        return None
    return math.atan2(h1, h0)


def sign_toward(axis: Vec3, toward: Vec3) -> Vec3:
    """축의 부호를 기준 벡터 쪽으로 맞춘다. 왼손 좌표계에서 외적 결과를 쓸 때 필요하다."""
    dot = axis[0] * toward[0] + axis[1] * toward[1] + axis[2] * toward[2]
    return axis if dot >= 0.0 else (-axis[0], -axis[1], -axis[2])


def _normalize(v: Vec3) -> Vec3:
    length = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if length == 0.0:
        return (0.0, 0.0, 0.0)
    return (v[0] / length, v[1] / length, v[2] / length)


def _dot3(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross3(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _orthogonalized(vector: Vec3, axis: Vec3) -> Vec3:
    """axis 성분을 뺀 뒤 정규화. axis에 평행하면 영벡터가 나온다."""
    projection = _dot3(vector, axis)
    return _normalize(
        (
            vector[0] - projection * axis[0],
            vector[1] - projection * axis[1],
            vector[2] - projection * axis[2],
        )
    )


def _fallback_horizontal(up_index: int) -> Vec3:
    """수평 축 쌍의 첫 축을 단위벡터로 쓴다. UP_INDEX가 바뀌면 이 축도 따라 바뀐다 —
    `(1.0, 0.0, 0.0)` 같은 고정 리터럴을 두지 않는다."""
    first, _ = _horizontal_indices(up_index)
    axes = [0.0, 0.0, 0.0]
    axes[first] = 1.0
    return (axes[0], axes[1], axes[2])


class Frame(NamedTuple):
    """forward를 기준으로 만든 정규직교 셋. `third`는 부호를 정하지 않는다."""

    forward: Vec3
    up: Vec3
    third: Vec3


def frame_around(forward: Vec3) -> Frame:
    """forward를 기준으로 한 정규직교 셋 (forward, up, third).

    `forward`는 입력을 정규화한 값이다. 영벡터는 받지 않는다 — 영벡터가 나오는
    상황(예: 제자리 프레임)은 호출자가 자기 도메인에서 이미 대체 방향으로 해소해
    두었어야 한다. 그 전제가 깨지면 `ValueError`다: 0으로 지어낸 방향을 조용히
    돌려주지 않는다.

    `up`은 월드 수직축을 forward에 직교화한 것이다. forward가 월드 수직축과
    평행하면(순수 수직 방향) 그 직교화가 영벡터가 된다 — 이 경우 수평 대체축
    (`_fallback_horizontal`, 수평 축 쌍의 첫 축)을 대신 직교화해 up을 다시 세운다.
    이 퇴화 경로에서 `up`의 값은 수평이다: 이름이 아니라 벡터가 그 축의 뜻이다.

    `third`는 forward와 up의 외적이고 **부호를 정하지 않는다**. 유니티는 왼손
    좌표계라 외적의 방향 감각이 오른손계와 반대이고, 무엇이 "맞는" 부호인지는
    forward·up만으로는 알 수 없다 — 그 답은 호출자의 기하(예: 스트로크라면 건
    위치)에 달려 있다. 호출자가 `sign_toward`로 정한다; 여기서는 손대칭성을
    가정하지 않는다.
    """
    normalized = _normalize(forward)
    if normalized == (0.0, 0.0, 0.0):
        raise ValueError(
            "frame_around: forward는 영벡터일 수 없다 — 호출자가 자기 도메인에서 "
            "그 퇴화를 먼저 해소해야 한다(예: 제자리 프레임의 대체 접선)."
        )

    up_index = UP_INDEX
    up = _orthogonalized(_up_vector(up_index), normalized)
    if up == (0.0, 0.0, 0.0):
        up = _orthogonalized(_fallback_horizontal(up_index), normalized)

    third = _normalize(_cross3(normalized, up))
    return Frame(forward=normalized, up=up, third=third)
