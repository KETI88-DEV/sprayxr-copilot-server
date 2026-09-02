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
from typing import Final

Vec3 = tuple[float, float, float]

UP_INDEX: Final = 1
UP: Final[Vec3] = (0.0, 1.0, 0.0)
HORIZONTAL_INDICES: Final[tuple[int, int]] = (0, 2)
# 유니티는 왼손 좌표계다. 외적의 방향 감각이 오른손계와 반대이므로 새 외적을 쓸 때마다
# 부호를 기준점으로 다시 정해야 한다 — `sign_toward`가 그 일을 한다.
HANDEDNESS: Final = "left"

# 수평 성분이 이보다 짧으면 방위가 정의되지 않는다(정확히 위/아래를 본 경우).
MIN_HORIZONTAL: Final = 1e-9


def vertical_of(v: Vec3) -> float:
    return v[UP_INDEX]


def horizontal_of(v: Vec3) -> tuple[float, float]:
    a, b = HORIZONTAL_INDICES
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
