#!/usr/bin/env python3
"""회전 표현 사이를 옮기고 회전공간에서 산포를 재는 최소 도구.

**표준 라이브러리만 쓴다.** numpy도 scipy도 쓰지 않는다. 이 저장소의 트레이너 컨테이너에
새 의존성을 넣지 않는다는 제약이 있고, 여기서 필요한 선형대수는 3×3 곱과 4×4 멱승법
하나뿐이라 직접 적는 편이 짧다.

왜 필요한가. `dataset_spec.md` §4.1은 오일러 각에 산술 통계를 쓰지 말라고 못박았고
그래서 지금까지 축별 **순환** 통계를 썼다. 그런데 순환 통계도 축별이면 오일러 좌표계에
갇힌다 — 세 축이 서로 독립이 아니고 특이점에서 결합하므로, X가 8.7°인데 Y·Z가 120°인
버킷이 나온다(`pose-lab-bucket-study.md` §4.2). 그 120°가 진짜 산포인지 오일러 결합의
지문인지는 **회전 표현에 무관한** 측도로만 갈린다. 그것이 쿼터니언 측지 산포다.

CSV의 오일러 규약은 벤더 문서로 확인된 적이 없다(설계 §10). 그래서 이 모듈은 규약을
가정하지 않는다 — 순서·부호를 인자로 받고, 어느 규약이 맞는지는 데이터가 고르게 한다
(`measured_pose.FACING`).
"""

from __future__ import annotations

import math
from typing import Final, Iterable, Sequence

Vec3 = tuple[float, float, float]
Mat3 = tuple[Vec3, Vec3, Vec3]
Quat = tuple[float, float, float, float]

AXIS_INDEX: Final[dict[str, int]] = {"X": 0, "Y": 1, "Z": 2}
EULER_ORDERS: Final[tuple[str, ...]] = ("XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX")
LOCAL_AXES: Final[dict[str, Vec3]] = {
    "+X": (1.0, 0.0, 0.0),
    "-X": (-1.0, 0.0, 0.0),
    "+Y": (0.0, 1.0, 0.0),
    "-Y": (0.0, -1.0, 0.0),
    "+Z": (0.0, 0.0, 1.0),
    "-Z": (0.0, 0.0, -1.0),
}

IDENTITY: Final[Mat3] = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

# 짐벌락 판정. |cos(중간각)|이 이보다 작으면 바깥 두 각이 분리되지 않는다.
GIMBAL_EPS: Final = 1e-7


def axis_matrix(axis: str, degrees: float) -> Mat3:
    """한 축 둘레의 우수 회전 행렬."""
    angle = math.radians(degrees)
    cos, sin = math.cos(angle), math.sin(angle)
    if axis == "X":
        return ((1.0, 0.0, 0.0), (0.0, cos, -sin), (0.0, sin, cos))
    if axis == "Y":
        return ((cos, 0.0, sin), (0.0, 1.0, 0.0), (-sin, 0.0, cos))
    if axis == "Z":
        return ((cos, -sin, 0.0), (sin, cos, 0.0), (0.0, 0.0, 1.0))
    raise ValueError(f"unknown axis: {axis}")


def matmul(left: Mat3, right: Mat3) -> Mat3:
    return tuple(  # pyright: ignore[reportReturnType]
        tuple(sum(left[row][k] * right[k][col] for k in range(3)) for col in range(3))
        for row in range(3)
    )


def matvec(matrix: Mat3, vector: Vec3) -> Vec3:
    return tuple(  # pyright: ignore[reportReturnType]
        sum(matrix[row][k] * vector[k] for k in range(3)) for row in range(3)
    )


def transpose(matrix: Mat3) -> Mat3:
    return tuple(  # pyright: ignore[reportReturnType]
        tuple(matrix[col][row] for col in range(3)) for row in range(3)
    )


def euler_matrix(order: str, angles_deg: Vec3, sign: float = 1.0) -> Mat3:
    """`R = R_{order[0]} · R_{order[1]} · R_{order[2]}`.

    `angles_deg`는 CSV 그대로 (X, Y, Z) 순이다. 어느 각이 언제 곱해지는지만 `order`가
    정하고 각은 자기 축에 붙어 있다. `sign=-1`은 좌수/우수 좌표계 차이를 흡수한다 —
    시뮬레이터가 어느 쪽인지 문서가 없으므로 두 부호를 다 후보로 둔다.
    """
    result = IDENTITY
    for axis in order:
        result = matmul(result, axis_matrix(axis, sign * angles_deg[AXIS_INDEX[axis]]))
    return result


def _parity(order: str) -> float:
    """(i, j, k)가 (0,1,2)의 짝수(순환) 치환이면 +1, 홀수면 −1."""
    i, j, k = (AXIS_INDEX[axis] for axis in order)
    return 1.0 if (j - i) % 3 == 1 and (k - j) % 3 == 1 else -1.0


def euler_from_matrix(order: str, matrix: Mat3, sign: float = 1.0) -> Vec3:
    """`euler_matrix`의 역. (X, Y, Z) 순의 도 단위로 되돌린다.

    Tait-Bryan(세 축이 모두 다른) 6개 순서에 대한 일반식이다. `ε = _parity(order)`일 때
    `sin(중간각) = ε · R[i][k]`이고 바깥 두 각은 같은 행/열에서 나온다. 중간각이
    ±90°에 붙으면(짐벌락) 바깥 두 각의 합/차만 결정되므로 뒤쪽 각을 0으로 두고 앞쪽에
    몰아준다 — 각을 지어내지 않고 같은 회전을 내는 대표값을 고른다.
    """
    i, j, k = (AXIS_INDEX[axis] for axis in order)
    eps = _parity(order)
    middle_sin = max(-1.0, min(1.0, eps * matrix[i][k]))
    middle = math.asin(middle_sin)
    if abs(math.cos(middle)) > GIMBAL_EPS:
        first = math.atan2(-eps * matrix[j][k], matrix[k][k])
        last = math.atan2(-eps * matrix[i][j], matrix[i][i])
    else:
        combined = math.atan2(eps * matrix[j][i], matrix[j][j])
        first = combined if middle_sin >= 0.0 else -combined
        last = 0.0
    out = [0.0, 0.0, 0.0]
    out[i], out[j], out[k] = first, middle, last
    return (
        math.degrees(out[0]) / sign,
        math.degrees(out[1]) / sign,
        math.degrees(out[2]) / sign,
    )


def matrix_quaternion(matrix: Mat3) -> Quat:
    """회전 행렬 → 단위 쿼터니언 `(w, x, y, z)`. 부호는 `w ≥ 0`으로 못 박는다.

    `q`와 `−q`는 같은 회전이다. 부호를 고정하지 않으면 평균과 측지거리에서 같은 회전이
    서로 반대편에 놓여 산포가 통째로 부푼다.
    """
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = (
            0.25 * scale,
            (matrix[2][1] - matrix[1][2]) / scale,
            (matrix[0][2] - matrix[2][0]) / scale,
            (matrix[1][0] - matrix[0][1]) / scale,
        )
    elif matrix[0][0] > matrix[1][1] and matrix[0][0] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2.0
        quat = (
            (matrix[2][1] - matrix[1][2]) / scale,
            0.25 * scale,
            (matrix[0][1] + matrix[1][0]) / scale,
            (matrix[0][2] + matrix[2][0]) / scale,
        )
    elif matrix[1][1] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2.0
        quat = (
            (matrix[0][2] - matrix[2][0]) / scale,
            (matrix[0][1] + matrix[1][0]) / scale,
            0.25 * scale,
            (matrix[1][2] + matrix[2][1]) / scale,
        )
    else:
        scale = math.sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2.0
        quat = (
            (matrix[1][0] - matrix[0][1]) / scale,
            (matrix[0][2] + matrix[2][0]) / scale,
            (matrix[1][2] + matrix[2][1]) / scale,
            0.25 * scale,
        )
    return canonical(normalize_quat(quat))


def quaternion_matrix(quat: Quat) -> Mat3:
    w, x, y, z = normalize_quat(quat)
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def normalize_quat(quat: Quat) -> Quat:
    length = math.sqrt(sum(component * component for component in quat))
    if length == 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    return (quat[0] / length, quat[1] / length, quat[2] / length, quat[3] / length)


def canonical(quat: Quat) -> Quat:
    """`w < 0`이면 뒤집는다. 같은 회전의 두 표현 중 하나를 고정한다."""
    if quat[0] < 0.0:
        return (-quat[0], -quat[1], -quat[2], -quat[3])
    return quat


def quat_dot(left: Quat, right: Quat) -> float:
    return sum(left[axis] * right[axis] for axis in range(4))


def geodesic_deg(left: Quat, right: Quat) -> float:
    """두 회전 사이의 측지 각 `θ = 2·acos(|q₁·q₂|)`. 도 단위, 범위 [0, 180].

    `|·|`이 있는 이유는 `q`와 `−q`가 같은 회전이기 때문이다. 이 값은 한 회전을 다른
    회전으로 옮기는 데 필요한 **최소 회전각**이고, 축이 어느 쪽이든 무관하다. 즉
    오일러 좌표계에 의존하지 않는다.
    """
    return math.degrees(2.0 * math.acos(min(1.0, abs(quat_dot(left, right)))))


def quaternion_mean(quats: Sequence[Quat], iterations: int = 64) -> Quat | None:
    """평균 회전. 4×4 산포행렬 `M = Σ qqᵀ`의 최대 고유벡터다(Markley 평균).

    이것이 프레셰(Fréchet) 평균의 닫힌 형태 근사이며, 쿼터니언 부호 모호성에 견딘다 —
    `qqᵀ`가 부호에 불변이기 때문이다. 고유벡터는 멱승법으로 뽑는다. 4×4라 scipy가
    필요 없다.

    표본이 회전공간에 고르게 퍼져 최대 고유값이 분리되지 않으면 평균 방향이 정의되지
    않는다. 그때는 각도를 지어내지 않고 None을 낸다 — 순환 평균에서 R≈0일 때와 같은
    처리다(`motion_key_sweep.circular_mean_deg`).
    """
    if not quats:
        return None
    scatter = [[0.0] * 4 for _ in range(4)]
    for quat in quats:
        for row in range(4):
            for col in range(4):
                scatter[row][col] += quat[row] * quat[col]

    vector = list(canonical(normalize_quat(quats[0])))
    for _ in range(iterations):
        nxt = [sum(scatter[row][col] * vector[col] for col in range(4)) for row in range(4)]
        length = math.sqrt(sum(value * value for value in nxt))
        if length <= 1e-12:
            return None
        vector = [value / length for value in nxt]
    return canonical((vector[0], vector[1], vector[2], vector[3]))


def geodesic_spread(quats: Sequence[Quat]) -> tuple[float, float] | None:
    """평균 회전 둘레의 측지 산포 `(mean θ, rms θ)`. 도 단위.

    `θᵢ = 2·acos(|qᵢ·q̄|)`이고 `q̄`는 `quaternion_mean`이다.

    - `mean θ` — 평균 각오차. "숙련자 대비 N도"의 N과 같은 단위다.
    - `rms θ = √(mean θ²)` — 표준편차와 같은 2차 모멘트. 순환 std와 같은 성격이라
      게이트 비교에서 축별 순환 std의 짝이 된다.

    둘을 다 내는 이유는 하나만 내면 게이트 통과 여부가 그 선택에 걸리기 때문이다.
    """
    if len(quats) < 2:
        return None
    mean = quaternion_mean(quats)
    if mean is None:
        return None
    angles = [geodesic_deg(quat, mean) for quat in quats]
    count = len(angles)
    average = sum(angles) / count
    rms = math.sqrt(sum(angle * angle for angle in angles) / count)
    return (average, rms)


def basis_matrix(columns: Iterable[Vec3]) -> Mat3:
    """열벡터 셋을 행렬로. `matvec(transpose(B), v)`가 로컬 성분이다."""
    cols = list(columns)
    return tuple(  # pyright: ignore[reportReturnType]
        tuple(cols[col][row] for col in range(3)) for row in range(3)
    )
