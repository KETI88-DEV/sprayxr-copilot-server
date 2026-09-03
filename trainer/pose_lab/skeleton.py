#!/usr/bin/env python3
"""전신 FK 스켈레톤을 실측 3점(머리·오른손·왼손)에 강체 정렬한다.

`pose_fk_csv.forward_kinematics`가 내는 20관절은 루트 `Pelvis`가 원점인 리그 로컬
좌표다. 화면에 그리려면 실측 좌표계(착탄점과 같은 월드)로 옮겨야 하는데, 뼈 길이는
리그가 이미 정한 값이므로 스케일을 다시 풀면 안 된다 — `level_mesh._umeyama`는
스케일까지 푸는 유사변환이라 여기 그대로 쓸 수 없다. 이 모듈은 스케일 없는 Kabsch만
푼다.

**SVD는 새로 쓰지 않는다.** `level_mesh._svd3`(내부적으로 `_jacobi_eigen_sym3`를
쓰는, 표준 라이브러리만으로 짠 3x3 SVD)를 그대로 재사용한다. `level_mesh.py`의
공개 API를 넓히지 않으려고, 승격 대신 이 모듈이 사설 이름을 직접 임포트하는 쪽을
골랐다 — 커밋 메시지에도 이 선택을 적는다.

채점(σ·교정·개선율)은 이 경로를 타지 않는다. 여기서 만드는 것은 화면에 그릴
관절 위치와, 그 배치가 실측과 얼마나 어긋나는지 보여주는 잔차뿐이다. FK 자체가
틀려도(리그 규약 미확정 — `pose_fk_csv` 모듈 독스트링) 지표는 오염되지 않는다.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Mapping

_SCRIPTS_DIR = str(Path(__file__).resolve().parents[1] / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import level_mesh  # noqa: E402  (사설 이름 `_svd3` 재사용 — 아래 참고)
import world_axes  # noqa: E402

Vec3 = tuple[float, float, float]
Matrix = tuple[Vec3, Vec3, Vec3]

# FK 관절명과 `measured_pose.MeasuredFrame` 필드명의 쌍. 셋 다 실측이다 — `hit`/`uv`/
# `yaw`처럼 파생값이 아니라 카메라(머리)·활성 건(오른손)·플래쉬(왼손)의 위치 그 자체다.
ANCHORS: tuple[tuple[str, str], ...] = (
    ("Head", "head"),
    ("RightHand", "right"),
    ("LeftHand", "left"),
)


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _mean(points: list[Vec3]) -> Vec3:
    n = len(points)
    return (
        sum(p[0] for p in points) / n,
        sum(p[1] for p in points) / n,
        sum(p[2] for p in points) / n,
    )


def _det3(m: Matrix) -> float:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def _matmul(left: Matrix, right: Matrix) -> Matrix:
    return tuple(
        tuple(sum(left[row][k] * right[k][col] for k in range(3)) for col in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def _matvec(m: Matrix, v: Vec3) -> Vec3:
    return (
        sum(m[0][k] * v[k] for k in range(3)),
        sum(m[1][k] * v[k] for k in range(3)),
        sum(m[2][k] * v[k] for k in range(3)),
    )


def apply(rotation: Matrix, translation: Vec3, v: Vec3) -> Vec3:
    """`rotation @ v + translation`. 테스트가 `kabsch`의 결과를 검증하는 데 쓴다."""
    rv = _matvec(rotation, v)
    return _add(rv, translation)


def kabsch(source: list[Vec3], target: list[Vec3]) -> tuple[Matrix, Vec3] | None:
    """`target ≈ rotation @ source + translation`을 최소제곱으로 푼다. 스케일 없음.

    뼈 길이는 리그가 이미 고정했으므로 여기서 다시 늘리거나 줄이지 않는다 —
    `level_mesh._umeyama`가 내는 스케일 항을 쓰지 않는 이유가 이것이다.

    대응점이 2개 미만이면 회전을 풀 수 없다(1개는 평행이동만, 0개는 그마저도
    없다). 그런 입력을 평행이동만 하는 항등 회전으로 조용히 돌려주는 대신
    `None`을 낸다 — 못 풀었다는 사실 자체가 결과다. 호출자(`place_skeleton`)가
    2개짜리 앵커에는 가상의 세 번째 대응점을 미리 얹어서 넘긴다.
    """
    n = len(source)
    if n != len(target) or n < 2:
        return None

    mu_s = _mean(source)
    mu_t = _mean(target)
    sc = [_sub(p, mu_s) for p in source]
    tc = [_sub(p, mu_t) for p in target]

    # 공분산 H = Σ tᵢ sᵢᵀ (Kabsch/Umeyama와 같은 배치: target이 "q", source가 "p").
    h = [[0.0, 0.0, 0.0] for _ in range(3)]
    for s, t in zip(sc, tc):
        for r in range(3):
            for c in range(3):
                h[r][c] += t[r] * s[c]
    h_mat: Matrix = tuple(tuple(row) for row in h)  # type: ignore[assignment]

    u, _svals, vt = level_mesh._svd3(h_mat)  # noqa: SLF001 (의도적 사설 이름 재사용)

    # 반사(det < 0) 방지: 마지막 특이벡터를 뒤집는다.
    d = 1.0 if _det3(u) * _det3(vt) >= 0.0 else -1.0
    diag: Matrix = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, d))
    rotation = _matmul(_matmul(u, diag), vt)

    translation = _sub(mu_t, _matvec(rotation, mu_s))
    return rotation, translation


def place_skeleton(
    local: Mapping[str, Vec3], frame: object
) -> tuple[dict[str, Vec3], float] | None:
    """`local`(FK 관절, Pelvis가 원점)을 `frame`의 실측 3점에 강체 정렬한다.

    반환하는 잔차는 **실제로 쓰인 앵커별 (정렬 후 위치 − 실측 위치) 거리의 RMS**다
    — 가상으로 얹은 up 보정점은 포함하지 않는다(그 점은 자세를 고정하는 도구이지
    측정값이 아니다).

    앵커가 2개(예: 왼손 미검출)면 회전에 한 자유도가 남는다 — 두 점만으로는 그 둘을
    잇는 축 둘레의 "롤"이 정해지지 않는다. 여기서는 `world_axes.UP`을 세 번째
    가상 대응점으로 넣어 그 자유도를 고정한다: 두 실측 앵커의 중점에서 월드 up
    방향으로 한 걸음 나간 점을, local 쪽도 같은 방식으로 만들어 세 번째 쌍으로
    준다. 두 좌표계 모두 같은 up 방향을 공유한다는(둘 다 유니티 Y-up 규약을 쓴다는)
    전제 위에 선 선택이다 — 다른 자세 고정 방법(예: 몸통 전방축)도 가능했지만
    이 모듈이 가진 정보(관절 위치뿐, 몸통 지향 없음)로는 up이 유일하게 값싼 기준이다.

    앵커가 1개 이하면 회전을 풀 근거가 아예 없다. `(local 그대로 평행이동, nan)`
    같은 값을 지어내는 대신 `None`을 낸다.
    """
    pairs: list[tuple[Vec3, Vec3]] = []
    for joint_name, field_name in ANCHORS:
        if joint_name not in local:
            continue
        measured = getattr(frame, field_name, None)
        if measured is None:
            continue
        pairs.append((local[joint_name], measured))

    if len(pairs) <= 1:
        return None

    sources = [p[0] for p in pairs]
    targets = [p[1] for p in pairs]

    if len(pairs) == 2:
        up = world_axes.UP
        mid_source = _mean(sources)
        mid_target = _mean(targets)
        sources = sources + [_add(mid_source, up)]
        targets = targets + [_add(mid_target, up)]

    solved = kabsch(sources, targets)
    if solved is None:
        return None
    rotation, translation = solved

    placed = {name: apply(rotation, translation, position) for name, position in local.items()}

    distances = [math.dist(apply(rotation, translation, src), tgt) for src, tgt in pairs]
    residual = math.sqrt(sum(d * d for d in distances) / len(distances))
    return placed, residual
